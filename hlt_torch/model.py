from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from beartype import beartype
from classifier_free_guidance_pytorch import (
    AttentionTextConditioner,
    TextConditioner,
    classifier_free_guidance,
)
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, einsum, nn
from zeta.nn import Residual

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# helper classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.0,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        window_size=7,
        num_mem_kv=4,
    ):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.mem_kv = nn.Parameter(torch.randn(2, self.heads, num_mem_kv, dim_head))

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        (
            batch,
            height,
            width,
            window_height,
            window_width,
            _,
            device,
            h,
        ) = (*x.shape, x.device, self.heads)

        x = self.norm(x)

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # null / memory / register kv

        mk, mv = map(
            lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]),
            self.mem_kv,
        )
        num_mem = mk.shape[-2]

        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)

        bias = F.pad(bias, (0, 0, num_mem, 0), value=0.0)

        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out,
            "b h (w1 w2) d -> b w1 w2 (h d)",
            w1=window_height,
            w2=window_width,
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
    ):
        super().__init__()
        assert isinstance(depth, tuple), (
            "depth needs to be tuple if integers indicating number of"
            " transformer blocks at that stage"
        )

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2**i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, (
            (layer_dim_in, layer_dim),
            layer_depth,
        ) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate,
                    ),
                    Rearrange(
                        "b d (x w1) (y w2) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # block-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
                    Rearrange(
                        "b d (w1 x) (w2 y) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # grid-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob=0.0,
        return_embeddings=False,
    ):
        x = self.conv_stem(x)

        cond_fns = iter(default(cond_fns, []))

        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)


# attention


class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        dropout=0.1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        attn_mask=None,
        cond_fn: Optional[Callable] = None,
    ):
        x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        q = q * self.scale

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        depth=6,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(
                            dim=dim, heads=heads, dropout=attn_dropout
                        ),
                        FeedForward(dim=dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask=None,
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
            x = (
                attn(
                    x,
                    attn_mask=attn_mask,
                    cond_fn=next(cond_fns, None),
                )
                + x
            )
            x = ff(x, cond_fn=next(cond_fns, None)) + x
        return x


# token learner module


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8, num_layers=2):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(
                dim * num_output_tokens,
                inner_dim,
                1,
                groups=num_output_tokens,
            ),
            nn.GELU(),
            nn.Conv2d(
                inner_dim,
                num_output_tokens,
                1,
                groups=num_output_tokens,
            ),
        )

    def forward(self, x):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x


# Robotic Transformer


@beartype
class HLTransformer(nn.Module):
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions=11,
        action_bins=256,
        depth=4,
        heads=8,
        dim_head=64,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
        cond_drop_prob=0.2,
        use_attn_conditioner=False,
        conditioner_kwargs: dict = {},
    ):
        super().__init__()
        self.vit = vit

        self.num_vit_stages = len(vit.cond_hidden_dims)

        conditioner_klass = (
            AttentionTextConditioner if use_attn_conditioner else TextConditioner
        )

        self.conditioner = conditioner_klass(
            hidden_dims=(
                *tuple(vit.cond_hidden_dims),
                *((vit.embed_dim,) * depth * 2),
            ),
            hiddens_channel_first=(
                *((True,) * self.num_vit_stages),
                *((False,) * depth * 2),
            ),
            cond_drop_prob=cond_drop_prob,
            **conditioner_kwargs,
        )

        self.token_learner = TokenLearner(
            dim=vit.embed_dim,
            ff_mult=token_learner_ff_mult,
            num_output_tokens=token_learner_num_output_tokens,
            num_layers=token_learner_num_layers,
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim=vit.embed_dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
        )

        self.cond_drop_prob = cond_drop_prob

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange("... (a b) -> ... a b", b=action_bins),
        )

    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob=0.0,
    ):
        assert exists(texts) ^ exists(text_embeds)
        cond_kwargs = dict(texts=texts, text_embeds=text_embeds)

        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns, _ = self.conditioner(
            **cond_kwargs,
            cond_drop_prob=cond_drop_prob,
            repeat_batch=(
                *((frames,) * self.num_vit_stages),
                *((1,) * self.transformer_depth * 2),
            ),
        )

        vit_cond_fns, transformer_cond_fns = (
            cond_fns[: -(depth * 2)],
            cond_fns[-(depth * 2) :],
        )

        video = rearrange(video, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(video, "* c h w")

        tokens = self.vit(
            images,
            texts=texts,
            cond_fns=vit_cond_fns,
            cond_drop_prob=cond_drop_prob,
            return_embeddings=True,
        )

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, "b f c n -> b (f n) c")

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype=torch.bool, device=device).triu(
            1
        )
        attn_mask = repeat(
            attn_mask,
            "i j -> (i r1) (j r2)",
            r1=self.num_learned_tokens,
            r2=self.num_learned_tokens,
        )

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(
            frames,
            learned_tokens.shape[-1],
            dtype=learned_tokens.dtype,
            device=learned_tokens.device,
        )

        learned_tokens = learned_tokens + repeat(
            pos_emb, "n d -> (n r) d", r=self.num_learned_tokens
        )

        # attention

        attended_tokens = self.transformer(
            learned_tokens,
            cond_fns=transformer_cond_fns,
            attn_mask=~attn_mask,
        )

        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=frames)

        logits = self.to_logits(pooled)
        return logits


class StatePolicy(nn.Module):
    """
    Initializes the StatePolicy module.

    Args:
        dim (int): The input dimension.
        dropout (int): The dropout rate.
        mult (int, optional): The multiplier for the hidden dimension. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim: int = 512,
        dropout: int = 0.1,
        mult: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.ffn = FeedForward(dim, mult, dropout=dropout)

    def forward(self, x: Tensor):
        """
        Performs a forward pass of the StatePolicy module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.ffn(x)


class HLT(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim_conv_stem: int,
        dim: int,
        dim_head: int,
        depth: Tuple[int],
        window_size: int,
        mbconv_expansion_rate: int,
        mbconv_shrinkage_rate: int,
        dropout: float,
        num_actions: int,
        hl_depth: int,
        hl_heads: int,
        hl_dim_head: int,
        cond_drop_prob: float,
    ):
        """
        Initializes the MaxViTWithHLTransformer module.

        Args:
            num_classes (int): The number of output classes.
            dim_conv_stem (int): The dimension of the convolutional stem.
            dim (int): The dimension of the transformer.
            dim_head (int): The dimension of each transformer head.
            depth (int): The depth of the transformer.
            window_size (int): The window size for the transformer.
            mbconv_expansion_rate (int): The expansion rate for the MBConv blocks.
            mbconv_shrinkage_rate (int): The shrinkage rate for the MBConv blocks.
            dropout (float): The dropout rate.
            num_actions (int): The number of actions.
            hl_depth (int): The depth of the HLTransformer.
            hl_heads (int): The number of heads in the HLTransformer.
            hl_dim_head (int): The dimension of each head in the HLTransformer.
            cond_drop_prob (float): The conditional dropout probability.
        """
        super().__init__()

        self.vit = MaxViT(
            num_classes=num_classes,
            dim_conv_stem=dim_conv_stem,
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate,
            dropout=dropout,
        )

        self.hl_transformer = HLTransformer(
            vit=self.vit,
            num_actions=num_actions,
            depth=hl_depth,
            heads=hl_heads,
            dim_head=hl_dim_head,
            cond_drop_prob=cond_drop_prob,
        )

    def forward(self, video: Tensor, instructions: Tensor) -> Tensor:
        """
        Performs a forward pass of the MaxViTWithHLTransformer module.

        Args:
            video (Tensor): The input video tensor.
            instructions (Tensor): The input instructions tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.hl_transformer(video, instructions)
