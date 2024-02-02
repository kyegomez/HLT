from hlt_torch.model import HLTransformer, MaxViT
import torch

vit = MaxViT(
    num_classes=1000,
    dim_conv_stem=64,
    dim=96,
    dim_head=32,
    depth=(2, 2, 5, 2),
    window_size=7,
    mbconv_expansion_rate=4,
    mbconv_shrinkage_rate=0.25,
    dropout=0.1,
)

model = HLTransformer(
    vit=vit,
    num_actions=11,
    depth=6,
    heads=8,
    dim_head=64,
    cond_drop_prob=0.2,
)

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    "bring me that apple sitting on the table",
    "please pass the butter",
]

train_logits = model(
    video, instructions
)  # (2, 6, 11, 256) # (batch, frames, actions, bins)
print(train_logits)
