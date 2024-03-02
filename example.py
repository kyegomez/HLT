import torch

from hlt_torch.model import HLT

# Import the necessary libraries
# Create an instance of the HLT model
model = HLT(
    num_classes=10,
    dim_conv_stem=64,
    dim=512,
    dim_head=64,
    depth=(4, 4, 4),
    window_size=8,
    mbconv_expansion_rate=4,
    mbconv_shrinkage_rate=2,
    dropout=0.1,
    num_actions=11,
    hl_depth=4,
    hl_heads=8,
    hl_dim_head=64,
    cond_drop_prob=0.2,
)

# Generate some dummy input tensors
video = torch.randn(
    1, 3, 16, 112, 112
)  # Shape: (batch_size, num_channels, num_frames, height, width)
instructions = torch.randn(
    1, 10, 512
)  # Shape: (batch_size, num_instructions, embedding_dim)

# Perform a forward pass through the model
output = model(video, instructions)

# Print the output tensor
print(output)
