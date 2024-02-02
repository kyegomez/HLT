[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Humanoid Locomotion Transformer
This is an implementation of the robotic transformer for humanoid robots from the premier paper from berkely: "Real-World Humanoid Locomotion with Reinforcement Learning". Here we implement the state policy model which is an MLP/FFN and a Transformer model that intakes both observation and action tokens to output the next action sequence.


## Install
`pip install hlt-torch`


## Usage
```python
# Import necessary modules
from hlt_torch.model import HLTransformer, MaxViT
import torch

# Define the MagitxViT model
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

# Define the HLTransformer model
model = HLTransformer(
    vit=vit,
    num_actions=11,
    depth=6,
    heads=8,
    dim_head=64,
    cond_drop_prob=0.2,
)

# Create a random video tensor
video = torch.randn(2, 3, 6, 224, 224)

# Define a list of instructions
instructions = [
    "bring me that apple sitting on the table",
    "please pass the butter",
]

# Pass the video and instructions through the model to get the train logits
train_logits = model(
    video, instructions
)  # (2, 6, 11, 256) # (batch, frames, actions, bins)

# Print the train logits
print(train_logits)


```

# License
MIT
