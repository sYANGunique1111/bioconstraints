import torch
from einops import rearrange

B, F, J, C = 2, 135, 17, 256
x = torch.randn(B, F, J, C)

# Spatial
x = rearrange(x, "b f n c -> (b f) n c")
# ... spatial block ...

# Temporal
x = rearrange(x, "(b f) n c -> (b n) f c", b=B)
print("x shape:", x.shape)

pruned_pos_embed = torch.randn(B, F, C)
print("pruned_pos_embed shape:", pruned_pos_embed.shape)

try:
    x = x + pruned_pos_embed
    print("Success")
except Exception as e:
    print("Error:", e)
