import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp

class CrossAttention(nn.Module):
    def __init__(
            self, 
            dim: int, 
            num_heads: int = 8, 
            qkv_bias: bool = False, 
            attn_drop: float = 0.0, 
            proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project Query (from input source 1)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Project Key and Value (from input source 2)
        # We project them together for efficiency
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q: Query input of shape (Batch, Seq_Q, Dim)
            x_kv: Key/Value input of shape (Batch, Seq_K, Dim)
            
        Note: Seq_Q and Seq_K can be different lengths.
        """
        B, N_q, C = x_q.shape
        B, N_k, _ = x_kv.shape

        # 1. Project Query
        # Shape: [B, N_q, num_heads, head_dim] -> permute to [B, num_heads, N_q, head_dim]
        q = self.q_proj(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. Project Key/Value
        # Shape: [B, N_k, 2 * dim] -> split -> [B, N_k, num_heads, head_dim]
        kv = self.kv_proj(x_kv).reshape(B, N_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # k, v are both [B, num_heads, N_k, head_dim]

        # 3. Scaled Dot Product Attention
        # PyTorch 2.0+ optimization (Flash Attention compatible)
        # The math naturally handles N_q != N_k because:
        # (N_q, d) x (d, N_k) -> (N_q, N_k) attention map
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale
        )

        # 4. Concatenate heads and project output
        # [B, num_heads, N_q, head_dim] -> [B, N_q, dim]
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4.0, 
        qkv_bias=False, 
        drop=0.0, 
        attn_drop=0.0, 
        drop_path=0.0
    ):
        super().__init__()
        
        # --- Part 1: Cross Attention ---
        self.norm1 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        self.cross_attn = CrossAttention(
            dim=dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        
        # --- Part 2: Feed Forward Network (FFN) ---
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            act_layer=nn.GELU,
            drop=drop
        )
        # DropPath (Stochastic Depth) is crucial for training deep ViTs
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv):
        # 1. Cross Attention Sub-layer
        # Note: Pre-Norm architecture (Norm applies before the operation)
        # Residual connection adds to the query stream (x_q)
        x_q = x_q + self.drop_path(self.cross_attn(
            self.norm1(x_q), 
            self.norm_kv(x_kv)
        ))
        
        # 2. FFN Sub-layer
        # Only processes the Query stream (x_q)
        x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q)))
        
        return x_q

# --- Usage Example ---
if __name__ == "__main__":
    # Settings
    dim = 256
    num_heads = 8
    
    # Define module
    cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
    
    # Create dummy inputs with DIFFERENT sequence lengths
    # Query: 1 image token (e.g., class token)
    query_src = torch.randn(1, 1, dim) 
    
    # KV: 196 image patches (14x14)
    kv_src = torch.randn(1, 196, dim)  

    # Forward pass
    output = cross_attn(query_src, kv_src)
    
    print(f"Query Source shape: {query_src.shape}")
    print(f"KV Source shape:    {kv_src.shape}")
    print(f"Output shape:       {output.shape}") 
    # Output matches Query length: [1, 1, 256]