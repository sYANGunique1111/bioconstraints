"""
MixSTE2: Mixed Spatio-Temporal Encoder
Revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from timm.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.comb:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, comb=comb)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HybridSpatialBlock(nn.Module):
    """
    Hybrid Spatial Block for processing joint and semantic groups separately.
    
    This block improves interaction between joint tokens and semantic tokens by:
    1. Normalizing them separately (LayerNorm for joint, F.layer_norm + gamma/beta for semantic)
    2. Applying MHSA to all tokens together
    3. Processing them through separate MLPs
    4. Merging back via concatenation
    
    Input: (B, T, P, C) where P = num_joints + num_groups
           - First num_joints tokens: joint groups
           - Remaining tokens: semantic groups
    Output: (B, T, P, C) - same shape
    """
    def __init__(self, dim, num_joints=17, num_groups=5, num_heads=8, mlp_ratio=2., 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = num_joints
        self.num_groups = num_groups
        
        # Separate normalizations for joint and semantic tokens
        self.joint_norm1 = norm_layer(dim)  # Standard LayerNorm for joints
        # For semantic: F.layer_norm + learnable gamma/beta
        self.semantic_gamma1 = nn.Parameter(torch.ones(1, 1, num_groups, dim))
        self.semantic_beta1 = nn.Parameter(torch.zeros(1, 1, num_groups, dim))
        
        # MHSA (applied to all tokens together)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, comb=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_drop = nn.Dropout(drop)
        
        # Separate MLPs for joint and semantic tokens
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.joint_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                            act_layer=act_layer, drop=drop)
        self.semantic_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                               act_layer=act_layer, drop=drop)
        
        # Separate norms before MLPs
        self.joint_norm2 = norm_layer(dim)
        self.semantic_gamma2 = nn.Parameter(torch.ones(1, 1, num_groups, dim))
        self.semantic_beta2 = nn.Parameter(torch.zeros(1, 1, num_groups, dim))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, P, C) where P = num_joints + num_groups
        
        Returns:
            x: (B, T, P, C) - same shape
        """
        B, T, P, C = x.shape
        assert P == self.num_joints + self.num_groups, \
            f"Expected {self.num_joints + self.num_groups} tokens, got {P}"
        
        # Store original for residual
        residual = x
        
        # Split into joint and semantic groups
        joint_tokens = x[:, :, :self.num_joints, :]  # (B, T, 17, C)
        semantic_tokens = x[:, :, self.num_joints:, :]  # (B, T, num_groups, C)
        
        # === Step 1: Separate Normalization ===
        # Joint: Standard LayerNorm
        joint_normed = self.joint_norm1(joint_tokens)
        
        # Semantic: F.layer_norm + gamma/beta
        semantic_normed = F.layer_norm(semantic_tokens, (semantic_tokens.shape[-1],), 
                                       weight=None, bias=None, eps=1e-6)
        semantic_normed = semantic_normed * self.semantic_gamma1 + self.semantic_beta1
        
        # Concatenate normalized tokens
        x_normed = torch.cat([joint_normed, semantic_normed], dim=2)  # (B, T, P, C)
        
        # === Step 2: MHSA with residual ===
        # Reshape for attention: (B*T, P, C)
        BT = B * T
        x_normed = x_normed.view(BT, P, C)
        
        # Apply MHSA
        attn_out = self.attn(x_normed)  # (B*T, P, C)
        attn_out = self.attn_drop(attn_out)
        
        # Reshape back and add residual
        attn_out = attn_out.view(B, T, P, C)
        x = residual + self.drop_path(attn_out)
        
        # === Step 3: Split and process through separate MLPs ===
        # Split again
        joint_tokens = x[:, :, :self.num_joints, :]
        semantic_tokens = x[:, :, self.num_joints:, :]
        
        # Joint MLP branch
        joint_normed = self.joint_norm2(joint_tokens)
        joint_mlp_out = self.joint_mlp(joint_normed)
        joint_out = joint_tokens + self.drop_path(joint_mlp_out)
        
        # Semantic MLP branch
        semantic_normed = F.layer_norm(semantic_tokens, (semantic_tokens.shape[-1],),
                                      weight=None, bias=None, eps=1e-6)
        semantic_normed = semantic_normed * self.semantic_gamma2 + self.semantic_beta2
        semantic_mlp_out = self.semantic_mlp(semantic_normed)
        semantic_out = semantic_tokens + self.drop_path(semantic_mlp_out)
        
        # === Step 4: Merge ===
        x = torch.cat([joint_out, semantic_out], dim=2)  # (B, T, P, C)
        
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention module where Q comes from one source and K,V from another.
    
    Used for joint tokens (Q) to attend to semantic tokens (K,V).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_input, kv_input):
        """
        Args:
            q_input: Query tokens (B, N_q, C)
            kv_input: Key-Value tokens (B, N_kv, C)
        
        Returns:
            Output tokens (B, N_q, C)
        """
        B, N_q, C = q_input.shape
        _, N_kv, _ = kv_input.shape
        
        # Generate Q, K, V
        q = self.q(q_input).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(kv_input).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv_input).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Cross-attention: Q from q_input, K,V from kv_input
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N_q, N_kv)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # (B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class HybridSpatialBlockV2(nn.Module):
    """
    Hybrid Spatial Block V2 with Cross-Attention (MHCA variant).
    
    This block uses cross-attention where joint tokens attend to semantic tokens:
    - Joint tokens as Query
    - Semantic tokens as Key and Value
    
    Workflow:
    1. Normalize both groups separately
    2. MHCA: joints (Q) attend to semantics (K,V)
    3. Update joints via residual + MHCA output
    4. Process both through separate MLPs with residuals
    5. Merge back
    
    Input: (B, T, P, C) where P = num_joints + num_groups
    Output: (B, T, P, C) - same shape
    """
    def __init__(self, dim, num_joints=17, num_groups=5, num_heads=8, mlp_ratio=2., 
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_joints = num_joints
        self.num_groups = num_groups
        
        # Separate normalizations for joint and semantic tokens (before cross-attention)
        self.joint_norm1 = norm_layer(dim)  # Standard LayerNorm for joints (Q)
        # For semantic (K,V): F.layer_norm + learnable gamma/beta
        self.semantic_gamma1 = nn.Parameter(torch.ones(1, 1, num_groups, dim))
        self.semantic_beta1 = nn.Parameter(torch.zeros(1, 1, num_groups, dim))
        
        # Cross-Attention: joints attend to semantics
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_drop = nn.Dropout(drop)
        
        # Separate MLPs for joint and semantic tokens
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.joint_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                            act_layer=act_layer, drop=drop)
        self.semantic_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                               act_layer=act_layer, drop=drop)
        
        # Separate norms before MLPs
        self.joint_norm2 = norm_layer(dim)
        self.semantic_norm2 = norm_layer(dim)
        # self.semantic_gamma2 = nn.Parameter(torch.ones(1, 1, num_groups, dim))
        # self.semantic_beta2 = nn.Parameter(torch.zeros(1, 1, num_groups, dim))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, P, C) where P = num_joints + num_groups
        
        Returns:
            x: (B, T, P, C) - same shape
        """
        B, T, P, C = x.shape
        assert P == self.num_joints + self.num_groups, \
            f"Expected {self.num_joints + self.num_groups} tokens, got {P}"
        
        # Split into joint and semantic groups
        joint_tokens = x[:, :, :self.num_joints, :]  # (B, T, 17, C)
        semantic_tokens = x[:, :, self.num_joints:, :]  # (B, T, num_groups, C)
        
        # === Step 1: Separate Normalization ===
        # Joint: Standard LayerNorm (for Q)
        joint_normed = self.joint_norm1(joint_tokens)
        
        # Semantic: F.layer_norm + gamma/beta (for K,V)
        semantic_normed = F.layer_norm(semantic_tokens, (semantic_tokens.shape[-1],), 
                                       weight=None, bias=None, eps=1e-6)
        semantic_normed = semantic_normed * self.semantic_gamma1 + self.semantic_beta1
        
        # === Step 2: Cross-Attention (joints attend to semantics) ===
        # Reshape for cross-attention: (B*T, N, C)
        BT = B * T
        joint_normed_flat = joint_normed.view(BT, self.num_joints, C)
        semantic_normed_flat = semantic_normed.view(BT, self.num_groups, C)
        
        # Cross-attention: Q from joints, K,V from semantics
        cross_attn_out = self.cross_attn(joint_normed_flat, semantic_normed_flat)  # (B*T, 17, C)
        cross_attn_out = self.attn_drop(cross_attn_out)
        
        # Reshape back and add residual  (only for joint tokens)
        cross_attn_out = cross_attn_out.view(B, T, self.num_joints, C)
        joint_tokens = joint_tokens + self.drop_path(cross_attn_out)
        
        # === Step 3: Process through separate MLPs ===
        # Joint MLP branch
        joint_normed = self.joint_norm2(joint_tokens)
        joint_mlp_out = self.joint_mlp(joint_normed)
        joint_out = joint_tokens + self.drop_path(joint_mlp_out)
        
        # Semantic MLP branch (semantics get their own MLP)
        # semantic_normed = F.layer_norm(semantic_tokens, (semantic_tokens.shape[-1],),
        #                               weight=None, bias=None, eps=1e-6)
        # semantic_normed = semantic_normed * self.semantic_gamma2 + self.semantic_beta2
        semantic_normed = self.semantic_norm2(semantic_tokens)
        semantic_mlp_out = self.semantic_mlp(semantic_normed)
        semantic_out = semantic_tokens + self.drop_path(semantic_mlp_out)
        
        # === Step 4: Merge ===
        x = torch.cat([joint_out, semantic_out], dim=2)  # (B, T, P, C)
        
        return x


class MixSTE2(nn.Module):
    """
    Mixed Spatio-Temporal Encoder for 2D-to-3D pose estimation.
    
    Args:
        num_frame (int): input frame number
        num_joints (int): joints number
        in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
        embed_dim_ratio (int): embedding dimension ratio
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
    """
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3  # output dimension is 3 (x, y, z)

        # Spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def STE_forward(self, x):
        """Spatial Transformer Encoder forward pass."""
        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_forward(self, x):
        """Temporal Transformer Encoder forward pass."""
        assert len(x.shape) == 3, "shape should be 3"
        b, f, _ = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)
        x = self.Temporal_norm(x)
        return x

    def ST_forward(self, x):
        """Alternating Spatio-Temporal forward pass."""
        assert len(x.shape) == 4, "shape should be 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 2D poses of shape (batch, frames, joints, 2)
            
        Returns:
            3D poses of shape (batch, frames, joints, 3)
        """
        b, f, n, c = x.shape
        
        x = self.STE_forward(x)
        x = self.TTE_forward(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_forward(x)
        x = self.head(x)
        x = x.view(b, f, n, -1)

        return x


class HybridMixSTEEmbedder(nn.Module):
    """
    Hybrid Pose Embedder that groups joints by body parts and applies temporal patching.
    
    Unlike HybridPoseEmbedder3_2 which flattens the output, this keeps spatial and temporal
    dimensions separate to fit MixSTE's alternating spatial-temporal processing.
    
    Output shape: (B, T_patches, num_groups, hidden_size)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=512, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        self.num_time_patches = num_frame // patch_size
        
        # Body part embedders (grouped joints -> 1 token per group per time patch)
        self.part_projs = nn.ModuleList()
        for group in self.joint_groups:
            dim = in_channels * patch_size * len(group)
            self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
        
        # Fixed Positional Embeddings (sinusoidal)
        # Temporal PE: for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: for each body part group
        body_pe = self._create_sinusoidal_pe(self.num_groups, hidden_size)
        self.register_buffer('body_pe', body_pe)  # (1, num_groups, hidden_size)
    
    def _create_sinusoidal_pe(self, num_positions, dim):
        """Create fixed sinusoidal positional embeddings."""
        pe = torch.zeros(1, num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            tokens: (B, T_patches, num_groups, hidden_size) - NOT flattened
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # Convert to (B, C, T, J) for grouped processing
        x_perm = x.permute(0, 3, 1, 2)  # (B, C, T, J)
        
        part_tokens = []
        for i, group in enumerate(self.joint_groups):
            # Select joints: (B, C, T, group_size)
            x_g = x_perm[:, :, :, group]
            
            # Reshape: (B, C, T_patches, P_t, group_size)
            x_g = x_g.reshape(B, C, T_patches, P_t, len(group))
            
            # Permute to (B, T_patches, C, P_t, group_size)
            x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
            
            # Flatten to (B, T_patches, C * P_t * group_size)
            x_g = x_g.view(B, T_patches, -1)                            
            
            # Project
            embed = self.part_projs[i](x_g)  # (B, T_patches, hidden_size)
            part_tokens.append(embed)
        
        # Stack part tokens: (B, T_patches, num_groups, hidden_size)
        part_tokens = torch.stack(part_tokens, dim=2)
        
        # Add positional embeddings
        # Temporal PE: broadcast across groups
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        # Body PE: broadcast across time patches
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_groups, D)
        
        part_tokens = part_tokens + temporal_pe_expanded + body_pe_expanded
        
        return part_tokens


class HybridJointSemanticEmbedder(nn.Module):
    """
    Hybrid Embedder combining joint-level and semantic-level processing.
    
    This embedder processes the same input in two parallel pathways:
    1. Joint Groups: Per-joint tokens using 2D convolution (kernel size: p×1)
       - Preserves original joint dimension (17 joints)
       - Treats input as "pose image" with spatial (J) and temporal (T) dimensions
       - Output: one token per joint per time patch
    
    2. Semantic Groups: Body-part tokens following HybridMixSTEEmbedder (OPTIONAL)
       - Groups joints by semantic body parts  
       - Output: one token per group per time patch
       - Enabled only if joint_groups is provided
    
    Output: Concatenated along spatial dimension (joint tokens first, then semantic)
    Shape: (B, T_patches, J + num_groups, hidden_size) if joint_groups provided
           (B, T_patches, J, hidden_size) if joint_groups is None
    """
    
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=512, joint_groups=None, bias=True):
        """
        Args:
            num_frame: Number of input frames
            num_joints: Number of joints (17 for H36M)
            patch_size: Temporal patch size (p)
            in_channels: Input channels (2 for 2D poses)
            hidden_size: Output embedding dimension
            joint_groups: Optional list of joint index lists for semantic grouping
                         If None, only joint groups are used (no semantic)
            bias: Whether to use bias in conv and linear projections
        """
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups) if joint_groups is not None else 0
        
        self.num_time_patches = num_frame // patch_size
        
        # ===== Joint Groups Module (2D Conv or Linear) =====
        self.use_linear_joint = (patch_size == 1)
        if self.use_linear_joint:
            # Use Linear (equivalent to 1x1 conv) for efficiency and to avoid DDP stride warnings
            self.joint_conv = nn.Linear(in_channels, hidden_size, bias=bias)
        else:
            # 2D conv with kernel (p, 1): p along temporal (T), 1 along spatial (J)
            # Conv2d operates on (B, C, height, width) where height=T, width=J
            # Input: (B, C, T, J) -> Output: (B, hidden_size, T//p, J)
            self.joint_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,  # Directly output to hidden_size
                kernel_size=(patch_size, 1),  # (temporal=p, spatial=1)
                stride=(patch_size, 1),        # No overlap
                padding=0,
                bias=bias
            )
        
        # Positional embeddings
        # Temporal PE: shared by both joint and semantic groups (same temporal division)
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, T_patches, hidden_size)
        
        # Joint Identity PE: for each joint
        joint_spatial_pe = self._create_sinusoidal_pe(num_joints, hidden_size)
        self.register_buffer('joint_spatial_pe', joint_spatial_pe)  # (1, num_joints, hidden_size)
        
        # ===== Semantic Groups Module (Optional) =====
        if self.joint_groups is not None:
            # Body part embedders (grouped joints -> 1 token per group per time patch)
            self.part_projs = nn.ModuleList()
            for group in self.joint_groups:
                dim = in_channels * patch_size * len(group)
                self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
            
            # Body Identity PE: for each body part group
            body_pe = self._create_sinusoidal_pe(self.num_groups, hidden_size)
            self.register_buffer('body_pe', body_pe)  # (1, num_groups, hidden_size)
    
    def _create_sinusoidal_pe(self, num_positions, dim):
        """Create fixed sinusoidal positional embeddings."""
        pe = torch.zeros(1, num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _process_joint_groups(self, x):
        """
        Process input through joint groups pathway.
        
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            joint_tokens: (B, T_patches, J, hidden_size)
        """
        if self.use_linear_joint:
            # Linear projection: (B, T, J, C) -> (B, T, J, hidden_size)
            # T_patches == T when patch_size == 1
            joint_tokens = self.joint_conv(x) 
        else:
            B, T, J, C = x.shape
            
            # Convert to (B, C, T, J) for 2D conv (treating as "pose image")
            x_conv = x.permute(0, 3, 1, 2)  # (B, C, T, J)
            
            # Apply 2D conv: (B, C, T, J) -> (B, hidden_size, T//p, J)
            conv_out = self.joint_conv(x_conv)  # (B, hidden_size, T_patches, J)
            
            # Rearrange to (B, T_patches, J, hidden_size)
            joint_tokens = conv_out.permute(0, 2, 3, 1)  # (B, T_patches, J, hidden_size)
        
        # Add positional embeddings
        # Temporal PE: broadcast across joints (dim 2)
        temporal_pe = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, hidden_size)
        # Joint PE: broadcast across time patches (dim 1)
        spatial_pe = self.joint_spatial_pe.unsqueeze(1)  # (1, 1, J, hidden_size)
        
        joint_tokens = joint_tokens + temporal_pe + spatial_pe
        
        return joint_tokens
    
    def _process_semantic_groups(self, x):
        """
        Process input through semantic groups pathway (following HybridMixSTEEmbedder).
        
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            semantic_tokens: (B, T_patches, num_groups, hidden_size)
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # Convert to (B, C, T, J) for grouped processing
        x_perm = x.permute(0, 3, 1, 2)  # (B, C, T, J)
        
        part_tokens = []
        for i, group in enumerate(self.joint_groups):
            # Select joints: (B, C, T, group_size)
            x_g = x_perm[:, :, :, group]
            
            # Reshape: (B, C, T_patches, P_t, group_size)
            x_g = x_g.reshape(B, C, T_patches, P_t, len(group))
            
            # Permute to (B, T_patches, C, P_t, group_size)
            x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
            
            # Flatten to (B, T_patches, C * P_t * group_size)
            x_g = x_g.view(B, T_patches, -1)
            
            # Project
            embed = self.part_projs[i](x_g)  # (B, T_patches, hidden_size)
            part_tokens.append(embed)
        
        # Stack part tokens: (B, T_patches, num_groups, hidden_size)
        part_tokens = torch.stack(part_tokens, dim=2)
        
        # Add positional embeddings
        # Temporal PE: broadcast across groups
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        # Body PE: broadcast across time patches
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_groups, D)
        
        part_tokens = part_tokens + temporal_pe_expanded + body_pe_expanded
        
        return part_tokens
    
    def forward(self, x):
        """
        Forward pass combining joint and semantic groups.
        
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            tokens: (B, T_patches, J + num_groups, hidden_size) if joint_groups is not None
                   (B, T_patches, J, hidden_size) if joint_groups is None
        """
        # Process joint groups (always)
        joint_tokens = self._process_joint_groups(x)  # (B, T_patches, J, hidden_size)
        
        # Process semantic groups (conditional on joint_groups)
        if self.joint_groups is not None:
            semantic_tokens = self._process_semantic_groups(x)  # (B, T_patches, num_groups, hidden_size)
            
            # Concatenate along spatial dimension: joint tokens first, then semantic
            tokens = torch.cat([joint_tokens, semantic_tokens], dim=2)  # (B, T_patches, J + num_groups, hidden_size)
        else:
            tokens = joint_tokens  # (B, T_patches, J, hidden_size)
        
        return tokens


class HybridMixSTEDecoder(nn.Module):
    """
    Decoder for HybridMixSTE that maps body-part tokens back to joint predictions.
    
    Supports two modes:
    - "overlap_average" (default): Each group predicts all its joints, overlapping 
      predictions are averaged. More expressive but has redundancy.
    - "group_only": Each group only predicts its own joints. No overlap, simpler.
    
    Input: (B, T_patches, num_groups, hidden_size)
    Output: (B, T, J, out_channels)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=512,
                 out_channels=3, joint_groups=None, decoder_mode="overlap_average"):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.decoder_mode = decoder_mode
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        self.num_time_patches = num_frame // patch_size
        
        # Part decoders: each predicts its group joints for patch_size frames
        self.part_heads = nn.ModuleList()
        for group in self.joint_groups:
            out_dim = patch_size * len(group) * out_channels
            self.part_heads.append(nn.Linear(hidden_size, out_dim))
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, num_groups, hidden_size) - body part tokens
            
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B = tokens.shape[0]
        T_patches = self.num_time_patches
        P_t = self.patch_size
        T = T_patches * P_t
        J = self.num_joints
        C_out = self.out_channels
        
        if self.decoder_mode == "overlap_average":
            return self._decode_overlap_average(tokens, B, T_patches, P_t, T, J, C_out)
        else:  # "group_only"
            return self._decode_group_only(tokens, B, T_patches, P_t, T, J, C_out)
    
    def _decode_overlap_average(self, tokens, B, T_patches, P_t, T, J, C_out):
        """Decode with overlapping predictions averaged."""
        # Initialize output accumulator and count for averaging overlaps
        output = torch.zeros(B, T, J, C_out, device=tokens.device, dtype=tokens.dtype)
        count = torch.zeros(B, T, J, 1, device=tokens.device, dtype=tokens.dtype)
        
        # Decode part tokens
        for i, group in enumerate(self.joint_groups):
            part_tokens = tokens[:, :, i, :]  # (B, T_patches, hidden_size)
            part_pred = self.part_heads[i](part_tokens)  # (B, T_patches, P_t * group_size * C_out)
            
            group_size = len(group)
            part_pred = part_pred.view(B, T_patches, P_t, group_size, C_out)
            part_pred = part_pred.view(B, T, group_size, C_out)
            
            # Add to corresponding joint positions (handles overlaps)
            for j, joint_idx in enumerate(group):
                output[:, :, joint_idx, :] = output[:, :, joint_idx, :] + part_pred[:, :, j, :]
                count[:, :, joint_idx, :] = count[:, :, joint_idx, :] + 1
        
        # Average overlapping predictions
        output = output / count.clamp(min=1)
        
        return output
    
    def _decode_group_only(self, tokens, B, T_patches, P_t, T, J, C_out):
        """Decode with each group only predicting its own joints (no overlap)."""
        output = torch.zeros(B, T, J, C_out, device=tokens.device, dtype=tokens.dtype)
        
        for i, group in enumerate(self.joint_groups):
            part_tokens = tokens[:, :, i, :]  # (B, T_patches, hidden_size)
            part_pred = self.part_heads[i](part_tokens)  # (B, T_patches, P_t * group_size * C_out)
            
            group_size = len(group)
            part_pred = part_pred.view(B, T_patches, P_t, group_size, C_out)
            part_pred = part_pred.view(B, T, group_size, C_out)
            
            # Directly assign to joint positions
            for j, joint_idx in enumerate(group):
                output[:, :, joint_idx, :] = part_pred[:, :, j, :]
        
        return output


class SimpleJointDecoder(nn.Module):
    """
    Simple decoder for per-joint tokens with temporal upsampling.
    
    Takes tokens from temporal patches and upsamples to full temporal resolution.
    Input: (B, T_patches, J, hidden_size)
    Output: (B, T, J, out_channels)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=512, out_channels=3):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        
        self.num_time_patches = num_frame // patch_size
        
        # Simple head: predict patch_size * out_channels per joint
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * out_channels)
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, J, hidden_size) - per-joint tokens
            
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B, T_patches, J, _ = tokens.shape
        P_t = self.patch_size
        C_out = self.out_channels
        
        # Apply head to each joint token
        output = self.head(tokens)  # (B, T_patches, J, P_t * C_out)
        
        # Reshape to (B, T_patches, J, P_t, C_out)
        output = output.view(B, T_patches, J, P_t, C_out)
        
        # Reshape to (B, T_patches * P_t, J, C_out) = (B, T, J, C_out)
        output = output.view(B, T_patches * P_t, J, C_out)
        
        return output


class DualGroupDecoder(nn.Module):
    """
    Dual-Group Decoder that combines predictions from both joint and semantic tokens.
    
    This decoder allows both joint tokens and semantic group tokens to predict joint 
    locations, then combines them via weighted averaging:
    
    - Joint tokens (17) → Each predicts all joints via temporal upsampling
    - Semantic tokens (5) → Each predicts only its own joints
    - For joints with multiple semantic predictions: Average within semantics first
    - Final prediction: weighted_mean(joint_pred, semantic_avg)
    
    Input: (B, T_patches, J + num_groups, hidden_size)
           - First J tokens: joint tokens
           - Remaining tokens: semantic group tokens
    Output: (B, T, J, out_channels) - final 3D joint predictions
    
    Args:
        num_frame: Total number of frames (T)
        num_joints: Number of joints (17 for H36M)
        patch_size: Temporal patch size
        hidden_size: Hidden dimension from transformer
        out_channels: Output channels (3 for 3D positions)
        joint_groups: List of joint index lists for each semantic group
        joint_weight: Weight for joint predictions (default: 0.5)
        semantic_weight: Weight for semantic predictions (default: 0.5)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=512, 
                 out_channels=3, joint_groups=None, joint_weight=0.5, semantic_weight=0.5):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.num_time_patches = num_frame // patch_size
        
        # Default H36M joint groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        # Weights for averaging
        self.joint_weight = joint_weight
        self.semantic_weight = semantic_weight
        assert abs(joint_weight + semantic_weight - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {joint_weight} + {semantic_weight} = {joint_weight + semantic_weight}"
        
        # Joint prediction head: predicts all joints
        self.joint_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size*17, patch_size * out_channels)
        )
        
        # Semantic prediction heads: each predicts only its own joints
        self.semantic_heads = nn.ModuleList()
        for group in joint_groups:
            group_size = len(group)
            self.semantic_heads.append(nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size*group_size, patch_size * group_size * out_channels)
            ))
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, J + num_groups, hidden_size)
                    First J tokens: joint tokens
                    Remaining tokens: semantic group tokens
        
        Returns:
            poses_3d: (B, T, J, out_channels) - combined predictions
        """
        B, T_patches, P, _ = tokens.shape
        P_t = self.patch_size
        C_out = self.out_channels
        J = self.num_joints
        T = self.num_frame
        
        assert P == J + self.num_groups, \
            f"Expected {J + self.num_groups} tokens, got {P}"
        
        # Split tokens
        joint_tokens = tokens[:, :, :J, :]  # (B, T_patches, 17, hidden_size)
        semantic_tokens = tokens[:, :, J:, :]  # (B, T_patches, num_groups, hidden_size)
        
        # === Step 1: Joint Predictions (all 17 joints) ===
        joint_preds = self.joint_head(joint_tokens)  # (B, T_patches, 17, P_t * C_out)
        joint_preds = joint_preds.view(B, T_patches, J, P_t, C_out)
        joint_preds = joint_preds.view(B, T, J, C_out)  # (B, T, 17, 3)
        
        # === Step 2: Semantic Predictions (only own joints) ===
        # We'll accumulate predictions and counts for each joint
        semantic_sum = torch.zeros_like(joint_preds)  # (B, T, 17, 3)
        semantic_count = torch.zeros(B, T, J, 1, device=tokens.device)  # (B, T, 17, 1)
        
        for i, (group, head) in enumerate(zip(self.joint_groups, self.semantic_heads)):
            group_token = semantic_tokens[:, :, i:i+1, :]  # (B, T_patches, 1, hidden_size)
            group_size = len(group)
            
            # Predict for this group's joints
            group_pred = head(group_token)  # (B, T_patches, 1, P_t * group_size * C_out)
            group_pred = group_pred.view(B, T_patches, P_t, group_size, C_out)
            group_pred = group_pred.view(B, T, group_size, C_out)  # (B, T, group_size, 3)
            
            # Accumulate predictions for each joint in this group
            for j, joint_idx in enumerate(group):
                semantic_sum[:, :, joint_idx, :] += group_pred[:, :, j, :]
                semantic_count[:, :, joint_idx, :] += 1
        
        # Average semantic predictions (handle overlaps within semantic groups)
        # Avoid division by zero (though shouldn't happen if all joints are covered)
        semantic_count = torch.clamp(semantic_count, min=1)
        semantic_avg = semantic_sum / semantic_count  # (B, T, 17, 3)
        
        # === Step 3: Weighted Combination ===
        final_pred = self.joint_weight * joint_preds + self.semantic_weight * semantic_avg
        
        return final_pred


class CrossAttentionDecoder(nn.Module):
    """
    Simplified decoder using Multi-Head Cross-Attention (MHCA).
    
    Joint tokens attend to semantic tokens to gather contextual information,
    then predict 3D poses via a simple prediction head.
    
    Architecture:
    1. Split tokens into joint and semantic
    2. Cross-attention: joints (Q) attend to semantics (K,V)
    3. Prediction head: LayerNorm + Linear → 3D poses
    4. Temporal upsampling
    
    Input: (B, T_patches, J + num_groups, hidden_size)
    Output: (B, T, J, out_channels)
    
    Args:
        num_frame: Total number of frames (T)
        num_joints: Number of joints (17 for H36M)
        patch_size: Temporal patch size
        hidden_size: Hidden dimension from transformer
        out_channels: Output channels (3 for 3D positions)
        num_groups: Number of semantic groups (default: 5)
        num_heads: Number of attention heads for cross-attention
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=512,
                 out_channels=3, num_groups=5, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.num_time_patches = num_frame // patch_size
        self.token_norm = nn.LayerNorm(hidden_size)
        
        # Cross-attention: joints attend to semantics
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Simple prediction head for each joint
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_size * out_channels)
        )
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, J + num_groups, hidden_size)
                    First J tokens: joint tokens
                    Remaining tokens: semantic group tokens
        
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B, T_patches, P, C = tokens.shape
        J = self.num_joints
        P_t = self.patch_size
        C_out = self.out_channels
        T = self.num_frame
        
        assert P == J + self.num_groups, \
            f"Expected {J + self.num_groups} tokens, got {P}"
        
        # Split tokens
        tokens = self.token_norm(tokens)
        joint_tokens = tokens[:, :, :J, :]  # (B, T_patches, 17, C)
        semantic_tokens = tokens[:, :, J:, :]  # (B, T_patches, num_groups, C)
        
        # Cross-attention: joints attend to semantics
        # Reshape to (B*T_patches, N, C) for cross-attention
        BT = B * T_patches
        joint_flat = joint_tokens.view(BT, J, C)
        semantic_flat = semantic_tokens.view(BT, self.num_groups, C)
        
        # Apply cross-attention
        updated_joints = self.cross_attn(joint_flat, semantic_flat) + joint_flat  # (B*T_patches, 17, C)
        
        # Reshape back
        updated_joints = updated_joints.view(B, T_patches, J, C)  # (B, T_patches, 17, C)
        
        # Predict 3D poses
        predictions = self.head(updated_joints)  # (B, T_patches, 17, P_t * C_out)
        
        # Temporal upsampling: reshape to (B, T, J, C_out)
        predictions = predictions.view(B, T_patches, J, P_t, C_out)
        predictions = predictions.view(B, T, J, C_out)
        
        return predictions


class DualGroupDecoderV2(nn.Module):
    """
    Simplified Dual-Group Decoder V2 with flattened tokens and learnable weight.
    
    Key simplifications from V1:
    - Flatten tokens before prediction
    - Both joint and semantic groups predict ALL 17 joints
    - Single learnable weight for combination (instead of fixed 0.5/0.5)
    
    Architecture:
    1. Flatten joint tokens: (B, T_patches, 17, C) → (B, T_patches, 17*C)
    2. Flatten semantic tokens: (B, T_patches, 5, C) → (B, T_patches, 5*C)
    3. Joint head: predicts all 17 joints
    4. Semantic head: predicts all 17 joints
    5. Combine: α * joint_pred + (1-α) * semantic_pred (α learnable)
    
    Input: (B, T_patches, J + num_groups, hidden_size)
    Output: (B, T, J, out_channels)
    
    Args:
        num_frame: Total number of frames
        num_joints: Number of joints (17 for H36M)
        patch_size: Temporal patch size
        hidden_size: Hidden dimension
        out_channels: Output channels (3 for 3D)
        num_groups: Number of semantic groups (default: 5)
        init_weight: Initial value for learnable weight (default: 0.5)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=512,
                 out_channels=3, num_groups=5, init_weight=0.5):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.num_time_patches = num_frame // patch_size
        
        # Flattened dimensions
        joint_flatten_dim = num_joints * hidden_size  # 17 * 512 = 8704
        semantic_flatten_dim = num_groups * hidden_size  # 5 * 512 = 2560
        
        # Output dimension: all joints, all frames in patch
        output_dim = num_joints * patch_size * out_channels  # 17 * 9 * 3 = 459
        
        # Joint prediction head (from flattened joint tokens)
        self.joint_head = nn.Sequential(
            nn.LayerNorm(joint_flatten_dim),
            nn.Linear(joint_flatten_dim, output_dim)
        )
        
        # Semantic prediction head (from flattened semantic tokens)
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(semantic_flatten_dim),
            nn.Linear(semantic_flatten_dim, output_dim)
        )
        
        # Learnable combination weight (initialized to init_weight)
        # final = alpha * joint + (1 - alpha) * semantic
        self.alpha = nn.Parameter(torch.tensor(init_weight), requires_grad=False)
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, J + num_groups, hidden_size)
        
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B, T_patches, P, C = tokens.shape
        J = self.num_joints
        P_t = self.patch_size
        C_out = self.out_channels
        T = self.num_frame
        
        assert P == J + self.num_groups, \
            f"Expected {J + self.num_groups} tokens, got {P}"
        
        # Split tokens
        joint_tokens = tokens[:, :, :J, :]  # (B, T_patches, 17, C)
        semantic_tokens = tokens[:, :, J:, :]  # (B, T_patches, num_groups, C)
        
        # === Step 1: Flatten tokens ===
        joint_flat = joint_tokens.reshape(B, T_patches, -1)  # (B, T_patches, 17*C)
        semantic_flat = semantic_tokens.reshape(B, T_patches, -1)  # (B, T_patches, 5*C)
        
        # === Step 2: Predict from flattened tokens ===
        joint_preds = self.joint_head(joint_flat)  # (B, T_patches, 17*P_t*C_out)
        semantic_preds = self.semantic_head(semantic_flat)  # (B, T_patches, 17*P_t*C_out)
        
        # === Step 3: Reshape to (B, T, J, C_out) ===
        joint_preds = joint_preds.reshape(B, T_patches, J, P_t, C_out)
        joint_preds = joint_preds.reshape(B, T, J, C_out)
        
        semantic_preds = semantic_preds.reshape(B, T_patches, J, P_t, C_out)
        semantic_preds = semantic_preds.reshape(B, T, J, C_out)
        
        # === Step 4: Learnable weighted combination ===
        # Clamp alpha to [0, 1] for stability
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha ∈ (0, 1)
        final_preds = alpha * joint_preds + (1 - alpha) * semantic_preds
        
        return final_preds


class HybridMixSTEWithJointConv(nn.Module):
    """
    Hybrid Mixed Spatio-Temporal Encoder using HybridJointSemanticEmbedder with dual-group processing.
    
    Key features:
    - Uses 2D convolution for temporal patching on per-joint basis (17 joint tokens)
    - Uses linear projection for semantic grouping (5 semantic tokens by default)
    - HybridSpatialBlock processes joint and semantic tokens with separate normalizations and MLPs
    - Regular temporal transformer blocks
    - Smaller temporal dimension (T_patches) compared to MixSTE2
    
    Args:
        num_frame (int): input frame number
        num_joints (int): joints number (17 for H36M)
        in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
        embed_dim_ratio (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        patch_size (int): temporal patch size
        joint_groups (list): list of joint index lists for semantic grouping (default: 5 body parts)
    """
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 patch_size=9, joint_groups=None):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        self.num_joints = num_joints
        self.num_time_patches = num_frame // patch_size
        
        # Default H36M 17-joint skeleton groups for semantic processing
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        # Embedder with semantic grouping
        self.embedder = HybridJointSemanticEmbedder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=embed_dim_ratio,
            joint_groups=joint_groups  # Enable semantic grouping
        )
        
        # Normalization for embedder output
        self.embedder_norm = norm_layer(embed_dim_ratio)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Hybrid Spatial Transformer blocks (dual-group processing)
        self.STEblocks = nn.ModuleList([
            HybridSpatialBlock(
                dim=embed_dim_ratio, num_joints=num_joints, num_groups=self.num_groups,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        # self.STEblocks = nn.ModuleList([
        #     HybridSpatialBlockV2(
        #         dim=embed_dim_ratio, num_joints=num_joints, num_groups=self.num_groups,
        #         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        #     for i in range(depth)])

        # Temporal Transformer blocks (attention over time patches)
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Decoder
        # self.decoder = CrossAttentionDecoder(
        #     num_frame=num_frame,
        #     num_joints=num_joints,
        #     patch_size=patch_size,
        #     hidden_size=embed_dim_ratio,
        #     out_channels=out_dim,
        #     num_groups=len(joint_groups),
        #     num_heads=4,
        #     proj_drop=drop_rate,
        #     attn_drop=attn_drop_rate
        # )
        self.decoder = DualGroupDecoderV2(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=embed_dim_ratio,
            out_channels=out_dim,
            num_groups=len(joint_groups)
        )
        # self.decoder = DualGroupDecoder(
        #     num_frame=num_frame,
        #     num_joints=num_joints,
        #     patch_size=patch_size,
        #     hidden_size=embed_dim_ratio,
        #     out_channels=out_dim,
        #     joint_groups=joint_groups,
        #     joint_weight=0.5,
        #     semantic_weight=0.5
        # )
        # self.decoder = SimpleJointDecoder(
        #     num_frame=num_frame,
        #     num_joints=num_joints,
        #     patch_size=patch_size,
        #     hidden_size=embed_dim_ratio,
        #     out_channels=out_dim
        # )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def ST_forward(self, x):
        """Alternating Spatio-Temporal forward pass."""
        assert len(x.shape) == 4, "shape should be 4"
        b, t, n, c = x.shape  # (B, T_patches, P, hidden_size) where P = 17 + num_groups
        
        # Assert spatial dimension includes both joint and semantic tokens
        expected_n = self.num_joints + self.num_groups
        assert n == expected_n, f"Spatial dimension must be {expected_n} (17 joint + {self.num_groups} semantic), got {n}"
        
        for i in range(self.block_depth):
            # Hybrid spatial attention (dual-group processing)
            # HybridSpatialBlock expects (B, T, P, C) and returns same shape
            x = self.STEblocks[i](x)  # (B, T, P, C)
            x = self.Spatial_norm(x)
            
            # Temporal attention (over time patches)
            x = rearrange(x, 'b t n c -> (b n) t c')
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) t c -> b t n c', n=n)
        
        return x

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 2D poses of shape (batch, frames, joints, 2)
            
        Returns:
            3D poses of shape (batch, frames, joints, 3)
        """
        b, f, j, c = x.shape
        
        # Embed: (B, T, J, 2) -> (B, T_patches, J + num_groups, hidden_size)
        x = self.embedder(x)
        x = self.embedder_norm(x)
        x = self.pos_drop(x)
        
        # Alternating spatial-temporal blocks
        x = self.ST_forward(x)  # -> (B, T_patches, J + num_groups, hidden_size)
        
        # Extract only joint tokens for decoding (first 17 tokens)
        # joint_tokens = x[:, :, :self.num_joints, :]  # (B, T_patches, 17, hidden_size)
        
        # Extract all tokens for decoding
        joint_tokens = x
        # Decode: (B, T_patches, 17, hidden_size) -> (B, T, 17, 3)
        x = self.decoder(joint_tokens)

        return x


class HybridMixSTE(nn.Module):
    """
    Hybrid Mixed Spatio-Temporal Encoder combining body-part grouping with 
    MixSTE's alternating spatial-temporal processing.
    
    Key differences from MixSTE2:
    - Spatial tokens are body part groups (5 by default) instead of individual joints (17)
    - Temporal dimension uses patching for efficiency
    - Uses sinusoidal positional embeddings instead of learnable
    
    Args:
        num_frame (int): input frame number
        num_joints (int): joints number
        in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
        embed_dim_ratio (int): embedding dimension ratio
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        patch_size (int): temporal patch size
        joint_groups (list): list of joint index lists for body parts
        decoder_mode (str): "overlap_average" or "group_only"
    """
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 patch_size=9, joint_groups=None, decoder_mode="overlap_average"):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        # Default joint groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        self.num_time_patches = num_frame // patch_size
        
        # Hybrid Embedder (outputs (B, T_patches, num_groups, hidden_size))
        self.embedder = HybridMixSTEEmbedder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=embed_dim_ratio,
            joint_groups=joint_groups
        )

        self.gamma = nn.Parameter(torch.ones(1, 1, self.num_groups, embed_dim_ratio))
        self.beta = nn.Parameter(torch.zeros(1, 1, self.num_groups, embed_dim_ratio))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Spatial Transformer blocks (attention over body part groups)
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # Temporal Transformer blocks (attention over time patches)
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Decoder
        self.decoder = HybridMixSTEDecoder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=embed_dim_ratio,
            out_channels=out_dim,
            joint_groups=joint_groups,
            decoder_mode=decoder_mode
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def STE_forward(self, x):
    #     """Spatial Transformer Encoder forward pass (attention over body part groups)."""
    #     b, t, n, c = x.shape  # (B, T_patches, num_groups, hidden_size)
    #     x = rearrange(x, 'b t n c -> (b t) n c')
    #     x = self.pos_drop(x)

    #     blk = self.STEblocks[0]
    #     x = blk(x)

    #     x = self.Spatial_norm(x)
    #     x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
    #     return x

    # def TTE_forward(self, x):
    #     """Temporal Transformer Encoder forward pass (attention over time patches)."""
    #     assert len(x.shape) == 3, "shape should be 3"
    #     x = self.pos_drop(x)
    #     blk = self.TTEblocks[0]
    #     x = blk(x)
    #     x = self.Temporal_norm(x)
    #     return x

    def ST_forward(self, x):
        """Alternating Spatio-Temporal forward pass."""
        assert len(x.shape) == 4, "shape should be 4"
        b, t, n, c = x.shape  # (B, T_patches, num_groups, hidden_size)
        for i in range(self.block_depth):
            x = rearrange(x, 'b t n c -> (b t) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) t c -> b t n c', n=n)
        
        return x

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 2D poses of shape (batch, frames, joints, 2)
            
        Returns:
            3D poses of shape (batch, frames, joints, 3)
        """
        b, f, j, c = x.shape
        
        # Embed: (B, T, J, 2) -> (B, T_patches, num_groups, hidden_size)
        x = self.embedder(x)
        x = F.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=1e-6)
        x = x * self.gamma + self.beta
        
        # First spatial-temporal pass
        # x = self.STE_forward(x)  # -> (B*num_groups, T_patches, hidden_size)
        # x = self.TTE_forward(x)  # -> (B*num_groups, T_patches, hidden_size)
        # x = rearrange(x, '(b n) t c -> b t n c', n=self.num_groups)
        
        # Alternating spatial-temporal blocks
        x = self.ST_forward(x)  # -> (B, T_patches, num_groups, hidden_size)
        
        # Decode: (B, T_patches, num_groups, hidden_size) -> (B, T, J, 3)
        x = self.decoder(x)

        return x


class HybridMixSTEEmbedderV2(nn.Module):
    """
    Hybrid Pose Embedder with proper spatiotemporal patchification (V2).
    
    Each (time_patch, group) combination is treated as a unique patch position
    and has its own projection layer, similar to ViT's patch embedding.
    
    Total projections: T_patches × N_groups
    
    Output shape: (B, T_patches, num_groups, hidden_size)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=512, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        self.num_time_patches = num_frame // patch_size
        
        # Create separate projection for each (time_patch, group) position
        # Organized as: patch_projs[group_idx][time_patch_idx]
        self.patch_projs = nn.ModuleList()
        for g_idx, group in enumerate(self.joint_groups):
            group_projs = nn.ModuleList()
            patch_dim = in_channels * patch_size * len(group)
            for t_idx in range(self.num_time_patches):
                group_projs.append(nn.Linear(patch_dim, hidden_size, bias=bias))
            self.patch_projs.append(group_projs)
        
        # Fixed Positional Embeddings (sinusoidal)
        # Temporal PE: for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: for each body part group
        body_pe = self._create_sinusoidal_pe(self.num_groups, hidden_size)
        self.register_buffer('body_pe', body_pe)  # (1, num_groups, hidden_size)
    
    def _create_sinusoidal_pe(self, num_positions, dim):
        """Create fixed sinusoidal positional embeddings."""
        pe = torch.zeros(1, num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            tokens: (B, T_patches, num_groups, hidden_size) - NOT flattened
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # Convert to (B, C, T, J) for grouped processing
        x_perm = x.permute(0, 3, 1, 2)  # (B, C, T, J)
        
        # Process each group and time patch with unique projection
        all_tokens = []
        for g_idx, group in enumerate(self.joint_groups):
            # Select joints: (B, C, T, group_size)
            x_g = x_perm[:, :, :, group]
            
            # Reshape: (B, C, T_patches, P_t, group_size)
            x_g = x_g.reshape(B, C, T_patches, P_t, len(group))
            
            # Permute to (B, T_patches, C, P_t, group_size)
            x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
            
            # Flatten spatial dims: (B, T_patches, C * P_t * group_size)
            x_g = x_g.view(B, T_patches, -1)
            
            # Apply unique projection for each time patch
            group_tokens = []
            for t_idx in range(T_patches):
                # Extract patch at time position t_idx: (B, patch_dim)
                patch = x_g[:, t_idx, :]
                # Project with position-specific projection
                token = self.patch_projs[g_idx][t_idx](patch)  # (B, hidden_size)
                group_tokens.append(token)
            
            # Stack: (B, T_patches, hidden_size)
            group_tokens = torch.stack(group_tokens, dim=1)
            all_tokens.append(group_tokens)
        
        # Stack all groups: (B, T_patches, num_groups, hidden_size)
        part_tokens = torch.stack(all_tokens, dim=2)
        
        # Add positional embeddings
        # Temporal PE: broadcast across groups
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        # Body PE: broadcast across time patches
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_groups, D)
        
        part_tokens = part_tokens + temporal_pe_expanded + body_pe_expanded
        
        return part_tokens


class HybridMixSTEV2(nn.Module):
    """
    Hybrid Mixed Spatio-Temporal Encoder V2 with proper per-patch patchification.
    
    Key differences from HybridMixSTE:
    - Uses HybridMixSTEEmbedderV2 with unique projection per (time_patch, group)
    - Simplified unified forward loop for all depth blocks
    
    Args:
        num_frame (int): input frame number
        num_joints (int): joints number
        in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
        embed_dim_ratio (int): embedding dimension ratio
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        patch_size (int): temporal patch size
        joint_groups (list): list of joint index lists for body parts
        decoder_mode (str): "overlap_average" or "group_only"
    """
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 patch_size=9, joint_groups=None, decoder_mode="overlap_average"):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3

        # Default joint groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],     # Left Arm
                [14, 15, 16]      # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        self.num_time_patches = num_frame // patch_size
        
        # V2 Embedder with proper per-patch projections
        self.embedder = HybridMixSTEEmbedderV2(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_chans,
            hidden_size=embed_dim_ratio,
            joint_groups=joint_groups
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        # Spatial Transformer blocks (attention over body part groups)
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # Temporal Transformer blocks (attention over time patches)
        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False)
            for i in range(depth)])

        self.Embedder_norm = norm_layer(embed_dim_ratio)
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        # Decoder
        self.decoder = HybridMixSTEDecoder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=embed_dim_ratio,
            out_channels=out_dim,
            joint_groups=joint_groups,
            decoder_mode=decoder_mode
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 2D poses of shape (batch, frames, joints, 2)
            
        Returns:
            3D poses of shape (batch, frames, joints, 3)
        """
        b, f, j, c = x.shape
        
        # Embed: (B, T, J, 2) -> (B, T_patches, num_groups, hidden_size)
        # V2 embedder handles proper patchification with unique projections per (t, g)
        x = self.embedder(x)
        x = self.pos_drop(x)
        x = self.Embedder_norm(x)
        
        t = self.num_time_patches
        n = self.num_groups
        
        # Alternating spatial-temporal blocks (all depth blocks in unified loop)
        for i in range(self.block_depth):
            # Spatial attention (over body part groups)
            x = rearrange(x, 'b t n c -> (b t) n c')
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            
            # Temporal attention (over time patches)
            x = rearrange(x, '(b t) n c -> (b n) t c', t=t)
            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) t c -> b t n c', n=n)
        
        # Decode: (B, T_patches, num_groups, hidden_size) -> (B, T, J, 3)
        x = self.decoder(x)

        return x


class BiomechMixSTE(MixSTE2):
    """
    MixSTE2 with biomechanical constraint support.
    
    This model has the same architecture as MixSTE2 but stores skeleton information
    (parent hierarchy, joint symmetry, angle limits) needed for computing
    biomechanical losses during training.
    
    The biomechanical losses are not computed inside the model - they are computed
    in the training script using the poses predicted by this model.
    
    Args:
        Same as MixSTE2, plus:
        skeleton_parents: List of parent joint indices (default: H36M 17-joint)
        left_joints: List of left-side joint indices for symmetry
        right_joints: List of right-side joint indices for symmetry
        angle_limits: Dict mapping joint index to (min_angle, max_angle) in degrees
    """
    
    # Default H36M 17-joint skeleton parents
    DEFAULT_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    
    # Left/Right joint pairs for symmetry
    DEFAULT_LEFT_JOINTS = [4, 5, 6, 11, 12, 13]   # Left leg + Left arm
    DEFAULT_RIGHT_JOINTS = [1, 2, 3, 14, 15, 16]  # Right leg + Right arm
    
    # Default joint angle limits in degrees [min, max]
    # Convention: 180° = straight limb, smaller = more bent/flexed
    DEFAULT_ANGLE_LIMITS = {
        # Knees: prevent over-flexion (30° min allows deep squat)
        2: (30.0, 180.0),   # Right knee
        5: (30.0, 180.0),   # Left knee
        # Elbows: prevent over-flexion
        12: (30.0, 180.0),  # Left elbow
        15: (30.0, 180.0),  # Right elbow
        # Hips: wide range of motion
        1: (30.0, 180.0),   # Right hip
        4: (30.0, 180.0),   # Left hip
        # Shoulders: very flexible
        11: (20.0, 180.0),  # Left shoulder
        14: (20.0, 180.0),  # Right shoulder
        # Spine joints: limited flexion
        7: (140.0, 180.0),  # Lower spine
        8: (120.0, 180.0),  # Upper spine/thorax
        9: (120.0, 180.0),  # Neck
    }
    
    def __init__(self, num_frame=243, num_joints=17, in_chans=2, embed_dim_ratio=512, depth=8,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 skeleton_parents=None, left_joints=None, right_joints=None, angle_limits=None):
        # Initialize parent class (MixSTE2)
        super().__init__(
            num_frame=num_frame,
            num_joints=num_joints,
            in_chans=in_chans,
            embed_dim_ratio=embed_dim_ratio,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
        )
        
        # Store biomechanical parameters
        self.skeleton_parents = skeleton_parents if skeleton_parents is not None else self.DEFAULT_PARENTS
        self.left_joints = left_joints if left_joints is not None else self.DEFAULT_LEFT_JOINTS
        self.right_joints = right_joints if right_joints is not None else self.DEFAULT_RIGHT_JOINTS
        self.angle_limits = angle_limits if angle_limits is not None else self.DEFAULT_ANGLE_LIMITS
    
    def get_skeleton_info(self):
        """Return skeleton information for biomechanical loss computation."""
        return {
            'parents': self.skeleton_parents,
            'left_joints': self.left_joints,
            'right_joints': self.right_joints,
            'angle_limits': self.angle_limits
        }
