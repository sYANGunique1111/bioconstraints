"""
Efficiency Models for 3D Human Pose Estimation.

This module contains efficient model architectures for video-based 2D-to-3D pose estimation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

import importlib.util
import os

# Direct import of merge module to avoid tome/__init__.py which has broken dependencies
_merge_path = os.path.join(os.path.dirname(__file__), 'tome', 'merge.py')
_spec = importlib.util.spec_from_file_location('tome_merge', _merge_path)
_merge_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_merge_module)
bipartite_soft_matching = _merge_module.bipartite_soft_matching


def downsample_pose_frames(pose_frames: torch.Tensor, r: int):
    """
    Downsamples the temporal dimension of a pose tensor using Bipartite Matching.

    Args:
        pose_frames: Tensor of shape (B, T, J, C)
        r: Number of frames to reduce (remove).

    Returns:
        Tensor of shape (B, T - r, J, C)
    """
    B, T, J, C = pose_frames.shape

    # 1. Flatten J and C to create a 'metric' tensor for similarity calculation.
    #    Shape becomes: (B, T, J*C)
    #    Here, each timestep T is treated as a token, and the entire pose (J*C) is its feature.
    metric = pose_frames.view(B, T, -1)

    # 2. Compute the Bipartite Matching
    #    This calculates the cosine similarity between frame groups and decides which to merge.
    #    We pass class_token=False because pose sequences usually don't have a CLS token.
    merge_fn, _ = bipartite_soft_matching(
        metric=metric, 
        r=r, 
        class_token=False, 
        distill_token=False
    )

    # 3. Apply the merge operation
    #    The merge_fn handles the averaging of matched tokens (frames).
    #    Output shape: (B, T_new, J*C)
    merged_flat = merge_fn(metric, mode="mean")

    # 4. Reshape back to the original pose structure (B, T_new, J, C)
    #    We rely on the inferred dimension (-1) for the new time length.
    pose_downsampled = merged_flat.view(B, -1, J, C)

    return pose_downsampled


# =============================================================================
# Basic Building Blocks
# =============================================================================

class Mlp(nn.Module):
    """MLP block with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
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
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# TwoStageGroupedPoseModel Components
# =============================================================================

class JointGrouping(nn.Module):
    """
    Joint Grouping module that groups joints by body parts and projects to hidden dimension.
    
    Groups joints according to predefined body part groupings, pads to D_max for efficient
    tensor operations, and projects using per-group independent linear projections.
    
    The projection is: (B, T, G, D) × (1, 1, G, D, C) -> (B, T, G, C)
    Each group g has its own independent projection matrix of shape (D, C).
    
    Input: (B, T, J, 2)
    Output: (B, T, G, C) where G is number of groups
    """
    def __init__(self, num_joints=17, in_channels=2, hidden_size=256, 
                 joint_groups=None, bias=True):
        super().__init__()
        self.num_joints = num_joints
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
        
        # Calculate D_max (max joints per group * in_channels)
        self.max_group_size = max(len(group) for group in joint_groups)
        self.d_max = self.max_group_size * in_channels
        
        # Per-group independent projections: (1, 1, G, D_max, C)
        # Each group has its own projection matrix
        self.proj_weight = nn.Parameter(
            torch.empty(1, 1, self.num_groups, self.d_max, hidden_size)
        )
        if bias:
            self.proj_bias = nn.Parameter(torch.zeros(1, 1, self.num_groups, hidden_size))
        else:
            self.register_parameter('proj_bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.proj_weight, a=math.sqrt(5))
        if self.proj_bias is not None:
            fan_in = self.d_max
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.proj_bias, -bound, bound)
        
        # Create index tensors for gathering joints (register as buffers for device transfer)
        # Shape: (num_groups, max_group_size) - padded with -1 for invalid indices
        group_indices = torch.full((self.num_groups, self.max_group_size), -1, dtype=torch.long)
        group_masks = torch.zeros(self.num_groups, self.max_group_size, dtype=torch.bool)
        
        for g_idx, group in enumerate(joint_groups):
            for j_idx, joint in enumerate(group):
                group_indices[g_idx, j_idx] = joint
                group_masks[g_idx, j_idx] = True
        
        self.register_buffer('group_indices', group_indices)
        self.register_buffer('group_masks', group_masks)
        
        # Fixed spatial positional embeddings for groups
        spatial_pe = self._create_sinusoidal_pe(self.num_groups, hidden_size)
        self.register_buffer('spatial_pe', spatial_pe)  # (1, G, hidden_size)
    
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
            x: (B, T, J, 2) - 2D poses
            
        Returns:
            grouped: (B, T, G, C) - grouped and projected tokens with spatial PE
        """
        B, T, J, C = x.shape
        
        # Create padded tensor: (B, T, G, max_group_size, C)
        # First, create zero-padded output
        grouped_raw = torch.zeros(B, T, self.num_groups, self.max_group_size, C, 
                                  device=x.device, dtype=x.dtype)
        
        # Gather joints for each group
        for g_idx, group in enumerate(self.joint_groups):
            # Select joints for this group: (B, T, group_size, C)
            group_joints = x[:, :, group, :]
            # Place in padded tensor
            grouped_raw[:, :, g_idx, :len(group), :] = group_joints
        
        # Flatten last two dims: (B, T, G, D_max)
        grouped_flat = grouped_raw.view(B, T, self.num_groups, self.d_max)
        
        # Per-group projection: (B, T, G, D) × (1, 1, G, D, C) -> (B, T, G, C)
        # Using einsum for efficient batched matrix multiplication per group
        # grouped_flat: (B, T, G, D), proj_weight: (1, 1, G, D, C)
        grouped = torch.einsum('btgd,xyGdc->btgc', grouped_flat, self.proj_weight)
        if self.proj_bias is not None:
            grouped = grouped + self.proj_bias
        
        # Add spatial positional embeddings: (1, 1, G, C) broadcast
        grouped = grouped + self.spatial_pe.unsqueeze(1)
        
        return grouped


class SpatialUpSample(nn.Module):
    """
    Upsample from group tokens to joint tokens.
    
    Input: (B, T, G, C)
    Output: (B, T, J, C)
    
    Performs: reshape to (B,T,C,G) -> Linear(G, J) -> reshape to (B,T,J,C)
    """
    def __init__(self, num_groups, num_joints, hidden_size):
        super().__init__()
        self.num_groups = num_groups
        self.num_joints = num_joints
        self.hidden_size = hidden_size
        
        # Linear projection from G to J (applied on last dim after transpose)
        self.proj = nn.Linear(num_groups, num_joints, bias=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, G, C)
            
        Returns:
            out: (B, T, J, C)
        """
        # (B, T, G, C) -> (B, T, C, G)
        x = x.permute(0, 1, 3, 2)
        
        # Linear: (B, T, C, G) -> (B, T, C, J)
        x = self.proj(x)
        
        # (B, T, C, J) -> (B, T, J, C)
        x = x.permute(0, 1, 3, 2)
        
        return x


class TwoStageGroupedPoseModel(nn.Module):
    """
    Two-Stage Grouped Pose Model for efficient 2D-to-3D pose estimation.
    
    Architecture:
    1. Joint Grouping: Groups joints by body parts, pads to D_max, shared projection
    2. Stage 1 Transformer (×N): Spatial-only attention on group tokens
    3. Spatial UpSample: Project from groups back to joints
    4. Temporal Reduction: Reduce temporal dimension using bipartite matching
    5. Stage 2 Transformer (×N): Alternating spatial-temporal attention
    6. Temporal UpSample: Restore original temporal dimension using unmerge
    7. Regression Head: Linear projection to 3D coordinates
    
    Args:
        num_frame: Number of input frames (T)
        num_joints: Number of joints (J), default 17 for H36M
        in_channels: Input channels (2 for 2D poses)
        out_channels: Output channels (3 for 3D poses)
        hidden_size: Hidden dimension (C)
        depth: Number of transformer blocks per stage (N)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        temporal_reduction_ratio: Fraction of frames to remove (default 0.5)
        joint_groups: List of joint index lists for grouping
    """
    def __init__(
        self, 
        num_frame=243, 
        num_joints=17, 
        in_channels=2, 
        out_channels=3,
        hidden_size=256, 
        depth=3, 
        num_heads=8, 
        mlp_ratio=4.,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.2,
        temporal_reduction_ratio=0.5,
        joint_groups=None,
        norm_layer=None
    ):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.temporal_reduction_ratio = temporal_reduction_ratio
        
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
        
        # Calculate reduced temporal dimension
        self.r = int(num_frame * temporal_reduction_ratio)  # frames to remove
        self.reduced_frames = num_frame - self.r
        
        # =====================================================================
        # Components
        # =====================================================================
        
        # Joint Grouping: (B, T, J, 2) -> (B, T, G, C)
        self.joint_grouping = JointGrouping(
            num_joints=num_joints,
            in_channels=in_channels,
            hidden_size=hidden_size,
            joint_groups=joint_groups,
            bias=True
        )
        
        # Stage 1: Spatial-only transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth * 2)]
        self.stage1_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=True, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[i], 
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.stage1_norm = norm_layer(hidden_size)
        
        # Spatial UpSample: (B, T, G, C) -> (B, T, J, C)
        self.spatial_upsample = SpatialUpSample(
            num_groups=self.num_groups,
            num_joints=num_joints,
            hidden_size=hidden_size
        )
        
        # Learnable temporal positional embedding for Stage 2 (after reduction)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.reduced_frames, hidden_size)
        )
        
        # Stage 2: Temporal-only transformer blocks
        # After temporal reduction, we only process temporal dimension
        self.stage2_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qkv_bias=True, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth + i], 
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.stage2_norm = norm_layer(hidden_size)
        
        # Regression head
        self.head = nn.Linear(hidden_size, out_channels)
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
        trunc_normal_(self.temporal_pos_embed, std=.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _temporal_reduction(self, x):
        """
        Apply temporal reduction using bipartite soft matching.
        
        Args:
            x: (B, T, J, C)
            
        Returns:
            reduced: (B, R, J, C) where R = T - r
            unmerge_fn: function to restore original temporal dimension
        """
        B, T, J, C = x.shape
        
        # Flatten J and C for similarity computation: (B, T, J*C)
        metric = x.contiguous().view(B, T, -1)
        
        # Compute bipartite matching
        merge_fn, unmerge_fn = bipartite_soft_matching(
            metric=metric,
            r=self.r,
            class_token=False,
            distill_token=False
        )
        
        # Apply merge: (B, T, J*C) -> (B, R, J*C)
        merged = merge_fn(metric, mode="mean")
        
        # Reshape back: (B, R, J, C)
        R = merged.shape[1]
        reduced = merged.view(B, R, J, C)
        
        return reduced, unmerge_fn
    
    def _temporal_upsample(self, x, unmerge_fn):
        """
        Restore original temporal dimension using unmerge.
        
        Args:
            x: (B, R, J, C)
            unmerge_fn: function from bipartite_soft_matching
            
        Returns:
            restored: (B, T, J, C)
        """
        B, R, J, C = x.shape
        
        # Flatten: (B, R, J*C)
        x_flat = x.contiguous().view(B, R, -1)
        
        # Unmerge: (B, R, J*C) -> (B, T, J*C)
        restored_flat = unmerge_fn(x_flat)
        
        # Reshape: (B, T, J, C)
        T = restored_flat.shape[1]
        restored = restored_flat.view(B, T, J, C)
        
        return restored
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input 2D poses of shape (B, T, J, 2)
            
        Returns:
            3D poses of shape (B, T, J, 3)
        """
        B, T, J, C = x.shape
        
        # =====================================================================
        # Stage 1: Spatial-only processing on grouped tokens
        # =====================================================================
        
        # Joint Grouping: (B, T, J, 2) -> (B, T, G, C)
        x = self.joint_grouping(x)
        G = x.shape[2]
        
        # Reshape for spatial attention: (B*T, G, C)
        x = rearrange(x, 'b t g c -> (b t) g c')
        x = self.pos_drop(x)
        
        # Stage 1 transformer blocks (spatial-only)
        for blk in self.stage1_blocks:
            x = blk(x)
        x = self.stage1_norm(x)
        
        # Reshape back: (B, T, G, C)
        x = rearrange(x, '(b t) g c -> b t g c', b=B, t=T)
        
        # Spatial UpSample: (B, T, G, C) -> (B, T, J, C)
        x = self.spatial_upsample(x)
        
        # =====================================================================
        # Temporal Reduction
        # =====================================================================
        
        # (B, T, J, C) -> (B, R, J, C)
        x, unmerge_fn = self._temporal_reduction(x)
        R = x.shape[1]
        
        # Add learnable temporal positional embeddings
        # x shape: (B, R, J, C), temporal_pos_embed shape: (1, R, C)
        # Need to broadcast: (1, R, 1, C) to match (B, R, J, C)
        # Handle potential size mismatch if R differs from self.reduced_frames
        if R == self.reduced_frames:
            x = x + self.temporal_pos_embed.unsqueeze(2)  # (1, R, 1, C)
        else:
            # Interpolate temporal PE if needed
            temporal_pe = F.interpolate(
                self.temporal_pos_embed.transpose(1, 2),  # (1, C, R_expected)
                size=R,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (1, R, C)
            x = x + temporal_pe.unsqueeze(2)  # (1, R, 1, C)
        
        x = self.pos_drop(x)
        
        # =====================================================================
        # Stage 2: Temporal-only processing
        # =====================================================================
        
        # Reshape for temporal attention: (B*J, R, C)
        x = rearrange(x, 'b r j c -> (b j) r c')
        
        for blk in self.stage2_blocks:
            x = blk(x)
        x = self.stage2_norm(x)
        
        # Reshape back: (B, R, J, C)
        x = rearrange(x, '(b j) r c -> b r j c', b=B, j=J)
        
        # =====================================================================
        # Temporal UpSample and Regression
        # =====================================================================
        
        # Temporal upsample: (B, R, J, C) -> (B, T, J, C)
        x = self._temporal_upsample(x, unmerge_fn)
        
        # Regression: (B, T, J, C) -> (B, T, J, 3)
        x = self.head(x)
        
        return x


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create a dummy batch of poses: Batch=2, Time=243, Joints=17, Coords=2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test TwoStageGroupedPoseModel
    model = TwoStageGroupedPoseModel(
        num_frame=243,
        num_joints=17,
        in_channels=2,
        out_channels=3,
        hidden_size=256,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        temporal_reduction_ratio=0.5
    ).to(device)
    
    input_pose = torch.randn(2, 243, 17, 2, device=device)
    
    print(f"Input Shape: {input_pose.shape}")
    
    with torch.no_grad():
        output = model(input_pose)
    
    print(f"Output Shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")