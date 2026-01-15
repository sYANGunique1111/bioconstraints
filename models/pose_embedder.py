import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed,Attention, Mlp
from .basic_modules import CrossAttentionBlock

class GroupedJointPatchEmbed(nn.Module):
    """
    Patch Embedding that groups specific spatial joints together while patchifying time linearly.
    """
    def __init__(self, input_size, patch_size, in_channels, hidden_size, bias=True, joint_groups=None):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
            
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default configuration if none provided
        if joint_groups is None:
             joint_groups = [[1,2,3],[4,5,6],[0,7,8,9,10],[11,12,13],[14,15,16]]
        self.joint_groups = joint_groups
        
        self.num_time_patches = input_size[0] // patch_size[0]
        self.num_groups = len(joint_groups)
        self.num_patches = self.num_time_patches * self.num_groups
        
        self.projs = nn.ModuleList()
        for group in self.joint_groups:
            # Input dimension for this group: C * patch_time * group_size
            dim = in_channels * patch_size[0] * len(group)
            self.projs.append(nn.Linear(dim, hidden_size, bias=bias))

    def forward(self, x):
        """
        x: (B, C, T, J)
        """
        B, C, T, J = x.shape
        P_t = self.patch_size[0]
        
        all_tokens = []
        for i, group in enumerate(self.joint_groups):
            # Select joints: (B, C, T, group_size)
            x_g = x[:, :, :, group]
            
            # Reshape: (B, C, T_patches, P_t, group_size)
            x_g = x_g.reshape(B, C, T // P_t, P_t, len(group))
            
            # Permute to (B, T_patches, C, P_t, group_size)
            # We want to flatten C, P_t, group_size
            x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
            
            # Flatten to (B, T_patches, flattened_dim)
            x_g = x_g.view(B, T // P_t, -1)
            
            # Project
            embed = self.projs[i](x_g) # (B, T_patches, hidden_size)
            all_tokens.append(embed)
            
        # Stack: (B, T_patches, NumGroups, hidden_size)
        # We order by Time then Group to match "spatial-temporal" flattening usually scanning spatial first
        tokens = torch.stack(all_tokens, dim=2)
        
        # Flatten to (B, N, hidden_size)
        tokens = tokens.flatten(1, 2)
        return tokens


# =============================================================================
# Hybrid Pose Embedder Components
# =============================================================================

class GlobalPatchEmbed(nn.Module):
    """
    Global Patch Embedding that embeds all joints together per time patch.
    This creates a holistic representation of the entire pose per temporal window.
    """
    def __init__(self, num_frame, num_joints, patch_size, in_channels, hidden_size, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.num_time_patches = num_frame // patch_size
        
        # Input dimension: all joints * channels * temporal patch size
        input_dim = num_joints * in_channels * patch_size
        self.proj = nn.Linear(input_dim, hidden_size, bias=bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) - 2D poses
            
        Returns:
            tokens: (B, T_patches, hidden_size)
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # Reshape: (B, T, J, C) -> (B, T_patches, P_t, J, C)
        x = x.reshape(B, T_patches, P_t, J, C)
        
        # Flatten all dimensions except batch and time patches
        # (B, T_patches, P_t * J * C)
        x = x.reshape(B, T_patches, -1)
        
        # Project to hidden size
        tokens = self.proj(x)  # (B, T_patches, hidden_size)
        
        return tokens


class HybridPoseEmbedder(nn.Module):
    """
    Hybrid Pose Embedder that combines global pose tokens and body part tokens.
    
    Creates 6 tokens per time patch:
    - 1 global token (all 17 joints)
    - 5 body part tokens (right leg, left leg, torso, left arm, right arm)
    
    Each token gets positional embedding = TemporalPE + BodyIdentityPE
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=256, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],       # Right Leg
                [4, 5, 6],       # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],    # Left Arm
                [14, 15, 16]     # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        self.num_token_types = 1 + self.num_groups  # 1 global + N body parts
        
        self.num_time_patches = num_frame // patch_size
        self.num_tokens = self.num_time_patches * self.num_token_types
        
        # Global embedder (all joints -> 1 token per time patch)
        self.global_embed = GlobalPatchEmbed(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            bias=bias
        )
        
        # Body part embedders (grouped joints -> 1 token per group per time patch)
        self.part_projs = nn.ModuleList()
        for group in self.joint_groups:
            dim = in_channels * patch_size * len(group)
            self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
        
        # Fixed Positional Embeddings (non-learnable, unique identifiers)
        # Temporal PE: fixed sinusoidal encoding for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: fixed sinusoidal encoding for each token type
        body_pe = self._create_sinusoidal_pe(self.num_token_types, hidden_size)
        self.register_buffer('body_pe', body_pe)  # (1, num_token_types, hidden_size)
    
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
            tokens: (B, T_patches * num_token_types, hidden_size)
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # 1. Global tokens: (B, T_patches, D)
        global_tokens = self.global_embed(x)  # (B, T_patches, hidden_size)
        global_tokens = global_tokens.unsqueeze(2)  # (B, T_patches, 1, hidden_size)
        
        # 2. Body part tokens
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
        
        # 3. Concatenate global and part tokens
        # (B, T_patches, 1 + num_groups, hidden_size)
        all_tokens = torch.cat([global_tokens, part_tokens], dim=2)
        
        # 4. Add positional embeddings
        # Temporal PE: broadcast across token types
        # (1, T_patches, 1, D) + (B, T_patches, num_token_types, D)
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        
        # Body PE: broadcast across time patches
        # (1, 1, num_token_types, D) + (B, T_patches, num_token_types, D)
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_token_types, D)
        
        # Combined PE
        all_tokens = all_tokens + temporal_pe_expanded + body_pe_expanded
        
        # 5. Flatten to sequence: (B, T_patches * num_token_types, hidden_size)
        tokens = all_tokens.flatten(1, 2)
        
        return tokens


class HybridPoseEmbedder2(nn.Module):
    """
    Hybrid Pose Embedder that combines global pose tokens and body part tokens.
    
    Creates 6 tokens per time patch:
    - 1 global token (all 17 joints)
    - 5 body part tokens (right leg, left leg, torso, left arm, right arm)
    
    Each token gets positional embedding = TemporalPE + BodyIdentityPE
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=256, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],       # Right Leg
                [4, 5, 6],       # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],    # Left Arm
                [14, 15, 16]     # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        self.num_token_types = 1 + self.num_groups  # 1 global + N body parts
        
        self.num_time_patches = num_frame // patch_size
        self.num_tokens = self.num_time_patches * self.num_token_types
        
        # Global embedder (all joints -> 1 token per time patch)
        self.global_embed = GlobalPatchEmbed(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            bias=bias
        )
        
        # Cross Attention Block
        self.cross_attn_block = CrossAttentionBlock(
            dim=hidden_size,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=bias,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0
        )
        # Body part embedders (grouped joints -> 1 token per group per time patch)
        self.part_projs = nn.ModuleList()
        for group in self.joint_groups:
            dim = in_channels * patch_size * len(group)
            self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
        
        # Fixed Positional Embeddings (non-learnable, unique identifiers)
        # Temporal PE: fixed sinusoidal encoding for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: fixed sinusoidal encoding for each token type
        body_pe = self._create_sinusoidal_pe(self.num_token_types, hidden_size)
        self.register_buffer('body_pe', body_pe)  # (1, num_token_types, hidden_size)
    
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
            tokens: (B, T_patches * num_token_types, hidden_size)
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # 1. Global tokens: (B, T_patches, D)
        global_tokens = self.global_embed(x)  # (B, T_patches, hidden_size)
        global_tokens = global_tokens.unsqueeze(2)  # (B, T_patches, 1, hidden_size)
        
        # 2. Body part tokens
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
        
        # 3. Concatenate global and part tokens
        # (B, T_patches, 1 + num_groups, hidden_size)
        all_tokens = torch.cat([global_tokens, part_tokens], dim=2)
        
        # 4. Add positional embeddings
        # Temporal PE: broadcast across token types
        # (1, T_patches, 1, D) + (B, T_patches, num_token_types, D)
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        
        # Body PE: broadcast across time patches
        # (1, 1, num_token_types, D) + (B, T_patches, num_token_types, D)
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_token_types, D)
        
        # Combined PE
        all_tokens = all_tokens + temporal_pe_expanded + body_pe_expanded

        # 5. Split global and part tokens after adding positional embeddings
        global_tokens = all_tokens[:, :, 0:1, :].flatten(1, 2)
        part_tokens = all_tokens[:, :, 1:, :].flatten(1, 2)
        
        # Refine global tokens using cross attention
        global_tokens = self.cross_attn_block(global_tokens, part_tokens)
        
        return global_tokens


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class HybridPoseEmbedder3(nn.Module):
    """
    Hybrid Pose Embedder that uses global tokens to modulate body part tokens via AdaLN.
    
    Pipeline:
    1. Generate global tokens (all 17 joints) and body part tokens (5 groups)
    2. Add positional embeddings to body part tokens
    3. Use global tokens to generate AdaLN (shift, scale) parameters per time patch
    4. Apply AdaLN modulation to body part tokens
    5. Output only body part tokens
    
    Output shape: (B, T_patches * num_groups, hidden_size)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=256, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],       # Right Leg
                [4, 5, 6],       # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],    # Left Arm
                [14, 15, 16]     # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        self.num_time_patches = num_frame // patch_size
        self.num_tokens = self.num_time_patches * self.num_groups  # Only body parts
        
        # Global embedder (all joints -> 1 token per time patch)
        self.global_embed = GlobalPatchEmbed(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            bias=bias
        )
        
        # Body part embedders (grouped joints -> 1 token per group per time patch)
        self.part_projs = nn.ModuleList()
        for group in self.joint_groups:
            dim = in_channels * patch_size * len(group)
            self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
        
        # Fixed Positional Embeddings for body parts only
        # Temporal PE: fixed sinusoidal encoding for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: fixed sinusoidal encoding for each body part group
        body_pe = self._create_sinusoidal_pe(self.num_groups, hidden_size)
        self.register_buffer('body_pe', body_pe)  # (1, num_groups, hidden_size)
        
        # AdaLN modulation layer: global token -> (shift, scale) for body parts
        # Each global token generates modulation for all body parts at that time step
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # LayerNorm for body part tokens before modulation
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
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
            tokens: (B, T_patches * num_groups, hidden_size) - body part tokens only
        """
        B, T, J, C = x.shape
        P_t = self.patch_size
        T_patches = T // P_t
        
        # 1. Generate global tokens: (B, T_patches, D)
        global_tokens = self.global_embed(x)  # (B, T_patches, hidden_size)
        
        # 2. Generate body part tokens
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
        
        # 3. Add positional embeddings to body part tokens
        # Temporal PE: broadcast across groups
        temporal_pe_expanded = self.temporal_pe.unsqueeze(2)  # (1, T_patches, 1, D)
        # Body PE: broadcast across time patches
        body_pe_expanded = self.body_pe.unsqueeze(1)  # (1, 1, num_groups, D)
        
        part_tokens = part_tokens + temporal_pe_expanded + body_pe_expanded
        
        # 4. Apply AdaLN modulation using global tokens
        # Generate shift and scale from global tokens: (B, T_patches, 2*D)
        modulation_params = self.adaLN_modulation(global_tokens)  # (B, T_patches, 2*D)
        shift, scale = modulation_params.chunk(2, dim=-1)  # Each: (B, T_patches, D)
        
        # Expand for broadcasting across groups
        shift = shift.unsqueeze(2)  # (B, T_patches, 1, D)
        scale = scale.unsqueeze(2)  # (B, T_patches, 1, D)
        
        # Apply modulation: normalize then modulate
        part_tokens = modulate(self.norm(part_tokens), shift, scale)
        
        # 5. Flatten to sequence: (B, T_patches * num_groups, hidden_size)
        tokens = part_tokens.flatten(1, 2)
        
        return tokens


class HybridPoseDecoder3(nn.Module):
    """
    Decoder for HybridPoseEmbedder3 that handles body-part-only tokens.
    
    Each body part token predicts its corresponding joints.
    Supports overlapping groups (same joint in multiple groups) by averaging predictions.
    
    Input: (B, T_patches * num_groups, hidden_size) - body part tokens only
    Output: (B, T, J, out_channels) - 3D poses
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=256,
                 out_channels=3, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        
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
            tokens: (B, T_patches * num_groups, hidden_size) - body part tokens
            
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B = tokens.shape[0]
        T_patches = self.num_time_patches
        P_t = self.patch_size
        T = T_patches * P_t
        J = self.num_joints
        C_out = self.out_channels
        
        # Reshape tokens: (B, T_patches, num_groups, hidden_size)
        tokens = tokens.view(B, T_patches, self.num_groups, self.hidden_size)
        
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
        output = output / count.clamp(min=1)  # clamp to avoid division by zero
        
        return output


class HybridPoseDecoder(nn.Module):
    """
    Decoder for HybridPoseEmbedder that unpacks tokens back to joint predictions.
    
    Each token type predicts its corresponding joints:
    - Global token -> all 17 joints
    - Part tokens -> their respective joint groups
    
    Overlapping predictions are averaged.
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=256,
                 out_channels=3, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        
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
        self.num_token_types = 1 + self.num_groups
        
        self.num_time_patches = num_frame // patch_size
        
        # Global decoder: predicts all joints for patch_size frames
        global_out_dim = patch_size * num_joints * out_channels
        self.global_head = nn.Linear(hidden_size, global_out_dim)
        
        # Part decoders: each predicts its group joints for patch_size frames
        self.part_heads = nn.ModuleList()
        for group in self.joint_groups:
            out_dim = patch_size * len(group) * out_channels
            self.part_heads.append(nn.Linear(hidden_size, out_dim))
        
        # Register buffer for joint coverage counting
        self.register_buffer('_initialized', torch.tensor(False))
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches * num_token_types, hidden_size)
            
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B = tokens.shape[0]
        T_patches = self.num_time_patches
        P_t = self.patch_size
        T = T_patches * P_t
        J = self.num_joints
        C_out = self.out_channels
        
        # Reshape tokens: (B, T_patches, num_token_types, hidden_size)
        tokens = tokens.view(B, T_patches, self.num_token_types, self.hidden_size)
        
        # Initialize output accumulator and count
        output = torch.zeros(B, T, J, C_out, device=tokens.device, dtype=tokens.dtype)
        count = torch.zeros(B, T, J, 1, device=tokens.device, dtype=tokens.dtype)
        
        # 1. Decode global tokens (index 0)
        global_tokens = tokens[:, :, 0, :]  # (B, T_patches, hidden_size)
        global_pred = self.global_head(global_tokens)  # (B, T_patches, P_t * J * C_out)
        global_pred = global_pred.view(B, T_patches, P_t, J, C_out)
        global_pred = global_pred.view(B, T, J, C_out)
        
        output = output + global_pred
        count = count + 1
        
        # 2. Decode part tokens (indices 1 to num_groups)
        for i, group in enumerate(self.joint_groups):
            part_tokens = tokens[:, :, i + 1, :]  # (B, T_patches, hidden_size)
            part_pred = self.part_heads[i](part_tokens)  # (B, T_patches, P_t * group_size * C_out)
            
            group_size = len(group)
            part_pred = part_pred.view(B, T_patches, P_t, group_size, C_out)
            part_pred = part_pred.view(B, T, group_size, C_out)
            
            # Add to corresponding joint positions
            for j, joint_idx in enumerate(group):
                output[:, :, joint_idx, :] = output[:, :, joint_idx, :] + part_pred[:, :, j, :]
                count[:, :, joint_idx, :] = count[:, :, joint_idx, :] + 1
        
        # 3. Average overlapping predictions
        output = output / count
        
        return output


class TransformerBlock(nn.Module):
    """
    Simple Transformer block with multi-head self-attention and MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x


class HybridPoseModel(nn.Module):
    """
    Complete model for 2D-to-3D pose estimation using hybrid pose embeddings.
    
    Architecture:
    1. HybridPoseEmbedder: Creates global + body part tokens
    2. Transformer backbone: N layers of MHSA
    3. HybridPoseDecoder: Decodes tokens back to 3D poses
    """
    def __init__(self, num_frame=243, num_joints=17, in_channels=2, out_channels=3,
                 patch_size=9, hidden_size=256, depth=4, num_heads=8, 
                 mlp_ratio=4.0, dropout=0.1, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
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
        
        # Embedder
        self.embedder = HybridPoseEmbedder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            joint_groups=joint_groups
        )
        
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Decoder
        self.decoder = HybridPoseDecoder(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=hidden_size,
            out_channels=out_channels,
            joint_groups=joint_groups
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, 2) - 2D poses
            
        Returns:
            poses_3d: (B, T, J, 3) - 3D poses
        """
        # Embed
        tokens = self.embedder(x)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Decode
        poses_3d = self.decoder(tokens)
        
        return poses_3d


class HybridPoseDecoder2(nn.Module):
    """
    Simplified decoder for HybridPoseEmbedder2 that only processes global tokens.
    Each global token predicts all joints for its time patch.
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, hidden_size=256,
                 out_channels=3):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        
        self.num_time_patches = num_frame // patch_size
        
        # Each global token predicts all joints for patch_size frames
        out_dim = patch_size * num_joints * out_channels
        self.head = nn.Linear(hidden_size, out_dim)
    
    def forward(self, tokens):
        """
        Args:
            tokens: (B, T_patches, hidden_size) - global tokens only
            
        Returns:
            poses_3d: (B, T, J, out_channels)
        """
        B = tokens.shape[0]
        T_patches = self.num_time_patches
        P_t = self.patch_size
        T = T_patches * P_t
        J = self.num_joints
        C_out = self.out_channels
        
        # Decode: (B, T_patches, hidden_size) -> (B, T_patches, P_t * J * C_out)
        output = self.head(tokens)
        
        # Reshape: (B, T_patches, P_t, J, C_out) -> (B, T, J, C_out)
        output = output.view(B, T_patches, P_t, J, C_out)
        output = output.view(B, T, J, C_out)
        
        return output


class HybridPoseModel2(nn.Module):
    """
    Complete model for 2D-to-3D pose estimation using HybridPoseEmbedder2.
    
    Architecture:
    1. HybridPoseEmbedder2: Creates global tokens refined by cross-attention with part tokens
    2. Transformer backbone: N layers of MHSA
    3. HybridPoseDecoder2: Decodes global tokens to 3D poses
    """
    def __init__(self, num_frame=243, num_joints=17, in_channels=2, out_channels=3,
                 patch_size=9, hidden_size=256, depth=4, num_heads=8, 
                 mlp_ratio=4.0, dropout=0.1, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
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
        
        # Embedder with cross-attention
        self.embedder = HybridPoseEmbedder2(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            joint_groups=joint_groups
        )
        
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Decoder (simplified for global tokens only)
        self.decoder = HybridPoseDecoder2(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=hidden_size,
            out_channels=out_channels
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, 2) - 2D poses
            
        Returns:
            poses_3d: (B, T, J, 3) - 3D poses
        """
        # Embed (returns only global tokens after cross-attention)
        tokens = self.embedder(x)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Decode
        poses_3d = self.decoder(tokens)
        
        return poses_3d


class HybridPoseModel3(nn.Module):
    """
    Complete model for 2D-to-3D pose estimation using HybridPoseEmbedder2.
    
    Architecture:
    1. HybridPoseEmbedder2: Creates global tokens refined by cross-attention with part tokens
    2. Transformer backbone: N layers of MHSA
    3. HybridPoseDecoder2: Decodes global tokens to 3D poses
    """
    def __init__(self, num_frame=243, num_joints=17, in_channels=2, out_channels=3,
                 patch_size=9, hidden_size=256, depth=4, num_heads=8, 
                 mlp_ratio=4.0, dropout=0.1, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
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
        
        # Embedder with cross-attention
        self.embedder = HybridPoseEmbedder3(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            joint_groups=joint_groups
        )
        
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Decoder
        self.decoder = HybridPoseDecoder3(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=hidden_size,
            out_channels=out_channels,
            joint_groups=joint_groups
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, 2) - 2D poses
            
        Returns:
            poses_3d: (B, T, J, 3) - 3D poses
        """
        # Embed (returns only global tokens after cross-attention)
        tokens = self.embedder(x)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Decode
        poses_3d = self.decoder(tokens)
        
        return poses_3d


class HybridPoseEmbedder3_2(nn.Module):
    """
    Simplified Pose Embedder without global tokens or AdaLN modulation.
    
    Pipeline:
    1. Generate body part tokens (grouped joints -> 1 token per group per time patch)
    2. Add positional embeddings (temporal + body identity)
    3. Output body part tokens directly
    
    Output shape: (B, T_patches * num_groups, hidden_size)
    """
    def __init__(self, num_frame=243, num_joints=17, patch_size=9, in_channels=2, 
                 hidden_size=256, joint_groups=None, bias=True):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # Default H36M 17-joint skeleton groups
        if joint_groups is None:
            joint_groups = [
                [1, 2, 3],       # Right Leg
                [4, 5, 6],       # Left Leg  
                [0, 7, 8, 9, 10], # Torso/Spine
                [11, 12, 13],    # Left Arm
                [14, 15, 16]     # Right Arm
            ]
        self.joint_groups = joint_groups
        self.num_groups = len(joint_groups)
        
        self.num_time_patches = num_frame // patch_size
        self.num_tokens = self.num_time_patches * self.num_groups
        
        # Body part embedders (grouped joints -> 1 token per group per time patch)
        self.part_projs = nn.ModuleList()
        for group in self.joint_groups:
            dim = in_channels * patch_size * len(group)
            self.part_projs.append(nn.Linear(dim, hidden_size, bias=bias))
        
        # Fixed Positional Embeddings
        # Temporal PE: fixed sinusoidal encoding for each time patch
        temporal_pe = self._create_sinusoidal_pe(self.num_time_patches, hidden_size)
        self.register_buffer('temporal_pe', temporal_pe)  # (1, num_time_patches, hidden_size)
        
        # Body Identity PE: fixed sinusoidal encoding for each body part group
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
            tokens: (B, T_patches * num_groups, hidden_size) - body part tokens
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
        
        # Flatten to sequence: (B, T_patches * num_groups, hidden_size)
        tokens = part_tokens.flatten(1, 2)
        
        return tokens


class HybridPoseModel3_2(nn.Module):
    """
    Complete model for 2D-to-3D pose estimation using HybridPoseEmbedder3_2.
    
    Architecture:
    1. HybridPoseEmbedder3_2: Creates body part tokens (no global tokens)
    2. Transformer backbone: N layers of MHSA
    3. HybridPoseDecoder3: Decodes body part tokens to 3D poses
    """
    def __init__(self, num_frame=243, num_joints=17, in_channels=2, out_channels=3,
                 patch_size=9, hidden_size=256, depth=4, num_heads=8, 
                 mlp_ratio=4.0, dropout=0.1, joint_groups=None):
        super().__init__()
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.patch_size = patch_size
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
        
        # Embedder (no global tokens)
        self.embedder = HybridPoseEmbedder3_2(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            joint_groups=joint_groups
        )
        
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Decoder (reuse HybridPoseDecoder3)
        self.decoder = HybridPoseDecoder3(
            num_frame=num_frame,
            num_joints=num_joints,
            patch_size=patch_size,
            hidden_size=hidden_size,
            out_channels=out_channels,
            joint_groups=joint_groups
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, J, 2) - 2D poses
            
        Returns:
            poses_3d: (B, T, J, 3) - 3D poses
        """
        # Embed (body part tokens only)
        tokens = self.embedder(x)
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Decode
        poses_3d = self.decoder(tokens)
        
        return poses_3d