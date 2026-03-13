import torch
import torch.nn as nn
from functools import partial
from timm.layers import DropPath
from einops import rearrange, repeat
from .token_selector import TokenSelector


def _parse_int_list(value, arg_name):
    if value is None:
        return []

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        return [int(v.strip()) for v in cleaned.split(",") if v.strip()]

    raise TypeError(f"{arg_name} must be int, list/tuple of int, comma-separated str, or None.")


def index_points(points, idx):
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        batch_size, num_tokens, channels = x.shape
        if cluster_num >= num_tokens:
            raise ValueError(f"cluster_num ({cluster_num}) must be smaller than current token count ({num_tokens}).")

        dist_matrix = torch.cdist(x, x) / (channels ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(batch_size, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


class TokenPruningMotion(nn.Module):
    """Token Pruning Motion (TPMo) - Selects tokens based on human motion magnitude.
    
    From H2OT paper §3.1.3: Computes motion from original 2D poses and selects
    top-k frames with largest motion variations.
    """
    def __init__(self, num_representative_tokens):
        """
        Args:
            num_representative_tokens (int): The target number of tokens to keep (r_m).
        """
        super().__init__()
        self.r_m = num_representative_tokens

    def get_motion_scores(self, input_2d_poses):
        """
        Calculates motion scores based on the input 2D poses.
        Formula from paper Eq.4: M = {0, s2-s1, s3-s2, ...}
        
        Args:
            input_2d_poses (torch.Tensor): Shape (B, F, J, 2) or (B, F, D)
        
        Returns:
            motion_scores (torch.Tensor): Shape (B, F)
        """
        # Calculate velocity (temporal difference)
        diff = input_2d_poses[:, 1:] - input_2d_poses[:, :-1]
        
        # Add a zero frame at the beginning to match original length F (paper Eq.4)
        zeros = torch.zeros_like(input_2d_poses[:, :1])
        motion = torch.cat([zeros, diff], dim=1)

        # Sum absolute motion across joints and coordinates to get score per frame
        if motion.dim() == 4: # (B, F, J, C)
            motion_scores = motion.abs().sum(dim=(2, 3))
        else: # (B, F, D)
            motion_scores = motion.abs().sum(dim=2)
            
        return motion_scores

    def forward(self, tokens, input_2d_poses):
        """
        Args:
            tokens (torch.Tensor): The feature tokens to prune. Shape (B, F, J, C) or (B, F, D).
            input_2d_poses (torch.Tensor): The original 2D poses used to calculate motion. 
                                           Shape must match temporal dimension F of tokens.
        
        Returns:
            pruned_tokens (torch.Tensor): Selected tokens. Shape (B, r_m, ...)
            indices (torch.Tensor): Indices of selected tokens (useful for debugging/viz).
        """
        B, F, *_ = tokens.shape
        
        # 1. Calculate Motion Scores
        scores = self.get_motion_scores(input_2d_poses) # (B, F)
        
        # 2. Select Top-k Frames
        _, indices = torch.topk(scores, k=self.r_m, dim=1)
        
        # 3. Sort indices to preserve temporal order (for TRI compatibility)
        indices, _ = torch.sort(indices, dim=1)
        
        # 4. Gather tokens
        if tokens.dim() == 4:
            B, F, J, C = tokens.shape
            gather_indices = indices.view(B, self.r_m, 1, 1).expand(B, self.r_m, J, C)
        else:
            B, F, D = tokens.shape
            gather_indices = indices.view(B, self.r_m, 1).expand(B, self.r_m, D)
            
        pruned_tokens = torch.gather(tokens, 1, gather_indices)
        
        return pruned_tokens, indices


class TokenPruningSampler(nn.Module):
    """Token Pruning Sampler (TPS) - Uniformly samples tokens along temporal dimension.
    
    From H2OT paper §3.1.4: Parameter-free linear sampling, ordered for TRI compatibility.
    """
    def __init__(self, num_representative_tokens):
        """
        Args:
            num_representative_tokens (int): The target number of tokens to keep (r_m).
        """
        super().__init__()
        self.r_m = num_representative_tokens

    def forward(self, tokens):
        """
        Args:
            tokens (torch.Tensor): The feature tokens to sample. Shape (B, F, J, C) or (B, F, D).
        
        Returns:
            sampled_tokens (torch.Tensor): Uniformly sampled tokens. Shape (B, r_m, ...)
            indices (torch.Tensor): Indices of sampled tokens. Shape (B, r_m).
        """
        B, F, *rest = tokens.shape
        
        # Generate uniform indices
        indices = torch.linspace(0, F - 1, self.r_m, device=tokens.device).long()
        indices = indices.unsqueeze(0).expand(B, -1) # (B, r_m)
        
        # Gather tokens
        if tokens.dim() == 4:
            _, _, J, C = tokens.shape
            gather_indices = indices.view(B, self.r_m, 1, 1).expand(B, self.r_m, J, C)
        else:
            D = rest[0]
            gather_indices = indices.view(B, self.r_m, 1).expand(B, self.r_m, D)
            
        sampled_tokens = torch.gather(tokens, 1, gather_indices)
        
        return sampled_tokens, indices


class TokenRecoveringInterpolation(nn.Module):
    """Token Recovering Interpolation (TRI) - Interpolates 3D poses after regression head.
    
    From H2OT paper §3.2.2: Parameter-free interpolation on 3D poses (not features).
    Must be applied AFTER regression head, unlike TRA which works on features.
    """
    def __init__(self, target_frames):
        """
        Args:
            target_frames (int): The target number of frames to recover to (F).
        """
        super().__init__()
        self.target_frames = target_frames

    def forward(self, poses_3d):
        """
        Args:
            poses_3d (torch.Tensor): Pruned 3D poses from regression head. 
                                     Shape (B, r_m, J, 3) or (B, r_m, D).
        
        Returns:
            recovered_poses (torch.Tensor): Interpolated to full length. Shape (B, F, J, 3) or (B, F, D).
        """
        B, r_m, *rest = poses_3d.shape
        
        if r_m == self.target_frames:
            return poses_3d  # No interpolation needed
        
        # Reshape for F.interpolate: (B, C, T) format
        if poses_3d.dim() == 4:  # (B, r_m, J, 3)
            _, _, J, C = poses_3d.shape
            poses_3d = rearrange(poses_3d, 'b t j c -> b (j c) t')
            interpolated = torch.nn.functional.interpolate(
                poses_3d, size=self.target_frames, mode='linear', align_corners=True
            )
            interpolated = rearrange(interpolated, 'b (j c) t -> b t j c', j=J, c=C)
        else:  # (B, r_m, D)
            D = rest[0]
            poses_3d = poses_3d.transpose(1, 2)  # (B, D, r_m)
            interpolated = torch.nn.functional.interpolate(
                poses_3d, size=self.target_frames, mode='linear', align_corners=True
            )
            interpolated = interpolated.transpose(1, 2)  # (B, F, D)
        
        return interpolated


def interpolate_pose_batch_with_indices(pruned_poses, kept_indices, target_seq_len):
    """Recover poses to full sequence length using original-frame kept indices."""
    if pruned_poses.ndim != 4:
        raise ValueError(f"pruned_poses must have shape (B, F', J, C), got {tuple(pruned_poses.shape)}")
    if kept_indices.ndim != 2:
        raise ValueError(f"kept_indices must have shape (B, F'), got {tuple(kept_indices.shape)}")
    if target_seq_len < 1:
        raise ValueError("target_seq_len must be >= 1")

    batch_size, reduced_frames, num_joints, channels = pruned_poses.shape
    if kept_indices.shape != (batch_size, reduced_frames):
        raise ValueError(
            f"kept_indices shape {tuple(kept_indices.shape)} does not match (B, F') = ({batch_size}, {reduced_frames})"
        )
    if reduced_frames < 1:
        raise ValueError("F' must be >= 1")

    x_old = kept_indices.to(device=pruned_poses.device, dtype=torch.float32)
    if reduced_frames > 1 and not torch.all(x_old[:, 1:] > x_old[:, :-1]):
        raise ValueError("kept_indices must be strictly increasing per batch")

    y_old = pruned_poses.view(batch_size, reduced_frames, -1)
    x_new = torch.arange(target_seq_len, device=pruned_poses.device, dtype=torch.float32)
    x_new = x_new.unsqueeze(0).expand(batch_size, -1).contiguous()

    if reduced_frames == 1:
        y_new = y_old[:, :1, :].expand(batch_size, target_seq_len, y_old.shape[-1])
        return y_new.view(batch_size, target_seq_len, num_joints, channels)

    right_indices = torch.searchsorted(x_old, x_new, side="right")
    right_indices = torch.clamp(right_indices, 1, reduced_frames - 1)
    left_indices = right_indices - 1

    def _gather_time(tensor, idx):
        feat_dim = tensor.shape[-1]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, feat_dim)
        return torch.gather(tensor, 1, gather_idx)

    x_left = torch.gather(x_old, 1, left_indices)
    x_right = torch.gather(x_old, 1, right_indices)
    y_left = _gather_time(y_old, left_indices)
    y_right = _gather_time(y_old, right_indices)

    eps = 1e-6
    weights = (x_new - x_left) / (x_right - x_left + eps)
    y_new = y_left + weights.unsqueeze(-1) * (y_right - y_left)
    return y_new.view(batch_size, target_seq_len, num_joints, channels)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_key, x_value):
        batch_size, query_len, channels = x_query.shape
        _, key_len, _ = x_key.shape

        q = self.linear_q(x_query).reshape(batch_size, query_len, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_key).reshape(batch_size, key_len, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_value).reshape(batch_size, key_len, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, query_len, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_hidden_dim,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class H2OTMixSTE(nn.Module):
    def __init__(self, args):
        super().__init__()

        depth = 8
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2

        self.output_frames = args.frames
        self.pool = nn.AdaptiveAvgPool1d(1)

        cluster_layer_indices = _parse_int_list(getattr(args, "hierarchical_layer_indices", None), "hierarchical_layer_indices")
        cluster_token_nums = _parse_int_list(getattr(args, "hierarchical_token_nums", None), "hierarchical_token_nums")
        if not cluster_layer_indices and not cluster_token_nums:
            cluster_layer_indices = [int(args.layer_index)]
            cluster_token_nums = [int(args.token_num)]

        if len(cluster_layer_indices) != len(cluster_token_nums):
            raise ValueError("hierarchical_layer_indices and hierarchical_token_nums must have the same length.")
        if len(set(cluster_layer_indices)) != len(cluster_layer_indices):
            raise ValueError("hierarchical_layer_indices contains duplicated indices.")
        if any(idx < 1 or idx >= depth for idx in cluster_layer_indices):
            raise ValueError(f"All hierarchical_layer_indices must be in [1, {depth - 1}].")

        paired_cluster = sorted(zip(cluster_layer_indices, cluster_token_nums), key=lambda x: x[0])
        self.cluster_layer_indices = [item[0] for item in paired_cluster]
        self.cluster_token_nums = [item[1] for item in paired_cluster]
        self.cluster_stage_by_block = {idx: stage for stage, idx in enumerate(self.cluster_layer_indices)}

        current_tokens = args.frames
        for token_num in self.cluster_token_nums:
            if token_num <= 0:
                raise ValueError("All hierarchical_token_nums must be positive.")
            if token_num >= current_tokens:
                raise ValueError(
                    f"Invalid hierarchical reduction path: token_num {token_num} must be smaller than previous stage {current_tokens}."
                )
            current_tokens = token_num

        recovery_layer_indices = _parse_int_list(getattr(args, "recovery_layer_indices", None), "recovery_layer_indices")
        recovery_token_nums = _parse_int_list(getattr(args, "recovery_token_nums", None), "recovery_token_nums")
        if len(recovery_layer_indices) != len(recovery_token_nums):
            raise ValueError("recovery_layer_indices and recovery_token_nums must have the same length.")
        if len(set(recovery_layer_indices)) != len(recovery_layer_indices):
            raise ValueError("recovery_layer_indices contains duplicated indices.")
        if any(idx < 1 or idx >= depth for idx in recovery_layer_indices):
            raise ValueError(f"All recovery_layer_indices must be in [1, {depth - 1}].")
        if any(token_num <= 0 or token_num > args.frames for token_num in recovery_token_nums):
            raise ValueError(f"All recovery_token_nums must be in [1, {args.frames}].")

        paired_recovery = sorted(zip(recovery_layer_indices, recovery_token_nums), key=lambda x: x[0])
        self.recovery_layer_indices = [item[0] for item in paired_recovery]
        self.recovery_token_nums = [item[1] for item in paired_recovery]
        self.recovery_stage_by_block = {idx: stage for stage, idx in enumerate(self.recovery_layer_indices)}
        self.recovery_on_hierarchy = bool(getattr(args, "recovery_on_hierarchy", False))

        # NEW: Strategy selection for pruning and recovery
        self.pruning_strategy = getattr(args, "pruning_strategy", "cluster")  # cluster | learned | motion | sampler
        self.recovery_strategy = getattr(args, "recovery_strategy", "attention")  # attention | interpolation
        
        # Validate strategy combinations
        valid_pruning = ["cluster", "learned", "motion", "sampler"]
        valid_recovery = ["attention", "interpolation"]
        if self.pruning_strategy not in valid_pruning:
            raise ValueError(f"pruning_strategy must be one of {valid_pruning}, got {self.pruning_strategy}")
        if self.recovery_strategy not in valid_recovery:
            raise ValueError(f"recovery_strategy must be one of {valid_recovery}, got {self.recovery_strategy}")

        drop_path_rate = 0.1
        drop_rate = 0.0
        attn_drop_rate = 0.0
        qkv_bias = True
        qk_scale = None

        num_heads = 8
        num_joints = args.n_joints

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(2, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))

        # Positional embeddings after token-count changes.
        if self.pruning_strategy in {"cluster", "learned"}:
            self.cluster_pos_embeds = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, token_num, embed_dim)) for token_num in self.cluster_token_nums]
            )
        else:
            self.cluster_pos_embeds = nn.ParameterList()  # Empty for other strategies

        # Cross-attention recovery infrastructure (only for attention recovery strategy)
        if self.recovery_strategy == "attention":
            query_token_sizes = sorted(set(self.recovery_token_nums + [args.frames]))
            self.recovery_query_tokens = nn.ParameterDict(
                {str(token_num): nn.Parameter(torch.zeros(1, token_num, embed_dim)) for token_num in query_token_sizes}
            )
        else:
            self.recovery_query_tokens = nn.ParameterDict()  # Empty for interpolation

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.TTEblocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Cross-attention modules (only for attention recovery)
        if self.recovery_strategy == "attention":
            query_token_sizes = sorted(set(self.recovery_token_nums + [args.frames]))
            self.cross_attentions = nn.ModuleDict(
                {
                    str(token_num): CrossAttention(
                        embed_dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop_rate,
                        proj_drop=drop_rate,
                    )
                    for token_num in query_token_sizes
                }
            )
        else:
            self.cross_attentions = nn.ModuleDict()  # Empty for interpolation
        
        # NEW: Pruning modules (motion, sampler)
        if self.pruning_strategy == "motion":
            # TPMo modules for each hierarchical stage
            self.motion_pruners = nn.ModuleList(
                [TokenPruningMotion(token_num) for token_num in self.cluster_token_nums]
            )
        elif self.pruning_strategy == "learned":
            self.token_selectors = nn.ModuleList(
                [TokenSelector(embed_dim, hidden_dim=mlp_hidden_dim, drop=drop_rate, gate_scale=0.1) for _ in self.cluster_token_nums]
            )
        elif self.pruning_strategy == "sampler":
            # TPS modules for each hierarchical stage
            self.samplers = nn.ModuleList(
                [TokenPruningSampler(token_num) for token_num in self.cluster_token_nums]
            )
        
        # NEW: Recovery module (interpolation)
        if self.recovery_strategy == "interpolation":
            self.tri_interpolator = TokenRecoveringInterpolation(args.frames)

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def _cluster_temporal_tokens(self, x, target_tokens, stage_idx):
        batch_size, num_tokens, _, _ = x.shape
        if target_tokens >= num_tokens:
            raise ValueError(
                f"Cannot cluster stage {stage_idx}: target_tokens ({target_tokens}) must be smaller than current tokens ({num_tokens})."
            )

        x_knn = rearrange(x, "b f n c -> b (f c) n")
        x_knn = self.pool(x_knn)
        x_knn = rearrange(x_knn, "b (f c) 1 -> b f c", f=num_tokens)

        index, _ = cluster_dpc_knn(x_knn, target_tokens, 2)
        index, _ = torch.sort(index, dim=-1)

        batch_ind = torch.arange(batch_size, device=x.device).unsqueeze(-1)
        x = x[batch_ind, index]

        x = rearrange(x, "b f n c -> (b n) f c")
        x = x + self.cluster_pos_embeds[stage_idx]
        x = rearrange(x, "(b n) f c -> b f n c", n=x.shape[0] // batch_size)
        return x
    
    def _sample_temporal_tokens(self, x, target_tokens, stage_idx):
        """TPS: Uniformly sample tokens along temporal dimension."""
        batch_size, num_tokens, num_joints, channels = x.shape
        if target_tokens >= num_tokens:
            return x
        
        # Use TPS module
        x_sampled_list = []
        for joint_idx in range(num_joints):
            x_joint = x[:, :, joint_idx, :]  # (B, F, C)
            x_joint_sampled, _ = self.samplers[stage_idx](x_joint)
            x_sampled_list.append(x_joint_sampled)
        
        x = torch.stack(x_sampled_list, dim=2)  # (B, r_m, J, C)
        return x
    
    def _learned_prune_temporal_tokens(self, x, target_tokens, stage_idx):
        batch_size, num_tokens, num_joints, _ = x.shape
        if target_tokens >= num_tokens:
            return x

        x, _, _ = self.token_selectors[stage_idx](x, target_tokens)
        x = rearrange(x, "b f n c -> (b n) f c")
        x = x + self.cluster_pos_embeds[stage_idx]
        x = rearrange(x, "(b n) f c -> b f n c", b=batch_size, n=num_joints)
        return x
    
    
    def _motion_prune_temporal_tokens(self, x, target_tokens, stage_idx, input_2d):
        """TPMo: Select tokens based on human motion magnitude."""
        batch_size, num_tokens, num_joints, channels = x.shape
        if target_tokens >= num_tokens:
            return x
        
        # Use TPMo module - compute motion scores
        # For hierarchical stages after the first, input_2d has original F frames
        # but x has been pruned, so we compute motion on current x instead
        if input_2d.shape[1] == num_tokens:
            # First stage: use original 2D poses
            motion_input = input_2d
        else:
            # Subsequent hierarchical stages: use current features as proxy
            # Take first 2 channels as "pseudo-2D" for motion scoring
            motion_input = x[:, :, :, :2]  # (B, current_F, J, 2)
        
        scores = self.motion_pruners[stage_idx].get_motion_scores(motion_input)  # (B, current_F)
        
        # Select top-k frames based on motion
        _, indices = torch.topk(scores, k=target_tokens, dim=1)
        indices, _ = torch.sort(indices, dim=1)  # Sort for temporal order
        
        # Gather tokens for all joints using the same indices
        gather_indices = indices.view(batch_size, target_tokens, 1, 1).expand(batch_size, target_tokens, num_joints, channels)
        x = torch.gather(x, 1, gather_indices)  # (B, r_m, J, C)
        
        return x
    
    def _prune_temporal_tokens(self, x, target_tokens, stage_idx, input_2d=None):
        """Dispatch pruning based on pruning_strategy."""
        if self.pruning_strategy == "cluster":
            return self._cluster_temporal_tokens(x, target_tokens, stage_idx)
        elif self.pruning_strategy == "learned":
            return self._learned_prune_temporal_tokens(x, target_tokens, stage_idx)
        elif self.pruning_strategy == "sampler":
            return self._sample_temporal_tokens(x, target_tokens, stage_idx)
        elif self.pruning_strategy == "motion":
            if input_2d is None:
                raise ValueError("TPMo pruning requires input_2d argument")
            return self._motion_prune_temporal_tokens(x, target_tokens, stage_idx, input_2d)
        else:
            raise ValueError(f"Unknown pruning_strategy: {self.pruning_strategy}")

    def _recover_temporal_tokens(self, x, target_tokens):
        batch_size, _, num_joints, _ = x.shape
        key = str(target_tokens)
        if key not in self.recovery_query_tokens:
            raise ValueError(f"Recovery query token size {target_tokens} was not initialized.")
        if key not in self.cross_attentions:
            raise ValueError(f"CrossAttention for token size {target_tokens} was not initialized.")

        x = rearrange(x, "b f n c -> (b n) f c")
        query = repeat(self.recovery_query_tokens[key], "() f c -> b f c", b=batch_size * num_joints)
        x = query + self.cross_attentions[key](query, x, x)
        x = rearrange(x, "(b n) f c -> b f n c", b=batch_size, n=num_joints)
        return x

    def forward(self, x, input_2d=None):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input 2D poses. Shape (B, F, J, 2).
            input_2d (torch.Tensor, optional): Original 2D poses for TPMo. 
                                               Only required if pruning_strategy="motion".
                                               Shape (B, F, J, 2).
        """
        batch_size, frames, joints, _ = x.shape
        
        # Store original 2D for TPMo (Q2: user says only use one pruning strategy at a time)
        if self.pruning_strategy == "motion" and input_2d is None:
            # If input_2d not provided externally, use x itself for motion scoring
            input_2d = x

        x = rearrange(x, "b f n c -> (b f) n c")
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, "(b f) n c -> (b n) f c", f=frames)
        x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        x = rearrange(x, "(b n) f c -> b f n c", n=joints)

        for block_idx in range(1, self.block_depth):
            if block_idx in self.cluster_stage_by_block:
                cluster_stage = self.cluster_stage_by_block[block_idx]
                target_tokens = self.cluster_token_nums[cluster_stage]
                # NEW: Use dispatch instead of direct _cluster_temporal_tokens
                x = self._prune_temporal_tokens(x, target_tokens, cluster_stage, input_2d)

            x = rearrange(x, "b f n c -> (b f) n c")
            x = self.STEblocks[block_idx](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, "(b f) n c -> (b n) f c", b=batch_size)

            x = self.TTEblocks[block_idx](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, "(b n) f c -> b f n c", n=joints)


            if self.recovery_on_hierarchy and block_idx in self.recovery_stage_by_block:
                recovery_stage = self.recovery_stage_by_block[block_idx]
                target_tokens = self.recovery_token_nums[recovery_stage]
                # Only use TRA here (TRI operates after head)
                if self.recovery_strategy == "attention":
                    x = self._recover_temporal_tokens(x, target_tokens)

        # NEW: Recovery dispatch based on recovery_strategy (Q3: paper-faithful placement)
        if self.recovery_strategy == "attention":
            # TRA: Recover features BEFORE regression head
            if x.shape[1] != self.output_frames:
                x = self._recover_temporal_tokens(x, self.output_frames)
            x = self.head(x)  # (B, F, J, 3)
        elif self.recovery_strategy == "interpolation":
            # TRI: Interpolate 3D poses AFTER regression head (paper §3.2.2)
            x = self.head(x)  # (B, r_m, J, 3) - pruned poses
            if x.shape[1] != self.output_frames:
                # Interpolate across joints dimension
                x_list = []
                for joint_idx in range(joints):
                    x_joint = x[:, :, joint_idx, :]  # (B, r_m, 3)
                    x_joint_interp = self.tri_interpolator(x_joint)  # (B, F, 3)
                    x_list.append(x_joint_interp)
                x = torch.stack(x_list, dim=2)  # (B, F, J, 3)
        else:
            raise ValueError(f"Unknown recovery_strategy: {self.recovery_strategy}")
        
        x = x.view(batch_size, -1, joints, 3)
        return x



class H2OTMixSTEInterp(H2OTMixSTE):
    """H2OTMixSTE variant that composes kept indices and applies index-aware interpolation."""
    def __init__(self, args):
        super().__init__(args)
        self.recovery_strategy = "interpolation"
        if not hasattr(self, "tri_interpolator"):
            self.tri_interpolator = TokenRecoveringInterpolation(self.output_frames)

    @staticmethod
    def _gather_global_indices(global_indices, local_indices):
        return torch.gather(global_indices, 1, local_indices)

    def _cluster_temporal_tokens_with_indices(self, x, target_tokens, stage_idx):
        batch_size, num_tokens, num_joints, channels = x.shape
        if target_tokens >= num_tokens:
            raise ValueError(
                f"Cannot cluster stage {stage_idx}: target_tokens ({target_tokens}) must be smaller than current tokens ({num_tokens})."
            )

        x_knn = rearrange(x, "b f n c -> b (f c) n")
        x_knn = self.pool(x_knn)
        x_knn = rearrange(x_knn, "b (f c) 1 -> b f c", f=num_tokens)

        local_indices, _ = cluster_dpc_knn(x_knn, target_tokens, 2)
        local_indices, _ = torch.sort(local_indices, dim=-1)

        gather_idx = local_indices[:, :, None, None].expand(batch_size, target_tokens, num_joints, channels)
        x = torch.gather(x, 1, gather_idx)
        x = rearrange(x, "b f n c -> (b n) f c")
        x = x + self.cluster_pos_embeds[stage_idx]
        x = rearrange(x, "(b n) f c -> b f n c", b=batch_size, n=num_joints)
        return x, local_indices

    def _sample_temporal_tokens_with_indices(self, x, target_tokens):
        batch_size, num_tokens, num_joints, channels = x.shape
        if target_tokens >= num_tokens:
            local_indices = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
            return x, local_indices

        local_indices = torch.linspace(0, num_tokens - 1, target_tokens, device=x.device).long()
        local_indices = local_indices.unsqueeze(0).expand(batch_size, -1)
        gather_idx = local_indices[:, :, None, None].expand(batch_size, target_tokens, num_joints, channels)
        x = torch.gather(x, 1, gather_idx)
        return x, local_indices

    def _learned_prune_temporal_tokens_with_indices(self, x, target_tokens, stage_idx):
        batch_size, num_tokens, num_joints, _ = x.shape
        if target_tokens >= num_tokens:
            local_indices = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
            return x, local_indices

        x, local_indices, _ = self.token_selectors[stage_idx](x, target_tokens)
        x = rearrange(x, "b f n c -> (b n) f c")
        x = x + self.cluster_pos_embeds[stage_idx]
        x = rearrange(x, "(b n) f c -> b f n c", b=batch_size, n=num_joints)
        return x, local_indices

    def _motion_prune_temporal_tokens_with_indices(self, x, target_tokens, stage_idx, input_2d):
        batch_size, num_tokens, num_joints, channels = x.shape
        if target_tokens >= num_tokens:
            local_indices = torch.arange(num_tokens, device=x.device).unsqueeze(0).expand(batch_size, -1)
            return x, local_indices

        if input_2d.shape[1] == num_tokens:
            motion_input = input_2d
        else:
            motion_input = x[:, :, :, :2]

        scores = self.motion_pruners[stage_idx].get_motion_scores(motion_input)
        _, local_indices = torch.topk(scores, k=target_tokens, dim=1)
        local_indices, _ = torch.sort(local_indices, dim=1)

        gather_idx = local_indices[:, :, None, None].expand(batch_size, target_tokens, num_joints, channels)
        x = torch.gather(x, 1, gather_idx)
        return x, local_indices

    def _prune_temporal_tokens_with_indices(self, x, target_tokens, stage_idx, input_2d=None):
        if self.pruning_strategy == "cluster":
            return self._cluster_temporal_tokens_with_indices(x, target_tokens, stage_idx)
        if self.pruning_strategy == "learned":
            return self._learned_prune_temporal_tokens_with_indices(x, target_tokens, stage_idx)
        if self.pruning_strategy == "sampler":
            return self._sample_temporal_tokens_with_indices(x, target_tokens)
        if self.pruning_strategy == "motion":
            if input_2d is None:
                raise ValueError("TPMo pruning requires input_2d argument")
            return self._motion_prune_temporal_tokens_with_indices(x, target_tokens, stage_idx, input_2d)
        raise ValueError(f"Unknown pruning_strategy: {self.pruning_strategy}")

    def forward(self, x, input_2d=None, return_metadata=False, return_pre_interp=False):
        batch_size, frames, joints, _ = x.shape
        # if self.pruning_strategy == "motion" and input_2d is None:
        if input_2d is None:
            input_2d = x

        global_indices = torch.arange(frames, device=x.device).unsqueeze(0).expand(batch_size, -1)
        stage_indices = []

        x = rearrange(x, "b f n c -> (b f) n c")
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, "(b f) n c -> (b n) f c", f=frames)
        x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        x = rearrange(x, "(b n) f c -> b f n c", n=joints)

        for block_idx in range(1, self.block_depth):
            if block_idx in self.cluster_stage_by_block:
                stage_idx = self.cluster_stage_by_block[block_idx]
                target_tokens = self.cluster_token_nums[stage_idx]
                x, local_indices = self._prune_temporal_tokens_with_indices(x, target_tokens, stage_idx, input_2d)
                global_indices = self._gather_global_indices(global_indices, local_indices)
                stage_indices.append(
                    {
                        "block_idx": block_idx,
                        "local_indices": local_indices,
                        "global_indices_after_stage": global_indices,
                    }
                )

            x = rearrange(x, "b f n c -> (b f) n c")
            x = self.STEblocks[block_idx](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, "(b f) n c -> (b n) f c", b=batch_size)
            x = self.TTEblocks[block_idx](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, "(b n) f c -> b f n c", n=joints)

        pre_interp_poses = self.head(x)
        recovered_poses = pre_interp_poses
        if recovered_poses.shape[1] != self.output_frames:
            recovered_poses = interpolate_pose_batch_with_indices(recovered_poses, global_indices, self.output_frames)

        recovered_poses = recovered_poses.view(batch_size, -1, joints, 3)
        pre_interp_poses = pre_interp_poses.view(batch_size, -1, joints, 3)
        if not return_metadata and not return_pre_interp:
            return recovered_poses

        metadata = {
            "final_kept_indices": global_indices,
            "stage_indices": stage_indices,
        }
        if return_pre_interp and return_metadata:
            return recovered_poses, pre_interp_poses, global_indices, metadata
        if return_pre_interp:
            return recovered_poses, pre_interp_poses, global_indices
        return recovered_poses, metadata


Model = H2OTMixSTE
