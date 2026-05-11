import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath
from einops import rearrange, repeat
from .token_selector import TokenSelector
from common.loss import (
    H36M_LEFT_JOINTS,
    H36M_PARENTS,
    H36M_RIGHT_JOINTS,
    DEFAULT_ANGLE_LIMITS,
    compute_bone_lengths,
    compute_joint_angles,
)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]

    return new_points


def cluster_dpc_knn(x, cluster_num, k, token_mask=None):
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            density = density * token_mask

        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return index_down, idx_cluster


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
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

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


def symmetry_penalty_per_frame(predicted, parents=None, left_joints=None, right_joints=None):
    if parents is None:
        parents = H36M_PARENTS
    if left_joints is None:
        left_joints = H36M_LEFT_JOINTS
    if right_joints is None:
        right_joints = H36M_RIGHT_JOINTS

    pred_lengths = compute_bone_lengths(predicted, parents)
    left_lengths = pred_lengths[..., left_joints]
    right_lengths = pred_lengths[..., right_joints]
    return torch.abs(left_lengths - right_lengths).mean(dim=-1)


def joint_angle_penalty_per_frame(predicted, parents=None, angle_limits=None, beta=0.1):
    if parents is None:
        parents = H36M_PARENTS
    if angle_limits is None:
        angle_limits = DEFAULT_ANGLE_LIMITS

    angles = compute_joint_angles(predicted, parents)
    penalties = []

    for joint_idx, (min_angle, max_angle) in angle_limits.items():
        joint_angles = angles[..., joint_idx]
        below_min = torch.clamp(min_angle - joint_angles, min=0.0)
        above_max = torch.clamp(joint_angles - max_angle, min=0.0)
        violations = below_min + above_max
        smooth_penalty = torch.where(
            violations < beta,
            0.5 * violations.square() / beta,
            violations - 0.5 * beta,
        )
        penalties.append(smooth_penalty)

    if not penalties:
        return torch.zeros_like(angles[..., 0])

    return torch.stack(penalties, dim=-1).mean(dim=-1)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        depth = 8
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2

        self.center = (args.frames - 1) // 2

        self.recover_num = args.frames
        self.token_num = args.token_num
        self.layer_index = args.layer_index
        self.pruning_strategy = getattr(args, "pruning_strategy", "learned")

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pos_embed_token = nn.Parameter(torch.zeros(1, self.token_num, embed_dim))

        drop_path_rate = 0.1
        drop_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        num_heads = 8
        num_joints = args.n_joints

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(2, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, depth=depth)
            for i in range(depth)])

        self.x_token = nn.Parameter(torch.zeros(1, self.recover_num, embed_dim))
        self.token_selector = None
        if self.pruning_strategy == "learned":
            self.token_selector = TokenSelector(
                embed_dim,
                hidden_dim=mlp_hidden_dim,
                drop=drop_rate,
                gate_scale=0.1,
            )

        self.cross_attention = Cross_Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , 3),
        )

    def _cluster_temporal_tokens(self, x):
        b, f, _, _ = x.shape
        x_knn = rearrange(x, 'b f n c -> b (f c) n')
        x_knn = self.pool(x_knn)
        x_knn = rearrange(x_knn, 'b (f c) 1 -> b f c', f=f)

        index, _ = cluster_dpc_knn(x_knn, self.token_num, 2)
        index, _ = torch.sort(index, dim=-1)

        batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
        return x[batch_ind, index]

    def _learned_prune_temporal_tokens(self, x):
        if self.token_selector is None:
            raise ValueError("Token selector is not initialized. Set pruning_strategy='learned' to use learned pruning.")
        selected, _, _ = self.token_selector(x, self.token_num)
        return selected

    def _encode_tokens(self, x):
        b, f, n, _ = x.shape

        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        for i in range(1, self.block_depth):
            if i == self.layer_index:
                if self.pruning_strategy == "cluster":
                    x = self._cluster_temporal_tokens(x)
                elif self.pruning_strategy == "learned":
                    x = self._learned_prune_temporal_tokens(x)
                else:
                    raise ValueError(f"Unsupported HOT pruning_strategy: {self.pruning_strategy}")

                x = rearrange(x, 'b f n c -> (b n) f c')
                x += self.pos_embed_token
                x = rearrange(x, '(b n) f c -> b f n c', n=n)

            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        return x

    def _recover_output(self, x):
        b, _, n, _ = x.shape
        x = rearrange(x, 'b f n c -> (b n) f c')
        x_token = repeat(self.x_token, '() f c -> b f c', b=b * n)
        x = x_token + self.cross_attention(x_token, x, x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        return self.head(x)

    def forward(self, x):
        x = self._encode_tokens(x)
        x = self._recover_output(x)
        return x


# --- Old ChunkedCompressionModel (commented out, replaced by new version below) ---
# class ChunkedCompressionModel(Model):
#     """HOT variant that compresses temporal embeddings by learnable chunk averaging."""
#
#     def __init__(self, args):
#         super().__init__(args)
#
#         if self.token_num <= 0 or self.token_num > args.frames:
#             raise ValueError(f"token_num must be in [1, {args.frames}] for chunked compression.")
#
#         chunk_size = args.frames // self.token_num
#         if chunk_size == 0:
#             raise ValueError("token_num must be no larger than frames for chunked compression.")
#
#         self.chunk_size = chunk_size
#         self.chunk_compress2_chunk_size = 9
#         self.chunk_compress2_tokens_per_chunk = 3
#
#         if self.layer_index == 0:
#             self.register_parameter('Temporal_pos_embed', None)
#             self.chunk_compression_weights = nn.Parameter(
#                 torch.full((1, self.token_num, chunk_size, 2), 1.0 / chunk_size)
#             )
#         else:
#             self.chunk_compression_weights = nn.Parameter(
#                 torch.full((1, self.token_num, chunk_size, args.channel), 1.0 / chunk_size)
#             )

#
#         # compress2_num_chunks = math.ceil(args.frames / self.chunk_compress2_chunk_size)
#         # compress2_weight_dim = 2 if self.layer_index == 0 else args.channel
#         # self.chunk_compression_weights2 = nn.Parameter(
#         #     torch.full(
#         #         (
#         #             1,
#         #             compress2_num_chunks,
#         #             self.chunk_compress2_tokens_per_chunk,
#         #             self.chunk_compress2_chunk_size,
#         #             compress2_weight_dim,
#         #         ),
#         #         1.0 / self.chunk_compress2_chunk_size,
#         #     )
#         # )
#         self.use_chunk_ortho_loss = getattr(args, "use_chunk_ortho_loss", False)
#         self.lambda_chunk_ortho = getattr(args, "lambda_chunk_ortho", 0.0)
#         self.latest_chunk_ortho_loss = None
#
#     def _chunk_compress(self, x, weights):
#         b, f, n, _ = x.shape
#         if self.token_num <= 0 or self.token_num > f:
#             raise ValueError(f"token_num must be in [1, {f}] for chunked compression.")
#
#         chunk_size = f // self.token_num
#         usable_f = self.token_num * chunk_size
#         if chunk_size != self.chunk_size or weights.shape[2] != chunk_size:
#             raise ValueError(
#                 f"Expected chunk_size={self.chunk_size} but got chunk_size={chunk_size}."
#             )
#
#         x = x[:, :usable_f]
#         x = rearrange(x, 'b (k s) n c -> (b n) k s c', k=self.token_num, s=chunk_size)
#         weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
#         x = (x * weights).sum(dim=2)
#         return rearrange(x, '(b n) k c -> b k n c', b=b, n=n)
#
#     # def _chunk_compress2(self, x, weights=None):
#     #     ...  (already commented out)
#
#     def _compute_chunk_orthogonality_loss(self, x):
#         if x.shape[1] <= 1:
#             return x.new_zeros(())
#
#         tokens = rearrange(x, 'b f n c -> b n f c')
#         tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
#         sim = torch.matmul(tokens, tokens.transpose(-1, -2))
#
#         diag_mask = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
#         sim = sim.masked_fill(diag_mask.view(1, 1, sim.shape[-1], sim.shape[-1]), 0.0)
#
#         denom = x.shape[0] * x.shape[1] * x.shape[2] * max(x.shape[1] - 1, 1)
#         return  (0.44 * (sim-0.5).square()).sum() / denom
#
#     def _encode_tokens(self, x):
#         self.latest_chunk_ortho_loss = None
#         b, f, n, _ = x.shape
#
#         if self.layer_index == 0:
#             x = self._chunk_compress(x, self.chunk_compression_weights)
#             f = x.shape[1]
#
#         x = rearrange(x, 'b f n c  -> (b f) n c')
#         x = self.Spatial_patch_to_embedding(x)
#         x += self.Spatial_pos_embed
#         x = self.pos_drop(x)
#         x = self.STEblocks[0](x)
#         x = self.Spatial_norm(x)
#
#         x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
#         if self.layer_index == 0:
#             x = x + self.pos_embed_token
#         else:
#             x = x +self.Temporal_pos_embed
#         x = self.pos_drop(x)
#         x = self.TTEblocks[0](x)
#         x = self.Temporal_norm(x)
#
#         x = rearrange(x, '(b n) f c -> b f n c', n=n)
#         if self.layer_index == 0 and self.use_chunk_ortho_loss:
#             self.latest_chunk_ortho_loss = self._compute_chunk_orthogonality_loss(x)
#         for i in range(1, self.block_depth):
#             if i == self.layer_index:
#                 x = self._chunk_compress(x, self.chunk_compression_weights)
#                 if self.use_chunk_ortho_loss:
#                     self.latest_chunk_ortho_loss = self._compute_chunk_orthogonality_loss(x)
#                 x = rearrange(x, 'b f n c -> (b n) f c')
#                 x = x + self.pos_embed_token
#                 x = rearrange(x, '(b n) f c -> b f n c', n=n)
#
#             x = rearrange(x, 'b f n c -> (b f) n c')
#             steblock = self.STEblocks[i]
#             tteblock = self.TTEblocks[i]
#
#             x = steblock(x)
#             x = self.Spatial_norm(x)
#             x = rearrange(x, '(b f) n c -> (b n) f c', b=b)
#
#             x = tteblock(x)
#             x = self.Temporal_norm(x)
#             x = rearrange(x, '(b n) f c -> b f n c', n=n)
#
#         return x
# --- End of old ChunkedCompressionModel ---


class ChunkedCompressionModel(Model):
    """HOT variant that compresses temporal embeddings by learnable chunk averaging.

    Supports multiple decoder modes for temporal recovery:
      - 'cross_attention': (default) learnable x_token + cross-attention (original HOT).
      - 'one_step_interp': project to 3D then linearly interpolate to full resolution.
      - 'one_step_upsample': ConvTranspose1d in feature space then project to 3D.
      - 'two_step_upsample': two ConvTranspose1d stages then project to 3D.
            Only for token_num=27 (27->81->243). Falls back to one_step_upsample.
      - 'two_step_mix': ConvTranspose1d to intermediate, project to 3D, then interpolate.
            For token_num=27 (27->81->243) or token_num=81 (81->162->243).
            Falls back to one_step_interp.
    """

    SUPPORTED_DECODER_MODES = (
        'cross_attention',
        'one_step_interp', 'one_step_upsample',
        'two_step_upsample', 'two_step_mix',
    )

    # {token_num: (intermediate_frames, stride1, stride2)}
    TWO_STEP_UPSAMPLE_CFG = {
        27: (81, 3, 3),  # 27 ->(s=3)-> 81 ->(s=3)-> 243
    }

    # {token_num: (intermediate_frames, stride1)}
    TWO_STEP_MIX_CFG = {
        27: (81, 3),   # 27 ->(s=3)-> 81  ->(interp)-> 243
        81: (162, 2),  # 81 ->(s=2)-> 162 ->(interp)-> 243
    }

    def __init__(self, args):
        super().__init__(args)

        if self.token_num <= 0 or self.token_num > args.frames:
            raise ValueError(f"token_num must be in [1, {args.frames}] for chunked compression.")

        base_chunk_size = args.frames // self.token_num
        if base_chunk_size == 0:
            raise ValueError("token_num must be no larger than frames for chunked compression.")

        self.chunking_scheme = getattr(args, "chunking_scheme", "even")
        self.chunk_lengths = self._resolve_chunk_lengths(args.frames, self.token_num, self.chunking_scheme)
        self.use_uneven_chunking = self.chunk_lengths is not None
        self.preserve_boundary_chunks = self.chunking_scheme == "corner_aligned"

        self.chunk_size = base_chunk_size
        self.max_chunk_size = max(self.chunk_lengths) if self.use_uneven_chunking else base_chunk_size

        if self.layer_index == 0:
            self.register_parameter('Temporal_pos_embed', None)
            self.chunk_compression_weights = nn.Parameter(
                self._build_chunk_weight_init(2)
            )
        else:
            self.chunk_compression_weights = nn.Parameter(
                self._build_chunk_weight_init(args.channel)
            )

        if self.use_uneven_chunking:
            chunk_lengths = torch.tensor(self.chunk_lengths, dtype=torch.long)
            chunk_offsets = torch.cat(
                [chunk_lengths.new_zeros(1), chunk_lengths.cumsum(dim=0)[:-1]],
                dim=0,
            )
            self.chunk_offsets_list = chunk_offsets.tolist()
            chunk_position_ids = torch.arange(self.max_chunk_size, dtype=torch.long).unsqueeze(0)
            chunk_valid_mask = (chunk_position_ids < chunk_lengths.unsqueeze(1)).view(1, self.token_num, self.max_chunk_size, 1)
            self.register_buffer("chunk_lengths_tensor", chunk_lengths, persistent=False)
            self.register_buffer("chunk_offsets", chunk_offsets, persistent=False)
            self.register_buffer("chunk_valid_mask", chunk_valid_mask, persistent=False)
        else:
            self.chunk_lengths_tensor = None
            self.chunk_offsets = None
            self.chunk_valid_mask = None
            self.chunk_offsets_list = None

        self.use_chunk_ortho_loss = getattr(args, "use_chunk_ortho_loss", False)
        self.lambda_chunk_ortho = getattr(args, "lambda_chunk_ortho", 0.0)
        self.latest_chunk_ortho_loss = None

        # --- Pairwise temporal flow (parallel, non-cascading) ---
        self.use_pairwise_flow = getattr(args, "use_pairwise_flow", False)
        if self.use_pairwise_flow:
            flow_dim = 2 if self.layer_index == 0 else args.channel
            self.flow_proj = nn.Linear(flow_dim, flow_dim)
            self.flow_gate = nn.Parameter(torch.full((1,), -5.0))  # sigmoid(-5) ≈ 0

        # --- Decoder mode ---
        embed_dim = args.channel
        self.decoder_mode = getattr(args, "decoder_mode", "cross_attention")
        assert self.decoder_mode in self.SUPPORTED_DECODER_MODES, \
            f"Unknown decoder_mode: {self.decoder_mode}. Must be one of {self.SUPPORTED_DECODER_MODES}"

        # Determine effective mode (handle fallbacks)
        self._effective_decoder_mode = self.decoder_mode

        if self.decoder_mode == 'two_step_upsample' \
                and self.token_num not in self.TWO_STEP_UPSAMPLE_CFG:
            print(f"[ChunkedCompressionModel] two_step_upsample not supported for "
                  f"token_num={self.token_num}. Falling back to one_step_upsample.")
            self._effective_decoder_mode = 'one_step_upsample'

        if self.decoder_mode == 'two_step_mix' \
                and self.token_num not in self.TWO_STEP_MIX_CFG:
            print(f"[ChunkedCompressionModel] two_step_mix not supported for "
                  f"token_num={self.token_num}. Falling back to one_step_interp.")
            self._effective_decoder_mode = 'one_step_interp'

        # Build decoder layers based on effective mode
        if self._effective_decoder_mode == 'cross_attention':
            # Parent already provides: self.x_token, self.cross_attention, self.head
            pass

        elif self._effective_decoder_mode == 'one_step_interp':
            # Only need head (already inherited). Remove unused cross_attention/x_token.
            del self.cross_attention
            self.register_parameter('x_token', None)
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )

        elif self._effective_decoder_mode == 'one_step_upsample':
            del self.cross_attention
            self.register_parameter('x_token', None)
            self.temporal_upsample = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=base_chunk_size, stride=base_chunk_size,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )

        elif self._effective_decoder_mode == 'two_step_upsample':
            del self.cross_attention
            self.register_parameter('x_token', None)
            _, s1, s2 = self.TWO_STEP_UPSAMPLE_CFG[self.token_num]
            self.temporal_upsample_1 = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=s1, stride=s1,
            )
            self.temporal_upsample_2 = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=s2, stride=s2,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )

        elif self._effective_decoder_mode == 'two_step_mix':
            del self.cross_attention
            self.register_parameter('x_token', None)
            self._intermediate, s1 = self.TWO_STEP_MIX_CFG[self.token_num]
            self.temporal_upsample_1 = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=s1, stride=s1,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )

    @classmethod
    def _resolve_chunk_lengths(cls, frames, token_num, chunking_scheme):
        if chunking_scheme == "even":
            return None
        if chunking_scheme == "corner_aligned":
            return cls._build_corner_aligned_chunk_lengths(frames, token_num)
        raise ValueError(
            f"Unknown chunking_scheme: {chunking_scheme}. Must be one of ('even', 'corner_aligned')."
        )

    @staticmethod
    def _build_corner_aligned_chunk_lengths(frames, token_num):
        if token_num < 2:
            raise ValueError(
                f"chunking_scheme='corner_aligned' requires token_num >= 2, but got {token_num}."
            )
        if frames < 2:
            raise ValueError(
                f"chunking_scheme='corner_aligned' requires frames >= 2, but got {frames}."
            )

        chunk_size = frames // token_num
        if chunk_size == 0:
            raise ValueError(
                f"chunking_scheme='corner_aligned' requires token_num <= frames, but got "
                f"frames={frames}, token_num={token_num}."
            )

        boundary_chunk_count = 2
        boundary_frames = 2
        regular_chunk_count = token_num - boundary_chunk_count
        extra_large_chunk_count = frames - boundary_frames - chunk_size * regular_chunk_count

        if extra_large_chunk_count < 0:
            raise ValueError(
                f"chunking_scheme='corner_aligned' produced a negative number of enlarged chunks for "
                f"frames={frames}, token_num={token_num}."
            )
        if extra_large_chunk_count % 2 != 0:
            raise ValueError(
                f"chunking_scheme='corner_aligned' requires an even number of enlarged chunks, but got "
                f"{extra_large_chunk_count} for frames={frames}, token_num={token_num}."
            )
        if extra_large_chunk_count > regular_chunk_count:
            raise ValueError(
                f"chunking_scheme='corner_aligned' requires {extra_large_chunk_count} enlarged chunks, "
                f"but only {regular_chunk_count} non-boundary chunks are available."
            )

        large_chunks_per_side = extra_large_chunk_count // 2
        regular_middle_chunk_count = regular_chunk_count - extra_large_chunk_count
        chunk_lengths = (
            [1]
            + [chunk_size + 1] * large_chunks_per_side
            + [chunk_size] * regular_middle_chunk_count
            + [chunk_size + 1] * large_chunks_per_side
            + [1]
        )

        if len(chunk_lengths) != token_num:
            raise ValueError(
                f"chunking_scheme='corner_aligned' generated {len(chunk_lengths)} chunks, expected {token_num}."
            )
        if sum(chunk_lengths) != frames:
            raise ValueError(
                f"chunking_scheme='corner_aligned' covers {sum(chunk_lengths)} frames, expected {frames}."
            )
        return chunk_lengths

    def _build_chunk_weight_init(self, channel_dim):
        if not self.use_uneven_chunking:
            return torch.full((1, self.token_num, self.chunk_size, channel_dim), 1.0 / self.chunk_size)

        weight_init = torch.zeros((1, self.token_num, self.max_chunk_size, channel_dim))
        for chunk_idx, chunk_len in enumerate(self.chunk_lengths):
            weight_init[:, chunk_idx, :chunk_len, :] = 1.0 / chunk_len
        return weight_init

    def _chunk_compress(self, x, weights):
        b, f, n, _ = x.shape
        if self.token_num <= 0 or self.token_num > f:
            raise ValueError(f"token_num must be in [1, {f}] for chunked compression.")

        if self.use_uneven_chunking:
            return self._chunk_compress_uneven(x, weights)

        chunk_size = f // self.token_num
        usable_f = self.token_num * chunk_size
        if chunk_size != self.chunk_size or weights.shape[2] != chunk_size:
            raise ValueError(
                f"Expected chunk_size={self.chunk_size} but got chunk_size={chunk_size}."
            )

        x = x[:, :usable_f]
        x = rearrange(x, 'b (k s) n c -> (b n) k s c', k=self.token_num, s=chunk_size)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        x = (x * weights).sum(dim=2)
        return rearrange(x, '(b n) k c -> b k n c', b=b, n=n)

    def _chunk_compress_uneven(self, x, weights):
        b, f, n, _ = x.shape
        if f != int(self.chunk_lengths_tensor.sum().item()):
            raise ValueError(
                f"Expected frames={int(self.chunk_lengths_tensor.sum().item())} for uneven chunking but got {f}."
            )
        if weights.shape[2] != self.max_chunk_size:
            raise ValueError(
                f"Expected max_chunk_size={self.max_chunk_size} but got {weights.shape[2]}."
            )

        chunked = x.new_zeros((b, self.token_num, self.max_chunk_size, n, x.shape[-1]))
        for chunk_idx, (start, chunk_len) in enumerate(zip(self.chunk_offsets_list, self.chunk_lengths)):
            chunked[:, chunk_idx, :chunk_len] = x[:, start:start + chunk_len]

        chunked = rearrange(chunked, 'b k s n c -> (b n) k s c')
        masked_weights = weights * self.chunk_valid_mask.to(dtype=weights.dtype)
        masked_weights = masked_weights / (masked_weights.sum(dim=2, keepdim=True) + 1e-6)
        compressed = (chunked * masked_weights).sum(dim=2)

        if self.preserve_boundary_chunks:
            compressed[:, 0] = chunked[:, 0, 0]
            compressed[:, -1] = chunked[:, -1, 0]

        return rearrange(compressed, '(b n) k c -> b k n c', b=b, n=n)

    def _pairwise_temporal_flow(self, x):
        """Parallel pairwise info flow: each chunk absorbs from its immediate predecessor.

        For all chunks simultaneously:
          C_1' = C_1  (unchanged, zero-padded predecessor)
          C_i' = C_i + sigmoid(gate) * Linear(C_{i-1})   for i = 2..K

        Uses original (pre-update) predecessor values — non-cascading.

        Args:
            x: (B, K, N, C) compressed chunk tokens.
        Returns:
            (B, K, N, C) chunks with predecessor information mixed in.
        """
        # Shift right: predecessor[i] = x[i-1], predecessor[0] = 0
        predecessor = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        flow = self.flow_proj(predecessor)
        return x + torch.sigmoid(self.flow_gate) * flow

    def _compute_chunk_orthogonality_loss(self, x):
        if x.shape[1] <= 1:
            return x.new_zeros(())

        tokens = rearrange(x, 'b f n c -> b n f c')
        tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(tokens, tokens.transpose(-1, -2))

        diag_mask = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(diag_mask.view(1, 1, sim.shape[-1], sim.shape[-1]), 0.0)

        denom = x.shape[0] * x.shape[1] * x.shape[2] * max(x.shape[1] - 1, 1)
        # The following is used for test quadratic loss of cosine similarity
        return (0.44 * (sim - 0.5).square()).sum() / denom

    def _encode_tokens(self, x):
        self.latest_chunk_ortho_loss = None
        b, f, n, _ = x.shape

        if self.layer_index == 0:
            x = self._chunk_compress(x, self.chunk_compression_weights)
            if self.use_pairwise_flow:
                x = self._pairwise_temporal_flow(x)
            f = x.shape[1]

        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        if self.layer_index == 0:
            x = x + self.pos_embed_token
        else:
            x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        if self.layer_index == 0 and self.use_chunk_ortho_loss:
            self.latest_chunk_ortho_loss = self._compute_chunk_orthogonality_loss(x)
        for i in range(1, self.block_depth):
            if i == self.layer_index:
                x = self._chunk_compress(x, self.chunk_compression_weights)
                if self.use_pairwise_flow:
                    x = self._pairwise_temporal_flow(x)
                if self.use_chunk_ortho_loss:
                    self.latest_chunk_ortho_loss = self._compute_chunk_orthogonality_loss(x)
                x = rearrange(x, 'b f n c -> (b n) f c')
                x = x + self.pos_embed_token
                x = rearrange(x, '(b n) f c -> b f n c', n=n)

            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        return x

    def _recover_output(self, x):
        b, f_enc, n, c = x.shape

        if self._effective_decoder_mode == 'cross_attention':
            # Original HOT recovery: learnable x_token + cross-attention
            x = rearrange(x, 'b f n c -> (b n) f c')
            x_token = repeat(self.x_token, '() f c -> b f c', b=b * n)
            x = x_token + self.cross_attention(x_token, x, x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)
            return self.head(x)

        if self._effective_decoder_mode == 'one_step_interp':
            sparse = self.head(x)  # (B, token_num, N, 3)
            sparse = rearrange(sparse, 'b t n c -> b (n c) t')
            out = F.interpolate(sparse, size=self.recover_num,
                                mode='linear', align_corners=True)
            return rearrange(out, 'b (n c) t -> b t n c', n=n, c=3)

        if self._effective_decoder_mode == 'one_step_upsample':
            x = rearrange(x, 'b t n c -> (b n) c t')
            x = self.temporal_upsample(x)  # (B*N, C, recover_num)
            x = rearrange(x, '(b n) c t -> b t n c', b=b, n=n)
            return self.head(x)

        if self._effective_decoder_mode == 'two_step_upsample':
            x = rearrange(x, 'b t n c -> (b n) c t')
            x = self.temporal_upsample_1(x)  # (B*N, C, intermediate)
            x = self.temporal_upsample_2(x)  # (B*N, C, recover_num)
            x = rearrange(x, '(b n) c t -> b t n c', b=b, n=n)
            return self.head(x)

        # two_step_mix
        x = rearrange(x, 'b t n c -> (b n) c t')
        x = self.temporal_upsample_1(x)  # (B*N, C, intermediate)
        x = rearrange(x, '(b n) c t -> b t n c', b=b, n=n)
        sparse = self.head(x)  # (B, intermediate, N, 3)
        sparse = rearrange(sparse, 'b t n c -> b (n c) t')
        out = F.interpolate(sparse, size=self.recover_num,
                            mode='linear', align_corners=True)
        return rearrange(out, 'b (n c) t -> b t n c', n=n, c=3)

    def forward(self, x):
        x = self._encode_tokens(x)
        x = self._recover_output(x)
        return x


class ChunkCompressMultiStepModel(ChunkedCompressionModel):
    """HOT variant with explicit multi-stage chunk compression."""

    def __init__(self, args):
        Model.__init__(self, args)

        self.stage_layer_indices = self._parse_stage_values(
            getattr(args, "hierarchical_layer_indices", "2,3"),
            "hierarchical_layer_indices",
        )
        self.stage_token_nums = self._parse_stage_values(
            getattr(args, "hierarchical_token_nums", "81,27"),
            "hierarchical_token_nums",
        )
        if len(self.stage_layer_indices) != len(self.stage_token_nums):
            raise ValueError(
                "hierarchical_layer_indices and hierarchical_token_nums must have the same length "
                f"(got {len(self.stage_layer_indices)} and {len(self.stage_token_nums)})."
            )
        if not self.stage_layer_indices:
            raise ValueError("At least one compression stage is required for ChunkCompressMultiStepModel.")
        if any(layer_idx < 0 or layer_idx >= self.block_depth for layer_idx in self.stage_layer_indices):
            raise ValueError(
                f"Compression layer indices must be within [0, {self.block_depth - 1}], "
                f"but got {self.stage_layer_indices}."
            )
        if sorted(self.stage_layer_indices) != self.stage_layer_indices:
            raise ValueError(
                f"Compression layer indices must be sorted in increasing order, but got {self.stage_layer_indices}."
            )
        if len(set(self.stage_layer_indices)) != len(self.stage_layer_indices):
            raise ValueError(f"Compression layer indices must be unique, but got {self.stage_layer_indices}.")
        if self.stage_token_nums[-1] != self.token_num:
            raise ValueError(
                f"args.token_num must match the final stage token count for multi-step chunk compression "
                f"(got token_num={self.token_num}, final_stage={self.stage_token_nums[-1]})."
            )

        self.chunking_scheme = getattr(args, "chunking_scheme", "even")
        self.use_chunk_ortho_loss = getattr(args, "use_chunk_ortho_loss", False)
        self.lambda_chunk_ortho = getattr(args, "lambda_chunk_ortho", 0.0)
        self.latest_chunk_ortho_loss = None

        self.use_pairwise_flow = getattr(args, "use_pairwise_flow", False)
        self.stage_flow_projs = nn.ModuleList() if self.use_pairwise_flow else None
        self.stage_flow_gates = nn.ParameterList() if self.use_pairwise_flow else None

        self.stage_chunk_compression_weights = nn.ParameterList()
        self.stage_pos_embed_token_names = []
        self.stage_configs = []
        self.stage_index_by_layer = {}

        prev_frames = args.frames
        for stage_idx, (layer_idx, token_num) in enumerate(zip(self.stage_layer_indices, self.stage_token_nums)):
            if token_num <= 0 or token_num > prev_frames:
                raise ValueError(
                    f"Stage {stage_idx} token_num must be in [1, {prev_frames}], but got {token_num}."
                )

            chunk_size = prev_frames // token_num
            if chunk_size == 0:
                raise ValueError(
                    f"Stage {stage_idx} token_num must be no larger than its input frames "
                    f"(frames={prev_frames}, token_num={token_num})."
                )

            chunk_lengths = self._resolve_chunk_lengths(prev_frames, token_num, self.chunking_scheme)
            use_uneven_chunking = chunk_lengths is not None
            max_chunk_size = max(chunk_lengths) if use_uneven_chunking else chunk_size
            weight_dim = 2 if layer_idx == 0 else args.channel
            self.stage_chunk_compression_weights.append(
                nn.Parameter(self._build_stage_chunk_weight_init(token_num, chunk_size, chunk_lengths, max_chunk_size, weight_dim))
            )

            if stage_idx == len(self.stage_token_nums) - 1:
                self.stage_pos_embed_token_names.append(None)
            else:
                pos_embed_name = f"_stage_{stage_idx}_pos_embed_token"
                self.register_parameter(
                    pos_embed_name,
                    nn.Parameter(torch.zeros(1, token_num, args.channel)),
                )
                self.stage_pos_embed_token_names.append(pos_embed_name)

            if self.use_pairwise_flow:
                self.stage_flow_projs.append(nn.Linear(weight_dim, weight_dim))
                self.stage_flow_gates.append(nn.Parameter(torch.full((1,), -5.0)))

            chunk_offsets_list = None
            valid_mask_name = None
            if use_uneven_chunking:
                chunk_lengths_tensor = torch.tensor(chunk_lengths, dtype=torch.long)
                chunk_offsets = torch.cat(
                    [chunk_lengths_tensor.new_zeros(1), chunk_lengths_tensor.cumsum(dim=0)[:-1]],
                    dim=0,
                )
                chunk_offsets_list = chunk_offsets.tolist()
                chunk_position_ids = torch.arange(max_chunk_size, dtype=torch.long).unsqueeze(0)
                chunk_valid_mask = (
                    chunk_position_ids < chunk_lengths_tensor.unsqueeze(1)
                ).view(1, token_num, max_chunk_size, 1)
                valid_mask_name = f"_stage_{stage_idx}_chunk_valid_mask"
                self.register_buffer(valid_mask_name, chunk_valid_mask, persistent=False)

            self.stage_configs.append(
                {
                    "layer_idx": layer_idx,
                    "token_num": token_num,
                    "input_frames": prev_frames,
                    "chunk_size": chunk_size,
                    "chunk_lengths": chunk_lengths,
                    "use_uneven_chunking": use_uneven_chunking,
                    "max_chunk_size": max_chunk_size,
                    "chunk_offsets_list": chunk_offsets_list,
                    "chunk_valid_mask_name": valid_mask_name,
                }
            )
            self.stage_index_by_layer[layer_idx] = stage_idx
            prev_frames = token_num

        self.decoder_mode = getattr(args, "decoder_mode", "cross_attention")
        assert self.decoder_mode in self.SUPPORTED_DECODER_MODES, \
            f"Unknown decoder_mode: {self.decoder_mode}. Must be one of {self.SUPPORTED_DECODER_MODES}"

        self._effective_decoder_mode = self.decoder_mode
        if self.decoder_mode == 'two_step_upsample' \
                and self.token_num not in self.TWO_STEP_UPSAMPLE_CFG:
            print(f"[ChunkCompressMultiStepModel] two_step_upsample not supported for "
                  f"token_num={self.token_num}. Falling back to one_step_upsample.")
            self._effective_decoder_mode = 'one_step_upsample'

        if self.decoder_mode == 'two_step_mix' \
                and self.token_num not in self.TWO_STEP_MIX_CFG:
            print(f"[ChunkCompressMultiStepModel] two_step_mix not supported for "
                  f"token_num={self.token_num}. Falling back to one_step_interp.")
            self._effective_decoder_mode = 'one_step_interp'

        self._init_decoder_layers(args.channel, args.frames // self.token_num)

    @staticmethod
    def _parse_stage_values(value, arg_name):
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(',') if part.strip()]
        elif isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            raise TypeError(f"{arg_name} must be a comma-separated string or a sequence, but got {type(value).__name__}.")

        try:
            parsed = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{arg_name} must contain only integers, but got {value}.") from exc

        return parsed

    @staticmethod
    def _build_stage_chunk_weight_init(token_num, chunk_size, chunk_lengths, max_chunk_size, channel_dim):
        if chunk_lengths is None:
            return torch.full((1, token_num, chunk_size, channel_dim), 1.0 / chunk_size)

        weight_init = torch.zeros((1, token_num, max_chunk_size, channel_dim))
        for chunk_idx, chunk_len in enumerate(chunk_lengths):
            weight_init[:, chunk_idx, :chunk_len, :] = 1.0 / chunk_len
        return weight_init

    def _init_decoder_layers(self, embed_dim, base_chunk_size):
        if self._effective_decoder_mode == 'cross_attention':
            return

        if self._effective_decoder_mode == 'one_step_interp':
            del self.cross_attention
            self.register_parameter('x_token', None)
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )
            return

        if self._effective_decoder_mode == 'one_step_upsample':
            del self.cross_attention
            self.register_parameter('x_token', None)
            self.temporal_upsample = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=base_chunk_size, stride=base_chunk_size,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )
            return

        if self._effective_decoder_mode == 'two_step_upsample':
            del self.cross_attention
            self.register_parameter('x_token', None)
            _, s1, s2 = self.TWO_STEP_UPSAMPLE_CFG[self.token_num]
            self.temporal_upsample_1 = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=s1, stride=s1,
            )
            self.temporal_upsample_2 = nn.ConvTranspose1d(
                in_channels=embed_dim, out_channels=embed_dim,
                kernel_size=s2, stride=s2,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 3),
            )
            return

        del self.cross_attention
        self.register_parameter('x_token', None)
        self._intermediate, s1 = self.TWO_STEP_MIX_CFG[self.token_num]
        self.temporal_upsample_1 = nn.ConvTranspose1d(
            in_channels=embed_dim, out_channels=embed_dim,
            kernel_size=s1, stride=s1,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def _chunk_compress_stage(self, x, stage_idx):
        cfg = self.stage_configs[stage_idx]
        weights = self.stage_chunk_compression_weights[stage_idx]
        b, f, n, _ = x.shape
        if cfg["token_num"] <= 0 or cfg["token_num"] > f:
            raise ValueError(
                f"Stage {stage_idx} token_num must be in [1, {f}] for chunked compression."
            )

        if cfg["use_uneven_chunking"]:
            expected_frames = sum(cfg["chunk_lengths"])
            if f != expected_frames:
                raise ValueError(
                    f"Stage {stage_idx} expected frames={expected_frames} for uneven chunking but got {f}."
                )
            if weights.shape[2] != cfg["max_chunk_size"]:
                raise ValueError(
                    f"Stage {stage_idx} expected max_chunk_size={cfg['max_chunk_size']} "
                    f"but got {weights.shape[2]}."
                )

            chunked = x.new_zeros((b, cfg["token_num"], cfg["max_chunk_size"], n, x.shape[-1]))
            for chunk_idx, (start, chunk_len) in enumerate(zip(cfg["chunk_offsets_list"], cfg["chunk_lengths"])):
                chunked[:, chunk_idx, :chunk_len] = x[:, start:start + chunk_len]

            chunked = rearrange(chunked, 'b k s n c -> (b n) k s c')
            chunk_valid_mask = getattr(self, cfg["chunk_valid_mask_name"]).to(dtype=weights.dtype)
            masked_weights = weights * chunk_valid_mask
            masked_weights = masked_weights / (masked_weights.sum(dim=2, keepdim=True) + 1e-6)
            compressed = (chunked * masked_weights).sum(dim=2)
            return rearrange(compressed, '(b n) k c -> b k n c', b=b, n=n)

        chunk_size = f // cfg["token_num"]
        usable_f = cfg["token_num"] * chunk_size
        if chunk_size != cfg["chunk_size"] or weights.shape[2] != cfg["chunk_size"]:
            raise ValueError(
                f"Stage {stage_idx} expected chunk_size={cfg['chunk_size']} but got chunk_size={chunk_size}."
            )

        x = x[:, :usable_f]
        x = rearrange(x, 'b (k s) n c -> (b n) k s c', k=cfg["token_num"], s=chunk_size)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        x = (x * weights).sum(dim=2)
        return rearrange(x, '(b n) k c -> b k n c', b=b, n=n)

    def _pairwise_temporal_flow_stage(self, x, stage_idx):
        predecessor = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        flow = self.stage_flow_projs[stage_idx](predecessor)
        return x + torch.sigmoid(self.stage_flow_gates[stage_idx]) * flow

    def _get_stage_pos_embed_token(self, stage_idx):
        pos_embed_name = self.stage_pos_embed_token_names[stage_idx]
        if pos_embed_name is None:
            return self.pos_embed_token
        return getattr(self, pos_embed_name)

    def _encode_tokens(self, x):
        self.latest_chunk_ortho_loss = None
        ortho_loss = None
        b, f, n, _ = x.shape

        stage0_idx = self.stage_index_by_layer.get(0)
        if stage0_idx is not None:
            x = self._chunk_compress_stage(x, stage0_idx)
            if self.use_pairwise_flow:
                x = self._pairwise_temporal_flow_stage(x, stage0_idx)
            f = x.shape[1]

        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        if stage0_idx is not None:
            x = x + self._get_stage_pos_embed_token(stage0_idx)
        else:
            x = x + self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        if stage0_idx is not None and self.use_chunk_ortho_loss:
            ortho_loss = self._compute_chunk_orthogonality_loss(x)

        for i in range(1, self.block_depth):
            stage_idx = self.stage_index_by_layer.get(i)
            if stage_idx is not None:
                x = self._chunk_compress_stage(x, stage_idx)
                if self.use_pairwise_flow:
                    x = self._pairwise_temporal_flow_stage(x, stage_idx)
                if self.use_chunk_ortho_loss:
                    current_ortho = self._compute_chunk_orthogonality_loss(x)
                    ortho_loss = current_ortho if ortho_loss is None else ortho_loss + current_ortho
                x = rearrange(x, 'b f n c -> (b n) f c')
                x = x + self._get_stage_pos_embed_token(stage_idx)
                x = rearrange(x, '(b n) f c -> b f n c', n=n)

            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        self.latest_chunk_ortho_loss = ortho_loss
        return x


class PreservedQueryModel(Model):
    def __init__(self, args):
        super().__init__(args)
        # This variant does not decode from a learned query token.
        self.register_parameter("x_token", None)

        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2
        num_joints = args.n_joints
        num_heads = 8
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        qkv_bias = True
        qk_scale = None
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.block_depth)]

        # Independent query generation head
        self.query_Spatial_patch_to_embedding = nn.Linear(2, embed_dim)
        self.query_Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.query_Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))
        self.query_pos_drop = nn.Dropout(p=drop_rate)
        self.query_STEblock = Block(
            dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        )
        self.query_TTEblock = Block(
            dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            depth=self.block_depth,
        )
        self.query_Spatial_norm = norm_layer(embed_dim)
        self.query_Temporal_norm = norm_layer(embed_dim)

        # Decoder mode: "cross_attention" (original) or "conv_upsample" (new)
        self.decoder_mode = getattr(args, "decoder_mode", "cross_attention")
        assert self.decoder_mode in ("cross_attention", "conv_upsample"), \
            f"Unknown decoder_mode for PreservedQueryModel: {self.decoder_mode}"

        if self.decoder_mode == "conv_upsample":
            del self.cross_attention
            upsample_stride = self.recover_num // self.token_num
            self.temporal_upsample = nn.ConvTranspose1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=upsample_stride,
                stride=upsample_stride,
            )
            self.encoded_norm = norm_layer(embed_dim)
            self.query_out_norm = norm_layer(embed_dim)

    def _generate_query(self, x):
        b, f, n, _ = x.shape
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.query_Spatial_patch_to_embedding(x)
        x += self.query_Spatial_pos_embed
        x = self.query_pos_drop(x)
        x = self.query_STEblock(x)
        x = self.query_Spatial_norm(x)
        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.query_Temporal_pos_embed
        x = self.query_pos_drop(x)
        x = self.query_TTEblock(x)
        x = self.query_Temporal_norm(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        return x

    def _recover_output(self, x, query):
        b, f_q, n, c = query.shape

        if self.decoder_mode == "cross_attention":
            x = rearrange(x, 'b f n c -> (b n) f c')
            query = rearrange(query, 'b f n c -> (b n) f c')
            query = query + self.cross_attention(query, x, x)
            query = rearrange(query, '(b n) f c -> b f n c', b=b, n=n)
            return self.head(query)

        # conv_upsample: upsample encoded tokens, then fuse with query via addition
        x = rearrange(x, 'b f n c -> (b n) c f')
        x = self.temporal_upsample(x)  # (B*N, C, frames)
        x = rearrange(x, '(b n) c f -> b f n c', b=b, n=n)

        # LayerNorm both branches to align activation scales
        x = self.encoded_norm(x)
        query = self.query_out_norm(query)

        # Fuse and regress to 3D
        fused = x + query
        return self.head(fused)

    def forward(self, x):
        query = self._generate_query(x)
        encoded = self._encode_tokens(x)
        output = self._recover_output(encoded, query)
        return output


class MultiHypothesisModel(Model):
    def __init__(self, args):
        super().__init__(args)
        self.num_hypotheses = getattr(args, "num_hypotheses", 4)
        self.symmetry_floor = getattr(args, "symmetry_floor", 1e-3)
        self.joint_angle_floor = getattr(args, "joint_angle_floor", 1e-3)
        self.score_eps = getattr(args, "score_eps", 1e-8)
        self.x_token = nn.Parameter(
            torch.zeros(1, self.num_hypotheses, self.recover_num, args.channel)
        )

    def _recover_all_hypotheses(self, x):
        b, _, n, _ = x.shape
        source = rearrange(x, 'b f n c -> (b n) f c')
        source = repeat(source, 'bn f c -> (bn h) f c', h=self.num_hypotheses)
        x_token = repeat(self.x_token, '() h f c -> (b n h) f c', b=b, n=n)
        recovered = x_token + self.cross_attention(x_token, source, source)
        recovered = rearrange(recovered, '(b n h) f c -> b h f n c', b=b, n=n, h=self.num_hypotheses)
        return self.head(recovered)

    def _select_best_hypothesis(self, hypotheses):
        sym_loss = symmetry_penalty_per_frame(hypotheses)
        joint_loss = joint_angle_penalty_per_frame(hypotheses)

        sym_adj = sym_loss + self.symmetry_floor
        joint_adj = joint_loss + self.joint_angle_floor

        norm_sym = sym_adj / (sym_adj.sum(dim=1, keepdim=True) + self.score_eps)
        norm_joint = joint_adj / (joint_adj.sum(dim=1, keepdim=True) + self.score_eps)
        scores = norm_sym * norm_joint

        best_idx = scores.argmin(dim=1)
        gather_idx = best_idx[:, None, :, None, None].expand(-1, 1, -1, hypotheses.shape[3], hypotheses.shape[4])
        best = hypotheses.gather(dim=1, index=gather_idx).squeeze(1)
        return best

    def forward(self, x):
        encoded = self._encode_tokens(x)
        hypotheses = self._recover_all_hypotheses(encoded)
        return self._select_best_hypothesis(hypotheses)


class OracleSelectionModel(Model):
    """Ablation variant: oracle temporal token selection using GT 2D poses.

    Instead of clustering or learned pruning, compares detected 2D poses with
    ground-truth 2D poses per frame (MSE) and keeps the top-K frames with
    the smallest error.  This serves as an upper-bound experiment for token
    selection quality.

    Key difference from :class:`Model`: the K best frames are selected from
    the raw input *before* any encoding.  Only K frames flow through the
    entire STE/TTE backbone (no mid-network pruning step).  Temporal positional
    embeddings are gathered per-sample so that each selected frame retains its
    correct position encoding.

    Forward signature: ``forward(detected_2d, gt_2d)`` instead of ``forward(x)``.
    """

    def __init__(self, args):
        args.pruning_strategy = 'cluster'
        super().__init__(args)
        # pos_embed_token is not used (no mid-network pruning step).
        self.register_parameter('pos_embed_token', None)
        self.oracle_mode = getattr(args, 'oracle_mode', 'global')

    # ------------------------------------------------------------------
    # Oracle index computation
    # ------------------------------------------------------------------
    def _compute_oracle_indices(self, x, gt_2d):
        """Compute oracle frame indices based on MSE between detected and GT 2D.

        Args:
            x:      Detected 2D poses, shape ``(B, F, J, 2)``.
            gt_2d:  Ground-truth 2D poses, shape ``(B, F, J, 2)``.

        Returns:
            Sorted global frame indices, shape ``(B, token_num)``.
        """
        b, T = x.shape[0], x.shape[1]

        if self.oracle_mode == 'chunked':
            chunk_size = T // self.token_num
            usable_T = self.token_num * chunk_size

            # (B, token_num, chunk_size, J, 2)
            x_chunks = x[:, :usable_T].reshape(b, self.token_num, chunk_size, *x.shape[2:])
            gt_chunks = gt_2d[:, :usable_T].reshape(b, self.token_num, chunk_size, *gt_2d.shape[2:])

            # Per-frame MSE within each chunk -> (B, token_num, chunk_size)
            per_chunk_mse = ((x_chunks - gt_chunks) ** 2).mean(dim=(-1, -2))

            # Best frame index within each chunk -> (B, token_num)
            best_in_chunk = per_chunk_mse.argmin(dim=-1)

            # Convert to global frame indices
            chunk_starts = torch.arange(self.token_num, device=x.device) * chunk_size
            indices = chunk_starts[None, :] + best_in_chunk  # (B, token_num)
        else:
            # Global top-K
            per_frame_mse = ((x - gt_2d) ** 2).mean(dim=(-1, -2))  # (B, F)
            _, indices = torch.topk(per_frame_mse, k=self.token_num, largest=False)
            indices, _ = torch.sort(indices, dim=-1)

        return indices

    # ------------------------------------------------------------------
    # Overridden encoder: no mid-network pruning, gathered temporal PE
    # ------------------------------------------------------------------
    def _encode_tokens(self, x, temporal_indices):
        """Encode pre-selected tokens with correct positional embeddings.

        Args:
            x: Pre-selected 2D poses, shape ``(B, K, J, 2)`` where K = token_num.
            temporal_indices: Original frame indices of the selected tokens,
                shape ``(B, K)``, used to gather from ``Temporal_pos_embed``.
        """
        b, f, n, _ = x.shape  # f = token_num (K)

        # ---- Spatial embedding (identical to parent) ----
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        # ---- Temporal embedding (gathered per-sample) ----
        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        c = x.shape[-1]
        tpe = self.Temporal_pos_embed.expand(b, -1, -1)           # (B, 243, C)
        idx = temporal_indices.unsqueeze(-1).expand(-1, -1, c)    # (B, K, C)
        temporal_pe = torch.gather(tpe, 1, idx)                   # (B, K, C)
        temporal_pe = repeat(temporal_pe, 'b f c -> (b n) f c', n=n)
        x += temporal_pe

        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        # ---- Remaining STE/TTE blocks ----
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            x = self.STEblocks[i](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)

            x = self.TTEblocks[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        return x

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, x, gt_2d):
        """
        Args:
            x:      Detected 2D poses, shape ``(B, F, J, 2)``.
            gt_2d:  Ground-truth 2D poses, shape ``(B, F, J, 2)``.

        Returns:
            Predicted 3D poses, shape ``(B, F, J, 3)``.
        """
        b = x.shape[0]

        indices = self._compute_oracle_indices(x, gt_2d)

        # Pre-select frames from raw input
        batch_ind = torch.arange(b, device=x.device).unsqueeze(-1)
        x = x[batch_ind, indices]  # (B, K, J, 2)

        # Encode (only K frames) and recover (back to full sequence)
        x = self._encode_tokens(x, indices)
        x = self._recover_output(x)
        return x


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser().parse_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.n_joints, args.out_joints = 17, 17
    args.token_num = 81
    args.layer_index = 3

    input_2d = torch.rand(1, args.frames, 17, 2)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/ 1000000)

        print(input_2d.shape, 1)
        output = model(input_2d)
        print(output.shape, 2)

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input_2d, ))
    print('macs: ', macs/1000000, 'params: ', params/1000000)
    macs, params = clever_format([macs*2, params], "%.3f")
    print(macs, params)
