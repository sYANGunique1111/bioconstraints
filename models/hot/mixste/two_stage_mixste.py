import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from .h2ot_mixste import Block, CrossAttention

class ScoringHead(nn.Module):
    """Lightweight head to evaluate 2D pose detection quality per frame."""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (B, F, J, C)
        x_pooled = x.mean(dim=2) # Pool across joints -> (B, F, C)
        scores = self.regressor(x_pooled).squeeze(-1) # -> (B, F)
        return scores


class TwoStageMixSTE(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.embed_dim = args.channel
        self.frames = args.frames
        self.num_joints = args.n_joints
        mlp_hidden_dim = args.channel * 2
        num_heads = 8
        drop_rate = 0.0
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # Base Embeddings
        self.Spatial_patch_to_embedding = nn.Linear(2, self.embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_joints, self.embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.frames, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # -----------------------------------------------------------------
        # STAGE 1: Pose Evaluator (Blocks 0 to 1)
        # -----------------------------------------------------------------
        self.stage1_depth = 4
        self.stage1_STE = nn.ModuleList([Block(dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_path=0.0, norm_layer=norm_layer) for _ in range(self.stage1_depth)])
        self.stage1_TTE = nn.ModuleList([Block(dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_path=0.0, norm_layer=norm_layer) for _ in range(self.stage1_depth)])
        
        self.scoring_head = ScoringHead(self.embed_dim, mlp_hidden_dim)
        
        # -----------------------------------------------------------------
        # STAGE 2: 3D Regressor (Blocks 2 to 7)
        # -----------------------------------------------------------------
        self.stage2_depth = 6
        self.stage2_STE = nn.ModuleList([Block(dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_path=0.0, norm_layer=norm_layer) for _ in range(self.stage2_depth)])
        self.stage2_TTE = nn.ModuleList([Block(dim=self.embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, drop_path=0.0, norm_layer=norm_layer) for _ in range(self.stage2_depth)])
        
        self.Spatial_norm = norm_layer(self.embed_dim)
        self.Temporal_norm = norm_layer(self.embed_dim)
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 3),
        )
        self.recovery_query_tokens = nn.ParameterDict({
            str(self.frames): nn.Parameter(torch.zeros(1, self.frames, self.embed_dim))
        })
        self.cross_attentions = nn.ModuleDict({
            str(self.frames): CrossAttention(
                self.embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=drop_rate,
            )
        })

    def _localized_window_pruning(self, features, scores, window_size, keep_k):
        """Selects top-K tokens within non-overlapping temporal windows."""
        B, F, J, C = features.shape
        if F % window_size != 0:
            raise ValueError(f"frames ({F}) must be divisible by window_size ({window_size})")
        if keep_k > window_size:
            raise ValueError(f"keep_k ({keep_k}) must be <= window_size ({window_size})")
        num_windows = F // window_size
        
        # Reshape to windows
        scores_windows = scores.view(B, num_windows, window_size)
        
        # Extract Top-K local indices
        _, topk_indices = torch.topk(scores_windows, keep_k, dim=-1)
        topk_indices, _ = torch.sort(topk_indices, dim=-1) # Chronological sort
        
        # Map to global temporal indices
        window_offsets = torch.arange(num_windows, device=features.device) * window_size
        window_offsets = window_offsets.view(1, num_windows, 1)
        global_indices = (topk_indices + window_offsets).view(B, -1)
        
        # Gather Features
        gather_idx = global_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, J, C)
        pruned_features = torch.gather(features, 1, gather_idx)
        
        # Gather original Temporal Positional Embeddings
        pos_embed = self.Temporal_pos_embed.expand(B, -1, -1)
        gather_pos_idx = global_indices.unsqueeze(-1).expand(-1, -1, C)
        pruned_pos_embed = torch.gather(pos_embed, 1, gather_pos_idx)
        
        return pruned_features, pruned_pos_embed, global_indices

    def _run_stage1(self, x, B, J):
        for i in range(self.stage1_depth):
            x = rearrange(x, "b f n c -> (b f) n c")
            x = self.stage1_STE[i](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, "(b f) n c -> (b n) f c", b=B)
            if i == 0:
                x = x + self.Temporal_pos_embed
            x = self.stage1_TTE[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, "(b n) f c -> b f n c", n=J)
        return x

    def _run_stage2(self, x, pruned_pos_embed, B, J):
        for i in range(self.stage2_depth):
            x = rearrange(x, "b f n c -> (b f) n c")
            x = self.stage2_STE[i](x)
            x = self.Spatial_norm(x)
            x = rearrange(x, "(b f) n c -> (b n) f c", b=B)
            if i == 0:
                pruned_pos = pruned_pos_embed.unsqueeze(1).expand(-1, J, -1, -1)
                pruned_pos = rearrange(pruned_pos, "b n f c -> (b n) f c")
                x = x + pruned_pos
            x = self.stage2_TTE[i](x)
            x = self.Temporal_norm(x)
            x = rearrange(x, "(b n) f c -> b f n c", n=J)
        return x

    def _recover_temporal_tokens(self, x, target_tokens):
        B, _, J, _ = x.shape
        key = str(target_tokens)
        if key not in self.recovery_query_tokens or key not in self.cross_attentions:
            raise ValueError(f"Recovery modules for token size {target_tokens} were not initialized")

        x = rearrange(x, "b f n c -> (b n) f c")
        query = repeat(self.recovery_query_tokens[key], "() f c -> b f c", b=B * J)
        x = query + self.cross_attentions[key](query, x, x)
        x = rearrange(x, "(b n) f c -> b f n c", b=B, n=J)
        return x

    def forward(self, x, stage=2, window_size=27, keep_k=15):
        """
        Args:
            x: Input 2D poses (B, F, J, 2)
            stage: 1 (returns scores for training) or 2 (returns 3D poses)
        """
        B, F, J, _ = x.shape
        
        # Initial Embeddings
        x = rearrange(x, "b f n c -> (b f) n c")
        x = self.Spatial_patch_to_embedding(x)
        x = x + self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, "(b f) n c -> b f n c", b=B)
        
        if stage == 1:
            x = self._run_stage1(x, B, J)
            return self.scoring_head(x)

        with torch.no_grad():
            x = self._run_stage1(x, B, J)
            scores = self.scoring_head(x)

        pruned_features, pruned_pos_embed, _ = self._localized_window_pruning(
            x, scores, window_size, keep_k
        )
        
        x = self._run_stage2(pruned_features, pruned_pos_embed, B, J)
        if x.shape[1] != self.frames:
            x = self._recover_temporal_tokens(x, self.frames)
        x = self.head(x)
        return x
