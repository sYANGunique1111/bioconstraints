import torch
import torch.nn as nn


class TokenSelector(nn.Module):
    """Learned temporal token selector with context-aware scoring and a bounded residual gate."""

    def __init__(self, embed_dim, hidden_dim=None, drop=0.0, gate_scale=0.1, eps=1e-6):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.score_norm = nn.LayerNorm(embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_scale = gate_scale
        self.eps = eps

    def _aggregate_token_features(self, tokens):
        if tokens.dim() == 4:
            return tokens.mean(dim=2)
        if tokens.dim() == 3:
            return tokens
        raise ValueError(f"tokens must be 3D or 4D, got shape {tuple(tokens.shape)}")

    def _gather_tokens(self, tokens, indices):
        batch_size, keep_tokens = indices.shape
        if tokens.dim() == 4:
            _, _, num_joints, channels = tokens.shape
            gather_idx = indices[:, :, None, None].expand(batch_size, keep_tokens, num_joints, channels)
        else:
            _, _, channels = tokens.shape
            gather_idx = indices[:, :, None].expand(batch_size, keep_tokens, channels)
        return torch.gather(tokens, 1, gather_idx)

    def forward(self, tokens, keep_tokens):
        batch_size, num_tokens = tokens.shape[:2]
        if keep_tokens <= 0:
            raise ValueError(f"keep_tokens must be positive, got {keep_tokens}")
        if keep_tokens > num_tokens:
            raise ValueError(f"keep_tokens ({keep_tokens}) must be <= current token count ({num_tokens})")
        if keep_tokens == num_tokens:
            indices = torch.arange(num_tokens, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
            score_tokens = self._aggregate_token_features(tokens)
            scores = self.scorer(self.score_norm(score_tokens)).squeeze(-1)
            return tokens, indices, scores

        score_tokens = self._aggregate_token_features(tokens)
        score_tokens = self.score_norm(score_tokens)
        global_context = score_tokens.mean(dim=1, keepdim=True)
        scores = self.scorer(score_tokens + global_context).squeeze(-1)

        score_mean = scores.mean(dim=1, keepdim=True)
        score_std = scores.std(dim=1, keepdim=True, unbiased=False)
        scores = (scores - score_mean) / (score_std + self.eps)

        topk_scores, topk_indices = torch.topk(scores, k=keep_tokens, dim=1)
        sorted_indices, sort_order = torch.sort(topk_indices, dim=1)
        sorted_scores = torch.gather(topk_scores, 1, sort_order)

        selected_tokens = self._gather_tokens(tokens, sorted_indices)
        gate = torch.sigmoid(sorted_scores)
        while gate.dim() < selected_tokens.dim():
            gate = gate.unsqueeze(-1)
        selected_tokens = selected_tokens * (1.0 + self.gate_scale * gate)

        return selected_tokens, sorted_indices, scores
