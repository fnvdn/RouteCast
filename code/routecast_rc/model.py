from __future__ import annotations

import torch
from torch import nn


class SharedExpertScorer(nn.Module):
    """Permutation-equivariant scorer for variable-size expert sets.

    The same MLP is applied independently to every candidate expert. Expert IDs
    are represented only through route-derived features, so permuting expert
    indices permutes outputs without changing the decision rule.
    """

    def __init__(self, expert_dim: int, context_dim: int = 0,
                 hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.expert_dim = expert_dim
        self.context_dim = context_dim
        in_dim = expert_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, expert_features: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if expert_features.ndim != 3:
            raise ValueError("expert_features must have shape [batch, experts, features]")
        if expert_features.shape[-1] != self.expert_dim:
            raise ValueError(f"expected {self.expert_dim} expert features")
        x = expert_features
        if self.context_dim:
            if context is None or context.shape[-1] != self.context_dim:
                raise ValueError("context is required and has the wrong width")
            x = torch.cat([x, context.unsqueeze(1).expand(-1, x.shape[1], -1)], -1)
        return self.net(x).squeeze(-1)


def temperature_scaled_ranking_weights(scores: torch.Tensor,
                                       temperature: float = 1.0):
    """Return unit-sum ranking weights after temperature scaling.

    The weights summarize relative score concentration. They are not
    categorical probabilities when multiple experts can be correct.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(scores / temperature, dim=-1)


def calibrated_probabilities(scores: torch.Tensor, temperature: float = 1.0):
    """Backward-compatible alias for temperature-scaled ranking weights."""
    return temperature_scaled_ranking_weights(scores, temperature)


def cumulative_mass_set(probabilities: torch.Tensor, mass: float,
                        max_k: int | None = None) -> torch.Tensor:
    """Return a mask using the smallest prefix reaching normalized score mass."""
    if not 0 < mass <= 1:
        raise ValueError("mass must be in (0, 1]")
    e = probabilities.shape[-1]
    k_limit = e if max_k is None else min(max_k, e)
    values, indices = probabilities.sort(dim=-1, descending=True)
    prefix = values[..., :k_limit].cumsum(-1)
    k = (prefix < mass).sum(-1).clamp(max=k_limit - 1) + 1
    ranks = torch.arange(k_limit, device=probabilities.device)
    selected_sorted = ranks.unsqueeze(0) < k.unsqueeze(-1)
    selected = torch.zeros_like(probabilities, dtype=torch.bool)
    selected.scatter_(1, indices[..., :k_limit], selected_sorted)
    return selected
