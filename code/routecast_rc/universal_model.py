from __future__ import annotations
import torch
from torch import nn


class UniversalRouteCast(nn.Module):
    """Shared predictor for MoEs with different layer/expert counts and native K."""

    def __init__(self, history: int = 8, hidden: int = 32):
        super().__init__()
        self.history = history
        self.temporal = nn.GRU(1, hidden, batch_first=True)
        self.temporal_head = nn.Linear(hidden, 1)
        # context: three branch max/margin/entropy + layer/E/K descriptors
        self.gate = nn.Sequential(nn.Linear(12, 48), nn.GELU(), nn.Linear(48, 3))
        self.refine = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))

    @staticmethod
    def _logit_prob(x):
        return torch.logit(x.clamp(1e-5, 1 - 1e-5))

    @staticmethod
    def _summary(scores):
        p = scores.softmax(-1)
        top = p.topk(min(2, p.shape[1]), 1).values
        margin = top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0)
        entropy = -(p * p.clamp_min(1e-9).log()).sum(1) / torch.log(
            torch.tensor(float(p.shape[1]), device=p.device)
        )
        return torch.stack([top[:, 0], margin, entropy], 1)

    def forward(self, static, history, context):
        # static [B,E,2]: popularity and transition probability
        # history [B,H,E]: previous-token activation indicators
        b, h, e = history.shape
        seq = history.permute(0, 2, 1).reshape(b * e, h, 1)
        temporal = self.temporal_head(self.temporal(seq)[0][:, -1]).reshape(b, e)
        pop = self._logit_prob(static[:, :, 0])
        trans = self._logit_prob(static[:, :, 1])
        branches = torch.stack([pop, trans, temporal], -1)
        summary = torch.cat([self._summary(x) for x in (pop, trans, temporal)], 1)
        weights = self.gate(torch.cat([summary, context], 1)).softmax(-1)
        fused = (branches * weights.unsqueeze(1)).sum(-1)
        local = torch.cat([static, temporal.unsqueeze(-1), branches], -1)
        return fused + self.refine(local).squeeze(-1), weights

