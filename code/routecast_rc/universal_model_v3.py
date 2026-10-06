from __future__ import annotations
import math
import torch
from torch import nn


class UniversalRouteCastV3(nn.Module):
    """One structurally-conditioned scorer shared by heterogeneous MoEs."""
    BRANCH_NAMES = ('popularity','one_step','two_hop','prefill','temporal','multiscale')

    def __init__(self, history: int = 16, hidden: int = 24, context_dim: int = 5, disabled_branches=()):
        super().__init__(); self.history=history
        unknown=set(disabled_branches)-set(self.BRANCH_NAMES)
        if unknown: raise ValueError(f'unknown disabled branches: {sorted(unknown)}')
        self.disabled_branches=tuple(sorted(set(disabled_branches)))
        self.enabled=tuple(name not in self.disabled_branches for name in self.BRANCH_NAMES)
        self.temporal=nn.GRU(1,hidden,batch_first=True); self.temporal_head=nn.Linear(hidden,1)
        self.multiscale=nn.Sequential(nn.Linear(6,24),nn.GELU(),nn.Linear(24,1))
        # six branches x (max, margin, entropy), plus structural context
        self.gate=nn.Sequential(nn.Linear(18+context_dim,64),nn.GELU(),nn.Linear(64,6))
        self.refine=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _logit(x): return torch.logit(x.clamp(1e-5,1-1e-5))
    @staticmethod
    def _summary(s):
        p=s.softmax(1); top=p.topk(min(2,p.shape[1]),1).values
        margin=top[:,0]-(top[:,1] if top.shape[1]>1 else 0)
        ent=-(p*p.clamp_min(1e-9).log()).sum(1)/math.log(p.shape[1])
        return torch.stack([top[:,0],margin,ent],1)
    def forward(self,static,history,context):
        b,h,e=history.shape
        if self.enabled[4]:
            seq=history.permute(0,2,1).reshape(b*e,h,1)
            temporal=self.temporal_head(self.temporal(seq)[0][:,-1]).reshape(b,e)
        else: temporal=torch.zeros((b,e),device=history.device,dtype=history.dtype)
        if self.enabled[5]:
            windows=[]
            for w in (1,2,4,8,h): windows.append(history[:,-min(w,h):].mean(1))
            idx=torch.arange(1,h+1,device=history.device,dtype=history.dtype).view(1,h,1)
            recency=(history*idx).amax(1)/h
            multi=self.multiscale(torch.stack([*windows,recency],-1)).squeeze(-1)
        else: multi=torch.zeros((b,e),device=history.device,dtype=history.dtype)
        uniform=torch.full((b,e),1/e,device=static.device,dtype=static.dtype)
        static_parts=[static[:,:,i] if self.enabled[i] else uniform for i in range(4)]
        static_effective=torch.stack(static_parts,-1)
        prior=[self._logit(static_parts[i]) for i in range(4)]
        branches=torch.stack([*prior,temporal,multi],-1)
        summaries=torch.cat([self._summary(branches[:,:,i]) for i in range(6)],1)
        gate_logits=self.gate(torch.cat([summaries,context],1))
        enabled=torch.tensor(self.enabled,device=gate_logits.device,dtype=torch.bool).view(1,-1)
        # -1e4 is effectively zero after softmax and remains representable in FP16.
        weights=gate_logits.masked_fill(~enabled,-1e4).softmax(-1)
        # Probability-mixture fusion is stable when branch score scales differ.
        logp=torch.stack([branches[:,:,i].log_softmax(1) for i in range(6)],-1)
        fused=torch.logsumexp(logp+weights.clamp_min(1e-8).log().unsqueeze(1),-1)
        correction=self.refine(torch.cat([static_effective,temporal.unsqueeze(-1),multi.unsqueeze(-1),branches[:,:,:4]],-1)).squeeze(-1)
        return fused+0.1*correction,weights
