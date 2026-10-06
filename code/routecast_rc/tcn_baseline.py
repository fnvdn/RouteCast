"""History-only causal TCN baseline shared across heterogeneous MoE models."""
from __future__ import annotations
import torch
from torch import nn

class CausalBlock(nn.Module):
    def __init__(self,channels:int,dilation:int):
        super().__init__(); self.trim=2*dilation
        self.conv=nn.Conv1d(channels,channels,3,padding=self.trim,dilation=dilation)
        self.norm=nn.GroupNorm(1,channels); self.act=nn.GELU()
    def forward(self,x):
        y=self.conv(x); y=y[...,:-self.trim] if self.trim else y
        return self.act(self.norm(y)+x)

class HistoryTCN(nn.Module):
    """Scores each expert from only its own binary activation history."""
    def __init__(self,history:int=16,channels:int=24):
        super().__init__(); self.history=history
        self.input=nn.Conv1d(1,channels,1)
        self.blocks=nn.Sequential(*(CausalBlock(channels,d) for d in (1,2,4,8)))
        self.head=nn.Linear(channels,1)
    def forward(self,static,history,context):
        b,h,e=history.shape; x=history.permute(0,2,1).reshape(b*e,1,h)
        z=self.blocks(self.input(x))[...,-1]; scores=self.head(z).reshape(b,e)
        return scores,None
