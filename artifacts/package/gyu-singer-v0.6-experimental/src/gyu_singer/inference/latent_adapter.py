"""Small FiLM adapter injected into SoulX's phrase conditioning latent."""
from __future__ import annotations

import torch
from torch import nn


class SoulXLatentAdapter(nn.Module):
    def __init__(self, identity_dim: int = 64, style_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(identity_dim + style_dim, 256), nn.SiLU(), nn.Linear(256, hidden_dim * 2))
        self.gate = nn.Parameter(torch.tensor(-2.2))

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if identity.ndim == 1: identity = identity[None]
        if style.ndim == 1: style = style[None]
        gamma, beta = self.condition(torch.cat([identity, style], dim=-1)).chunk(2, dim=-1)
        gamma = 0.05 * torch.tanh(gamma)[:, None, :]
        beta = 0.05 * torch.tanh(beta)[:, None, :]
        return hidden + torch.sigmoid(self.gate) * (hidden * gamma + beta)
