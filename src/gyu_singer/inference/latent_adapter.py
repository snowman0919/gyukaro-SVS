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


class _SoulXFiLMPath(nn.Module):
    def __init__(self, condition_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU(), nn.Linear(128, hidden_dim * 2))
        self.gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition[None]
        gamma, beta = self.condition(condition).chunk(2, dim=-1)
        delta = hidden * (0.1 * torch.tanh(gamma)[:, None, :]) + 0.1 * torch.tanh(beta)[:, None, :]
        return hidden + torch.sigmoid(self.gate) * delta


class GYUIdentityAdapter(_SoulXFiLMPath):
    """Identity-only FiLM path trained on actual SoulX decoder inputs."""


class GYUStyleAdapter(_SoulXFiLMPath):
    """Style-only FiLM path trained on actual SoulX decoder inputs."""


class SoulXRealLatentAdapters(nn.Module):
    def __init__(self, identity_dim: int = 64, style_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.identity = GYUIdentityAdapter(identity_dim, hidden_dim)
        self.style = GYUStyleAdapter(style_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor | None, style: torch.Tensor | None) -> torch.Tensor:
        if identity is not None:
            hidden = self.identity(hidden, identity)
        if style is not None:
            hidden = self.style(hidden, style)
        return hidden
