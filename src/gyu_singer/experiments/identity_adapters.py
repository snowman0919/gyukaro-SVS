from __future__ import annotations

import torch
from torch import nn


class IdentityFiLM(nn.Module):
    def __init__(self, hidden_size: int, identity_size: int, limit: float = 0.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.limit = limit
        self.affine = nn.Linear(identity_size, hidden_size * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(identity).chunk(2, dim=-1)
        return hidden + self.limit * (
            torch.tanh(gamma).unsqueeze(1) * hidden + torch.tanh(beta).unsqueeze(1)
        )


class LowRankIdentityResidual(nn.Module):
    def __init__(self, hidden_size: int, identity_size: int, rank: int = 8, limit: float = 0.25):
        super().__init__()
        self.limit = limit
        self.down = nn.Linear(hidden_size + identity_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        conditioning = identity.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        residual = self.up(torch.tanh(self.down(torch.cat((hidden, conditioning), dim=-1))))
        return hidden + self.limit * torch.tanh(residual)
