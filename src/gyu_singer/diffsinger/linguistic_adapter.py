from __future__ import annotations

import torch
from torch import nn


class KoreanLinguisticAdapter(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, foundation_size: int, limit: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.projection = nn.Linear(hidden_size, foundation_size, bias=False)
        self.limit = limit
        nn.init.zeros_(self.projection.weight)

    def forward(self, foundation_embedding: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return foundation_embedding + self.limit * torch.tanh(self.projection(self.embedding(token_ids)))


def freeze_for_linguistic_adaptation(foundation: nn.Module, adapter: KoreanLinguisticAdapter) -> None:
    foundation.requires_grad_(False)
    adapter.requires_grad_(True)
