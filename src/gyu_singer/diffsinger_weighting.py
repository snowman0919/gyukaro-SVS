"""Small loss helpers for the local DiffSinger training probes."""
from __future__ import annotations

import torch


def weighted_frame_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_token_ids: torch.Tensor,
    low_weight_ids: set[int],
    consonant_weight: float = 5.0,
) -> torch.Tensor:
    """Weight consonant frames without changing padding or vowel reconstruction."""
    valid = frame_token_ids > 0
    low_weight = torch.zeros_like(valid)
    for token_id in low_weight_ids:
        low_weight |= frame_token_ids == token_id
    weights = torch.where(low_weight, 1.0, consonant_weight) * valid
    frame_error = (prediction - target).abs().mean(dim=-1)
    return (frame_error * weights).sum() / weights.sum().clamp_min(1)
