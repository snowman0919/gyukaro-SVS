from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (predicted - target).square().mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1) if mask is not None else loss.mean()


def pitch_loss(predicted_log_f0: torch.Tensor, target_f0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
    target = target_f0.clamp_min(1.0).log()
    return (F.smooth_l1_loss(predicted_log_f0, target, reduction="none") * voiced).sum() / voiced.sum().clamp_min(1)


def teacher_distillation_loss(predicted: torch.Tensor, target: torch.Tensor, trust_weight: torch.Tensor) -> torch.Tensor:
    return ((predicted - target).square().mean(dim=-1) * trust_weight).sum() / trust_weight.sum().clamp_min(1e-6)
