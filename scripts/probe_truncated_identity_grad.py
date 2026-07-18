#!/usr/bin/env python3
"""Bounded final-WAV gradient feasibility probe for the v0.7 identity adapter."""
from __future__ import annotations

import torch
from torch import nn


def diffusion_step(flow, state, prompt, condition, index: int, total_steps: int, cfg: float):
    prompt_len = prompt.shape[1]
    target_len = state.shape[1]
    step = 1.0 / total_steps
    mask = torch.ones(condition.shape[0], condition.shape[1], device=condition.device)
    target_mask = torch.ones(condition.shape[0], target_len, device=condition.device)
    time = ((index + 0.5) * step) * torch.ones(
        state.shape[0], dtype=state.dtype, device=state.device
    )
    predicted = flow.diff_estimator(torch.cat([prompt, state], 1), time, condition, mask)
    predicted = predicted[:, prompt_len:, :]
    if cfg > 0:
        unconditioned = flow.diff_estimator(
            state, time, torch.zeros_like(condition)[:, :target_len, :], target_mask
        )
        predicted_std = predicted.std()
        guided = predicted + cfg * (predicted - unconditioned)
        rescaled = guided * predicted_std / guided.std()
        predicted = 0.75 * rescaled + 0.25 * guided
    return state + predicted * step


def full_reverse_diffusion(flow, prompt, condition, total_steps: int, cfg: float):
    target_len = condition.shape[1] - prompt.shape[1]
    state = torch.randn(
        condition.shape[0], target_len, flow.mel_dim,
        dtype=condition.dtype, device=condition.device,
    )
    with torch.no_grad():
        for index in range(total_steps):
            state = diffusion_step(flow, state, prompt, condition, index, total_steps, cfg)
    return state


def truncated_reverse_diffusion(
    flow, prompt, cond_frozen, cond_trainable, total_steps: int, grad_steps: int, cfg: float
):
    if not 1 <= grad_steps <= total_steps:
        raise ValueError("grad_steps must be within total_steps")
    target_len = cond_frozen.shape[1] - prompt.shape[1]
    state = torch.randn(
        cond_frozen.shape[0], target_len, flow.mel_dim,
        dtype=cond_frozen.dtype, device=cond_frozen.device,
    )
    with torch.no_grad():
        for index in range(total_steps - grad_steps):
            state = diffusion_step(flow, state, prompt, cond_frozen, index, total_steps, cfg)
    state = state.detach()
    for index in range(total_steps - grad_steps, total_steps):
        state = diffusion_step(flow, state, prompt, cond_trainable, index, total_steps, cfg)
    return state


class ToyFlow(nn.Module):
    mel_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def diff_estimator(self, state, time, condition, mask):
        del mask
        return self.projection(state) + 0.1 * condition + time[:, None, None]
