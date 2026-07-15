"""Small residual waveform refiner with frozen-backbone domain adapters."""
from __future__ import annotations

import torch
from torch import nn


class LowRankAudioAdapter(nn.Module):
    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.down = nn.Conv1d(channels, rank, 1)
        self.up = nn.Conv1d(rank, channels, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(torch.nn.functional.gelu(self.down(value)))


class RefinerBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, adapter_rank: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, groups=channels)
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.project = nn.Conv1d(channels * 2, channels, 1)
        self.singing_adapter = LowRankAudioAdapter(channels, adapter_rank)
        self.gyu_adapter = LowRankAudioAdapter(channels, adapter_rank)

    def forward(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        hidden = self.depthwise(self.norm(value))
        hidden = self.project(torch.nn.functional.gelu(self.expand(hidden)))
        value = value + .2 * hidden
        if mode in {"singing", "gyu"}:
            value = value + self.singing_adapter(value)
        if mode == "gyu":
            value = value + self.gyu_adapter(value)
        return value


class VocalAcousticRefiner(nn.Module):
    """Identity-initialized refiner; output cannot exceed a bounded correction."""
    def __init__(self, channels: int = 32, blocks: int = 9, adapter_rank: int = 8, max_residual: float = .12):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks, "adapter_rank": adapter_rank, "max_residual": max_residual}
        self.max_residual = max_residual
        self.input = nn.Conv1d(1, channels, 15, padding=7)
        self.blocks = nn.ModuleList(RefinerBlock(channels, 2 ** (index % 7), adapter_rank) for index in range(blocks))
        self.output = nn.Conv1d(channels, 1, 15, padding=7)
        nn.init.normal_(self.output.weight, std=1e-4); nn.init.zeros_(self.output.bias)

    def forward(self, audio: torch.Tensor, mode: str = "universal") -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio[:, None]
        hidden = self.input(audio)
        for block in self.blocks:
            hidden = block(hidden, mode)
        correction = self.max_residual * torch.tanh(self.output(hidden))
        return (audio + correction).clamp(-1, 1).squeeze(1)

    def train_stage(self, stage: str) -> int:
        for name, parameter in self.named_parameters():
            if stage == "universal":
                parameter.requires_grad = "_adapter" not in name
            elif stage == "singing":
                parameter.requires_grad = "singing_adapter" in name
            elif stage == "gyu":
                parameter.requires_grad = "gyu_adapter" in name
            else:
                raise ValueError(stage)
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

