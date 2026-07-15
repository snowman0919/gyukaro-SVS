"""Time-frequency refiner trained only on real pipeline degradation pairs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralAdapter(nn.Module):
    def __init__(self, channels: int, rank: int):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1)
        self.up = nn.Conv2d(rank, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class SpectralBlock(nn.Module):
    def __init__(self, channels: int, time_dilation: int, adapter_rank: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, (3, 5), padding=(1, 2 * time_dilation),
            dilation=(1, time_dilation), groups=channels,
        )
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels * 2, channels, 1)
        self.singing_adapter = SpectralAdapter(channels, adapter_rank)
        self.gyu_adapter = SpectralAdapter(channels, adapter_rank)

    def forward(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        hidden = self.depthwise(self.norm(value))
        value = value + 0.2 * self.project(F.gelu(self.expand(hidden)))
        if mode in {"singing", "gyu"}:
            value = value + self.singing_adapter(value)
        if mode == "gyu":
            value = value + self.gyu_adapter(value)
        return value


class SpectralAcousticRefiner(nn.Module):
    """Identity-initialized STFT-mask U-Net with domain-specific adapters."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        channels: int = 16,
        bottleneck_channels: int = 64,
        blocks: int = 6,
        adapter_rank: int = 8,
        max_log_gain: float = 0.8,
    ):
        super().__init__()
        if bottleneck_channels != channels * 4:
            raise ValueError("bottleneck_channels must equal channels * 4")
        self.config = {
            "n_fft": n_fft, "hop_length": hop_length, "channels": channels,
            "bottleneck_channels": bottleneck_channels, "blocks": blocks,
            "adapter_rank": adapter_rank, "max_log_gain": max_log_gain,
        }
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_log_gain = max_log_gain
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(4, channels), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(channels * 2, bottleneck_channels, 3, stride=(2, 1), padding=1),
            nn.GroupNorm(8, bottleneck_channels), nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            SpectralBlock(bottleneck_channels, 2 ** (index % 4), adapter_rank)
            for index in range(blocks)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels + channels * 2, channels * 2, 3, padding=1),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.GELU(),
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.GELU(),
        )
        self.mask = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)

    @staticmethod
    def _resize(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, audio: torch.Tensor, mode: str = "universal") -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError("audio must have shape [batch, samples]")
        spectrum = torch.stft(
            audio, self.n_fft, self.hop_length, window=self.window,
            center=True, pad_mode="constant", return_complex=True,
        )
        feature = torch.log1p(spectrum.abs())[:, None]
        e1 = self.enc1(feature)
        e2 = self.enc2(e1)
        hidden = self.enc3(e2)
        for block in self.blocks:
            hidden = block(hidden, mode)
        hidden = self.dec2(torch.cat((self._resize(hidden, e2), e2), dim=1))
        hidden = self.dec1(torch.cat((self._resize(hidden, e1), e1), dim=1))
        hidden = self.dec0(torch.cat((self._resize(hidden, feature), feature), dim=1))
        log_gain = self.max_log_gain * torch.tanh(self.mask(hidden).squeeze(1))
        refined = torch.istft(
            spectrum * torch.exp(log_gain), self.n_fft, self.hop_length,
            window=self.window, center=True, length=audio.shape[-1],
        )
        activity = F.avg_pool1d(audio.abs()[:, None], 481, stride=1, padding=240).squeeze(1)
        gate = (activity / 0.02).clamp(0, 1)
        return audio + gate * (refined - audio)

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
