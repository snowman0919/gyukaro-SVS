"""Compact neural acoustic-style adapter applied before the singing decoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import stft, istft


class GyuAcousticStyleAdapter(nn.Module):
    """Predict a spectral log-gain representation from timbre and style controls."""

    def __init__(self, feature_dim: int = 160, output_bins: int = 257):
        super().__init__()
        self.output_bins = output_bins
        self.network = nn.Sequential(nn.Linear(feature_dim + 13, 96), nn.SiLU(), nn.Linear(96, output_bins), nn.Tanh())

    def forward(self, reference_features: torch.Tensor, style_controls: torch.Tensor, preset: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(preset.long().clamp(0, 7), 8).to(reference_features.dtype)
        return self.network(torch.cat((reference_features, style_controls, one_hot), dim=-1))


def load_adapter(checkpoint: str | Path, device: str = "cpu") -> GyuAcousticStyleAdapter:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter = GyuAcousticStyleAdapter(**saved.get("config", {})).to(device).eval()
    adapter.load_state_dict(saved["model"])
    return adapter


def adapt_waveform(audio: np.ndarray, rate: int, adapter: GyuAcousticStyleAdapter, reference_features: torch.Tensor, style_controls: torch.Tensor, preset: torch.Tensor, strength: float = 1.0) -> np.ndarray:
    """Apply predicted log spectral gain to content; SoulX consumes this adapted content."""
    if strength == 0 or len(audio) < 512:
        return audio.astype(np.float32, copy=False)
    _, _, z = stft(audio.astype(np.float32), fs=rate, nperseg=512, noverlap=384, boundary="zeros")
    with torch.inference_mode():
        gain = adapter(reference_features[None], style_controls[None], preset[None])[0].cpu().numpy()
    gain = np.exp(gain * float(np.clip(strength, 0, 2)) * 0.25)
    gain = np.interp(np.arange(z.shape[0]), np.linspace(0, z.shape[0] - 1, len(gain)), gain)
    _, output = istft(z * gain[:, None], fs=rate, nperseg=512, noverlap=384, input_onesided=True)
    return output[: len(audio)].astype(np.float32)
