"""v0.7 renderer using separately trained real-latent identity/style paths."""
from __future__ import annotations

from pathlib import Path

import torch

from .v06 import GyuSingerV06Renderer


class GyuSingerV07Renderer(GyuSingerV06Renderer):
    # Sign-only calibration selected on quality_ko causal output proxies.
    STYLE_CALIBRATION = {"energetic": -0.5, "dark": -2.0}
    STYLE_SEMANTICS = {"neutral": "neutral", "soft": "relative_style_A_unverified", "breathy": "breathiness_proxy", "energetic": "energy_proxy", "dark": "relative_style_C_unverified", "bright": "relative_style_B_unverified"}

    def __init__(self, reference: str | Path, root: str | Path = ".", latent_adapter_enabled: bool = True):
        super().__init__(reference, root=root, latent_adapter_enabled=latent_adapter_enabled, latent_adapter_checkpoint="checkpoints/gyu_real_latent_adapters_v0.7.pt")

    def model_info(self) -> dict:
        info = super().model_info()
        return info | {"backend": "gyu-singer-v0.7", "model_version": "gyu-singer-v0.7-experimental", "latent_adapter_checkpoint": "checkpoints/gyu_real_latent_adapters_v0.7.pt", "latent_training": "actual SoulXSingerSVC.infer_segment.gt_decoder_inp", "identity_style_separable": True, "style_calibration": self.STYLE_CALIBRATION, "style_semantics": self.STYLE_SEMANTICS}

    def _style_vector(self, style: dict, device: torch.device) -> torch.Tensor:
        vector = super()._style_vector(style, device)
        neutral = super()._style_vector(style | {"preset": "neutral"}, device)
        return neutral + self.STYLE_CALIBRATION.get(style["preset"], 1.0) * (vector - neutral)

    def _content_style_preset(self, style: dict) -> str:
        # Hold the source/content path constant: v0.7 style acts inside SoulX.
        return "neutral"
