"""RC5 engineering path: canonical voicing/timing and safer SoulX decode."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .soulx import _Worker
from .v08 import GyuSingerV08Renderer


class GyuSingerV09Renderer(GyuSingerV08Renderer):
    """Internal post-RC4 experiment; deliberately not exposed as a backend."""
    def __init__(self, reference: str | Path, root: str | Path = "."):
        super().__init__(reference, root=root)
        self.soulx.close()
        soulx = self.cache / "soulx-singer"
        command = [
            str(self.soulx_python), "scripts/probe_soulx_score.py", "--worker", "--precision", "fp32",
            "--reference", str(self.reference),
            "--model", str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"),
            "--config", str(soulx / "soulxsinger/config/soulxsinger.yaml"),
            "--rmvpe", str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
            "--latent-adapter", str(self.root / "checkpoints/gyu_real_latent_adapters_v0.7.pt"),
        ]
        self.soulx = _Worker(command, self.root, os.environ | {"GYU_SINGER_CACHE": str(self.cache)})

    def _predict_pitch(self, score: dict) -> torch.Tensor:
        return self.pitch_controller.predict(score, canonical_timing=True)[0]

    def _target_f0(self, score: dict, duration: float, expressive: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        return self._canonical_f0(score, duration, expressive)

    def _decoder_options(self) -> dict:
        return {"n_steps": 64, "cfg": 2.0, "seed": 21}

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-v0.9-timing-experiment",
            "model_version": "gyu-singer-v0.9-timing-experiment",
            "rc4_preserved_backend": "gyu-singer-v0.8",
            "canonical_phone_score_timeline": True,
            "unvoiced_f0_zero": True,
            "soulx_n_steps": 64,
            "soulx_cfg": 2.0,
            "soulx_precision": "fp32",
            "human_listening_status": "not_a_release_candidate",
        }
