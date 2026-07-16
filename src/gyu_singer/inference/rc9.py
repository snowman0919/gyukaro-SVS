"""RC9 OpenUtau song candidate with independently controlled non-KO pitch."""
from __future__ import annotations

import torch

from .rc8 import GyuSingerRC8Renderer


class GyuSingerRC9Renderer(GyuSingerRC8Renderer):
    """Keep RC8 audio fixes while making EN/JA score and PITD authoritative."""

    def _predict_pitch(self, score: dict) -> torch.Tensor:
        predicted = self.pitch_controller.predict(score, canonical_timing=True)[0]
        # The personalized residual has real GYU singing supervision only in
        # Korean. EN/JA retain score/PITD F0 and SoulX's generic singing prior.
        return predicted if score["language"] == "ko" else torch.zeros_like(predicted)

    @classmethod
    def _content_warp_strength(cls, score: dict) -> float:
        # Japanese MMS alignment is not reliable enough for dense song lyrics.
        if cls._rapid(score) and score["language"] == "ja":
            return 0.0
        return super()._content_warp_strength(score)

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-rc9",
            "model_version": "1.0.0-rc.9-candidate",
            "rc8_baseline_backend": "gyu-singer-rc8",
            "personalized_prosody": "Korean only; EN/JA use nominal score plus user PITD",
            "rapid_japanese_content_warp": "disabled after local full-song isolation",
            "release_state": "OpenUtau song candidate; human listening pending",
            "final_v1_tagged": False,
        }
