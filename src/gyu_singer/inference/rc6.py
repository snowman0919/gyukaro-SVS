"""Post-RC5 acoustic-quality candidate; human listening is still pending."""
from __future__ import annotations

import numpy as np

from .acoustic_refiner import AcousticRefinerRuntime
from .rc5 import GyuSingerRC5Renderer


class GyuSingerRC6Renderer(GyuSingerRC5Renderer):
    refiner_strength = .25

    def __init__(self, reference, root="."):
        super().__init__(reference, root=root)
        self.acoustic_refiner = AcousticRefinerRuntime(self.root / "checkpoints/acoustic_refiner_universal.pt", device=self.pitch_controller.device)

    def render(self, score: dict) -> np.ndarray:
        baseline = super().render(score)
        refined = self.acoustic_refiner.process(baseline)
        audio = baseline + self.refiner_strength * (refined - baseline)
        peak = float(np.max(np.abs(audio)))
        return audio * min(1.0, .97 / max(peak, 1e-8))

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-rc6",
            "model_version": "1.0.0-rc.6-candidate",
            "release_state": "objective candidate; human listening pending",
            "acoustic_refiner": "universal residual backbone",
            "acoustic_refiner_strength": self.refiner_strength,
            "singing_refiner_adapter": "measured but disabled by default: voicing regression",
            "gyu_refiner_adapter": "measured but disabled by default: no production advantage over universal",
            "final_v1_tagged": False,
        }
