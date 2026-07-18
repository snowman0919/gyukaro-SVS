"""RC8 quality candidate: local timing fixes and stationary spectral gating."""
from __future__ import annotations

import numpy as np

from .rc6 import GyuSingerRC6Renderer
from .spectral_refiner import SpectralRefinerRuntime


class GyuSingerRC8Renderer(GyuSingerRC6Renderer):
    def __init__(self, reference, root="."):
        super().__init__(reference, root=root)
        self.spectral_refiner = SpectralRefinerRuntime(self.root / "checkpoints/acoustic_refiner_spectral_singing.pt", device=self.pitch_controller.device)

    def _decoder_options(self, score: dict | None = None) -> dict:
        if (
            score
            and score["language"] == "ko"
            and score["style"]["preset"] == "neutral"
            and not self._rapid(score)
            and not self._large_interval(score)
            and max(note["duration"] for note in score["notes"]) >= 2.5
        ):
            return {"n_steps": 64, "cfg": 1.5, "seed": 21}
        return super()._decoder_options(score)

    def render(self, score: dict) -> np.ndarray:
        baseline = super().render(score)
        refined = self.spectral_refiner.process(baseline)
        audio = baseline + .5 * (refined - baseline)
        return audio * min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-rc8", "model_version": "1.0.0-rc.8-candidate",
            "spectral_refiner": "checkpoints/acoustic_refiner_spectral_singing.pt",
            "spectral_strength": "fixed 0.5; rejected stronger stationary correction is disabled",
            "sustained_decoder": "64 steps / CFG 1.5 only for neutral KO notes >=2.5 s",
            "english_ay_voiced": True, "japanese_note_lexicon_fixed": True,
            "normal_ko_ctc_warp_strength": .05, "large_interval_steps": 50,
            "rapid_ko_path_changed": False, "final_v1_tagged": False,
        }
