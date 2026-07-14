"""v0.8 production selection: v0.5 prosody plus v0.7 real-latent adapters."""
from __future__ import annotations

from pathlib import Path

from .v07 import GyuSingerV07Renderer


class GyuSingerV08Renderer(GyuSingerV07Renderer):
    def __init__(self, reference: str | Path, root: str | Path = "."):
        super().__init__(reference, root=root)

    def model_info(self) -> dict:
        return super().model_info() | {
            "backend": "gyu-singer-v0.8",
            "model_version": "gyu-singer-v0.8",
            "production_prosody": "checkpoints/gyu_prosody_v0.5.pt",
            "selection_basis": "24-row independent score metrics plus KO/EN/JA causal identity and style ablations",
            "per_note_tts": False,
            "waveform_pitch_shift": False,
            "phase_vocoder_note_control": False,
        }
