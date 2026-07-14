"""Phrase-level neural SVS runtime. Does not invoke baseline DSP vocalizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.data import acoustic_reference_features
from gyu_singer.frontend import phonemize
from gyu_singer.model import TriSingerModel
from gyu_singer.score import normalize_score

from .codec import MossCodecDecoder


def load_hybrid_model(checkpoint: str | Path, device: str | None = None) -> TriSingerModel:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved.get("model_config", {})
    model = TriSingerModel(**config)
    model.load_state_dict(saved["model"])
    model.checkpoint_path = str(checkpoint)
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()


class HybridRenderer:
    sample_rate = 48000

    def __init__(self, model: TriSingerModel, codec: MossCodecDecoder, reference_path: str | Path):
        self.model, self.codec = model, codec
        self.device = next(model.parameters()).device
        self.reference_features = acoustic_reference_features(reference_path).to(self.device)

    def model_info(self) -> dict:
        return {"backend": "hybrid-svs", "model_version": "gyu-hybrid-v0.2", "checkpoint": getattr(self.model, "checkpoint_path", "in-memory"), "languages": ["ko", "en", "ja"], "sample_rate": self.sample_rate}

    def batch(self, score: dict) -> dict[str, torch.Tensor]:
        score = normalize_score(score)
        text = " ".join(note["lyric"] for note in score["notes"])
        frames = build_phrase_frames(phonemize(score["language"], text), score["notes"], score["curves"]["pitch"])
        controls = score["curves"]
        def control(name: str, default: float = 0.0) -> float:
            values = controls[name]
            return float(sum(point["value"] for point in values) / len(values)) if values else default
        style = torch.tensor([
            control("dynamics", 0.8), control("breathiness"), control("tension"), control("brightness"), control("vibrato"),
        ], device=self.device)
        return {
            "phoneme_ids": frames.phoneme_ids[None].to(self.device), "language_ids": frames.language_ids[None].to(self.device),
            "features": frames.features[None].to(self.device), "midi": frames.midi[None].to(self.device),
            "note_index": frames.note_index[None].to(self.device), "boundary": frames.boundary[None].to(self.device),
            "note_onset": frames.note_onset[None].to(self.device), "note_duration": frames.note_duration[None].to(self.device),
            "f0_hz": frames.f0_hz[None].to(self.device), "voiced": frames.voiced[None].to(self.device),
            "residual": frames.residual[None].to(self.device), "reference_features": self.reference_features[None],
            "style_preset": torch.tensor([{"neutral": 0, "soft": 1, "breathy": 2, "energetic": 3, "dark": 4, "bright": 5, "tense": 6, "vibrato": 7}[score["style"]["preset"]]], device=self.device), "style_controls": style[None],
        }

    def render(self, score: dict) -> np.ndarray:
        """One conditional-flow pass over whole phrase, then frozen codec decode."""
        batch = self.batch(score)
        latent = self.model.sample(batch) + self.model.singing_decoder(self.model.condition(batch)[0])
        audio = self.codec.decode(latent)[0].numpy()
        return audio / max(1.0, float(np.abs(audio).max()) / 0.92)

    def render_file(self, input_path: str | Path, output_path: str | Path) -> None:
        score = json.loads(Path(input_path).read_text())
        sf.write(output_path, self.render(score), self.sample_rate, subtype="PCM_24")
