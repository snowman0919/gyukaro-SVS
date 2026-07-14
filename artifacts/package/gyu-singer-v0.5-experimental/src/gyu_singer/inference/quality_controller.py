"""Score-only expressive-F0 controller used by the quality neural decoder."""
from __future__ import annotations

from pathlib import Path

import torch

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.frontend import phonemize
from gyu_singer.model import TriSingerModel
from gyu_singer.score import normalize_score


STYLE = {"neutral": 0, "soft": 1, "breathy": 2, "energetic": 3, "dark": 4, "bright": 5, "tense": 6, "vibrato": 7}


def condition_batch(score: dict, reference_features: torch.Tensor, device: str) -> tuple[dict[str, torch.Tensor], float]:
    score = normalize_score(score)
    text = " ".join(note["lyric"] for note in score["notes"])
    frames = build_phrase_frames(phonemize(score["language"], text), score["notes"], score["curves"]["pitch"])
    controls = score["curves"]
    def mean(name: str, default: float = 0.0) -> float:
        points = controls[name]
        return float(sum(point["value"] for point in points) / len(points)) if points else default
    def frame(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = getattr(frames, name)
        if dtype: value = value.to(dtype)
        return value[None].to(device)
    batch = {"phoneme_ids": frame("phoneme_ids", torch.long), "language_ids": frame("language_ids", torch.long), "features": frame("features"),
             "midi": frame("midi"), "note_index": frame("note_index", torch.long), "note_onset": frame("note_onset"), "note_duration": frame("note_duration"), "boundary": frame("boundary"), "f0_hz": frame("f0_hz"), "voiced": frame("voiced"), "residual": frame("residual"),
             "reference_features": reference_features[None].to(device), "style_preset": torch.tensor([STYLE[score["style"]["preset"]]], device=device),
             "style_controls": torch.tensor([[mean("dynamics", .8), mean("breathiness"), mean("tension"), mean("brightness"), mean("vibrato")]], device=device)}
    return batch, max(note["start"] + note["duration"] for note in score["notes"])


class QualityPitchController:
    """TriSinger conditioner and residual flow, bounded before neural decoding."""
    def __init__(self, checkpoint: str | Path, reference_features: torch.Tensor, device: str | None = None):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_semitones = float(saved["max_semitones"])
        self.residual_scale = float(saved.get("residual_scale", 1.0))
        self.model = TriSingerModel(**saved["model_config"]).to(self.device).eval()
        # v0.4 checkpoints predate the v0.5 teacher-identity branch.
        self.model.load_state_dict(saved["model"], strict=False)
        self.reference_features = reference_features.float().to(self.device)

    @torch.no_grad()
    def predict(self, score: dict) -> tuple[torch.Tensor, float]:
        batch, duration = condition_batch(score, self.reference_features, self.device)
        # Production path is score-only supervised pitch head; flow remains acoustic-latent training evidence.
        source = self.model.acoustic_source(batch)
        output = self.model(torch.zeros_like(source), torch.zeros(1, device=self.device), batch)
        return torch.tanh(output["pitch_residual"][0] / max(self.max_semitones, 1e-6)) * self.max_semitones * self.residual_scale, duration
