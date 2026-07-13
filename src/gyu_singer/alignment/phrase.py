"""Deterministic phrase-level phoneme/note frame alignment."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from gyu_singer.frontend import FrontendOutput, phonemize


@dataclass
class PhraseFrames:
    phoneme_ids: torch.Tensor
    language_ids: torch.Tensor
    features: torch.Tensor
    midi: torch.Tensor
    f0_hz: torch.Tensor
    voiced: torch.Tensor
    residual: torch.Tensor
    note_index: torch.Tensor
    boundary: torch.Tensor


def build_phrase_frames(frontend: FrontendOutput, notes: list[dict], frame_hz: float = 12.5) -> PhraseFrames:
    if not notes:
        raise ValueError("notes must not be empty")
    ordered = sorted(notes, key=lambda note: float(note.get("start", note.get("start_sec", 0))))
    total = max(float(note.get("start", note.get("start_sec", 0))) + float(note.get("duration", note.get("duration_sec", 0))) for note in ordered)
    frames = max(1, math.ceil(total * frame_hz))
    phoneme_ids = torch.zeros(frames, dtype=torch.long)
    language_ids = torch.zeros(frames, dtype=torch.long)
    features = torch.zeros(frames, len(frontend.features[0]))
    midi = torch.zeros(frames)
    note_index = torch.zeros(frames, dtype=torch.long)
    boundary = torch.zeros(frames)
    for index, note in enumerate(ordered):
        start = int(round(float(note.get("start", note.get("start_sec", 0))) * frame_hz))
        end = max(start + 1, int(round((float(note.get("start", note.get("start_sec", 0))) + float(note.get("duration", note.get("duration_sec", 0)))) * frame_hz)))
        end = min(frames, end)
        midi[start:end] = float(note["pitch"])
        note_index[start:end] = index
        boundary[start] = 1.0
        # Score lyric owns its note frames; no phrase-wide character stretching.
        lyric = str(note.get("lyric", ""))
        units = phonemize(frontend.language, lyric) if lyric.strip() else frontend
        positions = torch.linspace(0, len(units.phoneme_ids) - 1, end - start).round().long()
        phoneme_ids[start:end] = torch.tensor(units.phoneme_ids)[positions]
        language_ids[start:end] = torch.tensor(units.language_ids)[positions]
        features[start:end] = torch.tensor(units.features, dtype=torch.float32)[positions]
    # Silence gaps inherit nearest preceding content but remain unvoiced.
    for index in range(1, frames):
        if phoneme_ids[index] == 0:
            phoneme_ids[index] = phoneme_ids[index - 1]
            language_ids[index] = language_ids[index - 1]
            features[index] = features[index - 1]
    f0_hz = 440.0 * torch.pow(torch.tensor(2.0), (midi - 69.0) / 12.0)
    return PhraseFrames(
        phoneme_ids=phoneme_ids,
        language_ids=language_ids,
        features=features,
        midi=midi,
        f0_hz=f0_hz,
        voiced=torch.ones(frames),
        residual=torch.zeros(frames),
        note_index=note_index,
        boundary=boundary,
    )
