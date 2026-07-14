"""Deterministic phrase-level phoneme/note frame alignment."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
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
    note_onset: torch.Tensor
    note_duration: torch.Tensor
    boundary: torch.Tensor


def build_phrase_frames(frontend: FrontendOutput, notes: list[dict], pitch_curve: list[dict] | None = None, frame_hz: float = 12.5) -> PhraseFrames:
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
    note_onset = torch.zeros(frames)
    note_duration = torch.zeros(frames)
    boundary = torch.zeros(frames)
    for index, note in enumerate(ordered):
        start = int(round(float(note.get("start", note.get("start_sec", 0))) * frame_hz))
        end = max(start + 1, int(round((float(note.get("start", note.get("start_sec", 0))) + float(note.get("duration", note.get("duration_sec", 0)))) * frame_hz)))
        end = min(frames, end)
        midi[start:end] = float(note["pitch"])
        note_index[start:end] = index
        note_onset[start:end] = float(note.get("start", note.get("start_sec", 0))) / max(total, 1e-3)
        note_duration[start:end] = float(note.get("duration", note.get("duration_sec", 0))) / max(total, 1e-3)
        # A slur carries articulation across note onset instead of creating a hard boundary.
        boundary[start] = 0.0 if note.get("slur", False) else 1.0
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
    residual = torch.zeros(frames)
    if pitch_curve:
        times = torch.arange(frames) / frame_hz
        points_t = torch.tensor([point["time"] for point in pitch_curve])
        points_v = torch.tensor([point["value"] for point in pitch_curve])
        residual = torch.from_numpy(np.interp(times.numpy(), points_t.numpy(), points_v.numpy()).astype("float32"))
    f0_hz = 440.0 * torch.pow(torch.tensor(2.0), (midi + residual - 69.0) / 12.0)
    return PhraseFrames(
        phoneme_ids=phoneme_ids,
        language_ids=language_ids,
        features=features,
        midi=midi,
        f0_hz=f0_hz,
        voiced=torch.ones(frames),
        residual=residual,
        note_index=note_index,
        note_onset=note_onset,
        note_duration=note_duration,
        boundary=boundary,
    )
