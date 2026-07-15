"""Bounded score-timeline gate for stronger stationary spectral correction."""
from __future__ import annotations

import numpy as np

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.frontend import phonemize
from gyu_singer.score import normalize_score


def stationary_gate(score: dict, samples: int, sample_rate: int = 48_000) -> np.ndarray:
    score = normalize_score(score)
    # Measured only on sustained Korean vowels. Ordinary phrases lost lyric
    # identity when this stronger correction touched consonant transitions.
    if (
        score["language"] != "ko"
        or score["style"]["preset"] == "breathy"
        or max(note["duration"] for note in score["notes"]) < 2.5
    ):
        return np.zeros(samples, dtype="float32")
    text = " ".join(note["lyric"] for note in score["notes"])
    frames = build_phrase_frames(phonemize(score["language"], text), score["notes"], score["curves"]["pitch"], frame_hz=50, phoneme_alignment=score.get("phonemes"))
    f0 = frames.f0_hz.numpy()
    stable = f0 > 1
    change = np.full(len(f0), np.inf, dtype="float32")
    both = (f0[1:] > 1) & (f0[:-1] > 1)
    change[1:][both] = np.abs(1200 * np.log2(f0[1:][both] / f0[:-1][both]))
    stable &= change < 35
    boundary = np.r_[True, np.diff(frames.note_index.numpy()) != 0] | np.r_[True, np.diff(frames.voiced.numpy()) != 0]
    boundary = np.convolve(boundary.astype("float32"), np.ones(9), mode="same") > 0
    stable &= ~boundary
    keep = np.zeros_like(stable)
    edges = np.flatnonzero(np.diff(np.r_[False, stable, False]))
    for start, end in edges.reshape(-1, 2):
        if end - start >= 15:
            keep[start:end] = True
    positions = np.linspace(0, len(keep) - 1, samples)
    gate = np.interp(positions, np.arange(len(keep)), keep.astype("float32"))
    window = np.hanning(2 * round(.04 * sample_rate) + 1)
    return np.convolve(gate, window / window.sum(), mode="same").astype("float32")
