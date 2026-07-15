#!/usr/bin/env python3
"""Convert the Korean stress scores to the official MLP Singer input format."""
from __future__ import annotations

import json
from pathlib import Path

import mido
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data/cache/mlp-singer"
MODEL_MIN_MIDI = 53
MODEL_MAX_MIDI = 77


def write_midi(path: Path, notes: list[dict], tempo: int, transpose: int) -> None:
    ticks_per_beat = 480
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    events = []
    for note in notes:
        pitch = note["pitch"] + transpose
        events.extend([
            (note["start"], 1, mido.Message("note_on", note=pitch, velocity=100, time=0)),
            (note["start"] + note["duration"], 0, mido.Message("note_off", note=pitch, velocity=0, time=0)),
        ])
    cursor = 0.0
    for seconds, _, message in sorted(events):
        message.time = round(mido.second2tick(seconds - cursor, ticks_per_beat, tempo))
        track.append(message)
        cursor = seconds
    midi.save(path)


def write_case(case: str) -> dict:
    score = json.loads((ROOT / f"examples/review_{case}.json").read_text())
    notes = score["notes"]
    low = min(note["pitch"] for note in notes)
    high = max(note["pitch"] for note in notes)
    transpose = min(0, MODEL_MAX_MIDI - high)
    transpose = max(transpose, MODEL_MIN_MIDI - low)
    if not all(MODEL_MIN_MIDI <= note["pitch"] + transpose <= MODEL_MAX_MIDI for note in notes):
        raise RuntimeError(f"{case}: pitch span exceeds the pretrained model range")

    raw = MODEL_ROOT / "data/probe"
    for folder in (raw / "mid", raw / "txt"):
        folder.mkdir(parents=True, exist_ok=True)
    (raw / "txt" / f"{case}.txt").write_text(
        "".join(note["lyric"] for note in notes) + "\n"
    )
    (raw / "txt" / f"{case}_gyu.txt").write_text(
        "".join(note["lyric"] for note in notes) + "\n"
    )
    tempo = mido.bpm2tempo(score["tempo"])
    write_midi(raw / "mid" / f"{case}.mid", notes, tempo, transpose)
    write_midi(raw / "mid" / f"{case}_gyu.mid", notes, tempo, 0)
    f0_dir = ROOT / "artifacts/reports/mlp_singer_korean_probe/f0"
    f0_dir.mkdir(parents=True, exist_ok=True)
    ds = json.loads(
        (ROOT / f"artifacts/reports/diffsinger_score_native_pilot/{case}.ds").read_text()
    )
    np.save(f0_dir / f"{case}.npy", np.asarray(ds[0]["f0_seq"].split(), dtype=np.float32))
    return {
        "case": case,
        "lyrics": "".join(note["lyric"] for note in notes),
        "transpose_semitones": transpose,
        "pitch_range": [low + transpose, high + transpose],
        "timing_source": "independent stress score",
        "soulx_target_f0": str((f0_dir / f"{case}.npy").relative_to(ROOT)),
    }


def main() -> None:
    rows = [write_case(case) for case in ("rapid_ko", "large_interval_ko")]
    report = {
        "status": "diagnostic_inputs_ready",
        "model": "neosapience/mlp-singer official pretrained checkpoint",
        "model_license": "MIT",
        "model_revision": "7f4621ca04ee5e35c0e0a80b1fed785a55a51891",
        "score_native": True,
        "identity": "generic CSD singer; not GYU",
        "rows": rows,
    }
    target = ROOT / "artifacts/reports/mlp_singer_korean_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
