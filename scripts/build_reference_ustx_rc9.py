#!/usr/bin/env python3
"""Build the local-only editable RC9 OpenUtau score from measured reference data."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/rc9_reference"
PROJECT = WORK / "nonbreath_oblige_gyu_rc9.ustx"
REPORT = ROOT / "artifacts/reports/reference_song_rc9_project.json"
TICKS = 480


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(line: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[\u3040-\u30ff\u3400-\u9fff々]", line)


def allocate(weights: list[int], total: int) -> list[int]:
    raw = np.asarray(weights, dtype="float64") / sum(weights) * total
    result = np.floor(raw).astype(int)
    result[result == 0] = 1
    while result.sum() > total:
        candidates = np.flatnonzero(result > 1)
        result[candidates[np.argmin(raw[candidates] - result[candidates])]] -= 1
    for index in np.argsort(-(raw - result))[: total - result.sum()]:
        result[index] += 1
    return result.tolist()


def lyric_chunks(values: list[str], count: int) -> list[str]:
    edges = np.rint(np.linspace(0, len(values), count + 1)).astype(int)
    chunks = ["".join(values[edges[index]:edges[index + 1]]) for index in range(count)]
    previous = values[-1]
    for index, value in enumerate(chunks):
        if value:
            previous = value
        else:
            chunks[index] = "+" + previous[-1]
    return chunks


def vibrato(f0: np.ndarray, start: float, duration: float) -> dict:
    default = {"length": 0, "period": 180, "depth": 25, "in": 10, "out": 10, "shift": 0, "drift": 0, "vol_link": 0}
    if duration < .5:
        return default
    values = f0[round(start * 50):round((start + duration) * 50)]
    values = values[values > 0]
    if len(values) < 25:
        return default
    cents = 1200 * np.log2(values / np.median(values))
    cents -= np.polyval(np.polyfit(np.arange(len(cents)), cents, 1), np.arange(len(cents)))
    frequencies = np.fft.rfftfreq(len(cents), 1 / 50)
    spectrum = 2 * np.abs(np.fft.rfft(cents)) / len(cents)
    band = (frequencies >= 4) & (frequencies <= 8)
    if not np.any(band):
        return default
    index = np.flatnonzero(band)[np.argmax(spectrum[band])]
    if spectrum[index] < 8:
        return default
    return default | {
        "length": 60,
        "period": int(np.clip(round(1000 / frequencies[index]), 125, 250)),
        "depth": int(np.clip(round(spectrum[index]), 8, 100)),
    }


def note_dict(row: dict, position: int, duration: int, lyric: str, f0: np.ndarray) -> dict:
    return {
        "position": position, "duration": max(15, duration), "tone": row["pitch"], "lyric": lyric,
        "pitch": {"data": [{"x": -40, "y": 0, "shape": "io"}, {"x": 40, "y": 0, "shape": "io"}], "snap_first": True},
        "vibrato": vibrato(f0, row["start"], row["duration"]),
    }


def main() -> None:
    analysis = json.loads((ROOT / "artifacts/reports/reference_song_rc9_analysis.json").read_text())
    bpm = analysis["tempo"]["bpm_estimate"]
    notes = json.loads((WORK / "note_candidates.json").read_text())
    notes = [row for row in notes if 7.5 <= row["start"] <= 205 and 52 <= row["pitch"] <= 96]
    f0 = np.load(WORK / "consensus_f0_50hz.npy")
    lines = [tokens(line) for line in (ROOT / "lyrics.txt").read_text().splitlines() if tokens(line)]
    counts = allocate([len(line) for line in lines], len(notes))
    parts, boundaries = [], []
    cursor = 0
    pitch_points = vibrato_notes = 0
    seconds_to_ticks = bpm * TICKS / 60
    for line_index, (line, count) in enumerate(zip(lines, counts)):
        selected = notes[cursor:cursor + count]
        cursor += count
        chunks = lyric_chunks(line, count)
        absolute = [round(row["start"] * seconds_to_ticks) for row in selected]
        part_position = absolute[0]
        rendered_notes = []
        for index, (row, lyric) in enumerate(zip(selected, chunks)):
            measured_end = round((row["start"] + row["duration"]) * seconds_to_ticks)
            if index + 1 < len(selected):
                measured_end = absolute[index + 1]
            item = note_dict(row, absolute[index] - part_position, measured_end - absolute[index], lyric, f0)
            rendered_notes.append(item)
            vibrato_notes += item["vibrato"]["length"] > 0
        part_end = absolute[-1] + rendered_notes[-1]["duration"]
        xs, ys = [], []
        for frame in range(max(0, round(selected[0]["start"] * 50)), min(len(f0), round((selected[-1]["start"] + selected[-1]["duration"]) * 50))):
            if f0[frame] <= 0:
                continue
            tick = round(frame / 50 * seconds_to_ticks)
            active = next((row for row, start, note in zip(selected, absolute, rendered_notes)
                           if start <= tick < start + note["duration"]), None)
            if active is None:
                continue
            xs.append(tick - part_position)
            ys.append(int(np.clip(round(1200 * np.log2(f0[frame] / (440 * 2 ** ((active["pitch"] - 69) / 12)))), -300, 300)))
        pitch_points += len(xs)
        parts.append({
            "duration": part_end - part_position, "name": f"Reference phrase {line_index + 1:02d}",
            "track_no": 0, "position": part_position, "notes": rendered_notes,
            "curves": [{"abbr": "pitd", "xs": xs, "ys": ys}] if xs else [],
        })
        if line_index:
            previous_end = parts[-2]["position"] + parts[-2]["duration"]
            boundaries.append({"previous_part": line_index, "tick": part_position, "gap_seconds": round((part_position - previous_end) / seconds_to_ticks, 4)})
    project = {
        "name": "GYU RC9 local reference reconstruction",
        "comment": "Local evaluation only; score/lyrics are inferred from user-provided material and must not be packaged.",
        "ustx_version": "0.9",
        "time_signatures": [{"bar_position": 0, "beat_per_bar": 4, "beat_unit": 4}],
        "tempos": [{"position": 0, "bpm": bpm}],
        "tracks": [{"singer": "GYU-SINGER", "phonemizer": "OpenUtau.Core.DefaultPhonemizer",
                    "renderer_settings": {"renderer": "GYU-SINGER"}, "track_name": "GYU JA reference validation"}],
        "voice_parts": parts,
    }
    PROJECT.write_text(yaml.safe_dump(project, allow_unicode=True, sort_keys=False))
    loaded = yaml.safe_load(PROJECT.read_text())
    assert cursor == len(notes) and sum(len(part["notes"]) for part in loaded["voice_parts"]) == len(notes)
    report = {
        "status": "local_editable_project_built_openutau_validation_pending",
        "project": str(PROJECT.relative_to(ROOT)), "project_sha256": sha256(PROJECT),
        "distribution": "local evaluation only; project and lyrics excluded from Git and package",
        "score_labels": "inferred from two-of-three F0 consensus; not manual ground truth",
        "tempo_bpm": bpm, "parts": len(parts), "notes": len(notes), "lyrics_lines": len(lines),
        "generated_phonemes": "OpenUtau ValidateFull pending", "pitch_curve_points": pitch_points,
        "vibrato_notes": int(vibrato_notes), "phrase_boundaries": boundaries,
        "duration_seconds": round((parts[-1]["position"] + parts[-1]["duration"]) / seconds_to_ticks, 4),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "parts", "notes", "pitch_curve_points", "vibrato_notes", "duration_seconds")}, indent=2))


if __name__ == "__main__":
    main()
