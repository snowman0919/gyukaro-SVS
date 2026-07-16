#!/usr/bin/env python3
"""Measure lyric retention and unintended gaps on the local RC9 song."""
from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/rc9_reference"
sys.path.insert(0, str(ROOT / "scripts"))
from build_reference_ustx_rc9 import allocate, tokens  # noqa: E402


def normalized(text: str) -> str:
    return re.sub(r"[^A-Za-zぁ-んァ-ン一-龯]", "", text).lower()


def source_lines() -> tuple[list[str], list[tuple[float, float]], list[int]]:
    notes = json.loads((WORK / "note_candidates.json").read_text())
    notes = [row for row in notes if 7.5 <= row["start"] <= 205 and 52 <= row["pitch"] <= 96]
    lines = [line for line in (ROOT / "lyrics.txt").read_text().splitlines() if tokens(line)]
    counts = allocate([len(tokens(line)) for line in lines], len(notes))
    spans, ends, cursor = [], [], 0
    for count in counts:
        selected = notes[cursor:cursor + count]
        cursor += count
        spans.append((selected[0]["start"], selected[-1]["start"] + selected[-1]["duration"]))
        ends.append(cursor)
    return lines, spans, ends


def project_groups(ends: list[int]) -> list[tuple[int, int]]:
    project = yaml.safe_load((WORK / "nonbreath_oblige_gyu_rc9.ustx").read_text())
    groups, cursor, line = [], 0, 0
    for part in project["voice_parts"]:
        cursor += len(part["notes"])
        start = line
        while line < len(ends) and ends[line] <= cursor:
            line += 1
        groups.append((start, line))
    return groups


def transcribe(path: Path, lines: list[str], spans: list[tuple[float, float]], groups: list[tuple[int, int]], processor, model) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    rows = []
    for begin, end in groups:
        clip = audio[round(spans[begin][0] * rate):round(spans[end - 1][1] * rate)]
        divisor = np.gcd(rate, 16_000)
        clip = resample_poly(clip, 16_000 // divisor, rate // divisor).astype("float32")
        inputs = processor(clip, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(inputs.input_features.cuda().half(), language="ja", task="transcribe", max_new_tokens=256)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        expected = "".join(lines[begin:end])
        rows.append({
            "lines": [begin + 1, end],
            "similarity": round(SequenceMatcher(None, normalized(expected), normalized(transcript)).ratio(), 4),
            "weight": len(normalized(expected)),
        })
    return {
        "weighted_similarity": round(sum(row["similarity"] * row["weight"] for row in rows) / sum(row["weight"] for row in rows), 4),
        "median_similarity": round(float(np.median([row["similarity"] for row in rows])), 4),
        "groups": rows,
    }


def boundary_gaps(path: Path, spans: list[tuple[float, float]]) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    ratios = []
    for previous, following in zip(spans, spans[1:]):
        if following[0] - previous[1] > .1:
            continue
        center = round(following[0] * rate)
        near = audio[max(0, center - round(.03 * rate)):center + round(.03 * rate)]
        around = np.r_[
            audio[max(0, center - round(.2 * rate)):max(0, center - round(.06 * rate))],
            audio[center + round(.06 * rate):center + round(.2 * rate)],
        ]
        ratios.append(float(np.sqrt(np.mean(near ** 2) + 1e-12) / np.sqrt(np.mean(around ** 2) + 1e-12)))
    return {
        "near_contiguous_boundaries": len(ratios),
        "severe_energy_troughs_below_0_15": sum(value < .15 for value in ratios),
        "median_boundary_energy_ratio": round(float(np.median(ratios)), 4),
    }


def main() -> None:
    lines, spans, ends = source_lines()
    groups = project_groups(ends)
    paths = {
        "baseline": WORK / "rc9_listening_gate/01_full_openutau_rc9.wav",
        "candidate": WORK / "openutau_render.wav",
    }
    processor = AutoProcessor.from_pretrained(ROOT / "data/cache/whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ROOT / "data/cache/whisper-large-v3-turbo", torch_dtype=torch.float16).cuda().eval()
    variants = {
        name: {"audio": str(path.relative_to(ROOT)), "lyrics": transcribe(path, lines, spans, groups, processor, model),
               "continuity": boundary_gaps(path, spans)}
        for name, path in paths.items()
    }
    report = {
        "status": "human_listening_pending",
        "comparison_grid": "same inferred complete-lyric groups; Whisper large-v3-turbo is diagnostic, not human acceptance",
        "variants": variants,
        "copyright": "local evaluation audio and lyrics excluded from Git and package",
    }
    output = ROOT / "artifacts/reports/reference_song_rc9_continuity.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {"lyrics": row["lyrics"]["weighted_similarity"], **row["continuity"]}
                      for name, row in variants.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
