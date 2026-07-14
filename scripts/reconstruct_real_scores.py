#!/usr/bin/env python3
"""Reconstruct inferred score notes from RMVPE and recording-script priors."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import medfilt


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def note_segments(f0: np.ndarray) -> list[tuple[int, int, float]]:
    voiced = f0 > 1
    if voiced.sum() < 3: return []
    contour = 69 + 12 * np.log2(np.maximum(f0, 1) / 440)
    contour[~voiced] = np.interp(np.flatnonzero(~voiced), np.flatnonzero(voiced), contour[voiced])
    labels = np.rint(medfilt(contour, 5)).astype(int)
    parts: list[tuple[int, int, float]] = []
    start = None
    for index, active in enumerate(voiced):
        if active and start is None: start = index
        if start is not None and (not active or index == len(f0) - 1):
            end = index if not active else index + 1
            cursor = start
            for change in np.r_[np.where(np.diff(labels[start:end]) != 0)[0] + 1, end - start]:
                right = start + change
                if right > cursor: parts.append((cursor, right, float(np.median(labels[cursor:right]))))
                cursor = right
            start = None
    merged: list[tuple[int, int, float]] = []
    for part in parts:
        if merged and part[1] - part[0] < 2:
            left = merged.pop(); weight = (left[1] - left[0]) + (part[1] - part[0])
            merged.append((left[0], part[1], (left[2] * (left[1] - left[0]) + part[2] * (part[1] - part[0])) / weight))
        else: merged.append(part)
    return merged


def script_shape(row: dict, all_rows: list[dict]) -> str:
    index = int(row["id"].rsplit("_", 1)[1])
    peers = sorted(int(peer["id"].rsplit("_", 1)[1]) for peer in all_rows if peer["text"] == row["text"] and 158 <= int(peer["id"].rsplit("_", 1)[1]) <= 211)
    return ("same", "ascending", "descending")[peers.index(index)] if len(peers) == 3 else "free"


def viterbi(parts: list[tuple[int, int, float]], count: int, shape: str) -> list[tuple[int, int]]:
    n = len(parts)
    if n < count:
        return [(min(n - 1, round(i * n / count)), max(1, round((i + 1) * n / count))) for i in range(count)]
    duration = np.array([right - left for left, right, _ in parts], float); pitch = np.array([value for _, _, value in parts])
    prefix = np.r_[0, np.cumsum(duration)]; center = np.median(pitch)
    cost = np.full((count + 1, n + 1), np.inf); back = np.zeros((count + 1, n + 1), int); cost[0, 0] = 0
    for syllable in range(1, count + 1):
        for end in range(syllable, n - (count - syllable) + 1):
            for start in range(syllable - 1, end):
                group, weight = pitch[start:end], duration[start:end]
                mean = np.average(group, weights=weight)
                prior = abs(mean - center) if shape == "same" else (-mean if shape == "ascending" else mean if shape == "descending" else 0)
                value = cost[syllable - 1, start] + 2 * abs((prefix[end] - prefix[start]) / prefix[-1] - 1 / count) + .1 * np.std(group) + .02 * prior
                if value < cost[syllable, end]: cost[syllable, end], back[syllable, end] = value, start
    end, groups = n, []
    for syllable in range(count, 0, -1):
        start = back[syllable, end]; groups.append((start, end)); end = start
    return groups[::-1]


def reconstruct(row: dict, all_rows: list[dict]) -> dict:
    f0 = np.load(f"data/cache/hybrid_f0/{row['id']}.npy"); parts = note_segments(f0)
    units = re.findall(r"[가-힣]", row["text"]) or ["아"]
    if not parts: raise RuntimeError(f"no voiced RMVPE frames: {row['id']}")
    shape, notes = script_shape(row, all_rows), []
    if len(parts) < len(units):
        boundaries = np.linspace(parts[0][0], parts[-1][1], len(units) + 1).round().astype(int)
        for unit, start, end in zip(units, boundaries[:-1], boundaries[1:]):
            center = (start + end) / 2
            closest = min(parts, key=lambda part: abs((part[0] + part[1]) / 2 - center))
            notes.append({"pitch": int(np.clip(round(closest[2]), 36, 84)), "start": round(start / 12.5, 4), "duration": round(max(1, end - start) / 12.5, 4), "lyric": unit})
        return {"id": row["id"], "language": row["language"], "score_source": "inferred_rmvpe_piecewise_constant_plus_script_prior", "script_shape_prior": shape, "f0_frame_hz": 12.5, "notes": notes}
    for unit, (start, end) in zip(units, viterbi(parts, len(units), shape)):
        group = parts[start:end] or [parts[min(start, len(parts) - 1)]]
        weights = np.array([right - left for left, right, _ in group])
        notes.append({"pitch": int(np.clip(round(np.average([value for _, _, value in group], weights=weights)), 36, 84)), "start": round(group[0][0] / 12.5, 4), "duration": round(max(1, group[-1][1] - group[0][0]) / 12.5, 4), "lyric": unit})
    return {"id": row["id"], "language": row["language"], "score_source": "inferred_rmvpe_piecewise_constant_plus_script_prior", "script_shape_prior": shape, "f0_frame_hz": 12.5, "notes": notes}


def main() -> None:
    source = read("data/manifests/neural_supervision.jsonl")
    Path("data/manifests/reconstructed_real_scores.jsonl").write_text("".join(json.dumps(reconstruct(row, source), ensure_ascii=False) + "\n" for row in source))
    print({"rows": len(source)})


if __name__ == "__main__": main()
