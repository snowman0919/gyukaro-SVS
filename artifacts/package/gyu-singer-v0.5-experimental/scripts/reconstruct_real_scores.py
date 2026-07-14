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
        return {"id": row["id"], "language": row["language"], "score_source": "rmvpe_script_constrained_reconstruction", "label_status": "inferred", "script_shape_prior": shape, "f0_frame_hz": 12.5, "notes": notes, "f0_path": f"data/cache/hybrid_f0/{row['id']}.npy"}
    for unit, (start, end) in zip(units, viterbi(parts, len(units), shape)):
        group = parts[start:end] or [parts[min(start, len(parts) - 1)]]
        weights = np.array([right - left for left, right, _ in group])
        notes.append({"pitch": int(np.clip(round(np.average([value for _, _, value in group], weights=weights)), 36, 84)), "start": round(group[0][0] / 12.5, 4), "duration": round(max(1, group[-1][1] - group[0][0]) / 12.5, 4), "lyric": unit})
    return {"id": row["id"], "language": row["language"], "score_source": "rmvpe_script_constrained_reconstruction", "label_status": "inferred", "script_shape_prior": shape, "f0_frame_hz": 12.5, "notes": notes, "f0_path": f"data/cache/hybrid_f0/{row['id']}.npy"}


def quality(score: dict, row: dict) -> dict:
    """Score reconstruction confidence without treating inferred labels as truth."""
    f0 = np.load(score["f0_path"])
    voiced = f0 > 1
    if not voiced.any():
        return {"confidence": 0.0, "voiced_coverage": 0.0, "median_error_cents": None, "flags": ["no_voiced_frames"]}
    times = np.arange(len(f0), dtype=np.float32) / score["f0_frame_hz"]
    expected, observed = [], []
    for note in score["notes"]:
        active = voiced & (times >= note["start"]) & (times < note["start"] + note["duration"])
        if active.any():
            target = 440 * 2 ** ((note["pitch"] - 69) / 12)
            expected.extend([target] * int(active.sum()))
            observed.extend(f0[active].tolist())
    coverage = float(len(observed) / max(1, int(voiced.sum())))
    error = float(np.median(np.abs(1200 * np.log2(np.maximum(observed, 1) / np.maximum(expected, 1))))) if observed else 1200.0
    stability = float(np.clip(1.0 - np.std(1200 * np.log2(np.maximum(f0[voiced], 1) / np.median(f0[voiced]))) / 500.0, 0, 1))
    confidence = float(np.clip(.55 * coverage + .3 * np.clip(1 - error / 300, 0, 1) + .15 * stability, 0, 1))
    flags = []
    if coverage < .65: flags.append("low_voiced_coverage")
    if error > 250: flags.append("large_nominal_error_cents")
    if score["script_shape_prior"] == "free": flags.append("free_script_prior")
    return {"confidence": round(confidence, 4), "voiced_coverage": round(coverage, 4), "median_error_cents": round(error, 2), "flags": flags}


def main() -> None:
    source = read("data/manifests/neural_supervision.jsonl")
    scores = []
    for row in source:
        score = reconstruct(row, source)
        score["quality"] = quality(score, row)
        score["source_audio_path"] = row["audio_path"]
        score["text"] = row["text"]
        score["split"] = row.get("split", "train")
        scores.append(score)
    Path("data/manifests/reconstructed_real_scores.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scores))
    candidates = [{**x, "candidate_status": "review_required"} for x in scores]
    accepted = [{**x, "candidate_status": "accepted_inferred"} for x in scores if x["quality"]["confidence"] >= (.55 if x["script_shape_prior"] == "free" else .45) and x["quality"]["voiced_coverage"] >= .5]
    Path("data/manifests/real_score_candidates.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in candidates))
    Path("data/manifests/real_score_accepted.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in accepted))
    errors = [x["quality"]["median_error_cents"] for x in scores if x["quality"]["median_error_cents"] is not None]
    report = ["# Real score reconstruction (v0.5)", "", "Labels are inferred; no source annotation is claimed.", "", f"- candidates: {len(candidates)}", f"- accepted: {len(accepted)}", f"- median nominal-vs-RMVPE error (cents): {np.median(errors):.2f}", f"- p90 nominal-vs-RMVPE error (cents): {np.percentile(errors, 90):.2f}", "- source: `rmvpe_script_constrained_reconstruction`", "- fake base/base±1 pattern: not used by this manifest", ""]
    Path("artifacts/reports/real_score_review.md").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports/real_score_review.md").write_text("\n".join(report))
    Path("docs").mkdir(exist_ok=True)
    Path("docs/score_reconstruction_report.md").write_text("\n".join(report + ["", "Acceptance excludes low-confidence free-singing rows from v0.5 training. Every row retains `f0_path`, confidence, and quality flags for review."]))
    print({"candidates": len(candidates), "accepted": len(accepted), "p90_error_cents": round(float(np.percentile(errors, 90)), 2)})


if __name__ == "__main__": main()
