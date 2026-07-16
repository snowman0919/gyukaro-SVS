#!/usr/bin/env python3
"""Evaluate the local OpenUtau song render on its absolute 50 Hz timeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/rc9_reference"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data/cache/soulx-singer")]

from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402
from gyu_singer.inference.soulx import SoulXPhraseRenderer  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402


def pitch_metrics(observed: np.ndarray, target: np.ndarray) -> dict:
    both = (observed > 1) & (target > 1)
    cents = np.abs(1200 * np.log2(observed[both] / target[both]))
    correlation = np.corrcoef(np.log(observed[both]), np.log(target[both]))[0, 1] if both.sum() > 1 else np.nan
    return {
        "f0_correlation": round(float(correlation), 4) if np.isfinite(correlation) else None,
        "pitch_median_abs_cents": round(float(np.median(cents)), 2) if len(cents) else None,
        "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 2) if len(cents) else None,
        "gross_error_over_200_cents": round(float(np.mean(cents > 200)), 4) if len(cents) else None,
        "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 4) if len(cents) else None,
        "reference_voiced_recall": round(float(both.sum() / max(1, np.sum(target > 1))), 4),
        "rendered_voiced_precision": round(float(both.sum() / max(1, np.sum(observed > 1))), 4),
    }


def edge_error_ms(observed: np.ndarray, target: np.ndarray, first: bool) -> float | None:
    observed_edges = np.flatnonzero(observed > 1)
    target_edges = np.flatnonzero(target > 1)
    if not len(observed_edges) or not len(target_edges):
        return None
    index = 0 if first else -1
    return round(float((observed_edges[index] - target_edges[index]) * 20), 2)


def main() -> None:
    render = WORK / "openutau_render.wav"
    if not render.exists():
        raise FileNotFoundError("run scripts/render_reference_song_rc9.py first")
    project = yaml.safe_load((WORK / "nonbreath_oblige_gyu_rc9.ustx").read_text())
    requests = json.loads((WORK / "openutau_phrase_requests.json").read_text())
    reference_f0 = np.load(WORK / "consensus_f0_50hz.npy").astype("float32")
    extractor = F0Extractor(
        str(ROOT / "data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu", target_sr=24_000, hop_size=480, verbose=False,
    )
    observed = np.asarray(extractor.process(str(render), verbose=False), dtype="float32")
    count = min(len(observed), len(reference_f0))
    observed, reference_f0 = observed[:count], reference_f0[:count]
    bpm = float(project["tempos"][0]["bpm"])
    score_f0 = np.zeros(count, dtype="float32")
    for part, request in zip(project["voice_parts"], requests):
        start = round(float(part["position"]) / 480 * 60 / bpm * 50)
        duration = max(float(note["start"]) + float(note["duration"]) for note in request["notes"])
        local, _ = SoulXPhraseRenderer._canonical_f0(normalize_score(request), duration)
        end = min(count, start + len(local))
        score_f0[start:end] = local[:end - start]
    rows = []
    for index, (part, request) in enumerate(zip(project["voice_parts"], requests), 1):
        start = round(float(part["position"]) / 480 * 60 / bpm * 50)
        duration = max(float(note["start"]) + float(note["duration"]) for note in request["notes"])
        end = min(count, start + round(duration * 50))
        local_observed = observed[start:end]
        local_reference = reference_f0[start:end]
        local_score = score_f0[start:end]
        rows.append({
            "phrase": index,
            "language": request["language"],
            "start_seconds": round(start / 50, 3),
            "duration_seconds": round((end - start) / 50, 3),
            "onset_error_ms": edge_error_ms(local_observed, local_score, True),
            "offset_error_ms": edge_error_ms(local_observed, local_score, False),
            "reference_pitch": pitch_metrics(local_observed, local_reference),
            "score_pitch": pitch_metrics(local_observed, local_score),
        })
    valid_onsets = [abs(row["onset_error_ms"]) for row in rows if row["onset_error_ms"] is not None]
    score_both = (observed > 1) & (score_f0 > 1)
    score_union = (observed > 1) | (score_f0 > 1)
    overall = {
        "reference_pitch": pitch_metrics(observed, reference_f0),
        "score_pitch": pitch_metrics(observed, score_f0),
        "score_voicing_accuracy": round(float(np.mean((observed > 1) == (score_f0 > 1))), 4),
        "score_voiced_iou": round(float(score_both.sum() / max(1, score_union.sum())), 4),
        "phrase_onset_abs_error_median_ms": round(float(np.median(valid_onsets)), 2),
        "phrase_onset_abs_error_p90_ms": round(float(np.percentile(valid_onsets, 90)), 2),
    }
    reference_pitch = overall["reference_pitch"]
    score_pitch = overall["score_pitch"]
    gates = {
        "reference_f0_correlation_at_least_0_80": (reference_pitch["f0_correlation"] or 0) >= .80,
        "reference_pitch_median_below_50_cents": (reference_pitch["pitch_median_abs_cents"] or np.inf) < 50,
        "reference_pitch_p90_below_250_cents": (reference_pitch["pitch_p90_abs_cents"] or np.inf) < 250,
        "reference_gross_600_below_5_percent": (reference_pitch["gross_error_over_600_cents"] or 1) < .05,
        "score_pitch_p90_below_150_cents": (score_pitch["pitch_p90_abs_cents"] or np.inf) < 150,
        "score_voiced_iou_at_least_0_75": overall["score_voiced_iou"] >= .75,
        "phrase_onset_median_below_60_ms": overall["phrase_onset_abs_error_median_ms"] < 60,
    }
    report = {
        "status": "objective_pass_human_listening_pending" if all(gates.values()) else "objective_fail",
        "alignment": "fixed OpenUtau absolute phrase positions at 50 Hz; no per-phrase time stretching and no optimized lag; timing/voicing use the OpenUtau score while reference-pitch metrics use two-of-three estimator consensus",
        "overall": overall,
        "gates": gates,
        "phrases": rows,
        "local_audio": str(render.relative_to(ROOT)),
        "copyright": "local evaluation audio excluded from Git and package",
    }
    output = ROOT / "artifacts/reports/reference_song_rc9_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "overall": overall, "failed_gates": [key for key, value in gates.items() if not value]}, indent=2))
    if report["status"] == "objective_fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
