#!/usr/bin/env python3
"""Authoritative v0.6 prosody evaluation on independent PyIN-reviewed scores."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from gyu_singer.data import acoustic_reference_features
from gyu_singer.inference.quality_controller import QualityPitchController


def nominal(score: dict, times: np.ndarray) -> np.ndarray:
    notes = score["notes"]; output = []
    for time in times:
        note = next((note for note in notes if note["start"] <= time < note["start"] + note["duration"]), notes[-1])
        output.append(440 * 2 ** ((note["pitch"] - 69) / 12))
    return np.asarray(output, dtype="float32")


def vibrato(f0: np.ndarray, voiced: np.ndarray, rate: float = 12.5) -> tuple[float, float]:
    values = f0[voiced]
    if len(values) < 10: return 0.0, 0.0
    cents = 1200 * np.log2(np.maximum(values, 1) / np.median(values)); crossings = np.sum(np.diff(np.signbit(cents)) != 0)
    return float(crossings / 2 / (len(values) / rate)), float(np.percentile(cents, 95) - np.percentile(cents, 5))


def score_model(rows: list[dict], controller: QualityPitchController | None) -> list[dict]:
    output = []
    for row in rows:
        notes = [dict(note) for note in row["notes"]]; cursor = float(notes[0]["start"])
        for note in notes:
            note["start"] = cursor; cursor += float(note["duration"])
        score = {"language": row["language"], "tempo": 120, "notes": notes}; target = np.load(f"data/cache/hybrid_f0/{row['id']}.npy"); times = np.arange(len(target), dtype="float32") / 12.5; base = nominal(score, times); predicted = base
        if controller:
            residual, duration = controller.predict(score); residual = residual.cpu().numpy(); predicted = base * 2 ** (np.interp(times, np.linspace(0, duration, len(residual)), residual) / 12)
        voiced = target > 1; frame_error = 1200 * np.log2(np.maximum(predicted, 1) / np.maximum(target, 1)); error_cents = frame_error[voiced]; log_rmse = float(np.sqrt(np.mean((np.log(np.maximum(predicted[voiced], 1)) - np.log(np.maximum(target[voiced], 1))) ** 2)))
        onset = []
        transitions = []
        for index, note in enumerate(score["notes"]):
            active = voiced & (times >= note["start"]) & (times < note["start"] + min(.12, note["duration"]))
            if active.any(): onset.append(float(np.median(frame_error[active])) / 1.0)
            if index and active.any(): transitions.extend(frame_error[active].tolist())
        true_vibrato = vibrato(target, voiced); pred_vibrato = vibrato(predicted, voiced)
        output.append({"id": row["id"], "f0_correlation": round(float(np.corrcoef(predicted[voiced], target[voiced])[0, 1]), 4), "pitch_mae_cents": round(float(np.median(np.abs(error_cents))), 2), "log_f0_rmse": round(log_rmse, 5), "note_onset_residual_cents": round(float(np.median(np.abs(onset))), 2) if onset else None, "transition_contour_error_cents": round(float(np.median(np.abs(transitions))), 2) if transitions else None, "vibrato_rate_error_hz": round(abs(pred_vibrato[0] - true_vibrato[0]), 4), "vibrato_extent_error_cents": round(abs(pred_vibrato[1] - true_vibrato[1]), 2)})
    return output


def aggregate(rows: list[dict]) -> dict:
    numeric = {key: float(np.mean([row[key] for row in rows if row[key] is not None])) for key in rows[0] if key != "id" and all(isinstance(row[key], (int, float)) for row in rows if row[key] is not None)}
    return {key: round(value, 4) for key, value in numeric.items()}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--v06-checkpoint", default=None); args = parser.parse_args()
    source = [json.loads(line) for line in Path("data/manifests/manual_verified_scores.jsonl").read_text().splitlines() if line]
    reference = acoustic_reference_features("data/processed/master/216.wav")
    runs = {"nominal_verified_score": score_model(source, None), "v0.4_synthetic_controller": score_model(source, QualityPitchController("checkpoints/gyu_quality_pitch_controller.pt", reference)), "v0.5_real_gyu_controller": score_model(source, QualityPitchController("checkpoints/gyu_prosody_v0.5.pt", reference))}
    if args.v06_checkpoint:
        runs["v0.6_verified_plus_reconstructed_controller"] = score_model(source, QualityPitchController(args.v06_checkpoint, reference))
    report = {"score_set": "data/manifests/manual_verified_scores.jsonl", "rows": len(source), "score_independent_from_target_f0": all(row["verification"]["score_independent_from_target_f0"] for row in source), "runs": {name: {"aggregate": aggregate(values), "rows": values} for name, values in runs.items()}}
    Path("artifacts/reports/independent_prosody_evaluation.json").write_text(json.dumps(report, indent=2) + "\n"); Path("docs/independent_prosody_evaluation.md").write_text("# Independent prosody evaluation (v0.6)\n\n" + json.dumps(report, indent=2) + "\n")
    print({name: value["aggregate"] for name, value in report["runs"].items()})


if __name__ == "__main__": main()
