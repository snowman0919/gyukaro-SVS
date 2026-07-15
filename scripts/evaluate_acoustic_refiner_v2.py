#!/usr/bin/env python3
"""Held-out speaker evaluation for expanded real-pipeline refiner pairs."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_acoustic_refiner_pairs import audio48, target_metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics  # noqa: E402
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402


def main() -> None:
    manifests = (
        ROOT / "data/external/manifests/pipeline_degradation_pairs_libri_v2.jsonl",
        ROOT / "data/external/manifests/pipeline_degradation_pairs_v2.jsonl",
    )
    rows = [
        json.loads(line)
        for manifest in manifests
        for line in manifest.read_text().splitlines()
        if line and json.loads(line)["split"] == "test"
    ]
    refiners = {
        "universal_v2": AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_universal_v2.pt"),
        "singing_v2": AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_singing_v2.pt"),
    }
    results = []
    with tempfile.TemporaryDirectory(prefix="gyu-refiner-v2-") as temporary:
        for index, row in enumerate(rows, 1):
            source, target = audio48(row["degraded_input"]), audio48(row["clean_target"])
            variants = {"input": source}
            for stage, refiner in refiners.items():
                refined = refiner.process(source)
                for strength in (.25, .5, .75, 1.0):
                    variants[f"{stage}_{round(strength * 100):03d}"] = source + strength * (refined - source)
            for variant, audio in variants.items():
                path = Path(temporary) / f"{index}_{variant}.wav"
                sf.write(path, audio, 48000, subtype="FLOAT")
                results.append({
                    "id": row["id"], "dataset": row["dataset"], "variant": variant,
                    "residual_l1_from_input": round(float(np.mean(np.abs(audio - source))), 7),
                } | target_metrics(audio.copy(), target.copy()) | acoustics(path))
            print(f"{index}/{len(rows)} {row['id']}", flush=True)
    keys = (
        "target_log_spectral_l1", "target_spectral_convergence", "target_highband_log_l1",
        "target_envelope_l1", "hf_spike_p99_over_median", "spectral_flux_p95",
        "sample_jump_p999", "clip_fraction", "residual_l1_from_input",
    )
    aggregate = defaultdict(dict)
    variants = ("input",) + tuple(
        f"{stage}_{strength:03d}"
        for stage in refiners for strength in (25, 50, 75, 100)
    )
    for dataset in ("libritts_r", "vocalset"):
        for variant in variants:
            selected = [row for row in results if row["dataset"] == dataset and row["variant"] == variant]
            aggregate[dataset][variant] = {
                key: round(float(np.mean([row[key] for row in selected])), 6) for key in keys
            }
    report = {
        "status": "objective_held_out_evaluation", "test_rows": len(rows),
        "speaker_disjoint_from_training": True, "aggregate": dict(aggregate),
        "rows": [
            row for row in results
            if row["variant"] in {"input", "singing_v2_025"}
        ],
        "row_policy": "sample-wise input and selected 25% singing adapter only; full strength sweep retained in aggregates",
        "human_listening": "not_requested_until_RC_stress_improves",
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/acoustic_refiner_v2_evaluation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
