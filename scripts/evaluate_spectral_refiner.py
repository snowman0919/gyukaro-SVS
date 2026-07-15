#!/usr/bin/env python3
"""Held-out comparison of waveform-TCN and aligned spectral-mask refiners."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_acoustic_refiner_pairs import audio48, target_metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics  # noqa: E402
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402
from gyu_singer.inference.spectral_refiner import SpectralRefinerRuntime as SpectralRuntime  # noqa: E402
from train_spectral_refiner import alignment  # noqa: E402


def aligned(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    lag, _ = alignment(source, target)
    if lag > 0:
        target = target[lag:]
    elif lag < 0:
        source = source[-lag:]
    length = min(len(source), len(target))
    return source[:length], target[:length], lag


def main() -> None:
    manifests = (
        ROOT / "data/external/manifests/pipeline_degradation_pairs_libri_v2.jsonl",
        ROOT / "data/external/manifests/pipeline_degradation_pairs_v2.jsonl",
        ROOT / "data/external/manifests/pipeline_degradation_pairs.jsonl",
    )
    rows = []
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row["split"] != "test":
                continue
            if manifest.name == "pipeline_degradation_pairs.jsonl" and row["dataset"] != "real_gyu":
                continue
            rows.append(row)
    old = AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_singing_v2.pt")
    refiners = {
        "spectral_universal": SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_universal.pt"),
        "spectral_singing": SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt"),
        "spectral_gyu": SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_gyu.pt"),
    }
    results = []
    with tempfile.TemporaryDirectory(prefix="gyu-spectral-refiner-") as temporary:
        for index, row in enumerate(rows, 1):
            raw_source, raw_target = audio48(row["degraded_input"]), audio48(row["clean_target"])
            variants = {"input": raw_source, "waveform_singing_v2_025": raw_source + 0.25 * (old.process(raw_source) - raw_source)}
            for name, refiner in refiners.items():
                refined = refiner.process(raw_source)
                for strength in (0.1, 0.25, 0.5, 1.0):
                    variants[f"{name}_{round(strength * 100):03d}"] = raw_source + strength * (refined - raw_source)
            for variant, raw_audio in variants.items():
                audio, target, lag = aligned(raw_audio, raw_target)
                path = Path(temporary) / f"{index}_{variant}.wav"
                sf.write(path, audio, 48000, subtype="FLOAT")
                results.append({
                    "id": row["id"], "dataset": row["dataset"], "variant": variant,
                    "alignment_lag_samples": lag,
                    "residual_l1_from_input": round(float(np.mean(np.abs(raw_audio - raw_source))), 7),
                } | target_metrics(audio.copy(), target.copy()) | acoustics(path))
            print(f"{index}/{len(rows)} {row['id']}", flush=True)
    keys = (
        "target_log_spectral_l1", "target_spectral_convergence", "target_highband_log_l1",
        "target_envelope_l1", "hf_spike_p99_over_median", "spectral_flux_p95",
        "sample_jump_p999", "clip_fraction", "residual_l1_from_input",
    )
    variants = ("input", "waveform_singing_v2_025") + tuple(
        f"{name}_{strength:03d}"
        for name in refiners for strength in (10, 25, 50, 100)
    )
    aggregate = defaultdict(dict)
    for dataset in ("libritts_r", "vocalset", "real_gyu"):
        for variant in variants:
            selected = [row for row in results if row["dataset"] == dataset and row["variant"] == variant]
            aggregate[dataset][variant] = {
                key: round(float(np.mean([row[key] for row in selected])), 6) for key in keys
            }
    report = {
        "status": "objective_held_out_evaluation",
        "test_rows": len(rows), "speaker_disjoint_from_training": True,
        "target_alignment": "10 ms RMS envelope cross-correlation within +/-500 ms",
        "aggregate": dict(aggregate),
        "rows": [
            row for row in results
            if row["variant"] in {
                "input", "waveform_singing_v2_025", "spectral_singing_025", "spectral_gyu_025"
            }
        ],
        "human_listening": "not_requested_until_RC_stress_improves",
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/acoustic_refiner_spectral_evaluation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
