#!/usr/bin/env python3
"""Causal held-out comparison of universal, singing, and GYU refiners."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime
from evaluate_rc4_artifact_matrix import acoustics


def audio48(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    return resample_poly(mono, 48000, rate).astype("float32") if rate != 48000 else mono


def target_metrics(output: np.ndarray, target: np.ndarray) -> dict:
    length = min(len(output), len(target)); output, target = output[:length], target[:length]
    target *= np.sqrt(np.mean(output**2) + 1e-8) / np.sqrt(np.mean(target**2) + 1e-8)
    target *= min(1.0, .97 / max(float(np.max(np.abs(target))), 1e-8))
    _, _, estimate = stft(output, 48000, nperseg=1024, noverlap=768, boundary=None)
    _, _, truth = stft(target, 48000, nperseg=1024, noverlap=768, boundary=None)
    estimate, truth = np.abs(estimate) + 1e-5, np.abs(truth) + 1e-5
    log_l1 = float(np.mean(np.abs(np.log(estimate) - np.log(truth))))
    convergence = float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), 1e-8))
    frequencies = np.fft.rfftfreq(1024, 1 / 48000); high = frequencies >= 8000
    high_log_l1 = float(np.mean(np.abs(np.log(estimate[high]) - np.log(truth[high]))))
    frame = 240; usable = length // frame * frame
    envelope = float(np.mean(np.abs(np.mean(np.abs(output[:usable]).reshape(-1, frame), 1) - np.mean(np.abs(target[:usable]).reshape(-1, frame), 1))))
    return {"target_log_spectral_l1": round(log_l1, 6), "target_spectral_convergence": round(convergence, 6),
            "target_highband_log_l1": round(high_log_l1, 6), "target_envelope_l1": round(envelope, 6)}


def main() -> None:
    root = Path("artifacts/reports/acoustic_refiner_pair_evaluation"); audio_root = root / "audio"; audio_root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path("data/external/manifests/pipeline_degradation_pairs.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if row["split"] == "test"]
    refiners = {stage: AcousticRefinerRuntime(f"checkpoints/acoustic_refiner_{stage}.pt") for stage in ("universal", "singing", "gyu")}
    results = []
    for index, row in enumerate(rows, 1):
        source, target = audio48(row["degraded_input"]), audio48(row["clean_target"])
        variants = {"input": source}
        for stage, refiner in refiners.items():
            refined = refiner.process(source)
            for strength in (.25, .5, .75, 1.0):
                variants[f"{stage}_{round(strength * 100):03d}"] = source + strength * (refined - source)
        for variant, audio in variants.items():
            path = audio_root / f"{row['id']}_{variant}.wav"; sf.write(path, audio, 48000, subtype="PCM_24")
            results.append({"id": row["id"], "dataset": row["dataset"], "domain": row["domain"], "variant": variant, "path": str(path),
                            "residual_l1_from_input": round(float(np.mean(np.abs(audio - source))), 7)} | target_metrics(audio.copy(), target.copy()) | acoustics(path))
        print(f"{index}/{len(rows)} {row['id']}", flush=True)
    keys = ("target_log_spectral_l1", "target_spectral_convergence", "target_highband_log_l1", "target_envelope_l1", "hf_energy_ratio_p95", "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999", "clip_fraction", "residual_l1_from_input")
    aggregate = defaultdict(dict)
    for dataset in ("libritts_r", "vocalset", "real_gyu"):
        for variant in ("input",) + tuple(f"{stage}_{strength:03d}" for stage in ("universal", "singing", "gyu") for strength in (25, 50, 75, 100)):
            selected = [row for row in results if row["dataset"] == dataset and row["variant"] == variant]
            aggregate[dataset][variant] = {key: round(float(np.mean([row[key] for row in selected])), 6) for key in keys}
    report = {"status": "objective_diagnostic_human_listening_not_performed", "test_rows": len(rows), "aggregate": dict(aggregate), "rows": results,
              "promotion_rule": "a stage must improve its held-out target domain without material speech, F0, content, speaker, or listening regression"}
    root.mkdir(parents=True, exist_ok=True); (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
