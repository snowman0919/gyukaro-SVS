#!/usr/bin/env python3
"""Run the frozen, release-ineligible GTSinger-to-GYU diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gyu_singer.alignment.phrase import build_phrase_frames  # noqa: E402
from gyu_singer.frontend import phonemize  # noqa: E402


CASES = (
    "quality_ko",
    "rapid_ko",
    "large_interval_ko",
    "sustain_ko",
    "phrase_boundary_ko",
)
CASE_PATHS = {
    "quality_ko": "examples/quality_ko.json",
    "rapid_ko": "examples/review_rapid_ko.json",
    "large_interval_ko": "examples/review_large_interval_ko.json",
    "sustain_ko": "examples/review_sustain_ko.json",
    "phrase_boundary_ko": "examples/review_phrase_boundary_ko.json",
}
STRESS = {
    "quality_ko": "ordinary",
    "rapid_ko": "rapid",
    "large_interval_ko": "large_interval",
    "sustain_ko": "sustain",
    "phrase_boundary_ko": "phrase_boundary",
}
SEEDS = (7, 21, 42)
REFERENCES = tuple(f"data/processed/master/{value}.wav" for value in (212, 215, 216, 219, 220))
WORK = ROOT / "data/external/work/gtsinger_gyu_identity_diagnostic"
REPORT = ROOT / "artifacts/reports/gtsinger_gyu_identity_diagnostic"
MANIFEST = ROOT / "data/manifests/gtsinger_gyu_identity_protocol.json"
SOURCE_CHECKPOINT = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt"
INIT_CHECKPOINT = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity_init.ckpt"
VOCODER = ROOT / "data/external/work/diffsinger_score_native/vocoder_exported/model.ckpt"
SOURCE_SHA256 = "dd31b42469ef2caa307799212b30fa44b2f1b7186c2f3a14eae45a2a80a6da8a"
VOCODER_SHA256 = "0b6728a7e677afdf0d1abc8d1fc1ac376631f6055062d2578db7d8ae4ba24729"
DIFFSINGER_REVISION = "0619d61d5301c4340db442a15cf3e73e197e9101"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build_score_ds(score: dict) -> tuple[dict, dict]:
    """Convert one tracked KO score to one inferred phrase-level DiffSinger row."""
    text = "".join(note["lyric"] for note in score["notes"])
    frames = build_phrase_frames(phonemize(score["language"], text), score["notes"], frame_hz=50)
    content = sorted(frames.phoneme_durations, key=lambda row: row["start_frame"])
    phones, cursor, silence_frames = [], 0, 0
    for phone in content:
        start = int(phone["start_frame"])
        if start > cursor:
            phones.append({"symbol": "SP", "start_frame": cursor, "duration_frames": start - cursor})
            silence_frames += start - cursor
        phones.append(phone)
        cursor = start + int(phone["duration_frames"])
    if cursor < len(frames.f0_hz):
        phones.append({"symbol": "SP", "start_frame": cursor,
                       "duration_frames": len(frames.f0_hz) - cursor})
        silence_frames += len(frames.f0_hz) - cursor
    if not phones or sum(int(phone["duration_frames"]) for phone in phones) != len(frames.f0_hz):
        raise ValueError("phoneme durations do not cover the score F0 grid")
    row = {
        "offset": 0,
        "text": text,
        "ph_seq": " ".join(phone["symbol"] for phone in phones),
        "ph_dur": " ".join(f'{phone["duration_frames"] / 50:.6f}' for phone in phones),
        "f0_seq": " ".join(f"{value:.3f}" for value in frames.f0_hz.tolist()),
        "f0_timestep": 0.02,
        "spk_mix": {"gts_ja_soprano": 1.0},
    }
    return row, {
        "duration_seconds": round(len(frames.f0_hz) / 50, 6),
        "phoneme_count": len(phones),
        "frame_count": len(frames.f0_hz),
        "silence_gap_frames": silence_frames,
        "voiced_ratio": round(float(frames.voiced.mean()), 6),
        "timing_labels": "score_timed_inferred_split",
        "phoneme_sequence_source": "gyu_singer.frontend.phonemize",
        "phoneme_duration_source": "gyu_singer.alignment.build_phrase_frames",
        "target_f0_source": "nominal_score_f0_with_frontend_voicing",
    }


def distribution(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("distribution requires finite values")
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "standard_deviation": float(np.std(data)),
    }


def _heldout_limits(root: Path) -> tuple[float, float]:
    report_dir = root / "artifacts/reports/diffsinger_gtsinger_heldout_set"
    reports = [json.loads(path.read_text()) for path in sorted(report_dir.glob("evaluation_gtsja*.json"))]
    if len(reports) != 5:
        raise ValueError("five held-out reports are required to freeze artifact limits")
    source_hf = [
        row["reference_calibration"]["waveform_analysis"]["hf_spike_p99_over_median"]
        for row in reports
    ]
    soprano_jumps = [
        next(item for item in row["rows"] if item["label"] == "soprano")["sample_jump_p999"]
        for row in reports
    ]
    return 2 * max(source_hf), 1.1 * max(soprano_jumps)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def protocol_manifest(root: Path = ROOT) -> dict:
    hf_max, jump_max = _heldout_limits(root)
    split_rows = jsonl(root / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl")
    split_ids = {
        f"{split}_ids": sorted(row["id"] for row in split_rows if row["split"] == split)
        for split in ("train", "validation", "test")
    }
    cases = []
    for identifier in CASES:
        score_path = root / CASE_PATHS[identifier]
        score = json.loads(score_path.read_text())
        row, metadata = build_score_ds(score)
        cases.append({
            "id": identifier,
            "language": "ko",
            "split": "protected_evaluation",
            "stress_category": STRESS[identifier],
            "score_path": CASE_PATHS[identifier],
            "score_sha256": sha256(score_path),
            "expected_lyrics": row["text"],
            "expected_phonemes": row["ph_seq"].split(),
            "ds_path": f"data/external/work/gtsinger_gyu_identity_diagnostic/inputs/{identifier}.ds",
        } | metadata)
    model_files = {
        "foundation_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "foundation_checkpoint_sha256": sha256(root / SOURCE_CHECKPOINT.relative_to(ROOT)),
        "combined_vocabulary_initialization": str(INIT_CHECKPOINT.relative_to(ROOT)),
        "combined_vocabulary_initialization_sha256": sha256(root / INIT_CHECKPOINT.relative_to(ROOT)),
        "vocoder_checkpoint": str(VOCODER.relative_to(ROOT)),
        "vocoder_checkpoint_sha256": sha256(root / VOCODER.relative_to(ROOT)),
    }
    manifest = {
        "status": "frozen_before_rendering",
        "authoritative_spec": "docs/superpowers/specs/2026-07-18-gtsinger-gyu-preservation-identity-design.md",
        "design_commit": "6d8f933f9a0087a8b4f0b4b742aca61aaad255c3",
        "cases": cases,
        "seeds": list(SEEDS),
        "identity_references": [
            {"path": path, "sha256": sha256(root / path)} for path in REFERENCES
        ],
        "adaptation_splits": split_ids,
        "labels": "inferred score-timed phoneme split",
        "models": model_files | {
            "gtsinger_dataset_revision": "4426c862beed558b7e1cb8a4dce7e8c0c83bb208",
            "diffsinger_revision": DIFFSINGER_REVISION,
            "whisper": "data/cache/whisper-large-v3-turbo",
            "wavlm": "data/cache/wavlm-base-plus-sv",
            "ecapa": "data/cache/spkrec-ecapa-voxceleb",
            "rmvpe": "data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
            "rmvpe_sha256": sha256(root / "data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        },
        "adapter": {"type": "MelAdapter", "hidden": 64, "limit": 0.75, "initial_output_projection": "zero"},
        "optimizer": {
            "type": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001,
            "maximum_steps": 200, "evaluation_interval": 25, "maximum_checkpoints": 3,
            "early_stop_failed_intervals": 3,
        },
        "loss_weights": {
            "wavlm_identity": 0.5, "ecapa_identity": 0.5, "content": 2.0,
            "pitch_period": 2.0, "waveform": 1.0, "stft_256": 1.0,
            "stft_1024": 1.0, "stft_4096": 1.0, "adapter_delta": 0.1,
            "parameter_update": 1.0,
        },
        "gates": {
            "lyric_similarity_min": 0.8,
            "pitch_mae_regression_max_cents": 10.0,
            "pitch_p90_abs_cents_max": 100.0,
            "pitch_p90_regression_max_cents": 15.0,
            "gross_error_over_600_cents_max": 0.05,
            "voicing_accuracy_min": 0.8,
            "voicing_regression_max": 0.02,
            "clip_fraction_max": 0.0,
            "hf_spike_p99_over_median_max": hf_max,
            "sample_jump_p999_max": jump_max,
            "relative_hf_and_jump_max": 1.1,
            "wavlm_mean_improvement_min": 0.01,
            "ecapa_mean_improvement_min": 0.01,
            "wavlm_individual_regression_min": -0.02,
            "ecapa_individual_regression_min": -0.03,
            "required_phrase_seed_pass_ratio": 1.0,
        },
        "tool_versions": {
            name: _version(name) for name in (
                "torch", "torchaudio", "numpy", "scipy", "librosa", "transformers",
                "speechbrain", "onnxruntime", "lightning", "pytorch-lightning",
            )
        },
        "production_runtime_modified": False,
        "package_modified": False,
        "openutau_modified": False,
        "release_allowed": False,
    }
    validate_protocol(manifest)
    return manifest


def validate_protocol(protocol: dict) -> None:
    if [row["id"] for row in protocol["cases"]] != list(CASES):
        raise ValueError("protocol cases changed")
    if protocol["seeds"] != list(SEEDS):
        raise ValueError("protocol seeds changed")
    split_sets = [set(protocol["adaptation_splits"][f"{name}_ids"]) for name in ("train", "validation", "test")]
    if any(first & second for index, first in enumerate(split_sets) for second in split_sets[index + 1:]):
        raise ValueError("split leakage")
    if len({row["path"] for row in protocol["identity_references"]}) != len(REFERENCES):
        raise ValueError("identity references changed or duplicated")
    if protocol["models"]["foundation_checkpoint_sha256"] != SOURCE_SHA256:
        raise ValueError("foundation checkpoint hash mismatch")
    if protocol["models"]["vocoder_checkpoint_sha256"] != VOCODER_SHA256:
        raise ValueError("vocoder checkpoint hash mismatch")


REQUIRED_METRICS = (
    "lyric_similarity", "pitch_mae_cents", "pitch_p90_abs_cents",
    "gross_error_over_600_cents", "voicing_accuracy", "clip_fraction",
    "hf_spike_p99_over_median", "sample_jump_p999", "spectral_flux_p95",
)


def validate_matrix(rows: list[dict], protocol: dict, *, root: Path | None = None) -> None:
    expected = {(row["id"], seed) for row in protocol["cases"] for seed in protocol["seeds"]}
    actual = {(row.get("case"), row.get("seed")) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("matrix mismatch")
    for row in rows:
        for name in REQUIRED_METRICS:
            value = row.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"missing or non-finite metric: {name}")
        if not row.get("whisper_transcript") or len(row.get("audio_sha256", "")) != 64:
            raise ValueError("missing transcript or WAV SHA")
        if root is not None and not (root / row["audio_path"]).is_file():
            raise ValueError(f"missing WAV: {row['audio_path']}")


def _row_failures(row: dict, gates: dict) -> list[str]:
    failures = []
    if (row["lyric_similarity"] < gates["lyric_similarity_min"]
            or row.get("repetition_detected") or row.get("omission_detected")):
        failures.append("foundation_content_failure")
    if (row["pitch_p90_abs_cents"] > gates["pitch_p90_abs_cents_max"]
            or row["gross_error_over_600_cents"] > gates["gross_error_over_600_cents_max"]):
        failures.append("pitch_regression")
    if row["voicing_accuracy"] < gates["voicing_accuracy_min"]:
        failures.append("voicing_regression")
    if row["clip_fraction"] > gates["clip_fraction_max"]:
        failures.append("clipping_failure")
    if (row["hf_spike_p99_over_median"] > gates["hf_spike_p99_over_median_max"]
            or row["sample_jump_p999"] > gates["sample_jump_p999_max"]):
        failures.append("artifact_regression")
    return failures


def gate_foundation(rows: list[dict], protocol: dict) -> dict:
    validate_matrix(rows, protocol)
    decisions = []
    failures = []
    for row in rows:
        taxonomy = _row_failures(row, protocol["gates"])
        decision = {"case": row["case"], "seed": row["seed"], "pass": not taxonomy,
                    "failure_taxonomy": taxonomy}
        decisions.append(decision)
        if taxonomy:
            failures.append(decision)
    pass_count = len(decisions) - len(failures)
    aggregates = {name: distribution([float(row[name]) for row in rows]) for name in REQUIRED_METRICS}
    passed = pass_count == len(decisions)
    return {
        "status": "foundation_ko_gate_pass" if passed else "foundation_ko_gate_reject",
        "pass_count": pass_count,
        "total_count": len(decisions),
        "pass_ratio": pass_count / len(decisions),
        "identity_training_allowed": passed,
        "failures": failures,
        "decisions": decisions,
        "aggregate": aggregates,
    }


def _environment(root: Path) -> dict:
    import torch

    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(root)
    cache_revision = subprocess.run(
        ["git", "-C", str(root / "data/cache/diffsinger"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": (torch.cuda.get_device_properties(0).total_memory
                            if torch.cuda.is_available() else None),
        "system_memory_bytes": memory_bytes,
        "disk_total_bytes": disk.total,
        "disk_free_bytes_at_freeze": disk.free,
        "diffsinger_cache_checkout": cache_revision,
        "diffsinger_required_revision": DIFFSINGER_REVISION,
        "source_recordings_present_in_worktree": (root / "data/source").exists(),
    }


def freeze() -> None:
    manifest = protocol_manifest(ROOT)
    inputs = WORK / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        score = json.loads((ROOT / case["score_path"]).read_text())
        row, _ = build_score_ds(score)
        (ROOT / case["ds_path"]).write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "environment.json").write_text(
        json.dumps(_environment(ROOT), ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(MANIFEST.relative_to(ROOT)),
                      "cases": len(manifest["cases"]), "seeds": manifest["seeds"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze",))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()


if __name__ == "__main__":
    main()
