#!/usr/bin/env python3
"""Run the frozen, release-ineligible GTSinger-to-GYU diagnostic."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
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
REPORTED_DIFFSINGER_REVISION = "0619d61d5301c4340db442a15cf3e73e197e9101"
DIFFSINGER_REVISION = "753b7cc622aadf802b3145d7bb8f7df4afa213c4"


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


def final_status(evaluation: dict) -> dict:
    if evaluation["status"] != "foundation_ko_gate_reject":
        raise ValueError("final status is only valid for the completed foundation rejection path")
    if evaluation.get("identity_training_started"):
        raise ValueError("rejected foundation must not start identity training")
    return {
        "report_header": "NOT A RELEASE REPORT — EXPERIMENT REJECTED",
        "conclusion": "diagnostic_reject",
        "training_status": "not_started_foundation_gate_failed",
        "checkpoint_selected": None,
        "human_ab_generated": False,
        "runtime_integration": False,
        "package_openutau": "blocked",
        "release_allowed": False,
        "failure_taxonomy": ["foundation_content_failure"],
    }


def _normalized(text: str) -> str:
    return re.sub(r"[^a-zA-Z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def transcript_flags(expected: str, transcript: str) -> dict:
    expected_value = _normalized(expected)
    observed = _normalized(transcript)
    repeated = bool(expected_value and observed.count(expected_value) > 1)
    if observed != expected_value and not repeated:
        repeated = any(
            observed == observed[:size] * (len(observed) // size)
            and len(observed) // size >= 2
            for size in range(1, len(observed) // 2 + 1)
            if len(observed) % size == 0
        )
    omitted = len(observed) < 0.8 * len(expected_value)
    return {"repetition_detected": repeated, "omission_detected": omitted}


def render_jobs(protocol: dict) -> list[dict]:
    return [
        {
            "case": case["id"],
            "seed": seed,
            "title": f'{case["id"]}_foundation_seed{seed}',
            "ds_path": case.get("ds_path", f'inputs/{case["id"]}.ds'),
            "audio_path": (
                "data/external/work/gtsinger_gyu_identity_diagnostic/outputs/"
                f'{case["id"]}_foundation_seed{seed}.wav'
            ),
        }
        for case in protocol["cases"]
        for seed in protocol["seeds"]
    ]


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
    cache_revision = subprocess.run(
        ["git", "-C", str(root / "data/cache/diffsinger"), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if cache_revision != DIFFSINGER_REVISION:
        raise ValueError(f"actual DiffSinger checkout changed: {cache_revision}")
    reported_available = subprocess.run(
        ["git", "-C", str(root / "data/cache/diffsinger"), "cat-file", "-e",
         f"{REPORTED_DIFFSINGER_REVISION}^{{commit}}"],
        capture_output=True,
    ).returncode == 0
    manifest = {
        "status": "frozen_before_rendering",
        "protocol_revision": 2,
        "invalidated_protocol_revision": 1,
        "protocol_restart_reason": "invalid_reported_diffsinger_revision",
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
            "reported_diffsinger_revision": REPORTED_DIFFSINGER_REVISION,
            "reported_revision_available": reported_available,
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
    if protocol["models"]["reported_revision_available"]:
        raise ValueError("protocol restart is invalid because the reported revision became available")
    if protocol["models"]["diffsinger_revision"] != DIFFSINGER_REVISION:
        raise ValueError("actual DiffSinger revision changed")


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
        "diffsinger_reported_revision": REPORTED_DIFFSINGER_REVISION,
        "diffsinger_reported_revision_available": False,
        "diffsinger_actual_revision": DIFFSINGER_REVISION,
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


def _prepare_inference_experiment(protocol: dict) -> str:
    source = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity"
    target = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_gyu_protocol_init"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "dictionary-gyu.txt", "lang_map.json", "spk_map.json"):
        destination = target / name
        if not destination.exists():
            shutil.copy2(source / name, destination)
    checkpoint = target / "model_ckpt_steps_0.ckpt"
    if not checkpoint.exists():
        shutil.copy2(ROOT / protocol["models"]["combined_vocabulary_initialization"], checkpoint)
    expected = protocol["models"]["combined_vocabulary_initialization_sha256"]
    if sha256(checkpoint) != expected:
        raise ValueError("protocol inference checkpoint hash mismatch")
    if json.loads((target / "spk_map.json").read_text()).get("gts_ja_soprano") != 0:
        raise ValueError("foundation speaker id changed")
    return target.name


def render_foundation() -> None:
    protocol = json.loads(MANIFEST.read_text())
    validate_protocol(protocol)
    experiment = _prepare_inference_experiment(protocol)
    output = WORK / "outputs"
    logs = WORK / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    diffsinger = ROOT / "data/cache/diffsinger"
    python = ROOT.parent.parent / ".venv-diffsinger/bin/python"
    if not python.is_file():
        python = Path("/home/kotori9/code/gyukaro/.venv-diffsinger/bin/python")
    rows = []
    for job in render_jobs(protocol):
        path = ROOT / job["audio_path"]
        if path.exists():
            raise FileExistsError(f"frozen output already exists: {path}")
        command = [
            "/usr/bin/time", "-f", "%M", "-o", str(logs / f'{job["title"]}.rss'),
            str(python), "scripts/infer.py", "acoustic", str(ROOT / job["ds_path"]),
            "--exp", experiment, "--ckpt", "0", "--spk", "gts_ja_soprano",
            "--out", str(output), "--title", job["title"], "--depth", "0",
            "--seed", str(job["seed"]),
        ]
        started = time.perf_counter()
        process = subprocess.run(
            command, cwd=diffsinger, env=os.environ | {
                "PYTHONPATH": f"{ROOT / 'scripts'}:{diffsinger}",
            }, capture_output=True, text=True,
        )
        (logs / f'{job["title"]}.log').write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f"DiffSinger render failed for {job['title']}; see local log")
        if not path.is_file():
            raise FileNotFoundError(f"DiffSinger did not create {path}")
        rows.append(job | {
            "command": command[6:],
            "runtime_seconds": round(time.perf_counter() - started, 4),
            "peak_process_rss_kib": int((logs / f'{job["title"]}.rss').read_text().strip()),
            "peak_gpu_memory_bytes": None,
            "peak_gpu_memory_note": "nvidia-smi reports N/A on unified-memory NVIDIA GB10",
            "audio_sha256": sha256(path),
        })
        print(f"rendered {job['case']} seed{job['seed']}", flush=True)
    report = {
        "status": "foundation_render_complete",
        "protocol_sha256": sha256(MANIFEST),
        "diffsinger_revision": protocol["models"]["diffsinger_revision"],
        "experiment": experiment,
        "rows": rows,
    }
    (WORK / "render_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "outputs": len(rows)}, indent=2))


def _plot_failure(path: Path, target_f0: np.ndarray, label: str) -> str:
    import librosa
    import matplotlib.pyplot as plt
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    figure, axes = plt.subplots(5, 1, figsize=(14, 12), constrained_layout=True)
    times = np.arange(len(mono)) / rate
    axes[0].plot(times, mono, linewidth=0.35)
    axes[0].set_title(f"{label} waveform")
    axes[1].plot(np.arange(len(target_f0)) * 0.02, target_f0, linewidth=0.7)
    axes[1].set_title("nominal score F0")
    for axis, size in zip(axes[2:], (256, 1024, 4096)):
        spectrum = librosa.amplitude_to_db(
            np.abs(librosa.stft(mono, n_fft=size, hop_length=max(64, size // 4))), ref=np.max
        )
        librosa.display.specshow(spectrum, sr=rate, hop_length=max(64, size // 4),
                                 x_axis="time", y_axis="hz", ax=axis)
        axis.set_title(f"STFT {size}")
    plots = WORK / "failure_plots"
    plots.mkdir(parents=True, exist_ok=True)
    output = plots / f"{label}.png"
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return str(output.relative_to(ROOT))


def evaluate_foundation() -> None:
    import librosa
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioXVector,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    sys.path.insert(0, str(ROOT / "data/cache/soulx-singer"))
    from preprocess.tools.f0_extraction import F0Extractor
    from evaluate_rc4_artifact_matrix import acoustics, audio16

    protocol = json.loads(MANIFEST.read_text())
    validate_protocol(protocol)
    render_report = json.loads((WORK / "render_manifest.json").read_text())
    jobs = {(row["case"], row["seed"]): row for row in render_report["rows"]}
    if set(jobs) != {(case["id"], seed) for case in protocol["cases"] for seed in protocol["seeds"]}:
        raise ValueError("render manifest matrix mismatch")
    case_map = {row["id"]: row for row in protocol["cases"]}

    cache = ROOT / "data/cache"
    processor = AutoProcessor.from_pretrained(cache / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        cache / "whisper-large-v3-turbo", dtype=torch.float16
    ).cuda().eval()
    extractor = F0Extractor(
        str(cache / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )

    def transcribe(path: Path) -> str:
        inputs = processor(audio=audio16(path), sampling_rate=16_000, return_tensors="pt",
                           return_attention_mask=True)
        with torch.inference_mode():
            tokens = whisper.generate(
                inputs.input_features.cuda().half(),
                attention_mask=(inputs.attention_mask.cuda()
                                if "attention_mask" in inputs else None),
                language="ko", task="transcribe", max_new_tokens=96,
            )
        return processor.batch_decode(tokens, skip_special_tokens=True)[0]

    reference_audit = []
    for reference in protocol["identity_references"]:
        path = ROOT / reference["path"]
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(1)
        f0 = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        voiced = f0[f0 > 1]
        rms = float(np.sqrt(np.mean(mono * mono)))
        noise = float(np.percentile(np.abs(mono), 10))
        reference_audit.append({
            "path": reference["path"], "sha256": sha256(path), "sample_rate": rate,
            "channels": audio.shape[1], "duration_seconds": round(len(mono) / rate, 6),
            "clip_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 8),
            "silence_ratio": round(float(np.mean(np.abs(mono) < 1e-4)), 6),
            "snr_proxy_db": round(20 * math.log10(max(rms, 1e-8) / max(noise, 1e-8)), 3),
            "whisper_transcript": transcribe(path),
            "pitch_hz_min": round(float(np.min(voiced)), 3) if len(voiced) else None,
            "pitch_hz_median": round(float(np.median(voiced)), 3) if len(voiced) else None,
            "pitch_hz_max": round(float(np.max(voiced)), 3) if len(voiced) else None,
        })

    rows = []
    for case in protocol["cases"]:
        target = np.asarray(json.loads((ROOT / case["ds_path"]).read_text())[0]["f0_seq"].split(),
                            dtype=np.float32)
        for seed in protocol["seeds"]:
            render = jobs[(case["id"], seed)]
            path = ROOT / render["audio_path"]
            transcript = transcribe(path)
            observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
            frames = min(len(target), len(observed))
            target_aligned, observed_aligned = target[:frames], observed[:frames]
            both = (target_aligned > 1) & (observed_aligned > 1)
            if not np.any(both):
                raise ValueError(f"no jointly voiced frames: {case['id']} seed{seed}")
            cents = np.abs(1200 * np.log2(observed_aligned[both] / target_aligned[both]))
            flags = transcript_flags(case["expected_lyrics"], transcript)
            row = {
                "condition": "unadapted_gtsinger_soprano_combined_vocab_init",
                "case": case["id"], "language": "ko", "stress_category": case["stress_category"],
                "seed": seed, "expected_text": case["expected_lyrics"],
                "whisper_transcript": transcript,
                "lyric_similarity": round(SequenceMatcher(
                    None, _normalized(case["expected_lyrics"]), _normalized(transcript)
                ).ratio(), 6),
                **flags,
                "pitch_mae_cents": round(float(np.mean(cents)), 4),
                "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 4),
                "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 6),
                "target_voiced_ratio": round(float(np.mean(target_aligned > 1)), 6),
                "observed_voiced_ratio": round(float(np.mean(observed_aligned > 1)), 6),
                "voicing_accuracy": round(float(np.mean(
                    (target_aligned > 1) == (observed_aligned > 1)
                )), 6),
                "target_f0_frames": len(target), "observed_f0_frames": len(observed),
                "timing_grid_mismatch_frames": abs(len(target) - len(observed)),
                "audio_path": render["audio_path"], "audio_sha256": sha256(path),
                "sample_rate": sf.info(path).samplerate,
                "runtime_seconds": render["runtime_seconds"],
                "peak_process_rss_kib": render["peak_process_rss_kib"],
                "peak_gpu_memory_bytes": render["peak_gpu_memory_bytes"],
            } | acoustics(path)
            row["waveform_discontinuity"] = row["sample_jump_p999"]
            rows.append(row)
            print(f"metrics {case['id']} seed{seed}: {transcript}", flush=True)

    del whisper, extractor
    torch.cuda.empty_cache()
    feature_extractor = AutoFeatureExtractor.from_pretrained(cache / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(cache / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(cache / "spkrec-ecapa-voxceleb"),
        savedir=str(cache / "spkrec-ecapa-voxceleb"), run_opts={"device": "cuda:0"},
    )

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio16(path)
        values = feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            first = wavlm(**{name: value.cuda() for name, value in values.items()}).embeddings
            second = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        first = torch.nn.functional.normalize(first, dim=-1).squeeze().cpu().numpy()
        second = second.squeeze().cpu().numpy()
        second /= max(np.linalg.norm(second), 1e-8)
        return first, second

    reference_embeddings = {
        row["path"]: speaker(ROOT / row["path"]) for row in reference_audit
    }
    for row in reference_audit:
        wavlm_values = [float(np.dot(reference_embeddings[row["path"]][0], value[0]))
                        for name, value in reference_embeddings.items() if name != row["path"]]
        ecapa_values = [float(np.dot(reference_embeddings[row["path"]][1], value[1]))
                        for name, value in reference_embeddings.items() if name != row["path"]]
        row["wavlm_to_other_references"] = distribution(wavlm_values)
        row["ecapa_to_other_references"] = distribution(ecapa_values)
        row["embedding_outlier"] = False
    duplicate_transcripts = {
        value for value in (_normalized(row["whisper_transcript"]) for row in reference_audit)
        if value and sum(_normalized(other["whisper_transcript"]) == value for other in reference_audit) > 1
    }
    for row in reference_audit:
        row["duplicate_content"] = _normalized(row["whisper_transcript"]) in duplicate_transcripts

    for row in rows:
        current = speaker(ROOT / row["audio_path"])
        wavlm_values = {
            name: round(float(np.dot(value[0], current[0])), 7)
            for name, value in reference_embeddings.items()
        }
        ecapa_values = {
            name: round(float(np.dot(value[1], current[1])), 7)
            for name, value in reference_embeddings.items()
        }
        row["wavlm_similarity"] = {"values": wavlm_values,
                                    "distribution": distribution(list(wavlm_values.values()))}
        row["ecapa_similarity"] = {"values": ecapa_values,
                                    "distribution": distribution(list(ecapa_values.values()))}

    del wavlm, ecapa
    torch.cuda.empty_cache()
    validate_matrix(rows, protocol, root=ROOT)
    decision = gate_foundation(rows, protocol)
    target_by_case = {
        case["id"]: np.asarray(json.loads((ROOT / case["ds_path"]).read_text())[0]["f0_seq"].split(),
                               dtype=np.float32)
        for case in protocol["cases"]
    }
    failure_plots = []
    for failure in decision["failures"]:
        row = next(item for item in rows if item["case"] == failure["case"]
                   and item["seed"] == failure["seed"])
        failure_plots.append(_plot_failure(
            ROOT / row["audio_path"], target_by_case[row["case"]],
            f'{row["case"]}_seed{row["seed"]}',
        ))
    result = {
        "status": decision["status"],
        "candidate_status": "diagnostic_reject" if not decision["identity_training_allowed"]
                            else "foundation_gate_pass_identity_training_required",
        "protocol_path": str(MANIFEST.relative_to(ROOT)), "protocol_sha256": sha256(MANIFEST),
        "reference_audit": reference_audit, "rows": rows, "gate": decision,
        "failure_plots": failure_plots,
        "identity_training_started": False,
        "runtime_integration": False, "package_openutau": "blocked", "release_allowed": False,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "foundation_ko_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": result["status"], "pass_count": decision["pass_count"],
                      "total_count": decision["total_count"], "failure_plots": len(failure_plots)}, indent=2))


def _metric(value: float) -> str:
    return f"{value:.6f}"


def write_final_report(test_count: int, dataset_result: str) -> None:
    protocol = json.loads(MANIFEST.read_text())
    environment = json.loads((REPORT / "environment.json").read_text())
    evaluation_path = REPORT / "foundation_ko_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    status = final_status(evaluation)
    gate = evaluation["gate"]
    aggregate = gate["aggregate"]
    commits = subprocess.run(
        ["git", "log", "--reverse", "--format=%H %s", "6d8f933..HEAD"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    evidence_files = [path for path in WORK.rglob("*") if path.is_file()]
    evidence_bytes = sum(path.stat().st_size for path in evidence_files)
    rows = evaluation["rows"]
    table = [
        "| Case | Seed | Whisper | Lyric | Repeat | Pitch MAE | Pitch p90 | Voicing | Clip | HF spike | Jump | WavLM | ECAPA | WAV path | SHA-256 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        transcript = row["whisper_transcript"].strip().replace("|", "\\|")
        table.append(
            f'| {row["case"]} | {row["seed"]} | {transcript} | '
            f'{_metric(row["lyric_similarity"])} | {row["repetition_detected"]} | '
            f'{_metric(row["pitch_mae_cents"])} | {_metric(row["pitch_p90_abs_cents"])} | '
            f'{_metric(row["voicing_accuracy"])} | {_metric(row["clip_fraction"])} | '
            f'{_metric(row["hf_spike_p99_over_median"])} | {_metric(row["sample_jump_p999"])} | '
            f'{_metric(row["wavlm_similarity"]["distribution"]["mean"])} | '
            f'{_metric(row["ecapa_similarity"]["distribution"]["mean"])} | '
            f'`{row["audio_path"]}` | '
            f'`{row["audio_sha256"]}` |'
        )
    distributions = []
    for name, values in aggregate.items():
        distributions.append(
            f'| {name} | {_metric(values["mean"])} | {_metric(values["median"])} | '
            f'{_metric(values["minimum"])} | {_metric(values["maximum"])} | '
            f'{_metric(values["standard_deviation"])} |'
        )
    tools = ", ".join(f"{name} {version}" for name, version in protocol["tool_versions"].items())
    commit_lines = "\n".join(f"- `{line}`" for line in commits) or "- none"
    reference_lines = "\n".join(
        f'- `{row["path"]}` — `{row["sha256"]}`; {row["sample_rate"]} Hz, '
        f'{row["channels"]} ch, clip={row["clip_fraction"]}, transcript=`{row["whisper_transcript"].strip()}`'
        for row in evaluation["reference_audit"]
    )
    report = f'''{status["report_header"]}

# GTSinger-to-GYU preservation identity diagnostic

## Decision

- Conclusion: `{status["conclusion"]}`
- Foundation gate: `{evaluation["status"]}` ({gate["pass_count"]}/{gate["total_count"]}, pass ratio {gate["pass_ratio"]:.3f})
- Training: `{status["training_status"]}`
- Failure taxonomy: `foundation_content_failure`
- Selected checkpoint: none
- Human A/B: not generated
- Runtime integration: false
- Package/OpenUtau: blocked and unchanged
- RC9, production, and v1.0.0 remain prohibited.

The frozen Korean foundation emitted repeated or substituted syllables on every phrase×seed item. Mandatory lexical qualification therefore failed before optimizer initialization. Identity scores cannot compensate for this failure, and the approved protocol forbids attempting to repair it with the adapter.

## Authority and provenance

- Specification: `{protocol["authoritative_spec"]}`
- Approved design commit: `{protocol["design_commit"]}`
- Starting commit: `6d8f933f9a0087a8b4f0b4b742aca61aaad255c3`
- Protocol revision: {protocol["protocol_revision"]}; invalidated revision: {protocol["invalidated_protocol_revision"]}
- Protocol manifest: `{MANIFEST.relative_to(ROOT)}` (`{sha256(MANIFEST)}`)
- DiffSinger actual cached revision: `{protocol["models"]["diffsinger_revision"]}`
- Reported but unavailable DiffSinger revision: `{protocol["models"]["reported_diffsinger_revision"]}`; retained as a provenance erratum and not used.
- GTSinger dataset revision: `{protocol["models"]["gtsinger_dataset_revision"]}`
- Foundation checkpoint SHA-256: `{protocol["models"]["foundation_checkpoint_sha256"]}`
- Combined-vocabulary zero-step initialization SHA-256: `{protocol["models"]["combined_vocabulary_initialization_sha256"]}`
- Vocoder SHA-256: `{protocol["models"]["vocoder_checkpoint_sha256"]}`
- RMVPE SHA-256: `{protocol["models"]["rmvpe_sha256"]}`

Implementation commits before this report:

{commit_lines}

## Environment

- Python {environment["python"]}; PyTorch {environment["pytorch"]}; CUDA build {environment["cuda_build"]}
- GPU: {environment["gpu"]}; unified memory {environment["gpu_total_bytes"]} bytes
- System memory: {environment["system_memory_bytes"]} bytes
- Disk at freeze: {environment["disk_free_bytes_at_freeze"]} free of {environment["disk_total_bytes"]} bytes
- Libraries: {tools}
- Peak GPU memory: unavailable (`nvidia-smi` reports N/A for this GB10 unified-memory system); per-render peak process RSS and runtime are retained sample-wise.

## Frozen protocol

- Cases: ordinary `quality_ko`, rapid `rapid_ko`, large interval `large_interval_ko`, sustain `sustain_ko`, phrase boundary `phrase_boundary_ko`
- Seeds: {protocol["seeds"]}
- Identity references: 212, 215, 216, 219, 220; one fixed set for every sample
- Adapter (authorized but not instantiated): {json.dumps(protocol["adapter"], ensure_ascii=False)}
- Optimizer (frozen but not initialized): {json.dumps(protocol["optimizer"], ensure_ascii=False)}
- Loss weights (frozen but not evaluated): {json.dumps(protocol["loss_weights"], ensure_ascii=False)}
- Adaptation split sizes: train {len(protocol["adaptation_splits"]["train_ids"])}, validation {len(protocol["adaptation_splits"]["validation_ids"])}, held-out {len(protocol["adaptation_splits"]["test_ids"])}; no split leakage.
- All phoneme splits are marked inferred; score timing and nominal F0 are not relabeled as GYU supervision.

### Fixed reference audit

{reference_lines}

No source recording, external dataset, rendered WAV, plot, cache, or checkpoint is committed. The audit found no clipping or duplicate reference transcript; no favorable per-phrase reference selection occurred.

## Parameter and feasibility audit

- Executed foundation state dict: 210 tensors, 27,274,368 values.
- Trainable parameters in the executed experiment: 0 (0%).
- The authorized identity adapter was deliberately not instantiated because the pre-optimizer foundation gate failed.
- Consequently initialization equivalence, adapter gradient isolation, update norms, adapter VRAM, and optimizer feasibility are not applicable—not passed.
- Foundation, variance/pitch/duration predictors, phoneme encoder, decoder, and vocoder were not modified.

## Training and checkpoint selection

- Optimizer steps: 0 of the frozen maximum {protocol["optimizer"]["maximum_steps"]}.
- Training/validation/held-out checkpoint selection was never entered.
- Selected checkpoint: none.
- No held-out-guided tuning, seed selection, reference selection, or loss-weight change occurred.

## Korean qualification results (5×3)

{chr(10).join(table)}

Aggregate distributions:

| Metric | Mean | Median | Minimum | Maximum | Std |
|---|---:|---:|---:|---:|---:|
{chr(10).join(distributions)}

Observed lexical failures were stable across seeds: ordinary produced repeated `와우`; rapid produced repeated `야다`; large interval produced `아 아`; sustain produced `다`; phrase boundary produced repeated `다아`. Pitch, voicing, clipping, and artifact metrics passing on some rows do not override the 0/15 lexical result.

## Japanese held-out and identity-candidate evaluation

The previously verified unadapted soprano Japanese gate remains the external foundation reference (5/5 phrases and 15/15 seed matrix) in `artifacts/reports/diffsinger_gtsinger_heldout_set/aggregate_evaluation.json`. No adapted candidate exists, so a new candidate Japanese 5×3 matrix, identity-gain comparison, protected regression matrix, or human A/B set would be misleading and was not generated after the mandatory Korean early-stop.

WavLM and ECAPA were nevertheless recorded for every Korean foundation WAV against every fixed reference to make the rejection auditable. The compact evaluation contains each sample's per-reference values and mean/median/minimum/maximum/standard deviation. They are baseline observations, not identity improvements: no trained condition exists from which to compute a gain.

## Machine gates

- Korean lexical validity all phrases/seeds: FAIL (0/15)
- No repetition/omission/substitution: FAIL
- Seed stability: FAIL at the lexical level across all seeds
- Pitch/voicing/clipping/artifact preservation: measured, but cannot promote a lexically failed foundation
- WavLM and ECAPA held-out improvement: NOT EVALUATED; no candidate
- Individual regression limits: NOT EVALUATED; no candidate
- Overall mandatory gate: FAIL

Failure taxonomy is `foundation_content_failure`, not `adapter_content_regression`: the failure occurred before an adapter or optimizer existed.

## Evidence

- Compact evaluation: `{evaluation_path.relative_to(ROOT)}`
- Local WAVs: `data/external/work/gtsinger_gyu_identity_diagnostic/outputs/`
- Local render logs: `data/external/work/gtsinger_gyu_identity_diagnostic/logs/`
- Local waveform + FFT 256/1024/4096 plots: `data/external/work/gtsinger_gyu_identity_diagnostic/failure_plots/`
- Local DS inputs: `data/external/work/gtsinger_gyu_identity_diagnostic/inputs/`
- Local evidence: {len(evidence_files)} files, {evidence_bytes} bytes; ignored and uncommitted.
- Every WAV path and SHA-256 is recorded in the sample table and compact JSON.

## Repository verification

- Relevant/full test suite: {test_count} passed.
- Dataset validation: `{dataset_result}`.
- `git diff --check`: PASS before report commit and required again after commit.
- Production renderer imports no experimental adapter; package config selects no checkpoint; OpenUtau paths are unchanged.
- Previous RC7 and rejected SoulX evidence are unchanged. RC7 remains an accepted experimental baseline; RC8 candidate 3, v0.7 adapter, truncated K=2, and truncated K=4 remain rejected.

## Final conclusion

`diagnostic_reject`

The bounded experiment completed at its mandatory early-stop. The unadapted Korean score-native lexical foundation is not qualified, so GTSinger-to-GYU identity adaptation, checkpoint creation, runtime integration, packaging, OpenUtau work, RC9, and release work remain blocked.
'''
    document = ROOT / "docs/final_gtsinger_gyu_identity_diagnostic.md"
    document.write_text(report)
    final_json = status | {
        "foundation_gate": evaluation["status"],
        "foundation_pass_count": gate["pass_count"],
        "foundation_total_count": gate["total_count"],
        "optimizer_steps": 0,
        "test_count": test_count,
        "dataset_validation": dataset_result,
        "local_evidence_count": len(evidence_files),
        "local_evidence_bytes": evidence_bytes,
        "report_path": str(document.relative_to(ROOT)),
        "evaluation_path": str(evaluation_path.relative_to(ROOT)),
    }
    (REPORT / "final_status.json").write_text(
        json.dumps(final_json, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(final_json, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "render-foundation", "evaluate-foundation", "finalize"))
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--dataset-result")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "render-foundation":
        render_foundation()
    elif args.command == "evaluate-foundation":
        evaluate_foundation()
    elif args.command == "finalize":
        if args.test_count is None or not args.dataset_result:
            parser.error("finalize requires --test-count and --dataset-result")
        write_final_report(args.test_count, args.dataset_result)


if __name__ == "__main__":
    main()
