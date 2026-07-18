#!/usr/bin/env python3
"""Strict, fail-closed Korean score-native DiffSinger foundation gate."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_diffsinger_pjs_rapid import pitch_errors, voicing_accuracy
from evaluate_rc4_artifact_matrix import audio16
from prepare_diffsinger_gtsinger_ko_qualified import _audio_metrics, normalized_text
from run_gtsinger_gyu_identity_diagnostic import (
    build_score_ds,
    distribution,
    transcript_flags,
)

CASES = (
    "quality_ko",
    "rapid_ko",
    "large_interval_ko",
    "sustain_ko",
    "repeated_ko",
    "high_ko",
    "phrase_boundary_ko",
)
CASE_PATHS = {
    "quality_ko": "examples/quality_ko.json",
    "rapid_ko": "examples/review_rapid_ko.json",
    "large_interval_ko": "examples/review_large_interval_ko.json",
    "sustain_ko": "examples/review_sustain_ko.json",
    "repeated_ko": "examples/review_repeated_ko.json",
    "high_ko": "examples/review_high_ko.json",
    "phrase_boundary_ko": "examples/review_phrase_boundary_ko.json",
}
STRESS = {
    "quality_ko": "ordinary",
    "rapid_ko": "rapid",
    "large_interval_ko": "large_interval",
    "sustain_ko": "sustained",
    "repeated_ko": "repeated_note",
    "high_ko": "high_register",
    "phrase_boundary_ko": "phrase_boundary",
}
REQUIRED = (
    "lyric_similarity",
    "pitch_mae_cents",
    "pitch_p90_abs_cents",
    "gross_pitch_error_rate",
    "voicing_accuracy",
    "clipping_samples",
    "hf_spike_ratio_to_source",
    "sample_jump_ratio_to_source",
    "waveform_discontinuity_ratio_to_source",
    "stft_spike_ratio_to_source",
)
PROTOCOL_PATH = ROOT / "configs/gtsinger_ko_qualified_protocol.json"
SOURCE_SUMMARY = ROOT / "artifacts/reports/gtsinger_ko_source_qualification/summary.json"
SOURCE_MANIFEST = ROOT / "data/manifests/gtsinger_ko_source_qualified.jsonl"
CONFIG = ROOT / "configs/diffsinger_gtsinger_ko_qualified.yaml"
WORK = ROOT / "data/external/work/gtsinger_ko_qualified_foundation"
REPORT = ROOT / "artifacts/reports/gtsinger_ko_qualified_foundation/evaluation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_row(row: dict, gates: dict) -> list[str]:
    failures = []
    if (
        row["lyric_similarity"] < gates["lyric_similarity_min"]
        or row.get("repetition_detected")
        or row.get("omission_detected")
    ):
        failures.append("foundation_content_failure")
    if (
        row["pitch_mae_cents"] > gates["pitch_mae_row_max_cents"]
        or row["pitch_p90_abs_cents"] > gates["pitch_p90_max_cents"]
        or row["gross_pitch_error_rate"] > gates["gross_pitch_error_max"]
    ):
        failures.append("pitch_regression")
    if row["voicing_accuracy"] < gates["voicing_accuracy_min"]:
        failures.append("voicing_regression")
    if row["clipping_samples"] > gates["clipping_samples_max"]:
        failures.append("clipping_failure")
    artifact_checks = {
        "hf_spike_ratio_to_source": "hf_spike_ratio_to_source_max",
        "sample_jump_ratio_to_source": "sample_jump_ratio_to_source_max",
        "waveform_discontinuity_ratio_to_source": "waveform_discontinuity_ratio_to_source_max",
        "stft_spike_ratio_to_source": "stft_spike_ratio_to_source_max",
    }
    if any(row[name] > gates[limit] for name, limit in artifact_checks.items()):
        failures.append("artifact_regression")
    return failures


def summarize_checkpoint(rows: list[dict], protocol: dict) -> dict:
    expected = {(case, seed) for case in CASES for seed in protocol["seeds"]}
    actual = {(row.get("case"), row.get("seed")) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("foundation matrix mismatch")
    for row in rows:
        for name in REQUIRED:
            if not isinstance(row.get(name), (int, float)) or not math.isfinite(row[name]):
                raise ValueError(f"missing or non-finite metric: {name}")
        if not row.get("whisper_transcript") or len(row.get("audio_sha256", "")) != 64:
            raise ValueError("missing transcript or WAV SHA")
    gates = protocol["foundation_gates"]
    decisions = [
        {
            "case": row["case"],
            "seed": row["seed"],
            "failures": gate_row(row, gates),
        }
        for row in rows
    ]
    mean_pitch = float(np.mean([row["pitch_mae_cents"] for row in rows]))
    aggregate_failure = mean_pitch > gates["pitch_mae_mean_max_cents"]
    pass_count = sum(not item["failures"] for item in decisions)
    passed = pass_count == len(rows) and not aggregate_failure
    return {
        "step": rows[0]["checkpoint_step"],
        "status": "foundation_ko_gate_pass" if passed else "foundation_ko_gate_reject",
        "pass_count": pass_count,
        "pass_ratio": pass_count / len(rows),
        "training_identity_allowed": passed,
        "aggregate_pitch_failure": aggregate_failure,
        "decisions": decisions,
        "metrics": {
            name: distribution([float(row[name]) for row in rows]) for name in REQUIRED
        },
        "minimum_lyric_similarity": min(row["lyric_similarity"] for row in rows),
        "maximum_pitch_mae_cents": max(row["pitch_mae_cents"] for row in rows),
    }


def select_checkpoint(reports: list[dict]) -> dict:
    passed = [report for report in reports if report["status"] == "foundation_ko_gate_pass"]
    if not passed:
        return {"status": "foundation_ko_gate_reject", "selected_step": None}
    selected = max(
        passed,
        key=lambda report: (
            report["minimum_lyric_similarity"],
            -report["maximum_pitch_mae_cents"],
            -report["step"],
        ),
    )
    return {"status": "foundation_ko_gate_pass", "selected_step": selected["step"]}


def _frozen() -> dict:
    return json.loads((WORK / "frozen_protocol.json").read_text())


def freeze() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    source = json.loads(SOURCE_SUMMARY.read_text())
    if source.get("status") != "source_qualification_pass" or not source.get("training_allowed"):
        raise RuntimeError("source qualification did not pass")
    import yaml

    config = yaml.safe_load(CONFIG.read_text())
    dictionary = Path(config["dictionaries"]["ko"])
    inputs = WORK / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in CASES:
        score_path = ROOT / CASE_PATHS[case]
        score = json.loads(score_path.read_text())
        row, metadata = build_score_ds(score)
        row["spk_mix"] = {"gts_ko_soprano_2": 1.0}
        ds_path = inputs / f"{case}.ds"
        ds_path.write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
        cases.append({
            "id": case,
            "stress_category": STRESS[case],
            "score_path": CASE_PATHS[case],
            "score_sha256": sha256(score_path),
            "expected_text": row["text"],
            "ds_path": str(ds_path.relative_to(ROOT)),
            "ds_sha256": sha256(ds_path),
        } | metadata)
    checkpoints = []
    for step in protocol["checkpoints"]:
        path = ROOT / f"data/cache/diffsinger/checkpoints/gtsinger_ko_qualified/model_ckpt_steps_{step}.ckpt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"step": step, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    frozen = {
        "status": "frozen_before_render",
        "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "dictionary": str(dictionary),
        "dictionary_sha256": sha256(dictionary),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "cases": cases,
        "seeds": protocol["seeds"],
        "checkpoints": checkpoints,
    }
    (WORK / "frozen_protocol.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": frozen["status"], "cases": len(cases)}, indent=2))


def render() -> None:
    frozen = _frozen()
    diffsinger = ROOT / "data/cache/diffsinger"
    python = diffsinger / ".venv/bin/python"
    output = WORK / "outputs"
    logs = WORK / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in frozen["checkpoints"]:
        for case in frozen["cases"]:
            for seed in frozen["seeds"]:
                title = f"step{checkpoint['step']}_{case['id']}_seed{seed}"
                path = output / f"{title}.wav"
                command = [
                    str(python), "scripts/infer.py", "acoustic", str(ROOT / case["ds_path"]),
                    "--exp", "gtsinger_ko_qualified", "--ckpt", str(checkpoint["step"]),
                    "--spk", "gts_ko_soprano_2", "--out", str(output),
                    "--title", title, "--depth", "0", "--seed", str(seed),
                ]
                started = time.perf_counter()
                process = subprocess.run(command, cwd=diffsinger, capture_output=True, text=True)
                (logs / f"{title}.log").write_text(process.stdout + process.stderr)
                if process.returncode:
                    raise RuntimeError(f"render failed: {title}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append({
                    "checkpoint_step": checkpoint["step"], "case": case["id"],
                    "seed": seed, "audio_path": str(path.relative_to(ROOT)),
                    "audio_sha256": sha256(path),
                    "runtime_seconds": time.perf_counter() - started,
                })
                print(f"rendered {title}", flush=True)
    (WORK / "render_manifest.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")


def _ratio(value: float, baseline: float | None, name: str) -> float:
    if baseline is None or baseline <= 0:
        raise ValueError(f"missing or zero source baseline: {name}")
    return float(value / baseline)


def evaluate() -> None:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    sys.path.insert(0, str(ROOT / "data/cache/soulx-singer"))
    from preprocess.tools.f0_extraction import F0Extractor

    frozen = _frozen()
    source = json.loads(SOURCE_SUMMARY.read_text())
    baselines = source["accepted_source_p95"]
    protocol = json.loads(PROTOCOL_PATH.read_text())
    render_rows = json.loads((WORK / "render_manifest.json").read_text())["rows"]
    cases = {row["id"]: row for row in frozen["cases"]}
    processor = AutoProcessor.from_pretrained(ROOT / "data/cache/whisper-large-v3-turbo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        ROOT / "data/cache/whisper-large-v3-turbo", torch_dtype=dtype
    ).to(device).eval()
    extractor = F0Extractor(
        str(ROOT / "data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device=device, target_sr=24000, hop_size=480, verbose=False,
    )
    rows = []
    for rendered in render_rows:
        case = cases[rendered["case"]]
        path = ROOT / rendered["audio_path"]
        inputs = processor(
            audio16(path), sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        kwargs = {
            key: (value.to(device, dtype=dtype) if key == "input_features" else value.to(device))
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            tokens = whisper.generate(
                **kwargs, language="ko", task="transcribe", max_new_tokens=96
            )
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        target = np.asarray(
            json.loads((ROOT / case["ds_path"]).read_text())[0]["f0_seq"].split(),
            dtype=np.float32,
        )
        observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        errors = pitch_errors(target, observed)
        audio_metrics, _, _ = _audio_metrics(path)
        stft_ratios = [
            _ratio(audio_metrics[f"stft_spike_fft_{size}"], baselines[f"stft_spike_fft_{size}"], f"stft_{size}")
            for size in (256, 1024, 4096)
        ]
        row = rendered | {
            "language": "ko",
            "stress_category": case["stress_category"],
            "expected_text": case["expected_text"],
            "whisper_transcript": transcript,
            "lyric_similarity": SequenceMatcher(
                None, normalized_text(case["expected_text"]), normalized_text(transcript)
            ).ratio(),
            **transcript_flags(case["expected_text"], transcript),
            "pitch_mae_cents": float(np.mean(errors)),
            "pitch_p90_abs_cents": float(np.percentile(errors, 90)),
            "gross_pitch_error_rate": float(np.mean(errors > 600)),
            "voicing_accuracy": voicing_accuracy(target, observed),
            "clipping_samples": audio_metrics["clipping_samples"],
            "hf_spike_ratio_to_source": _ratio(
                audio_metrics["hf_spike_p99_over_median"], baselines["hf_spike_p99_over_median"], "hf"
            ),
            "sample_jump_ratio_to_source": _ratio(
                audio_metrics["sample_jump_p999"], baselines["sample_jump_p999"], "sample_jump"
            ),
            "waveform_discontinuity_ratio_to_source": _ratio(
                audio_metrics["waveform_discontinuity_p999"], baselines["waveform_discontinuity_p999"], "waveform"
            ),
            "stft_spike_ratio_to_source": max(stft_ratios),
            "stft_spike_ratios": dict(zip(("fft_256", "fft_1024", "fft_4096"), stft_ratios)),
        }
        rows.append(row)
        print(f"evaluated step{row['checkpoint_step']} {row['case']} seed{row['seed']}", flush=True)
    (WORK / "metrics.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n")


def _plot_failure(row: dict) -> str:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import soundfile as sf

    path = ROOT / row["audio_path"]
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(np.arange(len(mono)) / rate, mono, linewidth=0.35)
    axes[0].set_title("waveform")
    for axis, size in zip(axes[1:], (256, 1024, 4096)):
        hop = max(64, size // 4)
        spectrum = librosa.amplitude_to_db(
            np.abs(librosa.stft(mono, n_fft=size, hop_length=hop)), ref=np.max
        )
        librosa.display.specshow(
            spectrum, sr=rate, hop_length=hop, x_axis="time", y_axis="hz", ax=axis
        )
        axis.set_title(f"STFT {size}")
    output = WORK / "failure_plots" / (
        f"step{row['checkpoint_step']}_{row['case']}_seed{row['seed']}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return str(output.relative_to(ROOT))


def report() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    rows = json.loads((WORK / "metrics.json").read_text())["rows"]
    reports = [
        summarize_checkpoint(
            [row for row in rows if row["checkpoint_step"] == step], protocol
        )
        for step in protocol["checkpoints"]
    ]
    selection = select_checkpoint(reports)
    failed_keys = {
        (checkpoint["step"], decision["case"], decision["seed"])
        for checkpoint in reports
        for decision in checkpoint["decisions"]
        if decision["failures"]
    }
    failure_plots = [
        _plot_failure(row)
        for row in rows
        if (row["checkpoint_step"], row["case"], row["seed"]) in failed_keys
    ]
    result = {
        "status": selection["status"],
        "selected_step": selection["selected_step"],
        "identity_training_allowed": selection["status"] == "foundation_ko_gate_pass",
        "candidate_status": (
            "foundation_gate_pass_multilingual_plan_required"
            if selection["status"] == "foundation_ko_gate_pass"
            else "diagnostic_reject"
        ),
        "checkpoints": reports,
        "rows": rows,
        "failure_plots": failure_plots,
        "runtime_integration": False,
        "package_openutau": "blocked",
        "release_allowed": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected_step": result["selected_step"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "render", "evaluate", "report"))
    args = parser.parse_args()
    globals()[args.command]()


if __name__ == "__main__":
    main()
