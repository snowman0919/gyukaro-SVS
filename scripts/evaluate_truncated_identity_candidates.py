#!/usr/bin/env python3
"""Render and gate the four-condition truncated-identity held-out matrix."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft


CONDITIONS = ("identity_off", "current_v07", "k2", "k4")
FIXED_SEEDS = (7, 21, 42)
METRICS = (
    "wavlm_similarity", "ecapa_similarity", "lyric_similarity",
    "pitch_mae_cents", "voicing_accuracy", "hf_spike_p99_over_median",
    "sample_jump_p999", "clip_fraction", "runtime_seconds",
    "peak_cuda_allocated_bytes",
)


def summarize(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"mean": None, "median": None, "minimum": None, "std": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "std": float(np.std(array)),
    }


def _comparison_failure(candidate, baseline, metric, relation, limit):
    value, reference = candidate.get(metric), baseline.get(metric)
    if value is None or reference is None:
        return {"metric": metric, "reason": "missing_metric", "candidate": value, "baseline": reference}
    failed = value < reference - limit if relation == "higher" else value > reference + limit
    if not failed:
        return None
    return {
        "metric": metric, "reason": "individual_regression", "candidate": value,
        "baseline": reference, "allowed_delta": limit,
    }


def candidate_gates(rows: list[dict], candidate_name: str) -> dict:
    by = {(row["phrase"], row["seed"], row["condition"]): row for row in rows}
    candidates = [row for row in rows if row["condition"] == candidate_name]
    heldout = [row for row in candidates if row["split"] == "heldout"]
    failures, sample_pass = [], {}
    for candidate in candidates:
        key = (candidate["phrase"], candidate["seed"])
        baseline = by.get((*key, "identity_off"))
        if baseline is None:
            failures.append({"phrase": key[0], "seed": key[1], "metric": "comparison", "reason": "missing_identity_off"})
            sample_pass[key] = False
            continue
        sample_failures = []
        for metric, relation, limit in (
            ("wavlm_similarity", "higher", 0.02),
            ("ecapa_similarity", "higher", 0.03),
            ("lyric_similarity", "higher", 0.02),
            ("lyric_coverage", "higher", 0.02),
            ("voicing_accuracy", "higher", 0.01),
        ):
            failure = _comparison_failure(candidate, baseline, metric, relation, limit)
            if failure:
                sample_failures.append(failure)
        base_pitch = baseline.get("pitch_mae_cents")
        pitch_limit = None if base_pitch is None else max(2.0, 0.05 * base_pitch)
        if pitch_limit is None or candidate.get("pitch_mae_cents") is None:
            sample_failures.append({"metric": "pitch_mae_cents", "reason": "missing_metric"})
        else:
            failure = _comparison_failure(candidate, baseline, "pitch_mae_cents", "lower", pitch_limit)
            if failure:
                sample_failures.append(failure)
        for metric in ("hf_spike_p99_over_median", "sample_jump_p999"):
            base_value = baseline.get(metric)
            limit = None if base_value is None else max(base_value * 0.10, 1e-8)
            if limit is None:
                sample_failures.append({"metric": metric, "reason": "missing_metric"})
            else:
                failure = _comparison_failure(candidate, baseline, metric, "lower", limit)
                if failure:
                    sample_failures.append(failure)
        if candidate.get("clipping_samples", 1) != 0:
            sample_failures.append({
                "metric": "clipping_samples", "reason": "clipping", "candidate": candidate.get("clipping_samples"),
            })
        if candidate.get("repeated_expected_span") and not baseline.get("repeated_expected_span"):
            sample_failures.append({
                "metric": "repeated_expected_span", "reason": "repetition",
                "candidate": candidate["repeated_expected_span"],
            })
        if candidate.get("omission_detected") and not baseline.get("omission_detected"):
            sample_failures.append({"metric": "omission_detected", "reason": "omission"})
        sample_pass[key] = not sample_failures
        failures.extend({"phrase": key[0], "seed": key[1]} | failure for failure in sample_failures)

    def mean(condition, metric):
        selected = [
            row[metric] for row in rows
            if row["condition"] == condition and row["split"] == "heldout" and row.get(metric) is not None
        ]
        return float(np.mean(selected)) if selected else float("nan")

    delta_off = {
        metric: mean(candidate_name, metric) - mean("identity_off", metric)
        for metric in ("wavlm_similarity", "ecapa_similarity")
    }
    delta_current = {
        metric: mean(candidate_name, metric) - mean("current_v07", metric)
        for metric in ("wavlm_similarity", "ecapa_similarity")
    }
    mean_gates = {
        "wavlm_plus_0_01_vs_identity_off": delta_off["wavlm_similarity"] >= 0.01,
        "ecapa_plus_0_01_vs_identity_off": delta_off["ecapa_similarity"] >= 0.01,
        "wavlm_plus_0_005_vs_current_v07": delta_current["wavlm_similarity"] >= 0.005,
        "ecapa_plus_0_005_vs_current_v07": delta_current["ecapa_similarity"] >= 0.005,
    }
    ratio = sum(sample_pass.values()) / len(sample_pass) if sample_pass else 0.0
    rapid_keys = [key for key in sample_pass if by[(*key, candidate_name)]["split"] == "protected"]
    rapid_pass = bool(rapid_keys) and all(sample_pass[key] for key in rapid_keys)
    all_pass = bool(candidates) and ratio == 1.0 and rapid_pass and all(mean_gates.values())
    return {
        "status": "human_pending" if all_pass else "diagnostic_reject",
        "candidate": candidate_name,
        "phrase_seed_pass_ratio": ratio,
        "rapid_ko_pass": rapid_pass,
        "heldout_mean_delta_vs_identity_off": delta_off,
        "heldout_mean_delta_vs_current_v07": delta_current,
        "mean_gates": mean_gates,
        "individual_failures": failures,
        "sample_pass": {f"{phrase}:seed{seed}": value for (phrase, seed), value in sample_pass.items()},
    }


def dispose_candidate_checkpoints(paths: dict[str, Path], gates: dict[str, dict]) -> dict:
    disposition = {}
    for candidate, raw_path in paths.items():
        path = Path(raw_path)
        existed = path.exists()
        digest = _sha(path) if existed else None
        rejected = gates[candidate]["status"] == "diagnostic_reject"
        if rejected and existed:
            path.unlink()
        disposition[candidate] = {
            "status": gates[candidate]["status"], "path": str(path), "sha256_before_disposal": digest,
            "existed_before_disposal": existed, "deleted": rejected and existed,
            "retained_for_human_ab": not rejected and path.exists(), "runtime_integration": False,
        }
        training_report = path.parent / "training.json"
        if training_report.exists():
            training = json.loads(training_report.read_text())
            training["heldout_evaluation_status"] = gates[candidate]["status"]
            training["checkpoint_disposition"] = disposition[candidate]
            if rejected:
                training["rejected_checkpoint_path"] = training.get("checkpoint")
                training["checkpoint"] = None
                training["status"] = "diagnostic_reject"
            training_report.write_text(json.dumps(training, indent=2) + "\n")
    return disposition


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _checkpoint_state(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def render_matrix(args) -> dict:
    import torch

    root = Path.cwd()
    output = Path(args.output).resolve()
    (output / "render_failure.json").unlink(missing_ok=True)
    listening = output / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    rows = [row for row in _manifest(Path(args.manifest)) if row["split"] in {"heldout", "protected"}]
    if len(rows) != 6 or any(row["id"] == "heldout_ja" for row in rows):
        raise RuntimeError("held-out matrix must contain five heldout phrases plus protected Rapid KO and no heldout JA")

    sys.path[:0] = [str(root / "scripts"), str(root / "src"), str(root / "data/cache/soulx-singer")]
    from probe_soulx_score import initialize, render

    init_args = SimpleNamespace(
        seed=21, config=args.config, model=args.model, precision="fp32",
        latent_adapter=args.current_adapter, reference=args.reference, rmvpe=args.rmvpe,
    )
    model, config, reference, reference_f0 = initialize(init_args)
    adapter = model._gyu_latent_adapter
    states = {
        "current_v07": _checkpoint_state(Path(args.current_adapter)),
        "k2": _checkpoint_state(Path(args.k2)),
        "k4": _checkpoint_state(Path(args.k4)),
    }
    render_rows = []
    for row in rows:
        contour_path = root / row["f0_path"]
        contour = np.load(contour_path).astype(np.float32)
        source_path = root / row["source_path"]
        identity_path = root / row["identity_path"]
        warp_path = root / row["content_warp_path"] if row.get("content_warp_path") else None
        for seed in FIXED_SEEDS:
            for condition in CONDITIONS:
                if condition != "identity_off":
                    adapter.load_state_dict(states[condition])
                destination = listening / condition / f"{row['id']}_seed{seed}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                render(
                    model, config, reference, reference_f0, str(source_path), contour,
                    str(destination), None if condition == "identity_off" else str(identity_path),
                    n_steps=64, cfg=float(row["cfg"]), seed=seed, use_fp16=False,
                    content_warp_path=str(warp_path) if warp_path else None,
                    content_warp_strength=float(row.get("content_warp_strength", 0.0)),
                )
                torch.cuda.synchronize()
                record = {
                    "phrase": row["id"], "split": row["split"], "language": row["language"],
                    "seed": seed, "condition": condition,
                    "expected_lyrics": row["expected_lyrics"],
                    "path": str(destination.relative_to(root)), "sha256": _sha(destination),
                    "source_sha256": row["source_sha256"], "target_f0_sha256": row["f0_sha256"],
                    "identity_sha256": row["identity_sha256"] if condition != "identity_off" else None,
                    "cfg": float(row["cfg"]), "n_steps": 64, "precision": "fp32",
                    "runtime_seconds": round(time.perf_counter() - started, 4),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                render_rows.append(record)
                print(f"render {len(render_rows)}/72 {row['id']} seed{seed} {condition}", flush=True)
                (output / "render_manifest.json").write_text(json.dumps({
                    "status": "render_in_progress", "rows": render_rows, "runtime_integration": False,
                }, ensure_ascii=False, indent=2) + "\n")
    report = {
        "status": "render_complete", "conditions": list(CONDITIONS), "fixed_seeds": list(FIXED_SEEDS),
        "phrase_count": len(rows), "output_count": len(render_rows), "rows": render_rows,
        "heldout_ja_excluded": True, "rapid_ko_protected_only": True,
        "phrase_level_soulx_decode": True, "per_note_tts": False,
        "final_wav_stitching": False, "waveform_pitch_shift": False,
        "runtime_integration": False,
    }
    (output / "render_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _compact(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def _repeated_span(expected: str, observed: str):
    for size in range(len(expected), 1, -1):
        for start in range(len(expected) - size + 1):
            span = expected[start:start + size]
            if observed.count(span) > expected.count(span):
                return span
    return None


def _omission(expected: str, observed: str) -> tuple[bool, list[str], float]:
    matcher = SequenceMatcher(None, expected, observed)
    coverage = sum(block.size for block in matcher.get_matching_blocks()) / max(len(expected), 1)
    missing = [expected[i1:i2] for tag, i1, i2, _, _ in matcher.get_opcodes() if tag in {"delete", "replace"} and i2 - i1 >= 2]
    return coverage < 0.80 and bool(missing), missing, coverage


def _audio16(path: Path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    return resample_poly(mono, 16_000, rate).astype(np.float32) if rate != 16_000 else mono.astype(np.float32)


def _acoustics(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    frequencies, _, spectrum = stft(mono, fs=rate, nperseg=1024, noverlap=768, boundary=None)
    power = np.abs(spectrum).T ** 2 + 1e-12
    normalized = power / power.sum(1, keepdims=True)
    hf = normalized[:, frequencies >= min(8000, rate / 2 * 0.8)].sum(1)
    jumps = np.abs(np.diff(mono))
    return {
        "peak": round(float(np.max(np.abs(mono))), 7),
        "rms": round(float(np.sqrt(np.mean(mono * mono))), 7),
        "clipping_samples": int(np.sum(np.abs(mono) >= 0.999)),
        "clip_fraction": round(float(np.mean(np.abs(mono) >= 0.999)), 9),
        "hf_spike_p99_over_median": round(float(np.percentile(hf, 99) / max(np.median(hf), 1e-8)), 6),
        "sample_jump_p999": round(float(np.percentile(jumps, 99.9)), 7),
    }


def _pitch(path: Path, target_path: Path, extractor) -> dict:
    observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
    target_raw = np.load(target_path).astype(np.float32)
    target = np.interp(np.arange(len(observed)), np.linspace(0, len(observed) - 1, len(target_raw)), target_raw)
    target_voiced, observed_voiced = target > 1, observed > 1
    both = target_voiced & observed_voiced
    return {
        "target_voiced_ratio": round(float(target_voiced.mean()), 6),
        "observed_voiced_ratio": round(float(observed_voiced.mean()), 6),
        "voicing_accuracy": round(float(np.mean(target_voiced == observed_voiced)), 6),
        "pitch_mae_cents": None if not both.any() else round(float(np.mean(np.abs(
            1200 * np.log2(observed[both] / target[both])
        ))), 4),
    }


def _plot_comparison(root: Path, rows: list[dict], destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(CONDITIONS), 4, figsize=(20, 12), constrained_layout=True)
    for row_index, condition in enumerate(CONDITIONS):
        row = next(item for item in rows if item["condition"] == condition)
        audio, rate = sf.read(root / row["path"], dtype="float32", always_2d=True)
        mono = audio.mean(1)
        time_axis = np.arange(len(mono)) / rate
        axes[row_index, 0].plot(time_axis, mono, linewidth=0.3)
        axes[row_index, 0].set_title(f"{condition} waveform")
        for column, size in enumerate((256, 1024, 4096), 1):
            frequencies, times, spectrum = stft(mono, fs=rate, nperseg=size, noverlap=3 * size // 4, boundary=None)
            db = 20 * np.log10(np.abs(spectrum) + 1e-7)
            keep = frequencies <= min(12_000, rate / 2)
            axes[row_index, column].pcolormesh(times, frequencies[keep], db[keep], shading="auto", vmin=-100, vmax=0)
            axes[row_index, column].set_title(f"{condition} FFT {size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def analyze_matrix(args) -> dict:
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers.models.wavlm import modeling_wavlm

    root = Path.cwd()
    output = Path(args.output).resolve()
    (output / "analyze_failure.json").unlink(missing_ok=True)
    render_report = json.loads((output / "render_manifest.json").read_text())
    rows = render_report["rows"]
    if render_report.get("status") != "render_complete" or len(rows) != 72:
        raise RuntimeError("complete 72-output render manifest is required")
    corpus = {row["id"]: row for row in _manifest(Path(args.manifest))}
    sys.path[:0] = [str(root / "scripts"), str(root / "data/cache/soulx-singer")]
    from preprocess.tools.f0_extraction import F0Extractor
    from probe_truncated_identity_grad import (
        _centroids, _convert_legacy_wavlm_weight_norm, _disable_optional_peft,
        _freeze, _speaker_and_content,
    )

    extractor = F0Extractor(args.rmvpe, device="cuda", target_sr=24_000, hop_size=480, verbose=False)
    for index, row in enumerate(rows, 1):
        path = root / row["path"]
        row.update(_acoustics(path))
        row.update(_pitch(path, root / corpus[row["phrase"]]["f0_path"], extractor))
        print(f"pitch/acoustics {index}/72", flush=True)
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(args.whisper)
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(args.whisper, dtype=torch.float16).cuda().eval()
    for index, row in enumerate(rows, 1):
        audio = _audio16(root / row["path"])
        values = processor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            tokens = whisper.generate(
                values.input_features.cuda().half(), language=row["language"], task="transcribe", max_new_tokens=96,
            )
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        expected, observed = _compact(row["expected_lyrics"]), _compact(transcript)
        omission, missing, coverage = _omission(expected, observed)
        row.update({
            "whisper_transcript": transcript,
            "lyric_similarity": round(SequenceMatcher(None, expected, observed).ratio(), 6),
            "lyric_coverage": round(coverage, 6),
            "repeated_expected_span": _repeated_span(expected, observed),
            "omission_detected": omission,
            "missing_expected_spans": missing,
        })
        print(f"whisper {index}/72 {row['phrase']} seed{row['seed']} {row['condition']}: {transcript}", flush=True)
    del whisper
    torch.cuda.empty_cache()

    _disable_optional_peft(modeling_wavlm)
    wavlm = AutoModelForAudioXVector.from_pretrained(args.wavlm).cuda().eval()
    wavlm_state = torch.load(Path(args.wavlm) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    load_result = wavlm.load_state_dict(_convert_legacy_wavlm_weight_norm(wavlm_state), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"WavLM checkpoint conversion failed: {load_result}")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir=args.ecapa, run_opts={"device": "cuda"},
    )
    _freeze(wavlm)
    _freeze(ecapa.mods)
    references = [root / f"data/processed/master/{index}.wav" for index in range(171, 195)]
    wavlm_centroid, ecapa_centroid = _centroids(wavlm, ecapa, references, torch.device("cuda"))
    for index, row in enumerate(rows, 1):
        audio = torch.from_numpy(_audio16(root / row["path"]))[None].cuda()
        with torch.inference_mode():
            wavlm_value, ecapa_value, _ = _speaker_and_content(wavlm, ecapa, audio)
        row["wavlm_similarity"] = round(float((wavlm_value * wavlm_centroid).sum()), 7)
        row["ecapa_similarity"] = round(float((ecapa_value * ecapa_centroid).sum()), 7)
        print(f"identity {index}/72", flush=True)
    del wavlm, ecapa
    torch.cuda.empty_cache()

    plots = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phrase"], row["seed"])].append(row)
    for index, ((phrase, seed), selected) in enumerate(sorted(grouped.items()), 1):
        destination = output / "waveform_multires_stft" / f"{phrase}_seed{seed}.png"
        _plot_comparison(root, selected, destination)
        plots[f"{phrase}:seed{seed}"] = str(destination.relative_to(root))
        print(f"plot {index}/{len(grouped)}", flush=True)

    aggregate = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition and row["split"] == "heldout"]
        aggregate[condition] = {
            metric: summarize(row[metric] for row in selected if row.get(metric) is not None)
            for metric in METRICS
        }
    gates = {candidate: candidate_gates(rows, candidate) for candidate in ("k2", "k4")}
    eligible = [candidate for candidate, result in gates.items() if result["status"] == "human_pending"]
    checkpoint_disposition = dispose_candidate_checkpoints(
        {"k2": Path(args.k2), "k4": Path(args.k4)}, gates,
    )
    report = {
        "status": "human_pending" if eligible else "diagnostic_reject",
        "eligible_candidates": eligible,
        "human_listening": "pending" if eligible else "not_requested_gate_failure",
        "conditions": list(CONDITIONS), "fixed_seeds": list(FIXED_SEEDS),
        "heldout_phrase_count": 5, "protected_rapid_ko_count": 1,
        "heldout_ja": "excluded_content_source_failure",
        "identity_reference_centroid": "real GYU master rows 171..194",
        "aggregate_heldout": aggregate, "candidate_gates": gates,
        "checkpoint_disposition": checkpoint_disposition,
        "rows": rows, "waveform_multires_stft": plots,
        "constraints": {
            "phrase_level_soulx_decode": True, "total_steps": 64,
            "per_note_tts": False, "final_wav_stitching": False,
            "waveform_pitch_shift": False, "runtime_integration": False,
        },
        "release_allowed": False,
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    root = Path.cwd()
    soulx = root / "data/cache/soulx-singer"
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("render", "analyze"), required=True)
    parser.add_argument("--output", default="artifacts/reports/truncated_identity_evaluation")
    parser.add_argument("--manifest", default="data/manifests/truncated_identity_diagnostic.jsonl")
    parser.add_argument("--current-adapter", default="checkpoints/gyu_real_latent_adapters_v0.7.pt")
    parser.add_argument("--k2", default="artifacts/reports/truncated_identity_training/k2/identity_adapter_diagnostic.pt")
    parser.add_argument("--k4", default="artifacts/reports/truncated_identity_training/k4/identity_adapter_diagnostic.pt")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    parser.add_argument("--model", default=str(soulx / "pretrained_models/SoulX-Singer/model-svc.pt"))
    parser.add_argument("--config", default=str(soulx / "soulxsinger/config/soulxsinger.yaml"))
    parser.add_argument("--rmvpe", default=str(soulx / "pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"))
    parser.add_argument("--whisper", default="data/cache/whisper-large-v3-turbo")
    parser.add_argument("--wavlm", default="data/cache/wavlm-base-plus-sv")
    parser.add_argument("--ecapa", default="data/cache/spkrec-ecapa-voxceleb")
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    try:
        report = render_matrix(args) if args.stage == "render" else analyze_matrix(args)
    except Exception as error:
        report = {
            "status": "diagnostic_reject", "stage": args.stage,
            "error_type": type(error).__name__, "error": str(error),
            "runtime_integration": False, "release_allowed": False,
        }
        (Path(args.output) / f"{args.stage}_failure.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    print(json.dumps({
        "status": report["status"], "stage": args.stage,
        "output_count": report.get("output_count"), "eligible_candidates": report.get("eligible_candidates"),
    }, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
