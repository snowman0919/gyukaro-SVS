#!/usr/bin/env python3
"""Qualify GTSinger Korean source rows before any DiffSinger training."""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, stft


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gyu_singer.experiments.korean_phones import mms_alignment_target

DATASET = ROOT / "data/external/raw/gtsinger-lfs"
WORK = ROOT / "data/external/work/gtsinger_ko_source_qualification"
PROTOCOL = ROOT / "configs/gtsinger_ko_qualified_protocol.json"
MANIFEST = ROOT / "data/manifests/gtsinger_ko_source_qualified.jsonl"
SUMMARY = ROOT / "artifacts/reports/gtsinger_ko_source_qualification/summary.json"
CACHE = ROOT / "data/cache"


def normalized_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


def normalized_phones(row: dict) -> list[str]:
    aliases = {"<SP>": "SP", "<AP>": "AP"}
    return [aliases.get(value, value) for value in row["ph"]]


def row_rejections(row: dict, measured: dict, gates: dict) -> list[str]:
    reasons: list[str] = []
    arrays = (
        row.get("ph"),
        row.get("ph_durs"),
        row.get("ep_pitches"),
        row.get("ep_notedurs"),
    )
    if (
        not all(isinstance(value, (list, tuple)) and value for value in arrays)
        or len({len(value) for value in arrays if value is not None}) != 1
    ):
        reasons.append("metadata_shape")

    duration = float(measured["audio_duration_seconds"])
    if not gates["duration_min_seconds"] <= duration <= gates["duration_max_seconds"]:
        reasons.append("duration")
    if (
        row.get("ph_durs")
        and abs(sum(map(float, row["ph_durs"])) - duration)
        > gates["duration_delta_max_seconds"]
    ):
        reasons.append("duration_alignment")
    if measured["clipping_samples"] > gates["clipping_samples_max"]:
        reasons.append("clipping")
    if measured["whisper_similarity"] < gates["whisper_similarity_min"]:
        reasons.append("whisper")
    if measured["ctc_coverage"] < gates["ctc_coverage_min"]:
        reasons.append("ctc_coverage")
    if measured["ctc_unknown_ratio"] > gates["ctc_unknown_ratio_max"]:
        reasons.append("ctc_unknown")
    if gates["ctc_monotonic_required"] and not measured["ctc_monotonic"]:
        reasons.append("ctc_monotonic")
    if not any(float(value) > 0 for value in row.get("ep_pitches", [])):
        reasons.append("no_voiced_note")
    if not any(value not in ("<SP>", "<AP>") for value in row.get("ph", [])):
        reasons.append("no_lexical_phone")
    return reasons


def corpus_summary(rows: list[dict], minimums: dict) -> dict:
    counts = {
        "rows": len(rows),
        "duration_seconds": sum(float(row["audio_duration_seconds"]) for row in rows),
        "fast_rows": sum(row["pace"] == "fast" for row in rows),
        "high_register_rows": sum(row["range"] == "high" for row in rows),
        "sustained_rows": sum(
            float(row["max_phone_duration_seconds"]) >= 1.0 for row in rows
        ),
        "large_interval_rows": sum(
            float(row["max_interval_semitones"]) >= 7.0 for row in rows
        ),
    }
    failures = [
        name for name, threshold in minimums.items() if counts[name] < threshold
    ]
    return {
        "status": (
            "source_qualification_pass"
            if not failures
            else "foundation_source_gate_reject"
        ),
        "counts": counts,
        "failed_minimums": failures,
        "training_allowed": not failures,
    }


def song_splits(rows: list[dict]) -> dict[str, list[str]]:
    songs = sorted({row["item_name"].split("#")[3] for row in rows})
    heldout = set(songs[-2:])
    validation = set(songs[-4:-2])
    split_by_song = {
        song: (
            "test" if song in heldout else "validation" if song in validation else "train"
        )
        for song in songs
    }
    result = {"train": [], "validation": [], "test": []}
    for row in rows:
        song = row["item_name"].split("#")[3]
        result[split_by_song[song]].append(row["item_name"])
    return {name: sorted(values) for name, values in result.items()}


def ctc_metrics(
    log_probs: torch.Tensor, target_ids: torch.Tensor, unknown_count: int = 0
) -> dict:
    """Return fail-closed target-conditioned MMS-CTC alignment evidence."""
    import torchaudio

    target_count = int(target_ids.shape[1]) + int(unknown_count)
    if target_count == 0 or target_ids.shape[1] == 0:
        return {
            "ctc_coverage": 0.0,
            "ctc_unknown_ratio": 1.0,
            "ctc_monotonic": False,
            "ctc_aligned_frame_ratio": 0.0,
            "ctc_mean_log_score": None,
        }
    try:
        alignment, scores = torchaudio.functional.forced_align(log_probs, target_ids)
        spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    except (RuntimeError, ValueError):
        spans = []
    frames = int(log_probs.shape[1])
    monotonic = len(spans) == target_ids.shape[1] and all(
        left.end <= right.start for left, right in zip(spans, spans[1:])
    )
    return {
        "ctc_coverage": float(len(spans) / target_count),
        "ctc_unknown_ratio": float(unknown_count / target_count),
        "ctc_monotonic": monotonic,
        "ctc_aligned_frame_ratio": float(
            sum(span.end - span.start for span in spans) / max(frames, 1)
        ),
        "ctc_mean_log_score": (
            float(np.mean([float(span.score) for span in spans])) if spans else None
        ),
    }


def _run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def download_sources(protocol: dict) -> None:
    """Fetch only the authorized KO-Soprano-2 Control_Group material."""
    repo = protocol["dataset"]
    revision = protocol["dataset_revision"]
    environment = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
    if not (DATASET / ".git").is_dir():
        DATASET.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git", "clone", "--depth", "1", "--no-checkout",
            f"https://huggingface.co/datasets/{repo}", str(DATASET), env=environment,
        )
        _run("git", "fetch", "--depth", "1", "origin", revision, cwd=DATASET)
        _run("git", "sparse-checkout", "init", "--no-cone", cwd=DATASET)
        _run("git", "checkout", revision, cwd=DATASET, env=environment)
    patterns = (
        "Korean/KO-Soprano-2/**/Control_Group/*.wav",
        "processed/Korean/metadata.json",
        "dataset_license.md",
    )
    _run("git", "sparse-checkout", "add", *patterns, cwd=DATASET)
    _run(
        "git", "lfs", "pull",
        "--include=Korean/KO-Soprano-2/**/Control_Group/*.wav,processed/Korean/metadata.json,dataset_license.md",
        cwd=DATASET,
    )


def ensure_metadata(protocol: dict) -> Path:
    metadata = DATASET / "processed/Korean/metadata.json"
    if metadata.is_file():
        return metadata
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(
        protocol["dataset"],
        "processed/Korean/metadata.json",
        repo_type="dataset",
        revision=protocol["dataset_revision"],
    ))
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.symlink_to(cached)
    return metadata


def selected_rows(rows: list[dict], protocol: dict) -> list[dict]:
    return [
        row for row in rows
        if row.get("language") == protocol["language"]
        and row.get("singer") == protocol["singer"]
        and f"#{protocol['group']}#" in row.get("item_name", "")
    ]


def _audio_metrics(path: Path) -> tuple[dict, np.ndarray, int]:
    stereo, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = stereo.mean(axis=1)
    jumps = np.abs(np.diff(audio))
    curvature = np.abs(np.diff(audio, n=2))
    metrics = {
        "audio_duration_seconds": float(len(audio) / rate),
        "sample_rate": int(rate),
        "channels": int(stereo.shape[1]),
        "clipping_samples": int(np.count_nonzero(np.abs(stereo) >= 0.999)),
        "sample_jump_p999": float(np.percentile(jumps, 99.9)) if len(jumps) else 0.0,
        "waveform_discontinuity_p999": (
            float(np.percentile(curvature, 99.9)) if len(curvature) else 0.0
        ),
    }
    for size in (256, 1024, 4096):
        if len(audio) < size:
            metrics[f"stft_spike_fft_{size}"] = 0.0
            continue
        frequencies, _, spectrum = stft(
            audio, fs=rate, nperseg=size, noverlap=size * 3 // 4, boundary=None
        )
        power = np.abs(spectrum).T ** 2 + 1e-12
        normalized = power / power.sum(axis=1, keepdims=True)
        flux = np.sqrt(np.sum(np.diff(normalized, axis=0) ** 2, axis=1))
        metrics[f"stft_spike_fft_{size}"] = (
            float(np.percentile(flux, 99)) if len(flux) else 0.0
        )
        if size == 1024:
            hf = normalized[:, frequencies >= min(8000, rate * 0.4)].sum(axis=1)
            metrics["hf_spike_p99_over_median"] = float(
                np.percentile(hf, 99) / max(np.median(hf), 1e-8)
            )
    return metrics, audio, rate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_text(row: dict) -> str:
    value = row.get("txt", "")
    parts = value if isinstance(value, list) else [value]
    return "".join(part for part in parts if part not in ("<SP>", "<AP>"))


def _max_interval(row: dict) -> float:
    voiced = [float(value) for value in row.get("ep_pitches", []) if float(value) > 0]
    return max((abs(right - left) for left, right in zip(voiced, voiced[1:])), default=0.0)


def _p95_baselines(rows: list[dict]) -> dict:
    names = (
        "hf_spike_p99_over_median",
        "sample_jump_p999",
        "waveform_discontinuity_p999",
        "stft_spike_fft_256",
        "stft_spike_fft_1024",
        "stft_spike_fft_4096",
    )
    return {
        name: (float(np.percentile([row[name] for row in rows], 95)) if rows else None)
        for name in names
    }


def qualify(protocol: dict) -> dict:
    """Measure all selected source rows once and persist compact evidence."""
    import torchaudio
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    metadata = ensure_metadata(protocol)
    candidates = selected_rows(json.loads(metadata.read_text()), protocol)
    if not candidates:
        raise RuntimeError("no authorized GTSinger Korean source rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_path = CACHE / "whisper-large-v3-turbo"
    processor = AutoProcessor.from_pretrained(whisper_path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_path, torch_dtype=dtype
    ).to(device).eval()
    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    label_ids = {label: index for index, label in enumerate(labels)}
    mms = bundle.get_model().to(device).eval()

    measured_rows = []
    for index, source_row in enumerate(candidates, 1):
        path = DATASET / source_row["wav_fn"]
        if not path.is_file():
            raise FileNotFoundError(f"source missing; rerun with --download: {path}")
        audio_metrics, audio, rate = _audio_metrics(path)
        divisor = math.gcd(rate, 16000)
        aligned = (
            resample_poly(audio, 16000 // divisor, rate // divisor).astype("float32")
            if rate != 16000 else audio
        )
        whisper_inputs = processor(
            aligned, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        generation = {
            key: value.to(device, dtype if key == "input_features" else None)
            for key, value in whisper_inputs.items()
        }
        with torch.inference_mode():
            tokens = whisper.generate(
                **generation, language="ko", task="transcribe", max_new_tokens=128
            )
            emission, _ = mms(torch.from_numpy(aligned)[None].to(device))
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        expected = _expected_text(source_row)
        target_string = mms_alignment_target(expected)
        known = [label_ids[character] for character in target_string if character in label_ids]
        unknown_count = len(target_string) - len(known)
        ctc = ctc_metrics(
            emission.log_softmax(-1),
            torch.tensor([known], device=device, dtype=torch.int32),
            unknown_count,
        )
        measured = audio_metrics | ctc | {
            "whisper_transcript": transcript,
            "whisper_similarity": SequenceMatcher(
                None, normalized_text(expected), normalized_text(transcript)
            ).ratio(),
            "audio_sha256": _sha256(path),
        }
        rejections = row_rejections(source_row, measured, protocol["row_gates"])
        measured_rows.append(source_row | measured | {
            "source_path": str(path.relative_to(ROOT)),
            "expected_text": expected,
            "max_phone_duration_seconds": max(map(float, source_row["ph_durs"])),
            "max_interval_semitones": _max_interval(source_row),
            "accepted": not rejections,
            "rejections": rejections,
            "label_status": "dataset_metadata_plus_measured_whisper_and_target_conditioned_mms_ctc",
        })
        print(
            f"qualify {index}/{len(candidates)} accepted={not rejections} "
            f"whisper={measured['whisper_similarity']:.3f} ctc={ctc['ctc_coverage']:.3f}",
            flush=True,
        )

    accepted = [row for row in measured_rows if row["accepted"]]
    corpus = corpus_summary(accepted, protocol["corpus_minimums"])
    summary = {
        "schema": 1,
        "status": corpus["status"],
        "training_allowed": corpus["training_allowed"],
        "dataset": protocol["dataset"],
        "dataset_revision": protocol["dataset_revision"],
        "dataset_license": protocol["dataset_license"],
        "selection": {
            "language": protocol["language"],
            "singer": protocol["singer"],
            "group": protocol["group"],
        },
        "candidate_rows": len(measured_rows),
        "accepted_rows": len(accepted),
        "rejection_counts": dict(Counter(
            reason for row in measured_rows for reason in row["rejections"]
        )),
        "corpus": corpus,
        "accepted_source_p95": _p95_baselines(accepted),
        "label_status": "dataset_metadata_plus_measured_whisper_and_target_conditioned_mms_ctc",
        "human_verified": False,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "all_rows.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in measured_rows
    ))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in accepted
    ))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.download:
        download_sources(protocol)
    if not args.qualify:
        parser.error("--qualify is required")
    summary = qualify(protocol)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["training_allowed"] else 2)


if __name__ == "__main__":
    main()
