#!/usr/bin/env python3
"""Qualify GTSinger Korean source rows before any DiffSinger training."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import re
import shutil
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
DOCUMENT = ROOT / "docs/gtsinger_ko_qualified_foundation.md"
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


def write_training_data(rows: list[dict], dataset_root: Path, output: Path) -> dict:
    """Export accepted rows deterministically without complete-song leakage."""
    output = Path(output)
    wavs = output / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(row["item_name"].encode()).digest(),
    )
    item_splits = song_splits(ordered)
    split_lookup = {
        item_name: split
        for split, item_names in item_splits.items()
        for item_name in item_names
    }
    split_names = {"train": [], "validation": [], "test": []}
    phones: set[str] = set()
    names_by_item: dict[str, str] = {}
    with (output / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for index, row in enumerate(ordered):
            name = f"gtsko{index:04d}"
            names_by_item[row["item_name"]] = name
            source_value = row.get("source_path", row.get("wav_fn"))
            source = Path(source_value)
            if not source.is_absolute():
                source = dataset_root / source
            if not source.is_file():
                raise FileNotFoundError(source)
            sequence = normalized_phones(row)
            durations = [float(value) for value in row["ph_durs"]]
            audio_duration = sf.info(source).duration
            delta = audio_duration - sum(durations)
            if delta > 0.001:
                sequence.append("SP")
                durations.append(delta)
            elif delta < -0.001:
                durations[-1] += delta
            if durations[-1] <= 0 or abs(sum(durations) - audio_duration) >= 0.002:
                raise ValueError(
                    f"invalid alignment for {row['item_name']}: {delta:+.4f}s"
                )
            target = wavs / f"{name}.wav"
            if not target.exists():
                target.symlink_to(source.resolve())
            writer.writerow({
                "name": name,
                "ph_seq": " ".join(sequence),
                "ph_dur": " ".join(f"{value:.7f}" for value in durations),
            })
            split_names[split_lookup[row["item_name"]]].append(name)
            phones.update(sequence)

    songs_by_split = {
        split: {
            item_name.split("#")[3]
            for item_name in item_names
        }
        for split, item_names in item_splits.items()
    }
    leakage = any(
        songs_by_split[left] & songs_by_split[right]
        for index, left in enumerate(songs_by_split)
        for right in list(songs_by_split)[index + 1:]
    )
    if leakage:
        raise RuntimeError("complete-song split leakage")
    split_record = {
        "splits": {name: sorted(values) for name, values in split_names.items()},
        "songs": {name: sorted(values) for name, values in songs_by_split.items()},
        "names_by_item": names_by_item,
    }
    (output / "split.json").write_text(
        json.dumps(split_record, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "rows": len(ordered),
        "phones": sorted(phones),
        "song_split_leakage": leakage,
        "split_counts": {name: len(values) for name, values in split_names.items()},
    }


def build_config(summary: dict, raw_dir: Path, dictionary: Path) -> dict:
    """Build the frozen 15k medium DiffSinger foundation configuration."""
    import yaml

    split = json.loads((raw_dir / "split.json").read_text())["splits"]
    base = yaml.safe_load((ROOT / "configs/diffsinger_pjs_compact.yaml").read_text())
    work = ROOT / "data/external/work/diffsinger_score_native"
    base.update({
        "dictionaries": {"ko": str(dictionary)},
        "datasets": [{
            "raw_data_dir": str(raw_dir),
            "speaker": "gts_ko_soprano_2",
            "spk_id": 0,
            "language": "ko",
            "test_prefixes": split["validation"] + split["test"],
        }],
        "binary_data_dir": str(work / "binary_gtsinger_ko_qualified"),
        "binarizer_cls": "diffsinger_neutral_augmentation_binarizer.NeutralAugmentationBinarizer",
        "finetune_enabled": False,
        "finetune_ckpt_path": None,
        "finetune_strict_shapes": True,
        "frozen_params": ["model.diffusion"],
        "hidden_size": 256,
        "num_heads": 4,
        "enc_layers": 6,
        "backbone_args": {
            "num_channels": 512,
            "num_layers": 6,
            "kernel_size": 15,
            "dropout_rate": 0.0,
            "use_conditioner_cache": True,
            "glu_type": "atanglu",
        },
        "shallow_diffusion_args": {
            "train_aux_decoder": True,
            "train_diffusion": False,
            "aux_decoder_arch": "convnext",
            "aux_decoder_grad": 0.1,
            "aux_decoder_args": {
                "num_channels": 384,
                "num_layers": 6,
                "kernel_size": 7,
                "dropout_rate": 0.1,
            },
        },
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-2.0, 12.0], "scale": 1.0},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": 0.5},
            "random_time_stretching": {"enabled": True, "range": [0.95, 4.5], "scale": 2.0},
        },
        "max_updates": 15000,
        "val_check_interval": 1000,
        "num_ckpt_keep": 5,
        "max_batch_frames": 16000,
        "max_batch_size": 6,
        "optimizer_args": {"lr": 0.0003},
        "work_dir": str(CACHE / "diffsinger/checkpoints/gtsinger_ko_qualified"),
    })
    return base


def prepare_training(protocol: dict) -> dict:
    summary = json.loads(SUMMARY.read_text())
    if not summary.get("training_allowed"):
        raise RuntimeError("source qualification rejected; training export is forbidden")
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]
    raw = ROOT / "data/external/work/diffsinger_score_native/raw/gtsinger_ko_qualified"
    export = write_training_data(rows, ROOT, raw)
    dictionary = ROOT / "data/external/work/diffsinger_score_native/dictionary-gtsinger-ko-qualified.txt"
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    dictionary.write_text("".join(
        f"{phone}\t{phone}\n" for phone in export["phones"] if phone not in ("AP", "SP")
    ))
    config = build_config(summary, raw, dictionary)
    import yaml

    config_path = ROOT / "configs/diffsinger_gtsinger_ko_qualified.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "status": "ready_for_binarization",
        "config": str(config_path.relative_to(ROOT)),
        "raw_data_dir": str(raw),
        "dictionary": str(dictionary),
    } | export


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


def environment_evidence() -> dict:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    disk = shutil.disk_usage(ROOT)
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    revision = subprocess.run(
        ["git", "-C", str(CACHE / "diffsinger"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "system_memory_bytes": memory_bytes,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "diffsinger_revision": revision,
        "versions": {
            name: version(name)
            for name in ("torchaudio", "transformers", "numpy", "scipy", "soundfile")
        },
    }


def record_environment() -> dict:
    summary = json.loads(SUMMARY.read_text())
    summary["environment"] = environment_evidence()
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def write_source_report(test_count: int, dataset_result: str, smoke_result: str) -> dict:
    summary = record_environment()
    if summary["status"] != "foundation_source_gate_reject":
        raise ValueError("source rejection report requires a rejected source gate")
    summary["verification"] = {
        "pytest_passed": test_count,
        "dataset_validation": dataset_result,
        "voicebank_factory_smoke": smoke_result,
    }
    summary["protocol_sha256"] = _sha256(PROTOCOL)
    summary["accepted_manifest_sha256"] = _sha256(MANIFEST)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    counts = summary["corpus"]["counts"]
    failed = ", ".join(summary["corpus"]["failed_minimums"])
    environment = summary["environment"]
    report = f'''NOT A RELEASE REPORT — FOUNDATION SOURCE REJECTED

# GTSinger Korean source-qualified foundation diagnostic

## Decision

- Conclusion: `foundation_source_gate_reject`
- Training allowed: false
- Candidate source rows: {summary["candidate_rows"]}
- Accepted rows: {summary["accepted_rows"]}
- Accepted duration: {counts["duration_seconds"]:.6f} seconds ({counts["duration_seconds"] / 60:.3f} minutes)
- Failed frozen minimums: `{failed}`
- Binarization, optimizer training, checkpoint selection, rendering, multilingual adaptation, GYU identity adaptation, runtime integration, packaging, and OpenUtau work were not started.

The frozen source gate required at least 200 accepted rows and 1,800 seconds. Only {counts["rows"]} rows and {counts["duration_seconds"]:.3f} seconds passed all row-level evidence. Identity or acoustic training cannot repair source labels that failed qualification, so the mandatory early stop was applied without changing a threshold.

## Frozen source and evidence

- Dataset: `{summary["dataset"]}` at `{summary["dataset_revision"]}`
- License: `{summary["dataset_license"]}`; local non-commercial experiment only
- Selection: Korean / KO-Soprano-2 / Control_Group
- Label status: `{summary["label_status"]}`
- Human verified: false
- Accepted compact manifest: `data/manifests/gtsinger_ko_source_qualified.jsonl`
- Compact summary: `artifacts/reports/gtsinger_ko_source_qualification/summary.json`
- Full local row evidence: `data/external/work/gtsinger_ko_source_qualification/all_rows.jsonl`
- Local source cache: `data/external/raw/gtsinger-lfs/`
- Original project recordings under `data/source/`: unchanged and uncommitted
- Frozen protocol SHA-256: `{summary["protocol_sha256"]}`
- Accepted manifest SHA-256: `{summary["accepted_manifest_sha256"]}`

Row rejection counts: `{json.dumps(summary["rejection_counts"], ensure_ascii=False)}`. Accepted stress coverage was fast={counts["fast_rows"]}, high-register={counts["high_register_rows"]}, sustained={counts["sustained_rows"]}, and large-interval={counts["large_interval_rows"]}; these do not override the failed row-count and duration minimums.

## Environment

- Python {environment["python"]}; PyTorch {environment["torch"]}; CUDA build {environment["cuda_build"]}
- GPU: {environment["gpu"]}; reported total memory {environment["gpu_total_bytes"]} bytes
- System memory: {environment["system_memory_bytes"]} bytes
- Disk at evidence freeze: {environment["disk_free_bytes"]} bytes free of {environment["disk_total_bytes"]}
- DiffSinger checkout: `{environment["diffsinger_revision"]}`
- Tools: `{json.dumps(environment["versions"], ensure_ascii=False)}`

## Status boundary

- This is not a trained Korean foundation and not a neural GYU SVS package.
- No generated WAV is presented as a usable singer.
- Production renderer, RC7/RC8 decisions, package configuration, and OpenUtau paths remain unchanged.
- Public release remains unauthorized; GTSinger-derived work is governed by CC BY-NC-SA 4.0.

## Repository verification

- Full pytest: {test_count} passed
- Dataset validation: `{dataset_result}`
- Existing voicebank factory smoke: `{smoke_result}`
- `git diff --check`: required clean before and after the evidence commit
- Protected production renderer/package/OpenUtau paths: unchanged from `9b443ee`
- Committed WAV/checkpoint/cache/external dataset: none

## Next valid requirement

The next valid path is a new rights-controlled, score-native GYU recording corpus with independently verified lyrics, phonemes, durations, and scores that satisfies the same frozen source minimums. Lowering the gate or training on the rejected rows is not an acceptable workaround.
'''
    DOCUMENT.write_text(report)
    return {
        "status": summary["status"],
        "report": str(DOCUMENT.relative_to(ROOT)),
        "training_started": False,
        "runtime_modified": False,
        "package_openutau": "blocked",
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
            key: (value.to(device, dtype=dtype) if key == "input_features" else value.to(device))
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--record-environment", action="store_true")
    parser.add_argument("--report-source", action="store_true")
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--dataset-result")
    parser.add_argument("--smoke-result")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.download:
        download_sources(protocol)
    if not any((args.qualify, args.prepare, args.record_environment, args.report_source)):
        parser.error("--qualify, --prepare, --record-environment, or --report-source is required")
    if args.qualify:
        summary = qualify(protocol)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["training_allowed"]:
            raise SystemExit(2)
    if args.prepare:
        print(json.dumps(prepare_training(protocol), ensure_ascii=False, indent=2))
    if args.record_environment and not args.report_source:
        print(json.dumps(record_environment(), ensure_ascii=False, indent=2))
    if args.report_source:
        if args.test_count is None or not args.dataset_result or not args.smoke_result:
            parser.error("--report-source requires --test-count, --dataset-result, and --smoke-result")
        print(json.dumps(
            write_source_report(args.test_count, args.dataset_result, args.smoke_result),
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
