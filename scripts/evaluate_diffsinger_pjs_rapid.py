#!/usr/bin/env python3
"""Gate local PJS DiffSinger rapid-song probes without publishing lyrics/audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CACHE / "soulx-singer"))

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lyric_similarity(expected: list[str], transcript: str) -> float:
    return max(
        SequenceMatcher(None, normalized(value), normalized(transcript)).ratio()
        for value in expected
    )


def teacher_forced_nll(processor, model, audio: np.ndarray, expected: list[str]) -> float:
    inputs = processor(audio=audio, sampling_rate=16_000, return_tensors="pt")
    losses = []
    for text in expected:
        labels = processor.tokenizer(text, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            loss = model(input_features=inputs.input_features.cuda().half(), labels=labels).loss
        losses.append(float(loss))
    return min(losses)


def passes_gate(row: dict, *, asr_gate_valid: bool, reference_nll: float | None) -> bool:
    lyric_pass = (
        row["asr_lyric_similarity"] >= .8 if asr_gate_valid
        else reference_nll is not None and row["teacher_forced_lyric_nll"] <= reference_nll * 1.05
    )
    return bool(
        lyric_pass
        and row["pitch_p90_abs_cents"] <= 100
        and row["gross_error_over_600_cents"] <= .05
        and row["observed_voiced_ratio"] >= .8
        and row["clip_fraction"] == 0
    )


def pitch_errors(target: np.ndarray, observed: np.ndarray, *, max_frame_delta: int = 1) -> np.ndarray:
    """Compare equal-hop F0 without inventing pitch across unvoiced boundaries."""
    if abs(len(target) - len(observed)) > max_frame_delta:
        raise ValueError(
            f"F0 grid length mismatch exceeds {max_frame_delta} frame: "
            f"target={len(target)}, observed={len(observed)}"
        )
    frames = min(len(target), len(observed))
    target = target[:frames]
    observed = observed[:frames]
    both = (observed > 1) & (target > 1)
    if not np.any(both):
        raise ValueError("candidate and target have no jointly voiced F0 frames")
    return np.abs(1200 * np.log2(observed[both] / target[both]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=Path, required=True)
    parser.add_argument("--expected-text", action="append", required=True,
                        help="Expected lyric spelling; repeat for kanji/kana aliases")
    parser.add_argument("--candidate", action="append", required=True,
                        help="LABEL=/absolute/or/repository/path.wav")
    parser.add_argument("--reference-audio", type=Path,
                        help="Known-correct lyric audio used to calibrate the ASR/NLL gate")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/reports/diffsinger_pjs_rapid_evaluation.json")
    args = parser.parse_args()

    target = np.asarray(json.loads(args.ds.read_text())[0]["f0_seq"].split(), dtype=np.float32)
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24_000, hop_size=480, verbose=False,
    )

    reference_nll = None
    reference_asr_similarity = None
    if args.reference_audio is not None:
        reference_path = args.reference_audio if args.reference_audio.is_absolute() else ROOT / args.reference_audio
        reference_audio = audio16(reference_path)
        reference_inputs = processor(audio=reference_audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            reference_ids = whisper.generate(
                reference_inputs.input_features.cuda().half(),
                attention_mask=reference_inputs.attention_mask.cuda() if "attention_mask" in reference_inputs else None,
                language="ja", task="transcribe", max_new_tokens=64,
            )
        reference_transcript = processor.batch_decode(reference_ids, skip_special_tokens=True)[0]
        reference_asr_similarity = lyric_similarity(args.expected_text, reference_transcript)
        reference_nll = teacher_forced_nll(processor, whisper, reference_audio, args.expected_text)
    asr_gate_valid = reference_asr_similarity is None or reference_asr_similarity >= .8

    rows = []
    for value in args.candidate:
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        inputs = processor(audio=audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = whisper.generate(
                inputs.input_features.cuda().half(),
                attention_mask=inputs.attention_mask.cuda() if "attention_mask" in inputs else None,
                language="ja", task="transcribe",
                max_new_tokens=64,
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        cents = pitch_errors(target, observed)
        metrics = {
            "label": label,
            "audio_path": str(path.relative_to(ROOT)),
            "audio_sha256": sha256(path),
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(lyric_similarity(args.expected_text, transcript), 4),
            "teacher_forced_lyric_nll": round(
                teacher_forced_nll(processor, whisper, audio16(path), args.expected_text), 4
            ),
            "pitch_median_abs_cents": round(float(np.median(cents)), 2),
            "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 2),
            "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 4),
            "observed_voiced_ratio": round(float(np.mean(observed > 1)), 4),
        } | acoustics(path)
        metrics["pass"] = passes_gate(
            metrics, asr_gate_valid=asr_gate_valid, reference_nll=reference_nll
        )
        rows.append(metrics)

    candidate = max(rows, key=lambda row: (row["pass"], row["asr_lyric_similarity"],
                                           -row["pitch_p90_abs_cents"]))
    report = {
        "status": "source_probe_pass_human_pending" if candidate["pass"] else "source_probe_reject",
        "selected": candidate["label"],
        "gate": {
            "asr_lyric_similarity_min": .8 if asr_gate_valid else "invalid_for_this_reference",
            "teacher_forced_lyric_nll_max": (
                round(reference_nll * 1.05, 4) if reference_nll is not None else None
            ),
            "pitch_p90_abs_cents_max": 100,
            "gross_error_over_600_cents_max": .05,
            "observed_voiced_ratio_min": .8,
            "clip_fraction": 0,
        },
        "reference_calibration": {
            "used": args.reference_audio is not None,
            "free_asr_similarity": (round(reference_asr_similarity, 4)
                                    if reference_asr_similarity is not None else None),
            "teacher_forced_lyric_nll": (round(reference_nll, 4)
                                         if reference_nll is not None else None),
            "free_asr_gate_valid": asr_gate_valid,
        },
        "expected_text_sha256": [
            hashlib.sha256(normalized(value).encode("utf-8")).hexdigest()
            for value in args.expected_text
        ],
        "expected_text_characters": [len(normalized(value)) for value in args.expected_text],
        "rows": rows,
        "identity_adaptation_allowed": bool(candidate["pass"]),
        "release_allowed": False,
        "interpretation": "The known-correct reference calibrates lexical metrics; human listening is still mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
