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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=Path, required=True)
    parser.add_argument("--expected-text", action="append", required=True,
                        help="Expected lyric spelling; repeat for kanji/kana aliases")
    parser.add_argument("--candidate", action="append", required=True,
                        help="LABEL=/absolute/or/repository/path.wav")
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

    rows = []
    for value in args.candidate:
        label, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        inputs = processor(audio=audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = whisper.generate(
                inputs.input_features.cuda().half(), language="ja", task="transcribe",
                max_new_tokens=64,
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        observed = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        aligned = np.interp(
            np.arange(len(observed)), np.linspace(0, len(observed) - 1, len(target)), target
        )
        both = (observed > 1) & (aligned > 1)
        cents = np.abs(1200 * np.log2(observed[both] / aligned[both]))
        metrics = {
            "label": label,
            "audio_path": str(path.relative_to(ROOT)),
            "audio_sha256": sha256(path),
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(lyric_similarity(args.expected_text, transcript), 4),
            "pitch_median_abs_cents": round(float(np.median(cents)), 2),
            "pitch_p90_abs_cents": round(float(np.percentile(cents, 90)), 2),
            "gross_error_over_600_cents": round(float(np.mean(cents > 600)), 4),
            "observed_voiced_ratio": round(float(np.mean(observed > 1)), 4),
        } | acoustics(path)
        metrics["pass"] = (
            metrics["asr_lyric_similarity"] >= .8
            and metrics["pitch_p90_abs_cents"] <= 100
            and metrics["gross_error_over_600_cents"] <= .05
            and metrics["observed_voiced_ratio"] >= .8
            and metrics["clip_fraction"] == 0
        )
        rows.append(metrics)

    candidate = max(rows, key=lambda row: (row["pass"], row["asr_lyric_similarity"],
                                           -row["pitch_p90_abs_cents"]))
    report = {
        "status": "source_probe_pass_human_pending" if candidate["pass"] else "source_probe_reject",
        "selected": candidate["label"],
        "gate": {
            "asr_lyric_similarity_min": .8,
            "pitch_p90_abs_cents_max": 100,
            "gross_error_over_600_cents_max": .05,
            "observed_voiced_ratio_min": .8,
            "clip_fraction": 0,
        },
        "expected_text_sha256": [
            hashlib.sha256(normalized(value).encode("utf-8")).hexdigest()
            for value in args.expected_text
        ],
        "expected_text_characters": [len(normalized(value)) for value in args.expected_text],
        "rows": rows,
        "identity_adaptation_allowed": bool(candidate["pass"]),
        "release_allowed": False,
        "interpretation": "This objective source gate can reject a model; human listening is still mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
