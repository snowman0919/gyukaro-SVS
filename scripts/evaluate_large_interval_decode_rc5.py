#!/usr/bin/env python3
"""Evaluate the controlled large-interval decoder sweep."""

from __future__ import annotations

import json
import os
import sys
import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch


CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache"))
sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("large_interval_ko", "rapid_ko"), default="large_interval_ko")
    args = parser.parse_args()
    stem = "large_interval" if args.case == "large_interval_ko" else "rapid"
    root = Path(f"artifacts/reports/rc5_{stem}_decode")
    report = json.loads((root / "manifest.json").read_text())
    target = np.load(
        f"artifacts/reports/rc5_candidate_core/{args.case}/canonical_f0.npy"
    )
    extractor = F0Extractor(
        str(
            CACHE
            / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"
        ),
        device="cuda",
        target_sr=24000,
        hop_size=480,
        verbose=False,
    )
    rows = []
    for row in report["rows"]:
        path = Path(row["path"])
        rows.append(row | acoustics(path) | pitch(path, target, extractor))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            CACHE / "whisper-large-v3-turbo", dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    expected = normalized("높이 날아" if args.case == "large_interval_ko" else "빠르게 노래하자")
    for row in rows:
        inputs = processor(
            audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt"
        )
        with torch.inference_mode():
            ids = model.generate(
                inputs.input_features.cuda().half(),
                language="ko",
                task="transcribe",
                max_new_tokens=32,
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        actual = normalized(transcript)
        matcher = SequenceMatcher(None, expected, actual)
        row["asr_transcript"] = transcript
        row["asr_lyric_similarity"] = round(matcher.ratio(), 4)
        row["asr_lyric_coverage"] = round(
            sum(block.size for block in matcher.get_matching_blocks()) / len(expected), 4
        )
    accepted = [
        row
        for row in rows
        if row["asr_lyric_coverage"] >= 0.8 and row["pitch_mae_cents"] < 30
    ]
    baseline = None
    if args.case == "rapid_ko":
        stress = json.loads(Path("artifacts/reports/rc5_stress_candidate4/evaluation.json").read_text())
        baseline = next(row for row in stress["rows"] if row["case"] == args.case)
        metrics = (
            "pitch_mae_cents",
            "hf_energy_ratio_p95",
            "hf_spike_p99_over_median",
            "spectral_flux_p95",
            "sample_jump_p999",
        )
        accepted = [
            row
            for row in accepted
            if row["name"] != "s64_c2_seed21"
            and row["asr_lyric_coverage"] >= baseline["asr_lyric_coverage"]
            and all(row[name] <= baseline[name] for name in metrics)
        ]
    selected = min(accepted, key=lambda row: row["hf_spike_p99_over_median"]) if accepted else None
    result = {
        "status": "selected_not_human_reviewed" if selected else "no_strict_improvement",
        "case": args.case,
        "selection_rule": (
            "coverage>=baseline and no worse on pitch/HF/flux/jump"
            if args.case == "rapid_ko"
            else "coverage>=0.8, pitch_mae<30c; then lowest HF spike"
        ),
        "selected": selected,
        "baseline": baseline,
        "rows": rows,
    }
    (root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
