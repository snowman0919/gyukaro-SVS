#!/usr/bin/env python3
"""Evaluate the controlled large-interval decoder sweep."""

from __future__ import annotations

import json
import os
import sys
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
    root = Path("artifacts/reports/rc5_large_interval_decode")
    report = json.loads((root / "manifest.json").read_text())
    target = np.load(
        "artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy"
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
    expected = normalized("높이 날아")
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
    selected = (
        min(
            accepted,
            key=lambda row: (
                -row["asr_lyric_similarity"],
                row["hf_spike_p99_over_median"],
                row["spectral_flux_p95"],
            ),
        )
        if accepted
        else None
    )
    result = {
        "status": "selected_not_human_reviewed" if selected else "no_acceptable_decode",
        "selection_rule": "coverage>=0.8, pitch_mae<30c; then ASR, HF spike, flux",
        "selected": selected,
        "rows": rows,
    }
    (root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
