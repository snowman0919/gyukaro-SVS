#!/usr/bin/env python3
"""CTC-align every usable scripted GYU recording for acoustic-only training."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio

from align_real_phonemes import align, roman_syllables


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[dict]:
    path = ROOT / "data/manifests" / name
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    recordings = {row["id"]: row for row in read("real_recordings.jsonl")}
    independent = {row["id"] for row in read("manual_verified_scores.jsonl")}
    candidates = []
    rejected = []
    for segment in read("real_segments.jsonl"):
        recording = recordings[segment["id"]]
        reasons = []
        if segment["id"] in independent:
            reasons.append("independent_evaluation_holdout")
        if recording["corrupt"] or recording["clipping"]:
            reasons.append("audio_integrity")
        if recording["active_voice_duration_sec"] < 1.0:
            reasons.append("insufficient_active_voice")
        if segment["alignment_confidence"] < 0.7:
            reasons.append("script_alignment_confidence")
        if not roman_syllables(segment["text"])[0]:
            reasons.append("no_hangul_target")
        if reasons:
            rejected.append({"id": segment["id"], "reasons": reasons})
            continue
        candidates.append(segment | {"audio_path": str(ROOT / recording["pcm_master"])})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.MMS_FA
    labels = bundle.get_labels()
    model = bundle.get_model().to(device).eval()
    dictionary = {label: index for index, label in enumerate(labels)}
    output_rows = []
    for row in candidates:
        try:
            result = align(row, model, dictionary, labels, device)
        except (RuntimeError, KeyError, IndexError) as error:
            rejected.append({"id": row["id"], "reasons": [f"ctc_alignment:{type(error).__name__}"]})
            continue
        result.update({
            "label_status": "inferred",
            "training_role": "real_gyu_acoustic_alignment_not_score_ground_truth",
            "script_block": row["script_block"],
        })
        output_rows.append(result)

    output = ROOT / "data/manifests/real_phoneme_alignment_all.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows))
    report = {
        "status": "all_usable_real_recordings_aligned",
        "source_recordings": len(recordings),
        "aligned_rows": len(output_rows),
        "rejected_rows": len(rejected),
        "independent_rows_excluded": sum("independent_evaluation_holdout" in row["reasons"] for row in rejected),
        "alignment": "MMS multilingual CTC plus singing vowel-duration prior",
        "labels": "inferred; acoustic training only; not score ground truth",
        "rejections": rejected,
    }
    path = ROOT / "artifacts/reports/real_phoneme_alignment_all.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rejections"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
