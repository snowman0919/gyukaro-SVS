#!/usr/bin/env python3
"""Measure score control and artifact proxies for score-native pilot outputs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor

CASES = {
    "rapid_ko": ROOT / "examples/review_rapid_ko.json",
    "large_interval_ko": ROOT / "examples/review_large_interval_ko.json",
}
RC6 = {
    "rapid_ko": ROOT / "artifacts/reports/rc6_listening_gate/06_rapid_ko.wav",
    "large_interval_ko": ROOT / "artifacts/reports/rc6_listening_gate/08_large_interval_ko.wav",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = ROOT / "artifacts/reports/diffsinger_score_native_pilot"
    checkpoints = {step: ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_pilot/model_ckpt_steps_{step}.ckpt" for step in (1000, 2000)}
    targets = {
        case: np.array(json.loads((root / f"{case}.ds").read_text())[0]["f0_seq"].split(), dtype=np.float32)
        for case in CASES
    }
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    rows = []
    for case in CASES:
        paths = {"rc6": RC6[case]} | {
            f"steps{step}": root / "listening" / f"{case}_steps{step}.wav"
            for step in checkpoints
        }
        for model, path in paths.items():
            rows.append({
                "case": case,
                "model": model,
                "path": str(path.relative_to(ROOT)),
                "sample_rate": sf.info(path).samplerate,
            } | acoustics(path) | pitch(path, targets[case], extractor))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16
    ).cuda().eval()
    for row in rows:
        score = json.loads(CASES[row["case"]].read_text())
        expected = normalized("".join(note["lyric"] for note in score["notes"]))
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(
                inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_transcript"] = transcript
        row["asr_lyric_similarity"] = round(
            SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4
        )

    metrics = (
        "pitch_mae_cents", "voicing_accuracy", "hf_spike_p99_over_median",
        "spectral_flux_p95", "sample_jump_p999", "asr_lyric_similarity",
    )
    aggregate = {
        model: {
            metric: round(float(np.mean([row[metric] for row in rows if row["model"] == model and row[metric] is not None])), 6)
            for metric in metrics
        }
        for model in ("rc6", "steps1000", "steps2000")
    }
    eligible = (
        aggregate["steps2000"]["pitch_mae_cents"] <= 100
        and aggregate["steps2000"]["voicing_accuracy"] >= 0.8
        and aggregate["steps2000"]["asr_lyric_similarity"] >= 0.8
    )
    report = {
        "status": "objective_probe_pass_human_pending" if eligible else "objective_reject_undertrained",
        "human_listening": "pending" if eligible else "not_requested_objective_reject",
        "score_native": True,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "training_validation_loss": {"500": 0.22407, "1000": 0.21156, "1500": 0.21941, "2000": 0.19398},
        "checkpoint_sha256": {str(step): sha256(path) for step, path in checkpoints.items()},
        "aggregate": aggregate,
        "rows": rows,
        "interpretation": "Objective metrics can reject a pilot, but cannot pass listening or make this an RC.",
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
