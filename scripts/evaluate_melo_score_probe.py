#!/usr/bin/env python3
"""Measure Korean lexical retention and score control in the MeloTTS probe."""
from __future__ import annotations

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

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


TEXT = {"rapid_ko": "빠르게노래하자아", "large_interval_ko": "높이날아"}


def main() -> None:
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    root = ROOT / "artifacts/reports/melo_score_probe/listening"
    score_root = ROOT / "artifacts/reports/diffsinger_score_native_pilot"
    rows = []
    for case, expected in TEXT.items():
        target = np.array(
            json.loads((score_root / f"{case}.ds").read_text())[0]["f0_seq"].split(),
            dtype=np.float32,
        )
        for mode in ("predicted_duration", "exact_duration"):
            path = root / f"{case}_{mode}.wav"
            inputs = processor(audio=audio16(path), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = model.generate(
                    inputs.input_features.cuda().half(),
                    language="ko", task="transcribe", max_new_tokens=64,
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            rows.append({
                "case": case,
                "mode": mode,
                "path": str(path.relative_to(ROOT)),
                "asr_transcript": transcript,
                "asr_lyric_similarity": round(
                    SequenceMatcher(None, normalized(expected), normalized(transcript)).ratio(), 4
                ),
            } | acoustics(path) | pitch(path, target, extractor))
    metrics = (
        "asr_lyric_similarity", "pitch_mae_cents", "voicing_accuracy",
        "hf_spike_p99_over_median", "spectral_flux_p95", "sample_jump_p999",
    )
    aggregate = {
        mode: {
            metric: round(float(np.mean([
                row[metric] for row in rows if row["mode"] == mode and row[metric] is not None
            ])), 6)
            for metric in metrics
        }
        for mode in ("predicted_duration", "exact_duration")
    }
    lexical_pass = min(
        row["asr_lyric_similarity"] for row in rows if row["mode"] == "exact_duration"
    ) >= 0.8
    report = {
        "status": "lexical_prior_pass_pitch_adapter_required" if lexical_pass else "objective_reject_lexical_prior",
        "model": "myshell-ai/MeloTTS-Korean",
        "license": "MIT",
        "lexical_prior_pass": lexical_pass,
        "score_native": False,
        "exact_duration_control": True,
        "explicit_f0_control": False,
        "production_integration_allowed": False,
        "aggregate": aggregate,
        "rows": rows,
    }
    target = ROOT / "artifacts/reports/melo_score_probe/evaluation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
