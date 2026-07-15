#!/usr/bin/env python3
"""Rank the bounded FM-Singer AMS speaker sweep on score-native stress cases."""

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
sys.path[:0] = [str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


CASES = {"rapid_ko": "빠르게노래하자아", "large_interval_ko": "높이날아"}


def main() -> None:
    config = json.loads((CACHE / "fm-singer/FMSinger/config.json").read_text())
    speakers = {
        name.lower(): speaker_id
        for name, speaker_id in config["speaker_elf"].items()
        if 0 <= speaker_id <= 18
    }
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    f0_extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    targets = {
        case: np.asarray(
            json.loads((ROOT / f"artifacts/reports/diffsinger_score_native_pilot/{case}.ds").read_text())[0]["f0_seq"].split(),
            dtype=np.float32,
        )
        for case in CASES
    }
    listening = ROOT / "artifacts/reports/fm_singer_score_probe/listening"
    rows = []
    for speaker, speaker_id in speakers.items():
        for case, lyric in CASES.items():
            path = listening / f"{case}_exact_score_duration_{speaker}_p12.wav"
            inputs = processor(audio=audio16(path), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = whisper.generate(
                    inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            rows.append({
                "speaker": speaker,
                "speaker_id": speaker_id,
                "case": case,
                "path": str(path.relative_to(ROOT)),
                "asr_transcript": transcript,
                "asr_lyric_similarity": round(
                    SequenceMatcher(None, normalized(lyric), normalized(transcript)).ratio(), 4
                ),
            } | acoustics(path) | pitch(path, targets[case], f0_extractor))
    metrics = (
        "asr_lyric_similarity", "pitch_mae_cents", "voicing_accuracy",
        "hf_spike_p99_over_median", "spectral_flux_p95", "sample_jump_p999",
    )
    def mean_metric(speaker: str, metric: str):
        values = [
            row[metric] for row in rows
            if row["speaker"] == speaker and row[metric] is not None
        ]
        return round(float(np.mean(values)), 6) if values else None

    aggregate = {
        speaker: {
            metric: mean_metric(speaker, metric)
            for metric in metrics
        } | {
            "min_asr": min(row["asr_lyric_similarity"] for row in rows if row["speaker"] == speaker),
            "max_pitch_mae_cents": max(
                (row["pitch_mae_cents"] for row in rows if row["speaker"] == speaker and row["pitch_mae_cents"] is not None),
                default=None,
            ),
        }
        for speaker in speakers
    }
    ranking = sorted(
        speakers,
        key=lambda name: (
            aggregate[name]["min_asr"],
            aggregate[name]["asr_lyric_similarity"],
            aggregate[name]["voicing_accuracy"] or 0.0,
            -aggregate[name]["hf_spike_p99_over_median"],
        ),
        reverse=True,
    )
    report = {
        "status": "bounded_speaker_selection_only",
        "checkpoint_license": "unresolved_evaluation_only",
        "pitch_convention": "input MIDI +12; no waveform pitch shifting",
        "duration_control": "exact note totals with 3/vowel/3 phone allocation",
        "ranking": ranking,
        "aggregate": aggregate,
        "rows": rows,
        "selected": ranking[0],
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/fm_singer_score_probe/speaker_sweep.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "selected": ranking[0],
        "top5": [{"speaker": name} | aggregate[name] for name in ranking[:5]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
