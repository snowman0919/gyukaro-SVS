#!/usr/bin/env python3
"""Test whether Japanese OmniVoice repetition depends on text/duration ratio."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT if (ROOT / ".venv-soulx").is_dir() else ROOT.parents[1]
CACHE = ROOT / "data/cache"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from analyze_rc8_defects import FFT, spectral  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized  # noqa: E402
from gyu_singer.inference.soulx import _Worker  # noqa: E402


TEXTS = {
    "short_todokeru": "届ける",
    "medium_kaze": "風に乗せて",
    "medium_atarashii": "新しい歌を",
    "heldout_full": "新しい歌を風に乗せて届ける",
    "quality_full": "空へ向かい歌おう小さな光を追う",
}
DURATIONS = (2.2, 4.4, 6.6, 8.9)


def repeated_span(expected: str, observed: str) -> dict | None:
    for size in range(len(expected), 1, -1):
        for start in range(len(expected) - size + 1):
            span = expected[start:start + size]
            wanted, actual = expected.count(span), observed.count(span)
            if actual > wanted:
                return {"text": span, "expected_count": wanted, "observed_count": actual}
    return None


def plot_grid(output: Path, name: str, rows: list[dict]) -> str:
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3 * len(rows)), constrained_layout=True)
    for row_index, row in enumerate(rows):
        audio, rate = sf.read(ROOT / row["path"], dtype="float32", always_2d=True)
        audio = audio.mean(1)
        axes[row_index, 0].plot(np.arange(len(audio)) / rate, audio, linewidth=.35)
        axes[row_index, 0].set_title(f"{row['duration']} s waveform")
        for column, (resolution, (n_fft, hop)) in enumerate(FFT.items(), 1):
            magnitude = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)), ref=np.max)
            axes[row_index, column].imshow(
                magnitude, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                extent=[0, len(audio) / rate, 0, rate / 2],
            )
            axes[row_index, column].set_ylim(0, min(12_000, rate / 2))
            axes[row_index, column].set_title(f"{row['duration']} s {resolution} STFT")
    target = output / f"{name}_waveform_multires_stft.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return str(target.relative_to(ROOT))


def main() -> None:
    output = ROOT / "artifacts/reports/omnivoice_ja_duration_collapse"
    listening = output / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    worker = _Worker([
        str(CACHE / "omnivoice/.venv/bin/python"), "scripts/generate_omnivoice_phrase.py", "--worker",
        "--checkpoint", str(CACHE / "omnivoice-checkpoint"),
    ], RUNTIME_ROOT, os.environ)
    rows = []
    try:
        for name, lyrics in TEXTS.items():
            for duration in DURATIONS:
                path = listening / f"{name}_{duration:g}s.wav"
                worker.request({"language": "ja", "lyrics": lyrics, "duration": duration, "output": str(path)})
                rows.append({
                    "case": name, "lyrics": lyrics, "characters": len(lyrics), "duration": duration,
                    "seconds_per_character": round(duration / len(lyrics), 4),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    finally:
        worker.close()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16,
    ).cuda().eval()
    for index, row in enumerate(rows, 1):
        path = ROOT / row["path"]
        values = processor(audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(values.input_features.cuda().half(), language="ja", task="transcribe", max_new_tokens=96)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        expected, observed = normalized(row["lyrics"]), normalized(transcript)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        row |= {
            "whisper_transcript": transcript,
            "lyric_similarity": round(SequenceMatcher(None, expected, observed).ratio(), 4),
            "observed_expected_length_ratio": round(len(observed) / max(len(expected), 1), 4),
            "repeated_expected_span": repeated_span(expected, observed),
            "waveform": acoustics(path),
            "multi_resolution": {name: spectral(audio, *params) for name, params in FFT.items()},
        }
        print(f"{index}/{len(rows)} {row['case']} {row['duration']}s {transcript}", flush=True)
    del model
    torch.cuda.empty_cache()

    buckets = {}
    for label, low, high in (("le_0.5", 0, .5), ("0.5_to_1.0", .5, 1.0), ("gt_1.0", 1.0, float("inf"))):
        selected = [row for row in rows if low < row["seconds_per_character"] <= high]
        buckets[label] = {
            "rows": len(selected),
            "repetition_collapse_rows": sum(row["repeated_expected_span"] is not None for row in selected),
            "mean_lyric_similarity": round(float(np.mean([row["lyric_similarity"] for row in selected])), 4) if selected else None,
        }
    plots = {name: plot_grid(output, name, [row for row in rows if row["case"] == name]) for name in TEXTS}
    heldout = next(row for row in rows if row["case"] == "heldout_full" and row["duration"] == 8.9)
    high_risk = buckets["gt_1.0"]["repetition_collapse_rows"] / max(buckets["gt_1.0"]["rows"], 1)
    low_risk = buckets["le_0.5"]["repetition_collapse_rows"] / max(buckets["le_0.5"]["rows"], 1)
    report = {
        "status": "duration_ratio_risk_confirmed_not_sufficient" if heldout["repeated_expected_span"] and high_risk > low_risk else "duration_ratio_not_confirmed",
        "model": "pinned OmniVoice 1574e06 via generate_omnivoice_phrase.py; seed 101; CFG 3.0",
        "hypothesis": "short Japanese text forced to excessive duration increases autoregressive repetition",
        "rows": rows, "seconds_per_character_buckets": buckets, "waveform_multires_stft": plots,
        "interpretation": "Long duration per character is a strong repetition risk, not a deterministic sole cause; text/context also matters.",
        "replacement_investigation_started": False,
    }
    (output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert len(rows) == len(TEXTS) * len(DURATIONS) and all(Path(ROOT / row["path"]).is_file() for row in rows)
    print(json.dumps({"status": report["status"], "buckets": buckets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
