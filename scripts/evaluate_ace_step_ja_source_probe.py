#!/usr/bin/env python3
"""Gate two fixed-seed ACE-Step JA sources before any SoulX decode."""
from __future__ import annotations

import hashlib
import json
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
CACHE = RUNTIME_ROOT / "data/cache"
OUTPUT = ROOT / "artifacts/reports/ace_step_ja_content_source"
CASES = {
    "quality_ja": "空へ向かい歌おう小さな光を追う",
    "heldout_ja": "新しい歌を風に乗せて届ける",
}
sys.path[:0] = [str(ROOT / "scripts")]

from analyze_rc8_defects import FFT, spectral  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized  # noqa: E402
from probe_omnivoice_ja_duration_collapse import repeated_span  # noqa: E402


def source_gate(rows: list[dict]) -> bool:
    by_case = {row["case"]: row for row in rows}
    return set(by_case) == set(CASES) and all(
        by_case[case]["lyric_similarity"] >= .90 and not by_case[case]["repetition"]
        for case in CASES
    )


def main() -> None:
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16,
    ).cuda().eval()
    rows = []
    for case, lyrics in CASES.items():
        path = OUTPUT / "source" / f"{case}.wav"
        values = processor(audio16(path), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(values.input_features.cuda().half(), language="ja",
                                 task="transcribe", max_new_tokens=96)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        expected, observed = normalized(lyrics), normalized(transcript)
        repeat = repeated_span(expected, observed)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(1)
        rows.append({
            "case": case, "lyrics": lyrics, "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration_seconds": round(len(audio) / rate, 4),
            "whisper_transcript": transcript,
            "lyric_similarity": round(SequenceMatcher(None, expected, observed).ratio(), 4),
            "repeated_expected_span": repeat, "repetition": repeat is not None,
            "waveform": acoustics(path),
            "multi_resolution": {name: spectral(audio, *params) for name, params in FFT.items()},
        })
    del model
    torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 3 * len(rows)), constrained_layout=True)
    for index, row in enumerate(rows):
        audio, rate = sf.read(ROOT / row["path"], dtype="float32", always_2d=True)
        audio = audio.mean(1)
        axes[index, 0].plot(np.arange(len(audio)) / rate, audio, linewidth=.35)
        axes[index, 0].set_title(f"{row['case']}: waveform")
        for column, (resolution, (n_fft, hop)) in enumerate(FFT.items(), 1):
            magnitude = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)), ref=np.max,
            )
            axes[index, column].imshow(
                magnitude, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                extent=[0, len(audio) / rate, 0, rate / 2],
            )
            axes[index, column].set_ylim(0, min(12_000, rate / 2))
            axes[index, column].set_title(f"{row['case']}: {resolution} STFT")
    plot = OUTPUT / "waveform_multires_stft.png"
    fig.savefig(plot, dpi=120)
    plt.close(fig)

    passed = source_gate(rows)
    report = {
        "status": "source_gate_pass" if passed else "source_gate_reject",
        "model": "ACE-Step-v1-3.5B", "revision": "1bee4c9f5b43e30995f8d4d33b3919197ce1bd68",
        "license": "Apache-2.0", "seed": 101, "infer_steps": 20, "guidance_scale": 7,
        "rows": rows, "waveform_multires_stft": str(plot.relative_to(ROOT)),
        "soulx_decode_skipped": not passed, "runtime_integrated": False,
        "rc8_human_status": "pending", "rc9_started": False,
    }
    (OUTPUT / "source_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
