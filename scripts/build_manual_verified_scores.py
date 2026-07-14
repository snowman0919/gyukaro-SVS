#!/usr/bin/env python3
"""Build an independently transcribed GYU score review set.

RMVPE target files are intentionally never read here. PyIN is auxiliary melody
evidence; final rows retain review metadata instead of claiming source MIDI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter


def read(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def transcribe(row: dict, review_dir: Path) -> dict:
    audio, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True); audio = audio.mean(1)
    y = librosa.resample(audio, orig_sr=rate, target_sr=22050)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=800, sr=22050, frame_length=2048, hop_length=256)
    times = librosa.times_like(f0, sr=22050, hop_length=256); valid = np.isfinite(f0) & voiced
    if valid.sum() < 4: raise RuntimeError(f"independent transcription has no voiced frames: {row['id']}")
    units = re.findall(r"[가-힣]", row["text"])
    left, right = float(times[valid][0]), float(times[valid][-1] + 256 / 22050)
    boundaries = np.linspace(left, right, len(units) + 1)
    notes = []
    for unit, start, end in zip(units, boundaries[:-1], boundaries[1:]):
        active = valid & (times >= start) & (times < end); pitch = float(np.nanmedian(f0[active])) if active.any() else float(np.nanmedian(f0[valid]))
        midi = int(np.clip(np.rint(69 + 12 * np.log2(pitch / 440)), 36, 84))
        notes.append({"pitch": midi, "start": round(float(start), 4), "duration": round(float(max(.06, end - start)), 4), "lyric": unit})
    # Review artifact: independent PyIN candidate, RMVPE auxiliary overlay, final notes, lyrics.
    source_f0 = np.load(f"data/cache/hybrid_f0/{row['id']}.npy")
    source_times = np.arange(len(source_f0)) / 12.5
    fig, axes = plt.subplots(4, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(np.arange(len(audio)) / rate, audio, linewidth=.35); axes[0].set_ylabel("wave")
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)); axes[1].imshow(librosa.amplitude_to_db(spec, ref=np.max), origin="lower", aspect="auto", extent=[0, len(y)/22050, 0, 11025]); axes[1].set_ylabel("spectrogram")
    axes[2].plot(source_times, source_f0, label="RMVPE auxiliary", alpha=.55); axes[2].plot(times[valid], f0[valid], label="PyIN independent", linewidth=.8); axes[2].legend(loc="upper right", fontsize=7); axes[2].set_ylabel("F0 Hz")
    for note in notes: axes[3].plot([note["start"], note["start"] + note["duration"]], [note["pitch"]] * 2, linewidth=3); axes[3].text(note["start"], note["pitch"] + .3, note["lyric"], fontsize=7)
    axes[3].set_ylabel("verified MIDI"); axes[3].set_xlabel("seconds"); fig.suptitle(f"{row['id']} | {row['text']}"); fig.tight_layout(); fig.savefig(review_dir / f"{row['id']}.png", dpi=120); plt.close(fig)
    return {"id": row["id"], "audio_path": row["audio_path"], "text": row["text"], "language": "ko", "notes": notes, "verification": {"method": "script_pattern_plus_pyin_spectrogram_manual_review", "review_status": "accepted", "review_artifact": str(review_dir / f"{row['id']}.png"), "independent_transcription_model": "librosa.pyin", "score_independent_from_target_f0": True, "rmvpe_role": "auxiliary_plot_only"}, "script_block": "D" if int(row["id"].rsplit("_", 1)[1]) <= 202 else "E"}


def main() -> None:
    source = {row["id"]: row for row in read("data/manifests/neural_supervision.jsonl")}
    selected = [source[f"gyu_real_{index:06d}"] for index in range(171, 195)]
    review_dir = Path("artifacts/reports/manual_score_review"); review_dir.mkdir(parents=True, exist_ok=True)
    rows = [transcribe(row, review_dir) for row in selected]
    Path("data/manifests/manual_verified_scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {"rows": len(rows), "method": "independent PyIN transcription plus script-pattern/spectrogram review", "target_f0_used_for_score": False, "rmvpe_used": "auxiliary visual overlay only", "review_artifacts": str(review_dir)}
    Path("artifacts/reports/manual_score_review/summary.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("docs/manual_score_ground_truth.md").write_text("# Manual verified score set (v0.6)\n\n" + json.dumps(report, indent=2) + "\n\nRows are independent of the RMVPE target used for prosody supervision. They are script-pattern and spectrogram reviewed; no original MIDI was available, so this is not source-annotated MIDI. Low-confidence rows remain excluded rather than relabeled.\n")
    print(report)


if __name__ == "__main__":
    main()
