#!/usr/bin/env python3
"""Deterministic original-recording index. No destructive source edits."""
from __future__ import annotations

import argparse, json, re, subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

PHASES = [(106, 119, "A"), (120, 153, "B"), (154, 170, "C"), (171, 202, "D"), (203, 222, "E"), (223, 232, "F"), (233, 237, "G")]

def probe(path: Path) -> dict:
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], text=True)
    meta = json.loads(raw); stream = next(s for s in meta["streams"] if s["codec_type"] == "audio")
    return {"codec": stream.get("codec_name"), "sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"]), "duration_sec": float(meta["format"]["duration"])}

def decode(path: Path, wav: Path) -> np.ndarray:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(wav)], check=True)
    audio, _ = sf.read(wav, dtype="float32"); return audio

def f0_autocorr(audio: np.ndarray, rate: int = 48000) -> tuple[float, float, float, float]:
    values = []
    for start in range(0, max(0, len(audio) - 2048), 480):
        frame = audio[start:start+2048] * np.hanning(2048)
        if np.sqrt(np.mean(frame * frame)) < 0.006: continue
        corr = np.correlate(frame, frame, "full")[2047:]
        lo, hi = int(rate / 800), int(rate / 65)
        lag = lo + int(np.argmax(corr[lo:hi]))
        if corr[lag] / max(corr[0], 1e-8) > 0.35: values.append(rate / lag)
    if not values: return 0.0, 0.0, 0.0, 0.0
    a = np.array(values); return float(np.median(a)), float(np.min(a)), float(np.max(a)), len(a) / max(1, (len(audio) - 2048) // 480)

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", default="data/source"); ap.add_argument("--processed", default="data/processed/master"); ap.add_argument("--manifest", default="data/manifests/real_recordings.jsonl"); args = ap.parse_args()
    source, processed = Path(args.source), Path(args.processed); processed.mkdir(parents=True, exist_ok=True)
    rows = []
    number = lambda p: int(re.search(r"(\d+)\.m4a$", p.name).group(1))
    for path in sorted(source.glob("*.m4a"), key=number):
        index = number(path); wav = processed / f"{index}.wav"
        try:
            meta = probe(path); audio = decode(path, wav); corrupt = False
        except (subprocess.CalledProcessError, StopIteration, KeyError, ValueError):
            # Preserve bad original and record it; never silently omit a source take.
            meta = {"codec": "alac", "sample_rate": 48000, "channels": 1, "duration_sec": 0.0}; audio = np.zeros(1, np.float32); corrupt = True
        rms = float(np.sqrt(np.mean(audio * audio))); peak = float(np.max(np.abs(audio))); active = np.abs(audio) > 0.015
        fmed, fmin, fmax, voiced = f0_autocorr(audio); block = next(block for first, last, block in PHASES if first <= index <= last)
        flags = ["script_text_unverified"] + (["decode_failed"] if corrupt else [])
        rows.append({"id": f"gyu_real_{index:06d}", "source_file": path.name, "source_index": index, "pcm_master": str(wav), **meta, "pcm_format": "s24le", "integrated_loudness_lufs_approx": round(20*np.log10(max(rms, 1e-8)), 2), "peak": round(peak, 5), "rms": round(rms, 5), "silence_ratio": round(1-float(active.mean()), 4), "active_voice_duration_sec": round(float(active.mean())*meta["duration_sec"], 3), "f0_median_hz": round(fmed, 2), "f0_min_hz": round(fmin, 2), "f0_max_hz": round(fmax, 2), "voiced_frame_ratio": round(voiced, 4), "clipping": bool(peak >= .995), "corrupt": corrupt, "script_block": block, "script_item": "unverified_source_order", "alignment_confidence": 0.35, "quality_flags": flags})
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True); Path(args.manifest).write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows))
    print(json.dumps({"recordings": len(rows), "duration_sec": round(sum(x["duration_sec"] for x in rows), 2), "sample_rates": sorted(set(x["sample_rate"] for x in rows)), "codecs": sorted(set(x["codec"] for x in rows))}))

if __name__ == "__main__": main()
