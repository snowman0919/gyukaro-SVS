#!/usr/bin/env python3
"""Evaluate two final production phrases per language for the v1 release gate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


CASES = [(split, language) for split in ("quality", "heldout") for language in ("ko", "en", "ja")]


def compact(text: str) -> str:
    return re.sub(r"[^A-Za-z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def similarity(expected: str, actual: str) -> float:
    left, right = compact(expected), compact(actual)
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return float(1 - previous[-1] / max(len(left), len(right)))


def target_f0(score: dict, frames: int, seconds: float) -> np.ndarray:
    nominal = max(note["start"] + note["duration"] for note in score["notes"])
    times = np.arange(frames) * .02 * nominal / seconds
    values = np.zeros(frames, dtype=np.float32)
    pitch = score.get("curves", {}).get("pitch", [])
    residual = np.interp(times, [p["time"] for p in pitch], [p["value"] for p in pitch]) if pitch else 0
    for note in score["notes"]:
        active = (times >= note["start"]) & (times < note["start"] + note["duration"])
        values[active] = 440 * 2 ** ((note["pitch"] + (residual[active] if isinstance(residual, np.ndarray) else residual) - 69) / 12)
    return values


def audio_proxy(audio: np.ndarray) -> dict:
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) + 1e-10
    flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    return {"spectral_flatness": flatness, "crest_factor": float(np.max(np.abs(audio)) / max(np.sqrt(np.mean(audio ** 2)), 1e-8))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/reports/release_audio_v10.json")
    parser.add_argument("--soulx-root", default="data/cache/soulx-singer")
    parser.add_argument("--rmvpe", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")
    parser.add_argument("--whisper", default="data/cache/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sys.path.insert(0, args.soulx_root)
    from preprocess.tools.f0_extraction import F0Extractor

    identity = json.loads(Path("artifacts/reports/v07_identity_style_ablation.json").read_text())["identity_ablation"]["scores"]
    extractor = F0Extractor(args.rmvpe, device=args.device, target_sr=24_000, hop_size=480, verbose=False)
    rows = []
    for split, language in CASES:
        name = f"{split}_{language}"
        path = Path(f"artifacts/reports/v07_ablation_identity_student_{name}.wav")
        score = json.loads(Path(f"examples/{name}.json").read_text())
        stereo, rate = sf.read(path, dtype="float32", always_2d=True); audio = stereo.mean(1)
        f0 = np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32)
        target = target_f0(score, len(f0), len(audio) / rate)
        voiced = (f0 > 0) & (target > 0)
        cents = 1200 * np.log2(f0[voiced] / target[voiced])
        corr = float(np.corrcoef(np.log(f0[voiced]), np.log(target[voiced]))[0, 1]) if voiced.sum() > 2 else None
        student = identity[name]["variants"]["student"]
        silence = np.abs(audio) < .002
        rows.append({
            "id": name, "language": language, "audio": str(path), "score": f"examples/{name}.json",
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "expected_lyrics": "".join(n["lyric"] for n in score["notes"]),
            "sample_rate": rate, "channels": stereo.shape[1], "duration_seconds": round(len(audio) / rate, 4),
            "peak": round(float(np.max(np.abs(audio))), 6), "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
            "sample_silence_ratio": round(float(np.mean(silence)), 6), "voiced_ratio_rmvpe": round(float(np.mean(f0 > 0)), 4),
            "pitch_mae_cents": round(float(np.mean(np.abs(cents))), 2),
            "log_f0_correlation": None if corr is None or not np.isfinite(corr) else round(corr, 4),
            "wavlm_to_gyu": student["wavlm_to_gyu"], "ecapa_to_gyu": student["ecapa_to_gyu"],
        } | {key: round(value, 6) for key, value in audio_proxy(audio).items()})
    del extractor
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.whisper)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.whisper, torch_dtype=dtype).to(args.device).eval()
    for row in rows:
        audio, rate = sf.read(row["audio"], dtype="float32", always_2d=True); audio = audio.mean(1)
        if rate != 16_000:
            divisor = np.gcd(rate, 16_000); audio = resample_poly(audio, 16_000 // divisor, rate // divisor).astype(np.float32)
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(inputs.input_features.to(args.device, dtype), language=row["language"], task="transcribe", max_new_tokens=96)
        row["asr_transcript"] = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        row["asr_lyric_similarity"] = round(similarity(row["expected_lyrics"], row["asr_transcript"]), 4)

    languages = {}
    for language in ("ko", "en", "ja"):
        selected = [row for row in rows if row["language"] == language]
        languages[language] = {
            "phrases": len(selected),
            "mean_wavlm_to_gyu": round(float(np.mean([row["wavlm_to_gyu"] for row in selected])), 4),
            "mean_ecapa_to_gyu": round(float(np.mean([row["ecapa_to_gyu"] for row in selected])), 4),
            "mean_asr_lyric_similarity": round(float(np.mean([row["asr_lyric_similarity"] for row in selected])), 4),
            "mean_pitch_mae_cents": round(float(np.mean([row["pitch_mae_cents"] for row in selected])), 2),
            "mean_voiced_ratio": round(float(np.mean([row["voiced_ratio_rmvpe"] for row in selected])), 4),
            "prosody": "personalized real-GYU" if language == "ko" else "generic multilingual plus GYU identity/style",
        }
    gates = {
        "two_phrases_per_language": all(value["phrases"] == 2 for value in languages.values()),
        "formats_48k_mono": all(row["sample_rate"] == 48_000 and row["channels"] == 1 for row in rows),
        "no_clipping": all(row["peak"] < .999 for row in rows),
        "no_silence_anomaly": all(row["rms"] > .005 and row["sample_silence_ratio"] < .75 for row in rows),
        "pitch_adherence": all(row["pitch_mae_cents"] < 200 for row in rows),
        "lyrics_observed_by_asr": all(row["asr_lyric_similarity"] > 0 for row in rows),
        "speaker_metrics_present": all(np.isfinite(row["wavlm_to_gyu"]) and np.isfinite(row["ecapa_to_gyu"]) for row in rows),
    }
    report = {
        "backend": "gyu-singer-v0.8 production path", "phrases": rows, "languages": languages,
        "audio_quality_metric": "spectral flatness and crest factor are descriptive proxies; no MOS claim",
        "gates": {key: bool(value) for key, value in gates.items()}, "pass": all(gates.values()),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
