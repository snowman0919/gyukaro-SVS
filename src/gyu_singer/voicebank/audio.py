from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aif", ".aiff"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _pitch_hz(audio: np.ndarray, sample_rate: int) -> float | None:
    active = audio[np.abs(audio) > max(1e-4, float(np.max(np.abs(audio))) * 0.05)]
    if len(active) < sample_rate // 20:
        return None
    window = active[: min(len(active), sample_rate)] * np.hanning(min(len(active), sample_rate))
    spectrum = np.abs(np.fft.rfft(window))
    frequencies = np.fft.rfftfreq(len(window), 1 / sample_rate)
    mask = (frequencies >= 60) & (frequencies <= 1000)
    return round(float(frequencies[mask][np.argmax(spectrum[mask])]), 3) if np.any(mask) else None


def _signature(audio: np.ndarray, sample_rate: int) -> list[float]:
    if not len(audio):
        return [0.0] * 10
    spectrum = np.abs(np.fft.rfft(audio[: min(len(audio), sample_rate)] * np.hanning(min(len(audio), sample_rate))))
    bands = np.array_split(spectrum, 8)
    values = [float(np.mean(band)) for band in bands]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values] + [float(np.sqrt(np.mean(audio * audio))), float(np.mean(audio))]


def inspect_audio_directory(directory: Path) -> dict:
    rows = []
    for path in sorted(item for item in directory.iterdir() if item.suffix.lower() in AUDIO_SUFFIXES):
        try:
            audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
            mono = _mono(audio)
            peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
            rows.append({
                "file": path.name,
                "path": str(path.resolve()),
                "status": "ok",
                "audio_sha256": _sha256(path),
                "sample_rate": sample_rate,
                "channels": audio.shape[1],
                "duration_seconds": round(len(mono) / sample_rate, 6),
                "silence_ratio": round(float(np.mean(np.abs(mono) < 1e-4)), 6) if len(mono) else 1.0,
                "clipping_ratio": round(float(np.mean(np.abs(mono) >= 0.999)), 8) if len(mono) else 0.0,
                "dc_offset": round(float(np.mean(mono)), 8) if len(mono) else 0.0,
                "noise_estimate": round(float(np.percentile(np.abs(mono), 10)), 8) if len(mono) else 0.0,
                "peak": round(peak, 8),
                "pitch_hz_estimate": _pitch_hz(mono, sample_rate),
                "language_estimate": "unknown_auxiliary",
                "acoustic_signature": _signature(mono, sample_rate),
            })
        except Exception as error:
            rows.append({"file": path.name, "path": str(path.resolve()), "status": "corrupt", "error": type(error).__name__})
    hashes: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] == "ok":
            hashes.setdefault(row["audio_sha256"], []).append(row["file"])
    duplicate_groups = sorted(sorted(group) for group in hashes.values() if len(group) > 1)
    valid = [row for row in rows if row["status"] == "ok"]
    if valid:
        signatures = np.asarray([row["acoustic_signature"] for row in valid], dtype=np.float64)
        center = np.median(signatures, axis=0)
        distances = np.linalg.norm(signatures - center, axis=1)
        scale = float(np.median(np.abs(distances - np.median(distances)))) or 1.0
        for row, distance in zip(valid, distances):
            row["speaker_outlier_score"] = round(float(distance / scale), 6)
            row["speaker_outlier_method"] = "acoustic_signature_heuristic"
    near_duplicates = []
    for index, left in enumerate(valid):
        left_vector = np.asarray(left["acoustic_signature"][:8])
        for right in valid[index + 1:]:
            if left["audio_sha256"] == right["audio_sha256"]:
                continue
            right_vector = np.asarray(right["acoustic_signature"][:8])
            score = float(np.dot(left_vector, right_vector) / ((np.linalg.norm(left_vector) * np.linalg.norm(right_vector)) or 1.0))
            if score >= 0.999:
                near_duplicates.append({"files": [left["file"], right["file"]], "cosine": round(score, 6), "method": "spectral_signature_heuristic"})
    return {
        "status": "audio_inspection_complete",
        "source_overwritten": False,
        "rows": rows,
        "valid_count": len(valid),
        "corrupt_count": sum(row["status"] == "corrupt" for row in rows),
        "duplicate_groups": duplicate_groups,
        "near_duplicates": near_duplicates,
    }


def normalize_audio(source: Path, destination: Path, sample_rate: int = 48_000) -> dict:
    audio, source_rate = sf.read(source, always_2d=True, dtype="float32")
    mono = _mono(audio)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor).astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, sample_rate, subtype="PCM_16")
    return {"path": str(destination), "sample_rate": sample_rate, "channels": 1, "sha256": _sha256(destination)}


def energy_vad(path: Path) -> dict:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = _mono(audio)
    frame = max(1, int(sample_rate * 0.02))
    hop = max(1, int(sample_rate * 0.01))
    starts = list(range(0, max(len(mono) - frame + 1, 1), hop))
    energy = np.asarray([float(np.sqrt(np.mean(mono[start:start + frame] ** 2))) for start in starts])
    threshold = max(1e-4, float(np.max(energy)) * 0.1) if len(energy) else 1e-4
    active = [index for index, value in enumerate(energy) if value >= threshold]
    segments = []
    if active:
        first = previous = active[0]
        for index in active[1:]:
            if index != previous + 1:
                segments.append([round(starts[first] / sample_rate, 6), round(min(len(mono), starts[previous] + frame) / sample_rate, 6)])
                first = index
            previous = index
        segments.append([round(starts[first] / sample_rate, 6), round(min(len(mono), starts[previous] + frame) / sample_rate, 6)])
    return {"source": "energy_vad", "threshold_rms": round(threshold, 8), "segments_seconds": segments}
