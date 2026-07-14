#!/usr/bin/env python3
"""Measure full-song and phrase-boundary release quality on an OpenUtau export."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def frame_rms(audio: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    size, hop = max(1, round(rate * .02)), max(1, round(rate * .01))
    frames = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop]
    return np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64)), hop / rate


def longest_true(values: np.ndarray, step: float) -> float:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best * step


def centroid(audio: np.ndarray, rate: int) -> float:
    if not np.any(audio):
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    frequencies = np.fft.rfftfreq(len(audio), 1 / rate)
    return float(np.sum(spectrum * frequencies) / max(np.sum(spectrum), 1e-12))


def load_f0(path: Path, soulx_root: Path, checkpoint: Path, device: str) -> tuple[np.ndarray, float]:
    sys.path.insert(0, str(soulx_root))
    from preprocess.tools.f0_extraction import F0Extractor

    extractor = F0Extractor(str(checkpoint), device=device, target_sr=24_000, hop_size=480, verbose=False)
    return np.asarray(extractor.process(str(path), verbose=False), dtype=np.float32), .02


def median_f0(f0: np.ndarray, step: float, start: float, end: float) -> float | None:
    values = f0[max(0, round(start / step)):max(0, round(end / step))]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if len(values) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--manifest", default="artifacts/reports/longform_v10_manifest.json")
    parser.add_argument("--render-metrics", required=True)
    parser.add_argument("--output", default="artifacts/reports/longform_v10_quality.json")
    parser.add_argument("--soulx-root", default="data/cache/soulx-singer")
    parser.add_argument("--rmvpe", default="data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    path = Path(args.audio)
    manifest = json.loads(Path(args.manifest).read_text())
    render = json.loads(Path(args.render_metrics).read_text())
    stereo, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = stereo.mean(axis=1)
    duration = len(audio) / rate
    f0, f0_step = load_f0(path, Path(args.soulx_root), Path(args.rmvpe), args.device)

    boundaries = manifest["boundaries"]
    starts = [0.0] + [row["seconds"] + row["expected_gap_seconds"] for row in boundaries]
    ends = [row["seconds"] for row in boundaries] + [duration]
    phrases = []
    fingerprints: dict[str, list[int]] = {}
    for index, (start, end) in enumerate(zip(starts, ends), 1):
        segment = audio[round(start * rate):round(end * rate)]
        framed, step = frame_rms(segment, rate)
        fingerprint = hashlib.sha256(np.round(segment * 32767).astype("<i2").tobytes()).hexdigest()
        fingerprints.setdefault(fingerprint, []).append(index)
        phrases.append({
            "index": index, "start_seconds": round(start, 4), "end_seconds": round(end, 4),
            "rms": round(rms(segment), 6), "peak": round(float(np.max(np.abs(segment))), 6),
            "near_silence_ratio": round(float(np.mean(framed < .002)), 6),
            "longest_near_silence_seconds": round(longest_true(framed < .002, step), 4),
            "sha256_pcm16": fingerprint,
        })

    boundary_rows = []
    for index, boundary in enumerate(boundaries, 1):
        at, gap = boundary["seconds"], boundary["expected_gap_seconds"]
        sample = min(len(audio) - 1, max(1, round(at * rate)))
        left = audio[max(0, sample - round(.1 * rate)):sample]
        right_start = round((at + gap) * rate)
        right = audio[right_start:min(len(audio), right_start + round(.1 * rate))]
        left_rms, right_rms = rms(left), rms(right)
        energy_db = abs(20 * np.log10(max(left_rms, 1e-8) / max(right_rms, 1e-8)))
        spectral_hz = abs(centroid(left, rate) - centroid(right, rate))
        click = float(abs(audio[sample] - audio[sample - 1]))
        left_f0 = median_f0(f0, f0_step, at - .2, at)
        right_f0 = median_f0(f0, f0_step, at + gap, at + gap + .2)
        f0_cents = None if left_f0 is None or right_f0 is None else abs(1200 * np.log2(right_f0 / left_f0))
        if gap:
            gap_audio = audio[sample:round((at + gap) * rate)]
            gap_frames, gap_step = frame_rms(gap_audio, rate)
            silence_seconds = longest_true(gap_frames < .002, gap_step)
            silence_pass = silence_seconds >= max(0, gap - .08)
        else:
            around = audio[max(0, sample - round(.08 * rate)):min(len(audio), sample + round(.08 * rate))]
            around_frames, gap_step = frame_rms(around, rate)
            silence_seconds = longest_true(around_frames < .002, gap_step)
            # Up to 150 ms can be an intentional unvoiced onset/coda, not a dropped phrase.
            silence_pass = silence_seconds <= .15
        continuity_pass = True
        if boundary["continuous_pitch_expected"]:
            continuity_pass = f0_cents is not None and f0_cents <= 200
        checks = {
            "click": click <= .2,
            "energy": gap > 0 or energy_db <= 40,
            "spectral": gap > 0 or spectral_hz <= 2500,
            "silence": silence_pass,
            "f0": continuity_pass,
        }
        boundary_rows.append({
            "index": index, "seconds": round(at, 4), "expected_gap_seconds": round(gap, 4),
            "continuous_pitch_expected": boundary["continuous_pitch_expected"],
            "sample_jump": round(click, 6), "energy_delta_db": round(float(energy_db), 3),
            "spectral_centroid_delta_hz": round(float(spectral_hz), 2),
            "measured_silence_seconds": round(float(silence_seconds), 4),
            "f0_delta_cents": None if f0_cents is None else round(float(f0_cents), 2),
            "checks": {key: bool(value) for key, value in checks.items()}, "pass": all(checks.values()),
        })

    duplicate_groups = [indexes for indexes in fingerprints.values() if len(indexes) > 1]
    clip_fraction = float(np.mean(np.abs(audio) >= .999))
    gates = {
        "duration_2_to_4_minutes": 119.5 <= duration <= 240.5,
        "all_17_phrases_present": len(phrases) == 17 and all(row["rms"] >= .005 for row in phrases),
        "no_long_unexpected_phrase_silence": all(row["longest_near_silence_seconds"] <= .6 for row in phrases),
        "no_exact_duplicate_phrases": not duplicate_groups,
        "no_clipping": float(np.max(np.abs(audio))) < .999 and clip_fraction == 0,
        "all_boundaries_pass": all(row["pass"] for row in boundary_rows),
        "openutau_render_complete": render["failed_phrases"] == 0 and render["cache_misses"] == 17,
        "cache_reused": render["cache_hits"] == 17 and render["stale_cache_files_after_repeat"] == 0,
    }
    report = {
        "audio": str(path), "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sample_rate": rate, "channels": stereo.shape[1], "duration_seconds": round(duration, 4),
        "peak": round(float(np.max(np.abs(audio))), 6), "rms": round(rms(audio), 6),
        "clip_fraction": round(clip_fraction, 10), "rmvpe_frames": len(f0),
        "thresholds": {"near_silence_rms": .002, "max_phrase_silence_seconds": .6, "max_click": .2,
                       "max_continuous_energy_delta_db": 40, "max_continuous_spectral_delta_hz": 2500,
                       "max_continuous_f0_delta_cents": 200, "max_unvoiced_boundary_seconds": .15,
                       "gap_tolerance_seconds": .08, "duration_tolerance_seconds": .5},
        "phrases": phrases, "exact_duplicate_phrase_groups": duplicate_groups, "boundaries": boundary_rows,
        "render_metrics": render, "gates": {key: bool(value) for key, value in gates.items()}, "pass": all(gates.values()),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
