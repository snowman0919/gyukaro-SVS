#!/usr/bin/env python3
"""Analyze the user-provided reference song without publishing derived audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import median_filter
from scipy.signal import correlate, correlation_lags, medfilt, resample_poly


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
WORK = ROOT / "data/external/work/rc9_reference"
REPORT = ROOT / "artifacts/reports/reference_song_rc9_analysis.json"
sys.path.insert(0, str(CACHE / "soulx-singer"))

from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mono(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(1), rate


def align(mix: np.ndarray, instrumental: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, int]:
    analysis_rate = 2_000
    a = resample_poly(mix, analysis_rate, rate)
    b = resample_poly(instrumental, analysis_rate, rate)
    length = min(len(a), len(b), 200 * analysis_rate)
    start = 5 * analysis_rate
    values = correlate(a[start:length], b[start:length], mode="full", method="fft")
    lags = correlation_lags(length - start, length - start)
    allowed = np.abs(lags) <= round(.25 * analysis_rate)
    lag = int(lags[allowed][np.argmax(values[allowed])])
    sample_lag = round(lag * rate / analysis_rate)
    if sample_lag >= 0:
        mix = mix[sample_lag:]
    else:
        instrumental = instrumental[-sample_lag:]
    length = min(len(mix), len(instrumental))
    return mix[:length], instrumental[:length], sample_lag


def subtract_instrumental(mix: np.ndarray, instrumental: np.ndarray) -> tuple[np.ndarray, dict]:
    n_fft, hop = 4096, 1024
    mix_stft = librosa.stft(mix, n_fft=n_fft, hop_length=hop)
    inst_stft = librosa.stft(instrumental, n_fft=n_fft, hop_length=hop)
    cross = np.sum(mix_stft * np.conj(inst_stft), axis=1)
    inst_power = np.sum(np.abs(inst_stft) ** 2, axis=1) + 1e-8
    transfer = cross / inst_power
    predicted = transfer[:, None] * inst_stft
    frame_gain = np.real(np.sum(mix_stft * np.conj(predicted), axis=0)) / (np.sum(np.abs(predicted) ** 2, axis=0) + 1e-8)
    frame_gain = median_filter(np.clip(frame_gain, .7, 1.3), size=101, mode="nearest")
    residual_stft = mix_stft - predicted * frame_gain[None]
    residual = librosa.istft(residual_stft, hop_length=hop, length=len(mix)).astype("float32")
    before = float(np.mean(mix**2))
    after = float(np.mean(residual**2))
    coherence = np.abs(cross) ** 2 / ((np.sum(np.abs(mix_stft) ** 2, axis=1) + 1e-8) * inst_power)
    return residual, {
        "method": "aligned provided-off-vocal complex STFT subtraction",
        "n_fft": n_fft,
        "hop": hop,
        "frame_gain_median": round(float(np.median(frame_gain)), 6),
        "frame_gain_p05_p95": [round(float(value), 6) for value in np.quantile(frame_gain, [.05, .95])],
        "median_frequency_coherence": round(float(np.median(coherence)), 6),
        "residual_to_mix_rms": round(float(np.sqrt(after / max(before, 1e-12))), 6),
    }


def spectral(audio: np.ndarray, rate: int, n_fft: int, hop: int) -> dict:
    magnitude = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) + 1e-8
    power = magnitude**2
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    instability = np.mean(np.abs(np.diff(np.log1p(magnitude), axis=1)), axis=0)
    return {
        "n_fft": n_fft,
        "hop": hop,
        "window_ms": round(1000 * n_fft / rate, 3),
        "spectral_flatness_mean": round(float(np.mean(flatness)), 7),
        "spectral_instability_p95": round(float(np.quantile(instability, .95)), 6),
    }


def tempo(instrumental: np.ndarray, rate: int) -> tuple[float, float, np.ndarray]:
    start, end = 20 * rate, min(len(instrumental), 190 * rate)
    audio = librosa.resample(instrumental[start:end], orig_sr=rate, target_sr=22_050)
    onset = librosa.onset.onset_strength(y=audio, sr=22_050, hop_length=512)
    bpm, beats = librosa.beat.beat_track(onset_envelope=onset, sr=22_050, hop_length=512, start_bpm=152)
    beat_times = librosa.frames_to_time(beats, sr=22_050, hop_length=512) + 20
    return float(np.asarray(bpm).item()), float(np.median(np.diff(beat_times))), beat_times


def note_candidates(f0: np.ndarray, frame_hz: int = 50) -> list[dict]:
    voiced = f0 > 0
    midi = np.zeros_like(f0, dtype="float32")
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440)
    smoothed = midi.copy()
    edges = np.flatnonzero(np.diff(np.r_[False, voiced, False]))
    for start, end in edges.reshape(-1, 2):
        length = end - start
        kernel = min(9, length if length % 2 else length - 1)
        smoothed[start:end] = medfilt(midi[start:end], kernel_size=max(1, kernel))
    quantized = np.rint(smoothed).astype(int)
    rows = []
    start = 0
    while start < len(f0):
        if not voiced[start]:
            start += 1
            continue
        pitch = quantized[start]
        end = start + 1
        while end < len(f0) and voiced[end] and abs(quantized[end] - pitch) <= 0:
            end += 1
        if end - start >= 3:
            local = f0[start:end]
            rows.append({
                "start": round(start / frame_hz, 4), "duration": round((end - start) / frame_hz, 4),
                "pitch": int(round(float(np.median(69 + 12 * np.log2(local / 440))))),
                "median_f0_hz": round(float(np.median(local)), 3), "source": "two_of_three_f0_consensus_inferred",
            })
        start = end
    merged = []
    for row in rows:
        if merged and row["pitch"] == merged[-1]["pitch"] and row["start"] - (merged[-1]["start"] + merged[-1]["duration"]) <= .06:
            merged[-1]["duration"] = round(row["start"] + row["duration"] - merged[-1]["start"], 4)
        else:
            merged.append(row)
    return merged


def consensus_f0(*tracks: np.ndarray, maximum_cents: float = 100) -> tuple[np.ndarray, np.ndarray]:
    length = min(map(len, tracks))
    values = np.stack([track[:length] for track in tracks])
    values[(values < 150) | (values > 1100)] = 0
    output = np.zeros(length, dtype="float32")
    support = np.zeros(length, dtype="uint8")
    for frame in range(length):
        active = values[:, frame][values[:, frame] > 0]
        if len(active) < 2:
            continue
        cents = np.abs(1200 * np.log2(active[:, None] / active[None, :]))
        np.fill_diagonal(cents, np.inf)
        first, second = np.unravel_index(np.argmin(cents), cents.shape)
        if cents[first, second] <= maximum_cents:
            agreed = active[np.min(cents, axis=1) <= maximum_cents]
            output[frame] = float(np.exp(np.mean(np.log(agreed))))
            support[frame] = len(agreed)
    # Bridge only tiny estimator dropouts; longer gaps remain unverified.
    edges = np.flatnonzero(np.diff(np.r_[False, output > 0, False]))
    for end, start in zip(edges[1::2], edges[2::2]):
        if 0 < start - end <= 2 and output[end - 1] > 0 and output[start] > 0:
            output[end:start] = np.geomspace(output[end - 1], output[start], start - end + 2)[1:-1]
            support[end:start] = 2
    return output, support


def self_test() -> None:
    f0 = np.r_[np.zeros(2), np.full(5, 440), np.zeros(2), np.full(5, 880)].astype("float32")
    notes = note_candidates(f0)
    assert [row["pitch"] for row in notes] == [69, 81]
    consensus, support = consensus_f0(np.array([440, 0, 440]), np.array([442, 0, 880]), np.array([0, 0, 441]))
    assert support.tolist() == [2, 2, 2] and np.all(consensus > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", type=Path)
    parser.add_argument("--instrumental", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    mix_path = args.mix or next(ROOT.glob("{original}*.mp3"))
    instrumental_path = args.instrumental or next(ROOT.glob("{MR}*.mp3"))
    WORK.mkdir(parents=True, exist_ok=True)
    mix, rate = mono(mix_path)
    instrumental, instrumental_rate = mono(instrumental_path)
    if rate != instrumental_rate:
        instrumental = resample_poly(instrumental, rate, instrumental_rate).astype("float32")
    aligned_mix, aligned_instrumental, lag = align(mix, instrumental, rate)
    vocal, separation = subtract_instrumental(aligned_mix, aligned_instrumental)
    vocal_path = WORK / "vocal_estimate.wav"
    mix_path_local = WORK / "aligned_mix.wav"
    instrumental_path_local = WORK / "aligned_instrumental.wav"
    sf.write(vocal_path, vocal, rate, subtype="PCM_24")
    sf.write(mix_path_local, aligned_mix, rate, subtype="PCM_24")
    sf.write(instrumental_path_local, aligned_instrumental, rate, subtype="PCM_24")
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda" if torch.cuda.is_available() else "cpu", target_sr=24_000, hop_size=480, verbose=False,
    )
    f0 = extractor.process(str(vocal_path))
    mix_f0 = extractor.process(str(mix_path_local))
    np.save(WORK / "rmvpe_f0_50hz.npy", f0)
    np.save(WORK / "mix_rmvpe_f0_50hz.npy", mix_f0)
    analysis_audio = librosa.resample(vocal, orig_sr=rate, target_sr=16_000)
    pyin, pyin_voiced, _ = librosa.pyin(analysis_audio, fmin=65, fmax=1100, sr=16_000, frame_length=2048, hop_length=320)
    np.save(WORK / "pyin_f0_50hz.npy", pyin)
    count = min(len(f0), len(pyin))
    valid = (f0[:count] > 0) & pyin_voiced[:count] & np.isfinite(pyin[:count])
    disagreement = np.abs(1200 * np.log2(f0[:count][valid] / pyin[:count][valid]))
    pyin_track = np.nan_to_num(pyin, nan=0).astype("float32")
    pyin_track[~pyin_voiced] = 0
    consensus, support = consensus_f0(f0, mix_f0, pyin_track)
    np.save(WORK / "consensus_f0_50hz.npy", consensus)
    bpm, beat_period, beats = tempo(aligned_instrumental, rate)
    np.save(WORK / "beat_times.npy", beats)
    notes = note_candidates(consensus)
    (WORK / "note_candidates.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    report = {
        "status": "local_reference_analyzed_score_review_pending",
        "copyright": "user-provided evaluation material; source audio and derived stems excluded from Git and package",
        "sources": {
            "mix": {"local_path": mix_path.name, "sha256": sha256(mix_path), "duration": round(len(mix) / rate, 3)},
            "instrumental": {"local_path": instrumental_path.name, "sha256": sha256(instrumental_path), "duration": round(len(instrumental) / rate, 3)},
            "lyrics": {"local_path": "lyrics.txt", "sha256": sha256(ROOT / "lyrics.txt")},
        },
        "audio": {"sample_rate": rate, "aligned_duration": round(len(vocal) / rate, 3), "instrumental_sample_lag": lag,
                  "instrumental_offset_seconds": round(lag / rate, 6)},
        "separation": separation | {"local_vocal_estimate": str(vocal_path.relative_to(ROOT)), "human_review": "pending"},
        "fft_definitions": {
            "short": spectral(vocal, rate, 512, 128), "medium": spectral(vocal, rate, 2048, 512),
            "long": spectral(vocal, rate, 8192, 2048),
        },
        "tempo": {"bpm_estimate": round(bpm, 6), "median_beat_period_seconds": round(beat_period, 6), "beat_count": len(beats)},
        "pitch": {
            "rmvpe_frames": len(f0), "rmvpe_voiced_ratio": round(float(np.mean(f0 > 0)), 6),
            "pyin_frames": len(pyin), "pyin_voiced_ratio": round(float(np.mean(pyin_voiced)), 6),
            "agreement_frames": int(np.sum(valid)),
            "agreement_median_cents": round(float(np.median(disagreement)), 3) if len(disagreement) else None,
            "agreement_p90_cents": round(float(np.quantile(disagreement, .9)), 3) if len(disagreement) else None,
            "two_of_three_consensus_frames": int(np.sum(support >= 2)),
            "three_way_consensus_frames": int(np.sum(support == 3)),
            "consensus_voiced_ratio": round(float(np.mean(consensus > 0)), 6),
        },
        "note_candidates": {"count": len(notes), "local_path": str((WORK / "note_candidates.json").relative_to(ROOT)),
                            "method": "50 Hz residual-RMVPE/original-mix-RMVPE/pYIN two-of-three <=100 cents, 180 ms median filter, piecewise semitone runs, >=60 ms"},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "lag": lag, "bpm": report["tempo"]["bpm_estimate"],
                      "notes": len(notes), "f0_agreement_p90": report["pitch"]["agreement_p90_cents"]}, indent=2))


if __name__ == "__main__":
    main()
