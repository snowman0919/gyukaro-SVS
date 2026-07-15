#!/usr/bin/env python3
"""Measure RC8 defects on the frozen RC7 and its existing ablations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gyu_singer.inference.content_timing import score_phone_targets  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402


VARIANTS = {
    "B_no_new_spectral": "artifacts/reports/rc6_backend_candidate/manifest.json",
    "D_lower_025": "artifacts/reports/spectral_refiner_stress_s025/manifest.json",
    "A_C_rc7_current_050": "artifacts/reports/spectral_refiner_stress_s050/manifest.json",
    "D_higher_100": "artifacts/reports/spectral_refiner_stress_s100/manifest.json",
}
GROUPS = {
    "sustained_noise": ["sustained_ko"],
    "english_transition": ["en"],
    "korean_overconnection": ["ko_neutral", "phrase_boundary", "rapid_ko"],
    "japanese_weak_phoneme": ["ja"],
    "large_interval": ["large_interval_ko"],
}
INTERVAL_STAGES = {
    "omnivoice_source": "artifacts/reports/rc5_isolation/large_interval_ko/A_omnivoice_source.wav",
    "soulx_selected_s32_c2_seed21": "artifacts/reports/rc5_large_interval_decode/s32_c2_seed21.wav",
    "soulx_s50_c2_seed21": "artifacts/reports/rc5_large_interval_decode/s50_c2_seed21.wav",
    "rc6_waveform_refiner": "artifacts/reports/rc6_backend_candidate/listening/large_interval_ko.wav",
    "spectral_lower_025": "artifacts/reports/spectral_refiner_stress_s025/listening/large_interval_ko.wav",
    "rc7": "artifacts/reports/rc7_listening_gate/08_large_interval_ko.wav",
    "spectral_higher_100": "artifacts/reports/spectral_refiner_stress_s100/listening/large_interval_ko.wav",
}
FFT = {"short": (256, 64), "medium": (1024, 256), "long": (4096, 1024)}


def load(path: Path, rate: int = 48_000) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    return resample_poly(mono, rate, source_rate).astype("float32") if source_rate != rate else mono


def spectral(audio: np.ndarray, n_fft: int, hop: int) -> dict:
    magnitude = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft)) + 1e-8
    power = magnitude**2
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    logmag = np.log1p(magnitude)
    instability = np.mean(np.abs(np.diff(logmag, axis=1)), axis=0)
    floor_db = 20 * np.log10(np.quantile(magnitude, .2, axis=0) + 1e-8)
    return {
        "spectral_flatness_mean": round(float(np.mean(flatness)), 7),
        "spectral_instability_p95": round(float(np.quantile(instability, .95)), 6),
        "noise_floor_modulation_db_std": round(float(np.std(floor_db)), 4),
    }


def phone_metrics(audio: np.ndarray, score_path: Path) -> dict:
    score = normalize_score(json.loads(score_path.read_text()))
    phones, _ = score_phone_targets(score)
    rms = librosa.feature.rms(y=audio, frame_length=960, hop_length=960, center=False)[0]
    f0, voiced, _ = librosa.pyin(audio, fmin=65, fmax=1100, sr=48_000, frame_length=2048, hop_length=960)
    duration = len(audio) / 48_000
    scale = duration / max(note["start"] + note["duration"] for note in score["notes"])
    rows = []
    median_rms = float(np.median(rms[rms > 1e-6])) if np.any(rms > 1e-6) else 1e-6
    for phone in phones:
        start = max(0, round(phone["target_start"] * scale * 50))
        end = min(len(rms), max(start + 1, round(phone["target_end"] * scale * 50)))
        local = rms[start:end]
        expected = phone.get("voicing") in {"vowel", "voiced_consonant"}
        observed = voiced[start:min(end, len(voiced))]
        rows.append({
            "phoneme": phone["symbol"], "voicing": phone.get("voicing"),
            "start": round(phone["target_start"] * scale, 3), "end": round(phone["target_end"] * scale, 3),
            "rms_to_phrase_median": round(float(np.mean(local) / median_rms), 4),
            "unexpected_silence_fraction": round(float(np.mean(local < .1 * median_rms)), 4),
            "voiced_fraction": round(float(np.mean(observed)), 4) if len(observed) else None,
            "voicing_match": round(float(np.mean(observed == expected)), 4) if len(observed) else None,
        })
    weak = [row for row in rows if row["rms_to_phrase_median"] < .25]
    mismatched = [row for row in rows if row["voicing_match"] is not None and row["voicing_match"] < .5]
    return {
        "timeline_source": "score_timed_frontend_inferred",
        "phone_count": len(rows), "weak_phone_count": len(weak),
        "voicing_mismatch_count": len(mismatched),
        "weak_phones": weak, "voicing_mismatches": mismatched,
    }


def metrics(path: Path, score_path: Path) -> dict:
    audio = load(path)
    harmonic = librosa.effects.harmonic(audio, margin=2)
    noise = audio - harmonic
    hnr = 10 * np.log10((np.mean(harmonic**2) + 1e-10) / (np.mean(noise**2) + 1e-10))
    return {
        "path": str(path.relative_to(ROOT)), "duration": round(len(audio) / 48_000, 4),
        "harmonic_to_noise_proxy_db": round(float(hnr), 4),
        "multi_resolution": {name: spectral(audio, *params) for name, params in FFT.items()},
        "phones": phone_metrics(audio, score_path),
    }


def interval_f0(path: Path) -> dict:
    audio = load(path)[round(.8 * 48_000):round(1.6 * 48_000)]
    pyin, voiced, probability = librosa.pyin(audio, fmin=65, fmax=1100, sr=48_000, frame_length=4096, hop_length=480)
    yin = librosa.yin(audio, fmin=65, fmax=1100, sr=48_000, frame_length=4096, hop_length=480)
    valid = np.isfinite(pyin) & np.isfinite(yin) & voiced
    disagreement = np.abs(1200 * np.log2(np.maximum(pyin[valid], 1e-6) / np.maximum(yin[valid], 1e-6)))
    return {
        "path": str(path.relative_to(ROOT)), "region_seconds": [.8, 1.6],
        "pyin_median_hz": round(float(np.nanmedian(pyin)), 3),
        "yin_median_hz": round(float(np.nanmedian(yin)), 3),
        "estimator_disagreement_p90_cents": round(float(np.quantile(disagreement, .9)), 3) if len(disagreement) else None,
        "pyin_voiced_ratio": round(float(np.mean(voiced)), 4),
        "pyin_probability_mean": round(float(np.nanmean(probability)), 4),
        "multi_resolution": {name: spectral(audio, *params) for name, params in FFT.items()},
    }


def interval_plot() -> str:
    target = ROOT / "artifacts/reports/rc8_defect_review/large_interval_stages.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    names = ("omnivoice_source", "soulx_selected_s32_c2_seed21", "soulx_s50_c2_seed21", "rc6_waveform_refiner", "rc7")
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, names):
        audio = load(ROOT / INTERVAL_STAGES[name])[:round(2 * 48_000)]
        spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=4096, hop_length=240)), ref=np.max)
        axis.imshow(spectrum, origin="lower", aspect="auto", extent=[0, 2, 0, 24_000], cmap="magma", vmin=-80, vmax=0)
        axis.set_ylim(0, 4000); axis.set_ylabel("Hz"); axis.set_title(name)
        axis.axvline(1.2, color="cyan", linewidth=.8)
    axes[-1].set_xlabel("seconds")
    fig.savefig(target, dpi=140); plt.close(fig)
    return str(target.relative_to(ROOT))


def main() -> None:
    manifests = {name: json.loads((ROOT / path).read_text()) for name, path in VARIANTS.items()}
    rows = []
    for variant, manifest in manifests.items():
        for group, cases in GROUPS.items():
            for case in cases:
                item = manifest["files"][case]
                rows.append({"variant": variant, "group": group, "case": case} | metrics(ROOT / item["path"], ROOT / item["score"]))
    interval = {name: interval_f0(ROOT / path) for name, path in INTERVAL_STAGES.items()}
    report = {
        "status": "diagnosed_no_fix_selected",
        "rc7_source_commit": "ae8944070f3dc38e310b33f29d95f4bcd3c81def",
        "fft_definitions": {name: {"n_fft": n, "hop": hop, "window_ms": round(n / 48, 3)} for name, (n, hop) in FFT.items()},
        "groups": GROUPS, "ablation": VARIANTS, "rows": rows,
        "large_interval_first_region": interval,
        "large_interval_review_plot": interval_plot(),
        "limitations": [
            "The frozen stress scores lack editor-supplied phoneme spans; phone timing here is explicitly inferred.",
            "Japanese low-energy phones are review candidates, not automatic gain targets, because valid devoicing is not labeled.",
            "A fix is not selected until these measurements and listening agree.",
        ],
    }
    target = ROOT / "artifacts/reports/rc8_defect_diagnostics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert len(rows) == 7 * len(VARIANTS) and set(interval) == set(INTERVAL_STAGES)
    print(json.dumps({"status": report["status"], "rows": len(rows), "interval_stages": len(interval)}, indent=2))


if __name__ == "__main__":
    main()
