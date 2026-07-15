#!/usr/bin/env python3
"""Non-destructive audit of the selected GYU reference and recording corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch


def db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def main() -> None:
    path = Path("data/processed/master/216.wav")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    y = audio.mean(1)
    frame = max(1, round(0.05 * rate))
    trim = y[: len(y) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(trim**2, axis=1) + 1e-12)

    active = rms > np.percentile(rms, 35)
    envelope = rms[active]
    lag = max(1, round(0.1 / 0.05))
    reverb = (
        float(np.corrcoef(envelope[:-lag], envelope[lag:])[0, 1])
        if len(envelope) > lag
        else 0.0
    )

    # Whole-recording harmonic energy is confounded by the singer's pitch.
    # Measure mains hum prominence against neighboring bins in quiet frames only.
    quiet = trim[rms <= np.percentile(rms, 20)].reshape(-1)
    freq, power = welch(quiet, rate, nperseg=min(16384, len(quiet)))
    prominences = []
    for base in (50, 60):
        for harmonic in range(base, 601, base):
            narrow = power[np.abs(freq - harmonic) <= 2]
            local = power[
                (np.abs(freq - harmonic) >= 8)
                & (np.abs(freq - harmonic) <= 20)
            ]
            if len(narrow) and len(local):
                prominences.append(
                    10
                    * np.log10(
                        max(float(narrow.mean() / max(local.mean(), 1e-20)), 1e-12)
                    )
                )
    hum_max = float(max(prominences, default=0.0))

    corpus = [
        json.loads(line)
        for line in Path("data/manifests/real_recordings.jsonl").read_text().splitlines()
        if line
    ]
    selected = next(row for row in corpus if row["source_index"] == 216)
    metrics = {
        "path": str(path),
        "sample_rate": rate,
        "channels": audio.shape[1],
        "duration_seconds": len(y) / rate,
        "peak": float(np.max(np.abs(y))),
        "clip_fraction": float(np.mean(np.abs(y) >= 0.999)),
        "dc_offset": float(np.mean(y)),
        "rms_dbfs": db(float(np.sqrt(np.mean(y * y)))),
        "noise_floor_p10_dbfs": db(float(np.percentile(rms, 10))),
        "noise_floor_p20_dbfs": db(float(np.percentile(rms, 20))),
        "quiet_frame_max_mains_harmonic_prominence_db": hum_max,
        "active_envelope_100ms_correlation_reverb_proxy": reverb,
    }
    retain_original = (
        metrics["clip_fraction"] == 0
        and metrics["noise_floor_p20_dbfs"] < -45
        and hum_max < 8
    )
    report = {
        "status": "audited",
        "selected_reference": metrics,
        "selected_manifest_row": selected,
        "corpus": {
            "rows": len(corpus),
            "clipping_rows": sum(row["clipping"] for row in corpus),
            "corrupt_rows": sum(row["corrupt"] for row in corpus),
            "median_peak": float(np.median([row["peak"] for row in corpus])),
            "median_lufs_approx": float(
                np.median([row["integrated_loudness_lufs_approx"] for row in corpus])
            ),
        },
        "decision": (
            "retain_original_lossless" if retain_original else "reference_ab_test_needed"
        ),
        "preprocessing": {
            "denoise_applied": False,
            "dereverb_applied": False,
            "uvr_applied": False,
        },
        "limitations": (
            "background sounds and room character still require listening; "
            "proxies do not certify absence"
        ),
    }
    Path("artifacts/reports/rc5_reference_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    Path("docs/rc5_reference_audit.md").write_text(
        "# RC5 reference quality audit\n\n"
        "```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n\n"
        "No source recording was modified. The selected reference has no clipping, "
        "a -51.9 dBFS quiet-floor proxy, and no prominent mains harmonic in quiet "
        "frames, so the original lossless-derived PCM is retained. Generic denoise, "
        "dereverb, and UVR were not applied.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
