#!/usr/bin/env python3
"""Apply the fixed RC6/RC7 refiners to the measured 50-step interval decode."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_spectral_refiner import SpectralRuntime  # noqa: E402
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_interval_candidate"
    listening = root / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(ROOT / "artifacts/reports/rc5_large_interval_decode/s50_c2_seed21.wav", dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if rate != 48_000:
        audio = resample_poly(audio, 48_000, rate).astype("float32")
    waveform = AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_universal.pt")
    old = waveform.process(audio)
    base = audio + .25 * (old - audio)
    spectral = SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt")
    refined = spectral.process(base)
    rows = []
    for strength in (0.0, .25, .5):
        output = base + strength * (refined - base)
        output *= min(1.0, .97 / max(float(np.max(np.abs(output))), 1e-8))
        path = listening / f"large_interval_s50_spectral_{strength:g}.wav"
        sf.write(path, output, 48_000, subtype="PCM_24")
        rows.append({"strength": strength, "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {
        "status": "objective_and_human_evaluation_pending", "decoder": {"steps": 50, "cfg": 2.0, "seed": 21},
        "source": "artifacts/reports/rc5_large_interval_decode/s50_c2_seed21.wav",
        "waveform_refiner_strength": .25, "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
