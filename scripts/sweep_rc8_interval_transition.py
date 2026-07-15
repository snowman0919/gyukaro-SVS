#!/usr/bin/env python3
"""Sweep only the large-jump onset transition on the fixed RC8 source."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402
from gyu_singer.inference.spectral_refiner import SpectralRefinerRuntime  # noqa: E402
from gyu_singer.inference.v09 import GyuSingerV09Renderer, soften_large_jumps  # noqa: E402


def self_test() -> None:
    f0 = np.array([200, 200, 0, 400, 400, 300, 300], dtype="float32")
    score = {"notes": [
        {"pitch": 55, "start": 0, "duration": .06},
        {"pitch": 67, "start": .06, "duration": .04},
        {"pitch": 62, "start": .1, "duration": .04},
    ]}
    softened = soften_large_jumps(f0, score, .04)
    assert 200 < softened[3] < 400 and np.isclose(softened[4], 400)
    assert np.array_equal(softened[5:], f0[5:])


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_interval_transition"
    listening = root / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    score = json.loads((ROOT / "examples/review_large_interval_ko.json").read_text())
    base_f0 = np.load(ROOT / "artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy")
    source = ROOT / "artifacts/reports/rc5_isolation/large_interval_ko/production_adapted_source.wav"
    identity = ROOT / "artifacts/reports/rc5_isolation/large_interval_ko/identity.npy"
    style = ROOT / "artifacts/reports/rc5_isolation/large_interval_ko/style.npy"
    renderer = GyuSingerV09Renderer(ROOT / "data/processed/master/216.wav", root=ROOT)
    renderer.omnivoice.close()
    waveform = AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_universal.pt")
    spectral = SpectralRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt")
    rows = []
    try:
        for frames in (0, 2, 3, 4, 5, 6):
            started = time.perf_counter()
            contour = base_f0.copy() if frames == 0 else soften_large_jumps(base_f0, score, frames / 50)
            contour_path = root / f"f0_transition_{frames}.npy"
            np.save(contour_path, contour)
            with tempfile.TemporaryDirectory(prefix="gyu-rc8-interval-") as directory:
                decoded = Path(directory) / "decoded.wav"
                renderer.soulx.request({
                    "source": str(source), "f0_npy": str(contour_path),
                    "identity_npy": str(identity), "style_npy": str(style),
                    "n_steps": 50, "cfg": 2.0, "seed": 21, "output": str(decoded),
                })
                audio, rate = sf.read(decoded, dtype="float32", always_2d=True)
            audio = audio.mean(1)
            if rate != 48_000:
                audio = resample_poly(audio, 48_000, rate).astype("float32")
            refined = waveform.process(audio)
            audio += .25 * (refined - audio)
            refined = spectral.process(audio)
            audio += .5 * (refined - audio)
            audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
            path = listening / f"large_interval_transition_{frames}.wav"
            sf.write(path, audio, 48_000, subtype="PCM_24")
            rows.append({
                "variant": f"transition_{frames}", "transition_frames": frames,
                "transition_ms": frames * 20, "path": str(path.relative_to(ROOT)),
                "f0_path": str(contour_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "render_seconds": round(time.perf_counter() - started, 3),
            })
            print(frames, flush=True)
    finally:
        renderer.close()
    report = {
        "status": "objective_and_human_evaluation_pending",
        "fixed_conditions": {"steps": 50, "cfg": 2.0, "seed": 21, "precision": "fp32"},
        "change": "large jumps only; score and user pitch remain authoritative",
        "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        main()
