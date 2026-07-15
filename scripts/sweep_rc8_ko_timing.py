#!/usr/bin/env python3
"""Sweep score/CTC latent timing for normal KO while leaving Rapid KO untouched."""
from __future__ import annotations

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

from evaluate_spectral_refiner import SpectralRuntime  # noqa: E402
from gyu_singer.inference.acoustic_refiner import AcousticRefinerRuntime  # noqa: E402
from gyu_singer.inference.v09 import GyuSingerV09Renderer  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_ko_timing_sweep"
    listening = root / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/reports/rc5_isolation/ko_neutral/production_adapted_source.wav"
    f0 = ROOT / "artifacts/reports/rc5_candidate_core/ko_neutral/canonical_f0.npy"
    warp = ROOT / "artifacts/reports/rc5_latent_timing/ko_neutral/content_warp.npy"
    identity = ROOT / "artifacts/reports/rc5_isolation/ko_neutral/identity.npy"
    style = ROOT / "artifacts/reports/rc5_isolation/ko_neutral/style.npy"
    renderer = GyuSingerV09Renderer(ROOT / "data/processed/master/216.wav", root=ROOT)
    renderer.omnivoice.close()
    waveform = AcousticRefinerRuntime(ROOT / "checkpoints/acoustic_refiner_universal.pt")
    spectral = SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt")
    rows = []
    try:
        for strength in (0.0, .1, .25, .5):
            started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="gyu-rc8-ko-timing-") as temporary:
                decoded = Path(temporary) / "decoded.wav"
                renderer.soulx.request({
                    "source": str(source), "f0_npy": str(f0),
                    "content_warp_npy": str(warp), "content_warp_strength": strength,
                    "identity_npy": str(identity), "style_npy": str(style),
                    "n_steps": 32, "cfg": 1.5, "seed": 21, "output": str(decoded),
                })
                audio, rate = sf.read(decoded, dtype="float32", always_2d=True)
            audio = audio.mean(1)
            if rate != 48_000:
                audio = resample_poly(audio, 48_000, rate).astype("float32")
            old = waveform.process(audio)
            audio = audio + .25 * (old - audio)
            refined = spectral.process(audio)
            audio = audio + .5 * (refined - audio)
            audio *= min(1.0, .97 / max(float(np.max(np.abs(audio))), 1e-8))
            path = listening / f"ko_neutral_warp_{strength:g}.wav"
            sf.write(path, audio, 48_000, subtype="PCM_24")
            rows.append({"strength": strength, "path": str(path.relative_to(ROOT)), "sha256": sha(path), "render_seconds": round(time.perf_counter() - started, 3)})
            print(strength, flush=True)
    finally:
        renderer.close()
    report = {
        "status": "objective_and_human_evaluation_pending", "case": "ko_neutral",
        "protected_rapid_path_changed": False,
        "source_score_phone_center_offset_seconds": {"median": .3751, "p90": .6344, "max": .7741},
        "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
