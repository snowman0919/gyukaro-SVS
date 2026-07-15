#!/usr/bin/env python3
"""Render the bounded EN/JA frontend fixes without touching frozen RC7 files."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_spectral_refiner import SpectralRuntime  # noqa: E402
from gyu_singer.inference.rc6 import GyuSingerRC6Renderer  # noqa: E402


SCORES = {"en": "examples/quality_en.json", "ja": "examples/quality_ja.json"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "artifacts/reports/rc8_frontend_candidate"
    listening = output / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerRC6Renderer(ROOT / "data/processed/master/216.wav", root=ROOT)
    refiner = SpectralRuntime(ROOT / "checkpoints/acoustic_refiner_spectral_singing.pt")
    rows = {}
    try:
        for case, score in SCORES.items():
            started = time.perf_counter()
            source = renderer.render(json.loads((ROOT / score).read_text()))
            refined = refiner.process(source)
            candidate = source + .5 * (refined - source)
            candidate *= min(1.0, .97 / max(float(np.max(np.abs(candidate))), 1e-8))
            before_path = listening / f"{case}_rc6_frontend_fixed.wav"
            after_path = listening / f"{case}_rc8_frontend_fixed.wav"
            sf.write(before_path, source, 48_000, subtype="PCM_24")
            sf.write(after_path, candidate, 48_000, subtype="PCM_24")
            rows[case] = {
                "score": score, "rc6_path": str(before_path.relative_to(ROOT)),
                "candidate_path": str(after_path.relative_to(ROOT)),
                "rc6_sha256": sha(before_path), "candidate_sha256": sha(after_path),
                "render_seconds": round(time.perf_counter() - started, 3),
            }
            print(case, flush=True)
    finally:
        renderer.close()
    report = {
        "status": "objective_and_human_evaluation_pending",
        "baseline": "frozen RC7 at ae8944070f3dc38e310b33f29d95f4bcd3c81def",
        "changes": ["English ARPAbet AY classified as voiced", "Japanese score chunks resolved before phoneme timing"],
        "spectral_refiner": "checkpoints/acoustic_refiner_spectral_singing.pt",
        "strength": .5, "rows": rows,
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": list(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
