#!/usr/bin/env python3
"""Render the mandatory stress set through the actual resident RC6 backend."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.inference.rc6 import GyuSingerRC6Renderer


SCORES = {
    "ko_neutral": "examples/quality_ko.json", "en": "examples/quality_en.json", "rapid_ko": "examples/review_rapid_ko.json",
    "large_interval_ko": "examples/review_large_interval_ko.json", "ko_breathy": "artifacts/reports/rc5_stress_candidate/ko_breathy.json",
    "ko_energetic": "artifacts/reports/rc5_stress_candidate/ko_energetic.json", "ja": "examples/quality_ja.json",
    "sustained_ko": "examples/review_sustain_ko.json", "phrase_boundary": "examples/review_phrase_boundary_ko.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("artifacts/reports/rc6_backend_candidate"); listening = root / "listening"; listening.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerRC6Renderer("data/processed/master/216.wav"); files = {}
    try:
        info = renderer.model_info()
        for case, score in SCORES.items():
            output = listening / f"{case}.wav"; started = time.perf_counter(); renderer.render_file(score, output)
            files[case] = {"path": str(output), "score": score, "sha256": sha(output), "render_seconds": round(time.perf_counter() - started, 4),
                           "sample_rate": 48000, "backend": info["backend"], "refiner_strength": info["acoustic_refiner_strength"]}
            print(case, flush=True)
    finally:
        renderer.close()
    manifest = {"status": "objective_evaluation_pending", "name": "RC6 actual-backend acoustic candidate (not a tag or release)",
                "backend": info, "files": files, "human_review": "pending"}
    root.mkdir(parents=True, exist_ok=True); (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(files), "backend": info["backend"], "human_review": "pending"}, indent=2))


if __name__ == "__main__":
    main()
