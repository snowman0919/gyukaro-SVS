#!/usr/bin/env python3
"""Render/copy the compact v1 listening set without inventing human observations."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import soundfile as sf


def post(url: str, score: Path, output: Path) -> None:
    request = urllib.request.Request(url, data=score.read_bytes(), headers={"Content-Type": "application/json"}, method="POST")
    output.write_bytes(urllib.request.urlopen(request, timeout=600).read())


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--url", default="http://127.0.0.1:8765/render")
    args = parser.parse_args(); root = Path("artifacts/reports/listening_v10"); root.mkdir(parents=True, exist_ok=True)
    sources = {
        "ko_neutral": "artifacts/reports/v07_style_semantics_quality_neutral.wav",
        "ko_breathy": "artifacts/reports/v07_style_semantics_quality_breathy.wav",
        "ko_energetic": "artifacts/reports/v07_style_semantics_quality_energetic.wav",
        "en": "artifacts/reports/v07_ablation_identity_student_quality_en.wav",
        "ja": "artifacts/reports/v07_ablation_identity_student_quality_ja.wav",
    }
    items = []
    for name, source in sources.items():
        output = root / f"{name}.wav"; shutil.copy2(source, output); items.append((name, output, "existing final production output"))
    for name in ("rapid_ko", "sustain_ko", "large_interval_ko"):
        score = Path(f"examples/review_{name}.json"); output = root / f"{name}.wav"; post(args.url, score, output); items.append((name, output, str(score)))
    audio, rate = sf.read("artifacts/reports/longform_v10.wav", dtype="float32", always_2d=True)
    boundary = root / "phrase_boundary.wav"; sf.write(boundary, audio[round(13.4 * rate):round(14.9 * rate)], rate, subtype="PCM_16")
    items.append(("phrase_boundary", boundary, "continuous boundary 2 at 14.1429 seconds"))
    manifest = {"human_review_status": "pending", "subjective_scores": None, "items": []}
    for name, path, purpose in items:
        info = sf.info(path)
        manifest["items"].append({"id": name, "path": str(path), "purpose": purpose, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "sample_rate": info.samplerate, "channels": info.channels, "duration_seconds": round(info.duration, 4)})
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
