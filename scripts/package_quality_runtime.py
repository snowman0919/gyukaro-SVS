#!/usr/bin/env python3
"""Package the quality-proven frozen ACE-Step + SoulX phrase runtime."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


root = Path("artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime")
if root.exists(): shutil.rmtree(root)
for part in ("runtime", "scripts", "examples", "model"):
    (root / part).mkdir(parents=True)
shutil.copytree("src/gyu_singer", root / "runtime/gyu_singer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for script in ("generate_ace_phrase.py", "probe_soulx_score.py"):
    shutil.copy2(Path("scripts") / script, root / "scripts" / script)
for example in ("quality_ko.json", "quality_en.json", "quality_ja.json"):
    shutil.copy2(Path("examples") / example, root / "examples" / example)
shutil.copy2("data/processed/master/216.wav", root / "model/gyu_reference_216.wav")
(root / "requirements.txt").write_text("numpy\nscipy\nsoundfile\ntorch\ntorchaudio\n")
(root / "run.sh").write_text("#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")\"\n: \"${GYU_SINGER_CACHE:?set to the cache containing ace-step and soulx-singer}\"\n: \"${GYU_SOULX_PYTHON:?set to the SoulX pinned Python executable}\"\nPYTHONPATH=runtime python -m gyu_singer.cli --backend hybrid-soulx-phrase --reference model/gyu_reference_216.wav render examples/quality_ko.json --output output.wav\n")
(root / "README.md").write_text("# GYU Hybrid Singer v0.3 quality runtime\n\nWhole-phrase neural path: ACE-Step lyric-vocal generation -> SoulX-Singer SVC with an explicit full-score 50 Hz F0 contour. No per-note TTS, pitch-shift, time-stretch, or waveform concatenation.\n\nThis thin package intentionally excludes 13 GB of upstream model weights. Set `GYU_SINGER_CACHE` to a cache containing `ace-step/`, `ace-step-checkpoint/`, and `soulx-singer/`; set `GYU_SOULX_PYTHON` to the compatible SoulX environment. Then run `sh run.sh`.\n")
archive = Path("artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root.parent, root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
print(archive, digest)
