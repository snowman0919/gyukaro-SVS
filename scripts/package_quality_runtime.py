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
shutil.copy2("checkpoints/gyu_quality_pitch_controller.pt", root / "model/gyu_quality_pitch_controller.pt")
(root / "requirements.txt").write_text("numpy\nscipy\nsoundfile\ntorch\ntorchaudio\n")
(root / "run.sh").write_text("#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")\"\n: \"${GYU_SINGER_CACHE:?set cache path, or run sh bootstrap.sh /path/to/cache}\"\n: \"${GYU_SOULX_PYTHON:=$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python}\"\nexport GYU_SINGER_CACHE GYU_SOULX_PYTHON\nPYTHONPATH=runtime python -m gyu_singer.cli --backend hybrid-svs --reference model/gyu_reference_216.wav render examples/quality_ko.json --output output.wav\n")
(root / "bootstrap.sh").write_text("#!/bin/sh\nset -eu\nCACHE=${1:?usage: sh bootstrap.sh /path/to/cache}\nmkdir -p \"$CACHE\"\nif [ ! -d \"$CACHE/ace-step/.git\" ]; then git clone https://github.com/ace-step/ACE-Step.git \"$CACHE/ace-step\"; fi\ngit -C \"$CACHE/ace-step\" checkout 1bee4c9\npython3 -m venv \"$CACHE/ace-step/.venv\"\n\"$CACHE/ace-step/.venv/bin/pip\" install -r \"$CACHE/ace-step/requirements.txt\"\n\"$CACHE/ace-step/.venv/bin/pip\" install -U huggingface_hub\n\"$CACHE/ace-step/.venv/bin/hf\" download ACE-Step/ACE-Step-v1-3.5B --local-dir \"$CACHE/ace-step-checkpoint\"\nif [ ! -d \"$CACHE/soulx-singer/.git\" ]; then git clone https://github.com/Soul-AILab/SoulX-Singer.git \"$CACHE/soulx-singer\"; fi\ngit -C \"$CACHE/soulx-singer\" checkout 81aeb3a\npython3 -m venv \"$CACHE/soulx-singer/.venv\"\n\"$CACHE/soulx-singer/.venv/bin/pip\" install -r \"$CACHE/soulx-singer/requirements.txt\"\n\"$CACHE/soulx-singer/.venv/bin/pip\" install -U huggingface_hub\n\"$CACHE/soulx-singer/.venv/bin/hf\" download Soul-AILab/SoulX-Singer --local-dir \"$CACHE/soulx-singer/pretrained_models/SoulX-Singer\"\n\"$CACHE/soulx-singer/.venv/bin/hf\" download Soul-AILab/SoulX-Singer-Preprocess --local-dir \"$CACHE/soulx-singer/pretrained_models/SoulX-Singer-Preprocess\"\nprintf '%s\\n' \"export GYU_SINGER_CACHE=$CACHE\" \"export GYU_SOULX_PYTHON=$CACHE/soulx-singer/.venv/bin/python\"\n")
(root / "README.md").write_text("# GYU Hybrid Singer v0.3 quality runtime\n\nWhole-phrase neural path: TriSinger score/content/timbre/style conditioner -> flow-predicted expressive F0 residual -> ACE-Step lyric-vocal content -> SoulX-Singer SVC. SoulX receives the complete 50 Hz score-plus-controller F0 contour. No per-note TTS, pitch-shift, time-stretch, or waveform concatenation.\n\nThe package includes the 0.8 MB trained TriSinger pitch controller but excludes 13 GB upstream weights. On a CUDA host with Git and Python 3, run `sh bootstrap.sh /path/to/cache`; export the two paths it prints; then run `sh run.sh`. Bootstrap pins ACE-Step `1bee4c9` and SoulX-Singer `81aeb3a`, installs their isolated environments, and downloads the Apache-2.0 upstream models.\n")
archive = Path("artifacts/package/gyu-hybrid-singer-v0.3-quality-runtime.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root.parent, root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
print(archive, digest)
