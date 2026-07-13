#!/usr/bin/env python3
from __future__ import annotations
import hashlib, shutil, subprocess
from pathlib import Path

root = Path("artifacts/package/gyu-singer-v1-experimental")
if root.exists(): shutil.rmtree(root)
for part in ["model", "runtime/gyu_singer", "examples", "integrations/renderer_protocol", "config"]: (root/part).mkdir(parents=True, exist_ok=True)
shutil.copy2("checkpoints/gyu_v1_experimental.npz", root/"model/gyu_v1_experimental.npz")
for file in Path("src/gyu_singer").glob("*.py"): shutil.copy2(file, root/"runtime/gyu_singer"/file.name)
for file in Path("examples").glob("*.json"): shutil.copy2(file, root/"examples"/file.name)
for file in Path("integrations/renderer_protocol").glob("*"): shutil.copy2(file, root/"integrations/renderer_protocol"/file.name)
(root/"run.sh").write_text("#!/bin/sh\nPYTHONPATH=runtime python -m gyu_singer.cli --model model/gyu_v1_experimental.npz render examples/korean.json --output output.wav\n")
(root/"README.md").write_text("# GYU Singer v1-experimental\n\nRequires Python 3.11+, numpy, scipy, soundfile. Run `sh run.sh`. This is a real-GYU source-loop score renderer, not intelligible multilingual neural SVS.\n")
(root/"MODEL_CARD.md").write_text("Real GYU-derived voiced loop bank. Pitch controlled by resampling. Text/phoneme conditioning is not trained; lyrics are accepted for protocol compatibility only.\n")
(root/"LICENSES.md").write_text("Source recordings: authorized target-speaker material. Runtime: project source; numpy/scipy/soundfile retain their upstream licenses.\n")
archive = Path("artifacts/package/gyu-singer-v1-experimental.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root.parent, root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest(); Path(str(archive)+".sha256").write_text(digest+"  "+archive.name+"\n")
print(archive, digest)
