#!/usr/bin/env python3
from __future__ import annotations
import hashlib, shutil, subprocess
from pathlib import Path

root = Path("artifacts/package/gyu-singer-v1-experimental")
if root.exists(): shutil.rmtree(root)
for part in ["model", "runtime/gyu_singer", "examples", "integrations/renderer_protocol", "config"]: (root/part).mkdir(parents=True, exist_ok=True)
shutil.copy2("checkpoints/gyu_v1_experimental.npz", root/"model/gyu_v1_experimental.npz")
shutil.copytree("checkpoints/gyu_moss_nano_sft/checkpoint-last", root/"model/gyu_moss_nano_sft", ignore=shutil.ignore_patterns(".cache"))
shutil.copytree("data/cache/moss-audio-tokenizer-nano", root/"model/moss-audio-tokenizer-nano", ignore=shutil.ignore_patterns(".cache"))
shutil.copy2("data/source/Korea Digital Media High School 215.m4a", root/"model/gyu_reference.m4a")
for file in Path("src/gyu_singer").glob("*.py"): shutil.copy2(file, root/"runtime/gyu_singer"/file.name)
for file in Path("examples").glob("*.json"): shutil.copy2(file, root/"examples"/file.name)
for file in Path("integrations/renderer_protocol").glob("*"): shutil.copy2(file, root/"integrations/renderer_protocol"/file.name)
(root/"requirements.txt").write_text("numpy\nscipy\nsoundfile\ntorch\ntorchaudio\ntorchcodec\ntransformers\nonnxruntime\n")
(root/"install.sh").write_text("#!/bin/sh\npython -m pip install -r requirements.txt\n")
(root/"run.sh").write_text("#!/bin/sh\nPYTHONPATH=runtime python -m gyu_singer.cli --backend neural --model model/gyu_moss_nano_sft --audio-tokenizer model/moss-audio-tokenizer-nano --reference model/gyu_reference.m4a render examples/smoke.json --output output.wav\n")
(root/"README.md").write_text("# GYU Singer v1-experimental\n\n`sh install.sh`, then `sh run.sh`. Score notes drive fine-tuned GYU vocalization, MIDI pitch, timing, and dynamics. This is an experimental neural vocalizer, not full score-to-acoustic SVS.\n")
(root/"MODEL_CARD.md").write_text("MOSS-TTS-Nano foundation weights fine-tuned for 3 epochs/48 steps on 64 ASR-confirmed GYU singing phrases. Runtime adds authorized GYU reference conditioning and note-by-note pitch/time transforms. Long-note, consonant, and unseen multilingual lyric quality remain experimental.\n")
(root/"LICENSES.md").write_text("GYU reference recording: authorized target-speaker material. MOSS-TTS-Nano model card declares Apache-2.0. Runtime dependencies retain upstream licenses.\n")
archive = Path("artifacts/package/gyu-singer-v1-experimental.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root.parent, root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest(); Path(str(archive)+".sha256").write_text(digest+"  "+archive.name+"\n")
print(archive, digest)
