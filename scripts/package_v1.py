#!/usr/bin/env python3
"""Build self-contained experimental v1 hybrid package from trained local artifacts."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


root = Path("artifacts/package/gyu-singer-v1-experimental")
if root.exists(): shutil.rmtree(root)
for part in ("model", "runtime", "examples", "integrations/openutau", "integrations/renderer_protocol", "config"):
    (root / part).mkdir(parents=True, exist_ok=True)
shutil.copytree("src/gyu_singer", root / "runtime/gyu_singer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copytree("data/cache/moss-audio-tokenizer-nano", root / "model/moss-audio-tokenizer-nano", ignore=shutil.ignore_patterns(".cache", "__pycache__"))
shutil.copy2("checkpoints/gyu_hybrid_v0.2.pt", root / "model/gyu_hybrid_v0.2.pt")
shutil.copy2("data/processed/master/216.wav", root / "model/gyu_reference_216.wav")
shutil.copy2("examples/smoke.json", root / "examples/smoke.json")
for source, target in (("integrations/openutau", "integrations/openutau"), ("integrations/renderer_protocol", "integrations/renderer_protocol")):
    shutil.copytree(source, root / target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
(root / "requirements.txt").write_text("numpy\nsoundfile\ntorch\ntransformers\npyyaml\nscipy\ntorchaudio\ntorchcodec\n")
(root / "install.sh").write_text("#!/bin/sh\nset -eu\npython -m venv --system-site-packages .venv\n.venv/bin/python -m pip install -r requirements.txt\n")
(root / "run.sh").write_text("#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")\"\nPYTHONPATH=runtime .venv/bin/python -m gyu_singer.cli --backend hybrid-svs --checkpoint model/gyu_hybrid_v0.2.pt --audio-tokenizer model/moss-audio-tokenizer-nano --reference model/gyu_reference_216.wav render examples/smoke.json --output output.wav\n")
(root / "README.md").write_text("# GYU Singer v1-experimental\n\nCompact phrase-level neural SVS runtime. `sh install.sh`, then `sh run.sh`. It uses a 160-step compact conditional-flow checkpoint trained on real GYU anchors with inferred score timing plus weighted teacher representation loss. Output quality and non-Korean singing remain experimental; no source-loop or per-note DSP path is used by this package.\n")
(root / "MODEL_CARD.md").write_text("## Scope\n\nInput: Korean, English, or Japanese lyric-note JSON. Model: TriSinger conditional-flow acoustic-latent generator (3.0 MB checkpoint) decoded by frozen Apache-2.0 MOSS audio tokenizer. Training: 76 real GYU segments; 60 train/5 validation/5 test, real anchors only for acoustic target; 633 teacher rows representation-only at trust 0.05-0.20. Score for real anchors was inferred from speech duration, not ground-truth singing notation.\n\nThis is an experimental personalized SVS runtime, not a production-quality multilingual singer.\n")
(root / "LICENSES.md").write_text("GYU recordings: authorized target-speaker data; package redistributes only one authorized 48 kHz reference WAV. MOSS audio tokenizer: upstream Apache-2.0 model card. Runtime dependencies retain their upstream licenses.\n")
archive = Path("artifacts/package/gyu-singer-v1-experimental.zip")
if archive.exists(): archive.unlink()
shutil.make_archive(str(archive.with_suffix("")), "zip", root.parent, root.name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
print(archive, digest)
