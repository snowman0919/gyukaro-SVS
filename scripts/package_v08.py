#!/usr/bin/env python3
"""Build the measured v0.8 runtime without training-only teacher models."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


NAME = "gyu-singer-v0.8"
FILES = [
    Path("src/gyu_singer"),
    Path("scripts/probe_soulx_score.py"),
    Path("scripts/generate_omnivoice_phrase.py"),
    Path("checkpoints/gyu_prosody_v0.5.pt"),
    Path("checkpoints/gyu_teacher_identity_v0.5.pt"),
    Path("checkpoints/gyu_acoustic_style_adapter_v0.5.pt"),
    Path("checkpoints/gyu_identity_space_v0.6.pt"),
    Path("checkpoints/gyu_real_latent_adapters_v0.7.pt"),
    Path("data/processed/master/216.wav"),
    Path("examples/quality_ko.json"),
    Path("examples/quality_en.json"),
    Path("examples/quality_ja.json"),
    Path("docs/evaluation_v0.8.md"),
]


def main() -> None:
    root = Path("artifacts/package") / NAME
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for source in FILES:
        destination = root / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (root / "PACKAGE.json").write_text(json.dumps({"version": "v0.8", "backend": "gyu-singer-v0.8", "git_commit": commit, "external_cache_required": True, "teacher_models_bundled": False}, indent=2) + "\n")
    (root / "run.sh").write_text("#!/bin/sh\nset -eu\ncd \"$(dirname \"$0\")\"\n: \"${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE to the pinned model cache}\"\n: \"${GYU_SOULX_PYTHON:?set GYU_SOULX_PYTHON to the pinned SoulX Python}\"\nexport GYU_SINGER_CACHE GYU_SOULX_PYTHON\nPYTHONPATH=src python -m gyu_singer.cli --backend gyu-singer-v0.8 --reference data/processed/master/216.wav render \"${1:-examples/quality_ko.json}\" --output \"${2:-output.wav}\"\n")
    (root / "run.sh").chmod(0o755)
    archive = root.parent / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(str(Path(NAME) / path.relative_to(root)), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if path.name == "run.sh" else 0o100644) << 16
                output.writestr(info, path.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"package": str(archive), "sha256": digest, "git_commit": commit}))


if __name__ == "__main__":
    main()
