#!/usr/bin/env python3
"""Build the v0.9 runtime plus executable OpenUtau fork overlay."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


NAME = "gyu-singer-v0.9-openutau"
CHARACTER_LIBRARY_NAME = "GYU-SINGER"
CHARACTER_LIBRARY_DIR = Path("openutau_character_library") / CHARACTER_LIBRARY_NAME
CHARACTER_SOURCE_DIR = Path("gyu")


FILES = [
    Path("pyproject.toml"),
    Path("src/gyu_singer"),
    Path("scripts/probe_soulx_score.py"),
    Path("scripts/generate_omnivoice_phrase.py"),
    Path("scripts/test_openutau_v09_behavior.py"),
    Path("scripts/openutau_v09_runtime_smoke.sh"),
    Path("scripts/openutau_v09_ops.sh"),
    Path("scripts/openutau_v09_runtime.sh"),
    Path("scripts/openutau_v09_production_readiness.sh"),
    Path("scripts/openutau_v09_ops_check.sh"),
    Path("scripts/openutau_v09_full_runtime_readiness.sh"),
    Path("scripts/openutau_v09_ready_check.sh"),
    Path("scripts/openutau_v09_operational_check.sh"),
    Path("scripts/openutau_v09_collect_approval_record.sh"),
    Path("scripts/report_openutau_v09_readiness.py"),
    Path("scripts/verify_v09_runtime_paths.sh"),
    Path("checkpoints/gyu_prosody_v0.5.pt"),
    Path("checkpoints/gyu_teacher_identity_v0.5.pt"),
    Path("checkpoints/gyu_acoustic_style_adapter_v0.5.pt"),
    Path("checkpoints/gyu_identity_space_v0.6.pt"),
    Path("checkpoints/gyu_real_latent_adapters_v0.7.pt"),
    Path("data/processed/master/216.wav"),
    Path("examples/quality_ko.json"),
    Path("examples/quality_en.json"),
    Path("examples/quality_ja.json"),
    Path("examples/openutau_v09.ustx"),
    Path("examples/openutau_v10_longform.ustx"),
    Path("integrations/openutau"),
    Path("artifacts/reports/openutau_v09/behavior.json"),
    Path("docs/v0.6_baseline.md"),
    Path("docs/real_latent_dataset.md"),
    Path("docs/identity_adapter_v0.7.md"),
    Path("docs/style_adapter_v0.7.md"),
    Path("docs/evaluation_v0.8.md"),
    Path("docs/openutau_v0.9.md"),
]


def build_character_metadata_text() -> str:
    image_file = (
        "portraits/gh.png"
        if (CHARACTER_SOURCE_DIR / "gh.png").exists()
        else "portraits/0.png" if (CHARACTER_SOURCE_DIR / "0.png").exists()
        else "portraits/" + sorted([name.name for name in CHARACTER_SOURCE_DIR.glob("*.png")])[0]
    )
    return "\n".join([
        "name=GYU-SINGER",
        "author=GYU Singer project",
        "voice=GYU-SINGER",
        f"image={image_file}",
        "version=0.9-openutau",
        "text_file_encoding=utf-8",
        "default_phonemizer=OpenUtau.Core.DefaultPhonemizer",
        "",
    ])


def build_character_yaml() -> str:
    image_file = (
        "portraits/gh.png"
        if (CHARACTER_SOURCE_DIR / "gh.png").exists()
        else "portraits/0.png" if (CHARACTER_SOURCE_DIR / "0.png").exists()
        else "portraits/" + sorted([name.name for name in CHARACTER_SOURCE_DIR.glob("*.png")])[0]
    )
    return "\n".join([
        "name: GYU-SINGER",
        "text_file_encoding: utf-8",
        f"image: {image_file}",
        f"portrait: {image_file}",
        "portrait_opacity: 0.67",
        "author: GYU Singer project",
        "version: 0.9-openutau",
        "default_phonemizer: OpenUtau.Core.DefaultPhonemizer",
        "subbanks:",
        "  - color: \"\"",
        "    prefix: \"\"",
        "    suffix: \"\"",
        "    tone_ranges:",
        "      - C2-C6",
        "",
    ])


def copy_character_metadata(root: Path, source_dir: Path = CHARACTER_SOURCE_DIR) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"missing character source dir: {source_dir}")
    png_files = sorted(source_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"no png files found in {source_dir}")

    character_root = root / CHARACTER_LIBRARY_DIR
    portrait_root = character_root / "portraits"
    portrait_root.mkdir(parents=True, exist_ok=True)
    for png in png_files:
        shutil.copy2(png, portrait_root / png.name)

    urls = source_dir / "as" / "urls.txt"
    if urls.exists():
        as_dir = character_root / "as"
        as_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(urls, as_dir / "urls.txt")

    (character_root / "character.txt").write_text(build_character_metadata_text(), encoding="utf-8")
    (character_root / "character.yaml").write_text(build_character_yaml(), encoding="utf-8")
    (character_root / "README.md").write_text(
        "This is an OpenUtau character metadata bundle for the v0.9 runtime package.\n"
        "It is used for singer metadata discovery, portrait preview, and OpenUtau configuration.\n"
        "Actual GYU phrase rendering is performed through the maintained OpenUtau fork renderer.\n",
        encoding="utf-8",
    )


def copy(source: Path, root: Path) -> None:
    destination = root / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)


def main() -> None:
    root = Path("artifacts/package") / NAME
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    missing = [str(path) for path in FILES if not path.exists()]
    if missing: raise FileNotFoundError(f"package inputs missing: {missing}")
    for source in FILES: copy(source, root)
    copy_character_metadata(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "version": "v0.9", "backend": "gyu-singer-v0.8", "integration": "OpenUtau GYU-SINGER maintained fork",
        "git_commit": "see docs/v0.9_release_baseline.md", "external_cache_required": True, "training_teacher_models_bundled": False,
        "openutau_revision": "27573ac5c888d927119d5f65a207312d79194b1f",
        "soulx_revision": "81aeb3ae772c70093c3de74dc23c92d983801ae4",
        "omnivoice_revision": "1574e06a767808c9343740ba695e7515c3d484e2",
        "fish_source_revision": "e5e292632cb11e7a27b2b7487f58f612bc101e13",
        "moss_source_revision": "ad99ec5f26debf1d6c1a4dc8461b2bcb787ec9af",
        "moss_model_revision": "be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        "higgs_inference_required": False,
        "openutau_character_library": str(CHARACTER_LIBRARY_DIR),
    }
    (root / "PACKAGE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "serve.sh").write_text("""#!/bin/sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [ -z "${GYU_SINGER_CACHE:-}" ] && [ -d "$SCRIPT_DIR/data/cache" ]; then
  GYU_SINGER_CACHE="$SCRIPT_DIR/data/cache"
fi
: "${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE to the pinned model cache}"
if [ ! -d "$GYU_SINGER_CACHE/omnivoice/.venv/bin" ] && [ -d "$SCRIPT_DIR/data/cache/omnivoice/.venv/bin" ]; then
  GYU_SINGER_CACHE="$SCRIPT_DIR/data/cache"
fi
if [ ! -d "$GYU_SINGER_CACHE" ]; then
  echo "missing GYU_SINGER_CACHE: $GYU_SINGER_CACHE"
  exit 2
fi
if [ -z "${GYU_SOULX_PYTHON:-}" ]; then
  if [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -d "$GYU_SOULX_RUNTIME_DIR" ]; then
    if [ -x "$GYU_SOULX_RUNTIME_DIR/.venv/bin/python" ]; then
      GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/.venv/bin/python"
    elif [ -x "$GYU_SOULX_RUNTIME_DIR/bin/python" ]; then
      GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/bin/python"
    fi
  fi
  if [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$SCRIPT_DIR/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$SCRIPT_DIR/.venv-soulx/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
fi
export GYU_SOULX_PYTHON
: "${GYU_SOULX_PYTHON:?set GYU_SOULX_PYTHON to the pinned SoulX Python}"
if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON"
  exit 2
fi
if [ ! -x "$GYU_SINGER_CACHE/omnivoice/.venv/bin/python" ]; then
  echo "missing pinned OmniVoice runtime: $GYU_SINGER_CACHE/omnivoice/.venv/bin/python"
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,expandable_segments:True}"
export GYU_SINGER_CACHE GYU_SOULX_PYTHON
exec env PYTHONPATH=src "$GYU_SOULX_PYTHON" -m gyu_singer.cli --backend gyu-singer-v0.8 --reference data/processed/master/216.wav serve --port "${1:-8765}"
""")
    (root / "render.sh").write_text("""#!/bin/sh
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [ -z "${GYU_SINGER_CACHE:-}" ] && [ -d "$SCRIPT_DIR/data/cache" ]; then
  GYU_SINGER_CACHE="$SCRIPT_DIR/data/cache"
fi
: "${GYU_SINGER_CACHE:?set GYU_SINGER_CACHE to the pinned model cache}"
if [ ! -d "$GYU_SINGER_CACHE/omnivoice/.venv/bin" ] && [ -d "$SCRIPT_DIR/data/cache/omnivoice/.venv/bin" ]; then
  GYU_SINGER_CACHE="$SCRIPT_DIR/data/cache"
fi
if [ ! -d "$GYU_SINGER_CACHE" ]; then
  echo "missing GYU_SINGER_CACHE: $GYU_SINGER_CACHE"
  exit 2
fi
if [ -z "${GYU_SOULX_PYTHON:-}" ]; then
  if [ -n "${GYU_SOULX_RUNTIME_DIR:-}" ] && [ -d "$GYU_SOULX_RUNTIME_DIR" ]; then
    if [ -x "$GYU_SOULX_RUNTIME_DIR/.venv/bin/python" ]; then
      GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/.venv/bin/python"
    elif [ -x "$GYU_SOULX_RUNTIME_DIR/bin/python" ]; then
      GYU_SOULX_PYTHON="$GYU_SOULX_RUNTIME_DIR/bin/python"
    fi
  fi
  if [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$GYU_SINGER_CACHE/soulx-singer/.venv-soulx/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$SCRIPT_DIR/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$SCRIPT_DIR/.venv-soulx/bin/python"
  elif [ -z "${GYU_SOULX_PYTHON:-}" ] && [ -x "$HOME/.venv-soulx/bin/python" ]; then
    GYU_SOULX_PYTHON="$HOME/.venv-soulx/bin/python"
  fi
fi
export GYU_SOULX_PYTHON
: "${GYU_SOULX_PYTHON:?set GYU_SOULX_PYTHON to the pinned SoulX Python}"
if [ ! -x "$GYU_SOULX_PYTHON" ]; then
  echo "invalid GYU_SOULX_PYTHON: $GYU_SOULX_PYTHON"
  exit 2
fi
if [ ! -x "$GYU_SINGER_CACHE/omnivoice/.venv/bin/python" ]; then
  echo "missing pinned OmniVoice runtime: $GYU_SINGER_CACHE/omnivoice/.venv/bin/python"
  exit 2
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,expandable_segments:True}"
export GYU_SINGER_CACHE GYU_SOULX_PYTHON
exec env PYTHONPATH=src "$GYU_SOULX_PYTHON" -m gyu_singer.cli --backend gyu-singer-v0.8 --reference data/processed/master/216.wav render "${1:-examples/quality_ko.json}" --output "${2:-output.wav}"
    """)
    for path in (root / "serve.sh", root / "render.sh", root / "scripts/openutau_v09_runtime_smoke.sh", root / "scripts/openutau_v09_ops.sh", root / "scripts/openutau_v09_runtime.sh", root / "integrations/openutau/install_fork.sh", root / "integrations/openutau/test_resident_fork.sh"):
        path.chmod(0o755)
    archive = root.parent / f"{NAME}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            output.writestr(info, path.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"package": str(archive), "sha256": digest, "git_commit": commit}))


if __name__ == "__main__":
    main()
