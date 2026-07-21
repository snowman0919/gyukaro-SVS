#!/usr/bin/env python3
"""Build an OpenUtau official-character-library package from local metadata assets."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


NAME = "gyu-singer-openutau-character-library"
LIBRARY_DIRNAME = Path("GYU-SINGER")
CHARACTER_DIR = Path("gyu")
CHARACTER_LIBRARY_DIR = LIBRARY_DIRNAME


def _build_character_txt() -> str:
    image_file = (
        "portraits/gh.png"
        if (CHARACTER_DIR / "gh.png").exists()
        else "portraits/0.png" if (CHARACTER_DIR / "0.png").exists()
        else "portraits/" + sorted([path.name for path in CHARACTER_DIR.glob("*.png")])[0]
    )
    return "\n".join([
        "name=GYU-SINGER",
        "author=GYU Singer project",
        "voice=GYU-SINGER",
        f"image={image_file}",
        "version=0.9-openutau-official-library",
        "text_file_encoding=utf-8",
        "default_phonemizer=OpenUtau.Core.DefaultPhonemizer",
        "",
    ])



def _build_character_yaml() -> str:
    image_file = (
        "portraits/gh.png"
        if (CHARACTER_DIR / "gh.png").exists()
        else "portraits/0.png" if (CHARACTER_DIR / "0.png").exists()
        else "portraits/" + sorted([path.name for path in CHARACTER_DIR.glob("*.png")])[0]
    )
    return "\n".join([
        "name: GYU-SINGER",
        "text_file_encoding: utf-8",
        f"image: {image_file}",
        f"portrait: {image_file}",
        "portrait_opacity: 0.67",
        "author: GYU Singer project",
        "version: 0.9-openutau-official-library",
        "default_phonemizer: OpenUtau.Core.DefaultPhonemizer",
        "subbanks:",
        "  - color: \"\"",
        "    prefix: \"\"",
        "    suffix: \"\"",
        "    tone_ranges:",
        "      - C2-C6",
        "",
    ])


def _copy_metadata(root: Path) -> list[Path]:
    if not CHARACTER_DIR.exists():
        raise FileNotFoundError(f"missing character directory: {CHARACTER_DIR}")

    png_files = sorted(CHARACTER_DIR.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"no portrait png found in {CHARACTER_DIR}")

    portrait_dir = root / CHARACTER_LIBRARY_DIR / "portraits"
    portrait_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for image in png_files:
        target = portrait_dir / image.name
        shutil.copy2(image, target)
        copied.append(target)

    as_dir = CHARACTER_DIR / "as"
    if as_dir.exists():
        copied_urls = as_dir / "urls.txt"
        if copied_urls.exists():
            target_as = root / CHARACTER_LIBRARY_DIR / "as"
            target_as.mkdir(parents=True, exist_ok=True)
            shutil.copy2(copied_urls, target_as / "urls.txt")
            copied.append(target_as / "urls.txt")

    character_root = root / CHARACTER_LIBRARY_DIR
    (character_root / "character.txt").write_text(_build_character_txt(), encoding="utf-8")
    (character_root / "character.yaml").write_text(_build_character_yaml(), encoding="utf-8")
    (character_root / "README.md").write_text(
        "This is the OpenUtau singer discovery package (metadata only) for GYU-SINGER.\n"
        "It contains character metadata and portrait assets for official OpenUtau character\n"
        "discovery. Runtime rendering requires the separate official OpenUtau renderer\n"
        "integration path used by the v0.9 package.\n",
        encoding="utf-8",
    )
    return copied + [character_root / "character.txt", character_root / "character.yaml", character_root / "README.md"]


def _write_package_metadata(path: Path) -> None:
    metadata = {
        "version": "0.9-openutau-library",
        "kind": "openutau-character-library",
        "character": "GYU-SINGER",
        "runtime": "separate renderer required",
        "content_scope": ["metadata", "portraits", "as_urls"],
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    package_root = Path("artifacts/package") / NAME
    package_root.mkdir(parents=True, exist_ok=True)
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)

    copied = _copy_metadata(package_root)
    _write_package_metadata(package_root)

    zip_path = Path("artifacts/package") / f"{NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            handle.write(path, path.relative_to(package_root))

    digest = _sha256(zip_path)
    (Path("artifacts/package") / f"{zip_path.name}.sha256").write_text(
        f"{digest}  {zip_path.name}\n",
        encoding="utf-8",
    )

    alias = Path("artifacts/package") / "gyu-singer-v0.9-openutau-character.zip"
    shutil.copy2(zip_path, alias)
    (Path("artifacts/package") / f"{alias.name}.sha256").write_text(
        f"{digest}  {alias.name}\n",
        encoding="utf-8",
    )

    print(json.dumps({"package": str(zip_path), "sha256": digest, "alias": str(alias)}, indent=2))


if __name__ == "__main__":
    main()
