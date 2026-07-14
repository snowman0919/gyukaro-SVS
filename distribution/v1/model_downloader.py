#!/usr/bin/env python3
"""Install or verify exact external inference repositories and model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def source_path(cache: Path, item: dict) -> Path:
    mapping = {
        "OmniVoice source": cache / "omnivoice",
        "OmniVoice singing checkpoint": cache / "omnivoice-checkpoint",
        "SoulX-Singer source": cache / "soulx-singer",
        "SoulX-Singer checkpoint": cache / "soulx-singer/pretrained_models/SoulX-Singer",
        "SoulX preprocessing checkpoint": cache / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess",
    }
    return mapping[item["name"]]


def install_git(item: dict, target: Path, offline: Path | None) -> None:
    if offline:
        source = source_path(offline, item)
        if not source.exists():
            raise FileNotFoundError(f"offline cache missing {source}")
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".venv", "pretrained_models") if item["name"] == "SoulX-Singer source" else shutil.ignore_patterns(".venv"))
    elif not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", item["repository"], str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["revision"]], check=True)


def install_huggingface(item: dict, target: Path, offline: Path | None) -> None:
    if offline:
        source = source_path(offline, item)
        if not source.exists():
            raise FileNotFoundError(f"offline cache missing {source}")
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif not target.exists() or not all((target / name).exists() for name in item.get("checksums", {})):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=item["repository"], revision=item["revision"], local_dir=target)


def verify(items: list[dict], runtime: Path) -> None:
    for item in items:
        if item["name"] == "OpenUtau":
            continue
        target = runtime / item["install_path"]
        if not target.exists():
            raise FileNotFoundError(f"missing dependency: {target}")
        if item["kind"] == "git":
            actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
            if actual != item["revision"]:
                raise RuntimeError(f"{item['name']} revision {actual}, expected {item['revision']}")
        for relative, expected in item.get("checksums", {}).items():
            path = target / relative
            actual = digest(path)
            if actual != expected:
                raise RuntimeError(f"{path} sha256 {actual}, expected {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default=".runtime")
    parser.add_argument("--manifest", default="model-dependencies.json")
    parser.add_argument("--cache-source")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve(); runtime.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    items = [item for item in manifest["inference"] if item["name"] != "OpenUtau"]
    offline = Path(args.cache_source).resolve() if args.cache_source else None
    if not args.verify_only:
        for item in items:
            target = runtime / item["install_path"]
            (install_git if item["kind"] == "git" else install_huggingface)(item, target, offline)
    verify(items, runtime)
    print(json.dumps({"status": "ok", "verified": [item["name"] for item in items], "runtime": str(runtime)}, indent=2))


if __name__ == "__main__":
    main()
