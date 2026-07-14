#!/usr/bin/env python3
"""Verify installed revisions, packaged checkpoints, integration build, and optional smoke WAV."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import soundfile as sf


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--audio", action="append", default=[])
    args = parser.parse_args(); root = Path(args.root).resolve(); runtime = root / ".runtime"
    manifest = json.loads((root / "model-dependencies.json").read_text())
    for name, expected in manifest["packaged_project_checkpoints"].items():
        actual = sha(root / "checkpoints" / name)
        if actual != expected: raise RuntimeError(f"checkpoint {name}: {actual} != {expected}")
    expected_openutau = next(item["revision"] for item in manifest["inference"] if item["name"] == "OpenUtau")
    actual_openutau = subprocess.check_output(["git", "-C", str(runtime / "OpenUtau"), "rev-parse", "HEAD"], text=True).strip()
    if actual_openutau != expected_openutau: raise RuntimeError("OpenUtau revision mismatch")
    dll = runtime / "OpenUtau/OpenUtau/bin/Release/net8.0/OpenUtau.dll"
    if not dll.exists(): raise FileNotFoundError(dll)
    audio = []
    for path in args.audio:
        info = sf.info(path); row = {"path": path, "sample_rate": info.samplerate, "channels": info.channels, "seconds": info.duration}
        if info.samplerate != 48_000 or info.channels != 1 or info.duration < 1: raise RuntimeError(f"invalid smoke audio: {row}")
        audio.append(row)
    print(json.dumps({"status": "ok", "openutau": actual_openutau, "checkpoints": len(manifest["packaged_project_checkpoints"]), "audio": audio}, indent=2))


if __name__ == "__main__": main()
