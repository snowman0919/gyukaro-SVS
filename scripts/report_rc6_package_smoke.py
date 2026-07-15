#!/usr/bin/env python3
"""Verify and report an installed RC6 archive outside the repository."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="artifacts/package/gyu-singer-v1.0.0-rc6-candidate.zip")
    parser.add_argument("--installed-root", default="/tmp/gyu-rc6-clean/gyu-singer-v1.0")
    parser.add_argument("--output", default="artifacts/reports/rc6_package_smoke.json")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    root = Path(args.installed_root).resolve()
    metadata = json.loads((root / "PACKAGE.json").read_text())
    dependencies = json.loads((root / "model-dependencies.json").read_text())
    checkpoint_rows = []
    for name, expected in dependencies["packaged_project_checkpoints"].items():
        actual = sha(root / "checkpoints" / name)
        checkpoint_rows.append({"name": name, "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected})
    audio_rows = []
    for language in ("ko", "en", "ja"):
        path = root / ".runtime" / f"install-smoke-{language}.wav"
        info = sf.info(path)
        audio_rows.append({"language": language, "path": str(path), "sha256": sha(path), "sample_rate": info.samplerate,
                           "channels": info.channels, "seconds": info.duration,
                           "pass": info.samplerate == 48_000 and info.channels == 1 and info.duration >= 1})
    openutau = subprocess.check_output(["git", "-C", str(root / ".runtime/OpenUtau"), "rev-parse", "HEAD"], text=True).strip()
    checks = {
        "archive_sha_manifest": archive.with_suffix(archive.suffix + ".sha256").read_text().split()[0] == sha(archive),
        "candidate_not_final": metadata["final_v1_release"] is False and metadata["human_listening"] == "pending",
        "backend_rc6": metadata["backend"] == "gyu-singer-rc6" and "gyu-singer-rc6" in (root / "serve.sh").read_text(),
        "checkpoint_hashes": all(row["pass"] for row in checkpoint_rows),
        "openutau_revision": openutau == metadata["openutau_revision"],
        "multilingual_audio": all(row["pass"] for row in audio_rows),
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail", "archive": str(archive), "archive_sha256": sha(archive),
        "source_commit": metadata["source_commit"], "installed_root": str(root), "checks": checks,
        "openutau_revision": openutau, "checkpoints": checkpoint_rows, "audio": audio_rows,
        "installer_completed": True, "final_v1_release_created": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
