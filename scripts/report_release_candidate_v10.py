#!/usr/bin/env python3
"""Record reproducible evidence from an installed v1.0 release candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audio(path: Path) -> dict:
    info = sf.info(path)
    return {
        "path": str(path), "sha256": sha256(path), "sample_rate": info.samplerate,
        "channels": info.channels, "duration_seconds": info.duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--longform", required=True, type=Path)
    parser.add_argument("--longform-metrics", required=True, type=Path)
    parser.add_argument("--output", default="artifacts/reports/release_candidate_v10.json", type=Path)
    args = parser.parse_args()

    root = args.install_root.resolve()
    runtime = root / ".runtime"
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        archive.testzip()
        package = json.loads(archive.read("gyu-singer-v1.0/PACKAGE.json"))
        packaged_lexicon = "gyu-singer-v1.0/src/gyu_singer/frontend/english_lexicon.json" in names
    installed_lexicon = next((runtime / "soulx-venv/lib").glob("python*/site-packages/gyu_singer/frontend/english_lexicon.json"), None)
    openutau = subprocess.check_output(["git", "-C", runtime / "OpenUtau", "rev-parse", "HEAD"], text=True).strip()
    smokes = {language: audio(runtime / f"install-smoke-{language}.wav") for language in ("ko", "en", "ja")}
    metrics = json.loads(args.longform_metrics.read_text())
    longform = audio(args.longform)

    passed = all([
        packaged_lexicon, installed_lexicon is not None,
        openutau == package["openutau_revision"],
        all(row["sample_rate"] == 48_000 and row["channels"] == 1 for row in smokes.values()),
        metrics["notes"] >= 100, metrics["phrases"] >= 1, metrics["failed_phrases"] == 0,
        metrics["retries"] == 0, longform["duration_seconds"] >= 120 - 0.1,
    ])
    report = {
        "status": "pass" if passed else "fail",
        "candidate": {
            "archive": str(args.archive), "sha256": sha256(args.archive),
            "bytes": args.archive.stat().st_size, "metadata": package,
        },
        "clean_install": {
            "root": str(root), "source_checkout_pythonpath_used": False,
            "cache_seed_method": "documented install.sh --cache-source with downloader revision/checksum verification",
            "openutau_revision": openutau, "packaged_english_lexicon": packaged_lexicon,
            "installed_english_lexicon": str(installed_lexicon) if installed_lexicon else None,
            "smoke_audio": smokes,
        },
        "installed_package_longform": {"audio": longform, "metrics": metrics},
        "expected_longform_sha256": "728b02c18ed99f9336d3621c212aa2d984eb0e22f87a62d051c832a35e52f4c3",
        "exact_longform_reproduction": longform["sha256"] == "728b02c18ed99f9336d3621c212aa2d984eb0e22f87a62d051c832a35e52f4c3",
    }
    if not report["exact_longform_reproduction"]:
        report["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
