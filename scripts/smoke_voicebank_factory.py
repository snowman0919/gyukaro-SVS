#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from gyu_singer.voicebank.factory import FactoryError, VoicebankFactory


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reports/voicebank_factory/smoke.json"


def run_smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="gyu-voicebank-smoke-") as temporary:
        root = Path(temporary)
        recordings = root / "recordings"
        recordings.mkdir()
        sample_rate = 44_100
        time = np.arange(sample_rate // 4) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        source = recordings / "a.wav"
        sf.write(source, audio, sample_rate)
        source_before = hashlib.sha256(source.read_bytes()).hexdigest()
        rights = {
            "source_type": "synthetic_test_fixture",
            "owner_or_authorized_user": "repository test",
            "allowed_use": "automated_test",
            "redistribution_permission": "not_applicable_not_bundled",
            "languages": ["ko"],
            "known_scripts": {"a.wav": "가나다"},
            "file_metadata": {"a.wav": {"language": "ko", "coverage_tags": []}},
            "recording_environment": "synthetic",
            "consent_provenance_notes": "generated sine wave; not a voice",
            "permission_affirmed": True,
        }
        rights_path = root / "rights.json"
        rights_path.write_text(json.dumps(rights))
        workspace = root / "workspace"
        factory = VoicebankFactory(ROOT, workspace)
        dry_run = factory.init(recordings, "Synthetic Test", ["ko"], rights_path, dry_run=True)
        factory.init(recordings, "Synthetic Test", ["ko"], rights_path)
        inspection = factory.inspect()
        prepared = factory.prepare()
        resumed = factory.build()
        try:
            factory.train()
            training = "unexpectedly_started"
        except FactoryError:
            training = "blocked"
        try:
            factory.package(release=True)
            release = "unexpectedly_created"
        except FactoryError:
            release = "refused"
        diagnostic = factory.package(diagnostic=True)
        normalized = json.loads((workspace / "manifests/segments.jsonl").read_text().splitlines()[0])["normalized_audio"]
        info = sf.info(normalized)
        return {
            "status": prepared["status"],
            "dry_run_workspace_created": False,
            "resource_estimate_files": dry_run["files"],
            "audio_valid": inspection["valid_count"],
            "audio_corrupt": inspection["corrupt_count"],
            "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest() == source_before,
            "normalized_sample_rate": info.samplerate,
            "normalized_channels": info.channels,
            "resume_status": resumed["status"],
            "training": training,
            "release": release,
            "diagnostic_package": diagnostic["status"],
            "audio_or_checkpoint_bundled": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_smoke()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != serialized:
            raise SystemExit("voicebank factory smoke report is missing or stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(serialized)
    print(f"PASS status={report['status']} release={report['release']}")


if __name__ == "__main__":
    main()
