import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from gyu_singer.voicebank.audio import inspect_audio_directory
from gyu_singer.voicebank.factory import (
    FactoryError,
    VoicebankFactory,
    choose_transcript,
    deterministic_split,
    select_checkpoint,
)
from gyu_singer.voicebank.rights import validate_rights_manifest


ROOT = Path(__file__).resolve().parents[1]


def rights_manifest(input_dir: Path) -> dict:
    return {
        "source_type": "user_recordings",
        "owner_or_authorized_user": "Test Singer",
        "allowed_use": "local_voice_model_research",
        "redistribution_permission": "not_granted",
        "languages": ["ko"],
        "known_scripts": {"a.wav": "가나다"},
        "recording_environment": "test fixture",
        "consent_provenance_notes": "synthetic test audio",
        "permission_affirmed": True,
        "input_directory": str(input_dir),
    }


def write_tone(path: Path, frequency: float = 220.0, sample_rate: int = 44_100) -> None:
    time = np.arange(sample_rate // 4) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_rights_manifest_is_mandatory_and_complete(tmp_path):
    manifest = rights_manifest(tmp_path)
    validate_rights_manifest(manifest)
    del manifest["permission_affirmed"]
    with pytest.raises(FactoryError, match="rights manifest"):
        validate_rights_manifest(manifest)


def test_audio_inspection_detects_corruption_duplicates_and_preserves_sources(tmp_path):
    write_tone(tmp_path / "a.wav")
    (tmp_path / "b.wav").write_bytes((tmp_path / "a.wav").read_bytes())
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    before = hashlib.sha256((tmp_path / "a.wav").read_bytes()).hexdigest()

    report = inspect_audio_directory(tmp_path)

    assert report["corrupt_count"] == 1
    assert report["duplicate_groups"] == [["a.wav", "b.wav"]]
    assert report["rows"][0]["sample_rate"] == 44_100
    assert hashlib.sha256((tmp_path / "a.wav").read_bytes()).hexdigest() == before


def test_transcript_priority_and_untrusted_draft_exclusion():
    exact = choose_transcript("가나다", None, None, "가나다라", "ko")
    draft = choose_transcript(None, None, None, "가나다라", "ko")

    assert exact["source"] == "user_provided_exact_script"
    assert exact["training_trust"] == "high"
    assert draft["source"] == "untrusted_draft_transcript"
    assert draft["training_trust"] == "excluded_until_review"


def test_deterministic_split_rejects_duplicate_content_and_is_stable():
    rows = [
        {"id": f"clip-{index}", "audio_sha256": f"sha-{index}", "training_trust": "high"}
        for index in range(10)
    ]
    first = deterministic_split(rows)
    assert deterministic_split(list(reversed(rows))) == first
    leaked = rows + [{"id": "copy", "audio_sha256": "sha-0", "training_trust": "high"}]
    with pytest.raises(FactoryError, match="duplicate clips"):
        deterministic_split(leaked)


def test_checkpoint_selection_is_preservation_first_and_deterministic():
    candidates = [
        {"id": "late", "step": 20, "preservation_pass": True, "identity_score": 0.4, "update_norm": 0.2},
        {"id": "damaged", "step": 10, "preservation_pass": False, "identity_score": 0.9, "update_norm": 0.1},
        {"id": "small", "step": 10, "preservation_pass": True, "identity_score": 0.4, "update_norm": 0.1},
    ]
    assert select_checkpoint(candidates)["id"] == "small"


def test_factory_prepare_marks_alignment_and_stops_on_insufficient_coverage(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_tone(recordings / "a.wav")
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps(rights_manifest(recordings)))
    workspace = tmp_path / "workspace"
    factory = VoicebankFactory(ROOT, workspace)

    factory.init(recordings, "Test Singer", ["ko"], rights)
    result = factory.prepare()

    assert result["status"] == "dataset_needs_more_recording"
    row = json.loads((workspace / "manifests/segments.jsonl").read_text().splitlines()[0])
    assert row["alignment"]["classification"] in {"forced_aligned_low_confidence", "inferred_only"}
    assert row["vad"]["source"] == "energy_vad"
    assert row["vad"]["segments_seconds"]
    assert row["transcript"]["source"] == "user_provided_exact_script"
    info = sf.info(row["normalized_audio"])
    assert info.samplerate == 48_000 and info.channels == 1
    assert (workspace / "reports/recording_plan.json").is_file()
    with pytest.raises(FactoryError, match="coverage"):
        factory.train()


def test_factory_resume_dry_run_and_build_stop_are_observable(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_tone(recordings / "a.wav")
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps(rights_manifest(recordings)))
    workspace = tmp_path / "workspace"
    factory = VoicebankFactory(ROOT, workspace)

    estimate = factory.init(recordings, "Test Singer", ["ko"], rights, dry_run=True)
    assert estimate["status"] == "dry_run"
    assert not workspace.exists()
    factory.init(recordings, "Test Singer", ["ko"], rights)
    assert factory.init(recordings, "Test Singer", ["ko"], rights)["status"] == "already_complete"
    result = factory.build()
    assert result["status"] == "dataset_needs_more_recording"
    assert factory.build()["status"] == "dataset_needs_more_recording"
    assert json.loads((workspace / "state.json").read_text())["last_completed_phase"] == "prepare"
    assert (workspace / "factory.jsonl").read_text().count("\n") >= 3


def test_release_package_requires_human_approval_and_diagnostic_is_labeled(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "state.json").write_text(json.dumps({"last_completed_phase": "evaluate", "status": "voicebank_candidate_human_pending"}))
    factory = VoicebankFactory(ROOT, workspace)

    with pytest.raises(FactoryError, match="human approval"):
        factory.package(release=True)
    result = factory.package(diagnostic=True)
    assert result["status"] == "diagnostic_package_not_a_release"
    assert "NOT A RELEASE" in (workspace / "packages/voicebank-diagnostic/README.md").read_text()


def test_cli_dry_run_estimates_resources_without_creating_workspace(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_tone(recordings / "a.wav")
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps(rights_manifest(recordings)))
    workspace = tmp_path / "workspace"

    result = subprocess.run(
        [sys.executable, "-m", "gyu_singer.voicebank.cli", "init", "--input", str(recordings), "--name", "Test Singer", "--languages", "ko", "--workspace", str(workspace), "--rights-manifest", str(rights), "--dry-run"],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "dry_run"
    assert not workspace.exists()


def test_factory_configuration_forbids_network_acquisition():
    config = json.loads((ROOT / "configs/voicebank_factory.json").read_text())
    assert config["network_acquisition"] is False
    assert config["automatic_stt_ground_truth"] is False
    assert config["korean_whisper_role"] == "untrusted_draft_transcript"
    assert resources.files("gyu_singer").joinpath("voicebank_factory.json").read_bytes() == (
        ROOT / "configs/voicebank_factory.json"
    ).read_bytes()


def test_unreviewed_transcript_is_excluded_then_review_import_promotes_it(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    write_tone(recordings / "a.wav")
    manifest = rights_manifest(recordings)
    manifest["known_scripts"] = {}
    rights = tmp_path / "rights.json"
    rights.write_text(json.dumps(manifest))
    workspace = tmp_path / "workspace"
    factory = VoicebankFactory(ROOT, workspace)
    factory.init(recordings, "Test Singer", ["ko"], rights)

    first = factory.prepare()
    row = json.loads((workspace / "manifests/segments.jsonl").read_text().splitlines()[0])
    assert first["eligible_rows"] == 0
    assert row["transcript"]["source"] == "untrusted_draft_transcript"
    review = tmp_path / "review.csv"
    review.write_text("id,corrected_text,review_status\nclip-00000,가나다,accepted\n")
    second = factory.prepare(review)
    row = json.loads((workspace / "manifests/segments.jsonl").read_text().splitlines()[0])
    assert second["eligible_rows"] == 1
    assert row["transcript"]["source"] == "user_corrected_draft"


def test_voicebank_factory_smoke_report_is_reproducible():
    result = subprocess.run(
        [sys.executable, "scripts/smoke_voicebank_factory.py", "--check"],
        cwd=ROOT,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS status=dataset_needs_more_recording release=refused" in result.stdout
