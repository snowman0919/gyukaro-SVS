#!/usr/bin/env python3
"""Build deterministic GYU Singer v1.0 release-candidate bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path


NAME = "gyu-singer-v1.0"
CHECKPOINTS = [
    "gyu_prosody_v0.5.pt", "gyu_teacher_identity_v0.5.pt", "gyu_acoustic_style_adapter_v0.5.pt",
    "gyu_identity_space_v0.6.pt", "gyu_real_latent_adapters_v0.7.pt",
]
EXAMPLES = [
    "quality_ko.json", "quality_en.json", "quality_ja.json", "heldout_ko.json", "heldout_en.json", "heldout_ja.json",
    "review_rapid_ko.json", "review_sustain_ko.json", "review_large_interval_ko.json",
    "openutau_v09.ustx", "openutau_v10_longform.ustx",
]
EVIDENCE = [
    "openutau_upstream_v10.json", "runtime_v10_stress.json", "longform_v10_manifest.json",
    "longform_v10_render_metrics.json", "longform_v10_quality.json", "longform_v10_supervised.json",
    "release_audio_v10.json",
]


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--archive", default="artifacts/package/gyu-singer-v1.0.0-rc1.zip")
    args = parser.parse_args(); package_dir = Path("artifacts/package"); root = package_dir / NAME
    shutil.rmtree(root, ignore_errors=True); root.mkdir(parents=True)
    copy(Path("src"), root / "src"); copy(Path("pyproject.toml"), root / "pyproject.toml")
    for script in ("probe_soulx_score.py", "generate_omnivoice_phrase.py"):
        copy(Path("scripts") / script, root / "scripts" / script)
    for checkpoint in CHECKPOINTS: copy(Path("checkpoints") / checkpoint, root / "checkpoints" / checkpoint)
    copy(Path("data/processed/master/216.wav"), root / "model/gyu_reference_216.wav")
    copy(Path("integrations/openutau"), root / "integrations/openutau")
    for example in EXAMPLES: copy(Path("examples") / example, root / "examples" / example)
    for path in Path("distribution/v1").iterdir(): copy(path, root / path.name)
    for report in EVIDENCE: copy(Path("artifacts/reports") / report, root / "evidence" / report)
    copy(Path("artifacts/reports/listening_v10"), root / "listening")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "version": "1.0.0", "release_state": "release candidate", "source_commit": commit,
        "backend": "gyu-singer-v0.8", "openutau_revision": "27573ac5c888d927119d5f65a207312d79194b1f",
        "package_root": NAME, "per_note_tts": False, "waveform_pitch_shifting": False,
        "training_teachers_required_at_inference": False,
    }
    (root / "PACKAGE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for path in (root / "install.sh", root / "serve.sh", root / "render-example.sh", root / "launch-openutau.sh",
                 root / "model_downloader.py", root / "verify-install.py", root / "integrations/openutau/install_fork.sh",
                 root / "integrations/openutau/test_resident_fork.sh", root / "integrations/openutau/test_longform_fork.sh"):
        path.chmod(0o755)

    archive = Path(args.archive); archive.parent.mkdir(parents=True, exist_ok=True); archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(root.rglob("*")):
            if not path.is_file(): continue
            relative = Path(NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"package": str(archive), "sha256": digest, "bytes": archive.stat().st_size, "commit": commit}, indent=2))


if __name__ == "__main__": main()
