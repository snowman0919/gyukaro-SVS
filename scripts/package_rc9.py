#!/usr/bin/env python3
"""Build deterministic RC9 bytes after the song and human gates pass."""
from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from package_rc6 import CHECKPOINTS, EXAMPLES, copy, sha


NAME = "gyu-singer-rc9"
DOCS = [
    "rc7_baseline.md", "rc8_quality_fixes.md", "rc8_listening_report.md",
    "reference_song_analysis.md", "openutau_song_validation.md", "rc9_package.md", "final_rc9_report.md",
]
REPORTS = [
    "openutau_upstream_v10.json", "reference_song_rc9_analysis.json", "reference_song_rc9_project.json",
    "reference_song_rc9_runtime.json", "reference_song_rc9_evaluation.json", "reference_song_rc9_identity.json",
]


def main() -> None:
    package_dir = Path("artifacts/package")
    root = package_dir / NAME
    required = [Path("docs") / name for name in DOCS] + [Path("artifacts/reports") / name for name in REPORTS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RC9 gate evidence missing: {missing}")
    evaluation = json.loads(Path("artifacts/reports/reference_song_rc9_evaluation.json").read_text())
    identity = json.loads(Path("artifacts/reports/reference_song_rc9_identity.json").read_text())
    if evaluation["status"] != "objective_pass_human_listening_pending" or identity["status"] != "identity_nonregression_pass_human_pending":
        raise RuntimeError("RC9 objective song or identity gate has not passed")
    if "Overall status: RC9 achieved" not in Path("docs/final_rc9_report.md").read_text():
        raise RuntimeError("human listening acceptance is not recorded in docs/final_rc9_report.md")

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    copy(Path("src"), root / "src")
    copy(Path("pyproject.toml"), root / "pyproject.toml")
    for name in ("probe_soulx_score.py", "generate_omnivoice_phrase.py"):
        copy(Path("scripts") / name, root / "scripts" / name)
    checkpoints = CHECKPOINTS + ["acoustic_refiner_spectral_singing.pt"]
    for name in checkpoints:
        copy(Path("checkpoints") / name, root / "checkpoints" / name)
    copy(Path("data/processed/master/216.wav"), root / "model/gyu_reference_216.wav")
    copy(Path("integrations/openutau"), root / "integrations/openutau")
    for name in EXAMPLES:
        copy(Path("examples") / name, root / "examples" / name)
    for path in Path("distribution/v1").iterdir():
        copy(path, root / path.name)
    for name in DOCS:
        copy(Path("docs") / name, root / "evidence/docs" / name)
    for name in REPORTS:
        copy(Path("artifacts/reports") / name, root / "evidence" / name)

    for script in (root / "serve.sh", root / "render-example.sh"):
        script.write_text(script.read_text().replace("gyu-singer-rc5", "gyu-singer-rc9"))
    config = json.loads((root / "config.json").read_text())
    config.update({"backend": "gyu-singer-rc9", "release_state": "RC9 human-approved candidate; not final v1.0.0"})
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dependencies = json.loads((root / "model-dependencies.json").read_text())
    dependencies["packaged_project_checkpoints"]["acoustic_refiner_spectral_singing.pt"] = sha(Path("checkpoints/acoustic_refiner_spectral_singing.pt"))
    (root / "model-dependencies.json").write_text(json.dumps(dependencies, indent=2) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "version": "1.0.0-rc.9", "backend": "gyu-singer-rc9", "source_commit": commit,
        "release_state": "human-approved release candidate; not final v1.0.0",
        "openutau_revision": "27573ac5c888d927119d5f65a207312d79194b1f",
        "package_root": NAME, "human_listening": "pass", "final_v1_release": False,
        "per_note_tts": False, "waveform_pitch_shifting": False,
        "copyrighted_reference_material_included": False,
    }
    (root / "PACKAGE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "README.md").write_text(
        "# GYU Singer RC9\n\nHuman-approved OpenUtau production candidate; not final v1.0.0. "
        "It uses phrase-level OmniVoice content and SoulX decode, preserves the RC8 quality path, and keeps "
        "personalized prosody Korean-only. Install with `./install.sh --cache-source /path/to/pinned/cache`. "
        "The copyrighted local reference song, stems, lyrics, and reconstructed USTX are not included.\n"
    )
    for path in (
        root / "install.sh", root / "serve.sh", root / "render-example.sh", root / "launch-openutau.sh",
        root / "model_downloader.py", root / "verify-install.py", root / "integrations/openutau/install_fork.sh",
        root / "integrations/openutau/test_resident_fork.sh", root / "integrations/openutau/test_longform_fork.sh",
    ):
        path.chmod(0o755)

    archive = package_dir / f"{NAME}.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(str(relative), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = sha(archive)
    (package_dir / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"package": str(archive), "sha256": digest, "bytes": archive.stat().st_size, "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
