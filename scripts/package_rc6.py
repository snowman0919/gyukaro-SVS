#!/usr/bin/env python3
"""Build deterministic RC6 candidate bytes without changing the RC5 package."""
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
    "gyu_identity_space_v0.6.pt", "gyu_real_latent_adapters_v0.7.pt", "acoustic_refiner_universal.pt",
]
EXAMPLES = [
    "quality_ko.json", "quality_en.json", "quality_ja.json", "heldout_ko.json", "heldout_en.json", "heldout_ja.json",
    "review_rapid_ko.json", "review_sustain_ko.json", "review_large_interval_ko.json", "review_phrase_boundary_ko.json",
    "openutau_v09.ustx", "openutau_v10_longform.ustx",
]
REPORTS = [
    "external_acoustic_quality.json", "pipeline_degradation_pairs.json", "acoustic_refiner_universal.json",
    "acoustic_refiner_singing.json", "acoustic_refiner_gyu.json", "refiner_identity_evaluation.json",
    "runtime_rc6_stress.json", "longform_rc6_render_metrics.json", "longform_rc6_supervised.json",
    "rc6_backend_candidate/manifest.json", "rc6_backend_candidate/evaluation.json", "rc6_runtime_smoke/verification.json",
]
DOCS = [
    "dataset_and_license_audit.md", "rc4_artifact_isolation.md", "timing_voicing_fix.md", "acoustic_prior_training.md",
    "singing_prior_training.md", "gyu_acoustic_adaptation.md", "final_quality_evaluation.md", "final_v1.0_report.md",
]


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(source, destination)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="artifacts/package/gyu-singer-v1.0.0-rc6-candidate.zip")
    args = parser.parse_args()
    package_dir = Path("artifacts/package")
    root = package_dir / NAME
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    copy(Path("src"), root / "src")
    copy(Path("pyproject.toml"), root / "pyproject.toml")
    for script in ("probe_soulx_score.py", "generate_omnivoice_phrase.py"):
        copy(Path("scripts") / script, root / "scripts" / script)
    for checkpoint in CHECKPOINTS:
        copy(Path("checkpoints") / checkpoint, root / "checkpoints" / checkpoint)
    copy(Path("data/processed/master/216.wav"), root / "model/gyu_reference_216.wav")
    copy(Path("integrations/openutau"), root / "integrations/openutau")
    for example in EXAMPLES:
        copy(Path("examples") / example, root / "examples" / example)
    for path in Path("distribution/v1").iterdir():
        copy(path, root / path.name)
    for report in REPORTS:
        copy(Path("artifacts/reports") / report, root / "evidence" / report)
    for doc in DOCS:
        copy(Path("docs") / doc, root / "evidence/docs" / doc)
    copy(Path("artifacts/reports/rc6_listening_gate"), root / "listening")

    for script in (root / "serve.sh", root / "render-example.sh"):
        script.write_text(script.read_text().replace("gyu-singer-rc5", "gyu-singer-rc6"))
    config = json.loads((root / "config.json").read_text())
    config["backend"] = "gyu-singer-rc6"
    config["release_state"] = "candidate; human listening pending"
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    dependencies = json.loads((root / "model-dependencies.json").read_text())
    dependencies["packaged_project_checkpoints"]["acoustic_refiner_universal.pt"] = sha(Path("checkpoints/acoustic_refiner_universal.pt"))
    (root / "model-dependencies.json").write_text(json.dumps(dependencies, indent=2) + "\n")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "version": "1.0.0-rc.6-candidate", "release_state": "objective candidate; human listening pending",
        "source_commit": commit, "backend": "gyu-singer-rc6",
        "openutau_revision": "27573ac5c888d927119d5f65a207312d79194b1f", "package_root": NAME,
        "per_note_tts": False, "waveform_pitch_shifting": False, "final_v1_release": False,
        "human_listening": "pending", "training_teachers_required_at_inference": False,
    }
    (root / "PACKAGE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "README.md").write_text("""# GYU Singer RC6 candidate

This archive is an objective audio-quality candidate, not final `v1.0.0`. Human review of every file under `listening/` is mandatory before release. The runtime uses phrase-level OmniVoice content, SoulX decode, canonical voicing/F0 timing, and the universal acoustic refiner at 25%. It does not use per-note TTS or waveform pitch shifting.

Install with `./install.sh --cache-source /path/to/pinned/cache`, then run `./serve.sh` or `./render-example.sh`. See `evidence/docs/final_v1.0_report.md` for measured limitations.
""")
    (root / "MODEL_CARD.md").write_text("""# Model card: RC6 candidate

Status: human listening pending; not a final release. Backend: `gyu-singer-rc6`. Decoder: pinned SoulX-Singer. A 51,537-parameter bounded residual refiner uses the universal backbone at 25%; singing and GYU-specific adapters were measured and disabled because production results were worse. Korean has personalized prosody evidence; English and Japanese use generic multilingual prosody with GYU identity/style. Linux NVIDIA CUDA is the tested platform.
""")
    (root / "LIMITATIONS.md").write_text("""# Limitations

- Human listening has not approved RC6.
- Aggregate HF energy rose slightly even as HF spikes and waveform jumps fell.
- Large-interval and phrase-boundary voicing scores remain weak.
- English has a measured “Sing”/“Sink” ASR ambiguity; isolated sustained vowels are ASR-ambiguous.
- Personalized prosody evidence is Korean-only.
- Linux NVIDIA CUDA is the only release-tested platform; pinned model caches are roughly 10 GB.
""")
    (root / "CHANGELOG.md").write_text("""# Changelog

## 1.0.0-rc.6-candidate

- Added canonical phrase voicing/F0 timing and retained safer RC5 SoulX settings.
- Added a bounded, silence-gated universal residual refiner trained on actual SoulX degradation pairs.
- Disabled singing/GYU refiner adapters after production ablations regressed quality.
- Added nine-file human listening gate. Final release remains blocked.
""")

    for path in (
        root / "install.sh", root / "serve.sh", root / "render-example.sh", root / "launch-openutau.sh",
        root / "model_downloader.py", root / "verify-install.py", root / "integrations/openutau/install_fork.sh",
        root / "integrations/openutau/test_resident_fork.sh", root / "integrations/openutau/test_longform_fork.sh",
    ):
        path.chmod(0o755)

    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
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
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(json.dumps({"package": str(archive), "sha256": digest, "bytes": archive.stat().st_size, "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
