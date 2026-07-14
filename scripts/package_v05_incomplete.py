#!/usr/bin/env python3
"""Reproducibly package the honest v0.5-incomplete runtime."""
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    root = Path("."); output = root / "artifacts/package/gyu-singer-v0.5-incomplete.zip"; output.parent.mkdir(parents=True, exist_ok=True)
    files = [Path("src/gyu_singer"), Path("scripts/prepare_real_prosody.py"), Path("scripts/train_quality_pitch_controller.py"), Path("scripts/train_acoustic_style_adapter.py"), Path("scripts/train_teacher_representation.py"), Path("scripts/extract_teacher_representations.py"), Path("scripts/reconstruct_real_scores.py"), Path("scripts/align_real_phonemes.py"), Path("scripts/report_v05_alignment.py"), Path("scripts/report_gyu_prosody.py"), Path("scripts/package_v05_incomplete.py"), Path("docs/score_reconstruction_report.md"), Path("docs/phoneme_alignment_report.md"), Path("docs/gyu_prosody_analysis.md"), Path("docs/acoustic_style_adapter.md"), Path("docs/teacher_representation_distillation.md"), Path("docs/openutau_v0.5_readiness.md"), Path("checkpoints/gyu_prosody_v0.5.pt"), Path("checkpoints/gyu_acoustic_style_adapter_v0.5.pt"), Path("checkpoints/gyu_teacher_identity_v0.5.pt"), Path("data/manifests/real_score_accepted.jsonl"), Path("data/manifests/real_gyu_prosody.jsonl"), Path("data/manifests/real_phoneme_alignment.jsonl"), Path("examples/quality_ko.json")]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files, key=str):
            if path.is_dir():
                for child in sorted(path.rglob("*.py")):
                    archive.write(child, Path("gyu-singer-v0.5-incomplete") / child)
            elif path.exists(): archive.write(path, Path("gyu-singer-v0.5-incomplete") / path)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        archive.writestr("gyu-singer-v0.5-incomplete/PACKAGE.json", json.dumps({"status": "incomplete", "git_commit": commit, "backend": "gyu-singer-v0.5", "v0_4_fallback": False}, indent=2) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest(); output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n"); print({"package": str(output), "sha256": digest})


if __name__ == "__main__": main()
