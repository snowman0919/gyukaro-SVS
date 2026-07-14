#!/usr/bin/env python3
"""Package the reproducible but explicitly incomplete v0.6 experiment."""
from __future__ import annotations
import hashlib, json, shutil, zipfile
from pathlib import Path

NAME = "gyu-singer-v0.6-incomplete"
FILES = [
    Path("src/gyu_singer"), Path("scripts/probe_soulx_score.py"), Path("scripts/generate_omnivoice_phrase.py"),
    Path("scripts/build_teacher_internal_pairs.py"), Path("scripts/build_v06_prosody_manifest.py"), Path("scripts/extract_teacher_representations.py"), Path("scripts/train_identity_space_v06.py"),
    Path("scripts/init_latent_adapter_v06.py"), Path("scripts/evaluate_independent_prosody.py"), Path("scripts/evaluate_v06_identity_style_ablation.py"),
    Path("docs"), Path("examples/quality_ko.json"), Path("examples/quality_en.json"), Path("examples/quality_ja.json"),
    Path("checkpoints/gyu_prosody_v0.5.pt"), Path("checkpoints/gyu_prosody_v0.6.pt"), Path("checkpoints/gyu_acoustic_style_adapter_v0.5.pt"), Path("checkpoints/gyu_identity_space_v0.6.pt"), Path("checkpoints/gyu_latent_adapter_v0.6.pt"),
    Path("data/manifests/manual_verified_scores.jsonl"), Path("data/manifests/real_gyu_prosody_v06.jsonl"), Path("data/manifests/teacher_internal_pairs.jsonl"), Path("data/manifests/teacher_internal_representations.jsonl"), Path("data/cache/teacher_representations"),
]

root = Path("artifacts/package") / NAME
if root.exists(): shutil.rmtree(root)
root.mkdir(parents=True)
for source in FILES:
    if not source.exists(): continue
    destination = root / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("final_v0.6_report.md"))
    else: shutil.copy2(source, destination)
(root / "PACKAGE.json").write_text(json.dumps({"status": "incomplete", "backend": "gyu-singer-v0.6", "git_commit": "see repository final report", "reason": "independent prosody gain and clean full-path acceptance remain open"}, indent=2) + "\n")
archive = root.parent / f"{NAME}.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in root.rglob("*"):
        if path.is_file():
            info = zipfile.ZipInfo(str(Path(NAME) / path.relative_to(root)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
digest = hashlib.sha256(archive.read_bytes()).hexdigest(); archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n")
print({"package": str(archive), "sha256": digest, "status": "incomplete"})
