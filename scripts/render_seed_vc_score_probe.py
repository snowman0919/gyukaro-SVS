#!/usr/bin/env python3
"""Reproduce the bounded official Seed-VC score-source conversion controls."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
REPOSITORY = CACHE / "seed-vc"
OUTPUT = ROOT / "artifacts/reports/seed_vc_score_probe/listening"
MODEL_REVISION = "257283f9f41585055e8f858fba4fd044e5caed6e"
REPOSITORY_REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"
MODEL = CACHE / "huggingface/hub/models--Plachta--Seed-VC/snapshots" / MODEL_REVISION / "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
CONFIG = REPOSITORY / "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
SOURCES = ROOT / "artifacts/reports/mlp_singer_korean_probe/listening"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(source: Path, reference: Path, steps: int, cfg: float) -> Path:
    target = OUTPUT / f"vc_{source.stem}_{reference.stem}_1.0_{steps}_{cfg}.wav"
    if target.is_file():
        return target
    # Seed-VC pins an older huggingface_hub call signature. This in-process
    # compatibility shim changes no model code or tensor path.
    code = """
import runpy, sys
from modules.bigvgan import bigvgan
defaults = bigvgan.BigVGAN._from_pretrained.__func__.__kwdefaults__
if defaults is not None:
    defaults.setdefault('proxies', None)
    defaults.setdefault('resume_download', None)
runpy.run_path('inference.py', run_name='__main__')
"""
    command = [
        str(REPOSITORY / ".venv/bin/python"), "-c", code,
        "--source", str(source), "--target", str(reference),
        "--output", str(OUTPUT), "--diffusion-steps", str(steps),
        "--length-adjust", "1.0", "--inference-cfg-rate", str(cfg),
        "--f0-condition", "true", "--auto-f0-adjust", "false",
        "--checkpoint", str(MODEL), "--config", str(CONFIG), "--fp16", "false",
    ]
    subprocess.run(
        command, cwd=REPOSITORY, check=True,
        env=os.environ | {
            "HF_HOME": str(CACHE / "huggingface"),
            "HF_HUB_CACHE": str(CACHE / "huggingface/hub"),
        },
    )
    return target


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    references = (
        (ROOT / "data/processed/master/192.wav", 30, 0.7),
        (ROOT / "data/external/work/diffsinger_score_native/raw/gyu_coarticulated/wavs/gyu_real_000146.wav", 50, 0.3),
    )
    outputs = []
    for case in ("rapid_ko", "large_interval_ko"):
        source = SOURCES / f"{case}_c6_generated_e2e.wav"
        for reference, steps, cfg in references:
            path = run(source, reference, steps, cfg)
            outputs.append({
                "case": case, "path": str(path.relative_to(ROOT)),
                "reference": str(reference.relative_to(ROOT)),
                "steps": steps, "cfg": cfg, "sha256": sha256(path),
            })
    report = {
        "status": "rendered_evaluation_only",
        "repository_revision": REPOSITORY_REVISION,
        "model_revision": MODEL_REVISION,
        "model_sha256": sha256(MODEL),
        "declared_license": "GPL-3.0",
        "training_data_provenance_documented": False,
        "score_source_is_csd_noncommercial": True,
        "production_integration_allowed": False,
        "outputs": outputs,
    }
    target = ROOT / "artifacts/reports/seed_vc_score_probe/render.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
