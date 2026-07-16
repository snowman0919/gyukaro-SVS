#!/usr/bin/env python3
"""Freeze the qualified GTSinger source and train only its diffusion decoder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt"
FINAL = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_diffusion/model_ckpt_steps_2000.ckpt"
EVALUATION = ROOT / "artifacts/reports/diffsinger_gtsinger_ja_diffusion_evaluation.json"


def main() -> None:
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    parameters = sum(
        value.numel() for name, value in checkpoint["state_dict"].items()
        if name.startswith("model.diffusion.")
    )
    config = yaml.safe_load((ROOT / "configs/diffsinger_gtsinger_ja.yaml").read_text())
    config.update({
        "finetune_enabled": True,
        "finetune_ckpt_path": str(SOURCE),
        "finetune_strict_shapes": True,
        "frozen_params": ["model.fs2", "model.aux_decoder"],
        "max_updates": 2_000,
        "val_check_interval": 250,
        "num_ckpt_keep": 4,
        "lambda_aux_mel_loss": 0.0,
        "optimizer_args": {"lr": 3e-4},
    })
    config["shallow_diffusion_args"].update({
        "train_aux_decoder": False,
        "train_diffusion": True,
    })
    assert config["frozen_params"] == ["model.fs2", "model.aux_decoder"]
    assert config["shallow_diffusion_args"]["train_diffusion"] is True
    assert config["shallow_diffusion_args"]["train_aux_decoder"] is False
    target = ROOT / "configs/diffsinger_gtsinger_ja_diffusion.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "ready",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "trainable_path": "model.diffusion only",
        "frozen_path": ["model.fs2", "model.aux_decoder"],
        "diffusion_parameters": parameters,
        "max_updates": 2_000,
        "config": str(target.relative_to(ROOT)),
        "release_allowed": False,
    }
    if FINAL.is_file():
        report.update({
            "status": "trained_unevaluated",
            "final_checkpoint": str(FINAL.relative_to(ROOT)),
            "final_checkpoint_sha256": hashlib.sha256(FINAL.read_bytes()).hexdigest(),
            "final_checkpoint_step": 2_000,
            "selected_inference": None,
            "human_listening": "not_started",
        })
    if EVALUATION.is_file():
        evaluation = json.loads(EVALUATION.read_text())
        accepted = evaluation["status"] == "source_probe_pass_human_pending"
        report.update({
            "status": ("trained_objective_gate_pass_human_pending" if accepted
                       else "trained_candidate_rejected"),
            "objective_gate": evaluation["status"],
            "selected_inference": evaluation["selected"],
            "human_listening": ("pending" if accepted
                                else "fail_excessive_pitch_and_unintelligible"),
        })
        if not accepted:
            report["rejection_reason"] = (
                "Free Whisper transcripts are unrelated; the target/reference F0 and duration do not align."
            )
    output = ROOT / "artifacts/reports/diffsinger_gtsinger_ja_diffusion.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
