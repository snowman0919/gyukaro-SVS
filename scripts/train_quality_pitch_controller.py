#!/usr/bin/env python3
"""Train score-only TriSinger residual flow for the quality decoder F0 input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gyu_singer.data import acoustic_reference_features, read_jsonl
from gyu_singer.inference.quality_controller import condition_batch
from gyu_singer.losses import flow_matching_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/quality_pitch_controller.jsonl")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--max-semitones", type=float, default=.10)
    parser.add_argument("--teachers", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--teacher-style-supplement", default="data/manifests/teacher_style_supplement_weighted.jsonl")
    parser.add_argument("--teacher-loss-weight", type=float, default=.05)
    parser.add_argument("--output", default="checkpoints/gyu_quality_pitch_controller.pt")
    parser.add_argument("--report", default="artifacts/reports/quality_pitch_controller_training.json")
    args = parser.parse_args()
    torch.manual_seed(17); device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_jsonl(args.manifest)
    teachers = read_jsonl(args.teachers) + read_jsonl(args.teacher_style_supplement)
    reference = acoustic_reference_features("data/processed/master/216.wav").to(device)
    model = TriSingerModel(dim=args.dim, latent_dim=1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    history = []
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        batch, _ = condition_batch(row["score"], reference, device)
        target_f0 = torch.from_numpy(np.load(row["target_f0_path"]).astype("float32")).to(device)
        target_f0 = torch.nn.functional.interpolate(target_f0[None, None], size=batch["f0_hz"].shape[1], mode="linear", align_corners=False)[0, 0]
        voiced = target_f0 > 1
        target = torch.where(voiced, 12 * torch.log2(target_f0.clamp_min(1) / batch["f0_hz"][0]), torch.zeros_like(target_f0)).clamp(-args.max_semitones, args.max_semitones) / args.max_semitones
        target = target[None, :, None]
        time = torch.rand(1, device=device)
        source = model.acoustic_source(batch)
        output = model((1 - time[:, None, None]) * source + time[:, None, None] * target, time, batch)
        mask = voiced[None].float()
        source_loss = flow_matching_loss(source, target, mask)
        flow_loss = flow_matching_loss(output["velocity"], target - source, mask)
        teacher = teachers[(step - 1) % len(teachers)]
        preset = teacher.get("style", "neutral")
        teacher_score = {"language": teacher["language"], "tempo": 120, "style": {"preset": preset if preset in {"neutral", "soft", "breathy", "energetic", "dark", "bright", "tense", "vibrato"} else "neutral"}, "notes": [{"pitch": 60, "start": 0, "duration": max(.1, float(teacher["duration_sec"])), "lyric": teacher["text"]}]}
        teacher_batch, _ = condition_batch(teacher_score, reference, device)
        teacher_target = acoustic_reference_features(teacher["output_path"], strict_sample_rate=False).to(device)[None]
        teacher_loss = weighted_distillation_loss(model.distillation_prediction(teacher_batch), teacher_target, torch.tensor([teacher["trust_weight"]], device=device))
        loss = source_loss + flow_loss + args.teacher_loss_weight * teacher_loss
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 100 == 0: history.append({"step": step, "loss": round(float(loss.detach()), 6), "source": round(float(source_loss.detach()), 6), "flow": round(float(flow_loss.detach()), 6), "teacher": round(float(teacher_loss.detach()), 6)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": args.dim, "latent_dim": 1}, "max_semitones": args.max_semitones, "steps": args.steps, "input": "score_plus_controls_only", "target": "RMVPE_f0_residual"}, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"steps": args.steps, "dim": args.dim, "max_semitones": args.max_semitones, "rows": len(rows), "teacher_rows": len(teachers), "teacher_loss_weight": args.teacher_loss_weight, "history": history}, indent=2) + "\n")


if __name__ == "__main__": main()
