#!/usr/bin/env python3
"""Train score-only TriSinger residual flow against real GYU RMVPE targets."""
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
    parser.add_argument("--manifest", default="data/manifests/real_gyu_prosody.jsonl")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--max-semitones", type=float, default=2.0)
    parser.add_argument("--teachers", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--teacher-style-supplement", default="data/manifests/teacher_style_supplement_weighted.jsonl")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--output", default="checkpoints/gyu_prosody_v0.5.pt")
    parser.add_argument("--report", default="artifacts/reports/gyu_prosody_training_v0.5.json")
    args = parser.parse_args()
    torch.manual_seed(17); device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = read_jsonl(args.manifest)
    if not rows or not all(row.get("target") == "real_gyu_rmvpe_log_f0" for row in rows):
        raise ValueError("v0.5 prosody training requires real_gyu_prosody.jsonl rows")
    teachers = read_jsonl(args.teachers) + read_jsonl(args.teacher_style_supplement)
    reference = acoustic_reference_features("data/processed/master/216.wav").to(device)
    model = TriSingerModel(dim=args.dim, latent_dim=1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    history = []
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        batch, duration = condition_batch(row["score"], reference, device)
        target_f0 = torch.from_numpy(np.load(row["target_f0_path"]).astype("float32")).to(device)
        # Preserve absolute recording time; generic resize would leak timing mismatch into residuals.
        target_times = torch.arange(target_f0.numel(), device=device, dtype=torch.float32) / 12.5
        frame_times = torch.linspace(0, duration, batch["f0_hz"].shape[1], device=device)
        positions = torch.searchsorted(target_times, frame_times).clamp(1, target_f0.numel() - 1)
        left, right = positions - 1, positions
        weight = (frame_times - target_times[left]) / (target_times[right] - target_times[left]).clamp_min(1e-6)
        target_f0 = target_f0[left] * (1 - weight) + target_f0[right] * weight
        voiced = target_f0 > 1
        target = torch.where(voiced, 12 * torch.log2(target_f0.clamp_min(1) / batch["f0_hz"][0]), torch.zeros_like(target_f0)).clamp(-args.max_semitones, args.max_semitones) / args.max_semitones
        target = target[None, :, None]
        time = torch.rand(1, device=device)
        source = model.acoustic_source(batch)
        output = model((1 - time[:, None, None]) * source + time[:, None, None] * target, time, batch)
        mask = voiced[None].float()
        source_loss = flow_matching_loss(source, target, mask)
        flow_loss = flow_matching_loss(output["velocity"], target - source, mask)
        pitch_loss = (((output["pitch_residual"] - target[..., 0] * args.max_semitones) ** 2) * mask).sum() / mask.sum().clamp_min(1)
        teacher_loss = torch.zeros((), device=device)
        if args.teacher_loss_weight and teachers:
            teacher = teachers[(step - 1) % len(teachers)]
            preset = teacher.get("style", "neutral")
            teacher_score = {"language": teacher["language"], "tempo": 120, "style": {"preset": preset if preset in {"neutral", "soft", "breathy", "energetic", "dark", "bright", "tense", "vibrato"} else "neutral"}, "notes": [{"pitch": 60, "start": 0, "duration": max(.1, float(teacher["duration_sec"])), "lyric": teacher["text"]}]}
            teacher_batch, _ = condition_batch(teacher_score, reference, device)
            teacher_target = acoustic_reference_features(teacher["output_path"], strict_sample_rate=False).to(device)[None]
            teacher_loss = weighted_distillation_loss(model.distillation_prediction(teacher_batch), teacher_target, torch.tensor([teacher["trust_weight"]], device=device))
        loss = source_loss + flow_loss + pitch_loss + args.teacher_loss_weight * teacher_loss
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 100 == 0: history.append({"step": step, "loss": round(float(loss.detach()), 6), "source": round(float(source_loss.detach()), 6), "flow": round(float(flow_loss.detach()), 6), "pitch": round(float(pitch_loss.detach()), 6), "teacher": round(float(teacher_loss.detach()), 6)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": args.dim, "latent_dim": 1}, "max_semitones": args.max_semitones, "residual_scale": .25, "steps": args.steps, "input": "nominal_score_plus_controls_only", "target": "real_GYU_RMVPE_log_f0_residual", "actual_f0_condition": False}, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"steps": args.steps, "dim": args.dim, "max_semitones": args.max_semitones, "rows": len(rows), "teacher_rows": len(teachers) if args.teacher_loss_weight else 0, "teacher_loss_weight": args.teacher_loss_weight, "real_target": True, "actual_f0_condition": False, "history": history}, indent=2) + "\n")


if __name__ == "__main__": main()
