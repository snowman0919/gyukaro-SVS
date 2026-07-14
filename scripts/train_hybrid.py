#!/usr/bin/env python3
"""Train compact phrase-level CFM SVS on real anchors plus weighted teacher representation loss."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.data import acoustic_reference_features, read_jsonl
from gyu_singer.frontend import phonemize
from gyu_singer.losses import flow_matching_loss, log_pitch_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel


def resize(values: torch.Tensor, length: int) -> torch.Tensor:
    return F.interpolate(values[None, None].float(), size=length, mode="linear", align_corners=False)[0, 0]


def batch_from_row(row: dict, device: str, target_length: int | None = None, teacher_audio: bool = False) -> dict[str, torch.Tensor]:
    front = phonemize(row["language"], row["text"])
    frames = build_phrase_frames(front, row["score"]["notes"])
    length = target_length or len(frames.midi)
    if row.get("f0_path") and Path(row["f0_path"]).exists():
        actual_f0 = torch.from_numpy(np.load(row["f0_path"]).astype("float32"))
        actual_f0 = resize(actual_f0, len(frames.midi))
        voiced = (actual_f0 > 1).float()
        nominal = 440.0 * torch.pow(torch.tensor(2.0), (frames.midi - 69.0) / 12.0)
        frames.f0_hz = torch.where(voiced.bool(), actual_f0, nominal)
        frames.voiced = voiced
        frames.residual = torch.where(voiced.bool(), 12 * torch.log2(frames.f0_hz.clamp_min(1) / nominal), torch.zeros_like(nominal))
    def seq(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = getattr(frames, name)
        if value.ndim == 1: value = resize(value, length)
        else: value = F.interpolate(value.T[None], size=length, mode="linear", align_corners=False)[0].T
        if dtype: value = value.to(dtype)
        return value[None].to(device)
    return {"phoneme_ids": seq("phoneme_ids", torch.long), "language_ids": seq("language_ids", torch.long), "features": seq("features"),
            "midi": seq("midi"), "note_index": seq("note_index", torch.long), "note_onset": seq("note_onset"), "note_duration": seq("note_duration"), "boundary": seq("boundary"), "f0_hz": seq("f0_hz"),
            "voiced": seq("voiced"), "residual": seq("residual"), "reference_features": acoustic_reference_features(row["audio_path"], strict_sample_rate=not teacher_audio)[None].to(device),
            "style_preset": torch.zeros(1, dtype=torch.long, device=device), "style_controls": torch.tensor([[0.8, 0, 0, 0, 0]], device=device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/manifests/hybrid_train.jsonl")
    parser.add_argument("--teachers", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--teacher-style-supplement", default="data/manifests/teacher_style_supplement_weighted.jsonl")
    parser.add_argument("--latents", default="data/cache/hybrid_latents")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.15)
    parser.add_argument("--output", default="checkpoints/gyu_hybrid_v0.2.pt")
    parser.add_argument("--report", default="artifacts/reports/hybrid_training.json")
    args = parser.parse_args()
    torch.manual_seed(7); random.seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, teachers = read_jsonl(args.train), read_jsonl(args.teachers) + read_jsonl(args.teacher_style_supplement)
    model = TriSingerModel(dim=args.dim).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    history = []
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        target = torch.load(Path(args.latents) / f"{row['id']}.pt", map_location=device, weights_only=True)[None]
        batch = batch_from_row(row, device, target.shape[1])
        noise, time = torch.randn_like(target), torch.rand(1, device=device)
        output = model((1 - time[:, None, None]) * noise + time[:, None, None] * target, time, batch)
        acoustic = output["velocity"] + 0.10 * output["acoustic_bias"]
        loss_flow = flow_matching_loss(acoustic, target - noise) * float(row["trust_weight"])
        loss_pitch = log_pitch_loss(output["pitch_log_f0"], batch["f0_hz"], batch["voiced"])
        loss_teacher = torch.zeros((), device=device)
        if args.teacher_loss_weight:
            teacher = teachers[(step - 1) % len(teachers)]
            teacher_features = acoustic_reference_features(teacher["output_path"], strict_sample_rate=False).to(device)[None]
            teacher_row = {"language": teacher["language"], "text": teacher["text"], "audio_path": teacher["output_path"],
                           "score": {"notes": [{"pitch": 60, "start": 0, "duration": max(0.1, float(teacher["duration_sec"])), "lyric": teacher["text"]}]}}
            teacher_batch = batch_from_row(teacher_row, device, 8, teacher_audio=True)
            loss_teacher = weighted_distillation_loss(model.distillation_prediction(teacher_batch), teacher_features, torch.tensor([teacher["trust_weight"]], device=device))
        loss = loss_flow + 0.05 * loss_pitch + args.teacher_loss_weight * loss_teacher
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append({"step": step, "loss": round(float(loss.detach()), 6), "flow": round(float(loss_flow.detach()), 6), "pitch": round(float(loss_pitch.detach()), 6), "teacher": round(float(loss_teacher.detach()), 6)})
        if step % 20 == 0: print(history[-1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": args.dim}, "steps": args.steps, "teacher_loss": "weighted_representation_only"}, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"device": device, "steps": args.steps, "real_rows": sum(row["phase"] == "C_real_gyu" for row in rows), "pseudo_rows": sum(row["phase"] == "B_pseudo_singing" for row in rows), "teacher_rows": len(teachers), "teacher_loss_weight": args.teacher_loss_weight, "history": history, "checkpoint": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
