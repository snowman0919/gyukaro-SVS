#!/usr/bin/env python3
"""Train compact phrase-level CFM SVS on real anchors plus weighted teacher representation loss."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.data import acoustic_reference_features, read_jsonl
from gyu_singer.frontend import phonemize
from gyu_singer.losses import flow_matching_loss, log_pitch_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel


def resize(values: torch.Tensor, length: int) -> torch.Tensor:
    return F.interpolate(values[None, None].float(), size=length, mode="linear", align_corners=False)[0, 0]


def batch_from_row(row: dict, device: str, target_length: int | None = None) -> dict[str, torch.Tensor]:
    front = phonemize(row["language"], row["text"])
    frames = build_phrase_frames(front, row["score"]["notes"])
    length = target_length or len(frames.midi)
    def seq(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        value = getattr(frames, name)
        if value.ndim == 1: value = resize(value, length)
        else: value = F.interpolate(value.T[None], size=length, mode="linear", align_corners=False)[0].T
        if dtype: value = value.to(dtype)
        return value[None].to(device)
    return {"phoneme_ids": seq("phoneme_ids", torch.long), "language_ids": seq("language_ids", torch.long), "features": seq("features"),
            "midi": seq("midi"), "note_index": seq("note_index", torch.long), "boundary": seq("boundary"), "f0_hz": seq("f0_hz"),
            "voiced": seq("voiced"), "residual": seq("residual"), "reference_features": acoustic_reference_features(row["audio_path"])[None].to(device),
            "style_preset": torch.zeros(1, dtype=torch.long, device=device), "style_controls": torch.tensor([[0.8, 0, 0, 0, 0]], device=device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/manifests/hybrid_train.jsonl")
    parser.add_argument("--teachers", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--latents", default="data/cache/hybrid_latents")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--output", default="checkpoints/gyu_hybrid_v0.2.pt")
    parser.add_argument("--report", default="artifacts/reports/hybrid_training.json")
    args = parser.parse_args()
    torch.manual_seed(7); random.seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, teachers = read_jsonl(args.train), read_jsonl(args.teachers)
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
        teacher = teachers[(step - 1) % len(teachers)]
        teacher_features = acoustic_reference_features(teacher["output_path"]).to(device)[None]
        teacher_row = {"language": teacher["language"], "text": teacher["text"], "audio_path": teacher["output_path"],
                       "score": {"notes": [{"pitch": 60, "start": 0, "duration": max(0.1, float(teacher["duration_sec"])), "lyric": teacher["text"]}]}}
        teacher_batch = batch_from_row(teacher_row, device, 8)
        loss_teacher = weighted_distillation_loss(model.distillation_prediction(teacher_batch), teacher_features, torch.tensor([teacher["trust_weight"]], device=device))
        loss = loss_flow + 0.05 * loss_pitch + 0.15 * loss_teacher
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append({"step": step, "loss": round(float(loss.detach()), 6), "flow": round(float(loss_flow.detach()), 6), "pitch": round(float(loss_pitch.detach()), 6), "teacher": round(float(loss_teacher.detach()), 6)})
        if step % 20 == 0: print(history[-1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": args.dim}, "steps": args.steps, "teacher_loss": "weighted_representation_only"}, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"device": device, "steps": args.steps, "real_rows": len(rows), "teacher_rows": len(teachers), "history": history, "checkpoint": args.output}, indent=2) + "\n")


if __name__ == "__main__":
    main()
