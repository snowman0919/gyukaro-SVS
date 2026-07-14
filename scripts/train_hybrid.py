#!/usr/bin/env python3
"""Train compact phrase-level CFM SVS on real anchors plus weighted teacher representation loss."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.data import acoustic_reference_features, read_jsonl
from gyu_singer.frontend import phonemize
from gyu_singer.losses import flow_matching_loss, log_pitch_loss, weighted_distillation_loss
from gyu_singer.model import TriSingerModel
from gyu_singer.inference.codec import MossCodecDecoder


STYLE_PRESETS = {"neutral": 0, "soft": 1, "breathy": 2, "energetic": 3, "dark": 4, "bright": 5, "tense": 6, "vibrato": 7}

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
            "style_preset": torch.tensor([STYLE_PRESETS.get(row.get("style", "neutral"), 0)], dtype=torch.long, device=device), "style_controls": torch.tensor([[0.8, 0, 0, 0, 0]], device=device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/manifests/hybrid_train.jsonl")
    parser.add_argument("--validation", default="data/manifests/hybrid_valid.jsonl")
    parser.add_argument("--teachers", default="data/manifests/teacher_weighted.jsonl")
    parser.add_argument("--teacher-style-supplement", default="data/manifests/teacher_style_supplement_weighted.jsonl")
    parser.add_argument("--latents", default="data/cache/hybrid_latents")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.15)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-audio-every", type=int, default=1000)
    parser.add_argument("--validation-audio-dir", default="artifacts/validation/hybrid")
    parser.add_argument("--resume", help="checkpoint model weights to continue training")
    parser.add_argument("--output", default="checkpoints/gyu_hybrid_v0.2.pt")
    parser.add_argument("--report", default="artifacts/reports/hybrid_training.json")
    args = parser.parse_args()
    torch.manual_seed(7); random.seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, validation_rows = read_jsonl(args.train), read_jsonl(args.validation)
    teachers = read_jsonl(args.teachers) + read_jsonl(args.teacher_style_supplement)
    model = TriSingerModel(dim=args.dim).to(device).train()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False)["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    history, validation, validation_audio = [], [], []
    started = time.perf_counter()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    codec = MossCodecDecoder("data/cache/moss-audio-tokenizer-nano", device) if args.validation_audio_every else None

    @torch.inference_mode()
    def validate() -> dict:
        model.eval(); values = []
        for row in validation_rows:
            target = torch.load(Path(args.latents) / f"{row['id']}.pt", map_location=device, weights_only=True)[None]
            batch = batch_from_row(row, device, target.shape[1])
            output = model(torch.zeros_like(target), torch.full((1,), .5, device=device), batch)
            values.append(float(flow_matching_loss(output["velocity"], target) + .05 * log_pitch_loss(output["pitch_log_f0"], batch["f0_hz"], batch["voiced"])))
        model.train()
        return {"step": step, "acoustic_loss": round(sum(values) / len(values), 6)}
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        target = torch.load(Path(args.latents) / f"{row['id']}.pt", map_location=device, weights_only=True)[None]
        batch = batch_from_row(row, device, target.shape[1])
        noise, flow_time = torch.randn_like(target), torch.rand(1, device=device)
        output = model((1 - flow_time[:, None, None]) * noise + flow_time[:, None, None] * target, flow_time, batch)
        acoustic = output["velocity"] + 0.10 * output["acoustic_bias"]
        loss_flow = flow_matching_loss(acoustic, target - noise) * float(row["trust_weight"])
        loss_pitch = log_pitch_loss(output["pitch_log_f0"], batch["f0_hz"], batch["voiced"]) * float(row["trust_weight"])
        loss_teacher = torch.zeros((), device=device)
        if args.teacher_loss_weight:
            teacher = teachers[(step - 1) % len(teachers)]
            teacher_features = acoustic_reference_features(teacher["output_path"], strict_sample_rate=False).to(device)[None]
            teacher_row = {"language": teacher["language"], "text": teacher["text"], "style": teacher.get("style", "neutral"), "audio_path": teacher["output_path"],
                           "score": {"notes": [{"pitch": 60, "start": 0, "duration": max(0.1, float(teacher["duration_sec"])), "lyric": teacher["text"]}]}}
            teacher_batch = batch_from_row(teacher_row, device, 8, teacher_audio=True)
            loss_teacher = weighted_distillation_loss(model.distillation_prediction(teacher_batch), teacher_features, torch.tensor([teacher["trust_weight"]], device=device))
        loss = loss_flow + 0.05 * loss_pitch + args.teacher_loss_weight * loss_teacher
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append({"step": step, "loss": round(float(loss.detach()), 6), "flow": round(float(loss_flow.detach()), 6), "pitch": round(float(loss_pitch.detach()), 6), "teacher": round(float(loss_teacher.detach()), 6)})
        if step % 20 == 0: print(history[-1])
        if step % args.validate_every == 0:
            validation.append(validate()); print(validation[-1])
        if codec and step % args.validation_audio_every == 0:
            model.eval()
            row = validation_rows[0]
            target = torch.load(Path(args.latents) / f"{row['id']}.pt", map_location=device, weights_only=True)[None]
            sample = model.sample(batch_from_row(row, device, target.shape[1]))
            path = Path(args.validation_audio_dir) / f"step_{step:06d}.wav"; path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, codec.decode(sample)[0].numpy(), 48000)
            validation_audio.append(str(path)); model.train()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.eval().cpu().state_dict(), "model_config": {"dim": args.dim}, "steps": args.steps, "teacher_loss": "weighted_representation_only"}, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    report = {"device": device, "steps": args.steps, "resume": args.resume, "optimizer": "AdamW", "learning_rate": 2e-4, "batch_size": 1, "gradient_accumulation": 1, "precision": "torch_default", "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()), "real_rows": sum(row["phase"] == "C_real_gyu" for row in rows), "pseudo_rows": sum(row["phase"] == "B_pseudo_singing" for row in rows), "validation_rows": len(validation_rows), "teacher_rows": len(teachers), "teacher_loss_weight": args.teacher_loss_weight, "trust_weights": sorted({float(row["trust_weight"]) for row in rows}), "train_duration_sec": round(sum(sf.info(row["audio_path"]).duration for row in rows), 3), "wall_clock_sec": round(time.perf_counter() - started, 3), "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0, "history": history, "validation": validation, "validation_audio": validation_audio, "checkpoint": args.output}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
