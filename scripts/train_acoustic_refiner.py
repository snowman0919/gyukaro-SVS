#!/usr/bin/env python3
"""Train universal, singing, or GYU residual acoustic-refiner stages."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.model import VocalAcousticRefiner


def read_audio(path: str, rate: int = 48000) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    if source_rate != rate:
        mono = resample_poly(mono, rate, source_rate).astype("float32")
    return mono


def load_pairs(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for index, row in enumerate(rows, 1):
        degraded, target = read_audio(row["degraded_input"]), read_audio(row["clean_target"])
        length = min(len(degraded), len(target)); degraded, target = degraded[:length], target[:length]
        # Corpus mastering levels are unrelated to reconstruction quality. Match
        # target loudness to the engine input and learn spectral shape, not gain.
        target *= np.sqrt(np.mean(degraded**2) + 1e-8) / np.sqrt(np.mean(target**2) + 1e-8)
        target *= min(1.0, .97 / max(float(np.max(np.abs(target))), 1e-8))
        row["arrays"] = (degraded.astype("float32"), target.astype("float32"))
        print(f"load {index}/{len(rows)} {row['id']}", flush=True)
    return rows


def crop(row: dict, samples: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    source, target = row["arrays"]
    if len(source) < samples:
        amount = samples - len(source); source = np.pad(source, (0, amount)); target = np.pad(target, (0, amount))
    start = rng.randrange(0, len(source) - samples + 1)
    return source[start:start + samples], target[start:start + samples]


def spectral_loss(output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    total = output.new_tensor(0.0); values = {}
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        window = torch.hann_window(n_fft, device=output.device)
        estimate = torch.stft(output, n_fft, hop, window=window, return_complex=True).abs().clamp_min(1e-5)
        truth = torch.stft(target, n_fft, hop, window=window, return_complex=True).abs().clamp_min(1e-5)
        log = torch.mean(torch.abs(torch.log(estimate) - torch.log(truth)))
        convergence = torch.linalg.vector_norm(estimate - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-5)
        value = log + convergence; total = total + value; values[f"stft_{n_fft}"] = float(value.detach())
    envelope = torch.nn.functional.l1_loss(torch.nn.functional.avg_pool1d(output.abs()[:, None], 240, 120), torch.nn.functional.avg_pool1d(target.abs()[:, None], 240, 120))
    return total / 3 + envelope, values | {"envelope": float(envelope.detach())}


def stage_rows(rows: list[dict], stage: str, split: str) -> tuple[list[dict], list[dict]]:
    primary_dataset = {"universal": "libritts_r", "singing": "vocalset", "gyu": "real_gyu"}[stage]
    primary = [row for row in rows if row["dataset"] == primary_dataset and row["split"] == split]
    replay = [] if stage == "universal" else [row for row in rows if row["dataset"] != primary_dataset and row["split"] == split]
    return primary, replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("universal", "singing", "gyu"), required=True)
    parser.add_argument("--init")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-samples", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=194)
    parser.add_argument("--manifest", type=Path, default=Path("data/external/manifests/pipeline_degradation_pairs.jsonl"))
    parser.add_argument("--output")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_pairs(args.manifest); primary, replay = stage_rows(rows, args.stage, "train"); validation, _ = stage_rows(rows, args.stage, "validation")
    if args.init:
        saved = torch.load(args.init, map_location="cpu", weights_only=False); model = VocalAcousticRefiner(**saved["model_config"]); model.load_state_dict(saved["model"])
    else:
        if args.stage != "universal":
            parser.error("--init is required for adapter stages")
        model = VocalAcousticRefiner()
    trainable = model.train_stage(args.stage); model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    rng = random.Random(args.seed); history = []; started = time.perf_counter(); mode = args.stage
    for step in range(1, args.steps + 1):
        selected = []
        for _ in range(args.batch_size):
            pool = replay if replay and rng.random() < .25 else primary
            selected.append(rng.choice(pool))
        chunks = [crop(row, args.crop_samples, rng) for row in selected]
        source = torch.from_numpy(np.stack([item[0] for item in chunks])).to(device)
        target = torch.from_numpy(np.stack([item[1] for item in chunks])).to(device)
        optimizer.zero_grad(set_to_none=True); output = model(source, mode)
        reconstruction, parts = spectral_loss(output, target)
        residual = torch.mean(torch.abs(output - source)); loss = reconstruction + .05 * residual
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            entry = {"step": step, "loss": round(float(loss.detach()), 6), "residual_l1": round(float(residual.detach()), 7)} | {key: round(value, 6) for key, value in parts.items()}
            history.append(entry); print(json.dumps(entry), flush=True)
    model.eval(); validation_losses = []
    with torch.inference_mode():
        for row in validation:
            source, target = crop(row, args.crop_samples, random.Random(args.seed))
            source_t, target_t = torch.from_numpy(source)[None].to(device), torch.from_numpy(target)[None].to(device)
            value, _ = spectral_loss(model(source_t, mode), target_t); validation_losses.append(float(value))
    output = Path(args.output or f"checkpoints/acoustic_refiner_{args.stage}.pt"); output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "stage": args.stage, "model_config": model.config, "model": {key: value.cpu() for key, value in model.state_dict().items()},
               "training": {"steps": args.steps, "batch_size": args.batch_size, "crop_samples": args.crop_samples, "learning_rate": args.learning_rate, "seed": args.seed,
                            "trainable_parameters": trainable, "primary_rows": len(primary), "replay_rows": len(replay), "validation_rows": len(validation),
                            "target_loudness": "RMS-matched to degraded input; peak capped at 0.97", "random_noise_augmentation": False,
                            "loss": "three-resolution log-STFT + spectral convergence + envelope L1 + 0.05 residual L1",
                            "wall_clock_sec": round(time.perf_counter() - started, 3), "validation_loss": round(float(np.mean(validation_losses)), 6), "history": history},
               "parent": args.init}
    torch.save(payload, output)
    report = payload["training"] | {"stage": args.stage, "checkpoint": str(output), "total_parameters": sum(parameter.numel() for parameter in model.parameters())}
    Path(f"artifacts/reports/acoustic_refiner_{args.stage}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
