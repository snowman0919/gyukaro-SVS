#!/usr/bin/env python3
"""Train a tiny, bounded GYU FiLM adapter inside the score-native model."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data/cache/mlp-singer"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MODEL_ROOT))

from data.preprocess import Preprocessor  # noqa: E402
from train_mlp_singer_gyu_adapt import prepare_data, remap_model, sample_batch  # noqa: E402
from utils import AttrDict  # noqa: E402


SAVE_STEPS = (100, 400, 1000)


class GyuFilm(nn.Module):
    """A global speaker transform with bounded hidden-state modulation."""

    def __init__(self, channels: int, limit: float = 0.3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.limit = limit

    def forward(self, hidden: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        scale = self.limit * torch.tanh(self.gamma)
        shift = self.limit * torch.tanh(self.beta)
        return hidden * (1 + strength * scale) + strength * shift


class GyuResidual(nn.Module):
    """A zero-initialized, bounded low-rank hidden residual."""

    def __init__(self, channels: int, bottleneck: int = 16, limit: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.limit = limit

    def forward(self, hidden: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(hidden))))
        return hidden + strength * self.limit * torch.tanh(residual)


def hidden(model: nn.Module, notes: torch.Tensor, phonemes: torch.Tensor) -> torch.Tensor:
    pitch = model.pitch_embed(notes)
    text = model.text_embed(phonemes)
    return model.decoder(model.embed(torch.cat((text, pitch), dim=-1)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def validate(model, adapter, rows: list[tuple]) -> tuple[float, float, float]:
    target_losses = []
    base_losses = []
    dynamics = []
    for notes, phonemes, target in rows:
        starts = list(range(0, len(notes) - 192 + 1, 192))
        if starts[-1] != len(notes) - 192:
            starts.append(len(notes) - 192)
        for start in starts:
            n = notes[start : start + 192].unsqueeze(0).cuda()
            p = phonemes[start : start + 192].unsqueeze(0).cuda()
            y = target[start : start + 192].unsqueeze(0).cuda()
            h = hidden(model, n, p)
            base = model.proj(h)
            adapted = model.proj(adapter(h))
            base_losses.append(F.l1_loss(base, y).item())
            target_losses.append(F.l1_loss(adapted, y).item())
            dynamics.append(
                F.l1_loss(adapted[:, 1:] - adapted[:, :-1], base[:, 1:] - base[:, :-1]).item()
            )
    mean = lambda values: sum(values) / len(values)
    return mean(base_losses), mean(target_losses), mean(dynamics)


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    preprocess_config = AttrDict(
        json.loads((ROOT / "configs/mlp_singer_preprocess_gyu.json").read_text())
    )
    train_rows, validation_rows, data_stats = prepare_data(Preprocessor(preprocess_config))
    model, model_config = remap_model()
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    channels = model.proj.in_features
    adapter = GyuFilm(channels).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = random.Random(42)
    output = MODEL_ROOT / "checkpoints/gyu_film"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, max(SAVE_STEPS) + 1):
        notes, phonemes, target = sample_batch(train_rows, 16, generator)
        with torch.no_grad():
            h = hidden(model, notes, phonemes)
            base = model.proj(h)
        adapted = model.proj(adapter(h))
        target_loss = F.l1_loss(adapted, target)
        dynamics_loss = F.l1_loss(
            adapted[:, 1:] - adapted[:, :-1], base[:, 1:] - base[:, :-1]
        )
        magnitude_loss = (adapter.gamma.square().mean() + adapter.beta.square().mean())
        loss = target_loss + 0.5 * dynamics_loss + 0.001 * magnitude_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step in SAVE_STEPS:
            base_val, adapted_val, dynamics_val = validate(model, adapter, validation_rows)
            checkpoint = output / f"steps_{step}.pt"
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "limit": adapter.limit,
                    "channels": channels,
                    "step": step,
                },
                checkpoint,
            )
            row = {
                "step": step,
                "train_target_loss": round(float(target_loss), 6),
                "validation_base_loss": round(base_val, 6),
                "validation_adapted_loss": round(adapted_val, 6),
                "validation_dynamics_delta": round(dynamics_val, 6),
                "checkpoint_sha256": sha256(checkpoint),
            }
            history.append(row)
            print(row, flush=True)
    report = {
        "status": "bounded_latent_film_training_complete_objective_evaluation_pending",
        "base_model": "neosapience/mlp-singer@7f4621ca04ee5e35c0e0a80b1fed785a55a51891",
        "base_model_license": "MIT",
        "score_native": True,
        "speaker_target": "GYU",
        "data": data_stats,
        "adapter": {
            "location": "decoder hidden before frozen mel projection",
            "type": "bounded global FiLM",
            "channels": channels,
            "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
            "modulation_limit": adapter.limit,
        },
        "frozen": "phoneme embedding, pitch embedding, mixer decoder, mel projection, HiFi-GAN",
        "objective": "target mel L1 + 0.5 temporal-dynamics preservation + parameter regularization",
        "history": history,
        "production_integrated": False,
    }
    (ROOT / "artifacts/reports/mlp_singer_gyu_film_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
