#!/usr/bin/env python3
"""Train bounded score-native FiLM with two frozen speaker encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModelForAudioXVector


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
MODEL_ROOT = CACHE / "mlp-singer"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MODEL_ROOT))

from data.preprocess import Preprocessor  # noqa: E402
from train_mlp_singer_gyu_adapt import prepare_data, remap_model, sample_batch  # noqa: E402
from train_mlp_singer_gyu_film import GyuFilm, GyuResidual, hidden  # noqa: E402
from utils import AttrDict  # noqa: E402


SAVE_STEPS = (25, 50, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(audio: torch.Tensor) -> torch.Tensor:
    return (audio - audio.mean(-1, keepdim=True)) / torch.sqrt(
        audio.var(-1, keepdim=True, unbiased=False) + 1e-7
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=("film", "residual"), default="film")
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    preprocess_config = AttrDict(
        json.loads((ROOT / "configs/mlp_singer_preprocess_gyu.json").read_text())
    )
    train_rows, _, data_stats = prepare_data(Preprocessor(preprocess_config))
    model, _ = remap_model()
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    # The RVC/HiFi-GAN utility module shares the generic name `utils`; load the
    # score model first, then expose the vocoder implementation.
    sys.modules.pop("utils", None)
    sys.path.insert(0, str(MODEL_ROOT / "hifi-gan"))
    from env import AttrDict as HifiAttrDict
    from models import Generator

    hifi_config = HifiAttrDict(
        json.loads((MODEL_ROOT / "hifi-gan/config.json").read_text())
    )
    vocoder = Generator(hifi_config).cuda().eval()
    vocoder.load_state_dict(
        torch.load(
            MODEL_ROOT / "hifi-gan/g_02500000", map_location="cpu", weights_only=False
        )["generator"]
    )
    vocoder.remove_weight_norm()
    for parameter in vocoder.parameters():
        parameter.requires_grad = False

    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    for parameter in wavlm.parameters():
        parameter.requires_grad = False
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )
    for parameter in ecapa.mods.parameters():
        parameter.requires_grad = False

    if args.adapter == "film":
        adapter = GyuFilm(model.proj.in_features, limit=0.15).cuda()
    else:
        adapter = GyuResidual(model.proj.in_features, bottleneck=16, limit=0.1).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-3)
    generator = random.Random(42)
    output = MODEL_ROOT / f"checkpoints/gyu_speaker_{args.adapter}"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, max(SAVE_STEPS) + 1):
        notes, phonemes, target_mel = sample_batch(train_rows, 2, generator)
        with torch.no_grad():
            h = hidden(model, notes, phonemes)
            base_mel = model.proj(h)
            target_audio = vocoder(target_mel.transpose(1, 2)).squeeze(1)
            target_16k = AF.resample(target_audio, 22050, 16000)
            target_wavlm = F.normalize(
                wavlm(input_values=normalize(target_16k)).embeddings, dim=-1
            )
            target_ecapa = F.normalize(ecapa.encode_batch(target_16k).squeeze(1), dim=-1)

        adapted_mel = model.proj(adapter(h))
        adapted_audio = vocoder(adapted_mel.transpose(1, 2)).squeeze(1)
        adapted_16k = AF.resample(adapted_audio, 22050, 16000)
        adapted_wavlm = F.normalize(
            wavlm(input_values=normalize(adapted_16k)).embeddings, dim=-1
        )
        adapted_ecapa = F.normalize(ecapa.encode_batch(adapted_16k).squeeze(1), dim=-1)
        wavlm_loss = (1 - (adapted_wavlm * target_wavlm).sum(-1)).mean()
        ecapa_loss = (1 - (adapted_ecapa * target_ecapa).sum(-1)).mean()
        dynamics_loss = F.l1_loss(
            adapted_mel[:, 1:] - adapted_mel[:, :-1],
            base_mel[:, 1:] - base_mel[:, :-1],
        )
        base_distance = F.l1_loss(adapted_mel, base_mel)
        target_distance = F.l1_loss(adapted_mel, target_mel)
        loss = (
            wavlm_loss
            + ecapa_loss
            + 0.5 * dynamics_loss
            + 0.2 * base_distance
            + 0.05 * target_distance
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step in SAVE_STEPS:
            checkpoint = output / f"steps_{step}.pt"
            torch.save(
                {
                    "adapter": adapter.state_dict(),
                    "limit": adapter.limit,
                    "channels": model.proj.in_features,
                    "adapter_type": args.adapter,
                    "step": step,
                },
                checkpoint,
            )
            row = {
                "step": step,
                "wavlm_loss": round(float(wavlm_loss), 6),
                "ecapa_loss": round(float(ecapa_loss), 6),
                "dynamics_loss": round(float(dynamics_loss), 6),
                "base_mel_distance": round(float(base_distance), 6),
                "target_mel_distance": round(float(target_distance), 6),
                "checkpoint_sha256": sha256(checkpoint),
            }
            history.append(row)
            print(row, flush=True)
    report = {
        "status": "bounded_dual_speaker_film_training_complete_objective_evaluation_pending",
        "score_native": True,
        "data": data_stats,
        "adapter": {
            "location": "decoder hidden before frozen mel projection",
            "type": "bounded global FiLM" if args.adapter == "film" else "bounded low-rank residual",
            "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
            "modulation_limit": adapter.limit,
        },
        "speaker_supervision": ["microsoft/wavlm-base-plus-sv", "speechbrain/spkrec-ecapa-voxceleb"],
        "target": "real GYU mel reconstructed by the same frozen official HiFi-GAN",
        "frozen": "score model, mel projection, HiFi-GAN, WavLM, ECAPA",
        "history": history,
        "production_integrated": False,
    }
    (ROOT / f"artifacts/reports/mlp_singer_gyu_speaker_{args.adapter}_training.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
