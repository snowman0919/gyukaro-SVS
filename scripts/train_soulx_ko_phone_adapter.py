#!/usr/bin/env python3
"""Train only unused SoulX phone embeddings on real-GYU phrase reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "data/cache/soulx-singer"
sys.path[:0] = [str(ROOT / "src"), str(SOULX)]

from gyu_singer.inference.content_timing import roman_phone  # noqa: E402
from soulxsinger.models.soulxsinger import SoulXSinger  # noqa: E402
from soulxsinger.utils.data_processor import DataProcessor  # noqa: E402
from soulxsinger.utils.file_utils import load_config  # noqa: E402
from soulx_exact_ko_processor import ExactKoreanDataProcessor  # noqa: E402


ROMAN_TO_EN = {
    "a": "en_AA1", "ae": "en_AE1", "b": "en_B", "ch": "en_CH",
    "d": "en_D", "e": "en_EH1", "eo": "en_AO1", "eu": "en_UH1",
    "g": "en_G", "h": "en_HH", "i": "en_IY1", "j": "en_JH",
    "k": "en_K", "l": "en_L", "m": "en_M", "n": "en_N",
    "ng": "en_NG", "o": "en_OW1", "p": "en_P", "r": "en_R",
    "s": "en_S", "t": "en_T", "u": "en_UW1", "ui": "en_IY1",
    "w": "en_W", "y": "en_Y",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_token(phone: str, available: set[str]) -> str:
    roman = roman_phone(phone)
    roman = {
        "kk": "k", "tt": "t", "pp": "p", "ss": "s",
        "nh": "n", "lg": "l", "wa": "a", "wae": "ae",
        "wo": "eo", "yo": "o", "ya": "a", "ye": "e", "yeo": "eo",
    }.get(roman, roman)
    value = ROMAN_TO_EN.get(roman, "<SP>" if phone == "ko_onset_11" else "<UNK>")
    return value if value in available else "<UNK>"


def frame_f0(audio: np.ndarray, rate: int, frame_count: int) -> np.ndarray:
    audio = librosa.resample(audio.astype(np.float32), orig_sr=rate, target_sr=24000)
    values, _, _ = librosa.pyin(audio, fmin=60, fmax=900, sr=24000, hop_length=480)
    values = np.nan_to_num(values, nan=0.0).astype(np.float32)
    if len(values) == frame_count:
        return values
    return np.interp(
        np.linspace(0.0, 1.0, frame_count),
        np.linspace(0.0, 1.0, max(1, len(values))),
        values,
    ).astype(np.float32)


def metadata(row: dict) -> dict:
    groups: list[tuple[list[str], list[float]]] = []
    current: list[str] = []
    phone_durations: list[float] = []
    for phone, value in zip(row["ph_seq"], row["ph_dur"]):
        value = float(value)
        if phone in {"AP", "SP"}:
            if current:
                groups.append((current, phone_durations))
                current, phone_durations = [], []
            groups.append((["<SP>"], [value]))
        else:
            current.append(phone)
            phone_durations.append(value)
    if current:
        groups.append((current, phone_durations))
    durations = [sum(values) for _, values in groups]
    frame_count = int(sum(durations) * 50)
    audio, rate = sf.read(ROOT / row["audio_path"], dtype="float32", always_2d=True)
    f0 = frame_f0(audio.mean(1), rate, frame_count)
    phones = ["<SP>" if group == ["<SP>"] else "koexact:" + "|".join(
        f"{phone}@{value:.6f}" for phone, value in zip(group, values)
    ) for group, values in groups]
    return {
        "duration": " ".join(f"{value:.6f}" for value in durations),
        "phoneme": " ".join(phones),
        "note_pitch": " ".join("0" for _ in phones),
        "note_type": " ".join("1" if phone == "<SP>" else "2" for phone in phones),
        "f0": " ".join(f"{value:.3f}" for value in f0),
    }


def flow_loss(model: SoulXSinger, item: dict) -> torch.Tensor:
    waveform = item["waveform"].float()
    mel = model.mel(waveform)
    frames = min(mel.shape[1], item["mel2note"].shape[1], item["f0"].shape[1])
    mel = mel[:, :frames]
    mel2note = item["mel2note"][:, :frames]
    f0 = item["f0"][:, :frames]
    features = (
        model.note_pitch_encoder(item["note_pitch"])
        + model.note_type_encoder(item["note_type"])
        + model.note_text_encoder(item["phoneme"])
    )
    features = model.preflow(features)
    features = model.expand_states(features, mel2note)
    features = features + model.f0_encoder(model.f0_to_coarse(f0))
    mask = torch.ones(mel.shape[:2], device=mel.device)
    is_prompt = torch.zeros_like(mask)
    is_prompt[:, :max(5, int(frames * 0.2))] = 1
    noise, clean, prediction, final_mask, _ = model.cfm_decoder(
        mel, mask, features, is_prompt
    )
    target = clean - (1 - model.cfm_decoder.model.sigma) * noise
    return (
        F.l1_loss(prediction, target, reduction="none").float()
        * final_mask
    ).mean(dim=2).sum() / final_mask.sum().clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-updates", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/cache/soulx_ko_phone_adapter.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SoulX phone training requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [
        json.loads(line)
        for line in (ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl").read_text().splitlines()
    ]
    original_phones = json.loads(
        (SOULX / "soulxsinger/utils/phoneme/phone_set.json").read_text()
    )
    korean_phones = sorted({
        phone for row in rows for phone in row["ph_seq"] if phone not in {"AP", "SP"}
    })
    korean_tokens = korean_phones
    if len(original_phones) + len(korean_tokens) > 3000:
        raise RuntimeError("Korean phones exceed SoulX embedding capacity")
    phones = original_phones + korean_tokens
    phone_ids = {phone: index for index, phone in enumerate(phones)}

    config = load_config(str(SOULX / "soulxsinger/config/soulxsinger.yaml"))
    checkpoint = SOULX / "pretrained_models/SoulX-Singer/model.pt"
    model = SoulXSinger(config).cuda()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    model.train()
    model.cfm_decoder.model.cfg_drop_prob = 0.0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.note_text_encoder.weight
    embedding.requires_grad_(True)
    with torch.no_grad():
        available = set(original_phones)
        for phone, token in zip(korean_phones, korean_tokens):
            embedding[phone_ids[token]].copy_(embedding[phone_ids[source_token(phone, available)]])
    gradient_mask = torch.zeros_like(embedding)
    gradient_mask[[phone_ids[token] for token in korean_tokens]] = 1
    embedding.register_hook(lambda gradient: gradient * gradient_mask)
    optimizer = torch.optim.AdamW([embedding], lr=args.learning_rate, weight_decay=0.0)
    processor = ExactKoreanDataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=str(SOULX / "soulxsinger/utils/phoneme/phone_set.json"),
        device="cuda",
    )
    processor.phone2idx = phone_ids
    training = [row for row in rows if row["split"] == "train"]
    losses = []
    for update in range(args.max_updates):
        row = training[update % len(training)]
        item = processor.process(metadata(row), str(ROOT / row["audio_path"]))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(model, item)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([embedding], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(json.dumps({"update": update + 1, "id": row["id"], "loss": losses[-1]}), flush=True)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "soulx_ko_exact_phone_embedding_v3",
        "soulx_revision": "81aeb3ae772c70093c3de74dc23c92d983801ae4",
        "source_checkpoint_sha256": sha256(checkpoint),
        "phones": phones,
        "korean_phones": korean_phones,
        "korean_tokens": korean_tokens,
        "weights": embedding.detach().cpu()[[phone_ids[token] for token in korean_tokens]],
        "updates": args.max_updates,
        "learning_rate": args.learning_rate,
        "train_rows": len(training),
        "validation_rows": sum(row["split"] == "validation" for row in rows),
        "test_rows": sum(row["split"] == "test" for row in rows),
        "label_status": "inferred_singing_alignment_real_gyu_phrase_reconstruction",
        "losses": losses,
    }
    torch.save(payload, output)
    report = {
        key: value for key, value in payload.items() if key not in {"phones", "weights", "losses"}
    } | {
        "status": "bounded_embedding_probe_trained_not_evaluated",
        "trainable_parameters": len(korean_phones) * embedding.shape[1],
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "checkpoint": str(output.relative_to(ROOT)),
        "checkpoint_sha256": sha256(output),
    }
    target = ROOT / f"artifacts/reports/soulx_ko_phone_adapter_training_{args.max_updates}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
