#!/usr/bin/env python3
"""Prepare low-rate real-GYU singing adaptation from the Korean acoustic prior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_replay/model_ckpt_steps_300.ckpt"
TARGET = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_gyu_adapt_vocab.ckpt"


def tokens(path: Path) -> list[str]:
    values = {"AP", "SP"}
    values.update(line.split("\t", 1)[0] for line in path.read_text().splitlines() if line)
    return sorted(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    old_dictionary = WORK / "dictionary-gyu-zeroth.txt"
    new_dictionary = WORK / "dictionary-gyu-all-adapt.txt"
    old_tokens, new_tokens = tokens(old_dictionary), tokens(new_dictionary)
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]

    text_key = "model.fs2.txt_embed.weight"
    old_text = state[text_key]
    old_ids = {token: index + 1 for index, token in enumerate(old_tokens)}
    new_ids = {token: index + 1 for index, token in enumerate(new_tokens)}
    missing = sorted(set(new_tokens) - set(old_tokens))
    if missing:
        raise RuntimeError(f"GYU adaptation dictionary has unseen tokens: {missing}")
    new_text = torch.stack([old_text[0], *[old_text[old_ids[token]] for token in new_tokens]])
    state[text_key] = new_text

    speaker_key = "model.fs2.spk_embed.weight"
    old_speakers = state[speaker_key]
    if old_speakers.shape[0] != 25:
        raise RuntimeError(f"expected 25 prior speakers, found {old_speakers.shape[0]}")
    state[speaker_key] = old_speakers[:21].clone()

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "category": checkpoint.get("category")}, TARGET)

    config = yaml.safe_load((ROOT / "configs/diffsinger_gyu_all_adapt.yaml").read_text())
    config["finetune_ckpt_path"] = str(TARGET)
    config["max_updates"] = 300
    config["val_check_interval"] = 100
    config["optimizer_args"]["lr"] = 2e-6
    output = ROOT / "configs/diffsinger_zeroth_gyu_adapt.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))

    report = {
        "status": "two_stage_gyu_singing_adaptation_ready",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "checkpoint": str(TARGET.relative_to(ROOT)),
        "checkpoint_sha256": sha256(TARGET),
        "shared_text_embedding_max_abs_error": max(
            float((new_text[new_ids[token]] - old_text[old_ids[token]]).abs().max())
            for token in new_tokens
        ),
        "gyu_speaker_embedding_max_abs_error": float(
            (state[speaker_key][20] - old_speakers[20]).abs().max()
        ),
        "dropped_speakers": ["zeroth_187", "zeroth_191", "zeroth_201", "zeroth_214"],
        "training_data": "730 inferred-timing real-GYU contiguous singing segments",
        "learning_rate": config["optimizer_args"]["lr"],
        "max_updates": config["max_updates"],
        "config": str(output.relative_to(ROOT)),
    }
    (ROOT / "artifacts/reports/diffsinger_zeroth_gyu_adapt.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
