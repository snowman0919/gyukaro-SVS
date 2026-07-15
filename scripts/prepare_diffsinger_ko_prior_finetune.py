#!/usr/bin/env python3
"""Remap the 4k pilot vocabulary and configure a text-path-only adaptation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt"
TARGET = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000_ko_vocab.ckpt"


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
    old_tokens = tokens(WORK / "dictionary-gyu.txt")
    new_tokens = tokens(WORK / "dictionary-gyu-ko-prior.txt")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    key = "model.fs2.txt_embed.weight"
    old_weight = state[key]
    assert old_weight.shape[0] == len(old_tokens) + 1
    new_weight = old_weight.new_empty((len(new_tokens) + 1, old_weight.shape[1]))
    new_weight[0] = old_weight[0]
    old_ids = {token: index + 1 for index, token in enumerate(old_tokens)}
    new_ids = {token: index + 1 for index, token in enumerate(new_tokens)}
    analogs = {
        "ko_coda_17": ["ko_onset_7"],
        "ko_coda_2": ["ko_onset_1"],
        "ko_nucleus_11": ["ko_nucleus_10", "ko_nucleus_1"],
        "ko_nucleus_17": ["ko_nucleus_13", "ko_nucleus_5"],
        "ko_onset_10": ["ko_onset_9"],
        "ko_onset_15": ["ko_onset_0"],
    }
    for token, new_id in new_ids.items():
        if token in old_ids:
            new_weight[new_id] = old_weight[old_ids[token]]
        else:
            source_ids = [old_ids[value] for value in analogs[token]]
            new_weight[new_id] = old_weight[source_ids].mean(dim=0)
    state[key] = new_weight
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "category": checkpoint.get("category")}, TARGET)

    config_path = ROOT / "configs/diffsinger_ko_phoneme_prior.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.update({
        "finetune_enabled": True,
        "finetune_ckpt_path": str(TARGET),
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.fs2.dur_embed", "model.fs2.pitch_embed", "model.fs2.spk_embed",
            "model.aux_decoder", "model.diffusion",
        ],
        "max_updates": 500,
        "val_check_interval": 100,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "max_batch_frames": 20_000,
        "max_batch_size": 8,
        "num_ckpt_keep": 2,
    })
    config.setdefault("optimizer_args", {})["lr"] = 0.0001
    output_config = ROOT / "configs/diffsinger_ko_prior_finetune.yaml"
    output_config.write_text(yaml.safe_dump(config, sort_keys=False))

    preserved = max(
        float((new_weight[new_ids[token]] - old_weight[old_ids[token]]).abs().max())
        for token in old_ids.keys() & new_ids.keys()
    )
    report = {
        "status": "vocabulary_remapped_text_path_finetune_ready",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(SOURCE),
        "remapped_checkpoint": str(TARGET.relative_to(ROOT)),
        "remapped_checkpoint_sha256": sha256(TARGET),
        "old_vocabulary": len(old_tokens) + 1,
        "new_vocabulary": len(new_tokens) + 1,
        "shared_embedding_max_abs_error": preserved,
        "new_token_initialization": analogs,
        "trainable_scope": ["model.fs2.txt_embed", "model.fs2.encoder"],
        "frozen_scope": config["frozen_params"],
        "max_updates": config["max_updates"],
        "learning_rate": config["optimizer_args"]["lr"],
        "config": str(output_config.relative_to(ROOT)),
    }
    path = ROOT / "artifacts/reports/diffsinger_ko_prior_finetune.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
