#!/usr/bin/env python3
"""Remap the 4k singing pilot for bounded Zeroth-Korean acoustic adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="zeroth")
    parser.add_argument("--max-updates", type=int, default=300)
    parser.add_argument("--val-check-interval", type=int, default=100)
    args = parser.parse_args()
    target = ROOT / (
        f"data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000_{args.label}_vocab.ckpt"
    )
    text_target = ROOT / (
        f"data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000_{args.label}_text_vocab.ckpt"
    )
    old_tokens = tokens(WORK / "dictionary-gyu.txt")
    new_tokens = tokens(WORK / f"dictionary-gyu-{args.label}.txt")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]

    text_key = "model.fs2.txt_embed.weight"
    old_text = state[text_key]
    assert old_text.shape[0] == len(old_tokens) + 1
    new_text = old_text.new_empty((len(new_tokens) + 1, old_text.shape[1]))
    new_text[0] = old_text[0]
    old_ids = {token: index + 1 for index, token in enumerate(old_tokens)}
    new_ids = {token: index + 1 for index, token in enumerate(new_tokens)}
    initialization = {}
    for token, new_id in new_ids.items():
        if token in old_ids:
            new_text[new_id] = old_text[old_ids[token]]
            continue
        category = token.rsplit("_", 1)[0]
        sources = [value for value in old_tokens if value.startswith(category + "_")]
        assert sources, token
        new_text[new_id] = old_text[[old_ids[value] for value in sources]].mean(dim=0)
        initialization[token] = f"mean:{category}"
    state[text_key] = new_text

    speaker_key = "model.fs2.spk_embed.weight"
    old_speakers = state[speaker_key]
    assert old_speakers.shape[0] == 21
    new_speakers = old_speakers.new_empty((25, old_speakers.shape[1]))
    new_speakers[:21] = old_speakers
    new_speakers[21:] = old_speakers[:20].mean(dim=0)
    state[speaker_key] = new_speakers

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "category": checkpoint.get("category")}, target)

    text_tokens = tokens(WORK / f"dictionary-gyu-{args.label}-text.txt")
    text_state = state.copy()
    text_state[text_key] = torch.stack([
        new_text[0], *[new_text[new_ids[token]] for token in text_tokens]
    ])
    torch.save({"state_dict": text_state, "category": checkpoint.get("category")}, text_target)

    config = yaml.safe_load((ROOT / f"configs/diffsinger_{args.label}_prior.yaml").read_text())
    config.update({
        "finetune_enabled": True,
        "finetune_ckpt_path": str(target),
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.fs2.dur_embed", "model.fs2.pitch_embed",
        ],
        "max_updates": args.max_updates,
        "val_check_interval": args.val_check_interval,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "max_batch_frames": 20_000,
        "max_batch_size": 8,
        "num_ckpt_keep": 3,
    })
    config.setdefault("optimizer_args", {})["lr"] = 0.00005
    output_config = ROOT / f"configs/diffsinger_{args.label}_finetune.yaml"
    output_config.write_text(yaml.safe_dump(config, sort_keys=False))

    text_config = yaml.safe_load(
        (ROOT / f"configs/diffsinger_{args.label}_text_prior.yaml").read_text()
    )
    text_config.update({
        "finetune_enabled": True,
        "finetune_ckpt_path": str(text_target),
        "finetune_ignored_params": [],
        "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.fs2.dur_embed", "model.fs2.pitch_embed", "model.fs2.spk_embed",
            "model.aux_decoder", "model.diffusion",
        ],
        "max_updates": 600,
        "val_check_interval": 200,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "max_batch_frames": 20_000,
        "max_batch_size": 8,
        "num_ckpt_keep": 3,
    })
    text_config.setdefault("optimizer_args", {})["lr"] = 0.0001
    text_output = ROOT / f"configs/diffsinger_{args.label}_text_finetune.yaml"
    text_output.write_text(yaml.safe_dump(text_config, sort_keys=False))

    shared_error = max(
        float((new_text[new_ids[token]] - old_text[old_ids[token]]).abs().max())
        for token in old_ids.keys() & new_ids.keys()
    )
    speaker_error = float((new_speakers[:21] - old_speakers).abs().max())
    report = {
        "status": "zeroth_acoustic_finetune_ready",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(SOURCE),
        "remapped_checkpoint": str(target.relative_to(ROOT)),
        "remapped_checkpoint_sha256": sha256(target),
        "text_remapped_checkpoint": str(text_target.relative_to(ROOT)),
        "text_remapped_checkpoint_sha256": sha256(text_target),
        "old_vocabulary": len(old_tokens) + 1,
        "new_vocabulary": len(new_tokens) + 1,
        "text_only_vocabulary": len(text_tokens) + 1,
        "shared_embedding_max_abs_error": shared_error,
        "existing_speaker_embedding_max_abs_error": speaker_error,
        "new_token_initialization": initialization,
        "new_speaker_initialization": "mean of 20 generic VocalSet singer embeddings; trainable",
        "frozen_score_control": config["frozen_params"],
        "trainable_acoustic_path": ["text encoder", "speaker embedding", "aux decoder", "diffusion"],
        "max_updates": config["max_updates"],
        "learning_rate": config["optimizer_args"]["lr"],
        "config": str(output_config.relative_to(ROOT)),
        "text_only_config": str(text_output.relative_to(ROOT)),
    }
    path = ROOT / f"artifacts/reports/diffsinger_{args.label}_finetune.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert shared_error == 0 and speaker_error == 0
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
