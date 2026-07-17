#!/usr/bin/env python3
"""Prepare native DiffSinger speaker-conditioned LayerNorm adaptation."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity_init.ckpt"
EXPERIMENT = "gtsinger_ja_gyu_mixln"
MIX_LAYERS = (0, 2)


def convert_layer_norms(state: dict[str, torch.Tensor]) -> int:
    parameters = 0
    for layer in MIX_LAYERS:
        for norm in ("layer_norm1", "layer_norm2"):
            prefix = f"model.fs2.encoder.layers.{layer}.op.{norm}"
            weight = state.pop(f"{prefix}.weight")
            bias = state.pop(f"{prefix}.bias")
            state[f"{prefix}.affine.weight"] = weight.new_zeros((weight.numel() * 2, weight.numel()))
            state[f"{prefix}.affine.bias"] = torch.cat((bias, weight))
            parameters += state[f"{prefix}.affine.weight"].numel() + weight.numel() * 2
    return parameters


def main() -> None:
    phrase_rows = [json.loads(line) for line in
                   (ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl").read_text().splitlines()]
    source_config = yaml.safe_load((ROOT / "configs/diffsinger_gtsinger_ja.yaml").read_text())
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    speakers = state["model.fs2.spk_embed.weight"]
    expanded = speakers[0].repeat(21, 1)
    expanded[20] = speakers[1]
    state["model.fs2.spk_embed.weight"] = expanded
    mixln_parameters = convert_layer_norms(state)
    initial = ROOT / f"data/cache/diffsinger/checkpoints/{EXPERIMENT}_init.ckpt"
    torch.save(checkpoint, initial)

    frozen = [
        "model.fs2.txt_embed", "model.fs2.dur_embed", "model.fs2.pitch_embed",
        "model.fs2.key_shift_embed", "model.fs2.speed_embed", "model.fs2.stretch_embed",
        "model.fs2.stretch_embed_rnn", "model.fs2.encoder.layer_norm",
        "model.aux_decoder", "model.diffusion",
    ]
    for layer in range(6):
        prefix = f"model.fs2.encoder.layers.{layer}.op"
        frozen.extend((f"{prefix}.self_attn", f"{prefix}.ffn"))
        if layer not in MIX_LAYERS:
            frozen.extend((f"{prefix}.layer_norm1", f"{prefix}.layer_norm2"))

    source_dataset = source_config["datasets"][0]
    config = yaml.safe_load(
        (ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity_v2/config.yaml").read_text()
    )
    config.update({
        "dictionaries": {"gyu": str(WORK / "dictionary-gtsinger-ja-gyu.txt")},
        "datasets": [source_dataset, {
            "raw_data_dir": str(WORK / "raw/gyu_phrase_chunks"),
            "speaker": "gyu", "spk_id": 20, "language": "gyu",
            "test_prefixes": [row["id"] for row in phrase_rows if row["split"] != "train"],
        }],
        "binary_data_dir": str(WORK / f"binary_{EXPERIMENT}"),
        "num_spk": 21,
        "use_mix_ln": True,
        "mix_ln_layer": list(MIX_LAYERS),
        "finetune_ckpt_path": str(initial),
        "freezing_enabled": True,
        "frozen_params": frozen,
        "max_updates": 600,
        "val_check_interval": 200,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "max_batch_frames": 20_000,
        "max_batch_size": 8,
        "num_ckpt_keep": 3,
        "optimizer_args": {
            "optimizer_cls": "torch.optim.AdamW", "lr": 1e-5,
            "betas": [.9, .98], "weight_decay": 0,
        },
    })
    config_path = ROOT / f"configs/diffsinger_{EXPERIMENT}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "ready_for_binarization",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "initial_checkpoint": str(initial.relative_to(ROOT)),
        "real_gyu_phrase_rows": len(phrase_rows),
        "real_gyu_duration_minutes": round(sum(row["duration_seconds"] for row in phrase_rows) / 60, 3),
        "japanese_replay_dataset": source_dataset["raw_data_dir"],
        "mixln_layers": list(MIX_LAYERS),
        "mixln_parameters": mixln_parameters,
        "speaker_embedding_parameters": expanded.numel(),
        "labels": "GYU labels inferred singing-aware; Japanese replay preserves the lexical backbone",
        "source_recordings_modified": False,
        "release_allowed": False,
        "config": str(config_path.relative_to(ROOT)),
    }
    output = ROOT / f"artifacts/reports/diffsinger_{EXPERIMENT}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
