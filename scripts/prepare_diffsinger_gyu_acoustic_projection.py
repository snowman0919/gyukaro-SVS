#!/usr/bin/env python3
"""Prepare a bounded GYU adaptation of DiffSinger's final acoustic projection."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity_v2/model_ckpt_steps_300.ckpt"
EXPERIMENT = "gtsinger_ja_gyu_acoustic_projection"


def main() -> None:
    rows = [json.loads(line) for line in
            (ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl").read_text().splitlines()]
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    key = "model.fs2.spk_embed.weight"
    speakers = checkpoint["state_dict"][key]
    expanded = speakers[0].repeat(21, 1)
    expanded[20] = speakers[1]
    checkpoint["state_dict"][key] = expanded
    initial = ROOT / f"data/cache/diffsinger/checkpoints/{EXPERIMENT}_init.ckpt"
    torch.save(checkpoint, initial)

    config = yaml.safe_load(
        (ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_gyu_identity_v2/config.yaml").read_text()
    )
    config.update({
        "dictionaries": {"gyu": str(WORK / "dictionary-gtsinger-ja-gyu.txt")},
        "datasets": [{
            "raw_data_dir": str(WORK / "raw/gyu_phrase_chunks"),
            "speaker": "gyu", "spk_id": 20, "language": "gyu",
            "test_prefixes": [row["id"] for row in rows if row["split"] != "train"],
        }],
        "binary_data_dir": str(WORK / f"binary_{EXPERIMENT}"),
        "num_spk": 21,
        "finetune_ckpt_path": str(initial),
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.txt_embed", "model.fs2.dur_embed", "model.fs2.encoder",
            "model.fs2.pitch_embed", "model.fs2.key_shift_embed", "model.fs2.speed_embed",
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.aux_decoder.decoder.inconv", "model.aux_decoder.decoder.conv",
            "model.diffusion",
        ],
        "max_updates": 300,
        "val_check_interval": 100,
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
    outconv = sum(value.numel() for name, value in checkpoint["state_dict"].items()
                  if name.startswith("model.aux_decoder.decoder.outconv"))
    report = {
        "status": "ready_for_binarization",
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "initial_checkpoint": str(initial.relative_to(ROOT)),
        "real_gyu_phrase_rows": len(rows),
        "real_gyu_duration_minutes": round(sum(row["duration_seconds"] for row in rows) / 60, 3),
        "trainable_acoustic_projection_parameters": outconv,
        "speaker_embedding_row": 20,
        "labels": "inferred singing-aware phoneme alignment; target F0 is never an inference condition",
        "source_recordings_modified": False,
        "japanese_encoder_frozen": True,
        "release_allowed": False,
        "config": str(config_path.relative_to(ROOT)),
    }
    output = ROOT / f"artifacts/reports/diffsinger_{EXPERIMENT}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
