#!/usr/bin/env python3
"""Fit a GYU speaker embedding without changing the Japanese singing backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from prepare_diffsinger_gyu_segments import dictionary_tokens, remap_vocabulary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
SOURCE = ROOT / "data/cache/diffsinger/checkpoints/gtsinger_ja_source/model_ckpt_steps_15000.ckpt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand_speaker(checkpoint: Path) -> tuple[int, int]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    key = "model.fs2.spk_embed.weight"
    weight = saved["state_dict"][key]
    if weight.shape[0] != 1:
        raise ValueError(f"expected one source speaker, got {weight.shape[0]}")
    saved["state_dict"][key] = torch.cat((weight, weight.clone()), dim=0)
    torch.save(saved, checkpoint)
    return tuple(saved["state_dict"][key].shape)


def restore_shared_token_rows(
    source: Path,
    source_dictionary: Path,
    adapted: Path,
    adapted_dictionary: Path,
    target: Path,
    tokens: tuple[str, ...] = ("AP", "SP"),
) -> dict:
    """Restore language-shared silence rows after single-language adaptation."""
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    adapted_payload = torch.load(adapted, map_location="cpu", weights_only=False)
    key = "model.fs2.txt_embed.weight"
    source_embedding = source_payload["state_dict"][key]
    adapted_embedding = adapted_payload["state_dict"][key]
    source_ids = {
        token: index + 1 for index, token in enumerate(dictionary_tokens(source_dictionary))
    }
    adapted_ids = {
        token: index + 1 for index, token in enumerate(dictionary_tokens(adapted_dictionary))
    }
    before = []
    for token in tokens:
        if token not in source_ids or token not in adapted_ids:
            raise ValueError(f"shared token missing from dictionary: {token}")
        source_row = source_embedding[source_ids[token]]
        adapted_row = adapted_embedding[adapted_ids[token]]
        before.append(float((adapted_row - source_row).abs().max()))
        adapted_row.copy_(source_row)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapted_payload, target)
    after = [
        float((adapted_embedding[adapted_ids[token]]
               - source_embedding[source_ids[token]]).abs().max())
        for token in tokens
    ]
    return {
        "restored_tokens": list(tokens),
        "max_abs_error_before": max(before),
        "max_abs_error_after": max(after),
    }


def build_strict_identity_checkpoint(
    initial: Path,
    adapted: Path,
    dictionary: Path,
    target: Path,
    identity_speaker_id: int = 1,
    token_prefix: str = "ko_",
) -> dict:
    """Whitelist only target-language text rows and the target speaker row."""
    initial_payload = torch.load(initial, map_location="cpu", weights_only=False)
    adapted_payload = torch.load(adapted, map_location="cpu", weights_only=False)
    initial_state = initial_payload["state_dict"]
    adapted_state = adapted_payload["state_dict"]
    text_key = "model.fs2.txt_embed.weight"
    speaker_key = "model.fs2.spk_embed.weight"
    if initial_state.keys() != adapted_state.keys():
        raise ValueError("initial and adapted checkpoint keys differ")
    unexpected = []
    for key in initial_state:
        if initial_state[key].shape != adapted_state[key].shape:
            raise ValueError(f"checkpoint tensor shape differs: {key}")
        if key not in {text_key, speaker_key} and not torch.equal(
            initial_state[key], adapted_state[key]
        ):
            unexpected.append(key)
    tokens = dictionary_tokens(dictionary)
    token_ids = {token: index + 1 for index, token in enumerate(tokens)}
    copied_tokens = sorted(token for token in tokens if token.startswith(token_prefix))
    for token in copied_tokens:
        index = token_ids[token]
        initial_state[text_key][index].copy_(adapted_state[text_key][index])
    initial_state[speaker_key][identity_speaker_id].copy_(
        adapted_state[speaker_key][identity_speaker_id]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(initial_payload, target)
    return {
        "copied_text_tokens": copied_tokens,
        "copied_speaker_row": identity_speaker_id,
        "unexpected_changed_tensors_in_adapted": sorted(unexpected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--source-config", type=Path,
                        default=ROOT / "configs/diffsinger_gtsinger_ja.yaml")
    parser.add_argument("--label", default="gtsinger_ja_gyu_identity")
    args = parser.parse_args()
    source = args.source.resolve()
    source_config = args.source_config.resolve()

    japanese = WORK / "dictionary-gtsinger-ja.txt"
    korean = WORK / "dictionary-gyu-all-adapt.txt"
    combined = WORK / "dictionary-gtsinger-ja-gyu.txt"
    tokens = sorted({line.split("\t", 1)[0] for path in (japanese, korean)
                     for line in path.read_text().splitlines() if line})
    combined.write_text("".join(f"{token}\t{token}\n" for token in tokens))

    checkpoint = ROOT / f"data/cache/diffsinger/checkpoints/{args.label}_init.ckpt"
    vocabulary = remap_vocabulary(source, japanese, combined, checkpoint)
    speaker_shape = expand_speaker(checkpoint)

    manifest = [json.loads(line) for line in
                (ROOT / "data/manifests/diffsinger_gyu_all_segmented.jsonl").read_text().splitlines()]
    config = yaml.safe_load(source_config.read_text())
    config.update({
        "allow_unused_phonemes": True,
        "dictionaries": {"gyu": str(combined)},
        "datasets": [{
            "raw_data_dir": str(WORK / "raw/gyu_all_segmented"),
            "speaker": "gyu", "spk_id": 1, "language": "gyu",
            "test_prefixes": [row["id"] for row in manifest if row["split"] != "train"],
        }],
        "binary_data_dir": str(WORK / "binary_gtsinger_ja_gyu_identity"),
        "num_spk": 2,
        "finetune_enabled": True,
        "finetune_ckpt_path": str(checkpoint),
        "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": [
            "model.fs2.dur_embed", "model.fs2.encoder", "model.fs2.pitch_embed",
            "model.fs2.key_shift_embed", "model.fs2.speed_embed",
            "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.aux_decoder", "model.diffusion",
        ],
        "max_updates": 300,
        "val_check_interval": 100,
        "num_valid_plots": 0,
        "val_with_vocoder": False,
        "num_ckpt_keep": 3,
        "optimizer_args": {
            "optimizer_cls": "torch.optim.AdamW", "lr": 5e-4,
            "betas": [.9, .98], "weight_decay": 0,
        },
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": False, "range": [-2.0, 2.0], "scale": 1.0},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": .5},
            "random_time_stretching": {"enabled": False, "range": [.95, 1.05], "scale": 1.0},
        },
    })
    config_path = ROOT / f"configs/diffsinger_{args.label}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    report = {
        "status": "ready_for_binarization",
        "method": "frozen Japanese singing backbone; train GYU speaker and Korean text embeddings only",
        "source_checkpoint": str(source.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(source),
        "initial_checkpoint": str(checkpoint.relative_to(ROOT)),
        "initial_checkpoint_sha256": sha256(checkpoint),
        "speaker_embedding_shape": speaker_shape,
        "text_embedding_shape": list(state["model.fs2.txt_embed.weight"].shape),
        "trainable_parameters": int(state["model.fs2.txt_embed.weight"].numel()
                                    + state["model.fs2.spk_embed.weight"].numel()),
        "real_gyu_segments": len(manifest),
        "source_recordings_modified": False,
        "labels": "inferred singing-aware phoneme alignment",
        "config": str(config_path.relative_to(ROOT)),
        "decision_gate": "Japanese free Whisper + waveform/F0 + GYU speaker similarity + human listening",
        "release_allowed": False,
        "vocabulary": vocabulary,
    }
    output = ROOT / f"artifacts/reports/diffsinger_{args.label}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
