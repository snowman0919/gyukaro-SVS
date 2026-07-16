#!/usr/bin/env python3
"""Prepare the CC BY-SA 4.0 PJS Japanese singing source probe."""
from __future__ import annotations

import csv
import copy
import json
import pickle
from pathlib import Path

import soundfile as sf
import yaml

from prepare_diffsinger_gyu_segments import remap_vocabulary, sha256


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/external/work/diffsinger_score_native"
PJS = ROOT / "data/external/work/pjs/PJS_corpus_ver1.1"
LABELS = ROOT / "data/external/work/pjs/manual-labels/lab"


def main() -> None:
    raw = WORK / "raw/pjs"
    wavs = raw / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    phones, seconds = set(), 0.0
    with (raw / "transcriptions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "ph_seq", "ph_dur"))
        writer.writeheader()
        for index in range(1, 101):
            name = f"pjs{index:03d}"
            source = PJS / name / f"{name}_song.wav"
            rows = [line.split() for line in (LABELS / f"{name}.lab").read_text().splitlines()]
            sequence = ["SP" if row[2] == "pau" else "AP" if row[2] == "xx" else f"ja_{row[2]}" for row in rows]
            if index == 1:
                sequence[0] = "AP"
            durations = [(int(row[1]) - int(row[0])) / 10_000_000 for row in rows]
            audio_duration = sf.info(source).duration
            if audio_duration - sum(durations) > .001:
                sequence.append("SP")
                durations.append(audio_duration - sum(durations))
            assert abs(sum(durations) - audio_duration) < .001
            target = wavs / f"{name}.wav"
            if not target.exists():
                target.symlink_to(source)
            writer.writerow({"name": name, "ph_seq": " ".join(sequence),
                             "ph_dur": " ".join(f"{value:.7f}" for value in durations)})
            phones.update(sequence)
            seconds += audio_duration

    base = yaml.safe_load((ROOT / "configs/diffsinger_score_native.yaml").read_text())
    old_dictionary = Path(base["dictionaries"]["gyu"])
    dictionary = WORK / "dictionary-pjs.txt"
    dictionary.write_text("".join(f"{phone}\t{phone}\n" for phone in sorted(phones - {"AP", "SP"})))
    source_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_4000.ckpt"
    target_checkpoint = ROOT / "data/cache/diffsinger/checkpoints/pjs_source_vocab.ckpt"
    remap = remap_vocabulary(source_checkpoint, old_dictionary, dictionary, target_checkpoint)
    assert remap["shared_embedding_max_abs_error"] == 0

    base.update({
        "dictionaries": {"gyu": str(dictionary)},
        "datasets": [{"raw_data_dir": str(raw), "speaker": "pjs", "spk_id": 0,
                      "language": "gyu", "test_prefixes": [f"pjs{index:03d}" for index in range(91, 101)]}],
        "binary_data_dir": str(WORK / "binary_pjs"),
        "finetune_enabled": True, "finetune_ckpt_path": str(target_checkpoint),
        "finetune_ignored_params": [], "finetune_strict_shapes": True,
        "freezing_enabled": True,
        "frozen_params": ["model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
                          "model.fs2.dur_embed", "model.fs2.pitch_embed"],
        "max_updates": 1000, "val_check_interval": 100, "num_valid_plots": 0,
        "val_with_vocoder": False, "max_batch_frames": 20000, "max_batch_size": 8,
        "num_ckpt_keep": 3, "optimizer_args": {"lr": 2e-5},
    })
    config = ROOT / "configs/diffsinger_pjs_source.yaml"
    config.write_text(yaml.safe_dump(base, sort_keys=False))

    rapid = copy.deepcopy(base)
    rapid.update({
        "binary_data_dir": str(WORK / "binary_pjs_rapid"),
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_source/model_ckpt_steps_1000.ckpt"
        ),
        "finetune_strict_shapes": False,
        "use_speed_embed": True,
        "max_updates": 1000,
        "val_check_interval": 200,
        "num_ckpt_keep": 3,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": False, "range": [-5.0, 5.0], "scale": .75},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": .5},
            "random_time_stretching": {"enabled": True, "range": [.95, 4.5], "scale": 3.0},
        },
    })
    rapid_config = ROOT / "configs/diffsinger_pjs_rapid.yaml"
    rapid_config.write_text(yaml.safe_dump(rapid, sort_keys=False))

    lexical = copy.deepcopy(base)
    lexical.update({
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_source/model_ckpt_steps_1000.ckpt"
        ),
        "frozen_params": [
            "model.diffusion", "model.fs2.stretch_embed", "model.fs2.stretch_embed_rnn",
            "model.fs2.dur_embed", "model.fs2.pitch_embed", "model.fs2.spk_embed",
        ],
        "shallow_diffusion_args": {"train_aux_decoder": True, "train_diffusion": False},
        "lambda_aux_mel_loss": 1.0,
        "max_updates": 2000,
        "val_check_interval": 200,
        "num_ckpt_keep": 4,
        "optimizer_args": {"lr": 1e-4},
    })
    lexical_config = ROOT / "configs/diffsinger_pjs_lexical.yaml"
    lexical_config.write_text(yaml.safe_dump(lexical, sort_keys=False))

    compact = copy.deepcopy(base)
    compact.update({
        "finetune_enabled": False,
        "finetune_ckpt_path": None,
        "freezing_enabled": True,
        "frozen_params": ["model.diffusion"],
        "num_spk": 1,
        "hidden_size": 192,
        "num_heads": 2,
        "enc_layers": 4,
        "backbone_args": {
            "num_channels": 256, "num_layers": 4, "kernel_size": 15,
            "dropout_rate": 0.0, "use_conditioner_cache": True, "glu_type": "atanglu",
        },
        "shallow_diffusion_args": {
            "train_aux_decoder": True, "train_diffusion": False,
            "aux_decoder_arch": "convnext", "aux_decoder_grad": .1,
            "aux_decoder_args": {
                "num_channels": 256, "num_layers": 4, "kernel_size": 7, "dropout_rate": .1,
            },
        },
        "lambda_aux_mel_loss": 1.0,
        "max_updates": 5000,
        "val_check_interval": 500,
        "num_ckpt_keep": 3,
        "optimizer_args": {"lr": 3e-4},
    })
    compact_config = ROOT / "configs/diffsinger_pjs_compact.yaml"
    compact_config.write_text(yaml.safe_dump(compact, sort_keys=False))

    compact_stress = copy.deepcopy(compact)
    compact_stress.update({
        "binary_data_dir": str(WORK / "binary_pjs_compact_stress"),
        "finetune_enabled": True,
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_compact/model_ckpt_steps_3000.ckpt"
        ),
        "finetune_strict_shapes": False,
        "use_key_shift_embed": True,
        "use_speed_embed": True,
        "augmentation_args": {
            "random_pitch_shifting": {"enabled": True, "range": [-2.0, 12.0], "scale": 2.0},
            "fixed_pitch_shifting": {"enabled": False, "targets": [-5.0, 5.0], "scale": .5},
            "random_time_stretching": {"enabled": True, "range": [.95, 4.5], "scale": 3.0},
        },
        "max_updates": 3000,
        "val_check_interval": 500,
        "optimizer_args": {"lr": 1e-4},
    })
    compact_stress_config = ROOT / "configs/diffsinger_pjs_compact_stress.yaml"
    compact_stress_config.write_text(yaml.safe_dump(compact_stress, sort_keys=False))

    consonant = copy.deepcopy(compact_stress)
    consonant.update({
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_compact_stress/model_ckpt_steps_3000.ckpt"
        ),
        "task_cls": "diffsinger_consonant_task.ConsonantWeightedAcousticTask",
        "consonant_loss_weight": 5.0,
        "max_updates": 2000,
        "val_check_interval": 500,
        "optimizer_args": {"lr": 5e-5},
    })
    consonant_config = ROOT / "configs/diffsinger_pjs_consonant.yaml"
    consonant_config.write_text(yaml.safe_dump(consonant, sort_keys=False))

    compact_long = copy.deepcopy(compact_stress)
    compact_long.update({
        "finetune_ckpt_path": str(
            ROOT / "data/cache/diffsinger/checkpoints/pjs_compact_stress/model_ckpt_steps_3000.ckpt"
        ),
        "max_updates": 5000,
        "val_check_interval": 500,
        "optimizer_args": {"lr": 5e-5},
    })
    compact_long_config = ROOT / "configs/diffsinger_pjs_compact_long.yaml"
    compact_long_config.write_text(yaml.safe_dump(compact_long, sort_keys=False))
    report = {
        "status": "ready_for_binarization", "rows": 100, "train_rows": 90,
        "validation_rows": 10, "duration_minutes": round(seconds / 60, 3),
        "audio_sample_rate": 48_000, "license": "CC BY-SA 4.0",
        "source_archive_sha256": sha256(ROOT / "data/external/work/pjs/pjs_corpus_ver1.1.zip"),
        "manual_labels_revision": "cc08bead6bf2b06e88608a8ece12555bcc720ec9",
        "diffsinger_revision": "753b7cc622aadf802b3145d7bb8f7df4afa213c4",
        "vocabulary_remap": remap, "config": str(config.relative_to(ROOT)),
        "rapid_augmentation": {
            "config": str(rapid_config.relative_to(ROOT)),
            "source_checkpoint_step": 1000,
            "time_stretch_range": [.95, 4.5],
            "augmentation_scale": 3.0,
            "speed_conditioning": True,
            "purpose": "bounded probe for measured 20-59 ms rapid Japanese phoneme transitions",
        },
        "lexical_focus": {
            "config": str(lexical_config.relative_to(ROOT)),
            "source_checkpoint_step": 1000,
            "trainable_path": "text embedding + phoneme encoder + auxiliary mel decoder",
            "frozen_path": "diffusion + pitch/duration/stretch/speaker conditioning",
            "purpose": "measured phoneme-collapse probe after speed/pitch/step isolation failed",
        },
        "compact_source": {
            "config": str(compact_config.relative_to(ROOT)),
            "trainable_path": "compact phoneme/F0/duration encoder + auxiliary mel decoder",
            "frozen_path": "untrained diffusion stage during lexical admission",
            "purpose": "PJS-scale source model after the 69.1M transfer path failed lexical admission",
        },
        "compact_stress": {
            "config": str(compact_stress_config.relative_to(ROOT)),
            "source_checkpoint_step": 3000,
            "pitch_shift_range_semitones": [-2.0, 12.0],
            "time_stretch_range": [.95, 4.5],
            "purpose": "measured high-pitch, rapid-transition, and repeated-phrase coverage gaps",
        },
        "consonant_focus": {
            "config": str(consonant_config.relative_to(ROOT)),
            "source_checkpoint_step": 3000,
            "consonant_frame_l1_weight": 5.0,
            "purpose": "measured Japanese consonant deletion after pitch/voicing recovery",
        },
        "compact_long": {
            "config": str(compact_long_config.relative_to(ROOT)),
            "source_checkpoint_step": 3000,
            "additional_update_limit": 5000,
            "purpose": "bounded rapid-OOD continuation after held-out normal Japanese ASR passed 0.5116",
        },
        "role": "Japanese lexical score-native source probe; not GYU and not a release model",
    }
    binary = Path(base["binary_data_dir"])
    if (binary / "train.meta").is_file() and (binary / "valid.meta").is_file():
        train = pickle.load((binary / "train.meta").open("rb"))
        valid = pickle.load((binary / "valid.meta").open("rb"))
        report.update({"status": "binarization_pass", "binarized_train_rows": len(train["lengths"]),
                       "binarized_validation_rows": len(valid["lengths"])})
    output = ROOT / "artifacts/reports/diffsinger_pjs_source.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
