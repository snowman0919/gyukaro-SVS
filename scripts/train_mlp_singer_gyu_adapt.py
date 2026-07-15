#!/usr/bin/env python3
"""Bounded GYU timbre adaptation of the official Korean score-native model."""
from __future__ import annotations

import hashlib
import json
import random
import sys
import argparse
from pathlib import Path

import mido
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "data/cache/mlp-singer"
sys.path.insert(0, str(MODEL_ROOT))

from data.preprocess import Preprocessor  # noqa: E402
from model import MLPSinger  # noqa: E402
from utils import AttrDict  # noqa: E402


SEQUENCE_LENGTH = 192
NEW_MIN_NOTE = 36
NEW_NUM_PITCH = 49
SAVE_STEPS = (100, 200, 400)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_midi(path: Path, notes: list[dict]) -> None:
    ticks_per_beat = 1000
    tempo = 1_000_000
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    events = []
    ordered = sorted(notes, key=lambda note: float(note["start"]))
    for index, note in enumerate(ordered):
        end = float(note["start"]) + float(note["duration"])
        if index + 1 < len(ordered):
            end = min(end, float(ordered[index + 1]["start"]))
        events.extend([
            (float(note["start"]), 1, mido.Message("note_on", note=note["pitch"], velocity=100)),
            (end, 0, mido.Message("note_off", note=note["pitch"], velocity=0)),
        ])
    cursor = 0.0
    for seconds, _, message in sorted(events):
        message.time = round(mido.second2tick(seconds - cursor, ticks_per_beat, tempo))
        track.append(message)
        cursor = seconds
    midi.save(path)


def prepare_data(preprocessor: Preprocessor) -> tuple[list[tuple], list[tuple], dict]:
    rows = read_jsonl(ROOT / "data/manifests/diffsinger_score_native.jsonl")
    selected = [row for row in rows if row["split"] in {"train", "validation"}]
    if any(row["split"] == "independent_evaluation" for row in selected):
        raise RuntimeError("independent verified scores leaked into adaptation")
    raw = MODEL_ROOT / "data/gyu_adapt"
    for folder in (raw / "mid", raw / "txt"):
        folder.mkdir(parents=True, exist_ok=True)
    tensors = {"train": [], "validation": []}
    for row in selected:
        identifier = row["id"]
        midi_path = raw / "mid" / f"{identifier}.mid"
        text_path = raw / "txt" / f"{identifier}.txt"
        write_midi(midi_path, row["score_notes"])
        text_path.write_text("".join(note["lyric"] for note in row["score_notes"]) + "\n")
        tensor = preprocessor(midi_path, text_path, ROOT / row["audio_path"])
        if len(tensor[0]) < SEQUENCE_LENGTH:
            continue
        tensors[row["split"]].append(tensor)
    stats = {
        "train_rows": len(tensors["train"]),
        "validation_rows": len(tensors["validation"]),
        "training_score_status": "inferred_from_target_f0",
        "independent_verified_rows_used": 0,
        "source_audio_modified": False,
    }
    return tensors["train"], tensors["validation"], stats


def remap_model() -> tuple[MLPSinger, dict]:
    source_config = json.loads((MODEL_ROOT / "checkpoints/default/config.json").read_text())
    source_config["model"]["num_pitch"] = NEW_NUM_PITCH
    source_config["model"]["dropout"] = 0.1
    model = MLPSinger(AttrDict(source_config["model"]))
    checkpoint = torch.load(
        MODEL_ROOT / "checkpoints/default/model.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["model"]
    old = state["pitch_embed.weight"]
    expanded = torch.empty(NEW_NUM_PITCH, old.shape[1])
    expanded[0] = old[0]
    for index in range(1, NEW_NUM_PITCH):
        midi = NEW_MIN_NOTE + index
        source_index = min(max(midi, 53), 77) - 52
        expanded[index] = old[source_index]
    state["pitch_embed.weight"] = expanded
    model.load_state_dict(state)
    return model, source_config


def sample_batch(rows: list[tuple], batch_size: int, generator: random.Random) -> tuple:
    batch = [[], [], []]
    for _ in range(batch_size):
        notes, phonemes, mel = generator.choice(rows)
        start = generator.randrange(0, len(notes) - SEQUENCE_LENGTH + 1)
        for output, tensor in zip(batch, (notes, phonemes, mel)):
            output.append(tensor[start:start + SEQUENCE_LENGTH])
    return tuple(torch.stack(values).cuda() for values in batch)


@torch.no_grad()
def validate(model: MLPSinger, rows: list[tuple]) -> float:
    model.eval()
    losses = []
    for notes, phonemes, mel in rows:
        starts = list(range(0, len(notes) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH))
        if starts[-1] != len(notes) - SEQUENCE_LENGTH:
            starts.append(len(notes) - SEQUENCE_LENGTH)
        for start in starts:
            pred = model(
                notes[start:start + SEQUENCE_LENGTH].unsqueeze(0).cuda(),
                phonemes[start:start + SEQUENCE_LENGTH].unsqueeze(0).cuda(),
            )
            losses.append(F.l1_loss(pred, mel[start:start + SEQUENCE_LENGTH].unsqueeze(0).cuda()).item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "projection"), default="projection")
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    preprocess_config = AttrDict(
        json.loads((ROOT / "configs/mlp_singer_preprocess_gyu.json").read_text())
    )
    train_rows, validation_rows, data_stats = prepare_data(Preprocessor(preprocess_config))
    model, model_config = remap_model()
    model.cuda()
    if args.mode == "projection":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.proj.parameters():
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 5e-5 if args.mode == "projection" else 1e-5
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    generator = random.Random(42)
    output = MODEL_ROOT / f"checkpoints/gyu_adapt_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    history = []
    for step in range(1, max(SAVE_STEPS) + 1):
        if args.mode == "projection":
            model.eval()
            model.proj.train()
        else:
            model.train()
        notes, phonemes, mel = sample_batch(train_rows, 16, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(notes, phonemes), mel)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step in SAVE_STEPS:
            validation_loss = validate(model, validation_rows)
            checkpoint = output / f"steps_{step}.pt"
            torch.save({"model": model.state_dict(), "step": step}, checkpoint)
            history.append({
                "step": step,
                "train_loss": round(float(loss), 6),
                "validation_loss": round(validation_loss, 6),
                "checkpoint_sha256": sha256(checkpoint),
            })
            print(history[-1], flush=True)
    report = {
        "status": "bounded_adaptation_complete_objective_evaluation_pending",
        "base_model": "neosapience/mlp-singer@7f4621ca04ee5e35c0e0a80b1fed785a55a51891",
        "base_model_license": "MIT",
        "score_native": True,
        "speaker_target": "GYU",
        "data": data_stats,
        "pitch_embedding": {"min_note": NEW_MIN_NOTE, "num_pitch": NEW_NUM_PITCH},
        "adaptation_mode": args.mode,
        "training": {
            "batch_size": 16,
            "learning_rate": learning_rate,
            "steps": max(SAVE_STEPS),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        "history": history,
        "production_integrated": False,
    }
    target = ROOT / f"artifacts/reports/mlp_singer_gyu_adapt_{args.mode}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
