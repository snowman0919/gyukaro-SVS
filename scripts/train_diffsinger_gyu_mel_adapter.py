#!/usr/bin/env python3
"""Train a bounded GYU mel residual while freezing the DiffSinger foundation.

Real GYU phrase chunks provide the target mel.  The input mel is rendered from
the score-derived nominal F0 and the aligned phoneme timeline, never from the
real target RMVPE F0.  Japanese foundation mels receive an identity/preservation
loss so the adapter cannot improve its target loss by globally destroying the
source acoustic space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import h5py
import numpy as np
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HOP_SECONDS = 512 / 44_100
UNVOICED = {
    "AP", "SP", "c_ja", "h_ja", "k_ja", "p_ja", "s_ja", "t_ja", "ts_ja",
    "tɕ_ja", "ç_ja", "ɕ_ja", "ɸ_ja", "ʔ_ja", "i̥_ja", "ɨ̥_ja", "ɯ̥_ja",
    "ko_onset_1", "ko_onset_2", "ko_onset_3", "ko_onset_5", "ko_onset_6",
    "ko_onset_7", "ko_onset_9", "ko_onset_10", "ko_onset_11", "ko_onset_12",
    "ko_onset_14", "ko_onset_15", "ko_onset_16", "ko_onset_17",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def id_to_phone(dictionary: Path) -> dict[int, str]:
    phones = {"AP", "SP"}
    for line in dictionary.read_text().splitlines():
        if line:
            phones.update(line.split("\t", 1)[1].split())
    return {index + 1: phone for index, phone in enumerate(sorted(phones))}


def recover_names(metadata: dict, handle: h5py.File, chunks: dict[str, dict],
                  old_phones: dict[int, str], split: str) -> list[str]:
    if metadata.get("names") is not None:
        return list(metadata["names"])
    phone_ids = {phone: index for index, phone in old_phones.items()}
    candidates: dict[tuple[int, ...], list[dict]] = {}
    for chunk in chunks.values():
        if chunk["split"] != split:
            continue
        signature = tuple(phone_ids[phone] for phone in chunk["ph_seq"])
        candidates.setdefault(signature, []).append(chunk)
    names = []
    used: set[str] = set()
    for index in range(len(handle)):
        item = handle[str(index)]
        tokens = item["tokens"][()]
        mel2ph = item["mel2ph"][()]
        durations = np.asarray([(mel2ph == token).sum()
                                for token in range(1, len(tokens) + 1)])
        options = [row for row in candidates.get(tuple(int(value) for value in tokens), [])
                   if row["id"] not in used]
        if not options:
            raise ValueError("could not recover stable training-row name from tokens")
        def duration_error(row: dict) -> float:
            accumulated = np.round(np.cumsum(row["ph_dur"]) / HOP_SECONDS + .5).astype(int)
            expected = np.diff(accumulated, prepend=0)
            return float(np.max(np.abs(expected - durations)))
        selected = min(options, key=duration_error)
        used.add(selected["id"])
        names.append(selected["id"])
    return names


def nominal_f0(
    frames: int,
    frame_phones: list[str],
    chunk: dict,
    score: dict,
    aligned_phones: list[dict],
) -> np.ndarray:
    voiced = np.asarray([phone not in UNVOICED for phone in frame_phones])
    phrase_start = min(float(phone["start"]) for phone in aligned_phones)
    phrase_end = max(float(phone["start"]) + float(phone["duration"])
                     for phone in aligned_phones)
    notes = score["score_notes"]
    score_end = max(float(note["start"]) + float(note["duration"]) for note in notes)
    local = (np.arange(frames) + .5) * HOP_SECONDS
    absolute = float(chunk["source_start_seconds"]) + local
    normalized = np.clip((absolute - phrase_start) / max(phrase_end - phrase_start, 1e-6), 0, 1)
    score_time = normalized * max(score_end - 1e-6, 0)
    result = np.zeros(frames, dtype=np.float32)
    for note in notes:
        selected = ((score_time >= float(note["start"]))
                    & (score_time < float(note["start"]) + float(note["duration"])))
        result[selected & voiced] = 440 * 2 ** ((float(note["pitch"]) - 69) / 12)
    return result


class MelAdapter(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hidden: int, limit: float):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.fc1 = nn.Linear(mean.numel(), hidden)
        self.fc2 = nn.Linear(hidden, mean.numel())
        self.limit = limit
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, mel: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.fc1((mel - self.mean) / self.std))
        delta = self.limit * torch.tanh(self.fc2(hidden))
        return mel + strength * delta


def predict_pairs(model_path: Path, binary: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    export = model_path.parent
    model_phones = json.loads(next(export.glob("*.phonemes.json")).read_text())
    old_phones = id_to_phone(binary / "dictionary-gyu.txt")
    chunks = {row["id"]: row for row in jsonl(ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl")}
    scores = {row["id"]: row for row in jsonl(ROOT / "data/manifests/diffsinger_score_native.jsonl")}
    alignments = {row["id"]: row["phones"]
                  for row in jsonl(ROOT / "data/manifests/real_phoneme_alignment_all.jsonl")}
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {value.name for value in session.get_inputs()}
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "valid"):
        metadata_path = binary / f"{split}.meta"
        import pickle
        metadata = pickle.load(metadata_path.open("rb"))
        inputs, targets = [], []
        with h5py.File(binary / f"{split}.data") as handle:
            names = recover_names(metadata, handle, chunks, old_phones, split)
            for index, name in enumerate(names):
                row = handle[str(index)]
                old_tokens = row["tokens"][()]
                phone_sequence = [old_phones[int(token)] for token in old_tokens]
                tokens = np.asarray([[model_phones[phone] for phone in phone_sequence]], dtype=np.int64)
                mel2ph = row["mel2ph"][()]
                durations = np.asarray([[(mel2ph == token).sum()
                                         for token in range(1, len(old_tokens) + 1)]], dtype=np.int64)
                frame_phones = [phone_sequence[max(int(token) - 1, 0)] for token in mel2ph]
                chunk = chunks[name]
                score = scores[chunk["source_id"]]
                f0 = nominal_f0(len(mel2ph), frame_phones, chunk, score,
                                alignments[chunk["source_id"]])[None]
                feed: dict[str, np.ndarray] = {
                    "tokens": tokens, "durations": durations, "f0": f0,
                    "depth": np.asarray(0, dtype=np.float32),
                    "steps": np.asarray(20, dtype=np.int64),
                }
                if "gender" in input_names:
                    feed["gender"] = np.zeros_like(f0)
                if "velocity" in input_names:
                    feed["velocity"] = np.ones_like(f0)
                prediction = session.run(["mel"], feed)[0][0].astype(np.float32)
                target = row["mel"][()].astype(np.float32)
                if prediction.shape != target.shape:
                    raise ValueError(f"shape mismatch for {name}: {prediction.shape} != {target.shape}")
                inputs.append(prediction)
                targets.append(target)
        output[split] = (np.concatenate(inputs), np.concatenate(targets))
    return output


def source_preservation_frames(binary: Path, rows: int = 128) -> np.ndarray:
    with h5py.File(binary / "train.data") as handle:
        indices = np.linspace(0, len(handle) - 1, min(rows, len(handle)), dtype=int)
        return np.concatenate([handle[str(index)]["mel"][()].astype(np.float32)
                               for index in indices])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--limit", type=float, default=.75)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phrase_binary = ROOT / "data/external/work/diffsinger_score_native/binary_gyu_phrase_chunks"
    source_binary = ROOT / "data/external/work/diffsinger_score_native/binary_gtsinger_ja_soprano"
    pairs = predict_pairs(args.model.resolve(), phrase_binary)
    preserve = source_preservation_frames(source_binary)
    train_input, train_target = pairs["train"]
    valid_input, valid_target = pairs["valid"]
    normalizer = np.concatenate((train_input, preserve[:len(train_input)]))
    mean = torch.from_numpy(normalizer.mean(0).astype(np.float32))
    std = torch.from_numpy(np.maximum(normalizer.std(0), .1).astype(np.float32))
    model = MelAdapter(mean, std, args.hidden, args.limit).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_input_t = torch.from_numpy(train_input).to(device)
    train_target_t = torch.from_numpy(train_target).to(device)
    preserve_t = torch.from_numpy(preserve).to(device)
    history = []
    batch = min(2048, len(train_input_t), len(preserve_t))
    for step in range(1, args.steps + 1):
        gyu_index = torch.randint(len(train_input_t), (batch,), device=device)
        source_index = torch.randint(len(preserve_t), (batch,), device=device)
        adapted = model(train_input_t[gyu_index])
        preserved = model(preserve_t[source_index])
        target_loss = torch.nn.functional.l1_loss(adapted, train_target_t[gyu_index])
        preserve_loss = torch.nn.functional.l1_loss(preserved, preserve_t[source_index])
        residual = torch.mean((adapted - train_input_t[gyu_index]) ** 2)
        loss = target_loss + 1.5 * preserve_loss + .05 * residual
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append({
                "step": step,
                "target_l1": round(float(target_loss), 6),
                "source_preservation_l1": round(float(preserve_loss), 6),
                "residual_rms": round(float(residual.sqrt()), 6),
            })

    model.eval()
    with torch.inference_mode():
        valid_input_t = torch.from_numpy(valid_input).to(device)
        valid_target_t = torch.from_numpy(valid_target).to(device)
        valid_adapted = model(valid_input_t)
        validation = {
            "baseline_l1": round(float(torch.nn.functional.l1_loss(valid_input_t, valid_target_t)), 6),
            "adapted_l1": round(float(torch.nn.functional.l1_loss(valid_adapted, valid_target_t)), 6),
            "adapter_delta_rms": round(float(torch.mean((valid_adapted - valid_input_t) ** 2).sqrt()), 6),
            "source_preservation_l1": round(float(torch.nn.functional.l1_loss(
                model(preserve_t), preserve_t)), 6),
        }
    saved = {
        "model": model.cpu().state_dict(),
        "config": {"bins": 128, "hidden": args.hidden, "limit": args.limit},
        "input": "score-nominal-F0 foundation mel",
        "target": "real GYU phrase-chunk mel",
        "target_f0_used_as_condition": False,
        "source_preservation": "GTSinger Japanese source mels",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    report = {
        "status": "trained_objective_evaluation_pending",
        "checkpoint": str(args.output),
        "checkpoint_sha256": digest(args.output),
        "foundation_onnx": str(args.model),
        "foundation_onnx_sha256": digest(args.model),
        "train_phrase_frames": len(train_input),
        "validation_phrase_frames": len(valid_input),
        "source_preservation_frames": len(preserve),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "real_target_f0_condition_leakage": False,
        "score_source": "reconstructed script score time-warped to phoneme phrase bounds",
        "labels": "inferred singing-aware CTC timing",
        "independent_evaluation_song_in_training": False,
        "validation": validation,
        "history": history,
        "release_allowed": False,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
