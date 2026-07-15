#!/usr/bin/env python3
"""Probe SoulX score mode with Korean syllables projected onto its English phones."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
SOULX = ROOT / "data/cache/soulx-singer"
sys.path[:0] = [str(ROOT / "src"), str(SOULX)]

from gyu_singer.frontend import phonemize  # noqa: E402
from gyu_singer.alignment import build_phrase_frames  # noqa: E402
from gyu_singer.inference.content_timing import roman_phone  # noqa: E402
from soulxsinger.models.soulxsinger import SoulXSinger  # noqa: E402
from soulxsinger.utils.data_processor import DataProcessor  # noqa: E402
from soulxsinger.utils.file_utils import load_config  # noqa: E402


PHONE_MAP = {
    "a": "AA1", "ae": "AE1", "b": "B", "ch": "CH", "d": "D",
    "e": "EH1", "eo": "AO1", "eu": "UH1", "g": "G", "h": "HH",
    "i": "IY1", "j": "JH", "k": "K", "l": "L", "m": "M",
    "n": "N", "ng": "NG", "o": "OW1", "p": "P", "pp": "P",
    "r": "R", "s": "S", "ss": "S", "t": "T", "u": "UW1",
    "ui": "IY1", "w": "W", "y": "Y",
}
CASES = ("rapid_ko", "large_interval_ko")


def frame_f0(audio: np.ndarray, rate: int, frame_count: int) -> np.ndarray:
    """Extract prompt evidence on SoulX's 50 Hz grid; silence stays unvoiced."""
    audio = librosa.resample(audio.astype(np.float32), orig_sr=rate, target_sr=24000)
    values, _, _ = librosa.pyin(audio, fmin=60, fmax=900, sr=24000, hop_length=480)
    values = np.nan_to_num(values, nan=0.0).astype(np.float32)
    if len(values) == frame_count:
        return values
    source = np.linspace(0.0, 1.0, max(1, len(values)))
    target = np.linspace(0.0, 1.0, frame_count)
    return np.interp(target, source, values).astype(np.float32)


def english_phone(symbol: str) -> str | None:
    value = roman_phone(symbol)
    return PHONE_MAP.get(value) if value else None


def syllable_phone(lyric: str) -> str:
    phones = [english_phone(symbol) for symbol in phonemize("ko", lyric).symbols]
    phones = [phone for phone in phones if phone]
    if not phones:
        raise ValueError(f"no SoulX phone projection for {lyric!r}")
    return "en_" + "-".join(phones)


def phrase_row() -> dict:
    manifest = ROOT / "data/manifests/diffsinger_gyu_phrase_chunks.jsonl"
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["id"] == "gyu_real_000216_phrase00":
            return row
    raise RuntimeError("GYU prompt phrase missing")


def prompt_metadata(row: dict) -> dict:
    audio, rate = sf.read(ROOT / row["audio_path"], dtype="float32", always_2d=True)
    audio = librosa.resample(audio.mean(1), orig_sr=rate, target_sr=24000)
    f0, _, _ = librosa.pyin(audio, fmin=60, fmax=900, sr=24000, hop_length=480)
    fallback = float(np.nanmedian(f0))

    groups: list[tuple[list[str], float, float]] = []
    phones: list[str] = []
    duration = 0.0
    cursor = 0.0
    for symbol, length in zip(row["ph_seq"], row["ph_dur"]):
        length = float(length)
        if symbol == "SP":
            if phones:
                groups.append((phones, duration, cursor - duration))
                phones, duration = [], 0.0
            groups.append((["<SP>"], length, cursor))
        else:
            phone = english_phone(symbol)
            if phone:
                phones.append(phone)
            duration += length
        cursor += length
    if phones:
        groups.append((phones, duration, cursor - duration))

    values = {"duration": [], "phoneme": [], "note_pitch": [], "note_type": []}
    for phones, duration, start in groups:
        silence = phones == ["<SP>"]
        values["duration"].append(duration)
        values["phoneme"].append("<SP>" if silence else "en_" + "-".join(phones))
        if silence:
            midi = 0
        else:
            begin = max(0, int(start / 0.02))
            end = min(len(f0), max(begin + 1, int((start + duration) / 0.02)))
            hz = float(np.nanmedian(f0[begin:end]))
            if not np.isfinite(hz):
                hz = fallback
            midi = int(round(69 + 12 * math.log2(hz / 440)))
        values["note_pitch"].append(midi)
        values["note_type"].append(1 if silence else 2)

    total = sum(values["duration"])
    prompt_f0 = frame_f0(audio, 24000, int(round(total * 50)))
    return {
        "index": row["id"], "language": "English", "time": [0, round(total * 1000)],
        "duration": " ".join(f"{value:.6f}" for value in values["duration"]),
        "text": row["source_text"],
        "phoneme": " ".join(values["phoneme"]),
        "note_pitch": " ".join(map(str, values["note_pitch"])),
        "note_type": " ".join(map(str, values["note_type"])),
        "f0": " ".join(f"{value:.3f}" for value in prompt_f0),
        "label_status": "inferred_ctc_and_rmvpe_prompt_metadata",
    }


def target_metadata(case: str) -> dict:
    score = json.loads((ROOT / f"examples/review_{case}.json").read_text())
    notes = score["notes"]
    frontend = phonemize("ko", "".join(note["lyric"] for note in notes))
    frames = build_phrase_frames(
        frontend,
        notes,
        score.get("curves", {}).get("pitch", []),
        frame_hz=50,
        phoneme_alignment=score.get("phonemes"),
    )
    result = {
        "index": case, "language": "English",
        "time": [0, round(max(note["start"] + note["duration"] for note in notes) * 1000)],
        "duration": " ".join(f'{note["duration"]:.6f}' for note in notes),
        "text": "".join(note["lyric"] for note in notes),
        "phoneme": " ".join(syllable_phone(note["lyric"]) for note in notes),
        "note_pitch": " ".join(str(note["pitch"]) for note in notes),
        "note_type": " ".join("2" for _ in notes),
        "f0": " ".join(f"{value:.3f}" for value in frames.f0_hz.tolist()),
        "label_status": "independent_stress_score_with_arpabet_projection",
    }
    return result


def verified_prompt(output: Path) -> tuple[dict, Path]:
    selected = None
    for line in (ROOT / "data/manifests/manual_verified_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] == "gyu_real_000192":
            selected = row
            break
    if selected is None:
        raise RuntimeError("verified GYU prompt missing")
    notes = selected["notes"]
    start = notes[0]["start"]
    end = notes[-1]["start"] + notes[-1]["duration"]
    audio, rate = sf.read(ROOT / selected["audio_path"], dtype="float32", always_2d=True)
    audio = audio[int(start * rate):int(end * rate)]
    path = output / "verified_prompt_192.wav"
    sf.write(path, audio, rate, subtype="PCM_24")

    values = {"duration": [], "phoneme": [], "note_pitch": [], "note_type": []}
    cursor = start
    for note in notes:
        gap = note["start"] - cursor
        if gap >= .005:
            values["duration"].append(gap)
            values["phoneme"].append("<SP>")
            values["note_pitch"].append(0)
            values["note_type"].append(1)
        values["duration"].append(note["duration"])
        values["phoneme"].append(syllable_phone(note["lyric"]))
        values["note_pitch"].append(note["pitch"])
        values["note_type"].append(2)
        cursor = note["start"] + note["duration"]
    prompt_f0 = frame_f0(audio.mean(1), rate, int(round((end - start) * 50)))
    return {
        "index": selected["id"], "language": "English", "time": [0, round((end - start) * 1000)],
        "duration": " ".join(f"{value:.6f}" for value in values["duration"]),
        "text": selected["text"],
        "phoneme": " ".join(values["phoneme"]),
        "note_pitch": " ".join(map(str, values["note_pitch"])),
        "note_type": " ".join(map(str, values["note_type"])),
        "f0": " ".join(f"{value:.3f}" for value in prompt_f0),
        "label_status": "independent_verified_score_plus_reviewed_phoneme_alignment",
    }, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", choices=("verified", "inferred"), default="verified")
    parser.add_argument("--control", choices=("score", "melody"), default="score")
    args = parser.parse_args()

    output = ROOT / "artifacts/reports/soulx_score_native_ko_probe"
    listening = output / "listening"
    listening.mkdir(parents=True, exist_ok=True)

    config = load_config(str(SOULX / "soulxsinger/config/soulxsinger.yaml"))
    checkpoint = SOULX / "pretrained_models/SoulX-Singer/model.pt"
    model = SoulXSinger(config).cuda()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.half().eval()
    model.mel.float()
    processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=str(SOULX / "soulxsinger/utils/phoneme/phone_set.json"),
        device="cuda",
    )

    if args.prompt == "verified":
        prompt_meta, prompt_path = verified_prompt(output)
        suffix = "vp"
    else:
        prompt_row = phrase_row()
        prompt_meta = prompt_metadata(prompt_row)
        prompt_path = ROOT / prompt_row["audio_path"]
        suffix = "ip"
    prompt = processor.process(prompt_meta, str(prompt_path))
    (output / f"prompt_metadata_{suffix}.json").write_text(
        json.dumps([prompt_meta], ensure_ascii=False, indent=2) + "\n"
    )

    rows = []
    for case in CASES:
        target_meta = target_metadata(case)
        (output / f"{case}_metadata.json").write_text(
            json.dumps([target_meta], ensure_ascii=False, indent=2) + "\n"
        )
        target = processor.process(target_meta)
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            audio = model.infer(
                {"prompt": prompt, "target": target},
                auto_shift=False, pitch_shift=0, n_steps=args.steps,
                cfg=args.cfg, control=args.control, use_fp16=True,
            )
        path = listening / f"{case}_{args.control}_s{args.steps}_c{args.cfg:g}_{suffix}.wav"
        sf.write(path, audio.squeeze().float().cpu().numpy(), 24000)
        rows.append({"case": case, "path": str(path.relative_to(ROOT))})

    report = {
        "status": "rendered_not_evaluated",
        "model": "SoulX-Singer score-conditioned model.pt",
        "soulx_revision": "81aeb3ae772c70093c3de74dc23c92d983801ae4",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "prompt": str(prompt_path.relative_to(ROOT)),
        "prompt_label_status": prompt_meta["label_status"],
        "korean_phone_projection": "deterministic approximation onto trained English ARPAbet; not native Korean supervision",
        "score_native": True,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "control": args.control,
        "steps": args.steps, "cfg": args.cfg, "seed": args.seed,
        "rows": rows,
    }
    (output / f"render_{args.control}_{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
