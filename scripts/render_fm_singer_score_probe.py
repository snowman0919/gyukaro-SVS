#!/usr/bin/env python3
"""Bounded FM-Singer Korean score-native source probe.

This is evaluation-only while the pretrained checkpoint/data redistribution
terms remain unresolved.  It never mutates the production renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "data/cache/fm-singer"
MODEL_ROOT = VENDOR / "FMSinger"
CONFIG = MODEL_ROOT / "config.json"
CHECKPOINT = ROOT / "data/cache/fm-singer-weights/G_70000.pth"
OUTPUT = ROOT / "artifacts/reports/fm_singer_score_probe"

PAD = " "
PUNCTUATION = ",.!?-~…_"
CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
SYMBOLS = [PAD] + list(PUNCTUATION) + list(CHOSEONG) + list(JUNGSEONG) + [x + "_E" for x in JONGSEONG]
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, HParams(**value) if isinstance(value, dict) else value)


def decompose_hangul(char: str) -> tuple[str, str, str]:
    value = ord(char) - 0xAC00
    if not 0 <= value < 11172:
        raise ValueError(f"FM-Singer probe accepts pre-normalized Hangul syllables only: {char!r}")
    onset = CHOSEONG[value // 588]
    vowel = JUNGSEONG[(value % 588) // 28]
    coda_index = value % 28
    coda = "_" if coda_index == 0 else JONGSEONG[coda_index - 1] + "_E"
    return onset, vowel, coda


def score_tensors(
    score: dict, frame_rate: float, pitch_offset: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    phone_ids: list[int] = []
    pitches: list[int] = []
    duration_frames: list[int] = []
    for note in score["notes"]:
        lyric = str(note["lyric"])
        if len(lyric) != 1:
            raise ValueError(f"one Hangul syllable per note required, got {lyric!r}")
        phones = decompose_hangul(lyric)
        total = max(7, round(float(note["duration"]) * frame_rate))
        durations = [3, total - 6, 3]
        phone_ids.extend(SYMBOL_TO_ID[p] for p in phones)
        pitches.extend([int(note["pitch"]) + pitch_offset] * 3)
        duration_frames.extend(durations)
    seconds = np.asarray(duration_frames, dtype=np.float32) / frame_rate
    return (
        torch.tensor(phone_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(pitches, dtype=torch.long).unsqueeze(0),
        torch.from_numpy(seconds).unsqueeze(0),
        duration_frames,
    )


def load_model(device: torch.device):
    # The official inference path imports the training-only Cython MAS module.
    # Inference never calls it, so keep a fail-closed stub instead of patching
    # vendored source or requiring a compiler in the production project.
    mas = types.ModuleType("monotonic_align")
    mas.maximum_path = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("training-only monotonic alignment was invoked during inference")
    )
    sys.modules["monotonic_align"] = mas
    sys.path[:0] = [str(VENDOR), str(MODEL_ROOT)]
    from models import SynthesizerTrn

    hps = HParams(**json.loads(CONFIG.read_text()))
    model = SynthesizerTrn(hps)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device).eval()
    return model, hps, {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


@contextmanager
def exact_duration_override(model, frames: list[int], hps):
    original = model.duration_predictor.forward
    exact = torch.tensor(frames, dtype=torch.float32)

    def forward(_x, x_mask, spk_emb=None):
        values = exact.to(device=x_mask.device, dtype=x_mask.dtype).view(1, 1, -1)
        # Subtract a tiny epsilon so the official ceil() recovers the exact
        # integer instead of N+1 after log/exp floating-point roundoff.
        seconds = (values - 1e-4) * hps.data.hop_size / hps.data.sample_rate
        return torch.log1p(seconds) * x_mask

    model.duration_predictor.forward = forward
    try:
        yield
    finally:
        model.duration_predictor.forward = original


@contextmanager
def f0_multiplier_override(model, multiplier: float):
    """Correct the checkpoint's documented dataset pitch convention internally."""
    original = model.f0_decoder.forward

    def forward(*args, **kwargs):
        mel_f0, mask = original(*args, **kwargs)
        hz = (torch.pow(10.0, torch.clamp_min(mel_f0, 0) * 500.0 / 2595.0) - 1.0) * 700.0
        corrected = 2595.0 * torch.log10(1.0 + hz * multiplier / 700.0) / 500.0
        return corrected, mask

    model.f0_decoder.forward = forward
    try:
        yield
    finally:
        model.f0_decoder.forward = original


def scale_phone_durations(predicted: list[int], requested: list[int]) -> list[int]:
    """Preserve learned within-syllable timing while enforcing note totals."""
    if len(predicted) != len(requested) or len(predicted) % 3:
        raise ValueError("phone durations must contain three entries per syllable")
    result: list[int] = []
    for offset in range(0, len(predicted), 3):
        source = np.asarray(predicted[offset : offset + 3], dtype=np.float64)
        target_total = sum(requested[offset : offset + 3])
        raw = source / max(source.sum(), 1.0) * target_total
        values = np.maximum(1, np.floor(raw).astype(np.int64))
        while values.sum() < target_total:
            index = int(np.argmax(raw - values))
            values[index] += 1
        while values.sum() > target_total:
            choices = np.where(values > 1, values - raw, -np.inf)
            values[int(np.argmax(choices))] -= 1
        result.extend(int(value) for value in values)
    return result


def render(
    model,
    hps,
    score: dict,
    speaker_id: int,
    duration_override: list[int] | None,
    seed: int,
    pitch_offset: int = 0,
    f0_multiplier: float = 1.0,
) -> tuple[np.ndarray, list[int], list[int]]:
    frame_rate = hps.data.sample_rate / hps.data.hop_size
    phone, pitch, duration, frames = score_tensors(score, frame_rate, pitch_offset)
    device = next(model.parameters()).device
    phone, pitch, duration = phone.to(device), pitch.to(device), duration.to(device)
    lengths = torch.tensor([phone.shape[1]], dtype=torch.long, device=device)
    speaker = torch.tensor([speaker_id], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    duration_context = (
        exact_duration_override(model, duration_override, hps)
        if duration_override is not None
        else contextmanager(lambda: (yield))()
    )
    f0_context = (
        f0_multiplier_override(model, f0_multiplier)
        if not math.isclose(f0_multiplier, 1.0)
        else contextmanager(lambda: (yield))()
    )
    with duration_context, f0_context, torch.inference_mode():
        audio, _harm, _noise, _prior, predicted = model.infer(phone, lengths, pitch, duration, speaker)
    return audio[0, 0].float().cpu().numpy(), predicted[0].long().cpu().tolist(), frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", type=int, default=14, help="AMS14 in the official config")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--pitch-offsets", type=int, nargs="+", default=[0, 12])
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, hps, load_report = load_model(device)
    speaker_names = {value: key.lower() for key, value in hps.speaker_elf.__dict__.items()}
    speaker_label = speaker_names.get(args.speaker_id, f"speaker{args.speaker_id}")
    listening = OUTPUT / "listening"
    listening.mkdir(parents=True, exist_ok=True)
    cases = {
        "rapid_ko": ROOT / "examples/review_rapid_ko.json",
        "large_interval_ko": ROOT / "examples/review_large_interval_ko.json",
    }
    rows = []
    for case, path in cases.items():
        score = json.loads(path.read_text())
        for pitch_offset in args.pitch_offsets:
            _, _, _, requested_phone_frames = score_tensors(
                score, hps.data.sample_rate / hps.data.hop_size, pitch_offset
            )
            for f0_multiplier in ([1.0, 2.0] if pitch_offset == 0 else [1.0]):
                if args.exact_only:
                    official_audio, official_frames = None, []
                    modes = [("exact_score_duration", requested_phone_frames)]
                else:
                    official_audio, official_frames, _ = render(
                        model, hps, score, args.speaker_id, None, args.seed,
                        pitch_offset, f0_multiplier,
                    )
                    scaled_frames = scale_phone_durations(official_frames, requested_phone_frames)
                    modes = [
                        ("official_predicted_duration", None),
                        ("exact_score_duration", requested_phone_frames),
                        ("score_scaled_prediction", scaled_frames),
                    ]
                for variant, override in modes:
                    if override is None:
                        audio, predicted_frames = official_audio, official_frames
                    else:
                        audio, predicted_frames, _ = render(
                            model, hps, score, args.speaker_id, override, args.seed,
                            pitch_offset, f0_multiplier,
                        )
                    suffix = "" if pitch_offset == 0 else f"_p{pitch_offset:+d}".replace("+", "")
                    if not math.isclose(f0_multiplier, 1.0):
                        suffix += f"_f0x{f0_multiplier:g}"
                    output_path = listening / f"{case}_{variant}_{speaker_label}{suffix}.wav"
                    sf.write(output_path, np.clip(audio, -1.0, 1.0), hps.data.sample_rate, subtype="PCM_16")
                    rows.append({
                        "case": case,
                        "variant": variant,
                        "pitch_offset_semitones": pitch_offset,
                        "internal_f0_multiplier": f0_multiplier,
                        "path": str(output_path.relative_to(ROOT)),
                        "samples": len(audio),
                        "duration_seconds": len(audio) / hps.data.sample_rate,
                        "requested_frames": sum(requested_phone_frames),
                        "decoder_frames": sum(predicted_frames),
                        "predicted_phone_frames": predicted_frames,
                    })
    report = {
        "status": "evaluation_only_license_unresolved",
        "model": "FM-Singer",
        "repository_revision": "7245cca4d0a43280f2c4a3aab8a17ed75ba89529",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": "b7b01c46c89de19022d89dbb9084031534684d9a43a8f087aff681877e233427",
        "speaker_id": args.speaker_id,
        "speaker_label": speaker_label,
        "seed": args.seed,
        "load": load_report,
        "rows": rows,
    }
    (OUTPUT / "render.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
