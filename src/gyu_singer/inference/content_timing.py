"""CTC-guided phrase content timing correction for duration-locked TTS sources."""
from __future__ import annotations

import numpy as np
import torch

from gyu_singer.alignment import build_phrase_frames
from gyu_singer.frontend import phonemize


_KO_ONSET = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_KO_VOWEL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
_KO_CODA = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


class CTCAlignmentUnavailable(RuntimeError):
    """The generated phrase has too few CTC frames for the score phones."""


def roman_phone(symbol: str) -> str:
    if symbol.startswith("ko_onset_"):
        return _KO_ONSET[int(symbol.rsplit("_", 1)[1])]
    if symbol.startswith("ko_nucleus_"):
        return _KO_VOWEL[int(symbol.rsplit("_", 1)[1])]
    if symbol.startswith("ko_coda_"):
        return _KO_CODA[int(symbol.rsplit("_", 1)[1])]
    if symbol.startswith(("en_", "ja_")):
        return symbol[3:].replace("long", "")
    return ""


def score_phone_targets(score: dict, frame_hz: float = 50) -> tuple[list[dict], str]:
    text = " ".join(note["lyric"] for note in score["notes"])
    frames = build_phrase_frames(phonemize(score["language"], text), score["notes"], score["curves"]["pitch"], frame_hz=frame_hz, phoneme_alignment=score.get("phonemes"))
    targets, characters = [], ""
    for phone in sorted(frames.phoneme_durations, key=lambda row: row["start_frame"]):
        token = "".join(char for char in roman_phone(phone["symbol"]) if "a" <= char <= "z")
        if not token:
            continue
        targets.append(phone | {"ctc_token": token, "target_start": phone["start_frame"] / frame_hz, "target_end": (phone["start_frame"] + phone["duration_frames"]) / frame_hz})
        characters += token
    return targets, characters


def ctc_phone_alignment(audio: torch.Tensor, sample_rate: int, score: dict, model: torch.nn.Module, labels: tuple[str, ...]) -> dict:
    targets, characters = score_phone_targets(score)
    dictionary = {label: index for index, label in enumerate(labels)}
    missing = sorted(set(characters) - set(dictionary))
    if missing:
        raise ValueError(f"MMS CTC target has unsupported characters: {missing}")
    with torch.inference_mode():
        emission, _ = model(audio[None])
    token_ids = torch.tensor([[dictionary[char] for char in characters]], device=emission.device)
    repeated = sum(before == after for before, after in zip(characters, characters[1:]))
    required_frames = len(characters) + repeated
    if required_frames > emission.shape[1]:
        raise CTCAlignmentUnavailable(
            f"MMS CTC needs at least {required_frames} frames for {len(characters)} targets, "
            f"but generated content has {emission.shape[1]}"
        )
    import torchaudio
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), token_ids)
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
    seconds = audio.numel() / sample_rate / emission.shape[1]
    cursor, phones = 0, []
    for target in targets:
        count = len(target["ctc_token"]); group = spans[cursor:cursor + count]; cursor += count
        if len(group) != count:
            raise RuntimeError("MMS CTC returned incomplete target alignment")
        phones.append(target | {"source_start": group[0].start * seconds, "source_end": group[-1].end * seconds, "ctc_mean_log_score": float(np.mean([span.score for span in group]))})
    return {"method": "MMS_FA_character_CTC_plus_frontend_phone_grouping", "characters": characters, "phones": phones,
            "mean_log_score": float(np.mean([row["ctc_mean_log_score"] for row in phones]))}


def wsola_to_phone_timing(audio: np.ndarray, sample_rate: int, duration: float, alignment: dict) -> np.ndarray:
    """Pitch-preserving sample-domain WSOLA; this is not note pitch control."""
    output_length = max(1, round(duration * sample_rate)); frame, hop, search = 2048, 512, 384
    source_times = [0.0] + [(row["source_start"] + row["source_end"]) / 2 for row in alignment["phones"]] + [len(audio) / sample_rate]
    target_times = [0.0] + [(row["target_start"] + row["target_end"]) / 2 for row in alignment["phones"]] + [duration]
    pairs = [(s, t) for s, t in zip(source_times, target_times) if 0 <= s <= len(audio) / sample_rate and 0 <= t <= duration]
    source_times, target_times = map(np.asarray, zip(*pairs))
    keep = np.r_[True, (np.diff(source_times) > 1e-4) & (np.diff(target_times) > 1e-4)]
    source_times, target_times = source_times[keep], target_times[keep]
    padded = np.pad(audio.astype(np.float32), frame)
    output, weight = np.zeros(output_length + frame, np.float32), np.zeros(output_length + frame, np.float32)
    window = np.hanning(frame).astype(np.float32); previous = None
    for target_start in range(0, output_length, hop):
        mapped = np.interp((target_start + frame / 2) / sample_rate, target_times, source_times) * sample_rate
        expected = int(round(mapped - frame / 2)) + frame
        candidates = range(max(0, expected - search), min(len(padded) - frame, expected + search) + 1, 32)
        selected = expected
        if previous is not None:
            reference = previous[hop:]
            denominator_ref = np.linalg.norm(reference) + 1e-8
            selected = max(candidates, key=lambda start: float(np.dot(reference, padded[start:start + frame - hop]) / (denominator_ref * (np.linalg.norm(padded[start:start + frame - hop]) + 1e-8))))
        grain = padded[selected:selected + frame]
        end = min(len(output), target_start + frame); length = end - target_start
        output[target_start:end] += grain[:length] * window[:length]; weight[target_start:end] += window[:length]
        previous = grain
    return (output[:output_length] / np.maximum(weight[:output_length], 1e-4)).astype(np.float32)


def latent_content_warp(alignment: dict, source_duration: float, target_duration: float, frames: int) -> np.ndarray:
    """Map score-time hidden frames to CTC-aligned source hidden positions."""
    phones = alignment["phones"]
    target = np.array([(row["target_start"] + row["target_end"]) / 2 for row in phones])
    source = np.array([(row["source_start"] + row["source_end"]) / 2 for row in phones])
    target = np.r_[0.0, target, target_duration]
    source = np.r_[max(0.0, source[0] - target[1]), source, min(source_duration, source[-1] + target_duration - target[-2])]
    mapped = np.interp(np.arange(frames) / 50, target, source)
    return np.clip(mapped / max(source_duration, 1e-6), 0, 1).astype(np.float32)


def latent_content_hold(alignment: dict, source_duration: float, frames: int) -> np.ndarray:
    """Hold a stable CTC phone-center hidden through each score phone window."""
    mapped = np.full(frames, np.nan, np.float32)
    for row in alignment["phones"]:
        start, end = round(row["target_start"] * 50), max(round(row["target_start"] * 50) + 1, round(row["target_end"] * 50))
        mapped[start:min(frames, end)] = (row["source_start"] + row["source_end"]) / 2 / max(source_duration, 1e-6)
    known = np.flatnonzero(np.isfinite(mapped))
    if not len(known):
        raise ValueError("CTC alignment contains no phone windows")
    mapped[:known[0]] = mapped[known[0]]; mapped[known[-1] + 1:] = mapped[known[-1]]
    missing = np.flatnonzero(~np.isfinite(mapped)); mapped[missing] = mapped[known[np.searchsorted(known, missing, side="right").clip(max=len(known) - 1)]]
    return np.clip(mapped, 0, 1)


def ctc_voicing_mask(alignment: dict, duration: float, frames: int) -> np.ndarray:
    """Classify the generated content grid from CTC phone centers."""
    phones = alignment["phones"]; centers = np.array([(row["source_start"] + row["source_end"]) / 2 for row in phones]); bounds = np.r_[centers[0], (centers[:-1] + centers[1:]) / 2, centers[-1]]
    mask = np.zeros(frames, np.float32)
    for index, row in enumerate(phones):
        start, end = round(bounds[index] * 50), max(round(bounds[index] * 50) + 1, round(bounds[index + 1] * 50))
        mask[max(0, start):min(frames, end)] = float(row.get("voicing") in {"vowel", "voiced_consonant"})
    return mask
