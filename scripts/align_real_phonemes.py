#!/usr/bin/env python3
"""MMS CTC-force-align confirmed Korean GYU phrases, then apply singing timing priors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from gyu_singer.frontend import phonemize

ONSET = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
VOWEL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
CODA = ["", "g", "kk", "gs", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h"]


def roman_syllables(text: str) -> tuple[str, list[int], list[str]]:
    target, ends, units = "", [], []
    for char in text:
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172: continue
        onset, rest = divmod(code, 588); vowel, coda = divmod(rest, 28)
        target += ONSET[onset] + VOWEL[vowel] + CODA[coda]; ends.append(len(target)); units.append(char)
    return target, ends, units


def align(row: dict, model: torch.nn.Module, dictionary: dict[str, int], labels: tuple[str, ...], device: str) -> dict:
    target, ends, units = roman_syllables(row["text"])
    audio, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True); audio = audio.mean(axis=1)
    if rate != 16000: audio = resample_poly(audio, 16000, rate).astype("float32")
    waveform = torch.from_numpy(audio)[None].to(device)
    with torch.inference_mode(): emission, _ = model(waveform)
    alignment, scores = torchaudio.functional.forced_align(emission.log_softmax(-1), torch.tensor([[dictionary[char] for char in target]], device=device))
    spans = torchaudio.functional.merge_tokens(alignment[0], scores[0]); seconds = len(audio) / 16000 / len(alignment[0])
    character_spans = [(span.start * seconds, span.end * seconds) for span in spans]
    phones, previous, phone_index = [], 0, 0
    for syllable_index, (unit, end) in enumerate(zip(units, ends)):
        group = character_spans[previous:end]; previous = end
        start, finish = group[0][0], group[-1][1]; duration = max(.04, finish - start)
        output = phonemize("ko", unit)
        onset = min(.10, duration * .25) if len(output.symbols) > 1 else 0
        coda = min(.15, duration * .25) if len(output.symbols) == 3 else 0
        nucleus = max(.02, duration - onset - coda)
        for index, symbol in enumerate(output.symbols):
            part_duration = onset if index == 0 and onset else coda if index == len(output.symbols) - 1 and coda else nucleus
            part_start = start if index == 0 else start + onset if index == 1 else start + onset + nucleus
            phones.append({"phoneme_index": phone_index, "symbol": symbol, "syllable_index": syllable_index, "start": round(part_start, 4), "duration": round(part_duration, 4), "source": "mms_ctc_plus_singing_vowel_prior"}); phone_index += 1
    return {"id": row["id"], "alignment_source": "MMS_multilingual_CTC_forced_alignment_plus_singing_vowel_prior", "phones": phones}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"; bundle = torchaudio.pipelines.MMS_FA
    labels, model = bundle.get_labels(), bundle.get_model().to(device).eval(); dictionary = {label: index for index, label in enumerate(labels)}
    rows = [json.loads(line) for line in Path("data/manifests/neural_supervision.jsonl").read_text().splitlines() if line]
    output = Path("data/manifests/real_phoneme_alignment.jsonl")
    output.write_text("".join(json.dumps(align(row, model, dictionary, labels, device), ensure_ascii=False) + "\n" for row in rows))
    print({"rows": len(rows), "model": "torchaudio.pipelines.MMS_FA"})


if __name__ == "__main__": main()
