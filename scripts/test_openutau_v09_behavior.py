#!/usr/bin/env python3
"""Behavioral OpenUtau-contract test against the resident v0.8 renderer."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache"))
sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor


def load_bridge():
    spec = importlib.util.spec_from_file_location("gyu_openutau_bridge", "integrations/openutau/bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(url: str, score: dict, path: Path) -> None:
    request = urllib.request.Request(url.rstrip("/") + "/render", data=json.dumps(score).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    path.write_bytes(urllib.request.urlopen(request, timeout=600).read())


def audio16(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    return resample_poly(mono, 16000, rate).astype(np.float32) if rate != 16000 else mono


def transcript(path: Path, language: str, processor, model) -> str:
    inputs = processor(audio16(path), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(inputs.input_features, language=language, task="transcribe", max_new_tokens=64)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def normalized(text: str) -> str:
    return re.sub(r"[^a-zA-Z가-힣ぁ-んァ-ン一-龯]", "", text).lower()


def paired_pitch_shift(left: np.ndarray, right: np.ndarray) -> float:
    size = min(len(left), len(right))
    voiced = (left[:size] > 0) & (right[:size] > 0)
    return float(np.median(1200 * np.log2(right[:size][voiced] / left[:size][voiced]))) if voiced.any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-url", default="http://127.0.0.1:8765")
    parser.add_argument("--project", default="examples/openutau_v09.ustx")
    parser.add_argument("--output", default="artifacts/reports/openutau_v09/behavior.json")
    parser.add_argument("--existing-audio-dir", type=Path)
    args = parser.parse_args()
    output = Path(args.output); audio_dir = output.parent / "outputs"; audio_dir.mkdir(parents=True, exist_ok=True)
    bridge = load_bridge()
    base = bridge.ustx_score(args.project, "ko", 0)
    en = bridge.ustx_score(args.project, "en", 1)
    ja = bridge.ustx_score(args.project, "ja", 2)
    note_edit = deepcopy(base)
    for note in note_edit["notes"]: note["pitch"] += 2
    lyric_edit = deepcopy(base)
    for note, lyric in zip(lyric_edit["notes"], ("사랑을", "담아", "전해줘", "내마음")): note["lyric"] = lyric
    user_pitch = deepcopy(base)
    user_pitch["curves"]["pitch"] = [{"time": 0.0, "value": 1.0}, {"time": max(note["start"] + note["duration"] for note in base["notes"]), "value": 1.0}]
    energetic = deepcopy(base); energetic["style"] = {"preset": "energetic"}
    cases = {"ko": base, "en": en, "ja": ja, "note_pitch_edit": note_edit, "lyric_edit": lyric_edit,
             "user_pitch_edit": user_pitch, "style_energetic": energetic}
    paths = {name: (args.existing_audio_dir or audio_dir) / f"{name}.wav" for name in cases}
    if not args.existing_audio_dir:
        for name, score in cases.items(): render(args.render_url, score, paths[name])

    f0 = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
                     device="cpu", target_sr=24000, hop_size=480, verbose=False)
    contours = {name: f0.process(str(path), verbose=False) for name, path in paths.items()}
    note_shift = paired_pitch_shift(contours["ko"], contours["note_pitch_edit"])
    user_shift = paired_pitch_shift(contours["ko"], contours["user_pitch_edit"])
    rms = {}
    formats = {}
    for name, path in paths.items():
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        rms[name] = float(np.sqrt(np.mean(audio**2)))
        formats[name] = {"sample_rate": rate, "channels": audio.shape[1], "seconds": round(len(audio) / rate, 4)}
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float32).eval()
    base_transcript = transcript(paths["ko"], "ko", processor, asr)
    edited_transcript = transcript(paths["lyric_edit"], "ko", processor, asr)
    base_audio, edited_audio = audio16(paths["ko"]), audio16(paths["lyric_edit"])
    size = min(len(base_audio), len(edited_audio))
    lyric_waveform_difference = float(np.sqrt(np.mean((base_audio[:size] - edited_audio[:size]) ** 2)))
    sample_rate = formats["ko"]["sample_rate"] if "ko" in formats else 48000
    gates = {
        "ko_en_ja_expected_format": all(value["sample_rate"] == sample_rate and value["channels"] == 1 for value in formats.values()),
        "note_edit_changes_pitch": np.isfinite(note_shift) and note_shift > 100,
        "user_pitch_curve_changes_f0": np.isfinite(user_shift) and user_shift > 40,
        "lyric_edit_changes_content": normalized(base_transcript) != normalized(edited_transcript) and lyric_waveform_difference > .005,
        "energetic_style_changes_audio": abs(rms["style_energetic"] - rms["ko"]) > .001,
    }
    report = {
        "protocol": "actual OpenUtau GyuSingerRenderer.Render audio" if args.existing_audio_dir else "Native OpenUtau mapping is compiled/tested separately; these requests reproduce its phrase payload against the resident service.",
        "renderer_url": args.render_url, "project": args.project, "formats": formats,
        "note_edit_pitch_shift_cents": round(note_shift, 2), "user_pitch_curve_shift_cents": round(user_shift, 2),
        "base_transcript": base_transcript, "lyric_edit_transcript": edited_transcript,
        "lyric_waveform_rms_difference": round(lyric_waveform_difference, 6),
        "neutral_rms": round(rms["ko"], 6), "energetic_rms": round(rms["style_energetic"], 6),
        "gates": {key: bool(value) for key, value in gates.items()}, "pass": all(gates.values()),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
