#!/usr/bin/env python3
"""Run local Whisper large-v3-turbo over every GYU source recording."""
from __future__ import annotations

import json, re
from pathlib import Path

import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "data/cache/whisper-large-v3-turbo"
SOURCES = sorted(Path("data/source").glob("*.m4a"), key=lambda p: int(re.search(r"(\d+)\.m4a$", p.name).group(1)))
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
rows = []
for path in SOURCES:
    index = int(re.search(r"(\d+)\.m4a$", path.name).group(1))
    master = Path(f"data/processed/master/{index}.wav")
    audio, rate = sf.read(master if master.exists() else path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    audio = resample_poly(audio, 16000, rate) if rate != 16000 else audio
    chunks = [audio[i:i + 480000] for i in range(0, len(audio), 480000)]
    text = []
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(inputs.input_features.to("cuda", torch.float16), language="ko", task="transcribe", max_new_tokens=96)
        text.append(processor.batch_decode(tokens, skip_special_tokens=True)[0].strip())
    rows.append({"source_index": index, "source_file": path.name, "asr_model": "openai/whisper-large-v3-turbo", "language": "ko", "transcript": " ".join(x for x in text if x)})
    print(index, rows[-1]["transcript"], flush=True)
Path("data/manifests/asr_transcripts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
