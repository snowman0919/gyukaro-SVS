#!/usr/bin/env python3
"""Small real execution gate before scaling synthetic teacher data."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

model_path = "data/cache/moss-tts-nano"
tokenizer_path = "data/cache/moss-audio-tokenizer-nano"
items = [
    ("ko", "하늘에 빛이 내려와.", 215),
    ("en", "A quiet light is falling from the sky.", 216),
    ("ja", "空から静かな光が降りてくる。", 217),
]
Path("data/teacher").mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model._set_attention_implementation("sdpa")
model = model.to("cuda", dtype=torch.bfloat16).eval()
rows = []
for language, text, reference_index in items:
    output = Path(f"data/teacher/moss_nano_{language}.wav")
    result = model.inference(text=text, output_audio_path=output, mode="voice_clone", prompt_audio_path=f"data/source/Korea Digital Media High School {reference_index}.m4a", audio_tokenizer_type="moss-audio-tokenizer-nano", audio_tokenizer_pretrained_name_or_path=tokenizer_path, device="cuda", max_new_frames=80, do_sample=False)
    rows.append({"teacher":"OpenMOSS-Team/MOSS-TTS-Nano", "teacher_role":"small-pilot replacement; not requested 4B Local Transformer", "model_revision":"downloaded 2026-07-14", "language":language, "text":text, "reference_ids":[f"gyu_real_{reference_index:06d}"], "seed":None, "generation_config":{"max_new_frames":80,"do_sample":False,"attention":"sdpa"}, "output_path":str(output), "sample_rate":result["sample_rate"], "quality_status":"pending_filter"})
Path("data/manifests/teacher_pilot.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows))
