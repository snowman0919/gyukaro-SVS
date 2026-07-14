#!/usr/bin/env python3
"""Extract real frozen internal representations from Fish DAC and MOSS tokenizer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers


def rows(path: str, limit: int) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line][:limit]


def extract_moss(path: str, model) -> tuple[torch.Tensor, tuple[int, ...]]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 48000:
        raise ValueError(f"MOSS expects 48 kHz, got {rate}")
    wav = torch.from_numpy(audio.T.copy()).to(next(model.parameters()).device)
    with torch.inference_mode(): result = model.encode(wav, chunk_duration=4.0)
    hidden = result.encoder_hidden_states[0].float().mean(dim=-1)
    return hidden.cpu(), tuple(result.encoder_hidden_states.shape)


def load_fish():
    import hydra
    from omegaconf import OmegaConf
    sys.path.insert(0, "data/cache/fish-speech")
    cfg = OmegaConf.load("data/cache/fish-speech/fish_speech/configs/modded_dac_vq.yaml")
    model = hydra.utils.instantiate(cfg)
    state = torch.load("data/cache/fish-s2-pro/codec.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False); return model.eval()


def extract_fish(path: str, model) -> tuple[torch.Tensor, tuple[int, ...]]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); audio = audio.mean(1)
    wav = torch.from_numpy(audio[None, None].copy())
    if rate != 44100: wav = torch.nn.functional.interpolate(wav, size=round(wav.shape[-1] * 44100 / rate), mode="linear", align_corners=False)
    with torch.inference_mode(): hidden = model.encoder(wav)
    return hidden[0].float().mean(dim=-1).cpu(), tuple(hidden.shape)


def main() -> None:
    limit = 4; device = "cuda" if torch.cuda.is_available() else "cpu"; manifest = []
    moss = transformers.AutoModel.from_pretrained("data/cache/moss-audio-tokenizer-nano", trust_remote_code=True, local_files_only=True).to(device).eval()
    for row in rows("data/manifests/teacher_moss_local_v15.jsonl", limit):
        value, shape = extract_moss(row["output_path"], moss); path = Path("data/cache/teacher_representations/moss") / f"{row['id']}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save(value, path); manifest.append({"id": row["id"], "teacher": "moss_local_v15", "representation": "MOSS-Audio-Tokenizer-Nano.encoder_hidden_states", "shape": list(shape), "pooled_shape": list(value.shape), "path": str(path), "revision": "local:data/cache/moss-audio-tokenizer-nano"})
    del moss
    fish = load_fish()
    for row in rows("data/manifests/teacher_fish_s2_pro.jsonl", limit):
        value, shape = extract_fish(row["output_path"], fish); path = Path("data/cache/teacher_representations/fish") / f"{row['id']}.pt"; path.parent.mkdir(parents=True, exist_ok=True); torch.save(value, path); manifest.append({"id": row["id"], "teacher": "fish_s2_pro", "representation": "Fish-S2-Pro-DAC.encoder_hidden", "shape": list(shape), "pooled_shape": list(value.shape), "path": str(path), "revision": "local:data/cache/fish-s2-pro"})
    Path("data/manifests/teacher_internal_representations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    print({"rows": len(manifest), "teachers": sorted({row["teacher"] for row in manifest}), "device": device})


if __name__ == "__main__": main()
