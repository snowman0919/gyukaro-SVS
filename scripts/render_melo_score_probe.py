#!/usr/bin/env python3
"""Probe exact score timing through the MIT-licensed Korean MeloTTS decoder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
MELO = CACHE / "melo-tts"
MODEL = CACHE / "melo-tts-korean"
sys.path.insert(0, str(MELO))

from melo import commons  # noqa: E402
from melo.models import SynthesizerTrn  # noqa: E402


ONSET = list("ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ")
NUCLEUS = list("ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ")
CODA = {
    1: "ᆨ", 2: "ᆨ", 3: "ᆨ", 4: "ᆫ", 5: "ᆫ", 6: "ᆫ", 7: "ᆮ",
    8: "ᆯ", 9: "ᆨ", 10: "ᆷ", 11: "ᆸ", 12: "ᆯ", 13: "ᆯ", 14: "ᆸ",
    15: "ᆯ", 16: "ᆷ", 17: "ᆸ", 18: "ᆸ", 19: "ᆮ", 20: "ᆮ", 21: "ᆼ",
    22: "ᆮ", 23: "ᆮ", 24: "ᆨ", 25: "ᆮ", 26: "ᆸ", 27: "ᆮ",
}


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


def attributes(value):
    if isinstance(value, dict):
        return AttrDict({key: attributes(item) for key, item in value.items()})
    if isinstance(value, list):
        return [attributes(item) for item in value]
    return value


def melo_symbol(phone: str) -> str:
    category, index = phone.rsplit("_", 1)
    value = int(index)
    if category == "ko_onset":
        return ONSET[value]
    if category == "ko_nucleus":
        return NUCLEUS[value]
    if category == "ko_coda":
        return CODA[value]
    raise ValueError(f"unsupported Korean score phone: {phone}")


def load_model(device: str) -> tuple[SynthesizerTrn, AttrDict]:
    config = attributes(json.loads((MODEL / "config.json").read_text()))
    model = SynthesizerTrn(
        len(config.symbols),
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        n_speakers=config.data.n_speakers,
        num_tones=config.num_tones,
        num_languages=config.num_languages,
        **config.model,
    ).to(device).eval()
    checkpoint = torch.load(MODEL / "checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.dec.remove_weight_norm()
    return model, config


def inputs(row: dict, config: AttrDict, device: str) -> tuple[torch.Tensor, ...]:
    symbol_ids = {symbol: index for index, symbol in enumerate(config.symbols)}
    phones = [melo_symbol(phone) for phone in row["ph_seq"].split()]
    durations = [float(value) for value in row["ph_dur"].split()]
    phone_ids = [symbol_ids[phone] for phone in phones]
    frame_rate = config.data.sampling_rate / config.data.hop_length
    frame_durations = [max(1, round(duration * frame_rate)) for duration in durations]
    expanded_ids, expanded_durations = [0], [0]
    for phone_id, duration in zip(phone_ids, frame_durations):
        expanded_ids.extend((phone_id, 0))
        expanded_durations.extend((duration, 0))
    length = len(expanded_ids)
    x = torch.tensor(expanded_ids, dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([length], dtype=torch.long, device=device)
    tone = torch.full_like(x, 11)
    language = torch.full_like(x, 4)
    bert = torch.zeros(1, 1024, length, device=device)
    ja_bert = torch.zeros(1, 768, length, device=device)
    durations_tensor = torch.tensor(expanded_durations, dtype=torch.float32, device=device)[None, None]
    return x, x_lengths, tone, language, bert, ja_bert, durations_tensor


@torch.inference_mode()
def render_exact(model: SynthesizerTrn, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, x_lengths, tone, language, bert, ja_bert, durations = values
    sid = torch.zeros(1, dtype=torch.long, device=x.device)
    g = model.emb_g(sid).unsqueeze(-1)
    _, mean, _, x_mask = model.enc_p(x, x_lengths, tone, language, bert, ja_bert, g=g)
    y_lengths = durations.sum((1, 2)).long()
    y_mask = commons.sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
    attention = commons.generate_path(durations, attn_mask)
    mean = torch.matmul(attention.squeeze(1), mean.transpose(1, 2)).transpose(1, 2)
    latent = model.flow(mean, y_mask, g=g, reverse=True)
    return model.dec(latent * y_mask, g=g)[0, 0]


@torch.inference_mode()
def render_predicted(model: SynthesizerTrn, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, x_lengths, tone, language, bert, ja_bert, _ = values
    sid = torch.zeros(1, dtype=torch.long, device=x.device)
    return model.infer(
        x, x_lengths, sid, tone, language, bert, ja_bert,
        noise_scale=0.0, noise_scale_w=0.0, sdp_ratio=0.0,
    )[0][0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, config = load_model(args.device)
    source = ROOT / "artifacts/reports/diffsinger_score_native_pilot"
    output = ROOT / "artifacts/reports/melo_score_probe/listening"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in ("rapid_ko", "large_interval_ko"):
        row = json.loads((source / f"{case}.ds").read_text())[0]
        values = inputs(row, config, args.device)
        for mode, audio in (
            ("predicted_duration", render_predicted(model, values)),
            ("exact_duration", render_exact(model, values)),
        ):
            path = output / f"{case}_{mode}.wav"
            samples = audio.float().cpu().numpy()
            sf.write(path, samples, config.data.sampling_rate, subtype="PCM_24")
            rows.append({
                "case": case,
                "mode": mode,
                "path": str(path.relative_to(ROOT)),
                "sample_rate": config.data.sampling_rate,
                "duration_seconds": len(samples) / config.data.sampling_rate,
            })
    report = {
        "status": "diagnostic_render_complete",
        "model": "myshell-ai/MeloTTS-Korean",
        "model_revision": "0207e5a",
        "repository_revision": "209145371cff8fc3bd60d7be902ea69cbdb7965a",
        "license": "MIT",
        "checkpoint_sha256": "48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0",
        "score_native": False,
        "exact_duration_control": True,
        "explicit_f0_control": False,
        "bert_conditioning": "zeroed_for_bounded_interface_probe",
        "rows": rows,
    }
    target = ROOT / "artifacts/reports/melo_score_probe/render.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
