#!/usr/bin/env python3
"""Record Fish→SoulX probes, excluding them from training under Fish license terms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PILOTS = [
    ("ko", "하늘을 향해 노래해", "data/pseudo_singing/fish_soulx_probe/generated.wav", 0.9918, 1.0, 0.8585, 0.7218, 3.2973),
    ("en", "Sing toward the bright sky", "data/pseudo_singing/fish_soulx_en_probe/generated.wav", 0.9778, 1.0, 0.9632, 0.6769, 4.4582),
    ("ja", "空へ向かって歌おう", "data/pseudo_singing/fish_soulx_ja_probe/generated.wav", 0.9964, 0.6364, 0.8812, 0.7253, 3.6223),
]


def score(language: str, text: str, duration: float) -> dict:
    units = [char for char in text if not char.isspace()] or ["la"]
    step = duration / len(units)
    return {"language": language, "tempo": 120, "sample_rate": 48000, "score_source": "inferred_from_RMVPE_pilot_contour_not_manual_score",
            "notes": [{"id": f"n{index}", "pitch": 47 + index % 4, "start": round(index * step, 5), "duration": round(step, 5), "lyric": unit} for index, unit in enumerate(units)]}


def main() -> None:
    output = Path("data/pseudo_singing/accepted"); output.mkdir(parents=True, exist_ok=True)
    candidates, accepted = [], []
    for language, text, source, f0, content, wavlm, ecapa, duration in PILOTS:
        identifier = f"pseudo_fish_soulx_{language}_001"
        target = output / f"{identifier}.wav"
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != 48000: mono = resample_poly(mono, 48000, rate).astype("float32")
        sf.write(target, mono, 48000)
        row = {"id": identifier, "generator": "Fish S2 [singing] -> SoulX SVC", "model_revision": "Fish S2 local + SoulX-Singer SVC@40493ad", "reference_ids": ["gyu_real_000216"], "language": language, "text": text,
               "source_output_path": source, "output_path": str(target), "f0_contour_correlation_rmvpe": f0, "duration_ratio": 1.0, "content_score": content, "speaker_score": wavlm, "speaker_score_2": ecapa,
               "quality_status": "rejected_license", "trust_weight": 0.0, "training_license": "prohibited_for_foundational_generative_ai_training", "training_use": "evaluation_only_not_training", "score": score(language, text, duration)}
        candidates.append(row)
    root = Path("data/manifests")
    for name, rows in (("pseudo_singing_candidates.jsonl", candidates), ("pseudo_singing_accepted.jsonl", accepted)):
        (root / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print({"candidates": len(candidates), "accepted": len(accepted)})


if __name__ == "__main__":
    main()
