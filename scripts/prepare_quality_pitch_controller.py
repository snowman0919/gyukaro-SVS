#!/usr/bin/env python3
"""Create explicitly synthetic score/F0 rows for the quality pitch controller."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gyu_singer.inference.soulx import SoulXPhraseRenderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/manifests/quality_pitch_controller.jsonl")
    parser.add_argument("--cache", default="data/cache/quality_pitch_controller")
    args = parser.parse_args()
    sys.path.insert(0, "data/cache/soulx-singer")
    from preprocess.tools.f0_extraction import F0Extractor
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    renderer = SoulXPhraseRenderer("data/processed/master/216.wav", use_controller=False)
    f0 = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device="cuda" if torch.cuda.is_available() else "cpu", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    try:
        for path in sorted(Path("examples").glob("quality_*.json")):
            base = json.loads(path.read_text())
            for preset in ("neutral", "soft"):
                score = {**base, "style": {"preset": preset}}
                row_id = f"{path.stem}_{preset}"
                audio_path, f0_path = cache / f"{row_id}.wav", cache / f"{row_id}.npy"
                renderer.render_file(path if preset == "neutral" else write_score(cache / f"{row_id}.json", score), audio_path)
                np.save(f0_path, f0.process(str(audio_path), verbose=False).astype("float32"))
                rows.append({"id": row_id, "synthetic": True, "target_source": "generated_quality_runtime_RMVPE_inferred", "score": score, "target_f0_path": str(f0_path)})
    finally:
        renderer.close()
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print({"rows": len(rows), "output": args.output})


def write_score(path: Path, score: dict) -> Path:
    path.write_text(json.dumps(score, ensure_ascii=False))
    return path


if __name__ == "__main__": main()
