#!/usr/bin/env python3
"""Extract 12.5 Hz RMVPE contours for real/pseudo hybrid acoustic rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "data/cache/soulx-singer")
from preprocess.tools.f0_extraction import F0Extractor


def main() -> None:
    rows = [json.loads(line) for line in Path("data/manifests/hybrid_all.jsonl").read_text().splitlines() if line]
    output = Path("data/cache/hybrid_f0"); output.mkdir(parents=True, exist_ok=True)
    extractor = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device="cuda", target_sr=48000, hop_size=3840, verbose=False)
    for row in rows:
        target = Path(row.get("f0_path", output / f"{row['id']}.npy"))
        if target.exists(): continue
        target.parent.mkdir(parents=True, exist_ok=True)
        extractor.process(row["audio_path"], f0_path=str(target), verbose=False)
        print(row["id"])


if __name__ == "__main__":
    main()
