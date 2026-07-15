#!/usr/bin/env python3
"""Render CTC phone-hold timing for rapid or phrase-boundary stress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gyu_singer.inference.content_timing import latent_content_hold
from gyu_singer.inference.v09 import GyuSingerV09Renderer


CONFIG = {
    "rapid_ko": {
        "source": "artifacts/reports/rc5_isolation/rapid_ko/production_adapted_source.wav",
        "alignment": "artifacts/reports/rc5_content_timing/rapid_ko/alignment.json",
        "f0": "artifacts/reports/rc5_candidate_core/rapid_ko/canonical_f0.npy",
        "identity": "artifacts/reports/rc5_isolation/rapid_ko/identity.npy",
        "style": "artifacts/reports/rc5_isolation/rapid_ko/style.npy",
        "output": "artifacts/reports/rc5_rapid_hold/rapid.wav",
    },
    "phrase_boundary": {
        "source": "artifacts/reports/rc5_boundary_isolation/phrase_boundary/production_adapted_source.wav",
        "alignment": "artifacts/reports/rc5_boundary_timing/phrase_boundary/alignment.json",
        "f0": "artifacts/reports/rc5_stress_candidate4/phrase_boundary_target_f0.npy",
        "identity": "artifacts/reports/rc5_boundary_isolation/phrase_boundary/identity.npy",
        "style": "artifacts/reports/rc5_boundary_isolation/phrase_boundary/style.npy",
        "output": "artifacts/reports/rc5_boundary_hold/hold.wav",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CONFIG), default="rapid_ko")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    paths = CONFIG[args.case]
    source, f0 = Path(paths["source"]), np.load(paths["f0"])
    alignment = json.loads(Path(paths["alignment"]).read_text())
    output = Path(paths["output"])
    if args.strength != 1.0:
        output = output.with_stem(f"{output.stem}_{args.strength:g}")
    output.parent.mkdir(parents=True, exist_ok=True)
    warp = latent_content_hold(alignment, sf.info(source).duration, len(f0))
    warp_path = output.parent / "warp.npy"
    np.save(warp_path, warp)

    renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd())
    renderer.omnivoice.close()
    try:
        renderer.soulx.request(
            {
                "source": str(source),
                "f0_npy": paths["f0"],
                "content_warp_npy": str(warp_path),
                "content_warp_strength": args.strength,
                "identity_npy": paths["identity"],
                "style_npy": paths["style"],
                "n_steps": args.steps,
                "cfg": args.cfg,
                "seed": args.seed,
                "output": str(output),
            }
        )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
