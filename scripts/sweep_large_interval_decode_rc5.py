#!/usr/bin/env python3
"""Small decoder sweep for the remaining RC5 stress cases."""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

from gyu_singer.inference.v09 import GyuSingerV09Renderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("large_interval_ko", "rapid_ko"), default="large_interval_ko")
    args = parser.parse_args()
    stem = "large_interval" if args.case == "large_interval_ko" else "rapid"
    root = Path(f"artifacts/reports/rc5_{stem}_decode")
    root.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd())
    renderer.omnivoice.close()
    common = {
        "source": f"artifacts/reports/rc5_isolation/{args.case}/production_adapted_source.wav",
        "f0_npy": f"artifacts/reports/rc5_candidate_core/{args.case}/canonical_f0.npy",
        "identity_npy": f"artifacts/reports/rc5_isolation/{args.case}/identity.npy",
        "style_npy": f"artifacts/reports/rc5_isolation/{args.case}/style.npy",
    }
    if args.case == "rapid_ko":
        common["content_warp_npy"] = "artifacts/reports/rc5_rapid_hold/warp.npy"
    rows = []
    try:
        for steps, cfg in ((32, 1.5), (32, 2.0), (50, 1.5), (50, 2.0), (64, 1.5), (64, 2.0)):
            for seed in (11, 21, 37):
                name = f"s{steps}_c{cfg:g}_seed{seed}"
                output = root / f"{name}.wav"
                started = time.perf_counter()
                renderer.soulx.request(
                    common
                    | {
                        "n_steps": steps,
                        "cfg": cfg,
                        "seed": seed,
                        "output": str(output),
                    }
                )
                rows.append(
                    {
                        "name": name,
                        "path": str(output),
                        "steps": steps,
                        "cfg": cfg,
                        "seed": seed,
                        "render_seconds": round(time.perf_counter() - started, 3),
                    }
                )
    finally:
        renderer.close()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "rendered_not_selected",
                "case": args.case,
                "precision": "fp32",
                "same_source_f0_identity_style": True,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
