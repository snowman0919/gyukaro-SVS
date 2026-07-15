#!/usr/bin/env python3
"""Small decoder sweep after the Korean obstruent-voicing correction."""

from __future__ import annotations

import json
import time
from pathlib import Path

from gyu_singer.inference.v09 import GyuSingerV09Renderer


def main() -> None:
    root = Path("artifacts/reports/rc5_large_interval_decode")
    root.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerV09Renderer("data/processed/master/216.wav", root=Path.cwd())
    renderer.omnivoice.close()
    common = {
        "source": "artifacts/reports/rc5_isolation/large_interval_ko/production_adapted_source.wav",
        "f0_npy": "artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy",
        "identity_npy": "artifacts/reports/rc5_isolation/large_interval_ko/identity.npy",
        "style_npy": "artifacts/reports/rc5_isolation/large_interval_ko/style.npy",
    }
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
