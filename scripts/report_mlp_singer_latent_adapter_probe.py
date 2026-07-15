#!/usr/bin/env python3
"""Record the bounded latent-adapter rejection on independent stress scores."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    evaluation = json.loads(
        (ROOT / "artifacts/reports/mlp_singer_korean_probe/evaluation.json").read_text()
    )
    models = (
        "base_c6",
        "film_100_s100",
        "film_1000_s025",
        "speaker_film_25_s100",
        "speaker_residual_25_s100",
        "speaker_residual_100_s100",
    )
    report = {
        "status": "latent_identity_adapters_rejected",
        "production_integrated": False,
        "human_listening_requested": False,
        "base_model_role": "diagnostic only; CSD archive metadata is CC BY-NC-SA 4.0",
        "training_data": {
            "real_gyu_train_rows": 43,
            "real_gyu_validation_rows": 5,
            "score_status": "inferred_from_target_f0",
            "independent_verified_rows_used": 0,
        },
        "adapters": {
            "mel_l1_film": {
                "trainable_parameters": 576,
                "supervision": "real GYU mel L1 plus temporal dynamics preservation",
            },
            "dual_speaker_film": {
                "trainable_parameters": 576,
                "supervision": "frozen WavLM and ECAPA plus content-preservation losses",
            },
            "dual_speaker_residual": {
                "trainable_parameters": 9520,
                "supervision": "frozen WavLM and ECAPA plus content-preservation losses",
            },
        },
        "aggregate": {model: evaluation["aggregate"][model] for model in models},
        "acceptance": {
            "selected": evaluation["selected"],
            "required": "per-case ASR >= 0.75 and both WavLM/ECAPA >= base + 0.01",
        },
        "finding": (
            "The bounded adapters either reduce lyric retention or fail cross-encoder identity "
            "agreement. More MLP-Singer adaptation is stopped."
        ),
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/mlp_singer_latent_adapter_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
