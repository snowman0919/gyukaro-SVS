#!/usr/bin/env python3
"""Measure score control and artifact proxies for score-native pilot outputs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor

CASES = {
    "rapid_ko": ROOT / "examples/review_rapid_ko.json",
    "large_interval_ko": ROOT / "examples/review_large_interval_ko.json",
}
RC6 = {
    "rapid_ko": ROOT / "artifacts/reports/rc6_listening_gate/06_rapid_ko.wav",
    "large_interval_ko": ROOT / "artifacts/reports/rc6_listening_gate/08_large_interval_ko.wav",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = ROOT / "artifacts/reports/diffsinger_score_native_pilot"
    previous = json.loads((root / "evaluation.json").read_text())
    checkpoints = {
        **{f"steps{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_pilot_best_{step}.ckpt" for step in (1000, 2000, 4000, 6000)},
        **{f"prior{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_ko_prior_early/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"prior{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_ko_prior/model_ckpt_steps_{step}.ckpt" for step in (400, 500)},
        **{f"acoustic{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_ko_acoustic_prior/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"zeroth{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_prior/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"replay{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_replay/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300, 600)},
        **{f"text{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_text/model_ckpt_steps_{step}.ckpt" for step in (200, 400, 600)},
        **{f"segmented{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_segmented/model_ckpt_steps_{step}.ckpt" for step in (200, 400, 600)},
        **{f"allseg{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_all_segmented/model_ckpt_steps_{step}.ckpt" for step in (200, 400)},
        **{f"gyuadapt{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_gyu_adapt/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
    }
    targets = {
        case: np.array(json.loads((root / f"{case}.ds").read_text())[0]["f0_seq"].split(), dtype=np.float32)
        for case in CASES
    }
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    rows = []
    for case in CASES:
        paths = {"rc6": RC6[case]} | {
            model: root / "listening" / f"{case}_{model}.wav"
            for model in checkpoints
        }
        for model, path in paths.items():
            rows.append({
                "case": case,
                "model": model,
                "path": str(path.relative_to(ROOT)),
                "sample_rate": sf.info(path).samplerate,
            } | acoustics(path) | pitch(path, targets[case], extractor))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    for row in rows:
        score = json.loads(CASES[row["case"]].read_text())
        expected = normalized("".join(note["lyric"] for note in score["notes"]))
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(
                inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_transcript"] = transcript
        row["asr_lyric_similarity"] = round(
            SequenceMatcher(None, expected, normalized(transcript)).ratio(), 4
        )

    metrics = (
        "pitch_mae_cents", "voicing_accuracy", "hf_spike_p99_over_median",
        "spectral_flux_p95", "sample_jump_p999", "asr_lyric_similarity",
    )
    aggregate = {
        model: {
            metric: round(float(np.mean([row[metric] for row in rows if row["model"] == model and row[metric] is not None])), 6)
            for metric in metrics
        }
        for model in ("rc6", *checkpoints)
    }
    candidate = max(
        tuple(model for model in checkpoints if model.startswith(("prior", "acoustic", "zeroth", "replay", "text", "segmented", "allseg", "gyuadapt"))),
        key=lambda model: (
            aggregate[model]["asr_lyric_similarity"],
            aggregate[model]["voicing_accuracy"],
            -aggregate[model]["pitch_mae_cents"],
        ),
    )
    eligible = (
        aggregate[candidate]["pitch_mae_cents"] <= 100
        and aggregate[candidate]["voicing_accuracy"] >= 0.8
        and aggregate[candidate]["asr_lyric_similarity"] >= 0.8
    )
    report = {
        "status": "objective_probe_pass_human_pending" if eligible else "objective_reject_undertrained",
        "human_listening": "pending" if eligible else "not_requested_objective_reject",
        "score_native": True,
        "selected_objective_candidate": candidate,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "training_validation_loss": {
            "pilot500": 0.22407, "pilot1000": 0.21156, "pilot1500": 0.21941,
            "pilot2000": 0.19398, "pilot4000": 0.16934, "pilot6000": 0.20919,
            "early_prior100": 0.21004, "early_prior200": 0.248,
            "early_prior300": 0.19735, "prior400": 0.19322, "prior500": 0.19852,
            "acoustic100": 0.14203, "acoustic200": 0.1598, "acoustic300": 0.22716,
            "zeroth100": 0.20216, "zeroth200": 0.17436, "zeroth300": 0.1783,
            "replay100": 0.21528, "replay200": 0.16625, "replay300": 0.19704,
            "replay600": 0.16408,
            "text200": 0.26735, "text400": 0.28097, "text600": 0.26234,
            "segmented200": 0.17748, "segmented400": 0.19848, "segmented600": 0.20261,
            "allseg200": 0.25150, "allseg400": 0.22856,
            "gyuadapt100": 0.23699, "gyuadapt200": 0.23334, "gyuadapt300": 0.20713,
        },
        "validation_loss_warning": "Diffusion validation loss varied materially across identical initial checkpoints; objective stress renders select the candidate.",
        "checkpoint_sha256": {
            model: sha256(path) if path.is_file() else previous["checkpoint_sha256"][model]
            for model, path in checkpoints.items()
        },
        "aggregate": aggregate,
        "rows": rows,
        "interpretation": "Objective metrics can reject a pilot, but cannot pass listening or make this an RC.",
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
