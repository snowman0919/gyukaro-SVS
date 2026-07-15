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
        **{f"lexical{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_vocalset_lexical/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"twostage{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_zeroth_gyu_adapt/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"gyulex{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_gyu_lexical/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
        **{f"phrase{step}": ROOT / f"data/cache/diffsinger/checkpoints/gyu_score_native_gyu_phrase_chunks/model_ckpt_steps_{step}.ckpt" for step in (100, 200, 300)},
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
        tuple(model for model in checkpoints if model.startswith(("prior", "acoustic", "zeroth", "replay", "text", "segmented", "allseg", "gyuadapt", "lexical", "twostage", "gyulex", "phrase"))),
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
            "lexical100": 0.255342, "lexical200": 0.211553, "lexical300": 0.211924,
            "twostage100": 0.260722, "twostage200": 0.252005, "twostage300": 0.256777,
            "gyulex100": 0.201801, "gyulex200": 0.191747, "gyulex300": 0.172804,
            "phrase100": 0.195004, "phrase200": 0.279518, "phrase300": 0.205754,
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
    lexical = {
        "status": "objective_reject_no_korean_lexical_transfer",
        "human_listening": "not_requested_objective_reject",
        "training_data": {
            "vocalset_lexical_rows": 76,
            "vocalset_lexical_minutes": 8.365,
            "gyu_segment_rows": 730,
            "license": "CC BY 4.0",
        },
        "aggregate": {
            model: aggregate[model]
            for model in ("rc6", "gyuadapt100", "lexical100", "lexical200", "lexical300")
        },
        "decision": (
            "The public-domain English lexical singing prior is valid training data, "
            "but does not transfer Korean consonant identity or voicing to the GYU stress set. "
            "Do not integrate these checkpoints into the runtime."
        ),
    }
    (ROOT / "artifacts/reports/diffsinger_vocalset_lexical_evaluation.json").write_text(
        json.dumps(lexical, ensure_ascii=False, indent=2) + "\n"
    )
    two_stage = {
        "status": "objective_reject_repetition_collapse",
        "human_listening": "not_requested_objective_reject",
        "training_order": [
            "Zeroth Korean speech acoustic prior with VocalSet/GYU replay",
            "730 inferred-timing real-GYU singing segments at 2e-6 learning rate",
        ],
        "aggregate": {
            model: aggregate[model]
            for model in ("rc6", "gyuadapt100", "twostage100", "twostage200", "twostage300")
        },
        "decision": (
            "The two-stage prior retains score pitch but emits repeated syllables instead of lyrics. "
            "The 730-segment corpus over-samples recording exercises; filter to lexical phrases "
            "before any further direct-model adaptation."
        ),
    }
    (ROOT / "artifacts/reports/diffsinger_zeroth_gyu_adapt_evaluation.json").write_text(
        json.dumps(two_stage, ensure_ascii=False, indent=2) + "\n"
    )
    gyu_lexical = {
        "status": "objective_reject_isolated_syllable_training",
        "human_listening": "not_requested_objective_reject",
        "training_data": "243 lexical-filtered real-GYU segments; inferred timing",
        "aggregate": {
            model: aggregate[model]
            for model in ("rc6", "gyuadapt100", "gyulex100", "gyulex200", "gyulex300")
        },
        "decision": (
            "Filtering recording exercises reduces high-frequency spikes but does not restore lyrics. "
            "Most retained segments are still isolated syllables split at 250 ms gaps; rebuild "
            "multi-syllable phrase chunks before further adaptation."
        ),
    }
    (ROOT / "artifacts/reports/diffsinger_gyu_lexical_evaluation.json").write_text(
        json.dumps(gyu_lexical, ensure_ascii=False, indent=2) + "\n"
    )
    phrase_chunks = {
        "status": "objective_probe_pending",
        "human_listening": "pending_objective_gate",
        "training_data": {
            "rows": 81,
            "minutes": 4.742,
            "median_seconds": 3.454,
            "split_gap_seconds": 0.8,
            "alignment": "inferred",
            "independent_score_rows_used": False,
        },
        "aggregate": {
            model: aggregate[model]
            for model in ("rc6", "gyulex100", "phrase100", "phrase200", "phrase300")
        },
    }
    phrase_candidate = max(
        ("phrase100", "phrase200", "phrase300"),
        key=lambda model: (
            aggregate[model]["asr_lyric_similarity"],
            aggregate[model]["voicing_accuracy"],
            -aggregate[model]["pitch_mae_cents"],
        ),
    )
    phrase_eligible = (
        aggregate[phrase_candidate]["pitch_mae_cents"] <= 100
        and aggregate[phrase_candidate]["voicing_accuracy"] >= 0.8
        and aggregate[phrase_candidate]["asr_lyric_similarity"] >= 0.8
    )
    phrase_chunks |= {
        "status": "objective_probe_pass_human_pending" if phrase_eligible else "objective_reject_phrase_chunks_insufficient",
        "human_listening": "pending" if phrase_eligible else "not_requested_objective_reject",
        "selected_candidate": phrase_candidate,
        "decision": (
            "Phrase chunks pass the objective gate; conduct listening before integration."
            if phrase_eligible else
            "Longer coarticulated chunks alone do not recover stable Korean lyrics; do not integrate these checkpoints."
        ),
    }
    (ROOT / "artifacts/reports/diffsinger_gyu_phrase_chunks_evaluation.json").write_text(
        json.dumps(phrase_chunks, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": report["status"], "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
