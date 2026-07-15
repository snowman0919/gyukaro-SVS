#!/usr/bin/env python3
"""Evaluate Korean score-native source, SoulX conversion, and GYU adaptation gates."""
from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioXVector,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(CACHE / "soulx-singer"))

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


CASES = {
    "rapid_ko": "빠르게노래하자아",
    "large_interval_ko": "높이날아",
}


def audio_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    return resample_poly(audio, 16000, rate).astype("float32") if rate != 16000 else audio


def main() -> None:
    listening = ROOT / "artifacts/reports/mlp_singer_korean_probe/listening"
    variants = {
        "rc6": {
            "rapid_ko": ROOT / "artifacts/reports/rc6_listening_gate/06_rapid_ko.wav",
            "large_interval_ko": ROOT / "artifacts/reports/rc6_listening_gate/08_large_interval_ko.wav",
        },
        "base_c3": {case: listening / f"{case}_generated_e2e.wav" for case in CASES},
        "base_c6": {case: listening / f"{case}_c6_generated_e2e.wav" for case in CASES},
        "base_c6_soulx": {case: listening / f"{case}_c6_soulx_s32_c2.wav" for case in CASES},
        "rvc_e5": {case: listening / f"rvc_e5/{case}.wav" for case in CASES},
        "rvc_e5_plus15": {
            case: listening / f"rvc_e5_plus15/{case}.wav" for case in CASES
        },
    }
    for step, strength in ((100, 100), (400, 100), (1000, 25), (1000, 50), (1000, 100)):
        model = f"film_{step}_s{strength:03d}"
        variants[model] = {case: listening / model / f"{case}.wav" for case in CASES}
    for step in (25, 50, 100):
        model = f"speaker_film_{step}_s100"
        variants[model] = {case: listening / model / f"{case}.wav" for case in CASES}
        model = f"speaker_residual_{step}_s100"
        variants[model] = {case: listening / model / f"{case}.wav" for case in CASES}
    for step in (100, 200, 400):
        variants[f"full_{step}"] = {
            case: listening / f"gyu_steps{step}/{case}_gyu_generated_e2e.wav" for case in CASES
        }
        variants[f"projection_{step}"] = {
            case: listening / f"gyu_projection_steps{step}/{case}_gyu_generated_e2e.wav"
            for case in CASES
        }
    for strength in (0, 10, 25, 50):
        variants[f"blend_{strength:03d}"] = {
            case: listening / f"gyu_blend_{strength:03d}/{case}_gyu_generated_e2e.wav"
            for case in CASES
        }

    device = "cuda"
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    f0_extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device=device,
        target_sr=24000,
        hop_size=480,
        verbose=False,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio_16k(path)
        value = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            wavlm_value = wavlm(**{key: item.cuda() for key, item in value.items()}).embeddings
            ecapa_value = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        wavlm_value = torch.nn.functional.normalize(wavlm_value, dim=-1).squeeze().cpu().numpy()
        ecapa_value = ecapa_value.squeeze().cpu().numpy()
        ecapa_value /= max(np.linalg.norm(ecapa_value), 1e-8)
        return wavlm_value, ecapa_value

    reference = speaker(ROOT / "data/processed/master/216.wav")
    rows = []
    for model, paths in variants.items():
        for case, path in paths.items():
            target = np.load(ROOT / f"artifacts/reports/mlp_singer_korean_probe/f0/{case}.npy")
            if model in {"base_c3", "base_c6"} and case == "large_interval_ko":
                target = target * 2 ** (-2 / 12)
            inputs = processor(audio=audio16(path), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = whisper.generate(
                    inputs.input_features.cuda().half(),
                    language="ko",
                    task="transcribe",
                    max_new_tokens=64,
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            embeddings = speaker(path)
            rows.append({
                "model": model,
                "case": case,
                "path": str(path.relative_to(ROOT)),
                "transcript": transcript,
                "asr_lyric_similarity": round(
                    SequenceMatcher(
                        None, normalized(CASES[case]), normalized(transcript)
                    ).ratio(),
                    4,
                ),
                "wavlm_to_gyu": round(float(np.dot(reference[0], embeddings[0])), 5),
                "ecapa_to_gyu": round(float(np.dot(reference[1], embeddings[1])), 5),
            } | acoustics(path) | pitch(path, target, f0_extractor))

    metrics = (
        "asr_lyric_similarity",
        "pitch_mae_cents",
        "voicing_accuracy",
        "hf_spike_p99_over_median",
        "spectral_flux_p95",
        "sample_jump_p999",
        "wavlm_to_gyu",
        "ecapa_to_gyu",
    )
    aggregate = {
        model: {
            metric: round(float(np.mean([row[metric] for row in rows if row["model"] == model])), 6)
            for metric in metrics
        }
        for model in variants
    }
    blend_models = tuple(model for model in variants if model.startswith("blend_"))
    eligible = []
    for model in blend_models:
        model_rows = [row for row in rows if row["model"] == model]
        if (
            min(row["asr_lyric_similarity"] for row in model_rows) >= 0.75
            and min(row["voicing_accuracy"] for row in model_rows) >= 0.8
            and max(row["pitch_mae_cents"] for row in model_rows) <= 50
            and aggregate[model]["wavlm_to_gyu"] >= aggregate["blend_000"]["wavlm_to_gyu"] + 0.01
            and aggregate[model]["ecapa_to_gyu"] >= aggregate["blend_000"]["ecapa_to_gyu"] + 0.01
        ):
            eligible.append(model)
    selected_blend = max(
        eligible,
        key=lambda model: (
            aggregate[model]["wavlm_to_gyu"],
            aggregate[model]["asr_lyric_similarity"],
        ),
        default=None,
    )
    rvc_models = ("rvc_e5", "rvc_e5_plus15")
    eligible_rvc = []
    for model in rvc_models:
        model_rows = [row for row in rows if row["model"] == model]
        if (
            min(row["asr_lyric_similarity"] for row in model_rows) >= 0.75
            and min(row["voicing_accuracy"] for row in model_rows) >= 0.8
            and max(row["pitch_mae_cents"] for row in model_rows) <= 50
            and aggregate[model]["hf_spike_p99_over_median"]
            < aggregate["rc6"]["hf_spike_p99_over_median"]
            and aggregate[model]["wavlm_to_gyu"]
            >= aggregate["base_c6"]["wavlm_to_gyu"] + 0.01
            and aggregate[model]["ecapa_to_gyu"]
            >= aggregate["base_c6"]["ecapa_to_gyu"] + 0.01
        ):
            eligible_rvc.append(model)
    selected_rvc = max(
        eligible_rvc,
        key=lambda model: (
            aggregate[model]["asr_lyric_similarity"],
            aggregate[model]["wavlm_to_gyu"],
        ),
        default=None,
    )
    film_models = tuple(
        model
        for model in variants
        if model.startswith(("film_", "speaker_film_", "speaker_residual_"))
    )
    eligible_film = []
    for model in film_models:
        model_rows = [row for row in rows if row["model"] == model]
        if (
            min(row["asr_lyric_similarity"] for row in model_rows) >= 0.75
            and min(row["voicing_accuracy"] for row in model_rows) >= 0.8
            and max(row["pitch_mae_cents"] for row in model_rows) <= 50
            and aggregate[model]["hf_spike_p99_over_median"]
            < aggregate["rc6"]["hf_spike_p99_over_median"]
            and aggregate[model]["wavlm_to_gyu"]
            >= aggregate["base_c6"]["wavlm_to_gyu"] + 0.01
            and aggregate[model]["ecapa_to_gyu"]
            >= aggregate["base_c6"]["ecapa_to_gyu"] + 0.01
        ):
            eligible_film.append(model)
    selected_film = max(
        eligible_film,
        key=lambda model: (
            aggregate[model]["asr_lyric_similarity"],
            aggregate[model]["wavlm_to_gyu"],
        ),
        default=None,
    )
    selected = selected_film or selected_rvc or selected_blend
    report = {
        "status": "objective_probe_pass_human_pending" if selected else "score_native_source_pass_gyu_conversion_reject",
        "human_listening": "pending" if selected else "not_requested_objective_reject",
        "selected": selected,
        "primary_finding": (
            "The score-native source removes post-hoc content timing, while SoulX conversion destroys its lyric advantage."
        ),
        "score_native": True,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "independent_stress_scores": True,
        "adaptation_training_scores": "inferred; independent verified rows excluded",
        "aggregate": aggregate,
        "rows": rows,
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/mlp_singer_korean_probe/evaluation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "selected": selected, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
