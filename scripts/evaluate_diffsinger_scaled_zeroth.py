#!/usr/bin/env python3
"""Gate the scaled Zeroth-Korean score-native replay checkpoint."""
from __future__ import annotations

import hashlib
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
LISTENING = ROOT / "artifacts/reports/diffsinger_score_native_pilot/listening"
VARIANTS = {
    "rc6": {
        "rapid_ko": ROOT / "artifacts/reports/rc6_listening_gate/06_rapid_ko.wav",
        "large_interval_ko": ROOT / "artifacts/reports/rc6_listening_gate/08_large_interval_ko.wav",
    },
    "zeroth1h_replay300": {
        case: LISTENING / f"{case}_replay300.wav" for case in CASES
    },
    "zeroth3h5_replay300": {
        case: LISTENING / f"{case}_zeroth3h5_300.wav" for case in CASES
    },
    "soulx_score32": {
        case: ROOT / f"artifacts/reports/soulx_score_native_ko_probe/listening/{case}_s32_c2.wav"
        for case in CASES
    },
    "soulx_score32_verified": {
        case: ROOT / f"artifacts/reports/soulx_score_native_ko_probe/listening/{case}_s32_c2_vp.wav"
        for case in CASES
    },
    "soulx_melody32_verified": {
        case: ROOT / f"artifacts/reports/soulx_score_native_ko_probe/listening/{case}_melody_s32_c2_vp.wav"
        for case in CASES
    },
}


def audio_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    return resample_poly(audio, 16000, rate).astype("float32") if rate != 16000 else audio


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for paths in VARIANTS.values():
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    f0_extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    wavlm_processor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cuda:0"},
    )

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio_16k(path)
        values = wavlm_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            wavlm_value = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            ecapa_value = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        wavlm_value = torch.nn.functional.normalize(wavlm_value, dim=-1).squeeze().cpu().numpy()
        ecapa_value = ecapa_value.squeeze().cpu().numpy()
        ecapa_value /= max(np.linalg.norm(ecapa_value), 1e-8)
        return wavlm_value, ecapa_value

    reference = speaker(ROOT / "data/processed/master/216.wav")
    targets = {
        case: np.array(
            json.loads((ROOT / f"artifacts/reports/diffsinger_score_native_pilot/{case}.ds").read_text())[0]["f0_seq"].split(),
            dtype=np.float32,
        )
        for case in CASES
    }
    rows = []
    for model, paths in VARIANTS.items():
        for case, path in paths.items():
            inputs = processor(audio=audio16(path), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = whisper.generate(
                    inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            embeddings = speaker(path)
            rows.append({
                "model": model,
                "case": case,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "asr_transcript": transcript,
                "asr_lyric_similarity": round(
                    SequenceMatcher(None, normalized(CASES[case]), normalized(transcript)).ratio(), 4
                ),
                "wavlm_to_gyu": round(float(np.dot(reference[0], embeddings[0])), 5),
                "ecapa_to_gyu": round(float(np.dot(reference[1], embeddings[1])), 5),
            } | acoustics(path) | pitch(path, targets[case], f0_extractor))

    metrics = (
        "asr_lyric_similarity", "pitch_mae_cents", "voicing_accuracy",
        "hf_spike_p99_over_median", "spectral_flux_p95", "sample_jump_p999",
        "wavlm_to_gyu", "ecapa_to_gyu",
    )
    aggregate = {
        model: {
            metric: round(float(np.mean([row[metric] for row in rows if row["model"] == model])), 6)
            for metric in metrics
        }
        for model in VARIANTS
    }
    candidate = "zeroth3h5_replay300"
    candidate_rows = [row for row in rows if row["model"] == candidate]
    eligible = (
        min(row["asr_lyric_similarity"] for row in candidate_rows) >= 0.8
        and min(row["voicing_accuracy"] for row in candidate_rows) >= 0.8
        and max(row["pitch_mae_cents"] for row in candidate_rows) <= 100
        and aggregate[candidate]["hf_spike_p99_over_median"] < aggregate["rc6"]["hf_spike_p99_over_median"]
    )
    scale_signal = (
        aggregate[candidate]["asr_lyric_similarity"]
        > aggregate["zeroth1h_replay300"]["asr_lyric_similarity"] + 0.1
        and min(row["asr_lyric_similarity"] for row in candidate_rows) >= 0.4
    )
    checkpoint = ROOT / "data/cache/diffsinger/checkpoints/gyu_score_native_zeroth3h5_replay/model_ckpt_steps_300.ckpt"
    report = {
        "status": "objective_probe_pass_human_pending" if eligible else "objective_reject_scaled_speech_prior",
        "human_listening": "pending" if eligible else "not_requested_objective_reject",
        "candidate": candidate,
        "eligible": eligible,
        "scale_signal_supports_one_more_stage": scale_signal,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "training_prior": "3.544 h balanced Zeroth-Korean speech plus VocalSet/real-GYU replay; inferred CTC timing",
        "score_native": True,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "aggregate": aggregate,
        "rows": rows,
        "release_allowed": False,
        "interpretation": "Automated metrics may reject this prior but cannot pass release listening.",
    }
    target = ROOT / "artifacts/reports/diffsinger_zeroth3h5_evaluation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    soulx_candidate = "soulx_melody32_verified"
    soulx_rows = [row for row in rows if row["model"] == soulx_candidate]
    soulx_eligible = (
        min(row["asr_lyric_similarity"] for row in soulx_rows) >= 0.8
        and min(row["voicing_accuracy"] for row in soulx_rows) >= 0.8
        and max(row["pitch_mae_cents"] for row in soulx_rows) <= 100
        and aggregate[soulx_candidate]["hf_spike_p99_over_median"]
        < aggregate["rc6"]["hf_spike_p99_over_median"]
    )
    soulx_report = {
        "status": "objective_probe_pass_human_pending" if soulx_eligible else "objective_reject_non_native_korean_projection",
        "human_listening": "pending" if soulx_eligible else "not_requested_objective_reject",
        "eligible": soulx_eligible,
        "candidate": soulx_candidate,
        "model": "official SoulX-Singer score-conditioned model.pt",
        "control": "canonical 50 Hz voiced F0, direct note duration, and projected phonemes",
        "korean_phone_projection": "deterministic English ARPAbet approximation; not native Korean supervision",
        "score_native": True,
        "per_note_tts": False,
        "waveform_pitch_shifting": False,
        "aggregate": {
            model: aggregate[model]
            for model in ("rc6", "zeroth1h_replay300", "soulx_score32", "soulx_score32_verified", soulx_candidate)
        },
        "rows": soulx_rows,
        "release_allowed": False,
        "interpretation": "This bounded cross-phone projection may be rejected objectively; passing still requires human listening.",
    }
    (ROOT / "artifacts/reports/soulx_score_native_ko_probe/evaluation.json").write_text(
        json.dumps(soulx_report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "status": report["status"],
        "soulx_status": soulx_report["status"],
        "scale_signal_supports_one_more_stage": scale_signal,
        "aggregate": aggregate,
        "transcripts": {f'{row["model"]}:{row["case"]}': row["asr_transcript"] for row in rows},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
