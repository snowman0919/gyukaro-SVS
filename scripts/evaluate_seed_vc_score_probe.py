#!/usr/bin/env python3
"""Gate a bounded Seed-VC conversion of the score-native Korean source."""
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


CASES = {"rapid_ko": "빠르게노래하자아", "large_interval_ko": "높이날아"}
MLP = ROOT / "artifacts/reports/mlp_singer_korean_probe/listening"
SEED = ROOT / "artifacts/reports/seed_vc_score_probe/listening"
VARIANTS = {
    "score_source_c6": {case: MLP / f"{case}_c6_generated_e2e.wav" for case in CASES},
    "soulx_conversion": {case: MLP / f"{case}_c6_soulx_s32_c2.wav" for case in CASES},
    "rvc_e5_plus15": {case: MLP / f"rvc_e5_plus15/{case}.wav" for case in CASES},
    "seed_vc_s30_c07": {
        case: SEED / f"vc_{case}_c6_generated_e2e_192_1.0_30_0.7.wav" for case in CASES
    },
    "seed_vc_s50_c03_trimmed": {
        case: SEED / f"vc_{case}_c6_generated_e2e_gyu_real_000146_1.0_50_0.3.wav"
        for case in CASES
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_16k(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    return (
        resample_poly(audio, 16000, sample_rate).astype("float32")
        if sample_rate != 16000 else audio
    )


def main() -> None:
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", torch_dtype=torch.float16
    ).cuda().eval()
    extractor = F0Extractor(
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
            a = wavlm(**{key: value.cuda() for key, value in values.items()}).embeddings
            b = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0).cuda())
        a = torch.nn.functional.normalize(a, dim=-1).squeeze().cpu().numpy()
        b = b.squeeze().cpu().numpy()
        b /= max(np.linalg.norm(b), 1e-8)
        return a, b

    reference = speaker(ROOT / "data/processed/master/216.wav")
    targets = {
        case: np.array(json.loads(
            (ROOT / f"artifacts/reports/diffsinger_score_native_pilot/{case}.ds").read_text()
        )[0]["f0_seq"].split(), dtype=np.float32)
        for case in CASES
    }
    rows = []
    for model, paths in VARIANTS.items():
        for case, path in paths.items():
            values = processor(audio=audio16(path), sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                ids = whisper.generate(
                    values.input_features.cuda().half(), language="ko", task="transcribe",
                    max_new_tokens=64,
                )
            transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
            embeddings = speaker(path)
            rows.append({
                "model": model,
                "case": case,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "asr_transcript": transcript,
                "asr_lyric_similarity": round(SequenceMatcher(
                    None, normalized(CASES[case]), normalized(transcript)
                ).ratio(), 4),
                "wavlm_to_gyu": round(float(np.dot(reference[0], embeddings[0])), 5),
                "ecapa_to_gyu": round(float(np.dot(reference[1], embeddings[1])), 5),
            } | acoustics(path) | pitch(path, targets[case], extractor))

    metrics = (
        "asr_lyric_similarity", "pitch_mae_cents", "voicing_accuracy",
        "hf_spike_p99_over_median", "spectral_flux_p95", "sample_jump_p999",
        "wavlm_to_gyu", "ecapa_to_gyu",
    )
    aggregate = {
        model: {metric: round(float(np.mean([
            row[metric] for row in rows
            if row["model"] == model and row[metric] is not None
        ])), 6) for metric in metrics}
        for model in VARIANTS
    }
    seed_candidates = ("seed_vc_s30_c07", "seed_vc_s50_c03_trimmed")
    candidate = max(seed_candidates, key=lambda model: (
        aggregate[model]["asr_lyric_similarity"],
        aggregate[model]["voicing_accuracy"],
        -aggregate[model]["hf_spike_p99_over_median"],
    ))
    seed_rows = [row for row in rows if row["model"] == candidate]
    eligible = (
        min(row["asr_lyric_similarity"] for row in seed_rows) >= 0.8
        and min(row["voicing_accuracy"] for row in seed_rows) >= 0.8
        and max(row["pitch_mae_cents"] for row in seed_rows) <= 50
        and aggregate[candidate]["hf_spike_p99_over_median"]
        < aggregate["soulx_conversion"]["hf_spike_p99_over_median"]
    )
    report = {
        "status": "objective_probe_pass_license_pending" if eligible else "objective_reject_seed_vc",
        "eligible": eligible,
        "candidate": candidate,
        "model": "Plachta/Seed-VC",
        "model_revision": "257283f9f41585055e8f858fba4fd044e5caed6e",
        "repository_revision": "51383efd921027683c89e5348211d93ff12ac2a8",
        "declared_license": "GPL-3.0",
        "training_data_provenance_documented": False,
        "evaluation_only": True,
        "score_source_is_csd_noncommercial": True,
        "production_integration_allowed": False,
        "aggregate": aggregate,
        "rows": rows,
        "release_allowed": False,
    }
    target = ROOT / "artifacts/reports/seed_vc_score_probe/evaluation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
