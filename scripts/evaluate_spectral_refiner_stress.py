#!/usr/bin/env python3
"""Gate spectral-refiner strengths on the fixed nine-file human-failed RC6 set."""
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
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


VARIANTS = {
    "rc6": ROOT / "artifacts/reports/rc6_backend_candidate/manifest.json",
    **{
        f"spectral_s{strength}": ROOT / f"artifacts/reports/spectral_refiner_stress_s{strength}/manifest.json"
        for strength in ("010", "025", "050", "100")
    },
    **{
        f"spectral_gyu_s{strength}": ROOT / f"artifacts/reports/spectral_gyu_refiner_stress_s{strength}/manifest.json"
        for strength in ("050", "100")
    },
}
STRESS = {"rapid_ko", "large_interval_ko"}


def audio_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    return resample_poly(mono, 16000, rate).astype("float32") if rate != 16000 else mono


def main() -> None:
    manifests = {name: json.loads(path.read_text()) for name, path in VARIANTS.items()}
    scores = {
        case: json.loads((ROOT / row["score"]).read_text())
        for case, row in manifests["rc6"]["files"].items()
    }
    targets = {
        case: np.load(ROOT / f"artifacts/reports/rc6_backend_candidate/{case}_target_f0.npy")
        for case in scores
    }
    extractor = F0Extractor(
        str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cuda", target_sr=24000, hop_size=480, verbose=False,
    )
    rows = []
    for variant, manifest in manifests.items():
        for case, item in manifest["files"].items():
            path = ROOT / item["path"]
            rows.append({
                "variant": variant, "case": case, "path": item["path"],
                "sample_rate": sf.info(path).samplerate,
            } | acoustics(path) | pitch(path, targets[case], extractor))
    del extractor
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        CACHE / "whisper-large-v3-turbo", dtype=torch.float16
    ).cuda().eval()
    wavlm_processor = AutoFeatureExtractor.from_pretrained(CACHE / "wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained(CACHE / "wavlm-base-plus-sv").cuda().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(CACHE / "spkrec-ecapa-voxceleb"),
        savedir=str(CACHE / "spkrec-ecapa-voxceleb"), run_opts={"device": "cuda:0"},
    )

    def speaker(path: Path) -> tuple[np.ndarray, np.ndarray]:
        audio = audio_16k(path)
        inputs = wavlm_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            a = wavlm(**{key: value.cuda() for key, value in inputs.items()}).embeddings
            b = ecapa.encode_batch(torch.from_numpy(audio)[None].cuda())
        a = torch.nn.functional.normalize(a, dim=-1).squeeze().cpu().numpy()
        b = b.squeeze().cpu().numpy(); b /= max(np.linalg.norm(b), 1e-8)
        return a, b

    reference = speaker(ROOT / "data/processed/master/216.wav")
    for index, row in enumerate(rows, 1):
        score = scores[row["case"]]
        expected = normalized("".join(note["lyric"] for note in score["notes"]))
        path = ROOT / row["path"]
        inputs = processor(audio16(path), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            ids = whisper.generate(
                inputs.input_features.cuda().half(), language=score["language"],
                task="transcribe", max_new_tokens=64,
            )
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        matcher = SequenceMatcher(None, expected, normalized(transcript))
        embeddings = speaker(path)
        row.update({
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(matcher.ratio(), 4),
            "asr_lyric_coverage": round(
                sum(block.size for block in matcher.get_matching_blocks()) / max(len(expected), 1), 4
            ),
            "wavlm_to_gyu": round(float(np.dot(reference[0], embeddings[0])), 5),
            "ecapa_to_gyu": round(float(np.dot(reference[1], embeddings[1])), 5),
        })
        print(f"{index}/{len(rows)} {row['variant']} {row['case']}", flush=True)

    metrics = (
        "pitch_mae_cents", "voicing_accuracy", "hf_energy_ratio_p95",
        "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95",
        "sample_jump_p999", "clip_fraction", "asr_lyric_similarity",
        "asr_lyric_coverage", "wavlm_to_gyu", "ecapa_to_gyu",
    )
    aggregate = {}
    stress = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        aggregate[variant] = {
            metric: round(float(np.mean([row[metric] for row in selected if row[metric] is not None])), 6)
            for metric in metrics
        }
        selected = [row for row in selected if row["case"] in STRESS]
        stress[variant] = {
            metric: round(float(np.mean([row[metric] for row in selected if row[metric] is not None])), 6)
            for metric in metrics
        }
    baseline = aggregate["rc6"]
    candidates = []
    for variant in VARIANTS:
        if variant == "rc6":
            continue
        selected = [row for row in rows if row["variant"] == variant]
        if (
            aggregate[variant]["asr_lyric_similarity"] >= baseline["asr_lyric_similarity"] - 0.01
            and aggregate[variant]["voicing_accuracy"] >= baseline["voicing_accuracy"] - 0.01
            and aggregate[variant]["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2.0
            and min(row["asr_lyric_coverage"] for row in selected) >= 0.8
            and max(row["clip_fraction"] for row in selected) == 0
        ):
            candidates.append(variant)
    candidate = min(
        candidates,
        key=lambda name: aggregate[name]["hf_spike_p99_over_median"],
        default=None,
    )
    materially_improved = candidate is not None and (
        aggregate[candidate]["hf_spike_p99_over_median"]
        <= 0.85 * baseline["hf_spike_p99_over_median"]
    )
    report = {
        "status": "objective_candidate_human_pending" if materially_improved else "objective_reject_no_material_stress_gain",
        "candidate": candidate, "materially_improved": materially_improved,
        "human_listening": "pending" if materially_improved else "not_requested_objective_reject",
        "baseline": "RC6 human-failed; preserved unchanged",
        "aggregate_9": aggregate, "stress_rapid_interval": stress,
        "rows": rows, "release_allowed": False,
        "gate": {
            "asr_mean_regression_max": 0.01, "voicing_regression_max": 0.01,
            "pitch_regression_max_cents": 2.0, "minimum_per_file_lyric_coverage": 0.8,
            "minimum_hf_spike_reduction": 0.15, "human_listening_required": True,
        },
    }
    target = ROOT / "artifacts/reports/acoustic_refiner_spectral_stress_evaluation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "candidate": candidate,
        "aggregate_9": aggregate, "stress_rapid_interval": stress,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
