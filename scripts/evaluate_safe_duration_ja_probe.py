#!/usr/bin/env python3
"""Gate the bounded JA safe-duration diagnostic; never changes RC8 runtime."""


def candidate_gates(current: dict, candidate: dict) -> dict[str, bool]:
    return {
        "heldout_lyric_similarity_at_least_090": candidate["asr_lyric_similarity"] >= .90,
        "heldout_repetition_removed": not candidate["repetition_detected"],
        "pitch_nonregression": candidate["pitch_mae_cents"] <= current["pitch_mae_cents"] + 2,
        "voicing_nonregression": candidate["voicing_accuracy"] >= current["voicing_accuracy"] - .01,
        "hf_spike_nonregression": candidate["hf_spike_p99_over_median"] <= current["hf_spike_p99_over_median"] * 1.05,
        "sample_jump_nonregression": candidate["sample_jump_p999"] <= current["sample_jump_p999"] * 1.05,
        "wavlm_identity_nonregression": candidate["wavlm_to_gyu"] >= current["wavlm_to_gyu"] - .02,
        "ecapa_identity_nonregression": candidate["ecapa_to_gyu"] >= current["ecapa_to_gyu"] - .02,
    }


if __name__ == "__main__":
    import hashlib
    import json
    import sys
    from difflib import SequenceMatcher
    from pathlib import Path

    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioXVector,
    )

    root = Path(__file__).resolve().parents[1]
    cache = root / "data/cache"
    report_root = root / "artifacts/reports/omnivoice_safe_duration_ja"
    sys.path[:0] = [str(root / "scripts"), str(cache / "soulx-singer")]
    from analyze_rc8_defects import metrics as multires_metrics  # noqa: E402
    from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
    from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402
    from probe_rc8_ja_duplicate_span import plot_case  # noqa: E402

    expected = "新しい歌を風に乗せて届ける"
    paths = {
        "current_fp16_raw": report_root / "current_fp16_raw.wav",
        "candidate_fp16_raw": report_root / "candidate_fp16_raw.wav",
        "current_fp16_refined": report_root / "current_fp16_refined.wav",
        "candidate_fp16_refined": report_root / "candidate_fp16_refined.wav",
        "current_fp32_raw": report_root / "current_fp32_raw.wav",
        "candidate_fp32_raw": report_root / "candidate_fp32_raw.wav",
        "current_fp32_refined": report_root / "current_fp32_refined.wav",
        "candidate_fp32_refined": report_root / "candidate_fp32_refined.wav",
    }
    target_f0 = np.load(root / "artifacts/reports/rc8_ja_duplicate_span/heldout_ja/target_f0.npy")
    f0 = F0Extractor(
        str(cache / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"),
        device="cpu", target_sr=24_000, hop_size=480, verbose=False,
    )
    rows = {
        name: acoustics(path) | pitch(path, target_f0, f0)
        | multires_metrics(path, root / "examples/heldout_ja.json")
        for name, path in paths.items()
    }
    del f0

    whisper_rows = {}
    for name in ("whisper_early_gate.json", "whisper_refined_gate.json", "whisper_fp32_gate.json"):
        whisper_rows |= {row["file"]: row for row in json.loads((report_root / name).read_text())["rows"]}
    for name, path in paths.items():
        transcript = whisper_rows[path.name]["transcript"]
        actual = normalized(transcript)
        rows[name] |= {
            "asr_transcript": transcript,
            "asr_lyric_similarity": round(SequenceMatcher(None, normalized(expected), actual).ratio(), 4),
            "repetition_detected": len(actual) > 1.25 * len(normalized(expected)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    wavlm_processor = AutoFeatureExtractor.from_pretrained(cache / "wavlm-base-plus-sv", local_files_only=True)
    wavlm = AutoModelForAudioXVector.from_pretrained(cache / "wavlm-base-plus-sv", local_files_only=True).cpu().eval()
    ecapa = EncoderClassifier.from_hparams(
        source=str(cache / "spkrec-ecapa-voxceleb"), savedir=str(cache / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )
    embeddings = {}
    for name, path in ({"reference": root / "data/processed/master/216.wav"} | paths).items():
        audio = audio16(path)
        values = wavlm_processor(audio, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            wavlm_value = torch.nn.functional.normalize(wavlm(**values).embeddings, dim=-1).squeeze().numpy()
            ecapa_value = ecapa.encode_batch(torch.from_numpy(audio)[None]).squeeze().numpy()
        embeddings[name] = (wavlm_value, ecapa_value / max(np.linalg.norm(ecapa_value), 1e-8))
    for name in paths:
        rows[name] |= {
            "wavlm_to_gyu": round(float(np.dot(embeddings["reference"][0], embeddings[name][0])), 5),
            "ecapa_to_gyu": round(float(np.dot(embeddings["reference"][1], embeddings[name][1])), 5),
        }

    alignment = json.loads((report_root / "ctc_alignment.json").read_text())
    phones = alignment["phones"]
    unknown = np.array(["unknown" in phone["symbol"] for phone in phones])
    confidence = np.array([phone["ctc_mean_log_score"] >= -2.0 for phone in phones]) & ~unknown
    durations = np.array([phone["target_end"] - phone["target_start"] for phone in phones])
    target_centers = np.array([(phone["target_start"] + phone["target_end"]) / 2 for phone in phones])
    source_centers = np.array([(phone["source_start"] + phone["source_end"]) / 2 for phone in phones])
    collapse = json.loads((root / "artifacts/reports/omnivoice_ja_duration_collapse/evaluation.json").read_text())
    source_rows = [
        row for row in collapse["rows"]
        if row["case"] == "heldout_full" and row["duration"] in {6.6, 8.9}
    ]
    pair_gates = {
        precision: candidate_gates(rows[f"current_{precision}"], rows[f"candidate_{precision}"])
        for precision in ("fp16_raw", "fp16_refined", "fp32_raw", "fp32_refined")
    }
    baseline = json.loads((root / "artifacts/reports/rc8_candidate3_full/manifest.json").read_text())
    regression = {}
    for case, item in baseline["files"].items():
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        regression[case] = {"expected": item["sha256"], "actual": actual, "unchanged": actual == item["sha256"]}
    plot = plot_case(report_root, paths)
    report = {
        "status": "diagnostic_candidate_human_pending" if all(all(gates.values()) for gates in pair_gates.values()) else "diagnostic_reject",
        "runtime_integrated": False,
        "human_listening": "pending",
        "scope": "JA non-rapid heldout diagnostic only; quality JA and Rapid JA/KO are not routed",
        "hypothesis": "avoid OmniVoice repetition collapse by generating an exact shorter phrase, padding only the SoulX carrier, and monotonically remapping content latent to the score timeline",
        "source_whisper": source_rows,
        "ctc_evidence": {
            "method": alignment["method"],
            "diagnostic_human_verified_kana": ["あたらしいうたを", "かぜにのせて", "とどけ", "る"],
            "unknown_phoneme_ratio": round(float(unknown.mean()), 6),
            "target_phoneme_coverage": round(float(durations[confidence].sum() / durations.sum()), 6),
            "monotonic": bool(np.all(np.diff(target_centers) > 0) and np.all(np.diff(source_centers) > 0)),
            "mean_log_score": round(float(alignment["mean_log_score"]), 6),
            "removed_source_spans": [],
            "mapping": "CTC-aligned safe 6.6 s latent -> monotonic 8.9 s score timeline; padded carrier is not final WAV stitching",
        },
        "decoder": {
            "precision": ["fp16 diagnostic", "fp32 production recheck"], "n_steps": 64, "cfg": 2.0, "seed": 21,
            "refiners": "unchanged RC8 acoustic 0.25 then spectral 0.5",
        },
        "rows": rows,
        "waveform_multires_stft": plot,
        "existing_9_file_regression": regression,
        "gates": {
            **pair_gates,
            "quality_ja_nonregression": regression["ja"]["unchanged"],
            "existing_9_file_runtime_unchanged": all(item["unchanged"] for item in regression.values()),
            "production_fp32_and_refiners_verified": all(pair_gates["fp32_raw"].values()) and all(pair_gates["fp32_refined"].values()),
        },
        "constraints": {
            "phrase_level_soulx_decode": True,
            "per_note_tts": False,
            "final_wav_chunk_stitching": False,
            "waveform_pitch_shift": False,
        },
        "next_step": "human-listen current_fp32_refined.wav versus candidate_fp32_refined.wav; do not propose runtime integration without an explicit pass",
    }
    (report_root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
