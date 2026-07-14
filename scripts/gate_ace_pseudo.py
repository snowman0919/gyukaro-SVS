#!/usr/bin/env python3
"""Quality-gate ACE-Step→SoulX pseudo singing with RMVPE, WavLM, Whisper, and LID."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector, AutoModelForSpeechSeq2Seq, AutoProcessor

sys.path.insert(0, "data/cache/soulx-singer")
from preprocess.tools.f0_extraction import F0Extractor
from prepare_hybrid_data import inferred_score


def audio16(path: str) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True); mono = audio.mean(1)
    return resample_poly(mono, 16000, rate).astype("float32") if rate != 16000 else mono


def similarity(expected: str, actual: str) -> float:
    expected, actual = re.sub(r"\W", "", expected).lower(), re.sub(r"\W", "", actual).lower()
    if not expected or not actual: return 0.0
    previous = list(range(len(actual) + 1))
    for char in expected:
        current = [previous[0] + 1]
        for index, other in enumerate(actual): current.append(min(current[-1] + 1, previous[index + 1] + 1, previous[index] + (char != other)))
        previous = current
    return round(1 - previous[-1] / max(len(expected), len(actual)), 4)


def language(text: str) -> str | None:
    if re.search(r"[가-힣]", text): return "ko"
    if re.search(r"[ぁ-んァ-ン一-龯]", text): return "ja"
    if re.search(r"[a-zA-Z]", text): return "en"
    return None


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--input", default="data/manifests/ace_step_candidates.jsonl")
    parser.add_argument("--candidates", default="data/manifests/pseudo_singing_candidates.jsonl")
    parser.add_argument("--accepted", default="data/manifests/pseudo_singing_accepted.jsonl")
    parser.add_argument("--reference", default="data/processed/master/216.wav")
    args = parser.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained("data/cache/wavlm-base-plus-sv")
    wavlm = AutoModelForAudioXVector.from_pretrained("data/cache/wavlm-base-plus-sv").to(device).eval()
    def embed(audio: np.ndarray) -> np.ndarray:
        value = extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): result = wavlm(**{key: item.to(device) for key, item in value.items()}).embeddings
        return torch.nn.functional.normalize(result, dim=-1).squeeze().cpu().numpy()
    reference = embed(audio16(args.reference))
    processor = AutoProcessor.from_pretrained("data/cache/whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained("data/cache/whisper-large-v3-turbo", dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    rmvpe = F0Extractor("data/cache/soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt", device=device, target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for source in (json.loads(line) for line in Path(args.input).read_text().splitlines() if line):
        output = source["output_path"]; teacher = source["source_output_path"]
        target_f0, result_f0 = rmvpe.process(teacher, verbose=False), rmvpe.process(output, verbose=False)
        a, b = np.log2(target_f0[target_f0 > 1]), np.log2(result_f0[result_f0 > 1])
        correlation = 0.0 if min(len(a), len(b)) < 3 else float(np.corrcoef(np.interp(np.linspace(0, len(a) - 1, len(b)), np.arange(len(a)), a), b)[0, 1])
        source_audio, source_rate = sf.read(teacher, dtype="float32"); output_audio, output_rate = sf.read(output, dtype="float32")
        inputs = processor(audio16(output), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): tokens = asr.generate(inputs.input_features.to(device, torch.float16 if device == "cuda" else torch.float32), language=source["language"], task="transcribe", max_new_tokens=64)
        transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        content = similarity(source["text"], transcript); speaker = float(np.dot(reference, embed(audio16(output))))
        duration = len(output_audio) / output_rate / (len(source_audio) / source_rate)
        accepted = correlation >= .90 and .85 <= duration <= 1.15 and speaker >= .65 and content >= .10 and language(transcript) == source["language"]
        median_f0 = float(np.median(target_f0[target_f0 > 1])) if np.any(target_f0 > 1) else 220.0
        row = source | {"duration_sec": round(len(output_audio) / output_rate, 5), "f0_median_hz": round(median_f0, 3), "f0_contour_correlation_rmvpe": round(correlation, 4), "duration_ratio": round(duration, 4), "speaker_score": round(speaker, 4), "asr_transcript": transcript, "content_score": content, "language_detected": language(transcript), "training_license": "allowed", "trust_weight": .20 if accepted else 0.0, "quality_status": "accepted" if accepted else "rejected_gate", "training_use": "synthetic_pseudo_singing_acoustic_low_trust" if accepted else "evaluation_only_not_training"}
        if accepted:
            row["score"] = inferred_score(row | {"score_source": "inferred_from_RMVPE_pitch_median_and_duration_not_ground_truth"})
        rows.append(row); print(source["id"], row["quality_status"], flush=True)
    Path(args.candidates).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    Path(args.accepted).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row["quality_status"] == "accepted"))


if __name__ == "__main__":
    main()
