#!/usr/bin/env python3
"""Evaluate hidden-level CTC timing against RC4 and the rejected F0-only fix."""
from __future__ import annotations

import json
import os
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch

CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")); sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor


def main() -> None:
    root, fixed = Path("artifacts/reports/rc5_latent_timing"), Path("artifacts/reports/rc5_candidate_core"); manifest = json.loads((root / "manifest.json").read_text()); prior = json.loads((fixed / "evaluation.json").read_text())
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False); rows = []
    for case, data in manifest["cases"].items():
        target = np.load(fixed / case / "canonical_f0.npy")
        for label, item in data["renders"].items(): rows.append({"case": case, "label": label, "path": item["path"]} | acoustics(Path(item["path"])) | pitch(Path(item["path"]), target, extractor))
    del extractor; torch.cuda.empty_cache(); processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo"); model = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    for row in rows:
        score = json.loads(Path(manifest["cases"][row["case"]]["score"]).read_text()); expected = " ".join(note["lyric"] for note in score["notes"]); inputs = processor(audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): ids = model.generate(inputs.input_features.cuda().half(), language=score["language"], task="transcribe", max_new_tokens=64)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]; row["asr_lyric_similarity"] = round(SequenceMatcher(None, normalized(expected), normalized(row["asr_transcript"])).ratio(), 4)
    names = ("pitch_mae_cents", "voicing_accuracy", "observed_voiced_ratio", "hf_energy_ratio_p95", "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999", "clip_fraction", "asr_lyric_similarity")
    aggregate = {label: {name: round(float(np.mean([row[name] for row in rows if row["label"] == label and row[name] is not None])), 6) for name in names} for label in ("latent_timing_off_off", "latent_timing_full")}; aggregate = {"rc4": prior["aggregate"]["rc4"], "f0_only_full_rejected": prior["aggregate"]["fixed_full"]} | aggregate
    after, before = aggregate["latent_timing_full"], aggregate["rc4"]; delta = {name: round(after[name] - before[name], 6) for name in names}
    listening = root / "listening_before_after"; shutil.rmtree(listening, ignore_errors=True); listening.mkdir()
    for case, data in manifest["cases"].items():
        shutil.copy2(next(row["path"] for row in prior["rows"] if row["case"] == case and row["label"] == "rc4"), listening / f"{case}_rc4.wav")
        for label, item in data["renders"].items(): shutil.copy2(item["path"], listening / f"{case}_{label}.wav")
    status = "objective_candidate_human_pending" if after["asr_lyric_similarity"] >= before["asr_lyric_similarity"] - .03 and after["sample_jump_p999"] < before["sample_jump_p999"] else "rejected"
    report = {"status": status, "metric_warning": "human listening remains mandatory", "aggregate": aggregate, "latent_full_minus_rc4": delta, "rows": rows, "listening_directory": str(listening)}; (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(report | {"rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
