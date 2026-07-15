#!/usr/bin/env python3
"""Objective diagnostics for the canonical-timeline candidate; never passes listening."""
from __future__ import annotations

import argparse
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


def mean(rows: list[dict], name: str) -> float:
    return round(float(np.mean([row[name] for row in rows if row.get(name) is not None])), 6)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="artifacts/reports/rc5_candidate_core"); parser.add_argument("--rc4", default="artifacts/reports/rc5_isolation")
    args = parser.parse_args(); root, rc4 = Path(args.root), Path(args.rc4); manifest = json.loads((root / "manifest.json").read_text())
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows, source_alignment = [], {}
    for case, data in manifest["cases"].items():
        target = np.load(root / case / "canonical_f0.npy")
        source = Path(json.loads((rc4 / "matrix.json").read_text())["cases"][case]["matrix"]["A"]["path"])
        source_alignment[case] = pitch(source, target, extractor)
        for label, item in data["renders"].items():
            path = Path(item["path"]); contour = np.load(rc4 / case / "production_f0.npy") if label == "rc4" else target
            rows.append({"case": case, "label": label, "path": str(path)} | acoustics(path) | pitch(path, contour, extractor))
    del extractor; torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    for row in rows:
        score = json.loads(Path(manifest["cases"][row["case"]]["score"]).read_text()); expected = " ".join(note["lyric"] for note in score["notes"])
        inputs = processor(audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): ids = asr.generate(inputs.input_features.cuda().half(), language=score["language"], task="transcribe", max_new_tokens=64)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_transcript"], row["asr_lyric_similarity"] = transcript, round(SequenceMatcher(None, normalized(expected), normalized(transcript)).ratio(), 4)
    names = ("pitch_mae_cents", "voicing_accuracy", "observed_voiced_ratio", "hf_energy_ratio_p95", "hf_spike_p99_over_median", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999", "clip_fraction", "asr_lyric_similarity")
    aggregate = {label: {name: mean([row for row in rows if row["label"] == label], name) for name in names} for label in {row["label"] for row in rows}}
    before, after = aggregate["rc4"], aggregate["fixed_full"]
    delta = {name: round(after[name] - before[name], 6) for name in names}
    listening = root / "listening_before_after"; shutil.rmtree(listening, ignore_errors=True); listening.mkdir()
    for row in rows:
        if row["label"] in {"rc4", "fixed_off_off", "fixed_full"}: shutil.copy2(row["path"], listening / f"{row['case']}_{row['label']}.wav")
    report = {"status": "objective_pass_human_pending", "metric_warning": "objective diagnostics cannot pass human listening", "source_timing_alignment": source_alignment,
              "aggregate": aggregate, "fixed_full_minus_rc4": delta, "rows": rows, "listening_directory": str(listening),
              "interpretation": {"primary": "RC4 all-frame voiced F0 plus low-step/high-CFG SoulX decode", "adapter_gate": "retain only if fixed_full listening benefit exceeds fixed_off_off; objective deltas alone are insufficient", "content_timing": "OmniVoice has no score timing input; source/target voicing agreement is diagnostic evidence, not corrected CTC alignment"}}
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate, "fixed_full_minus_rc4": delta, "source_timing_alignment": source_alignment, "listening_directory": str(listening)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
