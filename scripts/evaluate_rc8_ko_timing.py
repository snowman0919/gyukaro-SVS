#!/usr/bin/env python3
"""Evaluate normal-KO timing sweep; inferred phone metrics cannot replace listening."""
from __future__ import annotations

import json
import os
import sys
import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import load as load48, phone_metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="append", default=[], metavar="LABEL=PATH",
        help="Add a rendered timing candidate to the fixed sweep.",
    )
    args = parser.parse_args()
    root = ROOT / "artifacts/reports/rc8_ko_timing_sweep"
    manifest = json.loads((root / "manifest.json").read_text())
    score_path = ROOT / "examples/quality_ko.json"
    expected = normalized("".join(note["lyric"] for note in json.loads(score_path.read_text())["notes"]))
    variants = [{"variant": "rc7", "strength": None, "path": "artifacts/reports/rc7_listening_gate/01_ko_neutral.wav"}] + [row | {"variant": f"warp_{row['strength']:g}"} for row in manifest["rows"]]
    for value in args.candidate:
        label, path = value.split("=", 1)
        variants.append({"variant": label, "strength": None, "path": path})
    target = np.load(ROOT / "artifacts/reports/rc6_backend_candidate/ko_neutral_target_f0.npy")
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for variant in variants:
        path = ROOT / variant["path"]
        rows.append({"variant": variant["variant"], "strength": variant.get("strength"), "path": variant["path"]} | acoustics(path) | pitch(path, target, extractor) | {"phones": phone_metrics(load48(path), score_path)})
    del extractor
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    for row in rows:
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_lyric_similarity"] = round(SequenceMatcher(None, expected, normalized(row["asr_transcript"])).ratio(), 4)
    baseline = rows[0]
    eligible = [row for row in rows[1:] if row["asr_lyric_similarity"] >= baseline["asr_lyric_similarity"] - .02 and row["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2 and row["voicing_accuracy"] >= baseline["voicing_accuracy"] - .01 and row["clip_fraction"] == 0]
    selected = min(eligible, key=lambda row: (row["phones"]["voicing_mismatch_count"], row["spectral_flux_p95"]), default=None)
    report = {
        "status": "human_listening_pending" if selected else "objective_reject",
        "selection": None if not selected else selected["variant"],
        "selection_warning": "phone spans are score-inferred; listening must confirm reduced overconnection without staccato",
        "rapid_ko_protected": True, "rows": rows,
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "eligible": [row["variant"] for row in eligible]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
