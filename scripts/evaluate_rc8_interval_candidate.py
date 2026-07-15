#!/usr/bin/env python3
"""Gate 50-step large-interval candidates against frozen RC7."""
from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import interval_f0  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_interval_candidate"
    manifest = json.loads((root / "manifest.json").read_text())
    variants = [{"variant": "rc7", "strength": .5, "path": "artifacts/reports/rc7_listening_gate/08_large_interval_ko.wav"}] + [row | {"variant": f"s50_spectral_{row['strength']:g}"} for row in manifest["rows"]]
    target = np.load(ROOT / "artifacts/reports/rc6_backend_candidate/large_interval_ko_target_f0.npy")
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for variant in variants:
        path = ROOT / variant["path"]
        rows.append({"variant": variant["variant"], "strength": variant["strength"], "path": variant["path"]} | acoustics(path) | pitch(path, target, extractor) | {"failure_region": interval_f0(path)})
    del extractor
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    expected = normalized("높이날아")
    for row in rows:
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=32)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]
        row["asr_lyric_similarity"] = round(SequenceMatcher(None, expected, normalized(row["asr_transcript"])).ratio(), 4)
    baseline = rows[0]
    eligible = [row for row in rows[1:] if row["asr_lyric_similarity"] >= .99 and row["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2 and row["voicing_accuracy"] >= baseline["voicing_accuracy"] - .01 and row["clip_fraction"] == 0]
    selected = min(eligible, key=lambda row: (abs(row["failure_region"]["pyin_median_hz"] - 387.3), row["sample_jump_p999"], row["hf_spike_p99_over_median"]), default=None)
    report = {
        "status": "human_listening_pending" if selected else "objective_reject",
        "selection": None if not selected else selected["variant"],
        "root_cause": "selected SoulX decode, before either refiner; dominant competing harmonic trajectory appears in the first ascending boundary",
        "rows": rows, "human_listening_required": True,
    }
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "selection": report["selection"], "eligible": [row["variant"] for row in eligible]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
