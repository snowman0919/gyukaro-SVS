#!/usr/bin/env python3
"""Compare the score-guide ACE phrase source with the frozen OmniVoice source."""
from __future__ import annotations

import json
import os
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
    root = Path("artifacts/reports/rc5_ace_score_source"); manifest = json.loads((root / "manifest.json").read_text()); matrix = json.loads(Path("artifacts/reports/rc5_isolation/matrix.json").read_text())
    targets = Path("artifacts/reports/rc5_candidate_core"); extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for case, data in manifest["cases"].items():
        target = np.load(targets / case / "canonical_f0.npy")
        for label, path in (("omnivoice", Path(matrix["cases"][case]["matrix"]["A"]["path"])), ("ace_score", Path(data["source"]))):
            rows.append({"case": case, "label": label, "path": str(path)} | acoustics(path) | pitch(path, target, extractor))
    del extractor; torch.cuda.empty_cache(); processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo"); model = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    for row in rows:
        score = json.loads(Path(manifest["cases"][row["case"]]["score"]).read_text()); expected = " ".join(note["lyric"] for note in score["notes"]); inputs = processor(audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): ids = model.generate(inputs.input_features.cuda().half(), language=score["language"], task="transcribe", max_new_tokens=64)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]; row["asr_lyric_similarity"] = round(SequenceMatcher(None, normalized(expected), normalized(row["asr_transcript"])).ratio(), 4)
    metrics = ("pitch_mae_cents", "voicing_accuracy", "observed_voiced_ratio", "hf_energy_ratio_p95", "spectral_flatness_mean", "spectral_flux_p95", "sample_jump_p999", "asr_lyric_similarity")
    aggregate = {label: {name: round(float(np.mean([row[name] for row in rows if row["label"] == label and row[name] is not None])), 6) for name in metrics} for label in ("omnivoice", "ace_score")}
    ace = aggregate["ace_score"]; decision = "eligible_for_soulx_probe" if ace["asr_lyric_similarity"] >= .8 and ace["voicing_accuracy"] >= aggregate["omnivoice"]["voicing_accuracy"] else "reject"
    report = {"status": "measured", "decision": decision, "aggregate": aggregate, "rows": rows, "warning": "source diagnostics only; downstream SoulX and listening are mandatory"}; (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(report | {"rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
