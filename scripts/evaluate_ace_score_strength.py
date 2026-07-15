#!/usr/bin/env python3
from __future__ import annotations

import json, os, sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch

CACHE = Path(os.environ.get("GYU_SINGER_CACHE", "data/cache")); sys.path.insert(0, str(CACHE / "soulx-singer"))
from preprocess.tools.f0_extraction import F0Extractor

def main():
    root = Path("artifacts/reports/rc5_ace_score_strength"); data = json.loads((root / "manifest.json").read_text()); target = np.load("artifacts/reports/rc5_candidate_core/large_interval_ko/canonical_f0.npy")
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False); rows = [row | acoustics(Path(row["path"])) | pitch(Path(row["path"]), target, extractor) for row in data["rows"]]; del extractor; torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo"); model = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval(); expected = "높이날아"
    for row in rows:
        inputs = processor(audio16(Path(row["path"])), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode(): ids = model.generate(inputs.input_features.cuda().half(), language="ko", task="transcribe", max_new_tokens=64)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]; row["asr_lyric_similarity"] = round(SequenceMatcher(None, expected, normalized(row["asr_transcript"])).ratio(), 4)
    selected = max(rows, key=lambda row: (row["asr_lyric_similarity"], row["voicing_accuracy"])); report = {"status": "eligible_for_soulx_probe" if selected["asr_lyric_similarity"] >= .8 else "reject", "selected_strength": selected["strength"], "rows": rows}; (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
