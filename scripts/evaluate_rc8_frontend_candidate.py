#!/usr/bin/env python3
"""Gate the bounded EN/JA frontend candidate against frozen RC7."""
from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("GYU_SINGER_CACHE", ROOT / "data/cache"))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(CACHE / "soulx-singer")]

from analyze_rc8_defects import load as load48, phone_metrics  # noqa: E402
from evaluate_rc4_artifact_matrix import acoustics, audio16, normalized, pitch  # noqa: E402
from gyu_singer.data import acoustic_reference_features  # noqa: E402
from gyu_singer.inference.quality_controller import QualityPitchController  # noqa: E402
from gyu_singer.inference.soulx import SoulXPhraseRenderer  # noqa: E402
from gyu_singer.score import normalize_score  # noqa: E402
from preprocess.tools.f0_extraction import F0Extractor  # noqa: E402


CASES = {
    "en": ("examples/quality_en.json", "artifacts/reports/rc7_listening_gate/04_en.wav"),
    "ja": ("examples/quality_ja.json", "artifacts/reports/rc7_listening_gate/05_ja.wav"),
}


def main() -> None:
    root = ROOT / "artifacts/reports/rc8_frontend_candidate"
    manifest = json.loads((root / "manifest.json").read_text())
    controller = QualityPitchController(ROOT / "checkpoints/gyu_prosody_v0.5.pt", acoustic_reference_features(ROOT / "data/processed/master/216.wav"))
    targets, scores = {}, {}
    for case, (score_path, _) in CASES.items():
        score = normalize_score(json.loads((ROOT / score_path).read_text()))
        duration = sf.info(ROOT / manifest["rows"][case]["candidate_path"]).duration
        expressive = controller.predict(score, canonical_timing=True)[0] * score["style"]["prosody_strength"]
        targets[case], _ = SoulXPhraseRenderer._canonical_f0(score, duration, expressive.cpu().numpy())
        scores[case] = score
    del controller
    torch.cuda.empty_cache()
    extractor = F0Extractor(str(CACHE / "soulx-singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"), device="cuda", target_sr=24000, hop_size=480, verbose=False)
    rows = []
    for case, (score_path, baseline_path) in CASES.items():
        paths = {"rc7": ROOT / baseline_path, "frontend_fixed": ROOT / manifest["rows"][case]["candidate_path"]}
        for variant, path in paths.items():
            rows.append({"case": case, "variant": variant, "path": str(path.relative_to(ROOT))} | acoustics(path) | pitch(path, targets[case], extractor) | {"phones": phone_metrics(load48(path), ROOT / score_path)})
    del extractor
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(CACHE / "whisper-large-v3-turbo")
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(CACHE / "whisper-large-v3-turbo", dtype=torch.float16).cuda().eval()
    for row in rows:
        score = scores[row["case"]]
        inputs = processor(audio16(ROOT / row["path"]), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            ids = asr.generate(inputs.input_features.cuda().half(), language=score["language"], task="transcribe", max_new_tokens=64)
        row["asr_transcript"] = processor.batch_decode(ids, skip_special_tokens=True)[0]
        expected = normalized("".join(note["lyric"] for note in score["notes"]))
        matcher = SequenceMatcher(None, expected, normalized(row["asr_transcript"]))
        row["asr_lyric_similarity"] = round(matcher.ratio(), 4)
    gate = {}
    for case in CASES:
        baseline = next(row for row in rows if row["case"] == case and row["variant"] == "rc7")
        candidate = next(row for row in rows if row["case"] == case and row["variant"] == "frontend_fixed")
        gate[case] = {
            "pass_objective_non_regression": candidate["asr_lyric_similarity"] >= baseline["asr_lyric_similarity"] - .02 and candidate["pitch_mae_cents"] <= baseline["pitch_mae_cents"] + 2 and candidate["voicing_accuracy"] >= baseline["voicing_accuracy"] - .01 and candidate["clip_fraction"] == 0,
            "asr_delta": round(candidate["asr_lyric_similarity"] - baseline["asr_lyric_similarity"], 4),
            "pitch_delta_cents": round(candidate["pitch_mae_cents"] - baseline["pitch_mae_cents"], 2),
            "voicing_delta": round(candidate["voicing_accuracy"] - baseline["voicing_accuracy"], 4),
            "weak_phone_delta": candidate["phones"]["weak_phone_count"] - baseline["phones"]["weak_phone_count"],
            "voicing_mismatch_delta": candidate["phones"]["voicing_mismatch_count"] - baseline["phones"]["voicing_mismatch_count"],
        }
    report = {"status": "human_listening_pending" if all(row["pass_objective_non_regression"] for row in gate.values()) else "objective_reject", "gate": gate, "rows": rows, "human_listening_required": True}
    (root / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
