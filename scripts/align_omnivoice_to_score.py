#!/usr/bin/env python3
"""Evaluate CTC→WSOLA timing correction for frozen OmniVoice phrase sources."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

from gyu_singer.inference.content_timing import ctc_phone_alignment, wsola_to_phone_timing
from gyu_singer.score import normalize_score


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rc4", default="artifacts/reports/rc5_isolation"); parser.add_argument("--output", default="artifacts/reports/rc5_content_timing")
    args = parser.parse_args(); rc4, output = Path(args.rc4), Path(args.output); shutil.rmtree(output, ignore_errors=True); output.mkdir(parents=True)
    matrix = json.loads((rc4 / "matrix.json").read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, model = torchaudio.pipelines.MMS_FA.get_labels(), torchaudio.pipelines.MMS_FA.get_model().to(device).eval()
    report = {"status": "ctc_timing_correction_rendered_not_selected", "model": "torchaudio.pipelines.MMS_FA", "method": "MMS character CTC grouped into frontend phones, then pitch-preserving sample-domain WSOLA", "phase_vocoder": False, "waveform_pitch_shift": False, "cases": {}}
    for case, data in matrix["cases"].items():
        score = normalize_score(json.loads(Path(data["score"]).read_text())); source = Path(data["matrix"]["A"]["path"])
        audio, rate = sf.read(source, dtype="float32", always_2d=True); mono = audio.mean(1)
        model_audio = resample_poly(mono, 16000, rate).astype("float32") if rate != 16000 else mono
        alignment = ctc_phone_alignment(torch.from_numpy(model_audio).to(device), 16000, score, model, labels)
        duration = max(note["start"] + note["duration"] for note in score["notes"]); corrected = wsola_to_phone_timing(mono, rate, duration, alignment)
        case_dir = output / case; case_dir.mkdir(); target = case_dir / "ctc_wsola_source.wav"; sf.write(target, corrected, rate, subtype="PCM_24")
        (case_dir / "alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2) + "\n")
        report["cases"][case] = {"score": data["score"], "source": str(source), "corrected": str(target), "phones": len(alignment["phones"]), "ctc_mean_log_score": alignment["mean_log_score"], "duration_seconds": duration}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cases": {case: {"phones": row["phones"], "ctc_mean_log_score": row["ctc_mean_log_score"]} for case, row in report["cases"].items()}}, indent=2))


if __name__ == "__main__": main()
