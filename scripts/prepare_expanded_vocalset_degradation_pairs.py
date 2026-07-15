#!/usr/bin/env python3
"""Select a quality-gated, speaker-disjoint VocalSet degradation-pair plan."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/manifests/score_native_vocalset_realized.jsonl"
TARGET = ROOT / "data/external/manifests/vocalset_degradation_plan_v2.jsonl"
REPORT = ROOT / "artifacts/reports/vocalset_degradation_plan_v2.json"
TECHNIQUES = (
    "scales/fast_forte", "scales/fast_piano", "scales/straight", "scales/vibrato",
    "scales/breathy", "scales/belt", "arpeggios/fast_forte", "arpeggios/fast_piano",
    "arpeggios/straight", "arpeggios/vibrato", "long_tones/straight", "long_tones/forte",
)


def stable_pick(rows: list[dict], salt: str) -> dict:
    return min(rows, key=lambda row: hashlib.sha256(f"{salt}:{row['id']}".encode()).digest())


def quality(path: Path) -> dict:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(1)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    clipping = float(np.mean(np.abs(mono) >= .999)) if len(mono) else 1.0
    dc = float(abs(np.mean(mono))) if len(mono) else 1.0
    frame = max(1, int(rate * .02)); usable = len(mono) // frame * frame
    frame_rms = np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12) if usable else np.array([0.0])
    noise, signal = np.percentile(frame_rms, [10, 75])
    snr = float(20 * np.log10(max(signal, 1e-8) / max(noise, 1e-6)))
    duration = len(mono) / rate
    gates = {
        "duration": .75 <= duration <= 30,
        "clipping": clipping <= .0001,
        "dc_offset": dc <= .02,
        "level": rms >= .004,
        "snr_proxy": snr >= 10,
        "isolated_vocal_provenance": True,
    }
    return {
        "duration_sec": round(duration, 4), "sample_rate": rate, "peak": round(peak, 6),
        "rms": round(rms, 6), "clipping_fraction": round(clipping, 8),
        "dc_offset_abs": round(dc, 7), "snr_proxy_db": round(snr, 3),
        "quality_gates": gates, "accepted": all(gates.values()),
    }


def main() -> None:
    source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["speaker"], row["technique"])].append(row)
    speakers = sorted({row["speaker"] for row in source_rows})
    split = {
        speaker: ("test" if index % 5 == 0 else "validation" if index % 5 == 1 else "train")
        for index, speaker in enumerate(speakers)
    }
    plans = []
    for speaker in speakers:
        reference_candidates = grouped[(speaker, "long_tones/straight")]
        reference = stable_pick(reference_candidates, f"reference:{speaker}")
        for technique in TECHNIQUES:
            source = stable_pick(grouped[(speaker, technique)], f"source:{speaker}:{technique}")
            metrics = quality(ROOT / source["audio_path"])
            if not metrics["accepted"]:
                continue
            plans.append({
                "id": f"pair_v2_{source['id']}", "dataset": "vocalset", "domain": "singing",
                "language": "nonlexical_vowel", "speaker": speaker, "source_id": source["id"],
                "clean_target": source["audio_path"], "reference": reference["audio_path"],
                "split": split[speaker], "trust_weight": 1.0, "identity_adapter": False,
                "style_adapter": False, "technique": technique,
                "phoneme": source["phoneme"], "phoneme_label_status": source["phoneme_label_status"],
                "f0_path": source["f0_path"], "f0_label_status": source["f0_label_status"],
                "license": "CC-BY-4.0", "quality": metrics,
            })
    TARGET.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans))
    report = {
        "status": "ready", "rows": len(plans), "source_rows": len(source_rows),
        "target_hours": round(sum(row["quality"]["duration_sec"] for row in plans) / 3600, 4),
        "speakers": len({row["speaker"] for row in plans}),
        "splits": Counter(row["split"] for row in plans),
        "techniques": Counter(row["technique"] for row in plans),
        "license": "CC-BY-4.0", "speaker_disjoint": all(
            len({row["split"] for row in plans if row["speaker"] == speaker}) == 1
            for speaker in speakers
        ),
        "random_noise_used": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=dict) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
