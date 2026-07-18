#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/korean_phone_reassessment/evaluation.json"
LOCAL = ROOT / "data/external/work/korean_phone_reassessment_review"


def main() -> None:
    import matplotlib.pyplot as plt

    report = json.loads(REPORT.read_text())
    protocol = json.loads((ROOT / "data/manifests/gtsinger_gyu_identity_protocol.json").read_text())
    cases = {row["id"]: row for row in protocol["cases"]}
    rows = {(row["case"], row["seed"]): row for row in report["rows"]}
    items, key = [], {}
    for case_id, case in cases.items():
        order = [7, 21]
        if hashlib.sha256(case_id.encode()).digest()[0] % 2:
            order.reverse()
        directory = LOCAL / case_id
        directory.mkdir(parents=True, exist_ok=True)
        blind_audio = {}
        for label, seed in zip(("A", "B"), order):
            source = ROOT / rows[(case_id, seed)]["audio_path"]
            target = directory / f"{label}.wav"
            shutil.copy2(source, target)
            blind_audio[label] = str(target.relative_to(ROOT))
        protected = directory / "C.wav"
        shutil.copy2(ROOT / rows[(case_id, 42)]["audio_path"], protected)

        ds = json.loads((ROOT / case["ds_path"]).read_text())[0]
        f0 = [float(value) for value in ds["f0_seq"].split()]
        figure, axis = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
        axis.plot([index * float(ds["f0_timestep"]) for index in range(len(f0))], f0)
        axis.set(title="Nominal score F0", xlabel="seconds", ylabel="Hz")
        plot = directory / "score_f0.png"
        figure.savefig(plot, dpi=120)
        plt.close(figure)

        items.append({
            "id": case_id, "stress_category": case["stress_category"],
            "expected_lyrics": case["expected_lyrics"], "score_path": case["score_path"],
            "expected_phones": case["expected_phonemes"],
            "blind_audio": blind_audio, "protected_audio": str(protected.relative_to(ROOT)),
            "phoneme_timeline": rows[(case_id, order[0])]["aligned_phones_or_posterior_path"],
            "pitch_plot": str(plot.relative_to(ROOT)),
        })
        key[case_id] = {"A": order[0], "B": order[1], "C": 42}
    manifest = {
        "status": "foundation_machine_inconclusive", "purpose": "lexical_seed_stability_only",
        "promotion_allowed": False, "candidate_comparison_available": False,
        "rubric": ["phoneme_identity", "syllable_omission", "syllable_insertion",
                   "consonant_clarity", "vowel_correctness", "timing", "pitch", "voicing",
                   "artifacts", "naturalness"],
        "items": items,
    }
    destination = REPORT.with_name("human_review_manifest.json")
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (LOCAL / "review_key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "items": len(items),
                      "local_root": str(LOCAL.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
