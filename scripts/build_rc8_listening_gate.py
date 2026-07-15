#!/usr/bin/env python3
"""Assemble the objective-passing RC8 files without mutating frozen RC7."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/rc8_listening_gate"
CASES = {
    "ko_neutral": ("01_ko_neutral.wav", "examples/quality_ko.json", "artifacts/reports/rc8_ko_warp_005/listening/ko_neutral.wav"),
    "ko_breathy": ("02_ko_breathy.wav", "artifacts/reports/rc5_stress_candidate/ko_breathy.json", "artifacts/reports/rc8_backend_candidate2/listening/ko_breathy.wav"),
    "ko_energetic": ("03_ko_energetic.wav", "artifacts/reports/rc5_stress_candidate/ko_energetic.json", "artifacts/reports/rc8_backend_candidate2/listening/ko_energetic.wav"),
    "en": ("04_en.wav", "examples/quality_en.json", "artifacts/reports/rc8_backend_candidate2/listening/en.wav"),
    "ja": ("05_ja.wav", "examples/quality_ja.json", "artifacts/reports/rc8_backend_candidate2/listening/ja.wav"),
    "rapid_ko": ("06_rapid_ko.wav", "examples/review_rapid_ko.json", "artifacts/reports/rc8_backend_candidate2/listening/rapid_ko.wav"),
    "sustained_ko": ("07_sustained_ko.wav", "examples/review_sustain_ko.json", "artifacts/reports/rc8_backend_candidate2/listening/sustained_ko.wav"),
    "large_interval_ko": ("08_large_interval_ko_fixed.wav", "examples/review_large_interval_ko.json", "artifacts/reports/rc8_interval_actual_backend/listening/large_interval_ko.wav"),
    "phrase_boundary": ("09_phrase_boundary.wav", "examples/review_phrase_boundary_ko.json", "artifacts/reports/rc8_backend_candidate2/listening/phrase_boundary.wav"),
}
RC7_NAMES = {case: name for case, (name, _, _) in CASES.items()}
RC7_NAMES["large_interval_ko"] = "08_large_interval_ko.wav"
COMPARE = ("ko_neutral", "en", "ja", "sustained_ko", "large_interval_ko", "rapid_ko")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    review_path = OUT / "listening_manifest.json"
    reviews = {}
    if review_path.is_file():
        reviews = {
            row["case"]: (row.get("verdict"), row.get("observation", ""))
            for row in json.loads(review_path.read_text()).get("files", [])
        }
    files, listening = {}, []
    for case, (name, score, source_name) in CASES.items():
        source = ROOT / source_name
        target = OUT / name
        shutil.copy2(source, target)
        relative = str(target.relative_to(ROOT))
        verdict, observation = reviews.get(case, (None, ""))
        row = {
            "case": case, "file": relative, "sha256": digest(target),
            "verdict": verdict, "observation": observation,
        }
        listening.append(row)
        files[case] = {
            "path": relative, "score": score, "sha256": row["sha256"],
            "sample_rate": 48_000, "backend": "gyu-singer-rc8",
        }

    verdicts = [row["verdict"] for row in listening]
    human_review = (
        "pass" if verdicts and all(value == "PASS" for value in verdicts)
        else "8_pass_1_fail_large_interval" if verdicts.count("PASS") == 8 and verdicts.count("FAIL") == 1
        else "pending"
    )

    before_after = []
    for case in COMPARE:
        before = ROOT / "artifacts/reports/rc7_listening_gate" / RC7_NAMES[case]
        after = OUT / CASES[case][0]
        for label, source in (("RC7_before", before), ("RC8_after", after)):
            target = OUT / f"compare_{case}_{label}.wav"
            shutil.copy2(source, target)
            before_after.append({
                "case": case, "label": label,
                "file": str(target.relative_to(ROOT)), "sha256": digest(target),
            })

    manifest = {
        "status": (
            "human_pass" if human_review == "pass"
            else "human_partial_pass_large_interval_failed" if human_review != "pending"
            else "objective_nonregression_human_pending"
        ),
        "candidate": "RC8 local-quality candidate; not a tag or release",
        "baseline": "immutable RC7 at ae8944070f3dc38e310b33f29d95f4bcd3c81def",
        "backend": {"backend": "gyu-singer-rc8", "final_v1_tagged": False},
        "objective_evidence": "artifacts/reports/rc8_listening_gate/evaluation.json",
        "large_interval_evidence": "artifacts/reports/rc8_interval_actual_backend/evaluation.json",
        "files": files, "human_review": human_review,
    }
    listening_manifest = {
        "status": "human_listening_required" if human_review == "pending" else manifest["status"],
        "candidate": manifest["candidate"], "source_baseline": manifest["baseline"],
        "objective_evidence": "artifacts/reports/rc8_listening_gate/evaluation.json",
        "large_interval_evidence": "artifacts/reports/rc8_interval_actual_backend/evaluation.json",
        "files": listening, "before_after": before_after,
        "required_response": "PASS/FAIL for each of nine files, defect observation, and overall RC8 suitability",
        "final_v1_release_allowed": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "listening_manifest.json").write_text(json.dumps(listening_manifest, ensure_ascii=False, indent=2) + "\n")
    evaluation_path = OUT / "evaluation.json"
    if human_review == "pass" and evaluation_path.is_file():
        evaluation = json.loads(evaluation_path.read_text())
        evaluation |= {
            "status": "objective_nonregression_human_pass",
            "human_listening": "pass",
            "large_interval_superseded_by": "artifacts/reports/rc8_interval_actual_backend/evaluation.json",
        }
        evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(files), "comparisons": len(before_after), "human_review": human_review}, indent=2))


if __name__ == "__main__":
    main()
