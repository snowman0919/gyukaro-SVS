#!/usr/bin/env python3
"""Build the compact human gate for the objectively selected spectral candidate."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reports/rc7_listening_gate"
CASES = (
    "ko_neutral", "ko_breathy", "ko_energetic", "en", "ja",
    "rapid_ko", "sustained_ko", "large_interval_ko", "phrase_boundary",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    evaluation = json.loads(
        (ROOT / "artifacts/reports/acoustic_refiner_spectral_stress_evaluation.json").read_text()
    )
    if evaluation["candidate"] != "spectral_s050" or not evaluation["materially_improved"]:
        raise RuntimeError("objective spectral_s050 gate has not passed")
    baseline = json.loads((ROOT / "artifacts/reports/rc6_backend_candidate/manifest.json").read_text())
    candidate = json.loads((ROOT / "artifacts/reports/spectral_refiner_stress_s050/manifest.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = []
    for index, case in enumerate(CASES, 1):
        source = ROOT / candidate["files"][case]["path"]
        path = OUTPUT / f"{index:02d}_{case}.wav"
        shutil.copy2(source, path)
        files.append({
            "case": case, "file": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "verdict": None, "observation": "",
        })
    comparisons = []
    for case in ("rapid_ko", "large_interval_ko", "en"):
        for label, manifest in (("RC6_before", baseline), ("spectral_after", candidate)):
            source = ROOT / manifest["files"][case]["path"]
            path = OUTPUT / f"compare_{case}_{label}.wav"
            shutil.copy2(source, path)
            comparisons.append({
                "case": case, "label": label,
                "file": str(path.relative_to(ROOT)), "sha256": sha256(path),
            })
    report = {
        "status": "human_listening_required",
        "candidate": "RC7 spectral-refiner candidate; not a tag or release",
        "source_baseline": "RC6 human-failed, preserved byte-for-byte",
        "checkpoint": "checkpoints/acoustic_refiner_spectral_singing.pt",
        "strength": 0.5,
        "objective_evidence": "artifacts/reports/acoustic_refiner_spectral_stress_evaluation.json",
        "files": files, "before_after": comparisons,
        "required_response": "PASS/FAIL for each of the nine files, brief observation, and overall release suitability",
        "final_v1_release_allowed": False,
    }
    (OUTPUT / "listening_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
