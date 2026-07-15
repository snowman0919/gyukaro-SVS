#!/usr/bin/env python3
"""Render the actual RC6 backend and compare with the fixed candidate bytes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gyu_singer.inference.rc6 import GyuSingerRC6Renderer


CASES = {"ko_neutral": "examples/quality_ko.json", "en": "examples/quality_en.json", "rapid_ko": "examples/review_rapid_ko.json", "large_interval_ko": "examples/review_large_interval_ko.json"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("artifacts/reports/rc6_runtime_smoke"); root.mkdir(parents=True, exist_ok=True)
    renderer = GyuSingerRC6Renderer("data/processed/master/216.wav")
    rows = []
    try:
        for case, score in CASES.items():
            output = root / f"{case}.wav"; renderer.render_file(score, output)
            candidate = Path("artifacts/reports/refiner_rc_candidate/listening") / f"{case}.wav"
            rows.append({"case": case, "runtime": str(output), "candidate": str(candidate), "runtime_sha256": sha(output), "candidate_sha256": sha(candidate), "exact_match": sha(output) == sha(candidate)})
            print(case, flush=True)
    finally:
        renderer.close()
    report = {"status": "pass" if all(row["exact_match"] for row in rows) else "fail", "backend": "gyu-singer-rc6", "comparison": "byte-for-byte against 25% refiner candidate", "rows": rows}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
