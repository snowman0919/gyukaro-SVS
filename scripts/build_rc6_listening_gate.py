#!/usr/bin/env python3
"""Assemble the mandatory compact human-listening gate without deciding it."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


CASES = ("ko_neutral", "ko_breathy", "ko_energetic", "en", "ja", "rapid_ko", "sustained_ko", "large_interval_ko", "phrase_boundary")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("artifacts/reports/rc6_listening_gate"); root.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(Path("artifacts/reports/rc6_backend_candidate/manifest.json").read_text())
    isolation = json.loads(Path("artifacts/reports/rc5_isolation/matrix.json").read_text())
    rows = []
    for index, case in enumerate(CASES, 1):
        output = root / f"{index:02d}_{case}.wav"; shutil.copy2(candidate["files"][case]["path"], output)
        rows.append({"case": case, "file": str(output), "sha256": sha(output), "verdict": None, "observation": ""})
    comparisons = []
    for case in ("ko_neutral", "rapid_ko", "large_interval_ko"):
        before = root / f"before_rc4_{case}.wav"; after = root / f"after_rc6_{case}.wav"
        shutil.copy2(isolation["cases"][case]["matrix"]["F"]["path"], before); shutil.copy2(candidate["files"][case]["path"], after)
        comparisons.append({"case": case, "before": str(before), "before_sha256": sha(before), "after": str(after), "after_sha256": sha(after)})
    manifest = {"status": "human_listening_pending", "candidate": "gyu-singer-rc6 universal refiner 25%", "files": rows, "rc4_before_after": comparisons,
                "required_response": {"per_file": "PASS or FAIL plus brief observation", "overall_release_suitability": "PASS or FAIL"},
                "final_v1_release_allowed": False}
    (root / "listening_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = ["# RC6 human listening gate", "", "Final v1.0 remains blocked. Listen to every numbered file and the three RC4/RC6 comparisons.", "",
             "Reply with `PASS` or `FAIL` plus one brief observation for each numbered file, then `Overall release suitability: PASS/FAIL`.", ""]
    lines += [f"- {row['case']}: PENDING" for row in rows]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(rows), "comparisons": len(comparisons), "directory": str(root)}, indent=2))


if __name__ == "__main__":
    main()
