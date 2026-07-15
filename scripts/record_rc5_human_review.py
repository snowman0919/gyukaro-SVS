#!/usr/bin/env python3
"""Record explicit human decision for immutable RC5 candidate4 audio."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", choices=("pass", "fail"), required=True)
    parser.add_argument("--reviewer", default="project owner")
    args = parser.parse_args()
    root = Path("artifacts/reports/rc5_stress_candidate4")
    files = sorted((root / "listening").glob("*.wav")) + sorted((root / "before_after").glob("*.wav"))
    review = {
        "decision": args.decision,
        "reviewer": args.reviewer,
        "recorded_date": date.today().isoformat(),
        "scope": "candidate4 nine-case stress set and matched RC4 before/after clips",
        "files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
        "final_v1_authorized": False,
    }
    (root / "human_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["status"] = f"human_listening_{args.decision}"
    manifest["human_review"] = "artifacts/reports/rc5_stress_candidate4/human_review.json"
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    evaluation = json.loads((root / "evaluation.json").read_text())
    evaluation["status"] = f"objective_pass_human_{args.decision}"
    evaluation["human_listening"] = args.decision
    (root / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n")
    before_after = json.loads((root / "before_after/manifest.json").read_text())
    before_after["status"] = f"human_listening_{args.decision}"
    (root / "before_after/manifest.json").write_text(json.dumps(before_after, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"decision": args.decision, "files": len(files), "report": str(root / "human_review.json")}, indent=2))


if __name__ == "__main__":
    main()
