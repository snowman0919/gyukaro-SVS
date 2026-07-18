#!/usr/bin/env python3
"""Apply the strict all-five-phrases GTSinger foundation gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = ("soprano", "tenor", "mix20")
SEEDS = (7, 21, 42)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_stability(paths: dict[str, dict[int, Path]], reports: dict[str, dict]) -> dict:
    phrases = {}
    for phrase, values in paths.items():
        hashes = {str(seed): _sha256(Path(path)) for seed, path in values.items()}
        report = reports[phrase]
        rows = {row["label"]: row for row in report["rows"]}
        quality = {
            str(seed): bool(
                report["reference_calibration"]["free_asr_similarity"] >= .8
                and rows[f"seed{seed}"]["pass"]
            )
            for seed in SEEDS
        }
        phrases[phrase] = {
            "sha256": hashes,
            "unique_sha256": len(set(hashes.values())),
            "quality_gate": quality,
            "all_quality_gates_pass": all(quality.values()),
        }
    return {
        "seeds": list(SEEDS),
        "phrases": phrases,
        "byte_identical": bool(phrases) and all(
            phrases[phrase]["unique_sha256"] == 1 for phrase in paths
        ),
        "stable": bool(phrases) and all(
            len(values) == len(SEEDS) and phrases[phrase]["all_quality_gates_pass"]
            for phrase, values in paths.items()
        ),
    }


def _row(report: dict, label: str) -> dict:
    return next(row for row in report["rows"] if row["label"] == label)


def _identity(rows: list[dict], baseline_rows: list[dict] | None) -> dict | None:
    if not rows or not all(row.get("identity_similarity") for row in rows):
        return None
    result = {
        metric: round(float(np.mean([row["identity_similarity"][metric]["mean"] for row in rows])), 5)
        for metric in ("wavlm", "ecapa")
    }
    if baseline_rows is None:
        return result
    deltas = {}
    for metric, collapse_limit in (("wavlm", -.02), ("ecapa", -.03)):
        names = rows[0]["identity_similarity"][metric]["values"]
        per_reference = {
            name: round(float(np.mean([
                row["identity_similarity"][metric]["values"][name]
                - baseline["identity_similarity"][metric]["values"][name]
                for row, baseline in zip(rows, baseline_rows)
            ])), 5)
            for name in names
        }
        deltas[metric] = {
            "mean": round(result[metric] - float(np.mean([
                row["identity_similarity"][metric]["mean"] for row in baseline_rows
            ])), 5),
            "per_reference": per_reference,
            "no_material_reference_collapse": min(per_reference.values()) >= collapse_limit,
        }
    result["versus_baseline"] = deltas
    result["improved"] = bool(
        deltas["wavlm"]["mean"] >= .005
        and deltas["ecapa"]["mean"] >= .005
        and deltas["wavlm"]["no_material_reference_collapse"]
        and deltas["ecapa"]["no_material_reference_collapse"]
    )
    return result


def aggregate_candidate_gate(
    reports: list[dict], label: str, baseline_label: str | None = None,
) -> dict:
    rows = [_row(report, label) for report in reports]
    baseline_rows = [_row(report, baseline_label) for report in reports] if baseline_label else None
    failures = []
    for index, (report, row) in enumerate(zip(reports, rows)):
        reference = report["reference_calibration"]
        baseline = baseline_rows[index] if baseline_rows else None
        checks = {
            "source_upper_bound": reference["free_asr_similarity"] >= .8,
            "lexical": row["asr_lyric_similarity"] >= .8,
            "pitch_p90": row["pitch_p90_abs_cents"] <= 100,
            "pitch_gross": row["gross_error_over_600_cents"] <= .05,
            "voicing": row["voicing_accuracy"] >= (baseline["voicing_accuracy"] - .02 if baseline else .8),
            "clipping": row["clip_fraction"] == 0,
            "hf_spike": row["hf_spike_p99_over_median"] <= (
                2 * reference["waveform_analysis"]["hf_spike_p99_over_median"]
            ),
            "sample_jump": baseline is None or row["sample_jump_p999"] <= 1.1 * baseline["sample_jump_p999"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append({"phrase": report["id"], "failed_gates": failed})
    identity = _identity(rows, baseline_rows)
    identity_required = label == "mix20"
    identity_pass = not identity_required or bool(identity and identity.get("improved"))
    audio_pass = not failures and len(rows) == 5
    return {
        "label": label,
        "status": "qualified_foundation_only" if audio_pass and identity_pass else "reject",
        "passed_phrases": len(rows) - len(failures),
        "failed_phrases": [failure["phrase"] for failure in failures],
        "failures": failures,
        "identity": identity,
        "identity_gate_required": identity_required,
        "identity_gate_pass": identity_pass,
        "metrics": {
            "lyric_similarity_mean": round(float(np.mean([row["asr_lyric_similarity"] for row in rows])), 5),
            "lyric_similarity_min": min(row["asr_lyric_similarity"] for row in rows),
            "pitch_p90_max": max(row["pitch_p90_abs_cents"] for row in rows),
            "gross_error_max": max(row["gross_error_over_600_cents"] for row in rows),
            "voicing_accuracy_min": min(row["voicing_accuracy"] for row in rows),
            "clipping_max": max(row["clip_fraction"] for row in rows),
            "hf_over_source_max": max(
                row["hf_spike_p99_over_median"]
                / report["reference_calibration"]["waveform_analysis"]["hf_spike_p99_over_median"]
                for report, row in zip(reports, rows)
            ),
            "sample_jump_max": max(row["sample_jump_p999"] for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path(
        "artifacts/reports/diffsinger_gtsinger_heldout_set"
    ))
    args = parser.parse_args()
    paths = sorted(args.report_dir.glob("evaluation_gtsja*.json"))
    if len(paths) != 5:
        raise SystemExit(f"expected five evaluation reports, found {len(paths)}")
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        report["id"] = path.stem.removeprefix("evaluation_")
        reports.append(report)
    candidates = {
        "soprano": aggregate_candidate_gate(reports, "soprano"),
        "tenor": aggregate_candidate_gate(reports, "tenor", "soprano"),
        "mix20": aggregate_candidate_gate(reports, "mix20", "tenor"),
    }
    seed_paths = {
        report["id"]: {
            seed: args.report_dir / "listening/seed_stability" / f"{report['id']}_soprano_seed{seed}.wav"
            for seed in SEEDS
        }
        for report in reports
    }
    seed_reports = {
        report["id"]: json.loads(
            (args.report_dir / "seed_stability" / f"evaluation_{report['id']}.json").read_text()
        )
        for report in reports
    }
    stability = seed_stability(seed_paths, seed_reports)
    foundation_pass = candidates["soprano"]["status"] == "qualified_foundation_only" and stability["stable"]
    if candidates["tenor"]["status"] != "qualified_foundation_only":
        candidates["mix20"]["status"] = "reject_unqualified_identity_baseline"
    report = {
        "status": "qualified_score_native_foundation_only" if foundation_pass else "foundation_reject",
        "foundation": "soprano" if foundation_pass else None,
        "candidates": candidates,
        "seed_stability": stability,
        "source_upper_bounds_all_pass": all(
            item["reference_calibration"]["free_asr_similarity"] >= .8 for item in reports
        ),
        "identity_adaptation_allowed": False,
        "new_training_allowed": False,
        "production_readiness": "FAIL",
        "rc9_openutau_package_release": "blocked",
        "v1.0.0": "prohibited",
        "license": "CC BY-NC-SA 4.0 evaluation only",
        "runtime_integration": False,
    }
    output = args.report_dir / "aggregate_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
