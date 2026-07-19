from __future__ import annotations

import math
import statistics


def edit_counts(expected: list[str], observed: list[str]) -> dict:
    table = [[(0, 0, 0, 0)] * (len(observed) + 1) for _ in range(len(expected) + 1)]
    for i in range(1, len(expected) + 1):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, len(observed) + 1):
        table[0][j] = (j, j, 0, 0)
    for i, target in enumerate(expected, 1):
        for j, actual in enumerate(observed, 1):
            if target == actual:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidates = []
            cost, ins, delete, sub = table[i][j - 1]
            candidates.append((cost + 1, ins + 1, delete, sub))
            cost, ins, delete, sub = table[i - 1][j]
            candidates.append((cost + 1, ins, delete + 1, sub))
            cost, ins, delete, sub = table[i - 1][j - 1]
            candidates.append((cost + 1, ins, delete, sub + 1))
            table[i][j] = min(candidates)
    _, insertions, deletions, substitutions = table[-1][-1]
    return {"insertions": insertions, "deletions": deletions, "substitutions": substitutions}


def _distribution(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("finite metric values required")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norms if norms else 0.0


def aggregate_phone_evidence(rows: list[dict]) -> dict:
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    similarities = []
    for group in by_case.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                similarities.append(_cosine(left["content_embedding"], right["content_embedding"]))
    return {
        "phone_error_rate": _distribution([row["phone_error_rate"] for row in rows]),
        "seed_content_consistency": statistics.fmean(similarities) if similarities else 1.0,
    }


def classify_alignment(source: str, confidence: float) -> str:
    if source == "manual":
        return "manual"
    if source != "forced":
        return "inferred_only"
    if confidence < 0.25:
        return "rejected"
    return "forced_aligned_high_confidence" if confidence >= 0.75 else "forced_aligned_low_confidence"


def korean_lexical_decision(evidence: dict, auxiliary_stt_observation: str | None = None) -> str:
    del auxiliary_stt_observation
    if (evidence.get("calibration_status") != "calibrated"
            or evidence.get("acoustic_representations", 0) < 2):
        return "foundation_machine_inconclusive"
    if evidence.get("catastrophic_content_collapse") or not evidence.get("acoustic_gates_pass"):
        return "foundation_diagnostic_reject"
    required = (
        evidence.get("phone_error_rate", 1.0) <= 0.25,
        evidence.get("alignment_confidence", 0.0) >= 0.75,
        evidence.get("boundary_deviation", 1.0) <= 0.15,
        evidence.get("duration_deviation", 1.0) <= 0.15,
        evidence.get("seed_content_consistency", 0.0) >= 0.9,
    )
    return "foundation_candidate_human_pending" if all(required) else "foundation_diagnostic_reject"
