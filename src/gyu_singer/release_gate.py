from __future__ import annotations

from dataclasses import dataclass


REQUIRED_GATES = (
    "approved_foundation",
    "approved_identity_candidate",
    "phone_centered_lexical_gate",
    "score_pitch_gate",
    "voicing_gate",
    "artifact_gate",
    "multi_seed_gate",
    "long_form_continuity_gate",
    "human_approval_record",
    "license_provenance_validation",
    "reproducible_package_manifest",
)


class ReleaseGateError(ValueError):
    pass


@dataclass(frozen=True)
class ReleaseDecision:
    status: str
    failed_gates: tuple[str, ...]


def decide_release(protocol: dict) -> ReleaseDecision:
    gates = protocol.get("gates", {})
    missing = set(REQUIRED_GATES) - set(gates)
    extra = set(gates) - set(REQUIRED_GATES)
    if missing or extra:
        raise ReleaseGateError(f"missing release gates or unknown gates: missing={sorted(missing)} extra={sorted(extra)}")
    for name, row in gates.items():
        if not {"passed", "method", "evidence"} <= row.keys() or not isinstance(row["passed"], bool):
            raise ReleaseGateError(f"invalid release gate: {name}")
    lexical = gates["phone_centered_lexical_gate"]
    if lexical["passed"] and lexical["method"] == "whisper_only":
        raise ReleaseGateError("Whisper cannot be the sole release lexical criterion")
    human = gates["human_approval_record"]
    if human["passed"] and not human["evidence"]:
        raise ReleaseGateError("human approval requires a recorded evidence path")
    failed = tuple(name for name in REQUIRED_GATES if not gates[name]["passed"])
    return ReleaseDecision("release_approved" if not failed else "release_blocked", failed)
