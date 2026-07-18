from __future__ import annotations


VALID_RESULTS = {
    "identity_diagnostic_reject",
    "identity_machine_inconclusive",
    "identity_candidate_human_pending",
}


def validate_identity_protocol(protocol: dict) -> None:
    required = {
        "foundation", "foundation_checkpoint_sha256", "identity_references", "seeds",
        "candidates", "preservation_metrics", "identity_metrics", "valid_results",
        "selected_candidate", "training_status", "optimizer_steps",
    }
    if not required <= protocol.keys():
        raise ValueError("identity protocol is incomplete")
    types = [row.get("type") for row in protocol["candidates"]]
    expected = ["fixed_gyu_speaker_embedding", "small_film", "low_rank_residual", "vocoder_conditioning"]
    if types != expected or protocol["candidates"][-1].get("enabled"):
        raise ValueError("identity candidates must remain ordered and vocoder conditioning disabled")
    if set(protocol["valid_results"]) != VALID_RESULTS:
        raise ValueError("identity result states drift")
    if len(set(protocol["seeds"])) != len(protocol["seeds"]) or len(set(protocol["identity_references"])) != len(protocol["identity_references"]):
        raise ValueError("identity protocol contains duplicate seeds or references")


def identity_training_authorized(protocol: dict, project_status: dict) -> bool:
    validate_identity_protocol(protocol)
    models = {row["identifier"]: row for row in project_status.get("models", [])}
    foundation = models.get(protocol["foundation"])
    if foundation is None:
        return False
    return (
        foundation["status"] in protocol["authorized_foundation_statuses"]
        and foundation["human_approval_state"] == protocol["required_human_approval_state"]
    )
