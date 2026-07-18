from __future__ import annotations


REQUIRED_RIGHTS_FIELDS = {
    "source_type",
    "owner_or_authorized_user",
    "allowed_use",
    "redistribution_permission",
    "languages",
    "known_scripts",
    "recording_environment",
    "consent_provenance_notes",
    "permission_affirmed",
}


def validate_rights_manifest(manifest: dict) -> None:
    from .factory import FactoryError

    missing = REQUIRED_RIGHTS_FIELDS - manifest.keys()
    if missing or manifest.get("permission_affirmed") is not True:
        raise FactoryError(f"rights manifest is incomplete or permission is not affirmed: {sorted(missing)}")
    if not manifest["owner_or_authorized_user"] or not manifest["allowed_use"]:
        raise FactoryError("rights manifest requires owner and allowed use")
    if not isinstance(manifest["languages"], list) or not manifest["languages"]:
        raise FactoryError("rights manifest requires at least one language")
