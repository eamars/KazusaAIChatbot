"""Native static character-profile loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kazusa_ai_chatbot.db.schemas import CharacterProfileSeedV1

_RUNTIME_OWNED_FIELDS = frozenset({
    "_id",
    "global_user_id",
    "self_image",
    "cognition_state",
    "updated_at",
    "mood",
    "global_vibe",
    "reflection_summary",
})
_REQUIRED_SEED_FIELDS = frozenset({
    "name",
    "personality_brief",
    "boundary_profile",
    "linguistic_texture_profile",
})
_REQUIRED_PROFILE_FIELDS = {
    "boundary_profile": (
        "self_integrity",
        "control_sensitivity",
        "relational_override",
        "control_intimacy_misread",
        "authority_skepticism",
        "compliance_strategy",
        "boundary_recovery",
    ),
    "linguistic_texture_profile": (
        "fragmentation",
        "hesitation_density",
        "counter_questioning",
        "softener_density",
        "formalism_avoidance",
        "abstraction_reframing",
        "direct_assertion",
        "emotional_leakage",
        "rhythmic_bounce",
        "self_deprecation",
    ),
}
_BOUNDARY_COMPLIANCE_STRATEGIES = frozenset({"resist", "evade", "comply"})
_BOUNDARY_RECOVERY_MODES = frozenset({
    "rebound",
    "delayed_rebound",
    "decay",
    "detach",
})


def _validate_profile_seed_payload(
    payload: Mapping[str, Any],
) -> CharacterProfileSeedV1:
    """Validate a decoded static profile payload."""

    runtime_fields = sorted(_RUNTIME_OWNED_FIELDS.intersection(payload))
    if runtime_fields:
        field_text = ", ".join(runtime_fields)
        raise ValueError(
            f"character profile seed contains runtime-owned field(s): "
            f"{field_text}"
        )

    missing_fields = sorted(_REQUIRED_SEED_FIELDS.difference(payload))
    if missing_fields:
        field_text = ", ".join(missing_fields)
        raise ValueError(
            f"character profile seed is missing required field(s): "
            f"{field_text}"
        )

    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise ValueError("character profile seed name must be non-empty")

    for field_name in (
        "personality_brief",
        "boundary_profile",
        "linguistic_texture_profile",
    ):
        if not isinstance(payload[field_name], Mapping) or not payload[field_name]:
            raise ValueError(
                f"character profile seed {field_name} must be a non-empty object"
            )

    for profile_name, field_names in _REQUIRED_PROFILE_FIELDS.items():
        profile = payload[profile_name]
        missing_fields = [
            field_name for field_name in field_names if field_name not in profile
        ]
        if missing_fields:
            field_text = ", ".join(missing_fields)
            raise ValueError(
                f"character profile seed {profile_name} is missing required "
                f"field(s): {field_text}"
            )
        for field_name in field_names:
            value = profile[field_name]
            if profile_name == "boundary_profile" and field_name in {
                "compliance_strategy",
                "boundary_recovery",
            }:
                allowed_values = (
                    _BOUNDARY_COMPLIANCE_STRATEGIES
                    if field_name == "compliance_strategy"
                    else _BOUNDARY_RECOVERY_MODES
                )
                if value not in allowed_values:
                    raise ValueError(
                        f"character profile seed {profile_name}.{field_name} "
                        f"must be one of {sorted(allowed_values)}"
                    )
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    f"character profile seed {profile_name}.{field_name} "
                    "must be a number between 0 and 1"
                )

    validated_payload = dict(payload)
    return_value = cast("CharacterProfileSeedV1", validated_payload)
    return return_value


def load_character_profile_seed(path: Path) -> CharacterProfileSeedV1:
    """Load and validate one absolute UTF-8 static profile seed."""

    profile_path = Path(path)
    if not profile_path.is_absolute():
        raise ValueError("character profile path must be absolute")
    if not profile_path.is_file():
        raise ValueError(
            f"character profile path must point to a file: {profile_path}"
        )

    try:
        raw_text = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"failed to read character profile {profile_path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"character profile is not valid JSON: {profile_path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("character profile seed root must be an object")
    return _validate_profile_seed_payload(payload)


def validate_character_profile_seed(
    seed: Mapping[str, Any],
) -> CharacterProfileSeedV1:
    """Validate an already decoded static profile seed."""

    return _validate_profile_seed_payload(seed)
