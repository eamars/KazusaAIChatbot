"""Canonical character identity profile loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from kazusa_ai_chatbot.character_identity_growth.models import (
    CharacterEffectiveIdentityV1,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
)


def _validate_profile_seed_payload(
    payload: Mapping[str, object],
) -> CharacterEffectiveIdentityV1:
    """Validate one decoded profile as a complete revision-zero identity."""

    return validate_effective_identity(payload)


def _decode_profile_text(
    *,
    raw_text: str,
    source_description: str,
) -> CharacterEffectiveIdentityV1:
    """Decode and validate one UTF-8 profile document."""

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"character profile is not valid JSON: "
            f"{source_description}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("character profile seed root must be an object")
    return _validate_profile_seed_payload(payload)


def load_character_profile_seed(
    path: Path,
) -> CharacterEffectiveIdentityV1:
    """Load and validate one UTF-8 canonical identity seed."""

    profile_path = Path(path)
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

    return _decode_profile_text(
        raw_text=raw_text,
        source_description=str(profile_path),
    )


def validate_character_profile_seed(
    seed: Mapping[str, object],
) -> CharacterEffectiveIdentityV1:
    """Validate an already decoded canonical identity seed."""

    return _validate_profile_seed_payload(seed)
