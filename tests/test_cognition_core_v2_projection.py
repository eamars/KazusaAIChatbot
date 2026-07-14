"""Checkpoint D semantic projection and prompt-boundary tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_duration,
    project_numeric_band,
    project_state_for_prompt,
    project_trend,
    validate_prompt_projection,
)


NOW = "2026-07-14T00:00:00Z"


def _constraints() -> dict[str, object]:
    """Return the character constraints as a separate read-only snapshot."""

    state = build_character_production_state(updated_at=NOW)
    return {
        "drives": state["drives"],
        "standards": state["standards"],
        "meaning_state": state["meaning_state"],
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "none"),
        (20, "very low"),
        (21, "low"),
        (40, "low"),
        (41, "moderate"),
        (60, "moderate"),
        (61, "high"),
        (80, "high"),
        (81, "very high"),
        (100, "very high"),
    ],
)
def test_numeric_projection_uses_frozen_semantic_bands(
    value: int,
    expected: str,
) -> None:
    """Keep model-facing scalar vocabulary independent from raw numbers."""

    assert project_numeric_band(value) == expected


def test_signed_projection_and_trend_are_bounded() -> None:
    """Signed axes and direction use the explicit frozen boundaries."""

    assert project_numeric_band(-61, signed=True) == "strongly negative"
    assert project_numeric_band(-20, signed=True) == "neutral or mixed"
    assert project_numeric_band(61, signed=True) == "strongly positive"
    assert project_trend(40, 44) == "rising"
    assert project_trend(44, 40) == "falling"
    assert project_trend(40, 43) == "stable"


def test_duration_projection_uses_semantic_time_labels() -> None:
    """Timestamps become bounded duration descriptors before model entry."""

    assert project_duration(NOW, "2026-07-14T00:09:59Z") == "immediate"
    assert project_duration(NOW, "2026-07-14T01:00:00Z") == "recent"
    assert project_duration(NOW, "2026-07-14T12:00:00Z") == "earlier"
    assert project_duration(NOW, "2026-07-16T00:00:00Z") == "within recent days"
    assert project_duration(NOW, "2026-07-22T00:00:00Z") == "older"


def test_state_projection_separates_user_state_and_character_constraints() -> None:
    """Prompt payloads contain descriptors while private bindings retain refs."""

    state = build_acquaintance_user_state(
        global_user_id="user-projection",
        updated_at=NOW,
    )
    projection = project_state_for_prompt(
        state,
        character_constraints=_constraints(),
    )

    assert projection.payload["character_constraints"]["drives"]["care"][
        "pressure"
    ] == "very low"
    assert "owner_user_id" not in projection.payload
    assert "updated_at" not in projection.payload
    assert projection.handle_to_ref["r1"]["entity_id"] == (
        "relationship:user:user-projection"
    )


def test_prompt_sentinel_guard_rejects_raw_state_keys() -> None:
    """A raw internal sentinel cannot cross the model-facing projection seam."""

    with pytest.raises(ValueError, match="raw state key"):
        validate_prompt_projection({"updated_at": "sentinel-1701"})
