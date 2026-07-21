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
        (0, "无"),
        (20, "极低"),
        (21, "低"),
        (40, "低"),
        (41, "中等"),
        (60, "中等"),
        (61, "高"),
        (80, "高"),
        (81, "极高"),
        (100, "极高"),
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

    assert project_numeric_band(-61, signed=True) == "强烈负向"
    assert project_numeric_band(-20, signed=True) == "中性或混合"
    assert project_numeric_band(61, signed=True) == "强烈正向"
    assert project_trend(40, 44) == "上升"
    assert project_trend(44, 40) == "下降"
    assert project_trend(40, 43) == "稳定"


def test_duration_projection_uses_semantic_time_labels() -> None:
    """Timestamps become bounded duration descriptors before model entry."""

    assert project_duration(NOW, "2026-07-14T00:09:59Z") == "即时"
    assert project_duration(NOW, "2026-07-14T01:00:00Z") == "近期"
    assert project_duration(NOW, "2026-07-14T12:00:00Z") == "较早"
    assert project_duration(NOW, "2026-07-16T00:00:00Z") == "最近几天内"
    assert project_duration(NOW, "2026-07-22T00:00:00Z") == "较久以前"


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
    ] == "极低"
    assert "owner_user_id" not in projection.payload
    assert "updated_at" not in projection.payload
    assert projection.payload["roles"] == {
        "当前角色": "当前角色",
        "当前用户": "当前用户",
    }
    standard_text = [
        row["description"]
        for row in projection.payload["character_constraints"]["standards"]
    ]
    assert standard_text == [
        "保持诚实",
        "避免造成不必要的伤害",
        "尊重个人边界",
        "履行已经接受的承诺",
        "保护尊严与自主性",
    ]
    assert projection.handle_to_ref["r1"]["entity_id"] == (
        "relationship:user:user-projection"
    )


def test_prompt_sentinel_guard_rejects_raw_state_keys() -> None:
    """A raw internal sentinel cannot cross the model-facing projection seam."""

    with pytest.raises(ValueError, match="raw state key"):
        validate_prompt_projection({"updated_at": "sentinel-1701"})
