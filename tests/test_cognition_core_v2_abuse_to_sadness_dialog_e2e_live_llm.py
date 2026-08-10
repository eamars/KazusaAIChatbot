"""Real-LLM dialog proof for a mechanically established abuse loss."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_cognition_core_v2_crying_sadness_e2e_live_llm import (
    _run_chat_sequence,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_abuse_to_sadness_e2e_cases.json"
)


def _load_case() -> dict[str, object]:
    """Load the shared Chinese abuse/cutoff input."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("abuse-to-sadness dialog fixture is invalid")
    return payload


async def test_live_abuse_to_sadness_renders_visible_dialog(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Render one abuse/cutoff turn from a sadness-causing event state."""

    payload = _load_case()
    event_spec = payload.get("abuse_event")
    turn_spec = payload.get("turn")
    if not isinstance(event_spec, Mapping):
        raise ValueError("abuse dialog event is invalid")
    if not isinstance(turn_spec, Mapping):
        raise ValueError("abuse dialog turn is invalid")
    event = deepcopy(dict(event_spec))
    event.update({
        "outcome_impact": -85,
        "harm": 50,
        "unfairness": 50,
        "intentionality": 80,
        "responsibility": 80,
        "repair_need": 80,
        "reparability": 10,
        "identity_threat": 70,
        "temporal_loss": 80,
    })
    turn = {
        "turn_id": "abuse_to_sadness_visible_dialog",
        "text": str(turn_spec["text"]),
        "observation_target": (
            "辱骂事件已经具有确定的负面结果；可见渲染应当呈现悲伤。"
        ),
        "expected_role_bindings": turn_spec.get(
            "expected_role_bindings"
        ),
    }
    manifest = await _run_chat_sequence(
        case_id="abuse_to_sadness_visible_dialog",
        turns=[turn],
        event_spec=event,
        seed_at_turn=0,
        expected_emotion="sadness",
        forbidden_emotion="fear",
        caplog=caplog,
    )
    turns = manifest.get("turns")
    if not isinstance(turns, list) or len(turns) != 1:
        raise AssertionError("abuse dialog run did not produce one turn")
    turn_summary = turns[0]
    if not isinstance(turn_summary, Mapping):
        raise AssertionError("abuse dialog turn summary is invalid")
    affect_emotions = turn_summary.get("affect_emotions")
    final_dialog = turn_summary.get("final_dialog")
    assert isinstance(affect_emotions, list)
    assert "sadness" in affect_emotions
    assert isinstance(final_dialog, list) and final_dialog
    print("abuse-to-sadness visible dialog:")
    for segment in final_dialog:
        print(str(segment))
