"""Behavior checks for persona diagnostic state."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import get_args, get_type_hints

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_schema"
EXPECTED_SYMBOLS = ["GlobalPersonaState"]




def test_persona_states_carry_attempt_diagnostics() -> None:
    """Both persona state contracts expose the shared reducer field."""

    module = import_module(MODULE_PATH)
    from kazusa_ai_chatbot.state import append_attempt_diagnostics

    for state_name in ("GlobalPersonaState", "CognitionState"):
        state_type = getattr(module, state_name)
        state_hints = get_type_hints(state_type, include_extras=True)
        assert "attempt_diagnostics" in state_hints
        assert append_attempt_diagnostics in get_args(
            state_hints["attempt_diagnostics"]
        )[1:]


def test_persona_supervisor_returns_attempt_diagnostics() -> None:
    """Persona orchestration initializes and returns the metadata accumulator."""

    source = Path(
        "src/kazusa_ai_chatbot/nodes/persona_supervisor2.py"
    ).read_text(encoding="utf-8")

    assert '"attempt_diagnostics": [],' in source
    assert '"attempt_diagnostics": list(' in source
    assert 'results.get("attempt_diagnostics", [])' in source


def test_persona_supervisor_returns_only_new_diagnostic_delta() -> None:
    """The internal persona graph starts empty and returns its own delta."""

    source = Path(
        "src/kazusa_ai_chatbot/nodes/persona_supervisor2.py"
    ).read_text(encoding="utf-8")
    function_source = source[source.index("async def persona_supervisor2"):]

    assert '"attempt_diagnostics": state.get' not in function_source
    assert function_source.count('"attempt_diagnostics": [],') == 1
    assert 'results.get("attempt_diagnostics", [])' in function_source


def test_persona_supervisor_diagnostic_delta_respects_sixteen_row_cap() -> None:
    """One persona row displaces the oldest inherited row at the cap."""

    from kazusa_ai_chatbot.state import append_attempt_diagnostics

    inherited = [
        {
            "schema_version": "episode_attempt_diagnostic.v1",
            "stage": "relevance",
            "error_code": f"inherited-{index:02d}",
            "attempt_count": 2,
            "safe_checkpoint": "pre_state_commit",
            "retryable": False,
            "final_status": "accepted_degraded",
        }
        for index in range(1, 17)
    ]
    persona_delta = [
        {
            **inherited[-1],
            "stage": "dialog",
            "error_code": "dialog-terminal",
        }
    ]

    merged = append_attempt_diagnostics(inherited, persona_delta)

    assert [row["error_code"] for row in merged] == [
        f"inherited-{index:02d}" for index in range(2, 17)
    ] + ["dialog-terminal"]
    assert [row["error_code"] for row in merged].count("dialog-terminal") == 1
