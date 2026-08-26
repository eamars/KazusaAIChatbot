"""Focused contracts for the top-level control console."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def _canonical_live_observation(
    *,
    run_id: str = "console-test-run",
    invocation_id: str = "console-test-invocation",
) -> dict:
    """Build one deterministic canonical live observation payload."""

    from datetime import datetime, timezone

    from kazusa_ai_chatbot.brain_service.cognition_observation_projection import (
        build_live_cognition_observation,
    )

    observation = build_live_cognition_observation(
        graph_result={},
        persona_state={},
        run_id=run_id,
        cognition_invocation_id=invocation_id,
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=None,
        failure_code="",
        generated_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    assert observation is not None
    return observation.model_dump(mode="json")


def _valid_service_spec() -> dict:
    """Return one valid registry service spec for mutation tests."""

    spec = {
        "id": "brain",
        "display_name": "Brain service",
        "kind": "backend",
        "command": ["python", "-m", "kazusa_ai_chatbot.main"],
        "cwd": ".",
        "env": {},
        "dependencies": [],
        "health_url": "http://127.0.0.1:8000/health",
        "autostart": False,
    }
    return spec


def test_service_contracts_reject_extra_fields_and_unbounded_strings() -> None:
    """Registry and API contracts should fail closed on unsafe input."""

    from control_console.contracts import (
        ConsoleDebugChatRequest,
        ProcessLogQuery,
        ServiceActionRequest,
        ServiceSpec,
    )

    extra_field = dict(_valid_service_spec())
    extra_field["pid"] = 1234
    with pytest.raises(ValidationError):
        ServiceSpec.model_validate(extra_field)

    shell_string = dict(_valid_service_spec())
    shell_string["command"] = ["python -m kazusa_ai_chatbot.main"]
    with pytest.raises(ValidationError):
        ServiceSpec.model_validate(shell_string)

    unbounded_reason = {"reason": "x" * 241}
    with pytest.raises(ValidationError):
        ServiceActionRequest.model_validate(unbounded_reason)

    huge_log_query = {"service_id": "brain", "limit": 501}
    with pytest.raises(ValidationError):
        ProcessLogQuery.model_validate(huge_log_query)

    huge_debug_body = {
        "channel_id": "debug",
        "user_id": "operator",
        "user_display_name": "Operator",
        "message_text": "x" * 4001,
    }
    with pytest.raises(ValidationError):
        ConsoleDebugChatRequest.model_validate(huge_debug_body)


def test_console_response_contract_uses_view_envelopes_for_bootstrap_and_debug() -> None:
    """Console responses should wrap, rather than reconstruct, Brain observations."""

    from control_console.contracts import (
        ConsoleCognitionObservationView,
        ConsoleDebugChatResponse,
        ControlConsoleBootstrapResponse,
    )
    from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
        CognitionRunObservationV1,
    )

    observation = CognitionRunObservationV1.model_validate(
        _canonical_live_observation()
    )
    view = ConsoleCognitionObservationView.model_validate({
        "view_kind": "debug_latest",
        "availability": "available",
        "reason_code": "",
        "generated_at": "2026-08-26T00:00:01Z",
        "observation": observation.model_dump(mode="json"),
    })
    response = ConsoleDebugChatResponse.model_validate({
        "request_id": "request-1",
        "brain_available": True,
        "request": {},
        "response": {},
        "tracking_id": None,
        "trace_id": "",
        "delivery_tracking_id": None,
        "llm_trace_id": "",
        "latency_ms": 1,
        "sent_at": "2026-08-26T00:00:02Z",
        "error": None,
        "cognition_observation": view.model_dump(mode="json"),
    })

    assert isinstance(response.cognition_observation, ConsoleCognitionObservationView)
    assert isinstance(
        response.cognition_observation.observation,
        CognitionRunObservationV1,
    )
    assert "cognition_observation" in ConsoleDebugChatResponse.model_fields
    assert "cognition_graph" not in ConsoleDebugChatResponse.model_fields
    assert "latest_cognition_observation" in (
        ControlConsoleBootstrapResponse.model_fields
    )
    assert "latest_self_cognition_observation" in (
        ControlConsoleBootstrapResponse.model_fields
    )
    assert "latest_cognition_graph" not in ControlConsoleBootstrapResponse.model_fields
    assert "latest_self_cognition_graph" not in (
        ControlConsoleBootstrapResponse.model_fields
    )

