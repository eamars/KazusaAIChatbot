"""Focused contracts for the top-level control console."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


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


def test_chain_run_projection_is_strict_bounded_and_optional() -> None:
    """Chain-run snapshots reject raw fields and default to not_reported."""

    from control_console.contracts import CognitionChainRunSnapshot

    missing = CognitionChainRunSnapshot()
    assert missing.status == "not_reported"
    assert missing.warning_codes == []

    valid = CognitionChainRunSnapshot.model_validate({
        "status": "completed",
        "chain_run_id": "cogchain_1",
        "run_id": "run-1",
        "llm_trace_id": "trace-1",
        "cognition_invocation_id": "invocation-1",
        "chain_model_name": "chain-model",
        "sidecar_model_name": "sidecar-model",
        "terminal_disposition": "complete",
        "started_at": "2026-08-20T00:00:00Z",
        "completed_at": "2026-08-20T00:00:01Z",
        "step_count": 8,
        "warning_codes": ["bounded_warning"],
    })
    assert valid.run_id == "run-1"
    assert valid.step_count == 8

    with pytest.raises(ValidationError):
        CognitionChainRunSnapshot.model_validate({
            "status": "completed",
            "raw_prompt": "forbidden",
        })

