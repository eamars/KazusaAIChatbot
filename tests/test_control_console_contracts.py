"""Focused contracts for the top-level control console."""

from __future__ import annotations

from pydantic import ValidationError
import pytest


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

