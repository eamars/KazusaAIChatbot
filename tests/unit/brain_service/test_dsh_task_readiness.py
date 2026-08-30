"""Executable tests for the Brain-side DSH task readiness contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_task_capability_is_available_only_when_full_dsh_runtime_is_ready() -> None:
    """Readiness validates the sidecar and Brain bridge as one closed carrier."""

    from kazusa_ai_chatbot.brain_service.contracts import (
        DshInteractionHealthResponseV1,
    )

    ready_payload = {
        "schema_version": "dsh_brain_interaction_health.v1",
        "status": "ready",
        "configured": True,
        "durable_store": True,
        "cognition_judge": True,
        "task_resolution": {
            "status": "ready",
            "sidecar_identity": "sidecar-v2",
            "brain_bridge_identity": "brain-v2",
        },
    }
    health = DshInteractionHealthResponseV1.model_validate(ready_payload)
    assert health.task_resolution.status == "ready"
    assert health.task_resolution.sidecar_identity == "sidecar-v2"

    unavailable_payload = {
        **ready_payload,
        "task_resolution": {
            "status": "unavailable",
            "sidecar_identity": "sidecar-v2",
            "brain_bridge_identity": "brain-v2",
        },
    }
    unavailable = DshInteractionHealthResponseV1.model_validate(unavailable_payload)
    assert unavailable.task_resolution.status == "unavailable"

    with pytest.raises(ValidationError):
        DshInteractionHealthResponseV1.model_validate({
            **ready_payload,
            "task_resolution": {
                "status": "ready",
                "sidecar_identity": "",
                "brain_bridge_identity": "brain-v2",
            },
        })
