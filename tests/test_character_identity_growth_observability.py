"""Health precedence tests for character identity growth."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.character_identity_growth.identity import (
    derive_growth_health_state,
)
from kazusa_ai_chatbot.event_logging import (
    record_character_identity_growth_event,
)
import kazusa_ai_chatbot.event_logging.recording as recording_module


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"receipt_status": "mismatch"}, "consumption_error"),
        ({"latest_run_lifecycle_state": "failed"}, "pipeline_error"),
        ({"latest_revision_number": 1}, "awaiting_consumption"),
        ({"ready_candidate_count": 1}, "promotion_ready"),
        ({"latest_reason_code": "review_rejected"}, "semantic_rejection"),
        ({"latest_reason_code": "privacy_blocked"}, "semantic_rejection"),
        ({"latest_reason_code": "candidate_emerging"}, "waiting_for_evidence"),
        ({"latest_reason_code": "cadence_wait"}, "waiting_for_evidence"),
        ({"latest_reason_code": "duplicate_root"}, "waiting_for_evidence"),
        (
            {
                "latest_revision_number": 1,
                "receipt_status": "consumed",
            },
            "healthy_active",
        ),
        ({}, "healthy_idle"),
    ],
)
def test_health_state_uses_closed_precedence(
    overrides: dict[str, object],
    expected: str,
) -> None:
    """Operator health should distinguish data scarcity from broken process."""

    arguments = {
        "latest_revision_number": 0,
        "receipt_status": None,
        "latest_run_lifecycle_state": None,
        "latest_reason_code": "not_routed",
        "ready_candidate_count": 0,
        "emerging_candidate_count": 0,
    }
    arguments.update(overrides)

    assert derive_growth_health_state(**arguments) == expected


def test_consumption_and_pipeline_errors_outrank_other_states() -> None:
    """A serious failure must not be masked by a ready or emerging candidate."""

    mismatch = derive_growth_health_state(
        latest_revision_number=2,
        receipt_status="mismatch",
        latest_run_lifecycle_state="failed",
        latest_reason_code="candidate_ready",
        ready_candidate_count=1,
        emerging_candidate_count=1,
    )
    pipeline = derive_growth_health_state(
        latest_revision_number=2,
        receipt_status="consumed",
        latest_run_lifecycle_state="failed",
        latest_reason_code="candidate_ready",
        ready_candidate_count=1,
        emerging_candidate_count=1,
    )

    assert mismatch == "consumption_error"
    assert pipeline == "pipeline_error"


@pytest.mark.asyncio
async def test_consumption_event_contains_only_receipt_metadata(
    monkeypatch,
) -> None:
    """Mirrored consumption telemetry must exclude identity and evidence text."""

    captured: dict[str, object] = {}

    async def write_event(document):
        captured.update(document)
        return str(document["event_id"])

    monkeypatch.setattr(
        recording_module.repository,
        "write_event",
        write_event,
    )
    result = await record_character_identity_growth_event(
        event_type="consumption",
        stage="latest_identity_reader",
        reason_code="revision_consumed",
        status="consumed",
        correlation_id="correlation-1",
        run_id="run-1",
        revision_number=3,
        consumer_count=8,
        projection_digest="a" * 64,
    )

    assert result["accepted"] is True
    assert captured["event_family"] == "character_identity_growth"
    assert captured["payload"] == {
        "stage": "latest_identity_reader",
        "reason_code": "revision_consumed",
        "consumer_count": 8,
        "projection_digest": "a" * 64,
        "revision_number": 3,
    }
    serialized = json.dumps(captured, sort_keys=True)
    assert "effective_identity" not in serialized
    assert "evidence_refs" not in serialized
    assert "change_diff" not in serialized
