"""Retained deterministic coverage for the canonical DSH task service."""

from __future__ import annotations

import importlib

import pytest

from tests.task_resolution_test_helpers import (
    InMemoryDshBindingStore,
    _context,
    _goal_continuation_ref,
    _resolution_ref,
)


def _request() -> dict[str, object]:
    """Build the canonical resolver request used by queue fixtures."""

    return {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve one bounded public question.",
        "reason": "The current response lacks required evidence.",
        "evidence_handles": ["e1"],
        "start_in_background": False,
        "goal_continuation_ref": _goal_continuation_ref(),
    }


def _terminal_exhaust() -> dict[str, object]:
    """Build a minimal terminal exhaust accepted by the DSH projection."""

    return {
        "kind": "terminal",
        "terminal": {
            "summary": "One bounded fact was resolved.",
            "findings": ["One bounded fact was resolved."],
            "completed_subgoals": ["Resolve one bounded public question."],
            "remaining_needs": [],
            "artifact_refs": [],
        },
        "evidence": [{
            "evidence_id": "evidence-1",
            "semantic_ref": "evidence-ref-1",
            "content_digest": "sha256:evidence-1",
        }],
    }


def test_inline_budget_defaults_are_bounded() -> None:
    """The task service exposes the approved bounded foreground interval."""

    config = importlib.import_module("kazusa_ai_chatbot.config")
    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")

    assert config.TASK_RESOLUTION_INLINE_BUDGET_SECONDS == 30.0
    assert service.MINIMUM_INLINE_BUDGET_SECONDS == 0.001
    assert service.MAXIMUM_INLINE_BUDGET_SECONDS == 120.0


@pytest.mark.asyncio
async def test_inline_without_runtime_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing Brain-composed runtime cannot enter a legacy fallback path."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    monkeypatch.setattr(service, "_TASK_RESOLUTION_RUNTIME", None)

    with pytest.raises(ValueError, match="Brain-composed DSH runtime"):
        await service.resolve_task_inline(
            _request(),
            _context(),
            inline_budget_seconds=30.0,
        )


@pytest.mark.asyncio
async def test_inline_runtime_checkpoint_is_projected_without_reclassification() -> None:
    """A cooperative runtime checkpoint maps to the stable task result carrier."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    context = _context()
    request = _request()
    session_id = service._task_session_id(
        request,
        service._context_for_service(context),
    )
    reference = _resolution_ref(
        session_id=session_id,
        thread_id="thread-task-001",
        segment_id="segment-task-001",
        activation_id="activation-task-001",
    )

    class Runtime:
        async def open(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "kind": "checkpointed",
                "checkpoint": reference,
            }

        async def request_checkpoint(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return reference

    result = await service.resolve_task_inline(
        request,
        context,
        inline_budget_seconds=30.0,
        runtime=Runtime(),
        binding_store=InMemoryDshBindingStore(),
    )

    assert result["status"] == "deferred"
    assert result["checkpoint"]["schema_version"] == "dsh_resolution_ref.v1"
    assert result["evidence"] == []


@pytest.mark.asyncio
async def test_background_start_queues_generation_zero_without_authority() -> None:
    """Background admission fails closed before a worker-owned DSH ref exists."""

    service = importlib.import_module("kazusa_ai_chatbot.task_resolution.service")
    with pytest.raises(ValueError, match="Brain-composed DSH runtime"):
        await service.start_task_resolution_in_background(
            _request(),
            _context(),
            source_trigger_source="user_message",
            source_platform_bot_id="debug-bot",
            requester_display_name="Test User",
        )
