"""Contract tests for scoped runtime pipeline coordination."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_background_admission_defers_for_same_scope_foreground() -> None:
    """Same-scope background work must wait behind foreground work."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )
    foreground = await coordinator.start_run(
        scope=scope,
        owner="service",
        precedence="foreground",
        run_kind="chat",
    )

    assert foreground.admitted is True
    assert foreground.handle is not None
    async with foreground.handle:
        background = await coordinator.start_run(
            scope=scope,
            owner="self_cognition",
            precedence="background",
            run_kind="group_self_cognition_review",
        )

    assert background.admitted is False
    assert background.handle is None
    assert background.defer_reason == "same_scope_foreground_active"


@pytest.mark.asyncio
async def test_request_cancellation_marks_same_scope_background_only() -> None:
    """Foreground cancellation requests should target matching background runs."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    target_scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )
    other_scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-2",
        channel_type="group",
    )
    target = await coordinator.start_run(
        scope=target_scope,
        owner="self_cognition",
        precedence="background",
        run_kind="group_self_cognition_review",
    )
    other = await coordinator.start_run(
        scope=other_scope,
        owner="self_cognition",
        precedence="background",
        run_kind="group_self_cognition_review",
    )

    assert target.handle is not None
    assert other.handle is not None
    async with target.handle, other.handle:
        cancelled_run_ids = coordinator.request_cancellation(
            scope=target_scope,
            requested_by="service",
            reason="same_scope_foreground_pending",
        )

        assert cancelled_run_ids == [target.handle.run_id]
        assert target.handle.cancelled() is True
        assert other.handle.cancelled() is False


@pytest.mark.asyncio
async def test_different_scope_background_survives_cancellation() -> None:
    """A foreground event in one channel must not cancel another channel."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    target_scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )
    other_scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-2",
        channel_type="group",
    )
    other = await coordinator.start_run(
        scope=other_scope,
        owner="self_cognition",
        precedence="background",
        run_kind="group_self_cognition_review",
    )

    assert other.handle is not None
    async with other.handle:
        cancelled_run_ids = coordinator.request_cancellation(
            scope=target_scope,
            requested_by="service",
            reason="same_scope_foreground_pending",
        )

        assert cancelled_run_ids == []
        assert other.handle.cancelled() is False


@pytest.mark.asyncio
async def test_cancelled_checkpoint_raises_pipeline_cancelled() -> None:
    """Cancelled handles should fail closed at deterministic checkpoints."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCancelled,
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )
    admission = await coordinator.start_run(
        scope=scope,
        owner="self_cognition",
        precedence="background",
        run_kind="group_self_cognition_review",
    )

    assert admission.handle is not None
    async with admission.handle:
        coordinator.request_cancellation(
            scope=scope,
            requested_by="service",
            reason="same_scope_foreground_pending",
        )
        with pytest.raises(PipelineCancelled) as exc_info:
            admission.handle.raise_if_cancelled("before_dispatch")

    cancellation = exc_info.value.cancellation
    assert cancellation.run_id == admission.handle.run_id
    assert cancellation.scope == scope
    assert cancellation.requested_by == "service"
    assert cancellation.reason == "same_scope_foreground_pending"
    assert cancellation.checkpoint == "before_dispatch"


@pytest.mark.asyncio
async def test_handle_context_manager_releases_after_exception() -> None:
    """Foreground handles must not leak after exception unwinds."""

    from kazusa_ai_chatbot.runtime_coordination import (
        PipelineCoordinator,
        PipelineScope,
    )

    coordinator = PipelineCoordinator()
    scope = PipelineScope(
        platform="qq",
        platform_channel_id="group-1",
        channel_type="group",
    )
    foreground = await coordinator.start_run(
        scope=scope,
        owner="service",
        precedence="foreground",
        run_kind="chat",
    )

    assert foreground.handle is not None
    with pytest.raises(RuntimeError):
        async with foreground.handle:
            raise RuntimeError("turn crashed")

    background = await coordinator.start_run(
        scope=scope,
        owner="self_cognition",
        precedence="background",
        run_kind="group_self_cognition_review",
    )

    assert background.admitted is True
    assert background.handle is not None
    async with background.handle:
        assert background.handle.cancelled() is False
