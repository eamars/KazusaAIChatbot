"""Public accepted-task lifecycle functions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from uuid import uuid4

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASK_REQUESTED_DELIVERY,
    ACCEPTED_TASK_SCHEMA_VERSION,
    AcceptedTaskCreateRequest,
    AcceptedTaskCreateResult,
    AcceptedTaskDoc,
    AcceptedTaskIdentityMaterial,
    AcceptedTaskStatusCheckRequest,
    AcceptedTaskStatusResult,
)
from kazusa_ai_chatbot.db import accepted_tasks as repository


def build_task_identity_key(request: Mapping[str, object]) -> str:
    """Build the active-task duplicate key from trusted scope and semantics.

    Args:
        request: Accepted-task creation request or equivalent mapping. The
            source message id is intentionally ignored because repeat turns and
            progress turns have distinct source messages.

    Returns:
        A stable SHA-256 identity string for active duplicate rejection.
    """

    material = _identity_material(request)
    serialized = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    identity_key = f"accepted_task:v2:{digest}"
    return identity_key


async def create_or_return_active_accepted_task(
    request: AcceptedTaskCreateRequest,
) -> AcceptedTaskCreateResult:
    """Create an enqueueing task or return the active duplicate.

    Args:
        request: Trusted semantic task request built after cognition accepted a
            delayed user-facing task.

    Returns:
        A created or existing active task row.
    """

    task_identity_key = build_task_identity_key(request)
    task = _build_enqueueing_task_doc(
        request,
        task_identity_key=task_identity_key,
    )
    result = await repository.insert_or_get_active_accepted_task(
        task,
        source_message_id=_text(request, "source_message_id"),
        observed_at=_text(request, "storage_timestamp_utc"),
    )
    return result


async def mark_accepted_task_pending(
    *,
    accepted_task_id: str,
    executor_ref: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Move an enqueueing task to pending before a job becomes claimable."""

    task = await repository.mark_accepted_task_pending(
        accepted_task_id=accepted_task_id,
        executor_ref=executor_ref,
        updated_at=updated_at,
    )
    return task


async def mark_accepted_task_enqueue_failed(
    *,
    accepted_task_id: str,
    failure_summary: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Mark a task enqueue failure and release its active duplicate key."""

    task = await repository.mark_accepted_task_enqueue_failed(
        accepted_task_id=accepted_task_id,
        failure_summary=failure_summary,
        updated_at=updated_at,
    )
    return task


async def recover_stale_enqueueing_tasks(
    *,
    stale_before_utc: str,
    recovered_at: str,
) -> int:
    """Release stale enqueueing locks left by an interrupted queue insert."""

    recovered_count = await repository.recover_stale_enqueueing_tasks(
        stale_before_utc=stale_before_utc,
        recovered_at=recovered_at,
    )
    return recovered_count


async def recover_stale_delivery_in_progress_tasks(
    *,
    stale_before_utc: str,
    recovered_at: str,
) -> int:
    """Recover interrupted delivery claims for a later retry."""

    recovered_count = await repository.recover_stale_delivery_in_progress_tasks(
        stale_before_utc=stale_before_utc,
        recovered_at=recovered_at,
    )
    return recovered_count


async def check_accepted_task_status(
    request: AcceptedTaskStatusCheckRequest,
) -> AcceptedTaskStatusResult:
    """Return the newest active task for a trusted progress-check scope."""

    task = await repository.find_active_accepted_task_for_scope(request)
    if task is None:
        result: AcceptedTaskStatusResult = {"status": "none"}
        return result
    result = {
        "status": "active",
        "task": task,
    }
    return result


async def load_open_coding_run_contexts_for_scope(
    *,
    source_platform: str,
    source_channel_id: str,
    requester_global_user_id: str,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Load current prompt-safe coding contexts for a trusted user turn."""

    contexts = await repository.load_open_coding_run_contexts_for_scope(
        source_platform=source_platform,
        source_channel_id=source_channel_id,
        requester_global_user_id=requester_global_user_id,
        limit=limit,
    )
    return contexts


async def mark_accepted_task_running(
    *,
    accepted_task_id: str,
    started_at: str,
) -> AcceptedTaskDoc | None:
    """Move a pending accepted task to running when the executor claims it."""

    task = await repository.mark_accepted_task_running(
        accepted_task_id=accepted_task_id,
        started_at=started_at,
    )
    return task


async def mark_tool_result_ready(
    *,
    accepted_task_id: str,
    artifact_text: str,
    result_summary: str,
    completed_at: str,
    result_kind: str = "resolved",
    completion_status: str = "resolved",
    remaining_needs: list[str] | None = None,
    coding_run_context: dict[str, object] | None = None,
) -> AcceptedTaskDoc | None:
    """Record a prompt-safe terminal task result for source-bound delivery."""

    task = await repository.mark_tool_result_ready(
        accepted_task_id=accepted_task_id,
        artifact_text=artifact_text,
        result_summary=result_summary,
        completed_at=completed_at,
        result_kind=result_kind,
        completion_status=completion_status,
        remaining_needs=remaining_needs or [],
        coding_run_context=coding_run_context,
    )
    return task


async def mark_accepted_task_failure_ready(
    *,
    accepted_task_id: str,
    failure_summary: str,
    completed_at: str,
    result_kind: str = "failed",
    remaining_needs: list[str] | None = None,
    coding_run_context: dict[str, object] | None = None,
) -> AcceptedTaskDoc | None:
    """Record a failed executor result and make it ready for delivery."""

    task = await repository.mark_accepted_task_failure_ready(
        accepted_task_id=accepted_task_id,
        failure_summary=failure_summary,
        completed_at=completed_at,
        result_kind=result_kind,
        remaining_needs=remaining_needs or [],
        coding_run_context=coding_run_context,
    )
    return task


async def mark_accepted_task_delivery_in_progress(
    *,
    accepted_task_id: str,
    delivery_tracking_id: str,
    updated_at: str,
) -> AcceptedTaskDoc | None:
    """Claim a ready accepted-task result for dispatcher delivery."""

    task = await repository.mark_accepted_task_delivery_in_progress(
        accepted_task_id=accepted_task_id,
        delivery_tracking_id=delivery_tracking_id,
        updated_at=updated_at,
    )
    return task


async def mark_future_speak_accepted_task_delivered(
    *,
    accepted_task_id: str,
    delivered_at: str,
) -> AcceptedTaskDoc | None:
    """Complete a running future-speak task after scheduling succeeds."""

    task = await repository.mark_future_speak_accepted_task_delivered(
        accepted_task_id=accepted_task_id,
        delivered_at=delivered_at,
    )
    return task


async def mark_accepted_task_delivered(
    *,
    accepted_task_id: str,
    delivered_conversation_message_id: str,
    delivered_at: str,
) -> AcceptedTaskDoc | None:
    """Mark an accepted task delivered and release duplicate suppression."""

    task = await repository.mark_accepted_task_delivered(
        accepted_task_id=accepted_task_id,
        delivered_conversation_message_id=delivered_conversation_message_id,
        delivered_at=delivered_at,
    )
    return task


async def mark_accepted_task_delivery_failed(
    *,
    accepted_task_id: str,
    failure_summary: str,
    failed_at: str,
) -> AcceptedTaskDoc | None:
    """Record a delivery failure while keeping the result visible to ops."""

    task = await repository.mark_accepted_task_delivery_failed(
        accepted_task_id=accepted_task_id,
        failure_summary=failure_summary,
        failed_at=failed_at,
    )
    return task


def _build_enqueueing_task_doc(
    request: AcceptedTaskCreateRequest,
    *,
    task_identity_key: str,
) -> AcceptedTaskDoc:
    """Build the durable enqueueing task row."""

    storage_timestamp_utc = _text(request, "storage_timestamp_utc")
    source_message_id = _text(request, "source_message_id")
    identity_material = _identity_material(request)
    task: AcceptedTaskDoc = {
        "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        "accepted_task_id": f"task-{uuid4().hex}",
        "task_identity_key": task_identity_key,
        "active_identity_key": task_identity_key,
        "task_identity_material": identity_material,
        "task_kind": _task_kind(request),
        "semantic_objective": _text(request, "semantic_objective"),
        "first_source_message_id": source_message_id,
        "related_source_message_ids": [source_message_id]
        if source_message_id
        else [],
        "source_trigger_source": _text(request, "source_trigger_source"),
        "state": "enqueueing",
        "completion_status": "none",
        "result_kind": "none",
        "executor_kind": "background_work",
        "executor_ref": "",
        "accepted_task_summary": _text(request, "accepted_task_summary"),
        "requested_delivery": ACCEPTED_TASK_REQUESTED_DELIVERY,
        "max_output_chars": int(request["max_output_chars"]),
        "source_platform": _text(request, "source_platform"),
        "source_channel_id": _text(request, "source_channel_id"),
        "source_channel_type": _text(request, "source_channel_type"),
        "source_platform_bot_id": _text(request, "source_platform_bot_id"),
        "source_character_name": _text(request, "source_character_name"),
        "requester_global_user_id": _text(request, "requester_global_user_id"),
        "requester_platform_user_id": _text(
            request,
            "requester_platform_user_id",
        ),
        "requester_display_name": _text(request, "requester_display_name"),
        "created_at": storage_timestamp_utc,
        "updated_at": storage_timestamp_utc,
        "started_at": "",
        "completed_at": "",
        "delivered_at": "",
        "result_summary": "",
        "artifact_text": "",
        "remaining_needs": [],
        "failure_summary": "",
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "last_progress_reported_at": "",
        "coding_run_context": {},
    }
    return task


def _identity_material(
    request: Mapping[str, object],
) -> AcceptedTaskIdentityMaterial:
    """Return canonical material for active duplicate matching."""

    material: AcceptedTaskIdentityMaterial = {
        "source_platform": _text(request, "source_platform"),
        "source_channel_id": _text(request, "source_channel_id"),
        "source_channel_type": _text(request, "source_channel_type"),
        "requester_global_user_id": _text(request, "requester_global_user_id"),
        "requester_platform_user_id": _text(
            request,
            "requester_platform_user_id",
        ),
        "semantic_objective": _normalized_semantic_text(
            request,
            "semantic_objective",
        ),
    }
    return material


def _normalized_semantic_text(
    request: Mapping[str, object],
    field_name: str,
) -> str:
    """Normalize structured semantic text without classifying its meaning."""

    text = _text(request, field_name)
    normalized = " ".join(text.split())
    return normalized


def _text(request: Mapping[str, object], field_name: str) -> str:
    value = request.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _task_kind(request: Mapping[str, object]) -> str:
    """Return one declared v2 lifecycle kind from trusted creation material."""

    value = _text(request, "task_kind")
    if value not in {"task_resolution", "future_speak", "coding_continuation"}:
        raise ValueError("task_kind is invalid")
    return value
