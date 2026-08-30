"""Public accepted-task lifecycle functions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
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
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.db import accepted_tasks as repository


async def claim_dsh_followup(
    *,
    control: Mapping[str, object],
    action_attempt_id: str,
    task: Mapping[str, object],
    binding: Mapping[str, object],
    repository: object | None = None,
) -> dict[str, object]:
    """Claim one typed follow-up for the same DSH session exactly once."""

    from kazusa_ai_chatbot.task_resolution.contracts import (
        validate_accepted_task_control,
    )

    validated = validate_accepted_task_control(control)
    task_id = _text(task, "accepted_task_id")
    expected_ref = f"accepted_task:{task_id}"
    if validated["accepted_task_ref"] != expected_ref:
        raise ValueError("accepted_task_ref does not match the affordance")
    if not action_attempt_id.strip():
        raise ValueError("action_attempt_id is required")
    followup_open = task.get("dsh_followup_open")
    if followup_open is not True:
        raise ValueError("accepted task follow-up is not open")
    session_id = _text(binding, "task_session_id")
    generation = binding.get("operation_generation")
    if not session_id or not isinstance(generation, int) or generation < 0:
        raise ValueError("DSH binding identity is invalid")
    next_generation = generation + 1
    target_repository = repository or globals()["repository"]
    method = getattr(target_repository, "create_followup", None)
    if not callable(method):
        raise TypeError("accepted-task repository lacks create_followup")
    value = method(
        accepted_task_id=task_id,
        task_session_id=session_id,
        operation=validated["operation"],
        instruction=validated["instruction"],
        action_attempt_id=action_attempt_id,
        operation_generation=next_generation,
        binding=dict(binding),
    )
    if hasattr(value, "__await__"):
        value = await value
    if not isinstance(value, Mapping):
        raise TypeError("accepted-task repository returned an invalid follow-up")
    return dict(value)


async def create_followup(
    *,
    accepted_task_id: str,
    task_session_id: str,
    operation: str,
    instruction: str | None,
    action_attempt_id: str,
    operation_generation: int,
    binding: Mapping[str, object],
    expected_revision: int | None = None,
    updated_at: str | None = None,
) -> AcceptedTaskDoc:
    """Create one generation-bound accepted-task continuation row."""

    if not isinstance(binding, Mapping):
        raise TypeError("DSH binding is required")
    bound_session = _text(binding, "task_session_id")
    if bound_session != task_session_id:
        raise ValueError("DSH binding session does not match follow-up")
    bound_generation = binding.get("operation_generation")
    if (
        not isinstance(bound_generation, int)
        or isinstance(bound_generation, bool)
        or operation_generation != bound_generation + 1
    ):
        raise ValueError("DSH follow-up generation does not match binding")
    if binding.get("state") != "terminal":
        raise ValueError("DSH follow-up requires a terminal binding")
    revision = expected_revision
    if revision is None:
        candidate = binding.get("revision", 0)
        if not isinstance(candidate, int) or isinstance(candidate, bool):
            raise ValueError("DSH binding revision is invalid")
        revision = candidate
    value = repository.create_followup(
        accepted_task_id=accepted_task_id,
        task_session_id=task_session_id,
        operation=operation,
        instruction=instruction,
        action_attempt_id=action_attempt_id,
        operation_generation=operation_generation,
        binding=dict(binding),
        expected_revision=revision,
        updated_at=updated_at,
    )
    if hasattr(value, "__await__"):
        value = await value
    if not isinstance(value, Mapping):
        raise TypeError("accepted-task repository returned an invalid follow-up")
    return dict(value)


def _text(value: Mapping[str, object], field_name: str) -> str:
    field_value = value.get(field_name)
    return field_value.strip() if isinstance(field_value, str) else ""


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
    *,
    dsh_task_session_id: str | None = None,
    dsh_operation_generation: int = 0,
    dsh_followup_open: bool = False,
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
        dsh_task_session_id=dsh_task_session_id,
        dsh_operation_generation=dsh_operation_generation,
        dsh_followup_open=dsh_followup_open,
    )
    result = await repository.insert_or_get_active_accepted_task(
        task,
        source_message_id=_text(request, "source_message_id"),
        observed_at=_text(request, "storage_timestamp_utc"),
    )
    return result


async def find_accepted_task_by_id(
    *,
    accepted_task_id: str,
) -> AcceptedTaskDoc | None:
    """Load one accepted-task row for durable DSH control resolution."""

    method = getattr(repository, "find_accepted_task_by_id", None)
    if not callable(method):
        raise TypeError("accepted-task repository lacks id lookup")
    value = method(accepted_task_id=accepted_task_id)
    if hasattr(value, "__await__"):
        value = await value
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("accepted-task repository returned an invalid task")
    return dict(value)


async def find_dsh_followup_by_action_attempt(
    *,
    task_session_id: str,
    action_attempt_id: str,
    operation_generation: int,
) -> AcceptedTaskDoc | None:
    """Load one durable follow-up replay without process-local state."""

    method = getattr(repository, "find_dsh_followup_by_action_attempt", None)
    if not callable(method):
        raise TypeError("accepted-task repository lacks follow-up replay lookup")
    value = method(
        task_session_id=task_session_id,
        action_attempt_id=action_attempt_id,
        operation_generation=operation_generation,
    )
    if hasattr(value, "__await__"):
        value = await value
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("accepted-task repository returned an invalid replay")
    return dict(value)


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
        task = await repository.find_open_dsh_followup_for_scope(request)
    if task is None:
        result: AcceptedTaskStatusResult = {"status": "none"}
        return result
    result = {
        "status": "active",
        "task": task,
    }
    return result


async def load_open_dsh_task_affordances_for_scope(
    *,
    platform: str,
    source_channel_id: str,
    requester_global_user_id: str,
    source_channel_type: str | None = None,
    requester_platform_user_id: str | None = None,
    limit: int = 3,
) -> list[dict[str, object]]:
    """Load prompt-safe DSH task affordances for one trusted scope."""

    affordances = await repository.load_open_dsh_task_affordances_for_scope(
        platform=platform,
        source_channel_id=source_channel_id,
        requester_global_user_id=requester_global_user_id,
        source_channel_type=source_channel_type,
        requester_platform_user_id=requester_platform_user_id,
        limit=limit,
    )
    return [dict(affordance) for affordance in affordances]


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
    )
    return task


async def mark_accepted_task_failure_ready(
    *,
    accepted_task_id: str,
    failure_summary: str,
    completed_at: str,
    result_kind: str = "failed",
    remaining_needs: list[str] | None = None,
) -> AcceptedTaskDoc | None:
    """Record a failed executor result and make it ready for delivery."""

    task = await repository.mark_accepted_task_failure_ready(
        accepted_task_id=accepted_task_id,
        failure_summary=failure_summary,
        completed_at=completed_at,
        result_kind=result_kind,
        remaining_needs=remaining_needs or [],
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
    dsh_task_session_id: str | None = None,
    dsh_operation_generation: int = 0,
    dsh_followup_open: bool = False,
) -> AcceptedTaskDoc:
    """Build the durable enqueueing task row."""

    storage_timestamp_utc = _text(request, "storage_timestamp_utc")
    source_message_id = _text(request, "source_message_id")
    identity_material = _identity_material(request)
    task_kind = _task_kind(request)
    continuation_ref = _validated_goal_continuation_ref(
        request,
        task_kind=task_kind,
    )
    task: AcceptedTaskDoc = {
        "schema_version": ACCEPTED_TASK_SCHEMA_VERSION,
        "accepted_task_id": f"task-{uuid4().hex}",
        "task_identity_key": task_identity_key,
        "active_identity_key": task_identity_key,
        "task_identity_material": identity_material,
        "task_kind": task_kind,
        "semantic_objective": _text(request, "semantic_objective"),
        "goal_continuation_ref": continuation_ref,
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
        "revision": 0,
    }
    if task_kind == "task_resolution":
        if not isinstance(dsh_task_session_id, str) or not dsh_task_session_id.strip():
            dsh_task_session_id = (
                "session-" + task_identity_key.rsplit(":", 1)[-1][:32]
            )
        if (
            isinstance(dsh_operation_generation, bool)
            or not isinstance(dsh_operation_generation, int)
            or dsh_operation_generation < 0
        ):
            raise ValueError("dsh_operation_generation is invalid")
        task.update({
            "dsh_task_session_id": dsh_task_session_id.strip(),
            "dsh_operation_generation": dsh_operation_generation,
            "dsh_followup_open": bool(dsh_followup_open),
            "dsh_followup_claim_action_attempt_id": None,
        })
    authority = request.get("scheduled_future_speech_authority")
    if isinstance(authority, dict):
        task["scheduled_future_speech_authority"] = deepcopy(authority)
    return task


def _identity_material(
    request: Mapping[str, object],
) -> AcceptedTaskIdentityMaterial:
    """Return canonical material for active duplicate matching."""

    task_kind = _task_kind(request)
    material: AcceptedTaskIdentityMaterial = {
        "task_kind": task_kind,
        "source_platform": _text(request, "source_platform"),
        "source_channel_id": _text(request, "source_channel_id"),
        "source_channel_type": _text(request, "source_channel_type"),
        "requester_global_user_id": _text(request, "requester_global_user_id"),
        "requester_platform_user_id": _text(
            request,
            "requester_platform_user_id",
        ),
    }
    if task_kind == "task_resolution":
        continuation_ref = _validated_goal_continuation_ref(
            request,
            task_kind=task_kind,
        )
        if continuation_ref is None:
            raise ValueError(
                "task_resolution identity requires goal_continuation_ref"
            )
        material["goal_continuation_ref"] = continuation_ref
    else:
        material["semantic_objective"] = _normalized_semantic_text(
            request,
            "semantic_objective",
        )
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
    if value not in {"task_resolution", "future_speak"}:
        raise ValueError("task_kind is invalid")
    return value


def _validated_goal_continuation_ref(
    request: Mapping[str, object],
    *,
    task_kind: str,
) -> GoalContinuationRefV1 | None:
    """Return the exact persisted continuation reference for one task kind.

    Task-resolution rows require the deterministic reference selected by
    cognition; future-speak carries an explicit null reference until its own
    continuation lineage is bound.
    """

    raw_ref = request.get("goal_continuation_ref")
    if raw_ref is None:
        if task_kind == "task_resolution":
            raise ValueError(
                "task_resolution accepted task requires goal_continuation_ref"
            )
        return None
    try:
        continuation_ref = validate_goal_continuation_ref(raw_ref)
    except CognitiveEpisodeValidationError as exc:
        raise ValueError(
            f"goal_continuation_ref: invalid reference: {exc}"
        ) from exc
    return continuation_ref
