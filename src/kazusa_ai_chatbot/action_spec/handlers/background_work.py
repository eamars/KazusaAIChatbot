"""Queue handlers for retained future-speak and DSH task controls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskCreateRequest
from kazusa_ai_chatbot.action_spec.models import (
    ActionValidationError,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import FUTURE_SPEAK_CAPABILITY
from kazusa_ai_chatbot.background_work import enqueue_background_work_request
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    build_scheduled_future_speech_authority,
    validate_scheduled_authority_proposal,
    validate_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    CHARACTER_TIME_ZONE,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.task_resolution.contracts import (
    validate_accepted_task_control,
)
from kazusa_ai_chatbot.time_boundary import local_llm_datetime_to_storage_utc_iso

BackgroundWorkEnqueueFunc = Callable[
    [BackgroundWorkQueueRequest],
    Awaitable[BackgroundWorkQueueResult],
]

_REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS = (
    "source_platform",
    "source_channel_id",
    "source_channel_type",
    "source_message_id",
    "source_platform_bot_id",
    "source_character_name",
    "source_trigger_source",
    "requester_global_user_id",
    "requester_platform_user_id",
    "requester_display_name",
)
async def handle_accepted_task_control(
    *,
    control: Mapping[str, object],
    affordance: Mapping[str, object],
    trusted_scope: Mapping[str, object],
    claim_followup: Callable[..., Awaitable[object]],
) -> dict[str, object]:
    """Bind a typed control to the advertised task and trusted source scope."""

    del trusted_scope
    validated = validate_accepted_task_control(control)
    advertised_ref = affordance.get("accepted_task_ref")
    if validated["accepted_task_ref"] != advertised_ref:
        raise ValueError("accepted task control reference is not advertised")
    allowed = affordance.get("allowed_next_actions")
    if not isinstance(allowed, list) or validated["operation"] not in allowed:
        raise ValueError("accepted task control operation is unavailable")
    value = claim_followup(
        control=dict(validated),
        affordance=dict(affordance),
    )
    if hasattr(value, "__await__"):
        value = await value
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "status": "claimed",
        "accepted_task_ref": validated["accepted_task_ref"],
    }


def validate_future_speak_action(
    action_spec: dict[str, Any],
) -> dict[str, Any]:
    """Validate the retained deterministic future-speak queue action."""

    validated = validate_action_spec(action_spec)
    if validated["kind"] != FUTURE_SPEAK_CAPABILITY:
        raise ActionValidationError("kind: expected future_speak")
    _validate_private_background_target(validated)
    params = validated["params"]
    _validate_requested_delivery_and_output_limit(params)
    trigger_at = _required_param_text(params, "trigger_at")
    try:
        local_llm_datetime_to_storage_utc_iso(trigger_at)
    except ValueError as exc:
        raise ActionValidationError(
            f"trigger_at: expected exact local YYYY-MM-DD HH:MM: {exc}"
        ) from exc
    _required_param_text(params, "continuation_objective")
    _validate_scheduled_authority_input(validated)
    return validated


def _validate_scheduled_authority_input(
    validated: Mapping[str, Any],
) -> None:
    """Require the planner-owned proposal and source identity inputs."""

    params = validated["params"]
    proposal = params.get("scheduled_authority_proposal")
    if proposal is None:
        raise ActionValidationError(
            "params.scheduled_authority_proposal: required for future_speak"
        )
    try:
        validate_scheduled_authority_proposal(proposal)
    except (CognitionContractError, ValueError) as exc:
        raise ActionValidationError(
            f"params.scheduled_authority_proposal: invalid: {exc}"
        ) from exc
    for field_name in (
        "source_episode_id",
        "source_message_id",
        "accepted_at_utc",
    ):
        _required_param_text(params, field_name)
    scope = validated["target"]["scope"]
    if _required_param_text(
        params,
        "source_message_id",
    ) != _required_scope_text(scope, "source_message_id"):
        raise ActionValidationError(
            "params.source_message_id: trusted scope mismatch"
        )


async def enqueue_future_speak_action(
    action_spec: dict[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None = None,
    source_llm_trace_id: str = "",
    scheduled_authority_proposal: Mapping[str, object] | None = None,
) -> BackgroundWorkQueueResult:
    """Persist a retained future-speak task and its deterministic worker job."""

    validated = validate_future_speak_action(action_spec)
    params = validated["params"]
    trigger_at = _required_param_text(params, "trigger_at")
    continuation_objective = _required_param_text(
        params,
        "continuation_objective",
    )
    task_summary = f"Schedule future speak for {trigger_at}: {continuation_objective}"
    authority = _build_scheduled_authority(
        validated=validated,
        proposal=(
            scheduled_authority_proposal
            if scheduled_authority_proposal is not None
            else params.get("scheduled_authority_proposal")
        ),
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        source_llm_trace_id=source_llm_trace_id,
    )
    return await _create_or_queue_accepted_task(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        source_llm_trace_id=source_llm_trace_id,
        task_kind="future_speak",
        semantic_objective=continuation_objective,
        accepted_task_summary=continuation_objective,
        requested_worker="future_speak",
        worker_payload={
            "trigger_at": trigger_at,
            "continuation_objective": continuation_objective,
        },
        enqueue_background_work_func=enqueue_background_work_func,
        task_brief=task_summary,
        scheduled_future_speech_authority=authority,
    )


def _validate_private_background_target(validated: Mapping[str, Any]) -> None:
    """Require the shared private user-scope queue boundary."""

    if validated["visibility"] != "private":
        raise ActionValidationError("visibility: expected private")
    if validated["urgency"] != "background":
        raise ActionValidationError("urgency: expected background")
    target = validated["target"]
    if target["owner"] != "background_work":
        raise ActionValidationError("owner: expected background_work")
    if target["target_kind"] != "current_user":
        raise ActionValidationError("target_kind: expected current_user")
    if target["target_id"] is not None:
        raise ActionValidationError("target_id: expected null")
    scope = target["scope"]
    for field_name in _REQUIRED_DELIVERY_TARGET_SCOPE_FIELDS:
        _required_scope_text(scope, field_name)
    if _required_scope_text(scope, "source_trigger_source") != "user_message":
        raise ActionValidationError(
            "scope.source_trigger_source: expected user_message"
        )


def _validate_requested_delivery_and_output_limit(
    params: Mapping[str, object],
) -> None:
    """Validate shared accepted-task queue settings."""

    if params.get("requested_delivery") != BACKGROUND_WORK_REQUESTED_DELIVERY:
        raise ActionValidationError("requested_delivery: unsupported value")
    max_output_chars = params.get("max_output_chars")
    if isinstance(max_output_chars, bool) or not isinstance(
        max_output_chars,
        int,
    ):
        raise ActionValidationError("max_output_chars: expected integer")
    if max_output_chars < 1:
        raise ActionValidationError("max_output_chars: expected positive integer")
    if max_output_chars > BACKGROUND_WORK_OUTPUT_CHAR_LIMIT:
        raise ActionValidationError("max_output_chars: exceeds configured limit")


def _build_scheduled_authority(
    *,
    validated: Mapping[str, Any],
    proposal: object,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    source_llm_trace_id: str,
) -> dict[str, Any]:
    """Construct and validate the immutable pre-persistence authority.

    The authority is created here, before accepted-task, job, schedule, and
    run ids exist, and is copied unchanged into every carrier.
    """

    if not isinstance(proposal, Mapping):
        raise ActionValidationError(
            "scheduled_authority_proposal: required for future_speak"
        )
    try:
        validated_proposal = validate_scheduled_authority_proposal(proposal)
    except (CognitionContractError, ValueError) as exc:
        raise ActionValidationError(
            f"scheduled_authority_proposal: invalid: {exc}"
        ) from exc
    if validated_proposal["temporal_alignment"] != "aligned":
        raise ActionValidationError(
            "scheduled_authority_proposal.temporal_alignment: expected aligned"
        )
    params = validated["params"]
    scope = validated["target"]["scope"]
    channel_type = _required_scope_text(scope, "source_channel_type")
    audience_kind = "group" if channel_type == "group" else "private"
    try:
        authority = build_scheduled_future_speech_authority(
            source_episode_id=_required_param_text(
                params,
                "source_episode_id",
            ),
            source_message_id=_required_scope_text(
                scope,
                "source_message_id",
            ),
            source_action_attempt_id=action_attempt_id,
            source_llm_trace_id=source_llm_trace_id,
            accepted_at_utc=_required_param_text(params, "accepted_at_utc"),
            timezone=CHARACTER_TIME_ZONE,
            trigger_local=_required_param_text(params, "trigger_at"),
            platform=_required_scope_text(scope, "source_platform"),
            channel_type=channel_type,
            audience_kind=audience_kind,
            semantic_objective=_required_param_text(
                params,
                "continuation_objective",
            ),
            authorized_content_summary=validated_proposal[
                "authorized_content_summary"
            ],
            authorized_detail_refs=validated_proposal[
                "authorized_detail_refs"
            ],
            goal_continuation_ref=validated.get("goal_continuation_ref"),
        )
    except (CognitionContractError, ValueError) as exc:
        raise ActionValidationError(
            f"scheduled future-speech authority is invalid: {exc}"
        ) from exc
    validate_scheduled_future_speech_authority(authority)
    return dict(authority)


async def _create_or_queue_accepted_task(
    validated: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    source_llm_trace_id: str,
    task_kind: str,
    semantic_objective: str,
    accepted_task_summary: str,
    requested_worker: str,
    worker_payload: dict[str, object],
    enqueue_background_work_func: BackgroundWorkEnqueueFunc | None,
    task_brief: str,
    scheduled_future_speech_authority: dict[str, Any] | None = None,
) -> BackgroundWorkQueueResult:
    """Create v2 accepted state and enqueue its single approved worker."""

    from kazusa_ai_chatbot.accepted_task.lifecycle import (
        create_or_return_active_accepted_task,
        mark_accepted_task_enqueue_failed,
        mark_accepted_task_pending,
    )

    create_request = _accepted_task_create_request(
        validated,
        storage_timestamp_utc=storage_timestamp_utc,
        task_kind=task_kind,
        semantic_objective=semantic_objective,
        accepted_task_summary=accepted_task_summary,
        scheduled_future_speech_authority=scheduled_future_speech_authority,
    )
    create_result = await create_or_return_active_accepted_task(create_request)
    accepted_task = create_result["task"]
    if (
        create_result["status"] == "already_active"
        and scheduled_future_speech_authority is not None
    ):
        _reject_active_duplicate_authority_mismatch(
            accepted_task,
            scheduled_future_speech_authority,
        )
    if (
        create_result["status"] == "created"
        and scheduled_future_speech_authority is not None
    ):
        accepted_task["scheduled_future_speech_authority"] = dict(
            scheduled_future_speech_authority
        )
    active_state = _task_text(accepted_task, "state")
    if create_result["status"] == "already_active" and active_state not in {
        "enqueueing",
        "pending",
    }:
        return _accepted_task_queue_result(
            accepted_task,
            accepted_task_state="already_active",
            status="pending",
            job_id=_task_text(accepted_task, "executor_ref"),
            result_summary="Accepted task is already active.",
            acknowledgement_constraint="progress_report_allowed",
            wait_guidance="non_numeric_wait",
        )

    queue_request = _queue_request_from_accepted_task(
        validated,
        accepted_task,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
        source_llm_trace_id=source_llm_trace_id,
        semantic_objective=semantic_objective,
        requested_worker=requested_worker,
        worker_payload=worker_payload,
        scheduled_future_speech_authority=scheduled_future_speech_authority,
    )
    job_id = queue_request["job_id"]
    if active_state == "pending":
        existing_job_id = _task_text(accepted_task, "executor_ref")
        if not existing_job_id:
            raise DatabaseOperationError(
                "pending accepted task is missing its background job id"
            )
        queue_request["job_id"] = existing_job_id
        job_id = existing_job_id
    else:
        pending_task = await mark_accepted_task_pending(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            executor_ref=job_id,
            updated_at=storage_timestamp_utc,
        )
        if pending_task is None:
            await mark_accepted_task_enqueue_failed(
                accepted_task_id=_task_text(
                    accepted_task,
                    "accepted_task_id",
                ),
                failure_summary="Accepted task could not become pending.",
                updated_at=storage_timestamp_utc,
            )
            raise DatabaseOperationError(
                "accepted task pending transition failed before job insert"
            )
    enqueue = enqueue_background_work_func or enqueue_background_work_request
    try:
        queue_result = await enqueue(queue_request)
    except (DatabaseOperationError, ValueError) as exc:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            failure_summary="Background task enqueue failed.",
            updated_at=storage_timestamp_utc,
        )
        raise DatabaseOperationError("accepted task enqueue failed") from exc
    if queue_result["job_id"] != job_id:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=_task_text(accepted_task, "accepted_task_id"),
            failure_summary="Background task returned an invalid job id.",
            updated_at=storage_timestamp_utc,
        )
        raise DatabaseOperationError(
            "accepted task enqueue returned an unexpected job id"
        )
    return _accepted_task_queue_result(
        accepted_task,
        accepted_task_state="scheduled",
        status="pending",
        job_id=queue_result["job_id"],
        result_summary="Accepted task scheduled.",
        acknowledgement_constraint="promise_allowed",
        wait_guidance="non_numeric_wait",
    )


def _reject_active_duplicate_authority_mismatch(
    stored_task: Mapping[str, Any],
    incoming_authority: Mapping[str, Any],
) -> None:
    """Reject an active duplicate whose stored authority differs.

    The stored authority is the durable commitment already accepted by the
    character. A changed trigger or objective must produce a new authority and
    a new accepted work item through the normal deterministic path instead of
    silently re-enqueueing under the existing task identity.
    """

    stored_authority = stored_task.get("scheduled_future_speech_authority")
    if stored_authority != incoming_authority:
        raise ActionValidationError(
            "active accepted task has a different scheduled authority"
        )


def _accepted_task_create_request(
    validated: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    task_kind: str,
    semantic_objective: str,
    accepted_task_summary: str,
    scheduled_future_speech_authority: dict[str, Any] | None = None,
) -> AcceptedTaskCreateRequest:
    """Build one trusted v2 accepted-task request from the action scope."""

    if task_kind != "future_speak":
        raise ActionValidationError("accepted task kind is unsupported")
    params = validated["params"]
    scope = validated["target"]["scope"]
    request: AcceptedTaskCreateRequest = {
        "task_kind": task_kind,
        "semantic_objective": semantic_objective,
        "accepted_task_summary": accepted_task_summary,
        "goal_continuation_ref": validated["goal_continuation_ref"],
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "source_trigger_source": _required_scope_text(
            scope,
            "source_trigger_source",
        ),
        "source_platform": _required_scope_text(scope, "source_platform"),
        "source_channel_id": _required_scope_text(scope, "source_channel_id"),
        "source_channel_type": _required_scope_text(scope, "source_channel_type"),
        "source_message_id": _required_scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _required_scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_scope_text(
            scope,
            "source_character_name",
        ),
        "requester_global_user_id": _required_scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _required_scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _required_scope_text(
            scope,
            "requester_display_name",
        ),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    if scheduled_future_speech_authority is not None:
        request["scheduled_future_speech_authority"] = dict(
            scheduled_future_speech_authority
        )
    return request


def _queue_request_from_accepted_task(
    validated: Mapping[str, Any],
    accepted_task: Mapping[str, Any],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
    source_llm_trace_id: str,
    semantic_objective: str,
    requested_worker: str,
    worker_payload: Mapping[str, object],
    scheduled_future_speech_authority: dict[str, Any] | None = None,
) -> BackgroundWorkQueueRequest:
    """Build one internal v2 job request from accepted-task state."""

    params = validated["params"]
    scope = validated["target"]["scope"]
    accepted_task_id = _task_text(accepted_task, "accepted_task_id")
    request: BackgroundWorkQueueRequest = {
        "job_id": f"job-{accepted_task_id.removeprefix('task-')}",
        "source_action_attempt_id": action_attempt_id,
        "source_llm_trace_id": source_llm_trace_id.strip(),
        "idempotency_key": f"background_work:{accepted_task_id}",
        "accepted_task_id": accepted_task_id,
        "task_identity_key": _task_text(accepted_task, "task_identity_key"),
        "semantic_objective": semantic_objective,
        "goal_continuation_ref": validated["goal_continuation_ref"],
        "requested_worker": requested_worker,
        "worker_payload": dict(worker_payload),
        "source_platform": _required_scope_text(scope, "source_platform"),
        "source_channel_id": _required_scope_text(scope, "source_channel_id"),
        "source_channel_type": _required_scope_text(scope, "source_channel_type"),
        "source_message_id": _required_scope_text(scope, "source_message_id"),
        "source_platform_bot_id": _required_scope_text(
            scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_scope_text(
            scope,
            "source_character_name",
        ),
        "requester_global_user_id": _required_scope_text(
            scope,
            "requester_global_user_id",
        ),
        "requester_platform_user_id": _required_scope_text(
            scope,
            "requester_platform_user_id",
        ),
        "requester_display_name": _required_scope_text(
            scope,
            "requester_display_name",
        ),
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": int(params["max_output_chars"]),
        "storage_timestamp_utc": storage_timestamp_utc,
    }
    if scheduled_future_speech_authority is not None:
        request["scheduled_future_speech_authority"] = dict(
            scheduled_future_speech_authority
        )
    return request


def _accepted_task_queue_result(
    task: Mapping[str, Any],
    *,
    accepted_task_state: str,
    status: str,
    job_id: str,
    result_summary: str,
    acknowledgement_constraint: str,
    wait_guidance: str,
) -> BackgroundWorkQueueResult:
    """Project durable state into a prompt-safe queue response."""

    result: BackgroundWorkQueueResult = {
        "status": status,
        "job_id": job_id,
        "job_ref": f"background_work_job:{job_id}" if job_id else "",
        "accepted_task_id": _task_text(task, "accepted_task_id"),
        "task_identity_key": _task_text(task, "task_identity_key"),
        "accepted_task_summary": _task_text(task, "accepted_task_summary"),
        "acknowledgement_constraint": acknowledgement_constraint,
        "wait_guidance": wait_guidance,
        "result_summary": result_summary,
    }
    result["accepted_task_state"] = accepted_task_state
    return result


def _required_param_text(params: Mapping[str, object], field_name: str) -> str:
    """Return one required semantic action parameter."""

    return _required_mapping_text(params, field_name, "params")


def _required_scope_text(scope: Mapping[str, object], field_name: str) -> str:
    """Return one required trusted delivery-scope field."""

    return _required_mapping_text(scope, field_name, "scope")


def _required_mapping_text(
    value: Mapping[str, object],
    field_name: str,
    label: str,
) -> str:
    """Require one non-empty text value from a typed mapping."""

    field_value = value.get(field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ActionValidationError(f"{label}.{field_name}: expected non-empty string")
    return field_value.strip()


def _task_text(task: Mapping[str, Any], field_name: str) -> str:
    """Read one prompt-safe text field from a durable accepted-task row."""

    value = task.get(field_name)
    if not isinstance(value, str):
        return ""
    return value.strip()
