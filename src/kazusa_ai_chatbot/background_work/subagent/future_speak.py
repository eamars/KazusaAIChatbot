"""Future-speak worker that schedules a later self-cognition message."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.handlers.future_cognition import (
    execute_future_cognition_action,
)
from kazusa_ai_chatbot.action_spec.models import ActionValidationError
from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_OUTPUT_CHAR_LIMIT
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

WORKER = "future_speak"
DESCRIPTION = (
    "Schedules accepted future reminders and delayed follow-up messages as "
    "later self-cognition slots."
)


async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> BackgroundWorkResult:
    """Schedule one future-cognition slot from a bound worker payload."""

    try:
        worker_payload = _worker_payload(decision)
        trigger_at = _required_payload_text(worker_payload, "trigger_at")
        continuation_objective = _required_payload_text(
            worker_payload,
            "continuation_objective",
        )
        action_attempt_id = _required_payload_text(
            worker_payload,
            "source_action_attempt_id",
        )
        source_scope = _source_scope(worker_payload)
        storage_timestamp_utc = _storage_timestamp_utc(worker_payload)
        action_spec = _build_future_cognition_action_spec(
            trigger_at=trigger_at,
            continuation_objective=continuation_objective,
            source_scope=source_scope,
            reason=decision["reason"],
        )
        future_result = await execute_future_cognition_action(
            action_spec,
            storage_timestamp_utc=storage_timestamp_utc,
            action_attempt_id=action_attempt_id,
        )
    except ActionValidationError as exc:
        result = _terminal_result(
            status="rejected",
            failure_summary=str(exc),
            result_summary="Future speak request was rejected.",
            max_output_chars=max_output_chars,
        )
        return result
    except DatabaseOperationError as exc:
        result = _terminal_result(
            status="failed",
            failure_summary=str(exc),
            result_summary="Future speak scheduling failed.",
            max_output_chars=max_output_chars,
        )
        return result

    scheduled_for = str(future_result["trigger_at"])
    artifact_text = _bounded_text(
        f"Future speak scheduled for {scheduled_for}.",
        limit=max_output_chars,
    )
    result: BackgroundWorkResult = {
        "status": "succeeded",
        "worker": WORKER,
        "artifact_text": artifact_text,
        "failure_summary": "",
        "result_summary": "Future speak scheduled.",
        "worker_metadata": {
            "calendar_schedule_id": future_result["calendar_schedule_id"],
            "calendar_run_id": future_result["calendar_run_id"],
            "trigger_at": future_result["trigger_at"],
            "skip_result_delivery": True,
        },
    }
    return result


def _build_future_cognition_action_spec(
    *,
    trigger_at: str,
    continuation_objective: str,
    source_scope: dict[str, object],
    reason: str,
) -> dict[str, object]:
    """Build the internal future-cognition action for the scheduler."""

    target_scope = dict(source_scope)
    target_scope["episode_type"] = "self_cognition"
    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": "trigger_future_cognition",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "system_event",
                "ref_id": "future_speak_background_work",
                "owner": "background_work",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": target_scope,
        },
        "params": {
            "episode_type": "self_cognition",
            "trigger_at": trigger_at,
            "continuation_objective": continuation_objective,
        },
        "urgency": "scheduled",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "scheduled_followup",
            "episode_type": "self_cognition",
            "max_depth": 1,
            "include_result_as": "scheduled_event",
        },
        "reason": reason,
    }
    return action_spec


def _terminal_result(
    *,
    status: str,
    failure_summary: str,
    result_summary: str,
    max_output_chars: int,
) -> BackgroundWorkResult:
    """Build one terminal worker result."""

    result: BackgroundWorkResult = {
        "status": status,
        "worker": WORKER,
        "artifact_text": "",
        "failure_summary": _bounded_text(failure_summary, limit=max_output_chars),
        "result_summary": result_summary,
        "worker_metadata": {"skip_result_delivery": False},
    }
    return result


def _worker_payload(
    decision: BackgroundWorkWorkerDecision,
) -> dict[str, object]:
    """Return the required deterministic worker payload."""

    worker_payload = decision.get("worker_payload")
    if not isinstance(worker_payload, Mapping):
        raise ActionValidationError("worker_payload: expected object")
    return_value = dict(worker_payload)
    return return_value


def _source_scope(payload: Mapping[str, object]) -> dict[str, object]:
    """Return delivery source scope required by scheduled self-cognition."""

    raw_source_scope = payload.get("source_scope")
    if not isinstance(raw_source_scope, Mapping):
        raise ActionValidationError("worker_payload.source_scope: expected object")
    source_scope = {
        "source_platform": _required_scope_text(
            raw_source_scope,
            "source_platform",
        ),
        "source_channel_id": _required_scope_text(
            raw_source_scope,
            "source_channel_id",
        ),
        "source_channel_type": _required_scope_text(
            raw_source_scope,
            "source_channel_type",
        ),
        "source_user_id": _required_scope_text(
            raw_source_scope,
            "source_user_id",
        ),
        "source_platform_bot_id": _required_scope_text(
            raw_source_scope,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_scope_text(
            raw_source_scope,
            "source_character_name",
        ),
        "source_message_id": _scope_text(raw_source_scope, "source_message_id"),
    }
    return source_scope


def _storage_timestamp_utc(payload: Mapping[str, object]) -> str:
    """Return the scheduling storage timestamp."""

    storage_timestamp_utc = _payload_text(payload, "storage_timestamp_utc")
    if not storage_timestamp_utc:
        storage_timestamp_utc = storage_utc_now_iso()
    return storage_timestamp_utc


def _required_payload_text(
    payload: Mapping[str, object],
    field_name: str,
) -> str:
    """Return one required worker-payload text field."""

    value = _payload_text(payload, field_name)
    if not value:
        raise ActionValidationError(
            f"worker_payload.{field_name}: expected non-empty string"
        )
    return value


def _payload_text(payload: Mapping[str, object], field_name: str) -> str:
    """Return one optional worker-payload text field."""

    raw_value = payload.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _required_scope_text(
    scope: Mapping[str, object],
    field_name: str,
) -> str:
    """Return one required source-scope text field."""

    value = _scope_text(scope, field_name)
    if not value:
        raise ActionValidationError(
            f"worker_payload.source_scope.{field_name}: "
            "expected non-empty string"
        )
    return value


def _scope_text(scope: Mapping[str, object], field_name: str) -> str:
    """Return one optional source-scope text field."""

    raw_value = scope.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    """Return stripped text capped to a local bound."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()[:limit]
    return return_value
