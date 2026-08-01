"""Deterministic background scheduling for accepted future-speak requests."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TypedDict


class FutureSpeakExecutionResult(TypedDict):
    """Prompt-safe completion facts from deterministic future scheduling."""

    artifact_text: str
    result_summary: str


async def execute_future_speak_job(
    job: Mapping[str, object],
) -> FutureSpeakExecutionResult:
    """Schedule one reviewed future-cognition request from its v2 job row.

    Args:
        job: Claimed v2 job with an exact future-speak worker payload.

    Returns:
        Prompt-safe scheduling confirmation for the job audit record.
    """

    payload = _future_speak_payload(job)
    trigger_at = _required_text(payload, "trigger_at")
    continuation_objective = _required_text(
        payload,
        "continuation_objective",
    )
    action_spec = _future_cognition_action_spec(
        job,
        trigger_at=trigger_at,
        continuation_objective=continuation_objective,
    )
    future_result = await _execute_future_cognition_action(
        action_spec,
        storage_timestamp_utc=_job_timestamp(job),
        action_attempt_id=_required_job_text(job, "source_action_attempt_id"),
    )
    scheduled_for = str(future_result["trigger_at"])
    result: FutureSpeakExecutionResult = {
        "artifact_text": f"Future speak scheduled for {scheduled_for}.",
        "result_summary": "Future speak scheduled.",
    }
    return result


def _future_speak_payload(job: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact deterministic future-speak payload union branch."""

    value = job.get("worker_payload")
    if not isinstance(value, Mapping):
        raise ValueError("worker_payload: expected object")
    payload = dict(value)
    if set(payload) != {"trigger_at", "continuation_objective"}:
        raise ValueError("worker_payload: fields are invalid")
    return payload


def _future_cognition_action_spec(
    job: Mapping[str, object],
    *,
    trigger_at: str,
    continuation_objective: str,
) -> dict[str, object]:
    """Build the internal deterministic schedule action from trusted job scope."""

    source_scope = {
        "source_platform": _required_job_text(job, "source_platform"),
        "source_channel_id": _required_job_text(job, "source_channel_id"),
        "source_channel_type": _required_job_text(job, "source_channel_type"),
        "source_user_id": _required_job_text(
            job,
            "requester_global_user_id",
        ),
        "source_platform_bot_id": _required_job_text(
            job,
            "source_platform_bot_id",
        ),
        "source_character_name": _required_job_text(
            job,
            "source_character_name",
        ),
        "source_message_id": _required_job_text(job, "source_message_id"),
        "episode_type": "self_cognition",
    }
    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": "trigger_future_cognition",
        "cognition_mode": "deliberative",
        "source_refs": [{
            "schema_version": "action_source_ref.v1",
            "ref_kind": "system_event",
            "ref_id": "future_speak_background_work",
            "owner": "background_work",
            "relationship": "basis",
            "evidence_refs": [],
        }],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": source_scope,
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
        "reason": "A reviewed future-speak task requested deterministic scheduling.",
    }
    return action_spec


def _job_timestamp(job: Mapping[str, object]) -> str:
    """Use the stored job timestamp for deterministic schedule provenance."""

    updated_at = job.get("updated_at")
    if isinstance(updated_at, str) and updated_at.strip():
        return updated_at.strip()
    raise ValueError("background job updated_at is required")


def _required_text(payload: Mapping[str, object], field_name: str) -> str:
    """Require one non-empty deterministic worker-payload field."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"worker_payload.{field_name}: expected non-empty text"
        )
    text = value.strip()
    return text


def _required_job_text(job: Mapping[str, object], field_name: str) -> str:
    """Require one trusted source field from the v2 background job."""

    value = job.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"background job {field_name}: expected non-empty text"
        )
    text = value.strip()
    return text


async def _execute_future_cognition_action(
    action_spec: dict[str, object],
    *,
    storage_timestamp_utc: str,
    action_attempt_id: str,
) -> dict[str, object]:
    """Run the action-owned scheduler after a future job is claimed.

    The late import keeps the worker's module boundary independent from the
    action-spec package during startup while retaining one executable scheduler
    contract for the claimed job.
    """

    module = import_module(
        "kazusa_ai_chatbot.action_spec.handlers.future_cognition",
    )
    execute_action = getattr(module, "execute_future_cognition_action")
    result = await execute_action(
        action_spec,
        storage_timestamp_utc=storage_timestamp_utc,
        action_attempt_id=action_attempt_id,
    )
    return result
