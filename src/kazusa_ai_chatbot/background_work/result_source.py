"""Build source-bound cognition inputs from completed background-work jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from kazusa_ai_chatbot.background_work.models import BackgroundWorkJobDoc
from kazusa_ai_chatbot.coding_agent.coding_run.ledger import (
    sanitize_coding_run_context,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    CognitiveEpisodeV1,
    EvidenceRefV1,
    GoalContinuationRefV1,
    TargetScopeV1,
    ToolResultReadyV1,
    build_tool_result_episode,
    validate_goal_continuation_ref,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TASK_RESOLUTION_RESULT_EVIDENCE_STATES,
    TASK_RESOLUTION_STATUSES,
    TaskResolutionContractError,
    TaskResolutionResultV1,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc

MAX_TOOL_RESULT_SEMANTIC_SUMMARY_CHARS = 2000
MAX_TOOL_RESULT_SOURCE_ITEMS = 8
MAX_TOOL_RESULT_SOURCE_ITEM_CHARS = 1200


class ToolResultCognitionSourceV1(TypedDict):
    """Typed tool outcome admitted to a later cognition episode.

    Every field is projected only from a validated stored
    ``TaskResolutionResultV1``; free-form job summaries never replace this
    typed contract.
    """

    source_kind: str
    source_id: str
    occurred_at: str
    semantic_summary: str
    semantic_objective: str
    task_status: str
    evidence_state: str
    evidence_excerpts: list[str]
    evidence_handles: list[str]
    remaining_needs: list[str]
    goal_continuation_ref: GoalContinuationRefV1


def build_result_ready_episode_from_job(
    job: BackgroundWorkJobDoc,
) -> CognitiveEpisodeV1:
    """Project one completed accepted-task job into a tool-result episode.

    The validated stored ``TaskResolutionResultV1`` is the authoritative
    result contract: objective, status, evidence state, evidence excerpts and
    handles, remaining needs, and the exact goal-continuation reference are
    projected from it. Job-level summary or failure text is never used as a
    result authority.
    """

    completed_at = job.get("completed_at") or job.get("updated_at") or job["created_at"]
    turn_clock = build_turn_clock_from_storage_utc(completed_at)
    accepted_task_id = job.get("accepted_task_id", "").strip()
    if not accepted_task_id:
        raise ValueError("accepted_task_id is required for result delivery")
    task_result = _stored_task_result(job)
    continuation_ref = task_result["goal_continuation_ref"]

    worker_metadata = job.get("worker_metadata")
    coding_run_context = None
    if isinstance(worker_metadata, Mapping):
        metadata_context = worker_metadata.get("coding_run_context")
        coding_run_context = sanitize_coding_run_context(metadata_context)
    outcome_summary = _result_semantic_summary(task_result)
    evidence_refs = _tool_result_evidence_refs(
        task_result=task_result,
        completed_at=completed_at,
    )
    target_scope: TargetScopeV1 = {
        "platform": job.get("source_platform", ""),
        "platform_channel_id": job.get("source_channel_id", ""),
        "channel_type": job.get("source_channel_type", ""),
        "current_platform_user_id": job.get(
            "requester_platform_user_id",
            "",
        ),
        "current_global_user_id": job.get("requester_global_user_id", ""),
        "current_display_name": job.get("requester_display_name", ""),
        "target_addressed_user_ids": [
            job.get("requester_global_user_id", "")
        ] if job.get("requester_global_user_id") else [],
        "target_broadcast": False,
    }
    result: ToolResultReadyV1 = {
        "schema_version": "tool_result_ready.v1",
        "task_id": accepted_task_id,
        "task_kind": "accepted_task",
        "semantic_summary": outcome_summary,
        "completed_at": completed_at,
        "target_scope": target_scope,
        "evidence_refs": evidence_refs,
        "result_ref": accepted_task_id,
        "source_platform_bot_id": job.get("source_platform_bot_id", ""),
        "source_character_name": job.get("source_character_name", ""),
        "source_message_id": job.get("source_message_id", ""),
        "goal_continuation_ref": continuation_ref,
    }
    if coding_run_context is not None:
        result["coding_run_context"] = dict(coding_run_context)
    episode = build_tool_result_episode(
        result=result,
        evidence_refs=evidence_refs,
        local_time_context=turn_clock["local_time_context"],
        created_at=completed_at,
    )
    episode["origin_metadata"]["source_llm_trace_id"] = str(
        job.get("source_llm_trace_id") or ""
    ).strip()
    episode["origin_metadata"]["source_background_work_job_id"] = str(
        job.get("job_id") or ""
    ).strip()
    cognition_source = _build_tool_result_cognition_source(
        task_result=task_result,
        semantic_summary=outcome_summary,
        accepted_task_id=accepted_task_id,
        completed_at=completed_at,
    )
    episode["percepts"][0]["content"]["cognition_source"] = cognition_source
    return episode


def validate_tool_result_cognition_source(
    value: object,
) -> ToolResultCognitionSourceV1:
    """Validate one typed tool-result source attached to an episode percept.

    The projection is result-owned: factual states carry only validated
    evidence excerpts, and every non-factual state carries no evidence at all.
    """

    if not isinstance(value, Mapping):
        raise ValueError("tool_result cognition source must be an object")
    if set(value) != set(ToolResultCognitionSourceV1.__annotations__):
        raise ValueError("tool_result cognition source fields are not exact")
    if value["source_kind"] != "tool_result":
        raise ValueError("tool_result cognition source kind is invalid")
    source_id = _source_text(value["source_id"], "source_id")
    occurred_at = _source_text(value["occurred_at"], "occurred_at")
    semantic_summary = _source_text(
        value["semantic_summary"],
        "semantic_summary",
        maximum=MAX_TOOL_RESULT_SEMANTIC_SUMMARY_CHARS,
    )
    semantic_objective = _source_text(
        value["semantic_objective"],
        "semantic_objective",
    )
    task_status = _source_enum(
        value["task_status"],
        "task_status",
        TASK_RESOLUTION_STATUSES,
    )
    evidence_state = _source_enum(
        value["evidence_state"],
        "evidence_state",
        TASK_RESOLUTION_RESULT_EVIDENCE_STATES,
    )
    evidence_excerpts = _source_text_list(
        value["evidence_excerpts"],
        "evidence_excerpts",
    )
    evidence_handles = _source_text_list(
        value["evidence_handles"],
        "evidence_handles",
    )
    remaining_needs = _source_text_list(
        value["remaining_needs"],
        "remaining_needs",
    )
    try:
        continuation_ref = validate_goal_continuation_ref(
            value["goal_continuation_ref"]
        )
    except CognitiveEpisodeValidationError as exc:
        raise ValueError(
            "tool_result cognition source goal_continuation_ref is invalid: "
            f"{exc}"
        ) from exc
    _validate_result_state_projection(
        task_status=task_status,
        evidence_state=evidence_state,
        evidence_excerpts=evidence_excerpts,
        evidence_handles=evidence_handles,
    )
    normalized: ToolResultCognitionSourceV1 = {
        "source_kind": "tool_result",
        "source_id": source_id,
        "occurred_at": occurred_at,
        "semantic_summary": semantic_summary,
        "semantic_objective": semantic_objective,
        "task_status": task_status,
        "evidence_state": evidence_state,
        "evidence_excerpts": evidence_excerpts,
        "evidence_handles": evidence_handles,
        "remaining_needs": remaining_needs,
        "goal_continuation_ref": continuation_ref,
    }
    return normalized


def _stored_task_result(
    job: Mapping[str, object],
) -> TaskResolutionResultV1:
    """Return the required validated typed task result persisted by the worker."""

    value = job.get("task_resolution_result")
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "task_resolution_result is required for result delivery"
        )
    try:
        result = validate_task_resolution_result(value)
    except TaskResolutionContractError as exc:
        raise ValueError(
            f"task_resolution_result: invalid typed result: {exc}"
        ) from exc
    return result


def _result_semantic_summary(result: TaskResolutionResultV1) -> str:
    """Project validated task evidence into the prompt-safe result summary.

    Factual states expose only validated evidence excerpts, with explicit
    remaining limitations retained for partial results. Every other state
    exposes only objective-scoped status or clarification text.
    """

    if result["status"] in {"resolved", "partial"}:
        summary_rows = list(result["evidence_excerpts"])
        if result["status"] == "partial" and result["remaining_needs"]:
            remaining_text = "; ".join(result["remaining_needs"])
            summary_rows.append(f"Remaining limitations: {remaining_text}")
    else:
        summary_rows = [result["prompt_safe_summary"]]
        if result["remaining_needs"]:
            remaining_text = "; ".join(result["remaining_needs"])
            summary_rows.append(f"Remaining needs: {remaining_text}")
    summary = "; ".join(summary_rows)
    bounded_summary = summary[:MAX_TOOL_RESULT_SEMANTIC_SUMMARY_CHARS]
    return bounded_summary


def _build_tool_result_cognition_source(
    *,
    task_result: TaskResolutionResultV1,
    semantic_summary: str,
    accepted_task_id: str,
    completed_at: str,
) -> ToolResultCognitionSourceV1:
    """Project the validated stored result into one typed cognition source."""

    cognition_source: ToolResultCognitionSourceV1 = {
        "source_kind": "tool_result",
        "source_id": accepted_task_id,
        "occurred_at": completed_at,
        "semantic_summary": semantic_summary,
        "semantic_objective": task_result["semantic_objective"],
        "task_status": task_result["status"],
        "evidence_state": task_result["evidence_state"],
        "evidence_excerpts": list(task_result["evidence_excerpts"]),
        "evidence_handles": list(task_result["evidence_handles"]),
        "remaining_needs": list(task_result["remaining_needs"]),
        "goal_continuation_ref": task_result["goal_continuation_ref"],
    }
    try:
        validated_source = validate_tool_result_cognition_source(
            cognition_source
        )
    except ValueError as exc:
        raise ValueError(
            f"tool_result cognition source is invalid: {exc}"
        ) from exc
    return validated_source


def _tool_result_evidence_refs(
    *,
    task_result: TaskResolutionResultV1,
    completed_at: str,
) -> list[EvidenceRefV1]:
    """Project typed task evidence handles into prompt-safe evidence refs."""

    refs: list[EvidenceRefV1] = []
    for handle, excerpt in zip(
        task_result["evidence_handles"],
        task_result["evidence_excerpts"],
        strict=True,
    ):
        refs.append({
            "schema_version": "evidence_ref.v1",
            "evidence_kind": "tool_result",
            "evidence_id": handle,
            "owner": "background_work",
            "excerpt": excerpt,
            "observed_at": completed_at,
        })
    return refs


def _validate_result_state_projection(
    *,
    task_status: str,
    evidence_state: str,
    evidence_excerpts: list[str],
    evidence_handles: list[str],
) -> None:
    """Enforce the deterministic factual-evidence policy on a typed source."""

    if task_status in {"resolved", "partial"}:
        if evidence_state not in {"complete", "partial"}:
            raise ValueError(
                "tool_result cognition source factual task state requires "
                "complete or partial evidence"
            )
        if not evidence_excerpts or len(evidence_handles) != len(
            evidence_excerpts
        ):
            raise ValueError(
                "tool_result cognition source factual state requires "
                "parallel validated evidence excerpts and handles"
            )
        return
    if evidence_state not in {"pending", "missing", "blocked"}:
        raise ValueError(
            "tool_result cognition source non-factual task state has an "
            "invalid evidence state"
        )
    if evidence_excerpts or evidence_handles:
        raise ValueError(
            "tool_result cognition source non-factual task state cannot "
            "expose evidence"
        )


def _source_text(
    value: object,
    field_name: str,
    *,
    maximum: int = MAX_TOOL_RESULT_SOURCE_ITEM_CHARS,
) -> str:
    """Return one bounded non-empty semantic string field."""

    if not isinstance(value, str):
        raise ValueError(
            f"tool_result cognition source {field_name} must be a string"
        )
    text = value.strip()
    if not text or len(text) > maximum:
        raise ValueError(
            f"tool_result cognition source {field_name} is invalid"
        )
    return text


def _source_enum(
    value: object,
    field_name: str,
    allowed_values: frozenset[str],
) -> str:
    """Return one closed enum field."""

    if not isinstance(value, str) or value not in allowed_values:
        raise ValueError(
            f"tool_result cognition source {field_name} is invalid"
        )
    return value


def _source_text_list(
    value: object,
    field_name: str,
) -> list[str]:
    """Return one bounded list of non-empty unique semantic strings."""

    if not isinstance(value, list) or len(value) > MAX_TOOL_RESULT_SOURCE_ITEMS:
        raise ValueError(
            f"tool_result cognition source {field_name} must be a bounded list"
        )
    rows = [
        _source_text(item, field_name)
        for item in value
    ]
    if len(rows) != len(set(rows)):
        raise ValueError(
            f"tool_result cognition source {field_name} items are duplicated"
        )
    return rows
