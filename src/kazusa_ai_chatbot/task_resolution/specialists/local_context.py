"""Local/private evidence specialist for bounded task resolution."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.local_context_resolver import (
    DEFAULT_OPTION_LIMITS,
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    LocalContextValidationError,
    project_local_context_packet,
    resolve_local_context,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionExecutionContextV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
)
from kazusa_ai_chatbot.task_resolution.specialists import (
    _bounded_text,
    _prompt_message_text,
    _require_handler_coding_objective_mode,
    _specialist_evidence,
    _specialist_result,
    _validated_handler_inputs,
)


SPECIALIST = "local_context"
_LOCAL_EVIDENCE_FIELDS = (
    "memory_evidence",
    "recall_evidence",
    "conversation_evidence",
    "external_evidence",
)


async def resolve_with_local_context(
    request: dict[str, object],
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskSpecialistResultV1:
    """Resolve one selected subgoal through the public local-context IO.

    The handler maps only canonical task state into the existing
    request/context/options triplet.  It does not choose another specialist,
    persist task state, or use adapter objects.
    """

    task_request, context = _validated_handler_inputs(request, execution_context)
    _require_handler_coding_objective_mode(
        task_request,
        specialist=SPECIALIST,
    )
    resolver_request = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        "objective": task_request["objective"],
        "source": "l2d",
        "reason": "Task resolution requested local context evidence.",
        "priority": "normal",
    }
    resolver_context = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        "character_name": context["character_name"],
        "platform": context["platform"],
        "platform_channel_id": context["channel_id"],
        "global_user_id": context["requester_global_user_id"],
        "user_name": context["requester_platform_user_id"],
        "scene_context": context["scene_context"],
        "local_time_context": dict(context["local_time_context"]),
        "prompt_message_context": dict(context["prompt_message_context"]),
        "chat_history_recent": list(context["chat_history_recent"]),
        "chat_history_wide": list(context["chat_history_wide"]),
        "conversation_progress": dict(context["conversation_progress"]),
        "original_user_request": _local_original_request(task_request, context),
        "current_timestamp_utc": context["current_timestamp_utc"],
        "current_platform_message_id": context["source_message_id"],
        "active_turn_platform_message_ids": list(
            context["active_turn_platform_message_ids"]
        ),
        "active_turn_conversation_row_ids": list(
            context["active_turn_conversation_row_ids"]
        ),
        "session_media_refs": list(context["session_media_refs"]),
    }
    resolver_options = {
        "schema_version": LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        **DEFAULT_OPTION_LIMITS,
    }
    try:
        packet = await resolve_local_context(
            resolver_request,
            resolver_context,
            resolver_options,
        )
        projection = project_local_context_packet(packet)
    except LocalContextValidationError:
        return _specialist_result(
            specialist=SPECIALIST,
            status="failed",
            remaining_needs=[task_request["objective"]],
            reason="The local-context resolver returned invalid public output.",
        )

    summary, provenance_refs = _local_evidence_projection(projection, task_request)
    limitations = _packet_text_items(packet, "knowledge_still_lacking")
    if not summary:
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(task_request),
            reason="Local context did not contain evidence for this subgoal.",
        )
    evidence = _specialist_evidence(
        request=task_request,
        specialist=SPECIALIST,
        summary=summary,
        provenance_refs=provenance_refs,
        limitations=limitations,
    )
    status = "partial" if limitations else "resolved"
    return _specialist_result(
        specialist=SPECIALIST,
        status=status,
        evidence=[evidence],
        completed_subgoals=[task_request["objective"]],
        remaining_needs=limitations,
        reason="Local context returned provenance-bearing evidence.",
    )


def _local_original_request(
    request: TaskSpecialistRequestV1,
    context: TaskResolutionExecutionContextV1,
) -> str:
    """Prefer supplied message context while retaining the selected subgoal."""

    message_text = _prompt_message_text(context)
    if message_text:
        return message_text
    return request["objective"]


def _local_evidence_projection(
    projection: Mapping[str, object],
    request: TaskSpecialistRequestV1,
) -> tuple[str, list[str]]:
    """Project local resolver output into one bounded evidence record."""

    summaries: list[str] = []
    provenance_refs: list[str] = []
    for field_name in _LOCAL_EVIDENCE_FIELDS:
        raw_rows = projection.get(field_name)
        if not isinstance(raw_rows, list):
            continue
        for index, raw_row in enumerate(raw_rows, start=1):
            summary = _evidence_row_summary(raw_row)
            if not summary:
                continue
            summaries.append(summary)
            provenance_refs.append(
                f"local_context:{request['task_node_id']}:{field_name}:{index}"
            )
            if len(summaries) == 3:
                break
        if len(summaries) == 3:
            break
    answer = projection.get("answer")
    if isinstance(answer, str) and answer.strip():
        summaries.insert(0, _bounded_text(answer))
    if not summaries:
        return "", []
    if not provenance_refs:
        provenance_refs.append(
            f"local_context:{request['task_node_id']}:resolver_packet"
        )
    return "; ".join(summaries)[:1200], provenance_refs


def _evidence_row_summary(value: object) -> str:
    """Return one safe evidence summary from a resolver projection row."""

    if isinstance(value, str) and value.strip():
        return _bounded_text(value)
    if not isinstance(value, Mapping):
        return ""
    for field_name in ("summary", "content", "text", "description"):
        raw_summary = value.get(field_name)
        if isinstance(raw_summary, str) and raw_summary.strip():
            return _bounded_text(raw_summary)
    return ""


def _packet_text_items(packet: object, field_name: str) -> list[str]:
    """Project bounded textual limitations from one public resolver packet."""

    if not isinstance(packet, Mapping):
        return []
    raw_items = packet.get(field_name)
    if not isinstance(raw_items, list):
        return []
    items: list[str] = []
    for raw_item in raw_items[:8]:
        if isinstance(raw_item, str) and raw_item.strip():
            items.append(_bounded_text(raw_item))
    return items


def _remaining_needs(request: TaskSpecialistRequestV1) -> list[str]:
    """Retain the canonical need when a local specialist refuses the task."""

    if request["remaining_needs"]:
        return list(request["remaining_needs"])
    return [request["objective"]]
