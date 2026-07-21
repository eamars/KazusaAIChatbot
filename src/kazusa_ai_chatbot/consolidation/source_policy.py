"""Structural source views and lane source-policy validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.utils import text_or_empty


ASSISTANT_ACCEPTANCE_SOURCE_KIND = "assistant_" + "final_dialog"
USER_MESSAGE_SOURCE_KIND = "user_message"
INTERNAL_THOUGHT_SOURCE_KIND = "internal_thought"
EPISODE_TRACE_SOURCE_KIND = "episode_trace"
REFLECTION_RUN_SOURCE_KIND = "reflection_run"
REFLECTION_USER_STYLE_SOURCE_KEY = "reflection_user_style_signal"
RAG_MEMORY_EVIDENCE_SOURCE_KIND = "rag_memory_evidence"
RAG_CONVERSATION_EVIDENCE_SOURCE_KIND = "rag_conversation_evidence"
RAG_EXTERNAL_EVIDENCE_SOURCE_KIND = "rag_external_evidence"
RAG_RECALL_EVIDENCE_SOURCE_KIND = "rag_recall_evidence"

_RAG_SOURCE_FIELDS = (
    ("memory_evidence", RAG_MEMORY_EVIDENCE_SOURCE_KIND),
    ("conversation_evidence", RAG_CONVERSATION_EVIDENCE_SOURCE_KIND),
    ("external_evidence", RAG_EXTERNAL_EVIDENCE_SOURCE_KIND),
    ("recall_evidence", RAG_RECALL_EVIDENCE_SOURCE_KIND),
)
_USER_MEMORY_SOURCE_KINDS = frozenset(
    (
        USER_MESSAGE_SOURCE_KIND,
        RAG_MEMORY_EVIDENCE_SOURCE_KIND,
        RAG_CONVERSATION_EVIDENCE_SOURCE_KIND,
        RAG_RECALL_EVIDENCE_SOURCE_KIND,
    )
)
_STYLE_SOURCE_KINDS = frozenset(
    (
        USER_MESSAGE_SOURCE_KIND,
        ASSISTANT_ACCEPTANCE_SOURCE_KIND,
        REFLECTION_RUN_SOURCE_KIND,
    )
)


def build_consolidation_source_views(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project prompt-safe transient source views from consolidation state.

    Args:
        state: Consolidator state carrying origin metadata, current text,
            dialog output, internal thought, RAG evidence, and optional
            reflection evidence.

    Returns:
        Source-view rows with stable source keys, source classes, compact
        summaries, and source refs derived from existing identifiers.
    """

    origin = state.get("consolidation_origin")
    if not isinstance(origin, Mapping):
        origin = {}

    source_views: list[dict[str, Any]] = []
    trigger_source = text_or_empty(origin.get("trigger_source"))
    user_summary = text_or_empty(state.get("decontexualized_input"))
    if trigger_source == USER_MESSAGE_SOURCE_KIND:
        source_views.append(
            _source_view(
                source_key="current_turn_user_message",
                source_kind=USER_MESSAGE_SOURCE_KIND,
                summary=user_summary,
                source_refs=_origin_source_refs(origin, USER_MESSAGE_SOURCE_KIND),
            )
        )

    dialog_summary = _compact_text(state.get("final_dialog"))
    if dialog_summary:
        source_views.append(
            _source_view(
                source_key=ASSISTANT_ACCEPTANCE_SOURCE_KIND,
                source_kind=ASSISTANT_ACCEPTANCE_SOURCE_KIND,
                summary=dialog_summary,
                source_refs=_origin_source_refs(
                    origin,
                    ASSISTANT_ACCEPTANCE_SOURCE_KIND,
                ),
            )
        )

    thought_summary = text_or_empty(state.get("internal_monologue"))
    if thought_summary or trigger_source == INTERNAL_THOUGHT_SOURCE_KIND:
        source_views.append(
            _source_view(
                source_key=INTERNAL_THOUGHT_SOURCE_KIND,
                source_kind=INTERNAL_THOUGHT_SOURCE_KIND,
                summary=thought_summary,
                source_refs=_origin_source_refs(
                    origin,
                    INTERNAL_THOUGHT_SOURCE_KIND,
                ),
            )
        )

    episode_trace = state.get("episode_trace_projection")
    if isinstance(episode_trace, Mapping) and episode_trace:
        source_views.append(
            _source_view(
                source_key=EPISODE_TRACE_SOURCE_KIND,
                source_kind=EPISODE_TRACE_SOURCE_KIND,
                summary=_compact_text(episode_trace),
                source_refs=_origin_source_refs(origin, EPISODE_TRACE_SOURCE_KIND),
            )
        )

    rag_result = state.get("rag_result")
    if isinstance(rag_result, Mapping):
        for field_name, source_kind in _RAG_SOURCE_FIELDS:
            evidence = rag_result.get(field_name)
            if _has_evidence(evidence):
                source_views.append(
                    _source_view(
                        source_key=source_kind,
                        source_kind=source_kind,
                        summary=_compact_text(evidence),
                        source_refs=_evidence_source_refs(
                            evidence,
                            source_kind=source_kind,
                            origin=origin,
                        ),
                    )
                )

    source_views.extend(_reflection_source_views(state, origin))
    return source_views


def validate_lane_source_policy(
    lane: str,
    source_views: list[dict[str, Any]],
    privacy_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate source classes for one lane without semantic text checks."""

    source_kinds = _source_kind_set(source_views)
    if not source_kinds:
        return {"accepted": False, "reason": "missing_sources"}
    if not _all_sources_have_refs(source_views):
        return {"accepted": False, "reason": "source_refs_missing"}

    if lane == "user_memory_units":
        return _accepted_if(
            bool(source_kinds & _USER_MEMORY_SOURCE_KINDS),
            "source_class_not_allowed",
        )
    if lane == "active_commitment":
        return _accepted_if(
            _has_user_and_assistant_sources(source_kinds),
            "source_class_not_allowed",
        )
    if lane == "character_self_guidance":
        return _accepted_if(
            _has_user_and_assistant_sources(source_kinds),
            "source_class_not_allowed",
        )
    if lane == "shared_memory_promotion":
        return _accepted_if(
            REFLECTION_RUN_SOURCE_KIND in source_kinds
            and _privacy_review_passed(privacy_review),
            "source_class_not_allowed",
        )
    if lane == "interaction_style_image":
        return _accepted_if(
            bool(source_kinds & _STYLE_SOURCE_KINDS),
            "source_class_not_allowed",
        )

    return {"accepted": False, "reason": "unknown_lane"}


def source_refs_from_views(source_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect source refs from accepted source-view rows."""

    source_refs: list[dict[str, Any]] = []
    for source_view in source_views:
        raw_refs = source_view.get("source_refs")
        if not isinstance(raw_refs, list):
            continue
        for raw_ref in raw_refs:
            if isinstance(raw_ref, Mapping):
                source_refs.append(dict(raw_ref))
    return source_refs


def _source_view(
    *,
    source_key: str,
    source_kind: str,
    summary: str,
    source_refs: list[dict[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one source-view row."""

    source_view = {
        "source_key": source_key,
        "source_kind": source_kind,
        "summary": summary,
        "source_refs": source_refs,
    }
    if extra:
        source_view.update(dict(extra))
    return source_view


def _origin_source_refs(
    origin: Mapping[str, Any],
    source_kind: str,
) -> list[dict[str, Any]]:
    """Build source refs from identifier-only consolidation origin metadata."""

    if not origin:
        return_value: list[dict[str, Any]] = []
        return return_value

    source_ref: dict[str, Any] = {"source": source_kind}
    _copy_text_field(origin, source_ref, "storage_timestamp_utc", "timestamp")
    for field_name in (
        "episode_id",
        "platform",
        "platform_channel_id",
        "channel_type",
        "platform_message_id",
        "current_platform_user_id",
        "current_global_user_id",
        "current_display_name",
    ):
        _copy_text_field(origin, source_ref, field_name, field_name)

    active_message_ids = origin.get("active_turn_platform_message_ids")
    if isinstance(active_message_ids, list):
        source_ref["active_turn_platform_message_ids"] = [
            message_id
            for message_id in active_message_ids
            if isinstance(message_id, str) and message_id.strip()
        ]
    active_row_ids = origin.get("active_turn_conversation_row_ids")
    if isinstance(active_row_ids, list):
        source_ref["active_turn_conversation_row_ids"] = [
            row_id
            for row_id in active_row_ids
            if isinstance(row_id, str) and row_id.strip()
        ]
    return_value = [source_ref]
    return return_value


def _copy_text_field(
    source: Mapping[str, Any],
    target: dict[str, Any],
    source_field: str,
    target_field: str,
) -> None:
    """Copy one non-empty text field between metadata dictionaries."""

    value = text_or_empty(source.get(source_field))
    if value:
        target[target_field] = value


def _compact_text(value: object) -> str:
    """Build a bounded text summary for source-view display and routing."""

    if isinstance(value, str):
        summary = value.strip()
    elif isinstance(value, list):
        parts = [
            _compact_text(item)
            if isinstance(item, Mapping)
            else text_or_empty(item)
            for item in value
        ]
        summary = "\n".join(part for part in parts if part)
    elif isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            item_text = text_or_empty(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        summary = "\n".join(parts)
    else:
        summary = text_or_empty(value)

    if len(summary) > 1000:
        summary = summary[:1000].rstrip()
    return summary


def _has_evidence(evidence: object) -> bool:
    """Return whether an evidence payload has source material."""

    if isinstance(evidence, list):
        return_value = bool(evidence)
        return return_value
    if isinstance(evidence, Mapping):
        return_value = bool(evidence)
        return return_value
    return_value = bool(text_or_empty(evidence))
    return return_value


def _evidence_source_refs(
    evidence: object,
    *,
    source_kind: str,
    origin: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project evidence-local refs or fall back to origin identifiers."""

    source_refs: list[dict[str, Any]] = []
    evidence_items = evidence if isinstance(evidence, list) else [evidence]
    for index, item in enumerate(evidence_items):
        if isinstance(item, Mapping):
            for ref_field in ("source_refs", "evidence_refs"):
                refs = item.get(ref_field)
                if isinstance(refs, list):
                    source_refs.extend(
                        dict(ref)
                        for ref in refs
                        if isinstance(ref, Mapping)
                    )
            if not source_refs:
                source_ref = {"source": source_kind, "evidence_index": index}
                _copy_text_field(item, source_ref, "timestamp", "timestamp")
                _copy_text_field(
                    item,
                    source_ref,
                    "conversation_row_id",
                    "conversation_row_id",
                )
                _copy_text_field(item, source_ref, "message_id", "message_id")
                source_reflection_run_ids = item.get("source_reflection_run_ids")
                if isinstance(source_reflection_run_ids, list):
                    for run_id in source_reflection_run_ids:
                        clean_run_id = text_or_empty(run_id)
                        if clean_run_id:
                            source_refs.append(
                                {
                                    "source": source_kind,
                                    "reflection_run_id": clean_run_id,
                                }
                            )
                if not source_refs:
                    source_refs.append(source_ref)

    if not source_refs:
        source_refs = _origin_source_refs(origin, source_kind)
    return source_refs


def _reflection_source_views(
    state: Mapping[str, Any],
    origin: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build reflection source views when reflection evidence is present."""

    source_views: list[dict[str, Any]] = []
    reflection_payload = _reflection_promotion_payload(state)
    reflection_run_ids = state.get("source_reflection_run_ids")

    if _has_evidence(reflection_payload) or isinstance(
        reflection_run_ids,
        list,
    ):
        source_refs = _evidence_source_refs(
            reflection_payload or [],
            source_kind=REFLECTION_RUN_SOURCE_KIND,
            origin=origin,
        )
        if isinstance(reflection_run_ids, list):
            for run_id in reflection_run_ids:
                clean_run_id = text_or_empty(run_id)
                if clean_run_id:
                    source_refs.append(
                        {
                            "source": REFLECTION_RUN_SOURCE_KIND,
                            "reflection_run_id": clean_run_id,
                        }
                    )

        source_views.append(
            _source_view(
                source_key=REFLECTION_RUN_SOURCE_KIND,
                source_kind=REFLECTION_RUN_SOURCE_KIND,
                summary=_compact_text(reflection_payload),
                source_refs=source_refs,
                extra=_reflection_source_view_extra(reflection_payload),
            )
        )

    style_payload = _reflection_style_payload(state)
    if _has_evidence(style_payload):
        source_views.append(
            _source_view(
                source_key=REFLECTION_USER_STYLE_SOURCE_KEY,
                source_kind=REFLECTION_RUN_SOURCE_KIND,
                summary=_compact_text(style_payload),
                source_refs=_evidence_source_refs(
                    style_payload,
                    source_kind=REFLECTION_RUN_SOURCE_KIND,
                    origin=origin,
                ),
                extra={"source_role": "user_style_signal"},
            )
        )
    return source_views


def _reflection_promotion_payload(state: Mapping[str, Any]) -> object:
    """Collect reflection-shaped promotion evidence from state or RAG."""

    reflection_payload = state.get("reflection_evidence")
    if _has_evidence(reflection_payload):
        return reflection_payload
    reflection_payload = state.get("reflection_run")
    if _has_evidence(reflection_payload):
        return reflection_payload
    return_value = _rag_reflection_payload(state)
    return return_value


def _rag_reflection_payload(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect promotion-shaped reflection evidence projected under RAG."""

    rag_result = state.get("rag_result")
    if not isinstance(rag_result, Mapping):
        return_value: list[dict[str, Any]] = []
        return return_value

    reflection_payload: list[dict[str, Any]] = []
    memory_evidence = rag_result.get("memory_evidence")
    evidence_items = (
        memory_evidence
        if isinstance(memory_evidence, list)
        else [memory_evidence]
    )
    for evidence in evidence_items:
        if not isinstance(evidence, Mapping):
            continue
        source_kind = text_or_empty(evidence.get("source_kind"))
        evidence_refs = evidence.get("evidence_refs")
        has_reflection_ref = False
        if isinstance(evidence_refs, list):
            for evidence_ref in evidence_refs:
                if (
                    isinstance(evidence_ref, Mapping)
                    and text_or_empty(evidence_ref.get("reflection_run_id"))
                ):
                    has_reflection_ref = True
                    break
        if source_kind == REFLECTION_RUN_SOURCE_KIND or has_reflection_ref:
            reflection_payload.append(dict(evidence))

    return reflection_payload


def _reflection_style_payload(state: Mapping[str, Any]) -> object:
    """Return reflection style evidence without mixing promotion metadata."""

    style_payload = state.get("user_style_signal")
    if _has_evidence(style_payload):
        return style_payload
    rag_result = state.get("rag_result")
    if not isinstance(rag_result, Mapping):
        return_value: dict[str, Any] = {}
        return return_value
    return_value = rag_result.get("user_style_signal") or {}
    return return_value


def _reflection_source_view_extra(
    reflection_payload: object,
) -> dict[str, Any]:
    """Project structural metadata attached to reflection source evidence."""

    evidence_items = (
        reflection_payload
        if isinstance(reflection_payload, list)
        else [reflection_payload]
    )
    for evidence in evidence_items:
        if not isinstance(evidence, Mapping):
            continue
        privacy_review = evidence.get("privacy_review")
        if isinstance(privacy_review, Mapping):
            return_value = {"privacy_review": dict(privacy_review)}
            return return_value
        source_role = text_or_empty(evidence.get("source_role"))
        if source_role:
            return_value = {"source_role": source_role}
            return return_value
    return_value: dict[str, Any] = {}
    return return_value


def _source_kind_set(source_views: list[dict[str, Any]]) -> set[str]:
    """Return source-kind classes from structurally valid source views."""

    source_kinds: set[str] = set()
    for source_view in source_views:
        source_kind = text_or_empty(source_view.get("source_kind"))
        if source_kind:
            source_kinds.add(source_kind)
    return source_kinds


def _all_sources_have_refs(source_views: list[dict[str, Any]]) -> bool:
    """Return whether every source view has at least one source ref."""

    for source_view in source_views:
        source_refs = source_view.get("source_refs")
        if not isinstance(source_refs, list) or not source_refs:
            return_value = False
            return return_value
    return_value = True
    return return_value


def _accepted_if(condition: bool, failure_reason: str) -> dict[str, Any]:
    """Build a source-policy decision dictionary."""

    if condition:
        return_value = {"accepted": True, "reason": "accepted"}
    else:
        return_value = {"accepted": False, "reason": failure_reason}
    return return_value


def _has_user_and_assistant_sources(source_kinds: set[str]) -> bool:
    """Return whether chat request and assistant acceptance classes are present."""

    return_value = (
        USER_MESSAGE_SOURCE_KIND in source_kinds
        and ASSISTANT_ACCEPTANCE_SOURCE_KIND in source_kinds
    )
    return return_value


def _privacy_review_passed(privacy_review: object) -> bool:
    """Validate the structural privacy-review gate for promoted memory."""

    if not isinstance(privacy_review, Mapping):
        return_value = False
        return return_value
    return_value = privacy_review.get("user_details_removed") is True
    return return_value
