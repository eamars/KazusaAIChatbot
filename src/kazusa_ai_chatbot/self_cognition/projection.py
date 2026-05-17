"""Build bounded self-cognition source packets for shared cognition."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.config import (
    SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT,
    SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT,
)
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.time_boundary import (
    format_storage_utc_for_llm,
    local_time_context_from_storage_utc,
)

_RAG_OUTPUT_TIME_FIELDS = frozenset(
    (
        "timestamp",
        "created_at",
        "updated_at",
        "due_at",
        "execute_at",
        "completed_at",
        "recorded_at",
    )
)


def build_source_packet(
    case: models.SelfCognitionCase,
    rag_output: dict[str, Any] | None = None,
) -> models.SourcePacket:
    """Project one dry-run case into a bounded model-facing packet.

    Args:
        case: External case-file data supplied by the dry-run caller.
        rag_output: Optional RAG2 result from one supervisor invocation.

    Returns:
        Source packet containing semantic labels and bounded visible evidence.
    """

    idle_timestamp_utc = _string_field(case, "idle_timestamp_utc")
    last_evidence_timestamp_utc = _string_field(
        case,
        "last_evidence_timestamp_utc",
    )
    local_time_context = local_time_context_from_storage_utc(
        idle_timestamp_utc,
    )
    target_scope = _target_scope(case)
    source_refs = _source_refs(case)
    visible_context = _visible_context(case)
    packet: models.SourcePacket = {
        "instruction": models.SELF_COGNITION_INPUT_TEXT,
        "case_name": _string_field(case, "case_name"),
        "idle_local_datetime": format_storage_utc_for_llm(
            idle_timestamp_utc,
        ),
        "last_evidence_local_datetime": format_storage_utc_for_llm(
            last_evidence_timestamp_utc,
        ),
        "local_time_context": local_time_context,
        "trigger_kind": _string_field(case, "trigger_kind"),
        "semantic_due_state": _optional_string_field(
            case,
            "semantic_due_state",
        ),
        "actionability": _string_field(case, "actionability"),
        "target_scope": target_scope,
        "source_refs": source_refs,
        "visible_context": visible_context,
        "agency_options": [
            "Silence is allowed when contact would add no value.",
            (
                "If outward contact is beneficial, select a visible `speak` "
                "action through the shared action-spec contract."
            ),
            (
                "If only internal tracking is useful, emit "
                f"{models.PROGRESS_MAINTENANCE_MARKER}."
            ),
            (
                "If the evidence should only be audited, emit "
                f"{models.AUDIT_ONLY_MARKER} or "
                f"{models.SILENT_NO_WRITE_MARKER}."
            ),
        ],
    }

    conversation_progress = case.get("conversation_progress")
    if isinstance(conversation_progress, dict):
        packet["conversation_progress"] = conversation_progress

    current_mood = case.get("current_mood")
    if isinstance(current_mood, str) and current_mood:
        packet["current_mood"] = current_mood

    global_vibe = case.get("global_vibe")
    if isinstance(global_vibe, str) and global_vibe:
        packet["global_vibe"] = global_vibe

    reflection_modifier = case.get("reflection_modifier")
    if isinstance(reflection_modifier, dict):
        packet["reflection_modifier"] = reflection_modifier

    if rag_output is not None:
        packet["rag_evidence"] = project_rag_output(rag_output)

    return packet


def render_source_packet_text(packet: models.SourcePacket) -> str:
    """Render the source packet into a compact percept body.

    Args:
        packet: Source packet produced by `build_source_packet`.

    Returns:
        Text no longer than the configured source-packet character limit.
    """

    lines = [
        packet['instruction'],
        '这是角色自己的空闲自检，不是用户新发来的消息，也不是外部命令。',
        '我只是在根据自己看得见的对话、承诺和当前心情，判断现在要不要自然地做点什么。',
        '',
        '# 当前自检',
        f'- case_name: {packet["case_name"]}',
        f'- idle_local_datetime: {packet["idle_local_datetime"]}',
        (
            '- last_evidence_local_datetime: '
            f'{packet["last_evidence_local_datetime"]}'
        ),
        f'- local_time_context: {_compact_value(packet["local_time_context"])}',
        f'- trigger_kind: {packet["trigger_kind"]}',
        f'- semantic_due_state: {packet["semantic_due_state"]}',
        f'- actionability: {packet["actionability"]}',
        '',
        '# 我可以选择的自然路线',
    ]
    for option in packet["agency_options"]:
        lines.append(f"- {option}")

    lines.extend(
        [
            '',
            '# 当前心情和氛围',
            f'- current_mood: {packet.get("current_mood", "")}',
            f'- global_vibe: {packet.get("global_vibe", "")}',
            (
                '- reflection_modifier: '
                f'{_compact_value(packet.get("reflection_modifier", {}))}'
            ),
            '',
            '# 目标对象',
            _render_target_scope(packet['target_scope']),
            '',
            '# 触发证据',
            _render_source_refs(packet['source_refs']),
            '',
            '# 最近可见对话',
            _render_visible_context(packet['visible_context']),
            '',
            '# 对话进度',
            _compact_value(packet.get('conversation_progress', {})),
            '',
            '# 检索补充',
            _compact_value(packet.get('rag_evidence', {})),
        ]
    )
    rendered_text = "\n".join(lines)
    clipped_text = _clip_text(
        rendered_text,
        SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT,
    )
    return clipped_text


def build_rag_request(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Build one bounded RAG2 supervisor request for a follow-up topic.

    Args:
        case: External case-file data supplied by the dry-run caller.

    Returns:
        Request artifact for one RAG supervisor invocation.
    """

    query = case.get("rag_query")
    if not isinstance(query, str) or not query.strip():
        query = _fallback_rag_query(case)
    target_scope = _target_scope(case)
    user_id = target_scope["user_id"] or ""
    idle_timestamp_utc = _string_field(case, "idle_timestamp_utc")
    visible_context = _rag_visible_context(case)
    user_profile = case.get("user_profile")
    if not isinstance(user_profile, dict):
        user_profile = {}
    display_name = user_profile.get("display_name")
    if not isinstance(display_name, str):
        display_name = ""
    character_profile = case.get("character_profile")
    if not isinstance(character_profile, dict):
        character_profile = {}
    prompt_message_context = {
        "body_text": query.strip(),
        "addressed_to_global_user_ids": [user_id] if user_id else [],
        "broadcast": target_scope["channel_type"] == "group",
        "mentions": [],
        "attachments": [],
    }
    request = {
        "query": query.strip(),
        "context": {
            "platform": target_scope["platform"],
            "platform_channel_id": target_scope["platform_channel_id"],
            "channel_type": target_scope["channel_type"],
            "global_user_id": user_id,
            "platform_user_id": user_id,
            "display_name": display_name,
            "user_name": display_name,
            "user_profile": user_profile,
            "character_profile": character_profile,
            "current_timestamp_utc": idle_timestamp_utc,
            "local_time_context": local_time_context_from_storage_utc(
                idle_timestamp_utc,
            ),
            "prompt_message_context": prompt_message_context,
            "channel_topic": _string_field(case, "channel_topic"),
            "chat_history_recent": visible_context,
            "chat_history_wide": visible_context,
            "reply_context": {},
            "indirect_speech_context": "",
            "conversation_progress": case.get("conversation_progress"),
            "conversation_episode_state": case.get(
                "conversation_episode_state",
            ),
            "promoted_reflection_context": case.get(
                "promoted_reflection_context",
            ),
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
        },
        "budget": {
            "rag_supervisor_invocations": (
                models.RAG_SUPERVISOR_INVOCATION_LIMIT
            ),
        },
    }
    return request


def project_rag_output(rag_output: dict[str, Any]) -> dict[str, Any]:
    """Project RAG2 output into the self-cognition evidence budget.

    Args:
        rag_output: Raw result returned by the RAG2 supervisor.

    Returns:
        Bounded dict retaining the factual answer and compact fact list.
    """

    projected_rag_output = _project_rag_time_fields(rag_output)
    rendered = json.dumps(
        projected_rag_output,
        ensure_ascii=False,
        sort_keys=True,
    )
    clipped = _clip_text(rendered, SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT)
    projected_output = {
        "bounded_json": clipped,
        "char_limit": SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT,
    }
    return projected_output


def validate_case_name(case: models.SelfCognitionCase) -> str:
    """Return a supported case name or raise for an unsupported case.

    Args:
        case: External case-file data supplied by the dry-run caller.

    Returns:
        The validated case name.

    Raises:
        ValueError: If the case name is missing or unsupported.
    """

    case_name = _string_field(case, "case_name")
    if case_name not in models.SUPPORTED_CASE_NAMES:
        raise ValueError(f"unsupported self-cognition case: {case_name}")
    return case_name


def _target_scope(
    case: models.SelfCognitionCase,
) -> models.SelfCognitionTargetScope:
    """Normalize the externally supplied target scope for model input."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    platform = value.get("platform")
    platform_channel_id = value.get("platform_channel_id")
    channel_type = value.get("channel_type")
    user_id = value.get("user_id")
    scope: models.SelfCognitionTargetScope = {
        "platform": platform if isinstance(platform, str) else "",
        "platform_channel_id": (
            platform_channel_id if isinstance(platform_channel_id, str) else ""
        ),
        "channel_type": channel_type if isinstance(channel_type, str) else "",
        "user_id": user_id if isinstance(user_id, str) else None,
    }
    return scope


def _source_refs(
    case: models.SelfCognitionCase,
) -> list[models.SelfCognitionSourceRef]:
    """Normalize source references while preserving only supported fields."""

    value = case.get("source_refs")
    if not isinstance(value, list):
        return_value: list[models.SelfCognitionSourceRef] = []
        return return_value

    refs: list[models.SelfCognitionSourceRef] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_ref: models.SelfCognitionSourceRef = {
            "source_kind": _string_field(item, "source_kind"),
            "source_id": _string_field(item, "source_id"),
            "summary": _string_field(item, "summary"),
        }
        due_at = item.get("due_at")
        if due_at is None:
            source_ref["due_at"] = None
        elif isinstance(due_at, str):
            source_ref["due_at"] = (
                format_storage_utc_for_llm(due_at) or None
            )
        refs.append(source_ref)
    return refs


def _visible_context(case: models.SelfCognitionCase) -> list[dict[str, Any]]:
    """Copy visible dialog rows and localize storage times for model input."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            row = dict(item)
            raw_timestamp = row.get("timestamp")
            if isinstance(raw_timestamp, str):
                row["timestamp"] = format_storage_utc_for_llm(raw_timestamp)
            rows.append(row)
    return rows


def _rag_visible_context(
    case: models.SelfCognitionCase,
) -> list[dict[str, Any]]:
    """Copy visible dialog rows for internal RAG runtime context."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _fallback_rag_query(case: models.SelfCognitionCase) -> str:
    """Build a compact retrieval query when the case omits one."""

    source_refs = _source_refs(case)
    summaries = [
        item["summary"]
        for item in source_refs
        if item.get("summary")
    ]
    if summaries:
        query = " ".join(summaries)
    else:
        query = "self-cognition bounded follow-up topic"
    return query


def _render_target_scope(
    target_scope: models.SelfCognitionTargetScope,
) -> str:
    """Render normalized target scope into source-packet text."""

    lines = [
        f'- platform: {target_scope["platform"]}',
        f'- platform_channel_id: {target_scope["platform_channel_id"]}',
        f'- channel_type: {target_scope["channel_type"]}',
        f'- user_id: {target_scope["user_id"]}',
    ]
    rendered = '\n'.join(lines)
    return rendered


def _render_source_refs(
    source_refs: list[models.SelfCognitionSourceRef],
) -> str:
    """Render source references into source-packet evidence bullets."""

    if not source_refs:
        return_value = '- none'
        return return_value

    lines: list[str] = []
    for source_ref in source_refs:
        source_kind = source_ref.get('source_kind', '')
        source_id = source_ref.get('source_id', '')
        due_at = source_ref.get('due_at')
        summary = source_ref.get('summary', '')
        lines.append(f'- {source_kind}:{source_id}')
        if due_at:
            lines.append(f'  due_at: {due_at}')
        if summary:
            lines.append(f'  summary: {summary}')
    rendered = '\n'.join(lines)
    return rendered


def _render_visible_context(rows: list[dict[str, Any]]) -> str:
    """Render visible dialog rows into source-packet evidence bullets."""

    if not rows:
        return_value = '- none'
        return return_value

    lines: list[str] = []
    for row in rows:
        timestamp = _string_field(row, 'timestamp')
        role = _string_field(row, 'role')
        display_name = _string_field(row, 'display_name')
        body_text = _string_field(row, 'body_text')
        if not body_text:
            body_text = _string_field(row, 'text')
        speaker = display_name or role
        lines.append(f'- {timestamp} {speaker}: {body_text}')
    rendered = '\n'.join(lines)
    return rendered


def _project_rag_time_fields(value: object) -> object:
    """Project storage UTC time fields in RAG evidence to local text."""

    if isinstance(value, dict):
        projected: dict[str, object] = {}
        for key, item in value.items():
            if key in _RAG_OUTPUT_TIME_FIELDS and isinstance(item, str):
                projected[key] = format_storage_utc_for_llm(item)
            else:
                projected[key] = _project_rag_time_fields(item)
        return projected

    if isinstance(value, list):
        projected_list = [
            _project_rag_time_fields(item)
            for item in value
        ]
        return projected_list

    return value


def _compact_value(value: object) -> str:
    """Render optional structured context as compact JSON text."""

    if value in ({}, [], None, ''):
        return_value = ''
        return return_value
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return rendered


def _clip_text(text: str, limit: int) -> str:
    """Clip model-facing text to a configured character limit."""

    if len(text) <= limit:
        return_value = text
        return return_value
    suffix = "\n[truncated]"
    body_limit = limit - len(suffix)
    clipped = text[:body_limit].rstrip()
    return_value = f"{clipped}{suffix}"
    return return_value


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value


def _optional_string_field(
    case: dict[str, Any],
    field_name: str,
) -> str | None:
    """Read an optional external string-or-null field safely."""

    value = case.get(field_name)
    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, str):
        return_value = None
        return return_value
    return_value = value
    return return_value
