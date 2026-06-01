"""Build bounded self-cognition source packets for shared cognition."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.config import SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.time_boundary import (
    format_storage_utc_for_llm,
    local_time_context_from_storage_utc,
)

_DUE_STATE_LABELS = {
    models.DUE_STATE_FUTURE_DUE: '未到期',
    models.DUE_STATE_DUE_NOW: '当前到期',
    models.DUE_STATE_PAST_DUE: '已过期',
}


def build_source_packet(
    case: models.SelfCognitionCase,
) -> models.SourcePacket:
    """Project one self-cognition source case into a bounded model packet.

    Args:
        case: Source data collected by the self-cognition worker.

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
        "instruction": _instruction_for_case(case),
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
    }

    conversation_progress = case.get("conversation_progress")
    if isinstance(conversation_progress, dict):
        packet["conversation_progress"] = conversation_progress

    group_activity_window = _group_activity_window(case)
    if group_activity_window is not None:
        packet["group_activity_window"] = group_activity_window

    current_mood = case.get("current_mood")
    if isinstance(current_mood, str) and current_mood:
        packet["current_mood"] = current_mood

    global_vibe = case.get("global_vibe")
    if isinstance(global_vibe, str) and global_vibe:
        packet["global_vibe"] = global_vibe

    reflection_modifier = case.get("reflection_modifier")
    if isinstance(reflection_modifier, dict):
        packet["reflection_modifier"] = reflection_modifier

    return packet


def render_source_packet_text(packet: models.SourcePacket) -> str:
    """Render the source packet into a compact percept body.

    Args:
        packet: Source packet produced by `build_source_packet`.

    Returns:
        Text no longer than the configured source-packet character limit.
    """

    lines = [packet['instruction']]
    reason_line = _source_packet_reason_line(packet)
    if reason_line:
        lines.append(reason_line)
    lines.extend([
        '',
        '# 当前聊天窗口',
        f'- idle_local_datetime: {packet["idle_local_datetime"]}',
        (
            '- last_evidence_local_datetime: '
            f'{packet["last_evidence_local_datetime"]}'
        ),
        f'- local_time_context: {_compact_value(packet["local_time_context"])}',
    ])
    source_state = _render_source_state(packet)
    if source_state:
        lines.extend(
            [
                '',
                '# 来源状态',
                source_state,
            ]
        )
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
            '# 聊天位置',
            _render_target_scope(packet['target_scope']),
            '',
            '# 来源依据',
            _render_source_refs(packet['source_refs']),
        ]
    )
    group_activity_window = packet.get('group_activity_window')
    if isinstance(group_activity_window, dict):
        lines.extend(
            [
                '',
                '# 群聊窗口信息',
                _compact_value(group_activity_window),
            ]
        )
    lines.extend(
        [
            '',
            '# 最近可见对话',
            _render_visible_context(packet['visible_context']),
            '',
            '# 对话进度',
            _compact_value(packet.get('conversation_progress', {})),
        ]
    )
    rendered_text = "\n".join(lines)
    clipped_text = _clip_text(
        rendered_text,
        SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT,
    )
    return clipped_text


def _instruction_for_case(case: models.SelfCognitionCase) -> str:
    """Return the model-facing data-source line for a source case."""

    trigger_kind = _string_field(case, "trigger_kind")
    if trigger_kind == models.TRIGGER_GROUP_CHAT_REVIEW:
        group_activity_window = _group_activity_window(case)
        if _is_directly_addressed_group_window(group_activity_window):
            return_value = '我刚看到群里刚刚发生的一段现场。里面有人把话题指向我。'
            return return_value
        return_value = (
            '我刚看到群里刚刚发生的一段现场。'
            '我之前没有插话，这段里也没有人把话题交给我。'
        )
        return return_value
    target_scope = _target_scope(case)
    if target_scope["channel_type"] == "private":
        return_value = '来源位置：我和对方私聊窗口的最近可见内容。'
        return return_value
    if target_scope["channel_type"] == "group":
        return_value = '来源位置：我所在群聊窗口的最近可见内容。'
        return return_value
    return_value = models.SELF_COGNITION_INPUT_TEXT
    return return_value


def _group_activity_window(
    case: models.SelfCognitionCase,
) -> dict[str, Any] | None:
    """Project the semantic group-window source-packet contract."""

    group_activity_window = case.get("group_activity_window")
    if isinstance(group_activity_window, dict):
        return_value = _sanitize_group_activity_window(group_activity_window)
        return return_value

    if _string_field(case, "trigger_kind") != models.TRIGGER_GROUP_CHAT_REVIEW:
        return_value = None
        return return_value

    conversation_progress = case.get("conversation_progress")
    if not isinstance(conversation_progress, dict):
        return_value = None
        return return_value
    source = conversation_progress.get("source")
    window_start = conversation_progress.get("window_start")
    window_end = conversation_progress.get("window_end")
    activity_labels = conversation_progress.get("activity_labels")
    if not isinstance(source, str):
        return_value = None
        return return_value
    if not isinstance(window_start, str):
        return_value = None
        return return_value
    if not isinstance(window_end, str):
        return_value = None
        return return_value
    if not isinstance(activity_labels, dict):
        return_value = None
        return return_value

    return_value = _sanitize_group_activity_window({
        "source": source,
        "window_start": window_start,
        "window_end": window_end,
        "semantic_labels": activity_labels,
    })
    return return_value


def _sanitize_group_activity_window(
    group_activity_window: dict[str, Any],
) -> dict[str, Any] | None:
    """Keep only the prompt-approved group-window evidence fields."""

    source = group_activity_window.get("source")
    window_start = group_activity_window.get("window_start")
    window_end = group_activity_window.get("window_end")
    semantic_labels = group_activity_window.get("semantic_labels")
    if not isinstance(source, str):
        return_value = None
        return return_value
    if not isinstance(window_start, str):
        return_value = None
        return return_value
    if not isinstance(window_end, str):
        return_value = None
        return return_value
    if not isinstance(semantic_labels, dict):
        return_value = None
        return return_value
    safe_labels = {
        key: value
        for key, value in semantic_labels.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    return_value = {
        "source": source,
        "window_start": window_start,
        "window_end": window_end,
        "semantic_labels": safe_labels,
    }
    return return_value


def _is_directly_addressed_group_window(
    group_activity_window: dict[str, Any] | None,
) -> bool:
    """Return whether semantic labels say the group window addressed the bot."""

    if group_activity_window is None:
        return_value = False
        return return_value
    semantic_labels = group_activity_window["semantic_labels"]
    bot_addressing = semantic_labels.get("bot_addressing", "")
    is_directly_addressed = bot_addressing == "directly_addressed"
    return is_directly_addressed


def _source_packet_reason_line(packet: models.SourcePacket) -> str:
    """Return why the current chat-window data is visible to the character."""

    if packet["trigger_kind"] == models.TRIGGER_GROUP_CHAT_REVIEW:
        return_value = ''
        return return_value
    if packet["target_scope"]["channel_type"] == "private":
        return_value = (
            '出现原因：我在这段私聊里，需要接上这段对话的时间线和约定。'
        )
        return return_value
    if packet["target_scope"]["channel_type"] == "group":
        return_value = (
            '出现原因：我在这个群聊里，刚看到这段群聊的时间线和现场感。'
        )
        return return_value
    return_value = '出现原因：我正在查看这段聊天的时间线和现场感。'
    return return_value


def _render_source_state(packet: models.SourcePacket) -> str:
    """Render neutral source-state facts without action route guidance."""

    semantic_due_state = packet['semantic_due_state']
    if semantic_due_state is None:
        return_value = ''
        return return_value
    due_state_label = _DUE_STATE_LABELS.get(
        semantic_due_state,
        semantic_due_state,
    )
    rendered = f'- 约定状态: {due_state_label}'
    return rendered


def validate_case_name(case: models.SelfCognitionCase) -> str:
    """Return a supported case name or raise for an unsupported case.

    Args:
        case: Self-cognition source case.

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


def _render_target_scope(
    target_scope: models.SelfCognitionTargetScope,
) -> str:
    """Render normalized target scope into source-packet text."""

    lines = [
        f'- platform: {target_scope["platform"]}',
        f'- channel_type: {target_scope["channel_type"]}',
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
