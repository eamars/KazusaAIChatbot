"""Build bounded self-cognition source packets for shared cognition."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.channel_scene_projection import (
    project_group_review_instruction_preamble,
)
from kazusa_ai_chatbot.config import SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    validate_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.conversation_progress import (
    project_conversation_progress_evidence,
    project_conversation_progress_scene,
)
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

    validate_case_contract(case)
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

    source_context = _source_context(case)
    if source_context is not None:
        packet["source_context"] = source_context

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
            '# 聊天位置',
            _render_target_scope(packet['target_scope']),
            '',
            '# 来源依据',
            _render_source_refs(packet['source_refs']),
        ]
    )
    source_context = packet.get('source_context')
    if isinstance(source_context, dict):
        lines.extend(
            [
                '',
                '# 来源上下文',
                _compact_value(source_context),
            ]
        )
    group_activity_window = _group_activity_window_from_packet(packet)
    if group_activity_window is not None:
        lines.extend(
            [
                '',
                '# 群聊窗口信息',
                _compact_value(group_activity_window),
            ]
        )
    thread_reference_context = _thread_reference_context(packet)
    if thread_reference_context:
        lines.extend(
            [
                '',
                '# 二人称指向边界',
                _render_thread_reference_context(thread_reference_context),
            ]
        )
    lines.extend(
        [
            '',
            '# 最近可见对话',
            _render_visible_context(packet['visible_context']),
            '',
            '# 对话进度',
            _compact_value(
                _render_conversation_progress(
                    packet.get('conversation_progress', {}),
                ),
            ),
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
        return_value = _group_review_instruction(case)
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


def _group_review_instruction(case: models.SelfCognitionCase) -> str:
    """Build instruction for a group_chat_review trigger.

    Prefer the LLM-generated scene digest when available.  The digest
    already describes whether the character spoke in the window and what
    happened after, making the instruction factually consistent with
    visible_context.

    When the digest is unavailable, fall back to a multi-signal
    deterministic framing that checks both ``assistant_presence`` and
    ``bot_addressing`` so the instruction never contradicts the evidence.
    """

    group_review_preamble = project_group_review_instruction_preamble(
        _string_field(case, "channel_topic"),
    )

    digest_text = _group_scene_digest_text(case)
    if digest_text:
        return_value = f'{group_review_preamble}\n{digest_text}'
        return return_value

    return_value = _deterministic_group_review_instruction(
        case,
        preamble=group_review_preamble,
    )
    return return_value


def _group_scene_digest_text(case: models.SelfCognitionCase) -> str:
    """Extract the scene digest string from a group review case."""

    source_context = case.get("source_context")
    if not isinstance(source_context, dict):
        return_value = ""
        return return_value
    group_scene_digest = source_context.get("group_scene_digest")
    if not isinstance(group_scene_digest, dict):
        return_value = ""
        return return_value
    digest = group_scene_digest.get("digest")
    if not isinstance(digest, str):
        return_value = ""
        return return_value
    return_value = digest.strip()
    return return_value


def _deterministic_group_review_instruction(
    case: models.SelfCognitionCase,
    *,
    preamble: str,
) -> str:
    """Multi-signal fallback when the scene digest is unavailable.

    Checks both ``assistant_presence`` and ``bot_addressing`` to build an
    instruction that never contradicts what ``visible_context`` contains.
    """

    group_activity_window = _group_activity_window(case)
    assistant_present = _is_assistant_present_group_window(
        group_activity_window,
    )
    directly_addressed = _is_directly_addressed_group_window(
        group_activity_window,
    )

    participation_line = (
        '现场里有当前角色之前的可见发言。'
        if assistant_present
        else '现场里当前角色之前没有插话。'
    )
    addressing_line = (
        '现场标签显示有人把话题指向当前角色。'
        if directly_addressed
        else '现场标签没有显示有人把话题交给当前角色。'
    )

    return_value = f'{preamble}{participation_line}{addressing_line}'
    return return_value


def _group_activity_window(
    case: models.SelfCognitionCase,
) -> dict[str, Any] | None:
    """Project the semantic group-window source-packet contract."""

    source_context = case.get("source_context")
    if not isinstance(source_context, dict):
        return_value = None
        return return_value
    group_activity_window = source_context.get("group_activity_window")
    if not isinstance(group_activity_window, dict):
        return_value = None
        return return_value
    return_value = _sanitize_group_activity_window(group_activity_window)
    return return_value


def _group_activity_window_from_packet(
    packet: models.SourcePacket,
) -> dict[str, Any] | None:
    """Read the projected group-window context from a source packet."""

    source_context = packet.get("source_context")
    if not isinstance(source_context, dict):
        return_value = None
        return return_value
    group_activity_window = source_context.get("group_activity_window")
    if not isinstance(group_activity_window, dict):
        return_value = None
        return return_value
    return_value = group_activity_window
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


def _source_context(
    case: models.SelfCognitionCase,
) -> dict[str, Any] | None:
    """Project source-owned group or scheduled context through allowlists."""

    raw_context = case.get("source_context")
    if not isinstance(raw_context, dict):
        return_value = None
        return return_value

    context_kind = raw_context.get("context_kind")
    if context_kind == "group_chat_review":
        group_activity_window = raw_context.get("group_activity_window")
        if not isinstance(group_activity_window, dict):
            return_value = None
            return return_value
        projected_window = _sanitize_group_activity_window(
            group_activity_window,
        )
        if projected_window is None:
            return_value = None
            return return_value
        projected_context: dict[str, Any] = {
            "context_kind": context_kind,
            "group_activity_window": projected_window,
            "conversation_evidence": _prompt_string_list(
                raw_context.get("conversation_evidence"),
            ),
        }
        participant_context = _project_participant_context(
            raw_context.get("participant_context"),
        )
        if participant_context is not None:
            projected_context["participant_context"] = participant_context
        thread_reference_context = _project_thread_reference_context(
            raw_context.get("thread_reference_context"),
        )
        if thread_reference_context is not None:
            projected_context["thread_reference_context"] = (
                thread_reference_context
            )
        group_scene_digest = _project_group_scene_digest(
            raw_context.get("group_scene_digest"),
        )
        if group_scene_digest is not None:
            projected_context["group_scene_digest"] = group_scene_digest
        return projected_context

    if context_kind == "scheduled_future_cognition":
        continuation_objective = raw_context.get("continuation_objective")
        continuation_mode = raw_context.get("continuation_mode")
        if not isinstance(continuation_objective, str):
            return_value = None
            return return_value
        if not isinstance(continuation_mode, str):
            return_value = None
            return return_value
        projected_context: dict[str, Any] = {
            "context_kind": context_kind,
            "continuation_objective": continuation_objective,
            "continuation_mode": continuation_mode,
        }
        projected_authority = _project_scheduled_authority(case)
        if projected_authority is not None:
            projected_context["scheduled_authority"] = projected_authority
        return_value = projected_context
        return return_value

    return_value = None
    return return_value


def _project_scheduled_authority(
    case: models.SelfCognitionCase,
) -> dict[str, Any] | None:
    """Project the bounded semantic authority without delivery identifiers."""

    authority = case.get("scheduled_future_speech_authority")
    if not isinstance(authority, dict):
        return_value = None
        return return_value
    authorized_content = authority.get("authorized_content")
    if not isinstance(authorized_content, dict):
        return_value = None
        return return_value
    target = authority.get("target")
    if not isinstance(target, dict):
        return_value = None
        return return_value
    detail_refs = [
        {
            "semantic_summary": _string_field(
                detail_ref,
                "semantic_summary",
            ),
            "provenance_role": _string_field(
                detail_ref,
                "provenance_role",
            ),
        }
        for detail_ref in authorized_content.get("detail_refs", [])
        if isinstance(detail_ref, dict)
    ]
    trigger = authority.get("trigger")
    trigger_utc = trigger.get("utc") if isinstance(trigger, dict) else None
    projected_authority: dict[str, Any] = {
        "objective": _string_field(authority, "semantic_objective"),
        "summary": _string_field(authorized_content, "summary"),
        "detail_refs": detail_refs,
        "audience_kind": _string_field(target, "audience_kind"),
        "local_due_datetime": format_storage_utc_for_llm(trigger_utc),
    }
    return projected_authority


def _project_participant_context(value: object) -> dict[str, Any] | None:
    """Keep semantic participant context fields without identity metadata."""

    if not isinstance(value, dict):
        return_value = None
        return return_value
    projected: dict[str, Any] = {}
    for field_name in (
        "source",
        "context_shape",
        "focus_mode",
        "guidance",
    ):
        field_value = value.get(field_name)
        if isinstance(field_value, str):
            projected[field_name] = field_value

    primary = value.get("primary_reply_target")
    if isinstance(primary, dict):
        projected_primary: dict[str, Any] = {}
        for field_name in (
            "display_name",
            "reply_target_fit",
            "relationship_label",
            "relationship_band",
        ):
            field_value = primary.get(field_name)
            if isinstance(field_value, str):
                projected_primary[field_name] = field_value
        role_in_window = _prompt_string_list(primary.get("role_in_window"))
        if role_in_window:
            projected_primary["role_in_window"] = role_in_window
        for field_name in (
            "engagement_guidelines",
            "nearby_conversation_evidence",
            "visible_samples",
        ):
            field_value = _prompt_string_list(primary.get(field_name))
            if field_value:
                projected_primary[field_name] = field_value
        if projected_primary:
            projected["primary_reply_target"] = projected_primary

    background_flow = value.get("background_flow")
    if isinstance(background_flow, dict):
        projected_background: dict[str, str] = {}
        for field_name in (
            "mode",
            "summary",
            "participant_count_label",
        ):
            field_value = background_flow.get(field_name)
            if isinstance(field_value, str):
                projected_background[field_name] = field_value
        if projected_background:
            projected["background_flow"] = projected_background

    if not projected:
        return_value = None
        return return_value
    return projected


def _project_thread_reference_context(value: object) -> dict[str, Any] | None:
    """Keep bounded second-person warnings without row or delivery ids."""

    if not isinstance(value, dict):
        return_value = None
        return return_value
    projected: dict[str, Any] = {}
    for field_name in ("source", "context_shape", "guidance"):
        field_value = value.get(field_name)
        if isinstance(field_value, str):
            projected[field_name] = field_value
    raw_rows = value.get("ambiguous_second_person_rows")
    if isinstance(raw_rows, list):
        rows: list[dict[str, str]] = []
        for raw_row in raw_rows[:3]:
            if not isinstance(raw_row, dict):
                continue
            row = {
                field_name: raw_row[field_name]
                for field_name in (
                    "speaker",
                    "sample",
                    "referent_status",
                    "basis",
                )
                if isinstance(raw_row.get(field_name), str)
            }
            if row:
                rows.append(row)
        if rows:
            projected["ambiguous_second_person_rows"] = rows
    if not projected:
        return_value = None
        return return_value
    return projected


def _project_group_scene_digest(value: object) -> dict[str, str] | None:
    """Keep the neutral group digest and its optional semantic summary."""

    if not isinstance(value, dict):
        return_value = None
        return return_value
    digest = value.get("digest")
    if not isinstance(digest, str) or not digest.strip():
        return_value = None
        return return_value
    projected = {"digest": digest}
    summary = value.get("summary")
    if isinstance(summary, str) and summary.strip():
        projected["summary"] = summary
    return projected


def _prompt_string_list(value: object) -> list[str]:
    """Return bounded prompt text items from an external list value."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value
    values = [item for item in value if isinstance(item, str)]
    return_value = values
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


def _is_assistant_present_group_window(
    group_activity_window: dict[str, Any] | None,
) -> bool:
    """Return whether semantic labels say the bot spoke in the window."""

    if group_activity_window is None:
        return_value = False
        return return_value
    semantic_labels = group_activity_window.get("semantic_labels", {})
    assistant_presence = semantic_labels.get("assistant_presence", "")
    is_present = assistant_presence == "present"
    return is_present


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


def _thread_reference_context(packet: models.SourcePacket) -> dict[str, Any]:
    """Return prompt-safe thread-reference context from source context."""

    source_context = packet.get('source_context')
    if not isinstance(source_context, dict):
        return_value: dict[str, Any] = {}
        return return_value
    thread_reference_context = source_context.get(
        'thread_reference_context',
    )
    if not isinstance(thread_reference_context, dict):
        return_value = {}
        return return_value
    return_value = thread_reference_context
    return return_value


def _render_thread_reference_context(context: dict[str, Any]) -> str:
    """Render bounded second-person reference warnings for cognition."""

    lines: list[str] = []
    guidance = _string_field(context, 'guidance')
    if guidance:
        lines.append(f'- guidance: {guidance}')

    rows = context.get('ambiguous_second_person_rows')
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            speaker = _string_field(row, 'speaker')
            sample = _string_field(row, 'sample')
            referent_status = _string_field(row, 'referent_status')
            basis = _string_field(row, 'basis')
            if not speaker or not sample:
                continue
            lines.append(f'- {speaker}: {sample}')
            if referent_status:
                lines.append(f'  referent_status: {referent_status}')
            if basis:
                lines.append(f'  basis: {basis}')

    if not lines:
        return_value = '- none'
        return return_value
    rendered = '\n'.join(lines)
    return rendered


_CONVERSATION_PROGRESS_FIELDS = frozenset({
    'schema_version',
    'episode_state_id',
    'status',
    'continuity',
    'turn_count',
    'current_thread',
    'character_stance',
    'user_goal',
    'current_blocker',
    'emotional_trajectory',
    'episode_narrative',
    'events',
    'overused_moves',
    'interaction_logical_turns',
    'compacted_block_refs',
})


def validate_case_contract(case: models.SelfCognitionCase) -> None:
    """Validate source-owned context before it reaches model projection."""

    trigger_kind = _string_field(case, 'trigger_kind')
    progress = case.get('conversation_progress')
    if progress is not None:
        _validate_conversation_progress(
            progress,
            occurred_at=_string_field(case, 'idle_timestamp_utc'),
        )

    source_context = case.get('source_context')
    if trigger_kind == models.TRIGGER_GROUP_CHAT_REVIEW:
        if 'conversation_progress' not in case or progress is not None:
            raise ValueError(
                'group review cases require conversation_progress=None'
            )
        _validate_group_source_context(source_context)
        return
    if trigger_kind == models.TRIGGER_SCHEDULED_FUTURE_COGNITION:
        if 'conversation_progress' not in case or progress is not None:
            raise ValueError(
                'scheduled cognition cases require conversation_progress=None'
            )
        if 'scheduled_future_speech_authority' in case:
            _validate_scheduled_authority_contract(case)
        _validate_scheduled_source_context(source_context)
        return
    if source_context is not None:
        raise ValueError(
            'source_context is only valid for group or scheduled cognition'
        )


def _validate_conversation_progress(
    value: object,
    *,
    occurred_at: str,
) -> None:
    """Validate the canonical V2 continuity projection when supplied."""

    if not isinstance(value, dict):
        raise ValueError('conversation_progress must be an object or None')
    if set(value) != _CONVERSATION_PROGRESS_FIELDS:
        raise ValueError('conversation_progress fields are not exact')
    if value.get('schema_version') != 'conversation_progress_prompt.v2':
        raise ValueError('conversation_progress schema is invalid')
    for field_name in (
        'episode_state_id',
        'status',
        'continuity',
        'current_thread',
        'character_stance',
        'user_goal',
        'current_blocker',
        'emotional_trajectory',
        'episode_narrative',
    ):
        if not isinstance(value.get(field_name), str):
            raise ValueError(
                f'conversation_progress.{field_name} must be text'
            )
    turn_count = value.get('turn_count')
    if (
        not isinstance(turn_count, int)
        or isinstance(turn_count, bool)
        or turn_count < 0
    ):
        raise ValueError('conversation_progress.turn_count is invalid')
    for field_name in (
        'events',
        'overused_moves',
        'interaction_logical_turns',
        'compacted_block_refs',
    ):
        if not isinstance(value.get(field_name), list):
            raise ValueError(
                f'conversation_progress.{field_name} must be a list'
            )
    try:
        project_conversation_progress_scene(value)
        project_conversation_progress_evidence(value, occurred_at)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            'conversation_progress nested fields are invalid'
        ) from exc


def _validate_group_source_context(value: object) -> None:
    """Validate the complete source-owned group context union member."""

    if not isinstance(value, dict):
        raise ValueError('group review source_context is required')
    required_fields = {
        'schema_version',
        'context_kind',
        'group_activity_window',
        'conversation_evidence',
    }
    optional_fields = {
        'participant_context',
        'thread_reference_context',
        'group_scene_digest',
    }
    if set(value) - required_fields - optional_fields:
        raise ValueError('group source_context fields are not exact')
    if value.get('schema_version') != (
        'self_cognition_group_source_context.v1'
    ) or value.get('context_kind') != 'group_chat_review':
        raise ValueError('group source_context schema is invalid')

    activity_window = value.get('group_activity_window')
    if not isinstance(activity_window, dict):
        raise ValueError('group activity window is required')
    if set(activity_window) != {
        'source',
        'window_start',
        'window_end',
        'semantic_labels',
    }:
        raise ValueError('group activity window fields are not exact')
    for field_name in ('source', 'window_start', 'window_end'):
        if not isinstance(activity_window.get(field_name), str):
            raise ValueError(
                f'group activity window.{field_name} must be text'
            )
    semantic_labels = activity_window.get('semantic_labels')
    if not isinstance(semantic_labels, dict) or any(
        not isinstance(key, str) or not isinstance(label, str)
        for key, label in semantic_labels.items()
    ):
        raise ValueError('group activity window labels are invalid')

    conversation_evidence = value.get('conversation_evidence')
    if not isinstance(conversation_evidence, list) or any(
        not isinstance(item, str) for item in conversation_evidence
    ):
        raise ValueError('group conversation evidence is invalid')
    for field_name in ('participant_context', 'thread_reference_context'):
        optional_value = value.get(field_name)
        if optional_value is not None and not isinstance(optional_value, dict):
            raise ValueError(f'group {field_name} is invalid')
    digest = value.get('group_scene_digest')
    if digest is not None:
        if not isinstance(digest, dict):
            raise ValueError('group scene digest is invalid')
        if set(digest) - {'digest', 'summary'}:
            raise ValueError('group scene digest fields are not exact')
        if not isinstance(digest.get('digest'), str):
            raise ValueError('group scene digest text is required')
        if 'summary' in digest and not isinstance(digest['summary'], str):
            raise ValueError('group scene digest summary is invalid')


def _validate_scheduled_source_context(value: object) -> None:
    """Validate the complete source-owned scheduled context union member."""

    if not isinstance(value, dict):
        raise ValueError('scheduled cognition source_context is required')
    if set(value) != {
        'schema_version',
        'context_kind',
        'continuation_objective',
        'continuation_mode',
    }:
        raise ValueError('scheduled source_context fields are not exact')
    if value.get('schema_version') != (
        'self_cognition_scheduled_source_context.v1'
    ) or value.get('context_kind') != 'scheduled_future_cognition':
        raise ValueError('scheduled source_context schema is invalid')
    for field_name in ('continuation_objective', 'continuation_mode'):
        if not isinstance(value.get(field_name), str):
            raise ValueError(
                f'scheduled source_context.{field_name} must be text'
            )


def _validate_scheduled_authority_contract(
    case: models.SelfCognitionCase,
) -> None:
    """Validate the immutable authority carried by scheduled-speech cases."""

    authority = case.get('scheduled_future_speech_authority')
    if not isinstance(authority, dict):
        raise ValueError(
            'scheduled cognition case requires scheduled_future_speech_authority'
        )
    try:
        validate_scheduled_future_speech_authority(authority)
    except (CognitionContractError, ValueError) as exc:
        raise ValueError(
            f'scheduled cognition authority is invalid: {exc}'
        ) from exc


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
) -> models.SelfCognitionPromptTargetScope:
    """Project channel semantics without delivery or identity metadata."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    platform = value.get("platform")
    channel_type = value.get("channel_type")
    scope: models.SelfCognitionPromptTargetScope = {
        "platform": platform if isinstance(platform, str) else "",
        "channel_type": channel_type if isinstance(channel_type, str) else "",
    }
    return scope


def _source_refs(
    case: models.SelfCognitionCase,
) -> list[models.SelfCognitionPromptSourceRef]:
    """Project source references without storage or scheduler identifiers."""

    value = case.get("source_refs")
    if not isinstance(value, list):
        return_value: list[models.SelfCognitionPromptSourceRef] = []
        return return_value

    refs: list[models.SelfCognitionPromptSourceRef] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_ref: models.SelfCognitionPromptSourceRef = {
            "source_kind": _string_field(item, "source_kind"),
            "summary": _string_field(item, "summary"),
            "due_at": None,
        }
        due_at = item.get("due_at")
        if isinstance(due_at, str):
            source_ref["due_at"] = (
                format_storage_utc_for_llm(due_at) or None
            )
        refs.append(source_ref)
    return refs


def _visible_context(
    case: models.SelfCognitionCase,
) -> list[models.SelfCognitionVisibleContextRow]:
    """Copy visible dialog rows and localize storage times for model input."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[models.SelfCognitionVisibleContextRow] = []
        return return_value

    rows: list[models.SelfCognitionVisibleContextRow] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        body_text = item.get("body_text")
        if not isinstance(body_text, str) or not body_text.strip():
            continue
        raw_timestamp = item.get("timestamp")
        timestamp = (
            format_storage_utc_for_llm(raw_timestamp)
            if isinstance(raw_timestamp, str)
            else ""
        )
        if not timestamp:
            continue
        row: models.SelfCognitionVisibleContextRow = {
            "role": _string_field(item, "role"),
            "display_name": _string_field(item, "display_name"),
            "timestamp": timestamp,
            "body_text": body_text.strip(),
        }
        rows.append(row)
    return rows


def _render_target_scope(
    target_scope: models.SelfCognitionPromptTargetScope,
) -> str:
    """Render normalized target scope into source-packet text."""

    lines = [
        f'- platform: {target_scope["platform"]}',
        f'- channel_type: {target_scope["channel_type"]}',
    ]
    rendered = '\n'.join(lines)
    return rendered


def _render_source_refs(
    source_refs: list[models.SelfCognitionPromptSourceRef],
) -> str:
    """Render source references into source-packet evidence bullets."""

    if not source_refs:
        return_value = '- none'
        return return_value

    lines: list[str] = []
    for source_ref in source_refs:
        source_kind = source_ref.get('source_kind', '')
        due_at = source_ref.get('due_at')
        summary = source_ref.get('summary', '')
        lines.append(f'- {source_kind}')
        if due_at:
            lines.append(f'  due_at: {due_at}')
        if summary:
            lines.append(f'  summary: {summary}')
    rendered = '\n'.join(lines)
    return rendered


def _render_visible_context(
    rows: list[models.SelfCognitionVisibleContextRow],
) -> str:
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
        speaker = display_name or role
        lines.append(f'- {timestamp} {speaker}: {body_text}')
    rendered = '\n'.join(lines)
    return rendered


def _render_conversation_progress(
    conversation_progress: dict[str, Any] | object,
) -> dict[str, Any] | object:
    """Return the canonical current-user continuity projection unchanged."""

    return_value = conversation_progress
    return return_value


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
