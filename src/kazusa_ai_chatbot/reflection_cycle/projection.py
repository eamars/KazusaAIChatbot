"""Prompt projection helpers for read-only reflection evaluation."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.reflection_cycle.models import (
    DAILY_REQUIRED_FIELDS,
    HOURLY_REQUIRED_FIELDS,
    READONLY_REFLECTION_ARTIFACT_PROMPT_PREVIEW_CHARS,
    READONLY_REFLECTION_DAILY_SLOT_TEXT_CHARS,
    READONLY_REFLECTION_MAX_MESSAGE_CHARS,
    PromptBuildResult,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)


_FORWARD_LOOKING_FIELDS = {"lore_candidates", "progress_projection", "open_loops"}


def build_hourly_reflection_payload(scope: ReflectionScopeInput) -> dict[str, Any]:
    """Build the JSON payload for one hourly reflection prompt.

    Args:
        scope: Message-bearing conversation scope with bounded transcript rows.

    Returns:
        JSON-serializable prompt payload with deidentified speaker references.
    """

    payload = {
        "evaluation_mode": "readonly_hourly_reflection",
        "scope_metadata": _scope_metadata_for_prompt(scope),
        "conversation": {
            "message_order": "chronological",
            "messages": _project_messages_for_prompt(scope.messages),
        },
        "review_questions": [
            "这个范围实际覆盖了哪些话题？",
            "参与者行为显示了什么，但不能暴露私人身份？",
            "角色未来在类似范围里可以如何改进回应？",
            "哪些隐私风险会阻碍未来持久化？",
        ],
    }
    return payload


def build_daily_synthesis_payload(
    *,
    input_set: ReflectionInputSet,
    channel_scope: ReflectionScopeInput,
    hourly_results: list[ReflectionLLMResult],
) -> dict[str, Any]:
    """Build the compact JSON payload for one channel's daily synthesis.

    Args:
        input_set: Read-only input collection and scope metadata.
        channel_scope: Selected channel represented by the hourly results.
        hourly_results: Parsed hourly reflection outputs and validation state.

    Returns:
        JSON-serializable prompt payload that excludes raw transcript rows and
        verbose hourly objects.
    """

    payload = {
        "evaluation_mode": "readonly_daily_synthesis",
        "window": {
            "requested_start": input_set.requested_start,
            "requested_end": input_set.requested_end,
            "fallback_used": input_set.fallback_used,
            "fallback_reason": input_set.fallback_reason,
        },
        "channel": {
            "channel_type": channel_scope.channel_type,
        },
        "active_hour_slots": _active_hour_slots_for_daily_prompt(
            hourly_scopes=input_set.selected_scopes,
            hourly_results=hourly_results,
        ),
        "review_questions": [
            "当天哪些活跃小时形成了连续或重复话题？",
            "哪些回应质量经验对未来对话有用？",
            "哪些隐私限制会阻碍未来持久化？",
            "只基于这些活跃小时槽时，本次日汇总的可靠性如何？",
        ],
    }
    return payload


def validate_hourly_reflection_output(parsed_output: dict[str, Any]) -> list[str]:
    """Return schema warnings for parsed hourly reflection output.

    Args:
        parsed_output: JSON object parsed from the LLM response.

    Returns:
        Validation warnings for missing fields or unsupported output fields.
    """

    warnings = _validate_required_fields(parsed_output, HOURLY_REQUIRED_FIELDS)
    warnings.extend(_validate_confidence(parsed_output))
    for field_name in _FORWARD_LOOKING_FIELDS:
        if field_name in parsed_output:
            warnings.append(f"出现未请求的前瞻字段: {field_name}")
    return_value = warnings
    return return_value


def validate_daily_synthesis_output(
    parsed_output: dict[str, Any],
    allowed_hours: set[str] | None = None,
) -> list[str]:
    """Return schema warnings for parsed daily synthesis output.

    Args:
        parsed_output: JSON object parsed from the LLM response.
        allowed_hours: Optional exact hour labels allowed in summaries.

    Returns:
        Validation warnings for missing fields or unsupported output fields.
    """

    warnings = _validate_required_fields(parsed_output, DAILY_REQUIRED_FIELDS)
    warnings.extend(_validate_confidence(parsed_output))
    if allowed_hours is not None:
        warnings.extend(_validate_daily_summary_hours(parsed_output, allowed_hours))
    for field_name in _FORWARD_LOOKING_FIELDS:
        if field_name in parsed_output:
            warnings.append(f"出现未请求的前瞻字段: {field_name}")
    return_value = warnings
    return return_value


def _scope_metadata_for_prompt(scope: ReflectionScopeInput) -> dict[str, Any]:
    """Project scope counters into descriptive labels for the LLM."""

    metadata = {
        "scope_ref": scope.scope_ref,
        "platform": scope.platform,
        "channel_type": scope.channel_type,
        "activity_labels": {
            "message_volume": _message_volume_label(scope.total_message_count),
            "assistant_presence": _assistant_presence_label(
                assistant_messages=scope.assistant_message_count,
                user_messages=scope.user_message_count,
            ),
            "participant_diversity": _participant_diversity_label(scope.messages),
            "window_span": _window_span_label(
                first_timestamp=scope.first_timestamp,
                last_timestamp=scope.last_timestamp,
            ),
        },
    }
    return metadata


def _active_hour_slots_for_daily_prompt(
    *,
    hourly_scopes: list[ReflectionScopeInput],
    hourly_results: list[ReflectionLLMResult],
) -> list[dict[str, Any]]:
    """Project hourly results into compact active-hour daily slots."""

    scope_by_ref = {
        scope.scope_ref: scope
        for scope in hourly_scopes
    }
    slots: list[dict[str, Any]] = []
    for result in hourly_results:
        scope = scope_by_ref[result.scope_ref]
        slot = _daily_slot_from_hourly_result(
            scope=scope,
            result=result,
        )
        slots.append(slot)
    return_value = slots
    return return_value


def _daily_slot_from_hourly_result(
    *,
    scope: ReflectionScopeInput,
    result: ReflectionLLMResult,
) -> dict[str, Any]:
    """Build one compact daily-facing active-hour slot."""

    parsed_output = result.parsed_output
    slot: dict[str, Any] = {
        "hour": _hour_start_label(scope.first_timestamp),
    }
    topic_summary = parsed_output.get("topic_summary")
    if topic_summary:
        slot["topic_summary"] = _trim_text(
            str(topic_summary),
            max_chars=READONLY_REFLECTION_DAILY_SLOT_TEXT_CHARS,
        )
    quality_feedback, quality_feedback_omitted_count = _compact_text_list(
        parsed_output.get("conversation_quality_feedback"),
    )
    if quality_feedback:
        slot["conversation_quality_feedback"] = quality_feedback
    if quality_feedback_omitted_count:
        slot["conversation_quality_feedback_omitted_count"] = (
            quality_feedback_omitted_count
        )
    privacy_notes, privacy_notes_omitted_count = _compact_text_list(
        parsed_output.get("privacy_notes")
    )
    if privacy_notes:
        slot["privacy_notes"] = privacy_notes
    if privacy_notes_omitted_count:
        slot["privacy_notes_omitted_count"] = privacy_notes_omitted_count
    confidence = parsed_output.get("confidence")
    if confidence:
        slot["confidence"] = str(confidence)
    if result.validation_warnings:
        validation_warnings, validation_warnings_omitted_count = _compact_text_list(
            result.validation_warnings,
        )
        slot["validation_warnings"] = validation_warnings
        if validation_warnings_omitted_count:
            slot["validation_warnings_omitted_count"] = (
                validation_warnings_omitted_count
            )
    if result.llm_skipped:
        slot["llm_status"] = "skipped"
    return_value = slot
    return return_value


def _compact_text_list(value: Any) -> tuple[list[str], int]:
    """Return the lead compact text item and omitted source item count."""

    if isinstance(value, list):
        raw_items = value
    elif value:
        raw_items = [value]
    else:
        raw_items = []
    selected_items = raw_items[:1]
    compact_items = [
        _trim_text(
            str(item),
            max_chars=READONLY_REFLECTION_DAILY_SLOT_TEXT_CHARS,
        )
        for item in selected_items
    ]
    omitted_count = max(0, len(raw_items) - len(selected_items))
    return_value = compact_items, omitted_count
    return return_value


def _hour_start_label(timestamp: str) -> str:
    """Return a UTC ISO hour-start label for a daily active-hour slot."""

    parsed_timestamp = datetime.fromisoformat(
        str(timestamp).replace("Z", "+00:00")
    )
    hour_start = parsed_timestamp.astimezone(timezone.utc).replace(
        minute=0,
        second=0,
        microsecond=0,
    )
    return_value = hour_start.isoformat()
    return return_value


def _project_messages_for_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project raw conversation rows into bounded prompt messages."""

    participant_refs = _build_participant_refs(messages)
    total_messages = len(messages)
    projected_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        projected_message = {
            "role": role,
            "speaker_ref": _speaker_ref(message, participant_refs),
            "time_position": _time_position_label(index, total_messages),
            "text": _trim_text(str(message.get("body_text", ""))),
        }
        attachment_context = _attachment_context(message)
        if attachment_context:
            projected_message["attachment_context"] = attachment_context
        projected_messages.append(projected_message)
    return projected_messages


def _build_participant_refs(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Build stable participant aliases for one prompt scope."""

    refs: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "user":
            continue
        key = _participant_key(message)
        if key in refs:
            continue
        refs[key] = f"participant_{len(refs) + 1}"
    return refs


def _participant_key(message: dict[str, Any]) -> str:
    """Return a stable key for a user row without exposing it to the prompt."""

    for field_name in ("global_user_id", "platform_user_id", "display_name"):
        value = str(message.get(field_name, "") or "").strip()
        if value:
            return_value = f"{field_name}:{value}"
            return return_value
    return_value = "anonymous_user"
    return return_value


def _speaker_ref(message: dict[str, Any], participant_refs: dict[str, str]) -> str:
    """Return the prompt-facing speaker reference for one message."""

    if message.get("role") == "assistant":
        return_value = "active_character"
        return return_value
    key = _participant_key(message)
    return_value = participant_refs.get(key, "participant_unknown")
    return return_value


def _attachment_context(message: dict[str, Any]) -> list[str]:
    """Return bounded attachment descriptions from a conversation row."""

    attachments = message.get("attachments")
    if not isinstance(attachments, list):
        return_value: list[str] = []
        return return_value
    descriptions: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        description = str(attachment.get("description", "") or "").strip()
        if description:
            descriptions.append(_trim_text(description, max_chars=160))
    return_value = descriptions[:3]
    return return_value


def build_prompt_result(
    *,
    system_prompt: str,
    human_payload: dict[str, Any],
    max_prompt_chars: int,
) -> PromptBuildResult:
    """Serialize a prompt payload and shrink message text if needed.

    Args:
        system_prompt: Instruction prompt supplied to the LLM.
        human_payload: JSON payload supplied as the human message.
        max_prompt_chars: Maximum combined system and human prompt characters.

    Returns:
        Serialized prompt text with prompt-budget diagnostics.
    """

    warnings: list[str] = []
    payload = copy.deepcopy(human_payload)
    human_prompt = _serialize_payload(payload)
    prompt_chars = len(system_prompt) + len(human_prompt)
    if prompt_chars > max_prompt_chars:
        warnings.append(
            f"Prompt 超出预算，已尝试截断: {prompt_chars} > {max_prompt_chars}"
        )
        payload = _truncate_prompt_payload(payload, max_prompt_chars, system_prompt)
        human_prompt = _serialize_payload(payload)
        prompt_chars = len(system_prompt) + len(human_prompt)
    if prompt_chars > max_prompt_chars:
        warnings.append(
            f"Prompt 截断后仍超出预算: {prompt_chars} > {max_prompt_chars}"
        )

    prompt_preview = (system_prompt + "\n" + human_prompt)[
        :READONLY_REFLECTION_ARTIFACT_PROMPT_PREVIEW_CHARS
    ]
    result = PromptBuildResult(
        system_prompt=system_prompt,
        human_payload=payload,
        human_prompt=human_prompt,
        prompt_chars=prompt_chars,
        prompt_preview=prompt_preview,
        validation_warnings=warnings,
    )
    return result


def _truncate_prompt_payload(
    payload: dict[str, Any],
    max_prompt_chars: int,
    system_prompt: str,
) -> dict[str, Any]:
    """Reduce the prompt payload until it fits the configured budget."""

    truncated = copy.deepcopy(payload)
    conversation = truncated.get("conversation")
    messages: list[Any] = []
    if isinstance(conversation, dict):
        raw_messages = conversation.get("messages")
        if isinstance(raw_messages, list):
            messages = raw_messages
    if isinstance(messages, list):
        while (
            len(system_prompt) + len(_serialize_payload(truncated))
            > max_prompt_chars
            and len(messages) > 8
        ):
            del messages[0]
        prompt_chars = len(system_prompt) + len(_serialize_payload(truncated))
        if prompt_chars > max_prompt_chars:
            for message in messages:
                if not isinstance(message, dict):
                    continue
                message["text"] = _trim_text(
                    str(message.get("text", "")),
                    max_chars=120,
                )
    return_value = truncated
    return return_value


def _serialize_payload(payload: dict[str, Any]) -> str:
    """Serialize a prompt payload in stable readable JSON."""

    return_value = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return return_value


def _trim_text(
    value: str,
    max_chars: int = READONLY_REFLECTION_MAX_MESSAGE_CHARS,
) -> str:
    """Trim one prompt-facing text field to a bounded character count."""

    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return_value = cleaned
        return return_value
    return_value = f"{cleaned[:max_chars - 3]}..."
    return return_value


def _message_volume_label(total_message_count: int) -> str:
    """Convert a raw message total into a descriptive prompt label."""

    if total_message_count <= 8:
        return_value = "短对话"
    elif total_message_count <= 30:
        return_value = "中等长度对话"
    elif total_message_count <= 80:
        return_value = "活跃对话"
    else:
        return_value = "高密度对话"
    return return_value


def _assistant_presence_label(*, assistant_messages: int, user_messages: int) -> str:
    """Convert assistant/user activity into a descriptive label."""

    if assistant_messages <= 0:
        return_value = "无角色参与"
    elif assistant_messages >= user_messages:
        return_value = "角色参与度高"
    elif assistant_messages * 3 >= user_messages:
        return_value = "角色参与度均衡"
    else:
        return_value = "角色参与度轻"
    return return_value


def _participant_diversity_label(messages: list[dict[str, Any]]) -> str:
    """Convert unique user participation into a descriptive label."""

    participant_keys = {
        _participant_key(message)
        for message in messages
        if message.get("role") == "user"
    }
    if len(participant_keys) <= 1:
        return_value = "单用户互动"
    elif len(participant_keys) <= 4:
        return_value = "小范围多用户互动"
    else:
        return_value = "广泛多用户互动"
    return return_value


def _window_span_label(*, first_timestamp: str, last_timestamp: str) -> str:
    """Return a coarse time-span label without exposing raw metric counts."""

    if (
        not first_timestamp
        or not last_timestamp
        or first_timestamp == last_timestamp
    ):
        return_value = "单次集中互动"
    else:
        return_value = "多轮时间窗口"
    return return_value


def _time_position_label(index: int, total_messages: int) -> str:
    """Return a coarse chronological position label for one message."""

    if total_messages <= 1:
        return_value = "单条"
    elif index == 0:
        return_value = "开场"
    elif index == total_messages - 1:
        return_value = "收尾"
    else:
        ratio = index / max(1, total_messages - 1)
        if ratio < 0.34:
            return_value = "前段"
        elif ratio < 0.67:
            return_value = "中段"
        else:
            return_value = "后段"
    return return_value


def _validate_required_fields(
    parsed_output: dict[str, Any],
    required_fields: tuple[str, ...],
) -> list[str]:
    """Validate that parsed output contains required schema fields."""

    warnings: list[str] = []
    if not isinstance(parsed_output, dict) or not parsed_output:
        warnings.append("解析结果为空或不是 JSON 对象")
        return warnings
    for field_name in required_fields:
        if field_name not in parsed_output:
            warnings.append(f"缺少必需字段: {field_name}")
    return_value = warnings
    return return_value


def _validate_confidence(parsed_output: dict[str, Any]) -> list[str]:
    """Validate the shared confidence field."""

    confidence = str(parsed_output.get("confidence", "") or "").lower()
    if confidence not in {"low", "medium", "high"}:
        return_value = ["`confidence` 必须是 low、medium 或 high"]
        return return_value
    return_value: list[str] = []
    return return_value


def _validate_daily_summary_hours(
    parsed_output: dict[str, Any],
    allowed_hours: set[str],
) -> list[str]:
    """Validate daily summaries copy hour labels from the input slots."""

    warnings: list[str] = []
    active_hour_summaries = parsed_output.get("active_hour_summaries")
    if not isinstance(active_hour_summaries, list):
        return_value: list[str] = []
        return return_value
    for summary in active_hour_summaries:
        if not isinstance(summary, dict):
            continue
        hour = str(summary.get("hour", "") or "")
        if hour and hour not in allowed_hours:
            warnings.append(f"active_hour_summaries.hour 未逐字复制输入 hour: {hour}")
    return_value = warnings
    return return_value
