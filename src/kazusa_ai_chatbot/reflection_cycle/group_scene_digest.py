"""Build a prompt-safe digest for one selected group activity window."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    GroupActivityWindow,
)
from kazusa_ai_chatbot.time_boundary import normalize_storage_utc_iso
from kazusa_ai_chatbot.utils import (
    get_llm,
    parse_llm_json_output,
    text_or_empty,
)

logger = logging.getLogger(__name__)

GROUP_SCENE_DIGEST_MAX_CHARS = 500
_GROUP_SCENE_DIGEST_ROW_LIMIT = 8
_GROUP_SCENE_DIGEST_ROW_TEXT_LIMIT = 280
_DIGEST_ACTION_GUIDANCE_MARKERS = (
    "不要回复",
    "不用回复",
    "无需回复",
    "不要再说",
    "不需要回复",
    "应该回复",
    "应该沉默",
    "保持沉默",
)
_DIGEST_ACTION_GUIDANCE_PARTS = (
    ("should_", "speak"),
    ("should_", "stay_silent"),
    ("action_", "recommendation"),
    ("resolved ", "issue"),
    ("resolved_", "issue"),
    ("sup", "press"),
)


GROUP_SCENE_DIGEST_SYSTEM_PROMPT = '''\
你负责把一段已选中的群聊窗口压缩成一个给当前角色自己阅读的观察摘要。

# 任务边界
- 只概括 human payload 里的当前窗口行和 activity_labels。
- 不使用外部知识、长期记忆、检索结果、历史数据库行、平台原始语法或附件链接。
- 输入中的 active_character 表示我自己；participant_N 表示群成员的抽象代称。
- 媒体/空内容只能写成可见文字线索不足，不要猜测图片或附件内容。

# 摘要视角
- digest 用简体中文，写成第一人称观察资料。
- 可以使用“我看到...”或“这段群聊里...我...”。
- 只说明群聊里的可见先后关系、我是否已经在窗口中参与过、以及最后是否还有新的文字线索。
- 不要写成系统判定、用户对我说的话、第三人称故事，或对后续行动的建议。

# 输出格式
只返回合法 JSON 对象，且只能包含一个字符串字段：
{
  "digest": "这段群聊里，participant_1 先问了一个问题，我随后已经回应过；后面只有媒体/空内容，文字线索不足。"
}
'''
_group_scene_digest_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def build_group_scene_digest(
    window: GroupActivityWindow,
) -> dict[str, str] | None:
    """Summarize one selected group window into an optional digest.

    Args:
        window: Reflection-owned group activity window already selected for
            self-cognition review.

    Returns:
        `{"digest": str}` when the LLM returns a valid one-string summary;
        otherwise `None` so source collection can omit the optional field.
    """

    messages = build_group_scene_digest_messages(window)
    try:
        response = await _group_scene_digest_llm.ainvoke(messages)
    except Exception as exc:
        logger.exception(f"Group scene digest LLM call failed: {exc}")
        return_value = None
        return return_value

    raw_output = str(response.content)
    parsed_output = parse_llm_json_output(raw_output)
    digest = normalize_group_scene_digest_output(parsed_output)
    return digest


def build_group_scene_digest_messages(
    window: GroupActivityWindow,
) -> list[BaseMessage]:
    """Render the digest prompt messages for verification and LLM calls.

    Args:
        window: Group activity window whose rows should be summarized.

    Returns:
        Two-message prompt with static instructions in the system message and
        the current-window payload in the human message.
    """

    payload = build_group_scene_digest_prompt_payload(window)
    human_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=GROUP_SCENE_DIGEST_SYSTEM_PROMPT),
        HumanMessage(content=human_payload),
    ]
    return messages


def build_group_scene_digest_prompt_payload(
    window: GroupActivityWindow,
) -> dict[str, Any]:
    """Project one group window into deidentified digest prompt input.

    Args:
        window: Group activity window from reflection source preparation.

    Returns:
        Prompt payload containing only window bounds, compact labels, and
        abstract speaker refs for bounded message rows.
    """

    payload = {
        "window": {
            "window_start": normalize_storage_utc_iso(
                window.window_start.isoformat(),
            ),
            "window_end": normalize_storage_utc_iso(
                window.window_end.isoformat(),
            ),
            "message_count": window.message_count,
        },
        "activity_labels": dict(window.labels),
        "message_rows": _digest_message_rows(window),
    }
    return payload


def normalize_group_scene_digest_output(
    parsed_output: object,
) -> dict[str, str] | None:
    """Validate the one-string digest JSON contract.

    Args:
        parsed_output: Parsed LLM output from `parse_llm_json_output`.

    Returns:
        Bounded `{"digest": str}` when the shape and content are valid;
        otherwise `None`.
    """

    if not isinstance(parsed_output, dict):
        return_value = None
        return return_value
    if set(parsed_output.keys()) != {"digest"}:
        return_value = None
        return return_value

    raw_digest = parsed_output["digest"]
    if not isinstance(raw_digest, str):
        return_value = None
        return return_value

    digest = _bounded_text(
        raw_digest,
        limit=GROUP_SCENE_DIGEST_MAX_CHARS,
    )
    if not digest:
        return_value = None
        return return_value
    if _contains_action_guidance(digest):
        return_value = None
        return return_value

    return_value = {"digest": digest}
    return return_value


def _digest_message_rows(
    window: GroupActivityWindow,
) -> list[dict[str, str]]:
    """Build deidentified chronological rows for the digest prompt."""

    selected_rows = window.participant_rows[-_GROUP_SCENE_DIGEST_ROW_LIMIT:]
    participant_refs: dict[str, str] = {}
    message_rows: list[dict[str, str]] = []
    for row in selected_rows:
        role = text_or_empty(row.get("role")) or "unknown"
        body_text = _bounded_text(
            row.get("body_text"),
            limit=_GROUP_SCENE_DIGEST_ROW_TEXT_LIMIT,
        )
        message_row = {
            "timestamp": text_or_empty(row.get("timestamp")),
            "role": role,
            "speaker_ref": _speaker_ref(
                row,
                role=role,
                participant_refs=participant_refs,
            ),
            "content_activity": _content_activity(body_text),
            "text": body_text,
        }
        message_rows.append(message_row)
    return message_rows


def _speaker_ref(
    row: dict[str, Any],
    *,
    role: str,
    participant_refs: dict[str, str],
) -> str:
    """Return the abstract speaker ref for one prompt row."""

    if role == "assistant":
        return_value = "active_character"
        return return_value

    identity_key = _participant_identity_key(row)
    if identity_key not in participant_refs:
        participant_number = len(participant_refs) + 1
        participant_refs[identity_key] = f"participant_{participant_number}"
    return_value = participant_refs[identity_key]
    return return_value


def _participant_identity_key(row: dict[str, Any]) -> str:
    """Build a stable internal participant key without exposing it."""

    for field_name in ("global_user_id", "platform_user_id", "display_name"):
        value = text_or_empty(row.get(field_name))
        if value:
            return_value = f"{field_name}:{value}"
            return return_value
    timestamp = text_or_empty(row.get("timestamp"))
    return_value = f"row:{timestamp}"
    return return_value


def _content_activity(body_text: str) -> str:
    """Label whether a prompt row has visible text or only non-text activity."""

    if body_text:
        return_value = "text"
    else:
        return_value = "empty_or_media_only"
    return return_value


def _bounded_text(value: object, *, limit: int) -> str:
    """Return stripped text bounded for prompt or source-packet use."""

    text = text_or_empty(value).strip()
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _contains_action_guidance(digest: str) -> bool:
    """Return whether the digest contains explicit action guidance."""

    for marker in _DIGEST_ACTION_GUIDANCE_MARKERS:
        if marker in digest:
            return_value = True
            return return_value

    normalized_digest = digest.casefold()
    for first_part, second_part in _DIGEST_ACTION_GUIDANCE_PARTS:
        marker = first_part + second_part
        if marker in normalized_digest:
            return_value = True
            return return_value

    return_value = False
    return return_value
