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
    CONVERSATION_HISTORY_LIMIT,
)
from kazusa_ai_chatbot.conversation_history_prompt_projection import (
    project_conversation_history_for_llm,
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
GROUP_SCENE_SUMMARY_MAX_CHARS = 160
_GROUP_SCENE_DIGEST_ROW_LIMIT = CONVERSATION_HISTORY_LIMIT
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
你负责把一段已选中的群聊现场压缩成当前角色自己阅读的观察资料。

# 输入含义
- 信息来源限定为 human payload 里的 message_lines 和 activity_labels。
- message_lines 是按时间顺序排列的聊天记录行，每行格式为 `[时间] 说话人: 内容`。
- message_lines 可能比当前 15 分钟窗口更宽，用来补足话题承接关系；不要把更早的行误认为当前窗口内的新发言。
- activity_labels.assistant_presence 表示窗口里是否有当前角色自己的发言。
- 当前角色自己的发言行，摘要中写成"我（说话人）"。
- 行内的 @名称 属于该行可见文字，表示该行发言者艾特或提到了这个名称。

# 摘要视角
- digest 用简体中文，写成第一人称观察资料。
- 说明群聊里的可见先后关系、被概括的发言者名称、我是否已经在窗口中参与过、以及我最后发言后是否还有新的文字线索。
- 再次提到发言者时继续使用其名称，让后续 cognition 能直接看出谁在说话。
- summary 用简体中文，写成很短的群聊话题概述，只描述主要参与者、话题和指向关系；不要写行动建议。

# 生成步骤
1. 按 message_lines 的顺序阅读，保持这个顺序。
2. 对需要概括的用户行，写说话人问、说或提到的可见内容；如果行内有 @名称，把 @名称 保留为该行文字的一部分。
3. 把当前角色自己的行当作原文引用处理，优先写成：我（说话人）说："内容"。
4. 引用内容只从该行复制或轻微压缩；引用里的 `你`、`我`、称呼和语气词保持原样。
5. 说话人放在引用外标记这行是谁说的；当前角色行的被回复者只来自行内原文。
6. 窗口里有当前角色自己的行时，概括每条的可见文字内容。
7. 最后补一句当前角色发言后的状态，先看 activity_labels.assistant_presence，再看 message_lines：
   - assistant_presence="present" 且最后一条角色行后还有用户行：写"我最后发言后，说话人 说/问/提到：..."。
   - assistant_presence="present" 且最后一行是角色行：写"我最后发言后没有新的文字线索"。
   - assistant_presence 不是 "present"：写“我没有在这个窗口中发言”。

# 输出格式
输出为一个 JSON 对象，包含必填字符串字段 digest，以及可选字符串字段 summary：
{
  "digest": "一段简体中文第一人称观察摘要",
  "summary": "一句很短的群聊话题概述"
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
        `{"digest": str}` plus an optional ``summary`` when the
        LLM returns valid strings; otherwise `None` so source collection can
        omit the optional field.
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
    """Project one group window into explicit-name digest prompt input.

    Args:
        window: Group activity window from reflection source preparation.

    Returns:
        Prompt payload containing only window bounds, compact labels, and
        visible display names for bounded message rows.
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
        "message_lines": _digest_message_lines(window),
    }
    return payload


def normalize_group_scene_digest_output(
    parsed_output: object,
) -> dict[str, str] | None:
    """Validate the digest JSON contract.

    Args:
        parsed_output: Parsed LLM output from `parse_llm_json_output`.

    Returns:
        Bounded `{"digest": str}` with an optional summary string when the
        shape and content are valid; otherwise `None`.
    """

    if not isinstance(parsed_output, dict):
        return_value = None
        return return_value
    allowed_keys = {"digest", "summary"}
    output_keys = set(parsed_output.keys())
    if "digest" not in output_keys or not output_keys.issubset(allowed_keys):
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
    if "summary" in parsed_output:
        raw_summary = parsed_output["summary"]
        if not isinstance(raw_summary, str):
            return return_value
        summary = _bounded_text(
            raw_summary,
            limit=GROUP_SCENE_SUMMARY_MAX_CHARS,
        )
        if summary:
            if _contains_action_guidance(summary):
                return return_value
            return_value["summary"] = summary
    return return_value


def _digest_message_lines(
    window: GroupActivityWindow,
) -> list[str]:
    """Build transcript lines for the digest prompt."""

    selected_rows = (
        window.digest_participant_rows[-_GROUP_SCENE_DIGEST_ROW_LIMIT:]
    )
    lines = project_conversation_history_for_llm(
        selected_rows,
        character_name='当前角色',
    )
    bounded_lines = [
        line[:_GROUP_SCENE_DIGEST_ROW_TEXT_LIMIT].rstrip()
        if len(line) > _GROUP_SCENE_DIGEST_ROW_TEXT_LIMIT
        else line
        for line in lines
    ]
    return bounded_lines


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
