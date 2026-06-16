"""Post-episode recorder for compact first-person residue rows."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from uuid import uuid4

from json_repair import repair_json
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.config import (

    CHARACTER_GLOBAL_USER_ID,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
    INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT,
    COGNITION_LLM_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.internal_monologue_residue.loader import (
    RESIDUE_COMPONENT,
    build_scope_key,
)
from kazusa_ai_chatbot.internal_monologue_residue.models import (
    InternalMonologueResidueRow,
    InternalMonologueResidueSourceRef,
    RecorderInput,
    RecorderValidationResult,
    ResidueRecordResult,
    ResidueScopeKind,
    ResidueSourceKind,
)

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
PROMPT_LEAKAGE_MARKERS = (
    "prompt",
    "schema",
    "json",
    "system message",
    "系统消息",
    "提示词",
    "输出格式",
    "输入格式",
    "生成流程",
    "语义表达层",
    "模型",
    "开发",
    "流程泄露",
    "作为语言模型",
)
THIRD_PERSON_SELF_MARKERS = (
    "角色",
)


_RECORDER_PROMPT = '''\
# 我的身份与环境
我的名字：{character_name}
我所在的环境：{ambient_condition}

# 语言政策
- JSON key、结构化字段名、ID、URL、代码、命令和模型标签保持原样。
- 由你新生成的 `residue_text` 必须使用简体中文；需要精确保留的人名、标题、别名或外部原文可保持原语言。
- 不添加翻译、双语复写或括号解释，除非源文本本身已经包含。

# 核心任务
你负责把我刚完成的 `internal_monologue` 压缩成最多一条短期私念残留。
这条残留只回答一个问题：还有什么私人原因会轻微影响我下一次理解？
如果没有这种原因，`residue_text` 写空字符串。

# 证据身份
- `internal_monologue` 是本轮已经生成的第一人称内心解释，是最高优先级证据。
- `incoming_residue_context` 是我此前留下的短期余波窗口，只用于判断延续、缓和或变旧；不要机械复述窗口内容。
- `ambient_evidence_summary` 只说明群聊、私聊、聊天完成后或自我回顾后的场景边界；不要把它写成报告。
- `current_speaker_display_name` 与 `exact_name_candidates` 只是可用称呼提示；缺少准确称呼时可以自然省略。

# 生成步骤
1. 先读取 `internal_monologue`，找出其中仍可能影响后续理解的主观原因。
2. 对照 `incoming_residue_context`，判断这是新原因、旧原因延续，还是旧余波已经被当前理解放轻。
3. 只保留一条最有用的私念；不要总结整段内心，也不要列出所有事实。
4. 若当前只是普通、已接住、已放下或没有持续影响的内容，输出空字符串。
5. 输出前检查：它必须像我私下留给自己的短句，而不是报告、行动计划、长期记忆或可见台词。

# 私念视角契约
- 写成自然第一人称，可以写我为什么仍在意、放松、防备、期待、迟疑或释然。
- 不要用我的名字称呼我自己。
- 若准确名字能让原因更清楚，可以使用 `current_speaker_display_name` 或 `exact_name_candidates`。
- 若关系指代更自然，也可以写对方、那个人、某人、他或她；不要为了命名而写不确定的人名。
- 不写可见回复、行动指令、长期用户事实、提示内容、模型信息、字段说明、格式说明或处理过程。
- 最长 {row_char_limit} 个字符。

# 输入格式
human payload 是以下 JSON：
{{
  "internal_monologue": "本轮已经生成的第一人称内心解释",
  "current_speaker_display_name": "当前说话者显示名，可能为空",
  "exact_name_candidates": ["可安全使用的名字"],
  "ambient_evidence_summary": "不含原始消息正文的场景摘要",
  "incoming_residue_context": "已有短期余波窗口，可能为空"
}}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
  "residue_text": "第一人称简体中文短句；没有值得保留的内容时为空字符串"
}}
'''
_llm_interface = LLInterface()
_recorder_llm = LLInterface()
_recorder_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="COGNITION_LLM",
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
    model=COGNITION_LLM_MODEL,
    temperature=0.25,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=COGNITION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=COGNITION_LLM_THINKING_ENABLED,
    ),
)


async def record_completed_episode_residue(
    *,
    completed_state: Mapping[str, object],
    current_timestamp_utc: str,
) -> ResidueRecordResult:
    """Record compact residue after a completed chat or self-cognition episode.

    Args:
        completed_state: Completed persona/self-cognition state snapshot.
        current_timestamp_utc: Storage UTC timestamp for the residue row.

    Returns:
        Sanitized write/skip status without residue text.
    """

    recorder_input = _build_recorder_input(completed_state)
    if recorder_input is None:
        result = _record_result(
            status="skipped_missing_input",
            source_kind="",
            scope_kind="",
            written=False,
            retry_count=0,
            validation_errors=[],
        )
        await _record_write_event(result)
        return result

    first_payload, first_latency_ms = await _call_recorder(
        recorder_input,
        repair_reason="",
    )
    first_validation = validate_recorder_output(
        first_payload,
        row_char_limit=INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT,
    )
    raw_first_output = first_payload.get("residue_text")
    if isinstance(raw_first_output, str):
        first_output = raw_first_output.strip()
    else:
        first_output = ""
    if first_validation["status"] == "empty":
        result = _record_result(
            status="empty_no_write",
            source_kind=recorder_input["source_kind"],
            scope_kind="",
            written=False,
            retry_count=0,
            validation_errors=[],
        )
        await _record_llm_event(
            status="empty",
            output_chars=0,
            retry_count=0,
            latency_ms=first_latency_ms,
        )
        await _record_write_event(result)
        return result

    retry_count = 0
    residue_text = first_output
    validation = first_validation
    if not validation["accepted"]:
        retry_count = 1
        repair_reason = validation["failure_reason"]
        retry_payload, retry_latency_ms = await _call_recorder(
            recorder_input,
            repair_reason=repair_reason,
        )
        raw_retry_output = retry_payload.get("residue_text")
        if isinstance(raw_retry_output, str):
            residue_text = raw_retry_output.strip()
        else:
            residue_text = ""
        validation = validate_recorder_output(
            retry_payload,
            row_char_limit=INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT,
        )
        await _record_llm_event(
            status="retry",
            output_chars=len(residue_text),
            retry_count=retry_count,
            latency_ms=retry_latency_ms,
        )

    if validation["status"] == "empty":
        result = _record_result(
            status="empty_no_write",
            source_kind=recorder_input["source_kind"],
            scope_kind="",
            written=False,
            retry_count=retry_count,
            validation_errors=[],
        )
        await _record_write_event(result)
        return result

    if not validation["accepted"]:
        result = _record_result(
            status="skipped_invalid",
            source_kind=recorder_input["source_kind"],
            scope_kind="",
            written=False,
            retry_count=retry_count,
            validation_errors=[validation["failure_reason"]],
        )
        await _record_write_event(result)
        return result

    row = _build_residue_row(
        completed_state=completed_state,
        residue_text=residue_text,
        current_timestamp_utc=current_timestamp_utc,
        source_kind=recorder_input["source_kind"],
    )
    try:
        await db.insert_internal_monologue_residue_row(row)
    except DatabaseOperationError as exc:
        logger.warning(f"Internal monologue residue write failed: {exc}")
        result = _record_result(
            status="write_failed",
            source_kind=recorder_input["source_kind"],
            scope_kind=row["scope_kind"],
            written=False,
            retry_count=retry_count,
            validation_errors=[],
        )
        await _record_write_event(result)
        return result

    result = _record_result(
        status="written",
        source_kind=recorder_input["source_kind"],
        scope_kind=row["scope_kind"],
        written=True,
        retry_count=retry_count,
        validation_errors=[],
        residue_id=row["residue_id"],
    )
    await _record_llm_event(
        status="succeeded",
        output_chars=len(residue_text),
        retry_count=retry_count,
        latency_ms=first_latency_ms,
    )
    await _record_write_event(result)
    return result


def validate_recorder_output(
    payload: Mapping[str, object],
    *,
    row_char_limit: int,
) -> RecorderValidationResult:
    """Validate one recorder payload without inspecting semantic content."""

    raw_residue_text = payload.get("residue_text")
    if not isinstance(raw_residue_text, str):
        result: RecorderValidationResult = {
            "accepted": False,
            "status": "invalid",
            "failure_reason": "missing_residue_text",
        }
        return result

    text = raw_residue_text.strip()
    if not text:
        result = {
            "accepted": True,
            "status": "empty",
            "failure_reason": "",
        }
        return result

    if len(text) > row_char_limit:
        result = {
            "accepted": False,
            "status": "invalid",
            "failure_reason": "row_char_limit",
        }
        return result

    folded_text = text.casefold()
    for marker in PROMPT_LEAKAGE_MARKERS:
        if marker.casefold() in folded_text:
            result = {
                "accepted": False,
                "status": "invalid",
                "failure_reason": "prompt_process_leakage",
            }
            return result

    for marker in THIRD_PERSON_SELF_MARKERS:
        if marker in text:
            result = {
                "accepted": False,
                "status": "invalid",
                "failure_reason": "third_person_self_reference",
            }
            return result

    result = {
        "accepted": True,
        "status": "accepted",
        "failure_reason": "",
    }
    return result


def render_recorder_system_prompt(
    character_name: str,
    ambient_condition: str,
) -> str:
    """Render the recorder system prompt used for the LLM call."""

    rendered_prompt = _RECORDER_PROMPT.format(
        character_name=character_name,
        ambient_condition=ambient_condition,
        row_char_limit=INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT,
    )
    return rendered_prompt


async def _call_recorder(
    recorder_input: RecorderInput,
    *,
    repair_reason: str,
) -> tuple[dict[str, object], int]:
    """Call the recorder LLM and parse its candidate JSON payload."""

    started_at = time.perf_counter()
    system_content = render_recorder_system_prompt(
        character_name=recorder_input["character_name"],
        ambient_condition=recorder_input["ambient_condition"],
    )
    payload = {
        "internal_monologue": recorder_input["internal_monologue"],
        "current_speaker_display_name": (
            recorder_input["current_speaker_display_name"]
        ),
        "exact_name_candidates": recorder_input["exact_name_candidates"],
        "ambient_evidence_summary": recorder_input["ambient_evidence_summary"],
        "incoming_residue_context": recorder_input["incoming_residue_context"],
    }
    if repair_reason:
        payload["validation_failure"] = repair_reason
        payload["repair_instruction"] = (
            "只修正结构问题；如果没有可保留余波，仍可输出空字符串。"
        )

    response = await _recorder_llm.ainvoke([
        SystemMessage(content=system_content),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_recorder_llm_config)
    parsed = _parse_recorder_json(str(response.content))
    latency_ms = int(
        (time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND
    )
    return_value = (parsed, latency_ms)
    return return_value


def _parse_recorder_json(raw_output: str) -> dict[str, object]:
    """Parse recorder JSON without logging raw model text."""

    stripped = raw_output.strip().strip("```").strip("json")
    try:
        parsed = repair_json(stripped, return_objects=True)
    except ValueError as exc:
        logger.warning(f"Recorder JSON parse failed: {exc}")
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed


def _build_recorder_input(
    completed_state: Mapping[str, object],
) -> RecorderInput | None:
    """Build minimal model-facing recorder input from completed state."""

    internal_monologue = _string_field(completed_state, "internal_monologue")
    if not internal_monologue:
        return None

    source_kind = _source_kind(completed_state)
    if source_kind is None:
        return None

    character_profile = completed_state.get("character_profile")
    if not isinstance(character_profile, Mapping):
        return None

    character_name = _string_field(character_profile, "name")
    if not character_name:
        character_name = "active character"

    recorder_input: RecorderInput = {
        "character_name": character_name,
        "ambient_condition": _ambient_condition(completed_state, source_kind),
        "source_kind": source_kind,
        "internal_monologue": internal_monologue,
        "current_speaker_display_name": _string_field(
            completed_state,
            "user_name",
        ),
        "exact_name_candidates": _exact_name_candidates(
            completed_state,
            character_name,
        ),
        "ambient_evidence_summary": _ambient_evidence_summary(
            completed_state,
            source_kind,
        ),
        "incoming_residue_context": _string_field(
            completed_state,
            "internal_monologue_residue_context",
        ),
    }
    return recorder_input


def _build_residue_row(
    *,
    completed_state: Mapping[str, object],
    residue_text: str,
    current_timestamp_utc: str,
    source_kind: ResidueSourceKind,
) -> InternalMonologueResidueRow:
    """Build the storage row for one accepted residue string."""

    character_profile = completed_state["character_profile"]
    if not isinstance(character_profile, Mapping):
        raise ValueError("completed_state.character_profile must be a mapping")

    character_id = _string_field(character_profile, "global_user_id")
    if not character_id:
        character_id = CHARACTER_GLOBAL_USER_ID
    platform = _string_field(completed_state, "platform")
    platform_channel_id = _string_field(completed_state, "platform_channel_id")
    channel_type = _string_field(completed_state, "channel_type")
    global_user_id = _string_field(completed_state, "global_user_id")
    scope_kind = _scope_kind(
        source_kind=source_kind,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        global_user_id=global_user_id,
    )
    scope_key = build_scope_key(
        character_id=character_id,
        scope_kind=scope_kind,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
    )
    row: InternalMonologueResidueRow = {
        "residue_id": uuid4().hex,
        "character_id": character_id,
        "scope_key": scope_key,
        "scope_kind": scope_kind,
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "global_user_id": global_user_id,
        "residue_text": residue_text,
        "source_kind": source_kind,
        "source_refs": _source_refs(completed_state),
        "created_at": current_timestamp_utc,
    }
    return row


def _source_kind(
    completed_state: Mapping[str, object],
) -> ResidueSourceKind | None:
    """Infer the supported source kind from the completed cognitive episode."""

    episode = completed_state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return None
    trigger_source = episode.get("trigger_source")
    if trigger_source == "user_message":
        return_value: ResidueSourceKind = "chat"
        return return_value
    if trigger_source == "internal_thought":
        return_value = "self_cognition"
        return return_value
    return None


def _scope_kind(
    *,
    source_kind: ResidueSourceKind,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    global_user_id: str,
) -> ResidueScopeKind:
    """Choose the most specific storage scope available for the episode."""

    if global_user_id and platform and platform_channel_id:
        return_value: ResidueScopeKind = "user_thread"
        return return_value
    if channel_type == "group" and platform and platform_channel_id:
        return_value = "group_scene"
        return return_value
    return_value = "character_global"
    return return_value


def _ambient_condition(
    completed_state: Mapping[str, object],
    source_kind: ResidueSourceKind,
) -> str:
    """Build a short system-message condition for the recorder."""

    channel_type = _string_field(completed_state, "channel_type")
    if source_kind == "self_cognition":
        source_label = "自我回顾后的余波"
    else:
        source_label = "聊天完成后的余波"
    if channel_type == "group":
        scene_label = "群聊环境"
    elif channel_type == "private":
        scene_label = "私聊环境"
    else:
        scene_label = "非普通聊天环境"
    condition = f"{source_label}；{scene_label}"
    return condition


def _ambient_evidence_summary(
    completed_state: Mapping[str, object],
    source_kind: ResidueSourceKind,
) -> str:
    """Summarize completed episode state without raw message bodies."""

    fields = [
        f"source_kind={source_kind}",
        f"channel_type={_string_field(completed_state, 'channel_type')}",
        f"logical_stance={_string_field(completed_state, 'logical_stance')}",
        f"character_intent={_string_field(completed_state, 'character_intent')}",
        f"emotional_appraisal={_string_field(completed_state, 'emotional_appraisal')}",
        f"interaction_subtext={_string_field(completed_state, 'interaction_subtext')}",
        f"social_distance={_string_field(completed_state, 'social_distance')}",
        f"relational_dynamic={_string_field(completed_state, 'relational_dynamic')}",
    ]
    final_dialog = completed_state.get("final_dialog")
    has_visible_dialog = isinstance(final_dialog, list) and bool(final_dialog)
    fields.append(f"visible_dialog_selected={has_visible_dialog}")
    summary = "; ".join(fields)
    return summary


def _exact_name_candidates(
    completed_state: Mapping[str, object],
    character_name: str,
) -> list[str]:
    """Return safe display names the recorder may use."""

    names = [character_name]
    user_name = _string_field(completed_state, "user_name")
    if user_name:
        names.append(user_name)
    deduped_names = list(dict.fromkeys(names))
    return deduped_names


def _source_refs(
    completed_state: Mapping[str, object],
) -> list[InternalMonologueResidueSourceRef]:
    """Build sanitized source identifiers without raw text or delivery ids."""

    refs: list[InternalMonologueResidueSourceRef] = []
    episode = completed_state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        episode_id = _string_field(episode, "episode_id")
        if episode_id:
            refs.append({"ref_kind": "cognitive_episode", "ref_id": episode_id})
        origin_metadata = episode.get("origin_metadata")
        if isinstance(origin_metadata, Mapping):
            platform_message_id = _string_field(
                origin_metadata,
                "platform_message_id",
            )
            if platform_message_id:
                refs.append({
                    "ref_kind": "platform_message",
                    "ref_id": platform_message_id,
                })
            active_row_ids = origin_metadata.get("active_turn_conversation_row_ids")
            if isinstance(active_row_ids, list):
                for row_id in active_row_ids:
                    if isinstance(row_id, str) and row_id:
                        refs.append({
                            "ref_kind": "conversation_row",
                            "ref_id": row_id,
                        })
    return refs[:8]


def _string_field(data: Mapping[str, object], field_name: str) -> str:
    """Return a stripped string field from an external mapping."""

    value = data.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _record_result(
    *,
    status: str,
    source_kind: str,
    scope_kind: str,
    written: bool,
    retry_count: int,
    validation_errors: list[str],
    residue_id: str = "",
) -> ResidueRecordResult:
    """Build a sanitized recorder result."""

    result: ResidueRecordResult = {
        "status": status,
        "source_kind": source_kind,
        "scope_kind": scope_kind,
        "written": written,
        "retry_count": retry_count,
        "validation_errors": validation_errors,
    }
    if residue_id:
        result["residue_id"] = residue_id
    return result


async def _record_llm_event(
    *,
    status: str,
    output_chars: int,
    retry_count: int,
    latency_ms: int,
) -> None:
    """Record sanitized recorder LLM telemetry."""

    await event_logging.record_llm_stage_event(
        component=RESIDUE_COMPONENT,
        stage_name="internal_monologue_residue_recorder",
        route_name="COGNITION_LLM",
        model_name=COGNITION_LLM_MODEL,
        status=status,
        prompt_chars=len(_RECORDER_PROMPT),
        output_chars=output_chars,
        parse_status="parsed",
        retry_count=retry_count,
        json_repair_used=False,
        duration_ms=latency_ms,
    )


async def _record_write_event(result: ResidueRecordResult) -> None:
    """Record sanitized write, skip, or retry outcome telemetry."""

    document_ref = result.get("residue_id", "") if result["written"] else ""
    await event_logging.record_database_operation_event(
        component=RESIDUE_COMPONENT,
        collection=db.INTERNAL_MONOLOGUE_RESIDUE_COLLECTION,
        operation_kind="record_completed_episode_residue",
        status=result["status"],
        idempotency_result=(
            f"source:{result['source_kind']};scope:{result['scope_kind']};"
            f"retry:{result['retry_count']}"
        ),
        latency_ms=0,
        document_ref=document_ref,
    )
