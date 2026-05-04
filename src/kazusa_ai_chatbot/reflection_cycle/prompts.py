"""LLM prompt contracts for read-only reflection evaluation."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    DailySynthesisResult,
    PromptBuildResult,
    READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS,
    READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS,
    ReflectionInputSet,
    ReflectionLLMResult,
    ReflectionScopeInput,
)
from kazusa_ai_chatbot.reflection_cycle.projection import (
    build_daily_synthesis_payload,
    build_hourly_reflection_payload,
    build_prompt_result,
    validate_daily_synthesis_output,
    validate_hourly_reflection_output,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


HOURLY_REFLECTION_SYSTEM_PROMPT = '''\
你负责为虚构聊天角色执行只读小时反思评估。

你会收到一个被监控频道中的有消息小时槽。该小时可能同时有用户和角色发言，也可能只有用户发言或只有角色发言。
你的任务是评估角色能从这段交流里学到什么。
该阶段只能评估，不能写入记忆，不能改变设定，不能发送消息。

# 语言政策
- 除结构化枚举值、schema key、participant_ref、scope_ref、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。
- `confidence` 与 `evidence_strength` 必须保持 schema 指定的英文枚举值。

# 核心任务
- `话题概括`: 只总结该范围实际发生的话题，不预测后续发展。
- `参与者观察`: 只使用 `participant_1`、`participant_2` 或 `active_character` 这类抽象标识，不暴露真实身份。
- `回应质量反馈`: 评价角色未来如何更好地接住类似对话。
- `隐私说明`: 标出任何会阻碍未来持久化的隐私或泄漏风险。

# 生成步骤
1. 先阅读 `scope_metadata`，理解频道类型和活动标签。
2. 按时间顺序阅读 `conversation.messages`，只把输入内容作为证据。
3. 分开处理话题回顾、参与者观察、回应质量反馈和隐私说明。
4. 参与者身份必须保持抽象；输出里的 `participant_ref` 只能使用输入中提供的 `speaker_ref` 值。
5. 只返回输出 schema 定义的字段。

# 输入格式
{
  "evaluation_mode": "readonly_hourly_reflection",
  "scope_metadata": {
    "scope_ref": "scope_x",
    "platform": "平台标签",
    "channel_type": "private|group|system|unknown",
    "activity_labels": {
      "message_volume": "描述性标签",
      "assistant_presence": "描述性标签",
      "participant_diversity": "描述性标签",
      "window_span": "描述性标签"
    }
  },
  "conversation": {
    "message_order": "chronological",
    "messages": [
      {
        "role": "user|assistant",
        "speaker_ref": "participant_1|active_character",
        "time_position": "开场|前段|中段|后段|收尾|单条",
        "text": "受限长度的消息文本"
      }
    ]
  },
  "review_questions": ["评估问题"]
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
  "topic_summary": "本 scope 的话题概括",
  "participant_observations": [
    {
      "participant_ref": "participant_1",
      "observation": "基于 transcript 的行为观察",
      "evidence_strength": "low|medium|high"
    }
  ],
  "conversation_quality_feedback": [
    "角色回应质量与未来改进反馈"
  ],
  "privacy_notes": [
    "隐私或泄漏风险；若没有则写无明显风险"
  ],
  "confidence": "low|medium|high"
}
'''
_hourly_reflection_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def run_hourly_reflection_llm(
    *,
    scope_ref: str,
    prompt: PromptBuildResult,
) -> ReflectionLLMResult:
    """Run the configured consolidation LLM for one hourly reflection prompt.

    Args:
        scope_ref: Non-identifying reference for the evaluated scope.
        prompt: Serialized prompt contract and prompt diagnostics.

    Returns:
        Raw and parsed LLM output plus validation warnings.
    """

    response = await _hourly_reflection_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    parsed_output = _parse_reflection_json(raw_output)
    validation_warnings = list(prompt.validation_warnings)
    validation_warnings.extend(validate_hourly_reflection_output(parsed_output))
    result = ReflectionLLMResult(
        scope_ref=scope_ref,
        prompt=prompt,
        raw_output=raw_output,
        parsed_output=parsed_output,
        validation_warnings=validation_warnings,
    )
    return result


def build_hourly_reflection_prompt(
    scope: ReflectionScopeInput,
) -> PromptBuildResult:
    """Build the bounded hourly reflection prompt.

    Args:
        scope: Message-bearing conversation scope to evaluate.

    Returns:
        Prompt text and prompt-budget diagnostics for the hourly LLM call.
    """

    payload = build_hourly_reflection_payload(scope)
    prompt = build_prompt_result(
        system_prompt=HOURLY_REFLECTION_SYSTEM_PROMPT,
        human_payload=payload,
        max_prompt_chars=READONLY_REFLECTION_HOURLY_PROMPT_MAX_CHARS,
    )
    return prompt


DAILY_SYNTHESIS_SYSTEM_PROMPT = '''\
你负责为虚构聊天角色执行只读日汇总评估。

你只会收到单个频道的活跃小时槽，不会收到原始 transcript，也不会收到完整小时反思对象。
你的任务是测试日汇总 prompt 是否能用最少必要信息合并小时反思。
该阶段只能评估，不能写入记忆，不能改变设定，不能发送消息。

# 语言政策
- 除结构化枚举值、schema key、participant_ref、scope_ref、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。
- `confidence` 必须保持 schema 指定的英文枚举值。

# 核心任务
- 合并 `active_hour_slots` 里的重复话题、回应质量模式、隐私风险和局限性。
- 只基于 `active_hour_slots` 与 `channel`，不要推断未提供的原始对话。
- 缺失的小时表示没有可用小时反思数据，不代表没有聊天，也不代表聊天质量好或坏。
- `active_hour_summaries.hour` 必须从 `active_hour_slots.hour` 中逐字复制，不要换算时区，不要改写格式。
- `*_omitted_count` 表示同类小时反思内容因预算被省略，只能作为局限性信号，不要把它当成具体内容。
- 输出用于人工评估日汇总是否有价值，不作为持久化写入合同。

# 生成步骤
1. 先阅读 `window` 与 `channel`，理解覆盖范围、回退状态和频道类型。
2. 按 `active_hour_slots` 的 `hour` 顺序阅读；每个槽代表一个被评估的有消息小时。
3. 忽略缺失的小时，不要为缺失小时补内容。
4. 需要引用小时字段时，只能逐字复制输入里的 `hour` 值。
5. 合并重复出现的话题、回应质量、隐私和局限性模式。
6. 不要编造活跃小时槽中不存在的细节。
7. 只返回输出 schema 定义的字段。

# 输入格式
{
  "evaluation_mode": "readonly_daily_synthesis",
  "window": {
    "requested_start": "ISO timestamp",
    "requested_end": "ISO timestamp",
    "fallback_used": false,
    "fallback_reason": ""
  },
  "channel": {
    "channel_type": "private|group|system|unknown"
  },
  "active_hour_slots": [
    {
      "hour": "UTC hour-start ISO timestamp",
      "topic_summary": "该小时的紧凑话题概括",
      "conversation_quality_feedback": ["紧凑回应质量反馈"],
      "conversation_quality_feedback_omitted_count": 0,
      "privacy_notes": ["紧凑隐私说明"],
      "privacy_notes_omitted_count": 0,
      "validation_warnings": ["紧凑验证警告"],
      "validation_warnings_omitted_count": 0,
      "confidence": "low|medium|high"
    }
  ],
  "review_questions": ["评估问题"]
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
  "day_summary": "本频道窗口内的总体概括",
  "active_hour_summaries": [
    {
      "hour": "exact hour value copied from active_hour_slots",
      "summary": "该活跃小时对日汇总的贡献"
    }
  ],
  "cross_hour_topics": [
    "跨活跃小时的重复或对照话题模式"
  ],
  "conversation_quality_patterns": [
    "对未来对话有用的回应质量模式"
  ],
  "privacy_risks": [
    "未来持久化前必须处理的隐私风险"
  ],
  "synthesis_limitations": [
    "本次日汇总薄弱或不完整的原因"
  ],
  "confidence": "low|medium|high"
}
'''
_daily_synthesis_llm = get_llm(
    temperature=0.2,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


async def run_daily_synthesis_llm(
    *,
    prompt: PromptBuildResult,
) -> DailySynthesisResult:
    """Run the configured consolidation LLM for daily synthesis.

    Args:
        prompt: Serialized prompt contract and prompt diagnostics.

    Returns:
        Raw and parsed LLM output plus validation warnings.
    """

    response = await _daily_synthesis_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    parsed_output = _parse_reflection_json(raw_output)
    validation_warnings = list(prompt.validation_warnings)
    allowed_hours = _daily_prompt_allowed_hours(prompt)
    validation_warnings.extend(
        validate_daily_synthesis_output(
            parsed_output,
            allowed_hours=allowed_hours,
        )
    )
    result = DailySynthesisResult(
        prompt=prompt,
        raw_output=raw_output,
        parsed_output=parsed_output,
        validation_warnings=validation_warnings,
    )
    return result


def build_daily_synthesis_prompt(
    *,
    input_set: ReflectionInputSet,
    channel_scope: ReflectionScopeInput,
    hourly_results: list[ReflectionLLMResult],
) -> PromptBuildResult:
    """Build the bounded daily synthesis prompt.

    Args:
        input_set: Collected read-only hourly scope metadata for one channel.
        channel_scope: Selected channel represented by the hourly results.
        hourly_results: Parsed hourly outputs and validation warnings.

    Returns:
        Prompt text and prompt-budget diagnostics for the daily LLM call.
    """

    payload = build_daily_synthesis_payload(
        input_set=input_set,
        channel_scope=channel_scope,
        hourly_results=hourly_results,
    )
    prompt = build_prompt_result(
        system_prompt=DAILY_SYNTHESIS_SYSTEM_PROMPT,
        human_payload=payload,
        max_prompt_chars=READONLY_REFLECTION_DAILY_PROMPT_MAX_CHARS,
    )
    return prompt


def build_skipped_hourly_result(
    scope: ReflectionScopeInput,
) -> ReflectionLLMResult:
    """Build a prompt-only hourly result when LLM execution is disabled.

    Args:
        scope: Message-bearing conversation scope to project into a prompt.

    Returns:
        Hourly result containing prompt diagnostics and an LLM-skipped marker.
    """

    prompt = build_hourly_reflection_prompt(scope)
    result = ReflectionLLMResult(
        scope_ref=scope.scope_ref,
        prompt=prompt,
        raw_output="",
        parsed_output={},
        validation_warnings=["已跳过 LLM 执行"],
        llm_skipped=True,
    )
    return result


def build_skipped_daily_result(
    *,
    input_set: ReflectionInputSet,
    channel_scope: ReflectionScopeInput,
    hourly_results: list[ReflectionLLMResult],
) -> DailySynthesisResult:
    """Build a prompt-only daily result when LLM execution is disabled.

    Args:
        input_set: Collected read-only hourly scope metadata for one channel.
        channel_scope: Selected channel represented by the hourly results.
        hourly_results: Hourly reflection results to project into synthesis.

    Returns:
        Daily result containing prompt diagnostics and an LLM-skipped marker.
    """

    prompt = build_daily_synthesis_prompt(
        input_set=input_set,
        channel_scope=channel_scope,
        hourly_results=hourly_results,
    )
    result = DailySynthesisResult(
        prompt=prompt,
        raw_output="",
        parsed_output={},
        validation_warnings=["已跳过 LLM 执行"],
        llm_skipped=True,
    )
    return result


def _parse_reflection_json(raw_output: str) -> dict:
    """Parse a reflection LLM response into a JSON object.

    Args:
        raw_output: Raw response content from the LLM.

    Returns:
        Parsed JSON object, or an empty object if the repair helper returns a
        non-object value.
    """

    parsed_output = parse_llm_json_output(raw_output)
    if not isinstance(parsed_output, dict):
        return_value: dict = {}
        return return_value
    return_value = parsed_output
    return return_value


def _daily_prompt_allowed_hours(prompt: PromptBuildResult) -> set[str]:
    """Return exact daily slot hour labels from a rendered prompt payload."""

    active_hour_slots = prompt.human_payload.get("active_hour_slots")
    if not isinstance(active_hour_slots, list):
        return_value: set[str] = set()
        return return_value
    allowed_hours: set[str] = set()
    for slot in active_hour_slots:
        if not isinstance(slot, dict):
            continue
        hour = str(slot.get("hour", "") or "")
        if hour:
            allowed_hours.add(hour)
    return_value = allowed_hours
    return return_value
