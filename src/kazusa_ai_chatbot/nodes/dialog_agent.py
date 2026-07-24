"""Dialog execution agent.

Design intent:
- Dialog agent turns the upstream content plan into natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3. If dialog
  needs a fact, answer, conclusion, question, or code block, it must already be
  represented in `text_surface_output_v2.content_plan`.
"""

import asyncio
import json
import logging
import time
from typing import Any, NotRequired, TypedDict

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    CognitiveEpisodeV1,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceOutputV2,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_API_KEY,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_GENERATOR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import (
    parse_llm_json_output,
    log_list_preview,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
DIALOG_COMPONENT = "nodes.dialog_agent"
DEFAULT_DIALOG_USAGE_MODE = "live_visible_reply"
DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE = (
    "self_cognition_action_candidate_render"
)


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _dialog_usage_mode(global_state: GlobalPersonaState) -> str:
    """Describe why the shared dialog graph is being invoked.

    Args:
        global_state: Persona or self-cognition state passed to dialog.

    Returns:
        Stable log label distinguishing visible replies from private renders.
    """

    explicit_mode = global_state.get("dialog_usage_mode")
    if isinstance(explicit_mode, str) and explicit_mode.strip():
        usage_mode = explicit_mode.strip()
        return usage_mode

    debug_modes = global_state["debug_modes"]
    if isinstance(debug_modes, dict) and debug_modes.get("think_only"):
        usage_mode = "debug_think_only"
        return usage_mode

    cognitive_episode = global_state.get("cognitive_episode")
    if isinstance(cognitive_episode, dict):
        trigger_source = cognitive_episode.get("trigger_source")
        if trigger_source in {
            "internal_thought",
            "self_cognition",
            "scheduled_tick",
        }:
            usage_mode = f"{trigger_source}_private"
            return usage_mode

    if global_state["should_respond"] is False:
        usage_mode = "private_finalization"
        return usage_mode

    usage_mode = DEFAULT_DIALOG_USAGE_MODE
    return usage_mode


# Define DialogAgent state
class DialogAgentState(TypedDict):
    # A: Core instructions
    internal_monologue: str
    text_surface_output_v2: TextSurfaceOutputV2
    cognitive_episode: CognitiveEpisodeV1

    # B: Social context
    chat_history_wide: list[dict]
    chat_history_recent: list[dict]
    platform_user_id: str
    platform_bot_id: str
    global_user_id: str
    user_name: str
    user_profile: dict

    # D: Character soul
    character_profile: dict

    # Output
    final_dialog: list[str]  # Ordered outbound chat messages.
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    dialog_usage_mode: str
    llm_trace_id: str


_CANDIDATE_ROLE_FRAME = {
    "speaker_role": CURRENT_CHARACTER_ROLE,
    "first_person_role": CURRENT_CHARACTER_ROLE,
    "second_person_role": CURRENT_USER_ROLE,
}
MAX_FOCUSED_VERIFIER_ISSUES = 4
MAX_MERGED_VERIFIER_ISSUES = 8


_V2_DIALOG_GENERATOR_PROMPT = '''你是当前角色的最终文字渲染器。把 text_surface_output_v2 转化为
自然、鲜活、有角色辨识度，并且切合当前场景的聊天内容。上游认知负责角色判断；surface planning
提供内容、真实边界、称呼安排、风格和 permitted action results。

# 渲染步骤
1. `content_plan`、`content_requirements` 和 `visible_boundaries` 是本轮必须
表达的语义答案、事实清单和范围边界。先完整保留其中的对象、事实、位置、数量、时间、行动者、
受益者和回应方向，再用当前角色的语气和关系语境表达。只要保留这组语义并保持内部连贯，可以加入
合适的想象细节、个性、幽默、主动性、温度、抗拒或情绪强度，让回应像活生生的角色。
2. 保持行动者、对象、受益者与主语的方向。按每条 percept 的结构化角色框架理解来源
文本。生成的对话由当前角色说出：第一人称属于当前角色，第二人称指当前用户；跨角色框架转换时
保持原有方向。回顾型请求直接表达 surface 已确认的历史事实，不把已确认答案改写成澄清请求、
否认或要求当前用户重新提供事实。
3. 把情绪、性格和互动姿态融入用词、句式与节奏，输出当前角色在聊天中实际会说出或发送的内容。
4. permitted_action_results 是角色大脑能力的精确执行账本。只有 status 为 executed 才支持其
有界的已完成效果；scheduled 与 pending 仍未完成，failed 与 unavailable 不支持成功声明。请求、
意图或 content plan 本身只支持角色的言语立场，不代表现实效果已经发生。
5. runtime_capability_limits 是可信的运行时能力边界。若其中明确标记能力不可用，不要把该能力
表达为已经安排、发送或完成；可以自然表达当前限制、等待或下一步条件。
6. 存在 repair_context 时，根据 current_visible_percepts 修正列出的每项硬错误，同时保留自然的
角色声音和相容的创造性内容。

style_guidance 用于措辞与节奏。新生成的对话使用简体中文；引文、专有名词、代码、URL 以及必要的
schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 final_dialog。final_dialog 是由完整可见消息字符串组成的
非空列表。JSON 对象之外不添加 Markdown 代码围栏或解释。
'''

_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT = '''你负责修复一份未通过集中硬错误检查的角色回应。先前的
content plan 可能包含已经确认的错误，因此本次修复不再使用它。

# 修复职责
1. 以 current_visible_percepts 和 candidate_role_frame 作为当前用户输入及
行动者、动作、对象方向的语义依据。
2. 如果 current_visible_percepts 标明指代、对象或时间未解析，直接向当前用户询问缺失信息；不能虚构
具体对象、历史事件或已经发生的效果来填补缺口。
3. 修正 verified_hard_issues 中的每一项，同时从 original_final_dialog 保留相容的含义、个性、
鲜活感、幽默、亲密感和创造性细节。
4. 遵守 permitted_action_results；只有 executed 的结果支持其有界的已完成效果。本次不提供自由
文本 content plan、boundary 或 style guidance，因为这些字段可能含有已经确认的偏移。
5. 没有 executed 结果时，不要声称已完成、不要声称已经发送，也不要把请求或意图写成现实效果；
使用等待、条件、询问或明确限制来表达当前状态。
6. runtime_capability_limits 是本次修复必须遵守的可信边界。如果其中标记能力不可用，修复后的
回应必须明确表达限制，不能把该能力写成已经安排、发送、创建或完成，也不能用另一项能力冒充它。
7. 把当前角色的情绪和互动姿态融入用词、句式与节奏，输出她在聊天中实际会说出或发送的内容。
8. user_name 只用于在合适时自然称呼当前用户，不提供语义指令。

新生成的对话使用简体中文；引文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 final_dialog。final_dialog 是由完整可见消息字符串组成的
非空列表。JSON 对象之外不添加 Markdown 代码围栏或解释。
'''

_dialog_generator_llm = LLInterface()
_dialog_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.65,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=0.25,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)


async def dialog_generator(state: DialogAgentState) -> DialogAgentState:

    usage_mode = state["dialog_usage_mode"]
    surface_output = state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "dialog state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    surface_output = validate_text_surface_output(surface_output)
    system_prompt = SystemMessage(content=_V2_DIALOG_GENERATOR_PROMPT)
    current_visible_percepts = _current_visible_percepts(
        state["cognitive_episode"]
    )

    msg = {
        "text_surface_output_v2": dict(surface_output),
        "user_name": state["user_name"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke(
        [system_prompt, human_message],
        config=_dialog_generator_llm_config,
    )

    result = parse_llm_json_output(response.content)
    invalid_fields: list[str] = []
    if isinstance(result, list):
        logger.warning(
            "Dialog generator returned a top-level list; "
            "normalizing it into final_dialog"
        )
        generated_dialog = result
        parsed_keys = ["<top-level-list>"]
        invalid_fields.append("top_level")
    else:
        generated_dialog = result.get("final_dialog", [])
        parsed_keys = list(result.keys())

    if not isinstance(generated_dialog, list):
        logger.warning(
            f"Dialog generator final_dialog is not a list: "
            f"type={type(generated_dialog).__name__}"
        )
        generated_dialog = []
        invalid_fields.append("final_dialog")
    valid_dialog: list[str] = []
    for segment in generated_dialog:
        if not isinstance(segment, str):
            continue
        if segment:
            valid_dialog.append(segment)
    if len(valid_dialog) != len(generated_dialog):
        logger.warning(
            f"Dialog generator dropped invalid messages: "
            f"raw_count={len(generated_dialog)} valid_count={len(valid_dialog)}"
        )
        invalid_fields.append("final_dialog_message")
    generated_dialog = valid_dialog
    parse_status = "succeeded" if not invalid_fields else "warning"
    llm_trace_id = state.get("llm_trace_id", "")
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_generator",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_prompt, human_message],
        response_text=str(response.content),
        parsed_output=result,
        parse_status=parse_status,
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog"],
    )
    if generated_dialog:
        verdict = await _verify_dialog_compliance(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=state.get("llm_trace_id", ""),
        )
        if not verdict["aligned"]:
            repair_issues = verdict["issues"]
            generated_dialog = await _repair_dialog_hard_failure(
                generated_dialog=generated_dialog,
                repair_issues=repair_issues,
                current_visible_percepts=current_visible_percepts,
                surface_output=surface_output,
                user_name=state["user_name"],
                llm_trace_id=llm_trace_id,
            )
            repaired_verdict = await _verify_dialog_compliance(
                surface_output=surface_output,
                generated_dialog=generated_dialog,
                current_visible_percepts=current_visible_percepts,
                llm_trace_id=llm_trace_id,
                post_repair=True,
            )
            if not repaired_verdict["aligned"]:
                await event_logging.record_model_contract_event(
                    component=DIALOG_COMPONENT,
                    stage_name="dialog_compliance",
                    violation_kind="semantic_dialog_misalignment",
                    missing_fields=[],
                    invalid_fields=repaired_verdict["issues"],
                    repair_used=True,
                    status="failed",
                    correlation_id=llm_trace_id,
                )
                raise StateContractError(
                    "dialog remains hard-invalid after one repair"
                )
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_compliance",
                violation_kind="semantic_dialog_misalignment",
                missing_fields=[],
                invalid_fields=repair_issues,
                repair_used=True,
                status="repaired",
                correlation_id=llm_trace_id,
            )
    generated_dialog_preview = (
        generated_dialog
        if isinstance(generated_dialog, list)
        else []
    )
    logger.debug(
        f"Dialog generator: "
        f"parsed_keys={parsed_keys} "
        f"messages={len(generated_dialog_preview)} "
        f"dialog={log_list_preview(generated_dialog_preview)}"
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_generator",
        route_name="generate",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if not invalid_fields else "warning",
        correlation_id=llm_trace_id,
    )
    if invalid_fields:
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_generator",
            violation_kind="invalid_dialog_output",
            missing_fields=[],
            invalid_fields=invalid_fields,
            repair_used=True,
            status="repaired",
            correlation_id=llm_trace_id,
        )

    return_value = {
        "final_dialog": generated_dialog,
    }
    return return_value


async def _repair_dialog_hard_failure(
    *,
    generated_dialog: list[str],
    repair_issues: list[str],
    current_visible_percepts: list[dict[str, Any]],
    surface_output: TextSurfaceOutputV2,
    user_name: str,
    llm_trace_id: str,
) -> list[str]:
    """Repair one verified hard error without a drifted content plan."""

    system_message = SystemMessage(
        content=_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    )
    payload = {
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "current_visible_percepts": current_visible_percepts,
        "original_final_dialog": generated_dialog,
        "permitted_action_results": list(
            surface_output["permitted_action_results"]
        ),
        "user_name": user_name,
        "verified_hard_issues": repair_issues,
    }
    runtime_limits = list(surface_output.get("runtime_capability_limits", []))
    if runtime_limits:
        payload["runtime_capability_limits"] = runtime_limits
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_generator_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    repaired_dialog = _validated_dialog_messages(parsed)
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_generator_repair",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog"],
    )
    return repaired_dialog


_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT = '''按语义而非字面重合检查一份角色回应。
current_visible_percepts 包含当前输入和结构化场景角色，candidate_role_frame 定义候选回应中的
代词归属。每条 percept 的 role_explicit_content 是上游 LLM 已解析的含义，其中“当前用户”和
“当前角色”是结构化角色枚举；用它判断嵌套的行动者、动作、对象方向，同时保留 content 作为证据。
存在 response_operation 时，以 response_owner_role、selection_owner_role、selection_required、
embedded_actor_role 和 embedded_target_role 为准。selection_required 为 true 时，让其他角色
代替 selection_owner_role 选择属于主语颠倒。

只有以下情况将 aligned 标为 false：
1. 候选回应内部存在冲突；
2. 候选回应与当前用户输入直接冲突；
3. 行动者、动作、对象、受益者或主语发生颠倒。分别解析 percept 的角色与
candidate_role_frame，再比较方向。

角色颠倒需要当前语法和语境形成唯一明确的读法。笑话、双关、省略以及存在多种合理角色读法的
措辞按 aligned 处理。

只要与当前输入和已解析角色连贯，合理虚构、相容的未来内容、玩笑式条件、鲜明个性、反问、偏移
和补充内容都不属于硬错误。本阶段不添加文风要求。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 aligned 和 issues。aligned 是布尔值；issues 是零到四条
互不重复的简短硬错误，每条最多 300 字符。aligned 为 true 时 issues 为空；为 false 时至少包含
一条问题。
'''
_dialog_semantic_fidelity_llm = LLInterface()
_dialog_semantic_fidelity_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.semantic_fidelity",
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)


async def _verify_dialog_semantic_fidelity(
    *,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check contradiction and resolved semantic-role direction."""

    system_message = SystemMessage(
        content=_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "current_visible_percepts": current_visible_percepts,
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_semantic_fidelity_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_semantic_fidelity_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_compliance_verdict(
        parsed,
        max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
    )
    trace_stage_name = (
        "dialog_semantic_fidelity_recheck"
        if post_repair
        else "dialog_semantic_fidelity_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_semantic_fidelity_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_semantic_fidelity_recheck"
            if post_repair
            else "dialog_semantic_fidelity"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


_V2_DIALOG_ROLE_DIRECTION_PROMPT = '''只核对一份角色回应是否完成必要回应并保持角色方向。
candidate_role_frame 定义回应中的代词归属；required_role_operations 是上游去语境节点已经解析的
结构化含义。其中“当前角色”表示当前角色，“当前用户”表示当前用户。

对每项 required operation，保持 response_owner_role、selection_owner_role、
embedded_actor_role 和 embedded_target_role。selection_required 为 true 时，
selection_owner_role 需要作出或表达所请求的选择。若回应改为要求其他角色完成该选择，或明确
颠倒 embedded actor 与 target，则 aligned 为 false。

当前角色可以拒绝、协商、附加条件或不执行某项动作，而不改变角色方向。笑话、双关、省略以及
存在多种合理角色读法的措辞按 aligned 处理。文风、新颖度、亲密程度、安全、动作执行与文笔质量
不属于本阶段。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 aligned 和 issues。aligned 是布尔值；issues 是零到四条
互不重复的简短角色方向问题，每条最多 300 字符。aligned 为 true 时 issues 为空；为 false 时
至少包含一条问题。
'''
_dialog_role_direction_llm = LLInterface()
_dialog_role_direction_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.role_direction",
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)


def _required_selection_role_operations(
    current_visible_percepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project only typed operations that require a semantic selection."""

    required_operations: list[dict[str, Any]] = []
    for percept in current_visible_percepts:
        operation = percept.get("response_operation")
        if not isinstance(operation, dict):
            continue
        if operation.get("selection_required") is not True:
            continue
        projected_percept: dict[str, Any] = {
            "input_source": percept.get("input_source", ""),
            "content": percept.get("content", ""),
            "role_explicit_content": percept.get(
                "role_explicit_content",
                "",
            ),
            "response_operation": {
                "operation": operation.get("operation", ""),
                "response_owner_role": operation.get(
                    "response_owner_role",
                    "",
                ),
                "selection_owner_role": operation.get(
                    "selection_owner_role",
                    "",
                ),
                "selection_required": True,
                "embedded_actor_role": operation.get(
                    "embedded_actor_role",
                    "",
                ),
                "embedded_target_role": operation.get(
                    "embedded_target_role",
                    "",
                ),
            },
        }
        required_operations.append(projected_percept)
    return required_operations


async def _verify_dialog_role_direction(
    *,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check nested role direction when typed input requires a selection."""

    required_operations = _required_selection_role_operations(
        current_visible_percepts
    )
    if not required_operations:
        return {"aligned": True, "issues": []}

    system_message = SystemMessage(
        content=_V2_DIALOG_ROLE_DIRECTION_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "required_role_operations": required_operations,
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_role_direction_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_role_direction_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_compliance_verdict(
        parsed,
        max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
    )
    trace_stage_name = (
        "dialog_role_direction_recheck"
        if post_repair
        else "dialog_role_direction_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_role_direction_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_role_direction_recheck"
            if post_repair
            else "dialog_role_direction"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


_V2_DIALOG_SURFACE_INTEGRITY_PROMPT = '''根据候选回应和精确的 permitted_action_results 核对
能力执行事实。

只有一种情况将 aligned 标为 false：候选回应声称角色大脑已经完成某项系统、工具、平台或其他
能力，但 permitted_action_results 中没有匹配的 executed 结果。完成声明必须受该结果的
action_kind、semantic_result 和 target_roles 约束。scheduled 或 pending 仍未完成；failed 或
unavailable 不支持成功声明。单纯的言语立场、请求、邀请，以及未来、条件或假设事件都不等同于
能力已经执行。

payload 可以包含 externally completed tool result 的 completed_source_evidence。
如果候选回应准确表达了该证据支持的事实，即使没有 executed action result，也按有依据处理。
不要把这类 source evidence 当作 action result，也不要把它当作声称新工具动作的许可。

runtime_capability_limits 是可信的运行时能力边界。若其中标记某项能力不可用，候选回应不能把该
能力说成已经安排、发送或完成；等待、条件、询问或明确限制属于有依据的表达。

合理虚构、创造性语言、个性、偏移和补充内容不属于本阶段的错误。本阶段不添加质量或文风要求。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 aligned 和 issues。issues 是零到四个互不重复的对象，
每个对象必须恰好包含 kind、evidence 和 explanation；kind 固定为 false_execution。evidence
复制候选回应中一段完全一致的非空文字，explanation 用一句话说明具体冲突。aligned 为 true 时
issues 为空；为 false 时至少包含一项。
'''
_dialog_surface_integrity_llm = LLInterface()
_dialog_surface_integrity_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.surface_integrity",
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)


async def _verify_dialog_surface_integrity(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check literal-speech boundaries and exact action execution truth."""

    system_message = SystemMessage(
        content=_V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "permitted_action_results": list(
            surface_output["permitted_action_results"]
        ),
        "completed_source_evidence": [
            dict(percept)
            for percept in current_visible_percepts
            if percept.get("input_source") == "tool_result"
        ],
    }
    runtime_limits = list(surface_output.get("runtime_capability_limits", []))
    if runtime_limits:
        payload["runtime_capability_limits"] = runtime_limits
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_surface_integrity_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_surface_integrity_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_surface_compliance_verdict(
        parsed,
        generated_dialog=generated_dialog,
    )
    trace_stage_name = (
        "dialog_surface_integrity_recheck"
        if post_repair
        else "dialog_surface_integrity_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_surface_integrity_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_surface_integrity_recheck"
            if post_repair
            else "dialog_surface_integrity"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


async def _verify_dialog_compliance(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Run the three focused checks and merge bounded verdict shapes."""

    semantic_verdict, role_verdict, surface_verdict = await asyncio.gather(
        _verify_dialog_semantic_fidelity(
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
        _verify_dialog_role_direction(
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
        _verify_dialog_surface_integrity(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
    )
    issues: list[str] = []
    combined_issues = (
        semantic_verdict["issues"]
        + role_verdict["issues"]
        + surface_verdict["issues"]
    )
    for issue in combined_issues:
        if issue not in issues:
            issues.append(issue)
    merged_verdict: dict[str, Any] = {
        "aligned": (
            semantic_verdict["aligned"]
            and role_verdict["aligned"]
            and surface_verdict["aligned"]
        ),
        "issues": issues,
    }
    return _validate_compliance_verdict(
        merged_verdict,
        max_issues=MAX_MERGED_VERIFIER_ISSUES,
    )



async def dialog_agent(
    global_state: GlobalPersonaState
) -> list[str]:
    """
    Dialog agent that renders dialogue from the canonical V2 surface output.
    """
    
    usage_mode = _dialog_usage_mode(global_state)
    surface_output = global_state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "persona state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    validate_text_surface_output(surface_output)
    content_plan_entry_count = 1
    sub_agent_builder = StateGraph(DialogAgentState)

    sub_agent_builder.add_node("generator", dialog_generator)
    sub_agent_builder.add_edge(START, "generator")
    sub_agent_builder.add_edge("generator", END)
    
    # Compile
    sub_graph = sub_agent_builder.compile()

    # Build initial state
    subState: DialogAgentState = {
        # A
        "internal_monologue": global_state["internal_monologue"],
        "text_surface_output_v2": surface_output,
        "cognitive_episode": global_state["cognitive_episode"],

        # B
        "chat_history_wide": global_state["chat_history_wide"],
        "chat_history_recent": global_state["chat_history_recent"],
        "platform_user_id": global_state["platform_user_id"],
        "platform_bot_id": global_state["platform_bot_id"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],

        # D
        "character_profile": global_state["character_profile"],
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": usage_mode,
        "llm_trace_id": global_state.get("llm_trace_id", ""),
    }
    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]

    logger.info(
        f"Dialog output: usage_mode={usage_mode} "
        f"dialog={log_list_preview(final_dialog)}"
    )
    logger.debug(
        f'Dialog metadata: usage_mode={usage_mode} '
        f'messages={len(final_dialog)}'
    )
    quality_status = "passed" if final_dialog else "empty"
    await event_logging.record_dialog_quality_event(
        component=DIALOG_COMPONENT,
        correlation_id="",
        usage_mode=usage_mode,
        quality_status=quality_status,
        retry_count=0,
        failure_codes=[] if final_dialog else ["empty_dialog"],
        content_plan_entry_count=content_plan_entry_count,
        status="succeeded",
    )

    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": [global_state["global_user_id"]] if final_dialog else [],
        "target_broadcast": False,
    }
    return return_value


def _current_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project current model-visible percepts within the shared prompt bound."""

    percepts = project_model_visible_percepts(episode)
    serialized = json.dumps(percepts, ensure_ascii=False)
    if len(serialized) > 24000:
        raise StateContractError("current visible percepts exceed dialog bounds")
    return percepts


def _validate_compliance_verdict(
    value: object,
    *,
    max_issues: int,
) -> dict[str, Any]:
    """Validate one exact verdict shape and its caller-owned issue bound."""

    if not isinstance(value, dict) or set(value) != {"aligned", "issues"}:
        raise StateContractError("dialog compliance fields are not exact")
    aligned = value["aligned"]
    issues = value["issues"]
    if not isinstance(aligned, bool):
        raise StateContractError("dialog compliance aligned must be boolean")
    if not isinstance(issues, list) or len(issues) > max_issues:
        raise StateContractError("dialog compliance issues are invalid")
    if len(issues) != len(set(issues)):
        raise StateContractError("dialog compliance issues are duplicated")
    if any(
        not isinstance(issue, str)
        or not issue.strip()
        or len(issue) > 300
        for issue in issues
    ):
        raise StateContractError("dialog compliance issue text is invalid")
    if aligned and issues:
        raise StateContractError("aligned dialog cannot contain issues")
    if not aligned and not issues:
        raise StateContractError("misaligned dialog requires issues")
    return {"aligned": aligned, "issues": list(issues)}


def _validated_dialog_messages(value: object) -> list[str]:
    """Validate the single repair result without adding semantic judgment."""

    if not isinstance(value, dict) or set(value) != {"final_dialog"}:
        raise StateContractError("dialog repair fields are not exact")
    messages = value["final_dialog"]
    if not isinstance(messages, list) or not messages:
        raise StateContractError("dialog repair messages are invalid")
    if any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise StateContractError("dialog repair message text is invalid")
    validated_messages = list(messages)
    return validated_messages


def _validate_surface_compliance_verdict(
    value: object,
    *,
    generated_dialog: list[str],
) -> dict[str, Any]:
    """Validate evidence-bearing surface issues and flatten them for repair."""

    if not isinstance(value, dict) or set(value) != {"aligned", "issues"}:
        raise StateContractError("surface compliance fields are not exact")
    aligned = value["aligned"]
    issues = value["issues"]
    if not isinstance(aligned, bool):
        raise StateContractError("surface compliance aligned must be boolean")
    if (
        not isinstance(issues, list)
        or len(issues) > MAX_FOCUSED_VERIFIER_ISSUES
    ):
        raise StateContractError("surface compliance issues are invalid")
    candidate_text = "\n".join(generated_dialog)
    normalized_rows: list[tuple[str, str, str]] = []
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {
            "kind",
            "evidence",
            "explanation",
        }:
            raise StateContractError("surface issue fields are not exact")
        kind = issue["kind"]
        evidence = issue["evidence"]
        explanation = issue["explanation"]
        if kind not in {
            "false_execution",
        }:
            raise StateContractError("surface issue kind is invalid")
        if (
            not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 120
            or evidence not in candidate_text
        ):
            raise StateContractError("surface issue evidence is invalid")
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 140
        ):
            raise StateContractError("surface issue explanation is invalid")
        normalized_rows.append((kind, evidence, explanation))
    if len(normalized_rows) != len(set(normalized_rows)):
        raise StateContractError("surface compliance issues are duplicated")
    if aligned and normalized_rows:
        raise StateContractError("aligned surface cannot contain issues")
    if not aligned and not normalized_rows:
        raise StateContractError("misaligned surface requires issues")
    normalized_issues = [
        f"{kind}: {evidence!r} - {explanation}"
        for kind, evidence, explanation in normalized_rows
    ]
    return {
        "aligned": aligned,
        "issues": normalized_issues,
    }
