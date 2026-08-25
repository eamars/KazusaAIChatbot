"""Dialog execution agent.

Design intent:
- Dialog agent turns the upstream content plan into natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3. If dialog
  needs a fact, answer, conclusion, question, or code block, it must already be
  represented in `text_surface_output_v2.content_plan`.
"""

import json
import logging
import re
import time
from typing import Any, NotRequired, TypedDict

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from kazusa_ai_chatbot import event_logging, llm_tracing
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    CognitiveEpisodeV1,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    TextSurfaceInput,
    TextSurfaceOutputV2,
    validate_text_surface_input_canonical,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_shared.surface_stages import (
    VISIBLE_CONTENT_AUTHORITY_GUIDANCE,
)
from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_API_KEY,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_GENERATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import (
    log_list_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
DIALOG_COMPONENT = "nodes.dialog_agent"
DEFAULT_DIALOG_USAGE_MODE = "live_visible_reply"
DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE = (
    "self_cognition_action_candidate_render"
)
DIALOG_GENERATOR_TOTAL_ATTEMPTS = V2_MODEL_TOTAL_ATTEMPTS
DIALOG_CANDIDATE_MAX_CHARS = 12000
_HTTP_URL_PATTERN = re.compile(
    r"https?://[^\s\\)>\]}\"']+",
    re.IGNORECASE,
)
_MAX_REQUIRED_SOURCE_URLS = 8


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


class DialogGenerationContractError(StateContractError):
    """Expose total dialog-generator exhaustion without candidate details."""

    error_code = "dialog_generator_exhausted"
    stage = "dialog_generation"
    attempt_count = DIALOG_GENERATOR_TOTAL_ATTEMPTS
    safe_checkpoint = "post_cognition_commit"
    retryable = False


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
    text_surface_input: NotRequired[TextSurfaceInput]
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


def _candidate_role_frame(
    surface_output: TextSurfaceOutputV2,
) -> dict[str, Any]:
    """Project the authoritative target wording rules for dialog owners."""

    second_person_allowed_handles = [
        row["handle"]
        for row in surface_output["addressee_plan"]
        if row["wording_policy"] == "second_person_allowed"
    ]
    typed_non_current_targets = [
        {
            "handle": row["handle"],
            "display_name": row["display_name"],
            "semantic_role": row["semantic_role"],
            "wording_policy": row["wording_policy"],
        }
        for row in surface_output["addressee_plan"]
        if (
            row["handle"].startswith("p")
            and row["handle"] not in second_person_allowed_handles
        )
    ]
    frame = dict(_CANDIDATE_ROLE_FRAME)
    if typed_non_current_targets:
        frame["second_person_allowed_handles"] = (
            second_person_allowed_handles
        )
        frame["typed_non_current_targets"] = typed_non_current_targets
    return frame


_V2_DIALOG_GENERATOR_PROMPT_TEMPLATE = '''你是当前角色的最终文字渲染器。把 text_surface_output_v2 转化为
自然、鲜活、有角色辨识度，并且切合当前场景的聊天内容。上游认知负责角色判断；surface planning
提供语义内容、称呼安排、delivery profile、lexical_avoidances 和 permitted action results。

# 最高优先级的断言边界
payload.epistemic_boundary 是 text_surface_output_v2 中同一字段的置前副本，是上游认知
确定的断言、解释与未知边界。它的权威高于 selected_surface_intent、content_plan 和
content_requirements 中任何更强的确定性。所有未被允许直接断言的功能、原因、来源、
意图或结果，都在同一句可见措辞中明确表达为猜测或未知。从句、前提句、原因连接和反问
也不能把推测升级为既定事实；未观察到的特征不能用来排除一种功能或可能性。
输出前逐句检查可见断言；任何超过 epistemic_boundary 的句子先改写为明确猜测或未知。

resolver_result 提供来源自有的执行结果；其中的 source-owned evidence_state、evidence_excerpts、
evidence_handles、prompt_safe_observation_handle 和 remaining_needs 属权威边界。
evidence_state=complete 时，只能依据 supplied evidence_excerpts；partial、pending、missing 或
blocked 时，表达答案缺口、等待状态或 typed blocker，不把 generic semantic_result 当事实。

{visible_content_authority_guidance}

# 渲染步骤
1. selected_surface_intent 是本轮语义锚点；epistemic_boundary 限定可见断言强度；
content_plan 和 content_requirements 展开所需事实、
理由和互动推进。relational_willingness（如果存在）是上游已选择的角色关系立场；按上方可见内容权威规则参与表达，
不重新选择未被当前语义选中的内容。
以这组权威语义组织对象、事实、行动者、受益者和回应方向。
2. 先整体阅读 selected_surface_intent、content_plan、content_requirements、
visible_boundaries、addressee_plan 和 delivery_profile，判断规划中的开场反应指向行动或关系本身，还是指向提问的
时机、突然程度或直接程度。可自由组合惊讶、羞赧、防御、调侃、嘴硬、表面勉强、间接表达、温柔、
热烈以及其他符合角色的情绪和特征。这些表达可以先于明确决定出现，并与后文共同传达同一已选决定。
在这条语义弧线内，自由加入相容的想象细节、个性、幽默、主动性、温度和创造性展开，形成当前角色
实际会说出或发送的鲜活回应。
3. 按每条 percept 的结构化角色框架和 addressee_plan 保持行动者、对象、受益者与主语方向。生成的
对话由当前角色说出：第一人称属于当前角色；只有 wording_policy 为 second_person_allowed 的
current_user 行允许用第二人称；typed third-party 行必须使用其 display_name 或明确第三人称，
不得把第三方改写成当前用户的“你”。跨角色框架转换时延续原有方向。回顾型请求直接表达 surface
已确认的历史事实。
4. 按 permitted_action_results 映射执行状态：executed 表达其有界的已完成效果；scheduled 与
pending 表达已记录、已排队或等待对应 worker；failed 与 unavailable 表达当前限制和可行下一步；
请求、意图或 content plan 表达角色的言语立场。
每个可见完成效果都必须由同一类型、同一效果的 executed 行精确支持；一个 executed 行不是对其他行动的概括授权。
action_kind=speak 只授权说出或发送 final_dialog 的文字，不授权肢体或面部动作、触碰、物体操作、拥抱、亲吻、感官效果或其他外部结果。
对未来外部效果的具体承诺也属于行动主张：登记、预留、排期、发送、交付、调用工具或稍后联络等承诺，必须有同一效果的 pending、scheduled 或 executed 行。
没有对应结果时，可以表达 response_plan 已选择的当前言语立场、愿望、提议或条件，不承诺具体外部执行将发生。
对于没有对应 executed 结果的物理或外部效果，只渲染言语上的接受、拒绝、提议、邀请或意图；
不用动作舞台提示、拟声、已完成断言或结果反问声称该行动已执行、已交付或已被接收。
content_plan 或 content_requirements 中的任何执行指令都低于 permitted_action_results 的事实权威。
resolver_result.status=succeeded 且 semantic_result 明确任务已接纳并继续工作时，可以表达已接纳、
继续处理和等待后续结果，但不能声称最终结果已经完成。
task_resolution_request 的 evidence_state=complete 只能依据 evidence_excerpts；不完整状态保留
remaining_needs 的缺口，不补写缺失引文或答案。
5. 按 runtime_capability_limits 表达可信的能力边界、等待状态和下一步条件。
6. 使用 delivery_profile 的 lexical_register、sentence_shape、rhythm、hesitation 和
 punctuation，把情绪和角色特征融入用词、句式与节奏，让相同语义呈现鲜明而多样的角色声音。
7. lexical_avoidances 是 surface planning 为本轮表达连续性提供的具体片段。避免逐字重复这些片段，
同时保留 content_plan、content_requirements、selected_surface_intent 和关系立场的原义。该列表只
影响措辞，不是主题许可、道德判断、拒绝理由或新的立场选择。
8. payload 存在 required_source_urls 时，只能使用列表内的 URL，并至少逐字复制其中一个。根据
content_plan 中实际呈现的事实选择直接相关来源；回应包含来自不同来源的实质性事实时，分别附上
足以支撑这些事实的 exact URL，避免让单一链接看似支持全部不同主张。
存在 required_source_urls 时，角色化展开只改变语气、节奏、互动和表达方式；产品规格、价格、库存、
数量、时间、因果关系和其他可外部核查事实必须来自 content_plan 或 content_requirements，不添加
权威 surface 未提供的新事实。
创造性展开不得增加 content_plan 未选择的立场、关系主张、用户意图或对话收束。
不得把已表达的关系性回应模式改写成新的主要收束，除非当前 response_goal 重新选择；角色
化细节只能服务已选择的语义，不得用语义换词恢复未选择的关系回报。

引文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出前不可跳过的合同检查
1. 逐句对照 payload.epistemic_boundary。对每个功能、原因、来源、意图、结果或排除性主张，
   边界未允许直接断言时，必须在同一句中明确使用猜测或未知措辞。不把缺少可见特征或证据改写成排除性事实。
   如果 content_plan 或 content_requirements 越界，以 epistemic_boundary 为准主动降低断言强度。
2. 逐句对照 permitted_action_results。每个动作舞台提示、身体反应、触碰、物体操作、拟声、感官反馈或外部结果，
   都必须有同一行动与效果的 executed 行。action_kind=speak 只支持 final_dialog 的文字；它不支持任何身体或外部效果。
   未来时的具体外部承诺同样必须有同一效果的 pending、scheduled 或 executed 行。删去不匹配的动作、括号舞台提示、拟声、感官反馈、结果反问和外部执行承诺，
   保留 response_plan 已选择的当前言语立场、愿望、提议或条件。

# 输出格式
字段必须恰好是 final_dialog。final_dialog 是由完整可见消息字符串组成的
非空列表。
'''


_V2_DIALOG_GENERATOR_PROMPT = _V2_DIALOG_GENERATOR_PROMPT_TEMPLATE.format(
    visible_content_authority_guidance=VISIBLE_CONTENT_AUTHORITY_GUIDANCE,
)

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
    """Render final dialog from the validated V2 surface within the bounded budget.

    The generator is the only model stage in the final wording path. Structural
    JSON parsing, message validation, and immutable source-URL verification are
    deterministic; no semantic verifier, scoring, or evaluator-driven repair
    follows generation.

    Args:
        state: Dialog state containing the canonical surface and scene episode.

    Returns:
        The accepted visible dialog paired with the surface that produced it.

    Raises:
        DialogGenerationContractError: If no structurally valid candidate with
            clean source-URL fidelity remains after the bounded producer
            opportunities.
    """

    usage_mode = state["dialog_usage_mode"]
    surface_output = state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "dialog state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    surface_output = validate_text_surface_output(surface_output)
    current_visible_percepts = _current_visible_percepts(
        state["cognitive_episode"]
    )
    required_source_urls = _completed_tool_result_source_urls(
        current_visible_percepts
    )
    llm_trace_id = state.get("llm_trace_id", "")
    accepted_dialog: list[str] | None = None

    for attempt_number in range(1, DIALOG_GENERATOR_TOTAL_ATTEMPTS + 1):
        generated_dialog, _ = await _render_dialog_candidate(
            surface_output=surface_output,
            user_name=state["user_name"],
            attempt_number=attempt_number,
            llm_trace_id=llm_trace_id,
            required_source_urls=required_source_urls,
        )
        if not generated_dialog:
            continue
        source_url_issues = _dialog_source_url_issues(
            generated_dialog,
            required_source_urls=required_source_urls,
        )
        if source_url_issues:
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_source_url_fidelity",
                violation_kind="source_url_fidelity",
                missing_fields=[],
                invalid_fields=source_url_issues,
                repair_used=attempt_number > 1,
                status="retrying",
                correlation_id=llm_trace_id,
            )
            continue
        accepted_dialog = list(generated_dialog)
        break

    if accepted_dialog is None:
        raise DialogGenerationContractError(
            "dialog generator exhausted candidates without a valid candidate"
        )
    return_value = {
        "final_dialog": accepted_dialog,
        "text_surface_output_v2": surface_output,
    }
    return return_value


async def _render_dialog_candidate(
    *,
    surface_output: TextSurfaceOutputV2,
    user_name: str,
    attempt_number: int,
    llm_trace_id: str,
    required_source_urls: list[str] | None = None,
) -> tuple[list[str], str | None]:
    """Render one candidate within the shared bounded attempt budget.

    Args:
        surface_output: Current validated semantic surface authority.
        user_name: Optional natural addressee name for final wording.
        attempt_number: One-based producer opportunity in the shared budget.
        llm_trace_id: Correlation identifier for protected model evidence.
        required_source_urls: Exact completed-evidence URLs that visible output
            may preserve.

    Returns:
        A bounded dialog and no failure kind, or an empty dialog with the
        classified provider or structure failure kind.
    """

    if not 1 <= attempt_number <= DIALOG_GENERATOR_TOTAL_ATTEMPTS:
        raise ValueError("dialog generator attempt number is invalid")
    validated_surface = validate_text_surface_output(surface_output)
    source_urls = list(required_source_urls or [])
    system_message = SystemMessage(content=_V2_DIALOG_GENERATOR_PROMPT)
    payload: dict[str, Any] = {
        "epistemic_boundary": validated_surface["epistemic_boundary"],
        "text_surface_output_v2": dict(validated_surface),
        "candidate_role_frame": _candidate_role_frame(validated_surface),
        "user_name": user_name,
    }
    if source_urls:
        payload["required_source_urls"] = source_urls
    stage_name = (
        "dialog_generator"
        if attempt_number == 1
        else "dialog_generator_retry"
    )
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    request_messages = [system_message, human_message]
    started_at = time.perf_counter()
    try:
        response = await _dialog_generator_llm.ainvoke(
            request_messages,
            config=_dialog_generator_llm_config,
        )
    except (
        OpenAIError,
        httpx.HTTPError,
        ConnectionError,
        OSError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text="",
            parsed_output={},
            parse_status="provider_error",
            status="failed",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["final_dialog"],
            sequence=attempt_number - 1,
        )
        logger.warning(
            f"Dialog candidate {attempt_number} provider failure: {exc}"
        )
        return [], "provider"

    response_text = str(getattr(response, "content", ""))
    parsed: object = {}
    try:
        parsed = parse_llm_json_output(response_text)
        generated_dialog = _validated_dialog_messages(parsed)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="contract_error",
            status="failed",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["final_dialog"],
            sequence=attempt_number - 1,
        )
        logger.warning(
            f"Dialog candidate {attempt_number} contract failure: {exc}"
        )
        return [], "structure"

    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=request_messages,
        response_text=response_text,
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog"],
        sequence=attempt_number - 1,
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=stage_name,
        route_name="generate",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=sum(
            len(str(message.content))
            for message in request_messages
        ),
        output_chars=len(response_text),
        parse_status="succeeded",
        retry_count=attempt_number - 1,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return generated_dialog, None


async def dialog_agent(
    global_state: GlobalPersonaState
) -> dict[str, Any]:
    """Render a public dialog result from the canonical V2 surface.

    Args:
        global_state: Persona graph state with a committed V2 surface.

    Returns:
        Visible dialog delivery fields paired with the dialog-accepted surface.
    """

    usage_mode = _dialog_usage_mode(global_state)
    surface_output = global_state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "persona state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    validate_text_surface_output(surface_output)
    surface_input = global_state.get("text_surface_input")
    if surface_input is not None and not isinstance(surface_input, dict):
        raise StateContractError(
            "persona state text_surface_input must be an object"
        )
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
    if isinstance(surface_input, dict):
        subState["text_surface_input"] = validate_text_surface_input_canonical(
            surface_input
        )
    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]
    accepted_surface = result["text_surface_output_v2"]
    if not isinstance(accepted_surface, dict):
        raise StateContractError(
            "dialog result missing accepted text_surface_output_v2"
        )
    accepted_surface = validate_text_surface_output(accepted_surface)

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
        "target_addressed_user_ids": (
            [global_state["global_user_id"]]
            if final_dialog
            else []
        ),
        "target_broadcast": False,
        "text_surface_output_v2": accepted_surface,
    }
    return return_value


def _completed_tool_result_source_urls(
    current_visible_percepts: list[dict[str, Any]],
) -> list[str]:
    """Extract exact HTTP source tokens from completed tool-result evidence."""

    source_urls: list[str] = []
    for percept in current_visible_percepts:
        if percept.get("input_source") != "tool_result":
            continue
        serialized_content = json.dumps(
            percept.get("content", {}),
            ensure_ascii=False,
            default=str,
        )
        for match in _HTTP_URL_PATTERN.finditer(serialized_content):
            source_url = match.group(0).rstrip(".,;:")
            if source_url in source_urls:
                continue
            source_urls.append(source_url)
            if len(source_urls) >= _MAX_REQUIRED_SOURCE_URLS:
                return source_urls
    return source_urls


def _dialog_source_url_issues(
    generated_dialog: list[str],
    *,
    required_source_urls: list[str],
) -> list[str]:
    """Validate immutable source URL tokens without judging dialog semantics."""

    if not required_source_urls:
        return []
    candidate_text = "\n".join(generated_dialog)
    candidate_urls = {
        match.group(0).rstrip(".,;:")
        for match in _HTTP_URL_PATTERN.finditer(candidate_text)
    }
    allowed_urls = set(required_source_urls)
    unexpected_urls = sorted(candidate_urls - allowed_urls)
    if unexpected_urls:
        return [
            "source_url_fidelity: candidate URL is absent from completed "
            "tool evidence"
        ]
    if not candidate_urls & allowed_urls:
        return [
            "source_url_fidelity: include at least one exact "
            "required_source_urls token"
        ]
    return []


def _current_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project current model-visible percepts within the shared prompt bound."""

    percepts = project_model_visible_percepts(episode)
    serialized = json.dumps(percepts, ensure_ascii=False)
    if len(serialized) > 24000:
        raise StateContractError("current visible percepts exceed dialog bounds")
    return percepts


def _validated_dialog_messages(value: object) -> list[str]:
    """Validate and normalize generated dialog without semantic judgment."""

    if not isinstance(value, dict) or set(value) != {"final_dialog"}:
        raise StateContractError("dialog output fields are not exact")
    messages = value["final_dialog"]
    if not isinstance(messages, list) or not messages:
        raise StateContractError("dialog output messages are invalid")
    if any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise StateContractError("dialog output message text is invalid")
    if sum(len(message) for message in messages) > DIALOG_CANDIDATE_MAX_CHARS:
        raise StateContractError("dialog output messages exceed text bound")
    validated_messages = [
        re.sub(
            r"(?:\r?\n)[ \t]*(?:(?:\r?\n)[ \t]*)+",
            "\n",
            message,
        )
        for message in messages
    ]
    return validated_messages
