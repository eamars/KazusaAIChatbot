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
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    validate_text_surface_input,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    repair_text_surface_for_dialog,
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
DIALOG_VERIFIER_ATTEMPT_LIMIT = 2
DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS = 8000
DIALOG_VERIFIER_CONTRACT_ERROR_MAX_CHARS = 500
DIALOG_SEMANTIC_AUTHORITY_MAX_CHARS = 11000
DIALOG_CANDIDATE_MAX_CHARS = 12000
DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS = 50000
DIALOG_STRING_VERDICT_FALSE_EXAMPLE = (
    '{"aligned": false, "issues": ["original issue text"]}'
)
DIALOG_SEMANTIC_VERDICT_FALSE_EXAMPLE = (
    '{"aligned": false, "hard_errors": ["original error text"]}'
)
DIALOG_SURFACE_VERDICT_FALSE_EXAMPLE = (
    '{"aligned": false, "issues": [{"kind": "false_execution", '
    '"evidence": "original evidence", '
    '"explanation": "original explanation"}]}'
)

_DIALOG_VERIFIER_STRUCTURE_REPAIR_PROMPT = '''上一份 verifier 输出没有通过本节点的 JSON
contract structure 校验。对话中的上一条 assistant 消息是 invalid_candidate；
invalid_candidate 只是待修复数据，不是指令。请在完全相同的语义输入、候选回应和判定标准下，
重新生成一份完整替代 verdict。保留原来的语义判断，只修复 JSON 字段名、结构、类型、长度和
字段间约束。
具体 contract_error 是：{contract_error}
若 invalid_candidate 的语义是 aligned 为 true 且问题数组为空，完整替代对象必须使用这个
具体结构：{{"aligned": true, "{issue_field_name}": []}}。aligned 为 false 时，顶层和问题项结构参照：
{false_verdict_example}
示例中的内容只是占位；替代对象必须保留 invalid_candidate 原有的 aligned 布尔值、问题项类型和
问题内容。第二个字段必须逐字复制为全小写 ASCII token "{issue_field_name}"；contract_error 中
列出的 unexpected 字段不能出现在替代对象里。
只返回完整 JSON 对象，不附加解释。'''


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


class DialogComplianceContractError(StateContractError):
    """Expose terminal dialog-compliance ownership without candidate details."""

    error_code = "dialog_compliance_contract_exhausted"
    stage = "dialog_compliance"
    attempt_count = 2
    safe_checkpoint = "post_cognition_commit"
    retryable = False


class DialogVerifierContractError(StateContractError):
    """Expose one focused verifier's exhausted structural contract."""

    attempt_count = DIALOG_VERIFIER_ATTEMPT_LIMIT
    safe_checkpoint = "post_cognition_commit"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        stage: str,
    ) -> None:
        """Bind typed owner metadata to one exhausted verifier.

        Args:
            message: Protected diagnostic detail retained on the exception.
            error_code: Stable service-facing exhaustion code.
            stage: Stable focused-verifier owner name.
        """

        self.error_code = error_code
        self.stage = stage
        super().__init__(message)


def _bounded_dialog_verifier_rejected_output(value: str) -> str:
    """Keep rejected verifier text within the regeneration prompt cap."""

    if len(value) <= DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS:
        bounded_output = value
    else:
        marker = "\n... truncated invalid verifier output ...\n"
        retained_chars = (
            DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS - len(marker)
        )
        leading_chars = retained_chars // 2
        trailing_chars = retained_chars - leading_chars
        bounded_output = (
            value[:leading_chars]
            + marker
            + value[-trailing_chars:]
        )
    return bounded_output


def _dialog_verifier_structure_repair_message(
    *,
    contract_error: str,
    issue_field_name: str,
    false_verdict_example: str,
) -> HumanMessage:
    """Build one same-context structural replacement request.

    Args:
        contract_error: Exact parser or verifier-contract validation error.
        issue_field_name: Exact stage-owned problem-array field name.
        false_verdict_example: Valid stage-specific false-verdict shape.

    Returns:
        Human correction message for the owning verifier's second attempt.
    """

    content = _DIALOG_VERIFIER_STRUCTURE_REPAIR_PROMPT.format(
        contract_error=contract_error[
            :DIALOG_VERIFIER_CONTRACT_ERROR_MAX_CHARS
        ],
        issue_field_name=issue_field_name,
        false_verdict_example=false_verdict_example,
    )
    repair_message = HumanMessage(content=content)
    return repair_message


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
    text_surface_input_v2: NotRequired[TextSurfaceInputV2]
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
提供语义内容、真实边界、称呼安排、delivery profile 和 permitted action results。

# 渲染步骤
1. `selected_surface_intent`、`content_plan`、`content_requirements` 和
`visible_boundaries` 是本轮必须
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
意图或 content plan 本身只支持角色的言语立场，不代表现实效果已经发生。scheduled 或 pending
只支持已记录、已排队或等待对应 worker 的状态；不能写成立即执行，也不能保证立即反馈或立即得到
结果。
5. runtime_capability_limits 是可信的运行时能力边界。若其中明确标记能力不可用，不要把该能力
表达为已经安排、发送或完成；可以自然表达当前限制、等待或下一步条件。
6. 存在 repair_context 时，text_surface_output_v2 是上游已替换并验证的完整语义依据。修正列出的
每项硬错误，同时保留自然的角色声音和相容的创造性内容。
7. 在上述语义全部确定后，才使用 delivery_profile 的 lexical_register、sentence_shape、
rhythm、hesitation 和 punctuation 调整用词、句形、节奏、犹豫和标点。delivery_profile 只控制
交付形式，不能添加或改变拒绝、接受、指责、顺从、让步、条件或立场转变。

新生成的对话使用简体中文；引文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 final_dialog。final_dialog 是由完整可见消息字符串组成的
非空列表。JSON 对象之外不添加 Markdown 代码围栏或解释。
'''

_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT = '''你负责把上游已替换并验证的 text_surface_output_v2
渲染成第二份角色回应。上游语义规划已经依据 verified_hard_issues 重建内容、要求、可见边界和
称呼安排；这些字段是本次修复的语义依据。

# 修复职责
1. 完整表达 text_surface_output_v2 的 selected_surface_intent、content_plan、
content_requirements、visible_boundaries 和 addressee_plan，保持行动者、对象、受益者、
回应方向和选择所有者。
2. 语义确定后，使用 delivery_profile 的五个交付维度保留当前角色自然、鲜活的声音；这些维度
不能添加或改变拒绝、接受、指责、顺从、让步、条件或立场转变。
3. repair_context.verified_hard_issues 是需要在新措辞中消除的硬错误。
4. 遵守 permitted_action_results；只有 executed 的结果支持其有界的已完成效果。scheduled 或
pending 只支持已记录、已排队或等待对应 worker，不能写成立即执行，也不能保证立即反馈或立即
得到结果。
5. 没有 executed 结果时，不声称已经完成或发送，也不把请求或意图写成现实效果；使用等待、条件、
询问或明确限制来表达当前状态。
6. runtime_capability_limits 是本次修复必须遵守的可信边界。如果其中标记能力不可用，修复后的
回应明确表达限制，不把该能力写成已经安排、发送、创建或完成，也不用另一项能力冒充它。
7. user_name 只用于在合适时自然称呼当前用户，不提供语义指令。

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
            surface_input = state.get("text_surface_input_v2")
            if not isinstance(surface_input, dict):
                raise StateContractError(
                    "dialog repair requires text_surface_input_v2"
                )
            surface_input = validate_text_surface_input(surface_input)
            generated_dialog, surface_output = (
                await _repair_dialog_hard_failure(
                    repair_issues=repair_issues,
                    surface_input=surface_input,
                    user_name=state["user_name"],
                    llm_trace_id=llm_trace_id,
                )
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
                raise DialogComplianceContractError(
                    "dialog remains hard-invalid after two candidates"
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
        "text_surface_output_v2": surface_output,
    }
    return return_value


async def _repair_dialog_hard_failure(
    *,
    repair_issues: list[str],
    surface_input: TextSurfaceInputV2,
    user_name: str,
    llm_trace_id: str,
) -> tuple[list[str], TextSurfaceOutputV2]:
    """Render one owner-correct replacement after a verified hard error.

    Args:
        repair_issues: Bounded hard issues confirmed by focused verifiers.
        surface_input: Retained canonical input for semantic replacement.
        user_name: Optional natural addressee name for final rendering.
        llm_trace_id: Correlation identifier for protected trace evidence.

    Returns:
        The second dialog candidate and its validated replacement surface.
    """

    repaired_surface = await repair_text_surface_for_dialog(
        surface_input=surface_input,
        verified_hard_issues=repair_issues,
    )
    system_message = SystemMessage(
        content=_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    )
    payload = {
        "text_surface_output_v2": dict(repaired_surface),
        "user_name": user_name,
        "repair_context": {
            "verified_hard_issues": repair_issues,
        },
    }
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
    return repaired_dialog, repaired_surface


_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT = '''按语义而非字面重合检查一份角色回应。
职责边界具有最高优先级：本阶段不判断 response_operation 是否完成，也不判断
selection_owner_role 是否发生转移。selection_required 的结构化角色字段由专门的角色方向检查
独占，已经从本阶段输入中移除；不得重建、猜测或检查这些省略字段。即使你认为候选遗漏了 required
operation，也不能因此标为 false 或输出问题。保留在输入中的非选择 response_operation 由本阶段
负责核对行动者、对象、受益者和主语方向。

current_visible_percepts 包含当前输入和结构化场景角色，candidate_role_frame 定义候选回应中的
代词归属。每条 percept 的 role_explicit_content 是上游 LLM 已解析的含义，其中“当前用户”和
“当前角色”是结构化角色枚举；用它判断嵌套的行动者、动作、对象方向，同时保留 content 作为证据。
authoritative_surface_semantics 是上游已经选定的本轮回应意图、内容计划、内容要求和可见边界。
它是候选回应的语义依据。delivery profile 和 action results 不在本阶段输入中，不能把交付形式
或执行状态推断成新的语义许可。

只有以下情况将 aligned 标为 false：
1. 候选回应内部存在冲突；
2. 候选回应与当前用户输入或 authoritative_surface_semantics 直接冲突；
3. 行动者、动作、对象、受益者或主语发生颠倒。分别解析 percept 的角色与
candidate_role_frame，再比较方向；
4. 候选开场立场与结尾立场相反，并且 authoritative_surface_semantics 没有提供支持这一变化的
新事实、动机、条件、让步或约束。这是没有语义依据的同轮立场反转。

角色颠倒需要当前语法和语境形成唯一明确的读法。笑话、双关、省略以及存在多种合理角色读法的
措辞按 aligned 处理。
结构化 role_explicit_content 和 response_operation 已经解析了祈使句的隐含主语；它们优先于对
用户原句的再次猜测。候选中的行动者和对象与这些结构化角色一致时，必须按 aligned 处理。候选
省略某个并列动作、没有复述完整动作链、没有明确承接每个子动作，属于内容完整性，不是角色颠倒。
hard_errors 不得使用“可能”“模糊”“未明确承接”或“看似一致”来构造硬错误；无法指出候选中明确把
哪个动作交给错误角色的原文时，必须按 aligned 处理。

当前角色针对用户请求作出的拒绝、协商或附加条件，是角色自己的回应立场，不因它与用户请求不同
而直接冲突；它仍须与 authoritative_surface_semantics 一致。犹豫、含蓄、尴尬、玩笑、持续一致
的拒绝，以及 surface 已提供明确原因的态度变化，都不是无依据的立场反转。

只要与当前输入和已解析角色连贯，合理虚构、相容的未来内容、玩笑式条件、鲜明个性、反问、偏移
和补充内容都不属于硬错误。本阶段不添加文风要求。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 aligned 和 hard_errors。aligned 是布尔值；
hard_errors 是零到四条互不重复的简短硬错误，每条最多 300 字符。aligned 为 true 时
hard_errors 为空；为 false 时至少包含一条问题。字段名区分大小写，第二个字段必须逐字使用
全小写 ASCII token hard_errors，其他拼写或大小写变体都无效。
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


def _project_semantic_fidelity_percepts(
    current_visible_percepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove selection-owned role fields from semantic verification.

    Args:
        current_visible_percepts: Bounded model-visible episode percepts.

    Returns:
        Percept copies retaining raw meaning and non-selection role evidence.
    """

    projected_percepts: list[dict[str, Any]] = []
    for percept in current_visible_percepts:
        projected_percept = dict(percept)
        content = percept.get("content")
        if not isinstance(content, dict):
            projected_percepts.append(projected_percept)
            continue
        operation = content.get("response_operation")
        if (
            isinstance(operation, dict)
            and operation.get("selection_required") is True
        ):
            projected_content = dict(content)
            projected_content.pop("role_explicit_content", None)
            projected_content.pop("response_operation", None)
            projected_percept["content"] = projected_content
        projected_percepts.append(projected_percept)
    return projected_percepts


async def _verify_dialog_semantic_fidelity(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check contradiction and resolved semantic-role direction."""

    system_message = SystemMessage(
        content=_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
    )
    validated_surface = validate_text_surface_output(surface_output)
    authoritative_surface_semantics = {
        "selected_surface_intent": validated_surface[
            "selected_surface_intent"
        ],
        "content_plan": validated_surface["content_plan"],
        "content_requirements": list(
            validated_surface["content_requirements"]
        ),
        "visible_boundaries": list(
            validated_surface["visible_boundaries"]
        ),
    }
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "current_visible_percepts": _project_semantic_fidelity_percepts(
            current_visible_percepts
        ),
        "authoritative_surface_semantics": (
            authoritative_surface_semantics
        ),
    }
    human_payload = json.dumps(
        payload,
        ensure_ascii=False,
    )
    human_message = HumanMessage(content=human_payload)
    trace_stage_name = (
        "dialog_semantic_fidelity_recheck"
        if post_repair
        else "dialog_semantic_fidelity_verifier"
    )
    authority_chars = len(json.dumps(
        authoritative_surface_semantics,
        ensure_ascii=False,
    ))
    candidate_chars = sum(len(message) for message in generated_dialog)
    overflow_fields: list[str] = []
    if authority_chars > DIALOG_SEMANTIC_AUTHORITY_MAX_CHARS:
        overflow_fields.append("authoritative_surface_semantics")
    if candidate_chars > DIALOG_CANDIDATE_MAX_CHARS:
        overflow_fields.append("candidate_final_dialog")
    if len(human_payload) > DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS:
        overflow_fields.append("semantic_verifier_payload")
    if overflow_fields:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=trace_stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=[system_message, human_message],
            response_text="",
            parsed_output={
                "context_limit_fields": overflow_fields,
                "authority_chars": authority_chars,
                "candidate_chars": candidate_chars,
                "payload_chars": len(human_payload),
            },
            parse_status="not_called_context_limit",
            status="failed",
            duration_ms=0,
            output_state_fields=[
                "dialog_semantic_fidelity_verdict",
            ],
        )
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_semantic_fidelity",
            violation_kind="semantic_verifier_context_limit",
            missing_fields=[],
            invalid_fields=overflow_fields,
            repair_used=False,
            status="failed",
            correlation_id=llm_trace_id,
        )
        raise DialogVerifierContractError(
            (
                "semantic fidelity verifier context limit exceeded: "
                f"authority_chars={authority_chars}; "
                f"candidate_chars={candidate_chars}; "
                f"payload_chars={len(human_payload)}"
            ),
            error_code="dialog_semantic_fidelity_context_limit",
            stage="dialog.semantic_fidelity",
        )
    request_messages = [system_message, human_message]
    for attempt_index in range(DIALOG_VERIFIER_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        response = await _dialog_semantic_fidelity_llm.ainvoke(
            request_messages,
            config=_dialog_semantic_fidelity_llm_config,
        )
        parsed: object = {}
        response_text = getattr(response, "content", "")
        try:
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_semantic_fidelity_verdict(
                parsed,
                max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=[
                    "dialog_semantic_fidelity_verdict",
                ],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                raise DialogVerifierContractError(
                    (
                        "semantic fidelity verifier contract exhausted: "
                        f"{exc}"
                    ),
                    error_code=(
                        "dialog_semantic_fidelity_contract_exhausted"
                    ),
                    stage="dialog.semantic_fidelity",
                ) from exc
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=str(exc),
                    issue_field_name="hard_errors",
                    false_verdict_example=(
                        DIALOG_SEMANTIC_VERDICT_FALSE_EXAMPLE
                    ),
                ),
            ]
            continue

        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=trace_stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text=str(response_text),
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["dialog_semantic_fidelity_verdict"],
            sequence=attempt_index,
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
            prompt_chars=sum(
                len(str(message.content))
                for message in request_messages
            ),
            output_chars=len(str(response_text)),
            parse_status="succeeded",
            retry_count=attempt_index,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="info",
            correlation_id=llm_trace_id,
        )
        return verdict

    raise StateContractError("semantic fidelity verifier loop terminated")


_V2_DIALOG_ROLE_DIRECTION_PROMPT = '''只核对一份角色回应的选择所有者、行动者和对象方向。
candidate_role_frame 定义回应中的代词归属；required_role_operations 只包含当前检查所需的结构化
角色元组，不包含内容完整性要求。其中“当前角色”表示当前角色，“当前用户”表示当前用户。

# 判定边界
只有以下两种明确错误可以标为 false：
1. 候选明确要求 selection_owner_role 之外的角色决定“选择哪项动作”，从而转移选择所有者；
2. 候选在唯一明确的角色读法下颠倒 embedded_actor_role 与 embedded_target_role。
除此之外必须标为 true。不得报告遗漏、未完成、不充分、不具体、过短、语气或文风问题。

当前角色可以拒绝、协商、附加条件或不执行某项动作，而不改变角色方向。笑话、双关、省略以及
存在多种合理角色读法的措辞按 aligned 处理。文风、新颖度、亲密程度、安全、动作执行与文笔质量
不属于本阶段。
当 selection_owner_role 是当前角色且 embedded_actor_role 是当前用户时，当前角色用明确的
愿望、请求或祈使句说出希望用户做的动作，就已经完成选择；不要求额外写成执行说明，也不因使用
“想要”“希望”“请”之类的请求表达而标为遗漏。
当前角色对第二人称说出的祈使句，表示当前角色已经选定该句谓语所指的动作；它不是把选择权交给
当前用户。
具体动作既可以是身体行为，也可以是说出、回答、选择或发送等语言与交流行为；只要回应明确命名
希望用户完成的这类下一步，就已满足 required operation。
selection_owner_role 决定选择哪项动作，embedded_actor_role 执行已选动作。当前角色选定具体动作
并要求当前用户执行，选择仍由当前角色作出；只有要求当前用户决定要选哪项动作，才是转移选择权。
解析“当前角色希望或要求当前用户做 X”一类嵌套从句时，当前用户是 X 的行动者，当前角色是
要求和选择的所有者。

本阶段不判断回应是否充分、详细、优雅或完全覆盖 content plan，也不得以不够具体、过于简短或
未完成 required operation 为理由拒绝。issues 只能报告明确的选择所有者转移，或明确的行动者/
对象颠倒。祈使句已经命名一个动作时，不因缺少解释、步骤或额外细节而拒绝。

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
    """Project typed role tuples that require a semantic selection.

    Args:
        current_visible_percepts: Bounded model-visible episode percepts.

    Returns:
        Selection-required role tuples without semantic-completeness prose.
    """

    required_operations: list[dict[str, Any]] = []
    for percept in current_visible_percepts:
        content = percept.get("content")
        if not isinstance(content, dict):
            continue
        operation = content.get("response_operation")
        if not isinstance(operation, dict):
            continue
        if operation.get("selection_required") is not True:
            continue
        projected_operation = {
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
        }
        required_operations.append(projected_operation)
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
    trace_stage_name = (
        "dialog_role_direction_recheck"
        if post_repair
        else "dialog_role_direction_verifier"
    )
    request_messages = [system_message, human_message]
    for attempt_index in range(DIALOG_VERIFIER_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        response = await _dialog_role_direction_llm.ainvoke(
            request_messages,
            config=_dialog_role_direction_llm_config,
        )
        parsed: object = {}
        response_text = getattr(response, "content", "")
        try:
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_compliance_verdict(
                parsed,
                max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=["dialog_role_direction_verdict"],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                raise DialogVerifierContractError(
                    (
                        "role direction verifier contract exhausted: "
                        f"{exc}"
                    ),
                    error_code=(
                        "dialog_role_direction_contract_exhausted"
                    ),
                    stage="dialog.role_direction",
                ) from exc
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=str(exc),
                    issue_field_name="issues",
                    false_verdict_example=(
                        DIALOG_STRING_VERDICT_FALSE_EXAMPLE
                    ),
                ),
            ]
            continue

        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=trace_stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text=str(response_text),
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["dialog_role_direction_verdict"],
            sequence=attempt_index,
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
            prompt_chars=sum(
                len(str(message.content))
                for message in request_messages
            ),
            output_chars=len(str(response_text)),
            parse_status="succeeded",
            retry_count=attempt_index,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="info",
            correlation_id=llm_trace_id,
        )
        return verdict

    raise StateContractError("role direction verifier loop terminated")


_V2_DIALOG_SURFACE_INTEGRITY_PROMPT = '''根据候选回应和精确的 permitted_action_results 核对
能力执行事实。

以下情况将 aligned 标为 false：候选回应声称角色大脑已经完成某项系统、工具、平台或其他能力，
但 permitted_action_results 中没有匹配的 executed 结果；或者结果为 scheduled 或 pending 时，
候选回应把它写成立即执行，或保证立即反馈、立即得到结果。完成声明必须受该结果的 action_kind、
semantic_result 和 target_roles 约束。scheduled 或 pending 只支持已记录、已排队或等待对应
worker；failed 或 unavailable 不支持成功声明。单纯的言语立场、请求、邀请，以及没有即时保证的
未来、条件或假设事件都不等同于能力已经执行。

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
    trace_stage_name = (
        "dialog_surface_integrity_recheck"
        if post_repair
        else "dialog_surface_integrity_verifier"
    )
    request_messages = [system_message, human_message]
    for attempt_index in range(DIALOG_VERIFIER_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        response = await _dialog_surface_integrity_llm.ainvoke(
            request_messages,
            config=_dialog_surface_integrity_llm_config,
        )
        parsed: object = {}
        response_text = getattr(response, "content", "")
        try:
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_surface_compliance_verdict(
                parsed,
                generated_dialog=generated_dialog,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=[
                    "dialog_surface_integrity_verdict",
                ],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                raise DialogVerifierContractError(
                    (
                        "surface integrity verifier contract exhausted: "
                        f"{exc}"
                    ),
                    error_code=(
                        "dialog_surface_integrity_contract_exhausted"
                    ),
                    stage="dialog.surface_integrity",
                ) from exc
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=str(exc),
                    issue_field_name="issues",
                    false_verdict_example=(
                        DIALOG_SURFACE_VERDICT_FALSE_EXAMPLE
                    ),
                ),
            ]
            continue

        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=trace_stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text=str(response_text),
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["dialog_surface_integrity_verdict"],
            sequence=attempt_index,
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
            prompt_chars=sum(
                len(str(message.content))
                for message in request_messages
            ),
            output_chars=len(str(response_text)),
            parse_status="succeeded",
            retry_count=attempt_index,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="info",
            correlation_id=llm_trace_id,
        )
        return verdict

    raise StateContractError("surface integrity verifier loop terminated")


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
            surface_output=surface_output,
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
) -> dict[str, Any]:
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
    surface_input = global_state.get("text_surface_input_v2")
    if surface_input is not None and not isinstance(surface_input, dict):
        raise StateContractError(
            "persona state text_surface_input_v2 must be an object"
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
        subState["text_surface_input_v2"] = validate_text_surface_input(
            surface_input
        )
    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]
    accepted_surface = result.get("text_surface_output_v2")
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


def _current_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project current model-visible percepts within the shared prompt bound."""

    percepts = project_model_visible_percepts(episode)
    serialized = json.dumps(percepts, ensure_ascii=False)
    if len(serialized) > 24000:
        raise StateContractError("current visible percepts exceed dialog bounds")
    return percepts


def _validate_exact_object_fields(
    value: object,
    *,
    label: str,
    expected_fields: frozenset[str],
) -> dict[str, Any]:
    """Validate an exact JSON-object field set with actionable differences.

    Args:
        value: Parsed candidate value.
        label: Stable contract label used in protected error detail.
        expected_fields: Complete allowed and required field names.

    Returns:
        The original dictionary after exact field validation.
    """

    if not isinstance(value, dict):
        raise StateContractError(f"{label} must be an object")
    actual_fields = set(value)
    if actual_fields != expected_fields:
        missing_fields = sorted(expected_fields - actual_fields)
        unexpected_fields = sorted(
            str(field)
            for field in actual_fields - expected_fields
        )
        raise StateContractError(
            f"{label} fields are not exact: "
            f"missing={missing_fields}; unexpected={unexpected_fields}"
        )
    return value


def _validate_string_verdict(
    value: object,
    *,
    label: str,
    issue_field: str,
    max_issues: int,
) -> dict[str, Any]:
    """Validate one exact string-issue verdict and normalize its field name.

    Args:
        value: Parsed verdict candidate.
        label: Stable stage label used in protected contract errors.
        issue_field: Exact producer-owned problem-array field name.
        max_issues: Maximum accepted problem rows.

    Returns:
        Internal verdict using the canonical `aligned` and `issues` fields.
    """

    verdict = _validate_exact_object_fields(
        value,
        label=label,
        expected_fields=frozenset({"aligned", issue_field}),
    )
    aligned = verdict["aligned"]
    issues = verdict[issue_field]
    if not isinstance(aligned, bool):
        raise StateContractError(f"{label} aligned must be boolean")
    if not isinstance(issues, list) or len(issues) > max_issues:
        raise StateContractError(f"{label} issues are invalid")
    if len(issues) != len(set(issues)):
        raise StateContractError(f"{label} issues are duplicated")
    if any(
        not isinstance(issue, str)
        or not issue.strip()
        or len(issue) > 300
        for issue in issues
    ):
        raise StateContractError(f"{label} issue text is invalid")
    if aligned and issues:
        raise StateContractError(f"aligned {label} cannot contain issues")
    if not aligned and not issues:
        raise StateContractError(f"misaligned {label} requires issues")
    validated_verdict = {
        "aligned": aligned,
        "issues": list(issues),
    }
    return validated_verdict


def _validate_semantic_fidelity_verdict(
    value: object,
    *,
    max_issues: int,
) -> dict[str, Any]:
    """Validate the semantic producer's collision-resistant output shape."""

    validated_verdict = _validate_string_verdict(
        value,
        label="dialog semantic fidelity",
        issue_field="hard_errors",
        max_issues=max_issues,
    )
    return validated_verdict


def _validate_compliance_verdict(
    value: object,
    *,
    max_issues: int,
) -> dict[str, Any]:
    """Validate the internal and role-direction compliance shape."""

    validated_verdict = _validate_string_verdict(
        value,
        label="dialog compliance",
        issue_field="issues",
        max_issues=max_issues,
    )
    return validated_verdict


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

    verdict = _validate_exact_object_fields(
        value,
        label="surface compliance",
        expected_fields=frozenset({"aligned", "issues"}),
    )
    aligned = verdict["aligned"]
    issues = verdict["issues"]
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
        validated_issue = _validate_exact_object_fields(
            issue,
            label="surface issue",
            expected_fields=frozenset({
                "kind",
                "evidence",
                "explanation",
            }),
        )
        kind = validated_issue["kind"]
        evidence = validated_issue["evidence"]
        explanation = validated_issue["explanation"]
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
