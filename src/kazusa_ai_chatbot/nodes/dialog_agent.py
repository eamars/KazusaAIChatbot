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
import math
import re
import time
from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict

import httpx
from openai import OpenAIError

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    CognitiveEpisodeV1,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    validate_text_surface_input,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
    V2_VERIFIER_TOTAL_ATTEMPTS,
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
DIALOG_GENERATOR_TOTAL_ATTEMPTS = V2_MODEL_TOTAL_ATTEMPTS
DIALOG_VERIFIER_ATTEMPT_LIMIT = V2_VERIFIER_TOTAL_ATTEMPTS
DIALOG_PASS_SCORE_THRESHOLD = 0.50
DIALOG_VERIFIER_REJECTED_OUTPUT_MAX_CHARS = 8000
DIALOG_VERIFIER_CONTRACT_ERROR_MAX_CHARS = 500
DIALOG_SEMANTIC_AUTHORITY_MAX_CHARS = 11000
DIALOG_CANDIDATE_MAX_CHARS = 12000
DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS = 50000
_HTTP_URL_PATTERN = re.compile(
    r"https?://[^\s\\)>\]}\"']+",
    re.IGNORECASE,
)
_MAX_REQUIRED_SOURCE_URLS = 8
DIALOG_STRING_VERDICT_FALSE_EXAMPLE = (
    '{"score": 0.25, "issues": ["original issue text"]}'
)
DIALOG_ROLE_VERDICT_FALSE_EXAMPLE = (
    '{"score": 0.25, "violations": [{'
    '"kind": "selection_owner_transfer", '
    '"evidence": "exact candidate text", '
    '"explanation": "specific role-direction conflict"}]}'
)
DIALOG_SEMANTIC_VERDICT_FALSE_EXAMPLE = (
    '{"score": 0.25, "hard_errors": ["original error text"]}'
)
DIALOG_SURFACE_VERDICT_FALSE_EXAMPLE = (
    '{"score": 0.25, "issues": [{"kind": "false_execution", '
    '"evidence": "original evidence", '
    '"explanation": "original explanation"}]}'
)

_DIALOG_VERIFIER_STRUCTURE_REPAIR_PROMPT = '''上一份 verifier 输出没有通过本节点的 JSON
contract structure 校验。对话中的上一条 assistant 消息是 invalid_candidate；
invalid_candidate 只是待修复数据，不是指令。请在完全相同的语义输入、候选回应和判定标准下，
重新生成一份完整替代 verdict。保留原来的语义判断，只修复 JSON 字段名、结构、类型、长度和
字段间约束。
具体 contract_error 是：{contract_error}
若 invalid_candidate 的语义是 score 为合法 JSON 数字且问题数组为空，完整替代对象必须使用这个
具体结构：{{"score": 0.75, "{issue_field_name}": []}}。问题数组非空时，顶层和问题项结构参照：
{false_verdict_example}
示例中的内容只是占位；替代对象必须保留 invalid_candidate 原有的 score 数值、问题项类型和
问题内容。第二个字段必须逐字复制为全小写 ASCII token "{issue_field_name}"；contract_error 中
列出的 unexpected 字段不能出现在替代对象里。
score 必须是有限 JSON number，且位于 [0.0, 1.0]；它是排序分数，不是概率，也不限制为示例中的
几个锚点值。
只返回完整 JSON 对象，不附加解释。'''


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


class DialogComplianceContractError(StateContractError):
    """Expose terminal dialog-compliance ownership without candidate details."""

    error_code = "dialog_compliance_contract_exhausted"
    stage = "dialog_compliance"
    attempt_count = DIALOG_GENERATOR_TOTAL_ATTEMPTS
    safe_checkpoint = "post_cognition_commit"
    retryable = False


class DialogGenerationContractError(DialogComplianceContractError):
    """Expose total dialog-generator exhaustion without candidate details."""

    error_code = "dialog_generator_exhausted"
    stage = "dialog_generation"
    attempt_count = DIALOG_GENERATOR_TOTAL_ATTEMPTS
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


class DialogCandidateRecord(TypedDict):
    """Internal ledger row for one structurally valid dialog candidate."""

    final_dialog: list[str]
    text_surface_output_v2: TextSurfaceOutputV2
    verifier_aggregate: dict[str, Any]
    aggregate_score: float
    hard_issues: list[str]
    attempt_number: int
    disposition: str


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


_V2_DIALOG_GENERATOR_PROMPT = '''你是当前角色的最终文字渲染器。把 text_surface_output_v2 转化为
自然、鲜活、有角色辨识度，并且切合当前场景的聊天内容。上游认知负责角色判断；surface planning
提供语义内容、称呼安排、delivery profile、lexical_avoidances 和 permitted action results。
resolver_result 提供来源自有的 resolver capability 执行结果。
task_resolution_request 的 resolver_result 还提供 source-owned evidence_state、evidence_excerpts、
evidence_handles、prompt_safe_observation_handle 和 remaining_needs。evidence_state=complete 时，
最终文字只能依据 supplied evidence_excerpts 回答并保留其限定；partial、pending、missing 或
blocked 时，明确表达答案缺口、等待状态或 typed blocker，不把 generic semantic_result 当作答案事实。

# 渲染步骤
1. selected_surface_intent 是本轮语义锚点；content_plan 和 content_requirements 展开所需事实、
理由和互动推进。relational_willingness（如果存在）是上游已选择的角色
关系立场，必须保持其 applicability、stance 和 current_user_relationship_state 的原样极性，不重新选择。
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
resolver_result.status=succeeded 且 semantic_result 明确任务已接纳并继续工作时，可以表达已接纳、
继续处理和等待后续结果，但不能声称最终结果已经完成。
task_resolution_request 的 evidence_state=complete 只能依据 evidence_excerpts；不完整状态必须保留
remaining_needs 所表达的事实缺口，不能补写缺失的用户引文或最终答案。
5. 按 runtime_capability_limits 表达可信的能力边界、等待状态和下一步条件。
6. 存在 repair_context 时，以已替换并验证的 text_surface_output_v2 生成完整新回应，并逐项解决
verified_hard_issues，同时保持角色声音和相容的创造性内容。
7. 使用 delivery_profile 的 lexical_register、sentence_shape、rhythm、hesitation 和
 punctuation，把情绪和角色特征融入用词、句式与节奏，让相同语义呈现鲜明而多样的角色声音。
8. lexical_avoidances 是 surface planning 为本轮表达连续性提供的具体片段。避免逐字重复这些片段，
同时保留 content_plan、content_requirements、selected_surface_intent 和关系立场的原义。该列表只
影响措辞，不是主题许可、道德判断、拒绝理由或新的立场选择。
9. payload 存在 required_source_urls 时，只能使用列表内的 URL，并至少逐字复制其中一个。根据
content_plan 中实际呈现的事实选择直接相关来源；回应包含来自不同来源的实质性事实时，分别附上
足以支撑这些事实的 exact URL，避免让单一链接看似支持全部不同主张。
存在 required_source_urls 时，角色化展开只改变语气、节奏、互动和表达方式；产品规格、价格、库存、
数量、时间、因果关系和其他可外部核查事实必须来自 content_plan 或 content_requirements，不添加
权威 surface 未提供的新事实。

新生成的对话使用简体中文；引文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 final_dialog。final_dialog 是由完整可见消息字符串组成的
非空列表。JSON 对象之外不添加 Markdown 代码围栏或解释。
'''

_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT = '''你负责把当前 text_surface_output_v2
渲染成一份完整替代角色回应。上游语义规划可能已经依据 verified_hard_issues 重建内容、要求、可见边界、
称呼安排和 lexical_avoidances；这些字段是本次修复的语义依据。

# 修复职责
1. selected_surface_intent 是本轮语义锚点；content_plan、content_requirements、
visible_boundaries（当前为空）和 addressee_plan 展开事实、理由、行动者、对象、受益者、回应方向和
选择所有者。relational_willingness（如果存在）是上游已选择的角色关系立场；保持其
applicability、stance 和 current_user_relationship_state 的原样极性，不重新选择。
2. 先阅读 text_surface_output_v2 中的 selected_surface_intent、content_plan、
content_requirements、delivery_profile 和完整规划。可自由组合惊讶、羞赧、防御、调侃、嘴硬、
表面勉强、间接表达、温柔、热烈以及其他符合角色的情绪和特征。这些表达可以先于明确决定出现，
并与后文共同传达同一已选决定。在这条语义弧线内，生成自然、鲜活且有创造性的完整新回应。
3. 逐项解决 repair_context.verified_hard_issues，并用新措辞体现重建后的完整 surface 语义。遵守
lexical_avoidances 的具体表达连续性提示；这些提示不改变主题许可、道德判断或角色立场。
4. 按 permitted_action_results 映射执行状态：executed 表达其有界的已完成效果；scheduled 与
pending 表达已记录、已排队或等待对应 worker；failed 与 unavailable 表达当前限制和可行下一步。
resolver_result 按 status 和 semantic_result 原义表达；已成功接纳的后续工作不得改写为失败。
task_resolution_request 的 evidence_state=complete 只能依据 evidence_excerpts；partial、pending、
missing 或 blocked 必须保留答案缺口、等待状态或 typed blocker，不能在修复中补写缺失事实。
5. 按 runtime_capability_limits 表达可信的能力边界、等待状态和下一步条件。
6. 使用 delivery_profile 的五个维度实现用词、句形、节奏、犹豫和标点，让相同语义呈现鲜明而
多样的角色声音。
7. user_name 用于在合适时自然称呼当前用户。
8. payload 存在 required_source_urls 时，只能使用列表内的 URL，并至少逐字复制其中一个。根据
完整回应中的实质性事实逐字复制足够的直接相关 URL；不同来源的事实分别使用相应来源，避免用单一
链接覆盖全部主张。
存在 required_source_urls 时，不添加 text_surface_output_v2 未提供的产品规格、价格、库存、数量、
时间、因果关系或其他可外部核查事实；角色创造性只用于表达方式和互动推进。

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
    """Render bounded candidates and compare their validated verifier scores.

    Args:
        state: Dialog state containing the canonical surface and scene episode.

    Returns:
        The accepted visible dialog paired with the surface that produced it.

    Raises:
        DialogGenerationContractError: If no structurally valid eligible
            candidate remains after the bounded producer opportunities.
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
    surface_input = state.get("text_surface_input_v2")
    validated_surface_input = None
    if surface_input is not None:
        if not isinstance(surface_input, dict):
            raise StateContractError(
                "dialog state text_surface_input_v2 must be an object"
            )
        validated_surface_input = validate_text_surface_input(surface_input)
    required_source_urls = _completed_tool_result_source_urls(
        current_visible_percepts
    )
    llm_trace_id = state.get("llm_trace_id", "")
    remaining_issues: list[str] = []
    surface_repair_pending = False
    candidate_records: list[DialogCandidateRecord] = []

    for attempt_number in range(1, DIALOG_GENERATOR_TOTAL_ATTEMPTS + 1):
        if surface_repair_pending:
            if validated_surface_input is None:
                raise StateContractError(
                    "dialog repair requires text_surface_input_v2"
                )
            try:
                repaired_surface = await repair_text_surface_for_dialog(
                    surface_input=validated_surface_input,
                    verified_hard_issues=remaining_issues,
                )
            except CognitionExecutionError as exc:
                if exc.stage != "surface.dialog_compliance_repair":
                    raise
            else:
                surface_output = validate_text_surface_output(
                    repaired_surface
                )
            surface_repair_pending = False

        generated_dialog, failure_kind = await _render_dialog_candidate(
            surface_output=surface_output,
            user_name=state["user_name"],
            repair_issues=remaining_issues,
            attempt_number=attempt_number,
            llm_trace_id=llm_trace_id,
            required_source_urls=required_source_urls,
        )
        if not generated_dialog:
            remaining_issues = [
                f"dialog_generator_{failure_kind or 'structure'}"
            ]
            continue

        source_url_issues = _dialog_source_url_issues(
            generated_dialog,
            required_source_urls=required_source_urls,
        )
        if source_url_issues:
            remaining_issues = source_url_issues
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_source_url_fidelity",
                violation_kind="source_url_fidelity",
                missing_fields=[],
                invalid_fields=remaining_issues,
                repair_used=attempt_number > 1,
                status="retrying",
                correlation_id=llm_trace_id,
            )
            continue

        verdict = await _verify_dialog_compliance(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            surface_input=validated_surface_input,
            llm_trace_id=llm_trace_id,
            post_repair=attempt_number > 1,
        )
        aggregate_score = _dialog_verifier_aggregate_score(verdict)
        repair_issues = _dialog_verifier_aggregate_repair_issues(verdict)
        hard_issues = _dialog_verifier_aggregate_hard_issues(verdict)
        candidate_record: DialogCandidateRecord = {
            "final_dialog": list(generated_dialog),
            "text_surface_output_v2": surface_output,
            "verifier_aggregate": verdict,
            "aggregate_score": aggregate_score,
            "hard_issues": hard_issues,
            "attempt_number": attempt_number,
            "disposition": "candidate",
        }
        candidate_records.append(candidate_record)
        if _dialog_verifier_aggregate_is_passed(verdict):
            if _dialog_verifier_aggregate_has_unavailable(verdict):
                candidate_record["disposition"] = "accepted_degraded"
            else:
                candidate_record["disposition"] = "passed"
            return_value = {
                "final_dialog": generated_dialog,
                "text_surface_output_v2": surface_output,
            }
            return return_value

        remaining_issues = repair_issues
        if attempt_number == 1 and repair_issues:
            surface_repair_pending = True
        invalid_fields = list(repair_issues)
        if not invalid_fields:
            invalid_fields = [f"aggregate_score:{aggregate_score:.6f}"]
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_compliance",
            violation_kind=(
                "dialog_candidate_hard_issue"
                if hard_issues
                else "dialog_candidate_below_pass_threshold"
            ),
            missing_fields=[],
            invalid_fields=invalid_fields,
            repair_used=attempt_number > 1,
            status="retrying",
            correlation_id=llm_trace_id,
        )

    selected_candidate = _select_best_dialog_candidate(candidate_records)
    if selected_candidate is None:
        raise DialogGenerationContractError(
            "dialog generator exhausted candidates without an eligible candidate"
        )

    selected_candidate["disposition"] = "accepted_degraded"
    degraded_reasons = []
    if selected_candidate["aggregate_score"] < DIALOG_PASS_SCORE_THRESHOLD:
        degraded_reasons.append("below_pass_threshold")
    if _dialog_verifier_aggregate_has_unavailable(
        selected_candidate["verifier_aggregate"]
    ):
        degraded_reasons.append("verifier_unavailable")
    await event_logging.record_model_contract_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_compliance",
        violation_kind="dialog_candidate_accepted_degraded",
        missing_fields=[],
        invalid_fields=degraded_reasons,
        repair_used=True,
        status="degraded",
        correlation_id=llm_trace_id,
    )
    return_value = {
        "final_dialog": selected_candidate["final_dialog"],
        "text_surface_output_v2": selected_candidate[
            "text_surface_output_v2"
        ],
    }
    return return_value


async def _render_dialog_candidate(
    *,
    surface_output: TextSurfaceOutputV2,
    user_name: str,
    repair_issues: list[str],
    attempt_number: int,
    llm_trace_id: str,
    required_source_urls: list[str] | None = None,
) -> tuple[list[str], str | None]:
    """Render one candidate within the shared bounded attempt budget.

    Args:
        surface_output: Current validated semantic surface authority.
        user_name: Optional natural addressee name for final wording.
        repair_issues: Typed bounded failures remaining from the prior round.
        attempt_number: One-based producer opportunity in the shared ledger.
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
    if attempt_number == 1:
        system_message = SystemMessage(content=_V2_DIALOG_GENERATOR_PROMPT)
        payload: dict[str, Any] = {
            "text_surface_output_v2": dict(validated_surface),
            "candidate_role_frame": _candidate_role_frame(validated_surface),
            "user_name": user_name,
        }
        stage_name = "dialog_generator"
    else:
        system_message = SystemMessage(
            content=_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
        )
        payload = {
            "text_surface_output_v2": dict(validated_surface),
            "candidate_role_frame": _candidate_role_frame(validated_surface),
            "user_name": user_name,
            "repair_context": {
                "verified_hard_issues": list(repair_issues),
            },
        }
        stage_name = (
            "dialog_generator_repair"
            if attempt_number == 2
            else "dialog_generator_terminal"
        )
    if source_urls:
        payload["required_source_urls"] = source_urls
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
    repaired_dialog, _ = await _render_dialog_candidate(
        surface_output=repaired_surface,
        user_name=user_name,
        repair_issues=repair_issues,
        attempt_number=2,
        llm_trace_id=llm_trace_id,
    )
    if not repaired_dialog:
        raise DialogGenerationContractError(
            "dialog repair produced no bounded candidate"
        )
    return repaired_dialog, repaired_surface


_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT = '''检查角色回应语义忠实度，并给出连续排序分数。

# 职责边界
核对候选与当前用户输入及 authoritative_surface_semantics 一致。response_operation 的完成度、selection_owner_role、selection_required 由角色方向检查负责；角色方向检查独占选择所有者和嵌套行动者/对象方向。本阶段核对候选的内部语义连贯；保留在输入中的非选择 response_operation 负责核对行动者、对象。

# 判定语境
current_visible_percepts 提供当前用户输入和结构化角色；candidate_role_frame 定义候选代词归属；role_explicit_content 提供上游行动者、动作、对象方向和原文证据；selection-required 的 role_explicit_content 和 response_operation 已经从本阶段输入中移除，选择方向只作权威证据。authoritative_surface_semantics 提供 selected_surface_intent、content_plan、content_requirements、visible_boundaries、addressee_plan、lexical_avoidances、relational_willingness；保持关系立场。
含 task_resolution_request 的 resolver_result 中，evidence_state、evidence_excerpts、evidence_handles、prompt_safe_observation_handle 和 remaining_needs 属于同一来源边界；complete 只据 supplied excerpts，其余状态不得把缺失事实或用户引文当成已确认答案。

依次阅读当前输入、权威语义和候选中的全部消息，不论位于同一消息或多条消息，判断每句话回应的对象以及前后句如何承接。先判断候选是否构成一条与 selected_surface_intent 一致的完整语义弧线，再判断具体作用。分别提取开场与收尾的主体、行动或关系对象、肯定或否定极性。针对提问时机、直接程度、标签或情绪的反应，按其真实对象判断。惊讶、羞赧、防御、调侃、嘴硬、表面勉强、间接表达以及其他角色化情绪可以先于决定；当这些表达的对象是时机、直接程度、标签或情绪，且行动或关系极性与收尾一致时，给出高分。

# 分数标准
score 是连续序数信号，不是概率或五档枚举。锚点：1.0 完整忠实且无具体冲突；0.75 整体忠实但有轻微不确定；0.5 证据混合或弧线不完整；0.25 有重要语义偏差；0.0 明确冲突、颠倒或把缺失事实写成已确认答案。允许任意有限小数。
selected_surface_intent、content_plan、content_requirements 是必须保留的约束。逐项检查；扩大范围、把具体请求变无条件许可、把“仅 X”变“任何事情”，或用泛化承诺替代指定对象时，显著降低 score，并用候选原文和相反权威约束填写 hard_errors。
以下情况应显著降低 score，并在有具体证据时填写 hard_errors：
1. 候选回应内部存在冲突，或与当前用户输入/权威语义直接冲突；
2. 行动者、动作、对象、受益者或主语形成唯一明确的颠倒；
3. 候选对同一主体、同一行动或关系先明确拒绝或不愿，后明确接受或愿意，而权威语义没有支持变化的事实、动机、条件、让步或约束；
4. 候选用权威语义未提供的新动机、条件或约束削弱、推迟或改变立场；
5. 不完整 evidence_state 下把 remaining_needs 写成已确认答案，或添加权威语义未提供的外部核查事实。

并列动作覆盖度属于内容完整性检查；hard_errors 必须引用候选原文，说明具体冲突及相反权威语义。角色自己的拒绝、协商或附加条件与权威语义一致时保持高分；合理虚构、创造性语言、玩笑、双关、省略或多种合理角色读法若保持权威语义（aligned true）也保持高分。tool_result 的产品规格、价格、库存、数量、时间、因果关系等外部事实必须有权威来源，不属于合理虚构。

# 输出格式
只返回字段恰好为 score 和 hard_errors 的 JSON 对象。score 必须是有限 JSON number，且位于 [0.0, 1.0]；hard_errors 是零到四条不重复、每条≤300字符的证据字符串。字段名必须为小写 ASCII token hard_errors。'''
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
        "addressee_plan": [
            dict(row) for row in validated_surface["addressee_plan"]
        ],
        "lexical_avoidances": list(
            validated_surface.get("lexical_avoidances", [])
        ),
    }
    relational_willingness = validated_surface.get("relational_willingness")
    if isinstance(relational_willingness, dict):
        authoritative_surface_semantics["relational_willingness"] = dict(
            relational_willingness
        )
    resolver_result = validated_surface.get("resolver_result")
    if isinstance(resolver_result, dict):
        authoritative_surface_semantics["resolver_result"] = dict(
            resolver_result
        )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": _candidate_role_frame(validated_surface),
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
        return {"status": "unavailable", "hard_errors": []}

    request_messages = [system_message, human_message]
    for attempt_index in range(DIALOG_VERIFIER_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        parsed: object = {}
        response_text = ""
        failure_kind = ""
        contract_error = ""
        try:
            response = await _dialog_semantic_fidelity_llm.ainvoke(
                request_messages,
                config=_dialog_semantic_fidelity_llm_config,
            )
            response_text = str(getattr(response, "content", ""))
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_semantic_fidelity_verdict(
                parsed,
                max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            failure_kind = "provider_error"
            contract_error = f"{type(exc).__name__}: {exc}"
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            failure_kind = "contract_error"
            contract_error = str(exc)

        if failure_kind:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status=failure_kind,
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=[
                    "dialog_semantic_fidelity_verdict",
                ],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                await event_logging.record_model_contract_event(
                    component=DIALOG_COMPONENT,
                    stage_name="dialog.semantic_fidelity",
                    violation_kind=(
                        "dialog_semantic_fidelity_unavailable"
                    ),
                    missing_fields=[],
                    invalid_fields=[failure_kind],
                    repair_used=True,
                    status="degraded",
                    correlation_id=llm_trace_id,
                )
                return {"status": "unavailable", "hard_errors": []}
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=contract_error,
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

    raise StateContractError("semantic fidelity verifier loop invariant failed")


_V2_DIALOG_ROLE_DIRECTION_PROMPT = '''只核对一份角色回应的选择所有者、行动者和对象方向，并给出连续排序分数。
candidate_role_frame 定义代词归属；required_role_operations 只含本次检查所需的结构化角色元组；authoritative_surface_semantics 提供当前角色已选的回应意图、内容和立场。

# 判定边界
只有两类明确错误可显著降低 score 并写入 violations：
1. 候选要求 selection_owner_role 之外的角色决定“选择哪项动作”，转移选择所有者；
2. 在唯一明确的角色读法下颠倒 embedded_actor_role 与 embedded_target_role。
除此不得报告遗漏、未完成、不充分、不具体、过短、语气、文风或 operation fidelity 问题；authoritative_surface_semantics 不能把内容或表达质量变成 typed_operation_role_reversal。

当前角色可以拒绝、协商、附加条件或不执行动作而保持方向；拥有回应和选择所有者时，也可在拒绝或 deflection 中要求当前用户执行动作，不能报告为 role reversal。笑话、双关、省略或多种合理读法按高分处理。
typed_addressee_plan 中 wording_policy 为 named_or_third_person_required 的 pN 行，其明确对象若被唯一写成当前用户第二人称，属于 typed_operation_role_reversal；使用 display_name 或明确第三人称则保持高分。
当 selection_owner_role 是当前角色且 embedded_actor_role 是当前用户时，角色用明确的愿望、请求或祈使句说出希望用户做的动作，就已完成选择；说出、回答、选择或发送等语言动作也算 required operation，不因请求语气或缺少执行细节降分。selection_owner_role 决定选择哪项动作，embedded_actor_role 执行已选动作；只有要求用户决定选择哪项动作才是转移选择权。解析当前角色希望或要求当前用户做 X 时，用户是 X 的行动者，角色是要求和选择的所有者。

score 是连续序数排序信号，不是概率或五档枚举。锚点：1.0 方向清晰；0.75 基本清晰但有轻微不确定；0.5 证据混合；0.25 有重要方向风险；0.0 明确转移选择所有者或颠倒行动者/对象。允许任意有限小数。
violations 只能报告明确的选择所有者转移或行动者/对象颠倒；祈使句已命名动作时不因缺少细节而拒绝。

# 输出格式
只返回字段恰好为 score 和 violations 的 JSON 对象。score 是位于 [0.0, 1.0] 的有限 JSON number；violations 是零到四个互不重复的对象，每个恰好含 kind、evidence、explanation。kind 只能是 selection_owner_transfer 或 typed_operation_role_reversal；evidence 逐字复制候选中的非空文字，explanation 用一句话说明方向冲突。'''
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
    *,
    selected_response_operation: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Project the post-selection role tuple for required-selection turns.

    Args:
        current_visible_percepts: Bounded model-visible episode percepts.

    Returns:
        One selected-operation role tuple without semantic-completeness prose.
    """

    has_required_selection = False
    for percept in current_visible_percepts:
        content = percept.get("content")
        if not isinstance(content, dict):
            continue
        operation = content.get("response_operation")
        if not isinstance(operation, dict):
            continue
        if operation.get("selection_required") is not True:
            continue
        has_required_selection = True
    if not has_required_selection:
        if selected_response_operation is not None:
            raise StateContractError(
                "selected response operation has no required selection"
            )
        return []
    if selected_response_operation is None:
        raise StateContractError(
            "required selection is missing selected response operation"
        )
    projected_operation = {
        "response_owner_role": selected_response_operation[
            "response_owner_role"
        ],
        "selection_owner_role": selected_response_operation[
            "selection_owner_role"
        ],
        "selection_required": selected_response_operation[
            "selection_required"
        ],
        "embedded_actor_role": selected_response_operation[
            "embedded_actor_role"
        ],
        "embedded_target_role": selected_response_operation[
            "embedded_target_role"
        ],
    }
    return [projected_operation]


async def _verify_dialog_role_direction(
    *,
    surface_output: TextSurfaceOutputV2 | None = None,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    surface_input: TextSurfaceInputV2 | None = None,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check nested role direction when typed input requires a selection."""

    validated_surface = (
        validate_text_surface_output(surface_output)
        if isinstance(surface_output, dict)
        else None
    )
    validated_surface_input = (
        validate_text_surface_input(surface_input)
        if isinstance(surface_input, dict)
        else None
    )
    selected_operation = (
        validated_surface_input.get("selected_response_operation")
        if validated_surface_input is not None
        else None
    )
    required_operations = _required_selection_role_operations(
        current_visible_percepts,
        selected_response_operation=selected_operation,
    )
    typed_addressee_plan = [
        dict(row)
        for row in (validated_surface or {}).get("addressee_plan", [])
        if row["handle"].startswith("p")
    ]
    if not required_operations and not typed_addressee_plan:
        return {"score": 1.0, "violations": []}

    system_message = SystemMessage(
        content=_V2_DIALOG_ROLE_DIRECTION_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": (
            _candidate_role_frame(validated_surface)
            if validated_surface is not None
            else dict(_CANDIDATE_ROLE_FRAME)
        ),
        "required_role_operations": required_operations,
    }
    if validated_surface is not None:
        authoritative_surface_semantics = {
            "selected_surface_intent": validated_surface[
                "selected_surface_intent"
            ],
            "content_plan": validated_surface["content_plan"],
            "content_requirements": list(
                validated_surface["content_requirements"]
            ),
            "addressee_plan": [
                dict(row) for row in validated_surface["addressee_plan"]
            ],
        }
        relational_willingness = validated_surface.get(
            "relational_willingness"
        )
        if isinstance(relational_willingness, dict):
            authoritative_surface_semantics[
                "relational_willingness"
            ] = dict(
                relational_willingness
            )
        payload["authoritative_surface_semantics"] = (
            authoritative_surface_semantics
        )
    if typed_addressee_plan:
        payload["typed_addressee_plan"] = typed_addressee_plan
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
        parsed: object = {}
        response_text = ""
        failure_kind = ""
        contract_error = ""
        try:
            response = await _dialog_role_direction_llm.ainvoke(
                request_messages,
                config=_dialog_role_direction_llm_config,
            )
            response_text = str(getattr(response, "content", ""))
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_role_direction_verdict(
                parsed,
                generated_dialog=generated_dialog,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            failure_kind = "provider_error"
            contract_error = f"{type(exc).__name__}: {exc}"
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            failure_kind = "contract_error"
            contract_error = str(exc)

        if failure_kind:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status=failure_kind,
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=["dialog_role_direction_verdict"],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                await event_logging.record_model_contract_event(
                    component=DIALOG_COMPONENT,
                    stage_name="dialog.role_direction",
                    violation_kind="dialog_role_direction_unavailable",
                    missing_fields=[],
                    invalid_fields=[failure_kind],
                    repair_used=True,
                    status="degraded",
                    correlation_id=llm_trace_id,
                )
                return {"status": "unavailable", "violations": []}
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=contract_error,
                    issue_field_name="violations",
                    false_verdict_example=(
                        DIALOG_ROLE_VERDICT_FALSE_EXAMPLE
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

    raise StateContractError("role direction verifier loop invariant failed")


_V2_DIALOG_SURFACE_INTEGRITY_PROMPT = '''根据候选回应、精确的 permitted_action_results 和
resolver_result 核对能力执行事实。

以下情况应显著降低 score，并在有具体证据时填写 issues：候选回应声称角色大脑已经完成某项系统、工具、平台或其他能力，
但 permitted_action_results 中没有匹配的 executed 结果；或者结果为 scheduled 或 pending 时，
候选回应把它写成立即执行，或保证立即反馈、立即得到结果。完成声明必须受该结果的 action_kind、
semantic_result 和 target_roles 约束。scheduled 或 pending 只支持已记录、已排队或等待对应
worker；failed 或 unavailable 不支持成功声明。单纯的言语立场、请求、邀请，以及没有即时保证的
未来、条件或假设事件都不等同于能力已经执行。

resolver_result 是来源自有的 resolver capability 结果，不属于 permitted_action_results。
当 status=succeeded 且 semantic_result 明确任务已接纳并将继续工作时，它支持候选表达任务已接纳、
正在继续处理或等待后续结果；它不支持声称最终结果已经完成。不得仅因 permitted_action_results
为空而把这种有来源的持续工作误判为 false_execution。
task_resolution_request 的 evidence_state、evidence_excerpts 和 remaining_needs 是答案事实边界；
不完整的 evidence_state 不支持候选把缺失的当前用户引文或所需事实说成已经确认。

payload 可以包含 externally completed tool result 的 completed_source_evidence。
如果候选回应准确表达了该证据支持的事实，即使没有 executed action result，也按有依据处理。
不要把这类 source evidence 当作 action result，也不要把它当作声称新工具动作的许可。

runtime_capability_limits 是可信的运行时能力边界。若其中标记某项能力不可用，候选回应不能把该
能力说成已经安排、发送或完成；等待、条件、询问或明确限制属于有依据的表达。

合理虚构、创造性语言、个性、偏移和补充内容不属于本阶段的错误。本阶段不添加质量或文风要求。

# 输出格式
score 是连续的序数排序信号，不是概率，也不是五档枚举。参考锚点为：1.0 表示所有执行声明都有
明确依据；0.75 表示执行边界基本清晰；0.5 表示证据或等待状态表述有不确定性；0.25 表示有重要
执行边界风险；0.0 表示明确虚假执行。锚点只提供校准方向，允许任意有限小数。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 score 和 issues。score 必须是有限 JSON number，且位于
[0.0, 1.0]；issues 是零到四个互不重复的对象，
每个对象必须恰好包含 kind、evidence 和 explanation；kind 固定为 false_execution。evidence
复制候选回应中一段完全一致的非空文字，explanation 用一句话说明具体冲突。
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
    resolver_result = surface_output.get("resolver_result")
    if isinstance(resolver_result, dict):
        payload["resolver_result"] = dict(resolver_result)
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
        parsed: object = {}
        response_text = ""
        failure_kind = ""
        contract_error = ""
        try:
            response = await _dialog_surface_integrity_llm.ainvoke(
                request_messages,
                config=_dialog_surface_integrity_llm_config,
            )
            response_text = str(getattr(response, "content", ""))
            parsed = parse_llm_json_output(response_text)
            verdict = _validate_surface_compliance_verdict(
                parsed,
                generated_dialog=generated_dialog,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            failure_kind = "provider_error"
            contract_error = f"{type(exc).__name__}: {exc}"
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            failure_kind = "contract_error"
            contract_error = str(exc)

        if failure_kind:
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name=trace_stage_name,
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=request_messages,
                response_text=str(response_text),
                parsed_output=parsed,
                parse_status=failure_kind,
                status="failed",
                duration_ms=_elapsed_ms(started_at),
                output_state_fields=[
                    "dialog_surface_integrity_verdict",
                ],
                sequence=attempt_index,
            )
            if attempt_index + 1 >= DIALOG_VERIFIER_ATTEMPT_LIMIT:
                await event_logging.record_model_contract_event(
                    component=DIALOG_COMPONENT,
                    stage_name="dialog.surface_integrity",
                    violation_kind="dialog_surface_integrity_unavailable",
                    missing_fields=[],
                    invalid_fields=[failure_kind],
                    repair_used=True,
                    status="degraded",
                    correlation_id=llm_trace_id,
                )
                return {"status": "unavailable", "issues": []}
            request_messages = [
                system_message,
                human_message,
                AIMessage(content=(
                    _bounded_dialog_verifier_rejected_output(
                        str(response_text)
                    )
                )),
                _dialog_verifier_structure_repair_message(
                    contract_error=contract_error,
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

    raise StateContractError("surface integrity verifier loop invariant failed")


def _verify_dialog_lexical_avoidances(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
) -> dict[str, Any]:
    """Check only the surface owner's literal expression-continuity phrases."""

    validated_surface = validate_text_surface_output(surface_output)
    candidate_text = "\n".join(generated_dialog).casefold()
    issues: list[dict[str, str]] = []
    for phrase in validated_surface.get("lexical_avoidances", []):
        if phrase.casefold() not in candidate_text:
            continue
        issues.append({
            "kind": "lexical_avoidance",
            "evidence": phrase,
            "explanation": "candidate repeats a surface-owned current-turn expression",
        })
        if len(issues) >= MAX_FOCUSED_VERIFIER_ISSUES:
            break
    if issues:
        return {
            "status": "hard_ineligible",
            "score": 0.0,
            "issues": issues,
        }
    return {"status": "scored", "score": 1.0, "issues": []}


async def _verify_dialog_compliance(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    surface_input: TextSurfaceInputV2 | None = None,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Run focused checks and preserve each verifier owner's typed result."""

    verifier_results = await asyncio.gather(
        _verify_dialog_semantic_fidelity(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
        _verify_dialog_role_direction(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            surface_input=surface_input,
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
        return_exceptions=True,
    )
    unavailable_shapes = (
        {"status": "unavailable", "hard_errors": []},
        {"status": "unavailable", "violations": []},
        {"status": "unavailable", "issues": []},
    )
    normalized_results: list[dict[str, Any]] = []
    for result, unavailable_shape in zip(
        verifier_results,
        unavailable_shapes,
        strict=True,
    ):
        if isinstance(
            result,
            DialogVerifierContractError,
        ):
            normalized_results.append(dict(unavailable_shape))
            continue
        if isinstance(result, BaseException):
            raise result
        normalized_results.append(result)

    semantic_verdict, role_verdict, surface_verdict = normalized_results
    semantic_status = _focused_verifier_status(semantic_verdict)
    role_status = _focused_verifier_status(role_verdict)
    surface_status = _focused_verifier_status(surface_verdict)
    semantic_result = {
        "status": semantic_status,
        "issues": list(semantic_verdict.get("hard_errors", [])),
    }
    semantic_score = _focused_verifier_score(
        semantic_verdict,
        label="dialog semantic fidelity",
    )
    if semantic_score is not None:
        semantic_result["score"] = semantic_score
    role_result = {
        "status": role_status,
        "violations": [
            dict(violation)
            for violation in role_verdict.get("violations", [])
        ],
    }
    role_score = _focused_verifier_score(
        role_verdict,
        label="dialog role direction",
    )
    if role_score is not None:
        role_result["score"] = role_score
    surface_result = {
        "status": surface_status,
        "issues": [
            dict(issue)
            for issue in surface_verdict.get("issues", [])
        ],
    }
    surface_score = _focused_verifier_score(
        surface_verdict,
        label="dialog surface integrity",
    )
    if surface_score is not None:
        surface_result["score"] = surface_score
    aggregate = {
        "semantic_fidelity": semantic_result,
        "role_direction": role_result,
        "surface_integrity": surface_result,
        "lexical_avoidance": _verify_dialog_lexical_avoidances(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
        ),
    }
    return aggregate


def _focused_verifier_status(verdict: Mapping[str, Any]) -> str:
    """Project one focused verdict to a score-availability status token."""

    status = verdict.get("status")
    if status in {"unavailable", "skipped"}:
        return status
    if (
        verdict.get("hard_errors")
        or verdict.get("violations")
        or verdict.get("issues")
    ):
        return "hard_ineligible"
    return "scored"


def _focused_verifier_score(
    verdict: Mapping[str, Any],
    *,
    label: str,
) -> float | None:
    """Return a validated focused score, or no score for skipped/outage owners."""

    status = verdict.get("status")
    if status in {"unavailable", "skipped"}:
        return None
    score = _validate_numeric_score(verdict.get("score"), label=label)
    return score


def _dialog_verifier_aggregate_score(
    aggregate: Mapping[str, Any],
) -> float:
    """Compute the bounded equal-weight geometric mean of available scores."""

    focused_scores: list[float] = []
    for owner in (
        "semantic_fidelity",
        "role_direction",
        "surface_integrity",
    ):
        owner_result = aggregate[owner]
        if not isinstance(owner_result, Mapping):
            raise StateContractError(
                f"dialog verifier aggregate owner {owner} must be an object"
            )
        status = owner_result.get("status")
        if status in {"unavailable", "skipped"}:
            continue
        owner_score = _validate_numeric_score(
            owner_result.get("score"),
            label=f"dialog verifier aggregate {owner}",
        )
        focused_scores.append(owner_score)

    if not focused_scores:
        return 0.0

    scores = list(focused_scores)
    lexical_result = aggregate["lexical_avoidance"]
    if not isinstance(lexical_result, Mapping):
        raise StateContractError(
            "dialog verifier aggregate owner lexical_avoidance "
            "must be an object"
        )
    lexical_status = lexical_result.get("status")
    if lexical_status not in {"unavailable", "skipped"}:
        scores.append(_validate_numeric_score(
            lexical_result.get("score"),
            label="dialog verifier aggregate lexical_avoidance",
        ))

    product = math.prod(scores)
    aggregate_score = product ** (1 / len(scores))
    if not math.isfinite(aggregate_score):
        raise StateContractError("dialog verifier aggregate score is not finite")
    bounded_score = min(1.0, max(0.0, float(aggregate_score)))
    return bounded_score


def _dialog_verifier_aggregate_is_eligible(
    aggregate: Mapping[str, Any],
) -> bool:
    """Return whether no focused or lexical hard issue blocks a candidate."""

    eligible = not _dialog_verifier_aggregate_hard_issues(aggregate)
    return eligible


def _dialog_verifier_aggregate_is_passed(
    aggregate: Mapping[str, Any],
) -> bool:
    """Return whether an eligible candidate reaches the pass threshold."""

    passed = (
        _dialog_verifier_aggregate_is_eligible(aggregate)
        and _dialog_verifier_aggregate_score(aggregate)
        >= DIALOG_PASS_SCORE_THRESHOLD
    )
    return passed


def _dialog_verifier_aggregate_has_unavailable(
    aggregate: Mapping[str, Any],
) -> bool:
    """Return whether a focused model owner exhausted its bounded attempts."""

    unavailable = any(
        aggregate[owner]["status"] == "unavailable"
        for owner in (
            "semantic_fidelity",
            "role_direction",
            "surface_integrity",
        )
    )
    return unavailable


def _dialog_verifier_aggregate_repair_issues(
    aggregate: Mapping[str, Any],
) -> list[str]:
    """Flatten typed verifier outcomes only for the next repair prompt.

    Args:
        aggregate: Exact owner-preserving verifier result for one candidate.

    Returns:
        Bounded unique issue text suitable for the next rendering attempt.
    """

    semantic = aggregate["semantic_fidelity"]
    role = aggregate["role_direction"]
    surface = aggregate["surface_integrity"]
    lexical = aggregate["lexical_avoidance"]
    combined_issues = list(semantic["issues"])
    combined_issues.extend(
        (
            f"{violation['kind']}: {violation['evidence']!r} - "
            f"{violation['explanation']}"
        )
        for violation in role["violations"]
    )
    combined_issues.extend(
        (
            f"{issue['kind']}: {issue['evidence']!r} - "
            f"{issue['explanation']}"
        )
        for issue in surface["issues"]
    )
    combined_issues.extend(
        (
            f"{issue['kind']}: {issue['evidence']!r} - "
            f"{issue['explanation']}"
        )
        for issue in lexical["issues"]
    )
    issues: list[str] = []
    for issue in combined_issues:
        if issue not in issues:
            issues.append(issue)
        if len(issues) >= MAX_MERGED_VERIFIER_ISSUES:
            break
    return issues


def _dialog_verifier_aggregate_hard_issues(
    aggregate: Mapping[str, Any],
) -> list[str]:
    """Flatten only deterministic hard issues that block degraded fallback.

    Explicit semantic hard errors, typed role violations, false-execution
    issues, and lexical violations remain hard because selecting a lower-
    scoring candidate cannot safely repair those claims. A low semantic score
    with no hard error remains comparable for degraded ranking.
    """

    semantic = aggregate["semantic_fidelity"]
    role = aggregate["role_direction"]
    surface = aggregate["surface_integrity"]
    lexical = aggregate["lexical_avoidance"]
    combined_issues = list(semantic["issues"])
    combined_issues.extend(
        (
            f"{violation['kind']}: {violation['evidence']!r} - "
            f"{violation['explanation']}"
        )
        for violation in role["violations"]
    )
    combined_issues.extend(
        (
            f"{issue['kind']}: {issue['evidence']!r} - "
            f"{issue['explanation']}"
        )
        for issue in surface["issues"]
    )
    combined_issues.extend(
        (
            f"{issue['kind']}: {issue['evidence']!r} - "
            f"{issue['explanation']}"
        )
        for issue in lexical["issues"]
    )
    issues: list[str] = []
    for issue in combined_issues:
        if issue not in issues:
            issues.append(issue)
        if len(issues) >= MAX_MERGED_VERIFIER_ISSUES:
            break
    return issues


def _select_best_dialog_candidate(
    candidates: list[DialogCandidateRecord],
) -> DialogCandidateRecord | None:
    """Select the highest-scoring eligible candidate with latest-attempt ties."""

    selected_candidate: DialogCandidateRecord | None = None
    for candidate in candidates:
        if candidate["hard_issues"]:
            continue
        if selected_candidate is None:
            selected_candidate = candidate
            continue
        if candidate["aggregate_score"] > selected_candidate[
            "aggregate_score"
        ]:
            selected_candidate = candidate
            continue
        if (
            candidate["aggregate_score"]
            == selected_candidate["aggregate_score"]
            and candidate["attempt_number"]
            >= selected_candidate["attempt_number"]
        ):
            selected_candidate = candidate
    return selected_candidate


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


def _validate_numeric_score(value: object, *, label: str) -> float:
    """Validate one finite JSON number in the closed score interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateContractError(f"{label} score must be a number")
    try:
        score = float(value)
    except (OverflowError, ValueError) as exc:
        raise StateContractError(
            f"{label} score must be finite and within [0.0, 1.0]"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise StateContractError(
            f"{label} score must be finite and within [0.0, 1.0]"
        )
    return score


def _validate_score_issue_consistency(
    score: float,
    issue_count: int,
    *,
    label: str,
) -> None:
    """Reject a passing score that simultaneously asserts a hard issue."""

    if issue_count and score >= DIALOG_PASS_SCORE_THRESHOLD:
        raise StateContractError(
            f"{label} score is inconsistent with non-empty issues"
        )


def _validate_string_verdict(
    value: object,
    *,
    label: str,
    issue_field: str,
    max_issues: int,
) -> dict[str, Any]:
    """Validate one exact scored verdict with bounded string issues.

    Args:
        value: Parsed verdict candidate.
        label: Stable stage label used in protected contract errors.
        issue_field: Exact producer-owned problem-array field name.
        max_issues: Maximum accepted problem rows.

    Returns:
        The canonical numeric score and producer-owned issue field.
    """

    verdict = _validate_exact_object_fields(
        value,
        label=label,
        expected_fields=frozenset({"score", issue_field}),
    )
    score = _validate_numeric_score(verdict["score"], label=label)
    issues = verdict[issue_field]
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
    _validate_score_issue_consistency(
        score,
        len(issues),
        label=label,
    )
    validated_verdict = {
        "score": score,
        issue_field: list(issues),
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


def _validate_role_direction_verdict(
    value: object,
    *,
    generated_dialog: list[str],
) -> dict[str, Any]:
    """Validate the role owner's two typed violation conditions.

    Args:
        value: Parsed role-direction model output.
        generated_dialog: Candidate text that must contain quoted evidence.

    Returns:
        Validated numeric score and typed violation rows.
    """

    verdict = _validate_exact_object_fields(
        value,
        label="dialog compliance",
        expected_fields=frozenset({"score", "violations"}),
    )
    score = _validate_numeric_score(
        verdict["score"],
        label="dialog role direction",
    )
    violations = verdict["violations"]
    if (
        not isinstance(violations, list)
        or len(violations) > MAX_FOCUSED_VERIFIER_ISSUES
    ):
        raise StateContractError(
            "dialog role direction violations are invalid"
        )

    candidate_text = "\n".join(generated_dialog)
    validated_violations: list[dict[str, str]] = []
    normalized_rows: list[tuple[str, str, str]] = []
    for violation in violations:
        validated_violation = _validate_exact_object_fields(
            violation,
            label="dialog role direction violation",
            expected_fields=frozenset({
                "kind",
                "evidence",
                "explanation",
            }),
        )
        kind = validated_violation["kind"]
        evidence = validated_violation["evidence"]
        explanation = validated_violation["explanation"]
        if kind not in {
            "selection_owner_transfer",
            "typed_operation_role_reversal",
        }:
            raise StateContractError(
                "dialog role direction violation kind is invalid"
            )
        if (
            not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 120
            or evidence not in candidate_text
        ):
            raise StateContractError(
                "dialog role direction violation evidence is invalid"
            )
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 300
        ):
            raise StateContractError(
                "dialog role direction violation explanation is invalid"
            )
        normalized_rows.append((kind, evidence, explanation))
        validated_violations.append({
            "kind": kind,
            "evidence": evidence,
            "explanation": explanation,
        })
    if len(normalized_rows) != len(set(normalized_rows)):
        raise StateContractError(
            "dialog role direction violations are duplicated"
        )
    _validate_score_issue_consistency(
        score,
        len(validated_violations),
        label="dialog role direction",
    )
    return {
        "score": score,
        "violations": validated_violations,
    }


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
    if sum(len(message) for message in messages) > DIALOG_CANDIDATE_MAX_CHARS:
        raise StateContractError("dialog repair messages exceed text bound")
    validated_messages = list(messages)
    return validated_messages


def _validate_surface_compliance_verdict(
    value: object,
    *,
    generated_dialog: list[str],
) -> dict[str, Any]:
    """Validate evidence-bearing surface issues without losing typed rows."""

    verdict = _validate_exact_object_fields(
        value,
        label="surface compliance",
        expected_fields=frozenset({"score", "issues"}),
    )
    score = _validate_numeric_score(
        verdict["score"],
        label="dialog surface integrity",
    )
    issues = verdict["issues"]
    if (
        not isinstance(issues, list)
        or len(issues) > MAX_FOCUSED_VERIFIER_ISSUES
    ):
        raise StateContractError("surface compliance issues are invalid")
    candidate_text = "\n".join(generated_dialog)
    normalized_rows: list[tuple[str, str, str]] = []
    validated_issues: list[dict[str, str]] = []
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
        validated_issues.append({
            "kind": kind,
            "evidence": evidence,
            "explanation": explanation,
        })
    if len(normalized_rows) != len(set(normalized_rows)):
        raise StateContractError("surface compliance issues are duplicated")
    _validate_score_issue_consistency(
        score,
        len(validated_issues),
        label="dialog surface integrity",
    )
    return {
        "score": score,
        "issues": validated_issues,
    }
