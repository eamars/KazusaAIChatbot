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
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    record_protected_chain_record,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    CognitiveEpisodeV1,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    TextSurfaceInput,
    TextSurfaceOutputV2,
    validate_terminal_text_seed,
    validate_text_surface_input_canonical,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
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
_DIALOG_REPAIR_ERROR_CAP = 500
_DIALOG_REPAIR_OUTPUT_CAP = 8000
_DIALOG_VISIBLE_PERCEPT_SCAN_MAX_CHARS = 24000


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
    attempt_diagnostics: NotRequired[list[dict[str, object]]]


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


_V2_DIALOG_GENERATOR_PROMPT = '''# 任务
你是当前角色的最终文字渲染器。把已验证的 `text_surface_output_v2` 转化为自然、鲜活、有角色辨识度并切合当前场景的可发送聊天内容。上游已经决定语义、事实边界、行动状态和称呼方向；你负责最终措辞、节奏与角色声音。

# 输入
`text_surface_output_v2` 是权威渲染合同：`selected_surface_intent`、`content_plan` 与 `content_requirements` 定义本轮可见语义；`epistemic_boundary` 定义断言、解释与未知；`addressee_plan` 与 `candidate_role_frame` 定义第一、第二和第三人称方向；`delivery_profile` 定义词语、句式、节奏、犹豫和标点；`relational_willingness` 提供已选择的关系姿态；`permitted_action_results`、`resolver_result` 与 `runtime_capability_limits` 提供行动、任务、证据和能力事实；`lexical_avoidances` 提供本轮具体的措辞避让。顶层 `epistemic_boundary` 是同一断言边界的置前副本；`user_name` 是当前用户的可见名称；`required_source_urls` 若存在，给出可引用的精确来源；`contract_repair` 若出现，只说明上一候选的结构或来源问题。

# 渲染步骤
1. 以 `selected_surface_intent` 为中心，完整实现 `content_plan` 与每条 `content_requirements`。关系姿态和角色表达用于塑造温度、主动性、直接程度与声音，并服务已选择的语义。
2. 逐句遵循 `epistemic_boundary`：可断言内容直接表达，解释内容使用明确推测语气，未知内容保留未知。可外部核查的事实来自内容计划、要求或完整证据摘录。
3. 按 `candidate_role_frame` 与 `addressee_plan` 保持行动者、对象、受益者和主语方向。第一人称属于当前角色；只有 `second_person_allowed` 的当前用户目标使用第二人称；其他目标使用其 `display_name` 或明确第三人称。
4. 按 `permitted_action_results` 表达真实外部执行状态：`executed` 表达同一行动与效果已完成，`pending` 或 `scheduled` 表达等待或未来执行，`failed` 或 `unavailable` 表达限制与可行下一步。没有匹配结果时，保留内容计划中的言语立场、愿望、提议或条件。
5. 按 `resolver_result` 表达任务与证据状态：完整结果使用 `evidence_excerpts`；部分结果表达已确认部分和 `remaining_needs`；等待、缺失、阻塞、不可用或失败状态表达相应缺口。已接纳并继续的任务表达接纳与等待后续结果。
6. 使用 `delivery_profile` 形成角色化用词、句式与节奏；避免逐字使用 `lexical_avoidances`，同时保持原语义。语言跟随当前可见对话、内容计划与角色声音；引文、专有名词、代码、URL、schema 与 enum token 保持原样。
7. `required_source_urls` 存在时，只使用该列表中的 URL，并逐字包含至少一个直接支持已呈现事实的 URL；多来源事实分别附上相应来源。
8. 若有 `contract_repair`，依据原输入重新生成完整结果，修正其指出的合同或来源问题。

# 输出
只返回一个 JSON 对象，并且唯一字段是 `final_dialog`。`final_dialog` 是由一条或多条非空可见消息字符串组成的列表，所有消息总长度不超过 12000 字符。不添加解释、Markdown 包装或其他字段。'''

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


def _set_dialog_repair_issues(
    repair_issues: list[str],
    *,
    reason: str,
    contract_error: str,
    invalid_candidate: str,
) -> None:
    """Replace the bounded feedback context with the latest failure."""

    repair_issues[:] = [
        reason,
        contract_error[:_DIALOG_REPAIR_ERROR_CAP],
        invalid_candidate[:_DIALOG_REPAIR_OUTPUT_CAP],
    ]


def _dialog_repair_block(repair_issues: list[str]) -> dict[str, str]:
    """Project bounded runtime diagnostics for the next dialog attempt."""

    bounded_issues = [*repair_issues, "", "", ""]
    return {
        "reason": bounded_issues[0],
        "contract_error": bounded_issues[1][:_DIALOG_REPAIR_ERROR_CAP],
        "invalid_candidate": bounded_issues[2][:_DIALOG_REPAIR_OUTPUT_CAP],
    }


def _normalize_dialog_source_urls(
    generated_dialog: list[str],
    *,
    required_source_urls: list[str],
) -> tuple[list[str], list[str], bool]:
    """Apply source-token repairs and return the normalized issues."""

    allowed_urls = set(required_source_urls)

    def replace_unexpected_url(match: re.Match[str]) -> str:
        token = match.group(0)
        normalized_token = token.rstrip(".,;:")
        if normalized_token in allowed_urls:
            return token
        return ""

    normalized_dialog = []
    for message in generated_dialog:
        normalized_message = _HTTP_URL_PATTERN.sub(
            replace_unexpected_url,
            message,
        ).strip()
        if normalized_message:
            normalized_dialog.append(normalized_message)

    normalized = normalized_dialog != generated_dialog
    normalized_issues = _dialog_source_url_issues(
        normalized_dialog,
        required_source_urls=required_source_urls,
    )
    if normalized_dialog and normalized_issues and not (
        set(_extract_dialog_source_urls(normalized_dialog)) & allowed_urls
    ):
        first_required_url = required_source_urls[0]
        candidate = f"{normalized_dialog[-1]}\n{first_required_url}"
        if sum(len(message) for message in normalized_dialog[:-1]) + len(
            candidate
        ) <= DIALOG_CANDIDATE_MAX_CHARS:
            normalized_dialog[-1] = candidate
            normalized = True
            normalized_issues = _dialog_source_url_issues(
                normalized_dialog,
                required_source_urls=required_source_urls,
            )
    return normalized_dialog, normalized_issues, normalized


def _extract_dialog_source_urls(messages: list[str]) -> list[str]:
    """Return normalized URL tokens from bounded dialog messages."""

    return [
        match.group(0).rstrip(".,;:")
        for match in _HTTP_URL_PATTERN.finditer("\n".join(messages))
    ]


def _project_dialog_content_plan(
    surface_output: TextSurfaceOutputV2,
    *,
    required_source_urls: list[str],
) -> list[str]:
    """Project validated upstream wording after deterministic URL sanitization."""

    allowed_urls = set(required_source_urls)

    def remove_unexpected_url(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.rstrip(".,;:") in allowed_urls:
            return token
        return ""

    def sanitize(seed: str, label: str) -> str:
        sanitized = _HTTP_URL_PATTERN.sub(remove_unexpected_url, seed).strip()
        try:
            validate_terminal_text_seed(sanitized, label)
        except CognitionContractError:
            return ""
        return sanitized

    content_plan = sanitize(surface_output["content_plan"], "content_plan")
    if content_plan:
        return [content_plan]
    selected_intent = sanitize(
        surface_output["selected_surface_intent"],
        "selected_surface_intent",
    )
    if not selected_intent:
        raise StateContractError(
            "dialog terminal projection has no deliverable selected intent"
        )
    return [selected_intent]


def _dialog_attempt_diagnostic(error_code: str) -> dict[str, object]:
    """Build the fixed diagnostic row for one accepted-degraded dialog."""

    return {
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "dialog_generation",
        "error_code": error_code,
        "attempt_count": DIALOG_GENERATOR_TOTAL_ATTEMPTS,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }


def _record_dialog_protected_attempt(
    *,
    stage_name: str,
    request_messages: list[SystemMessage | HumanMessage],
    raw_output: object,
    parsed_output: object,
    parse_status: str,
    status: str,
    attempt_number: int,
    validation_error: str,
    started_at: float,
) -> None:
    """Record the protected disposition for one dialog model attempt."""

    record_protected_chain_record({
        "stage": stage_name,
        "config": {
            "route_name": _dialog_generator_llm_config.route_name,
            "model": _dialog_generator_llm_config.model,
            "stage_name": stage_name,
        },
        "messages": [
            {"role": "system", "content": request_messages[0].content},
            {"role": "human", "content": request_messages[1].content},
        ],
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_status": parse_status,
        "status": status,
        "attempt_index": attempt_number,
        "validation_error": validation_error,
        "duration_ms": _elapsed_ms(started_at),
    })


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
    retained_candidates: list[list[str]] = []
    repair_issues: list[str] = []
    attempts_used = 0

    for attempt_number in range(1, DIALOG_GENERATOR_TOTAL_ATTEMPTS + 1):
        attempts_used = attempt_number
        generated_dialog, failure_kind = await _render_dialog_candidate(
            surface_output=surface_output,
            user_name=state["user_name"],
            repair_issues=repair_issues,
            attempt_number=attempt_number,
            llm_trace_id=llm_trace_id,
            required_source_urls=required_source_urls,
        )
        if failure_kind == "normalized":
            accepted_dialog = generated_dialog
            break
        if failure_kind == "source_url":
            if generated_dialog:
                retained_candidates.append(generated_dialog)
            source_error = repair_issues[1] if len(repair_issues) > 1 else (
                "source_url_fidelity: candidate did not preserve required "
                "source provenance"
            )
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_source_url_fidelity",
                violation_kind="source_url_fidelity",
                missing_fields=[],
                invalid_fields=[source_error],
                repair_used=attempt_number > 1,
                status="retrying",
                correlation_id=llm_trace_id,
            )
            continue
        if not generated_dialog:
            continue
        accepted_dialog = list(generated_dialog)
        break

    attempt_diagnostics: list[dict[str, object]] = []
    if accepted_dialog is None:
        if retained_candidates:
            accepted_dialog = retained_candidates[-1]
            quality_status = "accepted_degraded"
            failure_codes = ["source_url_fidelity"]
            attempt_diagnostics = [
                _dialog_attempt_diagnostic("dialog_source_url_degraded"),
            ]
        else:
            accepted_dialog = _project_dialog_content_plan(
                surface_output,
                required_source_urls=required_source_urls,
            )
            quality_status = "accepted_degraded"
            failure_codes = ["deterministic_surface_projection"]
            attempt_diagnostics = [
                _dialog_attempt_diagnostic(
                    "dialog_surface_projection_degraded",
                ),
            ]
    else:
        quality_status = "passed"
        failure_codes = []

    await event_logging.record_dialog_quality_event(
        component=DIALOG_COMPONENT,
        correlation_id=llm_trace_id,
        usage_mode=state["dialog_usage_mode"],
        quality_status=quality_status,
        retry_count=max(0, attempts_used - 1),
        failure_codes=failure_codes,
        content_plan_entry_count=1,
        status="succeeded",
    )
    return_value = {
        "final_dialog": accepted_dialog,
        "text_surface_output_v2": surface_output,
    }
    if attempt_diagnostics:
        return_value["attempt_diagnostics"] = attempt_diagnostics
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
        repair_issues: Bounded failure context from the prior attempt.
        attempt_number: One-based producer opportunity in the shared budget.
        llm_trace_id: Correlation identifier for protected model evidence.
        required_source_urls: Exact completed-evidence URLs that visible output
            may preserve.

    Returns:
        A bounded dialog with a clean, normalized, or source-fidelity
        disposition, or an empty dialog with the classified failure kind.
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
    if attempt_number > 1:
        payload["contract_repair"] = _dialog_repair_block(repair_issues)
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
        _record_dialog_protected_attempt(
            stage_name=stage_name,
            request_messages=request_messages,
            raw_output="",
            parsed_output={},
            parse_status="provider_error",
            status="provider_error",
            attempt_number=attempt_number,
            validation_error=str(exc),
            started_at=started_at,
        )
        logger.warning(
            f"Dialog candidate {attempt_number} provider failure: {exc}"
        )
        _set_dialog_repair_issues(
            repair_issues,
            reason="no_candidate",
            contract_error="",
            invalid_candidate="",
        )
        return [], "provider"

    response_text = str(getattr(response, "content", ""))
    parsed = parse_llm_json_output(response_text)
    try:
        generated_dialog = _validated_dialog_messages(parsed)
    except StateContractError as exc:
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
        _record_dialog_protected_attempt(
            stage_name=stage_name,
            request_messages=request_messages,
            raw_output=response_text,
            parsed_output=parsed,
            parse_status="contract_error",
            status="contract_fault",
            attempt_number=attempt_number,
            validation_error=str(exc),
            started_at=started_at,
        )
        logger.warning(
            f"Dialog candidate {attempt_number} contract failure: {exc}"
        )
        _set_dialog_repair_issues(
            repair_issues,
            reason="failed_contract",
            contract_error=str(exc),
            invalid_candidate=response_text,
        )
        return [], "structure"

    normalized_dialog, source_url_issues, normalized = (
        _normalize_dialog_source_urls(
            generated_dialog,
            required_source_urls=source_urls,
        )
    )
    if not normalized_dialog:
        source_url_error = (
            source_url_issues[0]
            if source_url_issues
            else (
                "source_url_fidelity: candidate contains no authored text "
                "after URL removal"
            )
        )
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
            validation_error=source_url_error,
        )
        _record_dialog_protected_attempt(
            stage_name=stage_name,
            request_messages=request_messages,
            raw_output=response_text,
            parsed_output=parsed,
            parse_status="contract_error",
            status="contract_fault",
            attempt_number=attempt_number,
            validation_error=source_url_error,
            started_at=started_at,
        )
        await event_logging.record_llm_stage_event(
            component=DIALOG_COMPONENT,
            stage_name=stage_name,
            route_name="generate",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            status="failed",
            prompt_chars=sum(
                len(str(message.content))
                for message in request_messages
            ),
            output_chars=len(response_text),
            parse_status="contract_error",
            retry_count=attempt_number - 1,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="warning",
            correlation_id=llm_trace_id,
        )
        _set_dialog_repair_issues(
            repair_issues,
            reason="source_url_fidelity",
            contract_error=source_url_error,
            invalid_candidate=response_text,
        )
        return [], "source_url"

    if source_url_issues:
        source_url_error = source_url_issues[0]
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
            validation_error=source_url_error,
        )
        _record_dialog_protected_attempt(
            stage_name=stage_name,
            request_messages=request_messages,
            raw_output=response_text,
            parsed_output=parsed,
            parse_status="contract_error",
            status="contract_fault",
            attempt_number=attempt_number,
            validation_error=source_url_error,
            started_at=started_at,
        )
        await event_logging.record_llm_stage_event(
            component=DIALOG_COMPONENT,
            stage_name=stage_name,
            route_name="generate",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            status="failed",
            prompt_chars=sum(
                len(str(message.content))
                for message in request_messages
            ),
            output_chars=len(response_text),
            parse_status="contract_error",
            retry_count=attempt_number - 1,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="warning",
            correlation_id=llm_trace_id,
        )
        _set_dialog_repair_issues(
            repair_issues,
            reason="source_url_fidelity",
            contract_error=source_url_error,
            invalid_candidate=response_text,
        )
        return normalized_dialog, "source_url"

    if normalized:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_trace_id,
            stage_name=stage_name,
            route_name="DIALOG_GENERATOR_LLM",
            model_name=DIALOG_GENERATOR_LLM_MODEL,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="normalized",
            status="succeeded",
            duration_ms=_elapsed_ms(started_at),
            output_state_fields=["final_dialog"],
            sequence=attempt_number - 1,
        )
        _record_dialog_protected_attempt(
            stage_name=stage_name,
            request_messages=request_messages,
            raw_output=response_text,
            parsed_output=parsed,
            parse_status="normalized",
            status="parsed",
            attempt_number=attempt_number,
            validation_error="",
            started_at=started_at,
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
            parse_status="normalized",
            retry_count=attempt_number - 1,
            json_repair_used=False,
            duration_ms=_elapsed_ms(started_at),
            severity="info",
            correlation_id=llm_trace_id,
        )
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_source_url_fidelity",
            violation_kind="source_url_fidelity",
            missing_fields=[],
            invalid_fields=[],
            repair_used=True,
            status="normalized",
            correlation_id=llm_trace_id,
        )
        return normalized_dialog, "normalized"

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
    _record_dialog_protected_attempt(
        stage_name=stage_name,
        request_messages=request_messages,
        raw_output=response_text,
        parsed_output=parsed,
        parse_status="succeeded",
        status="parsed",
        attempt_number=attempt_number,
        validation_error="",
        started_at=started_at,
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
    attempt_diagnostics = result.get("attempt_diagnostics", [])
    if not isinstance(attempt_diagnostics, list):
        raise StateContractError(
            "dialog result attempt_diagnostics must be a list"
        )
    if any(not isinstance(row, dict) for row in attempt_diagnostics):
        raise StateContractError(
            "dialog result attempt_diagnostics rows are invalid"
        )

    logger.info(
        f"Dialog output: usage_mode={usage_mode} "
        f"dialog={log_list_preview(final_dialog)}"
    )
    logger.debug(
        f'Dialog metadata: usage_mode={usage_mode} '
        f'messages={len(final_dialog)}'
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
        "attempt_diagnostics": [dict(row) for row in attempt_diagnostics],
    }
    return return_value


def _completed_tool_result_source_urls(
    current_visible_percepts: list[dict[str, Any]],
) -> list[str]:
    """Extract exact HTTP source tokens from completed tool-result evidence."""

    source_urls: list[str] = []
    remaining_scan_chars = _DIALOG_VISIBLE_PERCEPT_SCAN_MAX_CHARS
    for percept in current_visible_percepts:
        if remaining_scan_chars <= 0:
            break
        if percept.get("input_source") != "tool_result":
            continue
        serialized_content = json.dumps(
            percept.get("content", {}),
            ensure_ascii=False,
            default=str,
        )
        scan_content = serialized_content[:remaining_scan_chars]
        remaining_scan_chars -= len(scan_content)
        for match in _HTTP_URL_PATTERN.finditer(scan_content):
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
    candidate_urls = set(_extract_dialog_source_urls(generated_dialog))
    allowed_urls = set(required_source_urls)
    unexpected_urls = sorted(candidate_urls - allowed_urls)
    if unexpected_urls:
        return [
            (
                "source_url_fidelity: candidate URL is absent from completed "
                "tool evidence"
            ),
        ]
    if not candidate_urls & allowed_urls:
        return [
            (
                "source_url_fidelity: include at least one exact "
                "required_source_urls token"
            ),
        ]
    return []


def _current_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project current percepts for the bounded source-URL scan."""

    percepts = project_model_visible_percepts(episode)
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
