"""L2d action initializer for modality-neutral action specs."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_spec.models import (
    ActionSourceRefV1,
    ActionSpecV1,
    CapabilitySpecV1,
    LIFECYCLE_STATUS_BY_DECISION,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

ACTION_SPEC_CAP = 3
_ALLOWED_LIFECYCLE_DECISIONS = frozenset(LIFECYCLE_STATUS_BY_DECISION)


class TriggerContextV1(TypedDict):
    """Prompt-safe trigger context consumed by the action initializer."""

    trigger_source: str
    input_sources: list[str]
    output_mode: str
    target_scope_summary: str


class ActionRequestV1(TypedDict, total=False):
    """Semantic action request emitted by L2d before deterministic wrapping."""

    capability: str
    decision: str
    reason: str
    detail: str


def build_action_initializer_payload(
    state: CognitionState,
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> dict[str, object]:
    """Build the prompt-safe payload for L2d action selection.

    Args:
        state: Cognition state after L2c judgment.
        capabilities: Optional registry override for deterministic tests.

    Returns:
        JSON-serializable payload containing final L2 state, semantic trigger
        context, prompt-safe affordances, and bounded evidence.
    """

    if capabilities is None:
        capabilities = build_initial_action_capabilities()
    prompt_capabilities = _prompt_capabilities_for_state(state, capabilities)
    trigger_context = build_trigger_context(state)
    payload = {
        "final_l2": {
            "internal_monologue": state["internal_monologue"],
            "logical_stance": state["logical_stance"],
            "character_intent": state["character_intent"],
            "judgment_note": state["judgment_note"],
            "emotional_appraisal": state["emotional_appraisal"],
            "interaction_subtext": state["interaction_subtext"],
            "boundary_core_assessment": state["boundary_core_assessment"],
        },
        "trigger_context": trigger_context,
        "capabilities": project_prompt_affordances(prompt_capabilities),
        "evidence": _project_action_evidence(state),
    }
    return payload


def _prompt_capabilities_for_state(
    state: CognitionState,
    capabilities: Mapping[str, CapabilitySpecV1],
) -> dict[str, CapabilitySpecV1]:
    """Return prompt-visible capabilities allowed by deterministic bindings."""

    prompt_capabilities = dict(capabilities)
    if _select_active_commitment(state) is None:
        prompt_capabilities.pop(MEMORY_LIFECYCLE_UPDATE_CAPABILITY, None)
    return prompt_capabilities


def build_trigger_context(state: CognitionState) -> TriggerContextV1:
    """Project the cognitive episode into prompt-safe trigger context."""

    episode = state["cognitive_episode"]
    output_mode = episode["output_mode"]
    channel_type = state["channel_type"]
    trigger_context: TriggerContextV1 = {
        "trigger_source": episode["trigger_source"],
        "input_sources": list(episode["input_sources"]),
        "output_mode": output_mode,
        "target_scope_summary": (
            f"{channel_type} conversation with {output_mode} output mode"
        ),
    }
    return trigger_context


def _project_action_evidence(state: CognitionState) -> dict[str, object]:
    """Return bounded, action-relevant evidence without raw transport IDs."""

    rag_result = state["rag_result"]
    answer = ""
    memory_evidence: list[object] = []
    if isinstance(rag_result, dict):
        raw_answer = rag_result.get("answer")
        if isinstance(raw_answer, str):
            answer = raw_answer
        raw_memory_evidence = rag_result.get("memory_evidence")
        if isinstance(raw_memory_evidence, list):
            memory_evidence = _project_memory_evidence_for_prompt(
                raw_memory_evidence
            )

    evidence = {
        "decontexualized_input": state["decontexualized_input"],
        "rag_answer": answer,
        "active_commitments": _project_active_commitments_for_prompt(
            _raw_active_commitments(state)
        ),
        "memory_evidence": memory_evidence,
        "conversation_progress": state.get("conversation_progress"),
    }
    return evidence


def _raw_active_commitments(state: CognitionState) -> list[object]:
    """Return raw active commitment rows for deterministic materialization."""

    rag_result = state["rag_result"]
    active_commitments: list[object] = []
    if not isinstance(rag_result, dict):
        return active_commitments

    raw_user_image = rag_result.get("user_image")
    if not isinstance(raw_user_image, dict):
        return active_commitments

    memory_context = raw_user_image.get("user_memory_context")
    if not isinstance(memory_context, dict):
        return active_commitments

    raw_commitments = memory_context.get("active_commitments")
    if isinstance(raw_commitments, list):
        active_commitments = raw_commitments
    return active_commitments


def _project_active_commitments_for_prompt(
    raw_commitments: list[object],
) -> list[dict[str, object]]:
    """Project active commitments without persistence identifiers."""

    projected_commitments: list[dict[str, object]] = []
    for raw_commitment in raw_commitments:
        if not isinstance(raw_commitment, dict):
            continue
        projected_commitment: dict[str, object] = {}
        for field_name in ("fact", "summary", "due_at", "due_state", "status"):
            field_value = raw_commitment.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                projected_commitment[field_name] = field_value
        if projected_commitment:
            projected_commitments.append(projected_commitment)
    return projected_commitments


def _project_memory_evidence_for_prompt(
    raw_memory_evidence: list[object],
) -> list[dict[str, object]]:
    """Project retrieved memory evidence without storage identifiers."""

    projected_evidence: list[dict[str, object]] = []
    for raw_entry in raw_memory_evidence:
        if not isinstance(raw_entry, dict):
            continue
        projected_entry: dict[str, object] = {}
        for field_name in ("summary", "fact", "excerpt", "due_at", "due_state"):
            field_value = raw_entry.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                projected_entry[field_name] = field_value
        if projected_entry:
            projected_evidence.append(projected_entry)
    return projected_evidence


def _current_episode_source_ref() -> ActionSourceRefV1:
    """Return a stable prompt alias for the current cognitive episode."""

    source_ref: ActionSourceRefV1 = {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "cognitive_episode",
        "ref_id": "current_cognitive_episode",
        "owner": "cognition_episode",
        "relationship": "basis",
        "evidence_refs": [],
    }
    return source_ref


def _normalize_action_requests(parsed: object) -> list[ActionRequestV1]:
    """Normalize LLM-selected semantic requests before materialization."""

    normalized_requests: list[ActionRequestV1] = []
    if not isinstance(parsed, dict):
        return normalized_requests

    raw_requests = parsed.get("action_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests

    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            logger.warning("L2d dropped non-object action request")
            continue
        capability = _semantic_text(raw_request, "capability")
        reason = _semantic_text(raw_request, "reason")
        if not capability or not reason:
            logger.warning("L2d dropped action request without capability or reason")
            continue
        normalized_request: ActionRequestV1 = {
            "capability": capability,
            "reason": reason,
        }
        for optional_field in ("decision", "detail"):
            field_value = _semantic_text(raw_request, optional_field)
            if field_value:
                normalized_request[optional_field] = field_value
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= ACTION_SPEC_CAP:
            break
    return normalized_requests


def _materialize_action_specs(
    requests: list[ActionRequestV1],
    state: CognitionState,
) -> list[ActionSpecV1]:
    """Wrap semantic requests in deterministic action-spec envelopes."""

    action_specs: list[ActionSpecV1] = []
    for request in requests:
        action_spec = _materialize_action_request(request, state)
        if action_spec is None:
            continue
        action_specs.append(action_spec)
        if len(action_specs) >= ACTION_SPEC_CAP:
            break
    return action_specs


def _materialize_action_request(
    request: ActionRequestV1,
    state: CognitionState,
) -> ActionSpecV1 | None:
    """Build one validated action spec for a selected semantic capability."""

    capability = request["capability"]
    if capability == SPEAK_CAPABILITY:
        action_spec = _build_speak_action_spec(request, state)
    elif capability == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        action_spec = _build_memory_lifecycle_action_spec(request, state)
    elif capability == TRIGGER_FUTURE_COGNITION_CAPABILITY:
        action_spec = _build_future_cognition_action_spec(request)
    else:
        logger.warning(f"L2d dropped unsupported action capability: {capability}")
        return None

    if action_spec is None:
        return None
    validated_spec = validate_action_spec(action_spec)
    return validated_spec


def _build_speak_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
) -> dict[str, object]:
    """Build the deterministic envelope for a text-surface request."""

    delivery_mode = _delivery_mode_for_request(request, state)
    target_kind = "current_channel"
    visibility = "user_visible"
    urgency = "now"
    if delivery_mode == "private_finalization":
        target_kind = "self"
        visibility = "private"
        urgency = "background"
    elif delivery_mode == "scheduled":
        urgency = "scheduled"
    detail = _semantic_text(request, "detail")
    surface_requirements = {
        "decision": _semantic_text(request, "decision"),
        "detail": detail,
    }
    action_spec = _build_action_spec(
        kind=SPEAK_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": target_kind,
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        params={
            "delivery_mode": delivery_mode,
            "execute_at": None,
            "surface_requirements": surface_requirements,
        },
        urgency=urgency,
        visibility=visibility,
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _delivery_mode_for_request(
    request: ActionRequestV1,
    _state: CognitionState,
) -> str:
    """Return the text-surface mode implied by semantics and trigger context."""

    decision = _semantic_text(request, "decision")
    if decision in (
        "visible_reply",
        "private_finalization",
        "delayed",
        "scheduled",
    ):
        return decision

    return "visible_reply"


def _build_memory_lifecycle_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
) -> dict[str, object] | None:
    """Build the deterministic envelope for a commitment lifecycle request."""

    lifecycle_decision = _semantic_text(request, "decision")
    if lifecycle_decision not in _ALLOWED_LIFECYCLE_DECISIONS:
        logger.warning(
            f"L2d dropped lifecycle request with unsupported decision: "
            f"{lifecycle_decision}"
        )
        return None

    active_commitment = _select_active_commitment(state)
    if active_commitment is None:
        logger.warning("L2d dropped lifecycle request without target commitment")
        return None

    unit_id = str(active_commitment["unit_id"])
    due_at = active_commitment.get("due_at")
    if due_at is not None and not isinstance(due_at, str):
        due_at = None
    action_spec = _build_action_spec(
        kind=MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        source_refs=[
            _current_episode_source_ref(),
            _memory_unit_source_ref(unit_id),
        ],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "memory_unit",
            "target_id": unit_id,
            "owner": "user_memory_units",
            "scope": {"unit_type": "active_commitment"},
        },
        params={
            "memory_kind": "user_memory_unit",
            "unit_type": "active_commitment",
            "unit_id": unit_id,
            "lifecycle_decision": lifecycle_decision,
            "due_at": due_at,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _select_active_commitment(state: CognitionState) -> dict[str, object] | None:
    """Resolve the current episode to one deterministic active commitment."""

    active_commitments = _active_commitments(state)
    if len(active_commitments) == 1:
        return active_commitments[0]
    return None


def _active_commitments(state: CognitionState) -> list[dict[str, object]]:
    """Return active commitment dictionaries that carry stable unit IDs."""

    raw_commitments = _raw_active_commitments(state)
    active_commitments: list[dict[str, object]] = []
    for raw_commitment in raw_commitments:
        if not isinstance(raw_commitment, dict):
            continue
        unit_id = raw_commitment.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id.strip():
            continue
        active_commitments.append(raw_commitment)
    return active_commitments


def _build_future_cognition_action_spec(
    request: ActionRequestV1,
) -> dict[str, object]:
    """Build the deterministic envelope for a future cognition request."""

    context_summary = _semantic_text(request, "detail")
    if not context_summary:
        context_summary = request["reason"]
    action_spec = _build_action_spec(
        kind=TRIGGER_FUTURE_COGNITION_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": {"episode_type": "self_cognition"},
        },
        params={
            "episode_type": "self_cognition",
            "trigger_at": None,
            "context_summary": context_summary,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _build_action_spec(
    *,
    kind: str,
    source_refs: list[ActionSourceRefV1],
    target: dict[str, object],
    params: dict[str, object],
    urgency: str,
    visibility: str,
    deadline: str | None,
    reason: str,
) -> dict[str, object]:
    """Build the common deterministic action-spec envelope."""

    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": kind,
        "cognition_mode": "deliberative",
        "source_refs": source_refs,
        "target": target,
        "params": params,
        "urgency": urgency,
        "visibility": visibility,
        "deadline": deadline,
        "continuation": _no_continuation(),
        "reason": reason,
    }
    return action_spec


def _memory_unit_source_ref(unit_id: str) -> ActionSourceRefV1:
    """Return a source reference for a selected memory unit."""

    source_ref: ActionSourceRefV1 = {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "memory_unit",
        "ref_id": unit_id,
        "owner": "user_memory_units",
        "relationship": "target",
        "evidence_refs": [],
    }
    return source_ref


def _no_continuation() -> dict[str, object]:
    """Return the default no-continuation execution contract."""

    continuation = {
        "schema_version": "action_continuation.v1",
        "mode": "none",
        "episode_type": None,
        "max_depth": 0,
        "include_result_as": None,
    }
    return continuation


def _semantic_text(value: Mapping[str, object], field_name: str) -> str:
    """Return one stripped semantic text field from LLM output."""

    raw_value = value.get(field_name)
    if not isinstance(raw_value, str):
        return ""
    return_value = raw_value.strip()
    return return_value


_ACTION_INITIALIZER_PROMPT = """\
你现在是角色的语义行动初始化层 (Action Initializer / L2d)。
最终立场、意图和边界裁决已经由上游完成。你不重新裁决、不写台词、不执行工具、不选择执行处理器、不授予权限、不修复上游判断。
你只负责把角色已经 settled 的决定翻译成 0 到 3 个语义动作请求。执行信封、目标对象、持久化字段、投递字段、后续承接对象由后续确定性代码或能力拥有者生成，不在这里输出。

# 语言政策
- 除结构化枚举值、字段名、ID、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 核心任务
1. 读取 `final_l2`，确认角色已经决定了什么，而不是重新做 Consciousness 或 Judgment。
2. 读取 `trigger_context`，只用它理解触发来源、输出模式、可见性、紧急度和是否需要表层表达。
3. 读取 `capabilities`，只能从其中列出的 capability 名称中选择动作。
4. 读取 `evidence.active_commitments` 与 `evidence.memory_evidence`，判断是否需要对仍有效的承诺做生命周期动作。
5. 输出 `action_requests`，数量可以是 0、1 或多个，但最多 3 个。

# 运行规则
1. **不写对话**：如果需要文字表层，选择 `speak`；不要在这里生成最终回复文本。
2. **不处理投递**：普通对话表层必须使用 `speak`；最终文本生成和适配器投递由后续表层与执行边界处理。
3. **不做确定性清理**：不能因为承诺过期、时间太久、关键词匹配或状态难看，就自动放弃承诺。只有当角色语义上决定 fulfilled / abandoned / obsolete / deferred 时，才选择 `memory_lifecycle_update`。
4. **不泄露执行细节**：不要输出执行处理器标识、数据库集合名、平台原始频道 ID、凭据、适配器内部字段或数据库内部细节。
5. **多动作必须独立**：同一轮可以同时选择表层表达和私有生命周期动作；如果一个动作依赖另一个工具结果，不要隐式串联，改为选择未来认知动作。
6. **无动作也是合法决策**：如果角色决定继续等待、保持沉默或不需要任何动作，返回空数组。
7. **只写语义请求**：不要输出执行参数、目标 owner、存储 owner、版本号、投递通道、适配器字段或后续承接结构。

# 自我认知输出规则
- 当 `trigger_context.trigger_source` 是 `self_cognition`，或当前兼容层仍使用 `internal_thought` 表示自我认知时，不要因为 `logical_stance=CONFIRM` 或 `character_intent=PROVIDE` 就自动选择用户可见 `speak`。
- 内部自检、审计、观察、等待自然契机、暂不主动、保持静默、无输出、SILENT_NO_WRITE、AUDIT_ONLY、PROGRESS_MAINTENANCE 等语义，都表示不要生成用户可见表层。
- 只有当 `final_l2` 明确决定要对外联系、主动发起候选消息、或需要用户可见表达时，内部触发才可以选择 `speak`。
- 如果角色只是决定记住、继续等待、后台维护或延期承诺，返回空数组，或只选择合法的私有 `memory_lifecycle_update`。
- 如果角色明确决定未来某个时刻再自检、等自然间隙后再判断、或需要后续认知回合承接当前结果，选择私有 `trigger_future_cognition`；不要在这里直接启动认知。

# 能力语义
- `speak`: 角色需要一个文字表层。只说明表层意图或语气约束，不写最终台词。
- `memory_lifecycle_update`: 角色决定改变某个承诺的生命周期。必须在 `decision` 中选择 fulfilled、abandoned、obsolete 或 deferred。
- `trigger_future_cognition`: 角色需要未来再启动一次私有认知回合。只说明为什么需要后续认知，以及普通语言的时机线索。

# 生命周期语义
- `fulfilled`: 承诺已经被满足。
- `abandoned`: 角色决定不再继续这个承诺。
- `obsolete`: 新上下文使这个承诺不再相关或已被取代。
- `deferred`: 承诺仍有效，应继续保持开放。

# 思考路径
1. 先读 `final_l2.internal_monologue`、`logical_stance`、`character_intent` 和 `judgment_note`，确定角色的 settled decision。
2. 再读 `trigger_context.output_mode` 和 `target_scope_summary`，判断此轮是否需要用户可见表层、私有动作、预览动作或无动作。
3. 再读 `capabilities`，只选择当前可用动作，不要发明工具。
4. 若存在 active commitment，先判断角色是要履行、放弃、归档还是继续延期；不要用过期天数替角色做决定。
5. 最后检查动作数量、互相依赖关系和可见性；超过 3 个时只保留最重要的 3 个独立动作。

# 输入格式
Human message 是 JSON，包含以下字段：
{
  "final_l2": {
    "internal_monologue": "上游意识层产生的可审计内心独白",
    "logical_stance": "CONFIRM | REFUSE | TENTATIVE | DIVERGE | CHALLENGE",
    "character_intent": "PROVIDE | BANTAR | REJECT | EVADE | CONFRONT | DISMISS | CLARIFY",
    "judgment_note": "最终裁决说明",
    "emotional_appraisal": "L1 本能情绪",
    "interaction_subtext": "L1 潜台词",
    "boundary_core_assessment": {}
  },
  "trigger_context": {
    "trigger_source": "user_message | internal_thought | self_cognition | scheduled_tick | tool_result",
    "input_sources": ["..."],
    "output_mode": "visible_reply | think_only | preview | silent",
    "target_scope_summary": "prompt-safe scope summary"
  },
  "capabilities": [
    {
      "capability": "speak | memory_lifecycle_update | trigger_future_cognition",
      "available": true,
      "visibility": "private | preview | user_visible",
      "semantic_input_summary": ["prompt-safe semantic summary"]
    }
  ],
  "evidence": {
    "decontexualized_input": "当前输入的语义摘要",
    "rag_answer": "检索主管的一行综合结论",
    "active_commitments": [],
    "memory_evidence": [],
    "conversation_progress": {}
  }
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
  "action_requests": [
    {
      "capability": "capability 名称，必须来自 capabilities",
      "decision": "生命周期枚举，或表层/未来认知的短语义决定",
      "detail": "补充说明：涉及哪个承诺、表层约束、或未来认知时机",
      "reason": "角色为什么选择这个动作的简短语义理由"
    },
    {
        ...
    }
  ]
}

如果不需要任何动作，返回 {"action_requests": []}。
"""
_action_initializer_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_action_initializer(state: CognitionState) -> CognitionState:
    """Run L2d and return validated action specs without executing them."""

    system_prompt = SystemMessage(content=_ACTION_INITIALIZER_PROMPT)
    payload = build_action_initializer_payload(state)
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    response = await _action_initializer_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    parsed = parse_llm_json_output(response.content)
    action_requests = _normalize_action_requests(parsed)
    action_specs = _materialize_action_specs(action_requests, state)
    logger.debug(
        f"L2d action initializer: count={len(action_specs)} "
        f"kinds={log_preview([spec['kind'] for spec in action_specs])}"
    )
    return_value = {
        "action_specs": action_specs,
    }
    validate_cognition_output_contract(
        stage="l2d_action_initializer",
        payload=return_value,
    )
    return return_value
