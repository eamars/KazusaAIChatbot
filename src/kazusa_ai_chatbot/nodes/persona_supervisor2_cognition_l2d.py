"""L2d action initializer for modality-neutral action specs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_spec.models import (
    ActionSourceRefV1,
    ActionSpecV1,
    CapabilitySpecV1,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
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


class ActionRequestV1(TypedDict, total=False):
    """Semantic action request emitted by L2d before deterministic wrapping."""

    capability: str
    decision: str
    reason: str
    detail: str


def build_action_initializer_payload(
    state: CognitionState,
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> str:
    """Build the current-run semantic context for L2d action selection.

    Args:
        state: Cognition state after L2c judgment.
        capabilities: Optional registry override for deterministic tests.

    Returns:
        One prompt-safe semantic string containing only dynamic turn context.
    """

    if capabilities is None:
        capabilities = build_initial_action_capabilities()
    prompt_capabilities = dict(capabilities)
    action_context = _build_action_context_text(state, prompt_capabilities)
    return action_context


def _build_action_context_text(
    state: CognitionState,
    prompt_capabilities: Mapping[str, CapabilitySpecV1],
) -> str:
    """Render the current cognition state as one model-facing context string."""

    episode = state["cognitive_episode"]
    input_sources = ", ".join(episode["input_sources"])
    evidence = _project_action_evidence(state)
    commitment_text = _commitment_context_text(
        state,
        memory_lifecycle_visible=(
            MEMORY_LIFECYCLE_UPDATE_CAPABILITY in prompt_capabilities
        ),
    )
    context_lines = [
        "当前行动上下文：",
        (
            f"触发来源：{episode['trigger_source']}；"
            f"输入来源：{input_sources}；"
            f"输出要求：{episode['output_mode']}；"
            f"场景：{state['channel_type']} 对话。"
        ),
        (
            "已形成的决定："
            f"立场={state['logical_stance']}；"
            f"意图={state['character_intent']}；"
            f"裁决={state['judgment_note']}；"
            f"内心判断={state['internal_monologue']}"
        ),
        (
            "即时感受："
            f"{state['emotional_appraisal']}；"
            f"互动潜台词：{state['interaction_subtext']}。"
        ),
        (
            "边界与社交语境："
            f"边界={_compact_context_value(state['boundary_core_assessment'])}；"
            f"距离={state['social_distance']}；"
            f"强度={state['emotional_intensity']}；"
            f"氛围={state['vibe_check']}；"
            f"关系={state['relational_dynamic']}。"
        ),
        f"当前输入摘要：{evidence['decontexualized_input']}",
        f"检索结论：{evidence['rag_answer'] or '无'}",
        commitment_text,
        f"相关记忆：{_evidence_list_text(evidence['memory_evidence'])}",
        f"对话进度：{_compact_context_value(evidence['conversation_progress'])}",
    ]
    group_engagement_text = _group_engagement_context_text(state)
    if group_engagement_text:
        context_lines.append(group_engagement_text)
    action_context = "\n".join(context_lines)
    return action_context


def _group_engagement_context_text(state: CognitionState) -> str:
    """Return group-channel engagement evidence for group self-cognition."""

    if not _is_group_self_cognition_context(state):
        return_value = ""
        return return_value

    raw_context = state.get("group_engagement_action_context")
    if not isinstance(raw_context, Mapping):
        return_value = "群聊参与习惯：无"
        return return_value

    raw_guidelines = raw_context.get("engagement_guidelines")
    guidelines: list[str] = []
    if isinstance(raw_guidelines, list):
        guidelines = [
            item.strip()
            for item in raw_guidelines
            if isinstance(item, str) and item.strip()
        ]
    if not guidelines:
        return_value = "群聊参与习惯：无"
        return return_value

    confidence = ""
    raw_confidence = raw_context.get("confidence")
    if isinstance(raw_confidence, str) and raw_confidence.strip():
        confidence = raw_confidence.strip()

    guideline_text = "；".join(guidelines)
    if confidence:
        return_value = f"群聊参与习惯：{guideline_text}；confidence={confidence}"
        return return_value

    return_value = f"群聊参与习惯：{guideline_text}"
    return return_value


def _is_group_self_cognition_context(state: CognitionState) -> bool:
    """Return whether L2d is selecting for a group self-cognition source."""

    if state["channel_type"] != "group":
        return_value = False
        return return_value

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = False
        return return_value

    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal_monologue = (
        isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    is_self_cognition = (
        trigger_source == "internal_thought"
        and is_internal_monologue
    )
    return is_self_cognition


def _commitment_context_text(
    state: CognitionState,
    *,
    memory_lifecycle_visible: bool,
) -> str:
    """Return the prompt-safe commitment context without persistence IDs."""

    projected_commitments = _project_active_commitments_for_prompt(
        _raw_active_commitments(state)
    )
    if not projected_commitments:
        commitment_text = "活动承诺线索：无。"
        return commitment_text

    commitment_lines = _evidence_list_text(projected_commitments)
    if memory_lifecycle_visible:
        commitment_text = f"活动承诺线索：有；{commitment_lines}"
        return commitment_text

    commitment_text = f"活动承诺线索：当前不允许生命周期复核；{commitment_lines}"
    return commitment_text


def _evidence_list_text(entries: object) -> str:
    """Render prompt-safe evidence entries as compact prose."""

    if not isinstance(entries, list) or not entries:
        return "无"

    rendered_entries: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            rendered_entry = _compact_mapping_text(entry)
        else:
            rendered_entry = _compact_context_value(entry)
        if rendered_entry != "无":
            rendered_entries.append(rendered_entry)

    if not rendered_entries:
        return "无"
    evidence_text = "；".join(rendered_entries)
    return evidence_text


def _compact_mapping_text(value: Mapping[str, object]) -> str:
    """Render a dictionary as compact prompt-facing prose."""

    rendered_pairs: list[str] = []
    for key, raw_value in value.items():
        rendered_value = _compact_context_value(raw_value)
        if rendered_value == "无":
            continue
        rendered_pairs.append(f"{key}={rendered_value}")

    if not rendered_pairs:
        return "无"
    mapping_text = "，".join(rendered_pairs)
    return mapping_text


def _compact_context_value(value: object) -> str:
    """Render one dynamic context value for the human message string."""

    if value is None:
        return "无"
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return "无"
        return stripped_value
    if isinstance(value, Mapping):
        mapping_text = _compact_mapping_text(value)
        return mapping_text
    if isinstance(value, list):
        list_text = _evidence_list_text(value)
        return list_text

    context_value = str(value)
    return context_value


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
    for index, request in enumerate(requests):
        continuation_objective: str | None = None
        if request["capability"] == TRIGGER_FUTURE_COGNITION_CAPABILITY:
            continuation_objective = _future_cognition_objective(
                requests,
                future_request_index=index,
            )
        action_spec = _materialize_action_request(
            request,
            state,
            continuation_objective=continuation_objective,
        )
        if action_spec is None:
            continue
        action_specs.append(action_spec)
        if len(action_specs) >= ACTION_SPEC_CAP:
            break
    return action_specs


def _materialize_action_request(
    request: ActionRequestV1,
    state: CognitionState,
    *,
    continuation_objective: str | None = None,
) -> ActionSpecV1 | None:
    """Build one validated action spec for a selected semantic capability."""

    capability = request["capability"]
    if capability == SPEAK_CAPABILITY:
        action_spec = _build_speak_action_spec(request, state)
    elif capability == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        action_spec = _build_memory_lifecycle_action_spec(request, state)
    elif capability == TRIGGER_FUTURE_COGNITION_CAPABILITY:
        if _is_scheduled_future_cognition_source(state):
            logger.warning(
                "L2d dropped future-cognition request from scheduled "
                "future-cognition source"
            )
            return None
        action_spec = _build_future_cognition_action_spec(
            request,
            state,
            continuation_objective=continuation_objective,
        )
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
    _state: CognitionState,
) -> dict[str, object] | None:
    """Build the specialist route intent for commitment lifecycle review."""

    detail = _semantic_text(request, "detail")
    if not detail:
        detail = request["reason"]
    action_spec = _build_action_spec(
        kind=MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        },
        params={
            "review_kind": "active_commitment_lifecycle",
            "detail": detail,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _build_future_cognition_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
    *,
    continuation_objective: str | None,
) -> dict[str, object]:
    """Build the deterministic envelope for a future cognition request."""

    if continuation_objective is None:
        continuation_objective = _semantic_text(request, "detail")
    if not continuation_objective:
        continuation_objective = request["reason"]
    action_spec = _build_action_spec(
        kind=TRIGGER_FUTURE_COGNITION_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": _future_cognition_target_scope(state),
        },
        params={
            "episode_type": "self_cognition",
            "trigger_at": None,
            "continuation_objective": continuation_objective,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        continuation=_scheduled_followup_continuation(),
        reason=request["reason"],
    )
    return action_spec


def _future_cognition_objective(
    requests: list[ActionRequestV1],
    *,
    future_request_index: int,
) -> str:
    """Return the one-string objective for a future cognition handoff."""

    future_request = requests[future_request_index]
    continuation_objective = _semantic_text(future_request, "detail")
    if continuation_objective:
        return continuation_objective

    continuation_objective = future_request["reason"]
    return continuation_objective


def _future_cognition_target_scope(state: CognitionState) -> dict[str, object]:
    """Bind trusted source scope for the later scheduled cognition slot."""

    scope: dict[str, object] = {
        "episode_type": "self_cognition",
    }
    field_map = (
        ("platform", "source_platform"),
        ("platform_channel_id", "source_channel_id"),
        ("channel_type", "source_channel_type"),
        ("global_user_id", "source_user_id"),
        ("platform_bot_id", "source_platform_bot_id"),
    )
    for state_field, scope_field in field_map:
        field_value = _state_text(state, state_field)
        if field_value:
            scope[scope_field] = field_value

    character_name = _character_name_for_scope(state)
    if character_name:
        scope["source_character_name"] = character_name
    return scope


def _state_text(state: CognitionState, field_name: str) -> str:
    """Return one optional text value from the trusted cognition state."""

    raw_value = state.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _character_name_for_scope(state: CognitionState) -> str:
    """Return the active character name when present in trusted state."""

    profile = state.get("character_profile")
    if not isinstance(profile, dict):
        return_value = ""
        return return_value

    name = profile.get("name")
    if not isinstance(name, str):
        return_value = ""
        return return_value

    return_value = name.strip()
    return return_value


def _is_scheduled_future_cognition_source(state: CognitionState) -> bool:
    """Return whether this cycle was itself started by a future-cognition slot."""

    conversation_progress = state.get("conversation_progress")
    if not isinstance(conversation_progress, dict):
        return False

    source = conversation_progress.get("source")
    is_scheduled_source = source == "scheduled_future_cognition"
    return is_scheduled_source


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
    continuation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the common deterministic action-spec envelope."""

    action_continuation = continuation
    if action_continuation is None:
        action_continuation = _no_continuation()
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
        "continuation": action_continuation,
        "reason": reason,
    }
    return action_spec


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


def _scheduled_followup_continuation() -> dict[str, object]:
    """Return the bounded continuation contract for future cognition."""

    continuation = {
        "schema_version": "action_continuation.v1",
        "mode": "scheduled_followup",
        "episode_type": "self_cognition",
        "max_depth": 1,
        "include_result_as": "scheduled_event",
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
你是角色的语义行动选择层。
前序理解已经形成当前事件的立场、意图、边界判断和社交语境。
用户消息只包含本轮动态行动上下文。
你的任务是把角色已经形成的行动意图整理成 0 到 3 个语义动作请求。
行动请求只描述角色想做什么，保持在语义层；不要生成最终发言文本，不要执行动作。

# 语言政策
- 除字段名、枚举值、ID、URL、代码、命令、模型标签等必须保持原样的内容外，你新生成的内部自由文本字段使用简体中文。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言。
- 保留原语言内容时只保留原文，不另加旁注解释。

# 可选动作
- `speak` 是可见文字回复。选择它表示角色决定把话说到当前外部频道；之后才会交给 L3/dialog 渲染为可见文本。`detail` 写清这个可见回复要处理的具体对象、问题、承诺或待处理目标。
- `memory_lifecycle_update` 表示当前回合可能影响活动承诺的兑现、放弃、过时或延期，需要专门复核活动承诺生命周期。只选择复核需要，不选择具体承诺、别名、数据库目标或生命周期决定。
- `trigger_future_cognition` 表示角色需要在未来拿到或消费一个具体新信息后再想一次。`detail` 必须写清等待或消费什么具体新信息，以及它会继续处理哪个具体问题、任务或承诺。

# 选择流程
1. 先阅读当前行动上下文，判断角色现在是否真的要把某件事外部化为动作。
2. 内心独白是证据，不是动作。私人好奇、只想观察、保持沉默、维护进度、等待更自然时机，除非已经决定外部化，否则都不是 `speak`。
3. 只有当前场景给了足够清楚的可见发言理由，并且角色愿意把该内容发到当前频道，才选择 `speak`。
4. 群聊参与习惯只是群频道互动证据。它可以帮助判断当前场景是否有合适开口，但不能替代当前场景，也不能命令角色发言。
5. 当前活动承诺可能被本轮输入或已形成决定影响时，选择 `memory_lifecycle_update`，并在 `detail` 写清需要复核的语义原因。
6. 当前回合存在具体未完成问题，且继续处理依赖未来新信息时，选择 `trigger_future_cognition`。
7. 没有需要外部化或私有处理的真实动作时，返回空数组。
8. 同一轮可以选择多个彼此独立的动作，最多 3 个。

# 未来认知判断
未来认知动作承接需要下一轮思考的信息缺口。
合格的 `detail` 同时包含两部分：
- 未来要等待或消费的具体新信息；
- 该信息将继续处理的具体问题、任务或承诺。

普通等待、情绪余波、关系观察和更自然的时机属于对话进度，不生成未来认知动作。
如果未来认知动作与 `speak` 并列，`speak.detail` 写当前可见回复目标，`trigger_future_cognition.detail` 写下一轮思考目标。

# 输入格式
用户消息是一段中文行动上下文字符串，不是 JSON。
它描述当前回合的动态信息：触发来源、输出要求、已形成的决定、即时感受、社交语境、当前输入摘要、检索结论、活动承诺线索、相关记忆、对话进度，以及可能出现的群聊参与习惯。
当出现 `群聊参与习惯` 时，把它当作抽象参与证据；它只说明这个群频道通常如何承接、观察或推进，不能替代当前场景。

# 输出格式
只返回合法 JSON 字符串：
{
  "action_requests": [
    {
      "capability": "speak | memory_lifecycle_update | trigger_future_cognition",
      "decision": "文字表层或未来认知的简短语义决定；memory_lifecycle_update 可省略或留空",
      "detail": "一个精确语义字符串",
      "reason": "选择这个动作的简短语义理由"
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
    action_context = build_action_initializer_payload(state)
    human_message = HumanMessage(content=action_context)
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
        stage="l2d_action_selection",
        payload=return_value,
    )
    return return_value
