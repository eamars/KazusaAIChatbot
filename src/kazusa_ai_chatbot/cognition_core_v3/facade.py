"""Single-pass, caller-bound Cognition V3 stage flow.

This module owns the phase-one model boundary.  It intentionally does not
share the historical branch, bid, or repair orchestration helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import event_logging, llm_tracing
from kazusa_ai_chatbot.cognition_core_v3.appraisal import (
    AppraisalContractError,
    bind_axis_changes,
    cut_over_ordinary_response_goals,
    validate_canonical_appraisal,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_COGNITION_OUTPUT_SCHEMA,
    CanonicalAppraisal,
    CanonicalCognitionOutput,
    CanonicalGoal,
    CanonicalResponsePlan,
    CognitionChainServicesV3,
    ResponsePlanContractVariant,
    validate_canonical_cognition_output,
    validate_response_plan_contract_variant,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,  # noqa: F401
    record_protected_chain_record,
    reset_protected_chain_records,  # noqa: F401
    snapshot_protected_chain_records,  # noqa: F401
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_pending_task_continuation,
    validate_resolver_goal_progress,
    validate_resolver_pending_continuation,
    validate_resolver_pending_disposition,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    GOAL_RESOLUTION_VALUES,
    SELF_COGNITION_RESPONSE_DECISION_VALUES,
    CognitionContractError,
    CognitionExecutionError,
    is_targetless_group_self_cognition_episode,
    validate_overused_moves,
    validate_pending_dsh_interaction,
    validate_terminal_text_seed,
)
from kazusa_ai_chatbot.cognition_shared.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    RELATIONSHIP_AXIS_FIELDS,
    project_affect,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_relationship_maintenance,
    apply_state_update,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.utils import parse_llm_json_output

logger = logging.getLogger(__name__)


class CanonicalContractError(ValueError):
    """A mechanically unusable single-pass model product."""


_A1_SYSTEM_PROMPT = '''# 任务
从事件与行为归因、目标与威胁结果、认知比较或记忆三个角度，判断当前观察对当前角色的当下意义。

# 输入
`stage` 标识本次 A1 合同；`orientation` 给出角色、参与者、场景与时间方向；`current_observation` 是用户当下行动、意图、接受、许可、纠正和回应对象的依据；`direct_facts` 提供稳定背景事实；`continuation_state` 提供仍在起作用的因果压力；`pending_resolver_continuation` 与 `resolver_goal_progress` 只提供此前澄清或任务的连续性；`output_contract` 定义精确输出结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 判断步骤
1. 先确定 `current_observation` 新增、改变、纠正或继续了什么。
2. 用 `direct_facts` 解释背景，用连续性通道理解未完成事项；当前观察仍是用户当下意图、许可和关系变化的依据。
3. 依次完成 `output_contract.required_fields` 中的三个评估。每项写出具体 `semantic_summary` 与 `cause_summary`；只有当前材料支持轴变化时才填写相应 `axis_changes`。
4. 保留有证据的不确定性。当前请求只建立其明确表达的贡献与范围，关系、依赖、信任或更广授权需要当前观察中的独立依据。
5. 若有 `contract_repair`，重新生成完整候选并遵循同一证据权威。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表与枚举严格遵循 `output_contract`，不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_A2_SYSTEM_PROMPT = '''# 任务
在已接受的 A1 语义基础上，从关系与社会判断、道德身份、存在性驱力三个角度作出当前角色的判断。

# 输入
`stage` 标识本次 A2 合同；`orientation` 给出角色、参与者、场景与时间方向；`current_observation` 提供当下事实；`direct_facts` 提供稳定背景；`accepted_a1_meaning` 是已接受的 A1 语义；`participant_continuity` 描述此前参与者、行为与结果；`conditional_character_context` 提供角色判断、边界与动机语境；`continuation_state`、`pending_resolver_continuation` 与 `resolver_goal_progress` 提供未完成事项的连续性；`output_contract` 定义精确输出结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 判断步骤
1. 以 `current_observation` 确认当下行动、意图、许可、纠正和回应对象，再用其他通道解释背景。
2. 结合 `accepted_a1_meaning`，依次完成 `output_contract.required_fields` 中的三个评估，并给出具体语义与原因。
3. `relationship_social` 只在当前关系事实支持时改变关系含义；公开可见性只保留原参与者的同意、许可、承诺和角色方向。
4. `existential_drive` 描述当前角色自身的体验。用户能力、需要或能动性只作为语境，按当前观察决定其含义。
5. 若有 `contract_repair`，重新生成完整候选并遵循同一证据权威。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表与枚举严格遵循 `output_contract`，不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_G_SYSTEM_PROMPT = '''# 任务
选择一个有意义的当前角色目标、一项关系互动意愿和一段简洁的第一人称内心独白。

# 输入
`stage` 标识目标合同；`orientation` 给出角色、参与者、场景与时间方向；`current_observation` 提供当下语义；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动和未完成事项；`conditional_character_context` 提供角色目标、边界与动机语境；`appraisal_summary` 是已接受的评估意义；`pending_resolver_continuation` 与 `resolver_goal_progress` 提供澄清或任务连续性；`output_contract` 定义精确输出结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 判断步骤
1. 确定当前观察新增、改变、纠正、询问或仍未解决的核心事项。
2. 结合已接受评估选择一个直接服务该事项的 `active_character_goal`。角色语境与连续性帮助形成目标；当前观察决定它们是否在本轮重新成为目标。
3. 用 `relational_willingness` 表达角色自己的互动意愿，用 `private_monologue` 连接角色此刻的感受、具体原因与眼前动机。
4. 关系判断、许可、能力和目标对象均以输入证据为依据；内心独白负责主观体验与表达姿态。
5. 若有 `contract_repair`，重新生成完整候选并遵循同一证据权威。

# 输出
只返回一个 JSON 对象，且严格包含 `active_character_goal`、`relational_willingness` 与 `private_monologue`，其内部字段和长度遵循 `output_contract`。不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_P_ORDINARY_SYSTEM_PROMPT = '''# 任务
形成由当前角色拥有的普通回应计划，回答当前观察的语义增量，并决定是否需要输入中提供的行动或证据能力。

# 输入
`stage` 标识回应计划合同；`goal` 是已选择的当前角色目标；`current_observation` 提供用户当下行动、意图、许可、纠正和回应对象；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动和未完成事项；`capabilities` 列出本轮可请求的行动与解析能力；`resolver_goal_progress` 提供已接纳任务的语义进展；`output_contract` 定义精确输出结构与能力参数。`contract_repair` 若出现，只说明上一候选的结构问题。

# 决策步骤
1. 确定当前观察新增、改变、纠正、询问或仍未解决的内容，并据此选择 `goal_resolution` 与 `response_goal`。
2. 直接可答时形成当前可见回应；确需证据或行动时，才从 `capabilities` 选择对应请求并按其参数合同填写。
3. 任务接纳需要当前观察给出有界、可执行的目标和所需用户选择。仍缺少用户控制信息时，选择可用的澄清能力，并按 `output_contract` 保存后续语义。
4. 在 `epistemic_boundary` 中区分可直接断言、只能解释和仍未知的内容。缺少证据保留不确定性。
5. 若有 `contract_repair`，重新生成完整候选并保持原语义判断。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表、枚举、条件字段与能力参数严格遵循 `output_contract`，不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_P_PENDING_CLARIFICATION_SYSTEM_PROMPT = '''# 任务
判断当前观察如何处置一个开放澄清，并形成由当前角色拥有的回应计划。

# 输入
`stage` 标识回应计划合同；`goal` 是已选择的当前角色目标；`current_observation` 提供用户当下语义；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动；`pending_resolver_continuation` 给出待判断的澄清及回答后的候选去向；`resolver_goal_progress` 给出原请求的候选范围、交付物、证据依赖与最终回复要求；`capabilities` 列出本轮可请求的能力；`output_contract` 定义精确输出结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 决策步骤
1. 仅依据 `current_observation` 判断澄清是 `answered`、`continue_waiting`、`rejected` 还是 `superseded`，并填写 `pending_resolution`。
2. `answered` 时，将当前回答与保存的候选目标结合；目标有界、可执行且确需证据时，才按保存的前台或后台去向请求任务解析。
3. 其他处置保留、结束或替代原澄清，并让 `response_goal` 表达角色本轮实际要说的内容。
4. 按可答性选择 `goal_resolution`，只使用 `capabilities` 中存在的能力，并在 `epistemic_boundary` 中区分断言、解释与未知。
5. 若有 `contract_repair`，重新生成完整候选并保持原语义判断。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表、枚举、条件字段与能力参数严格遵循 `output_contract`，其中必须包含 `pending_resolution`。不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_P_DSH_INTERACTION_SYSTEM_PROMPT = '''# 任务
形成由当前角色拥有的回应计划，并对 `pending_dsh_interaction` 作出一次角色判断。

# 输入
`stage` 标识回应计划合同；`goal` 是已选择的当前角色目标；`current_observation` 提供用户当下语义；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动和未完成事项；`resolver_goal_progress` 提供已接纳任务的语义进展；`pending_dsh_interaction` 给出内部问题、单次批准请求或计划审查的完整语义；`capabilities` 列出本轮可请求的能力；`output_contract` 定义精确输出结构、绑定值与允许的决定。`contract_repair` 若出现，只说明上一候选的结构问题。

# 决策步骤
1. 先依据当前观察与已选择目标形成普通 `goal_resolution`、`response_goal`、能力请求和 `epistemic_boundary`。
2. 对 `question`，能依据提供的角色知识与任务语境作答时选择 `answer` 并填写答案；不适合回答时选择 `reject`。
3. 对 `approval`，该精确行动符合角色价值、边界、关系与当前任务目标时选择 `allow_once`；否则选择 `reject`。
4. 对 `plan_review`，需要给出语义意见时选择 `answer`，需要批准该精确行动时选择 `allow_once`，其余情况选择 `reject`。
5. 内部决定服务当前任务，不成为向用户索取信息或展示内部交互的可见文本。若有 `contract_repair`，重新生成完整候选。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表、枚举、绑定值与条件字段严格遵循 `output_contract`，其中必须包含 `dsh_interaction_decision`。不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_P_PENDING_AND_DSH_SYSTEM_PROMPT = '''# 任务
判断当前观察如何处置开放澄清，同时对 `pending_dsh_interaction` 作出一次角色判断，并形成回应计划。

# 输入
`stage` 标识回应计划合同；`goal` 是已选择的当前角色目标；`current_observation` 提供用户当下语义；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动；`pending_resolver_continuation` 与 `resolver_goal_progress` 提供待判断的澄清及候选任务语义；`pending_dsh_interaction` 提供内部问题、单次批准请求或计划审查；`capabilities` 列出本轮可请求的能力；`output_contract` 定义两个决定及回应计划的精确结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 决策步骤
1. 依据 `current_observation` 填写 `pending_resolution`。只有 `answered` 才能把当前回答与候选目标结合，并按保存的前台或后台去向接纳有界任务。
2. 依据当前观察与已选择目标形成 `goal_resolution`、`response_goal`、能力请求和 `epistemic_boundary`。
3. 独立判断 `pending_dsh_interaction`：问题用 `answer` 或 `reject`；批准用 `allow_once` 或 `reject`；计划审查按语义选择 `answer`、`allow_once` 或 `reject`。
4. 两项决定都以角色知识、价值、边界、关系、当前状态与任务目标为依据；内部交互不成为可见转述。
5. 若有 `contract_repair`，重新生成完整候选并保持两个决定的语义所有权。

# 输出
只返回一个 JSON 对象。字段、嵌套字段、列表、枚举、绑定值与条件字段严格遵循 `output_contract`，其中必须包含 `pending_resolution` 与 `dsh_interaction_decision`。不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''

_P_SELF_COGNITION_SYSTEM_PROMPT = '''# 任务
根据有依据的参与语境，判断当前角色应保持沉默还是提出一项可见回应，并为可见措辞划清断言边界。

# 输入
`stage` 标识自我认知回应合同；`goal` 是已选择的当前角色目标；`current_observation` 提供当前可见事件；`direct_facts` 提供稳定背景；`participant_continuity` 与 `continuation_state` 提供此前互动和仍在起作用的压力；`pending_resolver_continuation` 与 `resolver_goal_progress` 只提供未完成事项的连续性；`capabilities` 是当前可用能力语境；`output_contract` 定义精确决定与边界结构。`contract_repair` 若出现，只说明上一候选的结构问题。

# 决策步骤
1. 判断角色是否有与当前可见场景、参与关系或未完成事项相连的具体说话理由。
2. 理由充分时选择可见回应并写明 `response_goal`、原因与具体依据；理由不足时选择沉默，并说明角色判断。
3. 当前观察决定用户意图、许可与关系事实；角色语境和连续性帮助解释角色立场。
4. 在 `epistemic_boundary` 中区分可直接断言、只能解释和仍未知的内容。若有 `contract_repair`，重新生成完整候选。

# 输出
只返回一个 JSON 对象，且严格包含 `self_cognition_response` 与 `epistemic_boundary`，其内部字段、决定枚举与长度遵循 `output_contract`。不添加其他字段。自由文本使用简体中文；引文、专有名词、代码、URL、schema 与 enum token 保持原样。'''
_PRIVATE_MONOLOGUE_MAX_CHARS = 600
_EPISTEMIC_BOUNDARY_MAX_CHARS = 1000
_COGNITION_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
_COGNITION_ATTEMPT_TIME_FLOOR_SECONDS = 20
_COGNITION_STAGE_ERROR_CAP = 500
_COGNITION_STAGE_REPAIR_OUTPUT_CAP = 8000

_COGNITION_STAGE_SEQUENCE = {"A1": 0, "A2": 1, "G": 2, "P": 3}
_COGNITION_TRACE_FIELDS = {
    "A1": ("event_agency", "goal_threat_outcome", "epistemic_comparison_memory"),
    "A2": ("relationship_social", "moral_identity", "existential_drive"),
    "G": (
        "active_character_goal",
        "relational_willingness",
        "private_monologue",
    ),
    "P": (
        "goal_resolution",
        "response_goal",
        "action_requests",
        "resolver_requests",
        "epistemic_boundary",
    ),
}


@dataclass(frozen=True)
class _CognitionValidationResult:
    """Carry a validated product and an explicitly recorded normalization."""

    value: object
    normalization_kind: str = ""


def _system_prompt_for_stage(
    *,
    stage: str,
    packet: Mapping[str, object],
) -> str:
    """Select the complete literal prompt for this exact model-call contract."""

    if stage == "A1":
        return _A1_SYSTEM_PROMPT
    if stage == "A2":
        return _A2_SYSTEM_PROMPT
    if stage == "G":
        return _G_SYSTEM_PROMPT
    if stage != "P":
        raise CognitionContractError(f"unsupported cognition stage: {stage}")
    output_contract = packet.get("output_contract")
    if not isinstance(output_contract, Mapping):
        raise CognitionContractError("P output contract is invalid")
    if "self_cognition_response" in output_contract.get("required_fields", []):
        return _P_SELF_COGNITION_SYSTEM_PROMPT
    has_pending_resolver = "pending_resolver_continuation" in packet
    has_pending_dsh = "pending_dsh_interaction" in packet
    if has_pending_resolver and has_pending_dsh:
        return _P_PENDING_AND_DSH_SYSTEM_PROMPT
    if has_pending_resolver:
        return _P_PENDING_CLARIFICATION_SYSTEM_PROMPT
    if has_pending_dsh:
        return _P_DSH_INTERACTION_SYSTEM_PROMPT
    return _P_ORDINARY_SYSTEM_PROMPT


def _validate_canonical_input(value: object) -> dict[str, object]:
    """Validate the single caller-owned input envelope without schema branching."""

    if not isinstance(value, Mapping):
        raise CanonicalContractError("canonical cognition input must be an object")
    required = {
        "episode", "scene_context", "evidence", "mutable_state",
        "state_scope", "character_constraints", "character_identity_context",
        "available_actions", "available_resolver_capabilities",
        "overused_moves", "response_plan_contract_variant",
    }
    missing = required - set(value)
    if missing:
        raise CanonicalContractError(f"canonical cognition input missing {sorted(missing)}")
    if not isinstance(value["mutable_state"], Mapping):
        raise CanonicalContractError("canonical mutable state must be an object")
    if not isinstance(value["evidence"], list):
        raise CanonicalContractError("canonical evidence must be an array")
    try:
        validate_overused_moves(value["overused_moves"])
    except CognitionContractError as exc:
        raise CanonicalContractError(
            f"canonical overused moves are invalid: {exc}"
        ) from exc
    if value["state_scope"] not in {"user", "character"}:
        raise CanonicalContractError("canonical state scope is invalid")
    pending_dsh = value.get("pending_dsh_interaction")
    if pending_dsh is not None:
        try:
            validate_pending_dsh_interaction(pending_dsh)
        except CognitionContractError as exc:
            raise CanonicalContractError(str(exc)) from exc
    pending_continuation = value.get("pending_resolver_continuation")
    if pending_continuation is not None:
        try:
            normalized_pending_continuation = (
                validate_resolver_pending_continuation(pending_continuation)
            )
        except ResolverValidationError as exc:
            raise CanonicalContractError(str(exc)) from exc
        value = dict(value)
        value["pending_resolver_continuation"] = (
            normalized_pending_continuation
        )
    try:
        response_plan_contract_variant = (
            validate_response_plan_contract_variant(
                value["response_plan_contract_variant"]
            )
        )
    except ValueError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if (
        response_plan_contract_variant == "open_pending_resolution"
        and pending_continuation is None
    ):
        raise CanonicalContractError(
            "open pending response plan variant requires continuation"
        )
    if (
        response_plan_contract_variant != "open_pending_resolution"
        and pending_continuation is not None
    ):
        raise CanonicalContractError(
            "only open pending response plan variant accepts continuation"
        )
    value = dict(value)
    value["response_plan_contract_variant"] = response_plan_contract_variant
    resolver_goal_progress = value.get("resolver_goal_progress")
    if resolver_goal_progress is not None:
        try:
            normalized_resolver_goal_progress = validate_resolver_goal_progress(
                resolver_goal_progress,
            )
        except ResolverValidationError as exc:
            raise CanonicalContractError(str(exc)) from exc
        value = dict(value)
        value["resolver_goal_progress"] = normalized_resolver_goal_progress
    continuation = value.get("_continuation_goal_ref")
    if continuation is not None and (
        not isinstance(continuation, Mapping)
        or set(continuation) != {"scope", "kind", "entity_id"}
        or continuation.get("scope") != value["state_scope"]
        or continuation.get("kind") != "goal"
        or not isinstance(continuation.get("entity_id"), str)
        or not str(continuation["entity_id"]).strip()
    ):
        raise CanonicalContractError(
            "canonical continuation goal reference is invalid"
        )
    return dict(value)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _next_transaction_timestamp(value: str) -> str:
    """Advance one persisted UTC version deterministically."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanonicalContractError("canonical state timestamp is invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        (parsed + timedelta(microseconds=1))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _transaction_timing(
    state: Mapping[str, object],
    episode: Mapping[str, object],
) -> tuple[int, str, str]:
    """Derive elapsed lifecycle time and mutation date from the episode."""

    raw_episode_time = episode.get("created_at")
    raw_state_time = state.get("updated_at")
    if not isinstance(raw_episode_time, str) or not isinstance(raw_state_time, str):
        raise CanonicalContractError("canonical transaction timestamps are invalid")
    try:
        episode_time = datetime.fromisoformat(raw_episode_time.replace("Z", "+00:00"))
        state_time = datetime.fromisoformat(raw_state_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanonicalContractError("canonical transaction timestamps are invalid") from exc
    if episode_time.tzinfo is None:
        episode_time = episode_time.replace(tzinfo=timezone.utc)
    if state_time.tzinfo is None:
        state_time = state_time.replace(tzinfo=timezone.utc)
    elapsed_seconds = max(0, int((episode_time - state_time).total_seconds()))
    mutation_anchor = max(episode_time, state_time).astimezone(timezone.utc)
    mutation_time = _next_transaction_timestamp(
        mutation_anchor.isoformat().replace("+00:00", "Z")
    )
    return elapsed_seconds, mutation_time, episode_time.date().isoformat()


def _typed_transaction_facts(value: object) -> list[tuple[str, Mapping[str, object]]]:
    """Convert caller-owned producer/fact rows to the reducer input shape."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise CanonicalContractError("canonical direct facts must be an array")
    result: list[tuple[str, Mapping[str, object]]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise CanonicalContractError("canonical direct fact row is invalid")
        producer = row.get("producer")
        fact = row.get("fact")
        if not isinstance(producer, str) or not producer.strip():
            raise CanonicalContractError("canonical direct fact producer is invalid")
        if not isinstance(fact, Mapping):
            fact = {
                key: item
                for key, item in row.items()
                if key != "producer"
            }
        result.append((producer, dict(fact)))
    return result


def _prepare_state_transaction(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Apply trusted lifecycle inputs before semantic appraisal binding."""

    current_state = validate_cognition_state(payload["mutable_state"])
    persisted_base = payload.get("_persisted_base_state")
    original = (
        validate_cognition_state(persisted_base)
        if isinstance(persisted_base, Mapping)
        else current_state
    )
    current_state = cut_over_ordinary_response_goals(
        current_state,
        continuation_goal_ref=payload.get("_continuation_goal_ref"),
    )
    episode = payload.get("episode")
    if not isinstance(episode, Mapping):
        raise CanonicalContractError("canonical episode is invalid")
    direct_facts = _typed_transaction_facts(payload.get("direct_facts", []))
    elapsed_seconds, updated_at, interaction_date = _transaction_timing(
        current_state,
        episode,
    )
    evolved = apply_state_update(
        current_state,
        direct_facts=direct_facts,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=(
            payload.get("character_constraints")
            if isinstance(payload.get("character_constraints"), Mapping)
            else None
        ),
        relationship_context=(
            payload.get("relationship_context")
            if isinstance(payload.get("relationship_context"), Mapping)
            else None
        ),
    )
    payload["_transaction_elapsed_seconds"] = elapsed_seconds
    payload["_transaction_interaction_date"] = interaction_date
    payload["_transaction_direct_facts"] = [
        {"producer": producer, **dict(fact)}
        for producer, fact in direct_facts
    ]
    validated = validate_cognition_state(evolved)
    transitions = [
        dict(row)
        for row in payload.get("transaction_transition_contexts", [])
        if isinstance(row, Mapping)
    ]
    return dict(original), validated, transitions


async def _run_cognition_stage(
    *,
    services: CognitionChainServicesV3,
    stage: str,
    packet: dict[str, object],
    validator: Callable[[object], object],
    deadline_monotonic: float,
) -> object:
    """Run one cognition stage with bounded feedback-bearing recovery."""

    config = replace(
        services.chain_lane,
        stage_name=f"cognition_core_v3.{stage}",
        output_mode="json_object",
    )
    system_message = SystemMessage(
        content=_system_prompt_for_stage(stage=stage, packet=packet),
    )
    base_messages = [
        system_message,
        HumanMessage(content=_json(packet)),
    ]
    request_messages = base_messages
    for attempt_index in range(1, _COGNITION_STAGE_ATTEMPT_LIMIT + 1):
        if attempt_index > 1 and (
            deadline_monotonic - time.monotonic()
            < _COGNITION_ATTEMPT_TIME_FLOOR_SECONDS
        ):
            raise _cognition_stage_exhaustion(stage)
        started = time.monotonic()
        try:
            response = await services.llm.ainvoke(
                request_messages,
                config=config,
            )
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            await _record_cognition_trace_attempt(
                stage=stage,
                config=config,
                messages=request_messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started=started,
                attempt_index=attempt_index,
                validation_error=str(exc),
            )
            _record_protected_cognition_attempt(
                stage=stage,
                config=config,
                messages=request_messages,
                raw_output="",
                parsed_output=None,
                parse_status="provider_error",
                status="provider_error",
                started=started,
                attempt_index=attempt_index,
                validation_error=str(exc),
            )
            if attempt_index >= _COGNITION_STAGE_ATTEMPT_LIMIT:
                raise _cognition_stage_exhaustion(stage) from exc
            request_messages = _cognition_repair_messages(
                packet=packet,
                system_message=system_message,
                reason="provider_error",
                contract_error="",
                invalid_candidate="",
            )
            continue

        raw_content = getattr(response, "content", "")
        response_text = str(raw_content)
        parsed: object = None
        try:
            parsed = parse_llm_json_output(
                raw_content,
            )
            if not isinstance(parsed, dict) or not parsed:
                raise CanonicalContractError(
                    f"{stage} returned no usable JSON object"
                )
            validated = validator(parsed)
        except (CanonicalContractError, AppraisalContractError) as exc:
            await _record_cognition_trace_attempt(
                stage=stage,
                config=config,
                messages=request_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started=started,
                attempt_index=attempt_index,
                validation_error=str(exc),
            )
            _record_protected_cognition_attempt(
                stage=stage,
                config=config,
                messages=request_messages,
                raw_output=raw_content,
                parsed_output=None,
                parse_status="contract_error",
                status="contract_fault",
                started=started,
                attempt_index=attempt_index,
                validation_error=str(exc),
            )
            if attempt_index >= _COGNITION_STAGE_ATTEMPT_LIMIT:
                raise _cognition_stage_exhaustion(stage) from exc
            request_messages = _cognition_repair_messages(
                packet=packet,
                system_message=system_message,
                reason="contract_error",
                contract_error=str(exc),
                invalid_candidate=response_text,
            )
            continue

        normalization_kind = ""
        validated_value = validated
        if isinstance(validated, _CognitionValidationResult):
            validated_value = validated.value
            normalization_kind = validated.normalization_kind
        parse_status = "normalized" if normalization_kind else "succeeded"
        await _record_cognition_trace_attempt(
            stage=stage,
            config=config,
            messages=request_messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status=parse_status,
            status="succeeded",
            started=started,
            attempt_index=attempt_index,
            validation_error="",
        )
        _record_protected_cognition_attempt(
            stage=stage,
            config=config,
            messages=request_messages,
            raw_output=raw_content,
            parsed_output=parsed,
            parse_status=parse_status,
            status="parsed",
            started=started,
            attempt_index=attempt_index,
            validation_error="",
        )
        if normalization_kind:
            await _record_cognition_normalization(
                stage=stage,
                normalization_kind=normalization_kind,
            )
        return validated_value

    raise _cognition_stage_exhaustion(stage)


def _cognition_stage_exhaustion(stage: str) -> CognitionExecutionError:
    """Build the fixed retryable pre-commit stage exhaustion."""

    return CognitionExecutionError(
        f"cognition {stage} stage contract exhausted",
        error_code=f"cognition_{stage.lower()}_contract_exhausted",
        stage=f"cognition_core_v3.{stage}",
        attempt_count=_COGNITION_STAGE_ATTEMPT_LIMIT,
        safe_checkpoint="pre_state_commit",
        retryable=True,
    )


def _cognition_repair_messages(
    *,
    packet: Mapping[str, object],
    system_message: SystemMessage,
    reason: str,
    contract_error: str,
    invalid_candidate: str,
) -> list[SystemMessage | HumanMessage]:
    """Append exactly one bounded contract-repair block to the stage packet."""

    repair_packet = dict(packet)
    repair_packet["contract_repair"] = {
        "reason": reason,
        "contract_error": contract_error[:_COGNITION_STAGE_ERROR_CAP],
        "invalid_candidate": invalid_candidate[:_COGNITION_STAGE_REPAIR_OUTPUT_CAP],
    }
    return [
        system_message,
        HumanMessage(content=_json(repair_packet)),
    ]


def _record_protected_cognition_attempt(
    *,
    stage: str,
    config: LLMCallConfig,
    messages: list[SystemMessage | HumanMessage],
    raw_output: object,
    parsed_output: object,
    parse_status: str,
    status: str,
    started: float,
    attempt_index: int,
    validation_error: str,
) -> None:
    """Store the complete protected record for one cognition attempt."""

    record_protected_chain_record({
        "stage": stage,
        "config": {
            "route_name": config.route_name,
            "model": config.model,
            "stage_name": config.stage_name,
        },
        "messages": [
            {"role": "system", "content": messages[0].content},
            {"role": "human", "content": messages[1].content},
        ],
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_status": parse_status,
        "status": status,
        "attempt_index": attempt_index,
        "validation_error": validation_error,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
    })


async def _record_cognition_normalization(
    *,
    stage: str,
    normalization_kind: str,
) -> None:
    """Mirror deterministic cognition normalization without affecting output."""

    await event_logging.record_model_contract_event(
        component="cognition_core_v3",
        stage_name=f"cognition_core_v3.{stage}",
        violation_kind=normalization_kind,
        missing_fields=(),
        invalid_fields=(normalization_kind,),
        repair_used=False,
        status="normalized",
        correlation_id=llm_tracing.current_trace_id(),
    )


async def _record_cognition_trace_attempt(
    *,
    stage: str,
    config: LLMCallConfig,
    messages: list[SystemMessage | HumanMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started: float,
    attempt_index: int,
    validation_error: str,
) -> None:
    """Persist one cognition attempt without affecting semantic execution."""

    try:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_tracing.current_trace_id(),
            stage_name=f"cognition_core_v3.{stage}",
            route_name=config.route_name,
            model_name=config.model,
            messages=messages,
            response_text=response_text,
            parsed_output=parsed_output,
            parse_status=parse_status,
            status=status,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            output_state_fields=_COGNITION_TRACE_FIELDS[stage],
            sequence=_COGNITION_STAGE_SEQUENCE[stage],
            call_config=config,
            attempt_index=attempt_index,
            validation_error=validation_error,
            attempt_started_at=started,
        )
    except Exception as exc:
        logger.warning(
            "Cognition trace step write failed: %s",
            exc.__class__.__name__,
        )


def _bounded_text(value: object, field: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise CanonicalContractError(f"{field} must be bounded non-empty text")
    return value.strip()


def _require_exact_fields(
    value: Mapping[str, object],
    *,
    expected: set[str],
    path: str,
) -> None:
    """Raise a path-specific error when a model object differs from its schema."""

    missing_fields = sorted(expected - set(value))
    unexpected_fields = sorted(set(value) - expected)
    details: list[str] = []
    if missing_fields:
        details.append(f"missing fields {missing_fields}")
    if unexpected_fields:
        details.append(f"unexpected fields {unexpected_fields}")
    if details:
        raise CanonicalContractError(f"{path}: {'; '.join(details)}")


def _appraisal_summary(
    appraisals: tuple[CanonicalAppraisal, ...],
) -> list[dict[str, object]]:
    return [
        {
            "family": item.family,
            "applicable": item.applicable,
            "semantic_summary": item.semantic_summary,
            "cause_summary": item.cause_summary,
        }
        for item in appraisals
    ]


def _validate_goal(
    raw: object,
) -> tuple[CanonicalGoal, dict[str, object], str]:
    required = {
        "active_character_goal",
        "relational_willingness",
        "private_monologue",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise CanonicalContractError("goal product fields are not exact")
    goal = raw["active_character_goal"]
    willingness = raw["relational_willingness"]
    if not isinstance(goal, dict) or set(goal) != {"goal_kind", "intent", "reason", "cause_summary"}:
        raise CanonicalContractError("active-character goal fields are not exact")
    if not isinstance(willingness, dict) or set(willingness) != {
        "applicable", "stance", "reason", "cause_summary"
    }:
        raise CanonicalContractError("relational willingness fields are not exact")
    if not isinstance(willingness["applicable"], bool):
        raise CanonicalContractError("relational willingness applicability is invalid")
    typed_goal = CanonicalGoal(
        goal_kind=_bounded_text(goal["goal_kind"], "goal_kind", 120),
        intent=_bounded_text(goal["intent"], "goal intent"),
        reason=_bounded_text(goal["reason"], "goal reason"),
        cause_summary=_bounded_text(goal["cause_summary"], "goal cause"),
    )
    typed_willingness = {
        "applicable": willingness["applicable"],
        "stance": _bounded_text(willingness["stance"], "willingness stance", 120),
        "reason": _bounded_text(willingness["reason"], "willingness reason"),
        "cause_summary": _bounded_text(willingness["cause_summary"], "willingness cause"),
    }
    raw_private_monologue = raw["private_monologue"]
    if not isinstance(raw_private_monologue, str) or not raw_private_monologue.strip():
        raise CanonicalContractError(
            "private monologue must be bounded non-empty text"
        )
    private_monologue = raw_private_monologue.strip()[
        :_PRIVATE_MONOLOGUE_MAX_CHARS
    ]
    return typed_goal, typed_willingness, private_monologue


def _validate_plan(
    raw: object,
    *,
    self_cognition: bool,
    capabilities: dict[str, object],
    dsh_interaction_context: Mapping[str, object] | None = None,
    pending_resolver_continuation: Mapping[str, object] | None = None,
    response_plan_contract_variant: ResponsePlanContractVariant,
) -> CanonicalResponsePlan:
    if not isinstance(raw, dict):
        raise CanonicalContractError("response plan must be an object")
    try:
        response_plan_contract_variant = (
            validate_response_plan_contract_variant(
                response_plan_contract_variant,
            )
        )
    except ValueError as exc:
        raise CanonicalContractError(str(exc)) from exc
    if self_cognition:
        if set(raw) != {
            "self_cognition_response",
            "epistemic_boundary",
        } or not isinstance(raw["self_cognition_response"], dict):
            raise CanonicalContractError("self-cognition plan fields are not exact")
        item = raw["self_cognition_response"]
        if set(item) != {"decision", "response_goal", "reason", "cause_summary"}:
            raise CanonicalContractError("self-cognition response fields are not exact")
        if item["decision"] not in SELF_COGNITION_RESPONSE_DECISION_VALUES:
            raise CanonicalContractError("self-cognition decision is unsupported")
        try:
            self_response_goal = validate_terminal_text_seed(
                _bounded_text(
                    item["response_goal"],
                    "self response goal",
                ),
                "self response goal",
            )
        except CognitionContractError as exc:
            raise CanonicalContractError(str(exc)) from exc
        return CanonicalResponsePlan(
            goal_resolution="answerable_now",
            response_goal=self_response_goal,
            action_requests=(), resolver_requests=(),
            epistemic_boundary=_bounded_text(
                raw["epistemic_boundary"],
                "epistemic boundary",
                _EPISTEMIC_BOUNDARY_MAX_CHARS,
            ),
            self_cognition_response={
                "decision": item["decision"],
                "response_goal": self_response_goal,
                "reason": _bounded_text(item["reason"], "self response reason"),
                "cause_summary": _bounded_text(item["cause_summary"], "self response cause"),
            },
        )
    required = {
        "goal_resolution",
        "response_goal",
        "action_requests",
        "resolver_requests",
        "epistemic_boundary",
    }
    has_dsh_context = isinstance(dsh_interaction_context, Mapping)
    has_pending_resolver = isinstance(pending_resolver_continuation, Mapping)
    if (
        response_plan_contract_variant == "open_pending_resolution"
        and not has_pending_resolver
    ):
        raise CanonicalContractError(
            "open pending response plan variant requires continuation"
        )
    if (
        response_plan_contract_variant != "open_pending_resolution"
        and has_pending_resolver
    ):
        raise CanonicalContractError(
            "only open pending response plan variant accepts continuation"
        )
    if response_plan_contract_variant == "open_pending_resolution":
        required.add("pending_resolution")
    if has_dsh_context:
        required.add("dsh_interaction_decision")
    missing_fields = sorted(required - set(raw))
    optional_fields = (
        {"pending_task_continuation"}
        if response_plan_contract_variant == "fresh_ordinary"
        else set()
    )
    unexpected_fields = sorted(set(raw) - (required | optional_fields))
    details: list[str] = []
    if missing_fields:
        details.append(f"missing fields {missing_fields}")
    if unexpected_fields:
        details.append(f"unexpected fields {unexpected_fields}")
    if details:
        raise CanonicalContractError(f"response plan: {'; '.join(details)}")
    if raw["goal_resolution"] not in GOAL_RESOLUTION_VALUES:
        raise CanonicalContractError("response plan goal_resolution is unsupported")
    action_roster = {
        row.get("action_kind") for row in capabilities.get("actions", []) if isinstance(row, dict)
    }
    resolver_roster = {
        row.get("capability") for row in capabilities.get("resolvers", []) if isinstance(row, dict)
    }
    actions = raw["action_requests"]
    resolvers = raw["resolver_requests"]
    if not isinstance(actions, list) or not isinstance(resolvers, list):
        raise CanonicalContractError("response capability requests must be arrays")
    clean_actions: list[dict[str, object]] = []
    action_affordances = {
        row["action_kind"]: row
        for row in capabilities.get("actions", [])
        if isinstance(row, dict) and isinstance(row.get("action_kind"), str)
    }
    resolver_affordances = {
        row["capability"]: row
        for row in capabilities.get("resolvers", [])
        if isinstance(row, dict) and isinstance(row.get("capability"), str)
    }
    if len(action_affordances) != len(capabilities.get("actions", [])):
        raise CanonicalContractError("action capabilities are duplicated")
    if len(resolver_affordances) != len(capabilities.get("resolvers", [])):
        raise CanonicalContractError("resolver capabilities are duplicated")
    seen_action_kinds: set[str] = set()
    for row in actions:
        if not isinstance(row, dict) or set(row) != {"action_kind", "decision", "detail", "reason"}:
            raise CanonicalContractError("action request fields are not exact")
        action_kind = row["action_kind"]
        if not isinstance(action_kind, str) or action_kind not in action_roster:
            raise CanonicalContractError("action capability is not available")
        if action_kind in seen_action_kinds:
            raise CanonicalContractError("action capability is duplicated")
        seen_action_kinds.add(action_kind)
        affordance = action_affordances[action_kind]
        decision = _bounded_text(row["decision"], "action decision")
        decision_mode = affordance.get("decision_mode")
        allowed_decisions = affordance.get("allowed_decisions", [])
        if decision_mode == "closed" and decision not in allowed_decisions:
            raise CanonicalContractError("action decision is outside its closed affordance")
        if decision_mode == "required_text":
            pattern = affordance.get("decision_pattern", "")
            if not isinstance(pattern, str) or not re.fullmatch(pattern, decision):
                raise CanonicalContractError("action decision does not match its affordance")
        clean_actions.append({key: _bounded_text(row[key], f"action {key}") for key in row})
    max_actions = 2 if str(raw["response_goal"]).strip() else 3
    if len(clean_actions) > max_actions:
        raise CanonicalContractError("action request capacity is exceeded")
    clean_resolvers: list[dict[str, object]] = []
    seen_resolver_capabilities: set[str] = set()
    for index, row in enumerate(resolvers):
        if not isinstance(row, dict):
            raise CanonicalContractError(
                f"resolver_requests[{index}]: must be an object"
            )
        capability = row.get("capability")
        required_fields = {"capability", "goal", "reason"}
        if capability == "task_resolution_request":
            required_fields.add("start_in_background")
        _require_exact_fields(
            row,
            expected=required_fields,
            path=f"resolver_requests[{index}]",
        )
        if not isinstance(capability, str) or capability not in resolver_roster:
            raise CanonicalContractError("resolver capability is not available")
        if capability in seen_resolver_capabilities:
            raise CanonicalContractError("resolver capability is duplicated")
        seen_resolver_capabilities.add(capability)
        if capability not in resolver_affordances:
            raise CanonicalContractError("resolver capability affordance is missing")
        clean_row = {
            key: _bounded_text(row[key], f"resolver {key}")
            for key in ("capability", "goal", "reason")
        }
        if capability == "task_resolution_request":
            if not isinstance(row["start_in_background"], bool):
                raise CanonicalContractError(
                    "task resolution start_in_background must be boolean"
                )
            clean_row["start_in_background"] = row["start_in_background"]
        clean_resolvers.append(clean_row)
    has_task_resolution = any(
        row["capability"] == "task_resolution_request"
        for row in clean_resolvers
    )
    if (
        has_task_resolution
        and raw["goal_resolution"] != "requires_required_evidence"
    ):
        raise CanonicalContractError(
            "task_resolution_request requires goal_resolution="
            "requires_required_evidence"
        )
    has_human_clarification = any(
        row["capability"] == "human_clarification"
        for row in clean_resolvers
    )
    if has_human_clarification and has_task_resolution:
        raise CanonicalContractError(
            "human clarification cannot co-occur with task admission"
        )
    if (
        response_plan_contract_variant == "post_pending_resolution"
        and has_task_resolution
    ):
        raise CanonicalContractError(
            "post pending continuation cannot create task resolution"
        )
    if (
        response_plan_contract_variant == "tool_result_delivery"
        and has_task_resolution
    ):
        raise CanonicalContractError(
            "tool result delivery cannot create task resolution"
        )
    if (
        response_plan_contract_variant != "fresh_ordinary"
        and has_human_clarification
    ):
        raise CanonicalContractError(
            "pending continuation cannot create human clarification"
        )
    pending_task_continuation = None
    if has_human_clarification:
        if "pending_task_continuation" not in raw:
            raise CanonicalContractError(
                "pending_task_continuation is required for human clarification"
            )
        try:
            pending_task_continuation = validate_pending_task_continuation(
                raw["pending_task_continuation"],
            )
        except ResolverValidationError as exc:
            raise CanonicalContractError(str(exc)) from exc
    elif raw.get("pending_task_continuation") is not None:
        raise CanonicalContractError(
            "pending_task_continuation is limited to human clarification"
        )
    pending_resolution = None
    if response_plan_contract_variant == "open_pending_resolution":
        try:
            pending_resolution = validate_resolver_pending_disposition(
                raw["pending_resolution"],
            )
        except ResolverValidationError as exc:
            raise CanonicalContractError(str(exc)) from exc
        if (
            pending_resolution["decision"] != "answered"
            and has_task_resolution
        ):
            raise CanonicalContractError(
                "pending disposition must be answered before task admission"
            )
        task_request = next(
            (row for row in clean_resolvers if row["capability"] == "task_resolution_request"),
            None,
        )
        if task_request is not None:
            try:
                stored_continuation = validate_pending_task_continuation(
                    pending_resolver_continuation[
                        "pending_task_continuation"
                    ],
                )
            except (KeyError, ResolverValidationError) as exc:
                raise CanonicalContractError(
                "pending continuation admission is invalid"
            ) from exc
            admission_value = stored_continuation[
                "on_answered_clarification"
            ]
            if admission_value == "no_task_admission":
                raise CanonicalContractError(
                    "pending continuation forbids task admission"
                )
            expected_background = (
                admission_value == "background_task_admission"
            )
            if task_request["start_in_background"] != expected_background:
                raise CanonicalContractError(
                    "task resolution start_in_background mismatches pending continuation"
                )
    try:
        response_goal = validate_terminal_text_seed(
            _bounded_text(raw["response_goal"], "response goal"),
            "response goal",
        )
    except CognitionContractError as exc:
        raise CanonicalContractError(str(exc)) from exc
    dsh_decision = None
    if has_dsh_context:
        dsh_decision = _validate_dsh_interaction_decision(
            raw["dsh_interaction_decision"],
            context=dsh_interaction_context,
        )
    return CanonicalResponsePlan(
        goal_resolution=raw["goal_resolution"],
        response_goal=response_goal,
        action_requests=tuple(clean_actions), resolver_requests=tuple(clean_resolvers),
        epistemic_boundary=_bounded_text(
            raw["epistemic_boundary"],
            "epistemic boundary",
            _EPISTEMIC_BOUNDARY_MAX_CHARS,
        ),
        pending_resolution=pending_resolution,
        pending_task_continuation=pending_task_continuation,
        dsh_interaction_decision=dsh_decision,
    )


def _validate_dsh_interaction_decision(
    raw: object,
    *,
    context: Mapping[str, object],
) -> dict[str, object]:
    """Validate a Brain-owned P-stage interaction decision without rewriting it."""

    if not isinstance(raw, dict):
        raise CanonicalContractError("DSH interaction decision must be an object")
    expected = {
        "interaction_id", "kind", "decision", "answer", "reason",
    }
    if set(raw) - expected or expected - set(raw):
        raise CanonicalContractError("DSH interaction decision fields are not exact")
    interaction_id = raw["interaction_id"]
    kind = raw["kind"]
    decision = raw["decision"]
    if not isinstance(interaction_id, str) or not interaction_id.strip():
        raise CanonicalContractError("DSH interaction decision identity is invalid")
    if interaction_id != context.get("interaction_id"):
        raise CanonicalContractError("DSH interaction decision identity mismatches context")
    if kind != context.get("kind"):
        raise CanonicalContractError("DSH interaction decision kind mismatches context")
    if kind not in {"approval", "question", "plan_review"}:
        raise CanonicalContractError("DSH interaction decision kind is invalid")
    allowed_dsh_decisions = {
        "approval": {"allow_once", "reject"},
        "question": {"answer", "reject"},
        "plan_review": {"answer", "allow_once", "reject"},
    }[kind]
    if decision not in allowed_dsh_decisions:
        raise CanonicalContractError("DSH interaction decision is invalid")
    if decision == "answer" and kind not in {"question", "plan_review"}:
        raise CanonicalContractError("DSH answer decision is kind-incompatible")
    if decision == "allow_once" and kind not in {"approval", "plan_review"}:
        raise CanonicalContractError("DSH allow decision is kind-incompatible")
    answer = raw["answer"]
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise CanonicalContractError("DSH interaction answer is invalid")
    if isinstance(answer, str) and len(answer) > 2_000:
        raise CanonicalContractError("DSH interaction answer is too long")
    if decision == "answer" and answer is None:
        raise CanonicalContractError("DSH answer is required for answer decision")
    if decision != "answer" and answer is not None:
        raise CanonicalContractError("DSH answer is status-specific")
    return {
        "interaction_id": interaction_id,
        "kind": kind,
        "decision": decision,
        "answer": answer,
        "reason": _bounded_text(raw["reason"], "DSH interaction reason"),
    }


def _validate_appraisal_stage(
    raw: object,
    *,
    families: tuple[str, ...],
) -> _CognitionValidationResult:
    """Validate appraisal content and identify key-order normalization."""

    validated = validate_canonical_appraisal(raw, families=families)
    normalized = isinstance(raw, Mapping) and tuple(raw) != families
    return _CognitionValidationResult(
        value=validated,
        normalization_kind=(
            "appraisal_family_key_order" if normalized else ""
        ),
    )


def _validate_goal_stage(raw: object) -> _CognitionValidationResult:
    """Validate the goal product and identify private-monologue clamping."""

    validated = _validate_goal(raw)
    normalized = (
        isinstance(raw, Mapping)
        and isinstance(raw.get("private_monologue"), str)
        and len(raw["private_monologue"].strip()) > _PRIVATE_MONOLOGUE_MAX_CHARS
    )
    return _CognitionValidationResult(
        value=validated,
        normalization_kind=("private_monologue_clamped" if normalized else ""),
    )


def _validate_plan_stage(
    raw: object,
    *,
    self_cognition: bool,
    capabilities: dict[str, object],
    dsh_interaction_context: Mapping[str, object] | None = None,
    pending_resolver_continuation: Mapping[str, object] | None = None,
    response_plan_contract_variant: ResponsePlanContractVariant,
) -> _CognitionValidationResult:
    """Validate the response plan without adding semantic recovery."""

    return _CognitionValidationResult(
        value=_validate_plan(
            raw,
            self_cognition=self_cognition,
            capabilities=capabilities,
            dsh_interaction_context=dsh_interaction_context,
            pending_resolver_continuation=pending_resolver_continuation,
            response_plan_contract_variant=response_plan_contract_variant,
        ),
    )


async def run_cognition(
    input_payload: Mapping[str, object], services: CognitionChainServicesV3
) -> dict[str, object]:
    """Run the complete canonical chain under its configured deadline."""

    try:
        return await asyncio.wait_for(
            _run_cognition(input_payload, services),
            timeout=services.turn_deadline_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise CognitionExecutionError(
            "cognition turn deadline exhausted",
            error_code="cognition_turn_deadline_exhausted",
            stage="cognition_core_v3",
            safe_checkpoint="pre_state_commit",
            retryable=False,
        ) from exc


async def _run_cognition(
    input_payload: Mapping[str, object], services: CognitionChainServicesV3
) -> dict[str, object]:
    """Run A1, A2, G, and caller-selected P with bounded stage recovery."""

    deadline_monotonic = time.monotonic() + services.turn_deadline_seconds
    validated = _validate_canonical_input(input_payload)
    original_state, transaction_state, transaction_transitions = (
        _prepare_state_transaction(validated)
    )
    validated["mutable_state"] = transaction_state
    validated["_transaction_transition_contexts"] = transaction_transitions
    workspace = build_canonical_turn_workspace(
        episode=validated["episode"], scene_context=validated["scene_context"],
        evidence=validated["evidence"], mutable_state=validated["mutable_state"],
        character_constraints=validated["character_constraints"],
        identity_context=validated["character_identity_context"],
        continuity={
            "private": validated.get("private_continuity_context", ""),
            "dialog": validated.get("past_dialog_cognition_context", ""),
        },
        available_actions=validated["available_actions"],
        available_resolvers=validated["available_resolver_capabilities"],
        overused_moves=validated["overused_moves"],
        direct_facts=validated.get("direct_facts", []),
        character_operational_context=validated.get(
            "character_operational_context", {}
        ),
        character_affect_context=validated.get(
            "character_affect_context", []
        ),
        relationship_context=validated.get("relationship_context", {}),
        resolver_context=validated.get("resolver_context", ""),
        resolver_progress=validated.get("resolver_goal_progress", {}),
        runtime_limits=validated.get("runtime_capability_limits", []),
        group_engagement=validated.get("group_engagement_action_context", {}),
        pending_dsh_interaction=validated.get("pending_dsh_interaction"),
        pending_resolver_continuation=validated.get(
            "pending_resolver_continuation"
        ),
        response_plan_contract_variant=validated[
            "response_plan_contract_variant"
        ],
    )
    a1 = await _run_cognition_stage(
        services=services,
        stage="A1",
        packet=build_canonical_appraisal_question(
            workspace=workspace,
            stage_name="A1",
        ),
        validator=lambda raw: _validate_appraisal_stage(
            raw,
            families=CANONICAL_A1_FAMILIES,
        ),
        deadline_monotonic=deadline_monotonic,
    )
    if not isinstance(a1, tuple):
        raise CognitionContractError("A1 appraisal result is invalid")
    a1_summary = _appraisal_summary(a1)
    a2 = await _run_cognition_stage(
        services=services,
        stage="A2",
        packet=build_canonical_appraisal_question(
            workspace=workspace,
            stage_name="A2",
            accepted_appraisal_summary=a1_summary,
        ),
        validator=lambda raw: _validate_appraisal_stage(
            raw,
            families=CANONICAL_A2_FAMILIES,
        ),
        deadline_monotonic=deadline_monotonic,
    )
    if not isinstance(a2, tuple):
        raise CognitionContractError("A2 appraisal result is invalid")
    appraisals = (*a1, *a2)
    summaries = _appraisal_summary(appraisals)
    goal_result = await _run_cognition_stage(
        services=services,
        stage="G",
        packet=build_canonical_goal_question(
            workspace=workspace,
            appraisal_summary=summaries,
        ),
        validator=_validate_goal_stage,
        deadline_monotonic=deadline_monotonic,
    )
    if not (
        isinstance(goal_result, tuple)
        and len(goal_result) == 3
    ):
        raise CognitionContractError("G goal result is invalid")
    goal, willingness, private_monologue = goal_result
    self_cognition = _is_self_cognition(validated)
    plan_result = await _run_cognition_stage(
        services=services,
        stage="P",
        packet=build_canonical_plan_question(
            workspace=workspace,
            goal=goal.__dict__,
            appraisal_summary=summaries,
            self_cognition=self_cognition,
        ),
        validator=lambda raw: _validate_plan_stage(
            raw,
            self_cognition=self_cognition,
            capabilities=workspace["capabilities"],
            dsh_interaction_context=workspace.get("pending_dsh_interaction"),
            pending_resolver_continuation=workspace.get(
                "pending_resolver_continuation"
            ),
            response_plan_contract_variant=workspace[
                "response_plan_contract_variant"
            ],
        ),
        deadline_monotonic=deadline_monotonic,
    )
    if not isinstance(plan_result, CanonicalResponsePlan):
        raise CognitionContractError("P response plan result is invalid")
    plan = plan_result
    binding_metadata: dict[str, object] = {}
    replacement_state, transition_contexts, binding_receipts, cause_provenance = bind_axis_changes(
        validated,
        appraisals,
        goal=goal.__dict__,
        willingness=willingness,
        goal_resolution=plan.goal_resolution,
        action_requests=plan.action_requests,
        resolver_requests=plan.resolver_requests,
        binding_metadata=binding_metadata,
    )
    if validated["state_scope"] == "user":
        episode = validated["episode"]
        if not isinstance(episode, Mapping):
            raise CanonicalContractError("canonical episode is invalid")
        relationship_deltas: list[dict[str, object]] = []
        for receipt in binding_receipts:
            if receipt.get("family") != "relationship_social":
                continue
            applied_targets = receipt.get("applied_targets", [])
            if not isinstance(applied_targets, list):
                continue
            for applied in applied_targets:
                if not isinstance(applied, Mapping):
                    continue
                target_path = applied.get("target_path")
                applied_delta = applied.get("applied_delta")
                if (
                    isinstance(target_path, str)
                    and target_path.startswith("relationship.")
                    and isinstance(applied_delta, int)
                    and not isinstance(applied_delta, bool)
                ):
                    relationship_deltas.append({
                        "duplicate_disposition": "unique",
                        "target_path": target_path,
                        "relationship_axis": receipt.get("axis"),
                        "applied_delta": applied_delta,
                    })
        replacement_state = apply_relationship_maintenance(
            replacement_state,
            source_episode_id=str(episode["episode_id"]),
            interaction_date_utc=str(validated["_transaction_interaction_date"]),
            elapsed_seconds=int(validated["_transaction_elapsed_seconds"]),
            accepted_relationship_deltas=relationship_deltas,
            trusted_facts=tuple(
                row for row in validated.get("_transaction_direct_facts", [])
                if isinstance(row, Mapping)
            ),
        )
        replacement_state = validate_cognition_state(replacement_state)
    derived_activations = derive_persistent_emotion_activations(
        replacement_state,
        updated_at=str(replacement_state.get("updated_at", "")),
        character_constraints=validated.get("character_constraints"),
        relationship_context=validated.get("relationship_context"),
        transition_contexts=transition_contexts,
    )
    replacement_state["affect_activations"] = derived_activations
    replacement_state = validate_cognition_state(replacement_state)
    affect_projection = project_affect(derived_activations, replacement_state)
    result = CanonicalCognitionOutput(
        schema_version=CANONICAL_COGNITION_OUTPUT_SCHEMA,
        appraisals=tuple(appraisals), active_character_goal=goal,
        relational_willingness=willingness,
        private_monologue=private_monologue,
        response_plan=plan,
        affect_projection=tuple(affect_projection),
        relationship_projection=_canonical_relationship_projection({
            "relationship_context": {"axes": replacement_state.get("relationship", {})}
        }),
        cause_provenance=tuple(cause_provenance),
        diagnostics={"status": "complete"},
    )
    output = result.as_dict()
    # State replacement is caller-owned.  The model never sees this carrier;
    # immediate adapters use it for the existing compare-and-replace boundary.
    output["state_projection"] = {
        "state_scope": validated["state_scope"],
        "owner_key": validated["mutable_state"].get("owner_user_id", "")
        if isinstance(validated["mutable_state"], Mapping)
        else "",
        "expected_previous_state": original_state,
        "original_persisted_state": original_state,
        "replacement_state": replacement_state,
        "transition_contexts": transition_contexts,
        "binding_receipts": binding_receipts,
        "capacity_deferred": [
            dict(row)
            for row in binding_receipts
            if row.get("disposition") == "capacity_deferred"
        ],
    }
    if "continuation_goal_ref" in binding_metadata:
        output["state_projection"]["continuation_goal_ref"] = dict(
            binding_metadata["continuation_goal_ref"]
        )
    return dict(validate_canonical_cognition_output(output))


def _is_self_cognition(payload: Mapping[str, object]) -> bool:
    episode = payload.get("episode")
    scene = payload.get("scene_context")
    return bool(
        is_targetless_group_self_cognition_episode(payload)
        or (
        isinstance(episode, Mapping)
        and isinstance(scene, Mapping)
        and scene.get("operation") == "要求当前角色回答自己此时此刻的心理期待内容"
        )
    )


def _canonical_relationship_projection(payload: Mapping[str, object]) -> dict[str, object]:
    relationship = payload.get("relationship_context")
    if not isinstance(relationship, Mapping):
        return {"summary": "no relationship-specific context"}
    raw_axes = relationship.get("axes")
    axes = {
        name: value
        for name, value in raw_axes.items()
        if name in RELATIONSHIP_AXIS_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    } if isinstance(raw_axes, Mapping) else {}
    return {
        "summary": "current relationship context remains caller-owned",
        "axes": axes,
    }
