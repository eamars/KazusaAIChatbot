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
    validate_canonical_cognition_output,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,  # noqa: F401
    record_protected_chain_record,
    reset_protected_chain_records,  # noqa: F401
    snapshot_protected_chain_records,  # noqa: F401
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    A2_EXISTENTIAL_DRIVE_EVIDENCE_GUIDANCE,
    A2_RELATIONSHIP_STATE_EVIDENCE_GUIDANCE,
    BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE,
    CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
    G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE,
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
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


_STAGE_SYSTEM_PROMPTS = {
    "A1": '''# 任务
使用三个面向客观情境的评估类别，判断当前观察对当前角色意味着什么。

# 证据边界
{current_observation_authority}
`continuation_state` 只提供仍在起作用的因果压力。输入不会提供角色身份、
习惯、反思倾向或 `participant_continuity`，因为这些信息不能证明当前世界中的
事实。当前用户明确纠正自己的意思或感受时，把这项纠正当作当前观察；纠正本身
不是相反意思的证据。'''.format(
        current_observation_authority=CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
    ),
    "A2": '''# 任务
根据已接受的 A1 语义，使用关系与社会判断、道德身份、存在性驱力三个评估类别
作出判断。

# 证据边界
{current_observation_authority}
`participant_continuity` 只描述此前参与者、行为及结果，不能证明新的行为、
同意、承诺、许可或当前意图。`conditional_character_context` 只可影响当前角色的
判断、边界、动机和情绪解释。`overused_moves` 只描述此前已经发生的可见回应模式，
不表示当前事实、用户意图、禁止事项或下一步行动。
{existential_drive_evidence}
{relationship_state_evidence}'''.format(
        current_observation_authority=CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
        existential_drive_evidence=A2_EXISTENTIAL_DRIVE_EVIDENCE_GUIDANCE,
        relationship_state_evidence=A2_RELATIONSHIP_STATE_EVIDENCE_GUIDANCE,
    ),
    "G": '''# 任务
选择一个有意义的当前角色目标、一项关系互动意愿记录，以及一段简洁的第一人称
`private_monologue`。

# 内心独白边界
`private_monologue` 要连接角色此刻的感受、造成这种感受的具体原因，以及角色眼下
想保护、揭示、回避或追求的事。它不能用于确认事实、许可、能力、目标对象或状态
变化。先根据当前观察的新增加、改变、纠正、询问或未解决内容选择目标；已表达的回应
模式只能作为背景，除非当前用户继续、深化、实质改变或重新打开同一事项。
{current_observation_authority}
{background_goal_authority}
{relational_carrier_evidence}'''.format(
        current_observation_authority=CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
        background_goal_authority=BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE,
        relational_carrier_evidence=G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE,
    ),
    "P": '''# 任务
确定当前角色的回应目标和 `epistemic_boundary`，并且只选择输入中提供的能力。

# 断言边界
{current_observation_authority}
`participant_continuity` 只支持对先前参与者、行为及结果的描述。
`epistemic_boundary` 必须说明可见措辞可以断言什么、哪些内容只能作为解释，以及
哪些内容仍然未知。每一项未被观察到的功能、原因、来源、意图或结果，都只能作为
解释，并且在可见措辞中必须明确表达不确定性。没有观察到某项特征或缺少证据，都
不能据此作出否定断言，也不能排除任何可能。

先让回应目标回答当前观察新增加、改变、纠正、询问或仍未解决的内容，再让性格和关系
语境影响表达。当前用户明确纠正自己的意思或感受时，把这项纠正当作当前观察；纠正
本身不是相反意思的证据。已表达的回应模式只能作为背景连续性，不能成为未重新选择的
主要关系回报。
{background_goal_authority}

# 输出边界
`response_goal` 的值必须是一个非空的简体中文字符串，用一句话概括下游可见对话的目标。
只返回 `output_contract` 指定的字段。不得把任何输入权威通道名称写入输出对象。'''.format(
        current_observation_authority=CURRENT_OBSERVATION_AUTHORITY_GUIDANCE,
        background_goal_authority=BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE,
    ),
}

_EXACT_JSON_SYSTEM_SUFFIX = '''# 输出约束
严格按照 `output_contract` 返回一个 JSON 对象，并且只输出该 JSON 对象。所有可
自由填写的语义文本必须使用简体中文。契约字段名、枚举值、能力名和动作名必须原样
保留；引用原文、代码和 URL 也保持原样。不得输出内部标识符、句柄、路径或证据
引用。'''
_PRIVATE_MONOLOGUE_MAX_CHARS = 600
_EPISTEMIC_BOUNDARY_MAX_CHARS = 1000
_COGNITION_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
_COGNITION_ATTEMPT_TIME_FLOOR_SECONDS = 20
_COGNITION_STAGE_ERROR_CAP = 500
_COGNITION_STAGE_REPAIR_OUTPUT_CAP = 8000
_COGNITION_REPAIR_INSTRUCTION = (
    '保留原始的角色判断、情绪方向、关系方向、能力结果和事实；只修复字段集合、字段类型、长度、'
    '列表基数。只返回当前阶段规定的完整对象，不添加解释、markdown 或额外字段。'
)

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


def _validate_canonical_input(value: object) -> dict[str, object]:
    """Validate the single caller-owned input envelope without schema branching."""

    if not isinstance(value, Mapping):
        raise CanonicalContractError("canonical cognition input must be an object")
    required = {
        "episode", "scene_context", "evidence", "mutable_state",
        "state_scope", "character_constraints", "character_identity_context",
        "available_actions", "available_resolver_capabilities",
        "overused_moves",
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
        output_mode="text",
    )
    system_message = SystemMessage(content="\n\n".join((
        _STAGE_SYSTEM_PROMPTS[stage],
        _EXACT_JSON_SYSTEM_SUFFIX,
    )))
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
                reason=(
                    "上一轮模型调用未返回可用候选，请在相同语境下重新生成完整 JSON。"
                ),
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
                reason=(
                    "上一份候选未通过当前阶段的字段、类型、长度或列表基数校验。"
                ),
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
        "repair_instruction": _COGNITION_REPAIR_INSTRUCTION,
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
    dsh_reply: bool = False,
) -> CanonicalResponsePlan:
    if not isinstance(raw, dict):
        raise CanonicalContractError("response plan must be an object")
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
    if has_dsh_context:
        required.add("dsh_interaction_decision")
    if set(raw) != required or raw["goal_resolution"] not in GOAL_RESOLUTION_VALUES:
        raise CanonicalContractError("ordinary response plan fields are not exact")
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
    for row in resolvers:
        if not isinstance(row, dict) or set(row) != {"capability", "goal", "reason"}:
            raise CanonicalContractError("resolver request fields are not exact")
        capability = row["capability"]
        if not isinstance(capability, str) or capability not in resolver_roster:
            raise CanonicalContractError("resolver capability is not available")
        if capability in seen_resolver_capabilities:
            raise CanonicalContractError("resolver capability is duplicated")
        seen_resolver_capabilities.add(capability)
        if capability not in resolver_affordances:
            raise CanonicalContractError("resolver capability affordance is missing")
        clean_resolvers.append({key: _bounded_text(row[key], f"resolver {key}") for key in row})
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
            dsh_reply=dsh_reply,
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
        dsh_interaction_decision=dsh_decision,
    )


def _validate_dsh_interaction_decision(
    raw: object,
    *,
    context: Mapping[str, object],
    dsh_reply: bool = False,
) -> dict[str, object]:
    """Validate a Brain-owned P-stage interaction decision without rewriting it."""

    if not isinstance(raw, dict):
        raise CanonicalContractError("DSH interaction decision must be an object")
    expected = {
        "interaction_id", "kind", "decision", "answer", "response_goal",
        "relay_mode", "reason",
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
        "answer", "allow_once", "reject", "relay_to_user",
    }
    if dsh_reply:
        allowed_dsh_decisions.add("continue_waiting")
    if decision not in allowed_dsh_decisions:
        raise CanonicalContractError("DSH interaction decision is invalid")
    if decision == "answer" and kind not in {"question", "plan_review"}:
        raise CanonicalContractError("DSH answer decision is kind-incompatible")
    if decision == "allow_once" and kind not in {"approval", "plan_review"}:
        raise CanonicalContractError("DSH allow decision is kind-incompatible")
    answer = raw["answer"]
    response_goal = raw["response_goal"]
    relay_mode = raw["relay_mode"]
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise CanonicalContractError("DSH interaction answer is invalid")
    if isinstance(answer, str) and len(answer) > 2_000:
        raise CanonicalContractError("DSH interaction answer is too long")
    if response_goal is not None and (
        not isinstance(response_goal, str) or not response_goal.strip()
    ):
        raise CanonicalContractError("DSH response goal is invalid")
    if isinstance(response_goal, str) and len(response_goal) > 2_000:
        raise CanonicalContractError("DSH response goal is too long")
    if relay_mode is not None and relay_mode not in {
        "question", "approval", "plan_review",
    }:
        raise CanonicalContractError("DSH relay mode is invalid")
    if decision == "answer" and answer is None:
        raise CanonicalContractError("DSH answer is required for answer decision")
    if decision != "answer" and answer is not None:
        raise CanonicalContractError("DSH answer is status-specific")
    if decision == "relay_to_user" and (
        response_goal is None or relay_mode is None
    ):
        raise CanonicalContractError(
            "DSH response goal and relay mode are required for relay decision"
        )
    if decision != "relay_to_user" and (
        response_goal is not None or relay_mode is not None
    ):
        raise CanonicalContractError("DSH relay fields are status-specific")
    if decision == "continue_waiting" and not dsh_reply:
        raise CanonicalContractError(
            "continue_waiting is available only for a matched user reply"
        )
    return {
        "interaction_id": interaction_id,
        "kind": kind,
        "decision": decision,
        "answer": answer,
        "response_goal": response_goal,
        "relay_mode": relay_mode,
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
    dsh_reply: bool = False,
) -> _CognitionValidationResult:
    """Validate the response plan without adding semantic recovery."""

    return _CognitionValidationResult(
        value=_validate_plan(
            raw,
            self_cognition=self_cognition,
            capabilities=capabilities,
            dsh_interaction_context=dsh_interaction_context,
            dsh_reply=dsh_reply,
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
            dsh_reply=bool(validated.get("pending_dsh_reply", False)),
        ),
        validator=lambda raw: _validate_plan_stage(
            raw,
            self_cognition=self_cognition,
            capabilities=workspace["capabilities"],
            dsh_interaction_context=workspace.get("pending_dsh_interaction"),
            dsh_reply=bool(validated.get("pending_dsh_reply", False)),
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
