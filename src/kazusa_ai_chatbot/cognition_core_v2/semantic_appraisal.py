"""Scoped semantic appraisal with deterministic structural validation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
    CognitionContextLimitError,
    CognitionEvidenceV2,
    CognitionExecutionError,
    SemanticAppraisalResultV2,
    SemanticQuestionV2,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    PromptBudgetError,
    fit_evidence_texts_to_budget,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output


SEMANTIC_APPRAISAL_ATTEMPT_LIMIT = 2
SEMANTIC_APPRAISAL_ITEM_LIMIT = 8
SEMANTIC_APPRAISAL_PROMPT_CAP = 8000
SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP = 10000
SEMANTIC_APPRAISAL_ITEM_EXPLANATION_LIMIT = 120
MIN_PROMPT_EVIDENCE_TEXT_CHARS = 96
_SEMANTIC_APPRAISAL_RESULT_FIELDS = {
    "question_id",
    "selected_evidence_handles",
    "selected_role_handles",
    "propositions",
    "deltas",
    "explanation",
}
_SEMANTIC_APPRAISAL_ITEM_FIELDS = {
    "question_id",
    "proposition",
    "delta",
}

_PROPOSITION_SUBJECT_KINDS = {
    "goal_release": "goal",
    "goal_supersession": "goal",
    "goal_completed": "goal",
    "event_completed": "event",
    "threat_resolved": "threat",
    "event_repaired": "event",
    "knowledge_answered": "knowledge_gap",
}


SEMANTIC_APPRAISAL_PROMPT = '''你根据有界证据回答一个范围明确的语义问题。
只使用本次 prompt 允许的 handle 和语义描述。动作选择、对话生成、emotion id、生命周期状态与
事实补充不属于本阶段。只有在所给证据支持时，才返回语义命题和允许路径上的数值变化。
每个 proposition_kind 都是其所给语义定义已经成立的肯定式断言。
当前调用只生成一个 micro_appraisal item。proposition 和 delta 各自只能是一个对象或 null，
不能使用数组，也不能列举多个候选。没有尚未输出的必要项目时，两者都返回 null 以结束循环。

遵守每条证据的 source_kind。角色自己的反思或内部观察属于证据，不是当前用户的即时发言。
生成的文字不复述来源包标题、时间戳、传输摘要、schema key 或运行元数据。新生成的自由文本使用
简体中文；引用的用户原文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 question_id、proposition 和 delta。
proposition 与 delta 若不是 null，就必须引用提供的 evidence handle；未知或缺少支持的含义直接
返回 null。不要输出 explanation、selected_evidence_handles、selected_role_handles、propositions
或 deltas。

每个 proposition 对象必须恰好包含 proposition_kind、subject_handle、evidence_handles、
role_assignments 和 semantic_value，并可选包含 object_handle。每条 role assignment 必须恰好
包含 role 和 entity_handle。每个 delta 对象必须恰好包含 target_path、delta、
evidence_handles 和 reason。所给 role handle 与 delta path 按原值使用；不输出 kind、handle、
semantic_text、role_handles、path 或其他 proposition、delta 字段。
question.permitted_delta_path_domains 的每一项给出 state_field、handles 和 axes。
每个 target_path 必须从同一项中各取一个值，按 state_field.handle.axis 组合并原样输出。
delta 必须是 -40 到 40（含边界）的 JSON 整数，例如 -5、0 或 12；不得使用字符串、小数、
百分比或正负号小数比例。

semantic_value 是一句简洁描述，目标长度 120 字符且上限 200 字符，其中不重复标准、约束或证据
解释，也不使用数值；数值只放在 delta 字段。每条 delta reason 不超过 300 字符。role 必须取以下
固定 enum token：actor（行动者）、experiencer（体验者）、
target（对象）、object（客体）、affected_goal（受影响目标）或 affected_relationship（受影响关系）。
r1、ce1、ct1、ck1 等实体 handle 放在 entity_handle，不能放在 role。当前角色和当前用户的内部
角色句柄只用于结构化字段；中文自由文本使用“当前角色”“当前用户”或配置的角色名、用户显示名，
不要把内部角色句柄或英文角色称谓写入中文自由文本。固定 schema key 和 enum token 仍按原值输出。
ceN、ctN、ckN 表示候选事件、威胁或知识缺口，不是人物；人物的 actor、experiencer 或 target 使用
self 或 current_user。无法从允许 handle 准确分配角色时，role_assignments 使用空数组。

handle 域严格对应 question：subject_handle、object_handle 与 role_assignments[*].entity_handle
只能使用 question.permitted_role_handles；proposition、delta 的 evidence_handles 只能使用
question.evidence_handles；target_path
只能使用 question.permitted_delta_path_domains 允许的 state_field.handle.axis 组合。持久 event handle
使用 ev1..evN，evidence handle 使用 e1..eN，候选 event、threat、knowledge gap handle 分别使用
ce1..ceN、ct1..ctN、ck1..ckN。question.candidate_origin_evidence 是允许的 candidate handle 到其
来源 evidence handle 的唯一映射；任何 proposition、delta 的 target_path 或结构化 handle 使用
ceN、ctN 或 ckN 时，该对象的 evidence_handles 必须包含对应的来源 evidence handle。
输出前逐个对象检查：subject_handle、object_handle、role_assignments[*].entity_handle 或
target_path 中每出现一个 ceN、ctN、ckN，就把映射值加入该对象的 evidence_handles；无法加入时省略
该 candidate 或整个对象。
'''


async def appraise_semantic_question(
    question: SemanticQuestionV2,
    evidence: Sequence[CognitionEvidenceV2],
    projection: PromptProjectionV2,
    services: CognitionCoreServicesV2,
    *,
    validation_state: Mapping[str, Any],
) -> SemanticAppraisalResultV2:
    """Run one bounded family appraisal and return no state authority.

    Args:
        question: Scoped semantic question selected for this family.
        evidence: Typed evidence rows available to the question.
        projection: Prompt-safe state plus private handle bindings.
        services: Configured stage-specific model services.
        validation_state: Validated preliminary state used only for trial
            reduction of each candidate.

    Returns:
        The first structurally and reducer-compatible appraisal result.
    """

    config_by_question_kind = {
        "event_agency": services.appraisal_event_agency_config,
        "relationship_social": services.appraisal_relationship_social_config,
        "moral_identity": services.appraisal_moral_identity_config,
        "goal_threat_outcome": services.appraisal_goal_threat_outcome_config,
        "epistemic_comparison_memory": (
            services.appraisal_epistemic_comparison_memory_config
        ),
        "existential_drive": services.appraisal_existential_drive_config,
    }
    config = config_by_question_kind[question["question_kind"]]
    evidence_by_handle = {
        row["evidence_handle"]: {
            "handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "source_kind": row["evidence_ref"]["source_kind"],
        }
        for row in evidence
        if row["evidence_handle"] in question["evidence_handles"]
    }
    allowed_evidence_handles = set(question["evidence_handles"])
    candidate_origin_evidence = {
        candidate_handle: origin_handle
        for candidate_handle in question["permitted_role_handles"]
        if (
            origin_handle := _candidate_evidence_handle(
                candidate_handle,
                projection.handle_to_ref,
            )
        ) and origin_handle in allowed_evidence_handles
    }
    base_payload = {
        "question": {
            "question_id": question["question_id"],
            "question_kind": question["question_kind"],
            "semantic_question": question["semantic_question"],
            "permitted_role_handles": question["permitted_role_handles"],
            "candidate_origin_evidence": candidate_origin_evidence,
            "permitted_delta_path_domains": (
                _compact_permitted_delta_path_domains(
                    question["permitted_delta_paths"]
                )
            ),
            "permitted_proposition_kinds": list(
                question_proposition_kinds(question["question_kind"])
            ),
            "proposition_kind_semantics": (
                question_proposition_kind_semantics(
                    question["question_kind"]
                )
            ),
            "handle_field_domains": {
                "subject_handle": question["permitted_role_handles"],
                "object_handle": question["permitted_role_handles"],
                "entity_handle": question["permitted_role_handles"],
                "evidence_handles": question["evidence_handles"],
            },
            "role_handle_semantics": {
                "self": {
                    "structured_handle": "self",
                    "semantic_text_reference": '当前角色',
                },
                "current_user": {
                    "structured_handle": "current_user",
                    "semantic_text_reference": '当前用户',
                },
            },
        },
        "evidence": list(evidence_by_handle.values()),
        "state": _project_question_state(projection, question),
    }
    system_message = SystemMessage(content=SEMANTIC_APPRAISAL_PROMPT)
    accepted_result: SemanticAppraisalResultV2 | None = None
    for item_index in range(1, SEMANTIC_APPRAISAL_ITEM_LIMIT + 1):
        emitted_paths = {
            delta["target_path"]
            for delta in (
                accepted_result["deltas"] if accepted_result else []
            )
        }
        item_question = deepcopy(dict(question))
        item_question["permitted_delta_paths"] = [
            path
            for path in question["permitted_delta_paths"]
            if path not in emitted_paths
        ]
        payload = deepcopy(base_payload)
        payload["question"]["permitted_delta_path_domains"] = (
            _compact_permitted_delta_path_domains(
                item_question["permitted_delta_paths"]
            )
        )
        payload["question"]["micro_appraisal"] = {
            "item_index": item_index,
            "maximum_items": SEMANTIC_APPRAISAL_ITEM_LIMIT,
            "maximum_propositions": 1,
            "maximum_deltas": 1,
            "empty_lists_end_family": True,
            "emitted_proposition_signatures": (
                _emitted_proposition_signatures(accepted_result)
            ),
            "emitted_delta_paths": sorted(emitted_paths),
        }
        payload_text = _fit_appraisal_payload(payload)
        item_result, merged_result = await _appraise_semantic_item(
            question=question,
            item_question=item_question,
            evidence=evidence,
            evidence_by_handle=evidence_by_handle,
            projection=projection,
            validation_state=validation_state,
            accepted_result=accepted_result,
            services=services,
            config=config,
            system_message=system_message,
            payload_text=payload_text,
            item_index=item_index,
        )
        if not item_result["propositions"] and not item_result["deltas"]:
            final_result = (
                accepted_result
                if accepted_result is not None
                else item_result
            )
            return final_result
        accepted_result = merged_result

    if accepted_result is None:
        raise AssertionError("semantic appraisal item loop produced no result")
    return accepted_result


async def _appraise_semantic_item(
    *,
    question: SemanticQuestionV2,
    item_question: SemanticQuestionV2,
    evidence: Sequence[CognitionEvidenceV2],
    evidence_by_handle: Mapping[str, Mapping[str, str]],
    projection: PromptProjectionV2,
    validation_state: Mapping[str, Any],
    accepted_result: SemanticAppraisalResultV2 | None,
    services: CognitionCoreServicesV2,
    config: LLMCallConfig,
    system_message: SystemMessage,
    payload_text: str,
    item_index: int,
) -> tuple[SemanticAppraisalResultV2, SemanticAppraisalResultV2]:
    """Generate and validate one bounded appraisal item."""

    human_message = HumanMessage(content=payload_text)
    request_messages: list[BaseMessage] = [system_message, human_message]
    for attempt_index in range(SEMANTIC_APPRAISAL_ATTEMPT_LIMIT):
        started_at = time.perf_counter()
        raw_output: str | None = None
        parsed_output: object | None = None
        stage_id = (
            f"semantic_appraisal:{question['question_id']}:item_{item_index}"
        )
        if attempt_index:
            stage_id = f"{stage_id}:repair_{attempt_index}"
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
            ended_at = time.perf_counter()
            _record_semantic_appraisal_trace(
                config=config,
                question=question,
                messages=request_messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                started_at=started_at,
                attempt_index=attempt_index + 1,
                item_index=item_index,
                validation_error=str(exc),
            )
            capture_validation_stage(
                stage_id=stage_id,
                config=config,
                system_prompt=SEMANTIC_APPRAISAL_PROMPT,
                human_payload=payload_text,
                raw_output=None,
                parsed_output=None,
                parse_status="failed",
                started_at=started_at,
                ended_at=ended_at,
                error=str(exc),
            )
            if attempt_index + 1 >= SEMANTIC_APPRAISAL_ATTEMPT_LIMIT:
                raise CognitionExecutionError(
                    "semantic appraisal provider attempts exhausted",
                    error_code="semantic_appraisal_provider_exhausted",
                    stage="semantic_appraisal",
                    attempt_count=attempt_index + 1,
                    safe_checkpoint="pre_state_commit",
                    retryable=False,
                ) from exc
            request_messages = [system_message, human_message]
            continue
        raw_output = getattr(response, "content", "")
        try:
            parsed_output = parse_llm_json_output(
                raw_output,
                repair_trace_hook=(
                    failure_capsule.append_json_repair_attempt
                ),
            )
            parsed_output = _canonicalize_semantic_appraisal_item(parsed_output)
            parsed_output = _suppress_emitted_appraisal_components(
                parsed_output,
                accepted_result,
            )
            result = validate_semantic_appraisal_result(
                parsed_output,
                item_question,
                set(evidence_by_handle),
                projection.handle_to_ref,
                maximum_propositions=1,
                maximum_deltas=1,
                maximum_explanation_chars=(
                    SEMANTIC_APPRAISAL_ITEM_EXPLANATION_LIMIT
                ),
            )
            merged_result = _merge_semantic_appraisal_item(
                accepted_result,
                result,
            )
            validate_semantic_appraisal_result(
                merged_result,
                question,
                set(evidence_by_handle),
                projection.handle_to_ref,
            )
            trial_state = deepcopy(dict(validation_state))
            apply_semantic_appraisals(
                trial_state,
                [merged_result],
                evidence,
                projection.handle_to_ref,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            ended_at = time.perf_counter()
            _record_semantic_appraisal_trace(
                config=config,
                question=question,
                messages=request_messages,
                response_text=str(raw_output),
                parsed_output=parsed_output,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                attempt_index=attempt_index + 1,
                item_index=item_index,
                validation_error=str(exc),
            )
            capture_validation_stage(
                stage_id=stage_id,
                config=config,
                system_prompt=SEMANTIC_APPRAISAL_PROMPT,
                human_payload=payload_text,
                raw_output=raw_output,
                parsed_output=parsed_output,
                parse_status="failed",
                started_at=started_at,
                ended_at=ended_at,
                error=str(exc),
            )
            if attempt_index + 1 >= SEMANTIC_APPRAISAL_ATTEMPT_LIMIT:
                raise CognitionExecutionError(
                    "semantic appraisal contract attempts exhausted",
                    error_code="semantic_appraisal_contract_exhausted",
                    stage="semantic_appraisal",
                    attempt_count=attempt_index + 1,
                    safe_checkpoint="pre_state_commit",
                    retryable=False,
                ) from exc
            request_messages = _appraisal_repair_messages(
                system_message=system_message,
                human_message=human_message,
                invalid_candidate=str(raw_output),
                contract_error=str(exc),
            )
            continue

        ended_at = time.perf_counter()
        _record_semantic_appraisal_trace(
            config=config,
            question=question,
            messages=request_messages,
            response_text=str(raw_output),
            parsed_output=parsed_output,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            attempt_index=attempt_index + 1,
            item_index=item_index,
            validation_error="",
        )
        capture_validation_stage(
            stage_id=stage_id,
            config=config,
            system_prompt=SEMANTIC_APPRAISAL_PROMPT,
            human_payload=payload_text,
            raw_output=raw_output,
            parsed_output=parsed_output,
            parse_status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
        )
        return result, merged_result

    raise AssertionError("semantic appraisal item attempt loop did not terminate")


def _merge_semantic_appraisal_item(
    accepted_result: SemanticAppraisalResultV2 | None,
    item_result: SemanticAppraisalResultV2,
) -> SemanticAppraisalResultV2:
    """Merge one validated item into the bounded family result."""

    if accepted_result is None:
        merged_result = deepcopy(item_result)
        return merged_result
    prior_signatures = set(_emitted_proposition_signatures(accepted_result))
    item_signatures = _emitted_proposition_signatures(item_result)
    if any(signature in prior_signatures for signature in item_signatures):
        raise ValueError("semantic appraisal proposition is duplicated")
    return {
        "question_id": accepted_result["question_id"],
        "selected_evidence_handles": _ordered_handle_union(
            accepted_result["selected_evidence_handles"],
            item_result["selected_evidence_handles"],
        ),
        "selected_role_handles": _ordered_handle_union(
            accepted_result["selected_role_handles"],
            item_result["selected_role_handles"],
        ),
        "propositions": [
            *deepcopy(accepted_result["propositions"]),
            *deepcopy(item_result["propositions"]),
        ],
        "deltas": [
            *deepcopy(accepted_result["deltas"]),
            *deepcopy(item_result["deltas"]),
        ],
        "explanation": (
            f"{accepted_result['explanation']} {item_result['explanation']}"
        ),
    }


def _ordered_handle_union(
    first: Sequence[str],
    second: Sequence[str],
) -> list[str]:
    """Return one stable duplicate-free handle union."""

    return list(dict.fromkeys([*first, *second]))


def _derive_appraisal_selection_metadata(parsed: object) -> object:
    """Derive provenance selections from the model's structured item fields."""

    if (
        not isinstance(parsed, Mapping)
        or set(parsed) != _SEMANTIC_APPRAISAL_RESULT_FIELDS
        or not isinstance(parsed["propositions"], list)
        or not isinstance(parsed["deltas"], list)
    ):
        return parsed

    selected_evidence: list[str] = []
    selected_roles: list[str] = []
    for proposition in parsed["propositions"]:
        if not isinstance(proposition, Mapping):
            continue
        evidence = proposition.get("evidence_handles")
        if isinstance(evidence, list):
            selected_evidence.extend(
                handle for handle in evidence if isinstance(handle, str)
            )
        for field in ("subject_handle", "object_handle"):
            handle = proposition.get(field)
            if isinstance(handle, str):
                selected_roles.append(handle)
        assignments = proposition.get("role_assignments")
        if isinstance(assignments, list):
            for assignment in assignments:
                if not isinstance(assignment, Mapping):
                    continue
                handle = assignment.get("entity_handle")
                if isinstance(handle, str):
                    selected_roles.append(handle)

    for delta in parsed["deltas"]:
        if not isinstance(delta, Mapping):
            continue
        evidence = delta.get("evidence_handles")
        if isinstance(evidence, list):
            selected_evidence.extend(
                handle for handle in evidence if isinstance(handle, str)
            )
        path = delta.get("target_path")
        if isinstance(path, str) and len(path.split(".")) == 3:
            selected_roles.append(path.split(".")[1])

    normalized = deepcopy(dict(parsed))
    normalized["selected_evidence_handles"] = _ordered_handle_union(
        [],
        selected_evidence,
    )
    normalized["selected_role_handles"] = _ordered_handle_union(
        [],
        selected_roles,
    )
    return normalized


def _canonicalize_semantic_appraisal_item(parsed: object) -> object:
    """Convert one singular model item into the public aggregate shape."""

    if not isinstance(parsed, Mapping):
        raise ValueError("semantic micro-appraisal must return an object")
    if set(parsed) != _SEMANTIC_APPRAISAL_ITEM_FIELDS:
        raise ValueError(
            "semantic micro-appraisal fields must be exactly question_id, "
            "proposition, and delta"
        )
    proposition = parsed["proposition"]
    delta = parsed["delta"]
    canonical = {
        "question_id": parsed["question_id"],
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [] if proposition is None else [proposition],
        "deltas": [] if delta is None else [delta],
        "explanation": _derive_semantic_item_explanation(
            proposition,
            delta,
        ),
    }
    canonical_result = _derive_appraisal_selection_metadata(canonical)
    return canonical_result


def _derive_semantic_item_explanation(
    proposition: object,
    delta: object,
) -> str:
    """Derive bounded audit text from the item's authored semantic fields."""

    parts: list[str] = []
    if isinstance(proposition, Mapping):
        semantic_value = proposition.get("semantic_value")
        if isinstance(semantic_value, str) and semantic_value:
            parts.append(semantic_value)
    if isinstance(delta, Mapping):
        reason = delta.get("reason")
        if isinstance(reason, str) and reason:
            parts.append(reason)
    if not parts:
        if proposition is None and delta is None:
            return "No additional supported semantic item."
        return "Structured semantic item."
    explanation = " ".join(dict.fromkeys(parts))
    return explanation[:SEMANTIC_APPRAISAL_ITEM_EXPLANATION_LIMIT]


def _suppress_emitted_appraisal_components(
    parsed: object,
    accepted_result: SemanticAppraisalResultV2 | None,
) -> object:
    """Remove exact accepted components so repetition terminates the loop."""

    if (
        accepted_result is None
        or not isinstance(parsed, Mapping)
        or set(parsed) != _SEMANTIC_APPRAISAL_RESULT_FIELDS
        or not isinstance(parsed["propositions"], list)
        or not isinstance(parsed["deltas"], list)
    ):
        return parsed

    emitted_signatures = set(
        _emitted_proposition_signatures(accepted_result)
    )
    emitted_paths = {
        delta["target_path"] for delta in accepted_result["deltas"]
    }
    propositions = []
    for proposition in parsed["propositions"]:
        if isinstance(proposition, Mapping):
            kind = proposition.get("proposition_kind")
            subject = proposition.get("subject_handle")
            object_handle = proposition.get("object_handle", "")
            if all(
                isinstance(value, str)
                for value in (kind, subject, object_handle)
            ):
                signature = "|".join((kind, subject, object_handle))
                if signature in emitted_signatures:
                    continue
        propositions.append(proposition)

    deltas = []
    for delta in parsed["deltas"]:
        if (
            isinstance(delta, Mapping)
            and delta.get("target_path") in emitted_paths
        ):
            continue
        deltas.append(delta)

    normalized = deepcopy(dict(parsed))
    normalized["propositions"] = propositions
    normalized["deltas"] = deltas
    normalized["explanation"] = _derive_semantic_item_explanation(
        propositions[0] if propositions else None,
        deltas[0] if deltas else None,
    )
    normalized_result = _derive_appraisal_selection_metadata(normalized)
    return normalized_result


def _emitted_proposition_signatures(
    result: SemanticAppraisalResultV2 | None,
) -> list[str]:
    """Project emitted proposition identities for bounded loop exclusion."""

    if result is None:
        return []
    return [
        "|".join((
            proposition["proposition_kind"],
            proposition["subject_handle"],
            proposition.get("object_handle", ""),
        ))
        for proposition in result["propositions"]
    ]


def _record_semantic_appraisal_trace(
    *,
    config: LLMCallConfig,
    question: SemanticQuestionV2,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    attempt_index: int,
    item_index: int,
    validation_error: str,
) -> None:
    """Preserve one protected semantic-appraisal model boundary."""

    stage_name = (
        f"semantic_appraisal.{question['question_id']}.item_{item_index}"
    )
    if attempt_index > 1:
        stage_name = f"{stage_name}.repair"
    failure_capsule.append_model_attempt(
        stage_name=stage_name,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        config=config,
        branch_id=question["question_id"],
        attempt_index=attempt_index,
        validation_error=validation_error,
        started_at=started_at,
    )


def _appraisal_repair_messages(
    *,
    system_message: SystemMessage,
    human_message: HumanMessage,
    invalid_candidate: str,
    contract_error: str,
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Build one bounded replacement request from the latest invalid output.

    Args:
        system_message: Stable semantic-appraisal instructions.
        human_message: Canonical question and evidence payload.
        invalid_candidate: Latest model output that failed validation.
        contract_error: Validation detail used to direct structural repair.

    Returns:
        A same-context message sequence requesting a complete replacement.
    """

    repair_payload = {
        "repair_instruction": (
            "请在相同语义问题和证据范围内返回一个完整替代 JSON 对象，只修复 JSON、"
            "字段、类型、handle 和 contract 约束。顶层字段必须恰好是 question_id、"
            "proposition 和 delta；proposition 和 delta 各自只能是一个对象或 null，"
            "不得使用数组；不要输出 Markdown、解释段落或 JSON 以外的文字。"
        ),
        "contract_error": contract_error,
    }
    repair_payload_text = json.dumps(
        repair_payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    residual_candidate_chars = (
        SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP
        - len(str(human_message.content))
        - len(repair_payload_text)
    )
    if residual_candidate_chars < 0:
        raise CognitionContextLimitError(
            "semantic appraisal repair context exceeds the contract cap"
        )
    if len(invalid_candidate) > residual_candidate_chars:
        truncation_marker = '\n... 已截断的不合格候选 ...\n'
        if residual_candidate_chars <= len(truncation_marker):
            invalid_candidate = invalid_candidate[:residual_candidate_chars]
        else:
            retained_chars = residual_candidate_chars - len(truncation_marker)
            head_chars = (retained_chars + 1) // 2
            tail_chars = retained_chars - head_chars
            tail_text = (
                invalid_candidate[-tail_chars:]
                if tail_chars
                else ""
            )
            invalid_candidate = (
                invalid_candidate[:head_chars]
                + truncation_marker
                + tail_text
            )
    messages = [
        system_message,
        human_message,
        AIMessage(content=invalid_candidate),
        HumanMessage(content=repair_payload_text),
    ]
    dynamic_context_chars = sum(
        len(str(message.content))
        for message in messages
        if not isinstance(message, SystemMessage)
    )
    if dynamic_context_chars > SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP:
        raise CognitionContextLimitError(
            "semantic appraisal repair context exceeds the contract cap"
        )
    return messages


def validate_semantic_appraisal_result(
    parsed: object,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    maximum_propositions: int = 8,
    maximum_deltas: int = 8,
    maximum_explanation_chars: int = 1000,
) -> SemanticAppraisalResultV2:
    """Validate one appraisal without interpreting its semantic prose."""

    _validate_question_handle_authority(question, handle_to_ref)
    if not isinstance(parsed, Mapping):
        raise ValueError("semantic appraisal must return an object")
    if set(parsed) != _SEMANTIC_APPRAISAL_RESULT_FIELDS:
        raise ValueError("semantic appraisal fields are not exact")
    if parsed["question_id"] != question["question_id"]:
        raise ValueError("semantic appraisal question id does not match")
    if (
        not isinstance(parsed["propositions"], list)
        or len(parsed["propositions"]) > maximum_propositions
    ):
        raise ValueError("semantic propositions are invalid")
    if (
        not isinstance(parsed["deltas"], list)
        or len(parsed["deltas"]) > maximum_deltas
    ):
        raise ValueError("semantic deltas are invalid")
    _validate_all_candidate_evidence_bindings(
        parsed["propositions"],
        parsed["deltas"],
        handle_to_ref,
    )
    selected_evidence = _validate_handles(
        parsed["selected_evidence_handles"],
        evidence_handles,
        "selected evidence",
        minimum=0,
    )
    selected_evidence_set = set(selected_evidence)
    selected_roles = _validate_handles(
        parsed["selected_role_handles"],
        set(question["permitted_role_handles"]),
        "selected roles",
        minimum=0,
    )
    propositions = [
        _validate_proposition(
            row,
            question,
            selected_evidence_set,
            handle_to_ref,
        )
        for row in parsed["propositions"]
    ]
    deltas = [
        _validate_delta(
            row,
            question,
            selected_evidence_set,
            handle_to_ref,
        )
        for row in parsed["deltas"]
    ]
    paths = [delta["target_path"] for delta in deltas]
    if len(paths) != len(set(paths)):
        raise ValueError("one appraisal cannot duplicate a target path")
    explanation = parsed["explanation"]
    if (
        not isinstance(explanation, str)
        or not 1 <= len(explanation) <= maximum_explanation_chars
    ):
        raise ValueError("semantic appraisal explanation is invalid")
    return {
        "question_id": question["question_id"],
        "selected_evidence_handles": selected_evidence,
        "selected_role_handles": selected_roles,
        "propositions": propositions,
        "deltas": deltas,
        "explanation": explanation,
    }


def _validate_proposition(
    value: Any,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate one semantic proposition and its role assignments."""

    if not isinstance(value, Mapping):
        raise ValueError("semantic proposition must be an object")
    allowed = {
        "proposition_kind",
        "subject_handle",
        "evidence_handles",
        "role_assignments",
        "semantic_value",
    }
    if "object_handle" in value:
        allowed.add("object_handle")
    if set(value) != allowed:
        raise ValueError("semantic proposition fields are not exact")
    proposition_kind = value["proposition_kind"]
    if proposition_kind not in question_proposition_kinds(question["question_kind"]):
        raise ValueError("semantic proposition kind is not owned by question")
    subject = value["subject_handle"]
    if subject not in set(question["permitted_role_handles"]):
        raise ValueError("semantic proposition subject handle is not permitted")
    required_subject_kind = _PROPOSITION_SUBJECT_KINDS.get(proposition_kind)
    if (
        required_subject_kind is not None
        and handle_to_ref[subject]["kind"] != required_subject_kind
    ):
        raise ValueError("semantic proposition kind requires subject kind")
    if "object_handle" in value and value["object_handle"] not in set(
        question["permitted_role_handles"]
    ):
        raise ValueError("semantic proposition object handle is not permitted")
    if proposition_kind == "goal_supersession":
        if "object_handle" not in value:
            raise ValueError("goal supersession requires an object handle")
        if (
            not subject.startswith("g")
            or not value["object_handle"].startswith("g")
        ):
            raise ValueError("goal supersession requires two goal handles")
        if subject == value["object_handle"]:
            raise ValueError("goal supersession requires a distinct goal")
    cited = _validate_handles(
        value["evidence_handles"],
        evidence_handles,
        "proposition evidence",
    )
    assignments = value["role_assignments"]
    if not isinstance(assignments, list) or len(assignments) > 8:
        raise ValueError("semantic proposition roles are invalid")
    normalized_assignments: list[dict[str, str]] = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != {
            "role",
            "entity_handle",
        }:
            raise ValueError("semantic role assignment is invalid")
        if assignment["role"] not in {
            "actor",
            "experiencer",
            "target",
            "object",
            "affected_goal",
            "affected_relationship",
        }:
            raise ValueError("semantic role value is invalid")
        if assignment["entity_handle"] not in set(
            question["permitted_role_handles"]
        ):
            permitted_handles = sorted(
                set(question["permitted_role_handles"])
            )
            raise ValueError(
                "role_assignments[*].entity_handle must be one of "
                + json.dumps(permitted_handles)
            )
        normalized_assignments.append(dict(assignment))
    referenced_handles = [subject]
    if "object_handle" in value:
        referenced_handles.append(value["object_handle"])
    referenced_handles.extend(
        assignment["entity_handle"]
        for assignment in normalized_assignments
    )
    _validate_candidate_evidence_binding(
        referenced_handles,
        cited,
        handle_to_ref,
    )
    result = {
        "proposition_kind": proposition_kind,
        "subject_handle": subject,
        "evidence_handles": cited,
        "role_assignments": normalized_assignments,
        "semantic_value": _require_text(value.get("semantic_value")),
    }
    if "object_handle" in value:
        result["object_handle"] = value["object_handle"]
    return result


def _validate_delta(
    value: Any,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Validate one allowlisted semantic numeric delta."""

    if not isinstance(value, Mapping) or set(value) != {
        "target_path",
        "delta",
        "evidence_handles",
        "reason",
    }:
        raise ValueError("semantic delta fields are not exact")
    path = value["target_path"]
    if path not in set(question["permitted_delta_paths"]):
        raise ValueError(
            f"semantic delta path {path} is not owned by question"
        )
    delta = value["delta"]
    if (
        isinstance(delta, bool)
        or not isinstance(delta, int)
        or not -40 <= delta <= 40
    ):
        raise ValueError(
            "semantic delta must be a JSON integer from -40 through 40; "
            f"received {type(delta).__name__}"
        )
    cited = _validate_handles(
        value["evidence_handles"],
        evidence_handles,
        "delta evidence",
    )
    path_handle = path.split(".")[1]
    _validate_candidate_evidence_binding(
        [path_handle],
        cited,
        handle_to_ref,
    )
    return {
        "target_path": path,
        "delta": delta,
        "evidence_handles": cited,
        "reason": _require_text(value["reason"], maximum=300),
    }


def _validate_handles(
    value: Any,
    allowed: set[str],
    label: str,
    *,
    minimum: int = 1,
) -> list[str]:
    """Validate a bounded duplicate-free handle list."""

    if not isinstance(value, list) or not minimum <= len(value) <= 8:
        raise ValueError(
            f"{label} handles must contain between {minimum} and 8 items"
        )
    if any(not isinstance(handle, str) or handle not in allowed for handle in value):
        raise ValueError(f"{label} contains an unknown handle")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} handles are duplicated")
    return list(value)


def _validate_candidate_evidence_binding(
    candidate_handles: Sequence[str],
    cited_evidence_handles: Sequence[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Require every prompt-local candidate to cite its source evidence."""

    cited = set(cited_evidence_handles)
    for handle in candidate_handles:
        evidence_handle = _candidate_evidence_handle(handle, handle_to_ref)
        if evidence_handle is not None and evidence_handle not in cited:
            raise ValueError(
                f"causal candidate {handle} must cite its originating "
                f"evidence {evidence_handle}"
            )


def _validate_all_candidate_evidence_bindings(
    propositions: Sequence[Any],
    deltas: Sequence[Any],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Report every missing candidate origin needed by one replacement."""

    violations: set[tuple[str, str]] = set()
    for proposition in propositions:
        if not isinstance(proposition, Mapping):
            continue
        cited = proposition.get("evidence_handles")
        if not isinstance(cited, list):
            continue
        handles = [
            proposition.get("subject_handle"),
            proposition.get("object_handle"),
        ]
        assignments = proposition.get("role_assignments")
        if isinstance(assignments, list):
            handles.extend(
                assignment.get("entity_handle")
                for assignment in assignments
                if isinstance(assignment, Mapping)
            )
        _collect_missing_candidate_bindings(
            handles,
            cited,
            handle_to_ref,
            violations,
        )
    for delta in deltas:
        if not isinstance(delta, Mapping):
            continue
        path = delta.get("target_path")
        cited = delta.get("evidence_handles")
        if not isinstance(path, str) or not isinstance(cited, list):
            continue
        pieces = path.split(".")
        if len(pieces) != 3:
            continue
        _collect_missing_candidate_bindings(
            [pieces[1]],
            cited,
            handle_to_ref,
            violations,
        )
    if violations:
        bindings = ", ".join(
            f"{handle}->{evidence_handle}"
            for handle, evidence_handle in sorted(violations)
        )
        raise ValueError(
            "causal candidates must cite originating evidence: " + bindings
        )


def _collect_missing_candidate_bindings(
    candidate_handles: Sequence[Any],
    cited_evidence_handles: Sequence[Any],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    violations: set[tuple[str, str]],
) -> None:
    """Collect missing candidate origins from one proposition or delta."""

    cited = {
        handle for handle in cited_evidence_handles if isinstance(handle, str)
    }
    for handle in candidate_handles:
        if not isinstance(handle, str):
            continue
        evidence_handle = _candidate_evidence_handle(handle, handle_to_ref)
        if evidence_handle is not None and evidence_handle not in cited:
            violations.add((handle, evidence_handle))


def _candidate_evidence_handle(
    candidate_handle: str,
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Map one candidate handle back to its exact evidence handle."""

    ref = handle_to_ref.get(candidate_handle)
    if ref is None:
        return None
    entity_id = ref.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id.startswith("candidate:"):
        return None
    pieces = entity_id.split(":", maxsplit=2)
    if len(pieces) == 3 and pieces[1] in {
        "event",
        "threat",
        "knowledge_gap",
    }:
        return pieces[2]
    return None


def _validate_question_handle_authority(
    question: SemanticQuestionV2,
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> None:
    """Require every question handle to exist in the canonical projection."""

    canonical_handles = set(handle_to_ref)
    permitted_handles = set(question["permitted_role_handles"])
    if not permitted_handles <= canonical_handles:
        raise ValueError("semantic question contains a non-canonical role handle")
    for path in question["permitted_delta_paths"]:
        pieces = path.split(".")
        if len(pieces) >= 3 and pieces[1] not in canonical_handles:
            raise ValueError("semantic question contains a non-canonical path handle")


def _require_text(value: Any, maximum: int = 200) -> str:
    """Require bounded non-empty semantic text."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("semantic text is invalid")
    return value


def _fit_appraisal_payload(payload: dict[str, Any]) -> str:
    """Fit one appraisal after reducing state, then evidence text."""

    supplemental_order = (
        "knowledge_gaps",
        "events",
        "threats",
        "goals",
        "affect",
        "relationship",
        "roles",
    )
    state = payload["state"]
    if not isinstance(state, Mapping):
        raise ValueError(
            "semantic appraisal state projection is invalid"
        )
    evidence_rows = payload["evidence"]
    if not isinstance(evidence_rows, list):
        raise ValueError(
            "semantic appraisal evidence projection is invalid"
        )
    projected_state = dict(state)
    while True:
        candidate = dict(payload)
        candidate["state"] = projected_state
        payload_text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if len(payload_text) <= SEMANTIC_APPRAISAL_PROMPT_CAP:
            return payload_text
        removed = False
        for key in supplemental_order:
            value = projected_state.get(key)
            if isinstance(value, list) and value:
                projected_state[key] = value[:-1]
                removed = True
                break
            if key in projected_state and value:
                projected_state.pop(key)
                removed = True
                break
        if removed:
            continue
        try:
            fitted_payload = fit_evidence_texts_to_budget(
                candidate,
                evidence_rows,
                text_field="semantic_text",
                maximum_chars=SEMANTIC_APPRAISAL_PROMPT_CAP,
                minimum_text_chars=MIN_PROMPT_EVIDENCE_TEXT_CHARS,
            )
        except PromptBudgetError as exc:
            raise CognitionContextLimitError(
                "required semantic appraisal context exceeds the contract cap"
            ) from exc
        return fitted_payload


def _compact_permitted_delta_path_domains(
    permitted_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Group exact path handles that expose an equal axis set."""

    axes_by_field_and_handle: dict[str, dict[str, set[str]]] = {}
    for path in permitted_paths:
        pieces = path.split(".")
        if len(pieces) != 3 or any(not piece for piece in pieces):
            raise ValueError("semantic appraisal delta path is invalid")
        state_field, handle, axis = pieces
        axes_by_handle = axes_by_field_and_handle.setdefault(
            state_field,
            {},
        )
        axes_by_handle.setdefault(handle, set()).add(axis)

    domains: list[dict[str, Any]] = []
    for state_field in sorted(axes_by_field_and_handle):
        handles_by_axes: dict[tuple[str, ...], list[str]] = {}
        for handle, axes in axes_by_field_and_handle[state_field].items():
            axis_set = tuple(sorted(axes))
            handles_by_axes.setdefault(axis_set, []).append(handle)
        for axes in sorted(handles_by_axes):
            domains.append({
                "state_field": state_field,
                "handles": sorted(handles_by_axes[axes]),
                "axes": list(axes),
            })
    return domains


def _project_question_state(
    projection: PromptProjectionV2,
    question: SemanticQuestionV2,
) -> dict[str, Any]:
    """Expose only the state partition authorized for one appraisal family."""

    allowed = set(question["permitted_role_handles"])
    source = projection.payload
    result: dict[str, Any] = {}
    for field_name in ("goals", "threats", "events", "knowledge_gaps"):
        rows = source.get(field_name, [])
        selected = [
            dict(row)
            for row in rows
            if isinstance(row, Mapping) and row.get("handle") in allowed
        ]
        if selected:
            result[field_name] = selected
    roles = source.get("roles", {})
    if isinstance(roles, Mapping):
        selected_roles = {
            handle: summary
            for handle, summary in roles.items()
            if handle in allowed
        }
        if selected_roles:
            result["roles"] = selected_roles
    if "r1" in allowed and isinstance(source.get("relationship"), Mapping):
        result["relationship"] = dict(source["relationship"])
    constraints = _project_question_constraints(
        projection,
        source.get("character_constraints"),
        allowed,
    )
    if constraints:
        result["character_constraints"] = constraints
    identity = projection.identity_by_question.get(
        question["question_kind"],
        {},
    )
    if identity:
        result["character_identity"] = deepcopy(identity)
    return result


def _project_question_constraints(
    projection: PromptProjectionV2,
    constraints: Any,
    allowed: set[str],
) -> dict[str, Any]:
    """Filter fixed character constraints through permitted local handles."""

    if not isinstance(constraints, Mapping):
        return {}
    selected: dict[str, Any] = {}
    drive_ids = {
        ref["entity_id"]
        for handle, ref in projection.handle_to_ref.items()
        if handle in allowed and ref["kind"] == "drive"
    }
    drives = constraints.get("drives")
    if isinstance(drives, Mapping):
        selected_drives = {
            drive_id: dict(value)
            for drive_id, value in drives.items()
            if drive_id in drive_ids and isinstance(value, Mapping)
        }
        if selected_drives:
            selected["drives"] = selected_drives
    standards = constraints.get("standards")
    standard_indexes = sorted(
        int(handle[1:]) - 1
        for handle in allowed
        if handle.startswith("s") and handle[1:].isdigit()
    )
    if isinstance(standards, list) and standard_indexes:
        selected["standards"] = [
            dict(standards[index])
            for index in standard_indexes
            if 0 <= index < len(standards)
            and isinstance(standards[index], Mapping)
        ]
    meaning = constraints.get("meaning_state")
    if "m1" in allowed and isinstance(meaning, Mapping):
        selected["meaning_state"] = dict(meaning)
    return selected
