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
    capture_validation_event,
    capture_validation_stage,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_APPRAISAL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    question_proposition_kind_semantics,
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    PromptBudgetError,
    fit_evidence_texts_to_budget,
    reduce_constraints_projection,
    reduce_identity_projection,
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


SEMANTIC_APPRAISAL_ATTEMPT_LIMIT = V2_APPRAISAL_TOTAL_ATTEMPTS
SEMANTIC_APPRAISAL_ITEM_LIMIT = 8
SEMANTIC_APPRAISAL_PROMPT_CAP = 20000
SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP = 24000
SEMANTIC_APPRAISAL_ITEM_EXPLANATION_LIMIT = 120
MIN_PROMPT_EVIDENCE_TEXT_CHARS = 96
MAX_APPRAISAL_OBJECT_HANDLES = 8
MAX_APPRAISAL_SEMANTIC_TEXT_CHARS = 200
MAX_APPRAISAL_DELTA_REASON_CHARS = 300
MAX_ERROR_ALLOWLIST_ITEMS = 40
DELTA_LIMIT_NARROW = 10
DELTA_LIMIT_WIDE = 40
_DELTA_LIMIT_BY_STATE_FIELD = {
    "relationship": DELTA_LIMIT_NARROW,
    "meaning_state": DELTA_LIMIT_NARROW,
}
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
_SEMANTIC_DELTA_PATH_ERROR_PREFIX = "semantic delta path "
_SEMANTIC_DELTA_PATH_ERROR_RULE_SUFFIX = " is not owned by question"
_SEMANTIC_DELTA_PATH_ERROR_PERMITTED_PATHS_SEPARATOR = (
    "; permitted paths:"
)

_PROPOSITION_SUBJECT_KINDS = {
    "goal_release": "goal",
    "goal_supersession": "goal",
    "goal_completed": "goal",
    "event_completed": "event",
    "threat_resolved": "threat",
    "event_repaired": "event",
    "knowledge_answered": "knowledge_gap",
}
_PROPOSITION_SUBJECT_KIND_SETS = {
    "outcome_pending": frozenset({
        "goal",
        "event",
        "threat",
        "knowledge_gap",
    }),
}


def _delta_limit_for_state_field(state_field: str) -> int:
    """Return the reducer's per-event delta bound for one state field."""

    return _DELTA_LIMIT_BY_STATE_FIELD.get(state_field, DELTA_LIMIT_WIDE)


SEMANTIC_APPRAISAL_PROMPT = '''你负责根据有界证据回答一个范围明确的语义问题，并返回一个可验证的
micro_appraisal。本阶段只判断证据已经支持的含义；动作选择、对话生成、emotion id、生命周期状态、
持久化和事实补充属于其他阶段。每个 proposition_kind 都是所给语义定义已经成立的肯定式断言。

# 按这个顺序判断
1. 先读 question.question_id、question.question_kind 和 question.semantic_question，确定这一题要判断的含义。
   再读 evidence；每行的 source_kind 说明来源，角色自己的反思或内部观察仍是证据，不是当前用户的即时发言。
   只使用本次输入允许的 handle 和证据，不把来源包标题、时间戳、传输摘要、schema key 或运行元数据当成新事实。
2. 先完成句柄映射，再写语义文字。question.handle_field_domains 说明每个字段能用哪些 handle：subject_handle、
   object_handle 和 role_assignments[*].entity_handle 使用 permitted_role_handles，evidence_handles 使用其中的
   evidence handle。ceN、ctN、ckN 分别是候选事件、威胁和知识缺口，evN 是持久事件，eN 是证据；它们不是人物。
   人物角色使用 self 或 current_user，role 只使用 actor、experiencer、target、object、affected_goal、
   affected_relationship 这六个 enum token。证据标签或关系注释（如 beneficiary）只是待解读的事实，
   不是 role 值；不支持时省略该 assignment，不编造、不照抄。
3. 使用 question.candidate_origin_evidence 做来源核对。一个 proposition 或 delta 的 subject_handle、object_handle、
   role assignment 或 target_path 只要出现 ceN、ctN 或 ckN，就把映射出的来源 evidence handle 放进同一个对象的
   evidence_handles；这就是候选来源引用。找不到对应来源时，省略这个候选或整个对象。角色 handle 不能代替
   evidence handle，未知 handle 也不能靠猜测补齐。

# 继续按顺序完成输出
4. 若有数值变化，只从 question.permitted_delta_path_domains 选择一项。每项给出 state_field、handles、
   axes 和 delta_limit；从同一项各取一个 state_field、一个 handle、一个 axis，按 state_field.handle.axis
   原样拼成 target_path，不构造其他 state path。delta 必须是该项 delta_limit 范围内的整数。
   有证据支持的蓄意阻碍、明确伤害或边界侵害可选 harm，
   以及有支持时的 unfairness 和 intentionality；已发生且不可逆的损失可选负向 outcome_impact 或
   temporal_loss；污染或基本规范/边界受到侵害可选 contamination_risk 或 norm_violation。axis 只描述
   证据中的可观察后果，不得因此增添情绪、归因类别、未给出的角色或事实；不在允许表中的 axis 不能使用，
   证据不足时返回 null。
5. 每次只生成一个 micro_appraisal item。proposition 和 delta 各自只能是一个对象或 null，不能使用数组，也不能列举多个候选。
   没有新的受支持项目时，两者都返回 null 以结束循环。semantic_value 写简洁的简体中文，
   目标约 120 字、上限 200 字，不重复标准或证据解释，也不写数值；delta reason 使用简体中文且不超过 300 字。
   引用的用户原文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样，普通自由文本使用“当前角色”
   和“当前用户”，不要写内部角色句柄或英文角色称谓。

# 只返回这个对象
顶层字段必须恰好是 question_id、proposition、delta。不要输出 explanation、selected_evidence_handles、selected_role_handles、propositions 或 deltas。proposition 若存在，字段必须恰好是 proposition_kind、
subject_handle、evidence_handles、role_assignments、semantic_value，可选 object_handle；
role_assignments 是必填字段，证据不支持任何角色时写 []；每条 role assignment 只能有 role 和
entity_handle。delta 若存在，字段必须恰好是 target_path、delta、evidence_handles、reason。delta 必须是
所选路径所在域 delta_limit 范围内的 JSON 整数，例如 -5、0 或 10；不使用字符串、小数、百分比或比例。

# 输出前最后检查
确认 question_id 没有改写；每个结构化 handle 都来自自己的域；每个 ceN、ctN、ckN 都带有对应的来源 evidence
handle；target_path 完全来自一个允许的 state_field.handle.axis；文字没有把运行元数据写入语义。只返回 JSON 对象。

# 输出示例
{
  "question_id": "q:event_agency",
  "proposition": {"proposition_kind": "event_completed", "subject_handle": "ev1", "evidence_handles": ["e1"], "role_assignments": [{"role": "actor", "entity_handle": "current_user"}], "semantic_value": "这里写一句简体中文语义描述。"},
  "delta": {"target_path": "goals.g1.importance", "delta": 10, "evidence_handles": ["e1"], "reason": "这里写不超过三百字的简体中文原因。"}
}
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
    evidence_by_handle: dict[str, dict[str, str]] = {}
    for row in evidence:
        if row["evidence_handle"] not in question["evidence_handles"]:
            continue
        projected_row: dict[str, str] = {
            "handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "source_kind": row["evidence_ref"]["source_kind"],
        }
        if "memory_scope" in row:
            projected_row["memory_scope"] = row["memory_scope"]
        evidence_by_handle[row["evidence_handle"]] = projected_row
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
        payload_text, surviving_role_handles = _fit_appraisal_payload(
            payload,
            system_prompt_chars=len(system_message.content),
        )
        repair_allowed_values = {
            "handle_field_domains": deepcopy(
                payload["question"]["handle_field_domains"]
            ),
            "candidate_origin_evidence": deepcopy(
                payload["question"]["candidate_origin_evidence"]
            ),
            "permitted_delta_path_domains": deepcopy(
                payload["question"]["permitted_delta_path_domains"]
            ),
        }
        item_question["permitted_role_handles"] = [
            handle
            for handle in item_question["permitted_role_handles"]
            if handle in surviving_role_handles
        ]
        item_question["permitted_delta_paths"] = [
            path
            for path in item_question["permitted_delta_paths"]
            if len(path.split(".")) == 3
            and path.split(".")[1] in surviving_role_handles
        ]
        try:
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
                repair_allowed_values=repair_allowed_values,
                item_index=item_index,
            )
        except CognitionExecutionError as exc:
            if accepted_result is None:
                raise
            capture_validation_event(
                "semantic_appraisal_bounded_termination",
                {
                    "question_id": question["question_id"],
                    "item_index": item_index,
                    "error_code": exc.error_code,
                    "attempt_count": exc.attempt_count,
                    "accepted_proposition_count": len(
                        accepted_result["propositions"]
                    ),
                    "accepted_delta_count": len(accepted_result["deltas"]),
                    "disposition": "accepted_prefix",
                    "error": str(exc),
                },
            )
            return accepted_result
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
    repair_allowed_values: Mapping[str, Any],
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
            full_contract_error = str(exc)
            request_messages = _appraisal_repair_messages(
                system_message=system_message,
                human_message=human_message,
                invalid_candidate=str(raw_output),
                contract_error=_compact_semantic_contract_error(
                    full_contract_error
                ),
                allowed_values=repair_allowed_values,
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


def _compact_semantic_contract_error(contract_error: str) -> str:
    """Remove only the validator-owned permitted-path suffix.

    The unowned semantic-delta-path validator error already identifies the
    failed rule and exact path before appending the permitted-path domain. The
    domain is projected separately in ``allowed_values`` for repair requests,
    so only that exact validator suffix is removed from model-facing text.
    Other contract errors remain unchanged.

    Args:
        contract_error: Complete validator error captured before repair
            feedback projection.

    Returns:
        The compacted model-facing error or the original error unchanged.
    """

    if not contract_error.startswith(_SEMANTIC_DELTA_PATH_ERROR_PREFIX):
        return contract_error
    marker_index = contract_error.find(
        _SEMANTIC_DELTA_PATH_ERROR_PERMITTED_PATHS_SEPARATOR
    )
    if marker_index < 0:
        return contract_error
    if not contract_error[:marker_index].endswith(
        _SEMANTIC_DELTA_PATH_ERROR_RULE_SUFFIX
    ):
        return contract_error
    compacted_error = contract_error[:marker_index]
    return compacted_error


def _appraisal_repair_messages(
    *,
    system_message: SystemMessage,
    human_message: HumanMessage,
    invalid_candidate: str,
    contract_error: str,
    allowed_values: Mapping[str, Any],
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
            "请把 contract_error 当作唯一失败规则，在原来的语义问题和证据范围内完整重生成 JSON。"
            "只修正该规则涉及的字段、类型、handle 或允许路径，保留其他受支持含义；"
            "allowed_values 给出的现有域、候选来源和路径表是唯一允许值；"
            "proposition 的 role_assignments 是必填字段，证据不支持任何角色时写 []；"
            "role 只能是六个 enum token：actor、experiencer、target、object、affected_goal、"
            "affected_relationship；证据标签（如 beneficiary）不能写入 role，不支持时省略该 assignment。"
            "顶层字段必须恰好是 question_id、proposition 和 delta；两者各自只能是一个对象或 null，"
            "不能使用数组，也不能输出 Markdown、解释段落或 JSON 以外的文字。"
        ),
        "contract_error": contract_error,
        "allowed_values": dict(allowed_values),
    }
    repair_payload_text = json.dumps(
        repair_payload,
        ensure_ascii=False,
        sort_keys=False,
    )
    system_prompt_chars = len(str(system_message.content))
    residual_candidate_chars = (
        SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP
        - system_prompt_chars
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
    if (
        system_prompt_chars + dynamic_context_chars
        > SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP
    ):
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
        maximum=len(evidence_handles),
    )
    selected_evidence_set = set(selected_evidence)
    selected_roles = _validate_handles(
        parsed["selected_role_handles"],
        set(question["permitted_role_handles"]),
        "selected roles",
        minimum=0,
        maximum=len(question["permitted_role_handles"]),
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
    permitted_kinds = question_proposition_kinds(question["question_kind"])
    if proposition_kind not in permitted_kinds:
        raise ValueError(
            "semantic proposition kind "
            f"{proposition_kind!r} is not owned by question; "
            f"permitted kinds: {json.dumps(permitted_kinds)}"
        )
    subject = value["subject_handle"]
    if subject not in set(question["permitted_role_handles"]):
        raise ValueError(
            "semantic proposition subject handle "
            f"{subject!r} is not permitted; allowed role handles: "
            f"{_allowlist_hint(question['permitted_role_handles'])}"
        )
    required_subject_kind = _PROPOSITION_SUBJECT_KINDS.get(proposition_kind)
    if (
        required_subject_kind is not None
        and handle_to_ref[subject]["kind"] != required_subject_kind
    ):
        raise ValueError(
            "semantic proposition kind requires subject kind "
            f"{required_subject_kind!r}; received "
            f"{handle_to_ref[subject]['kind']!r}"
        )
    permitted_subject_kinds = _PROPOSITION_SUBJECT_KIND_SETS.get(
        proposition_kind
    )
    if (
        permitted_subject_kinds is not None
        and handle_to_ref[subject]["kind"] not in permitted_subject_kinds
    ):
        raise ValueError(
            "semantic proposition subject kind "
            f"{handle_to_ref[subject]['kind']!r} is not permitted for "
            f"{proposition_kind!r}; permitted kinds: "
            f"{json.dumps(sorted(permitted_subject_kinds))}"
        )
    if "object_handle" in value and value["object_handle"] not in set(
        question["permitted_role_handles"]
    ):
        raise ValueError(
            "semantic proposition object handle "
            f"{value['object_handle']!r} is not permitted; allowed role "
            f"handles: {_allowlist_hint(question['permitted_role_handles'])}"
        )
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
        "semantic_value": _require_text(
            value.get("semantic_value"),
            "semantic_value",
        ),
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
            f"semantic delta path {path!r} is not owned by question; "
            f"permitted paths: "
            f"{_allowlist_hint(question['permitted_delta_paths'])}"
        )
    delta = value["delta"]
    delta_limit = _delta_limit_for_state_field(path.split(".")[0])
    if (
        isinstance(delta, bool)
        or not isinstance(delta, int)
        or not -delta_limit <= delta <= delta_limit
    ):
        raise ValueError(
            "semantic delta must be a JSON integer from "
            f"{-delta_limit} through {delta_limit}; "
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
        "reason": _require_text(
            value["reason"],
            "reason",
            maximum=MAX_APPRAISAL_DELTA_REASON_CHARS,
        ),
    }


def _allowlist_hint(values: Sequence[str]) -> str:
    """Render a bounded sorted allowlist for one contract error message."""

    sorted_values = sorted(values)
    shown_values = sorted_values[:MAX_ERROR_ALLOWLIST_ITEMS]
    hint = json.dumps(shown_values)
    if len(sorted_values) > MAX_ERROR_ALLOWLIST_ITEMS:
        hint = (
            f"{hint} "
            f"(+{len(sorted_values) - MAX_ERROR_ALLOWLIST_ITEMS} more)"
        )
    return hint


def _validate_handles(
    value: Any,
    allowed: set[str],
    label: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_APPRAISAL_OBJECT_HANDLES,
) -> list[str]:
    """Validate a bounded duplicate-free handle list."""

    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(
            f"{label} handles must contain between {minimum} and {maximum} "
            f"items; allowed: {_allowlist_hint(allowed)}"
        )
    invalid_handles = [
        handle
        for handle in value
        if not isinstance(handle, str) or handle not in allowed
    ]
    if invalid_handles:
        rejected_text = json.dumps(
            sorted({str(handle) for handle in invalid_handles})
        )
        raise ValueError(
            f"{label} contains unknown handles {rejected_text}; "
            f"allowed: {_allowlist_hint(allowed)}"
        )
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


def _require_text(
    value: Any,
    label: str,
    *,
    maximum: int = MAX_APPRAISAL_SEMANTIC_TEXT_CHARS,
) -> str:
    """Require bounded non-empty semantic text."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{label} must be non-empty text up to {maximum} characters"
        )
    return value


def _fit_appraisal_payload(
    payload: dict[str, Any],
    *,
    system_prompt_chars: int,
) -> tuple[str, frozenset[str]]:
    """Fit one appraisal packet and return its text and surviving handles."""

    supplemental_order = (
        "knowledge_gaps",
        "events",
        "threats",
        "goals",
        "affect",
        "relationship",
        "character_operational_context",
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
    question = payload["question"]
    question_kind = (
        question.get("question_kind")
        if isinstance(question, Mapping)
        else None
    )
    if system_prompt_chars >= SEMANTIC_APPRAISAL_PROMPT_CAP:
        raise CognitionContextLimitError(
            "required semantic appraisal context exceeds the contract cap"
        )
    projected_state = dict(state)
    while True:
        candidate = dict(payload)
        candidate["state"] = projected_state
        payload_text = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=False,
        )
        if (
            system_prompt_chars + len(payload_text)
            <= SEMANTIC_APPRAISAL_PROMPT_CAP
        ):
            surviving_handles = frozenset(
                question.get("permitted_role_handles", ())
            )
            return payload_text, surviving_handles
        identity = projected_state.get("character_identity")
        if isinstance(identity, Mapping) and reduce_identity_projection(
            identity
        ):
            continue
        constraints = projected_state.get("character_constraints")
        if isinstance(constraints, Mapping) and reduce_constraints_projection(
            constraints
        ):
            continue
        removed = False
        removed_row: Any = None
        for key in supplemental_order:
            if (
                question_kind == "relationship_social"
                and key == "relationship"
            ):
                continue
            value = projected_state.get(key)
            if isinstance(value, list) and value:
                removed_row = value[-1]
                projected_state[key] = value[:-1]
                removed = True
                break
            if key in projected_state and value:
                projected_state.pop(key)
                removed = True
                break
        if removed:
            removed_handle: str | None = None
            if key in {
                "goals",
                "threats",
                "events",
                "knowledge_gaps",
            }:
                if isinstance(removed_row, Mapping):
                    row_handle = removed_row.get("handle")
                    if isinstance(row_handle, str):
                        removed_handle = row_handle
            elif key == "relationship":
                removed_handle = "r1"
            if removed_handle is not None:
                permitted_handles = question["permitted_role_handles"]
                if removed_handle in permitted_handles:
                    question["permitted_role_handles"] = [
                        handle
                        for handle in permitted_handles
                        if handle != removed_handle
                    ]
                field_domains = question["handle_field_domains"]
                for field_name in (
                    "subject_handle",
                    "object_handle",
                    "entity_handle",
                ):
                    values = field_domains[field_name]
                    if removed_handle in values:
                        field_domains[field_name] = [
                            value
                            for value in values
                            if value != removed_handle
                        ]
                origin_evidence = question["candidate_origin_evidence"]
                origin_evidence.pop(removed_handle, None)
                path_domains = question["permitted_delta_path_domains"]
                for domain in path_domains:
                    handles = domain["handles"]
                    if removed_handle in handles:
                        domain["handles"] = [
                            handle
                            for handle in handles
                            if handle != removed_handle
                        ]
                question["permitted_delta_path_domains"] = [
                    domain
                    for domain in path_domains
                    if domain["handles"]
                ]
            continue
        try:
            fitted_payload = fit_evidence_texts_to_budget(
                candidate,
                evidence_rows,
                text_field="semantic_text",
                maximum_chars=(
                    SEMANTIC_APPRAISAL_PROMPT_CAP - system_prompt_chars
                ),
                minimum_text_chars=MIN_PROMPT_EVIDENCE_TEXT_CHARS,
            )
        except PromptBudgetError as exc:
            raise CognitionContextLimitError(
                "required semantic appraisal context exceeds the contract cap"
            ) from exc
        surviving_handles = frozenset(
            question.get("permitted_role_handles", ())
        )
        return fitted_payload, surviving_handles


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
                "delta_limit": _delta_limit_for_state_field(state_field),
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
    if "r1" in allowed and isinstance(source.get("relationship"), Mapping):
        result["relationship"] = dict(source["relationship"])
    operational_context = _project_question_operational_context(
        source.get("character_operational_context"),
        question_kind=question["question_kind"],
    )
    if operational_context is not None:
        result["character_operational_context"] = operational_context
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


def _project_question_operational_context(
    value: Any,
    *,
    question_kind: str,
) -> dict[str, list[dict[str, Any]]] | None:
    """Restrict global posture to the appraisal families allowed to use it."""

    if not isinstance(value, Mapping):
        return None
    affect = value.get("affect")
    pressures = value.get("pressures")
    if not isinstance(affect, list) or not isinstance(pressures, list):
        return None
    if question_kind in {"event_agency", "epistemic_comparison_memory"}:
        return None
    selected_affect = [
        dict(row)
        for row in affect
        if isinstance(row, Mapping)
    ]
    selected_pressures = [
        dict(row)
        for row in pressures
        if isinstance(row, Mapping)
    ]
    if question_kind == "moral_identity":
        allowed_causes = {"boundary_pressure", "repair_pressure"}
        selected_affect = [
            row for row in selected_affect
            if row.get("cause_class") in allowed_causes
        ]
        selected_pressures = [
            row for row in selected_pressures
            if row.get("cause_class") in allowed_causes
        ]
    elif question_kind == "existential_drive":
        allowed_causes = {
            "meaning_pressure",
            "goal_pressure",
            "competence_pressure",
        }
        selected_affect = [
            row for row in selected_affect
            if row.get("cause_class") in allowed_causes
        ]
        selected_pressures = [
            row for row in selected_pressures
            if row.get("cause_class") in allowed_causes
        ]
    if not selected_affect and not selected_pressures:
        return None
    return {
        "affect": selected_affect,
        "pressures": selected_pressures,
    }


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
    meaning = constraints.get("meaning_state")
    if "m1" in allowed and isinstance(meaning, Mapping):
        selected["meaning_state"] = dict(meaning)
    return selected
