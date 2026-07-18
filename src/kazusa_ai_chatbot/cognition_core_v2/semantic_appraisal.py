"""Scoped semantic appraisal with deterministic structural validation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
    CognitionContextLimitError,
    CognitionEvidenceV2,
    SemanticAppraisalResultV2,
    SemanticQuestionV2,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    question_proposition_kinds,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


SEMANTIC_APPRAISAL_PROMPT = '''你根据有界证据回答一个范围明确的语义问题。
只使用本次 prompt 允许的 handle 和语义描述。动作选择、对话生成、emotion id、生命周期状态与
事实补充不属于本阶段。只有在所给证据支持时，才返回语义命题和允许路径上的数值变化。

遵守每条证据的 source_kind。角色自己的反思或内部观察属于证据，不是当前用户的即时发言。
生成的文字不复述来源包标题、时间戳、传输摘要、schema key 或运行元数据。新生成的自由文本使用
简体中文；引用的用户原文、专有名词、代码、URL 以及必要的 schema 或 enum token 保持原样。

# 输出格式
只返回一个 JSON 对象，字段必须恰好是 question_id、selected_evidence_handles、
selected_role_handles、propositions、deltas 和 explanation。每条 proposition 与 delta 都必须
引用提供的 evidence handle；未知或缺少支持的含义直接省略。

每个 proposition 对象必须恰好包含 proposition_kind、subject_handle、evidence_handles、
role_assignments 和 semantic_value，并可选包含 object_handle。每条 role assignment 必须恰好
包含 role 和 entity_handle。每个 delta 对象必须恰好包含 target_path、delta、
evidence_handles 和 reason。所给 role handle 与 delta path 按原值使用；不输出 kind、handle、
semantic_text、role_handles、path 或其他 proposition、delta 字段。

semantic_value 是一句简洁描述，目标长度 120 字符且上限 200 字符，其中不重复标准、约束或证据
解释，也不使用数值；数值只放在 delta 字段。每条 delta reason 不超过 300 字符，explanation
不超过 1000 字符。role 必须取 actor、experiencer、target、object、affected_goal 或
affected_relationship。r1、ce1、ct1、ck1 等实体 handle 放在 entity_handle，不能放在 role。
'''


async def appraise_semantic_question(
    question: SemanticQuestionV2,
    evidence: Sequence[CognitionEvidenceV2],
    projection: PromptProjectionV2,
    services: CognitionCoreServicesV2,
) -> SemanticAppraisalResultV2:
    """Run one bounded family appraisal and return no state authority."""

    evidence_by_handle = {
        row["evidence_handle"]: {
            "handle": row["evidence_handle"],
            "semantic_text": row["semantic_text"],
            "source_kind": row["evidence_ref"]["source_kind"],
        }
        for row in evidence
        if row["evidence_handle"] in question["evidence_handles"]
    }
    payload = {
        "question": {
            "question_id": question["question_id"],
            "question_kind": question["question_kind"],
            "semantic_question": question["semantic_question"],
            "permitted_role_handles": question["permitted_role_handles"],
            "permitted_delta_paths": question["permitted_delta_paths"],
            "permitted_proposition_kinds": list(
                question_proposition_kinds(question["question_kind"])
            ),
            "output_schema": {
                "propositions": "有证据支持的 proposition 对象列表",
                "deltas": "允许路径上的有符号整数 delta 列表",
                "evidence_handles": "每条 proposition 和 delta 都必须引用这些 handle",
            },
        },
        "evidence": list(evidence_by_handle.values()),
        "state": _project_question_state(projection, question),
    }
    payload_text = _fit_appraisal_payload(payload)
    started_at = time.perf_counter()
    raw_output: str | None = None
    parsed_output: object | None = None
    try:
        response = await services.llm.ainvoke(
            [
                SystemMessage(content=SEMANTIC_APPRAISAL_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=services.appraisal_config,
        )
        raw_output = response.content
        parsed_output = parse_llm_json_output(raw_output)
        result = validate_semantic_appraisal_result(
            parsed_output,
            question,
            set(evidence_by_handle),
            projection.handle_to_ref,
        )
    except Exception as exc:
        ended_at = time.perf_counter()
        capture_validation_stage(
            stage_id=f"semantic_appraisal:{question['question_id']}",
            config=services.appraisal_config,
            system_prompt=SEMANTIC_APPRAISAL_PROMPT,
            human_payload=payload_text,
            raw_output=raw_output,
            parsed_output=parsed_output,
            parse_status="failed",
            started_at=started_at,
            ended_at=ended_at,
            error=str(exc),
        )
        raise
    ended_at = time.perf_counter()
    capture_validation_stage(
        stage_id=f"semantic_appraisal:{question['question_id']}",
        config=services.appraisal_config,
        system_prompt=SEMANTIC_APPRAISAL_PROMPT,
        human_payload=payload_text,
        raw_output=raw_output,
        parsed_output=parsed_output,
        parse_status="succeeded",
        started_at=started_at,
        ended_at=ended_at,
    )
    return result


def validate_semantic_appraisal_result(
    parsed: object,
    question: SemanticQuestionV2,
    evidence_handles: set[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> SemanticAppraisalResultV2:
    """Validate one appraisal without interpreting its semantic prose."""

    _validate_question_handle_authority(question, handle_to_ref)
    if not isinstance(parsed, Mapping):
        raise ValueError("semantic appraisal must return an object")
    required = {
        "question_id",
        "selected_evidence_handles",
        "selected_role_handles",
        "propositions",
        "deltas",
        "explanation",
    }
    if set(parsed) != required:
        raise ValueError("semantic appraisal fields are not exact")
    if parsed["question_id"] != question["question_id"]:
        raise ValueError("semantic appraisal question id does not match")
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
    if not isinstance(parsed["propositions"], list) or len(parsed["propositions"]) > 8:
        raise ValueError("semantic propositions are invalid")
    propositions = [
        _validate_proposition(
            row,
            question,
            selected_evidence_set,
            handle_to_ref,
        )
        for row in parsed["propositions"]
    ]
    if not isinstance(parsed["deltas"], list) or len(parsed["deltas"]) > 8:
        raise ValueError("semantic deltas are invalid")
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
    if not isinstance(explanation, str) or not 1 <= len(explanation) <= 1000:
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
            raise ValueError("semantic role handle is not permitted")
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
        raise ValueError("semantic delta path is not owned by question")
    delta = value["delta"]
    if (
        isinstance(delta, bool)
        or not isinstance(delta, int)
        or not -40 <= delta <= 40
    ):
        raise ValueError("semantic delta must be an integer in range")
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
        raise ValueError(f"{label} handles are invalid")
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
                "causal candidate must cite its originating evidence"
            )


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
    """Drop only supplemental projected context in reverse order before failing."""

    supplemental_order = (
        "causal_candidates",
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
        raise CognitionContextLimitError(
            "semantic appraisal state projection is invalid"
        )
    projected_state = dict(state)
    while True:
        candidate = dict(payload)
        candidate["state"] = projected_state
        payload_text = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if len(payload_text) <= 8000:
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
        if not removed:
            raise CognitionContextLimitError(
                "required semantic appraisal context exceeds the contract cap"
            )


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
    candidates = [
        dict(row)
        for row in source.get("causal_candidates", [])
        if isinstance(row, Mapping)
        and row.get("handle") in allowed
        and row.get("evidence_handle") in set(question["evidence_handles"])
    ]
    if candidates:
        result["causal_candidates"] = candidates
    evidence = [
        dict(row)
        for row in source.get("evidence", [])
        if isinstance(row, Mapping)
        and row.get("handle") in set(question["evidence_handles"])
    ]
    if evidence:
        result["evidence"] = evidence
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
