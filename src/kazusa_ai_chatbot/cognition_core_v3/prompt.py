"""Canonical stage-local, model-handleless Cognition V3 prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_FAMILY_AXES,
    CANONICAL_SHIFT_VALUES,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    GOAL_RESOLUTION_VALUES,
    SELF_COGNITION_RESPONSE_DECISION_VALUES,
    project_evidence_provenance_role,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    RELATIONSHIP_AXIS_FIELDS,
    project_affect,
    project_numeric_band,
    project_relationship_axis,
)

A1_QUESTION_GUIDANCE = '''
按固定位置返回 A1 的三个评估类别：事件与行为归因、目标与威胁结果、认知比较或
记忆。以开放的语义判断和具体原因作为主要内容；`axis_changes` 只是可选的从属
证据。`current_observation` 和 `direct_facts` 可以用于确认当下发生了什么。
`continuation_state` 只提供仍在起作用的因果压力。
当前用户明确纠正自己的意思时，把这项纠正当作当前观察；纠正本身不是相反意思的证据。
只有新的当下证据支持不确定或不同判断时，才保留相应的不确定性。
'''
A2_QUESTION_GUIDANCE = '''
按固定位置返回 A2 的三个评估类别：关系与社会判断、道德身份、存在性驱力。以开放
的语义判断和具体原因作为主要内容；`axis_changes` 只是可选的从属证据。
`participant_continuity` 只描述此前参与者、行为及结果。
`conditional_character_context` 可以影响角色判断和边界，但不能用于确认当前事实、
同意、承诺、许可、能力或当前用户的意图。
'''
APPRAISAL_QUESTION_GUIDANCE = '''
返回一个 JSON 对象，并且严格包含本阶段要求的固定评估类别位置。每个位置都要保留
开放的 `semantic_summary` 和具体的 `cause_summary`。`axis_changes` 只是可选的
从属证据，只能使用列出的 `axis` 和一个 `shift` 值。当前角色始终是判断主体。
'''
GOAL_QUESTION_GUIDANCE = '''
严格返回一个由当前角色拥有的 `active_character_goal`、一项
`relational_willingness` 记录，以及一段简洁的第一人称 `private_monologue`。
内心独白要连接此刻的感受、具体原因和眼前动机。它可以影响表达方式，但不能用于
确认事实、许可、能力、目标对象或状态变化。即使请求仍不明确，也要让角色目标具有
实际意义；澄清、守住边界、暂缓判断或有依据地保持沉默，都是有效目标。
先确定当前观察新增加、改变、纠正、询问或仍未解决的内容，再选择一个对这项当前
语义增量有贡献的主要目标。已表达的回应模式只是背景连续性；只有当前用户继续、深化、实质改变或重新打开同一事项时，才把它作为当前目标。当前用户明确纠正自己的意思时，把
这项纠正当作当前观察；纠正本身不是相反意思的证据。
继续处理同一任务或话题，本身不会继续或重新打开角色此前使用或提出的回应方式、提议、要求、条件或关系性回报。
角色尚未得到回应的提议只能作为参与者连续性，不能当作当前用户的意图、接受、承诺或必须追求的当前目标。
只有当前用户回应、接受、拒绝、提及、询问、实质改变或明确重新打开该回应事项时，才可以再次选择它。
角色倾向可以影响语气和立场，但不能取代当前语义增量成为主要目标。
'''
ORDINARY_PLAN_GUIDANCE = '''
返回一份由当前角色拥有的回应计划。`response_goal` 描述可见对话的意图；
`action_requests` 和 `resolver_requests` 只能使用输入中提供的语义能力。
`epistemic_boundary` 必须说明可见措辞可以断言什么、哪些内容只能作为解释，以及
哪些内容仍然未知。每一项未经观察的功能、原因、来源、意图或结果，都必须留在解释
层，并在可见措辞中明确表达不确定性。缺少证据或没有观察到某项特征，都不能据此
作出否定断言，也不能排除任何可能。不得虚构能力或私有引用。严格只返回
`goal_resolution`、`response_goal`、`action_requests`、`resolver_requests` 和
`epistemic_boundary`。不得把输入中的权威通道名称复制到输出对象中。
先让回应目标回答当前观察新增加、改变、纠正、询问或仍未解决的内容，再决定如何
让角色的性格和关系语境影响表达。
已表达的回应模式只属于背景连续性；只有当前用户继续、深化、实质改变或重新打开同一事项时，才允许重新选择该模式。
当前用户明确纠正自己的意思时，把这项纠正当作当前观察；纠正本身不是相反意思的证据。
继续处理同一任务或话题，本身不会继续或重新打开角色此前使用或提出的回应方式、提议、要求、条件或关系性回报。
角色尚未得到回应的提议只能作为参与者连续性，不能当作当前用户的意图、接受、承诺或必须追求的当前目标。
只有当前用户回应、接受、拒绝、提及、询问、实质改变或明确重新打开该回应事项时，才可以再次选择它。
角色倾向可以影响语气和立场，但不能取代当前语义增量成为主要目标。
'''
SELF_PLAN_GUIDANCE = '''
返回独立的自我认知回应契约。根据输入中有依据的参与语境，判断角色应保持沉默还是
提出一项可见回复，并为任何可见措辞明确断言边界。
'''

_PRIVATE_SUFFIXES = (
    "_id", "_ids", "_handle", "_handles", "_ref", "_refs", "_path", "_paths",
)
_ALLOWED_CONTEXT_FIELDS = frozenset({
    "name", "role", "description", "summary", "standard", "boundary",
    "policy", "value", "meaning", "status", "lifecycle", "semantic_summary",
})
_ALLOWED_SCENE_FIELDS = frozenset({
    "operation", "character_role", "current_user_role", "public_group_scene",
    "local_time_context", "semantic_temporal_context", "scene_summary",
})
_IDENTITY_PARTITIONS = frozenset({
    "event_agency", "goal_threat_outcome", "epistemic_comparison_memory",
    "relationship_social", "moral_identity", "existential_drive",
    "goal_cognition",
})
_A1_QUESTIONS = frozenset({
    "q:event_agency", "q:goal_threat_outcome",
    "q:epistemic_comparison_memory",
})
_A2_QUESTIONS = frozenset({
    "q:relationship_social", "q:moral_identity", "q:existential_drive",
})
_ALL_QUESTIONS = frozenset(
    f"q:{family}" for family in _IDENTITY_PARTITIONS
    if family != "goal_cognition"
)
_TERMINAL_STATUSES = frozenset({
    "satisfied", "failed", "abandoned", "resolved", "replaced",
})


class PromptContractError(ValueError):
    """Raised when caller-owned semantic prompt input is malformed."""


def _safe_text(value: object, *, field: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise PromptContractError(f"{field} must be bounded text")
    return value


def _semantic_number(value: object, *, field: str, key: str) -> object:
    """Convert native telemetry into a bounded semantic band."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    number = float(value)
    if key in {"age", "year"}:
        return int(number)
    if 0 <= number <= 1:
        if number <= 0.2:
            return "极低"
        if number <= 0.4:
            return "低"
        if number <= 0.6:
            return "中等"
        if number <= 0.8:
            return "高"
        return "极高"
    if -100 <= number <= 100:
        try:
            return project_numeric_band(int(round(number)), signed=number < 0)
        except ValueError:
            pass
    raise PromptContractError(f"{field} numeric value is outside its semantic domain")


def _project_semantic_tree(value: object, *, field: str, depth: int = 0) -> object:
    """Project typed semantic context while excluding storage identity."""

    if depth > 5:
        return {}
    if isinstance(value, str):
        return _safe_text(value, field=field, maximum=3000)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            if (
                key in {"id", "ids", "entity_id", "source_id", "trace_id"}
                or any(key.endswith(suffix) for suffix in _PRIVATE_SUFFIXES)
            ):
                continue
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                result[key] = _semantic_number(
                    child,
                    field=f"{field}.{key}",
                    key=key,
                )
            else:
                result[key] = _project_semantic_tree(
                    child,
                    field=f"{field}.{key}",
                    depth=depth + 1,
                )
        return result
    if isinstance(value, list):
        return [
            _project_semantic_tree(item, field=field, depth=depth + 1)
            for item in value[:32]
        ]
    return str(value)


def _semantic_mapping(
    value: Mapping[str, object],
    *,
    allowed: frozenset[str],
    field: str,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, item in value.items():
        if key not in allowed or any(key.endswith(suffix) for suffix in _PRIVATE_SUFFIXES):
            continue
        if isinstance(item, str):
            output[key] = _safe_text(item, field=f"{field}.{key}")
        elif isinstance(item, (int, float, bool)) or item is None:
            output[key] = item
        elif isinstance(item, Mapping):
            output[key] = _semantic_mapping(item, allowed=allowed, field=f"{field}.{key}")
        elif isinstance(item, list):
            output[key] = [
                _semantic_mapping(row, allowed=allowed, field=f"{field}.{key}")
                if isinstance(row, Mapping) else row
                for row in item[:32]
            ]
    return output


def _role(value: object) -> str:
    if value in {CURRENT_CHARACTER_ROLE, "self", "active_character"}:
        return "active_character"
    if value in {CURRENT_USER_ROLE, "current_user"}:
        return "current_user"
    if isinstance(value, str) and value:
        return "named_participant"
    return "unknown_participant"


def _project_percepts(episode: Mapping[str, object]) -> list[dict[str, object]]:
    visible = project_model_visible_percepts(episode)
    rows: list[dict[str, object]] = []
    for percept in visible:
        if not isinstance(percept, Mapping):
            continue
        row: dict[str, object] = {
            "input_source": percept.get("input_source", "observation"),
            "percept_kind": percept.get("percept_kind", "observation"),
        }
        if isinstance(percept.get("semantic_text"), str):
            row["semantic_text"] = _safe_text(percept["semantic_text"], field="percept")
        elif isinstance(percept.get("text"), str):
            row["semantic_text"] = _safe_text(percept["text"], field="percept")
        for source, target in (
            ("speaker_role", "speaker_role"),
            ("addressee_role", "addressee_role"),
            ("first_person_role", "first_person_role"),
            ("implicit_imperative_subject_role", "implicit_imperative_subject_role"),
        ):
            if source in percept:
                row[target] = _role(percept[source])
        if isinstance(percept.get("participants"), list):
            row["participants"] = [
                _semantic_mapping(item, allowed=_ALLOWED_CONTEXT_FIELDS, field="participant")
                for item in percept["participants"] if isinstance(item, Mapping)
            ]
        rows.append(row)
    return rows


def _project_evidence(
    evidence: Sequence[Mapping[str, object]],
    *,
    allowed_questions: frozenset[str] | None = None,
    limit: int = 32,
) -> list[dict[str, object]]:
    rows: list[tuple[int, int, dict[str, object]]] = []
    for row in evidence:
        if not isinstance(row, Mapping):
            raise PromptContractError("evidence rows must be mappings")
        if row.get("visibility") not in {None, "model_visible"}:
            continue
        visible_to = row.get("visible_to")
        if allowed_questions is not None and isinstance(visible_to, list):
            if not allowed_questions.intersection(
                item for item in visible_to if isinstance(item, str)
            ):
                continue
        reference = row.get("evidence_ref")
        if not isinstance(reference, Mapping):
            raise PromptContractError("evidence provenance must be typed")
        text = row.get("semantic_text")
        if not isinstance(text, str) or not text.strip():
            raise PromptContractError("evidence semantic_text is required")
        source_kind = str(reference.get("source_kind", "unknown"))
        item = {
            "semantic_text": _safe_text(text, field="evidence", maximum=4000),
            "authority": str(row.get("authority", "supporting")),
            "source_kind": source_kind,
            "provenance_role": project_evidence_provenance_role(
                source_kind, row.get("memory_scope")
            ),
        }
        authority = item["authority"]
        if authority in {"current_event", "current_episode"} or source_kind in {
            "action_result", "resolver_observation", "tool_result",
        }:
            priority = 0
        elif source_kind in {"media_observation", "scheduler_event"}:
            priority = 1
        elif source_kind in {"conversation_progress", "promoted_memory", "promoted_reflection"}:
            priority = 2
        else:
            priority = 3
        rows.append((priority, len(rows), item))
    rows.sort(key=lambda row: (row[0], row[1]))
    return [item for _priority, _position, item in rows[:limit]]


def _project_identity_context(value: Mapping[str, object]) -> dict[str, object]:
    """Keep visible identity and personality descriptors for model judgment."""

    if _IDENTITY_PARTITIONS.intersection(value):
        return {
            key: _project_semantic_tree(value[key], field=f"identity.{key}")
            for key in value
            if key in _IDENTITY_PARTITIONS and isinstance(value[key], Mapping)
        }
    output: dict[str, object] = {}
    for key in ("name", "display_name", "personality", "personality_judgment", "identity", "summary"):
        item = value.get(key)
        if isinstance(item, str):
            output[key] = _safe_text(item, field=f"identity.{key}", maximum=3000)
        elif isinstance(item, Mapping):
            output[key] = _semantic_mapping(
                item,
                allowed=_ALLOWED_CONTEXT_FIELDS,
                field=f"identity.{key}",
            )
    return output


def _project_constraints(value: Mapping[str, object]) -> dict[str, object]:
    """Project standards, drives, meaning, and boundaries without IDs."""

    output: dict[str, object] = {}
    for key in ("standards", "drives", "meaning_state", "boundaries", "personality_judgment"):
        item = value.get(key)
        if isinstance(item, Mapping):
            output[key] = _project_semantic_tree(
                item,
                field=f"constraints.{key}",
            )
        elif key == "standards" and isinstance(item, list):
            standards: list[dict[str, object]] = []
            for row in item[:8]:
                if not isinstance(row, Mapping):
                    continue
                projected: dict[str, object] = {}
                if isinstance(row.get("description"), str):
                    projected["description"] = _safe_text(
                        row["description"],
                        field="constraints.standards.description",
                    )
                importance = row.get("importance")
                if isinstance(importance, int) and not isinstance(importance, bool):
                    projected["importance"] = project_numeric_band(importance)
                if projected:
                    standards.append(projected)
            output[key] = standards
        elif isinstance(item, list):
            output[key] = [
                _semantic_mapping(row, allowed=_ALLOWED_CONTEXT_FIELDS, field=f"constraints.{key}")
                for row in item[:32]
                if isinstance(row, Mapping)
            ]
        elif isinstance(item, str):
            output[key] = _safe_text(item, field=f"constraints.{key}")
    return output


def _project_relationship_context(value: Mapping[str, object]) -> dict[str, object]:
    """Project relationship axes and concrete evidence meaning."""

    raw_axes = value.get("axes")
    axis_source = raw_axes if isinstance(raw_axes, Mapping) else value
    output: dict[str, object] = {
        "axes": {
            axis: project_relationship_axis(axis, axis_source[axis])
            for axis in RELATIONSHIP_AXIS_FIELDS
            if isinstance(axis_source.get(axis), (int, float))
            and not isinstance(axis_source.get(axis), bool)
        },
    }
    for key in ("status", "summary", "semantic_summary", "cause_summary"):
        if isinstance(value.get(key), str):
            output[key] = _safe_text(value[key], field=f"relationship.{key}")
    evidence = value.get("evidence_refs")
    if isinstance(evidence, list):
        output["evidence_meaning"] = [
            _safe_text(row["semantic_summary"], field="relationship.evidence")
            for row in evidence[:8]
            if isinstance(row, Mapping) and isinstance(row.get("semantic_summary"), str)
        ]
    causal_context = value.get("causal_context")
    if isinstance(causal_context, list):
        output["causal_meaning"] = [
            _project_semantic_tree(
                {
                    key: row[key]
                    for key in ("entity_kind", "semantic_summary", "lifecycle", "salience")
                    if key in row
                },
                field="relationship.causal_meaning",
            )
            for row in causal_context[:8]
            if isinstance(row, Mapping) and isinstance(
                row.get("cause_summary") or row.get("semantic_summary"),
                str,
            )
        ]
    relationship_affect = value.get("affect")
    if isinstance(relationship_affect, list):
        output["affect"] = [
            _project_semantic_tree(row, field="relationship.affect")
            for row in relationship_affect[:8]
            if isinstance(row, Mapping)
        ]
    return output


def _project_affect_context(state: Mapping[str, object]) -> list[dict[str, object]]:
    """Project emotion identity and concrete cause without root references."""

    activation_rows = state.get("affect_activations")
    structured_activations = isinstance(activation_rows, list) and bool(
        activation_rows
    )
    value = activation_rows if structured_activations else state.get("affect", [])
    if not isinstance(value, list):
        return []
    if (
        structured_activations
        and all(
            isinstance(row, Mapping)
            and {"primary_root", "cause_status", "emotion_id", "score", "trend"}
            <= set(row)
            for row in value
        )
    ):
        projected = project_affect(value, state)
        result = []
        for activation, row in zip(value[:32], projected[:32]):
            if not isinstance(activation, Mapping):
                continue
            item = dict(row)
            item["cause_status"] = activation.get("cause_status", "active")
            primary_root = activation.get("primary_root")
            if isinstance(primary_root, Mapping) and primary_root.get("kind") == "relationship":
                relationship = state.get("relationship")
                if isinstance(relationship, Mapping):
                    for evidence in reversed(relationship.get("evidence_refs", [])):
                        if isinstance(evidence, Mapping) and isinstance(
                            evidence.get("semantic_summary"),
                            str,
                        ):
                            item["cause_summary"] = evidence["semantic_summary"]
                            break
            result.append(item)
        return result
    result: list[dict[str, object]] = []
    for row in value[:32]:
        if not isinstance(row, Mapping):
            continue
        item: dict[str, object] = {}
        emotion = row.get("emotion") or row.get("emotion_id")
        if isinstance(emotion, str):
            item["emotion"] = _safe_text(emotion, field="affect.emotion", maximum=120)
        for key in ("phase", "intensity", "score", "trend", "cause_status"):
            if isinstance(row.get(key), (str, int, float, bool)):
                item[key] = row[key]
        for key in ("cause_summary", "cause"):
            if isinstance(row.get(key), str) and row[key].strip():
                item["cause_summary"] = _safe_text(
                    row[key], field="affect.cause_summary", maximum=1000
                )
                break
        if item:
            result.append(item)
    return result


def _project_entities(
    state: Mapping[str, object],
    *,
    include_terminal: bool = True,
    collections: Sequence[str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    selected_collections = collections or (
        "goals", "threats", "active_events", "knowledge_gaps"
    )
    for collection in selected_collections:
        values = state.get(collection, [])
        if not isinstance(values, list):
            continue
        projected: list[dict[str, object]] = []
        for row in values[:32]:
            if not isinstance(row, Mapping):
                continue
            if not include_terminal and row.get("status") in _TERMINAL_STATUSES:
                continue
            item: dict[str, object] = {}
            for key in (
                "description", "semantic_summary", "cause_summary", "status",
                "goal_kind", "intent", "reason", "residual_pressure", "salience",
            ):
                value = row.get(key)
                if key in {"residual_pressure", "salience"}:
                    if isinstance(value, int) and not isinstance(value, bool):
                        item[key] = project_numeric_band(value)
                    continue
                if isinstance(value, (str, int, float, bool)):
                    item[key] = value
            if item:
                projected.append(item)
        result[collection] = projected
    result["affect_activations"] = _project_affect_context(state)
    relationship = state.get("relationship")
    if isinstance(relationship, Mapping):
        result["relationship"] = _project_relationship_context(relationship)
    return result


def _project_capabilities(
    actions: Sequence[Mapping[str, object]],
    resolvers: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "actions": [
            {
                "action_kind": _safe_text(
                    row["action_kind"], field="action.action_kind", maximum=120
                ),
                "description": _safe_text(
                    row.get("capability") or row.get("description") or "",
                    field="action.description",
                    maximum=1000,
                ),
                "decision_mode": _safe_text(
                    row.get("decision_mode") or "optional",
                    field="action.decision_mode",
                    maximum=40,
                ),
                "allowed_decisions": [
                    _safe_text(item, field="action.allowed_decisions", maximum=120)
                    for item in row.get("allowed_decisions", [])
                    if isinstance(item, str)
                ],
                "default_decision": _safe_text(
                    row.get("default_decision") or "",
                    field="action.default_decision",
                    maximum=120,
                ),
                "decision_pattern": _safe_text(
                    row.get("decision_pattern") or "",
                    field="action.decision_pattern",
                    maximum=200,
                ),
            }
            for row in actions if isinstance(row, Mapping)
        ],
        "resolvers": [
            {
                "capability": _safe_text(
                    row["capability"], field="resolver.capability", maximum=120
                ),
                "description": _safe_text(
                    row.get("semantic_capability")
                    or row.get("description")
                    or "",
                    field="resolver.description",
                    maximum=1200,
                ),
            }
            for row in resolvers if isinstance(row, Mapping)
        ],
    }


def _project_operational_context(value: Mapping[str, object]) -> dict[str, object]:
    """Preserve bounded affect and pressure context without source identity."""

    result: dict[str, object] = {}
    for key in ("affect", "pressures"):
        rows = value.get(key)
        if isinstance(rows, list):
            result[key] = [
                _project_semantic_tree(row, field=f"operational.{key}")
                for row in rows[:16]
                if isinstance(row, Mapping)
            ]
    return result


def _project_continuity(value: Mapping[str, object]) -> dict[str, object]:
    """Project private/dialog continuity as bounded semantic context."""

    return {
        key: _safe_text(value[key], field=f"continuity.{key}", maximum=2000)
        for key in ("private", "dialog")
        if isinstance(value.get(key), str) and value[key].strip()
    }


def _project_resolver_context(
    resolver_context: object,
    resolver_progress: Mapping[str, object] | None,
) -> dict[str, object]:
    """Keep resolver observations and progress semantic and bounded."""

    result = {
        "context": _safe_text(
            resolver_context if isinstance(resolver_context, str) else "",
            field="resolver.context",
            maximum=8000,
        ),
    }
    if resolver_progress:
        result["progress"] = _project_semantic_tree(
            resolver_progress,
            field="resolver.progress",
        )
    return result


def _stage_character_context(
    workspace: Mapping[str, object],
    *,
    identity_families: Sequence[str],
    include_constraints: bool,
    include_operational: bool = False,
) -> dict[str, object]:
    """Select only the identity/constraint context owned by one stage."""

    source = workspace["character_context"]
    if not isinstance(source, Mapping):
        return {}
    identity = source.get("identity", {})
    if isinstance(identity, Mapping) and _IDENTITY_PARTITIONS.intersection(identity):
        projected_identity = {
            family: identity[family]
            for family in identity_families
            if family in identity
        }
    else:
        projected_identity = dict(identity) if isinstance(identity, Mapping) else {}
    result: dict[str, object] = {"identity": projected_identity}
    if include_constraints:
        result["constraints"] = source.get("constraints", {})
    if include_operational:
        result["operational"] = source.get("operational", {})
    return result


def _partition_evidence_authority(
    evidence: Sequence[Mapping[str, object]],
    *,
    allowed_questions: frozenset[str],
) -> dict[str, list[dict[str, object]]]:
    """Partition one bounded evidence roster by semantic authority."""

    lanes = {
        "current_observation": [],
        "direct_facts": [],
        "participant_continuity": [],
        "conditional_character_context": [],
    }
    projected = _project_evidence(
        evidence,
        allowed_questions=allowed_questions,
        limit=32,
    )
    for row in projected:
        authority = row.get("authority")
        source_kind = row.get("source_kind")
        if authority == "conditional_character_guidance":
            lane = "conditional_character_context"
        elif authority == "participant_continuity":
            lane = "participant_continuity"
        elif (
            authority in {"current_event", "current_episode"}
            and source_kind in {
                "episode",
                "media_observation",
                "scheduler_event",
            }
        ):
            lane = "current_observation"
        else:
            lane = "direct_facts"
        lanes[lane].append(row)
    return lanes


def _stage_authority_lanes(
    workspace: Mapping[str, object],
    *,
    allowed_questions: frozenset[str],
) -> dict[str, object]:
    """Build the five explicit model-facing authority lanes."""

    observation = workspace["observation"]
    if not isinstance(observation, Mapping):
        raise PromptContractError("workspace observation must be a mapping")
    raw_evidence = workspace.get("evidence_rows", [])
    if not isinstance(raw_evidence, list):
        raise PromptContractError("workspace evidence rows must be a list")
    partitioned = _partition_evidence_authority(
        raw_evidence,
        allowed_questions=allowed_questions,
    )
    current_observation = dict(observation)
    current_observation["evidence"] = partitioned["current_observation"]
    direct_facts = {
        "evidence": partitioned["direct_facts"],
        "typed_facts": list(workspace.get("direct_facts", [])),
    }
    participant_continuity = list(
        partitioned["participant_continuity"]
    )
    continuity = workspace.get("continuity")
    if isinstance(continuity, Mapping):
        dialog_context = continuity.get("dialog")
        if isinstance(dialog_context, str) and dialog_context.strip():
            participant_continuity.append({
                "semantic_text": dialog_context,
                "authority": "participant_continuity",
                "source_kind": "past_dialog_context",
                "provenance_role": "prior_dialog_context",
            })
    return {
        "current_observation": current_observation,
        "direct_facts": direct_facts,
        "participant_continuity": participant_continuity,
        "conditional_evidence": partitioned[
            "conditional_character_context"
        ],
    }


def _overused_move_rows(
    workspace: Mapping[str, object],
) -> list[dict[str, str]]:
    """Project observed response moves as participant continuity evidence."""

    moves = workspace["overused_moves"]
    if not isinstance(moves, list):
        raise PromptContractError("workspace overused moves must be a list")
    return [
        {
            "semantic_text": move,
            "authority": "participant_continuity",
            "source_kind": "conversation_progress_overused_move",
            "provenance_role": "observed_response_move",
        }
        for move in moves
    ]


def _continuation_state(
    workspace: Mapping[str, object],
    *,
    include_goals: bool,
) -> dict[str, object]:
    """Project unresolved goals and active causal pressures only."""

    state = workspace.get("state")
    if not isinstance(state, Mapping):
        return {}
    fields = ["threats", "active_events", "knowledge_gaps"]
    if include_goals:
        fields.insert(0, "goals")
    result = {
        field: list(state.get(field, []))
        for field in fields
        if isinstance(state.get(field), list)
    }
    affect = state.get("affect_activations")
    if isinstance(affect, list):
        result["affect_activations"] = list(affect)
    return result


def _conditional_character_context(
    workspace: Mapping[str, object],
    *,
    identity_families: Sequence[str],
    conditional_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Project character context with explicit non-factual authority."""

    result = _stage_character_context(
        workspace,
        identity_families=identity_families,
        include_constraints=True,
        include_operational=True,
    )
    result["relationship"] = workspace.get("relationship_context", {})
    result["affect"] = workspace.get("affect_context", [])
    result["evidence"] = [dict(row) for row in conditional_evidence]
    continuity = workspace.get("continuity")
    if isinstance(continuity, Mapping):
        private_context = continuity.get("private")
        if isinstance(private_context, str) and private_context.strip():
            result["private_continuity"] = private_context
    return result


def build_canonical_turn_workspace(
    *,
    episode: Mapping[str, object],
    scene_context: Mapping[str, object],
    evidence: Sequence[Mapping[str, object]],
    mutable_state: Mapping[str, object],
    character_constraints: Mapping[str, object] | None = None,
    identity_context: Mapping[str, object] | None = None,
    continuity: Mapping[str, object] | None = None,
    available_actions: Sequence[Mapping[str, object]] = (),
    available_resolvers: Sequence[Mapping[str, object]] = (),
    overused_moves: Sequence[str],
    direct_facts: Sequence[Mapping[str, object]] = (),
    character_operational_context: Mapping[str, object] | None = None,
    character_affect_context: Sequence[Mapping[str, object]] | None = None,
    relationship_context: Mapping[str, object] | None = None,
    resolver_context: object = None,
    resolver_progress: Mapping[str, object] | None = None,
    runtime_limits: Sequence[Mapping[str, object]] = (),
    group_engagement: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(episode, Mapping) or not isinstance(scene_context, Mapping):
        raise PromptContractError("episode and scene_context must be mappings")
    visible = _project_percepts(episode)
    evidence_rows = _project_evidence(evidence, limit=128)
    role_bindings = [
        {
            key: row[key]
            for key in (
                "speaker_role", "addressee_role", "first_person_role",
                "implicit_imperative_subject_role",
            )
            if key in row
        }
        for row in visible if row.get("input_source") == "dialog"
    ]
    for row in evidence_rows:
        if row["authority"] in {"current_event", "current_episode"} and role_bindings:
            row["dialogue_role_binding"] = dict(role_bindings[0])
    orientation = {
        "response_owner": _role(scene_context.get("character_role", "active_character")),
        "selection_owner": _role(scene_context.get("character_role", "active_character")),
        "current_user": _role(scene_context.get("current_user_role", "current_user")),
        "operation": _safe_text(
            str(scene_context.get("operation", "回应当前观察")),
            field="orientation.operation",
            maximum=500,
        ),
    }
    character_context = {
        "constraints": _project_constraints(character_constraints or {}),
        "identity": _project_identity_context(identity_context or {}),
        "operational": _project_operational_context(
            character_operational_context or {}
        ),
    }
    observation = {
        "visible_observation": visible,
        "dialogue_role_bindings": role_bindings,
        "scene": {
            key: scene_context[key]
            for key in _ALLOWED_SCENE_FIELDS
            if key in scene_context and isinstance(scene_context[key], (str, int, float, bool))
        },
        "group_engagement": _project_semantic_tree(
            group_engagement or scene_context.get("group_engagement_action_context", {}),
            field="group_engagement",
        ),
    }
    state = _project_entities(mutable_state, include_terminal=False)
    affect_context = _project_affect_context(mutable_state)
    if character_affect_context:
        affect_context.extend(
            _project_affect_context({"affect": list(character_affect_context)})
        )
    return {
        "observation": observation,
        "evidence_rows": [dict(row) for row in evidence],
        "orientation": orientation,
        "state": state,
        "character_context": character_context,
        "relationship_context": _project_relationship_context(
            relationship_context or {}
        ),
        "affect_context": affect_context[:32],
        "direct_facts": [
            _project_semantic_tree(row, field="direct_fact")
            for row in direct_facts if isinstance(row, Mapping)
        ],
        "continuity": _project_continuity(continuity or {}),
        "overused_moves": list(overused_moves),
        "resolver_context": _project_resolver_context(
            resolver_context,
            resolver_progress,
        ),
        "runtime_limits": [
            _project_semantic_tree(row, field="runtime_limit")
            for row in runtime_limits if isinstance(row, Mapping)
        ],
        "capabilities": _project_capabilities(available_actions, available_resolvers),
    }


def _family_contract(families: Sequence[str]) -> dict[str, object]:
    return {
        "required_fields": list(families),
        "additionalProperties": False,
        "family_slots": {
            family: {
                "required_fields": [
                    "applicable", "semantic_summary", "cause_summary", "axis_changes",
                ],
                "additionalProperties": False,
                "axis_names": list(CANONICAL_FAMILY_AXES[family]),
                "shift_values": sorted(CANONICAL_SHIFT_VALUES),
                "axis_change_fields": ["axis", "shift", "reason"],
                "maximum_axis_changes": len(CANONICAL_FAMILY_AXES[family]),
            }
            for family in families
        },
    }


def build_canonical_appraisal_question(
    *,
    workspace: Mapping[str, object],
    stage_name: str,
    accepted_appraisal_summary: object | None = None,
) -> dict[str, object]:
    if stage_name not in {"A1", "A2"}:
        raise PromptContractError("appraisal stage must be A1 or A2")
    families = CANONICAL_A1_FAMILIES if stage_name == "A1" else CANONICAL_A2_FAMILIES
    allowed_questions = (
        _A1_QUESTIONS if stage_name == "A1" else _A2_QUESTIONS
    )
    lanes = _stage_authority_lanes(
        workspace,
        allowed_questions=allowed_questions,
    )
    packet: dict[str, object] = {
        "stage": stage_name,
        "guidance": (
            A1_QUESTION_GUIDANCE
            if stage_name == "A1"
            else A2_QUESTION_GUIDANCE
        ),
        "orientation": workspace["orientation"],
        "current_observation": lanes["current_observation"],
        "direct_facts": lanes["direct_facts"],
        "continuation_state": _continuation_state(
            workspace,
            include_goals=False,
        ),
        "output_contract": _family_contract(families),
    }
    if stage_name == "A2":
        packet["accepted_a1_meaning"] = accepted_appraisal_summary or []
        packet["participant_continuity"] = [
            *lanes["participant_continuity"],
            *_overused_move_rows(workspace),
        ]
        packet["conditional_character_context"] = (
            _conditional_character_context(
                workspace,
                identity_families=tuple(CANONICAL_A2_FAMILIES),
                conditional_evidence=lanes["conditional_evidence"],
            )
        )
        packet["continuation_state"] = _continuation_state(
            workspace,
            include_goals=True,
        )
    return packet


def build_canonical_goal_question(
    *,
    workspace: Mapping[str, object],
    appraisal_summary: object,
) -> dict[str, object]:
    semantic_appraisal = [
        {
            key: row[key]
            for key in ("family", "applicable", "semantic_summary", "cause_summary")
            if key in row
        }
        for row in appraisal_summary
        if isinstance(row, Mapping)
    ]
    lanes = _stage_authority_lanes(
        workspace,
        allowed_questions=_ALL_QUESTIONS,
    )
    return {
        "stage": "G",
        "guidance": GOAL_QUESTION_GUIDANCE,
        "orientation": workspace["orientation"],
        "current_observation": lanes["current_observation"],
        "direct_facts": lanes["direct_facts"],
        "participant_continuity": [
            *lanes["participant_continuity"],
            *_overused_move_rows(workspace),
        ],
        "conditional_character_context": _conditional_character_context(
            workspace,
            identity_families=("goal_cognition",),
            conditional_evidence=lanes["conditional_evidence"],
        ),
        "continuation_state": _continuation_state(
            workspace,
            include_goals=True,
        ),
        "appraisal_summary": semantic_appraisal,
        "output_contract": {
            "required_fields": [
                "active_character_goal",
                "relational_willingness",
                "private_monologue",
            ],
            "additionalProperties": False,
            "active_character_goal_fields": [
                "goal_kind", "intent", "reason", "cause_summary",
            ],
            "relational_willingness_fields": [
                "applicable", "stance", "reason", "cause_summary",
            ],
            "private_monologue": {
                "type": "string",
                "minimum_characters": 1,
                "maximum_characters": 600,
            },
        },
    }


def build_canonical_plan_question(
    *,
    workspace: Mapping[str, object],
    goal: Mapping[str, object],
    appraisal_summary: object,
    self_cognition: bool = False,
) -> dict[str, object]:
    if self_cognition:
        contract = {
            "required_fields": [
                "self_cognition_response",
                "epistemic_boundary",
            ],
            "additionalProperties": False,
            "self_cognition_fields": [
                "decision", "response_goal", "reason", "cause_summary",
            ],
            "allowed_decisions": sorted(SELF_COGNITION_RESPONSE_DECISION_VALUES),
            "epistemic_boundary": {
                "type": "string",
                "minimum_characters": 1,
                "maximum_characters": 1000,
            },
        }
        guidance = SELF_PLAN_GUIDANCE
    else:
        contract = {
            "required_fields": [
                "goal_resolution", "response_goal", "action_requests", "resolver_requests",
                "epistemic_boundary",
            ],
            "additionalProperties": False,
            "goal_resolution_values": sorted(GOAL_RESOLUTION_VALUES),
            "action_request_fields": ["action_kind", "decision", "detail", "reason"],
            "action_request_item_bounds": {"minimum": 0, "maximum": 3},
            "response_goal_action_reservation": 1,
            "maximum_action_requests_with_response_goal": 2,
            "resolver_request_fields": ["capability", "goal", "reason"],
            "resolver_request_item_bounds": {"minimum": 0, "maximum": 8},
            "epistemic_boundary": {
                "type": "string",
                "minimum_characters": 1,
                "maximum_characters": 1000,
            },
        }
        guidance = ORDINARY_PLAN_GUIDANCE
    lanes = _stage_authority_lanes(
        workspace,
        allowed_questions=_ALL_QUESTIONS,
    )
    return {
        "stage": "P",
        "guidance": guidance,
        "goal": goal,
        "current_observation": lanes["current_observation"],
        "direct_facts": lanes["direct_facts"],
        "participant_continuity": [
            *lanes["participant_continuity"],
            *_overused_move_rows(workspace),
        ],
        "continuation_state": _continuation_state(
            workspace,
            include_goals=True,
        ),
        "capabilities": workspace["capabilities"],
        "output_contract": contract,
    }


def build_turn_workspace_stage_contracts(
    *,
    workspace: Mapping[str, object],
    appraisal_summary: object = None,
    goal: Mapping[str, object] | None = None,
    self_cognition: bool = False,
) -> dict[str, dict[str, object]]:
    selected_goal = goal or {
        "goal_kind": "open_goal",
        "intent": "理解当前请求",
        "reason": "当前观察需要一个有依据的回应",
        "cause_summary": "当前观察",
    }
    return {
        "A1": build_canonical_appraisal_question(workspace=workspace, stage_name="A1"),
        "A2": build_canonical_appraisal_question(
            workspace=workspace,
            stage_name="A2",
            accepted_appraisal_summary=appraisal_summary or [],
        ),
        "G": build_canonical_goal_question(
            workspace=workspace, appraisal_summary=appraisal_summary or []
        ),
        "P": build_canonical_plan_question(
            workspace=workspace,
            goal=selected_goal,
            appraisal_summary=appraisal_summary or [],
            self_cognition=self_cognition,
        ),
    }


def semantic_role_summary(
    role_name: str,
    reference: Mapping[str, object],
    *,
    scene_context: Mapping[str, object],
) -> str:
    """Render a visible participant description for an immediate surface owner."""

    for binding in scene_context.get("participant_bindings", []):
        if isinstance(binding, Mapping) and binding.get("handle") == role_name:
            display_name = binding.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                return f"{role_name}={display_name.strip()}（群聊其他参与者）"
    return f"{role_name}=named participant"


__all__ = [
    "A1_QUESTION_GUIDANCE",
    "A2_QUESTION_GUIDANCE",
    "APPRAISAL_QUESTION_GUIDANCE",
    "PromptContractError",
    "build_canonical_appraisal_question",
    "build_canonical_goal_question",
    "build_canonical_plan_question",
    "build_canonical_turn_workspace",
    "build_turn_workspace_stage_contracts",
    "semantic_role_summary",
]
