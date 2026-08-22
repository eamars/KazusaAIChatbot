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

A1_QUESTION_GUIDANCE = '''
Return fixed A1 family slots for event agency, goal and threat outcome, and
epistemic comparison or memory. Keep semantic meaning and concrete causes
primary; axis changes are optional subordinate evidence.
'''
A2_QUESTION_GUIDANCE = '''
Return fixed A2 family slots for relationship and social judgment, moral
identity, and existential drive. Keep semantic meaning and concrete causes
primary; axis changes are optional subordinate evidence.
'''
APPRAISAL_QUESTION_GUIDANCE = '''
Return one JSON object with exactly the fixed family slots requested by this
stage. Each slot keeps an open semantic_summary and concrete cause_summary.
Axis changes are optional subordinate evidence and use only the listed axis and
one shift value. The active character remains the subject of judgment.
'''
GOAL_QUESTION_GUIDANCE = '''
Return exactly one active_character_goal owned by the active character and one
relational_willingness record. Keep the goal meaningful even when the request
is uncertain: clarify, preserve a boundary, defer judgment, or grounded
silence are valid goals.
'''
ORDINARY_PLAN_GUIDANCE = '''
Return a response plan owned by the active character. response_goal describes
visible dialog intent; action_requests and resolver_requests use only supplied
semantic capabilities. Do not invent capabilities or private references.
'''
SELF_PLAN_GUIDANCE = '''
Return the separate self-cognition response contract. Decide whether the
character should stay silent or propose a visible reply from the supplied
grounded participation context.
'''

_PRIVATE_SUFFIXES = (
    "_id", "_ids", "_handle", "_handles", "_ref", "_refs", "_path", "_paths",
)
_ALLOWED_ENTITY_FIELDS = frozenset({
    "description", "semantic_summary", "cause_summary", "status", "lifecycle",
    "state", "kind", "goal_kind", "intent", "reason", "residual_pressure",
    "salience", "source_kind", "evidence", "evidence_refs", "axis_values",
})
_ALLOWED_CONTEXT_FIELDS = frozenset({
    "name", "role", "description", "summary", "standard", "boundary",
    "policy", "value", "meaning", "status", "lifecycle", "semantic_summary",
})
_ALLOWED_SCENE_FIELDS = frozenset({
    "operation", "character_role", "current_user_role", "public_group_scene",
    "local_time_context", "semantic_temporal_context", "scene_summary",
})


class PromptContractError(ValueError):
    """Raised when caller-owned semantic prompt input is malformed."""


def _safe_text(value: object, *, field: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise PromptContractError(f"{field} must be bounded text")
    return value


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


def _project_evidence(evidence: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in evidence:
        if not isinstance(row, Mapping):
            raise PromptContractError("evidence rows must be mappings")
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
        rows.append(item)
    return rows


def _project_entities(state: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for collection in ("goals", "threats", "active_events", "knowledge_gaps"):
        values = state.get(collection, [])
        if not isinstance(values, list):
            continue
        result[collection] = [
            _semantic_mapping(row, allowed=_ALLOWED_ENTITY_FIELDS, field=collection)
            for row in values[:32] if isinstance(row, Mapping)
        ]
    affect = state.get("affect_activations", [])
    if isinstance(affect, list):
        result["affect_activations"] = [
            _semantic_mapping(row, allowed=frozenset({
                "emotion", "intensity", "score", "trend", "cause_summary",
                "cause_status", "primary_root", "root_refs",
            }), field="affect")
            for row in affect[:32] if isinstance(row, Mapping)
        ]
    relationship = state.get("relationship")
    if isinstance(relationship, Mapping):
        result["relationship"] = {
            key: value for key, value in relationship.items()
            if key in {"axes", "summary", "status"} and not any(
                key.endswith(suffix) for suffix in _PRIVATE_SUFFIXES
            )
        }
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
    direct_facts: Sequence[Mapping[str, object]] = (),
    character_operational_context: Mapping[str, object] | None = None,
    relationship_context: Mapping[str, object] | None = None,
    resolver_context: object = None,
    resolver_progress: Mapping[str, object] | None = None,
    runtime_limits: Sequence[Mapping[str, object]] = (),
    group_engagement: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(episode, Mapping) or not isinstance(scene_context, Mapping):
        raise PromptContractError("episode and scene_context must be mappings")
    visible = _project_percepts(episode)
    evidence_rows = _project_evidence(evidence)
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
            str(scene_context.get("operation", "respond to current observation")),
            field="orientation.operation",
            maximum=500,
        ),
    }
    character_context = {
        "constraints": _semantic_mapping(
            character_constraints or {}, allowed=_ALLOWED_CONTEXT_FIELDS, field="constraints"
        ),
        "identity": _semantic_mapping(
            identity_context or {}, allowed=_ALLOWED_CONTEXT_FIELDS, field="identity"
        ),
        "operational": _semantic_mapping(
            character_operational_context or {},
            allowed=_ALLOWED_CONTEXT_FIELDS,
            field="operational",
        ),
    }
    observation = {
        "visible_observation": visible,
        "evidence": evidence_rows,
        "dialogue_role_bindings": role_bindings,
        "scene": {
            key: scene_context[key]
            for key in _ALLOWED_SCENE_FIELDS
            if key in scene_context and isinstance(scene_context[key], (str, int, float, bool))
        },
        "group_engagement": _semantic_mapping(
            group_engagement or scene_context.get("group_engagement_action_context", {}),
            allowed=_ALLOWED_CONTEXT_FIELDS,
            field="group_engagement",
        ),
    }
    state = _project_entities(mutable_state)
    return {
        "observation": observation,
        "orientation": orientation,
        "state": state,
        "character_context": character_context,
        "relationship_context": _semantic_mapping(
            relationship_context or {}, allowed=_ALLOWED_CONTEXT_FIELDS, field="relationship"
        ),
        "direct_facts": [
            _semantic_mapping(row, allowed=_ALLOWED_CONTEXT_FIELDS, field="direct_fact")
            for row in direct_facts if isinstance(row, Mapping)
        ],
        "continuity": _semantic_mapping(
            continuity or {}, allowed=_ALLOWED_CONTEXT_FIELDS, field="continuity"
        ),
        "resolver_context": {
            "context": resolver_context if isinstance(resolver_context, str) else "",
            "progress": _semantic_mapping(
                resolver_progress or {}, allowed=_ALLOWED_CONTEXT_FIELDS, field="resolver_progress"
            ),
        },
        "runtime_limits": [
            _semantic_mapping(row, allowed=_ALLOWED_CONTEXT_FIELDS, field="runtime_limit")
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
    context: dict[str, object] = {}
    if stage_name == "A2":
        context["accepted_a1_meaning"] = accepted_appraisal_summary or []
        context["affect_context"] = workspace["state"].get("affect_activations", [])
        context["character_context"] = workspace["character_context"]
        context["relationship_context"] = workspace["relationship_context"]
    return {
        "stage": stage_name,
        "guidance": A1_QUESTION_GUIDANCE if stage_name == "A1" else A2_QUESTION_GUIDANCE,
        "orientation": workspace["orientation"],
        "observation": workspace["observation"],
        "context": context,
        "output_contract": _family_contract(families),
    }


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
    return {
        "stage": "G",
        "guidance": GOAL_QUESTION_GUIDANCE,
        "orientation": workspace["orientation"],
        "observation": workspace["observation"],
        "character_context": workspace["character_context"],
        "relationship_context": workspace["relationship_context"],
        "continuity": workspace["continuity"],
        "state": workspace["state"],
        "appraisal_summary": semantic_appraisal,
        "output_contract": {
            "required_fields": ["active_character_goal", "relational_willingness"],
            "additionalProperties": False,
            "active_character_goal_fields": [
                "goal_kind", "intent", "reason", "cause_summary",
            ],
            "relational_willingness_fields": [
                "applicable", "stance", "reason", "cause_summary",
            ],
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
            "required_fields": ["self_cognition_response"],
            "additionalProperties": False,
            "self_cognition_fields": [
                "decision", "response_goal", "reason", "cause_summary",
            ],
            "allowed_decisions": sorted(SELF_COGNITION_RESPONSE_DECISION_VALUES),
        }
        guidance = SELF_PLAN_GUIDANCE
    else:
        contract = {
            "required_fields": [
                "goal_resolution", "response_goal", "action_requests", "resolver_requests",
            ],
            "additionalProperties": False,
            "goal_resolution_values": sorted(GOAL_RESOLUTION_VALUES),
            "action_request_fields": ["action_kind", "decision", "detail", "reason"],
            "action_request_item_bounds": {"minimum": 0, "maximum": 3},
            "response_goal_action_reservation": 1,
            "maximum_action_requests_with_response_goal": 2,
            "resolver_request_fields": ["capability", "goal", "reason"],
            "resolver_request_item_bounds": {"minimum": 0, "maximum": 8},
        }
        guidance = ORDINARY_PLAN_GUIDANCE
    return {
        "stage": "P",
        "guidance": guidance,
        "orientation": workspace["orientation"],
        "goal": goal,
        "capabilities": workspace["capabilities"],
        "resolver_context": workspace["resolver_context"],
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
        "intent": "understand the current request",
        "reason": "the current observation requires a grounded response",
        "cause_summary": "current observation",
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
