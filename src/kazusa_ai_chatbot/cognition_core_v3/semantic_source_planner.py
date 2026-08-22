"""Deterministic six-family source planning for cognition appraisal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.cognition_shared.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    ROLE_ENTITY_KINDS,
    SEMANTIC_QUESTION_KINDS,
    CognitionEvidenceV2,
    SemanticQuestionV2,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    ENTITY_LIST_FIELDS,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    evidence_source_identity,
)

MAX_SEMANTIC_QUESTIONS_PER_EPISODE = 6
MAX_EVIDENCE_HANDLES_PER_QUESTION = 8

_GOAL_OUTCOME_ELIGIBLE_STATUSES = {
    "goal": frozenset({"pursuing", "blocked"}),
    "threat": frozenset({"active"}),
    "event": frozenset({"active"}),
    "knowledge_gap": frozenset({"open", "reduced"}),
}

QUESTION_KINDS = SEMANTIC_QUESTION_KINDS

_QUESTION_PROPOSITION_KINDS = {
    "event_agency": ("responsibility", "intentionality"),
    "relationship_social": ("social_meaning", "relationship_threat"),
    "moral_identity": ("norm_meaning",),
    "goal_threat_outcome": (
        "goal_release",
        "goal_supersession",
        "goal_completed",
        "event_completed",
        "threat_resolved",
        "event_repaired",
        "knowledge_answered",
        "outcome_pending",
    ),
    "epistemic_comparison_memory": (
        "comparison_meaning",
        "memory_cue",
    ),
    "existential_drive": ("meaning_relevance",),
}

_QUESTION_PROPOSITION_KIND_SEMANTICS = {
    "event_agency": {
        "responsibility": "事件主体对结果负有责任",
        "intentionality": "事件主体有意促成该结果",
    },
    "relationship_social": {
        "social_meaning": "该事件具有明确的社交含义",
        "relationship_threat": "该事件对现有关系构成威胁",
    },
    "moral_identity": {
        "norm_meaning": "该事件体现明确的规范或身份含义",
    },
    "goal_threat_outcome": {
        "goal_release": "主体目标已被明确放下",
        "goal_supersession": "主体目标已被另一个进行中的目标取代",
        "goal_completed": "主体目标已经完成",
        "event_completed": "主体事件已经完成",
        "threat_resolved": "主体威胁已经解除",
        "event_repaired": "主体事件所需的修复已经完成",
        "knowledge_answered": "主体知识缺口已经获得答案",
        "outcome_pending": "主体结果仍在进行并等待明确终态",
    },
    "epistemic_comparison_memory": {
        "comparison_meaning": "该事件体现明确的比较含义",
        "memory_cue": "该事件构成明确的记忆线索",
    },
    "existential_drive": {
        "meaning_relevance": "该事件与当前意义或驱动力明确相关",
    },
}

_QUESTION_DESCRIPTIONS = {
    "event_agency": (
        "只使用已经授权的事件证据与角色 handle，判断责任和意图。"
    ),
    "relationship_social": (
        "只判断当前角色与当前用户的 r1 社交含义与关系威胁，同时保持关系归属不变；"
        "第三方互动本身不改变或威胁 r1 时省略。"
    ),
    "moral_identity": (
        "判断已授权 event handle 的规范含义与修复相关性；证据中的第三方没有允许的"
        "人物 handle 时省略其 role assignment，不用 ceN 代替人物。delta axis 只使用 harm、"
        "unfairness、repair_need、norm_violation、identity_threat 或 exposure。"
    ),
    "goal_threat_outcome": (
        "判断现有 handle 是否已经达到目标、事件、威胁或知识缺口的明确终态。"
    ),
    "epistemic_comparison_memory": (
        "判断现有 handle 的比较含义、记忆线索与认知含义。"
    ),
    "existential_drive": (
        "判断意义相关性与驱动力压力，不增添新的目标。"
    ),
}


def plan_semantic_questions(
    evidence: Sequence[CognitionEvidenceV2],
    mutable_state: Mapping[str, Any],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> list[SemanticQuestionV2]:
    """Select at most one scoped question per evidence family.

    The planner uses only typed evidence provenance and mutable-state shape.
    It never searches user-language text for keywords and never chooses a
    route, action, or final response.
    """

    evidence_rows = _select_evidence_rows(evidence)
    if not evidence_rows:
        return []
    selected_kinds = [
        question_kind
        for question_kind in QUESTION_KINDS
        if any(f"q:{question_kind}" in row["visible_to"] for row in evidence_rows)
    ]
    handle_map = _index_projected_handles(handle_to_ref)
    questions: list[SemanticQuestionV2] = []
    for question_kind in selected_kinds[:MAX_SEMANTIC_QUESTIONS_PER_EPISODE]:
        question_id = f"q:{question_kind}"
        question_evidence = [
            row
            for row in evidence_rows
            if question_id in row["visible_to"]
        ][:MAX_EVIDENCE_HANDLES_PER_QUESTION]
        question_evidence_identities = {
            row["evidence_handle"]: evidence_source_identity(
                row["evidence_ref"]
            )
            for row in question_evidence
        }
        permitted_paths = _permitted_delta_paths(
            question_kind,
            handle_map,
            mutable_state,
            [row["evidence_handle"] for row in question_evidence],
            question_evidence_identities,
        )
        permitted_handles = _permitted_role_handles(
            question_kind,
            handle_map,
            mutable_state,
            [row["evidence_handle"] for row in question_evidence],
            question_evidence_identities,
        )
        permitted_assignment_handles = _permitted_role_assignment_handles(
            handle_to_ref,
            permitted_handles,
        )
        question = {
            "question_id": question_id,
            "question_kind": question_kind,
            "semantic_question": _QUESTION_DESCRIPTIONS[question_kind],
            "evidence_handles": [
                row["evidence_handle"] for row in question_evidence
            ],
            "permitted_role_handles": permitted_handles,
            "permitted_role_assignment_handles": (
                permitted_assignment_handles
            ),
            "permitted_delta_paths": permitted_paths,
            "dependencies": [],
        }
        questions.append(question)
    _assert_unique_delta_owners(questions)
    return questions


def question_proposition_kinds(question_kind: str) -> tuple[str, ...]:
    """Return the frozen proposition vocabulary for one question family."""

    try:
        return _QUESTION_PROPOSITION_KINDS[question_kind]
    except KeyError as exc:
        raise ValueError(f"unknown semantic question kind: {question_kind}") from exc


def question_proposition_kind_semantics(
    question_kind: str,
) -> dict[str, str]:
    """Return positive meanings for one question family's proposition kinds."""

    try:
        semantics = _QUESTION_PROPOSITION_KIND_SEMANTICS[question_kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown semantic question kind: {question_kind}"
        ) from exc
    return dict(semantics)


def _select_evidence_rows(
    evidence: Sequence[CognitionEvidenceV2],
) -> list[CognitionEvidenceV2]:
    """Select every valid typed row in source order."""

    selected: list[CognitionEvidenceV2] = []
    seen: set[str] = set()
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("semantic evidence row must be a mapping")
        handle = row["evidence_handle"]
        ref = row["evidence_ref"]
        source_kind = ref["source_kind"]
        if source_kind not in EVIDENCE_SOURCE_QUESTION_IDS:
            raise ValueError("semantic evidence source kind is invalid")
        if handle in seen:
            raise ValueError("semantic evidence handles must be unique")
        seen.add(handle)
        selected.append(row)
    return selected


def _index_projected_handles(
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Index the canonical projection map without deriving new handles."""

    mapping: dict[str, str] = {}
    for handle, ref in handle_to_ref.items():
        entity_id = ref.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("projected handle reference requires an entity id")
        index_key = (
            f"prompt_role:{handle}"
            if handle in {"self", "current_user"}
            else entity_id
        )
        if index_key in mapping:
            raise ValueError("projected entity ids must be unique")
        mapping[index_key] = handle
    return mapping


def _permitted_delta_paths(
    question_kind: str,
    handle_map: Mapping[str, str],
    state: Mapping[str, Any],
    evidence_handles: Sequence[str],
    evidence_identities: Mapping[str, tuple[str, str]],
) -> list[str]:
    """Return exactly the allowlisted target paths owned by one family."""

    paths: list[str] = []
    if question_kind == "event_agency":
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _source_linked_eligible_entities(
                state,
                "event",
                evidence_identities,
            )
            for axis in ("responsibility", "intentionality")
        )
    elif question_kind == "relationship_social":
        relationship = state.get("relationship")
        if isinstance(relationship, Mapping):
            paths.extend(
                f"relationship.r1.{axis}"
                for axis in (
                    "positive_regard",
                    "trust",
                    "attachment",
                    "desired_closeness",
                    "perceived_closeness",
                    "care",
                    "boundary_safety",
                    "exclusivity",
                    "unresolved_injury",
                )
            )
    elif question_kind == "moral_identity":
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _source_linked_eligible_entities(
                state,
                "event",
                evidence_identities,
            )
            for axis in (
                "harm",
                "unfairness",
                "exposure",
                "repair_need",
                "reparability",
                "norm_violation",
                "contamination_risk",
                "identity_threat",
            )
        )
    elif question_kind == "goal_threat_outcome":
        paths.extend(
            f"goals.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _goal_outcome_eligible_entities(
                state,
                "goal",
                evidence_identities,
            )
            for axis in (
                "obstruction",
                "expected_success",
                "controllability",
                "recoverability",
                "urgency",
            )
        )
        paths.extend(
            f"threats.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _goal_outcome_eligible_entities(
                state,
                "threat",
                evidence_identities,
            )
            for axis in (
                "likelihood",
                "expected_harm",
                "uncertainty",
                "controllability",
                "coping_potential",
                "residual_pressure",
            )
        )
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _goal_outcome_eligible_entities(
                state,
                "event",
                evidence_identities,
            )
            for axis in ("outcome_impact", "expectation_mismatch")
        )
    elif question_kind == "epistemic_comparison_memory":
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _source_linked_eligible_entities(
                state,
                "event",
                evidence_identities,
            )
            for axis in (
                "comparison_gap",
                "vastness",
                "memory_warmth",
                "temporal_loss",
            )
        )
        paths.extend(
            f"knowledge_gaps.{handle_map[entity['entity_id']]}.{axis}"
            for entity in _source_linked_eligible_entities(
                state,
                "knowledge_gap",
                evidence_identities,
            )
            for axis in (
                "relevance",
                "uncertainty",
                "learnability",
                "novelty",
                "model_accommodation",
            )
        )
    elif question_kind == "existential_drive":
        paths.extend(
            f"drives.{handle_map[drive_id]}.pressure"
            for drive_id in state.get("drives", {})
        )
        if isinstance(state.get("meaning_state"), Mapping):
            paths.extend(
                f"meaning_state.m1.{axis}"
                for axis in (
                    "purpose_coherence",
                    "agency",
                    "identity_continuity",
                )
            )
    else:
        raise ValueError(f"unknown semantic question kind: {question_kind}")
    paths.extend(
        _candidate_delta_paths(
            question_kind,
            handle_map,
            evidence_handles,
        )
    )
    return sorted(set(paths))


def _candidate_delta_paths(
    question_kind: str,
    handle_map: Mapping[str, str],
    evidence_handles: Sequence[str],
) -> list[str]:
    """Return allowlisted paths for episode-local causal candidates."""

    event_axes: tuple[str, ...] = ()
    threat_axes: tuple[str, ...] = ()
    gap_axes: tuple[str, ...] = ()
    if question_kind == "event_agency":
        event_axes = ("responsibility", "intentionality")
    elif question_kind == "relationship_social":
        threat_axes = ()
    elif question_kind == "moral_identity":
        event_axes = (
            "harm",
            "unfairness",
            "repair_need",
            "norm_violation",
            "identity_threat",
        )
    elif question_kind == "goal_threat_outcome":
        event_axes = ("outcome_impact", "expectation_mismatch")
        threat_axes = (
            "likelihood",
            "expected_harm",
            "uncertainty",
            "residual_pressure",
        )
    elif question_kind == "epistemic_comparison_memory":
        event_axes = ("comparison_gap", "vastness", "memory_warmth", "temporal_loss")
        gap_axes = (
            "relevance",
            "uncertainty",
            "learnability",
            "novelty",
            "model_accommodation",
        )
    elif question_kind == "existential_drive":
        event_axes = ()
    else:
        raise ValueError(f"unknown semantic question kind: {question_kind}")

    paths: list[str] = []
    for entity_id, handle in handle_map.items():
        if entity_id.startswith("candidate:") and not _candidate_is_permitted(
            entity_id,
            evidence_handles,
        ):
            continue
        if handle.startswith("ce"):
            paths.extend(f"active_events.{handle}.{axis}" for axis in event_axes)
        elif handle.startswith("ct"):
            paths.extend(f"threats.{handle}.{axis}" for axis in threat_axes)
        elif handle.startswith("ck"):
            paths.extend(f"knowledge_gaps.{handle}.{axis}" for axis in gap_axes)
    return paths


def _permitted_role_handles(
    question_kind: str,
    handle_map: Mapping[str, str],
    state: Mapping[str, Any],
    evidence_handles: Sequence[str],
    evidence_identities: Mapping[str, tuple[str, str]],
) -> list[str]:
    """Return only entity and role handles owned by one question family."""

    prefixes = {
        "event_agency": ("ev", "ce"),
        "relationship_social": ("r", "ct"),
        "moral_identity": ("ev", "ce"),
        "goal_threat_outcome": ("g", "t", "ev", "k", "ce", "ct", "ck"),
        "epistemic_comparison_memory": ("ev", "k", "ce", "ck"),
        "existential_drive": ("d", "m"),
    }
    try:
        permitted_prefixes = prefixes[question_kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown semantic question kind: {question_kind}"
        ) from exc
    source_filtered_prefixes = {
        "event_agency": ("ev",),
        "relationship_social": (),
        "moral_identity": ("ev",),
        "goal_threat_outcome": ("g", "t", "ev", "k"),
        "epistemic_comparison_memory": ("ev", "k"),
        "existential_drive": (),
    }[question_kind]
    source_linked_native_handles = {
        handle_map[entity["entity_id"]]
        for entity_kind in _question_native_entity_kinds(question_kind)
        for entity in _source_linked_eligible_entities(
            state,
            entity_kind,
            evidence_identities,
        )
    }
    handles: list[str] = []
    for entity_id, handle in handle_map.items():
        if handle in {"self", "current_user"}:
            handles.append(handle)
            continue
        if not handle.startswith(permitted_prefixes):
            continue
        if (
            not entity_id.startswith("candidate:")
            and handle.startswith(source_filtered_prefixes)
            and handle not in source_linked_native_handles
        ):
            continue
        if entity_id.startswith("candidate:") and not _candidate_is_permitted(
            entity_id,
            evidence_handles,
        ):
            continue
        handles.append(handle)
    return sorted(set(handles))


def _permitted_role_assignment_handles(
    handle_to_ref: Mapping[str, Mapping[str, str]],
    permitted_role_handles: Sequence[str],
) -> list[str]:
    """Return the family-local role-assignment handle domain.

    Role assignments accept only role-bearing handles: family-owned handles
    whose canonical reference kind is in ``ROLE_ENTITY_KINDS``, scene
    third-party participant handles, and the explicit ``self`` and
    ``current_user`` bindings. Causal candidates and lifecycle handles stay
    in the subject/object domain and are never exposed as assignment targets.
    """

    assignment_handles = {
        handle
        for handle in permitted_role_handles
        if handle in handle_to_ref
        and handle_to_ref[handle].get("kind") in ROLE_ENTITY_KINDS
    }
    assignment_handles.update(
        handle
        for handle, ref in handle_to_ref.items()
        if ref.get("kind") == "third_party"
    )
    assignment_handles.update(
        handle
        for handle in ("self", "current_user")
        if handle in handle_to_ref
    )
    return sorted(assignment_handles)


def _goal_outcome_eligible_entities(
    state: Mapping[str, Any],
    entity_kind: str,
    evidence_identities: Mapping[str, tuple[str, str]] | None = None,
) -> list[Mapping[str, Any]]:
    """Return native entities whose lifecycle permits an outcome appraisal."""

    field_name = ENTITY_LIST_FIELDS[entity_kind]
    eligible_statuses = _GOAL_OUTCOME_ELIGIBLE_STATUSES[entity_kind]
    entities = [
        entity
        for entity in state[field_name]
        if entity["status"] in eligible_statuses
    ]
    if evidence_identities is not None:
        entities = [
            entity
            for entity in entities
            if _entity_has_source_identity(entity, evidence_identities)
        ]
    return entities


def _question_native_entity_kinds(question_kind: str) -> tuple[str, ...]:
    """Return causal native kinds owned by one semantic question family."""

    native_kinds = {
        "event_agency": ("event",),
        "relationship_social": (),
        "moral_identity": ("event",),
        "goal_threat_outcome": tuple(_GOAL_OUTCOME_ELIGIBLE_STATUSES),
        "epistemic_comparison_memory": ("event", "knowledge_gap"),
        "existential_drive": (),
    }
    try:
        selected_kinds = native_kinds[question_kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown semantic question kind: {question_kind}"
        ) from exc
    return selected_kinds


def _source_linked_eligible_entities(
    state: Mapping[str, Any],
    entity_kind: str,
    evidence_identities: Mapping[str, tuple[str, str]],
) -> list[Mapping[str, Any]]:
    """Return lifecycle-eligible native rows linked to current evidence."""

    entities = _goal_outcome_eligible_entities(
        state,
        entity_kind,
        evidence_identities,
    )
    return entities


def _entity_has_source_identity(
    entity: Mapping[str, Any],
    evidence_identities: Mapping[str, tuple[str, str]],
) -> bool:
    """Return whether an entity cites one current question provenance row."""

    current_identities = set(evidence_identities.values())
    matches = {
        evidence_source_identity(evidence_ref)
        for evidence_ref in entity.get("evidence_refs", [])
    }
    return bool(matches & current_identities)


def _candidate_is_permitted(
    entity_id: str,
    evidence_handles: Sequence[str],
) -> bool:
    """Match a candidate to one exact authorized evidence handle."""

    return entity_id in {
        f"candidate:{kind}:{evidence_handle}"
        for kind in ("event", "threat", "knowledge_gap")
        for evidence_handle in evidence_handles
    }


def _assert_unique_delta_owners(
    questions: Sequence[SemanticQuestionV2],
) -> None:
    """Reject any path claimed by more than one selected question."""

    owners: dict[str, str] = {}
    for question in questions:
        for path in question["permitted_delta_paths"]:
            previous = owners.setdefault(path, question["question_id"])
            if previous != question["question_id"]:
                raise ValueError(f"semantic delta path has duplicate owners: {path}")


def entity_list_field_for_prompt_kind(kind: str) -> str:
    """Return the persistent list field for a canonical prompt kind."""

    try:
        return ENTITY_LIST_FIELDS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt entity kind: {kind}") from exc
