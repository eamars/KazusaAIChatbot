"""Deterministic six-family source planning for cognition appraisal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionEvidenceV2,
    SemanticQuestionV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    ENTITY_LIST_FIELDS,
    EVIDENCE_SOURCE_KINDS,
)


MAX_SEMANTIC_QUESTIONS_PER_EPISODE = 6
MAX_EVIDENCE_HANDLES_PER_QUESTION = 8

QUESTION_KINDS = (
    "event_agency",
    "relationship_social",
    "moral_identity",
    "goal_threat_outcome",
    "epistemic_comparison_memory",
    "existential_drive",
)

_QUESTION_PROPOSITION_KINDS = {
    "event_agency": ("responsibility", "intentionality"),
    "relationship_social": ("social_meaning", "relationship_threat"),
    "moral_identity": ("norm_meaning",),
    "goal_threat_outcome": (
        "goal_release",
        "goal_supersession",
        "completion_meaning",
        "resolution_meaning",
    ),
    "epistemic_comparison_memory": (
        "comparison_meaning",
        "memory_cue",
    ),
    "existential_drive": ("meaning_relevance",),
}

_SOURCE_TO_QUESTIONS = {
    "episode": QUESTION_KINDS,
    "promoted_memory": QUESTION_KINDS,
    "promoted_reflection": QUESTION_KINDS,
    "media_observation": QUESTION_KINDS,
    "action_result": (
        "event_agency",
        "relationship_social",
        "moral_identity",
        "goal_threat_outcome",
    ),
    "resolver_observation": (
        "event_agency",
        "relationship_social",
        "moral_identity",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
    ),
    "accepted_task_result": (
        "event_agency",
        "relationship_social",
        "moral_identity",
        "goal_threat_outcome",
        "epistemic_comparison_memory",
    ),
    "scheduler_event": ("goal_threat_outcome",),
}

_QUESTION_DESCRIPTIONS = {
    "event_agency": (
        "Assess responsibility and intentionality using only the authorized "
        "event evidence and role handles."
    ),
    "relationship_social": (
        "Assess social meaning and relationship threat without changing "
        "relationship ownership."
    ),
    "moral_identity": (
        "Assess norm meaning and repair relevance for authorized event handles."
    ),
    "goal_threat_outcome": (
        "Assess goal release, supersession, completion, and resolution meaning "
        "for existing handles."
    ),
    "epistemic_comparison_memory": (
        "Assess comparison, memory cues, and epistemic meaning for existing "
        "handles."
    ),
    "existential_drive": (
        "Assess meaning relevance and drive pressure without inventing a goal."
    ),
}


def plan_semantic_questions(
    evidence: Sequence[CognitionEvidenceV2],
    mutable_state: Mapping[str, Any],
    character_constraints: Mapping[str, Any],
    relationship_context: Mapping[str, Any] | None = None,
) -> list[SemanticQuestionV2]:
    """Select at most one scoped question per evidence family.

    The planner uses only typed evidence provenance and mutable-state shape.
    It never searches user-language text for keywords and never chooses a
    route, action, or final response.
    """

    del character_constraints
    evidence_rows = _select_evidence_rows(evidence)
    if not evidence_rows:
        return []
    source_kinds = {
        row["evidence_ref"]["source_kind"]
        for row in evidence_rows
    }
    selected_kinds = [
        question_kind
        for question_kind in QUESTION_KINDS
        if any(
            question_kind in _SOURCE_TO_QUESTIONS[source_kind]
            for source_kind in source_kinds
        )
    ]
    handle_map = _build_prompt_handles(
        mutable_state,
        relationship_context,
        evidence_rows,
    )
    questions: list[SemanticQuestionV2] = []
    for question_kind in selected_kinds[:MAX_SEMANTIC_QUESTIONS_PER_EPISODE]:
        permitted_paths = _permitted_delta_paths(
            question_kind,
            handle_map,
            mutable_state,
        )
        question = {
            "question_id": f"q:{question_kind}",
            "question_kind": question_kind,
            "semantic_question": _QUESTION_DESCRIPTIONS[question_kind],
            "evidence_handles": [
                row["evidence_handle"] for row in evidence_rows
            ],
            "permitted_role_handles": sorted(handle_map.values()),
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


def _select_evidence_rows(
    evidence: Sequence[CognitionEvidenceV2],
) -> list[CognitionEvidenceV2]:
    """Select the first eight valid typed rows in source order."""

    selected: list[CognitionEvidenceV2] = []
    seen: set[str] = set()
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("semantic evidence row must be a mapping")
        handle = row["evidence_handle"]
        ref = row["evidence_ref"]
        source_kind = ref["source_kind"]
        if source_kind not in EVIDENCE_SOURCE_KINDS:
            raise ValueError("semantic evidence source kind is invalid")
        if handle in seen:
            raise ValueError("semantic evidence handles must be unique")
        seen.add(handle)
        if len(selected) < MAX_EVIDENCE_HANDLES_PER_QUESTION:
            selected.append(row)
    return selected


def _build_prompt_handles(
    state: Mapping[str, Any],
    relationship_context: Mapping[str, Any] | None,
    evidence: Sequence[CognitionEvidenceV2],
) -> dict[str, str]:
    """Build a private persistent-id to prompt-handle mapping."""

    mapping: dict[str, str] = {}
    prefixes = {
        "goals": "g",
        "threats": "t",
        "active_events": "e",
        "knowledge_gaps": "k",
    }
    for field_name, prefix in prefixes.items():
        entities = state[field_name]
        for index, entity in enumerate(entities, start=1):
            mapping[entity["entity_id"]] = f"{prefix}{index}"
    relationship = state.get("relationship")
    if relationship is None and relationship_context is not None:
        relationship = relationship_context
    if isinstance(relationship, Mapping):
        mapping[relationship["relationship_id"]] = "r1"
    for index, drive_id in enumerate(state.get("drives", {}), start=1):
        mapping[drive_id] = f"d{index}"
    for index, standard in enumerate(state.get("standards", []), start=1):
        mapping[standard["standard_id"]] = f"s{index}"
    if isinstance(state.get("meaning_state"), Mapping):
        mapping["meaning:character"] = "m1"
    for index, row in enumerate(evidence, start=1):
        evidence_handle = row["evidence_handle"]
        mapping[f"candidate:event:{evidence_handle}"] = f"ce{index}"
        mapping[f"candidate:threat:{evidence_handle}"] = f"ct{index}"
        mapping[f"candidate:knowledge_gap:{evidence_handle}"] = f"ck{index}"
    return mapping


def _permitted_delta_paths(
    question_kind: str,
    handle_map: Mapping[str, str],
    state: Mapping[str, Any],
) -> list[str]:
    """Return exactly the allowlisted target paths owned by one family."""

    paths: list[str] = []
    if question_kind == "event_agency":
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in state["active_events"]
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
            for entity in state["active_events"]
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
            for entity in state["goals"]
            for axis in (
                "importance",
                "progress",
                "obstruction",
                "expected_success",
                "controllability",
                "recoverability",
                "urgency",
                "salience",
            )
        )
        paths.extend(
            f"threats.{handle_map[entity['entity_id']]}.{axis}"
            for entity in state["threats"]
            for axis in (
                "likelihood",
                "expected_harm",
                "uncertainty",
                "controllability",
                "coping_potential",
                "residual_pressure",
                "salience",
            )
        )
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in state["active_events"]
            for axis in ("outcome_impact", "expectation_mismatch")
        )
    elif question_kind == "epistemic_comparison_memory":
        paths.extend(
            f"active_events.{handle_map[entity['entity_id']]}.{axis}"
            for entity in state["active_events"]
            for axis in (
                "comparison_gap",
                "vastness",
                "memory_warmth",
                "temporal_loss",
            )
        )
        paths.extend(
            f"knowledge_gaps.{handle_map[entity['entity_id']]}.{axis}"
            for entity in state["knowledge_gaps"]
            for axis in (
                "relevance",
                "uncertainty",
                "learnability",
                "novelty",
                "model_accommodation",
                "salience",
            )
        )
    elif question_kind == "existential_drive":
        paths.extend(
            f"drives.{handle_map[drive_id]}.pressure"
            for drive_id in state.get("drives", {})
        )
        paths.extend(
            f"meaning_state.m1.{axis}"
            for axis in (
                "purpose_coherence",
                "agency",
                "identity_continuity",
                "salience",
            )
        )
    else:
        raise ValueError(f"unknown semantic question kind: {question_kind}")
    paths.extend(_candidate_delta_paths(question_kind, handle_map))
    return sorted(set(paths))


def _candidate_delta_paths(
    question_kind: str,
    handle_map: Mapping[str, str],
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
        threat_axes = ("likelihood", "expected_harm", "uncertainty", "residual_pressure")
    elif question_kind == "epistemic_comparison_memory":
        event_axes = ("comparison_gap", "vastness", "memory_warmth", "temporal_loss")
        gap_axes = (
            "relevance",
            "uncertainty",
            "learnability",
            "novelty",
            "model_accommodation",
            "salience",
        )
    elif question_kind == "existential_drive":
        event_axes = ()
    else:
        raise ValueError(f"unknown semantic question kind: {question_kind}")

    paths: list[str] = []
    for handle in handle_map.values():
        if handle.startswith("ce"):
            paths.extend(f"active_events.{handle}.{axis}" for axis in event_axes)
        elif handle.startswith("ct"):
            paths.extend(f"threats.{handle}.{axis}" for axis in threat_axes)
        elif handle.startswith("ck"):
            paths.extend(f"knowledge_gaps.{handle}.{axis}" for axis in gap_axes)
    return paths


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
