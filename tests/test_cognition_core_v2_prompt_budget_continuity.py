"""Production-derived prompt budgeting and degraded-continuity contracts."""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2 import (
    action_authorization as action_authorization_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    action_selection as action_selection_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v2 import (
    goal_cognition as goal_cognition_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    resolver_authorization as resolver_authorization_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    semantic_appraisal as semantic_appraisal_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    state_models as state_models_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    state_reducers as state_reducers_module,
)
from kazusa_ai_chatbot.cognition_core_v2 import surface as surface_module
from kazusa_ai_chatbot.cognition_core_v2 import workspace as workspace_module
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    CognitionContextLimitError,
    CognitionExecutionError,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    PromptBudgetError,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    QUESTION_KINDS,
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    _progress_age_descriptor,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
    canonical_identity_context,
    maximum_identity_context,
)

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "cognition_core_v2_prompt_budget_production_case.json"
)
INCIDENT_TIMESTAMP = "2026-07-28T04:19:18Z"
_CAPTURED_NEAR_CAP_CASES: tuple[dict[str, object], ...] = (
    {
        "case_id": "a1a573_near_cap_boundary_termination",
        "trace_id": "llmtrace_93482f08e4a74aa5af90adc6e6f5918a",
        "trace_path": (
            Path(__file__).parents[1]
            / "test_artifacts"
            / "diagnostics"
            / "cognition_trace_a1a573b590a3494786c4edebdee55342.json"
        ),
        "stage_name": "semantic_appraisal.q:goal_threat_outcome.item_1",
        "question_id": "q:goal_threat_outcome",
    },
    {
        "case_id": "caad1a_near_cap_boundary_termination",
        "trace_id": "llmtrace_caad1a9370cf4d859e8ea6233f1e473d",
        "trace_path": (
            Path(__file__).parents[1]
            / "test_artifacts"
            / "diagnostics"
            / (
                "postdraft_goal_bid_failure_llmtrace_"
                "caad1a9370cf4d859e8ea6233f1e473d.json"
            )
        ),
        "stage_name": "semantic_appraisal.q:goal_threat_outcome.item_1",
        "question_id": "q:goal_threat_outcome",
    },
    {
        "case_id": "df6eb4_near_cap_boundary_termination",
        "trace_id": "llmtrace_df6eb45b1bfc405fa0e781baa7ce8d76",
        "trace_path": (
            Path(__file__).parents[1]
            / "test_artifacts"
            / "diagnostics"
            / (
                "postdraft_goal_bid_failure_llmtrace_"
                "df6eb45b1bfc405fa0e781baa7ce8d76.json"
            )
        ),
        "stage_name": "semantic_appraisal.q:goal_threat_outcome.item_1",
        "question_id": "q:goal_threat_outcome",
    },
)


class _BoundaryReached(AssertionError):
    """Mark that a deterministic prompt reached the configured model owner."""


class _BoundaryProbeLLM:
    """Capture model requests and stop before any generated semantics."""

    def __init__(self) -> None:
        """Initialize an empty request ledger."""

        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        """Capture the exact request and stop the stage at its model boundary."""

        self.calls.append({
            "config": config,
            "system_prompt": str(getattr(messages[0], "content", "")),
            "human_payload": str(getattr(messages[-1], "content", "")),
        })
        raise _BoundaryReached("model boundary reached")


class _NoCallLLM:
    """Fail a test if a fail-contained preflight schedules model work."""

    def __init__(self) -> None:
        """Initialize the call counter."""

        self.call_count = 0

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        """Record and reject an unexpected model invocation."""

        del messages, config
        self.call_count += 1
        raise AssertionError("preflight fallback scheduled an LLM call")


class _ValidGoalLLM:
    """Capture one goal prompt and return one valid grounded draft."""

    def __init__(self) -> None:
        """Initialize an empty prompt ledger."""

        self.human_payloads: list[str] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        """Return a valid goal bid after recording the fitted prompt."""

        del config
        self.human_payloads.append(
            str(getattr(messages[-1], "content", ""))
        )
        result = {
            "intention": "answer the current request",
            "desired_outcome": "provide one grounded response",
            "concrete_detail": "use the retained evidence registry",
            "reason": "the current evidence supports a response",
            "private_monologue": "I should answer from the grounded context.",
            "target_role_handles": [],
            "evidence_handles": ["e1"],
            "expected_consequences": ["the current request receives an answer"],
            "confidence": "high",
            "relational_willingness": {
                "schema_version": "relational_willingness.v2",
                "applicability": "not_relationship_sensitive",
                "stance": "not_applicable",
                "current_user_relationship_state": "not_applicable",
                "reason": '当前回合证据不涉及关系立场判断',
                "evidence_handles": ["e1"],
            },
        }
        response = SimpleNamespace(content=json.dumps(result))
        return response


class _InvalidCandidateLLM:
    """Return one invalid object while retaining each attempted request."""

    def __init__(self, candidate_content: str = "{}") -> None:
        """Initialize an empty model-request ledger and invalid response."""

        self.calls: list[list[str]] = []
        self.candidate_content = candidate_content

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        """Capture one request and return a structurally invalid candidate."""

        del config
        self.calls.append([
            str(getattr(message, "content", ""))
            for message in messages
        ])
        return SimpleNamespace(content=self.candidate_content)


class _InvalidThenValidSurfaceLLM:
    """Fail the first surface attempt and accept the fitted repair packet."""

    def __init__(self) -> None:
        """Initialize the request ledger."""

        self.calls: list[list[str]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        """Return one invalid candidate followed by a valid visual result."""

        del config
        self.calls.append([
            str(getattr(message, "content", ""))
            for message in messages
        ])
        if len(self.calls) == 1:
            return SimpleNamespace(content="{}")
        return SimpleNamespace(content=json.dumps({
            "visual_directives": "valid visual directive",
        }))


def _fixture_document() -> dict[str, Any]:
    """Load the sanitized production-derived incident fixture."""

    document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("prompt budget fixture must contain an object")
    return document


def _production_evidence() -> list[dict[str, Any]]:
    """Build the exact eight-row focused evidence packet in source order."""

    rows = _fixture_document()["evidence_rows"]
    evidence: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows, start=1):
        source_kind = source_row["source_kind"]
        semantic_text = source_row["semantic_text"]
        row: dict[str, Any] = {
            "evidence_handle": f"e{index}",
            "evidence_ref": {
                "source_kind": source_kind,
                "source_id": source_row["source_id"],
                "occurred_at": INCIDENT_TIMESTAMP,
                "semantic_summary": semantic_text[:500],
            },
            "semantic_text": semantic_text,
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS[source_kind]
            ),
        }
        if source_kind == "episode":
            row["authority"] = "current_event"
        elif source_kind == "promoted_memory":
            row["authority"] = "character_world_context"
            row["memory_scope"] = "shared_character_or_world"
        elif source_kind == "conversation_evidence":
            row["authority"] = "participant_continuity"
        else:
            raise AssertionError(
                f"unsupported prompt-budget evidence source: {source_kind}"
            )
        evidence.append(row)
    return evidence


def _character_constraints() -> dict[str, Any]:
    """Build the production-shape fixed character constraint snapshot."""

    state = build_character_production_state(
        updated_at=INCIDENT_TIMESTAMP,
    )
    constraints = {
        "drives": deepcopy(state["drives"]),
        "standards": deepcopy(state["standards"]),
        "meaning_state": deepcopy(state["meaning_state"]),
        "personality_judgment": {
            "logic": "evidence-led intuitive judgment",
            "defense": "convert pressure into immediate grounded movement",
            "quirks": "lightly playful confidence",
            "taboos": "preserve character agency",
        },
    }
    return constraints


def _production_appraisal_context() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Any,
    list[dict[str, Any]],
]:
    """Rebuild the deterministic state, projection, and question prefix."""

    document = _fixture_document()
    state = build_acquaintance_user_state(
        global_user_id="production-case-user",
        updated_at=INCIDENT_TIMESTAMP,
    )
    state["knowledge_gaps"] = deepcopy(document["knowledge_gaps"])
    evidence = _production_evidence()
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        character_identity_context=canonical_identity_context(),
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    return state, evidence, projection, questions


def _appraisal_services(llm: object) -> SimpleNamespace:
    """Build stage configs for every semantic appraisal family."""

    services = SimpleNamespace(
        llm=llm,
        appraisal_event_agency_config=object(),
        appraisal_relationship_social_config=object(),
        appraisal_moral_identity_config=object(),
        appraisal_goal_threat_outcome_config=object(),
        appraisal_epistemic_comparison_memory_config=object(),
        appraisal_existential_drive_config=object(),
    )
    return services


def _load_captured_near_cap_case(
    case: Mapping[str, object],
) -> tuple[dict[str, Any], str]:
    """Load one preserved near-cap input and its invalid candidate."""

    trace_path = case.get("trace_path")
    trace_id = case.get("trace_id")
    stage_name = case.get("stage_name")
    if not isinstance(trace_path, Path):
        raise AssertionError("captured near-cap trace path is invalid")
    if not isinstance(trace_id, str) or not isinstance(stage_name, str):
        raise AssertionError("captured near-cap trace identity is invalid")
    if not trace_path.exists():
        pytest.skip(
            "protected near-cap diagnostic capture is unavailable: "
            f"{trace_path}"
        )
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    capsules = [
        capsule
        for capsule in trace.get("cognition_failure_capsules", [])
        if (
            isinstance(capsule, Mapping)
            and capsule.get("trace_id") == trace_id
        )
    ]
    if len(capsules) != 1:
        raise AssertionError(
            "captured near-cap trace must contain one matching capsule"
        )
    input_payload = capsules[0].get("input_payload")
    if not isinstance(input_payload, dict):
        raise AssertionError("captured near-cap input is not an object")
    attempts = [
        attempt
        for attempt in capsules[0].get("attempts", [])
        if (
            isinstance(attempt, Mapping)
            and attempt.get("stage_name") == stage_name
        )
    ]
    if len(attempts) != 1:
        raise AssertionError(
            "captured near-cap trace must contain one matching attempt"
        )
    attempt = attempts[0]
    validation_error = str(attempt.get("validation_error") or "")
    raw_response_text = str(attempt.get("raw_response_text") or "")
    if "semantic delta path" not in validation_error:
        raise AssertionError("captured near-cap error is not a path error")
    if "; permitted paths:" not in validation_error:
        raise AssertionError("captured near-cap error lost its path domain")
    if not raw_response_text:
        raise AssertionError("captured near-cap response is empty")
    return input_payload, raw_response_text


def _captured_near_cap_appraisal_context(
    input_payload: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    list[Mapping[str, Any]],
]:
    """Rebuild deterministic pre-appraisal context from a captured input."""

    payload = validate_cognition_core_input(
        _canonicalize_captured_evidence_fixture(input_payload)
    )
    previous_state = state_models_module.validate_cognition_state(
        payload["mutable_state"]
    )
    updated_at = facade_module._episode_updated_at(payload["episode"])
    elapsed_seconds = facade_module._cognition_elapsed_seconds(
        previous_state,
        updated_at,
    )
    fact_pairs = [
        (fact["producer"], facade_module._fact_without_producer(fact))
        for fact in payload["direct_facts"]
    ]
    relationship_context = facade_module._native_relationship_context(
        payload.get("relationship_context")
    )
    preliminary_state = state_reducers_module.apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
    )
    preliminary_state = state_reducers_module.create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=relationship_context,
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = state_models_module.validate_cognition_state(
        preliminary_state
    )
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context"
        ),
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    return payload, preliminary_state, projection, questions


def _canonicalize_captured_evidence_fixture(
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the current evidence contract to protected legacy captures."""

    replay_input = deepcopy(dict(input_payload))
    evidence = replay_input.get("evidence")
    if not isinstance(evidence, list):
        raise AssertionError("captured evidence fixture is not a list")
    current_event_occurred_at = next(
        evidence_row["evidence_ref"]["occurred_at"]
        for evidence_row in evidence
        if evidence_row["evidence_ref"]["source_kind"] == "episode"
    )
    authority_by_source_kind = {
        "episode": "current_event",
        "media_observation": "current_event",
        "action_result": "current_event",
        "scheduler_event": "current_event",
        "tool_result": "current_event",
        "conversation_evidence": "participant_continuity",
        "recall_evidence": "contextual_fact_only",
        "resolver_observation": "contextual_fact_only",
    }
    for evidence_row in evidence:
        if not isinstance(evidence_row, dict):
            raise AssertionError("captured evidence row is not a mapping")
        evidence_ref = evidence_row.get("evidence_ref")
        if not isinstance(evidence_ref, dict):
            raise AssertionError("captured evidence reference is not a mapping")
        source_kind = evidence_ref.get("source_kind")
        source_id = evidence_ref.get("source_id")
        if not isinstance(source_kind, str) or not isinstance(source_id, str):
            raise AssertionError("captured evidence source metadata is invalid")
        if source_kind == "promoted_memory":
            memory_scope = evidence_row.get("memory_scope")
            if memory_scope == "current_user_continuity":
                authority = "participant_continuity"
            elif memory_scope == "shared_character_or_world":
                authority = "character_world_context"
            else:
                raise AssertionError(
                    f"unsupported captured memory scope: {memory_scope}"
                )
        elif source_kind == "promoted_reflection":
            authority = (
                "conditional_character_guidance"
                if ":self_guidance:" in source_id
                else "character_world_context"
            )
        else:
            try:
                authority = authority_by_source_kind[source_kind]
            except KeyError as exc:
                raise AssertionError(
                    f"unsupported captured evidence source: {source_kind}"
                ) from exc
        evidence_row["authority"] = authority
        if (
            source_kind == "conversation_evidence"
            and source_id.startswith("conversation-progress-event:")
        ):
            occurred_at = evidence_ref.get("occurred_at")
            if not isinstance(occurred_at, str):
                raise AssertionError(
                    "captured progress evidence timestamp is invalid"
                )
            evidence_row["temporal_provenance"] = {
                "occurred_at": occurred_at,
                "age_descriptor": _progress_age_descriptor(
                    occurred_at,
                    current_event_occurred_at,
                ),
            }
    return replay_input


def _maximum_evidence(count: int = 32) -> list[dict[str, Any]]:
    """Build the maximum valid evidence cardinality with full-size text."""

    evidence: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        semantic_text = f"{index:02d}" + ("x" * 998)
        evidence.append({
            "evidence_handle": f"e{index}",
            "evidence_ref": {
                "source_kind": "promoted_memory",
                "source_id": f"memory-{index}",
                "occurred_at": INCIDENT_TIMESTAMP,
                "semantic_summary": semantic_text[:500],
            },
            "memory_scope": "shared_character_or_world",
            "semantic_text": semantic_text,
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["promoted_memory"]
            ),
            "authority": "character_world_context",
        })
    return evidence


def _maximum_character_constraints() -> dict[str, Any]:
    """Build the maximum prompt-visible valid character constraints."""

    constraints = _character_constraints()
    standard_ids = (
        "honesty",
        "avoid_harm",
        "respect_boundaries",
        "follow_through",
        "self_respect",
    )
    constraints["standards"] = [
        {
            "standard_id": standard_ids[index % len(standard_ids)],
            "description": f"{index:02d}" + ("s" * 420),
            "importance": 100,
        }
        for index in range(16)
    ]
    constraints["personality_judgment"] = {
        field_name: marker * 180
        for field_name, marker in (
            ("logic", "l"),
            ("defense", "d"),
            ("quirks", "q"),
            ("taboos", "t"),
        )
    }
    return constraints


def _maximum_scene_context() -> dict[str, Any]:
    """Build the maximum valid prompt-visible scene context."""

    return {
        "channel_scope": "private",
        "character_role": "c" * 500,
        "current_user_role": "u" * 500,
        "semantic_scene": "s" * 500,
        "public_group_scene": "",
        "conversation_continuity": "h" * 1000,
        "semantic_temporal_context": "t" * 500,
    }


def _maximum_runtime_capability_limits() -> list[str]:
    """Build all eight maximum-length runtime capability limits."""

    return [
        f"{index}" + ("r" * 499)
        for index in range(8)
    ]


def _maximum_prompt_state(scope: str) -> dict[str, Any]:
    """Build a valid state at every prompt-visible entity cardinality."""

    if scope == "character":
        state = build_character_production_state(
            updated_at=INCIDENT_TIMESTAMP,
        )
        for standard in state["standards"]:
            standard["description"] = "z" * 500
    else:
        state = build_acquaintance_user_state(
            global_user_id="maximum-prompt-user",
            updated_at=INCIDENT_TIMESTAMP,
        )

    entity_specs = {
        "goals": (
            16,
            "pursuing",
            state_models_module.GOAL_FIELDS,
        ),
        "threats": (
            16,
            "active",
            state_models_module.THREAT_FIELDS,
        ),
        "active_events": (
            32,
            "active",
            state_models_module.EVENT_FIELDS,
        ),
        "knowledge_gaps": (
            16,
            "open",
            state_models_module.GAP_FIELDS,
        ),
    }
    role_names = (
        "actor",
        "experiencer",
        "target",
        "object",
        "affected_goal",
        "affected_relationship",
    )
    for field_name, (count, status, fields) in entity_specs.items():
        entities = []
        for index in range(1, count + 1):
            entity = {
                "entity_id": f"{field_name}:{index}",
                "description": f"{index:02d}" + ("x" * 498),
                "salience": 100,
                "role_refs": [
                    {
                        "role": role_names[role_index % len(role_names)],
                        "entity_kind": "user",
                        "entity_id": f"role-user-{role_index}",
                    }
                    for role_index in range(8)
                ],
                "evidence_refs": [],
                "created_at": INCIDENT_TIMESTAMP,
                "updated_at": INCIDENT_TIMESTAMP,
                "status": status,
            }
            for numeric_field in fields - {"goal_kind", "status"}:
                entity[numeric_field] = 100
            if field_name == "goals":
                entity["goal_kind"] = state_models_module.GOAL_KINDS[
                    (index - 1) % len(state_models_module.GOAL_KINDS)
                ]
            entities.append(entity)
        state[field_name] = entities

    state["affect_activations"] = []
    for index, emotion_id in enumerate(
        state_models_module.EMOTION_IDS,
        start=1,
    ):
        root_ref = {
            "scope": scope,
            "kind": "event",
            "entity_id": state["active_events"][
                (index - 1) % len(state["active_events"])
            ]["entity_id"],
        }
        state["affect_activations"].append({
            "activation_id": f"emotion:{emotion_id}",
            "emotion_id": emotion_id,
            "primary_root": root_ref,
            "root_refs": [root_ref],
            "phase": "active",
            "score": 100,
            "peak_score": 100,
            "trend": "stable",
            "cause_status": "active",
            "started_at": INCIDENT_TIMESTAMP,
            "updated_at": INCIDENT_TIMESTAMP,
            "last_reinforced_at": INCIDENT_TIMESTAMP,
        })
    return state_models_module.validate_cognition_state(state)


def _maximum_valid_cognition_input(
    scope: str,
    *,
    identity_context: dict[str, dict[str, object]],
) -> dict[str, Any]:
    """Validate one complete maximum prompt-shape cognition input."""

    trigger_source = "user_message"
    if scope == "character":
        trigger_source = "self_cognition"
    payload = {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            trigger_source=trigger_source,
            content="maximum prompt-shape episode",
        ),
        "state_scope": scope,
        "mutable_state": _maximum_prompt_state(scope),
        "character_constraints": _maximum_character_constraints(),
        "character_identity_context": identity_context,
        "evidence": _maximum_evidence(),
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "v" * 8000,
        "scene_context": _maximum_scene_context(),
        "private_continuity_context": "p" * 1000,
        "runtime_capability_limits": (
            _maximum_runtime_capability_limits()
        ),
    }
    return validate_cognition_core_input(payload)


def _maximum_appraisal_context(
    question_kind: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Any,
    dict[str, Any],
]:
    """Build the largest valid source/state shape for one appraisal family."""

    scope = "character"
    if question_kind == "relationship_social":
        scope = "user"
    payload = _maximum_valid_cognition_input(
        scope,
        identity_context=maximum_identity_context(),
    )
    state = payload["mutable_state"]
    evidence = payload["evidence"]
    projection = project_state_for_prompt(
        state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    question = next(
        row for row in questions
        if row["question_kind"] == question_kind
    )
    return state, evidence, projection, question


def _maximum_goal_context() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Build the complete valid maximum required-goal context."""

    payload = _maximum_valid_cognition_input(
        "character",
        identity_context=maximum_identity_context(),
    )
    state = payload["mutable_state"]
    evidence = payload["evidence"]
    evidence[0]["evidence_ref"]["source_kind"] = "episode"
    evidence[0]["visible_to"] = list(EVIDENCE_SOURCE_QUESTION_IDS["episode"])
    projection = project_state_for_prompt(
        state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        evidence=evidence,
    )
    appraisal_results = [
        {
            "question_id": f"q:{question_kind}",
            "explanation": "a" * 1000,
            "propositions": [
                {"semantic_value": "m" * 200}
                for _ in range(8)
            ],
        }
        for question_kind in QUESTION_KINDS
    ]
    context = facade_module._branch_context(
        projection,
        state,
        evidence,
        appraisal_results,
        scene_context=payload["scene_context"],
        private_continuity_context=payload[
            "private_continuity_context"
        ],
    )
    assert "runtime_capability_limits" not in context
    context["goal_projection"] = {
        "goal_kind": "ordinary_response",
        "description": "g" * 500,
        "lifecycle": "active",
    }
    return context, evidence


def _pad_to_serialized_length(
    payload: dict[str, Any],
    container: dict[str, Any],
    target_chars: int,
) -> None:
    """Pad one JSON mapping to an exact ensure-ascii-false length."""

    container["padding"] = ""
    base_chars = len(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    padding_chars = target_chars - base_chars
    if padding_chars < 0:
        raise AssertionError("test payload already exceeds target length")
    container["padding"] = "x" * padding_chars
    assert len(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )) == target_chars


def _bid(
    branch_id: str = "ordinary_response",
    *,
    evidence_handles: list[str] | None = None,
) -> dict[str, Any]:
    """Build one complete admitted bid for owner-fallback tests."""

    bid = {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": f"goal:{branch_id}",
        },
        "intention": "answer the current grounded request",
        "desired_outcome": "preserve a coherent current interaction",
        "concrete_detail": "use only current evidence",
        "reason": "the admitted evidence supports this motive",
        "private_monologue": "I should respond deliberately.",
        "target_roles": [],
        "evidence_handles": evidence_handles or ["e1"],
        "expected_consequences": ["the interaction remains coherent"],
        "confidence": "high",
    }
    return bid


def _surface_input() -> dict[str, Any]:
    """Build a valid surface packet whose aggregate exceeds the stage cap."""

    long_text = "s" * 1000
    supporting_bids = [
        {
            "motive": long_text,
            "intention": long_text,
            "desired_outcome": long_text,
            "permitted_detail": long_text,
            "target_summaries": [],
            "expected_consequences": [],
        }
        for _ in range(7)
    ]
    payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            content="Render the selected grounded response.",
        ),
        "intention": {
            "route": "speech",
            "intention": "state the selected grounded response",
            "target_roles": [],
            "reason": "the current turn requires a visible response",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": supporting_bids,
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "runtime_capability_limits": [
            "No external action is available.",
        ],
        "interaction_style_context": "brief conversational speech",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Concise spoken clauses.",
        },
        "visual_character_context": "A neutral visual frame.",
    }
    return payload


def _valid_goal_draft() -> dict[str, Any]:
    """Build a latest-valid goal draft for selection overflow handling."""

    draft = {
        "intention": "make the required current selection",
        "desired_outcome": "retain character-owned choice",
        "concrete_detail": "state one grounded choice in this turn",
        "reason": "the current operation assigns selection to the character",
        "private_monologue": "I should make this choice myself.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the selection remains character-owned"],
        "confidence": "high",
    }
    return draft


def _count_exact_text_values(value: object, target: str) -> int:
    """Count one decoded text value across a nested JSON-compatible object."""

    if isinstance(value, dict):
        return sum(
            _count_exact_text_values(nested, target)
            for nested in value.values()
        )
    if isinstance(value, list):
        return sum(
            _count_exact_text_values(nested, target)
            for nested in value
        )
    return int(value == target)


@pytest.mark.asyncio
async def test_production_epistemic_case_reaches_bounded_model_boundary() -> None:
    """The exact repeated incident packet fits without duplicate authority."""

    state, evidence, projection, questions = _production_appraisal_context()
    document = _fixture_document()
    question = next(
        row
        for row in questions
        if row["question_kind"] == document["incident"]["question_kind"]
    )
    assert len(question["permitted_delta_paths"]) == document["incident"][
        "expected_permitted_delta_path_count"
    ]
    llm = _BoundaryProbeLLM()

    with pytest.raises(_BoundaryReached):
        await appraise_semantic_question(
            question,
            evidence,
            projection,
            _appraisal_services(llm),
            validation_state=state,
        )

    assert len(llm.calls) == 1
    human_payload = llm.calls[0]["human_payload"]
    assert len(human_payload) <= document["incident"]["prompt_cap_chars"]
    prompt_payload = json.loads(human_payload)
    assert "evidence" not in prompt_payload["state"]
    assert "permitted_delta_paths" not in prompt_payload["question"]
    assert [
        row["handle"] for row in prompt_payload["evidence"]
    ] == question["evidence_handles"]
    for row in prompt_payload["evidence"]:
        assert _count_exact_text_values(
            prompt_payload,
            row["semantic_text"],
        ) == 1

    reconstructed_paths = {
        f"{domain['state_field']}.{handle}.{axis}"
        for domain in prompt_payload["question"][
            "permitted_delta_path_domains"
        ]
        for handle in domain["handles"]
        for axis in domain["axes"]
    }
    assert reconstructed_paths == set(question["permitted_delta_paths"])


@pytest.mark.asyncio
@pytest.mark.parametrize("question_kind", QUESTION_KINDS)
async def test_each_maximum_appraisal_family_reaches_its_model_boundary(
    question_kind: str,
) -> None:
    """Every family fits its complete valid maximum prompt-visible shape."""

    state, evidence, projection, question = _maximum_appraisal_context(
        question_kind
    )
    llm = _BoundaryProbeLLM()

    with pytest.raises(_BoundaryReached):
        await appraise_semantic_question(
            question,
            evidence,
            projection,
            _appraisal_services(llm),
            validation_state=state,
        )

    assert len(llm.calls) == 1
    human_payload = llm.calls[0]["human_payload"]
    assert (
        len(llm.calls[0]["system_prompt"]) + len(human_payload)
        <= semantic_appraisal_module.SEMANTIC_APPRAISAL_PROMPT_CAP
    )
    prompt_payload = json.loads(human_payload)
    assert len(prompt_payload["evidence"]) == 8
    assert [
        row["handle"] for row in prompt_payload["evidence"]
    ] == question["evidence_handles"]
    assert "evidence" not in prompt_payload["state"]
    assert "permitted_delta_paths" not in prompt_payload["question"]


@pytest.mark.asyncio
async def test_appraisal_question_keeps_candidate_origin_contract() -> None:
    """Candidate origins stay in the retained question, not removable state."""

    state, evidence, projection, questions = _production_appraisal_context()
    projection = project_state_for_prompt(
        state,
        character_constraints=_character_constraints(),
        character_identity_context=canonical_identity_context(),
        scene_context={
            "participant_bindings": [
                {
                    "handle": f"p{index}",
                    "display_name": f"participant {index}",
                }
                for index in range(1, 6)
            ],
        },
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        state,
        projection.handle_to_ref,
    )
    question = next(
        item for item in questions if item["question_kind"] == "event_agency"
    )
    llm = _BoundaryProbeLLM()

    with pytest.raises(_BoundaryReached):
        await appraise_semantic_question(
            question,
            evidence,
            projection,
            _appraisal_services(llm),
            validation_state=state,
        )

    prompt_payload = json.loads(llm.calls[0]["human_payload"])
    origins = prompt_payload["question"]["candidate_origin_evidence"]
    assert origins
    assert set(origins.values()) <= set(question["evidence_handles"])
    assert all(handle.startswith("ce") for handle in origins)
    assert "causal_candidates" not in prompt_payload["state"]
    assert set(
        question["permitted_role_assignment_handles"]
    ) >= {"self", "current_user", "p1", "p2", "p3", "p4", "p5"}
    handle_domains = prompt_payload["question"]["handle_field_domains"]
    assert {"p1", "p2", "p3", "p4", "p5"} <= set(
        handle_domains["entity_handle"]
    )
    assert not any(
        handle.startswith("ce")
        for handle in handle_domains["entity_handle"]
    )
    assert any(
        handle.startswith("ce")
        for handle in handle_domains["subject_handle"]
    )


@pytest.mark.asyncio
async def test_appraisal_accumulates_one_item_then_stops_on_empty() -> None:
    """One accepted micro item is followed by one bounded terminator call."""

    class _MicroAppraisalLLM:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            payload = json.loads(str(getattr(messages[1], "content", "")))
            self.payloads.append(payload)
            question_id = payload["question"]["question_id"]
            if len(self.payloads) == 1:
                result = {
                    "question_id": question_id,
                    "proposition": {
                        "proposition_kind": "intentionality",
                        "subject_handle": "current_user",
                        "evidence_handles": ["e1"],
                        "role_assignments": [{
                            "role": "target",
                            "entity_handle": "self",
                        }],
                        "semantic_value": "当前用户有意作出当前承诺。",
                    },
                    "delta": None,
                }
            else:
                result = {
                    "question_id": question_id,
                    "proposition": None,
                    "delta": None,
                }
            return SimpleNamespace(
                content=json.dumps(result, ensure_ascii=False),
            )

    state, evidence, projection, questions = _production_appraisal_context()
    question = next(
        item for item in questions if item["question_kind"] == "event_agency"
    )
    llm = _MicroAppraisalLLM()

    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        _appraisal_services(llm),
        validation_state=state,
    )

    assert len(llm.payloads) == 2
    assert len(result["propositions"]) == 1
    assert result["deltas"] == []
    assert result["selected_evidence_handles"] == ["e1"]
    assert result["selected_role_handles"] == ["current_user", "self"]
    assert llm.payloads[0]["question"]["micro_appraisal"] == {
        "item_index": 1,
        "maximum_items": 8,
        "maximum_propositions": 1,
        "maximum_deltas": 1,
        "empty_lists_end_family": True,
        "emitted_proposition_signatures": [],
        "emitted_delta_paths": [],
    }
    assert llm.payloads[1]["question"]["micro_appraisal"][
        "emitted_proposition_signatures"
    ] == ["intentionality|current_user|"]


@pytest.mark.asyncio
async def test_appraisal_boundary_rejection_returns_accepted_prefix_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later boundary rejection keeps the accepted appraisal prefix."""

    class _PrefixThenInvalidLLM:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            payload = json.loads(str(getattr(messages[1], "content", "")))
            self.payloads.append(payload)
            question_id = payload["question"]["question_id"]
            if len(self.payloads) == 1:
                result = {
                    "question_id": question_id,
                    "proposition": {
                        "proposition_kind": "intentionality",
                        "subject_handle": "current_user",
                        "evidence_handles": ["e1"],
                        "role_assignments": [{
                            "role": "target",
                            "entity_handle": "self",
                        }],
                        "semantic_value": (
                            "the current user intentionally made a commitment"
                        ),
                    },
                    "delta": None,
                }
            else:
                result = {
                    "question_id": question_id,
                    "proposition": {
                        "proposition_kind": "intentionality",
                        "subject_handle": "unknown-role",
                        "evidence_handles": ["e1"],
                        "role_assignments": [],
                        "semantic_value": "invalid role handle",
                    },
                    "delta": None,
                }
            return SimpleNamespace(content=json.dumps(result))

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        semantic_appraisal_module,
        "capture_validation_event",
        lambda event_id, payload: events.append((event_id, dict(payload))),
    )
    state, evidence, projection, questions = _production_appraisal_context()
    question = next(
        item for item in questions if item["question_kind"] == "event_agency"
    )
    llm = _PrefixThenInvalidLLM()

    result = await appraise_semantic_question(
        question,
        evidence,
        projection,
        _appraisal_services(llm),
        validation_state=state,
    )

    assert len(llm.payloads) == 2
    assert len(result["propositions"]) == 1
    assert result["deltas"] == []
    boundary_events = [
        payload
        for event_id, payload in events
        if event_id == "semantic_appraisal_boundary_failure"
    ]
    assert len(boundary_events) == 1
    assert boundary_events[0]["failure_kind"] == (
        "producer_handle_domain_invalid"
    )
    assert boundary_events[0]["disposition"] == "terminal_rejection"
    bounded_event = next(
        payload
        for event_id, payload in events
        if event_id == "semantic_appraisal_bounded_termination"
    )
    assert bounded_event == {
        "question_id": question["question_id"],
        "item_index": 2,
        "error_code": "cognition_boundary_rejected",
        "attempt_count": 1,
        "accepted_proposition_count": 1,
        "accepted_delta_count": 0,
        "disposition": "accepted_prefix",
        "error": bounded_event["error"],
    }


def test_appraisal_suppresses_exact_repeats_as_no_progress() -> None:
    """Repeated accepted components become an empty bounded terminator."""

    proposition = {
        "proposition_kind": "social_meaning",
        "subject_handle": "current_user",
        "evidence_handles": ["e1"],
        "role_assignments": [],
        "semantic_value": "The relationship meaning is already represented.",
    }
    delta = {
        "target_path": "relationship.r1.trust",
        "delta": 5,
        "evidence_handles": ["e1"],
        "reason": "The trust change is already represented.",
    }
    accepted = {
        "question_id": "q:relationship_social",
        "selected_evidence_handles": ["e1"],
        "selected_role_handles": ["current_user", "r1"],
        "propositions": [proposition],
        "deltas": [delta],
        "explanation": "The accepted item is already represented.",
    }
    repeated = deepcopy(accepted)

    normalized = (
        semantic_appraisal_module._suppress_emitted_appraisal_components(
            repeated,
            accepted,
        )
    )

    assert normalized["propositions"] == []
    assert normalized["deltas"] == []
    assert normalized["selected_evidence_handles"] == []
    assert normalized["selected_role_handles"] == []


def test_persistent_event_handles_are_distinct_from_evidence_handles() -> None:
    """Persistent events use evN while evidence keeps the eN namespace."""

    _, evidence, projection, _ = _maximum_appraisal_context("event_agency")
    persistent_event_handles = [
        handle
        for handle, ref in projection.handle_to_ref.items()
        if ref["kind"] == "event"
        and not ref["entity_id"].startswith("candidate:")
    ]

    assert persistent_event_handles
    assert all(handle.startswith("ev") for handle in persistent_event_handles)
    assert all(row["evidence_handle"].startswith("e") for row in evidence)


def test_appraisal_exact_cap_and_cap_plus_one_are_distinct() -> None:
    """The appraisal owner accepts its exact payload cap and rejects one more."""

    evidence_rows = [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": "e" * 96,
    }]
    question: dict[str, Any] = {}
    payload = {
        "question": question,
        "evidence": evidence_rows,
        "state": {},
    }
    appraisal_cap = semantic_appraisal_module.SEMANTIC_APPRAISAL_PROMPT_CAP
    _pad_to_serialized_length(payload, question, appraisal_cap)

    fitted, _, _ = semantic_appraisal_module._fit_appraisal_payload(
        payload,
        system_prompt_chars=0,
    )

    assert len(fitted) == appraisal_cap
    question["padding"] += "x"
    with pytest.raises(CognitionContextLimitError):
        semantic_appraisal_module._fit_appraisal_payload(
            payload,
            system_prompt_chars=0,
        )


def test_goal_exact_aggregate_cap_and_cap_plus_one_are_distinct() -> None:
    """The goal owner counts its system prompt inside the aggregate cap."""

    evidence_rows = [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": "e" * 96,
    }]
    semantic_context: dict[str, Any] = {}
    payload = {
        "branch": {},
        "goal": {},
        "semantic_context": semantic_context,
        "evidence": evidence_rows,
        "role_handles": [],
        "role_summaries": {},
    }
    goal_cap = goal_cognition_module.GOAL_COGNITION_PROMPT_CAP
    system_prompt = (
        goal_cognition_module.GOAL_COGNITION_PROMPT
        + goal_cognition_module.CONTINUITY_AUTHORITY_INSTRUCTIONS
    )
    payload_cap = goal_cap - len(system_prompt)
    _pad_to_serialized_length(payload, semantic_context, payload_cap)

    fitted = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt=system_prompt,
    )

    assert (
        len(system_prompt) + len(fitted)
        == goal_cap
    )
    semantic_context["padding"] += "x"
    with pytest.raises(PromptBudgetError):
        goal_cognition_module._fit_goal_prompt_payload(
            payload,
            system_prompt=system_prompt,
        )


def test_goal_budget_retains_public_scene_and_private_continuity() -> None:
    """The two scene lanes remain model-visible when the payload fits."""

    payload = {
        "branch": {},
        "goal": {},
        "semantic_context": {
            "scene_context": {
                "public_group_scene": "PUBLIC_SCENE_SENTINEL",
                "conversation_continuity": "PRIVATE_CONTINUITY_SENTINEL",
            },
            "private_continuity_context": "PRIVATE_RESIDUE_SENTINEL",
        },
        "evidence": [],
        "role_handles": [],
        "role_summaries": {},
    }

    fitted = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt="",
    )
    fitted_payload = json.loads(fitted)

    assert fitted_payload["semantic_context"]["scene_context"] == {
        "public_group_scene": "PUBLIC_SCENE_SENTINEL",
        "conversation_continuity": "PRIVATE_CONTINUITY_SENTINEL",
    }
    assert fitted_payload["semantic_context"][
        "private_continuity_context"
    ] == "PRIVATE_RESIDUE_SENTINEL"


def test_goal_budget_drops_restored_optional_context_before_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private and group guidance cannot displace grounded episode evidence."""

    evidence_text = "E" * 200
    semantic_context = {
        "past_dialog_cognition_context": "P" * 1800,
        "group_engagement_action_context": {
            "engagement_guidelines": ["G" * 120] * 5,
            "confidence": "C" * 80,
        },
    }
    payload = {
        "branch": {},
        "goal": {},
        "semantic_context": semantic_context,
        "evidence": [{
            "handle": "e1",
            "source_kind": "episode",
            "semantic_text": evidence_text,
        }],
        "role_handles": [],
        "role_summaries": {},
    }
    required_payload = {
        **payload,
        "semantic_context": {},
    }
    required_chars = len(json.dumps(
        required_payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    monkeypatch.setattr(
        goal_cognition_module,
        "GOAL_COGNITION_PROMPT_CAP",
        required_chars,
    )

    fitted = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt="",
    )
    fitted_payload = json.loads(fitted)

    assert fitted_payload["semantic_context"] == {}
    assert fitted_payload["evidence"][0]["semantic_text"] == evidence_text


@pytest.mark.asyncio
async def test_action_budget_drops_group_guidance_before_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action planning drops valid advisory guidance before giving up."""

    group_context = {
        "engagement_guidelines": ["G" * 120] * 5,
        "confidence": "C" * 80,
    }
    first_probe = _BoundaryProbeLLM()
    call_kwargs = {
        "primary_bid": _bid(),
        "supporting_bids": [],
        "episode": {
            "trigger_source": "self_cognition",
            "output_mode": "think_only",
        },
        "evidence": _maximum_evidence(1),
        "available_actions": [],
        "available_resolvers": [],
        "resolver_context": "",
        "group_engagement_action_context": group_context,
    }

    with pytest.raises(_BoundaryReached):
        await action_selection_module.plan_actions(
            **call_kwargs,
            services=SimpleNamespace(
                llm=first_probe,
                action_planning_config=object(),
            ),
        )

    full_payload = json.loads(first_probe.calls[0]["human_payload"])
    reduced_payload = dict(full_payload)
    reduced_payload["group_engagement_action_context"] = {
        "engagement_guidelines": [],
        "confidence": "",
    }
    reduced_chars = len(json.dumps(
        reduced_payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    monkeypatch.setattr(
        action_selection_module,
        "ACTION_PLANNING_PROMPT_CAP",
        len(action_selection_module.ACTION_PLANNING_PROMPT) + reduced_chars,
    )
    reduced_probe = _BoundaryProbeLLM()

    with pytest.raises(_BoundaryReached):
        await action_selection_module.plan_actions(
            **call_kwargs,
            services=SimpleNamespace(
                llm=reduced_probe,
                action_planning_config=object(),
            ),
        )

    actual_payload = json.loads(reduced_probe.calls[0]["human_payload"])
    assert actual_payload["group_engagement_action_context"] == {
        "engagement_guidelines": [],
        "confidence": "",
    }


def test_required_selection_budget_preserves_mandatory_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before a call instead of erasing required selection facts."""

    required_operation_text = "required operation " + ("R" * 300)
    required_progress_text = (
        "completed event; actor=user; action=finish; object=prior choice; "
        + ("P" * 300)
    )
    optional_text = "O" * 1000

    def _payload() -> dict[str, Any]:
        evidence_rows = [{
            "handle": "e1",
            "source_kind": "episode",
            "semantic_text": required_operation_text,
        }, {
            "handle": "e2",
            "source_kind": "conversation_evidence",
            "semantic_text": required_progress_text,
        }, {
            "handle": "e3",
            "source_kind": "conversation_evidence",
            "semantic_text": optional_text,
        }]
        return {
            "branch": {},
            "goal": {},
            "semantic_context": {},
            "role_handles": [],
            "role_summaries": {},
            "required_selection_operations": [{
                "evidence_handle": "e1",
                "role_explicit_content": required_operation_text,
            }],
            "conversation_progress_evidence": [{
                "evidence_handle": "e2",
                "semantic_text": required_progress_text,
            }],
            "supporting_evidence": [evidence_rows[2]],
        }

    payload = _payload()
    serialized_chars = len(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    optional_floor = goal_cognition_module.MIN_PROMPT_EVIDENCE_TEXT_CHARS
    optional_reduction = len(optional_text) - optional_floor
    exact_optional_floor = serialized_chars - optional_reduction
    system_prompt = goal_cognition_module.REQUIRED_SELECTION_GOAL_PROMPT
    monkeypatch.setattr(
        goal_cognition_module,
        "GOAL_COGNITION_PROMPT_CAP",
        len(system_prompt) + exact_optional_floor,
    )

    fitted = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt=system_prompt,
    )
    fitted_payload = json.loads(fitted)

    assert fitted_payload["required_selection_operations"][0][
        "role_explicit_content"
    ] == required_operation_text
    assert fitted_payload["conversation_progress_evidence"][0][
        "semantic_text"
    ] == required_progress_text
    assert len(
        fitted_payload["supporting_evidence"][0]["semantic_text"]
    ) == optional_floor

    payload = _payload()
    monkeypatch.setattr(
        goal_cognition_module,
        "GOAL_COGNITION_PROMPT_CAP",
        len(system_prompt) + exact_optional_floor - 1,
    )

    with pytest.raises(PromptBudgetError):
        goal_cognition_module._fit_goal_prompt_payload(
            payload,
            system_prompt=system_prompt,
        )


def test_required_selection_regeneration_feedback_counts_toward_cap() -> None:
    """Fit replacement feedback and payload inside one aggregate budget."""

    payload = {
        "branch": {},
        "goal": {},
        "semantic_context": {},
        "role_handles": [],
        "role_summaries": {},
        "required_selection_operations": [{
            "evidence_handle": "e1",
            "role_explicit_content": "required operation",
        }],
        "conversation_progress_evidence": [{
            "evidence_handle": "e2",
            "semantic_text": "completed prior event",
        }],
        "supporting_evidence": [{
            "handle": "e3",
            "source_kind": "conversation_evidence",
            "semantic_text": "S" * 30000,
        }],
    }
    initial_system_prompt = (
        goal_cognition_module.REQUIRED_SELECTION_GOAL_PROMPT
    )
    regeneration_system_prompt = initial_system_prompt
    repair_payload = dict(payload)
    repair_payload["repair_feedback"] = {
        "validation_error": "selection goal draft fields are not exact",
        "goal_output_contract": {
            "top_level_fields": [
                "selection",
                "reason",
                "private_monologue",
                "target_role_handles",
                "evidence_handles",
                "expected_consequences",
                "confidence",
                "relational_willingness",
            ],
            "field_types": {
                "selection": "non_empty_string_max_500",
            },
        },
        "allowed_evidence_handles": ["e1", "e2", "e3"],
        "required_evidence_handles": ["e1"],
        "current_episode_evidence_handles": ["e1"],
        "allowed_role_handles": [],
        "role_handles_forbidden_in_evidence_handles": [],
        "max_evidence_handles": 9,
        "max_role_handles": 8,
        "invalid_draft": '{"evidence_handles":["r1"]}',
    }

    initial_payload = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt=initial_system_prompt,
    )
    regeneration_payload = goal_cognition_module._fit_goal_prompt_payload(
        repair_payload,
        system_prompt=regeneration_system_prompt,
    )

    assert (
        len(initial_system_prompt) + len(initial_payload)
        <= goal_cognition_module.GOAL_COGNITION_PROMPT_CAP
    )
    assert (
        len(regeneration_system_prompt) + len(regeneration_payload)
        <= goal_cognition_module.GOAL_COGNITION_PROMPT_CAP
    )
    assert regeneration_system_prompt == initial_system_prompt
    assert len(regeneration_payload) > len(initial_payload)
    fitted_repair_payload = json.loads(regeneration_payload)
    assert (
        fitted_repair_payload["repair_feedback"]["required_evidence_handles"]
        == ["e1"]
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"state": [], "evidence": []},
            "state projection is invalid",
        ),
        (
            {"state": {}, "evidence": {}},
            "evidence projection is invalid",
        ),
    ],
)
def test_malformed_appraisal_projection_is_an_invariant_failure(
    payload: dict[str, Any],
    message: str,
) -> None:
    """Malformed canonical projection stays distinct from budget exhaustion."""

    with pytest.raises(ValueError, match=message) as error_info:
        semantic_appraisal_module._fit_appraisal_payload(
            payload,
            system_prompt_chars=0,
        )

    assert not isinstance(
        error_info.value,
        CognitionContextLimitError,
    )


def test_shared_budget_truncates_low_priority_text_without_losing_rows() -> None:
    """Aggregate fitting preserves authority and truncates the last row first."""

    prompt_budget = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.prompt_budget"
    )
    rows = [
        {
            "handle": f"e{index}",
            "source_kind": "promoted_memory",
            "semantic_text": marker * 400,
        }
        for index, marker in enumerate(("A", "B", "C"), start=1)
    ]
    payload = {"contract": {"kind": "test"}, "evidence": rows}
    raw_chars = len(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    maximum_chars = raw_chars - 120

    serialized = prompt_budget.fit_evidence_texts_to_budget(
        payload,
        rows,
        text_field="semantic_text",
        maximum_chars=maximum_chars,
        minimum_text_chars=96,
    )

    assert len(serialized) <= maximum_chars
    fitted_rows = json.loads(serialized)["evidence"]
    assert [row["handle"] for row in fitted_rows] == ["e1", "e2", "e3"]
    assert [row["source_kind"] for row in fitted_rows] == [
        "promoted_memory",
        "promoted_memory",
        "promoted_memory",
    ]
    assert fitted_rows[0]["semantic_text"] == "A" * 400
    assert fitted_rows[1]["semantic_text"] == "B" * 400
    assert len(fitted_rows[2]["semantic_text"]) < 400
    assert fitted_rows[2]["semantic_text"].startswith("C" * 16)
    assert fitted_rows[2]["semantic_text"].endswith("C" * 16)


@pytest.mark.asyncio
async def test_irreducible_appraisal_context_is_omitted_per_question() -> None:
    """One irreducible appraisal records failure without aborting the turn."""

    async def _raise_context_limit() -> dict[str, Any]:
        raise CognitionContextLimitError("required packet cannot fit")

    task = asyncio.create_task(_raise_context_limit())
    results, failures, warnings = await facade_module._collect_appraisals(
        [task],
        [{
            "question_id": "q:epistemic_comparison_memory",
            "question_kind": "epistemic_comparison_memory",
        }],
    )

    assert results == []
    assert failures == {
        "q:epistemic_comparison_memory": (
            "semantic_appraisal_context_limit"
        ),
    }
    assert warnings == [
        "semantic_appraisal_failed:semantic_appraisal_context_limit",
    ]


@pytest.mark.asyncio
async def test_goal_prompt_fits_maximum_evidence_without_duplication() -> None:
    """Complete maximum goal context fits with one registry per concept."""

    semantic_context, evidence = _maximum_goal_context()
    expected_role_handles = set(semantic_context["_role_bindings"])
    llm = _ValidGoalLLM()

    bid = await goal_cognition_module.run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "goal:test"},
        semantic_context,
        evidence,
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert bid["evidence_handles"] == ["e1"]
    assert len(llm.human_payloads) == 1
    human_payload = llm.human_payloads[0]
    assert (
        len(
            goal_cognition_module.GOAL_COGNITION_PROMPT
            + goal_cognition_module.CONTINUITY_AUTHORITY_INSTRUCTIONS
        )
        + len(human_payload)
        <= goal_cognition_module.GOAL_COGNITION_PROMPT_CAP
    )
    prompt_payload = json.loads(human_payload)
    assert "evidence" not in prompt_payload["semantic_context"]
    assert "goal_projection" not in prompt_payload["semantic_context"]
    assert "role_summaries" not in prompt_payload["semantic_context"]
    assert "personality_judgment" not in (
        prompt_payload["semantic_context"]["character_constraints"]
    )
    assert set(
        prompt_payload["semantic_context"]["character_identity"]
    ) == {
        "core",
        "personality",
        "boundaries",
        "self_image",
    }
    assert set(prompt_payload["role_handles"]) == expected_role_handles
    assert set(prompt_payload["role_summaries"]) == expected_role_handles
    assert [
        row["handle"] for row in prompt_payload["evidence"]
    ] == [f"e{index}" for index in range(1, 33)]
    assert human_payload.count(
        prompt_payload["evidence"][0]["semantic_text"]
    ) == 1
    assert len(prompt_payload["evidence"][-1]["semantic_text"]) < 1000


@pytest.mark.asyncio
async def test_required_selection_producer_overflow_fails_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection production honors the same aggregate preflight cap."""

    llm = _NoCallLLM()
    monkeypatch.setattr(
        goal_cognition_module,
        "GOAL_COGNITION_PROMPT_CAP",
        1,
    )
    selection_operation = json.dumps({
        "role_explicit_content": "the current character must choose",
        "response_operation": {
            "operation": "make one concrete selection",
            "selection_required": True,
            "response_owner_role": "当前角色",
            "selection_owner_role": "当前角色",
            "embedded_actor_role": "当前用户",
            "embedded_target_role": "当前角色",
        },
    })
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-15T00:00:00Z",
            "semantic_summary": selection_operation,
        },
        "semantic_text": selection_operation,
        "visible_to": ["q:event_agency"],
        "authority": "current_event",
    }]

    with pytest.raises(CognitionExecutionError) as error_info:
        await goal_cognition_module.run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {"scope": "user", "kind": "goal", "entity_id": "goal:test"},
            {
                "_role_bindings": {},
                "role_summaries": {},
            },
            evidence,
            SimpleNamespace(
                llm=llm,
                goal_ordinary_response_config=object(),
            ),
        )

    assert error_info.value.error_code == "goal_cognition_context_limit"
    assert llm.call_count == 0


def test_required_selection_has_only_producer_repair_surface() -> None:
    """Keep structural repair with the producer, without a verifier stage."""

    assert not hasattr(
        goal_cognition_module,
        "_verify_required_selection_bid",
    )
    assert not hasattr(
        goal_cognition_module,
        "_enforce_required_selection_alignment",
    )
    assert not hasattr(
        goal_cognition_module,
        "REQUIRED_SELECTION_VERIFIER_PROMPT",
    )
    retired_name = "REQUIRED_SELECTION_" + "GOAL_REPAIR_PROMPT"
    assert not hasattr(goal_cognition_module, retired_name)
    assert hasattr(
        goal_cognition_module,
        "SELECTION_GOAL_REPAIR_INSTRUCTIONS",
    )


@pytest.mark.asyncio
async def test_workspace_preflight_overflow_uses_stable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workspace cap exhaustion keeps the first complete registry-order bid."""

    monkeypatch.setattr(
        workspace_module,
        "WORKSPACE_COLLAPSE_PROMPT_CAP",
        1,
        raising=False,
    )
    llm = _NoCallLLM()

    result = await workspace_module.collapse_bids(
        [
            _bid("social_care"),
            _bid("ordinary_response"),
        ],
        SimpleNamespace(
            llm=llm,
            workspace_collapse_config=object(),
        ),
        current_event=[],
        goal_context_by_ref={
            "goal:social_care": {
                "goal_handle": "goal:social_care",
                "goal_kind": "social_care",
                "description": "support the active social-care matter",
                "status": "pursuing",
                "salience": 40,
                "importance": 70,
                "progress": 20,
                "obstruction": 0,
                "urgency": 20,
            },
        },
    )

    assert result["primary_branch_id"] == "ordinary_response"
    assert result["supporting_bids"] == []
    assert result["suppressed_branch_ids"] == ["social_care"]
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_workspace_fits_matching_relevance_context_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long matching context is fitted while every bid remains selectable."""

    class _MatchingWorkspaceLLM:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.calls.append(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps({
                "primary_bid_handle": "b2",
                "supporting_bid_handles": [],
                "suppressed_bid_handles": ["b1"],
            }))

    monkeypatch.setattr(
        workspace_module,
        "WORKSPACE_COLLAPSE_PROMPT_CAP",
        len(workspace_module.COLLAPSE_PROMPT) + 4000,
    )
    llm = _MatchingWorkspaceLLM()
    matching_tail = "same concrete body boundary"
    current_event = [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": "current autonomy pressure " + "x" * 12000
        + matching_tail,
    }]
    goal_context_by_ref = {
        "goal:autonomy_boundary": {
            "goal_handle": "goal:autonomy_boundary",
            "goal_kind": "autonomy_boundary",
            "description": "persistent autonomy goal " + "y" * 12000
            + matching_tail,
            "status": "pursuing",
            "salience": 80,
            "importance": 90,
            "progress": 20,
            "obstruction": 70,
            "urgency": 80,
        },
    }

    result = await workspace_module.collapse_bids(
        [
            _bid("ordinary_response"),
            _bid("autonomy_boundary"),
        ],
        SimpleNamespace(
            llm=llm,
            workspace_collapse_config=object(),
        ),
        current_event=current_event,
        goal_context_by_ref=goal_context_by_ref,
    )

    assert result["primary_branch_id"] == "autonomy_boundary"
    assert len(llm.calls) == 1
    fitted_payload = llm.calls[0]
    assert set(fitted_payload["bids"]) == {"b1", "b2"}
    assert "..." in fitted_payload["current_event"][0]["semantic_text"]
    assert matching_tail in fitted_payload["current_event"][0]["semantic_text"]
    persistent_goal = fitted_payload["bids"]["b2"]["persistent_goal"]
    assert persistent_goal["goal_handle"] == "goal:autonomy_boundary"
    assert "..." in persistent_goal["description"]
    assert matching_tail in persistent_goal["description"]


@pytest.mark.asyncio
async def test_workspace_repair_overflow_uses_fallback_before_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-cap workspace request never schedules an over-cap repair."""

    bids = [
        _bid("social_care"),
        _bid("ordinary_response"),
    ]
    ordered = sorted(
        bids,
        key=lambda bid: workspace_module.branch_order_key(
            bid["branch_id"]
        ),
    )
    current_event: list[dict[str, str]] = []
    goal_context_by_ref = {
        "goal:social_care": {
            "goal_handle": "goal:social_care",
            "goal_kind": "social_care",
            "description": "support the active social-care matter",
            "status": "pursuing",
            "salience": 40,
            "importance": 70,
            "progress": 20,
            "obstruction": 0,
            "urgency": 20,
        },
    }
    prompt_payload = {
        "current_event": current_event,
        "bids": {
            f"b{index}": {
                "branch_id": bid["branch_id"],
                "persistent_goal": (
                    None
                    if bid["branch_id"] == "ordinary_response"
                    else goal_context_by_ref[
                        bid["goal_ref"]["entity_id"]
                    ]
                ),
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "reason": bid["reason"],
                "confidence": bid["confidence"],
            }
            for index, bid in enumerate(ordered, start=1)
        }
    }
    initial_chars = len(json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    monkeypatch.setattr(
        workspace_module,
        "WORKSPACE_COLLAPSE_PROMPT_CAP",
        len(workspace_module.COLLAPSE_PROMPT) + initial_chars,
    )
    llm = _InvalidCandidateLLM()

    result = await workspace_module.collapse_bids(
        bids,
        SimpleNamespace(
            llm=llm,
            workspace_collapse_config=object(),
        ),
        current_event=current_event,
        goal_context_by_ref=goal_context_by_ref,
    )

    assert result["primary_branch_id"] == "ordinary_response"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_action_planning_overflow_returns_empty_blocked_plan() -> None:
    """Action preflight overflow requests no capability and keeps speech viable."""

    llm = _NoCallLLM()

    result = await action_selection_module.plan_actions(
        primary_bid=_bid(),
        supporting_bids=[],
        episode={
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=_maximum_evidence(),
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=llm,
            action_planning_config=object(),
        ),
    )

    assert result["action_requests"] == []
    assert result["resolver_requests"] == []
    assert result["goal_resolution"] == "blocked"
    assert result["intention"]["route"] == "speech"
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_action_repair_overflow_returns_empty_before_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-cap planner request enters its empty owner disposition."""

    monkeypatch.setattr(
        action_selection_module,
        "ACTION_PLANNING_PROMPT_CAP",
        100,
    )
    llm = _InvalidCandidateLLM()

    decision = await action_selection_module._invoke_action_planner(
        services=SimpleNamespace(
            llm=llm,
            action_planning_config=object(),
        ),
        messages=[
            SystemMessage(content="planner"),
            HumanMessage(content="p" * 100),
        ],
        bid_handles={},
        action_handles={},
        resolver_handles={},
        current_goal_progress=None,
        required_resolver_evidence_dependency=None,
        runtime_capability_limits=[],
    )

    assert decision == (
        action_selection_module._empty_action_plan_decision()
    )
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_action_authorization_overflow_denies_all_candidates() -> None:
    """Authorization preflight overflow cannot authorize executable work."""

    evidence = _maximum_evidence(8)
    evidence_handles = [
        row["evidence_handle"] for row in evidence
    ]
    bid = _bid(evidence_handles=evidence_handles)
    action_requests = [
        {
            "bid_handle": "b1",
            "action_handle": f"a{index}",
            "decision": "enqueue",
            "semantic_goal": f"perform bounded effect {index}",
            "reason": "the admitted bid proposed this effect",
        }
        for index in range(1, 4)
    ]
    action_handles = {
        f"a{index}": {
            "action_kind": f"bounded_action_{index}",
            "capability": "perform one bounded durable effect",
            "permission": "allowed",
            "decision_mode": "closed",
            "allowed_decisions": ["enqueue"],
            "default_decision": "enqueue",
            "decision_pattern": "",
            "context_ref": "",
            "target_roles": [],
        }
        for index in range(1, 4)
    }
    llm = _NoCallLLM()

    result = await action_authorization_module.authorize_action_requests(
        action_requests=action_requests,
        bid_handles={"b1": bid},
        evidence=evidence,
        action_handles=action_handles,
        runtime_capability_limits=[],
        services=SimpleNamespace(
            llm=llm,
            action_authorization_config=object(),
        ),
    )

    assert result == []
    assert llm.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage_name",
    [
        "action_authorization",
        "resolver_authorization",
    ],
)
async def test_authorization_repair_overflow_denies_before_second_call(
    stage_name: str,
) -> None:
    """Shared authorization repairs use the calling owner's aggregate cap."""

    llm = _InvalidCandidateLLM()

    decisions = (
        await action_authorization_module.invoke_semantic_authorizer(
            services=SimpleNamespace(llm=llm),
            config=object(),
            messages=[
                SystemMessage(content="authorizer"),
                HumanMessage(content="a" * 100),
            ],
            candidate_handles=["c1"],
            stage_name=stage_name,
            output_state_fields=["authorized_requests"],
            prompt_cap=100,
        )
    )

    assert decisions == {"c1": False}
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_resolver_authorization_overflow_denies_all_candidates() -> None:
    """Resolver preflight overflow schedules no evidence work."""

    resolver_requests = [
        {
            "bid_handle": "b1",
            "resolver_handle": f"r{index}",
            "semantic_goal": f"retrieve bounded evidence {index}",
            "reason": "the admitted bid still needs this evidence",
        }
        for index in range(1, 4)
    ]
    resolver_handles = {
        f"r{index}": {
            "capability": f"resolver_{index}",
            "semantic_capability": "retrieve one bounded evidence source",
            "availability": "available",
        }
        for index in range(1, 4)
    }
    llm = _NoCallLLM()

    result = await resolver_authorization_module.authorize_resolver_requests(
        resolver_requests=resolver_requests,
        bid_handles={"b1": _bid()},
        evidence=_maximum_evidence(),
        resolver_handles=resolver_handles,
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=llm,
            resolver_authorization_config=object(),
        ),
    )

    assert result == []
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_appraisal_repair_uses_residual_budget_for_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-cap canonical payload keeps a bounded second attempt reachable."""

    state, evidence, projection, questions = _production_appraisal_context()
    monkeypatch.setattr(
        semantic_appraisal_module,
        "_fit_appraisal_payload",
        lambda payload, *, system_prompt_chars: (
            "p" * 7900,
            frozenset(),
            frozenset(),
        ),
    )
    invalid_candidate = json.dumps({"padding": "x" * 5000})
    llm = _InvalidCandidateLLM(invalid_candidate)

    with pytest.raises(
        CognitionExecutionError,
        match="contract attempts exhausted",
    ):
        await appraise_semantic_question(
            questions[0],
            evidence,
            projection,
            _appraisal_services(llm),
            validation_state=state,
        )

    assert len(llm.calls) == 2
    repair_dynamic_chars = sum(len(content) for content in llm.calls[1][1:])
    assert (
        len(llm.calls[1][0]) + repair_dynamic_chars
        <= semantic_appraisal_module.SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP
    )
    assert len(llm.calls[1][2]) <= len(invalid_candidate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    _CAPTURED_NEAR_CAP_CASES,
    ids=lambda case: str(case["case_id"]),
)
async def test_captured_near_cap_boundary_candidates_terminate_without_retry(
    case: Mapping[str, object],
) -> None:
    """Each captured boundary candidate ends its family after one call."""

    input_payload, historical_response = _load_captured_near_cap_case(case)
    payload, preliminary_state, projection, questions = (
        _captured_near_cap_appraisal_context(input_payload)
    )
    question_id = case.get("question_id")
    assert isinstance(question_id, str)
    matching_questions = [
        question
        for question in questions
        if question.get("question_id") == question_id
    ]
    assert len(matching_questions) == 1
    original_state = deepcopy(preliminary_state)
    llm = _InvalidCandidateLLM(historical_response)

    result = await appraise_semantic_question(
        matching_questions[0],
        payload["evidence"],
        projection,
        _appraisal_services(llm),
        validation_state=preliminary_state,
    )

    assert result == {
        "question_id": question_id,
        "selected_evidence_handles": [],
        "selected_role_handles": [],
        "propositions": [],
        "deltas": [],
        "explanation": "No additional supported semantic item.",
    }
    assert len(llm.calls) == 1
    assert preliminary_state == original_state


@pytest.mark.asyncio
async def test_text_surface_overflow_returns_validated_degraded_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text cap overflow projects the committed intention without model prose."""

    surface_stages_module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface_stages"
    )
    monkeypatch.setattr(surface_stages_module, "SURFACE_STAGE_PROMPT_CAP", 100)
    llm = _NoCallLLM()

    output = await surface_module.run_text_surface_planning(
        _surface_input(),
        SimpleNamespace(
            llm=llm,
            content_plan_config=object(),
            preference_config=object(),
        ),
    )

    assert output["content_plan"] == "state the selected grounded response"
    assert output["selected_surface_intent"] == (
        "state the selected grounded response"
    )
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_surface_overflow_reduces_bids_before_model_boundary() -> None:
    """Surface preflight retains the minimum supporting-bid set before calling."""

    llm = _BoundaryProbeLLM()

    with pytest.raises(_BoundaryReached):
        await surface_module.run_text_surface_planning(
            _surface_input(),
            SimpleNamespace(
                llm=llm,
                content_plan_config=object(),
                preference_config=object(),
            ),
        )

    assert len(llm.calls) == 2
    supporting_bid_counts = {
        len(json.loads(call["human_payload"])["surface"]["supporting_bids"])
        for call in llm.calls
    }
    assert min(supporting_bid_counts) < 7
    assert supporting_bid_counts <= set(range(2, 8))


@pytest.mark.asyncio
async def test_surface_repair_reuses_the_fitted_surface_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface repair carries the reduced projection into its retry packet."""

    surface_stages_module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface_stages"
    )
    monkeypatch.setattr(
        surface_stages_module,
        "SURFACE_STAGE_PROMPT_CAP",
        12_000,
    )
    llm = _InvalidThenValidSurfaceLLM()

    result = await surface_stages_module._run_surface_stage(
        payload=_surface_input(),
        system_prompt="surface",
        llm=llm,
        config=object(),
        stage_name="visual",
        validator=surface_stages_module._validate_visual_result,
        safe_checkpoint="pre_state_commit",
    )

    assert result == "valid visual directive"
    assert len(llm.calls) == 2
    initial_surface = json.loads(llm.calls[0][1])["surface"]
    repair_surface = json.loads(llm.calls[1][1])["surface"]
    assert len(initial_surface["supporting_bids"]) < 7
    assert repair_surface["supporting_bids"] == (
        initial_surface["supporting_bids"]
    )


@pytest.mark.asyncio
async def test_surface_repair_overflow_is_typed_before_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-cap surface request never schedules an over-cap repair."""

    payload = {"required_context": "s" * 100}
    initial_chars = len(json.dumps(
        {"surface": payload},
        ensure_ascii=False,
        sort_keys=True,
    ))
    surface_stages_module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface_stages"
    )
    monkeypatch.setattr(
        surface_stages_module,
        "SURFACE_STAGE_PROMPT_CAP",
        initial_chars,
    )
    monkeypatch.setattr(
        surface_stages_module,
        "SURFACE_STAGE_REPAIR_PROMPT_CAP",
        initial_chars,
    )
    llm = _InvalidCandidateLLM()

    with pytest.raises(CognitionExecutionError) as error_info:
        await surface_stages_module._run_surface_stage(
            payload=payload,
            system_prompt="surface",
            llm=llm,
            config=object(),
            stage_name="visual",
            validator=surface_stages_module._validate_visual_result,
            safe_checkpoint="pre_state_commit",
        )

    assert error_info.value.error_code == "surface_visual_context_limit"
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_visual_surface_overflow_is_typed_for_optional_omission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visual cap overflow reaches the existing optional-stage failure owner."""

    surface_stages_module = importlib.import_module(
        "kazusa_ai_chatbot.cognition_core_v2.surface_stages"
    )
    monkeypatch.setattr(surface_stages_module, "SURFACE_STAGE_PROMPT_CAP", 100)
    llm = _NoCallLLM()

    with pytest.raises(CognitionExecutionError) as error_info:
        await surface_module.run_visual_surface_planning(
            _surface_input(),
            SimpleNamespace(
                llm=llm,
                visual_config=object(),
            ),
        )

    assert error_info.value.stage == "surface.visual"
    assert llm.call_count == 0
