"""Production-derived prompt budgeting and degraded-continuity contracts."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib
import json
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
from kazusa_ai_chatbot.cognition_core_v2 import surface as surface_module
from kazusa_ai_chatbot.cognition_core_v2 import workspace as workspace_module
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContextLimitError,
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
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
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
    canonical_identity_context,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "cognition_core_v2_prompt_budget_production_case.json"
)
INCIDENT_TIMESTAMP = "2026-07-28T04:19:18Z"


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
        }
        response = SimpleNamespace(content=json.dumps(result))
        return response


class _InvalidCandidateLLM:
    """Return one invalid object while retaining each attempted request."""

    def __init__(self) -> None:
        """Initialize an empty model-request ledger."""

        self.calls: list[list[str]] = []

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
        return SimpleNamespace(content="{}")


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
        evidence.append({
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
        })
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
            "semantic_text": semantic_text,
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["promoted_memory"]
            ),
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
            "description": f"{index:02d}" + ("s" * 498),
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


def _maximum_valid_cognition_input(scope: str) -> dict[str, Any]:
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
        "character_identity_context": canonical_identity_context(),
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
    payload = _maximum_valid_cognition_input(scope)
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

    payload = _maximum_valid_cognition_input("character")
    state = payload["mutable_state"]
    evidence = payload["evidence"]
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
        runtime_capability_limits=payload[
            "runtime_capability_limits"
        ],
    )
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
    assert len(human_payload) <= 8000
    prompt_payload = json.loads(human_payload)
    assert len(prompt_payload["evidence"]) == 8
    assert [
        row["handle"] for row in prompt_payload["evidence"]
    ] == question["evidence_handles"]
    assert "evidence" not in prompt_payload["state"]
    assert "permitted_delta_paths" not in prompt_payload["question"]


def test_appraisal_exact_cap_and_cap_plus_one_are_distinct() -> None:
    """The appraisal owner accepts 8,000 and rejects irreducible 8,001."""

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
    _pad_to_serialized_length(payload, question, 8000)

    fitted = semantic_appraisal_module._fit_appraisal_payload(payload)

    assert len(fitted) == 8000
    question["padding"] += "x"
    with pytest.raises(CognitionContextLimitError):
        semantic_appraisal_module._fit_appraisal_payload(payload)


def test_goal_exact_cap_and_cap_plus_one_are_distinct() -> None:
    """The goal owner accepts 24,000 and rejects irreducible 24,001."""

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
    _pad_to_serialized_length(payload, semantic_context, 24000)

    fitted = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt="",
    )

    assert len(fitted) == 24000
    semantic_context["padding"] += "x"
    with pytest.raises(PromptBudgetError):
        goal_cognition_module._fit_goal_prompt_payload(
            payload,
            system_prompt="",
        )


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
            "output_mode": "private_state",
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
        reduced_chars,
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
            "conversation_progress_constraints": [{
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
    assert fitted_payload["conversation_progress_constraints"][0][
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
        "conversation_progress_constraints": [{
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
    regeneration_system_prompt = (
        goal_cognition_module._required_selection_regeneration_prompt(
            "selection goal draft fields are not exact",
            {"e1", "e2"},
        )
    )

    initial_payload = goal_cognition_module._fit_goal_prompt_payload(
        payload,
        system_prompt=initial_system_prompt,
    )
    regeneration_payload = goal_cognition_module._fit_goal_prompt_payload(
        payload,
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
    assert len(regeneration_payload) < len(initial_payload)


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
        semantic_appraisal_module._fit_appraisal_payload(payload)

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
        [{"question_id": "q:epistemic_comparison_memory"}],
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
    assert len(human_payload) <= 24000
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
            "selection_owner_role": "current character",
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


def test_required_selection_has_no_semantic_repair_surface() -> None:
    """Keep evaluator and replacement ownership absent from the module."""

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
        "REQUIRED_SELECTION_REPAIR_PROMPT",
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
    )

    assert result["primary_branch_id"] == "ordinary_response"
    assert result["supporting_bids"] == []
    assert result["suppressed_branch_ids"] == ["social_care"]
    assert llm.call_count == 0


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
    prompt_payload = {
        "bids": {
            f"b{index}": {
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
        initial_chars,
    )
    llm = _InvalidCandidateLLM()

    result = await workspace_module.collapse_bids(
        bids,
        SimpleNamespace(
            llm=llm,
            workspace_collapse_config=object(),
        ),
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
        runtime_capability_limits=[],
    )

    assert decision == (
        action_selection_module._empty_action_plan_decision()
    )
    assert len(llm.calls) == 1


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
    assert len(llm.calls) == 1


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
async def test_appraisal_repair_overflow_omits_before_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A near-cap appraisal repair becomes typed family-local omission."""

    state, evidence, projection, questions = _production_appraisal_context()
    monkeypatch.setattr(
        semantic_appraisal_module,
        "SEMANTIC_APPRAISAL_PROMPT_CAP",
        100,
    )
    monkeypatch.setattr(
        semantic_appraisal_module,
        "_fit_appraisal_payload",
        lambda payload: "p" * 100,
    )
    llm = _InvalidCandidateLLM()

    with pytest.raises(
        CognitionContextLimitError,
        match="repair context",
    ):
        await appraise_semantic_question(
            questions[0],
            evidence,
            projection,
            _appraisal_services(llm),
            validation_state=state,
        )

    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_text_surface_overflow_returns_validated_degraded_output() -> None:
    """Text cap overflow projects the committed intention without model prose."""

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
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_visual_surface_overflow_is_typed_for_optional_omission() -> None:
    """Visual cap overflow reaches the existing optional-stage failure owner."""

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
