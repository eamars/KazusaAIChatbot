"""Deterministic contracts for stage-owned Cognition Core V2 model routes."""

from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage
import pytest

from kazusa_ai_chatbot import config as config_module
from kazusa_ai_chatbot.cognition_core_v2 import action_authorization
from kazusa_ai_chatbot.cognition_core_v2 import action_selection
from kazusa_ai_chatbot.cognition_core_v2 import goal_cognition
from kazusa_ai_chatbot.cognition_core_v2 import semantic_appraisal
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    authorize_resolver_requests,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    PromptProjectionV2,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig


STAGE_ROUTES = (
    (
        "appraisal_event_agency_config",
        "COGNITION_LLM_APPRAISAL_EVENT_AGENCY",
    ),
    (
        "appraisal_relationship_social_config",
        "COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL",
    ),
    (
        "appraisal_moral_identity_config",
        "COGNITION_LLM_APPRAISAL_MORAL_IDENTITY",
    ),
    (
        "appraisal_goal_threat_outcome_config",
        "COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME",
    ),
    (
        "appraisal_epistemic_comparison_memory_config",
        "COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY",
    ),
    (
        "appraisal_existential_drive_config",
        "COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE",
    ),
    (
        "goal_ordinary_response_config",
        "COGNITION_LLM_GOAL_ORDINARY_RESPONSE",
    ),
    (
        "goal_active_branch_config",
        "COGNITION_LLM_GOAL_ACTIVE_BRANCH",
    ),
    (
        "workspace_collapse_config",
        "COGNITION_LLM_WORKSPACE_COLLAPSE",
    ),
    (
        "action_planning_config",
        "COGNITION_LLM_ACTION_PLANNING",
    ),
    (
        "action_authorization_config",
        "COGNITION_LLM_ACTION_AUTHORIZATION",
    ),
    (
        "resolver_authorization_config",
        "COGNITION_LLM_RESOLVER_AUTHORIZATION",
    ),
)

APPRAISAL_CONFIG_FIELDS = {
    "event_agency": "appraisal_event_agency_config",
    "relationship_social": "appraisal_relationship_social_config",
    "moral_identity": "appraisal_moral_identity_config",
    "goal_threat_outcome": "appraisal_goal_threat_outcome_config",
    "epistemic_comparison_memory": (
        "appraisal_epistemic_comparison_memory_config"
    ),
    "existential_drive": "appraisal_existential_drive_config",
}


def test_appraisal_routes_share_the_calibrated_completion_default() -> None:
    """The six bounded appraisal schemas use the calibrated code default."""

    assert getattr(
        config_module,
        "SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS",
        None,
    ) == 2_048


class _CapturingInvoker:
    """Return queued JSON responses while preserving every selected config."""

    def __init__(self, responses: list[object]) -> None:
        """Store response candidates in call order."""

        self.responses = list(responses)
        self.configs: list[LLMCallConfig] = []
        self.messages: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: LLMCallConfig,
    ) -> SimpleNamespace:
        """Capture the config and return the next queued response."""

        self.messages.append(list(messages))
        self.configs.append(config)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        content = json.dumps(response)
        result = SimpleNamespace(content=content)
        return result


def _config(route_name: str) -> LLMCallConfig:
    """Build one unique stage config for deterministic identity assertions."""

    config = LLMCallConfig(
        stage_name=route_name.lower(),
        route_name=route_name,
        base_url=f"http://{route_name.lower()}.invalid/v1",
        api_key="test-key",
        model=f"{route_name.lower()}-model",
        temperature=0.1,
        top_p=0.7,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
        thinking=LLMThinkingConfig(enabled=False),
    )
    return config


def _services(llm: _CapturingInvoker) -> CognitionCoreServicesV2:
    """Build the exact stage-owned service contract under test."""

    services = CognitionCoreServicesV2(
        llm=llm,
        appraisal_event_agency_config=_config(
            "COGNITION_LLM_APPRAISAL_EVENT_AGENCY"
        ),
        appraisal_relationship_social_config=_config(
            "COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL"
        ),
        appraisal_moral_identity_config=_config(
            "COGNITION_LLM_APPRAISAL_MORAL_IDENTITY"
        ),
        appraisal_goal_threat_outcome_config=_config(
            "COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME"
        ),
        appraisal_epistemic_comparison_memory_config=_config(
            "COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY"
        ),
        appraisal_existential_drive_config=_config(
            "COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE"
        ),
        goal_ordinary_response_config=_config(
            "COGNITION_LLM_GOAL_ORDINARY_RESPONSE"
        ),
        goal_active_branch_config=_config(
            "COGNITION_LLM_GOAL_ACTIVE_BRANCH"
        ),
        workspace_collapse_config=_config(
            "COGNITION_LLM_WORKSPACE_COLLAPSE"
        ),
        action_planning_config=_config(
            "COGNITION_LLM_ACTION_PLANNING"
        ),
        action_authorization_config=_config(
            "COGNITION_LLM_ACTION_AUTHORIZATION"
        ),
        resolver_authorization_config=_config(
            "COGNITION_LLM_RESOLVER_AUTHORIZATION"
        ),
    )
    return services


def _goal_draft() -> dict[str, object]:
    """Build one structurally valid goal-cognition response."""

    return {
        "intention": "respond to the current event",
        "desired_outcome": "advance the current interaction",
        "concrete_detail": "answer from the admitted evidence",
        "reason": "the current event supports a response",
        "private_monologue": "I want to answer this directly.",
        "target_role_handles": [],
        "evidence_handles": [],
        "expected_consequences": ["the interaction advances"],
        "confidence": "high",
    }


def _ordinary_goal_draft() -> dict[str, object]:
    """Build one structurally valid ordinary-response goal result."""

    draft = _goal_draft()
    draft["relational_willingness"] = {
        "schema_version": "relational_willingness.v2",
        "applicability": "not_relationship_sensitive",
        "stance": "not_applicable",
        "current_user_relationship_state": "not_applicable",
        "reason": "当前回合证据不涉及关系许可判断",
        "evidence_handles": ["e1"],
    }
    return draft


def _selection_goal_draft() -> dict[str, object]:
    """Build one authoritative required-selection producer response."""

    return {
        "selection": "choose one concrete current action",
        "reason": "the typed operation gives the character selection ownership",
        "private_monologue": "I will make this choice directly.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the user receives one concrete choice"],
        "confidence": "high",
    }


def _bid(branch_id: str) -> dict[str, object]:
    """Build one complete admitted bid for routing-only tests."""

    return {
        "branch_id": branch_id,
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        **_goal_draft(),
        "target_roles": [],
    }


def _evidence() -> dict[str, object]:
    """Build one bounded current-event evidence row."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-27T00:00:00Z",
            "semantic_summary": "the user made a current request",
        },
        "semantic_text": "the user made a current request",
        "visible_to": ["q:event_agency"],
    }


def test_service_contract_exposes_only_stage_owned_configs() -> None:
    """The aggregate config fields are replaced by thirteen stage fields."""

    field_names = tuple(field.name for field in fields(CognitionCoreServicesV2))

    assert field_names == ("llm", *(name for name, _ in STAGE_ROUTES))
    assert {
        "appraisal_" + "config",
        "goal_cognition_" + "config",
        "collapse_" + "config",
        "action_selection_" + "config",
    }.isdisjoint(field_names)


def test_production_builder_binds_each_exact_stage_route() -> None:
    """The connector builds every service field from its named route bundle."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_core_services,
    )

    services = build_cognition_core_services()
    route_names = {
        field_name: getattr(services, field_name).route_name
        for field_name, _ in STAGE_ROUTES
    }

    assert route_names == dict(STAGE_ROUTES)


def test_l3_surface_retains_the_generic_cognition_route() -> None:
    """The outside-Core L3 surface keeps its existing generic route binding."""

    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition
    from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface

    assert (
        persona_supervisor2_l3_surface._cognition_llm_config
        is persona_supervisor2_cognition._cognition_llm_config
    )
    assert (
        persona_supervisor2_l3_surface._cognition_llm_config.route_name
        == "COGNITION_LLM"
    )


def test_first_wave_route_split_matches_existing_parallel_owners() -> None:
    """The initial concurrent owners retain the approved two route groups."""

    first_source_fields = {
        "appraisal_event_agency_config",
        "goal_ordinary_response_config",
    }
    second_source_fields = {
        "appraisal_relationship_social_config",
        "appraisal_moral_identity_config",
        "appraisal_goal_threat_outcome_config",
        "appraisal_epistemic_comparison_memory_config",
        "appraisal_existential_drive_config",
        "goal_active_branch_config",
    }

    assert first_source_fields.isdisjoint(second_source_fields)
    assert first_source_fields | second_source_fields == {
        *APPRAISAL_CONFIG_FIELDS.values(),
        "goal_ordinary_response_config",
        "goal_active_branch_config",
    }


@pytest.mark.asyncio
async def test_each_appraisal_family_reuses_its_route_for_repair_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each appraisal family invokes and validates with one stage config."""

    validation_rows: list[dict[str, object]] = []
    monkeypatch.setattr(
        semantic_appraisal,
        "capture_validation_stage",
        lambda **values: validation_rows.append(values),
    )

    for question_kind, config_field in APPRAISAL_CONFIG_FIELDS.items():
        question_id = f"q:{question_kind}"
        valid_response = {
            "question_id": question_id,
            "proposition": None,
            "delta": None,
        }
        llm = _CapturingInvoker([
            {"invalid": "shape"},
            valid_response,
        ])
        services = _services(llm)
        expected_config = getattr(services, config_field)
        row_start = len(validation_rows)

        result = await semantic_appraisal.appraise_semantic_question(
            {
                "question_id": question_id,
                "question_kind": question_kind,
                "semantic_question": "Inspect this bounded semantic family.",
                "evidence_handles": [],
                "permitted_role_handles": [],
                "permitted_delta_paths": [],
            },
            [],
            PromptProjectionV2(payload={}, handle_to_ref={}),
            services,
            validation_state=build_acquaintance_user_state(
                global_user_id="routing-user",
                updated_at="2026-07-28T00:00:00Z",
            ),
        )

        assert result["question_id"] == question_id
        assert llm.configs == [expected_config, expected_config]
        assert [
            row["config"] for row in validation_rows[row_start:]
        ] == [expected_config, expected_config]


@pytest.mark.asyncio
async def test_goal_branches_reuse_their_own_route_for_repairs_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary and active branches keep generation, repair, and trace aligned."""

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        goal_cognition.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        goal_cognition.llm_tracing,
        "current_trace_id",
        lambda: "trace-goal-routing",
    )
    cases = (
        ("ordinary_response", "goal_ordinary_response_config"),
        ("autonomy_boundary", "goal_active_branch_config"),
    )

    for branch_id, config_field in cases:
        draft = (
            _ordinary_goal_draft()
            if branch_id == "ordinary_response"
            else _goal_draft()
        )
        invalid_response = {**draft, "unknown": "field"}
        llm = _CapturingInvoker([invalid_response, draft])
        services = _services(llm)
        expected_config = getattr(services, config_field)
        trace_start = trace_recorder.await_count
        evidence = [_evidence()] if branch_id == "ordinary_response" else []

        result = await goal_cognition.run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS[branch_id],
            {"scope": "user", "kind": "goal", "entity_id": "g1"},
            {"_role_bindings": {}, "role_summaries": {}},
            evidence,
            services,
        )

        assert result["branch_id"] == branch_id
        assert llm.configs == [expected_config, expected_config]
        trace_calls = trace_recorder.await_args_list[trace_start:]
        assert [call.kwargs["route_name"] for call in trace_calls] == [
            expected_config.route_name,
            expected_config.route_name,
        ]
        assert [call.kwargs["model_name"] for call in trace_calls] == [
            expected_config.model,
            expected_config.model,
        ]


@pytest.mark.asyncio
async def test_selection_producer_retry_reuses_goal_route_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural retries stay on the one producing goal route."""

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        goal_cognition.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        goal_cognition.llm_tracing,
        "current_trace_id",
        lambda: "trace-selection-routing",
    )
    llm = _CapturingInvoker([
        {"selection": ""},
        _selection_goal_draft(),
    ])
    services = _services(llm)
    expected_config = services.goal_ordinary_response_config
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-27T00:00:00Z",
            "semantic_summary": "the character must make one selection",
        },
        "semantic_text": json.dumps({
            "role_explicit_content": "the character must make one selection",
            "response_operation": {
                "selection_required": True,
                "selection_owner_role": "character",
            },
        }),
        "visible_to": ["q:event_agency"],
    }]

    result = await goal_cognition.run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["autonomy_boundary"],
        {"scope": "user", "kind": "goal", "entity_id": "g1"},
        {"_role_bindings": {}, "role_summaries": {}},
        evidence,
        services,
    )

    assert result["intention"] == _selection_goal_draft()["selection"]
    assert llm.configs == [expected_config, expected_config]
    assert (
        llm.messages[0][0].content
        == goal_cognition._ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
    )
    assert llm.messages[1][0].content.startswith(
        goal_cognition._ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
    )
    assert [
        call.kwargs["route_name"]
        for call in trace_recorder.await_args_list
    ] == [expected_config.route_name, expected_config.route_name]


def test_required_selection_has_no_independent_model_route() -> None:
    """Keep selection production on the existing dense ordinary-goal route."""

    service_fields = {
        field.name for field in fields(CognitionCoreServicesV2)
    }
    assert "required_selection_verifier_config" not in service_fields
    assert all(
        route_name != "COGNITION_LLM_REQUIRED_SELECTION_VERIFIER"
        for _field_name, route_name in STAGE_ROUTES
    )


@pytest.mark.asyncio
async def test_workspace_collapse_reuses_its_route_for_repair() -> None:
    """Collapse replacement calls remain on the workspace-collapse route."""

    llm = _CapturingInvoker([
        {"primary_bid_handle": "b1"},
        {
            "primary_bid_handle": "b1",
            "supporting_bid_handles": ["b2"],
            "suppressed_bid_handles": [],
        },
    ])
    services = _services(llm)

    result = await collapse_bids(
        [_bid("ordinary_response"), _bid("autonomy_boundary")],
        services,
        current_event=[{
            "handle": "e1",
            "source_kind": "episode",
            "semantic_text": "the user made a current request",
        }],
        goal_context_by_ref={
            "g1": {
                "goal_handle": "g1",
                "goal_kind": "autonomy_boundary",
                "description": "preserve the current autonomy boundary",
                "status": "pursuing",
                "salience": 50,
                "importance": 80,
                "progress": 10,
                "obstruction": 30,
                "urgency": 60,
            },
        },
    )

    expected_config = services.workspace_collapse_config
    assert result["primary_bid"]["branch_id"] == "ordinary_response"
    assert llm.configs == [expected_config, expected_config]


@pytest.mark.asyncio
async def test_action_planning_reuses_its_route_for_repair_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner replacements and protected traces use action planning config."""

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        action_selection.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        action_selection.llm_tracing,
        "current_trace_id",
        lambda: "trace-planning-routing",
    )
    valid_response = {
        "action_requests": [],
        "resolver_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "goal_resolution": "answerable_now",
    }
    llm = _CapturingInvoker([{"invalid": "shape"}, valid_response])
    services = _services(llm)

    result = await action_selection._invoke_action_planner(
        services=services,
        messages=[HumanMessage(content="{}")],
        bid_handles={},
        action_handles={},
        resolver_handles={},
        current_goal_progress=None,
        runtime_capability_limits=(),
    )

    expected_config = services.action_planning_config
    assert result == valid_response
    assert llm.configs == [expected_config, expected_config]
    assert [
        call.kwargs["route_name"]
        for call in trace_recorder.await_args_list
    ] == [expected_config.route_name, expected_config.route_name]


@pytest.mark.asyncio
async def test_action_authorization_reuses_its_route_for_repair_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action authorization calls and traces use the action-owned route."""

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        action_authorization.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        action_authorization.llm_tracing,
        "current_trace_id",
        lambda: "trace-action-authorization-routing",
    )
    llm = _CapturingInvoker([
        {"invalid": "shape"},
        {"decisions": {"c1": True}},
    ])
    services = _services(llm)
    request = {
        "bid_handle": "b1",
        "action_handle": "a1",
        "decision": "enqueue",
        "semantic_goal": "record the accepted bounded work",
        "reason": "the current request supports this effect",
    }

    result = await authorize_action_requests(
        action_requests=[request],
        bid_handles={"b1": {**_bid("ordinary_response"), "evidence_handles": ["e1"]}},
        evidence=[_evidence()],
        action_handles={"a1": {
            "action_kind": "background_work_request",
            "capability": "Record explicitly accepted bounded delayed work.",
            "permission": "allowed",
            "decision_mode": "closed",
            "allowed_decisions": ["enqueue"],
            "default_decision": "enqueue",
            "decision_pattern": "",
            "context_ref": "",
            "target_roles": [],
        }},
        runtime_capability_limits=(),
        services=services,
    )

    expected_config = services.action_authorization_config
    assert result == [request]
    assert llm.configs == [expected_config, expected_config]
    assert [
        call.kwargs["route_name"]
        for call in trace_recorder.await_args_list
    ] == [expected_config.route_name, expected_config.route_name]


@pytest.mark.asyncio
async def test_resolver_authorization_reuses_its_route_for_repair_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver authorization calls and traces use the resolver-owned route."""

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        action_authorization.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        action_authorization.llm_tracing,
        "current_trace_id",
        lambda: "trace-resolver-authorization-routing",
    )
    llm = _CapturingInvoker([
        {"invalid": "shape"},
        {"decisions": {"c1": True}},
    ])
    services = _services(llm)
    request = {
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "retrieve the unresolved local context",
        "reason": "the admitted bid still needs this evidence",
    }

    result = await authorize_resolver_requests(
        resolver_requests=[request],
        bid_handles={"b1": {**_bid("ordinary_response"), "evidence_handles": ["e1"]}},
        evidence=[_evidence()],
        resolver_handles={"r1": {
            "capability": "task_resolution_request",
            "semantic_capability": "Retrieve relevant local context.",
            "availability": "available",
        }},
        resolver_context="resolver_status=idle",
        services=services,
    )

    expected_config = services.resolver_authorization_config
    assert result == [request]
    assert llm.configs == [expected_config, expected_config]
    assert [
        call.kwargs["route_name"]
        for call in trace_recorder.await_args_list
    ] == [expected_config.route_name, expected_config.route_name]
