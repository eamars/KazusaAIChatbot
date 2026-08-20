"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py."""

from __future__ import annotations

from importlib import import_module

import pytest

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.action_spec.registry import (
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_core_selector import (
    run_cognition as selected_run_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.config import (
    CognitionRouteSettingV1,
    CognitionV3RouteSettingsV1,
)
from tests.test_cognition_chain_connector_mapping import (
    NOW,
    _core_output,
    _global_state,
)

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"
EXPECTED_SYMBOLS = [
    "build_cognition_input_from_global_state",
    "build_scene_context_from_global_state",
]
COGNITION_V2_ROUTE_KEYS = (
    "appraisal_event_agency",
    "appraisal_relationship_social",
    "appraisal_moral_identity",
    "appraisal_goal_threat_outcome",
    "appraisal_epistemic_comparison_memory",
    "appraisal_existential_drive",
    "goal_ordinary_response",
    "goal_active_branch",
    "workspace_collapse",
    "action_planning",
    "action_authorization",
    "resolver_authorization",
)


def test_persona_supervisor2_cognition_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_connector_builds_selected_engine_services_without_inactive_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the selected engine loader contributes model bindings."""

    v2_settings = {
        key: CognitionRouteSettingV1(
            base_url=f"http://{key}.example/v1",
            api_key=f"{key}-key",
            model=f"{key}-model",
            max_completion_tokens=8_192,
            thinking_enabled=False,
            context_window_tokens=None,
        )
        for key in COGNITION_V2_ROUTE_KEYS
    }
    v3_settings = CognitionV3RouteSettingsV1(
        chain=CognitionRouteSettingV1(
            base_url="http://chain.example/v1",
            api_key="chain-key",
            model="chain-model",
            max_completion_tokens=8_192,
            thinking_enabled=False,
            context_window_tokens=50_176,
        ),
        sidecar=None,
        subconscious_enabled=False,
        appraisal_group_count=6,
        turn_deadline_seconds=417,
    )

    def inactive_v3_loader() -> CognitionV3RouteSettingsV1:
        raise AssertionError("V2 construction read the inactive V3 family")

    monkeypatch.setattr(cognition_module, "COGNITION_CORE_ENGINE", "v2")
    monkeypatch.setattr(
        cognition_module,
        "load_cognition_v2_route_settings",
        lambda: v2_settings,
    )
    monkeypatch.setattr(
        cognition_module,
        "load_cognition_v3_route_settings",
        inactive_v3_loader,
    )

    v2_services = cognition_module.build_cognition_core_services()

    assert isinstance(v2_services, CognitionCoreServicesV2)
    assert v2_services.appraisal_event_agency_config.context_window_tokens is None
    assert v2_services.goal_active_branch_config.model == (
        "goal_active_branch-model"
    )

    def inactive_v2_loader() -> dict[str, CognitionRouteSettingV1]:
        raise AssertionError("V3 construction read the inactive V2 family")

    monkeypatch.setattr(cognition_module, "COGNITION_CORE_ENGINE", "v3")
    monkeypatch.setattr(
        cognition_module,
        "load_cognition_v2_route_settings",
        inactive_v2_loader,
    )
    monkeypatch.setattr(
        cognition_module,
        "load_cognition_v3_route_settings",
        lambda: v3_settings,
    )

    v3_services = cognition_module.build_cognition_core_services()

    assert isinstance(v3_services, CognitionChainServicesV3)
    assert v3_services.chain_lane.route_name == "COGNITION_V3_CHAIN_LLM"
    assert v3_services.chain_lane.context_window_tokens == 50_176
    assert v3_services.sidecar_lane is None
    assert v3_services.appraisal_group_count == 6
    assert v3_services.turn_deadline_seconds == 417


@pytest.mark.asyncio
async def test_user_cognition_commit_uses_compare_and_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit user cognition through the complete-state CAS boundary."""

    previous = build_acquaintance_user_state(
        global_user_id="node-cas-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    replacement = build_acquaintance_user_state(
        global_user_id="node-cas-user",
        updated_at="2026-08-18T00:01:00Z",
    )
    captured: dict[str, object] = {}

    async def compare_and_replace(
        owner_key: str,
        expected_state: dict[str, object],
        replacement_state: dict[str, object],
    ) -> bool:
        """Capture the state boundary and acknowledge the commit."""

        captured.update({
            "owner_key": owner_key,
            "expected_state": expected_state,
            "replacement_state": replacement_state,
        })
        return True

    async def record_commit(*args: object, **kwargs: object) -> None:
        """Keep the unit test focused on the persistence call."""

        del args, kwargs

    monkeypatch.setattr(
        cognition_module,
        "compare_and_replace_user_cognition_state",
        compare_and_replace,
    )
    monkeypatch.setattr(
        cognition_module,
        "_record_state_commit_event",
        record_commit,
    )

    await cognition_module._commit_cognition_state({
        "intention": {"selected_branch_id": "ordinary_response"},
        "state_update": {
            "state_scope": "user",
            "owner_key": "node-cas-user",
            "expected_previous_state": previous,
            "replacement_state": replacement,
        },
    })

    assert captured == {
        "owner_key": "node-cas-user",
        "expected_state": previous,
        "replacement_state": replacement,
    }


def test_group_self_cognition_proposal_materializes_existing_speak_surface(
    monkeypatch,
) -> None:
    """A speech intention uses the existing canonical speak materializer."""

    captured: list[list[dict[str, object]]] = []

    def materialize(
        requests: list[dict[str, object]],
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        del state
        captured.append(requests)
        return [{"kind": request["capability"]} for request in requests]

    monkeypatch.setattr(
        cognition_module,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "action_requests": [],
        "intention": {
            "route": "speech",
            "intention": "intervene in the current group scene",
            "target_roles": [],
            "reason": "The current scene supports a bounded intervention.",
        },
        "admitted_bid": {"evidence_handles": ["e1"]},
    }

    action_specs = cognition_module._materialize_v2_action_requests(
        output,
        {},
    )

    assert len(captured) == 2
    assert action_specs == [{"kind": SPEAK_CAPABILITY}]
    assert captured[1][0]["capability"] == SPEAK_CAPABILITY
    assert captured[1][0]["evidence_handles"] == ["e1"]


def test_promoted_reflection_preserves_source_updated_at() -> None:
    """Promoted rows retain their valid source time and omit invalid rows."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_lore': [{
                'memory_name': 'world context',
                'content': 'the setting remains stable',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': 'not-a-timestamp',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert len(evidence) == 1
    assert evidence[0]['evidence_ref']['occurred_at'] == (
        '2026-07-29T23:00:00Z'
    )


def test_promoted_self_guidance_is_goal_only_conditional_context() -> None:
    """Self-guidance carries conditional authority and no current fact role."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert evidence[0]['authority'] == 'conditional_character_guidance'
    assert evidence[0]['evidence_ref']['source_id'] == (
        'promoted-reflection:self_guidance:1'
    )


def test_future_speak_v2_bridge_preserves_validated_authority_proposal(
    monkeypatch,
) -> None:
    """The runtime V2 bridge keeps the validated proposal on future_speak."""

    captured: list[list[dict[str, object]]] = []
    proposal = {
        "schema_version": "scheduled_authority_proposal.v1",
        "temporal_alignment": "aligned",
        "authorized_content_summary": "在约定时间开始补偿考核。",
        "authorized_detail_refs": [
            {
                "evidence_handle": "e1",
                "semantic_summary": (
                    "当前对话明确约定在该时间开始补偿考核。"
                ),
                "provenance_role": "current_event",
            }
        ],
    }

    def materialize(
        requests: list[dict[str, object]],
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        del state
        captured.append(requests)
        return [
            {"kind": request["capability"]}
            for request in requests
            if isinstance(request, dict)
        ]

    monkeypatch.setattr(
        cognition_module,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "action_requests": [
            {
                "action_kind": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-10 13:00",
                "context_ref": "current episode",
                "semantic_goal": "在约定时间开始补偿考核。",
                "reason": "用户要求在未来时间开始补偿考核。",
                "target_roles": [],
                "evidence_handles": ["e1"],
                "scheduled_authority_proposal": proposal,
            }
        ],
        "intention": {
            "route": "silence",
            "intention": "stay silent",
            "target_roles": [],
            "reason": "No visible surface selected.",
        },
    }

    action_specs = cognition_module._materialize_v2_action_requests(
        output,
        {},
    )

    assert len(captured) == 1
    materialized_request = captured[0][0]
    assert materialized_request["capability"] == FUTURE_SPEAK_CAPABILITY
    assert materialized_request["scheduled_authority_proposal"] == proposal
    assert action_specs == [{"kind": FUTURE_SPEAK_CAPABILITY}]


async def _noop_event_recorder(*args: object, **kwargs: object) -> None:
    """Keep selected-engine tests focused on the persistence boundary."""

    del args, kwargs


@pytest.mark.asyncio
async def test_connector_calls_selected_engine_with_canonical_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connector hands one canonical input to the selected engine."""

    assert cognition_module.run_cognition is selected_run_cognition

    state = _global_state()
    captured_inputs: list[dict[str, object]] = []

    async def fake_run_cognition(cognition_input, services):
        del services
        captured_inputs.append(dict(cognition_input))
        return _core_output()

    async def fetch_user_state(owner_key: str) -> dict[str, object]:
        """Serve the fixture user base state without a database read."""

        assert owner_key == "user-1"
        return build_acquaintance_user_state(
            global_user_id=owner_key,
            updated_at=NOW,
        )

    async def fetch_character_state() -> dict[str, object]:
        """Serve the fixture character base state without a database read."""

        return build_character_production_state(updated_at=NOW)

    monkeypatch.setattr(
        cognition_module,
        "run_cognition",
        fake_run_cognition,
    )
    monkeypatch.setattr(
        cognition_module,
        "get_user_cognition_state",
        fetch_user_state,
    )
    monkeypatch.setattr(
        cognition_module,
        "get_character_cognition_state",
        fetch_character_state,
    )
    monkeypatch.setattr(
        cognition_module,
        "record_continuity_boundary_event",
        _noop_event_recorder,
    )

    await cognition_module.call_cognition_subgraph(state, commit=False)

    assert len(captured_inputs) == 1
    captured = captured_inputs[0]
    assert captured["schema_version"] == "cognition_core_input.v2"
    assert captured["state_scope"] == "user"
    assert captured["episode"]["episode_id"] == "episode-1"
    assert len(captured["evidence"]) == 1
    evidence_row = captured["evidence"][0]
    assert evidence_row["evidence_handle"] == "e1"
    assert evidence_row["evidence_ref"]["source_kind"] == "episode"


@pytest.mark.asyncio
async def test_selected_engine_output_commits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selected-engine output commits through the CAS boundary once."""

    state = _global_state()
    cas_calls: list[dict[str, object]] = []

    async def fake_run_cognition(cognition_input, services):
        del cognition_input, services
        return _core_output()

    async def fetch_user_state(owner_key: str) -> dict[str, object]:
        """Serve the fixture user base state without a database read."""

        assert owner_key == "user-1"
        return build_acquaintance_user_state(
            global_user_id=owner_key,
            updated_at=NOW,
        )

    async def fetch_character_state() -> dict[str, object]:
        """Serve the fixture character base state without a database read."""

        return build_character_production_state(updated_at=NOW)

    async def compare_and_replace(
        owner_key: str,
        expected_previous_state: dict[str, object],
        replacement_state: dict[str, object],
    ) -> bool:
        """Capture the state boundary and acknowledge the commit."""

        cas_calls.append({
            "owner_key": owner_key,
            "expected_previous_state": expected_previous_state,
            "replacement_state": replacement_state,
        })
        return True

    monkeypatch.setattr(
        cognition_module,
        "run_cognition",
        fake_run_cognition,
    )
    monkeypatch.setattr(
        cognition_module,
        "get_user_cognition_state",
        fetch_user_state,
    )
    monkeypatch.setattr(
        cognition_module,
        "get_character_cognition_state",
        fetch_character_state,
    )
    monkeypatch.setattr(
        cognition_module,
        "compare_and_replace_user_cognition_state",
        compare_and_replace,
    )
    monkeypatch.setattr(
        cognition_module,
        "_record_state_commit_event",
        _noop_event_recorder,
    )
    monkeypatch.setattr(
        cognition_module,
        "record_continuity_boundary_event",
        _noop_event_recorder,
    )

    await cognition_module.call_cognition_subgraph(state, commit=True)

    assert len(cas_calls) == 1
    cas_call = cas_calls[0]
    assert cas_call["owner_key"] == "user-1"
    state_update = _core_output()["state_update"]
    assert cas_call["replacement_state"] == state_update["replacement_state"]
