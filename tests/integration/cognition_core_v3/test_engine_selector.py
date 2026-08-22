"""Closed engine-selection behavior for the cognition core.

These cases verify the process-level selector without model calls: the
connector and selector share one selected-engine binding, a reloaded selector
resolves exactly the configured engine entrypoint, and an unknown engine value
fails startup with an error that names the allowed set.
"""

from __future__ import annotations

import importlib
import inspect
from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.cognition_core_selector as cognition_core_selector
import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as connector_module
import kazusa_ai_chatbot.self_cognition.runner as idle_runner
from kazusa_ai_chatbot import config
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    run_cognition as v2_run_cognition,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    run_cognition as v3_run_cognition,
)

_V2_ROUTE_KEYS = (
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


def test_connector_and_selector_share_one_selected_engine_binding() -> None:
    """Live and idle connector paths call the selector's one entrypoint."""

    assert connector_module.run_cognition is cognition_core_selector.run_cognition

    if config.COGNITION_CORE_ENGINE == "v2":
        assert cognition_core_selector.run_cognition is v2_run_cognition
    else:
        assert cognition_core_selector.run_cognition is v3_run_cognition


def test_reloaded_selector_resolves_the_configured_engine() -> None:
    """A reloaded selector binds exactly the configured engine entrypoint."""

    original_engine = config.COGNITION_CORE_ENGINE
    try:
        config.COGNITION_CORE_ENGINE = "v2"
        importlib.reload(cognition_core_selector)
        assert (
            cognition_core_selector.run_cognition is v2_run_cognition
        )

        config.COGNITION_CORE_ENGINE = "v3"
        importlib.reload(cognition_core_selector)
        assert cognition_core_selector.run_cognition is v3_run_cognition
        parameters = list(
            inspect.signature(v3_run_cognition).parameters
        )
        assert parameters == ["input_payload", "services"]
    finally:
        config.COGNITION_CORE_ENGINE = original_engine
        importlib.reload(cognition_core_selector)


def test_unknown_engine_value_fails_startup_naming_the_allowed_set() -> None:
    """An unknown engine value raises a ValueError naming v2 and v3."""

    with pytest.raises(ValueError) as excinfo:
        cognition_core_selector.resolve_engine_module("v9")

    message = str(excinfo.value)
    assert "Cognition core engine must be one of:" in message
    assert "v2" in message
    assert "v3" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine", "expected_type"),
    [
        ("v2", CognitionCoreServicesV2),
        ("v3", CognitionChainServicesV3),
    ],
)
async def test_live_and_idle_connectors_construct_the_same_selected_engine_family(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    expected_type: type[object],
) -> None:
    """The idle resolver calls the live connector's selected service builder."""

    v2_settings = {
        key: config.CognitionRouteSettingV1(
            base_url=f"http://{key}.example/v1",
            api_key=f"{key}-key",
            model=f"{key}-model",
            max_completion_tokens=8_192,
            thinking_enabled=False,
            context_window_tokens=None,
        )
        for key in _V2_ROUTE_KEYS
    }
    v3_settings = config.CognitionV3RouteSettingsV1(
        chain=config.CognitionRouteSettingV1(
            base_url="http://chain.example/v1",
            api_key="chain-key",
            model="chain-model",
            max_completion_tokens=8_192,
            thinking_enabled=False,
            context_window_tokens=50_176,
        ),
        sidecar=None,
        subconscious_enabled=False,
        turn_deadline_seconds=240,
    )
    monkeypatch.setattr(connector_module, "COGNITION_CORE_ENGINE", engine)
    monkeypatch.setattr(
        connector_module,
        "load_cognition_v2_route_settings",
        lambda: v2_settings,
    )
    monkeypatch.setattr(
        connector_module,
        "load_cognition_v3_route_settings",
        lambda: v3_settings,
    )

    live_services = connector_module.build_cognition_core_services()
    observed_idle_types: list[type[object]] = []
    assert idle_runner.call_cognition_subgraph is (
        connector_module.call_cognition_subgraph
    )

    async def empty_mapping(*_args: object, **_kwargs: object) -> dict:
        return {}

    async def run_selected_engine(
        _input_payload: object,
        services: object,
    ) -> dict:
        observed_idle_types.append(type(services))
        return {"state_update": {"state_scope": "user"}}

    idle_cognition_input = {
        "evidence": [],
        "scene_context": {},
        "group_engagement_action_context": {
            "engagement_guidelines": [],
            "confidence": "",
        },
    }
    monkeypatch.setattr(
        connector_module,
        "validate_cognitive_episode_v1",
        lambda _episode: None,
    )
    monkeypatch.setattr(
        connector_module,
        "_scope_caller",
        lambda _episode: object(),
    )
    monkeypatch.setattr(
        connector_module,
        "resolve_state_scope",
        lambda *_args, **_kwargs: ("user", "idle-owner"),
    )
    monkeypatch.setattr(
        connector_module,
        "_state_has_episode_identity_snapshot",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        connector_module,
        "_is_group_self_cognition_state",
        lambda _state: False,
    )
    monkeypatch.setattr(
        connector_module,
        "_supports_first_cycle_shared_memory_prewarm",
        lambda _state: False,
    )
    monkeypatch.setattr(
        connector_module,
        "get_user_cognition_state",
        empty_mapping,
    )
    monkeypatch.setattr(
        connector_module,
        "get_character_cognition_state",
        empty_mapping,
    )
    monkeypatch.setattr(
        connector_module,
        "build_cognition_input_from_global_state",
        lambda *_args, **_kwargs: idle_cognition_input,
    )
    monkeypatch.setattr(
        connector_module,
        "record_continuity_boundary_event",
        empty_mapping,
    )
    monkeypatch.setattr(
        connector_module,
        "run_cognition",
        run_selected_engine,
    )
    monkeypatch.setattr(
        connector_module,
        "_project_output_to_global_state",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        connector_module,
        "_episode_identity_state_update",
        lambda _state: {},
    )

    async def one_terminal_cycle(
        state: dict[str, object],
        *,
        call_cognition_subgraph_func,
        **_kwargs: object,
    ) -> dict:
        return await call_cognition_subgraph_func(state)

    monkeypatch.setattr(
        idle_runner,
        "ensure_initial_resolver_inputs",
        lambda state, *, max_cycles: dict(state),
    )
    monkeypatch.setattr(
        idle_runner,
        "call_cognition_resolver_loop",
        one_terminal_cycle,
    )
    commit = AsyncMock()
    monkeypatch.setattr(idle_runner, "commit_cognition_output", commit)

    await idle_runner._default_cognition_client({
        "cognitive_episode": {"episode_id": "idle-engine-selector"},
    })

    assert isinstance(live_services, expected_type)
    assert observed_idle_types == [expected_type]
    commit.assert_awaited_once()
