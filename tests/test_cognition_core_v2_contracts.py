"""Exact public V2 cognition and surface contract tests."""

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import run_cognition
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)

from llm_test_helpers import make_llm_call_config


NOW = "2026-07-14T00:00:00Z"


class _NoCallLLM:
    """Raise when a no-evidence episode reaches a model-owned stage."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del messages
        del config
        raise AssertionError("a grounded evidence-free episode must stay quiet")


class _Logger:
    """Provide the minimal V2 logger contract."""

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs


def _services() -> CognitionCoreServicesV2:
    """Build V2 service bindings for a deterministic no-call case."""

    return CognitionCoreServicesV2(
        llm=_NoCallLLM(),
        appraisal_config=make_llm_call_config("v2_appraisal"),
        goal_cognition_config=make_llm_call_config("v2_goal"),
        collapse_config=make_llm_call_config("v2_collapse"),
        action_selection_config=make_llm_call_config("v2_route"),
        parse_json=json.loads,
        logger=_Logger(),
    )


def _input() -> dict[str, object]:
    """Build the exact V2 input without raw prompt-owned state fields."""

    character = build_character_production_state(updated_at=NOW)
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "episode_id": "v2-contract-episode",
            "semantic_scene": "quiet private greeting",
        },
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="v2-contract-user",
            updated_at=NOW,
        ),
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": "quiet private greeting",
            "semantic_temporal_context": "immediate",
        },
    }


@pytest.mark.asyncio
async def test_v2_facade_returns_exact_one_scope_output() -> None:
    """The public facade returns one validated state update and no legacy shape."""

    output = await run_cognition(_input(), _services())
    validated = validate_cognition_core_output(output)

    assert validated["schema_version"] == "cognition_core_output.v2"
    assert validated["state_update"]["state_scope"] == "user"
    assert validated["intention"]["route"] == "silence"
    assert "admitted_bid" not in validated
