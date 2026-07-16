"""V2 conversation-history boundary tests."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import run_text_surface_planning
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    TextSurfaceServicesV2,
    validate_text_surface_input,
)
from llm_test_helpers import make_llm_call_config
from tests.cognition_core_v2_test_helpers import canonical_episode


class _PromptCaptureLLM:
    """Capture public L3 prompts and return the exact surface-stage shape."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        system = str(getattr(messages[0], "content", ""))
        self.prompts.append(str(getattr(messages[-1], "content", "")))
        if "exactly style_guidance" in system:
            result = {"style_guidance": "bounded"}
        elif "exactly content_plan" in system:
            result = {
                "content_plan": "bounded",
                "content_requirements": ["preserve the current addressee"],
            }
        elif "exactly visible_boundaries" in system:
            result = {
                "visible_boundaries": ["bounded"],
                "addressee_plan": ["bounded"],
            }
        else:
            raise AssertionError("unexpected text-surface stage")
        return SimpleNamespace(content=json.dumps(result))


def _surface_payload() -> dict[str, object]:
    """Build one canonical public L3 packet with private history metadata."""

    episode = canonical_episode(
        episode_id="history-policy",
        content="visible current-turn grounding",
        metadata={"chat_history": "RAW_HISTORY_SENTINEL"},
    )
    episode["percepts"].append({
        "percept_id": "private-history-percept",
        "input_source": "dialog_text",
        "content": "PRIVATE_HISTORY_SENTINEL",
        "visibility": "internal_only",
        "metadata": {"internal_monologue": "PRIVATE_MONOLOGUE_SENTINEL"},
    })
    return {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": "acknowledge the current turn",
            "target_roles": [],
            "reason": "the current percept is visible",
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "calm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief and natural",
        "character_voice_context": "reserved, analytical, and warm",
    }


@pytest.mark.asyncio
async def test_surface_episode_allows_only_semantic_history_summaries() -> None:
    """Canonical validation precedes prompt-safe visible-percept projection."""

    invalid_payload = deepcopy(_surface_payload())
    invalid_payload["episode"] = {
        "episode_summary": "retired semantic episode projection"
    }
    with pytest.raises(CognitionContractError):
        validate_text_surface_input(invalid_payload)

    llm = _PromptCaptureLLM()
    services = TextSurfaceServicesV2(
        llm=llm,
        style_config=make_llm_call_config("history_style"),
        content_plan_config=make_llm_call_config("history_content"),
        preference_config=make_llm_call_config("history_preference"),
    )
    await run_text_surface_planning(_surface_payload(), services)

    rendered = "\n".join(llm.prompts)
    assert "visible current-turn grounding" in rendered
    assert "RAW_HISTORY_SENTINEL" not in rendered
    assert "PRIVATE_HISTORY_SENTINEL" not in rendered
    assert "PRIVATE_MONOLOGUE_SENTINEL" not in rendered
