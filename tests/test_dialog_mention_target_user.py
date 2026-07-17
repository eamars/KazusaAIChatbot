"""Tests for dialog-authored inline mention tags under the V2 contract."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.llm_interface.contracts import BackendDescriptor, LLMResponse
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from tests.cognition_core_v2_test_helpers import canonical_episode


@pytest.fixture(autouse=True)
def _aligned_dialog_compliance(monkeypatch) -> None:
    """Keep mention tests focused on delivery-neutral authored text."""

    monkeypatch.setattr(
        dialog_module,
        "_dialog_compliance_llm",
        _CapturingLLM({"aligned": True, "issues": []}),
    )


class _CapturingLLM:
    """Capture dialog-generator messages and return a fixed payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages, *, config=None):
        del config
        self.messages = messages
        backend = BackendDescriptor(
            route_name="test",
            backend_kind="test",
            model_family="test",
            model="test",
            normalized_base_url="test",
            thinking_strategy="unsupported",
            confidence="high",
            generation=1,
        )
        return LLMResponse(
            content=json.dumps(self.payload),
            backend=backend,
            raw_response=None,
            usage={},
        )


def _dialog_state() -> dict:
    """Build a reusable native dialog state fixture."""

    return {
        "internal_monologue": "answer directly",
        "cognitive_episode": canonical_episode(
            episode_id="dialog-mention-target",
            content="Address the current user.",
        ),
        "text_surface_output_v2": {
            "schema_version": "text_surface_output.v2",
            "content_plan": "answer",
            "content_requirements": ["address the current user"],
            "visible_boundaries": [],
            "addressee_plan": ["current user"],
            "style_guidance": "brief",
            "selected_surface_intent": "answer",
            "permitted_action_results": [],
        },
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "platform-user-1",
        "platform_bot_id": "bot-1",
        "global_user_id": "global-user-1",
        "user_name": "User",
        "user_profile": {
            "global_user_id": "global-user-1",
            "cognition_state": {"owner_user_id": "global-user-1"},
        },
        "character_profile": {
            "name": "Kazusa",
            "personality_brief": {
                "logic": "precise",
                "tempo": "measured",
                "defense": "guarded",
                "quirks": "dry",
                "taboos": "physical action narration",
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.4,
                "hesitation_density": 0.2,
                "counter_questioning": 0.2,
                "softener_density": 0.3,
                "formalism_avoidance": 0.6,
                "abstraction_reframing": 0.4,
                "direct_assertion": 0.6,
                "emotional_leakage": 0.3,
                "rhythmic_bounce": 0.2,
                "self_deprecation": 0.1,
            },
        },
        "debug_modes": {},
        "should_respond": True,
        "dialog_usage_mode": "live_visible_reply",
    }


@pytest.mark.asyncio
async def test_dialog_generator_preserves_inline_tag_without_delivery_context(
    monkeypatch,
) -> None:
    """Dialog preserves authored tags without receiving delivery identifiers."""

    fake_llm = _CapturingLLM({"final_dialog": ["@User answer"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["@User answer"]}
    human_payload = json.loads(fake_llm.messages[1].content)
    assert human_payload["text_surface_output_v2"]["schema_version"] == (
        "text_surface_output.v2"
    )
    assert "platform_user_id" not in human_payload
    assert "global_user_id" not in human_payload


@pytest.mark.asyncio
async def test_dialog_generator_does_not_require_mention_flag(monkeypatch) -> None:
    """Dialog output does not require a separate mention control field."""

    fake_llm = _CapturingLLM({"final_dialog": ["answer"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_module.dialog_generator(_dialog_state())

    assert result == {"final_dialog": ["answer"]}


@pytest.mark.asyncio
async def test_dialog_agent_returns_no_mention_flag(monkeypatch) -> None:
    """Dialog delivery metadata remains separate from model-authored text."""

    fake_generator = _CapturingLLM({"final_dialog": ["@User answer"]})
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_generator)

    result = await dialog_module.dialog_agent(_dialog_state())

    assert result["final_dialog"] == ["@User answer"]
    assert "mention_target_user" not in result
    assert result["target_addressed_user_ids"] == ["global-user-1"]
