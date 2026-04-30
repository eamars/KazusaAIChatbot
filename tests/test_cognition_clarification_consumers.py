from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3_module


class _DummyResponse:
    """Small LangChain-like response wrapper for cognition tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy response.

        Args:
            content: LLM response content.
        """
        self.content = content


class _CapturingAsyncLLM:
    """Async LLM fake that records messages and returns a JSON payload."""

    def __init__(self, payload: dict) -> None:
        """Create a capturing fake LLM.

        Args:
            payload: JSON-serializable response payload.
        """
        self.payload = payload
        self.messages: list = []

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Record prompt messages and return the configured payload.

        Args:
            messages: Prompt messages supplied by the caller.

        Returns:
            Dummy response with JSON content.
        """
        self.messages = messages
        return _DummyResponse(json.dumps(self.payload))


def _judgment_state() -> dict:
    """Build the minimal state required by Judgment Core.

    Returns:
        Cognition-state subset for the L2c test.
    """
    return {
        "character_profile": {"name": "Kazusa"},
        "user_profile": {"affinity": 500},
        "internal_monologue": "I might answer if I knew the object.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "boundary_summary": "none",
            "behavior_primary": "comply",
            "behavior_secondary": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
            "identity_policy": "accept",
            "pressure_policy": "absorb",
            "trajectory": "stable",
        },
    }


def _content_anchor_state() -> dict:
    """Build the minimal state required by Content Anchor.

    Returns:
        Cognition-state subset for the L3b test.
    """
    return {
        "character_profile": {"name": "Kazusa"},
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
        ],
        "rag_result": {
            "answer": "",
            "user_image": {},
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {"unknown_slots": [], "loop_count": 0},
        },
        "internal_monologue": "I need to ask what these refers to.",
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "conversation_progress": None,
    }


@pytest.mark.asyncio
async def test_judgment_core_consumes_unresolved_referents(monkeypatch) -> None:
    """Judgment Core should force clarification from structured referents."""
    fake_llm = _CapturingAsyncLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "would otherwise answer",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", fake_llm)

    result = await l2_module.call_judgment_core_agent(_judgment_state())

    payload = json.loads(fake_llm.messages[1].content)
    assert payload["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]
    assert result["logical_stance"] == "TENTATIVE"
    assert result["character_intent"] == "CLARIFY"
    assert "不要用宽泛旧上下文" in result["judgment_note"]


@pytest.mark.asyncio
async def test_judgment_core_requires_referents(monkeypatch) -> None:
    """Judgment Core should enforce the structured referents contract."""

    fake_llm = _CapturingAsyncLLM({
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "judgment_note": "would otherwise answer",
    })
    monkeypatch.setattr(l2_module, "_judgement_core_llm", fake_llm)

    state = _judgment_state()
    state.pop("referents")

    with pytest.raises(KeyError):
        await l2_module.call_judgment_core_agent(state)


@pytest.mark.asyncio
async def test_content_anchor_agent_receives_referent_signal(monkeypatch) -> None:
    """Content Anchor should receive referent-derived clarification fields."""
    fake_llm = _CapturingAsyncLLM({
        "content_anchors": [
            "[DECISION] 先追问缺失对象",
            "[ANSWER] 你说的这些具体是指什么？",
            "[SCOPE] 简短追问即可",
        ]
    })
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", fake_llm)

    result = await l3_module.call_content_anchor_agent(_content_anchor_state())

    payload = json.loads(fake_llm.messages[1].content)
    assert payload["referents"] == [
        {"phrase": "这些", "referent_role": "object", "status": "unresolved"}
    ]
    assert result["content_anchors"][1].startswith("[ANSWER]")


@pytest.mark.asyncio
async def test_content_anchor_agent_requires_referents(monkeypatch) -> None:
    """Content Anchor should enforce the structured referents contract."""
    fake_llm = _CapturingAsyncLLM({
        "content_anchors": [
            "[DECISION] 先追问缺失对象",
            "[ANSWER] 你说的这些具体是指什么？",
            "[SCOPE] 简短追问即可",
        ]
    })
    monkeypatch.setattr(l3_module, "_content_anchor_agent_llm", fake_llm)

    state = _content_anchor_state()
    state.pop("referents")

    with pytest.raises(KeyError):
        await l3_module.call_content_anchor_agent(state)
