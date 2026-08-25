"""Deterministic scope-certificate tests for character self-guidance writes."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.consolidation import character_self_guidance as module
from kazusa_ai_chatbot.memory_evolution.models import MemoryPrivacyReview


def _scope_certificate(
    *,
    global_applicability: str = "global",
    target_specific_meaning_removed: bool = True,
    affects_identity_or_boundaries: bool = False,
    private_detail_risk: str = "low",
    user_details_removed: bool = True,
    reason: str = "The accepted behavior remains character-owned after scope review.",
) -> dict[str, Any]:
    """Build a certificate fixture without encoding a conversation scenario."""

    return {
        "global_applicability": global_applicability,
        "target_specific_meaning_removed": target_specific_meaning_removed,
        "affects_identity_or_boundaries": affects_identity_or_boundaries,
        "private_detail_risk": private_detail_risk,
        "user_details_removed": user_details_removed,
        "reason": reason,
    }


def _candidate_output(certificate: dict[str, Any]) -> dict[str, Any]:
    """Build the specialist/reviewer wire meaning around one certificate."""

    return {
        "action": "write",
        "memory_name": "General response guidance",
        "content": "Use the accepted behavior when it remains appropriate.",
        **certificate,
    }


def _review_output(certificate: dict[str, Any]) -> dict[str, Any]:
    """Build the reviewer wire meaning around one certificate."""

    return {
        "decision": "accept",
        "memory_name": "General response guidance",
        "content": "Use the accepted behavior when it remains appropriate.",
        **certificate,
    }


def _state() -> dict[str, Any]:
    """Build the minimum prompt and persistence state for scope tests."""

    return {
        "character_profile": {"name": "the active character"},
        "decontextualized_input": "A bounded request.",
        "final_dialog": ["An accepted response."],
        "chat_history_recent": [],
        "character_self_guidance_source_refs": [{
            "source_key": "current_turn_user_message",
            "source_kind": "user_message",
            "source_refs": [{"conversation_history_id": "row-1"}],
        }],
        "storage_timestamp_utc": "2026-08-24T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_self_guidance_write_requires_independent_global_scope_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disagreement between independent certificates fails the write closed."""

    specialist_certificate = _scope_certificate()
    reviewer_certificate = _scope_certificate(
        target_specific_meaning_removed=False,
    )
    responses = iter([
        SimpleNamespace(content=json.dumps(
            _candidate_output(specialist_certificate),
            ensure_ascii=False,
        )),
        SimpleNamespace(content=json.dumps(
            _review_output(reviewer_certificate),
            ensure_ascii=False,
        )),
    ])

    async def _invoke(messages, *, config):
        del messages, config
        return next(responses)

    monkeypatch.setattr(module._self_guidance_specialist_llm, "ainvoke", _invoke)
    monkeypatch.setattr(module._self_guidance_reviewer_llm, "ainvoke", _invoke)

    result = await module.character_self_guidance_specialist(_state())

    assert result == {"character_self_guidance": {}}


@pytest.mark.asyncio
async def test_reviewer_receives_candidate_meaning_without_specialist_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer receives meaning without the specialist's judgment."""

    specialist_certificate = _scope_certificate()
    captured: dict[str, Any] = {}

    async def _specialist_invoke(messages, *, config):
        del messages, config
        return SimpleNamespace(content=json.dumps(
            _candidate_output(specialist_certificate),
            ensure_ascii=False,
        ))

    async def _reviewer_invoke(messages, *, config):
        del config
        captured["human_payload"] = json.loads(messages[1].content)
        return SimpleNamespace(content=json.dumps(
            _review_output(specialist_certificate),
            ensure_ascii=False,
        ))

    monkeypatch.setattr(
        module._self_guidance_specialist_llm,
        "ainvoke",
        _specialist_invoke,
    )
    monkeypatch.setattr(
        module._self_guidance_reviewer_llm,
        "ainvoke",
        _reviewer_invoke,
    )

    result = await module.character_self_guidance_specialist(_state())

    assert result["character_self_guidance"]
    reviewer_payload = captured["human_payload"]
    candidate_payload = reviewer_payload["candidate"]
    assert candidate_payload == {
        "memory_name": "General response guidance",
        "content": "Use the accepted behavior when it remains appropriate.",
        "memory_type": "defense_rule",
    }
    serialized_payload = json.dumps(reviewer_payload, ensure_ascii=False)
    assert "specialist_scope_certificate" not in serialized_payload
    assert specialist_certificate["reason"] not in serialized_payload


@pytest.mark.asyncio
async def test_malformed_specialist_certificate_fails_closed_before_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed specialist structure stops before the reviewer call."""

    malformed_output = _candidate_output(_scope_certificate())
    del malformed_output["private_detail_risk"]
    reviewer_mock = AsyncMock()

    async def _specialist_invoke(messages, *, config):
        del messages, config
        return SimpleNamespace(content=json.dumps(
            malformed_output,
            ensure_ascii=False,
        ))

    monkeypatch.setattr(
        module._self_guidance_specialist_llm,
        "ainvoke",
        _specialist_invoke,
    )
    monkeypatch.setattr(
        module._self_guidance_reviewer_llm,
        "ainvoke",
        reviewer_mock,
    )

    result = await module.character_self_guidance_specialist(_state())

    assert result == {"character_self_guidance": {}}
    reviewer_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_reviewer_certificate_fails_closed_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed reviewer structure yields no candidate or durable write."""

    specialist_certificate = _scope_certificate()
    malformed_output = _review_output(specialist_certificate)
    del malformed_output["user_details_removed"]
    reviewer_calls = 0

    async def _specialist_invoke(messages, *, config):
        del messages, config
        return SimpleNamespace(content=json.dumps(
            _candidate_output(specialist_certificate),
            ensure_ascii=False,
        ))

    async def _reviewer_invoke(messages, *, config):
        nonlocal reviewer_calls
        del messages, config
        reviewer_calls += 1
        return SimpleNamespace(content=json.dumps(
            malformed_output,
            ensure_ascii=False,
        ))

    monkeypatch.setattr(
        module._self_guidance_specialist_llm,
        "ainvoke",
        _specialist_invoke,
    )
    monkeypatch.setattr(
        module._self_guidance_reviewer_llm,
        "ainvoke",
        _reviewer_invoke,
    )
    insert_mock = AsyncMock()
    monkeypatch.setattr(module, "insert_memory_unit", insert_mock)

    result = await module.character_self_guidance_specialist(_state())
    persisted = await module.persist_character_self_guidance_from_state({
        **_state(),
        **result,
    })

    assert reviewer_calls == 1
    assert result == {"character_self_guidance": {}}
    assert persisted is None
    insert_mock.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("specialist", "reviewer"))
async def test_non_object_stage_output_fails_closed_without_persistence(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Non-object parser results fail closed at either self-guidance stage."""

    specialist_certificate = _scope_certificate()
    parsed_results: list[Any]
    if stage == "specialist":
        parsed_results = [[]]
    else:
        parsed_results = [
            _candidate_output(specialist_certificate),
            "not an object",
        ]
    reviewer_calls = 0

    async def _specialist_invoke(messages, *, config):
        del messages, config
        return SimpleNamespace(content="ignored")

    async def _reviewer_invoke(messages, *, config):
        nonlocal reviewer_calls
        del messages, config
        reviewer_calls += 1
        return SimpleNamespace(content="ignored")

    def _parse_output(raw_output: str) -> Any:
        del raw_output
        return parsed_results.pop(0)

    monkeypatch.setattr(
        module._self_guidance_specialist_llm,
        "ainvoke",
        _specialist_invoke,
    )
    monkeypatch.setattr(
        module._self_guidance_reviewer_llm,
        "ainvoke",
        _reviewer_invoke,
    )
    monkeypatch.setattr(module, "parse_llm_json_output", _parse_output)
    insert_mock = AsyncMock()
    monkeypatch.setattr(module, "insert_memory_unit", insert_mock)

    result = await module.character_self_guidance_specialist(_state())
    persisted = await module.persist_character_self_guidance_from_state({
        **_state(),
        **result,
    })

    assert result == {"character_self_guidance": {}}
    assert persisted is None
    assert reviewer_calls == (0 if stage == "specialist" else 1)
    insert_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_guidance_persists_reviewer_scope_and_privacy_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence stores the final reviewer certificate rather than defaults."""

    reviewer_certificate = _scope_certificate(
        reason="The final review found no retained private detail.",
    )
    candidate = {
        "memory_name": "General response guidance",
        "content": "Use the accepted behavior when it remains appropriate.",
        "specialist_scope_certificate": _scope_certificate(),
        "reviewer_scope_certificate": reviewer_certificate,
    }
    captured: dict[str, Any] = {}

    async def _insert(*, document: dict[str, Any]) -> dict[str, Any]:
        captured["document"] = document
        return {
            "memory_unit_id": "memory-1",
            "lineage_id": "lineage-1",
            "memory_type": "defense_rule",
            "memory_name": document["memory_name"],
            "content": document["content"],
        }

    monkeypatch.setattr(module, "insert_memory_unit", _insert)

    result = await module.persist_character_self_guidance_from_state({
        **_state(),
        "character_self_guidance": candidate,
    })

    assert result is not None
    privacy_review = captured["document"]["privacy_review"]
    assert privacy_review["global_applicability"] == "global"
    assert privacy_review["target_specific_meaning_removed"] is True
    assert privacy_review["affects_identity_or_boundaries"] is False
    assert privacy_review["private_detail_risk"] == "low"
    assert privacy_review["user_details_removed"] is True
    assert privacy_review["boundary_assessment"] == reviewer_certificate["reason"]


@pytest.mark.asyncio
async def test_learned_memory_privacy_review_requires_exact_scope_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learned write requires the complete typed scope certificate."""

    assert {
        "global_applicability",
        "target_specific_meaning_removed",
        "affects_identity_or_boundaries",
    }.issubset(MemoryPrivacyReview.__annotations__)

    insert_mock = AsyncMock()
    monkeypatch.setattr(module, "insert_memory_unit", insert_mock)
    result = await module.persist_character_self_guidance_from_state({
        **_state(),
        "character_self_guidance": {
            "memory_name": "General response guidance",
            "content": "Use the accepted behavior when it remains appropriate.",
        },
    })

    assert result is None
    insert_mock.assert_not_awaited()
