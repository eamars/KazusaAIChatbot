"""Deterministic tests for first-cycle shared-memory prewarm helpers."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

import pytest
from openai import OpenAIError

from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_shared_memory_prewarm_outcome,
)
from kazusa_ai_chatbot.db.user_memory_units import build_user_memory_unit_doc
from kazusa_ai_chatbot.rag.memory_evidence.workers.user_memory import (
    _project_row,
)
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import canonical_user_message_episode


def _minimal_persona_state(
    *,
    character_name: str = "Kazusa",
    character_global_user_id: str = "character-1",
    user_input: str = "Need a memory-backed stance.",
    active_turn_platform_message_ids: list[str] | None = None,
    active_turn_conversation_row_ids: list[str] | None = None,
    chat_history_recent: list[dict[str, Any]] | None = None,
    chat_history_wide: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the smallest persona state accepted by the prewarm boundary."""

    turn_clock = build_turn_clock("2026-06-08 09:00:00")
    episode = canonical_user_message_episode(
        episode_id="prewarm-episode-1",
        percept_id="prewarm-percept-1",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input=user_input,
        platform="debug",
        platform_channel_id="prewarm-channel",
        channel_type="private",
        platform_message_id="prewarm-message",
        platform_user_id="platform-user-1",
        global_user_id="user-1",
        user_name="Test User",
        active_turn_platform_message_ids=active_turn_platform_message_ids,
        active_turn_conversation_row_ids=active_turn_conversation_row_ids,
        target_addressed_user_ids=[character_global_user_id],
        target_broadcast=False,
    )
    state = {
        "cognitive_episode": episode,
        "decontextualized_input": user_input,
        "referents": [],
        "character_profile": {
            "name": character_name,
            "global_user_id": character_global_user_id,
        },
        "user_profile": {"relationship_state": 500},
        "prompt_message_context": {
            "body_text": user_input,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [character_global_user_id],
            "broadcast": False,
        },
        "channel_topic": "prewarm test",
        "chat_history_recent": deepcopy(chat_history_recent or []),
        "chat_history_wide": deepcopy(chat_history_wide or []),
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": None,
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "global_user_id": "user-1",
        "user_name": "Test User",
        "platform": "debug",
        "platform_channel_id": "prewarm-channel",
        "platform_message_id": "prewarm-message",
        "platform_bot_id": "bot-1",
        "active_turn_platform_message_ids": list(
            active_turn_platform_message_ids or []
        ),
        "active_turn_conversation_row_ids": list(
            active_turn_conversation_row_ids or []
        ),
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
    }
    return state


def _empty_result() -> dict[str, Any]:
    """Build the expected empty prewarm RAG result."""

    rag_result = {
        "answer": "",
        "user_image": {
            "user_memory_context": empty_user_memory_context(),
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "resolver": "local_context_resolver",
            "iterations": 0,
            "node_count": 0,
            "resolved_node_count": 0,
            "blocked_node_count": 0,
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return rag_result


def _ready_outcome(
    *,
    memory_evidence: list[dict[str, Any]] | None = None,
    status: str = "completed",
    reason_code: str = "shared_memory_ready",
    attempted: bool = True,
    latency_ms: int = 1,
    merged_shared_count: int = 0,
) -> dict[str, Any]:
    """Build one candidate prewarm outcome for contract tests."""

    rag_result = _empty_result()
    if memory_evidence is None:
        memory_evidence = [{"summary": "Shared prewarm memory"}]
    rag_result["memory_evidence"] = list(memory_evidence)
    outcome = {
        "schema_version": "shared_memory_prewarm_outcome.v1",
        "status": status,
        "reason_code": reason_code,
        "attempted": attempted,
        "latency_ms": latency_ms,
        "retrieved_shared_count": len(rag_result["memory_evidence"]),
        "merged_shared_count": merged_shared_count,
        "rag_result": rag_result,
    }
    return outcome


def test_shared_memory_prewarm_outcome_validator_rejects_invalid_disposition_and_counts() -> None:
    """Prewarm status, reason, and count invariants are closed-world."""

    valid = _ready_outcome()
    validated = validate_shared_memory_prewarm_outcome(valid)
    assert validated == valid
    invalid_candidates = [
        {
            **valid,
            "status": "empty",
            "reason_code": "shared_memory_ready",
        },
        {
            **valid,
            "status": "completed",
            "reason_code": "worker_error",
        },
        {
            **valid,
            "retrieved_shared_count": 0,
        },
        {
            **valid,
            "merged_shared_count": 1,
        },
        {
            **valid,
            "status": "skipped",
            "reason_code": "not_first_cycle",
            "attempted": True,
            "latency_ms": 1,
            "retrieved_shared_count": 0,
            "rag_result": _empty_result(),
        },
    ]
    for candidate in invalid_candidates:
        with pytest.raises(ResolverValidationError):
            validate_shared_memory_prewarm_outcome(candidate)


def test_shared_memory_prewarm_outcome_enforces_exact_types_bounds_and_rag_shape() -> None:
    """The outcome validator enforces strict types, bounds, and RAG fields."""

    valid = _ready_outcome()
    invalid_candidates = [
        {**valid, "extra": True},
        {key: value for key, value in valid.items() if key != "status"},
        {**valid, "attempted": 1},
        {**valid, "latency_ms": True},
        {**valid, "latency_ms": 120001},
        {**valid, "retrieved_shared_count": True},
        {**valid, "rag_result": {**_empty_result(), "extra": True}},
        {
            **valid,
            "rag_result": {
                **_empty_result(),
                "user_memory_unit_candidates": [{"unit_id": "private"}],
            },
        },
        {
            **valid,
            "rag_result": {
                **_empty_result(),
                "memory_evidence": ["not a mapping"],
            },
        },
        {
            **valid,
            "rag_result": {
                **_empty_result(),
                "supervisor_trace": {"large": ["x" * 600] * 200},
            },
        },
    ]
    too_many_entries = _empty_result()
    too_many_entries["memory_evidence"] = [
        {"summary": str(index)} for index in range(25)
    ]
    invalid_candidates.append({**valid, "rag_result": too_many_entries})
    for candidate in invalid_candidates:
        with pytest.raises(ResolverValidationError):
            validate_shared_memory_prewarm_outcome(candidate)

    source = _ready_outcome()
    copied = validate_shared_memory_prewarm_outcome(source)
    source["rag_result"]["memory_evidence"][0]["summary"] = "mutated"
    assert copied["rag_result"]["memory_evidence"][0]["summary"] != (
        "mutated"
    )


def _typed_shared_rows() -> list[dict[str, Any]]:
    """Build mixed certified shared rows for the prewarm authority gate."""

    certificate = {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
    }
    return [
        {
            "_id": "prewarm-fact-row",
            "memory_unit_id": "prewarm-fact-unit",
            "memory_name": "fact",
            "content": "A prewarm world fact.",
            "memory_type": "fact",
            "source_kind": "conversation_extracted",
            "source_global_user_id": "",
            "authority": "conversation_accepted",
            "status": "active",
            "privacy_review": {
                **certificate,
                "boundary_assessment": "deidentified global meaning",
                "reviewer": "automated_llm",
            },
        },
        {
            "_id": "prewarm-guidance-row",
            "memory_unit_id": "prewarm-guidance-unit",
            "memory_name": "defense_rule",
            "content": "A prewarm self-guidance rule.",
            "memory_type": "defense_rule",
            "source_kind": "reflection_inferred",
            "source_global_user_id": "",
            "authority": "reflection_promoted",
            "status": "active",
            "privacy_review": {
                **certificate,
                "boundary_assessment": "deidentified global meaning",
                "reviewer": "automated_llm",
            },
        },
    ]


def _user_memory_writer_row(*, user_id: str, unit_id: str) -> dict[str, Any]:
    """Build one current-user row through the production writer shape."""

    writer_row = dict(build_user_memory_unit_doc(
        user_id,
        {
            "unit_id": unit_id,
            "unit_type": "objective_fact",
            "fact": "Private current-user continuity must not prewarm.",
            "subjective_appraisal": "A scoped continuity note.",
            "relationship_signal": "Keep it participant-scoped.",
        },
        storage_timestamp_utc="2026-05-24T07:41:21+00:00",
        unit_id=unit_id,
    ))
    projected_row = _project_row(writer_row, user_id)
    return projected_row


def _patch_persistent_memory_worker(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any] | BaseException,
) -> list[dict[str, Any]]:
    """Patch the shared persistent-memory worker and return captured calls."""

    calls: list[dict[str, Any]] = []

    class FakePersistentMemorySearchAgent:
        """Capture prewarm calls without touching live retrieval backends."""

        async def run(
            self,
            task: str,
            context: dict[str, Any],
            max_attempts: int = 3,
        ) -> dict[str, Any]:
            """Record the worker invocation and return the configured result."""

            calls.append({
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            })
            if isinstance(result, BaseException):
                raise result
            return_value = result
            return return_value

    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        FakePersistentMemorySearchAgent,
        raising=False,
    )
    return calls


def _forbid_full_local_context_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the test if prewarm enters the full RAG3 resolver."""

    async def resolve_local_context(
        _request: dict[str, Any],
        _context: dict[str, Any],
        _options: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject full local-context resolver calls from prewarm."""

        raise AssertionError("prewarm must not call resolve_local_context")

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        resolve_local_context,
    )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_uses_shared_persistent_memory_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm should call only the shared persistent-memory worker."""

    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [_typed_shared_rows()[0]],
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert len(calls) == 1
    worker_call = calls[0]
    context = worker_call["context"]
    assert worker_call["task"] == "Need a memory-backed stance."
    assert worker_call["max_attempts"] == 1
    assert context["prompt_message_context"]["body_text"] == (
        "Need a memory-backed stance."
    )
    assert context["character_profile"]["name"] == "Kazusa"
    assert outcome["status"] == "completed"
    assert outcome["reason_code"] == "shared_memory_ready"
    rag_result = outcome["rag_result"]
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert "A prewarm world fact." in repr(
        rag_result["memory_evidence"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("character_name", "character_global_user_id"),
    [
        ("一之濑明日奈", "active-asuna-global-id"),
        ("杏山千纱", "active-kazusa-global-id"),
    ],
    ids=["active_asuna", "active_kazusa"],
)
async def test_napcat_character_mention_recalls_and_merges_corresponding_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
    character_name: str,
    character_global_user_id: str,
) -> None:
    """Character mentions search content and preserve the seeded shared row."""

    state = _minimal_persona_state(
        character_name=character_name,
        character_global_user_id=character_global_user_id,
        user_input=f"@{character_name} #napcat",
    )
    state["prompt_message_context"]["raw_wire_text"] = (
        "[CQ:at,qq=3768713357] #napcat"
    )
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": character_global_user_id,
        "display_name": character_name,
        "entity_kind": "bot",
    }]
    source_state = deepcopy(state)
    seeded_row = {
        "_id": "seed-row-not-observable",
        "memory_unit_id": "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3",
        "memory_name": "napcat",
        "content": "A seeded shared defense rule for napcat requests.",
        "memory_type": "defense_rule",
        "source_kind": "seeded_manual",
        "source_global_user_id": "",
        "authority": "seed",
        "status": "active",
        "scope_type": "global",
        "privacy_review": {
            "global_applicability": "global",
            "target_specific_meaning_removed": True,
            "affects_identity_or_boundaries": False,
            "private_detail_risk": "low",
            "user_details_removed": True,
            "boundary_assessment": "deidentified global meaning",
            "reviewer": "seed_tool",
        },
    }
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [seeded_row],
            "attempts": 1,
        },
    )

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert state == source_state
    assert len(calls) == 1
    assert calls[0]["task"] == "#napcat"
    assert calls[0]["max_attempts"] == 1
    worker_context = calls[0]["context"]
    assert worker_context["character_profile"]["name"] == character_name
    assert worker_context["prompt_message_context"] is not (
        state["prompt_message_context"]
    )
    assert worker_context["prompt_message_context"]["body_text"] == "#napcat"
    assert worker_context["prompt_message_context"]["mentions"] == []
    model_visible_context = project_runtime_context_for_llm(
        worker_context,
        character_name=character_name,
    )
    model_visible_payload = json.dumps(
        model_visible_context,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    assert f"@{character_name}" not in model_visible_payload
    assert character_name not in model_visible_payload
    assert "character_profile" not in model_visible_context
    assert outcome["status"] == "completed"
    assert outcome["reason_code"] == "shared_memory_ready"
    assert outcome["retrieved_shared_count"] == 1
    evidence = outcome["rag_result"]["memory_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["memory_unit_id"] == (
        "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
    )

    merged, finalized = capabilities.merge_shared_memory_prewarm_outcome(
        _empty_result(),
        outcome,
    )
    assert finalized["status"] == "completed"
    assert finalized["reason_code"] == "shared_memory_merged"
    assert finalized["retrieved_shared_count"] == 1
    assert finalized["merged_shared_count"] == 1
    assert merged["memory_evidence"][0]["memory_unit_id"] == (
        "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("character_name", "character_global_user_id"),
    [
        ("一之濑明日奈", "active-asuna-global-id"),
        ("杏山千纱", "active-kazusa-global-id"),
    ],
    ids=["active_asuna", "active_kazusa"],
)
async def test_prewarm_model_projection_excludes_active_turn_history_rows(
    monkeypatch: pytest.MonkeyPatch,
    character_name: str,
    character_global_user_id: str,
) -> None:
    """Prewarm history omits only the typed current-turn rows."""

    active_platform_message_id = "active-platform-message"
    active_conversation_row_id = "active-conversation-row"
    active_body = f"@{character_name} #napcat"
    older_body = f"Earlier, {character_name} discussed a topic."
    active_row = {
        "conversation_row_id": active_conversation_row_id,
        "platform_message_id": active_platform_message_id,
        "role": "user",
        "display_name": "Test User",
        "body_text": active_body,
        "timestamp": "2026-06-08T08:59:00Z",
    }
    older_row = {
        "conversation_row_id": "older-conversation-row",
        "platform_message_id": "older-platform-message",
        "role": "user",
        "display_name": "Earlier User",
        "body_text": older_body,
        "timestamp": "2026-06-08T08:58:00Z",
    }
    state = _minimal_persona_state(
        character_name=character_name,
        character_global_user_id=character_global_user_id,
        user_input=active_body,
        active_turn_platform_message_ids=[active_platform_message_id],
        active_turn_conversation_row_ids=[active_conversation_row_id],
        chat_history_recent=[active_row, older_row],
        chat_history_wide=[active_row, older_row],
    )
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": character_global_user_id,
        "display_name": character_name,
        "entity_kind": "bot",
    }]
    source_state = deepcopy(state)
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
        },
    )

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert state == source_state
    assert outcome["status"] == "empty"
    assert outcome["reason_code"] == "worker_unresolved"
    assert len(calls) == 1
    assert calls[0]["task"] == "#napcat"
    assert calls[0]["max_attempts"] == 1
    worker_context = calls[0]["context"]
    model_visible_context = project_runtime_context_for_llm(
        worker_context,
        character_name=character_name,
    )
    assert all(
        row.get("platform_message_id") != active_platform_message_id
        for row in worker_context["chat_history_recent"]
    )
    assert all(
        row.get("platform_message_id") != active_platform_message_id
        for row in worker_context["chat_history_wide"]
    )
    recent_lines = model_visible_context["chat_history_recent"]
    wide_lines = model_visible_context["chat_history_wide"]
    assert all(active_body not in line for line in recent_lines)
    assert all(active_body not in line for line in wide_lines)
    assert any(older_body in line for line in recent_lines)
    assert any(older_body in line for line in wide_lines)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("character_name", "character_global_user_id"),
    [
        ("一之濑明日奈", "active-asuna-global-id"),
        ("杏山千纱", "active-kazusa-global-id"),
    ],
)
async def test_first_cycle_prewarm_skips_character_mention_only_query(
    monkeypatch: pytest.MonkeyPatch,
    character_name: str,
    character_global_user_id: str,
) -> None:
    """A character address without content has no memory retrieval target."""

    state = _minimal_persona_state(
        character_name=character_name,
        character_global_user_id=character_global_user_id,
        user_input=f"@{character_name}",
    )
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": character_global_user_id,
        "display_name": character_name,
        "entity_kind": "bot",
    }]
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [{
                "content": "Character-name memory must not be selected.",
                "source_system": "memory",
            }],
            "attempts": 1,
        },
    )

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert calls == []
    assert outcome["status"] == "skipped"
    assert outcome["reason_code"] == "empty_query_after_character_mention"
    assert outcome["rag_result"] == _empty_result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "body_text",
        "mention_display_name",
        "mention_global_user_id",
        "mention_entity_kind",
        "expected_task",
    ),
    [
        (
            "@爱音 之前说了什么",
            "爱音",
            "participant-global-id",
            "user",
            "@爱音 之前说了什么",
        ),
        (
            "@另一机器人 #napcat",
            "另一机器人",
            "other-bot-global-id",
            "bot",
            "@另一机器人 #napcat",
        ),
        (
            "@一之濑明日奈 #napcat",
            "一之濑明日奈",
            None,
            "bot",
            "@一之濑明日奈 #napcat",
        ),
    ],
)
async def test_first_cycle_prewarm_preserves_non_character_mention(
    monkeypatch: pytest.MonkeyPatch,
    body_text: str,
    mention_display_name: str,
    mention_global_user_id: str | None,
    mention_entity_kind: str,
    expected_task: str,
) -> None:
    """Only a matching active-character mention is structural addressing."""

    state = _minimal_persona_state(
        character_name="一之濑明日奈",
        character_global_user_id="active-asuna-global-id",
    )
    state["decontextualized_input"] = body_text
    state["prompt_message_context"]["body_text"] = body_text
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": mention_global_user_id,
        "display_name": mention_display_name,
        "entity_kind": mention_entity_kind,
    }]
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
        },
    )

    await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert len(calls) == 1
    assert calls[0]["task"] == expected_task
    assert calls[0]["context"]["prompt_message_context"]["body_text"] == (
        body_text
    )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_preserves_plain_character_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain character name remains a legitimate retrieval subject."""

    state = _minimal_persona_state(
        character_name="一之濑明日奈",
        character_global_user_id="active-asuna-global-id",
        user_input="一之濑明日奈喜欢什么",
    )
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
        },
    )

    await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert len(calls) == 1
    assert calls[0]["task"] == '一之濑明日奈喜欢什么'


@pytest.mark.asyncio
async def test_first_cycle_prewarm_removes_only_exact_character_mention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A longer authored literal must survive before the exact bot mention."""

    state = _minimal_persona_state(
        character_name="一之濑明日奈",
        character_global_user_id="active-asuna-global-id",
    )
    state["decontextualized_input"] = (
        '@一之濑明日奈-archive @一之濑明日奈 #napcat'
    )
    state["prompt_message_context"]["body_text"] = (
        '@一之濑明日奈-archive @一之濑明日奈 #napcat'
    )
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": "active-asuna-global-id",
        "display_name": "一之濑明日奈",
        "entity_kind": "bot",
    }]
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
        },
    )

    await capabilities.run_first_cycle_shared_memory_prewarm(state)

    assert len(calls) == 1
    assert calls[0]["task"] == '@一之濑明日奈-archive  #napcat'


@pytest.mark.asyncio
async def test_first_cycle_prewarm_projects_memory_without_answer_or_user_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only accepted shared-memory rows should reach projected memory evidence."""

    private_user_row = _user_memory_writer_row(
        user_id="user-1",
        unit_id="private-unit",
    )
    private_user_row.update({
        "source_system": "user_memory_units",
        "source_kind": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": "user-1",
        "authority": "scoped_continuity",
        "truth_status": "character_lore_or_interaction_continuity",
        "origin": "consolidated_interaction",
    })
    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [
                {
                    **_typed_shared_rows()[0],
                    "content": "Shared nonverbal input policy.",
                },
                private_user_row,
            ],
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    rag_result = outcome["rag_result"]
    rendered = repr(rag_result)
    assert len(calls) == 1
    assert outcome["status"] == "completed"
    assert outcome["reason_code"] == "shared_memory_ready"
    assert rag_result["answer"] == ""
    assert rag_result["user_memory_unit_candidates"] == []
    assert len(rag_result["memory_evidence"]) == 1
    assert "Shared nonverbal input policy." in rendered
    assert "Private current-user continuity must not prewarm." not in rendered
    assert "Candidate must not prewarm." not in rendered
    assert "user_memory_units" not in rendered


def test_merge_shared_memory_prewarm_outcome_counts_projected_entries_and_rejects_repeat_or_invalid_base() -> None:
    """Merge appends projected rows once and preserves the complete base."""

    base_rag_result = _empty_result()
    base_rag_result["answer"] = "base answer"
    base_rag_result["memory_evidence"] = [{"summary": "base memory"}]
    base_rag_result["media_evidence"] = [{"kind": "image"}]
    outcome = _ready_outcome(memory_evidence=[
        {"summary": "shared prewarm memory"},
        {"summary": "second shared memory"},
    ])

    merged, finalized = capabilities.merge_shared_memory_prewarm_outcome(
        base_rag_result,
        outcome,
    )

    assert merged["answer"] == "base answer"
    assert merged["media_evidence"] == [{"kind": "image"}]
    assert merged["memory_evidence"] == [
        {"summary": "base memory"},
        {"summary": "shared prewarm memory"},
        {"summary": "second shared memory"},
    ]
    assert base_rag_result["memory_evidence"] == [
        {"summary": "base memory"},
    ]
    assert finalized["reason_code"] == "shared_memory_merged"
    assert finalized["retrieved_shared_count"] == 2
    assert finalized["merged_shared_count"] == 2

    with pytest.raises(ResolverValidationError, match="prewarm_outcome_not_ready"):
        capabilities.merge_shared_memory_prewarm_outcome(merged, finalized)
    invalid_base = _empty_result()
    invalid_base["memory_evidence"] = {"not": "a list"}
    with pytest.raises(ResolverValidationError, match="base_rag_result_invalid"):
        capabilities.merge_shared_memory_prewarm_outcome(
            invalid_base,
            outcome,
        )
    empty_outcome = _ready_outcome(
        status="empty",
        reason_code="no_shared_memory",
        memory_evidence=[],
    )
    with pytest.raises(ResolverValidationError, match="prewarm_outcome_not_ready"):
        capabilities.merge_shared_memory_prewarm_outcome(
            base_rag_result,
            empty_outcome,
        )


@pytest.mark.asyncio
async def test_first_cycle_prewarm_returns_explicit_outcome_dispositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each worker disposition remains explicit at the prewarm boundary."""

    _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [{
                "content": "Private current-user continuity must not prewarm.",
                "source_system": "user_memory_units",
            }],
            "attempts": 1,
        },
    )
    _forbid_full_local_context_resolver(monkeypatch)
    unresolved = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert unresolved["status"] == "empty"
    assert unresolved["reason_code"] == "no_shared_memory"
    assert unresolved["attempted"] is True
    assert unresolved["retrieved_shared_count"] == 0

    _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
        },
    )
    worker_unresolved = (
        await capabilities.run_first_cycle_shared_memory_prewarm(
            _minimal_persona_state(),
        )
    )
    assert worker_unresolved["status"] == "empty"
    assert worker_unresolved["reason_code"] == "worker_unresolved"
    assert worker_unresolved["attempted"] is True
    assert worker_unresolved["retrieved_shared_count"] == 0

    _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": {"invalid": "worker result shape"},
            "attempts": 1,
        },
    )
    worker_contract_invalid = (
        await capabilities.run_first_cycle_shared_memory_prewarm(
            _minimal_persona_state(),
        )
    )
    assert worker_contract_invalid["status"] == "failed"
    assert worker_contract_invalid["reason_code"] == (
        "worker_contract_invalid"
    )
    assert worker_contract_invalid["attempted"] is True
    assert worker_contract_invalid["merged_shared_count"] == 0

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise KeyError("projection shape")

    with monkeypatch.context() as projection_patch:
        projection_patch.setattr(
            capabilities,
            "project_known_facts",
            fail_projection,
        )
        _patch_persistent_memory_worker(
            projection_patch,
            {
                "resolved": True,
                "result": [_typed_shared_rows()[0]],
                "attempts": 1,
            },
        )
        projection_failed = (
            await capabilities.run_first_cycle_shared_memory_prewarm(
                _minimal_persona_state(),
            )
        )
    assert projection_failed["status"] == "failed"
    assert projection_failed["reason_code"] == "projection_failed"
    assert projection_failed["attempted"] is True
    assert projection_failed["merged_shared_count"] == 0

    _patch_persistent_memory_worker(
        monkeypatch,
        OpenAIError("memory worker unavailable"),
    )
    failed = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert failed["status"] == "failed"
    assert failed["reason_code"] == "worker_error"
    assert failed["attempted"] is True
    assert failed["merged_shared_count"] == 0

    ready_calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": [_typed_shared_rows()[0]],
            "attempts": 1,
        },
    )
    ready = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )
    assert len(ready_calls) == 1
    assert ready["status"] == "completed"
    assert ready["reason_code"] == "shared_memory_ready"
    assert ready["retrieved_shared_count"] == 1
    assert ready["merged_shared_count"] == 0


@pytest.mark.asyncio
async def test_first_cycle_prewarm_propagates_cancellation_without_fabricating_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation reaches the caller without an artificial disposition."""

    calls = _patch_persistent_memory_worker(
        monkeypatch,
        asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await capabilities.run_first_cycle_shared_memory_prewarm(
            _minimal_persona_state(),
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_first_cycle_prewarm_preserves_self_guidance_conditional_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prewarm uses the same typed partition as normal shared-memory recall."""

    calls = _patch_persistent_memory_worker(
        monkeypatch,
        {
            "resolved": True,
            "result": _typed_shared_rows(),
            "attempts": 1,
        },
    )

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(
        _minimal_persona_state(),
    )

    assert len(calls) == 1
    rag_result = outcome["rag_result"]
    entries = rag_result["memory_evidence"]
    assert [entry["memory_type"] for entry in entries] == [
        "fact",
        "defense_rule",
    ]
    assert entries[1]["authority"] == "reflection_promoted"
    assert entries[1]["source_kind"] == "reflection_inferred"
    assert "A prewarm self-guidance rule." in entries[1]["content"]
    assert rag_result["user_memory_unit_candidates"] == []
