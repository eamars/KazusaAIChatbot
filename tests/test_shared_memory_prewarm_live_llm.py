"""Guarded live LLM/DB proof for the seeded ``#napcat`` prewarm path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.db import build_memory_doc, save_memory
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.rag.memory_evidence.workers import persistent_search
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from tests.test_shared_memory_prewarm import (
    _empty_result,
    _minimal_persona_state,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_LIVE_GUARD_ENV = "NAPCAT_PREWARM_LIVE_GUARD"
_LIVE_DATABASE_ENV = "NAPCAT_PREWARM_LIVE_DATABASE"
_MEMORY_UNIT_ID = "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
_LIVE_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1] / "test_artifacts" / "debug_runs"
)


class _CapturingLiveLLM:
    """Capture real generator prompts and responses without changing them."""

    def __init__(self, wrapped_llm: Any) -> None:
        """Store the real LLM and initialize the captured-call list."""

        self._wrapped_llm = wrapped_llm
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        config: Any = None,
    ) -> Any:
        """Invoke the real LLM and retain its prompt and raw response."""

        response = await self._wrapped_llm.ainvoke(messages, config=config)
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(message.content),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
        })
        return response


def _require_napcat_live_runtime() -> None:
    """Require an explicit database and execution guard for this live proof."""

    if os.environ.get(_LIVE_GUARD_ENV, "").strip() != "1":
        pytest.skip(
            f"set {_LIVE_GUARD_ENV}=1 to run the isolated napcat live proof"
        )
    expected_database = os.environ.get(_LIVE_DATABASE_ENV, "").strip()
    if not expected_database:
        pytest.skip(
            f"{_LIVE_DATABASE_ENV} must name the explicitly authorized database"
        )
    if os.environ.get("MONGODB_DB_NAME", "").strip() != expected_database:
        pytest.skip(
            "napcat live proof requires MONGODB_DB_NAME to match "
            f"{_LIVE_DATABASE_ENV}"
        )


def _safe_outcome_diagnostic(outcome: object) -> str:
    """Return only disposition/count/id fields from a live outcome."""

    if not isinstance(outcome, dict):
        return f"outcome_type={type(outcome).__name__}"
    rag_result = outcome.get("rag_result")
    evidence_ids: list[str] = []
    if isinstance(rag_result, dict):
        memory_evidence = rag_result.get("memory_evidence")
        if isinstance(memory_evidence, list):
            evidence_ids = [
                str(row.get("memory_unit_id"))
                for row in memory_evidence
                if isinstance(row, dict) and row.get("memory_unit_id")
            ]
    return (
        f"status={outcome.get('status')!r} "
        f"reason_code={outcome.get('reason_code')!r} "
        f"retrieved_shared_count={outcome.get('retrieved_shared_count')!r} "
        f"merged_shared_count={outcome.get('merged_shared_count')!r} "
        f"evidence_memory_unit_ids={evidence_ids!r}"
    )


@pytest.mark.asyncio
async def test_live_bare_tag_prewarm_recovers_missing_generator_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify bare-tag prewarm survives an incomplete real generator result."""

    _require_napcat_live_runtime()
    run_id = uuid4().hex
    channel_id = f"live-napcat-channel-{run_id}"
    platform_user_id = f"live-napcat-platform-user-{run_id}"
    global_user_id = str(uuid4())
    message_id = f"live-napcat-message-{run_id}"
    episode_id = f"live-napcat-episode-{run_id}"
    state = _minimal_persona_state(
        character_name='一之濑明日奈',
        character_global_user_id="active-asuna-global-id",
        user_input='@一之濑明日奈 #napcat',
    )
    state["global_user_id"] = global_user_id
    state["user_name"] = "NapCat isolated live test user"
    state["platform_channel_id"] = channel_id
    state["platform_message_id"] = message_id
    state["chat_history_recent"] = []
    state["chat_history_wide"] = []
    state["prompt_message_context"]["body_text"] = (
        '@一之濑明日奈 #napcat'
    )
    state["prompt_message_context"]["raw_wire_text"] = (
        '@一之濑明日奈 #napcat'
    )
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": "active-asuna-global-id",
        "display_name": '一之濑明日奈',
        "entity_kind": "bot",
    }]
    episode = state["cognitive_episode"]
    episode["episode_id"] = episode_id
    episode["origin_metadata"]["platform_message_id"] = message_id
    episode["target_scope"]["channel_type"] = "group"
    episode["target_scope"]["platform_channel_id"] = channel_id
    episode["target_scope"]["current_platform_user_id"] = (
        platform_user_id
    )
    episode["target_scope"]["current_global_user_id"] = global_user_id
    episode["target_scope"]["current_display_name"] = (
        "NapCat isolated live test user"
    )
    episode["target_scope"]["target_addressed_user_ids"] = [
        "active-asuna-global-id"
    ]

    memory_unit_id = f"live-napcat-memory-{run_id}"
    memory_doc = build_memory_doc(
        memory_name=f"live_napcat_{run_id}",
        content=(
            "#napcat is the seeded shared command for this isolated "
            "prewarm test."
        ),
        source_global_user_id="",
        memory_type="fact",
        source_kind="seeded_manual",
        confidence_note="isolated live prewarm test seed",
        status="active",
    )
    memory_doc["memory_unit_id"] = memory_unit_id
    memory_doc["lineage_id"] = memory_unit_id
    memory_doc["authority"] = "seed"
    await save_memory(memory_doc, storage_utc_now_iso())
    try:
        generator_llm = _CapturingLiveLLM(persistent_search._generator_llm)
        monkeypatch.setattr(
            persistent_search,
            "_generator_llm",
            generator_llm,
        )
        tool_calls: list[dict[str, Any]] = []
        original_tool = persistent_search._tool

        async def capture_tool(args: dict[str, Any]) -> object:
            """Record actual arguments before the real retrieval tool runs."""

            tool_calls.append(dict(args))
            result = await original_tool(args)
            return result

        monkeypatch.setattr(persistent_search, "_tool", capture_tool)
        outcome = await capabilities.run_first_cycle_shared_memory_prewarm(
            state
        )
    finally:
        database = await get_db()
        await database.memory.delete_one({"memory_unit_id": memory_unit_id})

    artifact = {
        "case_id": "live_bare_tag_prewarm_query_contract",
        "input": {
            "body_text": '@一之濑明日奈 #napcat',
            "decontextualized_input": '@一之濑明日奈 #napcat',
            "channel_type": "group",
            "history_recent_count": len(state["chat_history_recent"]),
            "history_wide_count": len(state["chat_history_wide"]),
            "channel_id": channel_id,
            "platform_user_id": platform_user_id,
            "global_user_id": global_user_id,
        },
        "generator_calls": generator_llm.calls,
        "tool_calls": tool_calls,
        "outcome": outcome,
        "judgment": (
            "The real generator must result in a valid persistent-memory "
            "query and the seeded shared row must be available to prewarm."
        ),
    }
    artifact_path = (
        _LIVE_ARTIFACT_ROOT
        / f"{artifact['case_id']}_{run_id}.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"napcat live artifact: {artifact_path}")

    diagnostic = _safe_outcome_diagnostic(outcome)
    assert tool_calls, diagnostic
    assert tool_calls[0].get("search_query"), json.dumps(
        {
            "diagnostic": diagnostic,
            "tool_args": tool_calls[0],
        },
        ensure_ascii=True,
        default=str,
    )
    assert outcome["status"] == "completed", diagnostic
    assert outcome["reason_code"] == "shared_memory_ready", diagnostic
    evidence = outcome["rag_result"]["memory_evidence"]
    assert any(
        row.get("memory_unit_id") == memory_unit_id
        for row in evidence
        if isinstance(row, dict)
    ), diagnostic


async def test_production_napcat_command_recalls_seeded_shared_memory() -> None:
    """Verify the real ``@character #napcat`` row survives prewarm merge."""

    _require_napcat_live_runtime()
    state = _minimal_persona_state()
    state["decontextualized_input"] = '@一之濑明日奈 #napcat'
    state["prompt_message_context"]["body_text"] = '@一之濑明日奈 #napcat'
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": "character-1",
        "display_name": "一之濑明日奈",
        "entity_kind": "bot",
    }]

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(state)
    print(f"napcat prewarm outcome: {_safe_outcome_diagnostic(outcome)}")

    assert outcome["status"] == "completed", _safe_outcome_diagnostic(outcome)
    assert outcome["reason_code"] == "shared_memory_ready"
    evidence = outcome["rag_result"]["memory_evidence"]
    assert any(
        row.get("memory_unit_id") == _MEMORY_UNIT_ID
        for row in evidence
        if isinstance(row, dict)
    ), _safe_outcome_diagnostic(outcome)

    merged_rag_result, merged_outcome = (
        capabilities.merge_shared_memory_prewarm_outcome(
            _empty_result(),
            outcome,
        )
    )
    print(
        "napcat merged outcome: "
        f"{_safe_outcome_diagnostic(merged_outcome)}"
    )
    assert merged_outcome["status"] == "completed"
    assert merged_outcome["reason_code"] == "shared_memory_merged"
    assert any(
        row.get("memory_unit_id") == _MEMORY_UNIT_ID
        for row in merged_rag_result["memory_evidence"]
        if isinstance(row, dict)
    )
