"""V2 resolver-evidence and cycle-zero prewarm tests for persona cognition."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from tests.test_cognition_chain_connector_mapping import (
    NOW,
    _core_output,
    _global_state,
)


def test_resolver_observation_reenters_as_typed_evidence_only() -> None:
    """Resolver output contributes evidence and has no cognition-state authority."""

    evidence, direct_facts = project_resolver_observation_for_cognition(
        {
            "observation_id": "resolver-observation-1",
            "capability": "task_resolution_request",
            "semantic_summary": "A prior promise is relevant.",
            "replacement_state": {"forbidden": True},
        },
        occurred_at="2026-06-08T00:00:00Z",
    )

    assert evidence["evidence_ref"]["source_kind"] == "resolver_observation"
    assert evidence["evidence_ref"]["source_id"] == "resolver-observation-1"
    assert direct_facts == []
    assert "replacement_state" not in evidence


@pytest.mark.asyncio
async def test_cycle_zero_prewarm_reaches_v2_memory_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production connector joins shared prewarm before V2 input mapping."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    prewarm = AsyncMock(return_value={
        "answer": "",
        "memory_evidence": [{
            "content": "PREWARM_MEMORY_SENTINEL",
            "source_system": "memory",
        }],
        "user_memory_unit_candidates": [],
    })
    run_cognition = AsyncMock(return_value=_core_output())
    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(connector, "run_cognition", run_cognition)

    update = await connector.call_cognition_subgraph(state, commit=False)

    prewarm.assert_awaited_once_with(state)
    cognition_input = run_cognition.await_args.args[0]
    memory_rows = [
        row
        for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_kind"] == "promoted_memory"
    ]
    assert [row["semantic_text"] for row in memory_rows] == [
        "PREWARM_MEMORY_SENTINEL"
    ]
    assert [row["memory_scope"] for row in memory_rows] == [
        "shared_character_or_world"
    ]
    assert update["rag_result"]["answer"] == ""
    assert "PREWARM_MEMORY_SENTINEL" in repr(
        update["rag_result"]["memory_evidence"]
    )


@pytest.mark.asyncio
async def test_later_cycle_does_not_repeat_shared_memory_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver recurrence reuses prior evidence without another prewarm."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 1,
        "observations": [],
    }
    prewarm = AsyncMock()
    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "run_cognition",
        AsyncMock(return_value=_core_output()),
    )

    await connector.call_cognition_subgraph(state, commit=False)

    prewarm.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_prewarm_preserves_existing_rag_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved prewarm leaves existing evidence and answer unchanged."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    state["rag_result"] = {
        "answer": "existing resolver answer",
        "memory_evidence": [{"content": "BASE_MEMORY_SENTINEL"}],
    }
    prewarm = AsyncMock(return_value={
        "answer": "",
        "memory_evidence": [],
        "user_memory_unit_candidates": [],
    })
    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
        raising=False,
    )
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "run_cognition",
        AsyncMock(return_value=_core_output()),
    )

    update = await connector.call_cognition_subgraph(state, commit=False)

    prewarm.assert_awaited_once_with(state)
    assert update["rag_result"] == state["rag_result"]


@pytest.mark.asyncio
async def test_prewarm_starts_before_mutable_state_load_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared retrieval overlaps independent cycle-zero state preparation."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    prewarm_started = asyncio.Event()

    async def run_prewarm(_: object) -> dict[str, object]:
        prewarm_started.set()
        return {
            "answer": "",
            "memory_evidence": [],
            "user_memory_unit_candidates": [],
        }

    async def load_user_state(_: str) -> dict[str, object]:
        await asyncio.wait_for(prewarm_started.wait(), timeout=0.5)
        state_value = build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )
        return state_value

    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        run_prewarm,
        raising=False,
    )
    monkeypatch.setattr(connector, "get_user_cognition_state", load_user_state)
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "run_cognition",
        AsyncMock(return_value=_core_output()),
    )

    await connector.call_cognition_subgraph(state, commit=False)

    assert prewarm_started.is_set()


@pytest.mark.asyncio
async def test_invalid_episode_starts_no_prewarm_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical episode validation precedes retrieval task creation."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    episode = state["cognitive_episode"]
    assert isinstance(episode, dict)
    episode["unexpected_field"] = "invalid"
    prewarm = AsyncMock()
    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
    )

    with pytest.raises(
        connector.CognitionExecutionError,
        match="fields are not exact",
    ):
        await connector.call_cognition_subgraph(state, commit=False)

    prewarm.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_self_cognition_requires_service_owned_style_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connector fails closed when its service-owned snapshot is absent."""

    state = _global_state()
    state["channel_type"] = "group"
    state["global_user_id"] = ""
    state["character_identity_epistemic_core_included"] = True
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    episode = state["cognitive_episode"]
    assert isinstance(episode, dict)
    episode["trigger_source"] = "self_cognition"
    target_scope = episode["target_scope"]
    assert isinstance(target_scope, dict)
    target_scope["channel_type"] = "group"
    target_scope["current_global_user_id"] = None
    target_scope["current_platform_user_id"] = None

    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )

    with pytest.raises(
        connector.CognitionExecutionError,
        match="interaction style turn snapshot is required",
    ):
        await connector.call_cognition_subgraph(state, commit=False)

    assert not hasattr(connector, "build_group_engagement_action_context")


@pytest.mark.asyncio
async def test_state_load_failure_cancels_prewarm_preparation_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed user-state load cancels and joins shared-memory prewarm."""

    state = _global_state()
    state["resolver_state"] = {
        "cycle_index": 0,
        "observations": [],
    }
    prewarm_started = asyncio.Event()
    prewarm_cancelled = asyncio.Event()
    preparation_tasks: list[asyncio.Task[object]] = []

    async def run_prewarm(_: object) -> dict[str, object]:
        task = asyncio.current_task()
        assert task is not None
        preparation_tasks.append(task)
        prewarm_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            prewarm_cancelled.set()
        raise AssertionError("cancelled prewarm resumed")

    async def fail_user_state_load(_: str) -> dict[str, object]:
        await prewarm_started.wait()
        raise RuntimeError("state load failed")

    monkeypatch.setattr(
        connector,
        "run_first_cycle_shared_memory_prewarm",
        run_prewarm,
    )
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        fail_user_state_load,
    )

    try:
        with pytest.raises(RuntimeError, match="state load failed"):
            await connector.call_cognition_subgraph(state, commit=False)

        assert prewarm_cancelled.is_set()
    finally:
        for task in preparation_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*preparation_tasks, return_exceptions=True)
