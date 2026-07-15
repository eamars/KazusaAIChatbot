"""V2 internal-thought cognition dry-run tests."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy

import pytest

from kazusa_ai_chatbot import internal_thought_cognition as dry_run_module
from kazusa_ai_chatbot.cognition_episode import validate_cognitive_episode
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
    resolve_state_scope,
)
from kazusa_ai_chatbot.internal_thought_cognition import (
    InternalThoughtCognitionDryRunError,
    build_internal_thought_cognitive_episode,
    run_internal_thought_cognition_dry_run,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    _scope_caller,
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
)


STORAGE_TIMESTAMP_UTC = "2026-05-09T21:30:00+00:00"
V2_TIMESTAMP = "2026-05-09T21:30:00Z"


def _residue() -> dict[str, str]:
    return {
        "residue_id": "residue-1",
        "internal_monologue": "I should reconsider the unresolved event.",
        "source": "runtime_internal_thought",
    }


def _action_latch() -> dict[str, str]:
    return {
        "latch_id": "latch-1",
        "action_text": "Pause before selecting any later action.",
        "latch_reason": "Keep the private recurrence audit-only.",
        "status": "audit_only",
    }


def _time_context() -> dict[str, str]:
    return local_time_context_from_storage_utc(STORAGE_TIMESTAMP_UTC)


def test_internal_thought_episode_selects_character_scope() -> None:
    """Internal thought enters the frozen character-scope V2 path."""

    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
    )

    assert episode["trigger_source"] == "internal_thought"
    assert episode["input_sources"] == ["internal_monologue"]
    assert resolve_state_scope(_scope_caller(episode)) == (
        "character",
        "global",
    )


def test_internal_thought_builder_creates_exact_valid_episode() -> None:
    """The retained source builder preserves its typed episode contract."""

    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        action_latch=_action_latch(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
    )

    validate_cognitive_episode(episode)
    assert episode["output_mode"] == "think_only"
    assert episode["percepts"][0]["input_source"] == "internal_monologue"
    payload = json.loads(episode["percepts"][0]["content"])
    assert payload == {
        "residue": _residue(),
        "action_latch": _action_latch(),
    }


@pytest.mark.parametrize(
    ("local_enabled", "global_enabled"),
    [(False, True), (True, False)],
)
def test_internal_thought_builder_disables_visual_directives(
    monkeypatch: pytest.MonkeyPatch,
    local_enabled: bool,
    global_enabled: bool,
) -> None:
    """Either explicit or global visual disablement reaches episode metadata."""

    monkeypatch.setattr(
        dry_run_module,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        global_enabled,
    )

    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        visual_directives_enabled=local_enabled,
    )

    assert episode["origin_metadata"]["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }


def test_internal_thought_builder_uses_empty_latch_payload_when_absent() -> None:
    """An omitted audit latch remains an explicit empty JSON object."""

    episode = build_internal_thought_cognitive_episode(
        residue=_residue(),
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
    )

    payload = json.loads(episode["percepts"][0]["content"])
    assert payload["action_latch"] == {}


def test_internal_thought_episode_rejects_visible_reply() -> None:
    """The audit-only source cannot directly request visible output."""

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="output_mode is not supported",
    ):
        build_internal_thought_cognitive_episode(
            residue=_residue(),
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
            output_mode="visible_reply",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("internal_monologue", "", "internal_monologue is empty"),
        ("internal_monologue", "x" * 4001, "exceeds 4000"),
        ("source", "user_message", "source is not supported"),
    ],
)
def test_internal_thought_builder_rejects_invalid_residue(
    field_name: str,
    value: str,
    message: str,
) -> None:
    """Private recurrence rejects empty, oversized, and mistyped residue."""

    residue = _residue()
    residue[field_name] = value

    with pytest.raises(InternalThoughtCognitionDryRunError, match=message):
        build_internal_thought_cognitive_episode(
            residue=residue,  # type: ignore[arg-type]
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("status", "executable", "status is not supported"),
        ("action_text", "x" * 1001, "action_text exceeds 1000"),
        ("latch_reason", "x" * 1001, "latch_reason exceeds 1000"),
    ],
)
def test_internal_thought_builder_rejects_invalid_action_latch(
    field_name: str,
    value: str,
    message: str,
) -> None:
    """The optional action candidate remains bounded and audit-only."""

    action_latch = _action_latch()
    action_latch[field_name] = value

    with pytest.raises(InternalThoughtCognitionDryRunError, match=message):
        build_internal_thought_cognitive_episode(
            residue=_residue(),
            action_latch=action_latch,  # type: ignore[arg-type]
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
        )


@pytest.mark.asyncio
async def test_internal_thought_residue_is_evidence_not_direct_fact() -> None:
    """Private residue remains typed evidence and has no reducer authority."""

    captured_states: list[dict[str, object]] = []

    async def cognition_client(state: dict[str, object]) -> dict[str, object]:
        captured_states.append(deepcopy(state))
        return {"cognition_core_output": {"intention": {"route": "silence"}}}

    audit = await run_internal_thought_cognition_dry_run(
        residue=_residue(),
        character_profile={"name": "Kazusa"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=cognition_client,
    )
    character_state = build_character_production_state(updated_at=V2_TIMESTAMP)
    payload = build_cognition_input_from_global_state(
        captured_states[0],  # type: ignore[arg-type]
        mutable_state=character_state,
        character_state=character_state,
    )

    assert audit["cognition_schema_version"] == "cognition_core_input.v2"
    assert audit["state_scope"] == "character"
    assert payload["state_scope"] == "character"
    assert payload["evidence"][0]["evidence_ref"]["source_kind"] == "episode"
    assert payload["direct_facts"] == []


@pytest.mark.asyncio
async def test_busy_internal_thought_reports_v2_scope_without_calling_core() -> None:
    """Busy-path audit keeps the same V2 contract metadata."""

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    audit = await run_internal_thought_cognition_dry_run(
        residue=_residue(),
        character_profile={"name": "Kazusa"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: True,
        call_cognition_subgraph_func=unexpected_client,
    )

    assert audit["status"] == "skipped_busy"
    assert audit["cognition_called"] is False
    assert audit["cognition_schema_version"] == "cognition_core_input.v2"
    assert audit["state_scope"] == "character"


@pytest.mark.asyncio
async def test_dry_run_rejects_output_mode_before_busy_probe() -> None:
    """Local contract validation precedes any runtime busy check."""

    busy_probe_calls = 0

    def busy_probe() -> bool:
        nonlocal busy_probe_calls
        busy_probe_calls += 1
        return True

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    with pytest.raises(
        InternalThoughtCognitionDryRunError,
        match="output_mode is not supported",
    ):
        await run_internal_thought_cognition_dry_run(
            residue=_residue(),
            character_profile={"name": "Test Character"},
            user_profile={},
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
            is_primary_interaction_busy=busy_probe,
            call_cognition_subgraph_func=unexpected_client,
            output_mode="visible_reply",  # type: ignore[arg-type]
        )

    assert busy_probe_calls == 0


@pytest.mark.asyncio
async def test_dry_run_empty_residue_skips_cognition() -> None:
    """The ordinary empty-residue case returns a bounded audit result."""

    residue = _residue()
    residue["internal_monologue"] = ""

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    audit = await run_internal_thought_cognition_dry_run(
        residue=residue,  # type: ignore[arg-type]
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=unexpected_client,
    )

    assert audit["status"] == "skipped_empty_residue"
    assert audit["cognition_called"] is False
    assert audit["skip_reason"] == "internal_thought_residue_empty"


@pytest.mark.asyncio
async def test_dry_run_calls_cognition_once_and_returns_v2_audit() -> None:
    """A valid private recurrence invokes the injected V2 boundary once."""

    captured_states: list[dict[str, object]] = []

    async def cognition_client(state: dict[str, object]) -> dict[str, object]:
        captured_states.append(deepcopy(state))
        return {
            "cognition_core_output": {
                "schema_version": "cognition_core_output.v2",
            },
            "cognition_state_committed": True,
        }

    audit = await run_internal_thought_cognition_dry_run(
        residue=_residue(),
        action_latch=_action_latch(),
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=cognition_client,
    )

    assert len(captured_states) == 1
    assert audit["status"] == "completed"
    assert audit["cognition_called"] is True
    assert audit["action_latch_id"] == "latch-1"
    assert audit["cognition_output_keys"] == [
        "cognition_core_output",
        "cognition_state_committed",
    ]


@pytest.mark.asyncio
async def test_dry_run_visual_disable_reaches_canonical_episode() -> None:
    """The visual flag is state metadata, not a prompt-side fallback."""

    captured_states: list[dict[str, object]] = []

    async def cognition_client(state: dict[str, object]) -> dict[str, object]:
        captured_states.append(deepcopy(state))
        return {"cognition_core_output": {}}

    await run_internal_thought_cognition_dry_run(
        residue=_residue(),
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=cognition_client,
        visual_directives_enabled=False,
    )

    state = captured_states[0]
    assert state["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
        "no_visual_directives": True,
    }
    assert state["cognitive_episode"]["origin_metadata"]["debug_modes"] == (
        state["debug_modes"]
    )


def test_dry_run_module_has_no_write_delivery_or_scheduler_calls() -> None:
    """Internal thought remains an audit-only caller outside live delivery."""

    source = inspect.getsource(dry_run_module)
    forbidden_tokens = (
        "call_consolidation_subgraph",
        "save_conversation",
        "save_assistant_message",
        "dispatcher.dispatch",
        "schedule_event",
    )

    assert all(token not in source for token in forbidden_tokens)
