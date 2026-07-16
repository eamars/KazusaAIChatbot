"""V2 promoted-reflection source-boundary tests."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_episode import validate_cognitive_episode
from kazusa_ai_chatbot.reflection_cycle import cognition_dry_run as dry_run_module
from kazusa_ai_chatbot.reflection_cycle.cognition_dry_run import (
    ReflectionCognitionDryRunError,
    build_reflection_signal_cognitive_episode,
    run_reflection_cognition_dry_run,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-14T00:00:00Z"
STORAGE_TIMESTAMP_UTC = "2026-07-14T00:00:00+00:00"


def _promoted_context() -> dict[str, object]:
    return {
        "promoted_lore": [{
            "memory_name": "quiet_space_preference",
            "content": "The character protects quiet shared spaces.",
            "memory_type": "lore",
            "updated_at": STORAGE_TIMESTAMP_UTC,
            "confidence_note": "Promoted from repeated reflection.",
        }],
        "promoted_self_guidance": [{
            "memory_name": "avoid_overcommitting",
            "content": "Keep reflective conclusions tentative in speech.",
            "memory_type": "self_guidance",
            "updated_at": STORAGE_TIMESTAMP_UTC,
            "confidence_note": "Promoted as private guidance.",
        }],
        "source_dates": ["2026-07-14"],
        "retrieval_notes": ["Only promoted rows are included."],
    }


def _time_context() -> dict[str, str]:
    return {
        "current_local_datetime": "2026-07-14 12:00",
        "current_local_weekday": "Tuesday",
    }


def _payload() -> dict[str, object]:
    character = build_character_production_state(updated_at=NOW)
    state = build_acquaintance_user_state(
        global_user_id="reflection-user",
        updated_at=NOW,
    )
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id="reflection-dry-run",
            trigger_source="reflection_signal",
            output_mode="think_only",
            current_global_user_id="reflection-user",
            content="a promoted reflection is available",
        ),
        "state_scope": "user",
        "mutable_state": state,
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "promoted_reflection",
                "source_id": "reflection:promoted-1",
                "occurred_at": NOW,
                "semantic_summary": "promoted guidance about staying concise",
            },
            "semantic_text": "promoted guidance about staying concise",
            "visible_to": list(
                EVIDENCE_SOURCE_QUESTION_IDS["promoted_reflection"]
            ),
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "scene_context": {
            "channel_scope": "internal",
            "character_role": "character",
            "semantic_scene": "a promoted reflection is available",
            "conversation_continuity": "No active public conversation.",
            "semantic_temporal_context": "recent",
        },
        "private_continuity_context": "I should retain only promoted guidance.",
    }


def test_promoted_reflection_is_valid_typed_evidence() -> None:
    """Allow only promoted reflection through the normal evidence contract."""

    validated = validate_cognition_core_input(_payload())
    assert validated["evidence"][0]["evidence_ref"]["source_kind"] == (
        "promoted_reflection"
    )


def test_promoted_reflection_selects_exact_source_owned_questions() -> None:
    """Keep reflection provenance while exposing only semantic text."""

    payload = _payload()
    questions = plan_semantic_questions(
        payload["evidence"],
        payload["mutable_state"],
        payload["character_constraints"],
    )
    assert [question["question_id"] for question in questions] == list(
        EVIDENCE_SOURCE_QUESTION_IDS["promoted_reflection"]
    )
    assert all(question["evidence_handles"] == ["e1"] for question in questions)


def test_reflection_builder_creates_exact_valid_episode() -> None:
    """The retained builder emits one canonical private reflection episode."""

    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=_promoted_context(),  # type: ignore[arg-type]
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
    )

    validate_cognitive_episode(episode)
    assert episode["trigger_source"] == "reflection_signal"
    assert episode["input_sources"] == ["reflection_artifact"]
    assert episode["output_mode"] == "think_only"
    assert json.loads(episode["percepts"][0]["content"]) == (
        _promoted_context()
    )
    assert episode["origin_metadata"]["debug_modes"] == {
        "think_only": True,
        "no_remember": True,
    }


def test_reflection_builder_rejects_empty_promoted_context() -> None:
    """Reflection recurrence requires at least one promoted content lane."""

    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="promoted reflection context is empty",
    ):
        build_reflection_signal_cognitive_episode(
            promoted_reflection_context={  # type: ignore[arg-type]
                "retrieval_notes": ["nothing promoted"],
            },
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
        )


def test_reflection_builder_rejects_visible_reply() -> None:
    """The private reflection source cannot directly request a reply."""

    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="output_mode is not supported",
    ):
        build_reflection_signal_cognitive_episode(
            promoted_reflection_context=_promoted_context(),  # type: ignore[arg-type]
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
            output_mode="visible_reply",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_reflection_dry_run_busy_path_skips_cognition() -> None:
    """A busy primary interaction returns a bounded V2 audit result."""

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context=_promoted_context(),  # type: ignore[arg-type]
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: True,
        call_cognition_subgraph_func=unexpected_client,
        output_mode="preview",
    )

    assert audit["status"] == "skipped_busy"
    assert audit["cognition_called"] is False
    assert audit["state_scope"] == "character"
    assert audit["cognition_schema_version"] == "cognition_core_input.v2"


@pytest.mark.asyncio
async def test_reflection_dry_run_validates_mode_before_busy_probe() -> None:
    """Output-mode validation precedes the runtime busy check."""

    def unexpected_probe() -> bool:
        raise AssertionError("busy probe must not run")

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    with pytest.raises(
        ReflectionCognitionDryRunError,
        match="output_mode is not supported",
    ):
        await run_reflection_cognition_dry_run(
            promoted_reflection_context=_promoted_context(),  # type: ignore[arg-type]
            character_profile={"name": "Test Character"},
            user_profile={},
            storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
            local_time_context=_time_context(),
            is_primary_interaction_busy=unexpected_probe,
            call_cognition_subgraph_func=unexpected_client,
            output_mode="visible_reply",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_reflection_dry_run_empty_context_skips_cognition() -> None:
    """An ordinary empty promoted context does not build an episode."""

    async def unexpected_client(state: dict[str, object]) -> dict[str, object]:
        raise AssertionError(f"unexpected cognition call: {state!r}")

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context={  # type: ignore[arg-type]
            "retrieval_notes": ["nothing promoted"],
        },
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=unexpected_client,
        output_mode="silent",
    )

    assert audit["status"] == "skipped_empty_context"
    assert audit["cognition_called"] is False
    assert audit["skip_reason"] == "promoted_reflection_context_empty"


@pytest.mark.asyncio
async def test_reflection_dry_run_calls_v2_boundary_once() -> None:
    """A non-busy promoted reflection reaches cognition exactly once."""

    captured_states: list[dict[str, object]] = []

    async def cognition_client(state: dict[str, object]) -> dict[str, object]:
        captured_states.append(deepcopy(state))
        return {
            "cognition_core_output": {
                "schema_version": "cognition_core_output.v2",
            },
            "cognition_state_committed": True,
        }

    audit = await run_reflection_cognition_dry_run(
        promoted_reflection_context=_promoted_context(),  # type: ignore[arg-type]
        character_profile={"name": "Test Character"},
        user_profile={},
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=_time_context(),
        is_primary_interaction_busy=lambda: False,
        call_cognition_subgraph_func=cognition_client,
    )

    assert len(captured_states) == 1
    assert captured_states[0]["cognitive_episode"]["trigger_source"] == (
        "reflection_signal"
    )
    assert audit["status"] == "completed"
    assert audit["cognition_output_keys"] == [
        "cognition_core_output",
        "cognition_state_committed",
    ]


def test_reflection_dry_run_module_has_no_side_effect_entrypoints() -> None:
    """The reflection recurrence module remains audit-only."""

    source = inspect.getsource(dry_run_module)
    forbidden_tokens = (
        "call_consolidation_subgraph",
        "save_conversation",
        "save_assistant_message",
        "dispatcher.dispatch",
        "schedule_event",
    )

    assert all(token not in source for token in forbidden_tokens)
