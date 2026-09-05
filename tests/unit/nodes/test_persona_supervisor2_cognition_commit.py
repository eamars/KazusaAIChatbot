"""Focused commit and continuation-lineage checks for canonical cognition."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    SharedMemoryPrewarmOutcomeV1,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    _validate_evidence_rows,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    reset_v2_attempt_ledger,
    snapshot_v2_shared_memory_prewarm_checkpoint,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_node
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from tests.cognition_test_helpers import canonical_user_message_episode
from tests.unit.cognition_core_v3.test_handleless_contract import _input


def _empty_rag_result() -> dict[str, object]:
    """Build the canonical base RAG mapping used by cognition tests."""

    return {
        "answer": "",
        "user_image": {},
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {},
    }


def _ready_prewarm_outcome() -> SharedMemoryPrewarmOutcomeV1:
    """Build one ready prewarm result for the cognition carrier tests."""

    outcome: SharedMemoryPrewarmOutcomeV1 = {
        "schema_version": "shared_memory_prewarm_outcome.v1",
        "status": "completed",
        "reason_code": "shared_memory_ready",
        "attempted": True,
        "latency_ms": 3,
        "retrieved_shared_count": 1,
        "merged_shared_count": 0,
        "rag_result": {
            **_empty_rag_result(),
            "memory_evidence": [{"summary": "prewarm evidence"}],
        },
    }
    return outcome


def _global_state(*, cycle_index: int) -> dict[str, object]:
    """Build the minimal graph state needed by the cognition connector."""

    payload = _input()
    episode = deepcopy(payload["episode"])
    timestamp = "2026-07-14T00:00:00Z"
    user_state = deepcopy(payload["mutable_state"])
    state: dict[str, object] = {
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-1",
            "personality_brief": {
                "logic": "grounded",
                "defense": "boundary-aware",
                "quirks": "observant",
                "taboos": "pretending certainty",
            },
        },
        "storage_timestamp_utc": timestamp,
        "local_time_context": {
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        "llm_trace_id": "cognition-prewarm-test",
        "user_input": "the current observation",
        "prompt_message_context": {
            "body_text": "the current observation",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": "debug",
        "platform_channel_id": "deterministic-channel",
        "channel_type": "private",
        "channel_name": "",
        "platform_message_id": "deterministic-message",
        "platform_user_id": "platform-user",
        "global_user_id": "user-1",
        "user_name": "Test User",
        "user_profile": {"relationship_state": 500},
        "platform_bot_id": "bot-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "public_group_scene": "",
        "debug_modes": {},
        "should_respond": True,
        "decontextualized_input": "the current observation",
        "referents": [],
        "rag_result": _empty_rag_result(),
        "resolver_state": {
            "cycle_index": cycle_index,
            "observations": [],
        },
        "conversation_progress": None,
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
        "internal_monologue": "",
        "cognition_state": user_state,
        "character_identity_revision_number": 1,
        "character_identity_context": {"name": "Test Character"},
        "character_identity_surface_context": {"name": "Test Character"},
        "character_identity_projection_digest": "identity-digest",
        "character_identity_consumer_kinds": ["cognition"],
        "character_identity_episode_id": episode["episode_id"],
        "character_identity_epistemic_core_included": False,
    }
    return state




def _historical_prewarm_state() -> dict[str, object]:
    """Build the typed QQ ingress shape from the historical regression."""

    character_name = "一之濑明日奈"
    character_global_user_id = "active-character-global-id"
    current_body = f"@{character_name} #napcat"
    active_platform_message_id = "active-platform-message-id"
    active_conversation_row_id = "active-conversation-row-id"
    timestamp = "2026-07-14T00:00:00Z"
    episode = canonical_user_message_episode(
        episode_id="historical-prewarm-caller-episode",
        percept_id="historical-prewarm-caller-percept",
        storage_timestamp_utc=timestamp,
        local_time_context={
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        user_input=current_body,
        platform="qq",
        platform_channel_id="historical-qq-group",
        channel_type="group",
        platform_message_id=active_platform_message_id,
        platform_user_id="3768713000",
        global_user_id="user-1",
        user_name="Test User",
        active_turn_platform_message_ids=[active_platform_message_id],
        active_turn_conversation_row_ids=[active_conversation_row_id],
        target_addressed_user_ids=[character_global_user_id],
        target_broadcast=False,
    )
    active_history_row = {
        "conversation_row_id": active_conversation_row_id,
        "platform_message_id": active_platform_message_id,
        "role": "user",
        "display_name": "Test User",
        "body_text": current_body,
        "timestamp": "2026-07-14T00:00:00Z",
    }
    older_history_row = {
        "conversation_row_id": "older-conversation-row-id",
        "platform_message_id": "older-platform-message-id",
        "role": "user",
        "display_name": "Earlier User",
        "body_text": f"Earlier, {character_name} discussed a topic.",
        "timestamp": "2026-07-13T23:59:00Z",
    }
    state = _global_state(cycle_index=0)
    state.update({
        "character_profile": {
            **state["character_profile"],
            "name": character_name,
            "global_user_id": character_global_user_id,
        },
        "storage_timestamp_utc": timestamp,
        "local_time_context": {
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        "cognitive_episode": episode,
        "llm_trace_id": "historical-prewarm-caller-trace",
        "user_input": current_body,
        "decontextualized_input": current_body,
        "prompt_message_context": {
            "body_text": current_body,
            "raw_wire_text": "[CQ:at,qq=3768713357] #napcat",
            "mentions": [{
                "global_user_id": character_global_user_id,
                "display_name": character_name,
                "entity_kind": "bot",
                "platform_user_id": "3768713357",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": [character_global_user_id],
            "broadcast": False,
        },
        "platform": "qq",
        "platform_channel_id": "historical-qq-group",
        "channel_type": "group",
        "channel_name": "Historical QQ Group",
        "platform_message_id": active_platform_message_id,
        "platform_user_id": "3768713000",
        "global_user_id": "user-1",
        "user_name": "Test User",
        "platform_bot_id": "3768713357",
        "active_turn_platform_message_ids": [active_platform_message_id],
        "active_turn_conversation_row_ids": [active_conversation_row_id],
        "chat_history_recent": [active_history_row, older_history_row],
        "chat_history_wide": [active_history_row, older_history_row],
        "character_identity_context": {"name": character_name},
        "character_identity_surface_context": {"name": character_name},
        "character_identity_episode_id": episode["episode_id"],
    })
    return state


def _seeded_shared_memory_row() -> dict[str, object]:
    """Build the certified seeded row used by the caller regression."""

    return {
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


def _cognition_output() -> dict[str, object]:
    """Build a minimal canonical output accepted by global projection."""

    return {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "observe",
            "intent": "understand the current observation",
            "reason": "the current observation needs a grounded response",
            "cause_summary": "a current observation arrived",
        },
        "private_monologue": "I will keep the observation grounded.",
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": "Keep unknown details unknown.",
        },
        "state_projection": {
            "state_scope": "user",
            "owner_key": "user-1",
            "replacement_state": {"state_scope": "user"},
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }


def _patch_cognition_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_real_cognition_input: bool = False,
) -> None:
    """Patch model, persistence, and unrelated diagnostics for carrier tests."""

    async def get_user_state(_owner: str) -> dict[str, object]:
        return deepcopy(_input()["mutable_state"])

    async def get_character_state() -> dict[str, object]:
        return build_character_production_state(
            updated_at="2026-07-14T00:00:00Z",
        )

    async def record_boundary(**_kwargs: object) -> None:
        return None

    def fake_build_input(
        state,
        *,
        mutable_state,
        character_state,
    ) -> dict[str, object]:
        return {
            "episode": state["cognitive_episode"],
            "scene_context": {},
            "evidence": [],
            "available_actions": [],
            "available_resolver_capabilities": [],
        }

    async def fake_run_cognition(_input, _services) -> dict[str, object]:
        return _cognition_output()

    monkeypatch.setattr(cognition_node, "get_user_cognition_state", get_user_state)
    monkeypatch.setattr(
        cognition_node,
        "get_character_cognition_state",
        get_character_state,
    )
    monkeypatch.setattr(
        cognition_node,
        "record_continuity_boundary_event",
        record_boundary,
    )
    if not use_real_cognition_input:
        monkeypatch.setattr(
            cognition_node,
            "build_cognition_input_from_global_state",
            fake_build_input,
        )
    monkeypatch.setattr(cognition_node, "run_cognition", fake_run_cognition)
    monkeypatch.setattr(cognition_node, "current_chain_scope", lambda: "test")


@pytest.mark.asyncio
async def test_cognition_state_preserves_shared_memory_prewarm_outcome_after_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real caller carries seeded prewarm evidence into cognition."""

    _patch_cognition_runtime(
        monkeypatch,
        use_real_cognition_input=True,
    )
    worker_calls: list[dict[str, object]] = []
    cognition_inputs: list[dict[str, object]] = []

    class FakePersistentMemorySearchAgent:
        """Capture the direct worker while retaining the production helper."""

        async def run(
            self,
            task: str,
            context: dict[str, object],
            max_attempts: int = 3,
        ) -> dict[str, object]:
            worker_calls.append({
                "task": task,
                "context": deepcopy(context),
                "max_attempts": max_attempts,
            })
            return {
                "resolved": True,
                "result": [_seeded_shared_memory_row()],
                "attempts": 1,
            }

    async def capture_run_cognition(
        cognition_input: dict[str, object],
        _services: object,
    ) -> dict[str, object]:
        cognition_inputs.append(deepcopy(cognition_input))
        return _cognition_output()

    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        FakePersistentMemorySearchAgent,
    )
    monkeypatch.setattr(
        cognition_node,
        "run_cognition",
        capture_run_cognition,
    )
    result = await cognition_node.call_cognition_subgraph(
        _historical_prewarm_state(),
        commit=False,
    )

    assert len(worker_calls) == 1
    worker_call = worker_calls[0]
    assert worker_call["task"] == "#napcat"
    assert worker_call["max_attempts"] == 1
    worker_context = worker_call["context"]
    model_visible_context = project_runtime_context_for_llm(
        worker_context,
        character_name="一之濑明日奈",
    )
    assert model_visible_context["prompt_message_context"]["body_text"] == (
        "#napcat"
    )
    assert model_visible_context["prompt_message_context"]["mentions"] == []
    assert all(
        row.get("platform_message_id") != "active-platform-message-id"
        for row in worker_context["chat_history_recent"]
    )
    assert all(
        row.get("platform_message_id") != "active-platform-message-id"
        for row in worker_context["chat_history_wide"]
    )
    assert any(
        "Earlier, 一之濑明日奈 discussed a topic." in line
        for line in model_visible_context["chat_history_recent"]
    )
    assert len(cognition_inputs) == 1
    promoted_memory = [
        row
        for row in cognition_inputs[0]["evidence"]
        if row["evidence_ref"]["source_kind"] == "promoted_memory"
    ]
    assert len(promoted_memory) == 1
    assert promoted_memory[0]["memory_metadata"]["stable_id"] == (
        "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
    )
    assert promoted_memory[0]["authority"] == "conditional_character_guidance"
    assert promoted_memory[0]["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
    )
    outcome = result["shared_memory_prewarm_outcome"]
    assert outcome["status"] == "completed"
    assert outcome["reason_code"] == "shared_memory_merged"
    assert outcome["retrieved_shared_count"] == 1
    assert outcome["merged_shared_count"] == 1
    merged_memory = result["rag_result"]["memory_evidence"]
    assert len(merged_memory) == 1
    merged_row = merged_memory[0]
    assert merged_row["memory_unit_id"] == (
        "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"
    )
    assert merged_row["memory_type"] == "defense_rule"
    assert merged_row["source_kind"] == "seeded_manual"
    assert merged_row["authority"] == "seed"
    assert merged_row["status"] == "active"
    assert merged_row["scope_type"] == "global"
    assert merged_row["privacy_review"] == {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "boundary_assessment": "deidentified global meaning",
        "reviewer": "seed_tool",
    }
    assert "_id" not in merged_row
    assert "source_global_user_id" not in merged_row


@pytest.mark.asyncio
async def test_cognition_records_noneligible_prewarm_without_starting_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later graph cycle records its skip before model cognition starts."""

    _patch_cognition_runtime(monkeypatch)
    worker_started = False

    async def prewarm(_state) -> SharedMemoryPrewarmOutcomeV1:
        nonlocal worker_started
        worker_started = True
        return _ready_prewarm_outcome()

    monkeypatch.setattr(
        cognition_node,
        "run_first_cycle_shared_memory_prewarm",
        prewarm,
    )
    result = await cognition_node.call_cognition_subgraph(
        _global_state(cycle_index=1),
        commit=False,
    )

    outcome = result["shared_memory_prewarm_outcome"]
    assert worker_started is False
    assert outcome["status"] == "skipped"
    assert outcome["reason_code"] == "not_first_cycle"

    unsupported_state = _global_state(cycle_index=0)
    unsupported_episode = deepcopy(unsupported_state["cognitive_episode"])
    unsupported_episode["trigger_source"] = "scheduled_tick"
    unsupported_state["cognitive_episode"] = unsupported_episode
    unsupported_result = await cognition_node.call_cognition_subgraph(
        unsupported_state,
        commit=False,
    )
    unsupported_outcome = unsupported_result[
        "shared_memory_prewarm_outcome"
    ]
    assert worker_started is False
    assert unsupported_outcome["status"] == "skipped"
    assert unsupported_outcome["reason_code"] == "unsupported_episode"


@pytest.mark.asyncio
async def test_cognition_cancellation_publishes_no_prewarm_outcome_or_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation clears the checkpoint before cognition observation."""

    _patch_cognition_runtime(monkeypatch)
    boundary_calls: list[dict[str, object]] = []

    async def record_boundary(**kwargs: object) -> None:
        boundary_calls.append(kwargs)

    async def cancelled_prewarm(_state) -> SharedMemoryPrewarmOutcomeV1:
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        cognition_node,
        "record_continuity_boundary_event",
        record_boundary,
    )
    monkeypatch.setattr(
        cognition_node,
        "run_first_cycle_shared_memory_prewarm",
        cancelled_prewarm,
    )
    ledger = create_v2_attempt_ledger("cognition-cancel-test")
    token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        with pytest.raises(asyncio.CancelledError):
            await cognition_node.call_cognition_subgraph(
                _global_state(cycle_index=0),
                commit=False,
            )
        assert snapshot_v2_shared_memory_prewarm_checkpoint() is None
        assert boundary_calls == []
    finally:
        reset_v2_attempt_ledger(token)


def test_dialog_semantic_projection_excludes_procedural_provider_metadata() -> None:
    """Cognition receives explicit dialog meaning without response mechanics."""

    episode = deepcopy(_input()["episode"])
    dialog_content = episode["percepts"][0]["content"]
    dialog_content["role_explicit_content"] = "当前用户请求当前角色回应。"
    dialog_content["response_operation"] = {
        "operation": "当前角色提供回复内容。",
        "response_owner_role": "当前角色",
        "response_content_provider_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "无",
        "embedded_target_role": "无",
    }

    projection = cognition_node._dialog_semantic_projection_text(episode)

    assert projection == dialog_content["role_explicit_content"]
    assert "response_content_provider_role" not in projection


@pytest.mark.asyncio
async def test_resolver_recurrence_commits_against_original_user_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _input()
    original = payload["mutable_state"]
    replacement = dict(original)
    replacement["relationship"] = dict(original["relationship"])
    replacement["relationship"]["trust"] = 10
    captured: dict[str, object] = {}

    async def replace_user(owner: str, expected: dict, next_state: dict) -> bool:
        captured["owner"] = owner
        captured["expected"] = expected
        captured["replacement"] = next_state
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_user_cognition_state",
        replace_user,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "user",
            "owner_key": "user-1",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(output)
    assert captured["owner"] == "user-1"
    assert captured["expected"] == original
    assert captured["replacement"] == replacement


@pytest.mark.asyncio
async def test_cognition_state_version_conflict_is_retryable_before_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale cognition base is classified for one clean graph retry."""

    payload = _input()
    original = payload["mutable_state"]
    replacement = deepcopy(original)
    replacement["relationship"] = dict(original["relationship"])
    replacement["relationship"]["trust"] = 10

    async def reject_commit(
        _owner: str,
        _expected: dict,
        _replacement: dict,
    ) -> bool:
        return False

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_user_cognition_state",
        reject_commit,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "user",
            "owner_key": "user-1",
            "expected_previous_state": original,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }

    with pytest.raises(
        cognition_node.CognitionExecutionError,
        match="version conflict",
    ) as error_info:
        await cognition_node.commit_cognition_output(output)

    error = error_info.value
    assert error.error_code == "version_conflict"
    assert error.stage == "cognition.persistence"
    assert error.safe_checkpoint == "pre_state_commit"
    assert error.retryable is True


def test_current_continuation_uses_exact_private_goal_ref() -> None:
    payload = _input()
    replacement = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=payload["mutable_state"]["updated_at"],
    )
    output = {
        "state_projection": {
            "continuation_goal_ref": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:user:current",
            },
        },
    }
    continuation = cognition_node._canonical_goal_continuation_ref(
        output,
        {"cognitive_episode": payload["episode"]},
        replacement,
    )
    assert continuation["goal_ref"]["entity_id"] == (
        "goal:ordinary_response:user:current"
    )


def test_global_projection_preserves_exact_private_monologue() -> None:
    """Global residue state receives G subjectivity rather than goal analysis."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar object",
            "reason": "the observation does not identify the object",
            "cause_summary": "an unfamiliar object appeared",
        },
        "private_monologue": (
            "I am curious, but I do not want to pretend I recognize it."
        ),
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what the unfamiliar object is",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe its visible form and leave its identity unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["internal_monologue"] == output["private_monologue"]
    assert projected["internal_monologue"] != (
        output["active_character_goal"]["reason"]
    )


def test_global_projection_supplies_consolidation_interaction_subtext() -> None:
    """Project the goal reason and private monologue into separate fields."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    reason = "the compass reacts to a direction I cannot see"
    private_monologue = "I should ask what makes the needle move before guessing."
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar compass",
            "reason": reason,
            "cause_summary": "the compass needle moved without an obvious cause",
        },
        "private_monologue": private_monologue,
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what makes the compass needle move",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe the movement and leave its cause unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["interaction_subtext"] == reason
    assert projected["internal_monologue"] == private_monologue
    assert projected["interaction_subtext"] != projected["internal_monologue"]


def test_rag_memory_authority_maps_self_guidance_to_conditional_context() -> None:
    """Canonical projected self-guidance gets a typed cognition row."""

    evidence = cognition_node._rag_evidence(
        {
            "memory_evidence": [{
                "memory_unit_id": "guidance-unit-1",
                "memory_type": "defense_rule",
                "content": "A certified character guidance rule.",
                "source_kind": "reflection_inferred",
                "scope_type": "global",
                "authority": "reflection_promoted",
                "status": "active",
                "privacy_review": {
                    "global_applicability": "global",
                    "target_specific_meaning_removed": True,
                    "affects_identity_or_boundaries": False,
                    "private_detail_risk": "low",
                    "user_details_removed": True,
                    "boundary_assessment": "deidentified global meaning",
                    "reviewer": "automated_llm",
                },
            }],
        },
        "2026-06-08T00:00:00Z",
        current_user_id="user-1",
    )

    assert len(evidence) == 1
    row = evidence[0]
    assert row["authority"] == "conditional_character_guidance"
    assert row["memory_scope"] == "shared_character_or_world"
    assert row["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:guidance-unit-1"
    )
    assert row["memory_metadata"]["stable_id"] == "guidance-unit-1"
    assert row["memory_metadata"]["memory_type"] == "defense_rule"
    assert row["memory_metadata"]["status"] == "active"
    assert row["memory_metadata"]["privacy_review"]["reviewer"] == (
        "automated_llm"
    )


def test_rag_memory_rejects_unmarked_promoted_memory_source() -> None:
    """Promoted-memory evidence requires its canonical typed source id."""

    row = {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "promoted_memory",
            "source_id": "memory:unmarked",
            "occurred_at": "2026-06-08T00:00:00Z",
            "semantic_summary": "unmarked memory",
        },
        "semantic_text": "unmarked memory",
        "visible_to": ["q:event_agency"],
        "authority": "character_world_context",
        "memory_scope": "shared_character_or_world",
        "memory_metadata": {},
    }
    with pytest.raises(CognitionContractError, match="canonical"):
        _validate_evidence_rows([row])


def test_promoted_reflection_context_rejects_incomplete_legacy_certificate() -> None:
    """Legacy reflection rows cannot enter cognition without certification."""

    evidence = cognition_node._promoted_reflection_evidence(
        {
            "promoted_lore": [{
                "memory_name": "Legacy memory",
                "content": "A bounded legacy memory without the current scope certificate.",
                "memory_unit_id": "legacy-fact-unit",
                "memory_type": "fact",
                "source_kind": "reflection_inferred",
                "source_global_user_id": "",
                "authority": "reflection_promoted",
                "status": "active",
                "scope_type": "global",
                "privacy_review": {
                    "private_detail_risk": "low",
                    "user_details_removed": True,
                    "boundary_assessment": "Generic deidentified background meaning.",
                    "reviewer": "automated_llm",
                },
                "updated_at": "2026-06-08T00:00:00Z",
            }],
        },
        "2026-06-08T00:00:00Z",
    )

    assert evidence == []


def test_promoted_reflection_context_maps_certified_rows_through_typed_memory_contract() -> None:
    """Certified reflection context uses the canonical promoted-memory shape."""

    privacy_review = {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "boundary_assessment": "The meaning is deidentified and global.",
        "reviewer": "automated_llm",
    }
    evidence = cognition_node._promoted_reflection_evidence(
        {
            "promoted_self_guidance": [{
                "memory_name": "Certified guidance",
                "content": "A certified character guidance rule.",
                "memory_unit_id": "reflection-guidance-1",
                "memory_type": "defense_rule",
                "source_kind": "reflection_inferred",
                "source_global_user_id": "",
                "authority": "reflection_promoted",
                "status": "active",
                "scope_type": "global",
                "privacy_review": privacy_review,
                "updated_at": "2026-06-08T00:00:00Z",
            }],
        },
        "2026-06-08T00:00:00Z",
    )

    assert len(evidence) == 1
    row = evidence[0]
    assert row["evidence_ref"]["source_kind"] == "promoted_memory"
    assert row["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:reflection-guidance-1"
    )
    assert row["authority"] == "conditional_character_guidance"
    assert row["memory_scope"] == "shared_character_or_world"
    assert row["memory_metadata"] == {
        "stable_id": "reflection-guidance-1",
        "memory_type": "defense_rule",
        "source_kind": "reflection_inferred",
        "authority": "reflection_promoted",
        "status": "active",
        "scope_type": "global",
        "privacy_review": privacy_review,
    }
    row["evidence_handle"] = "e1"
    _validate_evidence_rows(evidence)


@pytest.mark.asyncio
async def test_persona_character_commit_reads_canonical_state_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_character_production_state(updated_at="2026-01-01T00:00:00Z")
    replacement = build_character_production_state(updated_at="2026-01-01T00:00:00.000001Z")
    captured: dict[str, object] = {}

    async def replace_character(*, expected_updated_at: str, replacement: dict) -> bool:
        captured["expected"] = expected_updated_at
        captured["replacement"] = replacement
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_character_cognition_state",
        replace_character,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "character",
            "owner_key": "character",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(
        output,
        expected_character_updated_at=original["updated_at"],
    )
    assert captured["expected"] == original["updated_at"]
    assert captured["replacement"] == replacement
