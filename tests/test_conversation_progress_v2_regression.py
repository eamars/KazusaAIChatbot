"""Production-path regression and long-thread capacity proof for V2."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    RESPONSE_OPERATION_METADATA_KEY,
    ROLE_EXPLICIT_CONTENT_METADATA_KEY,
)
from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT_CAP,
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.conversation_progress.compaction import (
    ConversationCompactionContractError,
    validate_block,
)
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    compose_recorder_delta,
    event_handle_map,
    source_handle_map,
    validate_event_observation_batch,
    validate_scene_observation,
)
from kazusa_ai_chatbot.conversation_progress.history import (
    assemble_logical_turns,
    logical_turn_source_refs,
    select_recent_logical_turns,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_ACTIVE_BLOCK_REFS,
    MAX_ACTIVE_EVENTS,
    MAX_BLOCK_GRAPH_DEPTH,
    MAX_BLOCK_EVENTS,
    MAX_CONTINUATION_CHARS,
    MAX_PROGRESS_EVIDENCE_CHARS,
    MAX_PROGRESS_SCENE_CHARS,
    MAX_RECENT_TURN_REFS,
    MAX_REACHABLE_BLOCK_REFS,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
    continuation_projection_chars,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    prepare_progress_write,
    validate_active_packet,
)
from kazusa_ai_chatbot.db import conversation as conversation_db
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)
from tests.conversation_progress_v2_helpers import (
    SCOPE,
    SOURCE_REF,
    changed_event_observation,
    event,
    event_observation_batch,
    logical_turn,
    new_event_observation,
    packet,
    record_input,
    scene_observation,
    unchanged_event_observation,
)
from tests.conversation_progress_v2_simulation import (
    simulate_long_thread,
)

_COGNITION_NOW = "2026-07-28T09:30:00Z"
_FULL_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "conversation_progress_v2"
    / "asuna_houjing_long_thread.json"
)


def _load_full_fixture() -> dict[str, Any]:
    """Load the source-faithful long-thread fixture with exact top-level shape."""

    payload = json.loads(_FULL_FIXTURE_PATH.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version",
        "source",
        "scope",
        "messages",
    }:
        raise AssertionError("long-thread fixture fields changed")
    if (
        payload["schema_version"]
        != "conversation_progress_v2_regression_fixture.v1"
    ):
        raise AssertionError("long-thread fixture schema changed")
    if not isinstance(payload["scope"], dict):
        raise AssertionError("long-thread fixture scope is invalid")
    if not isinstance(payload["messages"], list):
        raise AssertionError("long-thread fixture messages are invalid")
    return payload


class _FixtureCursor:
    """Small Motor cursor double that preserves query ordering semantics."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self._limit = len(rows)

    def sort(self, field_name: str, direction: int) -> _FixtureCursor:
        self._rows.sort(
            key=lambda row: row[field_name],
            reverse=direction < 0,
        )
        return self

    def limit(self, limit: int) -> _FixtureCursor:
        self._limit = limit
        return self

    async def to_list(self, *, length: int) -> list[dict[str, object]]:
        return deepcopy(self._rows[:min(length, self._limit)])


class _FixtureCollection:
    """Apply the production participant query to the source-faithful rows."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.query: dict[str, object] = {}

    def find(
        self,
        query: dict[str, object],
        *,
        projection: dict[str, int],
    ) -> _FixtureCursor:
        if projection != {"embedding": 0}:
            raise AssertionError("participant projection changed")
        self.query = deepcopy(query)
        selected = [
            row for row in self._rows
            if _matches_participant_query(row, query)
        ]
        return _FixtureCursor(selected)


class _GoalPromptBoundaryReached(AssertionError):
    """Stop exactly when the completed event reaches the goal owner."""


class _GoalPromptCapture:
    """Capture the actual serialized cognition input without model semantics."""

    def __init__(self) -> None:
        self.system_prompt = ""
        self.human_payload = ""
        self.payload: dict[str, Any] = {}

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.system_prompt = str(getattr(messages[0], "content", ""))
        self.human_payload = str(getattr(messages[-1], "content", ""))
        self.payload = json.loads(self.human_payload)
        raise _GoalPromptBoundaryReached


def _matches_participant_query(
    row: dict[str, object],
    query: dict[str, object],
) -> bool:
    """Evaluate the exact subset of Mongo syntax emitted by the owner."""

    if row["platform"] != query["platform"]:
        return False
    if row["platform_channel_id"] != query["platform_channel_id"]:
        return False
    branches = query["$or"]
    if not isinstance(branches, list):
        raise AssertionError("participant query must contain an $or list")
    return any(_matches_query_branch(row, branch) for branch in branches)


def _matches_query_branch(
    row: dict[str, object],
    branch: dict[str, object],
) -> bool:
    """Match equality and array-membership fields from one query branch."""

    for field_name, expected in branch.items():
        actual = row.get(field_name)
        if isinstance(actual, list):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


async def _participant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, Any],
]:
    """Load participant rows through the production DB query function."""

    fixture = _load_full_fixture()
    scope = fixture["scope"]
    collection = _FixtureCollection(fixture["messages"])
    fake_db = SimpleNamespace(conversation_history=collection)

    async def _get_db() -> SimpleNamespace:
        return fake_db

    monkeypatch.setattr(conversation_db, "get_db", _get_db)
    rows = await conversation_db.get_participant_conversation_history(
        platform=scope["platform"],
        platform_channel_id=scope["platform_channel_id"],
        current_global_user_id=scope["participant_global_user_id"],
        platform_bot_id=scope["bot_platform_user_id"],
        excluded_row_ids=[],
        limit=128,
    )
    return list(rows), collection.query, fixture


def _character_profile(character_name: str) -> dict[str, Any]:
    """Build the minimal valid cognition identity used by the connector."""

    profile = canonical_character_identity(marker="progress-regression")
    profile["name"] = character_name
    profile["personality_brief"] = {
        "mbti": "test",
        "logic": "Advance the active thread using grounded chronology.",
        "tempo": "measured",
        "defense": "Preserve character judgment and explicit boundaries.",
        "quirks": "Prefer one concrete continuation.",
        "taboos": "Avoid accidental resets of completed interaction events.",
    }
    return profile


@pytest.mark.asyncio
async def test_source_faithful_regression_projects_key_details_to_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the original completed action reaches cognition intact."""

    participant_rows, query, fixture = await _participant_rows(monkeypatch)
    scope = fixture["scope"]
    participant_global_user_id = scope["participant_global_user_id"]
    character_name = next(
        row["display_name"]
        for row in fixture["messages"]
        if row["role"] == "assistant"
    )
    turns = assemble_logical_turns(
        rows=participant_rows,
        excluded_row_ids=[],
    )
    assert len(turns) == 23
    unrelated_user_ids = {
        row["global_user_id"]
        for row in fixture["messages"]
        if (
            row["role"] == "user"
            and row["global_user_id"] != participant_global_user_id
        )
    }
    assert unrelated_user_ids
    assert all(
        turn["global_user_id"] not in unrelated_user_ids
        for turn in turns
    )
    assert (
        query["$or"][0]["global_user_id"]
        == participant_global_user_id
    )
    segmented = [
        turn for turn in turns
        if (
            turn["role"] == "assistant"
            and turn["llm_trace_id"] == "trace_006"
        )
    ]
    assert len(segmented) == 1
    assert len(segmented[0]["fragments"]) == 7

    response_turn_index = turns.index(segmented[0])
    completion_user_turn = turns[response_turn_index - 1]
    assert completion_user_turn["role"] == "user"
    assert completion_user_turn["conversation_row_ids"] == ["row_0022"]
    completion_sources = logical_turn_source_refs([
        completion_user_turn,
        segmented[0],
    ])
    completion_event_sources = completion_sources
    assert completion_event_sources == [{
        "ref_kind": "conversation_row",
        "ref_id": completion_user_turn["conversation_row_ids"][0],
        "occurred_at": completion_user_turn["occurred_at"],
    }, {
        "ref_kind": "llm_trace",
        "ref_id": "trace_006",
        "occurred_at": segmented[0]["occurred_at"],
    }]
    current_record = record_input()
    current_record["scope"] = SCOPE.__class__(
        platform=scope["platform"],
        platform_channel_id=scope["platform_channel_id"],
        global_user_id=participant_global_user_id,
    )
    current_record["storage_timestamp_utc"] = segmented[0]["occurred_at"]
    current_record["character_name"] = character_name
    current_record["interaction_logical_turns"] = (
        select_recent_logical_turns(
            turns[:response_turn_index + 1],
            limit=10,
        )
    )
    current_record["current_turn_source_refs"] = completion_sources
    current_record["decontextualized_input"] = (
        completion_user_turn["fragments"][0]
    )
    current_record["final_dialog"] = list(segmented[0]["fragments"])
    completed_summary = (
        f"the current user completed {character_name}'s requested neck and "
        "shoulder massage, which she accepted and evaluated"
    )
    completed_outcome = (
        f"{character_name} relaxed and judged the massage at least passing"
    )
    new_event = new_event_observation(
        summary=completed_summary,
        lifecycle_change="concluded",
        relevance="decision",
        source_turn_handles=["current_input", "current_response"],
        actor="the current user",
        action="completed the requested massage",
        object_=f"{character_name}'s neck and shoulders",
    )
    new_event["beneficiary"] = character_name
    new_event["outcome"] = completed_outcome
    event_candidate = event_observation_batch(new_events=[new_event])
    event_updates = validate_event_observation_batch(
        event_candidate,
        record_input=current_record,
        supplied_event_handles=set(event_handle_map(current_record)),
        supplied_source_handles=set(source_handle_map(current_record)),
    )
    scene_update = validate_scene_observation(
        scene_observation(),
        record_input=current_record,
    )
    validated_delta = compose_recorder_delta(
        scene_observation=scene_update,
        event_updates=event_updates,
    )
    prepared = prepare_progress_write(
        record_input=current_record,
        delta=validated_delta,
    )
    active_packet = validate_active_packet(prepared.packet)
    completed_events = [
        row for row in active_packet["events"]
        if (
            row["state"] == "completed"
            and row["retention"] == "decision_critical"
        )
    ]
    assert len(completed_events) == 1
    completed_event = completed_events[0]
    assert completed_event["semantic_summary"] == completed_summary
    assert completed_event["actor"] == "the current user"
    assert completed_event["action"] == "completed the requested massage"
    assert (
        completed_event["object"]
        == f"{character_name}'s neck and shoulders"
    )
    assert completed_event["beneficiary"] == character_name
    assert completed_event["outcome"] == completed_outcome
    assert completed_event["source_refs"] == completion_event_sources

    final_user_turn = turns[-1]
    assert final_user_turn["conversation_row_ids"] == ["row_0074"]
    historical_turns = select_recent_logical_turns(
        turns[:-1],
        limit=10,
    )
    assert all(
        "row_0022" not in turn["conversation_row_ids"]
        for turn in historical_turns
    )
    progress_prompt = build_progress_prompt(
        active_packet=active_packet,
        interaction_logical_turns=historical_turns,
    )
    mutable_state = build_acquaintance_user_state(
        global_user_id=participant_global_user_id,
        updated_at=_COGNITION_NOW,
    )
    character_state = build_character_production_state(
        updated_at=_COGNITION_NOW,
    )
    episode = canonical_episode(
        episode_id="progress-regression-next-step",
        content=final_user_turn["fragments"][0],
        current_global_user_id=participant_global_user_id,
        metadata={
            ROLE_EXPLICIT_CONTENT_METADATA_KEY: (
                "The current user reports that the earlobe massage is "
                "complete and asks the current character to choose the next "
                "location."
            ),
            RESPONSE_OPERATION_METADATA_KEY: {
                "operation": (
                    "the current character chooses the next requested "
                    "touch location"
                ),
                "response_owner_role": CURRENT_CHARACTER_ROLE,
                "selection_owner_role": CURRENT_CHARACTER_ROLE,
                "selection_required": True,
                "embedded_actor_role": CURRENT_USER_ROLE,
                "embedded_target_role": CURRENT_CHARACTER_ROLE,
            },
        },
    )
    cognition_input = build_cognition_input_from_global_state(
        {
            "cognitive_episode": episode,
            "global_user_id": participant_global_user_id,
            "user_input": final_user_turn["fragments"][0],
            "decontextualized_input": final_user_turn["fragments"][0],
            "conversation_progress": progress_prompt,
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
            "character_profile": _character_profile(character_name),
        },
        mutable_state=mutable_state,
        character_state=character_state,
    )
    progress_evidence = [
        row for row in cognition_input["evidence"]
        if row["evidence_ref"]["source_id"] == (
            "conversation-progress-event:"
            f"{completed_event['event_id']}"
        )
    ]
    assert len(progress_evidence) == 1
    projected_event = progress_evidence[0]
    progress_handle = projected_event["evidence_handle"]
    projected_text = projected_event["semantic_text"]
    for required_detail in (
        completed_summary,
        "state=completed",
        "retention=decision_critical",
        "actor=the current user",
        "action=completed the requested massage",
        f"object={character_name}'s neck and shoulders",
        f"beneficiary={character_name}",
        f"outcome={completed_outcome}",
    ):
        assert required_detail in projected_text

    state_projection = project_state_for_prompt(
        cognition_input["mutable_state"],
        character_constraints=cognition_input["character_constraints"],
        character_identity_context=cognition_input.get(
            "character_identity_context",
            canonical_identity_context(),
        ),
        evidence=cognition_input["evidence"],
    )
    branch_context = facade._branch_context(
        state_projection,
        cognition_input["mutable_state"],
        cognition_input["evidence"],
        scene_context=cognition_input["scene_context"],
        private_continuity_context=(
            cognition_input["private_continuity_context"]
        ),
    )
    goal_capture = _GoalPromptCapture()
    with pytest.raises(_GoalPromptBoundaryReached):
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary-response",
            },
            branch_context,
            cognition_input["evidence"],
            SimpleNamespace(
                llm=goal_capture,
                goal_ordinary_response_config=object(),
                goal_active_branch_config=object(),
            ),
        )

    assert "conversation_evidence_relations" not in goal_capture.system_prompt
    assert (
        len(goal_capture.human_payload)
        <= GOAL_COGNITION_PROMPT_CAP
    )
    assert goal_capture.payload[
        "conversation_progress_constraints"
    ] == [{
        "evidence_handle": progress_handle,
        "semantic_text": projected_text,
    }]
    assert "evidence" not in goal_capture.payload
    prompt_event = goal_capture.payload[
        "conversation_progress_constraints"
    ][0]
    assert prompt_event["semantic_text"] == projected_text


def test_explicit_deliberate_repeat_can_reopen_completed_event() -> None:
    """Prove V2 permits model-owned reopening with fresh source lineage."""

    prior = packet(events=[event(
        event_id="completed-action",
        summary="the selected action was completed",
        state="completed",
        retention="decision_critical",
    )])
    current_turn = logical_turn(
        turn_id="row:deliberate-repeat-row",
        row_id="deliberate-repeat-row",
    )
    source_ref = {
        "ref_kind": "conversation_row",
        "ref_id": "deliberate-repeat-row",
        "occurred_at": current_turn["occurred_at"],
    }
    current_record = record_input(prior_packet=prior)
    current_record["interaction_logical_turns"] = [current_turn]
    current_record["current_turn_source_refs"] = [source_ref]
    event_candidate = event_observation_batch(existing_events=[
        changed_event_observation(
            event_handle="e1",
            summary=(
                "the user explicitly requests a deliberate repeat of the "
                "completed action"
            ),
            lifecycle_change="reopened",
            relevance="decision",
            source_turn_handles=["current_input"],
        ),
    ])
    event_updates = validate_event_observation_batch(
        event_candidate,
        record_input=current_record,
        supplied_event_handles=set(event_handle_map(current_record)),
        supplied_source_handles=set(source_handle_map(current_record)),
    )
    scene_update = validate_scene_observation(
        scene_observation(),
        record_input=current_record,
    )
    validated_delta = compose_recorder_delta(
        scene_observation=scene_update,
        event_updates=event_updates,
    )
    reopened = prepare_progress_write(
        record_input=current_record,
        delta=validated_delta,
    ).packet
    reopened_event = next(
        row for row in reopened["events"]
        if row["event_id"] == "completed-action"
    )
    assert reopened_event["state"] == "open"
    assert reopened_event["source_refs"] == [SOURCE_REF, source_ref]


@pytest.mark.parametrize("turn_count", [20, 50, 100])
def test_long_thread_capacity_preserves_critical_and_archived_events(
    turn_count: int,
) -> None:
    """Prove hard caps and lossless block lineage at required capacities."""

    simulation = simulate_long_thread(turn_count)
    active_packet = validate_active_packet(simulation.packet)
    reachable_block_ids = simulation.reachable_block_ids()
    reachable_event_ids = simulation.reachable_archived_event_ids()

    assert active_packet["turn_count"] == turn_count
    assert len(active_packet["events"]) <= MAX_ACTIVE_EVENTS
    assert len(active_packet["recent_turn_refs"]) <= MAX_RECENT_TURN_REFS
    assert (
        len(active_packet["compacted_block_refs"])
        <= MAX_ACTIVE_BLOCK_REFS
    )
    critical_events = [
        row for row in active_packet["events"]
        if row["retention"] == "decision_critical"
    ]
    assert len(critical_events) == 1
    assert critical_events[0]["state"] == "completed"
    assert all(row["source_refs"] for row in active_packet["events"])
    assert simulation.archived_event_ids <= reachable_event_ids
    assert set(active_packet["compacted_block_refs"]) <= reachable_block_ids
    assert simulation.protected_block_ids == reachable_block_ids
    assert len(reachable_block_ids) <= MAX_REACHABLE_BLOCK_REFS
    for block_id in reachable_block_ids:
        block = validate_block(simulation.blocks[block_id])
        assert all(row["source_refs"] for row in block["events"])

    progress_prompt = build_progress_prompt(
        active_packet=active_packet,
        interaction_logical_turns=simulation.interaction_turns[-10:],
    )
    scene_chars, evidence_chars = continuation_projection_chars(
        progress_prompt,
        active_packet["updated_at"],
    )
    assert scene_chars <= MAX_PROGRESS_SCENE_CHARS
    assert evidence_chars <= MAX_PROGRESS_EVIDENCE_CHARS
    assert scene_chars + evidence_chars <= MAX_CONTINUATION_CHARS
    assert simulation.compaction_count > 0
    if turn_count == 100:
        assert simulation.hierarchical_compaction_count > 0
        assert len(reachable_block_ids) > len(
            active_packet["compacted_block_refs"]
        )


def test_balanced_compaction_survives_old_depth_failure_frontier() -> None:
    """Keep turn 593 below depth while preserving every exact event fact."""

    simulation = simulate_long_thread(593)
    active_packet = validate_active_packet(simulation.packet)
    reachable_block_ids = simulation.reachable_block_ids()
    reachable_event_ids = simulation.reachable_archived_event_ids()
    maximum_level = max(
        simulation.blocks[block_id]["level"]
        for block_id in reachable_block_ids
    )
    all_event_ids = {
        event_row["event_id"]
        for event_row in active_packet["events"]
    } | reachable_event_ids

    assert active_packet["turn_count"] == 593
    assert maximum_level <= MAX_BLOCK_GRAPH_DEPTH
    assert len(reachable_block_ids) < MAX_REACHABLE_BLOCK_REFS
    assert len(all_event_ids) == 594
    assert simulation.archived_event_ids <= reachable_event_ids


def test_maximum_history_capacity_reaches_node_cap_and_fails_closed() -> None:
    """Use 128 blocks plus 24 active slots, then reject the next fact."""

    simulation = simulate_long_thread(1047)
    active_packet = validate_active_packet(simulation.packet)
    reachable_block_ids = simulation.reachable_block_ids()
    archived_event_ids = simulation.reachable_archived_event_ids()
    active_event_ids = {
        event_row["event_id"]
        for event_row in active_packet["events"]
    }

    assert active_packet["turn_count"] == 1047
    assert len(active_event_ids) == MAX_ACTIVE_EVENTS
    assert len(archived_event_ids) == (
        MAX_REACHABLE_BLOCK_REFS * MAX_BLOCK_EVENTS
    )
    assert len(active_event_ids | archived_event_ids) == 1048
    assert len(active_packet["compacted_block_refs"]) == (
        MAX_ACTIVE_BLOCK_REFS
    )
    assert len(reachable_block_ids) == MAX_REACHABLE_BLOCK_REFS
    assert max(
        simulation.blocks[block_id]["level"]
        for block_id in reachable_block_ids
    ) == 6
    assert simulation.archived_event_ids == archived_event_ids

    next_timestamp = "2026-07-29T01:28:00+00:00"
    next_row_id = "capacity-row-1048"
    next_turn = logical_turn(
        turn_id=f"row:{next_row_id}",
        row_id=next_row_id,
    )
    next_turn["occurred_at"] = next_timestamp
    next_turn["fragments"] = ["bounded continuation turn 1048"]
    next_record = record_input(prior_packet=active_packet)
    next_record["storage_timestamp_utc"] = next_timestamp
    next_record["decontextualized_input"] = next_turn["fragments"][0]
    next_record["interaction_logical_turns"] = [next_turn]
    next_record["current_turn_source_refs"] = [{
        "ref_kind": "conversation_row",
        "ref_id": next_row_id,
        "occurred_at": next_timestamp,
    }]
    prior_handles = event_handle_map(next_record)
    event_updates = validate_event_observation_batch(
        event_observation_batch(
            existing_events=[
                unchanged_event_observation(event_handle=handle)
                for handle in prior_handles
            ],
            new_events=[new_event_observation(
                summary="completed incidental event from turn 1048",
                lifecycle_change="concluded",
                relevance="history",
                source_turn_handles=["current_input"],
            )],
        ),
        record_input=next_record,
        supplied_event_handles=set(prior_handles),
        supplied_source_handles=set(source_handle_map(next_record)),
    )
    scene_update = validate_scene_observation(
        scene_observation(),
        record_input=next_record,
    )
    candidate_delta = compose_recorder_delta(
        scene_observation=scene_update,
        event_updates=event_updates,
    )
    packet_before_candidate = deepcopy(simulation.packet)
    blocks_before_candidate = deepcopy(simulation.blocks)

    with pytest.raises(
        ConversationProgressContractError,
        match="active packet events exceeds its hard cap",
    ):
        prepare_progress_write(
            record_input=next_record,
            delta=candidate_delta,
            active_blocks=[
                simulation.blocks[block_id]
                for block_id in reachable_block_ids
            ],
        )

    assert simulation.packet == packet_before_candidate
    assert simulation.blocks == blocks_before_candidate
    assert len(simulation.blocks) == MAX_REACHABLE_BLOCK_REFS
