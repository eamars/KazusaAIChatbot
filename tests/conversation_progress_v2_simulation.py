"""Deterministic production-contract simulation for long V2 threads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from kazusa_ai_chatbot.conversation_progress.compaction import (
    validate_block,
)
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    compose_recorder_delta,
    event_handle_map,
    source_handle_map,
    validate_event_observation_batch,
    validate_scene_observation,
)
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationEpisodeBlockV1,
    ConversationLogicalTurnV1,
    ConversationProgressRecordInput,
    ConversationProgressStateV2,
)
from kazusa_ai_chatbot.conversation_progress.repository import (
    prepare_progress_write,
    validate_active_packet,
)
from tests.conversation_progress_v2_helpers import (
    SCOPE,
    event_observation_batch,
    new_event_observation,
    scene_observation,
    unchanged_event_observation,
)

_BASE_TIME = datetime(2026, 7, 28, 8, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class LongThreadSimulation:
    """Final packet, block graph, and losslessness evidence for one replay."""

    packet: ConversationProgressStateV2
    blocks: dict[str, ConversationEpisodeBlockV1]
    interaction_turns: list[ConversationLogicalTurnV1]
    archived_event_ids: set[str]
    protected_block_ids: set[str]
    compaction_count: int
    hierarchical_compaction_count: int

    def reachable_block_ids(self) -> set[str]:
        """Return the transitive block graph protected by the final packet."""

        reachable: set[str] = set()
        pending = list(self.packet["compacted_block_refs"])
        while pending:
            block_id = pending.pop()
            if block_id in reachable:
                continue
            block = self.blocks[block_id]
            reachable.add(block_id)
            pending.extend(block["source_block_ids"])
        return reachable

    def reachable_archived_event_ids(self) -> set[str]:
        """Return every event represented in the protected block graph."""

        event_ids: set[str] = set()
        for block_id in self.reachable_block_ids():
            event_ids.update(
                event["event_id"]
                for event in self.blocks[block_id]["events"]
            )
        return event_ids


def simulate_long_thread(turn_count: int) -> LongThreadSimulation:
    """Replay settled turns through canonical delta and compaction boundaries."""

    if turn_count <= 0:
        raise ValueError("turn_count must be positive")

    active_packet: ConversationProgressStateV2 | None = None
    blocks: dict[str, ConversationEpisodeBlockV1] = {}
    turns: list[ConversationLogicalTurnV1] = []
    archived_event_ids: set[str] = set()
    protected_block_ids: set[str] = set()
    compaction_count = 0
    hierarchical_count = 0

    for turn_number in range(1, turn_count + 1):
        timestamp = (
            _BASE_TIME + timedelta(minutes=turn_number)
        ).isoformat()
        row_id = f"capacity-row-{turn_number:03d}"
        source_ref = {
            "ref_kind": "conversation_row",
            "ref_id": row_id,
            "occurred_at": timestamp,
        }
        logical_turn: ConversationLogicalTurnV1 = {
            "turn_id": f"row:{row_id}",
            "role": "user",
            "occurred_at": timestamp,
            "display_name": "Capacity User",
            "fragments": [f"bounded continuation turn {turn_number}"],
            "conversation_row_ids": [row_id],
            "llm_trace_id": "",
            "platform_user_id": "capacity-platform-user",
            "global_user_id": SCOPE.global_user_id,
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "reply_context": {},
        }
        turns.append(logical_turn)

        active_blocks = _reachable_blocks(
            active_packet=active_packet,
            blocks=blocks,
        )
        new_events = [new_event_observation(
            summary=f"completed incidental event from turn {turn_number}",
            lifecycle_change="concluded",
            relevance="history",
            source_turn_handles=["current_input"],
        )]
        if turn_number == 1:
            new_events.insert(0, new_event_observation(
                summary=(
                    "the previously selected action was completed "
                    "and evaluated"
                ),
                lifecycle_change="concluded",
                relevance="decision",
                source_turn_handles=["current_input"],
            ))
        record_input = _record_input(
            active_packet=active_packet,
            logical_turn=logical_turn,
            source_ref=source_ref,
        )
        prior_handles = event_handle_map(record_input)
        event_updates = validate_event_observation_batch(
            event_observation_batch(
                existing_events=[
                    unchanged_event_observation(event_handle=handle)
                    for handle in prior_handles
                ],
                new_events=new_events,
            ),
            record_input=record_input,
            supplied_event_handles=set(prior_handles),
            supplied_source_handles=set(source_handle_map(record_input)),
        )
        scene_update = validate_scene_observation(
            scene_observation(),
            record_input=record_input,
        )
        validated_delta = compose_recorder_delta(
            scene_observation=scene_update,
            event_updates=event_updates,
        )
        prepared = prepare_progress_write(
            record_input=record_input,
            delta=validated_delta,
            active_blocks=active_blocks,
        )
        protected_block_ids = set(prepared.protected_block_ids)

        if prepared.block is not None:
            compaction_count += 1
            block = deepcopy(prepared.block)
            blocks[block["block_id"]] = block
            if block["source_block_ids"]:
                hierarchical_count += 1
            for source_block_id in prepared.source_block_ids:
                blocks[source_block_id]["superseded_by_block_id"] = (
                    block["block_id"]
                )
                validate_block(blocks[source_block_id])
            archived_event_ids.update(
                event["event_id"] for event in block["events"]
            )
        active_packet = prepared.packet
        validate_active_packet(active_packet)

    if active_packet is None:
        raise AssertionError("positive simulation did not produce a packet")
    for block in blocks.values():
        validate_block(block)
    return LongThreadSimulation(
        packet=active_packet,
        blocks=blocks,
        interaction_turns=turns,
        archived_event_ids=archived_event_ids,
        protected_block_ids=protected_block_ids,
        compaction_count=compaction_count,
        hierarchical_compaction_count=hierarchical_count,
    )


def _reachable_blocks(
    *,
    active_packet: ConversationProgressStateV2 | None,
    blocks: dict[str, ConversationEpisodeBlockV1],
) -> list[ConversationEpisodeBlockV1]:
    """Load the complete in-memory graph protected by active roots."""

    if active_packet is None:
        return []
    reachable: list[ConversationEpisodeBlockV1] = []
    seen: set[str] = set()
    pending = list(active_packet["compacted_block_refs"])
    while pending:
        block_id = pending.pop(0)
        if block_id in seen:
            raise AssertionError("simulation block graph is not a tree")
        block = blocks[block_id]
        seen.add(block_id)
        reachable.append(block)
        pending.extend(block["source_block_ids"])
    return reachable


def _record_input(
    *,
    active_packet: ConversationProgressStateV2 | None,
    logical_turn: ConversationLogicalTurnV1,
    source_ref: dict[str, str],
) -> ConversationProgressRecordInput:
    """Build one exact settled-turn input for the production write preparer."""

    return {
        "scope": SCOPE,
        "storage_timestamp_utc": logical_turn["occurred_at"],
        "character_name": "Capacity Character",
        "prior_episode_state": deepcopy(active_packet),
        "decontextualized_input": logical_turn["fragments"][0],
        "interaction_logical_turns": [deepcopy(logical_turn)],
        "current_turn_source_refs": [deepcopy(source_ref)],
        "turn_outcome": "visible_response",
        "content_plan": {
            "semantic_content": "continue the active thread",
            "surface_intent": "respond",
        },
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "final_dialog": ["bounded response"],
        "boundary_profile": {
            "self_integrity": 0.8,
            "control_sensitivity": 0.5,
            "compliance_strategy": "resist",
            "relational_override": 0.4,
            "control_intimacy_misread": 0.1,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.5,
        },
    }
