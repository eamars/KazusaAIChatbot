"""Deterministic pending-turn settlement and relevance-work ordering."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import heapq
import itertools
import time
from typing import Any, Literal

from kazusa_ai_chatbot.relevance import (
    FrontlineDecision,
    SettledRelevanceDecision,
    validate_frontline_decision,
    validate_settled_relevance_decision,
)


Scope = tuple[str, str, str]
FrontlineEvaluator = Callable[[Mapping[str, Any]], Awaitable[FrontlineDecision]]
SettledEvaluator = Callable[
    ["AssessmentLease", Mapping[str, Any]],
    Awaitable[SettledRelevanceDecision],
]


@dataclass(slots=True)
class PersistedChatFragment:
    """One persisted message retained as an assembled-turn fragment."""

    arrival_sequence: int
    scope: Scope
    author_platform_user_id: str
    author_global_user_id: str
    platform_message_id: str
    conversation_row_id: str
    storage_timestamp_utc: str
    enqueue_monotonic: float
    body_text: str
    semantic_target_labels: tuple[str, ...] = ()
    reply_target_label: str = "none"
    media_labels: tuple[str, ...] = ()
    media_descriptions: tuple[dict[str, Any], ...] = ()
    attachments: tuple[dict[str, Any], ...] = ()
    additional_media_present: bool = False
    request: Any | None = None
    future: Any | None = None
    pipeline_run_handle: Any | None = None
    queue_item: Any | None = None


@dataclass(slots=True)
class _PendingTurn:
    """Private mutable state for one candidate logical turn."""

    turn_id: str
    scope: Scope
    author_platform_user_id: str
    author_global_user_id: str
    leader_sequence: int
    response_owner_sequence: int
    fragments: list[PersistedChatFragment]
    status: Literal[
        "SETTLING",
        "ASSESSMENT_READY",
        "ASSESSING",
        "RUNNING",
        "DONE",
    ]
    created_at_monotonic: float
    last_fragment_at_monotonic: float
    eligible_at_monotonic: float
    hard_deadline_monotonic: float
    version: int = 1
    settled_assessment_count: int = 0
    wait_used: bool = False
    last_decision: SettledRelevanceDecision | None = None


@dataclass(frozen=True, slots=True)
class FrontlineOutcome:
    """Deterministic result after applying a frontline decision."""

    action: str
    turn_id: str = ""
    version: int = 0
    frozen_turn_id: str = ""


@dataclass(frozen=True, slots=True)
class AssessmentLease:
    """Versioned lease for one settled relevance assessment."""

    turn_id: str
    version: int
    observation_status: Literal[
        "more_time_available",
        "observation_complete",
    ]
    leader_sequence: int
    fragments: tuple[PersistedChatFragment, ...] = ()


@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    """Deterministic result after applying settled relevance."""

    response_action: str
    turn_id: str
    version: int
    wait_used: bool = False
    stale: bool = False
    claimable: bool = False


@dataclass(slots=True)
class _RelevanceWork:
    """One queued relevance call and its awaiting result future."""

    callback: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]


class TurnSettlementCoordinator:
    """Own pending turns, deadlines, ready order, and relevance serialization."""

    def __init__(
        self,
        *,
        frontline_evaluator: FrontlineEvaluator,
        settled_evaluator: SettledEvaluator,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the coordinator with both semantic stage callables."""

        self._frontline_evaluator = frontline_evaluator
        self._settled_evaluator = settled_evaluator
        self._clock = clock
        self._pending_turns: dict[str, _PendingTurn] = {}
        self._ready_heap: list[tuple[float, int, str, int]] = []
        self._turn_counter = itertools.count(1)
        self._state_condition = asyncio.Condition()
        self._work_lock = asyncio.Lock()
        self._work_queue: list[_RelevanceWork] = []
        self._work_task: asyncio.Task[None] | None = None

    async def evaluate_frontline(
        self,
        state: Mapping[str, Any],
    ) -> FrontlineDecision:
        """Evaluate one message through the FIFO relevance executor."""

        async def _evaluate() -> FrontlineDecision:
            return await self._frontline_evaluator(state)

        result = await self._run_relevance_work(_evaluate)
        return_value = validate_frontline_decision(result)
        return return_value

    async def apply_frontline_decision(
        self,
        fragment: PersistedChatFragment,
        decision: FrontlineDecision,
    ) -> FrontlineOutcome:
        """Apply one validated frontline action to deterministic turn state."""

        validated = validate_frontline_decision(decision)
        async with self._state_condition:
            action = validated["intake_action"]
            if action == "discard":
                return_value = FrontlineOutcome(action="discard")
                return return_value

            if action == "append":
                pending_turn = self._find_slot_turn_locked(
                    fragment,
                    validated["append_target"],
                )
                if pending_turn is None:
                    raise ValueError(
                        "frontline append target is not an open matching turn"
                    )
                self._append_fragment_locked(pending_turn, fragment)
                return_value = FrontlineOutcome(
                    action="append",
                    turn_id=pending_turn.turn_id,
                    version=pending_turn.version,
                )
                self._state_condition.notify_all()
                return return_value

            frozen_turn_id = self._freeze_oldest_open_turn_locked(fragment)
            pending_turn = self._create_turn_locked(fragment)
            return_value = FrontlineOutcome(
                action="start",
                turn_id=pending_turn.turn_id,
                version=pending_turn.version,
                frozen_turn_id=frozen_turn_id,
            )
            self._state_condition.notify_all()
            return return_value

    async def wait_for_assessment_ready(self) -> AssessmentLease:
        """Lease the globally earliest ready turn for settled relevance."""

        while True:
            async with self._state_condition:
                now = self._clock()
                lease = self._pop_ready_lease_locked(now)
                if lease is not None:
                    return lease

                next_eligible = self._next_eligible_time_locked()
                timeout = None
                if next_eligible is not None:
                    timeout = max(0.0, next_eligible - now)
                try:
                    if timeout is None:
                        await self._state_condition.wait()
                    else:
                        await asyncio.wait_for(
                            self._state_condition.wait(),
                            timeout=max(timeout, 0.001),
                        )
                except asyncio.TimeoutError:
                    continue

    async def evaluate_settled(
        self,
        lease: AssessmentLease,
        state: Mapping[str, Any],
    ) -> SettledRelevanceDecision:
        """Evaluate one leased turn through the same FIFO relevance executor."""

        evaluation_state = dict(state)
        evaluation_state["observation_status"] = lease.observation_status

        async def _evaluate() -> SettledRelevanceDecision:
            return await self._settled_evaluator(lease, evaluation_state)

        result = await self._run_relevance_work(_evaluate)
        validated = validate_settled_relevance_decision(
            result,
            observation_status=lease.observation_status,
        )
        return validated

    async def apply_settled_decision(
        self,
        lease: AssessmentLease,
        decision: SettledRelevanceDecision,
    ) -> SettlementOutcome:
        """Apply settled relevance, wait once, or expose a claimable proceed."""

        validated = validate_settled_relevance_decision(
            decision,
            observation_status=lease.observation_status,
        )
        async with self._state_condition:
            pending_turn = self._pending_turns.get(lease.turn_id)
            if pending_turn is None or pending_turn.status == "DONE":
                return_value = SettlementOutcome(
                    response_action="ignore",
                    turn_id=lease.turn_id,
                    version=lease.version,
                    stale=True,
                )
                return return_value

            if pending_turn.version != lease.version:
                pending_turn.status = "SETTLING"
                pending_turn.eligible_at_monotonic = (
                    pending_turn.hard_deadline_monotonic
                )
                self._schedule_ready_locked(pending_turn)
                return_value = SettlementOutcome(
                    response_action="ignore",
                    turn_id=pending_turn.turn_id,
                    version=pending_turn.version,
                    stale=True,
                )
                self._state_condition.notify_all()
                return return_value

            pending_turn.last_decision = validated
            if validated["response_action"] == "wait":
                if pending_turn.wait_used:
                    raise ValueError("settled wait extension was already used")
                pending_turn.wait_used = True
                pending_turn.status = "SETTLING"
                pending_turn.eligible_at_monotonic = (
                    pending_turn.hard_deadline_monotonic
                )
                self._schedule_ready_locked(pending_turn)
                return_value = SettlementOutcome(
                    response_action="wait",
                    turn_id=pending_turn.turn_id,
                    version=pending_turn.version,
                    wait_used=True,
                )
                self._state_condition.notify_all()
                return return_value

            pending_turn.status = "ASSESSMENT_READY"
            if validated["response_action"] == "ignore":
                pending_turn.status = "DONE"
            return_value = SettlementOutcome(
                response_action=validated["response_action"],
                turn_id=pending_turn.turn_id,
                version=pending_turn.version,
                wait_used=pending_turn.wait_used,
                claimable=validated["response_action"] == "proceed",
            )
            self._state_condition.notify_all()
            return return_value

    async def claim_for_cognition(self, turn_id: str, version: int) -> bool:
        """Atomically claim a matching settled proceed for cognition."""

        async with self._state_condition:
            pending_turn = self._pending_turns.get(turn_id)
            if pending_turn is None:
                return_value = False
                return return_value
            if pending_turn.version != version:
                return_value = False
                return return_value
            if pending_turn.status != "ASSESSMENT_READY":
                return_value = False
                return return_value
            if pending_turn.last_decision is None:
                return_value = False
                return return_value
            if pending_turn.last_decision["response_action"] != "proceed":
                return_value = False
                return return_value
            pending_turn.status = "RUNNING"
            return_value = True
            return return_value

    async def _run_relevance_work(
        self,
        callback: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Queue one semantic call and await its FIFO result."""

        future = asyncio.get_running_loop().create_future()
        work = _RelevanceWork(callback=callback, future=future)
        async with self._work_lock:
            self._work_queue.append(work)
            if self._work_task is None:
                self._work_task = asyncio.create_task(
                    self._drain_relevance_work()
                )
        result = await future
        return result

    async def _drain_relevance_work(self) -> None:
        """Run queued relevance callbacks one at a time in insertion order."""

        while True:
            async with self._work_lock:
                if not self._work_queue:
                    self._work_task = None
                    return
                work = self._work_queue.pop(0)
            try:
                result = await work.callback()
            except BaseException as exc:
                if not work.future.done():
                    work.future.set_exception(exc)
            else:
                if not work.future.done():
                    work.future.set_result(result)

    def _create_turn_locked(
        self,
        fragment: PersistedChatFragment,
    ) -> _PendingTurn:
        """Create one candidate turn and schedule its first eligibility."""

        now = self._clock()
        turn_id = f"turn-{next(self._turn_counter)}"
        is_private = fragment.scope[2] == "private"
        hard_deadline = now if is_private else now + 10.0
        eligible_at = now if is_private else min(now + 6.0, hard_deadline)
        pending_turn = _PendingTurn(
            turn_id=turn_id,
            scope=fragment.scope,
            author_platform_user_id=fragment.author_platform_user_id,
            author_global_user_id=fragment.author_global_user_id,
            leader_sequence=fragment.arrival_sequence,
            response_owner_sequence=fragment.arrival_sequence,
            fragments=[fragment],
            status="SETTLING",
            created_at_monotonic=now,
            last_fragment_at_monotonic=now,
            eligible_at_monotonic=eligible_at,
            hard_deadline_monotonic=hard_deadline,
        )
        self._pending_turns[turn_id] = pending_turn
        self._schedule_ready_locked(pending_turn)
        return pending_turn

    def _append_fragment_locked(
        self,
        pending_turn: _PendingTurn,
        fragment: PersistedChatFragment,
    ) -> None:
        """Append one fragment and invalidate any prior assessment."""

        now = self._clock()
        pending_turn.fragments.append(fragment)
        pending_turn.last_fragment_at_monotonic = now
        pending_turn.version += 1
        pending_turn.status = "SETTLING"
        pending_turn.eligible_at_monotonic = min(now + 6.0, pending_turn.hard_deadline_monotonic)
        self._schedule_ready_locked(pending_turn)

    def _freeze_oldest_open_turn_locked(
        self,
        fragment: PersistedChatFragment,
    ) -> str:
        """Freeze the oldest matching open turn when three slots are full."""

        open_turns = self._open_turns_locked(fragment)
        if len(open_turns) < 3:
            return_value = ""
            return return_value

        oldest = open_turns[0]
        oldest.status = "ASSESSMENT_READY"
        oldest.eligible_at_monotonic = self._clock()
        self._schedule_ready_locked(oldest)
        return_value = oldest.turn_id
        return return_value

    def _open_turns_locked(
        self,
        fragment: PersistedChatFragment,
    ) -> list[_PendingTurn]:
        """Return at most the current matching open turns in leader order."""

        turns = [
            turn
            for turn in self._pending_turns.values()
            if turn.status in {"SETTLING", "ASSESSING"}
            and turn.scope == fragment.scope
            and turn.author_platform_user_id == fragment.author_platform_user_id
        ]
        turns.sort(key=lambda turn: turn.leader_sequence)
        return_value = turns
        return return_value

    def _find_slot_turn_locked(
        self,
        fragment: PersistedChatFragment,
        append_target: str,
    ) -> _PendingTurn | None:
        """Map a model-facing slot to one matching open turn."""

        if append_target not in {"open_1", "open_2", "open_3"}:
            return None
        open_turns = self._open_turns_locked(fragment)
        slot_index = int(append_target.rsplit("_", maxsplit=1)[1]) - 1
        if slot_index >= len(open_turns):
            return None
        return_value = open_turns[slot_index]
        return return_value

    def _schedule_ready_locked(self, pending_turn: _PendingTurn) -> None:
        """Add one versioned turn token to the global ready heap."""

        heapq.heappush(
            self._ready_heap,
            (
                pending_turn.eligible_at_monotonic,
                pending_turn.leader_sequence,
                pending_turn.turn_id,
                pending_turn.version,
            ),
        )

    def _pop_ready_lease_locked(self, now: float) -> AssessmentLease | None:
        """Pop the earliest due current-version turn."""

        while self._ready_heap:
            eligible_at, _sequence, turn_id, version = self._ready_heap[0]
            if eligible_at > now:
                return None
            heapq.heappop(self._ready_heap)
            pending_turn = self._pending_turns.get(turn_id)
            if pending_turn is None:
                continue
            if pending_turn.version != version:
                continue
            if pending_turn.status not in {"SETTLING", "ASSESSMENT_READY"}:
                continue
            pending_turn.status = "ASSESSING"
            pending_turn.settled_assessment_count += 1
            observation_status = (
                "observation_complete"
                if now >= pending_turn.hard_deadline_monotonic
                or pending_turn.wait_used
                else "more_time_available"
            )
            return_value = AssessmentLease(
                turn_id=turn_id,
                version=version,
                observation_status=observation_status,
                leader_sequence=pending_turn.leader_sequence,
                fragments=tuple(pending_turn.fragments),
            )
            return return_value
        return None

    def _next_eligible_time_locked(self) -> float | None:
        """Return the next live timer used by the wait loop."""

        while self._ready_heap:
            eligible_at, _sequence, turn_id, version = self._ready_heap[0]
            pending_turn = self._pending_turns.get(turn_id)
            if pending_turn is None:
                heapq.heappop(self._ready_heap)
                continue
            if pending_turn.version != version or pending_turn.status not in {
                "SETTLING",
                "ASSESSMENT_READY",
            }:
                heapq.heappop(self._ready_heap)
                continue
            return_value = eligible_at
            return return_value
        return None
