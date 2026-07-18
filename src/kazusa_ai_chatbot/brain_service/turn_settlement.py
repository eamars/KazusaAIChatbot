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
_MAX_PRELUDE_SCOPES = 64
_MAX_CONTINUITY_SCOPES = 512
_BOT_CONTINUITY_ACTIVE_SECONDS = 180.0
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
    media_description_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict,
    )
    media_description_attempted_keys: set[str] = field(default_factory=set)
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
    accepts_fragments: bool = True


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
    response_owner_sequence: int
    fragments: tuple[PersistedChatFragment, ...] = ()
    latest_bot_continuity: str = ""


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
        self._silent_preludes: dict[
            tuple[Scope, str],
            list[PersistedChatFragment],
        ] = {}
        self._latest_bot_continuity: dict[
            tuple[Scope, str],
            tuple[str, float],
        ] = {}
        self._pending_ingress: dict[int, tuple[Scope, str, float]] = {}
        self._ingress_blocked_turns: set[str] = set()
        self._ready_heap: list[tuple[float, int, str, int]] = []
        self._turn_counter = itertools.count(1)
        self._state_condition = asyncio.Condition()
        self._work_lock = asyncio.Lock()
        self._work_queue: list[_RelevanceWork] = []
        self._work_task: asyncio.Task[None] | None = None

    def register_ingress(
        self,
        *,
        sequence: int,
        scope: Scope,
        author_platform_user_id: str,
        enqueue_monotonic: float,
    ) -> None:
        """Register one queued input before settlement can cross its boundary."""

        self._pending_ingress[sequence] = (
            scope,
            author_platform_user_id,
            enqueue_monotonic,
        )

    async def complete_ingress(self, sequence: int) -> None:
        """Release one registered input after its frontline action is applied."""

        async with self._state_condition:
            self._complete_ingress_locked(sequence)
            self._state_condition.notify_all()

    async def build_frontline_state(
        self,
        fragment: PersistedChatFragment,
    ) -> dict[str, Any]:
        """Build candidates from the same state used for slot application."""

        async with self._state_condition:
            open_turns = self._open_turns_locked(fragment)
            preludes = self._recent_preludes_locked(fragment)
            return_value = {
                "conversation_scope": fragment.scope[2],
                "current_message": {
                    "body_text": fragment.body_text,
                    "semantic_target_labels": list(
                        fragment.semantic_target_labels
                    ),
                    "reply_target_label": fragment.reply_target_label,
                    "media_labels": list(fragment.media_labels),
                },
                "open_turns": [
                    self._open_turn_descriptor(turn)
                    for turn in open_turns[:3]
                ],
                "recent_preludes": [
                    self._prelude_descriptor(prelude)
                    for prelude in preludes
                ],
                "latest_bot_continuity": (
                    self._recent_bot_continuity_locked(
                        self._prelude_key(fragment),
                    )
                ),
            }
        return return_value

    async def record_bot_continuity(
        self,
        *,
        scope: Scope,
        author_platform_user_id: str,
        dialog_text: str,
    ) -> None:
        """Retain the latest same-scope bot dialog for frontline continuity."""

        async with self._state_condition:
            key = (scope, author_platform_user_id)
            if dialog_text.strip():
                clean_text = dialog_text.strip()
                if len(clean_text) > 400:
                    clean_text = (
                        clean_text[:200]
                        + "..."
                        + clean_text[-197:]
                    )
                if key not in self._latest_bot_continuity and len(
                    self._latest_bot_continuity
                ) >= _MAX_CONTINUITY_SCOPES:
                    oldest_key = next(iter(self._latest_bot_continuity))
                    self._latest_bot_continuity.pop(oldest_key, None)
                self._latest_bot_continuity[key] = (
                    clean_text,
                    self._clock(),
                )
            else:
                self._latest_bot_continuity.pop(key, None)

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
                self._record_prelude_locked(fragment)
                self._complete_ingress_locked(fragment.arrival_sequence)
                return_value = FrontlineOutcome(action="discard")
                self._state_condition.notify_all()
                return return_value

            if action == "append":
                pending_turn = self._find_slot_turn_locked(
                    fragment,
                    validated["append_target"],
                )
                if pending_turn is None:
                    self._record_prelude_locked(fragment)
                    self._complete_ingress_locked(fragment.arrival_sequence)
                    return_value = FrontlineOutcome(action="discard")
                    self._state_condition.notify_all()
                    return return_value
                self._append_fragment_locked(pending_turn, fragment)
                self._complete_ingress_locked(fragment.arrival_sequence)
                return_value = FrontlineOutcome(
                    action="append",
                    turn_id=pending_turn.turn_id,
                    version=pending_turn.version,
                )
                self._state_condition.notify_all()
                return return_value

            promoted_preludes = self._select_preludes_locked(
                fragment,
                validated["prelude_targets"],
            )
            frozen_turn_id = self._freeze_oldest_open_turn_locked(fragment)
            pending_turn = self._create_turn_locked(
                fragment,
                promoted_preludes=promoted_preludes,
            )
            self._complete_ingress_locked(fragment.arrival_sequence)
            return_value = FrontlineOutcome(
                action="start",
                turn_id=pending_turn.turn_id,
                version=pending_turn.version,
                frozen_turn_id=frozen_turn_id,
            )
            self._state_condition.notify_all()
            return return_value

    async def append_collapsed_private_fragment(
        self,
        fragment: PersistedChatFragment,
        *,
        turn_id: str,
    ) -> FrontlineOutcome:
        """Attach a queue-coalesced private fragment to its exact survivor."""

        async with self._state_condition:
            pending_turn = self._pending_turns.get(turn_id)
            valid_target = (
                pending_turn is not None
                and fragment.scope[2] == "private"
                and pending_turn.scope == fragment.scope
                and pending_turn.author_platform_user_id
                == fragment.author_platform_user_id
                and pending_turn.accepts_fragments
                and pending_turn.status
                in {"SETTLING", "ASSESSING", "ASSESSMENT_READY"}
            )
            if not valid_target or pending_turn is None:
                self._complete_ingress_locked(fragment.arrival_sequence)
                self._state_condition.notify_all()
                return_value = FrontlineOutcome(action="discard")
                return return_value

            self._append_fragment_locked(pending_turn, fragment)
            self._complete_ingress_locked(fragment.arrival_sequence)
            self._state_condition.notify_all()
            return_value = FrontlineOutcome(
                action="append",
                turn_id=pending_turn.turn_id,
                version=pending_turn.version,
            )
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
                pending_turn.accepts_fragments = False
            return_value = SettlementOutcome(
                response_action=validated["response_action"],
                turn_id=pending_turn.turn_id,
                version=pending_turn.version,
                wait_used=pending_turn.wait_used,
                claimable=validated["response_action"] == "proceed",
            )
            if pending_turn.status == "DONE":
                self._pending_turns.pop(pending_turn.turn_id, None)
                self._ingress_blocked_turns.discard(pending_turn.turn_id)
            self._state_condition.notify_all()
            return return_value

    async def complete_failed_assessment(
        self,
        lease: AssessmentLease,
    ) -> bool:
        """Close a current failed assessment without inventing a semantic action.

        Args:
            lease: Versioned assessment lease whose operational failure has
                already been classified by the service boundary.

        Returns:
            True when this lease closed the current pending turn. False when
            the lease was stale or the turn had already completed.
        """

        async with self._state_condition:
            pending_turn = self._pending_turns.get(lease.turn_id)
            if pending_turn is None or pending_turn.status == "DONE":
                return_value = False
                return return_value

            if pending_turn.version != lease.version:
                pending_turn.status = "SETTLING"
                pending_turn.eligible_at_monotonic = (
                    pending_turn.hard_deadline_monotonic
                )
                self._schedule_ready_locked(pending_turn)
                self._state_condition.notify_all()
                return_value = False
                return return_value

            if pending_turn.status != "ASSESSING":
                raise ValueError(
                    "failed settled assessment must own an assessing turn"
                )

            pending_turn.status = "DONE"
            pending_turn.accepts_fragments = False
            self._pending_turns.pop(pending_turn.turn_id, None)
            self._ingress_blocked_turns.discard(pending_turn.turn_id)
            self._state_condition.notify_all()
            return_value = True
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
            if self._has_pending_ingress_locked(pending_turn):
                self._block_turn_for_ingress_locked(pending_turn)
                self._state_condition.notify_all()
                return_value = False
                return return_value
            pending_turn.status = "RUNNING"
            pending_turn.accepts_fragments = False
            return_value = True
            return return_value

    async def complete_cognition(self, turn_id: str, version: int) -> None:
        """Close and release one successfully claimed cognition turn."""

        async with self._state_condition:
            pending_turn = self._pending_turns.get(turn_id)
            if (
                pending_turn is not None
                and pending_turn.version == version
                and pending_turn.status == "RUNNING"
            ):
                pending_turn.status = "DONE"
                self._pending_turns.pop(turn_id, None)
                self._ingress_blocked_turns.discard(turn_id)
                self._state_condition.notify_all()

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
            except asyncio.CancelledError:
                if not work.future.done():
                    work.future.cancel()
                raise
            except Exception as exc:
                if not work.future.done():
                    work.future.set_exception(exc)
            else:
                if not work.future.done():
                    work.future.set_result(result)

    def _create_turn_locked(
        self,
        fragment: PersistedChatFragment,
        *,
        promoted_preludes: tuple[PersistedChatFragment, ...] = (),
    ) -> _PendingTurn:
        """Create one candidate turn and schedule its first eligibility."""

        fragments = sorted(
            [*promoted_preludes, fragment],
            key=lambda item: item.arrival_sequence,
        )
        created_at = fragments[0].enqueue_monotonic
        last_fragment_at = fragments[-1].enqueue_monotonic
        turn_id = f"turn-{next(self._turn_counter)}"
        is_private = fragment.scope[2] == "private"
        hard_deadline = created_at if is_private else created_at + 10.0
        eligible_at = (
            last_fragment_at
            if is_private
            else min(last_fragment_at + 6.0, hard_deadline)
        )
        pending_turn = _PendingTurn(
            turn_id=turn_id,
            scope=fragment.scope,
            author_platform_user_id=fragment.author_platform_user_id,
            author_global_user_id=fragment.author_global_user_id,
            leader_sequence=fragments[0].arrival_sequence,
            response_owner_sequence=fragment.arrival_sequence,
            fragments=fragments,
            status="SETTLING",
            created_at_monotonic=created_at,
            last_fragment_at_monotonic=last_fragment_at,
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

        pending_turn.fragments.append(fragment)
        pending_turn.last_fragment_at_monotonic = fragment.enqueue_monotonic
        pending_turn.version += 1
        pending_turn.status = "SETTLING"
        pending_turn.eligible_at_monotonic = min(
            fragment.enqueue_monotonic + 6.0,
            pending_turn.hard_deadline_monotonic,
        )
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
        oldest.accepts_fragments = False
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
            if turn.status in {"SETTLING", "ASSESSING", "ASSESSMENT_READY"}
            and turn.accepts_fragments
            and turn.scope == fragment.scope
            and turn.author_platform_user_id == fragment.author_platform_user_id
            and fragment.enqueue_monotonic <= turn.hard_deadline_monotonic
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
            if self._has_pending_ingress_locked(pending_turn):
                self._block_turn_for_ingress_locked(pending_turn)
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
                response_owner_sequence=pending_turn.response_owner_sequence,
                fragments=tuple(pending_turn.fragments),
                latest_bot_continuity=self._recent_bot_continuity_locked(
                    (
                        pending_turn.scope,
                        pending_turn.author_platform_user_id,
                    ),
                ),
            )
            return return_value
        return None

    def _prelude_key(
        self,
        fragment: PersistedChatFragment,
    ) -> tuple[Scope, str]:
        """Return the same-author and same-scope prelude key."""

        return_value = (fragment.scope, fragment.author_platform_user_id)
        return return_value

    def _recent_preludes_locked(
        self,
        fragment: PersistedChatFragment,
    ) -> list[PersistedChatFragment]:
        """Return at most two silent fragments from the prior ten seconds."""

        key = self._prelude_key(fragment)
        candidates = self._silent_preludes.get(key, [])
        recent = [
            candidate
            for candidate in candidates
            if 0.0
            <= fragment.enqueue_monotonic - candidate.enqueue_monotonic
            <= 10.0
            and candidate.arrival_sequence != fragment.arrival_sequence
        ]
        recent = recent[-2:]
        if recent:
            self._silent_preludes[key] = recent
        else:
            self._silent_preludes.pop(key, None)
        return_value = recent
        return return_value

    def _record_prelude_locked(self, fragment: PersistedChatFragment) -> None:
        """Retain one bounded group-silent fragment for possible promotion."""

        if fragment.scope[2] != "group":
            return
        target_labels = set(fragment.semantic_target_labels)
        if "character" not in target_labels and (
            "other_participant" in target_labels
            or fragment.reply_target_label in {
                "other_participant",
                "unknown_participant",
            }
        ):
            return
        if not fragment.body_text.strip() and not fragment.media_labels:
            return
        key = self._prelude_key(fragment)
        if key not in self._silent_preludes and len(
            self._silent_preludes
        ) >= _MAX_PRELUDE_SCOPES:
            oldest_key = next(iter(self._silent_preludes))
            self._silent_preludes.pop(oldest_key, None)
        candidates = self._silent_preludes.setdefault(key, [])
        candidates.append(fragment)
        self._silent_preludes[key] = candidates[-2:]

    def _select_preludes_locked(
        self,
        fragment: PersistedChatFragment,
        targets: list[str],
    ) -> tuple[PersistedChatFragment, ...]:
        """Map supplied prelude slots to retained fragments and consume them."""

        candidates = self._recent_preludes_locked(fragment)
        selected: list[PersistedChatFragment] = []
        for target in targets:
            slot_index = int(target.rsplit("_", maxsplit=1)[1]) - 1
            if slot_index >= len(candidates):
                continue
            candidate = candidates[slot_index]
            if candidate not in selected:
                selected.append(candidate)
        if selected:
            key = self._prelude_key(fragment)
            self._silent_preludes[key] = [
                candidate
                for candidate in candidates
                if candidate not in selected
            ]
            if not self._silent_preludes[key]:
                self._silent_preludes.pop(key, None)
        return_value = tuple(selected)
        return return_value

    @staticmethod
    def _prelude_descriptor(
        fragment: PersistedChatFragment,
    ) -> dict[str, str]:
        """Project one retained prelude without internal identity."""

        summary = fragment.body_text
        if not summary and fragment.media_labels:
            summary = ", ".join(fragment.media_labels)
        return_value = {
            "summary": summary,
            "target_summary": ", ".join(
                fragment.semantic_target_labels
            ),
            "reply_summary": fragment.reply_target_label,
        }
        return return_value

    def _recent_bot_continuity_locked(
        self,
        key: tuple[Scope, str],
    ) -> str:
        """Return active same-scene bot dialog and expire stale evidence."""

        continuity = self._latest_bot_continuity.get(key)
        if continuity is None:
            return_value = ""
            return return_value

        dialog_text, recorded_at = continuity
        age_seconds = self._clock() - recorded_at
        if 0.0 <= age_seconds <= _BOT_CONTINUITY_ACTIVE_SECONDS:
            return_value = dialog_text
            return return_value

        self._latest_bot_continuity.pop(key, None)
        return_value = ""
        return return_value

    @staticmethod
    def _open_turn_descriptor(turn: _PendingTurn) -> dict[str, str]:
        """Project one open turn from its chronological fragments."""

        opening = turn.fragments[0]
        latest = turn.fragments[-1]
        return_value = {
            "author_relation": "same_author",
            "latest_intent": latest.body_text,
            "opening_excerpt": opening.body_text,
            "target_summary": ", ".join(latest.semantic_target_labels),
            "reply_summary": latest.reply_target_label,
            "media_summary": ", ".join(latest.media_labels),
        }
        return return_value

    def _has_pending_ingress_locked(self, turn: _PendingTurn) -> bool:
        """Return whether pre-deadline same-author input awaits frontline."""

        return_value = any(
            scope == turn.scope
            and author == turn.author_platform_user_id
            and enqueue_monotonic <= turn.hard_deadline_monotonic
            for scope, author, enqueue_monotonic in self._pending_ingress.values()
        )
        return return_value

    def _block_turn_for_ingress_locked(self, turn: _PendingTurn) -> None:
        """Remove a due turn from assessment until registered input settles."""

        turn.status = "SETTLING"
        self._ingress_blocked_turns.add(turn.turn_id)

    def _complete_ingress_locked(self, sequence: int) -> None:
        """Release ingress and requeue turns whose watermark is now clear."""

        self._pending_ingress.pop(sequence, None)
        for turn_id in list(self._ingress_blocked_turns):
            turn = self._pending_turns.get(turn_id)
            if turn is None:
                self._ingress_blocked_turns.discard(turn_id)
                continue
            if self._has_pending_ingress_locked(turn):
                continue
            turn.eligible_at_monotonic = max(
                turn.eligible_at_monotonic,
                self._clock(),
            )
            self._schedule_ready_locked(turn)
            self._ingress_blocked_turns.discard(turn_id)

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
