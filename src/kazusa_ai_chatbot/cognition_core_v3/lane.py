"""Process-local FIFO coordination for Cognition V3 model lanes."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    current_v2_attempt_ledger,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot.llm_interface.detection import normalize_base_url

SidecarStreamKind = Literal[
    "l1",
    "json_repair",
    "action_authorization",
    "resolver_authorization",
]
LaneIdentity = tuple[int, str, str]
AttemptIdentity = tuple[str, int, int, str, str, int, int, int]

_AUTHORIZATION_ATTEMPT_CAP = 3
_PRIMARY_STREAM_KIND = "primary"
_ATTEMPT_COORDINATE_FIELDS = (
    "cognition_invocation_id",
    "graph_attempt",
    "branch_id",
    "producing_stage",
    "local_attempt",
    "cumulative_producer_attempt",
    "configured_limit",
)
_ATTEMPT_COORDINATE_FIELD_SET = frozenset(_ATTEMPT_COORDINATE_FIELDS)
_SIDECAR_STREAM_KINDS: frozenset[str] = frozenset(
    {
        "l1",
        "json_repair",
        "action_authorization",
        "resolver_authorization",
    }
)


class LaneContractError(RuntimeError):
    """A lane claim violates identity, ownership, or deadline rules."""


class LaneDeadlineError(TimeoutError):
    """A queued lane claim cannot start before its owning turn deadline."""


class SidecarAdmissionError(RuntimeError):
    """A logical sidecar producer exceeds its invocation-local allowance."""


@dataclass(frozen=True)
class LaneClaim:
    """Diagnostics captured when one FIFO ticket becomes the lane owner."""

    identity: LaneIdentity
    ticket: int
    stream_kind: str
    queue_wait_ms: int
    in_flight_at_start: int


def _consume_cleanup_task_result(task: asyncio.Task[None]) -> None:
    """Observe a detached lane cleanup task without changing cancellation."""

    if not task.cancelled():
        task.exception()


async def _observe_l1_task(
    task: asyncio.Task[object],
) -> tuple[bool, bool]:
    """Drain one L1 task and classify child cancellation versus failure."""

    try:
        await task
    except asyncio.CancelledError:
        return True, False
    except Exception:  # noqa: BLE001 - advisory task boundary
        return False, True
    return False, False


class _FifoLane:
    """One fair non-reentrant async lane shared by one resident model."""

    def __init__(self, identity: LaneIdentity) -> None:
        """Initialize an idle lane for one normalized invoker/route identity."""

        self.identity = identity
        self._condition = asyncio.Condition()
        self._waiters: deque[tuple[int, asyncio.Task[object]]] = deque()
        self._next_ticket = 1
        self._owner: asyncio.Task[object] | None = None
        self._in_flight = 0
        self._maximum_in_flight = 0

    async def _release_owner(self, task: asyncio.Task[object]) -> None:
        """Release one owner in a task that is independent of its caller."""

        async with self._condition:
            if self._owner is task:
                self._owner = None
                self._in_flight = 0
                self._condition.notify_all()

    @property
    def in_flight(self) -> int:
        """Return the current request count for diagnostics."""

        return self._in_flight

    @property
    def maximum_in_flight(self) -> int:
        """Return the greatest observed request count for this coordinator."""

        return self._maximum_in_flight

    @asynccontextmanager
    async def claim(
        self,
        *,
        stream_kind: str,
        deadline_monotonic: float | None,
    ) -> AsyncIterator[LaneClaim]:
        """Claim the lane in FIFO order and release it under every exit path.

        Args:
            stream_kind: Stable logical stream label recorded with the claim.
            deadline_monotonic: Optional absolute monotonic deadline checked
                before admission and while the ticket waits.

        Yields:
            The ticket and queue diagnostics for the admitted owner.

        Raises:
            LaneContractError: The current task already owns or awaits the lane.
            LaneDeadlineError: The ticket cannot start before its deadline.
        """

        task = asyncio.current_task()
        if task is None:
            raise LaneContractError("Lane claims require an active asyncio task")
        started_monotonic = time.monotonic()

        async with self._condition:
            if self._owner is task or any(
                waiting_task is task for _, waiting_task in self._waiters
            ):
                raise LaneContractError("A task cannot reacquire a lane it owns")
            if (
                deadline_monotonic is not None
                and started_monotonic >= deadline_monotonic
            ):
                raise LaneDeadlineError("Lane deadline expired before ticket admission")

            ticket = self._next_ticket
            self._next_ticket += 1
            waiter = (ticket, task)
            self._waiters.append(waiter)
            try:
                await self._wait_until_owner(
                    waiter,
                    deadline_monotonic=deadline_monotonic,
                )
                if (
                    deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic
                ):
                    raise LaneDeadlineError(
                        "Lane deadline expired while queued"
                    )
            except (asyncio.CancelledError, LaneDeadlineError):
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                raise

            self._waiters.popleft()
            self._owner = task
            self._in_flight = 1
            self._maximum_in_flight = max(
                self._maximum_in_flight,
                self._in_flight,
            )

        queue_wait_ms = max(
            0,
            round((time.monotonic() - started_monotonic) * 1000),
        )
        admitted_claim = LaneClaim(
            identity=self.identity,
            ticket=ticket,
            stream_kind=stream_kind,
            queue_wait_ms=queue_wait_ms,
            in_flight_at_start=self._in_flight,
        )
        try:
            yield admitted_claim
        finally:
            cleanup_task = asyncio.create_task(self._release_owner(task))
            cleanup_task.add_done_callback(_consume_cleanup_task_result)
            await asyncio.shield(cleanup_task)

    async def _wait_until_owner(
        self,
        waiter: tuple[int, asyncio.Task[object]],
        *,
        deadline_monotonic: float | None,
    ) -> None:
        """Wait until one ticket is first and the resident lane is idle."""

        while self._owner is not None or self._waiters[0] != waiter:
            if deadline_monotonic is None:
                await self._condition.wait()
                continue

            remaining_seconds = deadline_monotonic - time.monotonic()
            if remaining_seconds <= 0:
                raise LaneDeadlineError("Lane deadline expired while queued")
            try:
                await asyncio.wait_for(
                    self._condition.wait(),
                    timeout=remaining_seconds,
                )
            except TimeoutError as exc:
                raise LaneDeadlineError(
                    "Lane deadline expired while queued"
                ) from exc


class PrimaryLaneCoordinator:
    """FIFO owner for complete primary chains and recurrence tails."""

    def __init__(self, identity: LaneIdentity, llm: LLMInvoker) -> None:
        """Bind one primary coordinator to its exact resident model key."""

        self._llm = llm
        self._lane = _FifoLane(identity)

    @property
    def identity(self) -> LaneIdentity:
        """Return the exact invoker, endpoint, and model registry key."""

        return self._lane.identity

    @property
    def in_flight(self) -> int:
        """Return whether a primary chain currently owns this lane."""

        return self._lane.in_flight

    @asynccontextmanager
    async def claim(
        self,
        *,
        deadline_monotonic: float | None = None,
    ) -> AsyncIterator[LaneClaim]:
        """Hold the primary lane until the complete owned sequence exits."""

        async with self._lane.claim(
            stream_kind=_PRIMARY_STREAM_KIND,
            deadline_monotonic=deadline_monotonic,
        ) as admitted_claim:
            yield admitted_claim


class SidecarCoordinator:
    """One serialized stream for L1, repair, and authorization requests."""

    def __init__(self, identity: LaneIdentity, llm: LLMInvoker) -> None:
        """Bind one sidecar coordinator to its exact resident model key."""

        self._llm = llm
        self._lane = _FifoLane(identity)

    @property
    def identity(self) -> LaneIdentity:
        """Return the exact invoker, endpoint, and model registry key."""

        return self._lane.identity

    @property
    def in_flight(self) -> int:
        """Return the current number of sidecar requests."""

        return self._lane.in_flight

    @property
    def maximum_in_flight(self) -> int:
        """Return the greatest observed sidecar concurrency."""

        return self._lane.maximum_in_flight

    @asynccontextmanager
    async def claim(
        self,
        *,
        stream_kind: SidecarStreamKind,
        invocation_state: SidecarInvocationState,
        deadline_monotonic: float | None = None,
    ) -> AsyncIterator[LaneClaim]:
        """Serialize one known sidecar stream under the shared FIFO."""

        if stream_kind not in _SIDECAR_STREAM_KINDS:
            raise LaneContractError(
                f"Unknown sidecar stream kind {stream_kind!r}"
            )
        try:
            async with self._lane.claim(
                stream_kind=stream_kind,
                deadline_monotonic=deadline_monotonic,
            ) as admitted_claim:
                invocation_state.record_sidecar_claim(admitted_claim)
                yield admitted_claim
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                invocation_state.record_cancellation(task)
            raise


def _canonical_attempt_identity_and_disposition(
    attempt_coordinates: Mapping[str, object],
    *,
    expected_stage: str | None = None,
) -> tuple[AttemptIdentity, str]:
    """Resolve exact coordinates against the bound canonical attempt ledger.

    Args:
        attempt_coordinates: Coordinates returned by the shared V2 producer
            attempt authority for the model call that emitted a candidate.
        expected_stage: Optional exact producer owner required by a sidecar
            authorization stream.

    Returns:
        The invocation/epoch-qualified attempt identity and current producer
        disposition.

    Raises:
        SidecarAdmissionError: The coordinates are malformed, foreign, stale,
            or owned by a different producer stage.
    """

    if frozenset(attempt_coordinates) != _ATTEMPT_COORDINATE_FIELD_SET:
        raise SidecarAdmissionError(
            "Sidecar admission requires exact producer attempt coordinates"
        )

    canonical_ledger = current_v2_attempt_ledger()
    if canonical_ledger is None:
        raise SidecarAdmissionError(
            "Sidecar admission requires a bound producer attempt ledger"
        )

    matching_record = None
    for record_index in range(len(canonical_ledger.attempts) - 1, -1, -1):
        if (
            canonical_ledger.attempt_epochs[record_index]
            != canonical_ledger.epoch
        ):
            continue
        record = canonical_ledger.attempts[record_index]
        if all(
            record[field_name] == attempt_coordinates[field_name]
            for field_name in _ATTEMPT_COORDINATE_FIELDS
        ):
            matching_record = record
            break

    if matching_record is None:
        raise SidecarAdmissionError(
            "Sidecar attempt coordinates are not reserved in this invocation"
        )
    if (
        expected_stage is not None
        and matching_record["producing_stage"] != expected_stage
    ):
        raise SidecarAdmissionError(
            f"Sidecar stream requires a live {expected_stage} attempt"
        )

    attempt_identity: AttemptIdentity = (
        matching_record["cognition_invocation_id"],
        canonical_ledger.epoch,
        matching_record["graph_attempt"],
        matching_record["branch_id"],
        matching_record["producing_stage"],
        matching_record["local_attempt"],
        matching_record["cumulative_producer_attempt"],
        matching_record["configured_limit"],
    )
    disposition = matching_record["attempt_disposition"]
    return attempt_identity, disposition


@dataclass
class SidecarAdmissionLedger:
    """Invocation-local logical producer and attempt admission authority."""

    _l1_reserved: bool = field(default=False, init=False)
    _repair_candidates: set[str] = field(default_factory=set, init=False)
    _repair_attempts: set[AttemptIdentity] = field(
        default_factory=set,
        init=False,
    )
    _action_attempts: dict[int, int] = field(default_factory=dict, init=False)
    _resolver_attempts: dict[int, int] = field(default_factory=dict, init=False)
    _action_attempt_identities: dict[int, set[AttemptIdentity]] = field(
        default_factory=dict,
        init=False,
    )
    _resolver_attempt_identities: dict[int, set[AttemptIdentity]] = field(
        default_factory=dict,
        init=False,
    )
    _action_branch_ids: dict[int, str] = field(default_factory=dict, init=False)
    _resolver_branch_ids: dict[int, str] = field(
        default_factory=dict,
        init=False,
    )
    _action_last_coordinates: dict[int, Mapping[str, object]] = field(
        default_factory=dict,
        init=False,
    )
    _resolver_last_coordinates: dict[int, Mapping[str, object]] = field(
        default_factory=dict,
        init=False,
    )
    _action_finished: set[int] = field(default_factory=set, init=False)
    _resolver_finished: set[int] = field(default_factory=set, init=False)

    def reserve_l1(self) -> None:
        """Reserve the sole L1 producer for one cold invocation."""

        if self._l1_reserved:
            raise SidecarAdmissionError(
                "A cold invocation admits exactly one L1 producer"
            )
        self._l1_reserved = True

    def reserve_json_repair(
        self,
        *,
        candidate_id: str,
        attempt_coordinates: Mapping[str, object],
    ) -> None:
        """Reserve one repair for a candidate from one live producer attempt."""

        clean_candidate_id = candidate_id.strip()
        if not clean_candidate_id:
            raise SidecarAdmissionError(
                "JSON repair admission requires a raw candidate identity"
            )
        if clean_candidate_id in self._repair_candidates:
            raise SidecarAdmissionError(
                "Each raw candidate admits at most one JSON repair call"
            )

        attempt_identity, disposition = (
            _canonical_attempt_identity_and_disposition(attempt_coordinates)
        )
        if disposition != "started":
            raise SidecarAdmissionError(
                "JSON repair requires a live producer attempt"
            )
        if attempt_identity in self._repair_attempts:
            raise SidecarAdmissionError(
                "One producer attempt admits at most one JSON repair call"
            )

        self._repair_candidates.add(clean_candidate_id)
        self._repair_attempts.add(attempt_identity)

    def reserve_action_authorization(
        self,
        *,
        cycle_index: int,
        attempt_coordinates: Mapping[str, object],
    ) -> int:
        """Reserve the next X1 attempt for one cognition cycle."""

        self._validate_cycle_index(cycle_index)
        if cycle_index in self._action_finished:
            raise SidecarAdmissionError(
                "The X1 authorization producer already finished"
            )
        attempt_number = self._reserve_authorization_attempt(
            self._action_attempts,
            self._action_attempt_identities,
            self._action_branch_ids,
            self._action_last_coordinates,
            cycle_index=cycle_index,
            label="X1",
            expected_stage="action_authorization",
            attempt_coordinates=attempt_coordinates,
        )
        return attempt_number

    def finish_action_authorization(self, *, cycle_index: int) -> None:
        """Close X1, including a cycle where deterministic checks skipped it."""

        self._validate_cycle_index(cycle_index)
        if cycle_index in self._action_finished:
            raise SidecarAdmissionError(
                "The X1 authorization producer already finished"
            )
        self._require_last_attempt_finished(
            self._action_last_coordinates,
            cycle_index=cycle_index,
            label="X1",
            expected_stage="action_authorization",
        )
        self._action_finished.add(cycle_index)

    def reserve_resolver_authorization(
        self,
        *,
        cycle_index: int,
        attempt_coordinates: Mapping[str, object],
    ) -> int:
        """Reserve the next X2 attempt after X1 has reached disposition."""

        self._validate_cycle_index(cycle_index)
        if cycle_index not in self._action_finished:
            raise SidecarAdmissionError(
                "X1 must finish before the X2 authorization producer starts"
            )
        if cycle_index in self._resolver_finished:
            raise SidecarAdmissionError(
                "The X2 authorization producer already finished"
            )
        attempt_number = self._reserve_authorization_attempt(
            self._resolver_attempts,
            self._resolver_attempt_identities,
            self._resolver_branch_ids,
            self._resolver_last_coordinates,
            cycle_index=cycle_index,
            label="X2",
            expected_stage="resolver_authorization",
            attempt_coordinates=attempt_coordinates,
        )
        return attempt_number

    def finish_resolver_authorization(self, *, cycle_index: int) -> None:
        """Close X2 after its accepted or deny-all disposition."""

        self._validate_cycle_index(cycle_index)
        if cycle_index not in self._action_finished:
            raise SidecarAdmissionError(
                "X1 must finish before the X2 authorization producer finishes"
            )
        if cycle_index in self._resolver_finished:
            raise SidecarAdmissionError(
                "The X2 authorization producer already finished"
            )
        self._require_last_attempt_finished(
            self._resolver_last_coordinates,
            cycle_index=cycle_index,
            label="X2",
            expected_stage="resolver_authorization",
        )
        self._resolver_finished.add(cycle_index)

    @staticmethod
    def _validate_cycle_index(cycle_index: int) -> None:
        """Require a non-negative integer cognition cycle index."""

        if (
            not isinstance(cycle_index, int)
            or isinstance(cycle_index, bool)
            or cycle_index < 0
        ):
            raise SidecarAdmissionError(
                "Authorization cycle indexes must be non-negative integers"
            )

    @staticmethod
    def _reserve_authorization_attempt(
        attempts: dict[int, int],
        attempt_identities: dict[int, set[AttemptIdentity]],
        branch_ids: dict[int, str],
        last_coordinates: dict[int, Mapping[str, object]],
        *,
        cycle_index: int,
        label: str,
        expected_stage: str,
        attempt_coordinates: Mapping[str, object],
    ) -> int:
        """Bind one X1 or X2 call to a live canonical attempt reservation."""

        attempts_used = attempts.get(cycle_index, 0)
        if attempts_used >= _AUTHORIZATION_ATTEMPT_CAP:
            raise SidecarAdmissionError(
                f"{label} admits at most three attempts per cognition cycle"
            )

        SidecarAdmissionLedger._require_last_attempt_finished(
            last_coordinates,
            cycle_index=cycle_index,
            label=label,
            expected_stage=expected_stage,
        )
        attempt_identity, disposition = (
            _canonical_attempt_identity_and_disposition(
                attempt_coordinates,
                expected_stage=expected_stage,
            )
        )
        if disposition != "started":
            raise SidecarAdmissionError(
                f"{label} requires a live producer attempt"
            )
        if (
            attempt_coordinates["configured_limit"]
            != _AUTHORIZATION_ATTEMPT_CAP
        ):
            raise SidecarAdmissionError(
                f"{label} must retain the canonical three-attempt limit"
            )

        cycle_attempt_identities = attempt_identities.setdefault(
            cycle_index,
            set(),
        )
        if attempt_identity in cycle_attempt_identities:
            raise SidecarAdmissionError(
                f"{label} cannot reuse a producer attempt reservation"
            )

        branch_id = str(attempt_coordinates["branch_id"])
        existing_branch_id = branch_ids.get(cycle_index)
        if existing_branch_id is not None and existing_branch_id != branch_id:
            raise SidecarAdmissionError(
                f"{label} admits one producer identity per cognition cycle"
            )

        attempt_number = attempts_used + 1
        attempts[cycle_index] = attempt_number
        cycle_attempt_identities.add(attempt_identity)
        branch_ids[cycle_index] = branch_id
        last_coordinates[cycle_index] = dict(attempt_coordinates)
        return attempt_number

    @staticmethod
    def _require_last_attempt_finished(
        last_coordinates: Mapping[int, Mapping[str, object]],
        *,
        cycle_index: int,
        label: str,
        expected_stage: str,
    ) -> None:
        """Reject a new or terminal producer transition while an attempt lives."""

        previous_coordinates = last_coordinates.get(cycle_index)
        if previous_coordinates is None:
            return
        _, disposition = _canonical_attempt_identity_and_disposition(
            previous_coordinates,
            expected_stage=expected_stage,
        )
        if disposition == "started":
            raise SidecarAdmissionError(
                f"{label} has a live attempt that must finish first"
            )


@dataclass
class SidecarInvocationState:
    """Own the optional L1 task and its preemption diagnostics."""

    l1_preempted_by_repair: bool = field(default=False, init=False)
    _l1_task: asyncio.Task[object] | None = field(default=None, init=False)
    _stream_counts: dict[str, int] = field(
        default_factory=lambda: {
            "l1": 0,
            "json_repair": 0,
            "action_authorization": 0,
            "resolver_authorization": 0,
        },
        init=False,
    )
    _queue_wait_ms_total: int = field(default=0, init=False)
    _maximum_in_flight: int = field(default=0, init=False)
    _cancelled_tasks: set[asyncio.Task[object]] = field(
        default_factory=set,
        init=False,
    )
    _l1_warning: str | None = field(default=None, init=False)

    @property
    def cancellation_count(self) -> int:
        """Return the number of distinct sidecar tasks cancellation targeted."""

        cancellation_count = len(self._cancelled_tasks)
        return cancellation_count

    def record_sidecar_claim(self, claim: LaneClaim) -> None:
        """Record one admitted sidecar call in invocation-local diagnostics."""

        if claim.stream_kind not in _SIDECAR_STREAM_KINDS:
            raise LaneContractError(
                f"Unknown admitted sidecar stream {claim.stream_kind!r}"
            )
        if claim.in_flight_at_start != 1:
            raise LaneContractError(
                "A sidecar claim must start with exactly one request in flight"
            )
        self._stream_counts[claim.stream_kind] += 1
        self._queue_wait_ms_total += claim.queue_wait_ms
        self._maximum_in_flight = max(
            self._maximum_in_flight,
            claim.in_flight_at_start,
        )

    def record_cancellation(self, task: asyncio.Task[object]) -> None:
        """Record one task cancellation exactly once for this invocation."""

        self._cancelled_tasks.add(task)

    def diagnostics(self) -> dict[str, int | bool]:
        """Project the exact sidecar counters consumed by V3 diagnostics."""

        diagnostics: dict[str, int | bool] = {
            "l1_stream_count": self._stream_counts["l1"],
            "json_repair_call_count": self._stream_counts["json_repair"],
            "action_auth_attempt_count": self._stream_counts[
                "action_authorization"
            ],
            "resolver_auth_attempt_count": self._stream_counts[
                "resolver_authorization"
            ],
            "sidecar_queue_wait_ms_total": self._queue_wait_ms_total,
            "sidecar_max_in_flight": self._maximum_in_flight,
            "l1_preempted_by_repair": self.l1_preempted_by_repair,
            "sidecar_cancellation_count": self.cancellation_count,
        }
        return diagnostics

    def register_l1_task(self, task: asyncio.Task[object]) -> None:
        """Register the one L1 task started for this cold invocation."""

        if self._l1_task is not None:
            raise SidecarAdmissionError(
                "A cold invocation admits exactly one registered L1 task"
            )
        if task.done():
            raise SidecarAdmissionError(
                "The registered L1 task must still be active"
            )
        self._l1_task = task

    def consume_l1_warning(self) -> str | None:
        """Consume the bounded warning recorded for a failed L1 task."""

        warning = self._l1_warning
        self._l1_warning = None
        return warning

    async def preempt_l1_for_repair(self) -> bool:
        """Cancel and drain unfinished L1 work before repair claims the lane.

        Returns:
            True when an active L1 task was cancelled and drained, or False
            when no unfinished L1 task owns or awaits the sidecar.
        """

        task = self._l1_task
        if task is None:
            return False
        current_task = asyncio.current_task()
        if task is current_task:
            raise LaneContractError(
                "An L1 task cannot preempt itself for JSON repair"
            )
        initial_cancelling = (
            current_task.cancelling()
            if current_task is not None
            else 0
        )

        if task.done():
            try:
                task.result()
            except asyncio.CancelledError:
                return False
            except Exception:  # noqa: BLE001 - advisory task boundary
                self._l1_warning = "sidecar_l1_unavailable"
            if (
                current_task is not None
                and current_task.cancelling() > initial_cancelling
            ):
                raise asyncio.CancelledError
            return False

        self.record_cancellation(task)
        task.cancel()
        drain_task = asyncio.create_task(_observe_l1_task(task))
        outer_cancelled = False
        while True:
            try:
                _, child_failed = await asyncio.shield(
                    drain_task
                )
                break
            except asyncio.CancelledError:
                outer_cancelled = True
                continue
        if child_failed:
            self._l1_warning = "sidecar_l1_unavailable"
        self.l1_preempted_by_repair = True
        if (
            outer_cancelled
            or (
                current_task is not None
                and current_task.cancelling() > initial_cancelling
            )
        ):
            raise asyncio.CancelledError
        return True


_PRIMARY_LANE_REGISTRY: dict[LaneIdentity, PrimaryLaneCoordinator] = {}
_SIDECAR_LANE_REGISTRY: dict[LaneIdentity, SidecarCoordinator] = {}


def _lane_identity(llm: LLMInvoker, config: LLMCallConfig) -> LaneIdentity:
    """Build the exact process-local registry key for one resident lane."""

    if llm is None:
        raise LaneContractError("Lane coordination requires an LLM invoker")
    normalized_url = normalize_base_url(config.base_url)
    model = config.model.strip()
    if not normalized_url or not model:
        raise LaneContractError(
            "Lane coordination requires a non-empty endpoint and model"
        )
    identity = (id(llm), normalized_url, model)
    return identity


def primary_lane_coordinator(
    llm: LLMInvoker,
    config: LLMCallConfig,
) -> PrimaryLaneCoordinator:
    """Return the process-local primary coordinator for one exact lane key."""

    identity = _lane_identity(llm, config)
    coordinator = _PRIMARY_LANE_REGISTRY.get(identity)
    if coordinator is None:
        coordinator = PrimaryLaneCoordinator(identity, llm)
        _PRIMARY_LANE_REGISTRY[identity] = coordinator
    return coordinator


def sidecar_lane_coordinator(
    llm: LLMInvoker,
    config: LLMCallConfig,
) -> SidecarCoordinator:
    """Return the process-local sidecar coordinator for one exact lane key."""

    identity = _lane_identity(llm, config)
    coordinator = _SIDECAR_LANE_REGISTRY.get(identity)
    if coordinator is None:
        coordinator = SidecarCoordinator(identity, llm)
        _SIDECAR_LANE_REGISTRY[identity] = coordinator
    return coordinator


__all__ = [
    "LaneClaim",
    "LaneContractError",
    "LaneDeadlineError",
    "LaneIdentity",
    "PrimaryLaneCoordinator",
    "SidecarAdmissionError",
    "SidecarAdmissionLedger",
    "SidecarCoordinator",
    "SidecarInvocationState",
    "SidecarStreamKind",
    "primary_lane_coordinator",
    "sidecar_lane_coordinator",
]
