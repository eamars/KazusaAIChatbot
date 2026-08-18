"""Bounded parallel chain execution with global attempt caps and cancellation.

A wave runs every registered chain concurrently as an owned task; stages inside
one chain run serially in exact registry order. Each stage owner has a total
attempt limit tracked by one invocation-wide ledger, boundary-class failures are
terminal rejections with zero repair calls, structural exhaustion records the
contract-exhausted error code, and each stage is optional so independently valid
later stages still run after an earlier-stage failure. Cancelling owned tasks
materializes no partial effects: a cancelled or failed chain contributes nothing
to the result set while its completed attempt reservations remain diagnostic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, Sequence

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    BOUNDARY_REJECTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    TERMINAL_BOUNDARY_CLASSES,
    StageFailure,
    StageResult,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import ALL_CHAINS


class ExecutorContractError(RuntimeError):
    """Fail-closed executor topology or attempt-cap violation."""


@dataclass(frozen=True)
class StageAttemptOutcome:
    """Raw producer outcome for one bounded attempt before disposition.

    ``failure_class`` uses the closed contract failure vocabulary; it is None
    when the attempt was accepted, so no parallel failure vocabulary exists.
    ``detail`` carries the owner validator's exact structural violation or
    boundary context for non-accepted outcomes so a local repair request can
    restate the precise error without inventing a new failure class.
    """

    accepted: bool
    local_state: dict[str, object] | None
    semantic_summary: str | None
    failure_class: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ExecutorContext:
    """Deterministic context handed to one stage producer attempt."""

    chain_name: str
    stage_name: str
    attempt_number: int
    accepted_prefix: tuple[StageResult, ...]


StageProducer = Callable[[ExecutorContext], Awaitable[StageAttemptOutcome]]


@dataclass(frozen=True)
class ChainTaskSpec:
    """One registered chain to run in a wave with its ordered stage producers.

    ``stages`` must equal the exact registry stage order for ``chain_name``;
    model-created stages, reordered stages, and missing producer bindings are
    rejected before any task starts.
    """

    chain_name: str
    stages: tuple[str, ...]
    producers: Mapping[str, StageProducer]


@dataclass(frozen=True)
class ChainOutcome:
    """Materialized results for one cleanly completed chain."""

    chain_name: str
    results: tuple[StageResult, ...]


@dataclass(frozen=True)
class WaveResult:
    """Wave materialization separating clean outcomes from isolated failures.

    ``outcomes`` holds only chains whose every stage reached a recorded typed
    disposition; cancelled or failed chains appear in their own fields so no
    partial effect leaks into the result set.
    """

    outcomes: Mapping[str, ChainOutcome]
    cancelled_chains: tuple[str, ...]
    failed_chains: dict[str, str]


@dataclass
class AttemptLedger:
    """Global bounded attempt arithmetic for one cognition invocation.

    Reservations are per-owner and never exceed the configured total limit;
    exhaustion is a recorded typed outcome rather than an exception path.
    """

    limits: Mapping[str, int]
    _counts: dict[str, int] = field(default_factory=dict)

    def attempts_used(self, owner_name: str) -> int:
        """Return completed attempt reservations for one owner so far."""
        return self._counts.get(owner_name, 0)

    def can_reserve(self, owner_name: str) -> bool:
        """Check whether the next reservation stays inside the owner's cap.

        Raises:
            ExecutorContractError: Unknown owners or non-positive configured
                limits fail fast at configuration time.
        """
        if owner_name not in self.limits:
            raise ExecutorContractError(f"Unknown attempt-cap owner {owner_name!r}")
        limit = self.limits[owner_name]
        if limit <= 0:
            raise ExecutorContractError(f"Owner {owner_name!r} has a non-positive attempt limit")
        return self.attempts_used(owner_name) < limit

    def reserve(self, owner_name: str) -> int:
        """Reserve the next attempt number for one owner.

        Returns:
            The 1-based attempt number just reserved.

        Raises:
            ExecutorContractError: Reservations beyond the configured cap are a
                programming error and fail fast instead of silently overrunning
                the global arithmetic.
        """
        if not self.can_reserve(owner_name):
            raise ExecutorContractError(f"Owner {owner_name!r} attempt cap already exhausted")
        next_number = self.attempts_used(owner_name) + 1
        self._counts[owner_name] = next_number
        return next_number


def validate_chain_spec(spec: ChainTaskSpec) -> None:
    """Validate one chain task spec against the immutable registry.

    Args:
        spec: The chain, its stage sequence, and producer bindings to check.

    Raises:
        ExecutorContractError: Unknown chains, stage sequences that deviate from
            the exact registry order, or missing producer bindings fail fast
            before any model call starts; model-created stages are rejected at
            this boundary.
    """
    chain = next((candidate for candidate in ALL_CHAINS if candidate.name == spec.chain_name), None)
    if chain is None:
        raise ExecutorContractError(f"Unknown registered chain {spec.chain_name!r}")
    if spec.stages != chain.stages:
        raise ExecutorContractError(
            f"Chain {spec.chain_name!r} stages must match the exact registry order {chain.stages}"
        )
    for stage in chain.stages:
        if stage not in spec.producers:
            raise ExecutorContractError(f"Chain {spec.chain_name!r} has no producer bound for stage {stage!r}")


@dataclass
class WaveHandle:
    """Owned parallel-wave task set supporting bounded cancellation."""

    _tasks: dict[str, asyncio.Task] = field(default_factory=dict)

    def cancel(self) -> None:
        """Cancel every owned wave task.

        Cancelling stops in-flight stage attempts; no partial chain result is
        materialized by ``complete`` for a cancelled chain.
        """
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

    async def complete(self) -> WaveResult:
        """Materialize clean chain outcomes and isolate every failure route.

        Returns:
            A wave result whose ``outcomes`` contain only chains whose stages
            all reached recorded typed dispositions; cancelled chains list in
            ``cancelled_chains`` and producer-exception chains in
            ``failed_chains`` keyed by exception class name, so sibling chains
            keep their materialized results.
        """
        outcomes: dict[str, ChainOutcome] = {}
        cancelled_chains: list[str] = []
        failed_chains: dict[str, str] = {}

        for chain_name in sorted(self._tasks):
            task = self._tasks[chain_name]
            try:
                outcome = await task
            except asyncio.CancelledError:
                cancelled_chains.append(chain_name)
            except Exception as exc:  # noqa: BLE001 - producer isolation is the contract here
                failed_chains[chain_name] = type(exc).__name__
            else:
                outcomes[chain_name] = outcome

        return WaveResult(
            outcomes=outcomes,
            cancelled_chains=tuple(cancelled_chains),
            failed_chains=failed_chains,
        )


def start_wave(specs: Sequence[ChainTaskSpec], *, ledger: AttemptLedger) -> WaveHandle:
    """Start one parallel wave of registered chains as owned tasks.

    Args:
        specs: Chain task specs for the wave; every chain runs concurrently and
            stages inside each chain run serially in registry order.
        ledger: The invocation-wide attempt-cap arithmetic shared by all owners.

    Returns:
        The handle owning every started wave task.

    Raises:
        ExecutorContractError: Topology validation fails fast before any task
            starts, so an invalid spec never leaves a half-started wave behind.
    """
    for spec in specs:
        validate_chain_spec(spec)
    if len({spec.chain_name for spec in specs}) != len(specs):
        raise ExecutorContractError("One wave cannot run the same registered chain twice")

    handle = WaveHandle()
    for spec in specs:
        handle._tasks[spec.chain_name] = asyncio.create_task(_run_chain(spec, ledger))
    return handle


async def _run_chain(
    spec: ChainTaskSpec,
    ledger: AttemptLedger,
) -> ChainOutcome:
    """Run one chain's registered stages serially under the shared ledger.

    Each stage consumes bounded attempts until acceptance or a terminal boundary
    rejection; structural exhaustion records the contract-exhausted error code.
    Every stage is optional, so an earlier-stage failure does not remove later
    independently valid stages from the registry order.
    """
    accepted_prefix: list[StageResult] = []
    results: list[StageResult] = []

    for stage_name in spec.stages:
        producer = spec.producers[stage_name]
        result: StageResult | None = None

        while ledger.can_reserve(stage_name):
            attempt_number = ledger.reserve(stage_name)
            outcome = await producer(
                ExecutorContext(
                    chain_name=spec.chain_name,
                    stage_name=stage_name,
                    attempt_number=attempt_number,
                    accepted_prefix=tuple(accepted_prefix),
                )
            )

            if outcome.accepted:
                result = StageResult(
                    chain_name=spec.chain_name,
                    stage_name=stage_name,
                    accepted=True,
                    local_state=outcome.local_state,
                    semantic_summary=outcome.semantic_summary,
                )
                break

            if outcome.failure_class in TERMINAL_BOUNDARY_CLASSES:
                result = _boundary_rejection_result(
                    spec.chain_name, stage_name, outcome.failure_class
                )
                break

        if result is None:
            attempts_used = ledger.attempts_used(stage_name)
            result = StageResult(
                chain_name=spec.chain_name,
                stage_name=stage_name,
                accepted=False,
                local_state=None,
                semantic_summary=None,
                failure=StageFailure(
                    chain_name=spec.chain_name,
                    stage_name=stage_name,
                    failure_class=EXHAUSTION_FAILURE_CLASS,
                    error_code=APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
                    repair_attempted=attempts_used >= 2,
                ),
            )

        if result.accepted:
            accepted_prefix.append(result)
        results.append(result)

    return ChainOutcome(chain_name=spec.chain_name, results=tuple(results))


def _boundary_rejection_result(
    chain_name: str,
    stage_name: str,
    boundary_class: str,
) -> StageResult:
    """Record a terminal boundary rejection with zero repair calls.

    Args:
        chain_name: The registered chain owning the rejected stage.
        stage_name: The stage whose candidate failed a boundary-class check.
        boundary_class: The exact producer-reported terminal boundary class;
            the record keeps that closed-set value rather than inventing a new
            vocabulary.

    Returns:
        A non-accepted stage result carrying the exact boundary-rejected error
        code and ``repair_attempted=False``; the rejection is terminal for this
        stage while independently valid later stages keep running.
    """
    return StageResult(
        chain_name=chain_name,
        stage_name=stage_name,
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=StageFailure(
            chain_name=chain_name,
            stage_name=stage_name,
            failure_class=boundary_class,
            error_code=BOUNDARY_REJECTED_ERROR_CODE,
            repair_attempted=False,
        ),
    )
