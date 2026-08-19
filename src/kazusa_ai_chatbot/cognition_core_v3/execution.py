"""Bounded parallel chain execution with global attempt caps and cancellation.

A wave runs every registered chain concurrently as an owned task; stages inside
one chain run serially in exact registry order. Each stage owner has a total
attempt limit tracked by one invocation-wide ledger, boundary-class failures are
terminal rejections with zero repair calls, exhaustion records the
owner-specific error code (goal owners split provider versus structure on the
final attempt class while appraisal and terminal owners keep the contract-
exhausted code), each stage mirrors its attempts into the shared V2 invocation
ledger when one is active so producer budgets stay 1:1 with V3's local caps,
and each stage is optional so independently valid later stages still run after
an earlier-stage failure. Cancelling owned tasks materializes no partial
effects: a cancelled or failed chain contributes nothing to the result set while
its completed attempt reservations remain diagnostic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping, Sequence

from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2AttemptBudgetExhausted,
    current_v2_attempt_ledger,
    record_v2_attempt_disposition,
    record_v2_branch_disposition,
    reserve_v2_model_attempt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import GOAL_KINDS
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    BOUNDARY_REJECTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
    GOAL_BID_PROVIDER_EXHAUSTED_ERROR_CODE,
    GOAL_BID_STRUCTURE_EXHAUSTED_ERROR_CODE,
    PROVIDER_FAILURE_CLASS,
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


def _v2_mirror_coordinates(
    chain_name: str,
    stage_name: str,
) -> tuple[str, str]:
    """Map one V3 stage to its shared V2 ledger owner and branch key.

    Goal-kind stages mirror under the goal-bid producer keyed by their kind so
    each goal chain keeps the exact V2 per-branch budget; appraisal and
    terminal stages mirror under the semantic-appraisal producer keyed by
    ``chain:stage`` so every stage keeps its own full question budget, matching
    V2's one-question-one-budget parity without starving later stages of a
    multi-stage chain. The mapped caps are built from the same constants as
    V3's local ledger, keeping both counts 1:1 per stage.
    """
    if stage_name in GOAL_KINDS:
        return "goal_bid_structure", stage_name
    return "semantic_appraisal", f"{chain_name}:{stage_name}"


def _exhaustion_error_code(
    stage_name: str,
    last_failure_class: str | None,
) -> str:
    """Select the owner-specific exhaustion error code for one stage.

    Goal-kind stages split on the final attempt's failure class exactly as the
    V2 goal loop does (provider failures exhaust under the provider code,
    everything else under the structure code); appraisal and terminal stages
    keep the semantic-appraisal contract-exhausted code unchanged.
    """
    if stage_name not in GOAL_KINDS:
        return APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    if last_failure_class == PROVIDER_FAILURE_CLASS:
        return GOAL_BID_PROVIDER_EXHAUSTED_ERROR_CODE
    return GOAL_BID_STRUCTURE_EXHAUSTED_ERROR_CODE


def _reserve_v2_mirror_attempt(
    stage: str,
    branch_id: str,
    local_attempt: int,
) -> Mapping[str, object] | None:
    """Reserve one mirrored attempt in the shared V2 invocation ledger.

    Args:
        stage: The mapped V2 producer owner for this V3 stage.
        branch_id: The mapped V2 branch key for this chain or goal kind.
        local_attempt: The 1-based attempt number from the local ledger.

    Returns:
        The reserved coordinates for later disposition recording, or None when
        no invocation ledger is bound so standalone executor runs keep working
        without a context-bound ledger.

    Raises:
        V2AttemptBudgetExhausted: When the shared producer budget is exhausted
            before this model call; the calling stage fails closed at its
            shared-budget boundary instead of issuing another call.
    """
    if current_v2_attempt_ledger() is None:
        return None
    return reserve_v2_model_attempt(
        stage=stage, branch_id=branch_id, local_attempt=local_attempt
    )


def _record_v2_mirror_disposition(
    v2_coordinates: Mapping[str, object] | None,
    disposition: str,
) -> None:
    """Record one mirrored attempt disposition in the shared V2 ledger.

    Standalone execution without a bound invocation ledger skips silently so
    executor unit tests keep running outside a cognition graph.
    """
    if v2_coordinates is None or current_v2_attempt_ledger() is None:
        return
    record_v2_attempt_disposition(v2_coordinates, disposition=disposition)


async def _run_chain(
    spec: ChainTaskSpec,
    ledger: AttemptLedger,
) -> ChainOutcome:
    """Run one chain's registered stages serially under the shared ledger.

    Each stage consumes bounded attempts until acceptance or a terminal boundary
    rejection; exhaustion records the owner-specific error code and goal-kind
    stages additionally record their shared-branch exhausted disposition. When
    an active V2 invocation ledger exists, every attempt mirrors into its
    shared producer budget with V2 disposition semantics (accepted/recovered on
    acceptance, regenerate for non-final failures, exhausted for final ones,
    denied for terminal boundary rejections); a shared-budget exhaustion stops
    the stage without another model call. Every stage is optional, so an
    earlier-stage failure does not remove later independently valid stages from
    the registry order.
    """
    accepted_prefix: list[StageResult] = []
    results: list[StageResult] = []

    for stage_name in spec.stages:
        producer = spec.producers[stage_name]
        result: StageResult | None = None
        mirror_stage, mirror_branch = _v2_mirror_coordinates(
            spec.chain_name, stage_name
        )
        last_failure_class: str | None = None

        while ledger.can_reserve(stage_name):
            attempt_number = ledger.reserve(stage_name)
            try:
                v2_coordinates = _reserve_v2_mirror_attempt(
                    mirror_stage, mirror_branch, attempt_number
                )
            except V2AttemptBudgetExhausted:
                # The shared producer budget is exhausted before this model
                # call; fail the stage closed and let the trailing exhaustion
                # block record its owner-specific outcome.
                break

            outcome = await producer(
                ExecutorContext(
                    chain_name=spec.chain_name,
                    stage_name=stage_name,
                    attempt_number=attempt_number,
                    accepted_prefix=tuple(accepted_prefix),
                )
            )

            if outcome.accepted:
                _record_v2_mirror_disposition(
                    v2_coordinates,
                    "accepted" if attempt_number == 1 else "recovered",
                )
                result = StageResult(
                    chain_name=spec.chain_name,
                    stage_name=stage_name,
                    accepted=True,
                    local_state=outcome.local_state,
                    semantic_summary=outcome.semantic_summary,
                )
                break

            last_failure_class = outcome.failure_class

            if outcome.failure_class in TERMINAL_BOUNDARY_CLASSES:
                _record_v2_mirror_disposition(v2_coordinates, "denied")
                result = _boundary_rejection_result(
                    spec.chain_name, stage_name, outcome.failure_class
                )
                break

            shared_budget_exhausted = (
                v2_coordinates is not None
                and v2_coordinates["cumulative_producer_attempt"]
                >= v2_coordinates["configured_limit"]
            )
            if shared_budget_exhausted or not ledger.can_reserve(stage_name):
                _record_v2_mirror_disposition(v2_coordinates, "exhausted")
            else:
                _record_v2_mirror_disposition(v2_coordinates, "regenerate")

        if result is None:
            attempts_used = ledger.attempts_used(stage_name)
            error_code = _exhaustion_error_code(
                stage_name, last_failure_class
            )
            if (
                stage_name in GOAL_KINDS
                and current_v2_attempt_ledger() is not None
            ):
                record_v2_branch_disposition(
                    branch_id=mirror_branch,
                    disposition="exhausted",
                    error_code=error_code,
                )
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
                    error_code=error_code,
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
