"""Bounded parent-checkpoint recovery for the persona cognition path."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeVar
from uuid import uuid4

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    current_v2_attempt_ledger,
    enable_guarded_v2_attempt_ledger,
    set_v2_attempt_epoch,
    set_v2_parent_recovery_metadata,
)
from kazusa_ai_chatbot.llm_tracing import guardrail_capsule

ParentRecoveryDisposition = Literal[
    "not_attempted",
    "blocked_by_service_retry",
    "recovered",
    "exhausted",
]
ReplayClaimOwner = Literal["service_graph", "parent_checkpoint"]
ServicesT = TypeVar("ServicesT")
ParentCognitionRunner = Callable[
    [CognitionCoreInputV2, ServicesT],
    Awaitable[CognitionCoreOutputV2],
]

PARENT_RECOVERY_ERROR_CODES = frozenset({
    "goal_bid_structure_exhausted",
    "goal_bid_provider_exhausted",
})
PARENT_RECOVERY_EPOCH = 1
PARENT_RECOVERY_MAX_REPLAYS = 1


@dataclass
class CognitionRetryCoordinator:
    """Own one invocation-wide replay token and parent recovery epoch."""

    cognition_invocation_id: str
    replay_claimed: bool = False
    claimed_by: ReplayClaimOwner | None = None
    epoch: int = 0
    parent_recovery_attempted: bool = False
    parent_recovery_disposition: ParentRecoveryDisposition = "not_attempted"
    checkpoint_sha256: str = ""
    cycle_index: int = 0
    first_error: dict[str, object] | None = None
    second_error: dict[str, object] | None = None

    def claim_replay(self, owner: ReplayClaimOwner) -> bool:
        """Atomically consume the only invocation-wide replay token."""

        if self.replay_claimed:
            return False
        self.replay_claimed = True
        self.claimed_by = owner
        return True

    def claim_parent_checkpoint(
        self,
        exception: CognitionExecutionError,
        *,
        checkpoint_sha256: str,
        cycle_index: int,
    ) -> bool:
        """Claim parent recovery after the typed eligibility checks pass."""

        if not is_parent_recovery_eligible(exception):
            return False
        if self.replay_claimed:
            if self.claimed_by == "service_graph":
                self.parent_recovery_disposition = (
                    "blocked_by_service_retry"
                )
                if self.first_error is None:
                    self.first_error = _error_metadata(exception)
                    self.checkpoint_sha256 = checkpoint_sha256
                    self.cycle_index = cycle_index
            return False
        if not self.claim_replay("parent_checkpoint"):
            return False
        self.parent_recovery_attempted = True
        self.parent_recovery_disposition = "not_attempted"
        self.epoch = PARENT_RECOVERY_EPOCH
        self.checkpoint_sha256 = checkpoint_sha256
        self.cycle_index = cycle_index
        self.first_error = _error_metadata(exception)
        return True

    def finish_parent_recovery(
        self,
        disposition: ParentRecoveryDisposition,
        exception: BaseException | None = None,
    ) -> None:
        """Record the bounded result of the one parent child execution."""

        if disposition not in {"recovered", "exhausted"}:
            raise ValueError("parent recovery disposition is invalid")
        self.parent_recovery_disposition = disposition
        if exception is not None:
            self.second_error = _error_metadata(exception)

    def snapshot(self) -> dict[str, object]:
        """Return bounded metadata for protected tracing and service policy."""

        return {
            "cognition_invocation_id": self.cognition_invocation_id,
            "replay_claimed": self.replay_claimed,
            "claimed_by": self.claimed_by or "",
            "epoch": self.epoch,
            "parent_recovery_attempted": self.parent_recovery_attempted,
            "parent_recovery_disposition": self.parent_recovery_disposition,
            "checkpoint_sha256": self.checkpoint_sha256,
            "cycle_index": self.cycle_index,
            "first_error": dict(self.first_error or {}),
            "second_error": dict(self.second_error or {}),
        }


_CURRENT_COORDINATOR: ContextVar[CognitionRetryCoordinator | None] = (
    ContextVar("cognition_v2_retry_coordinator", default=None)
)


def create_cognition_retry_coordinator(
    cognition_invocation_id: str | None = None,
) -> CognitionRetryCoordinator:
    """Create one unbound coordinator for a queued persona invocation."""

    invocation_id = str(cognition_invocation_id or "").strip()
    if not invocation_id:
        invocation_id = uuid4().hex
    return CognitionRetryCoordinator(
        cognition_invocation_id=invocation_id,
    )


def current_cognition_retry_coordinator() -> CognitionRetryCoordinator | None:
    """Return the coordinator bound to the current async context."""

    return _CURRENT_COORDINATOR.get()


def bind_cognition_retry_coordinator(
    coordinator: CognitionRetryCoordinator,
) -> Token[CognitionRetryCoordinator | None]:
    """Bind one coordinator across service graph attempts."""

    if not isinstance(coordinator, CognitionRetryCoordinator):
        raise TypeError("cognition retry coordinator is invalid")
    return _CURRENT_COORDINATOR.set(coordinator)


def reset_cognition_retry_coordinator(
    token: Token[CognitionRetryCoordinator | None],
) -> None:
    """Restore the coordinator context preceding one service invocation."""

    _CURRENT_COORDINATOR.reset(token)


def is_parent_recovery_eligible(exception: BaseException) -> bool:
    """Return whether one escaped child failure may enter parent recovery."""

    if isinstance(exception, ParentRecoveryExhaustedError):
        return False
    if not isinstance(exception, CognitionExecutionError):
        return False
    if exception.safe_checkpoint != "pre_state_commit":
        return False
    if exception.error_code not in PARENT_RECOVERY_ERROR_CODES:
        return False
    ledger = current_v2_attempt_ledger()
    if ledger is None or not exception.branch_id:
        return True
    branch_disposition = ledger.branch_dispositions.get(exception.branch_id)
    if not isinstance(branch_disposition, Mapping):
        return True
    return branch_disposition.get("disposition") != "recovered_by_sibling"


async def run_guarded_cognition(
    input_payload: CognitionCoreInputV2,
    services: ServicesT,
    *,
    run_child: ParentCognitionRunner[ServicesT],
) -> CognitionCoreOutputV2:
    """Run one cognition child and at most one parent-checkpoint replay."""

    coordinator = current_cognition_retry_coordinator()
    if coordinator is None:
        output = await run_child(input_payload, services)
        return output

    checkpoint = deepcopy(input_payload)
    checkpoint_sha256 = _cognition_input_digest(checkpoint)
    cycle_index = _cycle_index(checkpoint)
    if current_v2_attempt_ledger() is not None:
        enable_guarded_v2_attempt_ledger()
        set_v2_attempt_epoch(coordinator.epoch)
    try:
        output = await run_child(deepcopy(checkpoint), services)
        return output
    except CognitionExecutionError as first_exception:
        if not coordinator.claim_parent_checkpoint(
            first_exception,
            checkpoint_sha256=checkpoint_sha256,
            cycle_index=cycle_index,
        ):
            raise

        guardrail_session = guardrail_capsule.begin_guardrail_capsule(
            trace_id=llm_tracing.current_trace_id(),
            scope="persona_stage_1",
            cycle_index=cycle_index,
            checkpoint_sha256=checkpoint_sha256,
        )
        guardrail_capsule.record_guardrail_trigger(
            guardrail_session,
            error=first_exception,
        )
        if current_v2_attempt_ledger() is not None:
            set_v2_parent_recovery_metadata(
                disposition="attempted",
                claimed_by="parent_checkpoint",
                epoch=PARENT_RECOVERY_EPOCH,
                checkpoint_sha256=checkpoint_sha256,
            )
            set_v2_attempt_epoch(PARENT_RECOVERY_EPOCH)
        try:
            recovered_output = await run_child(deepcopy(checkpoint), services)
        except asyncio.CancelledError:
            guardrail_capsule.discard_guardrail_capsule(
                guardrail_capsule.current_guardrail_capsule(),
            )
            raise
        except CognitionExecutionError as second_exception:
            coordinator.finish_parent_recovery(
                "exhausted",
                second_exception,
            )
            if current_v2_attempt_ledger() is not None:
                set_v2_parent_recovery_metadata(
                    disposition="exhausted",
                    claimed_by="parent_checkpoint",
                    epoch=PARENT_RECOVERY_EPOCH,
                    checkpoint_sha256=checkpoint_sha256,
                )
            raise ParentRecoveryExhaustedError(
                first_error=first_exception,
                second_error=second_exception,
                checkpoint_sha256=checkpoint_sha256,
                recovery_epoch=PARENT_RECOVERY_EPOCH,
            ) from second_exception

        coordinator.finish_parent_recovery("recovered")
        if current_v2_attempt_ledger() is not None:
            set_v2_parent_recovery_metadata(
                disposition="recovered",
                claimed_by="parent_checkpoint",
                epoch=PARENT_RECOVERY_EPOCH,
                checkpoint_sha256=checkpoint_sha256,
            )
        return_value = recovered_output
        return return_value


class ParentRecoveryExhaustedError(CognitionExecutionError):
    """Report a failed second child without allowing another outer replay."""

    def __init__(
        self,
        *,
        first_error: BaseException,
        second_error: BaseException,
        checkpoint_sha256: str,
        recovery_epoch: int,
    ) -> None:
        """Project the second child and bounded first-error metadata."""

        second_metadata = _error_metadata(second_error)
        first_metadata = _error_metadata(first_error)
        super().__init__(
            "parent cognition recovery exhausted",
            error_code=str(second_metadata["error_code"]),
            branch_id=str(second_metadata["branch_id"]),
            stage=str(second_metadata["stage"]),
            attempt_count=int(second_metadata["attempt_count"]),
            safe_checkpoint="pre_state_commit",
            retryable=False,
        )
        self.parent_recovery_attempted = True
        self.parent_recovery_disposition = "exhausted"
        self.first_error_code = str(first_metadata["error_code"])
        self.first_error_stage = str(first_metadata["stage"])
        self.first_error_branch_id = str(first_metadata["branch_id"])
        self.first_error_attempt_count = int(
            first_metadata["attempt_count"]
        )
        self.parent_checkpoint_digest = checkpoint_sha256
        self.recovery_epoch = recovery_epoch


def _cognition_input_digest(input_payload: CognitionCoreInputV2) -> str:
    """Hash one canonical JSON cognition input without retaining its content."""

    serialized_input = json.dumps(
        input_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized_input.encode("utf-8")).hexdigest()
    return digest


def _cycle_index(input_payload: CognitionCoreInputV2) -> int:
    """Read the bounded resolver cycle coordinate from canonical input."""

    cycle_index = input_payload.get("resolver_cycle_index", 0)
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int):
        return 0
    return max(0, cycle_index)


def _error_metadata(exception: BaseException) -> dict[str, object]:
    """Project bounded error coordinates without retaining exception text."""

    attempt_count = getattr(exception, "attempt_count", 1)
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        attempt_count = 1
    return {
        "error_code": str(
            getattr(exception, "error_code", "internal_invariant")
        ),
        "stage": str(getattr(exception, "stage", "")),
        "branch_id": str(getattr(exception, "branch_id", "")),
        "attempt_count": max(1, attempt_count),
    }
