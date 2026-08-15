"""Bounded retry and terminal-disposition metadata for V2 model owners."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal, Mapping, TypedDict, cast
from uuid import uuid4


V2_MODEL_TOTAL_ATTEMPTS = 3
V2_APPRAISAL_TOTAL_ATTEMPTS = 2

V2AttemptFailureKind = Literal[
    "provider",
    "parse",
    "structure",
    "semantic",
]
V2AttemptDisposition = Literal[
    "started",
    "accepted",
    "recovered",
    "accepted_degraded",
    "regenerate",
    "exhausted",
    "recovered_by_sibling",
    "retry_graph",
    "empty",
    "denied",
    "skipped",
    "unrecoverable",
]
V2SafeCheckpoint = Literal[
    "pre_state_commit",
    "post_cognition_commit",
]


class V2AttemptRecord(TypedDict):
    """Bounded diagnostic metadata for one V2 model owner."""

    stage: str
    failure_kind: V2AttemptFailureKind | None
    attempt_count: int
    total_attempt_limit: int
    selected_attempt: int | None
    disposition: V2AttemptDisposition
    safe_checkpoint: V2SafeCheckpoint


class V2ModelOwnerPolicy(TypedDict):
    """Attempt limit and exhausted disposition for one V2 model owner."""

    total_attempt_limit: int
    exhausted_disposition: V2AttemptDisposition


class V2AttemptCoordinates(TypedDict):
    """Invocation-wide coordinates for one reserved producer call."""

    cognition_invocation_id: str
    graph_attempt: int
    branch_id: str
    producing_stage: str
    local_attempt: int
    cumulative_producer_attempt: int
    configured_limit: int


class V2AttemptLedgerRecord(V2AttemptCoordinates):
    """Protected coordinates plus the producing stage's disposition."""

    attempt_disposition: V2AttemptDisposition


class V2BranchDispositionRecord(TypedDict):
    """Terminal branch outcome retained by protected diagnostics."""

    branch_id: str
    disposition: V2AttemptDisposition
    error_code: str


@dataclass
class V2InvocationAttemptLedger:
    """Share model-call budgets across one cognition invocation."""

    cognition_invocation_id: str
    graph_attempt: int = 1
    producer_attempts: dict[tuple[str, str], int] = field(
        default_factory=dict,
        repr=False,
    )
    attempts: list[V2AttemptLedgerRecord] = field(default_factory=list)
    branch_dispositions: dict[str, V2BranchDispositionRecord] = field(
        default_factory=dict,
    )
    guarded: bool = False
    epoch: int = 0
    guarded_producer_attempts: dict[tuple[int, str, str], int] = field(
        default_factory=dict,
        repr=False,
    )
    guarded_attempts: list[dict[str, object]] = field(
        default_factory=list,
        repr=False,
    )
    attempt_epochs: list[int] = field(
        default_factory=list,
        repr=False,
    )
    guarded_epoch_branch_dispositions: dict[
        tuple[int, str],
        V2BranchDispositionRecord,
    ] = field(default_factory=dict, repr=False)
    parent_recovery: dict[str, object] = field(
        default_factory=dict,
        repr=False,
    )


class V2AttemptBudgetExhausted(RuntimeError):
    """Raised before a model call that would exceed its invocation budget."""

    def __init__(
        self,
        *,
        stage: str,
        branch_id: str,
        configured_limit: int,
    ) -> None:
        """Retain the exhausted producer key and configured bound."""

        super().__init__(
            f"{stage} attempt budget exhausted for branch {branch_id}"
        )
        self.stage = stage
        self.branch_id = branch_id
        self.configured_limit = configured_limit


V2_MODEL_OWNER_POLICIES: dict[str, V2ModelOwnerPolicy] = {
    "image_descriptor": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "message_decontextualizer": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "semantic_appraisal": {
        "total_attempt_limit": V2_APPRAISAL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "goal_bid_structure": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "unrecoverable",
    },
    "workspace_collapse": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "action_planning": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "empty",
    },
    "action_authorization": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "denied",
    },
    "resolver_authorization": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "denied",
    },
    "surface_content_plan": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "surface_preference": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
    "surface_visual": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "skipped",
    },
    "dialog_generator": {
        "total_attempt_limit": V2_MODEL_TOTAL_ATTEMPTS,
        "exhausted_disposition": "accepted_degraded",
    },
}

_ATTEMPT_RECORD_FIELDS = frozenset({
    "stage",
    "failure_kind",
    "attempt_count",
    "total_attempt_limit",
    "selected_attempt",
    "disposition",
    "safe_checkpoint",
})
_FAILURE_KINDS = frozenset({
    "provider",
    "parse",
    "structure",
    "semantic",
})
_DISPOSITIONS = frozenset({
    "started",
    "accepted",
    "recovered",
    "accepted_degraded",
    "regenerate",
    "exhausted",
    "recovered_by_sibling",
    "retry_graph",
    "empty",
    "denied",
    "skipped",
    "unrecoverable",
})
_SAFE_CHECKPOINTS = frozenset({
    "pre_state_commit",
    "post_cognition_commit",
})

_CURRENT_ATTEMPT_LEDGER: ContextVar[V2InvocationAttemptLedger | None] = (
    ContextVar("cognition_v2_attempt_ledger", default=None)
)


def create_v2_attempt_ledger(
    cognition_invocation_id: str | None = None,
) -> V2InvocationAttemptLedger:
    """Create one unbound invocation ledger with a stable non-content id."""

    invocation_id = str(cognition_invocation_id or "").strip()
    if not invocation_id:
        invocation_id = uuid4().hex
    return V2InvocationAttemptLedger(
        cognition_invocation_id=invocation_id,
    )


def current_v2_attempt_ledger() -> V2InvocationAttemptLedger | None:
    """Return the context-local invocation ledger, when one is bound."""

    return _CURRENT_ATTEMPT_LEDGER.get()


def bind_v2_attempt_ledger(
    ledger: V2InvocationAttemptLedger,
    *,
    graph_attempt: int,
) -> Token[V2InvocationAttemptLedger | None]:
    """Bind one ledger for a specific service graph attempt."""

    if (
        isinstance(graph_attempt, bool)
        or not isinstance(graph_attempt, int)
        or graph_attempt < 1
    ):
        raise ValueError("graph attempt must be a positive integer")
    ledger.graph_attempt = graph_attempt
    return _CURRENT_ATTEMPT_LEDGER.set(ledger)


def reset_v2_attempt_ledger(
    token: Token[V2InvocationAttemptLedger | None],
) -> None:
    """Restore the attempt-ledger context that preceded one binding."""

    _CURRENT_ATTEMPT_LEDGER.reset(token)


def enable_guarded_v2_attempt_ledger() -> None:
    """Enable epoch-aware producer accounting for one guarded invocation."""

    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        raise RuntimeError("guarded V2 attempts require an invocation ledger")
    ledger.guarded = True


def set_v2_attempt_epoch(epoch: int) -> None:
    """Select the bounded epoch used by the guarded producer ledger."""

    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError("V2 attempt epoch must be an integer")
    if epoch not in (0, 1):
        raise ValueError("V2 attempt epoch is outside the two-epoch bound")
    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        raise RuntimeError("V2 attempt epoch requires an invocation ledger")
    ledger.epoch = epoch


def set_v2_parent_recovery_metadata(
    *,
    disposition: str,
    claimed_by: str,
    epoch: int,
    checkpoint_sha256: str,
) -> None:
    """Store bounded parent-recovery metadata beside the V1 ledger."""

    if disposition not in {"attempted", "recovered", "exhausted"}:
        raise ValueError("V2 parent recovery disposition is invalid")
    if claimed_by != "parent_checkpoint":
        raise ValueError("V2 parent recovery owner is invalid")
    if epoch != 1:
        raise ValueError("V2 parent recovery epoch is invalid")
    if len(checkpoint_sha256) != 64:
        raise ValueError("V2 parent recovery digest is invalid")
    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        return
    ledger.parent_recovery = {
        "disposition": disposition,
        "claimed_by": claimed_by,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "max_replays": 1,
    }


def reserve_v2_model_attempt(
    *,
    stage: str,
    branch_id: str,
    local_attempt: int,
) -> V2AttemptCoordinates:
    """Consume one call from an invocation-wide producer/branch budget."""

    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        raise RuntimeError("V2 model attempt requires an invocation ledger")
    if stage not in V2_MODEL_OWNER_POLICIES:
        raise ValueError("V2 model attempt stage is invalid")
    if not isinstance(branch_id, str) or not branch_id.strip():
        raise ValueError("V2 model attempt branch id is required")
    if (
        isinstance(local_attempt, bool)
        or not isinstance(local_attempt, int)
        or local_attempt < 1
    ):
        raise ValueError("V2 local attempt must be a positive integer")

    configured_limit = V2_MODEL_OWNER_POLICIES[stage][
        "total_attempt_limit"
    ]
    if ledger.guarded:
        guarded_key = (ledger.epoch, stage, branch_id)
        consumed = ledger.guarded_producer_attempts.get(guarded_key, 0)
    else:
        producer_key = (stage, branch_id)
        consumed = ledger.producer_attempts.get(producer_key, 0)
    if consumed >= configured_limit:
        raise V2AttemptBudgetExhausted(
            stage=stage,
            branch_id=branch_id,
            configured_limit=configured_limit,
        )

    cumulative_attempt = consumed + 1
    if ledger.guarded:
        ledger.guarded_producer_attempts[guarded_key] = cumulative_attempt
    else:
        ledger.producer_attempts[producer_key] = cumulative_attempt
    coordinates: V2AttemptCoordinates = {
        "cognition_invocation_id": ledger.cognition_invocation_id,
        "graph_attempt": ledger.graph_attempt,
        "branch_id": branch_id,
        "producing_stage": stage,
        "local_attempt": local_attempt,
        "cumulative_producer_attempt": cumulative_attempt,
        "configured_limit": configured_limit,
    }
    record = cast(V2AttemptLedgerRecord, {
        **coordinates,
        "attempt_disposition": "started",
    })
    ledger.attempts.append(record)
    ledger.attempt_epochs.append(ledger.epoch)
    if ledger.guarded:
        ledger.guarded_attempts.append({
            **coordinates,
            "epoch": ledger.epoch,
            "attempt_disposition": "started",
        })
    return coordinates


def record_v2_attempt_disposition(
    coordinates: Mapping[str, object],
    *,
    disposition: V2AttemptDisposition,
) -> None:
    """Attach one producer-owned disposition to a reserved attempt."""

    if disposition not in _DISPOSITIONS:
        raise ValueError("V2 attempt disposition is invalid")
    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        raise RuntimeError("V2 attempt disposition requires an invocation ledger")
    for record in reversed(ledger.attempts):
        if all(record.get(key) == value for key, value in coordinates.items()):
            record["attempt_disposition"] = disposition
            break
    else:
        raise ValueError("V2 attempt coordinates are not reserved")
    if ledger.guarded:
        for record in reversed(ledger.guarded_attempts):
            if (
                record.get("epoch") == ledger.epoch
                and all(
                    record.get(key) == value
                    for key, value in coordinates.items()
                )
            ):
                record["attempt_disposition"] = disposition
                return
        raise ValueError("guarded V2 attempt coordinates are not reserved")


def record_v2_branch_disposition(
    *,
    branch_id: str,
    disposition: V2AttemptDisposition,
    error_code: str = "",
) -> None:
    """Retain the latest terminal outcome for one cognition branch."""

    if disposition not in _DISPOSITIONS:
        raise ValueError("V2 branch disposition is invalid")
    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        return
    ledger.branch_dispositions[branch_id] = {
        "branch_id": branch_id,
        "disposition": disposition,
        "error_code": error_code,
    }
    if ledger.guarded:
        ledger.guarded_epoch_branch_dispositions[(ledger.epoch, branch_id)] = {
            "branch_id": branch_id,
            "disposition": disposition,
            "error_code": error_code,
        }


def snapshot_v2_attempt_ledger() -> dict[str, object] | None:
    """Copy bounded ledger metadata for protected failure diagnostics."""

    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None:
        return None
    if ledger.guarded:
        attempts = [
            dict(record)
            for index, record in enumerate(ledger.attempts)
            if ledger.attempt_epochs[index] == ledger.epoch
        ]
        branch_dispositions = [
            dict(ledger.guarded_epoch_branch_dispositions[(ledger.epoch, branch_id)])
            for epoch, branch_id in sorted(
                ledger.guarded_epoch_branch_dispositions
            )
            if epoch == ledger.epoch
        ]
    else:
        attempts = [dict(record) for record in ledger.attempts]
        branch_dispositions = [
            dict(ledger.branch_dispositions[branch_id])
            for branch_id in sorted(ledger.branch_dispositions)
        ]
    return {
        "schema_version": "cognition_attempt_ledger.v1",
        "cognition_invocation_id": ledger.cognition_invocation_id,
        "attempts": attempts,
        "branch_dispositions": branch_dispositions,
    }


def snapshot_v2_guarded_attempt_ledger() -> dict[str, object] | None:
    """Return bounded two-epoch metadata for the outer guardrail capsule."""

    ledger = _CURRENT_ATTEMPT_LEDGER.get()
    if ledger is None or not ledger.guarded:
        return None
    epochs: list[dict[str, object]] = []
    for epoch in (0, 1):
        attempts = [
            dict(record)
            for record in ledger.guarded_attempts
            if record.get("epoch") == epoch
        ]
        branch_dispositions = [
            dict(ledger.guarded_epoch_branch_dispositions[(epoch, branch_id)])
            for current_epoch, branch_id in sorted(
                ledger.guarded_epoch_branch_dispositions
            )
            if current_epoch == epoch
        ]
        if attempts or branch_dispositions:
            epochs.append({
                "epoch": epoch,
                "attempts": attempts,
                "branch_dispositions": branch_dispositions,
            })
    return {
        "schema_version": "cognition_attempt_ledger.v2",
        "cognition_invocation_id": ledger.cognition_invocation_id,
        "epochs": epochs,
        "parent_recovery": dict(ledger.parent_recovery),
    }


def validate_v2_attempt_record(
    value: Mapping[str, object],
) -> V2AttemptRecord:
    """Validate bounded diagnostic metadata without accepting model content.

    Args:
        value: Candidate metadata produced by a V2 model-owning stage.

    Returns:
        A copied record containing only the approved diagnostic fields.
    """

    if set(value) != _ATTEMPT_RECORD_FIELDS:
        raise ValueError("V2 attempt record fields are not exact")

    stage = value["stage"]
    if (
        not isinstance(stage, str)
        or stage not in V2_MODEL_OWNER_POLICIES
    ):
        raise ValueError("V2 attempt record stage is invalid")

    total_attempt_limit = value["total_attempt_limit"]
    owner_limit = V2_MODEL_OWNER_POLICIES[stage]["total_attempt_limit"]
    if (
        isinstance(total_attempt_limit, bool)
        or not isinstance(total_attempt_limit, int)
        or total_attempt_limit != owner_limit
    ):
        raise ValueError("V2 attempt record total limit is invalid")

    attempt_count = value["attempt_count"]
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or not 1 <= attempt_count <= total_attempt_limit
    ):
        raise ValueError("V2 attempt record attempt count is invalid")

    selected_attempt = value["selected_attempt"]
    if selected_attempt is not None and (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or not 1 <= selected_attempt <= attempt_count
    ):
        raise ValueError("V2 attempt record selected attempt is invalid")

    failure_kind = value["failure_kind"]
    if failure_kind is not None and failure_kind not in _FAILURE_KINDS:
        raise ValueError("V2 attempt record failure kind is invalid")

    disposition = value["disposition"]
    if disposition not in _DISPOSITIONS:
        raise ValueError("V2 attempt record disposition is invalid")

    safe_checkpoint = value["safe_checkpoint"]
    if safe_checkpoint not in _SAFE_CHECKPOINTS:
        raise ValueError("V2 attempt record safe checkpoint is invalid")

    validated = cast(V2AttemptRecord, dict(value))
    return validated
