"""Protected bounded capture for parent-checkpoint guardrail recovery."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from kazusa_ai_chatbot.config import DEBUG_LOG_TTL_DAYS, LLM_TRACE_CAPTURE_MODE
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_OWNER_POLICIES,
)
from kazusa_ai_chatbot.db import DatabaseBackendError
from kazusa_ai_chatbot.db import llm_tracing as db_llm_tracing
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


GuardrailDisposition = Literal["recovered", "exhausted"]
GUARDRAIL_CAPSULE_WRITE_TIMEOUT_SECONDS = 0.25
MAX_CAPSULE_IDENTIFIER_CHARS = 128
MAX_CAPSULE_ATTEMPTS_PER_EPOCH = 64
MAX_CAPSULE_BRANCHES_PER_EPOCH = 32
MAX_CAPSULE_EPOCHS = 2
MAX_CAPSULE_ATTEMPT_COUNT = 32
MAX_CAPSULE_GRAPH_ATTEMPT = 2
MAX_CAPSULE_LOCAL_ATTEMPT = 3
MAX_CAPSULE_CUMULATIVE_ATTEMPT = 3
MAX_CAPSULE_CONFIGURED_LIMIT = 3

_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_:-."
)
_ATTEMPT_DISPOSITIONS = frozenset({
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
_PARENT_RECOVERY_DISPOSITIONS = frozenset({
    "attempted",
    "recovered",
    "exhausted",
})
_GUARDED_ATTEMPT_FIELDS = frozenset({
    "cognition_invocation_id",
    "graph_attempt",
    "branch_id",
    "producing_stage",
    "local_attempt",
    "cumulative_producer_attempt",
    "configured_limit",
    "epoch",
    "attempt_disposition",
})
_BRANCH_DISPOSITION_FIELDS = frozenset({
    "branch_id",
    "disposition",
    "error_code",
})
_PARENT_RECOVERY_FIELDS = frozenset({
    "disposition",
    "claimed_by",
    "epoch",
    "checkpoint_sha256",
    "max_replays",
})
_CAPSULE_MODEL_OWNER_LIMITS = {
    stage: int(policy["total_attempt_limit"])
    for stage, policy in V2_MODEL_OWNER_POLICIES.items()
}

logger = logging.getLogger(__name__)


@dataclass
class GuardrailCapsuleSession:
    """Hold only bounded parent-recovery coordinates for one service turn."""

    trace_id: str
    scope: str
    cycle_index: int
    checkpoint_sha256: str
    guardrail_invocation_id: str = field(default_factory=lambda: uuid4().hex)
    trigger: dict[str, object] | None = None
    finished: bool = False
    context_token: Token[GuardrailCapsuleSession | None] | None = field(
        default=None,
        repr=False,
    )


_CURRENT_SESSION: ContextVar[GuardrailCapsuleSession | None] = ContextVar(
    "cognition_v2_parent_guardrail_capsule",
    default=None,
)
_PENDING_PERSISTENCE_TASKS: set[asyncio.Task[None]] = set()


def begin_guardrail_capsule(
    *,
    trace_id: str,
    scope: str,
    cycle_index: int,
    checkpoint_sha256: str,
) -> GuardrailCapsuleSession | None:
    """Bind a bounded outer capture when protected tracing is available."""

    trace_ref = _bounded_identifier(trace_id)
    if LLM_TRACE_CAPTURE_MODE == "off" or not trace_ref:
        return None
    if scope != "persona_stage_1":
        raise ValueError("guardrail capsule scope is invalid")
    if isinstance(cycle_index, bool) or not isinstance(cycle_index, int):
        raise ValueError("guardrail capsule cycle index is invalid")
    if not _is_sha256_digest(checkpoint_sha256):
        raise ValueError("guardrail capsule checkpoint digest is invalid")
    session = GuardrailCapsuleSession(
        trace_id=trace_ref,
        scope=scope,
        cycle_index=max(0, cycle_index),
        checkpoint_sha256=checkpoint_sha256,
    )
    context_token = _CURRENT_SESSION.set(session)
    session.context_token = context_token
    return session


def current_guardrail_capsule() -> GuardrailCapsuleSession | None:
    """Return the context-local outer capsule, when one is active."""

    return _CURRENT_SESSION.get()


def discard_guardrail_capsule(
    session: GuardrailCapsuleSession | None,
) -> None:
    """Drop an unfinished outer session and restore its prior context."""

    if session is None or session.finished:
        return
    session.finished = True
    _restore_prior_session(session)


def record_guardrail_trigger(
    session: GuardrailCapsuleSession | None,
    *,
    error: CognitionExecutionError,
) -> None:
    """Retain only the bounded coordinates that triggered parent recovery."""

    if session is None or session.finished:
        return
    session.trigger = {
        "error_code": _bounded_identifier(error.error_code),
        "stage": _bounded_identifier(error.stage),
        "branch_id": _bounded_identifier(error.branch_id),
        "attempt_count": _bounded_attempt_count(error.attempt_count),
    }


def finish_guardrail_capsule(
    session: GuardrailCapsuleSession | None,
    *,
    coordinator_snapshot: Mapping[str, object],
    attempt_ledger: Mapping[str, object] | None,
    disposition: GuardrailDisposition,
) -> str:
    """Persist bounded outer metadata after parent recovery or discard it."""

    if session is None:
        return ""
    if session.finished:
        return session.guardrail_invocation_id
    session.finished = True
    _restore_prior_session(session)
    if session.trigger is None:
        return ""
    if disposition not in {"recovered", "exhausted"}:
        raise ValueError("guardrail capsule disposition is invalid")

    parent_recovery = {
        "disposition": disposition,
        "claimed_by": "parent_checkpoint",
        "epoch": 1,
        "max_replays": 1,
    }
    capsule = {
        "schema_version": "cognition_parent_guardrail_capsule.v1",
        "trace_id": session.trace_id,
        "guardrail_invocation_id": session.guardrail_invocation_id,
        "scope": session.scope,
        "cycle_index": session.cycle_index,
        "checkpoint_sha256": session.checkpoint_sha256,
        "trigger": dict(session.trigger),
        "parent_recovery": parent_recovery,
        "attempt_ledger": _project_attempt_ledger(attempt_ledger),
    }
    document = _capsule_document(capsule)
    _schedule_persistence(document)
    return session.guardrail_invocation_id


def _restore_prior_session(session: GuardrailCapsuleSession) -> None:
    """Restore the context that preceded one outer capsule."""

    context_token = session.context_token
    if context_token is None:
        return
    _CURRENT_SESSION.reset(context_token)


def _capsule_document(capsule: Mapping[str, object]) -> dict[str, object]:
    """Build one protected trace-step document with retention metadata."""

    trace_id = str(capsule["trace_id"])
    invocation_id = str(capsule["guardrail_invocation_id"])
    created_at = storage_utc_now_iso()
    document = {
        "step_id": f"{trace_id}_cognition_guardrail_{invocation_id}",
        "trace_id": trace_id,
        "sequence": 0,
        "stage_name": "cognition_parent_guardrail",
        "capture_reason": "cognition_parent_guardrail",
        "cognition_invocation_id": invocation_id,
        "capsule": dict(capsule),
        "created_at": created_at,
        "expires_at": expiry_from_storage_iso(
            created_at,
            ttl_days=DEBUG_LOG_TTL_DAYS,
        ),
    }
    return document


def _schedule_persistence(document: Mapping[str, object]) -> None:
    """Schedule a bounded protected write without delaying the chat turn."""

    persistence_coroutine = _persist_capsule(document)
    try:
        task = asyncio.create_task(persistence_coroutine)
    except RuntimeError as exc:
        persistence_coroutine.close()
        logger.warning(f"Guardrail capsule task scheduling failed: {exc}")
        return
    _PENDING_PERSISTENCE_TASKS.add(task)
    task.add_done_callback(_observe_persistence_task)


async def _persist_capsule(document: Mapping[str, object]) -> None:
    """Write one outer capsule with a short bounded timeout."""

    try:
        await asyncio.wait_for(
            db_llm_tracing.insert_trace_step(document),
            timeout=GUARDRAIL_CAPSULE_WRITE_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except (
        ConnectionError,
        OSError,
        DatabaseBackendError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        logger.warning(
            f"Guardrail capsule persistence failed: {exc}"
        )


def _observe_persistence_task(task: asyncio.Task[None]) -> None:
    """Consume a protected write task and keep its failure observable."""

    _PENDING_PERSISTENCE_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except (
        ConnectionError,
        OSError,
        DatabaseBackendError,
        RuntimeError,
        TimeoutError,
    ) as exc:
        logger.warning(
            f"Guardrail capsule persistence task failed: {exc}"
        )


def _bounded_identifier(value: object) -> str:
    """Return one opaque identifier only when its shape is safe and bounded."""

    if not isinstance(value, str):
        return ""
    if not value or len(value) > MAX_CAPSULE_IDENTIFIER_CHARS:
        return ""
    if not all(character in _IDENTIFIER_CHARACTERS for character in value):
        return ""
    return value


def _is_sha256_digest(value: object) -> bool:
    """Return whether a value is an exact lowercase-or-uppercase SHA-256 ref."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _bounded_attempt_count(value: object) -> int:
    """Clamp an internal attempt count to the diagnostic schema bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return min(MAX_CAPSULE_ATTEMPT_COUNT, max(1, value))


def _project_attempt_ledger(
    attempt_ledger: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project only the exact bounded V2 ledger fields into the outer capsule."""

    empty_ledger = {
        "schema_version": "cognition_attempt_ledger.v2",
        "epochs": [],
        "parent_recovery": {},
    }
    if not isinstance(attempt_ledger, Mapping):
        return empty_ledger

    raw_epochs = attempt_ledger.get("epochs")
    projected_epochs: list[dict[str, object]] = []
    if isinstance(raw_epochs, (list, tuple)):
        for raw_epoch in list(raw_epochs)[:MAX_CAPSULE_EPOCHS]:
            projected_epoch = _project_epoch(raw_epoch)
            if projected_epoch is not None:
                projected_epochs.append(projected_epoch)

    raw_parent_recovery = attempt_ledger.get("parent_recovery")
    parent_recovery = _project_parent_recovery(raw_parent_recovery)
    return {
        "schema_version": "cognition_attempt_ledger.v2",
        "epochs": projected_epochs,
        "parent_recovery": parent_recovery,
    }


def _project_epoch(value: object) -> dict[str, object] | None:
    """Project one exact epoch record and discard malformed records."""

    if not isinstance(value, Mapping):
        return None
    epoch = value.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        return None
    if epoch not in (0, 1):
        return None

    projected_attempts: list[dict[str, object]] = []
    raw_attempts = value.get("attempts")
    if isinstance(raw_attempts, (list, tuple)):
        for raw_attempt in list(raw_attempts)[:MAX_CAPSULE_ATTEMPTS_PER_EPOCH]:
            projected_attempt = _project_attempt(
                raw_attempt,
                enclosing_epoch=epoch,
            )
            if projected_attempt is not None:
                projected_attempts.append(projected_attempt)

    projected_branches: list[dict[str, str]] = []
    raw_branches = value.get("branch_dispositions")
    if isinstance(raw_branches, (list, tuple)):
        for raw_branch in list(raw_branches)[:MAX_CAPSULE_BRANCHES_PER_EPOCH]:
            projected_branch = _project_branch(raw_branch)
            if projected_branch is not None:
                projected_branches.append(projected_branch)

    return {
        "epoch": epoch,
        "attempts": projected_attempts,
        "branch_dispositions": projected_branches,
    }


def _project_attempt(
    value: object,
    *,
    enclosing_epoch: int,
) -> dict[str, object] | None:
    """Project one attempt only when its key set and scalar values are exact."""

    if not isinstance(value, Mapping):
        return None
    if set(value) != _GUARDED_ATTEMPT_FIELDS:
        return None
    string_fields = (
        "cognition_invocation_id",
        "branch_id",
        "producing_stage",
    )
    projected: dict[str, object] = {}
    for field_name in string_fields:
        identifier = _bounded_identifier(value[field_name])
        if not identifier:
            return None
        projected[field_name] = identifier
    coordinate_bounds = {
        "graph_attempt": (1, MAX_CAPSULE_GRAPH_ATTEMPT),
        "local_attempt": (1, MAX_CAPSULE_LOCAL_ATTEMPT),
        "cumulative_producer_attempt": (
            1,
            MAX_CAPSULE_CUMULATIVE_ATTEMPT,
        ),
        "configured_limit": (1, MAX_CAPSULE_CONFIGURED_LIMIT),
        "epoch": (0, MAX_CAPSULE_EPOCHS - 1),
    }
    for field_name, (minimum, maximum) in coordinate_bounds.items():
        numeric_value = value[field_name]
        if (
            isinstance(numeric_value, bool)
            or not isinstance(numeric_value, int)
            or numeric_value < minimum
            or numeric_value > maximum
        ):
            return None
        projected[field_name] = numeric_value
    if projected["epoch"] != enclosing_epoch:
        return None
    expected_limit = _CAPSULE_MODEL_OWNER_LIMITS.get(
        str(projected["producing_stage"])
    )
    if expected_limit != projected["configured_limit"]:
        return None
    if (
        projected["local_attempt"] > projected["configured_limit"]
        or projected["cumulative_producer_attempt"]
        > projected["configured_limit"]
    ):
        return None
    disposition = value["attempt_disposition"]
    if (
        not isinstance(disposition, str)
        or disposition not in _ATTEMPT_DISPOSITIONS
    ):
        return None
    projected["attempt_disposition"] = disposition
    return {
        "cognition_invocation_id": projected["cognition_invocation_id"],
        "graph_attempt": projected["graph_attempt"],
        "branch_id": projected["branch_id"],
        "producing_stage": projected["producing_stage"],
        "local_attempt": projected["local_attempt"],
        "cumulative_producer_attempt": projected[
            "cumulative_producer_attempt"
        ],
        "configured_limit": projected["configured_limit"],
        "epoch": projected["epoch"],
        "attempt_disposition": projected["attempt_disposition"],
    }


def _project_branch(value: object) -> dict[str, str] | None:
    """Project one exact bounded branch disposition."""

    if not isinstance(value, Mapping):
        return None
    if set(value) != _BRANCH_DISPOSITION_FIELDS:
        return None
    branch_id = _bounded_identifier(value["branch_id"])
    error_code = _bounded_identifier(value["error_code"])
    disposition = value["disposition"]
    if not branch_id or not isinstance(disposition, str):
        return None
    if disposition not in _ATTEMPT_DISPOSITIONS:
        return None
    return {
        "branch_id": branch_id,
        "disposition": disposition,
        "error_code": error_code,
    }


def _project_parent_recovery(value: object) -> dict[str, object]:
    """Project the parent-recovery sidecar without accepting arbitrary fields."""

    if not isinstance(value, Mapping):
        return {}
    if set(value) != _PARENT_RECOVERY_FIELDS:
        return {}
    disposition = value["disposition"]
    claimed_by = value["claimed_by"]
    epoch = value["epoch"]
    max_replays = value["max_replays"]
    checkpoint_sha256 = value["checkpoint_sha256"]
    if (
        not isinstance(disposition, str)
        or disposition not in _PARENT_RECOVERY_DISPOSITIONS
        or claimed_by != "parent_checkpoint"
        or epoch != 1
        or max_replays != 1
        or not _is_sha256_digest(checkpoint_sha256)
    ):
        return {}
    return {
        "disposition": disposition,
        "claimed_by": claimed_by,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "max_replays": max_replays,
    }
