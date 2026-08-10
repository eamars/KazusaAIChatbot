"""Durable target adapter for one character operational carry-over update."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from uuid import uuid4

from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CharacterCarryoverServicesV1,
    _reduce_apply_decision,
    run_character_carryover_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_character_elapsed_decay,
)
from kazusa_ai_chatbot.consolidation.source_policy import (
    validate_character_operational_sources,
)
from kazusa_ai_chatbot.db.character import get_character_cognition_state
from kazusa_ai_chatbot.db.post_turn_lifecycle import (
    build_character_operational_lifecycle_record,
    claim_character_operational_receipt,
    commit_character_operational_update,
    complete_character_operational_receipt,
)
from kazusa_ai_chatbot.db.schemas import (
    CharacterOperationalClaimV1,
    CharacterOperationalReceiptV1,
)
from kazusa_ai_chatbot.llm_tracing.failure_capsule import mark_current_failure
from kazusa_ai_chatbot.time_boundary import (
    parse_storage_utc_datetime,
    storage_utc_now,
    storage_utc_now_iso,
)


CHARACTER_OPERATIONAL_LEASE_SECONDS = 45

logger = logging.getLogger(__name__)


class _OperationalReceipt(dict[str, Any]):
    """Keep the exact receipt mapping usable by existing attribute callers."""

    @property
    def status(self) -> str:
        """Return the terminal status without changing the mapping contract."""

        return str(self["status"])


@dataclass(frozen=True)
class CharacterOperationalExecutionContext:
    """A claimed durable receipt and its immutable optimistic base."""

    claim: CharacterOperationalClaimV1
    base_state: Mapping[str, Any] | None
    lease_owner: str
    registered_at: str


async def prepare_character_operational_target(
    *,
    source_episode_id: str,
    sequence: int,
    effective_at: str,
    delivery_tracking_id: str = "",
    created_at: str | None = None,
) -> CharacterOperationalExecutionContext:
    """Claim an episode receipt before the response can become observable."""

    _require_timestamp(effective_at, "effective_at")
    registered_at = storage_utc_now_iso()
    lifecycle_created_at = created_at or effective_at
    _require_timestamp(lifecycle_created_at, "created_at")
    lease_owner = uuid4().hex
    try:
        base_state = await _await_if_needed(get_character_cognition_state())
        if not isinstance(base_state, Mapping):
            raise ValueError("character cognition state is not a mapping")
        lifecycle_record = build_character_operational_lifecycle_record(
            source_episode_id=source_episode_id,
            created_at=lifecycle_created_at,
            delivery_tracking_id=delivery_tracking_id,
        )
        claim = await claim_character_operational_receipt(
            lifecycle_record=lifecycle_record,
            sequence=sequence,
            base_updated_at=_require_timestamp(
                base_state.get("updated_at"),
                "character state updated_at",
            ),
            registered_at=registered_at,
            lease_owner=lease_owner,
            lease_expires_at=_lease_expires_at(registered_at),
        )
    except Exception as exc:
        logger.error(
            f"character operational claim failed: "
            f"error_code=persistence_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="operational_claim_failed",
            stage_name="character_operational_claim",
            details={"error_code": "persistence_failed"},
            exception=exc,
        )
        claim = {
            "claim_status": "terminal",
            "receipt": _in_memory_failure_receipt(
                source_episode_id=source_episode_id,
                sequence=sequence,
                registered_at=registered_at,
                error_code="persistence_failed",
            ),
        }
        base_state = None
    return CharacterOperationalExecutionContext(
        claim=claim,
        base_state=base_state,
        lease_owner=lease_owner,
        registered_at=registered_at,
    )


async def run_character_operational_target(
    *,
    source_episode_id: str,
    sequence: int,
    evidence: Sequence[Mapping[str, Any]],
    effective_at: str,
    services: CharacterCarryoverServicesV1,
    execution_context: CharacterOperationalExecutionContext | None = None,
) -> CharacterOperationalReceiptV1:
    """Run at most one durable source-free carry-over update for an episode."""

    context = execution_context
    if context is None:
        context = await prepare_character_operational_target(
            source_episode_id=source_episode_id,
            sequence=sequence,
            effective_at=effective_at,
    )
    if context.claim["claim_status"] != "claimed":
        return _receipt(context.claim["receipt"])
    if context.base_state is None:
        return _in_memory_failure_receipt(
            source_episode_id=source_episode_id,
            sequence=sequence,
            registered_at=context.registered_at,
            error_code="persistence_failed",
        )
    remaining_seconds = _remaining_lease_seconds(context)
    if remaining_seconds <= 0:
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="timed_out",
            error_code="deadline_exceeded",
            attempt_count=0,
        )

    try:
        validated_sources = validate_character_operational_sources(
            [dict(row) for row in evidence],
        )
    except ValueError:
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code="source_policy_rejected",
            attempt_count=0,
        )

    try:
        base_state = _effective_character_state(
            context.base_state,
            effective_at=effective_at,
        )
        carryover_result = await asyncio.wait_for(
            run_character_carryover_cognition(
                source_episode_id=source_episode_id,
                evidence=_carryover_evidence(validated_sources),
                base_state=base_state,
                effective_at=effective_at,
                services=services,
            ),
            timeout=remaining_seconds,
        )
    except asyncio.TimeoutError:
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="timed_out",
            error_code="deadline_exceeded",
            attempt_count=0,
        )
    except Exception as exc:
        logger.error(
            f"character operational carry-over execution failed: "
            f"error_code=transaction_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="carryover_execution_error",
            stage_name="character_operational_carryover",
            details={"error_code": "transaction_failed"},
            exception=exc,
        )
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code="transaction_failed",
            attempt_count=0,
        )
    attempt_count = int(getattr(carryover_result, "attempts", 0))
    if carryover_result.disposition == "no_change":
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="no_change",
            error_code=None,
            attempt_count=attempt_count,
        )
    if carryover_result.disposition == "degraded":
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="no_change",
            error_code=(
                getattr(carryover_result, "error_code", None)
                or "state_rejected"
            ),
            attempt_count=attempt_count,
        )
    if (
        carryover_result.disposition != "apply"
        or carryover_result.state_update is None
    ):
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code=(
                getattr(carryover_result, "error_code", None)
                or "state_rejected"
            ),
            attempt_count=attempt_count,
        )

    committed = await _commit_or_conflict(
        source_episode_id=source_episode_id,
        lease_owner=context.lease_owner,
        base_state=base_state,
        replacement=carryover_result.state_update["replacement_state"],
        completed_at=storage_utc_now_iso(),
    )
    if isinstance(committed, Mapping):
        return _receipt(committed)
    if committed != "version_conflict":
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code=committed,
            attempt_count=attempt_count,
        )

    reloaded_base = await _reload_effective_character_state(effective_at)
    if reloaded_base is None:
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code="version_conflict",
            attempt_count=attempt_count,
        )
    semantic_appraisal = carryover_result.decision.semantic_appraisal
    if not isinstance(semantic_appraisal, Mapping):
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code="state_rejected",
            attempt_count=attempt_count,
        )
    reapplied = _reduce_apply_decision(
        base_state=reloaded_base,
        effective_at=effective_at,
        decision_payload={"semantic_appraisal": semantic_appraisal},
        evidence=_carryover_evidence(validated_sources),
    )
    if reapplied is None:
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code="state_rejected",
            attempt_count=attempt_count,
        )
    committed = await _commit_or_conflict(
        source_episode_id=source_episode_id,
        lease_owner=context.lease_owner,
        base_state=reloaded_base,
        replacement=reapplied["replacement_state"],
        completed_at=storage_utc_now_iso(),
    )
    if isinstance(committed, Mapping):
        return _receipt(committed)
    if committed != "version_conflict":
        return await _complete_or_in_memory_failure(
            source_episode_id=source_episode_id,
            sequence=sequence,
            lease_owner=context.lease_owner,
            registered_at=context.registered_at,
            status="failed",
            error_code=committed,
            attempt_count=attempt_count,
        )
    return await _complete_or_in_memory_failure(
        source_episode_id=source_episode_id,
        sequence=sequence,
        lease_owner=context.lease_owner,
        registered_at=context.registered_at,
        status="failed",
        error_code="version_conflict",
        attempt_count=attempt_count,
    )


async def _commit_or_conflict(
    *,
    source_episode_id: str,
    lease_owner: str,
    base_state: Mapping[str, Any],
    replacement: Mapping[str, Any],
    completed_at: str,
) -> CharacterOperationalReceiptV1 | Literal[
    "version_conflict",
    "state_rejected",
    "transaction_failed",
]:
    """Commit one validated replacement or return its optimistic conflict."""

    try:
        return await commit_character_operational_update(
            source_episode_id=source_episode_id,
            lease_owner=lease_owner,
            expected_updated_at=_require_timestamp(
                base_state.get("updated_at"),
                "character state updated_at",
            ),
            replacement=replacement,
            completed_at=_require_timestamp(completed_at, "completed_at"),
        )
    except ValueError as exc:
        if "base version is stale" in str(exc):
            return "version_conflict"
        if "updated_at" in str(exc):
            return "state_rejected"
        logger.error(
            f"character operational commit failed: "
            f"error_code=transaction_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="operational_commit_failed",
            stage_name="character_operational_carryover",
            details={"error_code": "transaction_failed"},
            exception=exc,
        )
        return "transaction_failed"
    except RuntimeError as exc:
        logger.error(
            f"character operational commit failed: "
            f"error_code=transaction_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="operational_commit_failed",
            stage_name="character_operational_carryover",
            details={"error_code": "transaction_failed"},
            exception=exc,
        )
        return "transaction_failed"


async def _reload_effective_character_state(
    effective_at: str,
) -> Mapping[str, Any] | None:
    """Load the latest valid base for the one allowed stale-proposal reapply."""

    try:
        state = await _await_if_needed(get_character_cognition_state())
        if not isinstance(state, Mapping):
            return None
        return _effective_character_state(state, effective_at=effective_at)
    except Exception as exc:
        logger.error(
            f"character operational state reload failed: "
            f"error_code=state_reload_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="operational_reload_failed",
            stage_name="character_operational_carryover",
            details={"error_code": "state_reload_failed"},
            exception=exc,
        )
        return None


def _effective_character_state(
    state: Mapping[str, Any],
    *,
    effective_at: str,
) -> Mapping[str, Any]:
    """Apply pure elapsed fading once before a semantic character write."""

    base_updated_at = _require_timestamp(
        state.get("updated_at"),
        "character state updated_at",
    )
    elapsed_seconds = max(
        0,
        int(
            (
                parse_storage_utc_datetime(effective_at)
                - parse_storage_utc_datetime(base_updated_at)
            ).total_seconds()
        ),
    )
    return apply_character_elapsed_decay(
        state,
        elapsed_seconds=elapsed_seconds,
    )


async def _complete_or_in_memory_failure(
    *,
    source_episode_id: str,
    sequence: int,
    lease_owner: str,
    registered_at: str,
    status: Literal["no_change", "failed", "timed_out"],
    error_code: str | None,
    attempt_count: int,
) -> CharacterOperationalReceiptV1:
    """Terminalize the claim or return a bounded degraded receipt on failure."""

    try:
        receipt = await complete_character_operational_receipt(
            source_episode_id=source_episode_id,
            lease_owner=lease_owner,
            status=status,
            completed_at=storage_utc_now_iso(),
            error_code=error_code,
            attempt_count=attempt_count,
        )
        return _receipt(receipt)
    except Exception as exc:
        logger.error(
            f"character operational completion failed: "
            f"error_code=persistence_failed "
            f"exception_type={exc.__class__.__name__}"
        )
        mark_current_failure(
            failure_kind="operational_completion_failed",
            stage_name="character_operational_carryover",
            details={"error_code": "persistence_failed"},
            exception=exc,
        )
        return _in_memory_failure_receipt(
            source_episode_id=source_episode_id,
            sequence=sequence,
            registered_at=registered_at,
            error_code="persistence_failed",
            attempt_count=attempt_count,
        )


def _carryover_evidence(
    source_views: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Map trusted router source views into opaque carry-over evidence rows."""

    return [
        {
            "evidence_handle": f"evidence:{source_view['source_key']}",
            "source_kind": source_view["source_kind"],
            "source_id": source_view["source_id"],
            "occurred_at": source_view["occurred_at"],
            "semantic_summary": "character operational event",
            "semantic_text": source_view["semantic_text"],
        }
        for source_view in source_views
    ]


def _lease_expires_at(registered_at: str) -> str:
    """Return the fixed bounded receipt lease expiry in storage UTC."""

    return (
        parse_storage_utc_datetime(registered_at)
        + timedelta(seconds=CHARACTER_OPERATIONAL_LEASE_SECONDS)
    ).isoformat()


def _remaining_lease_seconds(
    context: CharacterOperationalExecutionContext,
) -> float:
    """Return the one absolute receipt deadline remaining for all work."""

    receipt = context.claim["receipt"]
    lease_expires_at = receipt.get("lease_expires_at")
    if not isinstance(lease_expires_at, str):
        return 0.0
    return max(
        0.0,
        (
            parse_storage_utc_datetime(lease_expires_at)
            - storage_utc_now()
        ).total_seconds(),
    )


def _in_memory_failure_receipt(
    *,
    source_episode_id: str,
    sequence: int,
    registered_at: str,
    error_code: str,
    attempt_count: int = 0,
) -> CharacterOperationalReceiptV1:
    """Build the explicit non-durable failure returned after claim failure."""

    return _receipt({
        "schema_version": "character_operational_receipt.v1",
        "source_episode_id": source_episode_id,
        "status": "failed",
        "sequence": sequence,
        "durable": False,
        "base_updated_at": registered_at,
        "committed_updated_at": "",
        "registered_at": registered_at,
        "completed_at": registered_at,
        "lease_owner": "",
        "lease_expires_at": "",
        "attempt_count": attempt_count,
        "error_code": error_code,
    })


def _receipt(value: Mapping[str, Any]) -> CharacterOperationalReceiptV1:
    """Copy one exact durable receipt into the public adapter projection."""

    return _OperationalReceipt(value)


async def _await_if_needed(value: Any) -> Any:
    """Await repository functions while retaining direct deterministic seams."""

    if inspect.isawaitable(value):
        return await value
    return value


def _require_timestamp(value: object, label: str) -> str:
    """Validate a storage UTC timestamp at the target boundary."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    try:
        parse_storage_utc_datetime(value)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return value


__all__ = [
    "CharacterOperationalExecutionContext",
    "prepare_character_operational_target",
    "run_character_operational_target",
]
