"""Compatibility layer for generic action attempts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kazusa_ai_chatbot.db.self_cognition import (
    SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION,
    list_self_cognition_action_attempts,
    upsert_self_cognition_action_attempt,
)

ACTION_ATTEMPT_LEDGER_COLLECTION = SELF_COGNITION_ACTION_ATTEMPTS_COLLECTION


def build_action_idempotency_key(action_spec: dict[str, Any]) -> str:
    """Build a stable idempotency key from the semantic action identity."""

    payload = {
        "schema_version": action_spec.get("schema_version"),
        "kind": action_spec.get("kind"),
        "source_refs": _source_ref_identities(action_spec),
        "target": action_spec.get("target"),
        "params": action_spec.get("params"),
        "deadline": action_spec.get("deadline"),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return_value = f"action_spec:v1:{digest}"
    return return_value


def read_action_attempt_compat(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize old and new action-attempt rows for callers."""

    normalized = dict(row)
    normalized.setdefault("cognition_mode", None)
    normalized.setdefault("handler_owner", None)
    normalized.setdefault("validation_status", "legacy_unvalidated")
    normalized.setdefault("continuation_status", "legacy_not_recorded")
    normalized.setdefault("execution_result", None)
    normalized.setdefault("errors", [])
    return normalized


def build_action_attempt_record(
    action_spec: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    recorded_at: str,
    execution_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an additive action-attempt record for the existing ledger."""

    idempotency_key = eval_result.get("idempotency_key")
    if not isinstance(idempotency_key, str) or not idempotency_key:
        idempotency_key = build_action_idempotency_key(action_spec)
    attempt_id = f"action_attempt:{idempotency_key.removeprefix('action_spec:v1:')}"
    source_ref = _first_source_ref(action_spec)
    continuation_status = _continuation_status(action_spec)
    ok = eval_result.get("ok") is True
    validation_status = "accepted" if ok else "rejected"
    status = "candidate" if ok else "rejected"
    record = {
        "attempt_id": attempt_id,
        "run_id": "",
        "trigger_id": "",
        "source_kind": source_ref.get("ref_kind", ""),
        "source_id": source_ref.get("ref_id", ""),
        "target_scope": action_spec.get("target"),
        "action_kind": action_spec.get("kind"),
        "due_at": _due_at(action_spec),
        "idempotency_key": idempotency_key,
        "status": status,
        "dispatch_status": "",
        "scheduled_event_ids": [],
        "recorded_at": recorded_at,
        "action_spec_schema_version": action_spec.get("schema_version"),
        "cognition_mode": action_spec.get("cognition_mode"),
        "validation_status": validation_status,
        "handler_owner": eval_result.get("handler_owner"),
        "continuation_status": continuation_status,
        "execution_result": execution_result,
        "errors": list(eval_result.get("errors") or []),
    }
    return record


async def upsert_action_attempt(record: dict[str, Any]) -> None:
    """Persist an action-attempt record in the existing ledger collection."""

    await upsert_self_cognition_action_attempt(record)


async def list_action_attempts(*, limit: int = 1000) -> list[dict[str, Any]]:
    """Return recent action-attempt rows with compatibility defaults."""

    rows = await list_self_cognition_action_attempts(limit=limit)
    normalized_rows = [read_action_attempt_compat(row) for row in rows]
    return normalized_rows


def _source_ref_identities(action_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable source-ref identity fields for hashing."""

    source_refs = action_spec.get("source_refs")
    if not isinstance(source_refs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    identities = []
    for source_ref in source_refs:
        if not isinstance(source_ref, dict):
            continue
        identity = {
            "ref_kind": source_ref.get("ref_kind"),
            "ref_id": source_ref.get("ref_id"),
            "owner": source_ref.get("owner"),
            "relationship": source_ref.get("relationship"),
        }
        identities.append(identity)
    return identities


def _first_source_ref(action_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the first source reference or an empty mapping."""

    source_refs = action_spec.get("source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        return_value: dict[str, Any] = {}
        return return_value
    first = source_refs[0]
    if not isinstance(first, dict):
        return_value = {}
        return return_value
    return_value = first
    return return_value


def _due_at(action_spec: dict[str, Any]) -> str | None:
    """Read optional due-at metadata from params."""

    params = action_spec.get("params")
    if not isinstance(params, dict):
        return_value = None
        return return_value
    value = params.get("due_at")
    if not isinstance(value, str):
        return_value = None
        return return_value
    return_value = value
    return return_value


def _continuation_status(action_spec: dict[str, Any]) -> str:
    """Summarize the continuation request for ledger compatibility."""

    continuation = action_spec.get("continuation")
    if not isinstance(continuation, dict):
        return_value = "missing"
        return return_value
    mode = continuation.get("mode")
    if mode == "none":
        return_value = "none_requested"
        return return_value
    return_value = "requested_not_executed"
    return return_value
