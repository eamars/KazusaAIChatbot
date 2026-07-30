"""Failure-only protected capture for Cognition Core V2 invocations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from typing import Literal
from uuid import uuid4

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.config import DEBUG_LOG_TTL_DAYS, LLM_TRACE_CAPTURE_MODE
from kazusa_ai_chatbot.db import llm_tracing as db_llm_tracing
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


logger = logging.getLogger(__name__)

CapsuleOutcome = Literal["partial_failure", "terminal_failure"]
FAILURE_CAPSULE_WRITE_TIMEOUT_SECONDS = 0.25


@dataclass
class FailureCapsuleSession:
    """Hold exact evidence for one context-local cognition invocation."""

    trace_id: str
    cognition_invocation_id: str
    entrypoint: str
    input_payload: object
    input_sha256: str
    attempts: list[dict[str, object]] = field(default_factory=list)
    failure_events: list[dict[str, object]] = field(default_factory=list)
    secret_values: set[str] = field(default_factory=set, repr=False)
    has_failed_attempt: bool = False
    finished: bool = False
    context_token: Token[FailureCapsuleSession | None] | None = field(
        default=None,
        repr=False,
    )


_CURRENT_SESSION: ContextVar[FailureCapsuleSession | None] = ContextVar(
    "cognition_v2_failure_capsule",
    default=None,
)
_PENDING_PERSISTENCE_TASKS: set[asyncio.Task[None]] = set()


def begin_failure_capsule(
    *,
    trace_id: str,
    entrypoint: str,
    input_payload: object,
) -> FailureCapsuleSession | None:
    """Bind an invocation-local exact-input buffer when tracing is enabled.

    Args:
        trace_id: Protected turn trace that owns this invocation.
        entrypoint: Public Cognition V2 entrypoint being executed.
        input_payload: Raw public arguments captured before validation.

    Returns:
        The bound session, or ``None`` when protected tracing is disabled or
        no turn trace is available.
    """

    if LLM_TRACE_CAPTURE_MODE == "off" or not trace_id:
        return None

    try:
        payload_snapshot = _snapshot_value(input_payload)
        serialized_input = json.dumps(
            payload_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception as exc:
        _warn_capture_failure("setup", exc)
        return None

    input_sha256 = hashlib.sha256(
        serialized_input.encode("utf-8")
    ).hexdigest()
    session = FailureCapsuleSession(
        trace_id=trace_id,
        cognition_invocation_id=uuid4().hex,
        entrypoint=entrypoint,
        input_payload=payload_snapshot,
        input_sha256=input_sha256,
    )
    try:
        context_token = _CURRENT_SESSION.set(session)
    except Exception as exc:
        _warn_capture_failure("binding", exc)
        return None
    session.context_token = context_token
    return session


def append_model_attempt(
    *,
    stage_name: str,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    config: LLMCallConfig | None = None,
    route_name: str = "",
    model_name: str = "",
    branch_id: str = "",
    attempt_index: int = 0,
    validation_error: str = "",
    started_at: float | None = None,
) -> None:
    """Append one exact model attempt to the active protected session.

    Args:
        stage_name: Stable semantic stage identity.
        messages: Exact request messages supplied to the model.
        response_text: Exact normalized response, or empty text on provider
            failure.
        parsed_output: Parser output available before contract evaluation.
        parse_status: Parser or contract disposition.
        status: Final attempt status.
        config: Full call configuration projected without its API key.
        route_name: Route identity used when no call config is available.
        model_name: Model identity used when no call config is available.
        branch_id: Optional semantic branch or question identity.
        attempt_index: One-based attempt number when the owner exposes it.
        validation_error: Concrete provider or contract failure text.
        started_at: Monotonic start used only to retain real call order.
    """

    session = _CURRENT_SESSION.get()
    if session is None or session.finished:
        return

    try:
        secret_values: set[str] = set()
        api_key = getattr(config, "api_key", "")
        if isinstance(api_key, str) and api_key:
            secret_values.add(api_key)
            session.secret_values.add(api_key)
        resolved_attempt_index = attempt_index
        if resolved_attempt_index <= 0:
            resolved_attempt_index = 1 + sum(
                1
                for attempt_record in session.attempts
                if (
                    attempt_record["stage_name"] == stage_name
                    and attempt_record["branch_id"] == branch_id
                )
            )
        attempt = {
            "stage_name": stage_name,
            "branch_id": branch_id,
            "attempt_index": resolved_attempt_index,
            "config": _project_call_config(
                config,
                route_name=route_name,
                model_name=model_name,
            ),
            "messages": [
                {
                    "role": _message_role(message),
                    "content": _redact_text(
                        _message_content(message),
                        secret_values,
                    ),
                }
                for message in messages
            ],
            "raw_response_text": _redact_text(
                response_text,
                secret_values,
            ),
            "parsed_output": _redact_value(
                _snapshot_value(parsed_output),
                secret_values,
            ),
            "parse_status": parse_status,
            "validation_error": _redact_text(
                validation_error,
                secret_values,
            ),
            "status": status,
            "_started_at": started_at,
        }
    except Exception as exc:
        _warn_capture_failure("attempt snapshot", exc)
        return

    session.attempts.append(attempt)
    if status != "succeeded" or parse_status != "succeeded":
        session.has_failed_attempt = True


def append_json_repair_attempt(
    *,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    config: LLMCallConfig,
    validation_error: str,
    started_at: float,
) -> None:
    """Record one canonical JSON-repair call in the active capsule."""

    mark_current_failure(
        failure_kind="recovered_json_repair",
        stage_name="json_repair",
        details={},
    )
    append_model_attempt(
        stage_name="json_repair",
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        config=config,
        branch_id="json_repair",
        validation_error=validation_error,
        started_at=started_at,
    )


def mark_failure(
    session: FailureCapsuleSession | None,
    *,
    failure_kind: str,
    stage_name: str,
    details: Mapping[str, object],
) -> None:
    """Attach one producer-owned partial or terminal failure disposition.

    Args:
        session: Invocation session returned by ``begin_failure_capsule``.
        failure_kind: Stable failure category owned by the producing stage.
        stage_name: Stage or public entrypoint that observed the failure.
        details: Protected structured detail needed for replay and diagnosis.
    """

    if session is None or session.finished:
        return
    try:
        failure_event = {
            "failure_kind": failure_kind,
            "stage_name": stage_name,
            "details": _snapshot_value(details),
        }
    except Exception as exc:
        _warn_capture_failure("failure snapshot", exc)
        return
    session.failure_events.append(failure_event)


def mark_current_failure(
    *,
    failure_kind: str,
    stage_name: str,
    details: Mapping[str, object],
) -> None:
    """Attach a failure disposition to the active invocation, if any."""

    session = _CURRENT_SESSION.get()
    mark_failure(
        session,
        failure_kind=failure_kind,
        stage_name=stage_name,
        details=details,
    )


def finish_failure_capsule(
    session: FailureCapsuleSession | None,
    *,
    outcome: CapsuleOutcome | None,
    exception: BaseException | None = None,
) -> str:
    """Discard a clean session or schedule one protected failure row.

    Args:
        session: Invocation session returned by ``begin_failure_capsule``.
        outcome: Explicit failure outcome, or ``None`` to infer partial
            promotion from buffered failed attempts and failure events.
        exception: Original terminal exception retained as protected evidence.

    Returns:
        The persisted invocation identity, or an empty string when the session
        was disabled, clean, or unavailable.
    """

    if session is None:
        return ""
    if session.finished:
        return session.cognition_invocation_id

    session.finished = True
    _restore_prior_session(session)
    effective_outcome = outcome
    if effective_outcome is None and (
        session.has_failed_attempt or session.failure_events
    ):
        effective_outcome = "partial_failure"
    if effective_outcome is None:
        return ""

    try:
        attempts = _ordered_attempts(session.attempts)
        exception_payload = None
        if exception is not None:
            exception_payload = {
                "type": exception.__class__.__name__,
                "message": _redact_text(
                    str(exception),
                    session.secret_values,
                ),
            }
        capsule = {
            "schema_version": "cognition_failure_capsule.v1",
            "trace_id": session.trace_id,
            "cognition_invocation_id": session.cognition_invocation_id,
            "entrypoint": session.entrypoint,
            "input_payload": session.input_payload,
            "input_sha256": session.input_sha256,
            "attempts": attempts,
            "failure_events": list(session.failure_events),
            "outcome": effective_outcome,
            "exception": exception_payload,
        }
        document = _capsule_document(capsule)
    except Exception as exc:
        _warn_capture_failure("finalization", exc)
        return session.cognition_invocation_id

    _schedule_persistence(document)
    return session.cognition_invocation_id


def _restore_prior_session(session: FailureCapsuleSession) -> None:
    """Restore the ContextVar state that preceded one invocation."""

    context_token = session.context_token
    if context_token is None:
        return
    try:
        _CURRENT_SESSION.reset(context_token)
    except Exception as exc:
        _warn_capture_failure("context reset", exc)


def _ordered_attempts(
    attempts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return copied attempts in real model-call start order."""

    indexed_attempts = list(enumerate(attempts))
    indexed_attempts.sort(
        key=lambda indexed: (
            indexed[1].get("_started_at") is None,
            indexed[1].get("_started_at")
            if indexed[1].get("_started_at") is not None
            else indexed[0],
            indexed[0],
        )
    )
    ordered: list[dict[str, object]] = []
    for _, attempt in indexed_attempts:
        persisted_attempt = {
            key: value
            for key, value in attempt.items()
            if key != "_started_at"
        }
        ordered.append(persisted_attempt)
    return ordered


def _capsule_document(capsule: Mapping[str, object]) -> dict[str, object]:
    """Build one protected trace-step row using shared trace retention."""

    trace_id = str(capsule["trace_id"])
    invocation_id = str(capsule["cognition_invocation_id"])
    created_at = storage_utc_now_iso()
    document = {
        "step_id": f"{trace_id}_cognition_capsule_{invocation_id}",
        "trace_id": trace_id,
        "sequence": 0,
        "stage_name": "cognition_failure_capsule",
        "capture_reason": "cognition_failure_capsule",
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
    """Schedule protected persistence without awaiting the response path."""

    persistence_coroutine = _persist_capsule(document)
    try:
        task = asyncio.create_task(persistence_coroutine)
    except Exception as exc:
        persistence_coroutine.close()
        _warn_capture_failure("task scheduling", exc)
        return
    try:
        _PENDING_PERSISTENCE_TASKS.add(task)
        task.add_done_callback(_observe_persistence_task)
    except Exception as exc:
        _PENDING_PERSISTENCE_TASKS.discard(task)
        _warn_capture_failure("task observation", exc)


async def _persist_capsule(document: Mapping[str, object]) -> None:
    """Write one protected row while containing storage-side failures."""

    try:
        await asyncio.wait_for(
            db_llm_tracing.insert_trace_step(document),
            timeout=FAILURE_CAPSULE_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        _warn_capture_failure("persistence", exc)


def _observe_persistence_task(task: asyncio.Task[None]) -> None:
    """Release one background task and observe unexpected task termination."""

    _PENDING_PERSISTENCE_TASKS.discard(task)
    if task.cancelled():
        logger.warning("Cognition failure capsule persistence was cancelled")
        return
    try:
        task.result()
    except Exception as exc:
        _warn_capture_failure("task completion", exc)


def _project_call_config(
    config: LLMCallConfig | None,
    *,
    route_name: str,
    model_name: str,
) -> dict[str, object]:
    """Project generation settings while excluding provider credentials."""

    if config is None:
        projected = {
            "route_name": route_name,
            "base_url": "",
            "model": model_name,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_completion_tokens": None,
            "presence_penalty": None,
            "timeout_seconds": None,
            "thinking_enabled": False,
        }
        return projected

    projected = {
        "route_name": config.route_name,
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_completion_tokens": config.max_completion_tokens,
        "presence_penalty": config.presence_penalty,
        "timeout_seconds": config.timeout_seconds,
        "thinking_enabled": config.thinking.enabled,
    }
    return projected


def _message_role(message: BaseMessage) -> str:
    """Return the stable LangChain role label for one exact prompt message."""

    role = getattr(message, "type", "")
    if isinstance(role, str) and role.strip():
        message_role = role.strip()
    else:
        message_role = message.__class__.__name__
    return message_role


def _message_content(message: BaseMessage) -> str:
    """Return text content for one exact prompt message."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        message_content = content
    else:
        message_content = str(content)
    return message_content


def _snapshot_value(value: object) -> object:
    """Copy an arbitrary bounded value into deterministic JSON/BSON-safe data."""

    copied_value = deepcopy(value)
    snapshot = _json_safe_value(copied_value)
    return snapshot


def _json_safe_value(value: object) -> object:
    """Project copied values into protected JSON/BSON-compatible structures."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        projected_datetime = value.isoformat()
        return projected_datetime
    if is_dataclass(value):
        dataclass_value = asdict(value)
        projected_dataclass = _json_safe_value(dataclass_value)
        return projected_dataclass
    if isinstance(value, Mapping):
        projected = {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
        return projected
    if isinstance(value, (list, tuple)):
        projected = [_json_safe_value(item) for item in value]
        return projected
    if isinstance(value, (set, frozenset)):
        projected = [_json_safe_value(item) for item in value]
        projected.sort(key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ))
        return projected
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped_value = model_dump(mode="json")
        projected_dump = _json_safe_value(dumped_value)
        return projected_dump
    projected_value = str(value)
    return projected_value


def _redact_value(value: object, secret_values: set[str]) -> object:
    """Remove known route credentials from protected attempt evidence."""

    if isinstance(value, str):
        redacted_text = _redact_text(value, secret_values)
        return redacted_text
    if isinstance(value, dict):
        projected = {
            key: _redact_value(item, secret_values)
            for key, item in value.items()
        }
        return projected
    if isinstance(value, list):
        projected = [
            _redact_value(item, secret_values)
            for item in value
        ]
        return projected
    return value


def _redact_text(value: str, secret_values: set[str]) -> str:
    """Replace configured route credentials without changing other evidence."""

    redacted_value = value
    for secret_value in secret_values:
        if secret_value:
            redacted_value = redacted_value.replace(
                secret_value,
                "[REDACTED_API_KEY]",
            )
    return redacted_value


def _warn_capture_failure(operation: str, exc: BaseException) -> None:
    """Report capture unavailability without exposing protected exception text."""

    logger.warning(
        f"Cognition failure capsule {operation} failed: "
        f"{exc.__class__.__name__}"
    )
