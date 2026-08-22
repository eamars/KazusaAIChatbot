"""V3 public diagnostics projection and protected chain trace metadata.

The V3 public stage-trace record carries exactly the V2 validation-capture
stage field set so existing observability surfaces keep their contract, plus
two protected chain-scope fields: the registered chain name and the attempt
number within that stage's owner cap. Configuration identity is projected the
same way as in V2: route and generation settings only, never credentials.
Protected failure metadata crosses the boundary as closed typed values alone;
raw candidate text, validator prose, provider exception messages, and provider
metadata stay inside the harness-owned capture and never appear in protected
chain records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Protocol

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    StageFailure,
    StageResult,
)

# Exact V2 public stage-trace field set preserved by the V3 record projection.
STAGE_TRACE_PUBLIC_FIELDS = (
    "stage_id",
    "branch_id",
    "config",
    "system_prompt",
    "human_payload",
    "raw_output",
    "parsed_output",
    "parse_status",
    "started_at_monotonic",
    "ended_at_monotonic",
    "duration_ms",
    "error",
)

# Protected chain-scope fields added to the V2 public field set.
PROTECTED_CHAIN_FIELDS = (
    "chain_name",
    "attempt_number",
)

CONFIG_IDENTITY_FIELDS = (
    "stage_name",
    "route_name",
    "base_url",
    "model",
    "temperature",
    "top_p",
    "top_k",
    "max_completion_tokens",
    "presence_penalty",
    "timeout_seconds",
    "thinking_enabled",
)

PROTECTED_FAILURE_FIELDS = (
    "chain_name",
    "stage_name",
    "failure_class",
    "error_code",
    "repair_attempted",
)

CHAIN_STEP_FIELDS = (
    "step_id",
    "stage_kind",
    "lane_kind",
    "sidecar_stream_kind",
    "status",
    "attempt_count",
    "duration_ms",
    "queue_wait_ms",
    "in_flight_at_start",
    "prompt_chars",
    "new_suffix_chars",
    "estimated_prompt_tokens",
    "reserved_completion_tokens",
    "estimated_total_context_tokens",
    "active_total_ceiling_tokens",
    "extension_available",
    "extension_used",
    "estimated_new_suffix_tokens",
    "declared_shared_prefix_chars",
    "cache_class",
    "parse_status",
    "repair_count",
    "disposition",
    "warning_codes",
)

CHAIN_LEDGER_FIELDS = (
    "declared_context_window_tokens",
    "normal_total_ceiling_tokens",
    "extended_total_ceiling_tokens",
    "active_total_ceiling_tokens",
    "extension_available",
    "extension_used",
    "max_estimated_prompt_tokens",
    "max_reserved_completion_tokens",
    "max_estimated_total_context_tokens",
    "reanchor_used",
)

CHAIN_SIDECAR_FIELDS = (
    "l1_stream_count",
    "json_repair_call_count",
    "action_auth_attempt_count",
    "resolver_auth_attempt_count",
    "queue_wait_ms_total",
    "max_in_flight",
    "l1_preempted_by_repair",
    "cancellation_count",
)

CHAIN_LEDGER_DEFAULTS: dict[str, object] = {
    "declared_context_window_tokens": 0,
    "normal_total_ceiling_tokens": 0,
    "extended_total_ceiling_tokens": 0,
    "active_total_ceiling_tokens": 0,
    "extension_available": False,
    "extension_used": False,
    "max_estimated_prompt_tokens": 0,
    "max_reserved_completion_tokens": 0,
    "max_estimated_total_context_tokens": 0,
    "reanchor_used": False,
}

CHAIN_SIDECAR_DEFAULTS: dict[str, object] = {
    "l1_stream_count": 0,
    "json_repair_call_count": 0,
    "action_auth_attempt_count": 0,
    "resolver_auth_attempt_count": 0,
    "queue_wait_ms_total": 0,
    "max_in_flight": 0,
    "l1_preempted_by_repair": False,
    "cancellation_count": 0,
}

_MAX_CHAIN_STEPS = 96
_MAX_SESSION_EVENTS = 16
_MAX_DEGRADATION_MARKERS = 32
_MAX_WARNING_CODES = 32


class SidecarDiagnosticsProvider(Protocol):
    """Provide the lane-owned scalar sidecar counters."""

    def diagnostics(self) -> Mapping[str, object]:
        """Return the current invocation-local sidecar counters."""


@dataclass
class CognitionChainDiagnosticsScope:
    """Context-local producer state for one bounded V3 chain run.

    The scope deliberately separates protected transcript content from the
    sanitized step and aggregate projections.  Only the former is handed to
    the protected trace writer; event and database projections consume the
    closed typed fields below.
    """

    run_id: str = ""
    chain_run_id: str = ""
    source_kind: str = "unknown"
    llm_trace_id: str = ""
    cognition_invocation_id: str = ""
    started_at_utc: str = ""
    started_monotonic: float = 0.0
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, object]] = field(default_factory=list)
    session_events: list[str] = field(default_factory=list)
    degradation_markers: list[str] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)
    accepted_messages: tuple[tuple[str, str], ...] = ()
    token_ledger: dict[str, object] = field(default_factory=dict)
    sidecar: dict[str, object] = field(default_factory=dict)
    sidecar_provider: SidecarDiagnosticsProvider | None = None


_CURRENT_CHAIN_SCOPE: ContextVar[
    CognitionChainDiagnosticsScope | None
] = ContextVar("cognition_v3_chain_scope", default=None)


def bind_protected_chain_records(
    *,
    run_id: str = "",
    source_kind: str = "unknown",
    llm_trace_id: str = "",
    cognition_invocation_id: str = "",
) -> Token[CognitionChainDiagnosticsScope | None]:
    """Bind one typed V3 chain scope in the current execution context."""

    scope = CognitionChainDiagnosticsScope(
        run_id=run_id.strip(),
        source_kind=source_kind.strip() or "unknown",
        llm_trace_id=llm_trace_id.strip(),
        cognition_invocation_id=cognition_invocation_id.strip(),
    )
    scope_token = _CURRENT_CHAIN_SCOPE.set(scope)
    return scope_token


def current_chain_scope() -> CognitionChainDiagnosticsScope | None:
    """Return the active typed chain scope, when one is bound."""

    return _CURRENT_CHAIN_SCOPE.get()


def configure_chain_scope(
    *,
    run_id: str | None = None,
    source_kind: str | None = None,
    llm_trace_id: str | None = None,
    cognition_invocation_id: str | None = None,
) -> None:
    """Fill missing exact correlation values on the active chain scope."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    if run_id is not None and run_id.strip():
        scope.run_id = run_id.strip()
    if source_kind is not None and source_kind.strip():
        scope.source_kind = source_kind.strip()
    if llm_trace_id is not None and llm_trace_id.strip():
        scope.llm_trace_id = llm_trace_id.strip()
    if cognition_invocation_id is not None and cognition_invocation_id.strip():
        scope.cognition_invocation_id = cognition_invocation_id.strip()


def snapshot_protected_chain_records() -> tuple[dict[str, Any], ...]:
    """Return protected records from the active typed chain scope."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return ()
    return tuple(dict(record) for record in scope.protected_records)


def reset_protected_chain_records(
    token: Token[CognitionChainDiagnosticsScope | None],
) -> None:
    """Restore the chain scope that preceded ``token``."""

    _CURRENT_CHAIN_SCOPE.reset(token)


def record_protected_chain_record(record: Mapping[str, Any]) -> None:
    """Append one internal protected record without exposing it publicly."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is not None:
        scope.protected_records.append(dict(record))


def bind_chain_sidecar_state(
    provider: SidecarDiagnosticsProvider | None,
) -> None:
    """Attach the invocation-local sidecar counters to the active scope."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is not None:
        scope.sidecar_provider = provider


def record_current_sidecar_aggregate() -> None:
    """Refresh the sanitized sidecar projection from the active lane state."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None or scope.sidecar_provider is None:
        return
    record_sidecar_aggregate(scope.sidecar_provider.diagnostics())


def record_chain_step(record: Mapping[str, object]) -> None:
    """Append one exact sanitized chain step, bounded to 96 rows."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None or len(scope.steps) >= _MAX_CHAIN_STEPS:
        return
    defaults: dict[str, object] = {
        "step_id": "",
        "stage_kind": "",
        "lane_kind": "primary",
        "sidecar_stream_kind": "",
        "status": "skipped",
        "attempt_count": 0,
        "duration_ms": 0,
        "queue_wait_ms": 0,
        "in_flight_at_start": False,
        "prompt_chars": 0,
        "new_suffix_chars": 0,
        "estimated_prompt_tokens": 0,
        "reserved_completion_tokens": 0,
        "estimated_total_context_tokens": 0,
        "active_total_ceiling_tokens": 0,
        "extension_available": False,
        "extension_used": False,
        "estimated_new_suffix_tokens": 0,
        "declared_shared_prefix_chars": 0,
        "cache_class": "unknown",
        "parse_status": "not_run",
        "repair_count": 0,
        "disposition": "skipped",
        "warning_codes": [],
    }
    defaults.update({
        key: value
        for key, value in record.items()
        if key in CHAIN_STEP_FIELDS
    })
    warning_codes = defaults["warning_codes"]
    if not isinstance(warning_codes, Sequence) or isinstance(
        warning_codes,
        (str, bytes, bytearray),
    ):
        defaults["warning_codes"] = []
    else:
        defaults["warning_codes"] = [
            str(code)[:80]
            for code in warning_codes
            if isinstance(code, str) and code.strip()
        ][:8]
    step = {field_name: defaults[field_name] for field_name in CHAIN_STEP_FIELDS}
    scope.steps.append(step)


def record_registered_step(
    *,
    step_id: str,
    stage_kind: str,
    status: str,
    disposition: str,
    attempt_count: int = 0,
) -> None:
    """Record one deterministic or skipped registered stage marker."""

    record_chain_step({
        "step_id": step_id,
        "stage_kind": stage_kind,
        "status": status,
        "disposition": disposition,
        "attempt_count": attempt_count,
        "parse_status": "deterministic",
        "cache_class": "deterministic",
    })


def record_accepted_transcript(
    messages: Sequence[tuple[str, str]],
    *,
    system_content: str | None = None,
) -> None:
    """Store only accepted role/content pairs for protected capture."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    existing_system = next(
        (
            content
            for role, content in scope.accepted_messages
            if role == "system"
        ),
        "",
    )
    system_value = (
        system_content
        if isinstance(system_content, str) and system_content
        else existing_system
    )
    accepted_messages = tuple(
        (role, content)
        for role, content in messages
        if role in {"human", "assistant"}
        and isinstance(content, str)
    )
    scope.accepted_messages = (
        (("system", system_value),) if system_value else ()
    ) + accepted_messages


def record_chain_system_head(system_content: str) -> None:
    """Retain the static system head in the protected accepted transcript."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None or not isinstance(system_content, str) or not system_content:
        return
    accepted_messages = tuple(
        (role, content)
        for role, content in scope.accepted_messages
        if role != "system"
    )
    scope.accepted_messages = (("system", system_content),) + accepted_messages


def record_token_ledger(values: Mapping[str, object]) -> None:
    """Replace the bounded token aggregate with validated scalar values."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    projected = dict(scope.token_ledger)
    for key in CHAIN_LEDGER_FIELDS:
        value = values.get(key)
        if isinstance(value, (int, bool)) and not isinstance(value, float):
            projected[key] = (
                bool(value)
                if key in {"extension_available", "extension_used", "reanchor_used"}
                else int(value)
            )
    scope.token_ledger = {
        key: projected.get(key, default)
        for key, default in CHAIN_LEDGER_DEFAULTS.items()
    }


def record_sidecar_aggregate(values: Mapping[str, object]) -> None:
    """Replace the sidecar aggregate with the lane-owned counters."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    aliases = {
        "sidecar_queue_wait_ms_total": "queue_wait_ms_total",
        "sidecar_max_in_flight": "max_in_flight",
        "sidecar_cancellation_count": "cancellation_count",
    }
    projected = dict(scope.sidecar)
    for key, value in values.items():
        canonical_key = aliases.get(key, key)
        if (
            canonical_key in CHAIN_SIDECAR_FIELDS
            and isinstance(value, (int, bool))
        ):
            projected[canonical_key] = value
    scope.sidecar = {
        key: projected.get(key, default)
        for key, default in CHAIN_SIDECAR_DEFAULTS.items()
    }


def record_session_event(event: str) -> None:
    """Append one bounded session transition marker."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is not None and event.strip() and len(scope.session_events) < _MAX_SESSION_EVENTS:
        scope.session_events.append(event.strip()[:80])


def record_degradation_marker(marker: str) -> None:
    """Append one bounded accepted-degradation marker."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if (
        scope is not None
        and marker.strip()
        and len(scope.degradation_markers) < _MAX_DEGRADATION_MARKERS
        and marker.strip() not in scope.degradation_markers
    ):
        scope.degradation_markers.append(marker.strip()[:120])


def record_warning_codes(codes: Sequence[str]) -> None:
    """Append unique bounded warning codes from a validated output."""

    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    for code in codes:
        if (
            isinstance(code, str)
            and code.strip()
            and code.strip() not in scope.warning_codes
            and len(scope.warning_codes) < _MAX_WARNING_CODES
        ):
            scope.warning_codes.append(code.strip()[:120])


def project_config_identity(config: object) -> dict[str, object]:
    """Keep route identity and generation settings without exposing API keys.

    Args:
        config: An LLM call configuration owned by the stage boundary.

    Returns:
        The exact V2 config-identity projection; credential attributes are
        never read or named in the result.
    """
    thinking = getattr(config, "thinking", None)
    projected = {
        "stage_name": getattr(config, "stage_name", None),
        "route_name": getattr(config, "route_name", None),
        "base_url": getattr(config, "base_url", None),
        "model": getattr(config, "model", None),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "max_completion_tokens": getattr(config, "max_completion_tokens", None),
        "presence_penalty": getattr(config, "presence_penalty", None),
        "timeout_seconds": getattr(config, "timeout_seconds", None),
        "thinking_enabled": getattr(thinking, "enabled", None),
    }
    return projected


def build_chain_trace_record(
    *,
    chain_name: str,
    stage_id: str,
    config: object,
    system_prompt: str,
    human_payload: str,
    raw_output: str | None,
    parsed_output: object | None,
    parse_status: str,
    started_at: float,
    ended_at: float,
    branch_id: str | None = None,
    attempt_number: int = 1,
    error: str | None = None,
) -> dict[str, object]:
    """Build one protected chain trace record for a stage attempt.

    The public field set matches the V2 validation-capture stage record
    exactly; ``chain_name`` and ``attempt_number`` are the only additions.
    Configuration is projected through :func:`project_config_identity`, so no
    credential attribute ever enters the record.

    Args:
        chain_name: Registered chain owning this stage attempt.
        stage_id: Stable registered stage identity for the attempt.
        config: Stage-bound LLM configuration, projected without credentials.
        system_prompt: Static prompt supplied to the model.
        human_payload: Current-run dynamic prompt payload.
        raw_output: Normalized raw model output when invocation succeeded.
        parsed_output: Parser result before structural validation, if any.
        parse_status: Stage parse or validation status for evidence review.
        started_at: Monotonic stage start time.
        ended_at: Monotonic stage end time.
        branch_id: Optional activated goal branch identity.
        attempt_number: 1-based attempt position within the owner cap.
        error: Concrete failure text when the stage failed.

    Returns:
        The protected chain trace record with the exact public field set.
    """
    return {
        "chain_name": chain_name,
        "stage_id": stage_id,
        "branch_id": branch_id,
        "config": project_config_identity(config),
        "system_prompt": system_prompt,
        "human_payload": human_payload,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "parse_status": parse_status,
        "started_at_monotonic": started_at,
        "ended_at_monotonic": ended_at,
        "duration_ms": max(0, int((ended_at - started_at) * 1000)),
        "attempt_number": attempt_number,
        "error": error,
    }


def project_protected_chain_failure(failure: StageFailure) -> dict[str, object]:
    """Project one typed stage failure into protected chain metadata.

    Only closed typed fields cross the protection boundary: chain and stage
    identity, the bounded failure class, the exact error code, and the repair
    disposition. Raw candidate text, validator prose, provider exception
    messages, and provider metadata never appear in this projection; raw
    evidence stays inside the harness-owned capture.

    Args:
        failure: The typed stage failure record from a bounded stage attempt.

    Returns:
        The protected failure metadata with exactly its closed field set.
    """
    return {
        "chain_name": failure.chain_name,
        "stage_name": failure.stage_name,
        "failure_class": failure.failure_class,
        "error_code": failure.error_code,
        "repair_attempted": failure.repair_attempted,
    }


def project_protected_chain_result(result: StageResult) -> dict[str, object]:
    """Project one stage result into protected chain metadata.

    The projection carries the acceptance outcome for every stage and adds the
    typed failure fields only when a bounded attempt exhausted or terminated;
    it never carries local-state payloads, semantic summaries, raw output, or
    provider metadata.

    Args:
        result: The bounded stage execution result from the chain executor.

    Returns:
        The protected result metadata for observability surfaces.
    """
    record: dict[str, object] = {
        "chain_name": result.chain_name,
        "stage_name": result.stage_name,
        "accepted": result.accepted,
    }
    if result.failure is not None:
        record["failure"] = project_protected_chain_failure(result.failure)
    return record
