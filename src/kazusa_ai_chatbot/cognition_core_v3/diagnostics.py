"""Protected, invocation-local diagnostics for the canonical four-stage flow."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CognitionChainDiagnosticsScope:
    run_id: str = ""
    chain_run_id: str = ""
    source_kind: str = "unknown"
    llm_trace_id: str = ""
    cognition_invocation_id: str = ""
    protected_records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, value: Mapping[str, Any]) -> None:
        self.protected_records.append(dict(value))


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
    scope = CognitionChainDiagnosticsScope(
        run_id=run_id.strip(),
        source_kind=source_kind.strip() or "unknown",
        llm_trace_id=llm_trace_id.strip(),
        cognition_invocation_id=cognition_invocation_id.strip(),
    )
    return _CURRENT_CHAIN_SCOPE.set(scope)


def current_chain_scope() -> CognitionChainDiagnosticsScope | None:
    return _CURRENT_CHAIN_SCOPE.get()


def configure_chain_scope(
    *,
    run_id: str | None = None,
    source_kind: str | None = None,
    llm_trace_id: str | None = None,
    cognition_invocation_id: str | None = None,
) -> None:
    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return
    if run_id and run_id.strip():
        scope.run_id = run_id.strip()
    if source_kind and source_kind.strip():
        scope.source_kind = source_kind.strip()
    if llm_trace_id and llm_trace_id.strip():
        scope.llm_trace_id = llm_trace_id.strip()
    if cognition_invocation_id and cognition_invocation_id.strip():
        scope.cognition_invocation_id = cognition_invocation_id.strip()


def record_protected_chain_record(record: Mapping[str, Any]) -> None:
    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is not None:
        scope.record(record)


def snapshot_protected_chain_records() -> tuple[dict[str, Any], ...]:
    scope = _CURRENT_CHAIN_SCOPE.get()
    if scope is None:
        return ()
    return tuple(dict(item) for item in scope.protected_records)


def reset_protected_chain_records(
    token: Token[CognitionChainDiagnosticsScope | None],
) -> None:
    _CURRENT_CHAIN_SCOPE.reset(token)


__all__ = [
    "CognitionChainDiagnosticsScope",
    "bind_protected_chain_records",
    "configure_chain_scope",
    "current_chain_scope",
    "record_protected_chain_record",
    "reset_protected_chain_records",
    "snapshot_protected_chain_records",
]
