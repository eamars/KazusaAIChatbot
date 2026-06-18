"""Local JSONL audit writer for privileged console actions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid

from pydantic import ValidationError

from control_console.contracts import ControlAuditEvent
from control_console.redaction import redact_mapping


class LocalAuditWriter:
    """Append and read sanitized control-console audit events."""

    def __init__(self, path: Path) -> None:
        """Create a JSONL audit writer."""

        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        *,
        event_type: str,
        operator_id: str,
        service_id: str = "",
        target: dict[str, Any] | None = None,
        previous_state: dict[str, Any] | None = None,
        new_state: dict[str, Any] | None = None,
        reason: str = "",
        request_id: str | None = None,
    ) -> ControlAuditEvent:
        """Append one sanitized audit event and return the model."""

        event = ControlAuditEvent(
            event_id=f"cc-audit-{uuid.uuid4().hex}",
            event_type=event_type,
            operator_id=operator_id,
            service_id=service_id,
            target=redact_mapping(target or {}),
            previous_state=redact_mapping(previous_state) if previous_state else None,
            new_state=redact_mapping(new_state) if new_state else None,
            reason=reason[:240],
            created_at=datetime.now(timezone.utc),
            request_id=request_id or f"cc-req-{uuid.uuid4().hex[:12]}",
        )
        event_json = event.model_dump_json()
        try:
            with self._path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(f"{event_json}\n")
        except OSError as exc:
            raise RuntimeError(f"cannot write control audit event: {exc}") from exc
        return event

    def read_recent(self, *, limit: int) -> list[ControlAuditEvent]:
        """Read recent audit events as model objects."""

        if not self._path.exists():
            events: list[ControlAuditEvent] = []
            return events

        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot read control audit events: {exc}") from exc

        selected_lines = lines[-limit:]
        events: list[ControlAuditEvent] = []
        for line in reversed(selected_lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                event = ControlAuditEvent.model_validate(payload)
            except (json.JSONDecodeError, ValidationError):
                continue
            events.append(event)
        return events
