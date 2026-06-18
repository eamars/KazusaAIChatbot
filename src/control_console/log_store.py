"""Bounded redacted process-log storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import json
import uuid

from pydantic import ValidationError

from control_console.contracts import ProcessLogLine
from control_console.redaction import redact_text


MAX_LOG_FILE_LINES = 2000


class ProcessLogStore:
    """Append and tail bounded local process logs."""

    def __init__(self, log_dir: Path) -> None:
        """Create a process log store."""

        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def append_line(
        self,
        *,
        service_id: str,
        stream: Literal["stdout", "stderr", "supervisor"],
        line: str,
    ) -> ProcessLogLine:
        """Append one redacted process-log line."""

        log_line = ProcessLogLine(
            service_id=service_id,
            stream=stream,
            line=redact_text(line),
            created_at=datetime.now(timezone.utc),
            cursor=f"log-{uuid.uuid4().hex}",
        )
        path = self._path_for(service_id)
        try:
            existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            existing.append(log_line.model_dump_json())
            bounded = existing[-MAX_LOG_FILE_LINES:]
            path.write_text("\n".join(bounded) + "\n", encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot write process log: {exc}") from exc
        return log_line

    def tail(self, *, service_id: str, limit: int) -> list[ProcessLogLine]:
        """Return the most recent redacted log lines for one service."""

        path = self._path_for(service_id)
        if not path.exists():
            lines: list[ProcessLogLine] = []
            return lines

        try:
            raw_lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"cannot read process log: {exc}") from exc

        selected_lines = raw_lines[-limit:]
        lines: list[ProcessLogLine] = []
        for line in selected_lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                log_line = ProcessLogLine.model_validate(payload)
            except (json.JSONDecodeError, ValidationError):
                continue
            lines.append(log_line)
        return lines

    def _path_for(self, service_id: str) -> Path:
        """Return the local log path for one service id."""

        safe_name = service_id.replace(".", "_")
        path = self._log_dir / f"{safe_name}.jsonl"
        return path
