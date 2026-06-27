"""Durable diagnostic event tracing for code-writing workflow runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

DIAGNOSTIC_ROOT_NAME = "writing_diagnostics"
DIAGNOSTIC_EVENTS_NAME = "events.jsonl"


class WritingDiagnosticTracer:
    """Append public-safe workflow timing events to a request workspace."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        session_id: str,
    ) -> None:
        self._path: Path | None = None
        self._counter = 0
        self._starts: dict[str, float] = {}

        root = Path(workspace_root).expanduser().resolve(strict=False)
        try:
            trace_dir = ensure_path_inside(
                root / DIAGNOSTIC_ROOT_NAME / session_id,
                root,
            )
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._path = ensure_path_inside(
                trace_dir / DIAGNOSTIC_EVENTS_NAME,
                root,
            )
        except (OSError, PathSafetyError):
            self._path = None

    @property
    def path(self) -> str:
        """Return the trace path for review, or an empty string if disabled."""

        if self._path is None:
            return ""
        return str(self._path)

    def event(
        self,
        *,
        stage: str,
        event: str,
        **fields: object,
    ) -> None:
        """Append one diagnostic event without changing workflow behavior."""

        payload = self._event_payload(
            stage=stage,
            event=event,
            fields=fields,
        )
        self._append(payload)

    def start(self, *, stage: str, **fields: object) -> str:
        """Record a stage start and return the call id for the matching end."""

        call_id = self._next_call_id(stage)
        self._starts[call_id] = time.perf_counter()
        payload_fields = {"call_id": call_id, **fields}
        self.event(stage=stage, event="start", **payload_fields)
        return call_id

    def end(self, *, stage: str, call_id: str, **fields: object) -> None:
        """Record a stage end with elapsed time when a start event exists."""

        started_at = self._starts.pop(call_id, None)
        payload_fields = {"call_id": call_id, **fields}
        if started_at is not None:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            payload_fields["duration_ms"] = elapsed_ms
        self.event(stage=stage, event="end", **payload_fields)

    def _next_call_id(self, stage: str) -> str:
        self._counter += 1
        compact_stage = _compact_stage(stage)
        call_id = f"{self._counter:04d}-{compact_stage}"
        return call_id

    def _event_payload(
        self,
        *,
        stage: str,
        event: str,
        fields: dict[str, object],
    ) -> dict[str, object]:
        self._counter += 1
        payload = {
            "event_index": self._counter,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            **fields,
        }
        return payload

    def _append(self, payload: dict[str, object]) -> None:
        if self._path is None:
            return

        line = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            with self._path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(line)
                file_handle.write("\n")
        except OSError:
            self._path = None


def _compact_stage(stage: str) -> str:
    compact = "".join(
        character if character.isalnum() else "_"
        for character in stage
    )
    compact = compact.strip("_")
    if not compact:
        compact = "stage"
    return compact[:60]
