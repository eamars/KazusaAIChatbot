"""Atomic local service-state store for console-owned child processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json
import os


SNAPSHOT_FILENAME = "services.json"
TMP_FILENAME = "services.json.tmp"


class ProcessStore:
    """Persist desired state and console-owned process metadata."""

    def __init__(self, state_dir: Path) -> None:
        """Create a process store rooted at a local state directory."""

        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._state_dir / SNAPSHOT_FILENAME
        self._tmp_path = self._state_dir / TMP_FILENAME

    def load_snapshot(self) -> dict[str, Any]:
        """Load the current state snapshot, returning an empty shape if absent."""

        if not self._path.exists():
            snapshot: dict[str, Any] = {"version": 0, "services": {}}
            return snapshot

        try:
            raw_text = self._path.read_text(encoding="utf-8")
            snapshot = json.loads(raw_text)
        except OSError as exc:
            raise RuntimeError(f"cannot read process state snapshot: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"process state snapshot is invalid JSON: {exc}") from exc
        return snapshot

    def set_desired_state(
        self,
        service_id: str,
        desired_state: Literal["running", "stopped"],
    ) -> None:
        """Persist the desired state for one service."""

        snapshot = self.load_snapshot()
        service = self._service_record(snapshot=snapshot, service_id=service_id)
        service["desired_state"] = desired_state
        self._write_snapshot(snapshot)

    def record_process_owner(
        self,
        *,
        service_id: str,
        pid: int,
        generation: str,
        command_fingerprint: str,
    ) -> None:
        """Persist ownership metadata for one console-created child process."""

        snapshot = self.load_snapshot()
        service = self._service_record(snapshot=snapshot, service_id=service_id)
        service["pid"] = pid
        service["generation"] = generation
        service["command_fingerprint"] = command_fingerprint
        service["actual_state"] = "running"
        self._write_snapshot(snapshot)

    def update_service(self, service_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """Merge values into one service snapshot and write atomically."""

        snapshot = self.load_snapshot()
        service = self._service_record(snapshot=snapshot, service_id=service_id)
        service.update(values)
        self._write_snapshot(snapshot)
        return service

    def service_version(self, service_id: str) -> int:
        """Return the stored version for one service."""

        snapshot = self.load_snapshot()
        services = snapshot["services"]
        service = services.get(service_id, {})
        version = int(service.get("version", 0))
        return version

    def _service_record(self, *, snapshot: dict[str, Any], service_id: str) -> dict[str, Any]:
        """Return a mutable service record inside the snapshot."""

        services = snapshot.setdefault("services", {})
        service = services.setdefault(service_id, {"version": 0})
        service["version"] = int(service.get("version", 0)) + 1
        snapshot["version"] = int(snapshot.get("version", 0)) + 1
        return service

    def _write_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Write the snapshot with an atomic replace."""

        text = json.dumps(snapshot, indent=2, sort_keys=True)
        try:
            self._tmp_path.write_text(text, encoding="utf-8")
            os.replace(self._tmp_path, self._path)
        except OSError as exc:
            raise RuntimeError(f"cannot write process state snapshot: {exc}") from exc
