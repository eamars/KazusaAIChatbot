"""Argv-only local child-process supervisor for the control console."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import socket
from typing import Any
from urllib.parse import urlparse
import uuid

from control_console.audit import LocalAuditWriter
from control_console.contracts import ServiceRuntimeState, ServiceSpec
from control_console.log_store import ProcessLogStore
from control_console.process_store import ProcessStore


ENDPOINT_CONFLICT_TIMEOUT_SECONDS = 0.2
ENDPOINT_CONFLICT_MESSAGE = (
    "configured endpoint is already in use by an unmanaged process"
)


class ServiceLifecycleError(RuntimeError):
    """Raised when a service lifecycle action cannot be completed."""


class ProcessSupervisor:
    """Manage only registry-declared child processes created by this console."""

    def __init__(
        self,
        *,
        services: dict[str, ServiceSpec],
        store: ProcessStore,
        log_store: ProcessLogStore,
        audit_writer: LocalAuditWriter,
    ) -> None:
        """Create a process supervisor for one registry."""

        self._services = services
        self._store = store
        self._log_store = log_store
        self._audit_writer = audit_writer
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    def service_version(self, service_id: str) -> int:
        """Return the current service state version."""

        version = self._store.service_version(service_id)
        return version

    def service_state(self, service_id: str) -> ServiceRuntimeState:
        """Return one service state projected from local state."""

        self._refresh_process_exit(service_id)
        self._refresh_endpoint_conflict(service_id)
        spec = self._services[service_id]
        snapshot = self._store.load_snapshot()
        raw_service = snapshot["services"].get(service_id, {})
        state = _runtime_state_from_snapshot(spec=spec, raw_service=raw_service)
        return state

    def all_service_states(self) -> list[ServiceRuntimeState]:
        """Return current state for every registered service."""

        self.refresh_process_states()
        states = [self.service_state(service_id) for service_id in self._services]
        return states

    def refresh_process_states(self) -> None:
        """Detect crashed child processes and persist bounded state."""

        for service_id in list(self._services):
            self._refresh_process_exit(service_id)
            self._refresh_endpoint_conflict(service_id)

    async def start_service(
        self,
        *,
        service_id: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Start one registry service using `asyncio.create_subprocess_exec`."""

        spec = self._services[service_id]
        self._ensure_dependencies_running(service_id)
        request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
        self._audit_writer.write_event(
            event_type="service_start_requested",
            operator_id=operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            reason=reason,
            request_id=request_id,
        )
        self._ensure_endpoint_available(service_id)
        self._store.set_desired_state(service_id, "running")
        cwd = _resolve_cwd(spec.cwd)
        env = os.environ.copy()
        env.update(spec.env)
        process = await asyncio.create_subprocess_exec(
            *spec.command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[service_id] = process
        generation = f"generation-{uuid.uuid4().hex}"
        fingerprint = _command_fingerprint(spec.command)
        self._store.record_process_owner(
            service_id=service_id,
            pid=process.pid,
            generation=generation,
            command_fingerprint=fingerprint,
        )
        self._store.update_service(
            service_id,
            {
                "actual_state": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stopped_at": None,
                "exit_code": None,
                "last_error_preview": None,
            },
        )
        self._log_store.append_line(
            service_id=service_id,
            stream="supervisor",
            line=f"service started pid={process.pid}",
        )
        audit_event = self._audit_writer.write_event(
            event_type="service_started",
            operator_id=operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            new_state={"actual_state": "running", "pid": process.pid},
            reason=reason,
            request_id=request_id,
        )
        state = self.service_state(service_id)
        response = _action_response(
            action="start",
            request_id=request_id,
            audit_event_id=audit_event.event_id,
            state=state,
        )
        return response

    async def stop_service(
        self,
        *,
        service_id: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Stop one console-owned child process."""

        request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
        for dependent_id in self._running_dependents(service_id):
            await self.stop_service(
                service_id=dependent_id,
                operator_id=operator_id,
                reason=f"dependency stop before {service_id}: {reason}",
            )
        self._audit_writer.write_event(
            event_type="service_stop_requested",
            operator_id=operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            reason=reason,
            request_id=request_id,
        )
        self._store.set_desired_state(service_id, "stopped")
        process = self._processes.get(service_id)
        self._verify_owned_process(service_id=service_id, process=process)
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        if process is not None:
            self._processes.pop(service_id, None)
        self._store.update_service(
            service_id,
            {
                "actual_state": "stopped",
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "pid": None,
                "exit_code": getattr(process, "returncode", 0),
            },
        )
        self._log_store.append_line(
            service_id=service_id,
            stream="supervisor",
            line="service stopped",
        )
        audit_event = self._audit_writer.write_event(
            event_type="service_stopped",
            operator_id=operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            new_state={"actual_state": "stopped"},
            reason=reason,
            request_id=request_id,
        )
        state = self.service_state(service_id)
        response = _action_response(
            action="stop",
            request_id=request_id,
            audit_event_id=audit_event.event_id,
            state=state,
        )
        return response

    async def restart_service(
        self,
        *,
        service_id: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Restart one service with stop then start semantics."""

        request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
        self._audit_writer.write_event(
            event_type="service_restart_requested",
            operator_id=operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            reason=reason,
            request_id=request_id,
        )
        if service_id in self._processes:
            await self.stop_service(
                service_id=service_id,
                operator_id=operator_id,
                reason=reason,
            )
        start_response = await self.start_service(
            service_id=service_id,
            operator_id=operator_id,
            reason=reason,
        )
        start_response["action"] = "restart"
        return start_response

    async def shutdown_owned_services(self, *, operator_id: str, reason: str) -> None:
        """Stop every console-owned child service in dependency-aware order."""

        for service_id in reversed(list(self._services)):
            state = self.service_state(service_id)
            if state.actual_state == "running":
                await self.stop_service(
                    service_id=service_id,
                    operator_id=operator_id,
                    reason=reason,
                )

    def _ensure_dependencies_running(self, service_id: str) -> None:
        """Reject start when declared dependencies are not available."""

        spec = self._services[service_id]
        for dependency_id in spec.dependencies:
            dependency_state = self.service_state(dependency_id)
            if not _dependency_is_available(dependency_state):
                self._store.update_service(
                    service_id,
                    {
                        "actual_state": "unavailable",
                        "last_error_preview": (
                            f"dependency {dependency_id} is not running"
                        ),
                    },
                )
                raise ServiceLifecycleError(
                    f"dependency {dependency_id} is not running for {service_id}"
                )

    def _ensure_endpoint_available(self, service_id: str) -> None:
        """Refuse to spawn over a configured endpoint already in use."""

        spec = self._services[service_id]
        if spec.health_url is None:
            return
        if not _endpoint_is_listening(spec.health_url):
            return

        self._mark_conflict(service_id, ENDPOINT_CONFLICT_MESSAGE)
        raise ServiceLifecycleError(f"{ENDPOINT_CONFLICT_MESSAGE}: {service_id}")

    def _refresh_endpoint_conflict(self, service_id: str) -> None:
        """Keep service state truthful when an unmanaged endpoint is live."""

        spec = self._services[service_id]
        if spec.health_url is None:
            return
        process = self._processes.get(service_id)
        if process is not None and process.returncode is None:
            return

        raw_service = self._store.load_snapshot()["services"].get(service_id, {})
        actual_state = raw_service.get("actual_state", "stopped")
        last_error_preview = raw_service.get("last_error_preview")
        if _endpoint_is_listening(spec.health_url):
            if (
                actual_state != "conflict"
                or last_error_preview != ENDPOINT_CONFLICT_MESSAGE
            ):
                self._mark_conflict(service_id, ENDPOINT_CONFLICT_MESSAGE)
            return

        if (
            actual_state == "conflict"
            and last_error_preview == ENDPOINT_CONFLICT_MESSAGE
        ):
            self._store.update_service(
                service_id,
                {
                    "actual_state": "stopped",
                    "last_error_preview": None,
                    "pid": None,
                },
            )

    def _running_dependents(self, service_id: str) -> list[str]:
        """Return running services that depend on a target service."""

        dependents: list[str] = []
        for candidate_id, candidate in self._services.items():
            if service_id not in candidate.dependencies:
                continue
            state = self.service_state(candidate_id)
            if state.actual_state == "running":
                dependents.append(candidate_id)
        return dependents

    def _verify_owned_process(
        self,
        *,
        service_id: str,
        process: asyncio.subprocess.Process | None,
    ) -> None:
        """Verify persisted owner metadata before terminating a process."""

        state = self.service_state(service_id)
        if state.pid is None and process is None:
            return
        if process is None:
            self._mark_conflict(service_id, "no console-owned process handle")
            raise ServiceLifecycleError(f"service {service_id} is not console-owned")
        expected_fingerprint = _command_fingerprint(self._services[service_id].command)
        if state.pid != process.pid or state.command_fingerprint != expected_fingerprint:
            self._mark_conflict(service_id, "process ownership metadata mismatch")
            raise ServiceLifecycleError(
                f"service {service_id} ownership metadata mismatch"
            )

    def _mark_conflict(self, service_id: str, message: str) -> None:
        """Persist conflict state without touching the process."""

        self._store.update_service(
            service_id,
            {
                "actual_state": "conflict",
                "last_error_preview": message,
            },
        )
        self._log_store.append_line(
            service_id=service_id,
            stream="supervisor",
            line=message,
        )

    def _refresh_process_exit(self, service_id: str) -> None:
        """Persist crash state for a child process that has exited."""

        process = self._processes.get(service_id)
        if process is None or process.returncode is None:
            return
        state = self._store.load_snapshot()["services"].get(service_id, {})
        if state.get("actual_state") != "running":
            return
        exit_code = process.returncode
        self._store.update_service(
            service_id,
            {
                "actual_state": "crashed",
                "exit_code": exit_code,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "last_error_preview": f"process exited with code {exit_code}",
            },
        )
        self._log_store.append_line(
            service_id=service_id,
            stream="supervisor",
            line=f"service crashed exit_code={exit_code}",
        )
        self._audit_writer.write_event(
            event_type="service_crashed",
            operator_id="system",
            service_id=service_id,
            target={"service_id": service_id},
            new_state={"actual_state": "crashed", "exit_code": exit_code},
        )


def _dependency_is_available(state: ServiceRuntimeState) -> bool:
    """Return whether a dependency can serve a child process."""

    if state.actual_state == "running":
        return True
    if (
        state.actual_state == "conflict"
        and state.last_error_preview == ENDPOINT_CONFLICT_MESSAGE
    ):
        return True
    return False


def _runtime_state_from_snapshot(
    *,
    spec: ServiceSpec,
    raw_service: dict[str, Any],
) -> ServiceRuntimeState:
    """Project a service runtime state from registry and local snapshot data."""

    state = ServiceRuntimeState(
        id=spec.id,
        display_name=spec.display_name,
        kind=spec.kind,
        desired_state=raw_service.get("desired_state", "stopped"),
        actual_state=raw_service.get("actual_state", "stopped"),
        pid=raw_service.get("pid"),
        generation=raw_service.get("generation"),
        command_fingerprint=raw_service.get("command_fingerprint"),
        exit_code=raw_service.get("exit_code"),
        dependencies=list(spec.dependencies),
        last_error_preview=raw_service.get("last_error_preview"),
        version=int(raw_service.get("version", 0)),
    )
    return state


def _resolve_cwd(cwd: str | None) -> Path | None:
    """Resolve a registry working directory for subprocess execution."""

    if cwd is None:
        return None
    path = Path(cwd).resolve()
    return path


def _command_fingerprint(command: list[str]) -> str:
    """Return a stable non-secret fingerprint for a service argv."""

    joined_command = "\0".join(command)
    fingerprint = hashlib.sha256(joined_command.encode("utf-8")).hexdigest()
    return fingerprint


def _endpoint_is_listening(health_url: str) -> bool:
    """Return whether a service endpoint is already accepting TCP connections."""

    parsed = urlparse(health_url)
    host = parsed.hostname
    if host is None:
        return False
    try:
        port = _endpoint_port(parsed.scheme, parsed.port)
        with socket.create_connection(
            (host, port),
            timeout=ENDPOINT_CONFLICT_TIMEOUT_SECONDS,
        ):
            return True
    except (OSError, ValueError):
        return False


def _endpoint_port(scheme: str, configured_port: int | None) -> int:
    """Return an explicit or scheme-default TCP port for endpoint probing."""

    if configured_port is not None:
        return configured_port
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    raise ValueError(f"unsupported endpoint scheme: {scheme}")


def _action_response(
    *,
    action: str,
    request_id: str,
    audit_event_id: str,
    state: ServiceRuntimeState,
) -> dict[str, Any]:
    """Build the shared lifecycle action response."""

    response: dict[str, Any] = {
        "request_id": request_id,
        "service": state.model_dump(mode="json"),
        "action": action,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "audit_event_id": audit_event_id,
    }
    return response
