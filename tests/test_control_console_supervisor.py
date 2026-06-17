"""Process-supervisor tests for managed local services."""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest


class _FakeProcess:
    """Small subprocess fake used to prove supervisor call shape."""

    def __init__(self) -> None:
        """Create a running fake process."""

        self.pid = 4242
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = None
        self.stderr = None

    def terminate(self) -> None:
        """Record graceful termination."""

        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        """Record forced termination."""

        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        """Return the current process exit code."""

        return_code = self.returncode if self.returncode is not None else 0
        return return_code


class _FakeStream:
    """Async readline stream for subprocess log-drain tests."""

    def __init__(self, lines: list[str]) -> None:
        """Create a finite stream from text lines."""

        self._lines = [f"{line}\n".encode("utf-8") for line in lines]

    async def readline(self) -> bytes:
        """Return one encoded line, then EOF."""

        await asyncio.sleep(0)
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProcessWithStreams(_FakeProcess):
    """Fake process with stdout and stderr pipes."""

    def __init__(self) -> None:
        """Create a fake process that emits two bounded log lines."""

        super().__init__()
        self.stdout = _FakeStream(["adapter connected"])
        self.stderr = _FakeStream(["prompt=secret should redact"])


def _service_spec(service_id: str, tmp_path, *, dependencies=None):
    """Build a safe service spec for supervisor tests."""

    from control_console.contracts import ServiceSpec

    command_name = service_id.replace(".", "_")
    spec = ServiceSpec.model_validate({
        "id": service_id,
        "display_name": service_id,
        "kind": "adapter" if service_id.startswith("adapter") else "backend",
        "command": ["python", "-m", command_name],
        "cwd": str(tmp_path),
        "dependencies": dependencies or [],
    })
    return spec


@pytest.mark.asyncio
async def test_start_stop_restart_uses_argv_no_shell_and_records_audit(
    monkeypatch,
    tmp_path,
) -> None:
    """Lifecycle actions must use argv subprocesses and write audit events."""

    from control_console.audit import LocalAuditWriter
    from control_console.contracts import ServiceSpec
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    calls: list[dict] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return_value = _FakeProcess()
        return return_value

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    spec = ServiceSpec.model_validate({
        "id": "brain",
        "display_name": "Brain service",
        "kind": "backend",
        "command": ["python", "-m", "kazusa_ai_chatbot.main"],
        "cwd": str(tmp_path),
    })
    audit_writer = LocalAuditWriter(tmp_path / "audit.jsonl")
    supervisor = ProcessSupervisor(
        services={"brain": spec},
        store=ProcessStore(tmp_path / "state"),
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=audit_writer,
    )

    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="start for test",
    )
    await supervisor.restart_service(
        service_id="brain",
        operator_id="operator",
        reason="restart for test",
    )
    await supervisor.stop_service(
        service_id="brain",
        operator_id="operator",
        reason="stop for test",
    )

    assert calls
    assert calls[0]["args"] == ("python", "-m", "kazusa_ai_chatbot.main")
    assert calls[0]["kwargs"].get("shell") in (None, False)
    event_types = [event.event_type for event in audit_writer.read_recent(limit=20)]
    assert "service_start_requested" in event_types
    assert "service_restart_requested" in event_types
    assert "service_stop_requested" in event_types


@pytest.mark.asyncio
async def test_dependency_order_requires_brain_before_adapter_and_stops_dependents(
    monkeypatch,
    tmp_path,
) -> None:
    """Adapters cannot start before dependencies and brain stop stops dependents."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor, ServiceLifecycleError

    calls: list[tuple[str, ...]] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = kwargs
        calls.append(tuple(args))
        return _FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    services = {
        "brain": _service_spec("brain", tmp_path),
        "adapter.debug": _service_spec(
            "adapter.debug",
            tmp_path,
            dependencies=["brain"],
        ),
    }
    audit_writer = LocalAuditWriter(tmp_path / "audit.jsonl")
    supervisor = ProcessSupervisor(
        services=services,
        store=ProcessStore(tmp_path / "state"),
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=audit_writer,
    )

    with pytest.raises(ServiceLifecycleError):
        await supervisor.start_service(
            service_id="adapter.debug",
            operator_id="operator",
            reason="start adapter too early",
        )
    early_adapter_state = supervisor.service_state("adapter.debug")
    assert early_adapter_state.actual_state == "unavailable"
    assert early_adapter_state.last_error_preview == (
        "dependency brain is not running"
    )

    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="start brain",
    )
    await supervisor.start_service(
        service_id="adapter.debug",
        operator_id="operator",
        reason="start adapter",
    )
    await supervisor.stop_service(
        service_id="brain",
        operator_id="operator",
        reason="stop graph",
    )

    assert calls == [
        ("python", "-m", "brain"),
        ("python", "-m", "adapter_debug"),
    ]
    assert supervisor.service_state("adapter.debug").actual_state == "stopped"
    assert supervisor.service_state("brain").actual_state == "stopped"


@pytest.mark.asyncio
async def test_start_clears_previous_exit_code_and_error_preview(
    monkeypatch,
    tmp_path,
) -> None:
    """A successful start should not display stale failure metadata."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = args
        _ = kwargs
        return _FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    store = ProcessStore(tmp_path / "state")
    store.update_service(
        "brain",
        {
            "actual_state": "crashed",
            "exit_code": 1,
            "last_error_preview": "process exited with code 1",
        },
    )
    supervisor = ProcessSupervisor(
        services={"brain": _service_spec("brain", tmp_path)},
        store=store,
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
    )

    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="restart after crash",
    )

    state = supervisor.service_state("brain")
    assert state.actual_state == "running"
    assert state.exit_code is None
    assert state.last_error_preview is None


@pytest.mark.asyncio
async def test_start_drains_child_stdout_stderr_to_redacted_process_logs(
    monkeypatch,
    tmp_path,
) -> None:
    """Child process logs must be visible without leaking sensitive text."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = args
        _ = kwargs
        return _FakeProcessWithStreams()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    log_store = ProcessLogStore(tmp_path / "logs")
    supervisor = ProcessSupervisor(
        services={"brain": _service_spec("brain", tmp_path)},
        store=ProcessStore(tmp_path / "state"),
        log_store=log_store,
        audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
    )

    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="capture child logs",
    )
    await asyncio.sleep(0.05)

    lines = log_store.tail(service_id="brain", limit=10)
    streams = {(line.stream, line.line) for line in lines}
    assert ("stdout", "adapter connected") in streams
    assert any(line.stream == "stderr" and "[redacted]" in line.line for line in lines)
    assert all("secret" not in line.line for line in lines)


@pytest.mark.asyncio
async def test_start_marks_conflict_when_configured_endpoint_is_occupied(
    monkeypatch,
    tmp_path,
) -> None:
    """Start should not spawn a child over an unmanaged configured endpoint."""

    from control_console.audit import LocalAuditWriter
    from control_console.contracts import ServiceSpec
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor, ServiceLifecycleError

    calls: list[tuple[str, ...]] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = kwargs
        calls.append(tuple(args))
        return _FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    stop_listener = threading.Event()

    def accept_connections() -> None:
        """Drain probe connections until the test closes the listener."""

        listener.settimeout(0.1)
        while not stop_listener.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            with connection:
                pass

    listener_thread = threading.Thread(target=accept_connections, daemon=True)
    listener_thread.start()
    try:
        port = listener.getsockname()[1]
        spec = ServiceSpec.model_validate({
            "id": "brain",
            "display_name": "Brain service",
            "kind": "backend",
            "command": ["python", "-m", "kazusa_ai_chatbot.main"],
            "cwd": str(tmp_path),
            "health_url": f"http://127.0.0.1:{port}/health",
        })
        supervisor = ProcessSupervisor(
            services={"brain": spec},
            store=ProcessStore(tmp_path / "state"),
            log_store=ProcessLogStore(tmp_path / "logs"),
            audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
        )

        initial_state = supervisor.service_state("brain")
        assert initial_state.actual_state == "conflict"
        assert initial_state.last_error_preview == (
            "configured endpoint is already in use by an unmanaged process"
        )

        with pytest.raises(ServiceLifecycleError):
            await supervisor.start_service(
                service_id="brain",
                operator_id="operator",
                reason="start with occupied endpoint",
            )

        state = supervisor.service_state("brain")
        assert state.actual_state == "conflict"
        assert state.last_error_preview == (
            "configured endpoint is already in use by an unmanaged process"
        )
        assert calls == []
    finally:
        stop_listener.set()
        listener.close()
        listener_thread.join(timeout=1.0)


@pytest.mark.asyncio
async def test_start_allows_adapter_when_dependency_endpoint_is_unmanaged_available(
    monkeypatch,
    tmp_path,
) -> None:
    """Adapters may start against a live unmanaged brain HTTP endpoint."""

    from control_console.audit import LocalAuditWriter
    from control_console.contracts import ServiceSpec
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    calls: list[tuple[str, ...]] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = kwargs
        calls.append(tuple(args))
        return _FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    stop_listener = threading.Event()

    def accept_connections() -> None:
        """Drain probe connections until the test closes the listener."""

        listener.settimeout(0.1)
        while not stop_listener.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            with connection:
                pass

    listener_thread = threading.Thread(target=accept_connections, daemon=True)
    listener_thread.start()
    try:
        port = listener.getsockname()[1]
        brain = ServiceSpec.model_validate({
            "id": "brain",
            "display_name": "Brain service",
            "kind": "backend",
            "command": ["python", "-m", "kazusa_ai_chatbot.main"],
            "cwd": str(tmp_path),
            "health_url": f"http://127.0.0.1:{port}/health",
        })
        adapter = _service_spec(
            "adapter.debug",
            tmp_path,
            dependencies=["brain"],
        )
        supervisor = ProcessSupervisor(
            services={"brain": brain, "adapter.debug": adapter},
            store=ProcessStore(tmp_path / "state"),
            log_store=ProcessLogStore(tmp_path / "logs"),
            audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
        )

        assert supervisor.service_state("brain").actual_state == "conflict"

        await supervisor.start_service(
            service_id="adapter.debug",
            operator_id="operator",
            reason="start debug against live brain",
        )

        assert calls == [("python", "-m", "adapter_debug")]
        assert supervisor.service_state("adapter.debug").actual_state == "running"
    finally:
        stop_listener.set()
        listener.close()
        listener_thread.join(timeout=1.0)


@pytest.mark.asyncio
async def test_stop_refuses_stale_or_unowned_process_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    """Stop must not kill when persisted ownership metadata is inconsistent."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor, ServiceLifecycleError

    async def fake_create_subprocess_exec(*args, **kwargs):
        _ = args
        _ = kwargs
        return _FakeProcess()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    store = ProcessStore(tmp_path / "state")
    supervisor = ProcessSupervisor(
        services={"brain": _service_spec("brain", tmp_path)},
        store=store,
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
    )
    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="start brain",
    )
    store.update_service("brain", {"command_fingerprint": "external-fingerprint"})

    with pytest.raises(ServiceLifecycleError):
        await supervisor.stop_service(
            service_id="brain",
            operator_id="operator",
            reason="stop stale",
        )

    assert supervisor.service_state("brain").actual_state == "conflict"


@pytest.mark.asyncio
async def test_shutdown_and_crash_refresh_cover_owned_process_edges(
    monkeypatch,
    tmp_path,
) -> None:
    """Supervisor should stop owned services and mark crashed children."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    processes: list[_FakeProcess] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        process = _FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    store = ProcessStore(tmp_path / "state")
    supervisor = ProcessSupervisor(
        services={"brain": _service_spec("brain", tmp_path)},
        store=store,
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
    )

    assert supervisor.service_version("brain") == 0
    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="start for shutdown",
    )
    await supervisor.shutdown_owned_services(
        operator_id="operator",
        reason="test shutdown",
    )
    assert processes[0].terminated is True
    assert supervisor.service_state("brain").actual_state == "stopped"

    await supervisor.start_service(
        service_id="brain",
        operator_id="operator",
        reason="start for crash",
    )
    processes[1].returncode = 9
    state = supervisor.service_state("brain")

    assert state.actual_state == "crashed"
    assert state.exit_code == 9


@pytest.mark.asyncio
async def test_log_drain_and_task_cleanup_edge_paths(tmp_path) -> None:
    """Private log-drain helpers should handle stream errors and cleanup."""

    from control_console.audit import LocalAuditWriter
    from control_console.log_store import ProcessLogStore
    from control_console.process_store import ProcessStore
    from control_console.supervisor import ProcessSupervisor

    class BrokenStream:
        async def readline(self):
            raise ValueError("stream closed")

    supervisor = ProcessSupervisor(
        services={"brain": _service_spec("brain", tmp_path)},
        store=ProcessStore(tmp_path / "state"),
        log_store=ProcessLogStore(tmp_path / "logs"),
        audit_writer=LocalAuditWriter(tmp_path / "audit.jsonl"),
    )

    await supervisor._drain_process_stream(
        service_id="brain",
        stream_name="stdout",
        stream=BrokenStream(),
    )
    assert "log drain stopped" in supervisor._log_store.tail(
        service_id="brain",
        limit=1,
    )[0].line

    supervisor._discard_log_task("brain", asyncio.create_task(asyncio.sleep(0)))
    pending_task = asyncio.create_task(asyncio.sleep(10))
    supervisor._log_tasks["brain"] = {pending_task}
    supervisor._cancel_log_tasks("brain")

    assert pending_task.cancelled() or pending_task.cancelling()


def test_supervisor_small_helper_edges() -> None:
    """Small helper branches should stay explicit and covered."""

    from control_console.supervisor import (
        _endpoint_is_listening,
        _endpoint_port,
        _resolve_cwd,
    )

    assert _resolve_cwd(None) is None
    assert _endpoint_port("https", None) == 443
    assert _endpoint_port("http", None) == 80
    with pytest.raises(ValueError):
        _endpoint_port("ftp", None)
    assert _endpoint_is_listening("http:///missing-host") is False
