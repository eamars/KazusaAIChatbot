"""Live LLM user-wire acceptance nodes for the DSH task runtime."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.action_spec.attempt_ledger import (
    ACTION_ATTEMPT_LEDGER_COLLECTION,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    PENDING_TASK_CONTINUATION_VERSION,
    RESOLVER_PENDING_RESUME_VERSION,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.pending import (
    RESOLVER_PENDING_HIL_ACTION_KIND,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.test_agentic_resolver_live_llm import _require_live_backend

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
SIDECAR_ENTRY = (
    PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
)
DSH_RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v2"
ISOLATED_TEST_DATABASE = "_test_kazusa_live_llm"
EPHEMERAL_TEST_DATABASE_PREFIX = "_test_kazusa_"
EPHEMERAL_TEST_DATABASE_GUARD_ENV = "KAZUSA_EPHEMERAL_TEST_DATABASE_GUARD"
EPHEMERAL_TEST_DATABASE_NAME_ENV = "KAZUSA_EPHEMERAL_TEST_DATABASE_NAME"
_PLAN3_EVIDENCE_ERRORS = (
    AssertionError,
    asyncio.TimeoutError,
    KeyError,
    OSError,
    PyMongoError,
    RuntimeError,
    sqlite3.Error,
    TypeError,
    ValueError,
)


class _Plan3ChildProcess:
    """Own one live-gate child process and its inspectable output files."""

    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        process: asyncio.subprocess.Process,
        stdout_file: BinaryIO,
        stderr_file: BinaryIO,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        self.name = name
        self.command = command
        self.process = process
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path


class _Plan3LoopbackAdapterServer:
    """Capture the registered debug adapter's real HTTP callbacks."""

    def __init__(self, *, platform: str, shared_secret: str) -> None:
        if not platform.strip():
            raise ValueError("loopback adapter platform is required")
        if not shared_secret.strip():
            raise ValueError("loopback adapter shared secret is required")
        self._platform = platform
        self._shared_secret = shared_secret
        self._lock = Lock()
        self._capability_payloads: list[dict[str, Any]] = []
        self._delivery_payloads: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Serve the two authenticated callback endpoints."""

            def do_POST(self) -> None:
                owner._handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="dsh-plan3-loopback-adapter",
            daemon=True,
        )

    @property
    def callback_url(self) -> str:
        """Return the local callback base URL."""

        address = self._server.server_address
        return f"http://127.0.0.1:{address[1]}"

    def start(self) -> None:
        """Start the local HTTP callback boundary."""

        self._thread.start()

    def stop(self) -> None:
        """Stop the callback boundary and join its owned thread."""

        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("loopback adapter thread did not stop")

    def capability_payloads(self) -> list[dict[str, Any]]:
        """Return an immutable-enough callback snapshot for assertions."""

        with self._lock:
            return [dict(item) for item in self._capability_payloads]

    def delivery_payloads(self) -> list[dict[str, Any]]:
        """Return an immutable-enough outbound delivery snapshot."""

        with self._lock:
            return [dict(item) for item in self._delivery_payloads]

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        """Validate auth, record a typed callback, and return its receipt."""

        expected_authorization = f"Bearer {self._shared_secret}"
        if request.headers.get("Authorization") != expected_authorization:
            self._respond(request, 401, {"available": False})
            return
        raw_length = request.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._respond(request, 400, {"error": "invalid content length"})
            return
        try:
            payload_value = json.loads(
                request.rfile.read(length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond(request, 400, {"error": "invalid JSON"})
            return
        if not isinstance(payload_value, dict):
            self._respond(request, 400, {"error": "JSON object required"})
            return
        if request.path == "/send_message/capability":
            with self._lock:
                self._capability_payloads.append(dict(payload_value))
            self._respond(request, 200, {"available": True})
            return
        if request.path != "/send_message":
            self._respond(request, 404, {"error": "unknown callback"})
            return

        message_id = f"dsh-plan3-adapter-message-{uuid4().hex}"
        sent_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        delivery = {
            **payload_value,
            "message_id": message_id,
            "sent_at": sent_at,
        }
        with self._lock:
            self._delivery_payloads.append(delivery)
        self._respond(
            request,
            200,
            {
                "platform": self._platform,
                "channel_id": payload_value.get("channel_id", ""),
                "message_id": message_id,
                "sent_at": sent_at,
            },
        )

    @staticmethod
    def _respond(
        request: BaseHTTPRequestHandler,
        status_code: int,
        payload: Mapping[str, object],
    ) -> None:
        """Return one compact JSON callback response."""

        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.send_response(status_code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(encoded)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(encoded)


def _loopback_endpoint(name: str, value: str, *, required_path: str | None = None) -> tuple[str, int]:
    """Validate one configured HTTP endpoint before local process startup."""

    parsed = urlsplit(value.strip())
    if parsed.scheme != "http":
        raise AssertionError(f"{name} must use an HTTP URL for local startup")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AssertionError(f"{name} contains unsupported URL credentials or suffix")
    hostname = parsed.hostname
    if not hostname:
        raise AssertionError(f"{name} must include a loopback hostname")
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        raise AssertionError(f"{name} must use a loopback hostname")
    if required_path is not None and parsed.path != required_path:
        raise AssertionError(f"{name} must use the {required_path} path")
    if parsed.port is None:
        raise AssertionError(f"{name} must include an explicit configured port")
    if not 1 <= parsed.port <= 65535:
        raise AssertionError(f"{name} has an invalid configured port")
    return hostname, parsed.port


def _assert_port_available(name: str, hostname: str, port: int) -> None:
    """Prove the configured local port is free before claiming ownership."""

    address_family = socket.AF_INET6 if ":" in hostname else socket.AF_INET
    probe = socket.socket(address_family, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        if probe.connect_ex((hostname, port)) == 0:
            raise AssertionError(
                f"configured {name} port {port} is already occupied"
            )
    finally:
        probe.close()
    listener = socket.socket(address_family, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((hostname, port))
    except OSError as exc:
        raise AssertionError(
            f"configured {name} port {port} is not available: {exc.__class__.__name__}"
        ) from exc
    finally:
        listener.close()


def _runtime_environment(tmp_path: Path) -> dict[str, str]:
    """Build child settings from pytest-loaded config and isolated test paths."""

    database_name = f"{EPHEMERAL_TEST_DATABASE_PREFIX}plan3_{uuid4().hex}"
    _bind_plan3_parent_database(database_name)
    environment = os.environ.copy()
    data_root = (tmp_path / "dsh-data").resolve()
    workspace_root = (tmp_path / "workspace").resolve()
    data_root.mkdir(parents=True, exist_ok=False)
    workspace_root.mkdir(parents=True, exist_ok=False)
    environment.update({
        "AGENTIC_RESOLVER_WORKSPACE_ROOT": str(workspace_root),
        "KAZUSA_DSH_DATA_ROOT": str(data_root),
        "KAZUSA_DSH_PYTHON_EXECUTABLE": str(PYTHON_EXECUTABLE.resolve()),
        "MONGODB_DB_NAME": database_name,
        "KAZUSA_TEST_DB_GUARD": "1",
        EPHEMERAL_TEST_DATABASE_GUARD_ENV: "1",
        EPHEMERAL_TEST_DATABASE_NAME_ENV: database_name,
        "TASK_RESOLUTION_INLINE_BUDGET_SECONDS": "120.0",
        "PYTHONPATH": str((PROJECT_ROOT / "src").resolve()),
    })
    return environment


def _bind_plan3_parent_database(database_name: str) -> None:
    """Bind parent diagnostics to one exact reserved ephemeral database."""

    if not database_name.startswith(EPHEMERAL_TEST_DATABASE_PREFIX):
        raise AssertionError("Plan 3 case database is outside the reserved prefix")
    os.environ["MONGODB_DB_NAME"] = database_name
    os.environ["KAZUSA_TEST_DB_GUARD"] = "1"
    os.environ[EPHEMERAL_TEST_DATABASE_GUARD_ENV] = "1"
    os.environ[EPHEMERAL_TEST_DATABASE_NAME_ENV] = database_name
    from kazusa_ai_chatbot.db import _client as client_module

    client_module.MONGODB_DB_NAME = database_name


async def _seed_plan3_character_profile(
    environment: Mapping[str, str],
) -> None:
    """Seed the canonical identity before a Brain child starts."""

    database_name = environment.get("MONGODB_DB_NAME", "")
    if (
        not database_name
        or not database_name.startswith(EPHEMERAL_TEST_DATABASE_PREFIX)
    ):
        raise AssertionError("Plan 3 seed target is not an ephemeral database")
    from kazusa_ai_chatbot.character_profile import load_character_profile_seed
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import _client as client_module
    from kazusa_ai_chatbot.db.character_identity_growth import ensure_seed_identity

    await client_module.close_db()
    _bind_plan3_parent_database(database_name)
    client_module._assert_guarded_database_name()
    seed = load_character_profile_seed(PROJECT_ROOT / "personalities" / "example.json")
    await ensure_seed_identity(
        character_id=CHARACTER_GLOBAL_USER_ID,
        seed=seed,
    )


async def _drop_plan3_database(environment: Mapping[str, str]) -> None:
    """Drop only the exact guarded ephemeral database owned by one case."""

    database_name = environment.get("MONGODB_DB_NAME", "")
    if not database_name.startswith(EPHEMERAL_TEST_DATABASE_PREFIX):
        raise AssertionError("Plan 3 cleanup target is outside the reserved prefix")
    from kazusa_ai_chatbot.db import _client as client_module

    _bind_plan3_parent_database(database_name)
    client_module.MONGODB_DB_NAME = database_name
    client_module._assert_guarded_database_name()
    await client_module.close_db()
    cleanup_client = client_module.AsyncIOMotorClient(client_module.MONGODB_URI)
    try:
        await cleanup_client.drop_database(database_name)
    finally:
        cleanup_client.close()


async def _launch_plan3_child(
    *,
    name: str,
    command: list[str],
    environment: Mapping[str, str],
    artifact_dir: Path,
    artifact_stamp: str,
) -> _Plan3ChildProcess:
    """Start one hidden child and retain only its owned process handles."""

    stdout_path = artifact_dir / f"{name}_{artifact_stamp}.stdout.log"
    stderr_path = artifact_dir / f"{name}_{artifact_stamp}.stderr.log"
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    creation_flags = (
        subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=dict(environment),
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creation_flags,
        )
    except BaseException:
        stdout_file.close()
        stderr_file.close()
        raise
    return _Plan3ChildProcess(
        name=name,
        command=command,
        process=process,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _process_artifact(child: _Plan3ChildProcess) -> dict[str, object]:
    """Project process identity without environment secrets."""

    return {
        "name": child.name,
        "pid": child.process.pid,
        "command": child.command,
        "stdout_path": str(child.stdout_path),
        "stderr_path": str(child.stderr_path),
    }


async def _stop_plan3_child(child: _Plan3ChildProcess) -> None:
    """Stop and close exactly one child process owned by this test."""

    if child.process.returncode is None:
        child.process.terminate()
    try:
        await asyncio.wait_for(child.process.wait(), timeout=10)
    except asyncio.TimeoutError:
        if child.process.returncode is None:
            child.process.kill()
        await child.process.wait()
    finally:
        child.stdout_file.close()
        child.stderr_file.close()


async def _probe_brain_health(
    client: httpx.AsyncClient,
    *,
    brain_url: str,
    brain_secret: str,
) -> dict[str, object]:
    """Probe authenticated Brain readiness while preserving safe diagnostics."""

    try:
        response = await client.get(
            f"{brain_url.rstrip('/')}/runtime/dsh/health",
            headers={"Authorization": f"Bearer {brain_secret}"},
        )
    except httpx.RequestError as exc:
        return {"status": "unreachable", "error_class": exc.__class__.__name__}
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return {
        "http_status": response.status_code,
        "payload": payload,
        "ready": response.status_code == 200
        and isinstance(payload, Mapping)
        and payload.get("status") == "ready",
    }


async def _probe_sidecar_health(
    client: httpx.AsyncClient,
    *,
    sidecar_url: str,
    rpc_token: str,
) -> dict[str, object]:
    """Probe authenticated sidecar readiness through its JSON-RPC owner."""

    request_payload = {
        "jsonrpc": "2.0",
        "id": f"plan3-health-{time.time_ns()}",
        "method": "system.health",
        "params": {"protocol_version": DSH_RPC_PROTOCOL_VERSION},
    }
    try:
        response = await client.post(
            sidecar_url,
            headers={
                "Authorization": f"Bearer {rpc_token}",
                "Content-Type": "application/json",
            },
            json=request_payload,
        )
    except httpx.RequestError as exc:
        return {"status": "unreachable", "error_class": exc.__class__.__name__}
    try:
        payload = response.json()
    except ValueError:
        payload = None
    result = payload.get("result") if isinstance(payload, Mapping) else None
    return {
        "http_status": response.status_code,
        "payload": payload,
        "ready": response.status_code == 200
        and isinstance(result, Mapping)
        and result.get("status") == "ready",
    }


async def _wait_for_plan3_readiness(
    *,
    children: list[_Plan3ChildProcess],
    environment: Mapping[str, str],
    readiness_path: Path,
) -> dict[str, object]:
    """Wait until both owned services report their authenticated readiness."""

    child_by_name = {child.name: child for child in children}
    deadline = time.monotonic() + 90
    latest: dict[str, object] = {}
    started_at = time.perf_counter()
    timeout = httpx.Timeout(5.0, connect=1.0, write=2.0, pool=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.monotonic() < deadline:
            for child in children:
                if child.process.returncode is not None:
                    raise AssertionError(
                        f"{child.name} exited before readiness with code "
                        f"{child.process.returncode}"
                    )
            brain_probe = await _probe_brain_health(
                client,
                brain_url=environment["KAZUSA_DSH_BRAIN_URL"].rstrip("/"),
                brain_secret=environment["KAZUSA_DSH_BRAIN_SHARED_SECRET"],
            )
            sidecar_probe = await _probe_sidecar_health(
                client,
                sidecar_url=environment["KAZUSA_DSH_SIDECAR_URL"],
                rpc_token=environment["KAZUSA_DSH_RPC_TOKEN"],
            )
            latest = {
                "brain": brain_probe,
                "sidecar": sidecar_probe,
                "duration_ms": round((time.perf_counter() - started_at) * 1000),
            }
            if brain_probe.get("ready") and sidecar_probe.get("ready"):
                _write_json(readiness_path, latest)
                return latest
            await asyncio.sleep(0.25)
    _write_json(readiness_path, latest)
    names = ", ".join(child_by_name)
    raise AssertionError(f"Plan 3 child readiness timed out for {names}")


def _require_plan3_inline_live_backend() -> None:
    """Require the shared live route and the Plan 3 audit stores."""

    _require_live_backend()
    required = (
        "KAZUSA_DSH_DATA_ROOT",
        "KAZUSA_CONTROL_BRAIN_SHARED_SECRET",
        "CHARACTER_GLOBAL_USER_ID",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(
            "Plan 3 inline live configuration is missing: "
            f"{', '.join(missing)}"
        )
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        pytest.fail("Plan 3 inline live test database guard is not enabled")
    configured_database_name = os.environ.get("MONGODB_DB_NAME", "").strip()
    if not configured_database_name.startswith(EPHEMERAL_TEST_DATABASE_PREFIX):
        pytest.fail(
            "Plan 3 inline live test requires a reserved ephemeral database "
            f"with prefix {EPHEMERAL_TEST_DATABASE_PREFIX!r}"
        )


def _write_json(path: Path, value: object) -> None:
    """Write one UTF-8 inspectable live-run artifact."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=str,
    )
    path.write_text(serialized, encoding="utf-8")


def _read_plan3_dsh_events(
    *,
    data_root: str,
    resolution_thread_id: str,
    segment_id: str,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read the exact DSH root session through SQLite read-only mode."""

    import zstandard

    from agentic_resolver.contracts import DSH_RELEASE

    if not resolution_thread_id.strip() or not segment_id.strip():
        pytest.fail("DSH session identity is incomplete for the live audit")
    data_root = data_root.strip()
    if not data_root:
        pytest.fail("KAZUSA_DSH_DATA_ROOT is unavailable for the live audit")
    session_identity = f"{resolution_thread_id}\0{segment_id}".encode()
    session_id = "kazusa-resolution-" + sha256(session_identity).hexdigest()[:32]
    store_path = Path(data_root) / "dsh" / DSH_RELEASE / "sessions.sqlite"
    if not store_path.is_file():
        pytest.fail("the DSH session store is unavailable for the live audit")

    uri = f"file:{store_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        session_rows = connection.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchall()
        if session_rows != [(session_id,)]:
            pytest.fail("the DSH identity did not resolve one root session")
        event_rows = connection.execute(
            "SELECT seq, type, data FROM events "
            "WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()

    decoder = zstandard.ZstdDecompressor()
    events: list[tuple[int, str, dict[str, Any]]] = []
    for sequence, event_type, raw_data in event_rows:
        event_data: object = raw_data
        if isinstance(event_data, bytes):
            try:
                event_data = decoder.decompress(event_data).decode("utf-8")
            except (UnicodeDecodeError, zstandard.ZstdError) as exc:
                pytest.fail(
                    f"DSH {event_type} at sequence {sequence} cannot be decoded: "
                    f"{exc}"
                )
        if not isinstance(event_data, str):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not JSON text"
            )
        try:
            parsed = json.loads(event_data)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is malformed: {exc}"
            )
        if not isinstance(parsed, dict):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not an object"
            )
        events.append((sequence, event_type, parsed))
    return events


async def _read_plan3_trace_documents(
    database: Any,
    trace_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Read one protected trace run and its ordered raw step rows."""

    run_value = await database["llm_trace_runs"].find_one(
        {"trace_id": trace_id},
        {"_id": 0},
    )
    trace_run = dict(run_value) if isinstance(run_value, Mapping) else None
    cursor = (
        database["llm_trace_steps"]
        .find({"trace_id": trace_id}, {"_id": 0})
        .sort("sequence", 1)
    )
    step_rows = await cursor.to_list(length=None)
    trace_steps = [dict(row) for row in step_rows if isinstance(row, Mapping)]
    return trace_run, trace_steps


async def _wait_for_plan3_trace_finalization(
    database: Any,
    trace_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Wait for the post-response service trace finalizer to settle one run."""

    deadline = time.perf_counter() + timeout_seconds
    latest: Mapping[str, Any] | None = None
    while time.perf_counter() < deadline:
        row = await database["llm_trace_runs"].find_one(
            {"trace_id": trace_id},
            {"_id": 0},
        )
        if isinstance(row, Mapping):
            latest = row
            if str(row.get("status") or "") != "running":
                return dict(row)
        await asyncio.sleep(0.2)
    if latest is None:
        raise AssertionError("Plan 3 live chat did not persist its trace run")
    raise AssertionError(
        "Plan 3 live chat trace remained running: "
        f"{latest.get('status', '')}"
    )


async def _capture_plan3_background_delivery_traces(
    database: Any,
    *,
    background_work_job_id: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Capture finalized child traces for every delivery attempt of one job.

    Args:
        database: The guarded Mongo database that still contains live trace rows.
        background_work_job_id: The exact accepted background-work identity.

    Returns:
        Ordered companion trace artifacts and any capture errors by child trace.
    """

    if not background_work_job_id:
        companion_traces: list[dict[str, Any]] = []
        capture_errors: list[str] = []
        return companion_traces, capture_errors

    try:
        trace_rows = await _plan3_read_rows(
            database,
            "llm_trace_runs",
            {"source_background_work_job_id": background_work_job_id},
        )
    except _PLAN3_EVIDENCE_ERRORS as exc:
        companion_traces = []
        capture_errors = [
            f"companion_trace_lookup:{exc.__class__.__name__}: {exc}"
        ]
        return companion_traces, capture_errors

    ordered_trace_rows = sorted(
        trace_rows,
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("trace_id") or ""),
        ),
    )
    companion_traces = []
    capture_errors = []
    for trace_row in ordered_trace_rows:
        candidate_trace_id = trace_row.get("trace_id")
        if not isinstance(candidate_trace_id, str) or not candidate_trace_id.strip():
            capture_errors.append("companion_trace: missing trace_id")
            continue
        trace_id = candidate_trace_id.strip()
        try:
            trace_run = await _wait_for_plan3_trace_finalization(
                database,
                trace_id,
                timeout_seconds=30.0,
            )
            _, trace_steps = await _read_plan3_trace_documents(database, trace_id)
        except _PLAN3_EVIDENCE_ERRORS as exc:
            capture_errors.append(
                f"companion_trace:{trace_id}:{exc.__class__.__name__}: {exc}"
            )
            continue
        companion_traces.append(
            {
                "trace_id": trace_id,
                "trace_run": trace_run,
                "trace_steps": trace_steps,
            }
        )
    return companion_traces, capture_errors


async def _wait_for_plan3_deliveries(
    adapter: _Plan3LoopbackAdapterServer,
    *,
    count: int,
    channel_id: str | None = None,
    timeout_seconds: float = 420.0,
) -> list[dict[str, Any]]:
    """Wait for exactly the requested scoped adapter deliveries."""

    deadline = time.perf_counter() + timeout_seconds
    latest: list[dict[str, Any]] = []
    while time.perf_counter() < deadline:
        latest = [
            payload
            for payload in adapter.delivery_payloads()
            if channel_id is None or payload.get("channel_id") == channel_id
        ]
        if len(latest) >= count:
            return latest
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"expected {count} adapter deliveries, observed {len(latest)}"
    )


async def _wait_for_plan3_scoped_rows(
    database: Any,
    *,
    collection_name: str,
    query: Mapping[str, object],
    minimum_count: int = 1,
    timeout_seconds: float = 420.0,
) -> list[dict[str, Any]]:
    """Poll one Mongo collection for the exact isolated run scope."""

    deadline = time.perf_counter() + timeout_seconds
    latest: list[dict[str, Any]] = []
    while time.perf_counter() < deadline:
        cursor = database[collection_name].find(dict(query), {"_id": 0})
        rows = await cursor.to_list(length=None)
        latest = [dict(row) for row in rows if isinstance(row, Mapping)]
        if len(latest) >= minimum_count:
            return latest
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"collection {collection_name!r} did not produce "
        f"{minimum_count} scoped row(s); observed {len(latest)}"
    )


def _plan3_artifact_directory(case_id: str, suffix: str) -> tuple[Path, str]:
    """Create one durable directory for a single real-LLM acceptance case."""

    artifact_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = (
        PROJECT_ROOT
        / "test_artifacts"
        / "dsh_plan3_e2e"
        / f"{case_id}_{artifact_stamp}_{suffix[:8]}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir, artifact_stamp


async def _start_plan3_services(
    *,
    tmp_path: Path,
    artifact_dir: Path,
    artifact_stamp: str,
    environment: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[_Plan3ChildProcess], dict[str, object]]:
    """Start the configured local Brain and pinned DSH sidecar as owned children."""

    if environment is None:
        environment = _runtime_environment(tmp_path)
    await _seed_plan3_character_profile(environment)
    brain_url = environment["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    sidecar_url = environment["KAZUSA_DSH_SIDECAR_URL"]
    brain_hostname, brain_port = _loopback_endpoint(
        "KAZUSA_DSH_BRAIN_URL",
        brain_url,
    )
    sidecar_hostname, sidecar_port = _loopback_endpoint(
        "KAZUSA_DSH_SIDECAR_URL",
        sidecar_url,
        required_path="/rpc",
    )
    if brain_port == sidecar_port:
        raise AssertionError("Brain and sidecar must use distinct configured ports")
    _assert_port_available("Brain", brain_hostname, brain_port)
    _assert_port_available("sidecar", sidecar_hostname, sidecar_port)
    node_executable = shutil.which("node")
    if node_executable is None:
        raise AssertionError("the pinned sidecar requires node on PATH")
    if not SIDECAR_ENTRY.is_file():
        raise AssertionError(f"pinned sidecar build is missing: {SIDECAR_ENTRY}")
    if not PYTHON_EXECUTABLE.is_file():
        raise AssertionError(
            f"pinned Python executable is missing: {PYTHON_EXECUTABLE}"
        )

    child_processes: list[_Plan3ChildProcess] = []
    try:
        sidecar_child = await _launch_plan3_child(
            name="sidecar",
            command=[node_executable, str(SIDECAR_ENTRY)],
            environment=environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(sidecar_child)
        brain_child = await _launch_plan3_child(
            name="brain",
            command=[
                str(PYTHON_EXECUTABLE),
                "-m",
                "uvicorn",
                "kazusa_ai_chatbot.service:app",
                "--host",
                brain_hostname,
                "--port",
                str(brain_port),
            ],
            environment=environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(brain_child)
        _write_json(
            artifact_dir / "processes.json",
            [_process_artifact(child) for child in child_processes],
        )
        readiness = await _wait_for_plan3_readiness(
            children=child_processes,
            environment=environment,
            readiness_path=artifact_dir / "readiness.json",
        )
    except BaseException as exc:
        _write_json(
            artifact_dir / "startup_failure.json",
            {
                "error_class": exc.__class__.__name__,
                "error": str(exc),
                "processes": [
                    _process_artifact(child) for child in child_processes
                ],
            },
        )
        for child in reversed(child_processes):
            await _stop_plan3_child(child)
        raise
    return environment, child_processes, readiness


def _plan3_chat_request(
    *,
    platform: str,
    channel_id: str,
    user_id: str,
    bot_id: str,
    display_name: str,
    channel_name: str,
    message_id: str,
    text: str,
    no_remember: bool,
) -> dict[str, object]:
    """Build one ordinary public chat request for a stable test identity."""

    return {
        "platform": platform,
        "platform_channel_id": channel_id,
        "channel_type": "private",
        "platform_message_id": message_id,
        "platform_user_id": user_id,
        "platform_bot_id": bot_id,
        "display_name": display_name,
        "channel_name": channel_name,
        "content_type": "text",
        "message_envelope": {
            "body_text": text,
            "raw_wire_text": text,
            "mentions": [],
            "reply": None,
            "attachments": [],
            "addressed_to_global_user_ids": [bot_id],
            "broadcast": False,
        },
        "local_timestamp": build_turn_clock()["local_timestamp"],
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": no_remember,
        },
    }


async def _post_plan3_chat(
    client: httpx.AsyncClient,
    *,
    brain_url: str,
    control_secret: str,
    request_payload: Mapping[str, object],
) -> dict[str, Any]:
    """Send one real public chat request and return its typed JSON object."""

    response = await client.post(
        f"{brain_url.rstrip('/')}/chat",
        headers={
            "X-Kazusa-Control-Console": "debug-v1",
            "X-Kazusa-Control-Console-Auth": control_secret,
        },
        json=dict(request_payload),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, Mapping)
    response_payload = dict(payload)
    return response_payload


def _plan3_visible_text(response_payload: Mapping[str, object]) -> str:
    """Join non-empty Brain messages for human inspection and grounding checks."""

    messages = response_payload.get("messages")
    assert isinstance(messages, list) and messages
    visible_text = "\n".join(
        item.strip()
        for item in messages
        if isinstance(item, str) and item.strip()
    )
    assert visible_text
    return visible_text


async def _plan3_read_rows(
    database: Any,
    collection_name: str,
    query: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Read one exact scoped Mongo collection without changing live state."""

    cursor = database[collection_name].find(dict(query), {"_id": 0})
    values = await cursor.to_list(length=None)
    rows = [dict(value) for value in values if isinstance(value, Mapping)]
    return rows


def _plan3_event_evidence(
    event: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Extract validated-looking evidence receipts from one DSH event shape."""

    containers: list[Mapping[str, Any]] = []
    for key in ("meta", "data"):
        value = event.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    candidates: list[object] = []
    for container in containers:
        candidates.append(container.get("kazusa"))
        evidence_value = container.get("evidence")
        if isinstance(evidence_value, list):
            candidates.extend(evidence_value)
        kazusa_value = container.get("kazusa")
        if isinstance(kazusa_value, Mapping):
            nested_evidence = kazusa_value.get("evidence")
            if isinstance(nested_evidence, list):
                candidates.extend(nested_evidence)
    receipts = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("schema_version") == "evidence_receipt.v2"
    ]
    return receipts


def _plan3_authoritative_response_trace_id(
    response: Mapping[str, object],
) -> str:
    """Return the public debug trace identity before outcome assertions."""

    trace_value = response.get("trace_id")
    if isinstance(trace_value, str) and trace_value.strip():
        return trace_value.strip()
    operational_error = response.get("operational_error")
    if not isinstance(operational_error, Mapping):
        return ""
    error_trace_value = operational_error.get("trace_id")
    if isinstance(error_trace_value, str) and error_trace_value.strip():
        return error_trace_value.strip()
    return ""


def _plan3_turn_trace_query(
    *,
    platform: str,
    platform_channel_id: str,
    platform_message_id: str,
) -> dict[str, str]:
    """Build the stored trace-run identity for one public input turn."""

    return {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "platform_message_id": platform_message_id,
    }


def _plan3_turn_001_pending_query(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    source_message_id: str,
) -> dict[str, str]:
    """Build the canonical action-attempt scope for one open HIL row."""

    return {
        "action_kind": RESOLVER_PENDING_HIL_ACTION_KIND,
        "trigger_id": source_message_id,
        "target_scope.platform": platform,
        "target_scope.platform_channel_id": platform_channel_id,
        "target_scope.global_user_id": global_user_id,
        "target_scope.source_message_id": source_message_id,
    }


def _plan3_validate_turn_001_pending_row(
    row: Mapping[str, object],
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    source_message_id: str,
    original_goal: str,
) -> dict[str, Any]:
    """Validate the durable selection clarification before the answer turn."""

    assert row.get("action_kind") == RESOLVER_PENDING_HIL_ACTION_KIND
    assert row.get("trigger_id") == source_message_id
    assert row.get("validation_status") == "accepted"
    assert row.get("status") == "waiting_for_user"
    assert row.get("continuation_status") == "waiting_for_user"
    target_scope = row.get("target_scope")
    assert isinstance(target_scope, Mapping)
    assert target_scope == {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "global_user_id": global_user_id,
        "source_message_id": source_message_id,
    }
    pending_resume = row.get("resolver_pending_resume")
    assert isinstance(pending_resume, Mapping)
    validated_pending = validate_resolver_pending_resume(pending_resume)
    assert validated_pending["schema_version"] == RESOLVER_PENDING_RESUME_VERSION
    assert validated_pending["capability_kind"] == "human_clarification"
    assert validated_pending["status"] == "waiting_for_user"
    assert validated_pending["platform"] == platform
    assert validated_pending["platform_channel_id"] == platform_channel_id
    assert validated_pending["global_user_id"] == global_user_id
    assert validated_pending["source_message_id"] == source_message_id
    assert validated_pending["prompt_safe_original_goal"] == original_goal
    question = validated_pending["prompt_safe_question"]
    assert "plan3_real_user_e2e/alpha.txt" in question
    assert "plan3_real_user_e2e/beta.txt" in question
    pending_task_continuation = validated_pending["pending_task_continuation"]
    assert pending_task_continuation == {
        "schema_version": PENDING_TASK_CONTINUATION_VERSION,
        "on_answered_clarification": "background_task_admission",
    }
    execution_result = row.get("execution_result")
    assert isinstance(execution_result, Mapping)
    assert execution_result.get("status") == "waiting_for_user"
    assert execution_result.get("pending_resume") == validated_pending
    return dict(validated_pending)


async def _capture_plan3_failure_evidence(
    *,
    artifact_dir: Path,
    environment: Mapping[str, str] | None,
    database: Any | None,
    trace_id: str,
    trace_query: Mapping[str, object] | None = None,
    binding_query: Mapping[str, object] | None = None,
    mongo_queries: Mapping[str, Mapping[str, object]] | None = None,
    dsh_identity: tuple[str, str] | None = None,
    known_binding: Mapping[str, object] | None = None,
    known_dsh_events: list[tuple[int, str, dict[str, Any]]] | None = None,
    known_trace_run: Mapping[str, Any] | None = None,
    known_trace_steps: list[dict[str, Any]] | None = None,
    callback_payloads: list[dict[str, Any]] | None = None,
    background_work_job_id: str | None = None,
    turn_001_trace_id: str = "",
    known_turn_001_trace_run: Mapping[str, Any] | None = None,
    known_turn_001_trace_steps: list[dict[str, Any]] | None = None,
    turn_001_pending_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Snapshot live evidence while Brain remains alive after an assertion."""

    snapshot: dict[str, Any] = {
        "database_name": (
            environment.get("MONGODB_DB_NAME", "")
            if environment is not None
            else ""
        ),
        "trace_id": trace_id,
        "trace_run": dict(known_trace_run) if known_trace_run else None,
        "trace_steps": list(known_trace_steps or []),
        "turn_001": {
            "trace_id": turn_001_trace_id,
            "trace_run": (
                dict(known_turn_001_trace_run)
                if known_turn_001_trace_run
                else None
            ),
            "trace_steps": list(known_turn_001_trace_steps or []),
            "pending_rows": list(turn_001_pending_rows or []),
        },
        "companion_traces": [],
        "bindings": [],
        "dsh_events": list(known_dsh_events or []),
        "mongo_state": {},
        "callbacks": list(callback_payloads or []),
        "capture_errors": [],
    }
    active_database = database
    if active_database is None and environment is not None:
        try:
            database_name = environment.get("MONGODB_DB_NAME", "").strip()
            _bind_plan3_parent_database(database_name)
            from kazusa_ai_chatbot.db._client import get_db

            active_database = await get_db()
        except _PLAN3_EVIDENCE_ERRORS as exc:
            snapshot["capture_errors"].append(
                f"database:{exc.__class__.__name__}: {exc}"
            )
    if active_database is not None:
        if binding_query is not None:
            try:
                binding_rows = await _plan3_read_rows(
                    active_database,
                    "dsh_task_bindings",
                    binding_query,
                )
                snapshot["bindings"] = binding_rows
            except _PLAN3_EVIDENCE_ERRORS as exc:
                snapshot["capture_errors"].append(
                    f"bindings:{exc.__class__.__name__}: {exc}"
                )
        if known_binding is not None and not snapshot["bindings"]:
            snapshot["bindings"] = [dict(known_binding)]
        if mongo_queries is not None:
            for collection_name, query in mongo_queries.items():
                try:
                    snapshot["mongo_state"][collection_name] = (
                        await _plan3_read_rows(
                            active_database,
                            collection_name,
                            query,
                        )
                    )
                except _PLAN3_EVIDENCE_ERRORS as exc:
                    snapshot["capture_errors"].append(
                        f"{collection_name}:{exc.__class__.__name__}: {exc}"
                    )
        if not trace_id and trace_query is not None:
            try:
                trace_rows = await _plan3_read_rows(
                    active_database,
                    "llm_trace_runs",
                    trace_query,
                )
            except _PLAN3_EVIDENCE_ERRORS as exc:
                trace_rows = []
                snapshot["capture_errors"].append(
                    f"trace_lookup:{exc.__class__.__name__}: {exc}"
                )
            for row in trace_rows:
                candidate = row.get("trace_id")
                if isinstance(candidate, str) and candidate.strip():
                    trace_id = candidate.strip()
                    break
        snapshot["trace_id"] = trace_id
        if trace_id:
            try:
                snapshot["trace_run"] = await _wait_for_plan3_trace_finalization(
                    active_database,
                    trace_id,
                    timeout_seconds=30.0,
                )
                _, snapshot["trace_steps"] = await _read_plan3_trace_documents(
                    active_database,
                    trace_id,
                )
            except _PLAN3_EVIDENCE_ERRORS as exc:
                snapshot["capture_errors"].append(
                    f"trace:{exc.__class__.__name__}: {exc}"
                )
        companion_traces, companion_errors = (
            await _capture_plan3_background_delivery_traces(
                active_database,
                background_work_job_id=background_work_job_id,
            )
        )
        snapshot["companion_traces"] = companion_traces
        snapshot["capture_errors"].extend(companion_errors)
        resolved_identity = dsh_identity
        if resolved_identity is None and snapshot["bindings"]:
            candidate_binding = snapshot["bindings"][0]
            if isinstance(candidate_binding, Mapping):
                thread_id = candidate_binding.get("resolution_thread_id")
                segment_id = candidate_binding.get("segment_id")
                if isinstance(thread_id, str) and isinstance(segment_id, str):
                    resolved_identity = (thread_id, segment_id)
        if resolved_identity is not None and environment is not None:
            try:
                snapshot["dsh_events"] = _read_plan3_dsh_events(
                    data_root=environment["KAZUSA_DSH_DATA_ROOT"],
                    resolution_thread_id=resolved_identity[0],
                    segment_id=resolved_identity[1],
                )
            except _PLAN3_EVIDENCE_ERRORS as exc:
                snapshot["capture_errors"].append(
                    f"dsh_events:{exc.__class__.__name__}: {exc}"
                )
    _write_json(artifact_dir / "failure_evidence.json", snapshot)
    _write_json(artifact_dir / "mongo_state.json", snapshot["mongo_state"])
    _write_json(
        artifact_dir / "dsh_lineage.json",
        {
            "bindings": snapshot["bindings"],
            "events": snapshot["dsh_events"],
        },
    )
    _write_json(
        artifact_dir / "trace.json",
        {
            "trace_id": snapshot["trace_id"],
            "trace_run": snapshot["trace_run"],
            "trace_steps": snapshot["trace_steps"],
            "companion_traces": snapshot["companion_traces"],
            "turn_001": snapshot["turn_001"],
        },
    )
    return snapshot


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_e2e_inline_grounded_resolution_reenters_full_cognition(
    tmp_path: Path,
) -> None:
    """Run one grounded inline task through the full cognition loop in P3-P3."""

    _require_plan3_inline_live_backend()
    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    marker = f"DSH_PLAN3_INLINE_{suffix}"
    platform = f"dsh-plan3-inline-{suffix}"
    platform_channel_id = f"dsh-plan3-inline-channel-{suffix}"
    platform_user_id = f"dsh-plan3-inline-user-{suffix}"
    platform_message_id = f"dsh-plan3-inline-message-{suffix}"
    runtime_environment: dict[str, str] | None = None

    try:
        runtime_environment = _runtime_environment(tmp_path)
        await _seed_plan3_character_profile(runtime_environment)
        brain_url = runtime_environment["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
        sidecar_url = runtime_environment["KAZUSA_DSH_SIDECAR_URL"]
        brain_hostname, brain_port = _loopback_endpoint(
            "KAZUSA_DSH_BRAIN_URL",
            brain_url,
        )
        sidecar_hostname, sidecar_port = _loopback_endpoint(
            "KAZUSA_DSH_SIDECAR_URL",
            sidecar_url,
            required_path="/rpc",
        )
        if brain_port == sidecar_port:
            raise AssertionError(
                "Brain and sidecar must use distinct configured ports"
            )
        _assert_port_available("Brain", brain_hostname, brain_port)
        _assert_port_available("sidecar", sidecar_hostname, sidecar_port)
        node_executable = shutil.which("node")
        if node_executable is None:
            raise AssertionError("the pinned sidecar requires node on PATH")
        if not SIDECAR_ENTRY.is_file():
            raise AssertionError(f"pinned sidecar build is missing: {SIDECAR_ENTRY}")
        if not PYTHON_EXECUTABLE.is_file():
            raise AssertionError(
                f"pinned Python executable is missing: {PYTHON_EXECUTABLE}"
            )
    except _PLAN3_EVIDENCE_ERRORS as exc:
        startup_artifact_dir, _ = _plan3_artifact_directory(
            "inline_grounded_resolution_startup",
            suffix,
        )
        _write_json(
            startup_artifact_dir / "startup_failure.json",
            {
                "schema_version": "dsh_plan3_inline_startup_failure.v1",
                "case_id": "inline_grounded_resolution_reenters_full_cognition",
                "marker": marker,
                "brain_url": os.environ.get("KAZUSA_DSH_BRAIN_URL"),
                "sidecar_url": os.environ.get("KAZUSA_DSH_SIDECAR_URL"),
                "error_class": exc.__class__.__name__,
                "error": str(exc),
                "database_name": (
                    runtime_environment.get("MONGODB_DB_NAME", "")
                    if runtime_environment is not None
                    else ""
                ),
                "processes_started": [],
            },
        )
        cleanup_disposition = "not_attempted"
        if runtime_environment is not None:
            try:
                await _drop_plan3_database(runtime_environment)
            except _PLAN3_EVIDENCE_ERRORS as cleanup_error:
                cleanup_disposition = (
                    f"failed:{cleanup_error.__class__.__name__}"
                )
            else:
                cleanup_disposition = "dropped"
        _write_json(
            startup_artifact_dir / "run.json",
            {
                "case_id": "inline_grounded_resolution_reenters_full_cognition",
                "error_class": exc.__class__.__name__,
                "error": str(exc),
                "database_cleanup": cleanup_disposition,
                "processes": [],
            },
        )
        (startup_artifact_dir / "behavior_audit_conclusions.md").write_text(
            "# Plan 3 inline startup failure\n\n"
            f"- Error: `{exc.__class__.__name__}`\n"
            f"- Database cleanup: `{cleanup_disposition}`\n",
            encoding="utf-8",
        )
        print(f"DSH_PLAN3_INLINE_STARTUP_ARTIFACT={startup_artifact_dir}")
        raise

    assert runtime_environment is not None
    character_global_user_id = runtime_environment["CHARACTER_GLOBAL_USER_ID"].strip()
    workspace_root = Path(
        runtime_environment["AGENTIC_RESOLVER_WORKSPACE_ROOT"].strip()
    ).resolve()

    fixture_parent = workspace_root / "plan3-live-inline" / f"case-{suffix}"
    fixture_path = fixture_parent / "fact.txt"
    fixture_parent.mkdir(parents=True, exist_ok=False)
    fixture_content = f"Plan 3 inline evidence marker: {marker}\n"
    fixture_path.write_text(fixture_content, encoding="utf-8")
    relative_fixture_path = fixture_path.relative_to(workspace_root).as_posix()
    brain_conversation_ref = (
        f"user_message:{platform}:{platform_channel_id}:{platform_message_id}"
    )
    user_text = (
        "Use your native DSH workspace task to inspect the file "
        f"{relative_fixture_path}, read it as evidence, and submit one "
        "terminal resolution containing the exact marker "
        f"{marker}. Then answer me with the verified result, keeping the "
        "visible wording grounded in the submitted evidence."
    )
    request_payload: dict[str, object] = {
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": "private",
        "platform_message_id": platform_message_id,
        "platform_user_id": platform_user_id,
        "platform_bot_id": character_global_user_id,
        "display_name": f"Plan 3 inline user {suffix[:8]}",
        "channel_name": f"Plan 3 inline {suffix[:8]}",
        "content_type": "text",
        "message_envelope": {
            "body_text": user_text,
            "raw_wire_text": user_text,
            "mentions": [],
            "reply": None,
            "attachments": [],
            "addressed_to_global_user_ids": [character_global_user_id],
            "broadcast": False,
        },
        "local_timestamp": build_turn_clock()["local_timestamp"],
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": True,
        },
    }
    artifact_dir, artifact_stamp = _plan3_artifact_directory(
        "inline_grounded_resolution",
        suffix,
    )
    artifact_payload: dict[str, object] = {
        "schema_version": "dsh_plan3_inline_live_artifact.v1",
        "case_id": "inline_grounded_resolution_reenters_full_cognition",
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "platform_user_id": platform_user_id,
        "platform_message_id": platform_message_id,
        "brain_conversation_ref": brain_conversation_ref,
        "workspace_relative_fixture": relative_fixture_path,
        "fixture_content": fixture_content,
        "marker": marker,
        "request": request_payload,
        "observation_target": {
            "terminal_dsh_evidence": True,
            "task_resolution_result_v1": True,
            "full_cognition_recurrence": True,
            "brain_owned_dialog": True,
            "exact_thread_segment_session_lineage": True,
        },
        "startup": {
            "brain_url": brain_url,
            "brain_port": brain_port,
            "sidecar_url": sidecar_url,
            "sidecar_port": sidecar_port,
            "mongo_database_name": runtime_environment["MONGODB_DB_NAME"],
            "dsh_data_root": runtime_environment["KAZUSA_DSH_DATA_ROOT"],
            "workspace_root": runtime_environment[
                "AGENTIC_RESOLVER_WORKSPACE_ROOT"
            ],
        },
    }
    _write_json(artifact_dir / "turn_001_request.json", request_payload)
    response_payload: object = None
    health_payload: object = None
    binding: Mapping[str, object] | None = None
    terminal_events: list[dict[str, Any]] = []
    dsh_events: list[tuple[int, str, dict[str, Any]]] = []
    trace_run: dict[str, Any] | None = None
    trace_steps: list[dict[str, Any]] = []
    trace_id = ""
    http_status: int | None = None
    readiness_payload: dict[str, object] | None = None
    child_processes: list[_Plan3ChildProcess] = []
    database: Any | None = None
    visible_text = ""
    started_at = time.perf_counter()

    try:
        brain_secret = runtime_environment["KAZUSA_DSH_BRAIN_SHARED_SECRET"].strip()
        control_secret = runtime_environment[
            "KAZUSA_CONTROL_BRAIN_SHARED_SECRET"
        ].strip()
        sidecar_child = await _launch_plan3_child(
            name="sidecar",
            command=[node_executable, str(SIDECAR_ENTRY)],
            environment=runtime_environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(sidecar_child)
        _write_json(
            artifact_dir / "processes.json",
            [_process_artifact(child) for child in child_processes],
        )
        brain_child = await _launch_plan3_child(
            name="brain",
            command=[
                str(PYTHON_EXECUTABLE),
                "-m",
                "uvicorn",
                "kazusa_ai_chatbot.service:app",
                "--host",
                brain_hostname,
                "--port",
                str(brain_port),
            ],
            environment=runtime_environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(brain_child)
        _write_json(
            artifact_dir / "processes.json",
            [_process_artifact(child) for child in child_processes],
        )
        readiness_payload = await _wait_for_plan3_readiness(
            children=child_processes,
            environment=runtime_environment,
            readiness_path=artifact_dir / "readiness.json",
        )
        artifact_payload["readiness"] = readiness_payload
        timeout = httpx.Timeout(
            600.0,
            connect=15.0,
            write=15.0,
            pool=15.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            health_response = await client.get(
                f"{brain_url}/runtime/dsh/health",
                headers={"Authorization": f"Bearer {brain_secret}"},
            )
            health_payload = health_response.json()
            artifact_payload["health_http_status"] = health_response.status_code
            artifact_payload["health"] = health_payload
            assert health_response.status_code == 200
            assert isinstance(health_payload, Mapping)
            assert health_payload.get("status") == "ready"
            assert health_payload.get("configured") is True
            assert health_payload.get("durable_store") is True
            assert health_payload.get("cognition_judge") is True
            task_health = health_payload.get("task_resolution")
            assert isinstance(task_health, Mapping)
            assert task_health.get("status") == "ready"

            turn_started_at = time.perf_counter()
            chat_response = await client.post(
                f"{brain_url}/chat",
                headers={
                    "X-Kazusa-Control-Console": "debug-v1",
                    "X-Kazusa-Control-Console-Auth": control_secret,
                },
                json=request_payload,
            )
            http_status = chat_response.status_code
            turn_duration_ms = round(
                (time.perf_counter() - turn_started_at) * 1000
            )
            artifact_payload["turn_001_duration_ms"] = turn_duration_ms
            assert chat_response.status_code == 200
            response_payload = chat_response.json()
            assert isinstance(response_payload, Mapping)
            artifact_payload["response"] = response_payload

        assert isinstance(response_payload, Mapping)
        assert response_payload.get("content_type") == "text"
        assert response_payload.get("operational_error") is None
        messages = response_payload.get("messages")
        assert isinstance(messages, list) and messages
        visible_text = "\n".join(
            item.strip()
            for item in messages
            if isinstance(item, str) and item.strip()
        )
        assert visible_text
        assert marker in visible_text
        assert response_payload.get("trace_id") == ""

        graph = response_payload.get("cognition_graph")
        assert isinstance(graph, Mapping)
        assert graph.get("schema_version") == "cognition_run_observation.v1"
        assert graph.get("run_kind") == "live_turn"
        assert graph.get("status") == "completed"
        graph_correlation = graph.get("correlation")
        assert isinstance(graph_correlation, Mapping)
        trace_value = graph_correlation.get("llm_trace_id")
        assert isinstance(trace_value, str) and trace_value.strip()
        trace_id = trace_value
        sections = graph.get("sections")
        assert isinstance(sections, list)
        section_ids = {
            section.get("section_id")
            for section in sections
            if isinstance(section, Mapping)
        }
        assert {
            "action.requests",
            "action.results",
            "action.continuation",
        } <= section_ids

        database = await get_db()
        binding_rows = await database["dsh_task_bindings"].find(
            {
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": platform_channel_id,
                "source_scope.source_message_id": platform_message_id,
            },
            {"_id": 0},
        ).to_list(length=None)
        assert len(binding_rows) == 1
        binding = dict(binding_rows[0])
        assert binding.get("state") == "consumed_inline"
        assert binding.get("operation_generation") == 0
        assert binding.get("current_accepted_task_id") is None
        assert binding.get("current_background_work_job_id") is None

        task_session_id = binding.get("task_session_id")
        resolution_thread_id = binding.get("resolution_thread_id")
        segment_id = binding.get("segment_id")
        assert isinstance(task_session_id, str) and task_session_id.strip()
        assert isinstance(resolution_thread_id, str)
        assert resolution_thread_id.strip()
        assert isinstance(segment_id, str) and segment_id.strip()
        resolution_ref = binding.get("resolution_ref")
        assert isinstance(resolution_ref, Mapping)
        assert resolution_ref.get("resolution_thread_id") == resolution_thread_id
        assert resolution_ref.get("segment_id") == segment_id
        assert resolution_ref.get("dsh_session_id") == task_session_id
        for identity_key in (
            "activation_id",
            "lease_epoch",
            "document_revision",
            "last_committed_seq",
        ):
            assert identity_key in resolution_ref

        source_scope = binding.get("source_scope")
        assert isinstance(source_scope, Mapping)
        assert source_scope.get("platform") == platform
        assert source_scope.get("channel_id") == platform_channel_id
        assert source_scope.get("source_message_id") == platform_message_id
        requester_global_user_id = source_scope.get("requester_global_user_id")
        assert isinstance(requester_global_user_id, str)
        assert requester_global_user_id.strip()

        start_spec = binding.get("start_spec")
        assert isinstance(start_spec, Mapping)
        resolver_request = start_spec.get("resolver_request")
        assert isinstance(resolver_request, Mapping)
        assert resolver_request.get("capability") == "task_resolution_request"
        assert resolver_request.get("start_in_background") is False
        semantic_goal = resolver_request.get("semantic_goal")
        assert isinstance(semantic_goal, str) and semantic_goal.strip()
        execution_context = start_spec.get("execution_context")
        assert isinstance(execution_context, Mapping)
        assert execution_context.get("brain_conversation_ref") == (
            brain_conversation_ref
        )
        assert execution_context.get("source_message_id") == platform_message_id
        active_message_ids = execution_context.get(
            "active_turn_platform_message_ids"
        )
        assert isinstance(active_message_ids, list)
        assert platform_message_id in active_message_ids

        latest_result = binding.get("latest_task_resolution_result")
        assert isinstance(latest_result, Mapping)
        assert latest_result.get("schema_version") == "task_resolution_result.v1"
        assert latest_result.get("status") in {"resolved", "partial"}
        assert latest_result.get("semantic_objective") == semantic_goal
        assert latest_result.get("goal_continuation_ref") == (
            resolver_request.get("goal_continuation_ref")
        )
        assert latest_result.get("evidence_state") in {"complete", "partial"}
        evidence = latest_result.get("evidence")
        evidence_excerpts = latest_result.get("evidence_excerpts")
        evidence_handles = latest_result.get("evidence_handles")
        assert isinstance(evidence, list) and evidence
        assert isinstance(evidence_excerpts, list)
        assert isinstance(evidence_handles, list)
        assert any(
            isinstance(row, Mapping) and row.get("specialist") == "dsh"
            for row in evidence
        )

        dsh_events = _read_plan3_dsh_events(
            data_root=runtime_environment["KAZUSA_DSH_DATA_ROOT"],
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
        )
        tool_calls = [
            event
            for _, event_type, event in dsh_events
            if event_type == "tool/call"
        ]
        tool_results = [
            event
            for _, event_type, event in dsh_events
            if event_type == "tool/result"
        ]
        assert tool_calls
        assert tool_results
        terminal_events = [
            event
            for event in tool_results
            if isinstance(event.get("meta"), Mapping)
            and isinstance(event["meta"].get("kazusa"), Mapping)
            and event["meta"]["kazusa"].get("kind")
            == "terminal_resolution_v2"
        ]
        assert len(terminal_events) == 1
        terminal_events = [dict(event) for event in terminal_events]
        assert marker in json.dumps(terminal_events[0], ensure_ascii=False)
        terminal_receipt = terminal_events[0]["meta"]["kazusa"]
        assert terminal_receipt.get("resolution_thread_id") == (
            resolution_thread_id
        )
        assert terminal_receipt.get("segment_id") == segment_id
        assert terminal_receipt.get("brain_conversation_ref") == (
            brain_conversation_ref
        )
        call_ids = {
            event.get("callId")
            for event in tool_calls
            if isinstance(event.get("callId"), str)
        }
        result_call_ids = {
            item.get("toolCallId")
            for event in tool_results
            for item in (
                event.get("message", {}).get("content", [])
                if isinstance(event.get("message"), Mapping)
                and isinstance(event["message"].get("content"), list)
                else []
            )
            if isinstance(item, Mapping)
            and isinstance(item.get("toolCallId"), str)
        }
        assert call_ids & result_call_ids

        trace_run = await _wait_for_plan3_trace_finalization(
            database,
            trace_id,
        )
        _, trace_steps = await _read_plan3_trace_documents(database, trace_id)
        assert trace_run.get("status") == "succeeded"
        assert trace_run.get("platform") == platform
        assert trace_run.get("platform_channel_id") == platform_channel_id
        assert trace_run.get("platform_message_id") == platform_message_id
        assert trace_run.get("global_user_id") == requester_global_user_id
        assert trace_run.get("final_dialog_count", 0) >= 1
        stage_names = [
            row.get("stage_name")
            for row in trace_steps
            if isinstance(row.get("stage_name"), str)
        ]
        stage_counts = Counter(stage_names)
        required_cognition_stages = (
            "cognition_core_v3.A1",
            "cognition_core_v3.A2",
            "cognition_core_v3.G",
            "cognition_core_v3.P",
        )
        assert all(
            stage_counts[stage_name] >= 2
            for stage_name in required_cognition_stages
        )

        artifact_payload.update({
            "visible_text": visible_text,
            "trace_id": trace_id,
            "binding": binding,
            "dsh_events": dsh_events,
            "trace_run": trace_run,
            "trace_steps": trace_steps,
            "cognition_stage_counts": dict(stage_counts),
        })
        _write_json(artifact_dir / "response.json", response_payload)
        _write_json(
            artifact_dir / "dsh_lineage.json",
            {
                "binding": binding,
                "events": dsh_events,
                "terminal_receipts": terminal_events,
            },
        )
        _write_json(
            artifact_dir / "trace.json",
            {"trace_run": trace_run, "trace_steps": trace_steps},
        )
    except httpx.RequestError as exc:
        artifact_payload["http_error_class"] = exc.__class__.__name__
        pytest.fail(
            "Plan 3 Brain live HTTP request failed: "
            f"{exc.__class__.__name__}"
        )
    finally:
        failure_snapshot = await _capture_plan3_failure_evidence(
            artifact_dir=artifact_dir,
            environment=runtime_environment,
            database=database,
            trace_id=trace_id,
            trace_query={
                "platform": platform,
                "platform_channel_id": platform_channel_id,
                "platform_message_id": platform_message_id,
            },
            binding_query={
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": platform_channel_id,
                "source_scope.source_message_id": platform_message_id,
            },
            mongo_queries={
                "accepted_tasks": {
                    "source_platform": platform,
                    "source_channel_id": platform_channel_id,
                },
                "background_work_jobs": {
                    "source_platform": platform,
                    "source_channel_id": platform_channel_id,
                },
                "dsh_interaction_store": {
                    "brain_conversation_ref": brain_conversation_ref,
                },
            },
            dsh_identity=(
                (str(binding["resolution_thread_id"]), str(binding["segment_id"]))
                if isinstance(binding, Mapping)
                and isinstance(binding.get("resolution_thread_id"), str)
                and isinstance(binding.get("segment_id"), str)
                else None
            ),
            known_binding=binding,
            known_dsh_events=dsh_events,
            known_trace_run=trace_run,
            known_trace_steps=trace_steps,
        )
        if not trace_id and isinstance(failure_snapshot.get("trace_id"), str):
            trace_id = failure_snapshot["trace_id"]
        if trace_run is None and isinstance(failure_snapshot.get("trace_run"), Mapping):
            trace_run = dict(failure_snapshot["trace_run"])
        if not trace_steps and isinstance(failure_snapshot.get("trace_steps"), list):
            trace_steps = list(failure_snapshot["trace_steps"])
        if not dsh_events and isinstance(failure_snapshot.get("dsh_events"), list):
            dsh_events = list(failure_snapshot["dsh_events"])
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        artifact_payload["duration_ms"] = duration_ms
        artifact_payload["http_status"] = http_status
        artifact_payload["health"] = health_payload
        artifact_payload["readiness"] = readiness_payload
        artifact_payload["trace_id"] = trace_id
        artifact_payload["binding"] = binding
        artifact_payload["dsh_events"] = dsh_events
        artifact_payload["trace_run"] = trace_run
        artifact_payload["trace_steps"] = trace_steps
        for child in reversed(child_processes):
            await _stop_plan3_child(child)
        artifact_payload["processes"] = [
            _process_artifact(child) for child in child_processes
        ]
        _write_json(artifact_dir / "run.json", artifact_payload)
        conclusion_lines = [
            "# Plan 3 inline real-LLM gate",
            "",
            f"- Marker: `{marker}`",
            f"- Brain conversation ref: `{brain_conversation_ref}`",
            f"- HTTP status: `{http_status}`",
            f"- Trace id retained: `{bool(trace_id)}`",
            (
                "- Binding terminal and inline: `"
                f"{binding is not None and binding.get('state') == 'consumed_inline'}`"
            ),
            f"- DSH terminal receipt count: `{len(terminal_events)}`",
            (
                "- Terminal evidence contains marker: `"
                f"{marker in json.dumps(terminal_events, ensure_ascii=False)}`"
            ),
            f"- Protected trace step count: `{len(trace_steps)}`",
            (
                "- Qualitative judgment: DSH terminal evidence is inspected "
                "before the V1 binding result; the Brain response is required "
                "to carry the grounded marker; the retained cognition trace "
                "must show two full A1/A2/G/P passes for terminal-result "
                "recurrence."
            ),
        ]
        (artifact_dir / "behavior_audit_conclusions.md").write_text(
            "\n".join(conclusion_lines) + "\n",
            encoding="utf-8",
        )
        cleanup_disposition = "not_attempted"
        if runtime_environment is not None:
            try:
                await _drop_plan3_database(runtime_environment)
            except _PLAN3_EVIDENCE_ERRORS as cleanup_error:
                cleanup_disposition = (
                    f"failed:{cleanup_error.__class__.__name__}"
                )
            else:
                cleanup_disposition = "dropped"
        artifact_payload["database_cleanup"] = cleanup_disposition
        _write_json(artifact_dir / "run.json", artifact_payload)
        with (artifact_dir / "behavior_audit_conclusions.md").open(
            "a",
            encoding="utf-8",
        ) as conclusions_file:
            conclusions_file.write(
                f"- Database cleanup: `{cleanup_disposition}`\n"
            )
        fixture_path.unlink(missing_ok=True)
        try:
            fixture_parent.rmdir()
        except OSError:
            pass
        print(f"DSH_PLAN3_INLINE_ARTIFACT={artifact_dir}")


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_e2e_real_debug_user_prerequisite_is_resolved_before_dsh_admission(
    tmp_path: Path,
) -> None:
    """Prove ordinary clarification precedes beta-only DSH admission."""

    _require_plan3_inline_live_backend()
    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    platform = "debug"
    platform_channel_id = f"plan3-real-user-channel-{suffix}"
    platform_user_id = f"plan3-real-user-{suffix}"
    character_global_user_id = os.environ["CHARACTER_GLOBAL_USER_ID"].strip()
    callback_secret = f"plan3-real-user-callback-{suffix}"
    turn_001_message_id = f"plan3-real-user-turn-001-{suffix}"
    turn_002_message_id = f"plan3-real-user-turn-002-{suffix}"
    alpha_marker = "PLAN3_E2E_ALPHA_NOT_SELECTED"
    beta_marker = "PLAN3_E2E_BETA_SELECTED"
    turn_001_text = (
        "Please handle this in the background. In the task workspace, there "
        "are two files: plan3_real_user_e2e/alpha.txt and "
        "plan3_real_user_e2e/beta.txt. Before opening either file, ask me "
        "which one I want you to summarize. After I answer, read only that "
        "file and report its marker."
    )
    turn_002_text = "Use plan3_real_user_e2e/beta.txt."
    brain_conversation_ref = (
        f"user_message:{platform}:{platform_channel_id}:"
        f"{turn_001_message_id}"
    )
    artifact_dir, artifact_stamp = _plan3_artifact_directory(
        "prerequisite_admission",
        suffix,
    )
    adapter = _Plan3LoopbackAdapterServer(
        platform=platform,
        shared_secret=callback_secret,
    )
    adapter_started = False
    child_processes: list[_Plan3ChildProcess] = []
    response_001: Mapping[str, Any] | None = None
    response_002: Mapping[str, Any] | None = None
    readiness_payload: dict[str, object] | None = None
    registration: Mapping[str, object] | None = None
    final_delivery: dict[str, Any] | None = None
    binding: dict[str, Any] | None = None
    task: dict[str, Any] | None = None
    job: dict[str, Any] | None = None
    job_id = ""
    dsh_events: list[tuple[int, str, dict[str, Any]]] = []
    trace_run: dict[str, Any] | None = None
    trace_steps: list[dict[str, Any]] = []
    trace_id = ""
    turn_001_trace_id = ""
    turn_001_trace_run: dict[str, Any] | None = None
    turn_001_trace_steps: list[dict[str, Any]] = []
    turn_001_pending_query: dict[str, object] = {}
    turn_001_pending_rows: list[dict[str, Any]] = []
    requester_global_user_id = ""
    visible_001 = ""
    task_session_id = ""
    resolution_thread_id = ""
    segment_id = ""
    terminal_events: list[dict[str, Any]] = []
    failure: str | None = None
    artifact_payload: dict[str, object] = {
        "schema_version": "dsh_plan3_prerequisite_artifact.v1",
        "case_id": "real_debug_user_prerequisite_is_resolved_before_dsh_admission",
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "platform_user_id": platform_user_id,
        "brain_conversation_ref": brain_conversation_ref,
        "database_name": "",
        "markers": {"alpha": alpha_marker, "beta": beta_marker},
        "observation_targets": {
            "turn_001": (
                "ordinary character clarification with no task, binding, "
                "interaction, job, or file read"
            ),
            "turn_002": "beta-only DSH execution and one normal final delivery",
        },
    }

    def chat_payload(message_id: str, text: str) -> dict[str, object]:
        """Build one public ChatRequest with no native interaction control."""

        return {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "channel_type": "private",
            "platform_message_id": message_id,
            "platform_user_id": platform_user_id,
            "platform_bot_id": character_global_user_id,
            "display_name": f"Plan 3 real user {suffix[:8]}",
            "channel_name": f"Plan 3 real debug {suffix[:8]}",
            "content_type": "text",
            "message_envelope": {
                "body_text": text,
                "raw_wire_text": text,
                "mentions": [],
                "reply": None,
                "attachments": [],
                "addressed_to_global_user_ids": [character_global_user_id],
                "broadcast": False,
            },
            "local_timestamp": build_turn_clock()["local_timestamp"],
            "debug_modes": {
                "listen_only": False,
                "think_only": False,
                "no_remember": False,
            },
        }

    turn_001_request = chat_payload(turn_001_message_id, turn_001_text)
    turn_002_request = chat_payload(turn_002_message_id, turn_002_text)
    runtime_environment: dict[str, str] | None = None
    fixture_parent: Path | None = None
    alpha_path: Path | None = None
    beta_path: Path | None = None
    database: Any | None = None
    started_at = time.perf_counter()
    try:
        runtime_environment = _runtime_environment(tmp_path)
        artifact_payload["database_name"] = runtime_environment["MONGODB_DB_NAME"]
        await _seed_plan3_character_profile(runtime_environment)
        brain_url = runtime_environment["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
        sidecar_url = runtime_environment["KAZUSA_DSH_SIDECAR_URL"]
        brain_hostname, brain_port = _loopback_endpoint(
            "KAZUSA_DSH_BRAIN_URL",
            brain_url,
        )
        sidecar_hostname, sidecar_port = _loopback_endpoint(
            "KAZUSA_DSH_SIDECAR_URL",
            sidecar_url,
            required_path="/rpc",
        )
        if brain_port == sidecar_port:
            raise AssertionError(
                "Brain and sidecar must use distinct configured ports"
            )
        _assert_port_available("Brain", brain_hostname, brain_port)
        _assert_port_available("sidecar", sidecar_hostname, sidecar_port)
        node_executable = shutil.which("node")
        if node_executable is None:
            raise AssertionError("the pinned sidecar requires node on PATH")
        if not SIDECAR_ENTRY.is_file():
            raise AssertionError(f"pinned sidecar build is missing: {SIDECAR_ENTRY}")
        if not PYTHON_EXECUTABLE.is_file():
            raise AssertionError(
                f"pinned Python executable is missing: {PYTHON_EXECUTABLE}"
            )

        workspace_root = Path(
            runtime_environment["AGENTIC_RESOLVER_WORKSPACE_ROOT"]
        ).resolve()
        fixture_parent = workspace_root / "plan3_real_user_e2e"
        fixture_parent.mkdir(parents=True, exist_ok=False)
        alpha_path = fixture_parent / "alpha.txt"
        beta_path = fixture_parent / "beta.txt"
        alpha_path.write_text(alpha_marker + "\n", encoding="utf-8")
        beta_path.write_text(beta_marker + "\n", encoding="utf-8")
        database = await get_db()
    except _PLAN3_EVIDENCE_ERRORS as exc:
        failure = f"{exc.__class__.__name__}: {exc}"
        if alpha_path is not None:
            alpha_path.unlink(missing_ok=True)
        if beta_path is not None:
            beta_path.unlink(missing_ok=True)
        if fixture_parent is not None:
            try:
                fixture_parent.rmdir()
            except OSError:
                pass
        artifact_payload.update({
            "failure": failure,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
            "processes": [],
            "callback_deliveries": [],
            "database_cleanup": "not_attempted",
        })
        _write_json(artifact_dir / "run.json", artifact_payload)
        (artifact_dir / "behavior_audit_conclusions.md").write_text(
            "# Plan 3 pre-admission setup failure\n\n"
            f"- Error: `{failure}`\n"
            "- Evidence was unavailable before Brain startup.\n"
            "- Database cleanup: `not_attempted`\n",
            encoding="utf-8",
        )
        cleanup_disposition = "not_attempted"
        if runtime_environment is not None:
            try:
                await _drop_plan3_database(runtime_environment)
            except _PLAN3_EVIDENCE_ERRORS as cleanup_error:
                cleanup_disposition = (
                    f"failed:{cleanup_error.__class__.__name__}"
                )
            else:
                cleanup_disposition = "dropped"
        artifact_payload["database_cleanup"] = cleanup_disposition
        _write_json(artifact_dir / "run.json", artifact_payload)
        with (artifact_dir / "behavior_audit_conclusions.md").open(
            "a",
            encoding="utf-8",
        ) as conclusions_file:
            conclusions_file.write(
                f"- Database cleanup: `{cleanup_disposition}`\n"
            )
        print(f"DSH_PLAN3_PREREQUISITE_ARTIFACT={artifact_dir}")
        raise

    def scoped_query(collection_name: str, source_message_id: str) -> dict[str, object]:
        """Build one exact source-scoped query for the isolated run."""

        if collection_name == "dsh_task_bindings":
            return {
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": platform_channel_id,
                "source_scope.source_message_id": source_message_id,
            }
        if collection_name == "accepted_tasks":
            return {
                "schema_version": "accepted_task.v2",
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
                "first_source_message_id": source_message_id,
            }
        if collection_name == "background_work_jobs":
            return {
                "schema_version": "background_work_job.v2",
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
                "source_message_id": source_message_id,
            }
        raise ValueError(f"unsupported Plan 3 collection: {collection_name}")

    async def rows(
        collection_name: str,
        query: Mapping[str, object],
    ) -> list[dict[str, Any]]:
        """Read one bounded set of rows from the isolated database."""

        cursor = database[collection_name].find(dict(query), {"_id": 0})
        values = await cursor.to_list(length=None)
        return [dict(value) for value in values if isinstance(value, Mapping)]

    started_at = time.perf_counter()
    try:
        _write_json(artifact_dir / "turn_001_request.json", turn_001_request)
        adapter.start()
        adapter_started = True
        sidecar_child = await _launch_plan3_child(
            name="sidecar",
            command=[node_executable, str(SIDECAR_ENTRY)],
            environment=runtime_environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(sidecar_child)
        brain_child = await _launch_plan3_child(
            name="brain",
            command=[
                str(PYTHON_EXECUTABLE),
                "-m",
                "uvicorn",
                "kazusa_ai_chatbot.service:app",
                "--host",
                brain_hostname,
                "--port",
                str(brain_port),
            ],
            environment=runtime_environment,
            artifact_dir=artifact_dir,
            artifact_stamp=artifact_stamp,
        )
        child_processes.append(brain_child)
        _write_json(
            artifact_dir / "processes.json",
            [_process_artifact(child) for child in child_processes],
        )
        readiness_payload = await _wait_for_plan3_readiness(
            children=child_processes,
            environment=runtime_environment,
            readiness_path=artifact_dir / "readiness.json",
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                600.0,
                connect=15.0,
                write=15.0,
                pool=15.0,
            )
        ) as client:
            brain_secret = runtime_environment[
                "KAZUSA_DSH_BRAIN_SHARED_SECRET"
            ].strip()
            control_secret = runtime_environment[
                "KAZUSA_CONTROL_BRAIN_SHARED_SECRET"
            ].strip()
            health_response = await client.get(
                f"{brain_url}/runtime/dsh/health",
                headers={"Authorization": f"Bearer {brain_secret}"},
            )
            health_payload = health_response.json()
            assert health_response.status_code == 200
            assert isinstance(health_payload, Mapping)
            assert health_payload.get("status") == "ready"
            registration_response = await client.post(
                f"{brain_url}/runtime/adapters/register",
                json={
                    "platform": platform,
                    "callback_url": adapter.callback_url,
                    "platform_bot_id": character_global_user_id,
                    "shared_secret": callback_secret,
                    "timeout_seconds": 30.0,
                },
            )
            assert registration_response.status_code == 200
            registration_payload = registration_response.json()
            assert isinstance(registration_payload, Mapping)
            registration = dict(registration_payload)
            turn_001_started_at = time.perf_counter()
            first_response = await client.post(
                f"{brain_url}/chat",
                headers={
                    "X-Kazusa-Control-Console": "debug-v1",
                    "X-Kazusa-Control-Console-Auth": control_secret,
                },
                json=turn_001_request,
            )
            assert first_response.status_code == 200
            first_payload = first_response.json()
            assert isinstance(first_payload, Mapping)
            response_001 = dict(first_payload)
            artifact_payload["turn_001_duration_ms"] = round(
                (time.perf_counter() - turn_001_started_at) * 1000
            )
        _write_json(artifact_dir / "turn_001_response.json", response_001)
        turn_001_trace_id = _plan3_authoritative_response_trace_id(response_001)
        assert turn_001_trace_id
        turn_001_trace_run = await _wait_for_plan3_trace_finalization(
            database,
            turn_001_trace_id,
        )
        _, turn_001_trace_steps = await _read_plan3_trace_documents(
            database,
            turn_001_trace_id,
        )
        assert turn_001_trace_run.get("status") == "succeeded"
        assert turn_001_trace_run.get("platform") == platform
        assert turn_001_trace_run.get("platform_channel_id") == platform_channel_id
        assert turn_001_trace_run.get("platform_message_id") == turn_001_message_id
        requester_global_user_id = turn_001_trace_run.get("global_user_id")
        assert isinstance(requester_global_user_id, str)
        assert requester_global_user_id.strip()
        _write_json(
            artifact_dir / "turn_001_trace.json",
            {
                "trace_id": turn_001_trace_id,
                "trace_run": turn_001_trace_run,
                "trace_steps": turn_001_trace_steps,
            },
        )
        assert response_001.get("operational_error") is None
        messages_001 = response_001.get("messages")
        assert isinstance(messages_001, list)
        visible_001 = "\n".join(
            item.strip()
            for item in messages_001
            if isinstance(item, str) and item.strip()
        )
        assert visible_001
        assert alpha_marker not in visible_001
        assert beta_marker not in visible_001

        turn_001_binding_rows = await rows(
            "dsh_task_bindings",
            scoped_query("dsh_task_bindings", turn_001_message_id),
        )
        turn_001_task_rows = await rows(
            "accepted_tasks",
            scoped_query("accepted_tasks", turn_001_message_id),
        )
        turn_001_job_rows = await rows(
            "background_work_jobs",
            scoped_query("background_work_jobs", turn_001_message_id),
        )
        turn_001_interactions = await rows(
            "dsh_interaction_store",
            {"brain_conversation_ref": brain_conversation_ref},
        )
        assert not turn_001_binding_rows
        assert not turn_001_task_rows
        assert not turn_001_job_rows
        assert not turn_001_interactions
        assert alpha_path.read_text(encoding="utf-8") == alpha_marker + "\n"
        assert beta_path.read_text(encoding="utf-8") == beta_marker + "\n"
        turn_001_pending_query = _plan3_turn_001_pending_query(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=requester_global_user_id,
            source_message_id=turn_001_message_id,
        )
        turn_001_pending_rows = await rows(
            ACTION_ATTEMPT_LEDGER_COLLECTION,
            turn_001_pending_query,
        )
        assert len(turn_001_pending_rows) == 1
        _plan3_validate_turn_001_pending_row(
            turn_001_pending_rows[0],
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=requester_global_user_id,
            source_message_id=turn_001_message_id,
            original_goal=turn_001_text,
        )
        _write_json(
            artifact_dir / "turn_001_observation.json",
            {
                "response": response_001,
                "visible_text": visible_001,
                "trace_id": turn_001_trace_id,
                "trace_run": turn_001_trace_run,
                "trace_steps": turn_001_trace_steps,
                "pending_query": turn_001_pending_query,
                "pending_rows": turn_001_pending_rows,
                "binding_rows": turn_001_binding_rows,
                "task_rows": turn_001_task_rows,
                "job_rows": turn_001_job_rows,
                "interaction_rows": turn_001_interactions,
                "file_markers_unchanged": True,
            },
        )
        (artifact_dir / "turn_001_log.txt").write_text(
            f"visible_text={visible_001}\n"
            f"duration_ms={artifact_payload['turn_001_duration_ms']}\n",
            encoding="utf-8",
        )

        _write_json(artifact_dir / "turn_002_request.json", turn_002_request)
        (artifact_dir / "turn_002_log.txt").write_text(
            f"request={json.dumps(turn_002_request, ensure_ascii=False)}\n",
            encoding="utf-8",
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                600.0,
                connect=15.0,
                write=15.0,
                pool=15.0,
            )
        ) as client:
            second_response = await client.post(
                f"{brain_url}/chat",
                headers={
                    "X-Kazusa-Control-Console": "debug-v1",
                    "X-Kazusa-Control-Console-Auth": control_secret,
                },
                json=turn_002_request,
            )
            assert second_response.status_code == 200
            second_payload = second_response.json()
            assert isinstance(second_payload, Mapping)
            response_002 = dict(second_payload)
        _write_json(artifact_dir / "turn_002_response.json", response_002)
        trace_id = _plan3_authoritative_response_trace_id(response_002)
        assert response_002.get("operational_error") is None

        task_query = {
            "schema_version": "accepted_task.v2",
            "source_platform": platform,
            "source_channel_id": platform_channel_id,
        }
        task_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="accepted_tasks",
            query=task_query,
            timeout_seconds=420.0,
        )
        task = next(
            (
                row
                for row in task_rows
                if row.get("first_source_message_id") == turn_002_message_id
            ),
            None,
        )
        if task is None:
            raise AssertionError(
                "Turn 2 did not admit an accepted task from the beta choice"
            )
        task_id = task.get("accepted_task_id")
        assert isinstance(task_id, str) and task_id.strip()
        job_query = {
            "schema_version": "background_work_job.v2",
            "source_platform": platform,
            "source_channel_id": platform_channel_id,
        }
        job_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="background_work_jobs",
            query=job_query,
            timeout_seconds=420.0,
        )
        job = next(
            (
                row
                for row in job_rows
                if row.get("source_message_id") == turn_002_message_id
            ),
            None,
        )
        if job is None:
            raise AssertionError(
                "Turn 2 did not enqueue the selected beta task"
            )
        job_id = job.get("job_id")
        assert isinstance(job_id, str) and job_id.strip()
        binding_query = {
            "schema_version": "dsh_task_binding.v1",
            "source_scope.platform": platform,
            "source_scope.channel_id": platform_channel_id,
        }
        binding_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="dsh_task_bindings",
            query=binding_query,
            timeout_seconds=420.0,
        )
        binding = next(
            (
                row
                for row in binding_rows
                if isinstance(row.get("source_scope"), Mapping)
                and row["source_scope"].get("source_message_id")
                == turn_002_message_id
            ),
            None,
        )
        if binding is None:
            raise AssertionError("Turn 2 did not create its DSH task binding")
        final_delivery_rows = await _wait_for_plan3_deliveries(
            adapter,
            count=1,
            channel_id=platform_channel_id,
            timeout_seconds=420.0,
        )
        assert len(final_delivery_rows) == 1
        final_delivery = dict(final_delivery_rows[0])
        final_text = final_delivery.get("text")
        assert isinstance(final_text, str) and final_text.strip()
        _write_json(
            artifact_dir / "final_delivery_callback.json",
            final_delivery,
        )

        delivered_job_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="background_work_jobs",
            query={
                "schema_version": "background_work_job.v2",
                "job_id": job_id,
                "status": "delivered",
                "delivery_state": "delivered",
                "task_resolution_result.schema_version": (
                    "task_resolution_result.v1"
                ),
            },
            timeout_seconds=420.0,
        )
        assert len(delivered_job_rows) == 1
        job = delivered_job_rows[0]
        delivered_task_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="accepted_tasks",
            query={
                "schema_version": "accepted_task.v2",
                "accepted_task_id": task_id,
                "state": "delivered",
                "completion_status": {"$in": ["resolved", "partial"]},
            },
            timeout_seconds=420.0,
        )
        assert len(delivered_task_rows) == 1
        task = delivered_task_rows[0]
        terminal_binding_rows = await _wait_for_plan3_scoped_rows(
            database,
            collection_name="dsh_task_bindings",
            query={
                "schema_version": "dsh_task_binding.v1",
                "current_accepted_task_id": task_id,
                "current_background_work_job_id": job_id,
                "state": "terminal",
                "resolution_ref.schema_version": "dsh_resolution_ref.v1",
                "latest_task_resolution_result.schema_version": (
                    "task_resolution_result.v1"
                ),
            },
            timeout_seconds=420.0,
        )
        assert len(terminal_binding_rows) == 1
        binding = terminal_binding_rows[0]
        assert job.get("status") == "delivered"
        assert job.get("delivery_state") == "delivered"
        assert task.get("state") == "delivered"
        assert task.get("completion_status") in {"resolved", "partial"}

        task_result = job.get("task_resolution_result")
        assert isinstance(task_result, Mapping)
        assert task_result.get("schema_version") == "task_resolution_result.v1"
        assert task_result.get("status") in {"resolved", "partial"}
        assert beta_marker in json.dumps(task_result, ensure_ascii=False)
        assert alpha_marker not in json.dumps(task_result, ensure_ascii=False)
        resolution_ref = binding.get("resolution_ref")
        assert isinstance(resolution_ref, Mapping)
        resolution_thread_id = resolution_ref.get("resolution_thread_id")
        segment_id = resolution_ref.get("segment_id")
        task_session_id = binding.get("task_session_id")
        assert isinstance(resolution_thread_id, str) and resolution_thread_id.strip()
        assert isinstance(segment_id, str) and segment_id.strip()
        assert isinstance(task_session_id, str) and task_session_id.strip()
        assert resolution_ref.get("dsh_session_id") == task_session_id
        dsh_events = _read_plan3_dsh_events(
            data_root=runtime_environment["KAZUSA_DSH_DATA_ROOT"],
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
        )
        terminal_events = [
            event
            for _, event_type, event in dsh_events
            if event_type == "tool/result"
            and isinstance(event.get("meta"), Mapping)
            and isinstance(event["meta"].get("kazusa"), Mapping)
            and event["meta"]["kazusa"].get("kind") == "terminal_resolution_v2"
        ]
        assert len(terminal_events) == 1
        terminal_receipt = terminal_events[0]["meta"]["kazusa"]
        assert terminal_receipt.get("resolution_thread_id") == resolution_thread_id
        assert terminal_receipt.get("segment_id") == segment_id
        assert beta_marker in json.dumps(terminal_receipt, ensure_ascii=False)

        trace_candidates = await rows(
            "llm_trace_runs",
            {"source_background_work_job_id": job_id},
        )
        if trace_candidates:
            trace_id = trace_candidates[0].get("trace_id")
            assert isinstance(trace_id, str) and trace_id.strip()
            trace_run = await _wait_for_plan3_trace_finalization(
                database,
                trace_id,
            )
            _, trace_steps = await _read_plan3_trace_documents(database, trace_id)
            assert trace_run.get("status") == "succeeded"
            stage_names = {
                row.get("stage_name")
                for row in trace_steps
                if isinstance(row.get("stage_name"), str)
            }
            assert {
                "cognition_core_v3.A1",
                "cognition_core_v3.A2",
                "cognition_core_v3.G",
                "cognition_core_v3.P",
            } <= stage_names
            _write_json(
                artifact_dir / "trace.json",
                {"trace_run": trace_run, "trace_steps": trace_steps},
            )
        else:
            raise AssertionError("background DSH trace was not persisted")
        artifact_payload.update(
            {
                "turn_001_response": response_001,
                "turn_002_response": response_002,
                "registration": registration,
                "task": task,
                "job": job,
                "binding": binding,
                "dsh_events": dsh_events,
                "trace_run": trace_run,
                "trace_steps": trace_steps,
                "final_delivery": final_delivery,
                "visible_turn_001": visible_001,
            }
        )
        _write_json(
            artifact_dir / "dsh_lineage.json",
            {
                "task_session_id": task_session_id,
                "resolution_thread_id": resolution_thread_id,
                "segment_id": segment_id,
                "binding": binding,
                "dsh_events": dsh_events,
                "terminal_receipts": terminal_events,
            },
        )
        post_terminal_delivery_rows = await _wait_for_plan3_deliveries(
            adapter,
            count=1,
            channel_id=platform_channel_id,
            timeout_seconds=30.0,
        )
        assert len(post_terminal_delivery_rows) == 1
        assert dict(post_terminal_delivery_rows[0]) == final_delivery
        assert beta_marker in final_text
        assert alpha_marker not in final_text
    except httpx.RequestError as exc:
        failure = f"{exc.__class__.__name__}: {exc}"
        pytest.fail(
            f"Plan 3 public chat request failed: {exc.__class__.__name__}"
        )
    except Exception as exc:
        failure = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        failure_mongo_queries: dict[str, Mapping[str, object]] = {
            "accepted_tasks": {
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
            },
            "background_work_jobs": {
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
            },
            "dsh_interaction_store": {
                "brain_conversation_ref": brain_conversation_ref,
            },
        }
        if turn_001_pending_query:
            failure_mongo_queries[ACTION_ATTEMPT_LEDGER_COLLECTION] = (
                turn_001_pending_query
            )
        failure_snapshot = await _capture_plan3_failure_evidence(
            artifact_dir=artifact_dir,
            environment=runtime_environment,
            database=database,
            trace_id=trace_id,
            trace_query=_plan3_turn_trace_query(
                platform=platform,
                platform_channel_id=platform_channel_id,
                platform_message_id=turn_002_message_id,
            ),
            binding_query={
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": platform_channel_id,
            },
            mongo_queries=failure_mongo_queries,
            dsh_identity=(
                (str(binding["resolution_thread_id"]), str(binding["segment_id"]))
                if isinstance(binding, Mapping)
                and isinstance(binding.get("resolution_thread_id"), str)
                and isinstance(binding.get("segment_id"), str)
                else None
            ),
            known_binding=binding,
            known_dsh_events=dsh_events,
            known_trace_run=trace_run,
            known_trace_steps=trace_steps,
            callback_payloads=adapter.delivery_payloads(),
            background_work_job_id=job_id,
            turn_001_trace_id=turn_001_trace_id,
            known_turn_001_trace_run=turn_001_trace_run,
            known_turn_001_trace_steps=turn_001_trace_steps,
            turn_001_pending_rows=turn_001_pending_rows,
        )
        if not trace_id and isinstance(failure_snapshot.get("trace_id"), str):
            trace_id = failure_snapshot["trace_id"]
        if trace_run is None and isinstance(failure_snapshot.get("trace_run"), Mapping):
            trace_run = dict(failure_snapshot["trace_run"])
        if not trace_steps and isinstance(failure_snapshot.get("trace_steps"), list):
            trace_steps = list(failure_snapshot["trace_steps"])
        if not dsh_events and isinstance(failure_snapshot.get("dsh_events"), list):
            dsh_events = list(failure_snapshot["dsh_events"])
        for child in reversed(child_processes):
            await _stop_plan3_child(child)
        if adapter_started:
            adapter.stop()
        artifact_payload["duration_ms"] = round(
            (time.perf_counter() - started_at) * 1000
        )
        artifact_payload["processes"] = [
            _process_artifact(child) for child in child_processes
        ]
        artifact_payload["callback_deliveries"] = adapter.delivery_payloads()
        artifact_payload["readiness"] = readiness_payload
        if failure is not None:
            artifact_payload["failure"] = failure
        _write_json(artifact_dir / "run.json", artifact_payload)
        conclusion_lines = [
            "# Plan 3 pre-admission prerequisite gate",
            "",
            f"Turn 1 response captured: {isinstance(response_001, Mapping)}",
            f"Turn 2 response captured: {isinstance(response_002, Mapping)}",
            f"Final callback count: {len(adapter.delivery_payloads())}",
            (
                "Beta marker in final callback: "
                f"{beta_marker in json.dumps(final_delivery or {}, ensure_ascii=False)}"
            ),
            (
                "Alpha marker in final callback: "
                f"{alpha_marker in json.dumps(final_delivery or {}, ensure_ascii=False)}"
            ),
            (
                "Qualitative judgment: Turn 1 is ordinary clarification with "
                "no DSH admission; Turn 2 is the only task admission and the "
                "final beta result returns through normal Brain delivery."
            ),
        ]
        (artifact_dir / "behavior_audit_conclusions.md").write_text(
            "\n".join(conclusion_lines) + "\n",
            encoding="utf-8",
        )
        for name, value in (
            ("turn_001_response.json", response_001),
            (
                "turn_001_trace.json",
                {
                    "trace_id": turn_001_trace_id,
                    "trace_run": turn_001_trace_run,
                    "trace_steps": turn_001_trace_steps,
                },
            ),
            ("turn_002_response.json", response_002),
            ("final_delivery_callback.json", final_delivery or {}),
            ("dsh_lineage.json", {}),
            ("trace.json", {"trace_run": trace_run, "trace_steps": trace_steps}),
        ):
            path = artifact_dir / name
            if not path.is_file():
                _write_json(path, value)
        for fixture_path in (alpha_path, beta_path):
            fixture_path.unlink(missing_ok=True)
        try:
            fixture_parent.rmdir()
        except OSError:
            pass
        cleanup_disposition = "not_attempted"
        if runtime_environment is not None:
            try:
                await _drop_plan3_database(runtime_environment)
            except _PLAN3_EVIDENCE_ERRORS as cleanup_error:
                cleanup_disposition = (
                    f"failed:{cleanup_error.__class__.__name__}"
                )
            else:
                cleanup_disposition = "dropped"
        artifact_payload["database_cleanup"] = cleanup_disposition
        _write_json(artifact_dir / "run.json", artifact_payload)
        with (artifact_dir / "behavior_audit_conclusions.md").open(
            "a",
            encoding="utf-8",
        ) as conclusions_file:
            conclusions_file.write(
                f"Database cleanup: `{cleanup_disposition}`\n"
            )
        print(f"DSH_PLAN3_PREREQUISITE_ARTIFACT={artifact_dir}")


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_e2e_public_research_and_media_use_native_web_and_kazusa_semantic_evidence(
    tmp_path: Path,
) -> None:
    """Prove native web and public-media evidence reach one grounded result."""

    _require_plan3_inline_live_backend()
    from io import BytesIO

    from PIL import Image

    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    marker = f"DSH_PLAN3_PUBLIC_MEDIA_{suffix}"
    platform = f"dsh-plan3-public-research-{suffix}"
    channel_id = f"dsh-plan3-public-research-channel-{suffix}"
    user_id = f"dsh-plan3-public-research-user-{suffix}"
    message_id = f"dsh-plan3-public-research-message-{suffix}"
    bot_id = os.environ["CHARACTER_GLOBAL_USER_ID"].strip()
    media_url = (
        "https://raw.githubusercontent.com/eamars/KazusaAIChatbot/"
        "59357e591f762f46b7492f12be42752daff25632/resources/avatar.png"
    )
    expected_media_sha256 = (
        "3bbe03444a93c736945916353845cc96d63e7e0126b01cc77a19bb4fadd0de5b"
    )
    artifact_dir, artifact_stamp = _plan3_artifact_directory(
        "public_research_media",
        suffix,
    )
    runtime_environment: dict[str, str] | None = _runtime_environment(tmp_path)
    child_processes: list[_Plan3ChildProcess] = []
    readiness: dict[str, object] | None = None
    database: Any | None = None
    response_payload: dict[str, Any] | None = None
    binding: dict[str, Any] | None = None
    dsh_events: list[tuple[int, str, dict[str, Any]]] = []
    trace_run: dict[str, Any] | None = None
    trace_steps: list[dict[str, Any]] = []
    trace_id = ""
    request_payload: dict[str, object] | None = None
    visible_text = ""
    media_preflight: dict[str, object] = {}
    started_at = time.perf_counter()

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
            trust_env=False,
        ) as media_client:
            media_response = await media_client.get(media_url)
        assert media_response.status_code == 200
        media_bytes = media_response.content
        media_digest = sha256(media_bytes).hexdigest()
        assert media_digest == expected_media_sha256
        with Image.open(BytesIO(media_bytes)) as image:
            media_size = image.size
        assert media_size == (900, 900)
        media_preflight = {
            "url": media_url,
            "status_code": media_response.status_code,
            "byte_count": len(media_bytes),
            "sha256": media_digest,
            "size": media_size,
        }
        _write_json(artifact_dir / "media_preflight.json", media_preflight)

        runtime_environment, child_processes, readiness = (
            await _start_plan3_services(
                tmp_path=tmp_path,
                artifact_dir=artifact_dir,
                artifact_stamp=artifact_stamp,
                environment=runtime_environment,
            )
        )
        user_text = (
            "Use the native Standard web tools to research the public source "
            f"behind this immutable image URL: {media_url}. Then call the "
            "retained semantic capability kazusa_inspect_public_media with "
            "the same URL and the bounded visual question 'What subject and "
            "dominant colors are visibly present?'. Use one additional "
            "retained Kazusa semantic evidence capability when it is relevant, "
            f"then submit one grounded terminal result containing {marker}. "
            "Do not invoke complex_task_resolver, RAG2, or any legacy executor."
        )
        request_payload = _plan3_chat_request(
            platform=platform,
            channel_id=channel_id,
            user_id=user_id,
            bot_id=bot_id,
            display_name=f"Plan 3 public researcher {suffix[:8]}",
            channel_name=f"Plan 3 public research {suffix[:8]}",
            message_id=message_id,
            text=user_text,
            no_remember=True,
        )
        _write_json(artifact_dir / "request.json", request_payload)
        brain_url = runtime_environment["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
        control_secret = runtime_environment[
            "KAZUSA_CONTROL_BRAIN_SHARED_SECRET"
        ].strip()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                600.0,
                connect=15.0,
                write=15.0,
                pool=15.0,
            )
        ) as client:
            response_payload = await _post_plan3_chat(
                client,
                brain_url=brain_url,
                control_secret=control_secret,
                request_payload=request_payload,
            )
        _write_json(artifact_dir / "response.json", response_payload)
        assert response_payload.get("content_type") == "text"
        assert response_payload.get("operational_error") is None
        visible_text = _plan3_visible_text(response_payload)
        assert marker in visible_text
        graph = response_payload.get("cognition_graph")
        assert isinstance(graph, Mapping)
        graph_correlation = graph.get("correlation")
        assert isinstance(graph_correlation, Mapping)
        trace_value = graph_correlation.get("llm_trace_id")
        assert isinstance(trace_value, str) and trace_value.strip()
        trace_id = trace_value

        database = await get_db()
        binding_rows = await _plan3_read_rows(
            database,
            "dsh_task_bindings",
            {
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": channel_id,
                "source_scope.source_message_id": message_id,
            },
        )
        assert len(binding_rows) == 1
        binding = binding_rows[0]
        thread_id = binding.get("resolution_thread_id")
        segment_id = binding.get("segment_id")
        assert isinstance(thread_id, str) and thread_id.strip()
        assert isinstance(segment_id, str) and segment_id.strip()
        dsh_events = _read_plan3_dsh_events(
            data_root=runtime_environment["KAZUSA_DSH_DATA_ROOT"],
            resolution_thread_id=thread_id,
            segment_id=segment_id,
        )
        serialized_events = json.dumps(dsh_events, ensure_ascii=False)
        assert "kazusa_inspect_public_media" in serialized_events
        assert "public_media" in serialized_events
        assert "complex_task_resolver" not in serialized_events
        assert "rag2" not in serialized_events.lower()
        tool_calls = [
            event for _, event_type, event in dsh_events
            if event_type == "tool/call"
        ]
        tool_results = [
            event for _, event_type, event in dsh_events
            if event_type == "tool/result"
        ]
        assert any(
            media_url in json.dumps(event, ensure_ascii=False)
            for event in tool_calls
        )
        media_results = [
            event
            for event in tool_results
            if "kazusa_inspect_public_media"
            in json.dumps(event, ensure_ascii=False)
        ]
        assert media_results
        public_media_receipts = [
            receipt
            for event in tool_results
            for receipt in _plan3_event_evidence(event)
            if receipt.get("source_kind") == "public_media"
        ]
        assert len(public_media_receipts) == 1
        receipt_text = json.dumps(
            public_media_receipts[0],
            ensure_ascii=False,
        )
        assert public_media_receipts[0].get("semantic_ref", "").startswith(
            "public-media:"
        )
        provenance = public_media_receipts[0].get("provenance")
        assert isinstance(provenance, Mapping)
        assert provenance.get("tool_name") == "kazusa_inspect_public_media"
        assert receipt_text
        semantic_results = [
            event
            for event in tool_results
            if "kazusa_" in json.dumps(event, ensure_ascii=False)
        ]
        assert len(semantic_results) >= 2
        terminal_events = [
            event
            for event in tool_results
            if isinstance(event.get("meta"), Mapping)
            and isinstance(event["meta"].get("kazusa"), Mapping)
            and event["meta"]["kazusa"].get("kind")
            == "terminal_resolution_v2"
        ]
        assert len(terminal_events) == 1
        assert marker in json.dumps(terminal_events[0], ensure_ascii=False)
        trace_run = await _wait_for_plan3_trace_finalization(
            database,
            trace_id,
        )
        _, trace_steps = await _read_plan3_trace_documents(database, trace_id)
        assert trace_run.get("status") == "succeeded"
        _write_json(
            artifact_dir / "dsh_lineage.json",
            {
                "binding": binding,
                "events": dsh_events,
                "terminal_receipts": terminal_events,
            },
        )
        _write_json(
            artifact_dir / "trace.json",
            {"trace_run": trace_run, "trace_steps": trace_steps},
        )
    finally:
        failure_snapshot = await _capture_plan3_failure_evidence(
            artifact_dir=artifact_dir,
            environment=runtime_environment,
            database=database,
            trace_id=trace_id,
            trace_query={
                "platform": platform,
                "platform_channel_id": channel_id,
                "platform_user_id": user_id,
            },
            binding_query={
                "schema_version": "dsh_task_binding.v1",
                "source_scope.platform": platform,
                "source_scope.channel_id": channel_id,
                "source_scope.source_message_id": message_id,
            },
            mongo_queries={
                "accepted_tasks": {
                    "source_platform": platform,
                    "source_channel_id": channel_id,
                },
                "background_work_jobs": {
                    "source_platform": platform,
                    "source_channel_id": channel_id,
                },
                "dsh_interaction_store": {
                    "source_scope.platform": platform,
                    "source_scope.channel_id": channel_id,
                },
            },
            dsh_identity=(
                (str(binding["resolution_thread_id"]), str(binding["segment_id"]))
                if isinstance(binding, Mapping)
                and isinstance(binding.get("resolution_thread_id"), str)
                and isinstance(binding.get("segment_id"), str)
                else None
            ),
            known_binding=binding,
            known_dsh_events=dsh_events,
            known_trace_run=trace_run,
            known_trace_steps=trace_steps,
        )
        if not trace_id and isinstance(failure_snapshot.get("trace_id"), str):
            trace_id = failure_snapshot["trace_id"]
        if trace_run is None and isinstance(failure_snapshot.get("trace_run"), Mapping):
            trace_run = dict(failure_snapshot["trace_run"])
        if not trace_steps and isinstance(failure_snapshot.get("trace_steps"), list):
            trace_steps = list(failure_snapshot["trace_steps"])
        if not dsh_events and isinstance(failure_snapshot.get("dsh_events"), list):
            dsh_events = list(failure_snapshot["dsh_events"])
        for child in reversed(child_processes):
            await _stop_plan3_child(child)
        for name, value in (
            ("request.json", request_payload or {}),
            ("response.json", response_payload or {}),
            ("readiness.json", readiness or {}),
            ("dsh_lineage.json", {"binding": binding, "events": dsh_events}),
            ("trace.json", {"trace_run": trace_run, "trace_steps": trace_steps}),
            ("mongo_state.json", {}),
            ("media_preflight.json", media_preflight),
        ):
            path = artifact_dir / name
            if not path.is_file():
                _write_json(path, value)
        _write_json(
            artifact_dir / "processes.json",
            [_process_artifact(child) for child in child_processes],
        )
        conclusion_lines = [
            "# Public research and media acceptance",
            "",
            f"- Immutable media digest verified: `{media_preflight.get('sha256') == expected_media_sha256}`",
            f"- Visible grounded result captured: `{bool(visible_text)}`",
            f"- Public-media receipt count: `{len([receipt for _, event_type, event in dsh_events if event_type == 'tool/result' for receipt in _plan3_event_evidence(event) if receipt.get('source_kind') == 'public_media'])}`",
            f"- Protected trace retained: `{bool(trace_id)}`",
            f"- Duration ms: `{round((time.perf_counter() - started_at) * 1000)}`",
            "- Judgment: native web and the retained semantic gateway provide bounded evidence; the public-media bytes stay outside model-facing receipts and the terminal result remains grounded.",
        ]
        (artifact_dir / "behavior_audit_conclusions.md").write_text(
            "\n".join(conclusion_lines) + "\n",
            encoding="utf-8",
        )
        cleanup_disposition = "not_attempted"
        if runtime_environment is not None:
            try:
                await _drop_plan3_database(runtime_environment)
            except _PLAN3_EVIDENCE_ERRORS as cleanup_error:
                cleanup_disposition = (
                    f"failed:{cleanup_error.__class__.__name__}"
                )
            else:
                cleanup_disposition = "dropped"
        _write_json(
            artifact_dir / "run.json",
            {
                "database_name": runtime_environment.get("MONGODB_DB_NAME", "")
                if runtime_environment is not None
                else "",
                "cleanup": cleanup_disposition,
                "request": request_payload or {},
                "response": response_payload or {},
                "readiness": readiness or {},
                "media_preflight": media_preflight,
                "trace_id": trace_id,
            },
        )
        with (artifact_dir / "behavior_audit_conclusions.md").open(
            "a",
            encoding="utf-8",
        ) as conclusions_file:
            conclusions_file.write(
                f"Database cleanup: `{cleanup_disposition}`\n"
            )
        print(f"DSH_PLAN3_PUBLIC_RESEARCH_ARTIFACT={artifact_dir}")
