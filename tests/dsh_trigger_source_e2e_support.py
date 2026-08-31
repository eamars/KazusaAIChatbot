"""Isolated live-E2E support for the DSH trigger-source sign-off matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
SIDECAR_ENTRY = (
    PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
)
ARTIFACT_ROOT = PROJECT_ROOT / "test_artifacts" / "dsh_trigger_source_e2e"
EPHEMERAL_DATABASE_PREFIX = "_test_kazusa_dsh_e2e_"
DSH_RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v2"
CASE_PROCESS_TIMEOUT_SECONDS = 30 * 60
LIVE_HTTP_TIMEOUT_SECONDS = 10 * 60
SOURCE_TRACE_SETTLEMENT_TIMEOUT_SECONDS = 90.0


@dataclass(frozen=True)
class TriggerSourceCaseSpec:
    """Stable source, admission, and local-evidence definition for one node."""

    case_id: str
    trigger_source: str
    expects_dsh_entry: bool
    workspace_files: Mapping[str, str]
    expected_evidence_files: tuple[str, ...] = ()


CASE_SPECS = {
    "user_message_local_fact": TriggerSourceCaseSpec(
        case_id="user_message_local_fact",
        trigger_source="user_message",
        expects_dsh_entry=True,
        workspace_files={
            "rollout/status_note.txt": (
                "The rollout owner is Mira. The checksum review must finish "
                "before rollout begins."
            ),
        },
        expected_evidence_files=("status_note.txt",),
    ),
    "user_message_background_summary": TriggerSourceCaseSpec(
        case_id="user_message_background_summary",
        trigger_source="user_message",
        expects_dsh_entry=True,
        workspace_files={
            "handover/incident_note.txt": (
                "The cache alert is stable. Rowan owns the follow-up, and the "
                "next review happens after the morning metrics arrive."
            ),
        },
        expected_evidence_files=("incident_note.txt",),
    ),
    "internal_thought_file_check": TriggerSourceCaseSpec(
        case_id="internal_thought_file_check",
        trigger_source="internal_thought",
        expects_dsh_entry=True,
        workspace_files={
            "internal/health_note.txt": (
                "The service is stable. The next checkpoint is the queue-depth "
                "review."
            ),
        },
        expected_evidence_files=("health_note.txt",),
    ),
    "internal_thought_comparison": TriggerSourceCaseSpec(
        case_id="internal_thought_comparison",
        trigger_source="internal_thought",
        expects_dsh_entry=True,
        workspace_files={
            "internal/early_shift.txt": (
                "Early shift: the search index was rebuilding and Ava owned "
                "the observation."
            ),
            "internal/late_shift.txt": (
                "Late shift: the search index finished rebuilding and Dev now "
                "owns the verification."
            ),
        },
        expected_evidence_files=("early_shift.txt", "late_shift.txt"),
    ),
    "self_cognition_targetless_group": TriggerSourceCaseSpec(
        case_id="self_cognition_targetless_group",
        trigger_source="self_cognition",
        expects_dsh_entry=False,
        workspace_files={
            "group/shared_status.txt": (
                "The shared status note belongs to the channel, not to an "
                "invented private user."
            ),
        },
    ),
    "self_cognition_promoted_group": TriggerSourceCaseSpec(
        case_id="self_cognition_promoted_group",
        trigger_source="self_cognition",
        expects_dsh_entry=False,
        workspace_files={
            "group/promoted_status.txt": (
                "The promoted channel rule keeps shared facts separate from "
                "private user details."
            ),
        },
    ),
    "scheduled_tick_commitment_due": TriggerSourceCaseSpec(
        case_id="scheduled_tick_commitment_due",
        trigger_source="scheduled_tick",
        expects_dsh_entry=True,
        workspace_files={
            "scheduled/commitment_note.txt": (
                "The due commitment belongs to Priya. Verification starts "
                "after the deployment window opens."
            ),
        },
        expected_evidence_files=("commitment_note.txt",),
    ),
    "scheduled_tick_future": TriggerSourceCaseSpec(
        case_id="scheduled_tick_future",
        trigger_source="scheduled_tick",
        expects_dsh_entry=True,
        workspace_files={
            "scheduled/future_note.txt": (
                "The rollout condition is satisfied because the canary is "
                "healthy and the error budget remains intact."
            ),
        },
        expected_evidence_files=("future_note.txt",),
    ),
    "tool_result_resolved": TriggerSourceCaseSpec(
        case_id="tool_result_resolved",
        trigger_source="tool_result",
        expects_dsh_entry=False,
        workspace_files={},
    ),
    "tool_result_failed": TriggerSourceCaseSpec(
        case_id="tool_result_failed",
        trigger_source="tool_result",
        expects_dsh_entry=False,
        workspace_files={},
    ),
}


class _ChildProcess:
    """Own one nested process and its durable output streams."""

    def __init__(
        self,
        *,
        name: str,
        process: asyncio.subprocess.Process,
        stdout_file: BinaryIO,
        stderr_file: BinaryIO,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        self.name = name
        self.process = process
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path


class _LoopbackAdapterServer:
    """Capture real capability probes and dispatcher callbacks."""

    def __init__(self, *, platform: str, shared_secret: str) -> None:
        self._platform = platform
        self._shared_secret = shared_secret
        self._lock = Lock()
        self._capability_payloads: list[dict[str, Any]] = []
        self._delivery_payloads: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Serve the authenticated adapter callback endpoints."""

            def do_POST(self) -> None:
                owner._handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="dsh-trigger-source-loopback-adapter",
            daemon=True,
        )

    @property
    def callback_url(self) -> str:
        """Return the bound loopback callback URL."""

        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def start(self) -> None:
        """Start the callback server."""

        self._thread.start()

    def stop(self) -> None:
        """Stop and join the callback server."""

        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("loopback adapter thread did not stop")

    def capability_payloads(self) -> list[dict[str, Any]]:
        """Return a copy of all capability probes."""

        with self._lock:
            return [dict(value) for value in self._capability_payloads]

    def delivery_payloads(self) -> list[dict[str, Any]]:
        """Return a copy of all delivery callbacks."""

        with self._lock:
            return [dict(value) for value in self._delivery_payloads]

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        """Authenticate and record one adapter request."""

        expected = f"Bearer {self._shared_secret}"
        if request.headers.get("Authorization") != expected:
            self._respond(request, 401, {"available": False})
            return
        try:
            length = int(request.headers.get("Content-Length") or "")
            payload = json.loads(request.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._respond(request, 400, {"error": "invalid request"})
            return
        if not isinstance(payload, dict):
            self._respond(request, 400, {"error": "JSON object required"})
            return
        if request.path == "/send_message/capability":
            with self._lock:
                self._capability_payloads.append(dict(payload))
            self._respond(request, 200, {"available": True})
            return
        if request.path != "/send_message":
            self._respond(request, 404, {"error": "unknown callback"})
            return

        message_id = f"dsh-e2e-adapter-{uuid4().hex}"
        sent_at = datetime.now(UTC).isoformat()
        delivery = {
            **payload,
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
                "channel_id": payload.get("channel_id", ""),
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
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.send_response(status_code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(encoded)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(encoded)


def _write_json(path: Path, value: object) -> None:
    """Write one inspectable UTF-8 JSON artifact."""

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _free_loopback_port() -> int:
    """Return one currently free loopback TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _artifact_directory(case_id: str) -> Path:
    """Create one durable artifact directory for a matrix node."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = ARTIFACT_ROOT / f"{case_id}_{stamp}_{uuid4().hex[:8]}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    return artifact_dir


def _require_live_configuration() -> None:
    """Require the externally configured real-model routes."""

    if os.environ.get("KAZUSA_RUN_LIVE_LLM") != "1":
        pytest.skip("set KAZUSA_RUN_LIVE_LLM=1 for real-model coverage")
    required = (
        "AGENTIC_RESOLVER_LLM_API_KEY",
        "AGENTIC_RESOLVER_LLM_BASE_URL",
        "AGENTIC_RESOLVER_LLM_MODEL",
        "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS",
        "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS",
        "AGENTIC_RESOLVER_LLM_THINKING_ENABLED",
        "CHARACTER_GLOBAL_USER_ID",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(
            "DSH trigger-source live configuration is missing: "
            + ", ".join(missing)
        )
    if not PYTHON_EXECUTABLE.is_file():
        pytest.fail(f"project virtualenv Python is missing: {PYTHON_EXECUTABLE}")
    if not SIDECAR_ENTRY.is_file():
        pytest.fail(f"pinned DSH sidecar build is missing: {SIDECAR_ENTRY}")
    if shutil.which("node") is None:
        pytest.fail("the pinned DSH sidecar requires node on PATH")


def _case_environment(
    *,
    tmp_path: Path,
    database_name: str,
    brain_port: int,
    sidecar_port: int,
) -> dict[str, str]:
    """Build a clean case process environment without exposing secrets."""

    workspace_root = (tmp_path / "workspace").resolve()
    data_root = (tmp_path / "dsh-data").resolve()
    workspace_root.mkdir(parents=True, exist_ok=False)
    data_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    python_path_parts = [
        str((PROJECT_ROOT / "src").resolve()),
        str(PROJECT_ROOT.resolve()),
    ]
    existing_python_path = environment.get("PYTHONPATH", "").strip()
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path_parts),
            "MONGODB_DB_NAME": database_name,
            "KAZUSA_TEST_DB_GUARD": "1",
            "KAZUSA_EPHEMERAL_TEST_DATABASE_GUARD": "1",
            "KAZUSA_EPHEMERAL_TEST_DATABASE_NAME": database_name,
            "AGENTIC_RESOLVER_WORKSPACE_ROOT": str(workspace_root),
            "KAZUSA_DSH_DATA_ROOT": str(data_root),
            "KAZUSA_DSH_PYTHON_EXECUTABLE": str(PYTHON_EXECUTABLE.resolve()),
            "KAZUSA_DSH_BRAIN_URL": f"http://127.0.0.1:{brain_port}",
            "KAZUSA_DSH_SIDECAR_URL": f"http://127.0.0.1:{sidecar_port}/rpc",
            "KAZUSA_DSH_RPC_TOKEN": f"rpc-{uuid4().hex}",
            "KAZUSA_DSH_BRAIN_SHARED_SECRET": f"brain-{uuid4().hex}",
            "KAZUSA_DSH_TOOL_GATEWAY_SECRET": f"gateway-{uuid4().hex}",
            "KAZUSA_DSH_CAPABILITY_TOKEN": f"capability-{uuid4().hex}",
            "KAZUSA_CONTROL_BRAIN_SHARED_SECRET": f"control-{uuid4().hex}",
            "CALENDAR_SCHEDULER_ENABLED": "false",
            "SELF_COGNITION_ENABLED": "false",
            "BACKGROUND_WORK_WORKER_ENABLED": "false",
            "REFLECTION_CYCLE_ENABLED": "false",
            "SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED": "true",
            "SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED": "true",
            "CHARACTER_SLEEP_LOCAL_PERIOD": "",
            "COGNITION_VISUAL_DIRECTIVES_ENABLED": "false",
            "LLM_TRACE_CAPTURE_MODE": "full",
            "TASK_RESOLUTION_INLINE_BUDGET_SECONDS": "120.0",
        }
    )
    return environment


def _write_workspace_files(
    spec: TriggerSourceCaseSpec,
    *,
    workspace_root: Path,
) -> list[dict[str, str]]:
    """Materialize natural local evidence declared by a case."""

    records: list[dict[str, str]] = []
    for relative_path, content in spec.workspace_files.items():
        destination = (workspace_root / relative_path).resolve()
        if workspace_root not in destination.parents:
            raise AssertionError("workspace fixture escaped its isolated root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        records.append(
            {
                "relative_path": relative_path,
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
            }
        )
    return records


async def run_trigger_source_case(case_id: str, tmp_path: Path) -> None:
    """Run one matrix node in a configuration-isolated child process."""

    _require_live_configuration()
    try:
        spec = CASE_SPECS[case_id]
    except KeyError as exc:
        raise AssertionError(f"unknown DSH E2E case: {case_id}") from exc

    artifact_dir = _artifact_directory(case_id)
    case_tmp = (tmp_path / case_id).resolve()
    case_tmp.mkdir(parents=True, exist_ok=False)
    brain_port = _free_loopback_port()
    sidecar_port = _free_loopback_port()
    if brain_port == sidecar_port:
        sidecar_port = _free_loopback_port()
    database_name = f"{EPHEMERAL_DATABASE_PREFIX}{uuid4().hex}"
    environment = _case_environment(
        tmp_path=case_tmp,
        database_name=database_name,
        brain_port=brain_port,
        sidecar_port=sidecar_port,
    )
    workspace_records = _write_workspace_files(
        spec,
        workspace_root=Path(environment["AGENTIC_RESOLVER_WORKSPACE_ROOT"]),
    )
    _write_json(
        artifact_dir / "case_spec.json",
        {
            "case_id": spec.case_id,
            "trigger_source": spec.trigger_source,
            "expects_dsh_entry": spec.expects_dsh_entry,
            "expected_evidence_files": list(spec.expected_evidence_files),
            "workspace_files": workspace_records,
            "database_name": database_name,
        },
    )

    stdout_path = artifact_dir / "case_process.stdout.log"
    stderr_path = artifact_dir / "case_process.stderr.log"
    command = [
        str(PYTHON_EXECUTABLE),
        str(Path(__file__).resolve()),
        "--execute-case",
        case_id,
        "--artifact-dir",
        str(artifact_dir.resolve()),
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creation_flags,
        )
        try:
            return_code = await asyncio.wait_for(
                process.wait(),
                timeout=CASE_PROCESS_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            process.terminate()
            with suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=10)
            if process.returncode is None:
                process.kill()
                await process.wait()
            _write_json(
                artifact_dir / "parent_timeout.json",
                {
                    "case_id": case_id,
                    "timeout_seconds": CASE_PROCESS_TIMEOUT_SECONDS,
                },
            )
            pytest.fail(
                f"{case_id} exceeded its isolated process deadline; "
                f"artifacts: {artifact_dir}"
            )

    result_path = artifact_dir / "case_result.json"
    if not result_path.is_file():
        pytest.fail(
            f"{case_id} produced no case_result.json (exit {return_code}); "
            f"artifacts: {artifact_dir}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cleanup_path = artifact_dir / "cleanup.json"
    cleanup = (
        json.loads(cleanup_path.read_text(encoding="utf-8"))
        if cleanup_path.is_file()
        else {}
    )
    print(f"DSH trigger-source artifact: {artifact_dir}")
    if return_code != 0 or result.get("technical_status") != "passed":
        pytest.fail(
            f"{case_id} hard gates failed: {result.get('failures', [])}; "
            f"exit={return_code}; cleanup={cleanup}; artifacts={artifact_dir}"
        )
    if cleanup.get("database_dropped") is not True:
        pytest.fail(
            f"{case_id} did not confirm guarded database cleanup; "
            f"artifacts: {artifact_dir}"
        )


async def _prepare_case_database(artifact_dir: Path) -> None:
    """Bootstrap and seed the guarded database with bounded Mongo retries."""

    from pymongo.errors import PyMongoError

    from kazusa_ai_chatbot.character_profile import load_character_profile_seed
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import _client as client_module
    from kazusa_ai_chatbot.db import db_bootstrap
    from kazusa_ai_chatbot.db.character_identity_growth import (
        ensure_seed_identity,
    )

    database_name = os.environ.get("MONGODB_DB_NAME", "").strip()
    if not database_name.startswith(EPHEMERAL_DATABASE_PREFIX):
        raise AssertionError("character seed target is not a guarded database")
    client_module.MONGODB_DB_NAME = database_name
    client_module._assert_guarded_database_name()
    seed = load_character_profile_seed(
        PROJECT_ROOT / "personalities" / "example.json"
    )
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(3):
        started_at = time.perf_counter()
        try:
            await client_module.close_db()
            await db_bootstrap()
            await ensure_seed_identity(
                character_id=CHARACTER_GLOBAL_USER_ID,
                seed=seed,
            )
        except PyMongoError as exc:
            attempts.append({
                "attempt": attempt_index + 1,
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
                "error_class": type(exc).__name__,
                "error": str(exc),
                "status": "retryable_failure",
            })
            if attempt_index == 2:
                _write_json(
                    artifact_dir / "database_preparation.json",
                    {"attempts": attempts, "prepared": False},
                )
                raise
            await asyncio.sleep(0.5 * (attempt_index + 1))
        else:
            attempts.append({
                "attempt": attempt_index + 1,
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
                "status": "prepared",
            })
            _write_json(
                artifact_dir / "database_preparation.json",
                {"attempts": attempts, "prepared": True},
            )
            return
        finally:
            await client_module.close_db()


async def _drop_case_database() -> None:
    """Drop only the exact guarded database owned by this case process."""

    from pymongo.errors import PyMongoError

    from kazusa_ai_chatbot.db import _client as client_module

    database_name = os.environ.get("MONGODB_DB_NAME", "").strip()
    if not database_name.startswith(EPHEMERAL_DATABASE_PREFIX):
        raise AssertionError("cleanup target is outside the reserved prefix")
    if os.environ.get("KAZUSA_EPHEMERAL_TEST_DATABASE_NAME") != database_name:
        raise AssertionError("cleanup target does not match the case guard")
    client_module.MONGODB_DB_NAME = database_name
    client_module._assert_guarded_database_name()
    for attempt_index in range(3):
        cleanup_client = None
        try:
            await client_module.close_db()
            cleanup_client = client_module.AsyncIOMotorClient(
                client_module.MONGODB_URI
            )
            await cleanup_client.drop_database(database_name)
            return
        except PyMongoError:
            if attempt_index == 2:
                raise
            await asyncio.sleep(0.5 * (attempt_index + 1))
        finally:
            if cleanup_client is not None:
                cleanup_client.close()


async def _launch_sidecar(artifact_dir: Path) -> _ChildProcess:
    """Launch the pinned DSH sidecar with inspectable output."""

    node_executable = shutil.which("node")
    if node_executable is None:
        raise AssertionError("node is unavailable for the DSH sidecar")
    stdout_path = artifact_dir / "sidecar.stdout.log"
    stderr_path = artifact_dir / "sidecar.stderr.log"
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        process = await asyncio.create_subprocess_exec(
            node_executable,
            str(SIDECAR_ENTRY),
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creation_flags,
        )
    except BaseException:
        stdout_file.close()
        stderr_file.close()
        raise
    return _ChildProcess(
        name="sidecar",
        process=process,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


async def _stop_child(child: _ChildProcess) -> None:
    """Stop one exact nested process and close its log streams."""

    if child.process.returncode is None:
        child.process.terminate()
    try:
        await asyncio.wait_for(child.process.wait(), timeout=15)
    except TimeoutError:
        if child.process.returncode is None:
            child.process.kill()
        await child.process.wait()
    finally:
        child.stdout_file.close()
        child.stderr_file.close()


async def _wait_for_runtime_readiness(
    *,
    brain_task: asyncio.Task[Any],
    sidecar: _ChildProcess,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Wait for the live Brain and DSH sidecar health contracts."""

    brain_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    sidecar_url = os.environ["KAZUSA_DSH_SIDECAR_URL"]
    brain_secret = os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"]
    rpc_token = os.environ["KAZUSA_DSH_RPC_TOKEN"]
    started_at = time.perf_counter()
    deadline = time.monotonic() + 120
    latest: dict[str, Any] = {}
    timeout = httpx.Timeout(5.0, connect=1.0, write=2.0, pool=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while time.monotonic() < deadline:
            if brain_task.done():
                if brain_task.cancelled():
                    raise RuntimeError("Brain task was cancelled before readiness")
                brain_error = brain_task.exception()
                if brain_error is not None:
                    raise RuntimeError(
                        "Brain exited before readiness: "
                        f"{type(brain_error).__name__}: {brain_error}"
                    ) from brain_error
                raise RuntimeError("Brain exited before readiness")
            if sidecar.process.returncode is not None:
                raise AssertionError(
                    "DSH sidecar exited before readiness with code "
                    f"{sidecar.process.returncode}"
                )
            brain_probe: dict[str, Any]
            try:
                brain_response = await client.get(
                    f"{brain_url}/runtime/dsh/health",
                    headers={"Authorization": f"Bearer {brain_secret}"},
                )
                brain_payload = brain_response.json()
                brain_probe = {
                    "http_status": brain_response.status_code,
                    "payload": brain_payload,
                    "ready": (
                        brain_response.status_code == 200
                        and isinstance(brain_payload, Mapping)
                        and brain_payload.get("status") == "ready"
                    ),
                }
            except (httpx.RequestError, ValueError) as exc:
                brain_probe = {
                    "ready": False,
                    "error_class": type(exc).__name__,
                }

            sidecar_request = {
                "jsonrpc": "2.0",
                "id": f"health-{time.time_ns()}",
                "method": "system.health",
                "params": {"protocol_version": DSH_RPC_PROTOCOL_VERSION},
            }
            try:
                sidecar_response = await client.post(
                    sidecar_url,
                    headers={
                        "Authorization": f"Bearer {rpc_token}",
                        "Content-Type": "application/json",
                    },
                    json=sidecar_request,
                )
                sidecar_payload = sidecar_response.json()
                sidecar_result = (
                    sidecar_payload.get("result")
                    if isinstance(sidecar_payload, Mapping)
                    else None
                )
                sidecar_probe = {
                    "http_status": sidecar_response.status_code,
                    "payload": sidecar_payload,
                    "ready": (
                        sidecar_response.status_code == 200
                        and isinstance(sidecar_result, Mapping)
                        and sidecar_result.get("status") == "ready"
                    ),
                }
            except (httpx.RequestError, ValueError) as exc:
                sidecar_probe = {
                    "ready": False,
                    "error_class": type(exc).__name__,
                }
            latest = {
                "brain": brain_probe,
                "sidecar": sidecar_probe,
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
            }
            if brain_probe["ready"] and sidecar_probe["ready"]:
                _write_json(artifact_dir / "readiness.json", latest)
                return latest
            await asyncio.sleep(0.25)
    _write_json(artifact_dir / "readiness.json", latest)
    raise AssertionError("Brain and sidecar readiness timed out")


async def _serve_embedded_brain(server: Any) -> None:
    """Run Uvicorn without allowing its startup exit to escape the case."""

    try:
        await server.serve()
    except SystemExit as exc:
        raise RuntimeError(
            f"embedded Brain server exited with code {exc.code}"
        ) from exc


async def _stop_embedded_brain(
    *,
    server: Any | None,
    server_task: asyncio.Task[Any] | None,
) -> dict[str, Any]:
    """Close the exact embedded server task and report how it stopped."""

    result: dict[str, Any] = {
        "server_stopped": False,
        "server_shutdown_mode": "not_started",
        "server_exit_error": None,
        "warnings": [],
        "errors": [],
    }
    if server is None or server_task is None:
        result["server_stopped"] = True
        return result

    server.should_exit = True
    if not server_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(server_task), timeout=45)
            result["server_shutdown_mode"] = "graceful"
        except TimeoutError:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - retain shutdown evidence.
                result["server_exit_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            result["server_shutdown_mode"] = "forced_cancel"
            result["warnings"].append(
                "embedded Brain exceeded graceful shutdown deadline"
            )
        except Exception as exc:  # noqa: BLE001 - retain server exit evidence.
            result["server_shutdown_mode"] = "exited_with_error"
            result["server_exit_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["server_shutdown_mode"] = "already_stopped"

    result["server_stopped"] = server_task.done()
    if server_task.done() and not server_task.cancelled():
        task_error = server_task.exception()
        if task_error is not None:
            result["server_exit_error"] = (
                f"{type(task_error).__name__}: {task_error}"
            )
    if not result["server_stopped"]:
        result["errors"].append("embedded Brain task remained active")
    return result


async def _register_adapter(
    adapter: _LoopbackAdapterServer,
) -> dict[str, Any]:
    """Register the live debug adapter through the public runtime route."""

    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID

    callback_secret = adapter._shared_secret
    brain_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{brain_url}/runtime/adapters/register",
            json={
                "platform": "debug",
                "callback_url": adapter.callback_url,
                "platform_bot_id": CHARACTER_GLOBAL_USER_ID,
                "shared_secret": callback_secret,
                "timeout_seconds": 30.0,
            },
        )
    if response.status_code != 200:
        raise AssertionError(
            f"adapter registration failed: {response.status_code} {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("adapter registration response is not an object")
    return dict(payload)


def _chat_request(
    *,
    channel_id: str,
    user_id: str,
    message_id: str,
    text: str,
) -> dict[str, object]:
    """Build one ordinary private debug chat request."""

    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    return {
        "platform": "debug",
        "platform_channel_id": channel_id,
        "channel_type": "private",
        "platform_message_id": message_id,
        "platform_user_id": user_id,
        "platform_bot_id": CHARACTER_GLOBAL_USER_ID,
        "display_name": "DSH E2E User",
        "channel_name": "DSH E2E Private",
        "content_type": "text",
        "message_envelope": {
            "body_text": text,
            "raw_wire_text": text,
            "mentions": [],
            "reply": None,
            "attachments": [],
            "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
        },
        "local_timestamp": build_turn_clock()["local_timestamp"],
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
    }


async def _post_chat(request_payload: Mapping[str, object]) -> dict[str, Any]:
    """Submit one request through the public Brain HTTP boundary."""

    brain_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    control_secret = os.environ["KAZUSA_CONTROL_BRAIN_SHARED_SECRET"]
    timeout = httpx.Timeout(
        LIVE_HTTP_TIMEOUT_SECONDS,
        connect=10.0,
        read=LIVE_HTTP_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{brain_url}/chat",
            headers={
                "X-Kazusa-Control-Console": "debug-v1",
                "X-Kazusa-Control-Console-Auth": control_secret,
            },
            json=dict(request_payload),
        )
    if response.status_code != 200:
        raise AssertionError(
            f"public chat returned {response.status_code}: {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("public chat response is not an object")
    return dict(payload)


async def _read_collection(
    collection_name: str,
    query: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    """Read one guarded collection without ObjectId fields."""

    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    cursor = database[collection_name].find(dict(query or {}), {"_id": 0})
    rows = await cursor.to_list(length=None)
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _dsh_store_path() -> Path:
    """Return the current case's isolated standard-session store path."""

    from agentic_resolver.contracts import DSH_RELEASE

    return (
        Path(os.environ["KAZUSA_DSH_DATA_ROOT"])
        / "dsh"
        / DSH_RELEASE
        / "sessions.sqlite"
    )


def _binding_sidecar_session_id(binding: Mapping[str, object]) -> str:
    """Derive the canonical sidecar session id for one persisted binding."""

    thread_id = str(binding.get("resolution_thread_id") or "").strip()
    segment_id = str(binding.get("segment_id") or "").strip()
    if not thread_id or not segment_id:
        return ""
    identity = f"{thread_id}\0{segment_id}".encode()
    return "kazusa-resolution-" + sha256(identity).hexdigest()[:32]


def _read_session_events(
    connection: sqlite3.Connection,
    session_id: str,
) -> list[dict[str, Any]]:
    """Read and decode every persisted event for one exact session id."""

    import zstandard

    rows = connection.execute(
        "SELECT seq, type, data FROM events "
        "WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    decoder = zstandard.ZstdDecompressor()
    events: list[dict[str, Any]] = []
    for sequence, event_type, raw_data in rows:
        data = raw_data
        if isinstance(data, bytes):
            data = decoder.decompress(data).decode("utf-8")
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            events.append({
                "sequence": sequence,
                "event_type": event_type,
                "data": parsed,
            })
    return events


def _read_all_dsh_sessions(
    bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enumerate sidecar sessions independently of Brain binding rows."""

    store_path = _dsh_store_path()
    if not store_path.is_file():
        return []
    uri = f"file:{store_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        schema_row = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'sessions'",
        ).fetchone()
        if schema_row is None:
            return []
        session_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT id FROM sessions ORDER BY id",
            ).fetchall()
        ]
        binding_ids_by_session: dict[str, list[str]] = {}
        for binding in bindings:
            session_id = _binding_sidecar_session_id(binding)
            if not session_id:
                continue
            binding_ids_by_session.setdefault(session_id, []).append(
                str(binding.get("task_session_id") or "")
            )
        return [
            {
                "session_id": session_id,
                "binding_task_session_ids": binding_ids_by_session.get(
                    session_id,
                    [],
                ),
                "events": _read_session_events(connection, session_id),
            }
            for session_id in session_ids
        ]


async def _capture_case_evidence(
    *,
    source_output: Mapping[str, object],
    adapter: _LoopbackAdapterServer,
) -> dict[str, Any]:
    """Capture the stable stores and protected traces before cleanup."""

    collection_names = (
        "dsh_task_bindings",
        "accepted_tasks",
        "background_work_jobs",
        "internal_action_latches",
        "calendar_runs",
        "calendar_schedules",
        "self_cognition_group_review_windows",
        "self_cognition_action_attempts",
        "llm_trace_runs",
        "llm_trace_steps",
        "conversation_history",
    )
    mongo_state: dict[str, list[dict[str, Any]]] = {}
    for collection_name in collection_names:
        mongo_state[collection_name] = await _read_collection(collection_name)
    bindings = mongo_state["dsh_task_bindings"]
    dsh_sessions = _read_all_dsh_sessions(bindings)
    evidence = {
        "source_output": dict(source_output),
        "mongo_state": mongo_state,
        "dsh_sessions": dsh_sessions,
        "adapter": {
            "capability_payloads": adapter.capability_payloads(),
            "delivery_payloads": adapter.delivery_payloads(),
        },
    }
    return evidence


def _case_identity(case_id: str) -> dict[str, str]:
    """Build stable, case-scoped adapter and user identity fields."""

    suffix = sha256(case_id.encode("utf-8")).hexdigest()[:12]
    return {
        "platform": "debug",
        "channel_id": f"dsh-e2e-private-{suffix}",
        "user_id": f"dsh-e2e-user-{suffix}",
        "platform_user_id": f"dsh-e2e-platform-user-{suffix}",
        "display_name": "DSH E2E User",
        "source_message_id": f"dsh-e2e-source-{suffix}",
    }


async def _seed_user_context(
    *,
    identity: Mapping[str, str],
    body_text: str,
    timestamp: datetime,
) -> None:
    """Seed one real user profile and one private source conversation row."""

    from kazusa_ai_chatbot.db import create_user_profile
    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    user_id = identity["user_id"]
    existing = await database.user_profiles.find_one(
        {"global_user_id": user_id},
        {"_id": 1},
    )
    if existing is None:
        await create_user_profile(
            {
                "global_user_id": user_id,
                "display_name": identity["display_name"],
                "platform_accounts": [
                    {
                        "platform": identity["platform"],
                        "platform_user_id": identity["platform_user_id"],
                        "display_name": identity["display_name"],
                    }
                ],
            }
        )
    timestamp_text = timestamp.isoformat()
    await database.conversation_history.insert_one(
        {
            "platform": identity["platform"],
            "platform_channel_id": identity["channel_id"],
            "channel_type": "private",
            "channel_name": "DSH E2E Private",
            "role": "user",
            "platform_message_id": identity["source_message_id"],
            "platform_user_id": identity["platform_user_id"],
            "global_user_id": user_id,
            "display_name": identity["display_name"],
            "body_text": body_text,
            "raw_wire_text": body_text,
            "content_type": "text",
            "addressed_to_global_user_ids": [],
            "mentions": [],
            "broadcast": False,
            "attachments": [],
            "timestamp": timestamp_text,
            "received_at": timestamp_text,
        }
    )


async def _character_profile_with_platform_bot(service: Any) -> dict[str, Any]:
    """Load the live profile and attach the active adapter bot identity."""

    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID

    profile = await service._current_character_profile_snapshot()
    projected = dict(profile)
    projected["platform_bot_id"] = CHARACTER_GLOBAL_USER_ID
    return projected


async def _drain_background_work(service: Any) -> list[dict[str, Any]]:
    """Run bounded production worker/delivery ticks until no job remains active."""

    from kazusa_ai_chatbot.background_work.runtime import (
        run_background_work_runtime_tick,
    )

    tick_results: list[dict[str, Any]] = []
    for _ in range(8):
        active_before = await _read_collection(
            "background_work_jobs",
            {
                "status": {
                    "$in": [
                        "queued",
                        "in_progress",
                        "completed",
                        "failed",
                        "delivery_failed",
                    ]
                },
                "delivery_state": {"$ne": "delivered"},
            },
        )
        if not active_before:
            break
        tick = await run_background_work_runtime_tick(
            is_primary_interaction_busy=lambda: False,
            deliver_result_episode_func=(
                service._deliver_accepted_task_result_episode
            ),
        )
        tick_results.append(dict(tick))
        active_after = await _read_collection(
            "background_work_jobs",
            {
                "status": {
                    "$in": [
                        "queued",
                        "in_progress",
                        "completed",
                        "failed",
                        "delivery_failed",
                    ]
                },
                "delivery_state": {"$ne": "delivered"},
            },
        )
        if not active_after:
            break
        if tick.get("processed_count") == 0 and tick.get(
            "delivery_processed_count"
        ) == 0:
            break
    return tick_results


async def _run_user_message_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Enter cognition through public ``POST /chat``."""

    identity = _case_identity(spec.case_id)
    if spec.case_id == "user_message_local_fact":
        text = (
            "Please check rollout/status_note.txt and tell me who owns the "
            "rollout and what must happen first."
        )
    else:
        text = (
            "Please handle this in the background: read "
            "handover/incident_note.txt and send me a brief summary when the "
            "work is finished."
        )
    request = _chat_request(
        channel_id=identity["channel_id"],
        user_id=identity["platform_user_id"],
        message_id=identity["source_message_id"],
        text=text,
    )
    response = await _post_chat(request)
    background_ticks = await _drain_background_work(service)
    return {
        "entrypoint": "POST /chat",
        "identity": identity,
        "input_text": text,
        "response": response,
        "background_ticks": background_ticks,
    }


async def _run_internal_thought_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Issue and consume one real durable internal-action latch."""

    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import issue_internal_action_latch
    from kazusa_ai_chatbot.self_cognition.worker import (
        run_self_cognition_worker_tick,
    )

    identity = _case_identity(spec.case_id)
    now = datetime.now(UTC).replace(microsecond=0)
    if spec.case_id == "internal_thought_file_check":
        objective = (
            "Before continuing the planned update, inspect "
            "internal/health_note.txt and determine the current service state "
            "and next checkpoint."
        )
    else:
        objective = (
            "Before continuing the planned update, compare "
            "internal/early_shift.txt with internal/late_shift.txt and "
            "determine what changed."
        )
    await _seed_user_context(
        identity=identity,
        body_text="Please keep the operational update grounded in the notes.",
        timestamp=now - timedelta(minutes=3),
    )
    latch = await issue_internal_action_latch(
        source_episode_id=f"episode:{spec.case_id}",
        source_action_attempt_id=f"action-attempt:{spec.case_id}",
        continuation_objective=objective,
        evidence_refs=[
            {
                "evidence_id": f"evidence:{spec.case_id}",
                "excerpt": objective,
            }
        ],
        target_scope={
            "platform": identity["platform"],
            "platform_channel_id": identity["channel_id"],
            "channel_type": "private",
            "current_platform_user_id": identity["platform_user_id"],
            "current_global_user_id": identity["user_id"],
            "current_display_name": identity["display_name"],
            "target_addressed_user_ids": [identity["user_id"]],
            "target_broadcast": False,
            "source_platform_bot_id": CHARACTER_GLOBAL_USER_ID,
        },
        privacy_scope="private",
        continuation_depth=0,
        now=now.isoformat(),
    )
    profile = await _character_profile_with_platform_bot(service)
    worker_result = await run_self_cognition_worker_tick(
        now=now,
        is_primary_interaction_busy=lambda: False,
        character_profile=profile,
        adapter_registry_provider=lambda: service._adapter_registry,
        pipeline_coordinator=service._pipeline_coordinator,
        max_cases=1,
    )
    background_ticks = await _drain_background_work(service)
    return {
        "entrypoint": "self_cognition.worker internal latch claim",
        "identity": identity,
        "continuation_objective": objective,
        "latch_id": latch["latch_id"],
        "worker_result": asdict(worker_result),
        "background_ticks": background_ticks,
    }


async def _run_scheduled_tick_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Seed one due calendar producer and let the real worker claim it."""

    from kazusa_ai_chatbot.calendar_scheduler import handlers, models, repository
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import insert_user_memory_units
    from kazusa_ai_chatbot.self_cognition.worker import (
        run_self_cognition_worker_tick,
    )

    identity = _case_identity(spec.case_id)
    now = datetime.now(UTC).replace(microsecond=0)
    due_at = (now - timedelta(minutes=1)).isoformat()
    await _seed_user_context(
        identity=identity,
        body_text="The scheduled follow-up should use the local source note.",
        timestamp=now - timedelta(minutes=5),
    )

    if spec.case_id == "scheduled_tick_commitment_due":
        units = await insert_user_memory_units(
            identity["user_id"],
            [
                {
                    "unit_id": f"commitment-{spec.case_id}",
                    "unit_type": "active_commitment",
                    "fact": (
                        "I promised to inspect scheduled/commitment_note.txt "
                        "when this commitment became due and report its owner "
                        "and prerequisite."
                    ),
                    "subjective_appraisal": (
                        "The due check should be completed from the source."
                    ),
                    "relationship_signal": (
                        "Following through preserves continuity with the user."
                    ),
                    "status": "active",
                    "due_at": due_at,
                }
            ],
            storage_timestamp_utc=(now - timedelta(minutes=6)).isoformat(),
            include_embeddings=False,
        )
        unit = units[0]
        schedule_result = (
            await handlers.reconcile_active_commitment_calendar_schedule(
                unit,
                repository=repository,
                storage_timestamp_utc=now.isoformat(),
            )
        )
        source_objective = unit["fact"]
    else:
        source_objective = (
            "At this due time, inspect scheduled/future_note.txt and determine "
            "whether the planned rollout condition is satisfied."
        )
        source_scope = {
            "source_platform": identity["platform"],
            "source_channel_id": identity["channel_id"],
            "source_channel_type": "private",
            "source_user_id": identity["user_id"],
            "source_message_id": identity["source_message_id"],
            "source_platform_bot_id": CHARACTER_GLOBAL_USER_ID,
            "source_character_name": "Kazusa",
            "guild_id": None,
            "bot_role": "user",
        }
        schedule = models.build_one_time_calendar_schedule(
            trigger_kind=models.TRIGGER_FUTURE_COGNITION,
            due_at=due_at,
            payload={
                "episode_type": "self_cognition",
                "trigger_at": due_at,
                "continuation_objective": source_objective,
                "source_action_attempt_id": f"attempt:{spec.case_id}",
                "source_refs": [],
                "continuation": {
                    "mode": "scheduled_followup",
                    "episode_type": "self_cognition",
                    "max_depth": 1,
                    "include_result_as": "scheduled_event",
                },
            },
            source_scope=source_scope,
            idempotency_key=f"dsh-e2e:{spec.case_id}",
            storage_timestamp_utc=now.isoformat(),
        )
        run = models.build_calendar_run_from_schedule(
            schedule,
            due_at=due_at,
            storage_timestamp_utc=now.isoformat(),
        )
        await repository.upsert_calendar_schedule(schedule)
        await repository.upsert_calendar_run(run)
        schedule_result = {
            "status": "scheduled",
            "run_id": run["run_id"],
        }

    profile = await _character_profile_with_platform_bot(service)
    worker_result = await run_self_cognition_worker_tick(
        now=now,
        is_primary_interaction_busy=lambda: False,
        character_profile=profile,
        adapter_registry_provider=lambda: service._adapter_registry,
        pipeline_coordinator=service._pipeline_coordinator,
        max_cases=1,
    )
    background_ticks = await _drain_background_work(service)
    return {
        "entrypoint": "calendar run through self_cognition.worker",
        "identity": identity,
        "source_objective": source_objective,
        "schedule_result": schedule_result,
        "worker_result": asdict(worker_result),
        "background_ticks": background_ticks,
    }


def _group_message(
    *,
    role: str,
    body_text: str,
    timestamp: datetime,
    message_id: str,
    participant_user_id: str,
    character_user_id: str,
) -> dict[str, Any]:
    """Build one reflection-scope message without raw platform syntax."""

    is_assistant = role == "assistant"
    return {
        "role": role,
        "body_text": body_text,
        "timestamp": timestamp.isoformat(),
        "display_name": "Kazusa" if is_assistant else "Group Participant",
        "platform_message_id": message_id,
        "platform_user_id": (
            character_user_id if is_assistant else participant_user_id
        ),
        "global_user_id": (
            character_user_id if is_assistant else participant_user_id
        ),
        "addressed_to_global_user_ids": (
            [] if is_assistant else [character_user_id]
        ),
        "mentions": [],
    }


async def _run_group_self_cognition_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Run reflection-owned group review with no fabricated target user."""

    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import create_user_profile
    from kazusa_ai_chatbot.reflection_cycle import repository
    from kazusa_ai_chatbot.reflection_cycle import worker as reflection_worker
    from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput

    identity = _case_identity(spec.case_id)
    participant_user_id = identity["user_id"]
    database_profiles = await _read_collection(
        "user_profiles",
        {"global_user_id": participant_user_id},
    )
    if not database_profiles:
        await create_user_profile(
            {
                "global_user_id": participant_user_id,
                "display_name": "Group Participant",
            }
        )

    now = datetime.now(UTC).replace(microsecond=0)
    scope_ref = f"dsh-group-review:{spec.case_id}:{uuid4().hex[:8]}"
    channel_id = f"dsh-e2e-group-{uuid4().hex[:8]}"
    if spec.case_id == "self_cognition_promoted_group":
        hour_start = (now - timedelta(hours=2)).replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        messages = [
            _group_message(
                role="user",
                body_text=(
                    "The standing channel rule is to keep shared facts "
                    "separate from private user details."
                ),
                timestamp=hour_start + timedelta(minutes=5),
                message_id=f"{spec.case_id}-user-1",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
            _group_message(
                role="assistant",
                body_text=(
                    "I will preserve that shared-versus-private boundary in "
                    "this channel."
                ),
                timestamp=hour_start + timedelta(minutes=7),
                message_id=f"{spec.case_id}-assistant-1",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
            _group_message(
                role="user",
                body_text=(
                    "That rule still matters in the next part of the "
                    "conversation."
                ),
                timestamp=hour_start + timedelta(minutes=65),
                message_id=f"{spec.case_id}-user-2",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
            _group_message(
                role="assistant",
                body_text=(
                    "I will keep applying the same channel boundary as the "
                    "conversation continues."
                ),
                timestamp=hour_start + timedelta(minutes=67),
                message_id=f"{spec.case_id}-assistant-2",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
        ]
    else:
        messages = [
            _group_message(
                role="user",
                body_text=(
                    "Could someone check group/shared_status.txt before the "
                    "channel settles its shared policy?"
                ),
                timestamp=now - timedelta(minutes=25),
                message_id=f"{spec.case_id}-user",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
            _group_message(
                role="assistant",
                body_text=(
                    "I will review the channel context before deciding how to "
                    "respond to the group."
                ),
                timestamp=now - timedelta(minutes=23),
                message_id=f"{spec.case_id}-assistant",
                participant_user_id=participant_user_id,
                character_user_id=CHARACTER_GLOBAL_USER_ID,
            ),
        ]
    scope = ReflectionScopeInput(
        scope_ref=scope_ref,
        platform="debug",
        platform_channel_id=channel_id,
        channel_type="group",
        assistant_message_count=sum(
            message["role"] == "assistant" for message in messages
        ),
        user_message_count=sum(
            message["role"] == "user" for message in messages
        ),
        total_message_count=len(messages),
        first_timestamp=str(messages[0]["timestamp"]),
        last_timestamp=str(messages[-1]["timestamp"]),
        messages=messages,
    )

    promotion: dict[str, Any] = {"mode": "none"}
    if spec.case_id == "self_cognition_promoted_group":
        hourly_run_ids: list[str] = []
        for _ in range(2):
            hourly_result = await reflection_worker._run_hourly_reflection_for_scope(
                now=now,
                channel_scope=scope,
                dry_run=False,
                is_primary_interaction_busy=lambda: False,
            )
            hourly_run_ids.extend(str(value) for value in hourly_result.run_ids)
        if len(hourly_run_ids) != 2:
            raise AssertionError(
                "promoted group case did not settle two hourly roots"
            )
        first_hourly = await repository.reflection_run_by_id(hourly_run_ids[0])
        if first_hourly is None:
            raise AssertionError("promoted group hourly root is unavailable")
        character_local_date = str(first_hourly["character_local_date"])

        class _ExpectedHourlyRuns:
            async def expected_hourly_runs_for_character_local_date(
                self,
                *,
                character_local_date: str,
            ) -> list[object]:
                if character_local_date != expected_local_date:
                    return []
                return [
                    reflection_worker.ExpectedDailyChannelHourlyRuns(
                        channel_scope=scope,
                        expected_run_ids=list(hourly_run_ids),
                    )
                ]

        expected_local_date = character_local_date
        daily_result = (
            await reflection_worker._run_daily_channel_reflection_cycle(
                character_local_date=character_local_date,
                dry_run=False,
                is_primary_interaction_busy=lambda: False,
                phase_run_provider=_ExpectedHourlyRuns(),
            )
        )
        promotion = {
            "mode": "hourly_then_daily",
            "hourly_run_ids": hourly_run_ids,
            "daily_result": asdict(daily_result),
        }

    worker_result = (
        await reflection_worker._run_group_self_cognition_review_for_scope(
            now=now,
            channel_scope=scope,
            is_primary_interaction_busy=lambda: False,
            adapter_registry_provider=lambda: service._adapter_registry,
            pipeline_coordinator=service._pipeline_coordinator,
        )
    )
    return {
        "entrypoint": "reflection_cycle group self-cognition review",
        "scope_ref": scope_ref,
        "channel_id": channel_id,
        "messages": messages,
        "promotion": promotion,
        "worker_result": asdict(worker_result),
    }


def _task_scene_context() -> dict[str, str]:
    """Build one exact prompt-safe task scene for result recurrence."""

    return {
        "channel_scope": "private",
        "character_role": "Kazusa",
        "current_user_role": "DSH E2E User",
        "semantic_scene": "A completed background task is ready for delivery.",
        "public_group_scene": "",
        "conversation_continuity": "The result continues the accepted task.",
        "semantic_temporal_context": "The result is ready now.",
    }


def _goal_continuation_ref(case_id: str, source_message_id: str) -> dict[str, Any]:
    """Build the canonical result-continuation identity for one case."""

    from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref

    return build_goal_continuation_ref(
        source_episode_id=f"episode:{case_id}",
        source_message_id=source_message_id,
        branch_id="result-delivery",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": f"goal:{case_id}",
        },
    )


def _task_execution_context(
    *,
    identity: Mapping[str, str],
    continuation_ref: Mapping[str, object],
    now: datetime,
) -> dict[str, Any]:
    """Build one exact V2 execution context for a seeded accepted task."""

    return {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Kazusa",
        "platform": identity["platform"],
        "channel_id": identity["channel_id"],
        "channel_type": "private",
        "requester_global_user_id": identity["user_id"],
        "requester_platform_user_id": identity["platform_user_id"],
        "requester_display_name": identity["display_name"],
        "source_message_id": identity["source_message_id"],
        "source_platform_bot_id": os.environ["CHARACTER_GLOBAL_USER_ID"],
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "",
        "brain_conversation_ref": (
            f"chat:debug:{identity['channel_id']}:{identity['user_id']}"
        ),
        "scene_context": _task_scene_context(),
        "goal_continuation_ref": dict(continuation_ref),
        "local_time_context": {"local_time": now.isoformat()},
        "prompt_message_context": {
            "body_text": "Complete the accepted background task."
        },
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "Kazusa is delivering a grounded task result.",
        "conversation_summary": "The user is waiting for the accepted result.",
        "current_timestamp_utc": now.isoformat(),
        "active_turn_platform_message_ids": [identity["source_message_id"]],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }


def _typed_task_result(
    *,
    case_id: str,
    continuation_ref: Mapping[str, object],
    resolved: bool,
) -> dict[str, Any]:
    """Build and validate one result-owned recurrence input."""

    from kazusa_ai_chatbot.task_resolution.contracts import (
        validate_task_resolution_result,
    )

    if resolved:
        status = "resolved"
        evidence_state = "complete"
        summary = (
            "The bounded source check completed and the requested handover "
            "fact is available."
        )
        evidence_excerpts = [summary]
        evidence_handles = [summary]
        evidence = [
            {
                "schema_version": "task_resolution_evidence.v1",
                "evidence_id": f"evidence:{case_id}",
                "task_node_id": f"task-node:{case_id}",
                "specialist": "dsh",
                "summary": summary,
                "provenance_refs": [f"local-result:{case_id}"],
                "limitations": [],
            }
        ]
        completed_subgoals = ["Checked the bounded source."]
        remaining_needs: list[str] = []
    else:
        status = "failed"
        evidence_state = "blocked"
        summary = (
            "The bounded source check could not complete because its source "
            "was unavailable."
        )
        evidence_excerpts = []
        evidence_handles = []
        evidence = []
        completed_subgoals = []
        remaining_needs = ["The unavailable source must be restored."]
    result = {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded handover source check.",
        "status": status,
        "scene_context": _task_scene_context(),
        "goal_continuation_ref": dict(continuation_ref),
        "evidence_state": evidence_state,
        "evidence_excerpts": evidence_excerpts,
        "evidence_handles": evidence_handles,
        "prompt_safe_summary": summary,
        "evidence": evidence,
        "completed_subgoals": completed_subgoals,
        "remaining_needs": remaining_needs,
        "checkpoint": {},
        "coding_run_context": {},
    }
    return dict(validate_task_resolution_result(result))


async def _run_tool_result_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Deliver one real completed job through the tool-result source owner."""

    from kazusa_ai_chatbot.accepted_task import (
        create_or_return_active_accepted_task,
        mark_accepted_task_failure_ready,
        mark_accepted_task_pending,
        mark_accepted_task_running,
        mark_tool_result_ready,
    )
    from kazusa_ai_chatbot.background_work.delivery import (
        run_background_work_delivery_tick,
    )
    from kazusa_ai_chatbot.background_work.jobs import (
        enqueue_background_work_request,
    )
    from kazusa_ai_chatbot.background_work.models import background_work_job_ref
    from kazusa_ai_chatbot.background_work.result_source import (
        build_result_ready_episode_from_job,
        validate_tool_result_cognition_source,
    )
    from kazusa_ai_chatbot.db.background_work_jobs import (
        claim_background_work_job,
        complete_background_work_job,
        fail_background_work_job,
    )

    identity = _case_identity(spec.case_id)
    now = datetime.now(UTC).replace(microsecond=0)
    await _seed_user_context(
        identity=identity,
        body_text="Please send the accepted background result when it is ready.",
        timestamp=now - timedelta(minutes=2),
    )
    continuation_ref = _goal_continuation_ref(
        spec.case_id,
        identity["source_message_id"],
    )
    create_result = await create_or_return_active_accepted_task(
        {
            "task_kind": "task_resolution",
            "semantic_objective": "Resolve one bounded handover source check.",
            "accepted_task_summary": "Check the bounded handover source.",
            "goal_continuation_ref": continuation_ref,
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
            "source_trigger_source": "user_message",
            "source_platform": identity["platform"],
            "source_channel_id": identity["channel_id"],
            "source_channel_type": "private",
            "source_message_id": identity["source_message_id"],
            "source_platform_bot_id": os.environ["CHARACTER_GLOBAL_USER_ID"],
            "source_character_name": "Kazusa",
            "requester_global_user_id": identity["user_id"],
            "requester_platform_user_id": identity["platform_user_id"],
            "requester_display_name": identity["display_name"],
            "storage_timestamp_utc": now.isoformat(),
        },
        dsh_task_session_id=f"session-{uuid4().hex}",
    )
    task = create_result["task"]
    accepted_task_id = str(task["accepted_task_id"])
    job_id = f"job-{uuid4().hex}"
    execution_context = _task_execution_context(
        identity=identity,
        continuation_ref=continuation_ref,
        now=now,
    )
    await enqueue_background_work_request(
        {
            "job_id": job_id,
            "source_action_attempt_id": f"attempt:{spec.case_id}",
            "source_llm_trace_id": "",
            "idempotency_key": f"background-work:{spec.case_id}",
            "accepted_task_id": accepted_task_id,
            "task_identity_key": str(task["task_identity_key"]),
            "semantic_objective": "Resolve one bounded handover source check.",
            "goal_continuation_ref": continuation_ref,
            "requested_worker": "task_orchestrator",
            "worker_payload": {
                "schema_version": "task_orchestrator_worker_payload.v2",
                "operation": "open_dsh_resolution",
                "task_session_id": str(task["dsh_task_session_id"]),
                "operation_generation": 0,
                "control": None,
            },
            "task_execution_context": execution_context,
            "source_platform": identity["platform"],
            "source_channel_id": identity["channel_id"],
            "source_channel_type": "private",
            "source_message_id": identity["source_message_id"],
            "source_platform_bot_id": os.environ["CHARACTER_GLOBAL_USER_ID"],
            "source_character_name": "Kazusa",
            "requester_global_user_id": identity["user_id"],
            "requester_platform_user_id": identity["platform_user_id"],
            "requester_display_name": identity["display_name"],
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 3000,
            "storage_timestamp_utc": now.isoformat(),
        }
    )
    pending_task = await mark_accepted_task_pending(
        accepted_task_id=accepted_task_id,
        executor_ref=background_work_job_ref(job_id),
        updated_at=now.isoformat(),
    )
    if pending_task is None:
        raise AssertionError("seeded accepted task did not become pending")
    lease_owner = f"dsh-e2e-seed-{uuid4().hex}"
    claimed_job = await claim_background_work_job(
        lease_owner=lease_owner,
        lease_seconds=300,
        now_utc=now.isoformat(),
        max_attempts=3,
    )
    if claimed_job is None or claimed_job.get("job_id") != job_id:
        raise AssertionError("seeded background job was not claimable")
    running_task = await mark_accepted_task_running(
        accepted_task_id=accepted_task_id,
        started_at=now.isoformat(),
    )
    if running_task is None:
        raise AssertionError("seeded accepted task did not become running")
    resolved = spec.case_id == "tool_result_resolved"
    task_result = _typed_task_result(
        case_id=spec.case_id,
        continuation_ref=continuation_ref,
        resolved=resolved,
    )
    completed_at = (now + timedelta(seconds=1)).isoformat()
    if resolved:
        terminal_job = await complete_background_work_job(
            job_id=job_id,
            lease_owner=lease_owner,
            task_resolution_result=task_result,
            artifact_text=task_result["prompt_safe_summary"],
            result_summary=task_result["prompt_safe_summary"],
            completed_at=completed_at,
        )
        ready_task = await mark_tool_result_ready(
            accepted_task_id=accepted_task_id,
            artifact_text=task_result["prompt_safe_summary"],
            result_summary=task_result["prompt_safe_summary"],
            completed_at=completed_at,
        )
    else:
        terminal_job = await fail_background_work_job(
            job_id=job_id,
            lease_owner=lease_owner,
            failure_summary=task_result["prompt_safe_summary"],
            result_summary=task_result["prompt_safe_summary"],
            failed_at=completed_at,
            task_resolution_result=task_result,
        )
        ready_task = await mark_accepted_task_failure_ready(
            accepted_task_id=accepted_task_id,
            failure_summary=task_result["prompt_safe_summary"],
            completed_at=completed_at,
            remaining_needs=list(task_result["remaining_needs"]),
        )
    if terminal_job is None or ready_task is None:
        raise AssertionError("seeded result did not reach its ready state")
    episode = build_result_ready_episode_from_job(terminal_job)
    cognition_source = validate_tool_result_cognition_source(
        episode["percepts"][0]["content"]["cognition_source"]
    )
    delivery_result = await run_background_work_delivery_tick(
        deliver_result_episode_func=service._deliver_accepted_task_result_episode,
        limit=1,
    )
    return {
        "entrypoint": "background_work.delivery",
        "identity": identity,
        "accepted_task_id": accepted_task_id,
        "job_id": job_id,
        "seeded_task_result": task_result,
        "episode_trigger_source": episode["trigger_source"],
        "cognition_source": cognition_source,
        "delivery_result": delivery_result,
    }


def _binding_trigger_source(binding: Mapping[str, object]) -> str:
    """Read the canonical source from one persisted DSH start context."""

    start_spec = binding.get("start_spec")
    if not isinstance(start_spec, Mapping):
        return ""
    context = start_spec.get("execution_context")
    if not isinstance(context, Mapping):
        return ""
    value = context.get("source_trigger_source")
    return value.strip() if isinstance(value, str) else ""


def _terminal_dsh_event_count(evidence: Mapping[str, object]) -> int:
    """Count typed terminal-resolution events across captured root sessions."""

    count = 0
    sessions = evidence.get("dsh_sessions")
    if not isinstance(sessions, list):
        return 0
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        events = session.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if event.get("event_type") != "tool/result":
                continue
            data = event.get("data")
            if not isinstance(data, Mapping):
                continue
            meta = data.get("meta")
            kazusa = meta.get("kazusa") if isinstance(meta, Mapping) else None
            if (
                isinstance(kazusa, Mapping)
                and kazusa.get("kind") == "terminal_resolution_v2"
            ):
                count += 1
    return count


def _source_trace_rows(
    spec: TriggerSourceCaseSpec,
    evidence: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Select trace-run rows belonging to the source under test."""

    mongo_state = evidence.get("mongo_state")
    if not isinstance(mongo_state, Mapping):
        return []
    raw_rows = mongo_state.get("llm_trace_runs")
    rows = [dict(row) for row in raw_rows or [] if isinstance(row, Mapping)]
    source_output = evidence.get("source_output")
    source = source_output if isinstance(source_output, Mapping) else {}
    if spec.trigger_source == "user_message":
        response = source.get("response")
        response_mapping = response if isinstance(response, Mapping) else {}
        trace_id = str(response_mapping.get("trace_id") or "").strip()
        if not trace_id:
            graph = response_mapping.get("cognition_graph")
            correlation = (
                graph.get("correlation") if isinstance(graph, Mapping) else None
            )
            if isinstance(correlation, Mapping):
                trace_id = str(correlation.get("llm_trace_id") or "").strip()
        return [row for row in rows if row.get("trace_id") == trace_id]
    if spec.trigger_source == "tool_result":
        job_id = str(source.get("job_id") or "")
        return [
            row
            for row in rows
            if row.get("source_background_work_job_id") == job_id
        ]
    if spec.trigger_source == "internal_thought":
        return [
            row
            for row in rows
            if "self_cognition:internal-thought:" in str(
                row.get("platform_message_id") or ""
            )
        ]
    if spec.trigger_source == "scheduled_tick":
        return [
            row for row in rows if str(row.get("source_calendar_run_id") or "")
        ]
    return [
        row
        for row in rows
        if str(source.get("scope_ref") or "")
        and str(source.get("scope_ref") or "")
        in str(row.get("platform_message_id") or "")
        and str(row.get("platform_message_id") or "").startswith(
            "self_cognition:group_activity_window:"
        )
    ]


async def _wait_for_source_trace_settlement(
    spec: TriggerSourceCaseSpec,
    source_output: Mapping[str, object],
) -> dict[str, Any]:
    """Wait until the source-bound trace exists and leaves running state."""

    started_at = time.perf_counter()
    deadline = time.monotonic() + SOURCE_TRACE_SETTLEMENT_TIMEOUT_SECONDS
    latest_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        trace_rows = await _read_collection("llm_trace_runs")
        evidence = {
            "source_output": source_output,
            "mongo_state": {"llm_trace_runs": trace_rows},
        }
        latest_rows = _source_trace_rows(spec, evidence)
        if latest_rows and all(
            str(row.get("status") or "") != "running"
            for row in latest_rows
        ):
            return {
                "settled": True,
                "trace_ids": [row.get("trace_id") for row in latest_rows],
                "statuses": [row.get("status") for row in latest_rows],
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000
                ),
            }
        await asyncio.sleep(0.25)
    return {
        "settled": False,
        "trace_ids": [row.get("trace_id") for row in latest_rows],
        "statuses": [row.get("status") for row in latest_rows],
        "duration_ms": round((time.perf_counter() - started_at) * 1000),
    }


def _delivery_payloads(evidence: Mapping[str, object]) -> list[dict[str, Any]]:
    """Return captured outward adapter messages."""

    adapter = evidence.get("adapter")
    if not isinstance(adapter, Mapping):
        return []
    values = adapter.get("delivery_payloads")
    return [dict(value) for value in values or [] if isinstance(value, Mapping)]


def _worker_succeeded(source_output: Mapping[str, object]) -> bool:
    """Return whether one source worker processed a case without failure."""

    result = source_output.get("worker_result")
    if not isinstance(result, Mapping):
        return False
    return (
        result.get("processed_count") == 1
        and result.get("failed_count") == 0
        and result.get("deferred") is False
    )


def _evaluate_case(
    spec: TriggerSourceCaseSpec,
    evidence: Mapping[str, object],
) -> tuple[dict[str, bool], list[str]]:
    """Evaluate stable integration contracts without judging exact wording."""

    from kazusa_ai_chatbot.task_resolution.contracts import (
        TaskResolutionContractError,
        validate_task_resolution_result,
    )

    mongo_state = evidence.get("mongo_state")
    mongo = mongo_state if isinstance(mongo_state, Mapping) else {}
    bindings = [
        dict(value)
        for value in mongo.get("dsh_task_bindings", [])
        if isinstance(value, Mapping)
    ]
    accepted_tasks = [
        dict(value)
        for value in mongo.get("accepted_tasks", [])
        if isinstance(value, Mapping)
    ]
    jobs = [
        dict(value)
        for value in mongo.get("background_work_jobs", [])
        if isinstance(value, Mapping)
    ]
    traces = _source_trace_rows(spec, evidence)
    source_output_value = evidence.get("source_output")
    source_output = (
        source_output_value
        if isinstance(source_output_value, Mapping)
        else {}
    )
    sessions = [
        dict(value)
        for value in evidence.get("dsh_sessions", [])
        if isinstance(value, Mapping)
    ]
    checks: dict[str, bool] = {
        "source_trace_present": bool(traces),
        "source_trace_finalized": bool(traces)
        and all(str(row.get("status") or "") != "running" for row in traces),
    }

    if spec.expects_dsh_entry:
        checks["one_source_bound_dsh_binding"] = len(bindings) == 1
        checks["binding_preserves_trigger_source"] = (
            len(bindings) == 1
            and _binding_trigger_source(bindings[0]) == spec.trigger_source
        )
        checks["one_binding_matched_sidecar_session"] = (
            len(bindings) == 1
            and len(sessions) == 1
            and str(bindings[0].get("task_session_id") or "")
            in sessions[0].get("binding_task_session_ids", [])
        )
        latest_result = (
            bindings[0].get("latest_task_resolution_result")
            if len(bindings) == 1
            else None
        )
        validated_result: Mapping[str, object] | None = None
        try:
            validated_result = validate_task_resolution_result(latest_result)
        except (TaskResolutionContractError, TypeError, ValueError):
            validated_result = None
        checks["typed_terminal_task_result"] = (
            validated_result is not None
            and validated_result.get("status") in {"resolved", "partial"}
            and bool(validated_result.get("evidence"))
        )
        result_text = json.dumps(
            validated_result or {},
            ensure_ascii=False,
            default=str,
        ).lower()
        checks["expected_local_evidence_referenced"] = all(
            file_name.lower() in result_text
            for file_name in spec.expected_evidence_files
        )
        checks["typed_dsh_terminal_event"] = (
            _terminal_dsh_event_count(evidence) >= 1
        )
    else:
        checks["dsh_non_entry_preserved"] = len(bindings) == 0
        checks["no_sidecar_session_created"] = len(sessions) == 0

    if spec.trigger_source == "user_message":
        response = source_output.get("response")
        response_mapping = response if isinstance(response, Mapping) else {}
        messages = response_mapping.get("messages")
        visible_messages = [
            value
            for value in messages or []
            if isinstance(value, str) and value.strip()
        ]
        callbacks = _delivery_payloads(evidence)
        checks["public_chat_completed"] = not bool(
            response_mapping.get("operational_error")
        )
        checks["coherent_visible_answer_path"] = bool(
            visible_messages or callbacks
        )
        if spec.case_id == "user_message_background_summary":
            checks["durable_background_lineage"] = (
                len(accepted_tasks) == 1
                and len(jobs) == 1
                and accepted_tasks[0].get("state") == "delivered"
                and jobs[0].get("status") == "delivered"
                and jobs[0].get("delivery_state") == "delivered"
            )
            checks["background_result_callback"] = len(callbacks) == 1
    elif spec.trigger_source == "internal_thought":
        latches = [
            value
            for value in mongo.get("internal_action_latches", [])
            if isinstance(value, Mapping)
        ]
        checks["internal_worker_succeeded"] = _worker_succeeded(source_output)
        checks["latch_claim_consumed"] = (
            len(latches) == 1
            and latches[0].get("status") == "consumed"
            and bool(latches[0].get("consumed_episode_id"))
        )
    elif spec.trigger_source == "scheduled_tick":
        runs = [
            value
            for value in mongo.get("calendar_runs", [])
            if isinstance(value, Mapping)
        ]
        checks["scheduled_worker_succeeded"] = _worker_succeeded(source_output)
        checks["calendar_run_completed"] = (
            len(runs) == 1 and runs[0].get("status") == "completed"
        )
    elif spec.trigger_source == "self_cognition":
        ledgers = [
            value
            for value in mongo.get("self_cognition_group_review_windows", [])
            if isinstance(value, Mapping)
        ]
        checks["group_review_worker_succeeded"] = _worker_succeeded(
            source_output
        )
        checks["group_review_window_settled"] = (
            sum(value.get("status") == "reviewed" for value in ledgers) == 1
        )
    else:
        delivery = source_output.get("delivery_result")
        delivery_mapping = delivery if isinstance(delivery, Mapping) else {}
        callbacks = _delivery_payloads(evidence)
        cognition_source = source_output.get("cognition_source")
        cognition_mapping = (
            cognition_source if isinstance(cognition_source, Mapping) else {}
        )
        checks["tool_result_source_preserved"] = (
            source_output.get("episode_trigger_source") == "tool_result"
            and cognition_mapping.get("source_kind") == "tool_result"
        )
        checks["result_delivery_settled"] = (
            delivery_mapping.get("processed_count") == 1
            and delivery_mapping.get("delivered_count") == 1
            and delivery_mapping.get("failed_count") == 0
            and len(accepted_tasks) == 1
            and accepted_tasks[0].get("state") == "delivered"
            and len(jobs) == 1
            and jobs[0].get("status") == "delivered"
            and jobs[0].get("delivery_state") == "delivered"
        )
        checks["one_nonempty_result_callback"] = (
            len(callbacks) == 1
            and isinstance(callbacks[0].get("text"), str)
            and bool(str(callbacks[0]["text"]).strip())
        )
        expected_status = (
            "resolved" if spec.case_id == "tool_result_resolved" else "failed"
        )
        expected_evidence_state = (
            "complete" if expected_status == "resolved" else "blocked"
        )
        checks["typed_source_outcome_preserved"] = (
            cognition_mapping.get("task_status") == expected_status
            and cognition_mapping.get("evidence_state")
            == expected_evidence_state
        )

    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


def _write_evidence_artifacts(
    *,
    artifact_dir: Path,
    spec: TriggerSourceCaseSpec,
    evidence: Mapping[str, object],
    checks: Mapping[str, bool],
) -> None:
    """Write the technical dossier and review input without a behavior verdict."""

    mongo_state_value = evidence.get("mongo_state")
    mongo_state = (
        dict(mongo_state_value)
        if isinstance(mongo_state_value, Mapping)
        else {}
    )
    trace_runs = mongo_state.pop("llm_trace_runs", [])
    trace_steps = mongo_state.pop("llm_trace_steps", [])
    _write_json(artifact_dir / "source_execution.json", evidence["source_output"])
    _write_json(artifact_dir / "mongo_state.json", mongo_state)
    _write_json(
        artifact_dir / "dsh_lineage.json",
        {
            "bindings": mongo_state.get("dsh_task_bindings", []),
            "sessions": evidence.get("dsh_sessions", []),
        },
    )
    _write_json(
        artifact_dir / "trace.json",
        {
            "runs": trace_runs,
            "steps": trace_steps,
            "source_trace_rows": _source_trace_rows(spec, evidence),
        },
    )
    _write_json(artifact_dir / "callbacks.json", evidence.get("adapter", {}))
    bindings = mongo_state.get("dsh_task_bindings", [])
    task_results = [
        binding.get("latest_task_resolution_result")
        for binding in bindings
        if isinstance(binding, Mapping)
        and isinstance(binding.get("latest_task_resolution_result"), Mapping)
    ]
    _write_json(
        artifact_dir / "behavior_review_input.json",
        {
            "schema_version": "dsh_trigger_source_behavior_review_input.v1",
            "case_id": spec.case_id,
            "trigger_source": spec.trigger_source,
            "expected_integration_disposition": (
                "dsh_entry" if spec.expects_dsh_entry else "dsh_non_entry"
            ),
            "source_input_and_execution": evidence.get("source_output", {}),
            "typed_task_results": task_results,
            "outward_callbacks": _delivery_payloads(evidence),
            "source_trace_ids": [
                row.get("trace_id") for row in _source_trace_rows(spec, evidence)
            ],
            "automated_contract_checks": dict(checks),
            "review_questions": [
                "Was the source interpreted according to its real ownership?",
                "Was DSH entry or non-entry appropriate for the available authority?",
                "Is any factual result grounded in the retained typed evidence?",
                "Is the visible response, silence, or failure report coherent and in character?",
                "Did the result avoid inventing success, identity, or permission?",
            ],
            "behavior_decision": None,
        },
    )


async def _execute_source_case(
    spec: TriggerSourceCaseSpec,
    *,
    service: Any,
) -> dict[str, Any]:
    """Dispatch one case to its production trigger owner."""

    if spec.trigger_source == "user_message":
        return await _run_user_message_case(spec, service=service)
    if spec.trigger_source == "internal_thought":
        return await _run_internal_thought_case(spec, service=service)
    if spec.trigger_source == "self_cognition":
        return await _run_group_self_cognition_case(spec, service=service)
    if spec.trigger_source == "scheduled_tick":
        return await _run_scheduled_tick_case(spec, service=service)
    if spec.trigger_source == "tool_result":
        return await _run_tool_result_case(spec, service=service)
    raise AssertionError(f"unsupported trigger source: {spec.trigger_source}")


async def _run_case_process(
    *,
    case_id: str,
    artifact_dir: Path,
) -> int:
    """Own the complete live runtime, evidence, and cleanup for one case."""

    spec = CASE_SPECS[case_id]
    started_at = time.perf_counter()
    adapter = _LoopbackAdapterServer(
        platform="debug",
        shared_secret=f"adapter-{uuid4().hex}",
    )
    sidecar: _ChildProcess | None = None
    server: Any | None = None
    server_task: asyncio.Task[Any] | None = None
    source_output: dict[str, Any] = {}
    evidence: dict[str, Any] | None = None
    checks: dict[str, bool] = {}
    failures: list[str] = []
    error: dict[str, str] | None = None
    cleanup: dict[str, Any] = {
        "server_stopped": False,
        "server_shutdown_mode": "not_started",
        "server_exit_error": None,
        "adapter_stopped": False,
        "sidecar_stopped": False,
        "database_dropped": False,
        "warnings": [],
        "errors": [],
    }
    try:
        await _prepare_case_database(artifact_dir)
        import uvicorn

        from kazusa_ai_chatbot import service

        sidecar = await _launch_sidecar(artifact_dir)
        adapter.start()
        brain_port = urlsplit(os.environ["KAZUSA_DSH_BRAIN_URL"]).port
        if brain_port is None:
            raise AssertionError("case Brain URL has no explicit port")
        server_config = uvicorn.Config(
            service.app,
            host="127.0.0.1",
            port=brain_port,
            log_level="info",
            access_log=False,
            lifespan="on",
        )
        server = uvicorn.Server(server_config)
        server.install_signal_handlers = lambda: None
        server_task = asyncio.create_task(
            _serve_embedded_brain(server),
            name=f"dsh-e2e-brain-{case_id}",
        )
        _write_json(
            artifact_dir / "processes.json",
            {
                "case_process_pid": os.getpid(),
                "brain": {
                    "mode": "embedded_uvicorn",
                    "port": brain_port,
                },
                "sidecar": {
                    "pid": sidecar.process.pid,
                    "stdout_path": str(sidecar.stdout_path),
                    "stderr_path": str(sidecar.stderr_path),
                },
            },
        )
        await _wait_for_runtime_readiness(
            brain_task=server_task,
            sidecar=sidecar,
            artifact_dir=artifact_dir,
        )
        registration = await _register_adapter(adapter)
        _write_json(artifact_dir / "adapter_registration.json", registration)
        source_output = await _execute_source_case(spec, service=service)
        source_output["source_trace_settlement"] = (
            await _wait_for_source_trace_settlement(spec, source_output)
        )
        evidence = await _capture_case_evidence(
            source_output=source_output,
            adapter=adapter,
        )
        checks, failures = _evaluate_case(spec, evidence)
        _write_evidence_artifacts(
            artifact_dir=artifact_dir,
            spec=spec,
            evidence=evidence,
            checks=checks,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the case boundary dossier.
        error = {
            "error_class": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failures.append(f"case_exception:{type(exc).__name__}")
        _write_json(artifact_dir / "case_exception.json", error)
        if sidecar is not None:
            try:
                evidence = await _capture_case_evidence(
                    source_output=source_output,
                    adapter=adapter,
                )
                _write_evidence_artifacts(
                    artifact_dir=artifact_dir,
                    spec=spec,
                    evidence=evidence,
                    checks=checks,
                )
            except Exception as capture_exc:  # noqa: BLE001 - retain root error.
                _write_json(
                    artifact_dir / "evidence_capture_exception.json",
                    {
                        "error_class": type(capture_exc).__name__,
                        "error": str(capture_exc),
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        server_cleanup = await _stop_embedded_brain(
            server=server,
            server_task=server_task,
        )
        cleanup["server_stopped"] = server_cleanup["server_stopped"]
        cleanup["server_shutdown_mode"] = server_cleanup[
            "server_shutdown_mode"
        ]
        cleanup["server_exit_error"] = server_cleanup["server_exit_error"]
        cleanup["warnings"].extend(server_cleanup["warnings"])
        cleanup["errors"].extend(server_cleanup["errors"])
        if server_cleanup["server_exit_error"] and error is None:
            cleanup["errors"].append(
                f"server:{server_cleanup['server_exit_error']}"
            )
        try:
            adapter.stop()
            cleanup["adapter_stopped"] = True
        except Exception as exc:  # noqa: BLE001 - cleanup must continue.
            cleanup["errors"].append(
                f"adapter:{type(exc).__name__}: {exc}"
            )
        if sidecar is not None:
            try:
                await _stop_child(sidecar)
                cleanup["sidecar_stopped"] = True
            except Exception as exc:  # noqa: BLE001 - cleanup must continue.
                cleanup["errors"].append(
                    f"sidecar:{type(exc).__name__}: {exc}"
                )
        try:
            await _drop_case_database()
            cleanup["database_dropped"] = True
        except Exception as exc:  # noqa: BLE001 - cleanup must be reported.
            cleanup["errors"].append(
                f"database:{type(exc).__name__}: {exc}"
            )
        _write_json(artifact_dir / "cleanup.json", cleanup)

    if cleanup["errors"]:
        failures.append("cleanup_incomplete")
    technical_status = "passed" if not failures else "failed"
    _write_json(
        artifact_dir / "case_result.json",
        {
            "schema_version": "dsh_trigger_source_case_result.v1",
            "case_id": case_id,
            "trigger_source": spec.trigger_source,
            "expects_dsh_entry": spec.expects_dsh_entry,
            "technical_status": technical_status,
            "checks": checks,
            "failures": failures,
            "error": error,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        },
    )
    return 0 if technical_status == "passed" else 1


def _parse_args() -> argparse.Namespace:
    """Parse the private case-process command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-case", choices=tuple(CASE_SPECS))
    parser.add_argument("--artifact-dir", type=Path)
    return parser.parse_args()


def _main() -> int:
    """Run the isolated child mode used by the pytest wrappers."""

    args = _parse_args()
    if args.execute_case is None or args.artifact_dir is None:
        raise SystemExit("--execute-case and --artifact-dir are required")
    return asyncio.run(
        _run_case_process(
            case_id=args.execute_case,
            artifact_dir=args.artifact_dir.resolve(),
        )
    )


if __name__ == "__main__":
    raise SystemExit(_main())
