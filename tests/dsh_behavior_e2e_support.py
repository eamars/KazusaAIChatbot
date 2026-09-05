"""Isolated real-model support for DSH behavior contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import httpx
import pytest

from experiments.dsh_process_support import (
    SidecarProcess,
    start_configured_sidecar,
)
from tests.dsh_database_test_support import GuardedDshDiagnostics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
ARTIFACT_ROOT = PROJECT_ROOT / "test_artifacts" / "dsh_behavior_e2e"
DATABASE_PREFIX = "_test_kazusa_dsh_behavior_"
CHILD_TIMEOUT_SECONDS = 30 * 60
HTTP_TIMEOUT_SECONDS = 10 * 60

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class BehaviorCase:
    """One user-observable behavior contract and its local evidence."""

    case_id: str
    workspace_files: Mapping[str, str]
    user_inputs: list[str]
    interaction_inputs: list[dict[str, str]]
    behavior_contract: str
    input_kind: str
    hard_gates: list[str]
    behavior_rubric: list[str]
    acceptable_variation: list[str]
    forbidden_failure_modes: list[str]
    trace_required: list[str]


@dataclass
class BehaviorResources:
    """Record process evidence owned by the live scenario."""

    artifact_dir: Path
    processes: list[dict[str, object]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    cleanup: list[dict[str, object]] = field(default_factory=list)


class LiveProviderRecorder:
    """Forward real sidecar provider traffic and retain usage-bearing replies."""

    def __init__(self, upstream_base_url: str, artifact_dir: Path) -> None:
        self.calls: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                started = time.perf_counter()
                with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                    result = client.post(
                        upstream_base_url.rstrip("/") + self.path,
                        content=body,
                        headers={
                            "Authorization": self.headers.get("Authorization", ""),
                            "Content-Type": self.headers["Content-Type"],
                        },
                    )
                owner.calls.append({
                    "request": json.loads(body), "status": result.status_code,
                    "response_body": result.text,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                })
                _write_json(artifact_dir / "dsh_provider_calls.json", owner.calls)
                self.send_response(result.status_code)
                self.send_header("Content-Type", result.headers["Content-Type"])
                self.send_header("Content-Length", str(len(result.content)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(result.content)

            def log_message(self, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        """Join the real provider recording relay after its sidecar stops."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("live provider recording relay did not stop")


class CallbackAdapter:
    """Capture adapter capability checks and visible outbound delivery."""

    def __init__(self, *, shared_secret: str) -> None:
        self.shared_secret = shared_secret
        self._lock = Lock()
        self._capabilities: list[dict[str, Any]] = []
        self._deliveries: list[dict[str, Any]] = []
        self._started = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                owner._handle(self)

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="dsh-behavior-adapter",
            daemon=True,
        )

    @property
    def callback_url(self) -> str:
        """Return the bound loopback callback URL."""

        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def capabilities(self) -> list[dict[str, Any]]:
        """Return captured capability probes."""

        with self._lock:
            return [dict(row) for row in self._capabilities]

    @property
    def deliveries(self) -> list[dict[str, Any]]:
        """Return captured visible deliveries."""

        with self._lock:
            return [dict(row) for row in self._deliveries]

    def start(self) -> None:
        """Start the owned callback server."""

        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the owned callback server."""

        if not self._started:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("adapter callback thread did not stop")
        self._started = False

    def _handle(self, request: BaseHTTPRequestHandler) -> None:
        expected = f"Bearer {self.shared_secret}"
        if request.headers.get("Authorization") != expected:
            self._respond(request, 401, {"available": False})
            return
        try:
            length = int(request.headers.get("Content-Length", "0"))
            payload = json.loads(request.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._respond(request, 400, {"error": "invalid request"})
            return
        if not isinstance(payload, dict):
            self._respond(request, 400, {"error": "object required"})
            return
        if request.path == "/send_message/capability":
            with self._lock:
                self._capabilities.append(dict(payload))
            self._respond(request, 200, {"available": True})
            return
        if request.path != "/send_message":
            self._respond(request, 404, {"error": "unknown callback"})
            return
        delivered = {
            **payload,
            "message_id": f"dsh-behavior-{uuid4().hex}",
            "sent_at": _utc_now(),
        }
        with self._lock:
            self._deliveries.append(delivered)
        self._respond(
            request,
            200,
            {
                "platform": "debug",
                "channel_id": payload.get("channel_id", ""),
                "message_id": delivered["message_id"],
                "sent_at": delivered["sent_at"],
            },
        )

    @staticmethod
    def _respond(
        request: BaseHTTPRequestHandler,
        status_code: int,
        payload: Mapping[str, object],
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.send_response(status_code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(body)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(body)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _require_live_configuration() -> None:
    if os.environ.get("KAZUSA_RUN_LIVE_LLM") != "1":
        pytest.skip("set KAZUSA_RUN_LIVE_LLM=1 for real-model coverage")
    required = (
        "AGENTIC_RESOLVER_LLM_API_KEY",
        "AGENTIC_RESOLVER_LLM_BASE_URL",
        "AGENTIC_RESOLVER_LLM_MODEL",
        "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS",
        "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS",
        "AGENTIC_RESOLVER_LLM_THINKING_ENABLED",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail("missing real-model configuration: " + ", ".join(missing))
    if not PYTHON_EXECUTABLE.is_file():
        pytest.fail(f"project interpreter is missing: {PYTHON_EXECUTABLE}")


def _artifact_directory(case_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACT_ROOT / f"{case_id}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _case_environment(case: BehaviorCase, tmp_path: Path) -> dict[str, str]:
    brain_port = _free_port()
    sidecar_port = _free_port()
    while sidecar_port == brain_port:
        sidecar_port = _free_port()
    workspace_root = (tmp_path / "workspace").resolve()
    data_root = (tmp_path / "dsh-data").resolve()
    workspace_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    for relative_path, content in case.workspace_files.items():
        target = workspace_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    database_name = f"{DATABASE_PREFIX}{uuid4().hex}"
    environment = os.environ.copy()
    python_path = [str((PROJECT_ROOT / "src").resolve()), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update({
        "PYTHONPATH": os.pathsep.join(python_path),
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
        "KAZUSA_CONTROL_BRAIN_SHARED_SECRET": f"control-{uuid4().hex}",
        "CALENDAR_SCHEDULER_ENABLED": "false",
        "SELF_COGNITION_ENABLED": "false",
        "BACKGROUND_WORK_WORKER_ENABLED": "true",
        "REFLECTION_CYCLE_ENABLED": "false",
        "CHARACTER_SLEEP_LOCAL_PERIOD": "",
        "COGNITION_VISUAL_DIRECTIVES_ENABLED": "false",
        "LLM_TRACE_CAPTURE_MODE": "full",
        "TASK_RESOLUTION_INLINE_BUDGET_SECONDS": "120.0",
        "NODE_ENV": "test",
    })
    return environment


async def run_live_behavior_case(case: BehaviorCase, tmp_path: Path) -> None:
    """Run one real-model contract in a guarded child process."""

    _require_live_configuration()
    case_id = case.case_id
    artifact_dir = _artifact_directory(case_id)
    case_root = tmp_path / case_id
    environment = _case_environment(case, case_root)
    _write_json(artifact_dir / "case_contract.json", asdict(case))
    command = [
        str(PYTHON_EXECUTABLE),
        str(Path(__file__).resolve()),
        "--case-file",
        str(artifact_dir / "case_contract.json"),
        "--artifact-dir",
        str(artifact_dir.resolve()),
    ]
    stdout_path = artifact_dir / "case.stdout.log"
    stderr_path = artifact_dir / "case.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                | subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            ),
        )
        try:
            return_code = await asyncio.wait_for(
                process.wait(),
                timeout=CHILD_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()
            pytest.fail(
                f"{case_id} exceeded its deadline; artifacts={artifact_dir}",
            )
    result_path = artifact_dir / "case_result.json"
    if not result_path.is_file():
        pytest.fail(
            f"{case_id} produced no result (exit {return_code}); "
            f"artifacts={artifact_dir}",
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"DSH behavior artifact: {artifact_dir}")
    if return_code != 0 or result.get("technical_status") != "passed":
        pytest.fail(
            f"{case_id} technical gates failed: {result.get('failures')}; "
            f"artifacts={artifact_dir}",
        )


async def _prepare_database(artifact_dir: Path) -> None:
    from kazusa_ai_chatbot.character_profile import load_character_profile_seed
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db import db_bootstrap
    from kazusa_ai_chatbot.db.character_identity_growth import ensure_seed_identity

    seed = load_character_profile_seed(PROJECT_ROOT / "personalities" / "example.json")
    await db_bootstrap()
    await ensure_seed_identity(character_id=CHARACTER_GLOBAL_USER_ID, seed=seed)
    _write_json(artifact_dir / "database_preparation.json", {
        "database_name": os.environ["MONGODB_DB_NAME"], "status": "prepared",
    })


async def _seed_user(global_user_id: str, platform_user_id: str) -> None:
    from kazusa_ai_chatbot.db import create_user_profile, get_user_profile

    existing = await get_user_profile(global_user_id)
    if not existing:
        await create_user_profile({
            "global_user_id": global_user_id,
            "display_name": "DSH Behavior User",
            "platform_accounts": [{
                "platform": "debug", "platform_user_id": platform_user_id,
                "display_name": "DSH Behavior User",
            }],
        })


async def _wait_for_brain_bridge(brain_task: asyncio.Task[Any]) -> None:
    base_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    deadline = time.monotonic() + 120
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if brain_task.done():
                raise RuntimeError("Brain stopped before bridge readiness")
            try:
                response = await client.get(
                    f"{base_url}/runtime/dsh/bridge-health",
                )
                payload = response.json()
                if response.status_code == 200 and payload.get("status") == "ready":
                    return
            except (httpx.RequestError, ValueError):
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError("Brain bridge readiness timed out")


async def _wait_for_runtime_readiness() -> dict[str, object]:
    base_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    secret = os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"]
    deadline = time.monotonic() + 120
    latest: dict[str, object] = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(
                    f"{base_url}/runtime/dsh/health",
                    headers={"Authorization": f"Bearer {secret}"},
                )
                payload = response.json()
                latest = {
                    "http_status": response.status_code,
                    "payload": payload,
                }
                if response.status_code == 200 and payload.get("status") == "ready":
                    return latest
            except (httpx.RequestError, ValueError) as exc:
                latest = {"error_class": type(exc).__name__}
            await asyncio.sleep(0.2)
    raise RuntimeError(f"DSH runtime readiness timed out: {latest}")


async def _register_adapter(adapter: CallbackAdapter) -> dict[str, Any]:
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID

    base_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/runtime/adapters/register",
            json={
                "platform": "debug",
                "callback_url": adapter.callback_url,
                "platform_bot_id": CHARACTER_GLOBAL_USER_ID,
                "shared_secret": adapter.shared_secret,
                "timeout_seconds": 30.0,
            },
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("adapter registration returned a non-object")
    return dict(payload)


def _chat_request(case_id: str, text: str) -> dict[str, object]:
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    suffix = uuid4().hex
    return {
        "platform": "debug",
        "platform_channel_id": f"dsh-behavior-channel-{case_id}",
        "channel_type": "private",
        "platform_message_id": f"dsh-behavior-message-{suffix}",
        "platform_user_id": f"dsh-behavior-user-{case_id}",
        "platform_bot_id": CHARACTER_GLOBAL_USER_ID,
        "display_name": "DSH Behavior User",
        "channel_name": "DSH Behavior Private",
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


async def _post_chat(case_id: str, text: str) -> dict[str, Any]:
    base_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    timeout = httpx.Timeout(
        HTTP_TIMEOUT_SECONDS,
        connect=10.0,
        read=HTTP_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )
    request = _chat_request(case_id, text)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/chat",
            headers={
                "X-Kazusa-Control-Console": "debug-v1",
                "X-Kazusa-Control-Console-Auth": os.environ[
                    "KAZUSA_CONTROL_BRAIN_SHARED_SECRET"
                ],
            },
            json=request,
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("chat response returned a non-object")
    result = {"request": request, "response": dict(payload)}
    return result


async def _post_internal_interaction(scenario: Mapping[str, str]) -> dict[str, Any]:
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV2,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import (
        semantic_catalog_digest,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
        content_digest,
    )

    user_id = "dsh-behavior-internal-user"
    platform_user_id = "dsh-behavior-internal-platform-user"
    await _seed_user(user_id, platform_user_id)
    now = datetime.now(UTC)
    scope = {
        "platform": "debug",
        "platform_channel_id": "dsh-behavior-internal-channel",
        "global_user_id": user_id,
    }
    request = DshBrainInteractionRequestV2.from_mapping({
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": f"interaction-{uuid4().hex}",
        "kind": scenario["kind"],
        "resolution_thread_id": f"thread-{uuid4().hex}",
        "segment_id": f"segment-{uuid4().hex}",
        "activation_id": f"activation-{uuid4().hex}",
        "lease_epoch": 1,
        "dsh_call_id": f"call-{uuid4().hex}",
        "tool_name": "submit_resolution" if scenario["kind"] == "approval" else None,
        "operation_id": f"operation-{uuid4().hex}",
        "operation_payload_digest": "sha256:behavior-operation",
        "arguments_digest": "sha256:behavior-arguments",
        "transient_detail": scenario["detail"],
        "brain_conversation_ref": "chat:debug:dsh-behavior-internal",
        "platform": "debug",
        "platform_channel_id": scope["platform_channel_id"],
        "global_user_id": user_id,
        "scope_fingerprint": content_digest(scope),
        "audience_fingerprint": content_digest({"audience": user_id}),
        "profile_version": "kazusa-resolver-standard-v2",
        "catalog_digest": semantic_catalog_digest(),
        "model_route_digest": "sha256:configured-live-route",
        "workspace_fingerprint": content_digest({
            "workspace_root": os.environ["AGENTIC_RESOLVER_WORKSPACE_ROOT"],
        }),
        "policy_epoch": "dsh-standard-policy-v2",
        "issued_reference_digest": "sha256:behavior-references",
        "nonce": f"nonce-{uuid4().hex}",
        "issued_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (
            now + timedelta(minutes=5)
        ).isoformat().replace("+00:00", "Z"),
        "issuer": "dsh-behavior-test",
        "mac": "unsigned",
    })
    signed = sign_request(
        request,
        secret=os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"].encode("utf-8"),
    )
    base_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{base_url}/runtime/dsh/interactions",
            headers={
                "Authorization": "Bearer "
                + os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"],
            },
            json=signed.to_dict(),
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("interaction response returned a non-object")
    result = {"request": signed.to_dict(), "response": dict(payload)}
    return result


async def _wait_for_deferred_delivery(
    adapter: CallbackAdapter,
    diagnostics: GuardedDshDiagnostics,
    channel_id: str,
) -> None:
    deadline = time.monotonic() + HTTP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if adapter.deliveries and await diagnostics.delivery_completed(channel_id):
            return
        await asyncio.sleep(0.5)
    raise RuntimeError("deferred DSH delivery did not settle")


def _latest_task_results(evidence: Mapping[str, object]) -> list[dict[str, Any]]:
    mongo = evidence["mongo"]
    if not isinstance(mongo, Mapping):
        return []
    results = []
    for binding in mongo.get("dsh_task_bindings", []):
        if not isinstance(binding, Mapping):
            continue
        result = binding.get("latest_task_resolution_result")
        if isinstance(result, Mapping):
            results.append(dict(result))
    return results


def _evaluate_case(case: BehaviorCase, evidence: Mapping[str, Any]) -> tuple[dict[str, bool], list[str]]:
    """Check public response contracts; leave semantic judgments to review."""

    mongo = evidence["mongo"]
    response = evidence["response"]
    results = _latest_task_results(evidence)
    checks = {
        "cognition_trace_retained": bool(mongo["llm_trace_steps"]),
        "scenario_contract_retained": bool(evidence["contract"]),
        "no_operational_failure": all(
            turn["response"]["operational_error"] is None
            for turn in response["turns"]
        ),
    }
    channel = evidence["channel_id"]
    if case.case_id == "foreground":
        turns = response["turns"]
        visible = "\n".join(turns[-1]["response"]["messages"])
        checks.update({
            "ambiguity_response_visible": bool(turns[0]["response"]["messages"]),
            "clarified_response_visible": bool(visible),
            "source_owner_literal_preserved": "Mira" in visible,
            "conversation_correlated_dsh_entry": any(
                row["source_scope"]["source_message_id"] in {
                    turn["request"]["platform_message_id"] for turn in turns
                }
                for row in mongo["dsh_task_bindings"]
            ),
            "grounded_result_available": any(result["evidence"] for result in results),
        })
    elif case.case_id == "deferred":
        deliveries = evidence["adapter"]["delivery_payloads"]
        jobs = mongo["background_work_jobs"]
        visible = "\n".join(row["text"] for row in deliveries)
        checks.update({
            "correlated_dsh_entry": bool(mongo["dsh_task_bindings"]),
            "correlated_background_owner": bool(jobs),
            "background_job_delivered": any(row.get("delivery_state") == "delivered" for row in jobs),
            "one_eligible_delivery": len(deliveries) == 1,
            "correct_audience": all(row["channel_id"] == channel for row in deliveries),
            "source_owner_literal_preserved": "Rowan" in visible,
            "grounded_result_available": any(result["evidence"] for result in results),
        })
    elif case.case_id == "research":
        visible = "\n".join([
            *response["turns"][-1]["response"]["messages"],
            *(row["text"] for row in evidence["adapter"]["delivery_payloads"]),
        ])
        checks.update({
            "correlated_dsh_entry": bool(mongo["dsh_task_bindings"]),
            "grounded_result_available": any(result["evidence"] for result in results),
            "documented_class_visible": "DictReader" in visible,
            "documentation_cited": "https://docs.python.org/3/library/csv.html" in visible,
        })
    elif case.case_id == "workspace":
        visible = "\n".join([
            *response["turns"][-1]["response"]["messages"],
            *(row["text"] for row in evidence["adapter"]["delivery_payloads"]),
        ])
        artifacts = evidence["workspace_artifacts"]
        report = artifacts["report.json"]
        checks.update({
            "correlated_dsh_entry": bool(mongo["dsh_task_bindings"]),
            "program_created": bool(artifacts["summarize.py"]),
            "computed_report_correct": report == {"row_count": 2, "total": 27.5},
            "task_resolved": any(result["status"] == "resolved" for result in results),
            "result_visible": bool(visible),
        })
    else:
        interactions = response["interactions"]
        for ordinal, interaction in enumerate(interactions):
            request = interaction["request"]
            decision = interaction["response"]
            allowed = {"answer", "reject"} if request["kind"] == "question" else {"allow_once", "reject"}
            checks[f"interaction_{ordinal}_kind_contract"] = decision["decision"] in allowed
            checks[f"interaction_{ordinal}_reason"] = bool(decision["reason"])
            checks[f"interaction_{ordinal}_durable_match"] = any(
                row["interaction_id"] == request["interaction_id"]
                and row["decision_state"] == decision["decision"]
                for row in mongo["dsh_interactions"]
            )
        checks["answerable_question_answered"] = interactions[0]["response"]["decision"] == "answer"
        checks["unsupported_success_rejected"] = interactions[1]["response"]["decision"] == "reject"
        visible = ""
    checks["internal_identifiers_absent_from_visible_text"] = all(
        value not in visible
        for row in mongo["dsh_task_bindings"]
        for key in (
            "resolution_thread_id", "segment_id", "task_session_id",
            "current_accepted_task_id", "current_background_work_job_id",
        )
        if isinstance(value := row.get(key), str) and value
    )
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


async def _stop_brain(server: Any, task: asyncio.Task[Any]) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)
    except TimeoutError as exc:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise RuntimeError("Brain graceful shutdown exceeded 45 seconds") from exc


async def _execute_case(case: BehaviorCase, artifact_dir: Path) -> int:
    import uvicorn

    provider = None
    if case.case_id != "internal":
        provider = LiveProviderRecorder(os.environ["AGENTIC_RESOLVER_LLM_BASE_URL"], artifact_dir)
        os.environ["AGENTIC_RESOLVER_LLM_BASE_URL"] = provider.base_url

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.llm_interface import LLInterface
    from kazusa_ai_chatbot.llm_tracing import current_trace_id

    case_id = case.case_id
    started_at = time.perf_counter()
    recorder = BehaviorResources(artifact_dir=artifact_dir)
    diagnostics = GuardedDshDiagnostics(
        os.environ["MONGODB_URI"], os.environ["MONGODB_DB_NAME"],
        os.environ["KAZUSA_EPHEMERAL_TEST_DATABASE_NAME"],
    )
    adapter = (
        CallbackAdapter(shared_secret=f"adapter-{uuid4().hex}")
        if case_id != "internal" else None
    )
    sidecar: SidecarProcess | None = None
    server: Any | None = None
    brain_task: asyncio.Task[Any] | None = None
    evidence: dict[str, object] = {}
    checks: dict[str, bool] = {}
    failures: list[str] = []
    cleanup = {
        "brain_stopped": False,
        "sidecar_stopped": False,
        "adapter_stopped": False,
        "database_dropped": False,
        "errors": [],
    }
    error: dict[str, str] | None = None
    model_calls: list[dict[str, Any]] = []
    original_ainvoke = LLInterface.ainvoke
    response: dict[str, Any] = {"turns": [], "interactions": []}
    channel_id = (
        "dsh-behavior-internal-channel" if case_id == "internal"
        else f"dsh-behavior-channel-{case_id}"
    )

    async def observed_ainvoke(interface: Any, messages: Any, *, config: Any) -> Any:
        """Record real invocation cost while forwarding its exact contract."""
        call_started = time.perf_counter()
        response = await original_ainvoke(interface, messages, config=config)
        model_calls.append({
            "trace_id": current_trace_id(), "route": config.route_name,
            "model": config.model, "base_url": config.base_url,
            "max_completion_tokens": config.max_completion_tokens,
            "duration_ms": round((time.perf_counter() - call_started) * 1000),
            "usage": dict(response.usage),
            "messages": [{"type": message.type, "content": message.content} for message in messages],
            "raw_response": response.content,
        })
        _write_json(artifact_dir / "real_model_calls.json", model_calls)
        return response

    LLInterface.ainvoke = observed_ainvoke
    try:
        await _prepare_database(artifact_dir)
        brain_port = int(
            os.environ["KAZUSA_DSH_BRAIN_URL"].rsplit(":", maxsplit=1)[1],
        )
        server = uvicorn.Server(uvicorn.Config(
            service.app,
            host="127.0.0.1",
            port=brain_port,
            log_level="info",
            access_log=False,
            lifespan="on",
        ))
        server.install_signal_handlers = lambda: None
        brain_task = asyncio.create_task(
            server.serve(),
            name=f"dsh-behavior-brain-{case_id}",
        )
        await _wait_for_brain_bridge(brain_task)
        if adapter is not None:
            sidecar = await asyncio.to_thread(
                start_configured_sidecar, recorder, name="dsh-sidecar",
                environment=os.environ.copy(),
            )
            readiness = await _wait_for_runtime_readiness()
            _write_json(artifact_dir / "readiness.json", readiness)
            adapter.start()
            registration = await _register_adapter(adapter)
            _write_json(artifact_dir / "adapter_registration.json", registration)
        for text in case.user_inputs:
            turn_response = await _post_chat(case_id, text)
            response["turns"].append({"input": text, **turn_response})
            _write_json(artifact_dir / f"turn_{len(response['turns'])}.json", response["turns"][-1])
        if case_id in {"deferred", "research", "workspace"}:
            assert adapter is not None
            if await diagnostics.has_background_work(channel_id):
                await _wait_for_deferred_delivery(adapter, diagnostics, channel_id)
        for scenario in case.interaction_inputs:
            interaction = await _post_internal_interaction(scenario)
            response["interactions"].append(interaction)
            _write_json(artifact_dir / f"interaction_{len(response['interactions'])}.json", interaction)
        interaction_ids = [row["request"]["interaction_id"] for row in response["interactions"]]
        mongo = await diagnostics.snapshot(channel_id, interaction_ids)
        evidence = {
            "contract": asdict(case), "response": response, "channel_id": channel_id,
            "adapter": {
                "capability_payloads": adapter.capabilities if adapter else [],
                "delivery_payloads": adapter.deliveries if adapter else [],
            },
            "mongo": mongo,
            "cost": {
                "planned_cognition_episodes": len(case.user_inputs) + len(case.interaction_inputs) + (case_id == "deferred"),
                "brain_llm_calls": len(model_calls),
                "brain_usage_records": [row["usage"] for row in model_calls],
                "dsh_llm_calls": len(provider.calls) if provider is not None else 0,
                "dsh_provider_artifact": "dsh_provider_calls.json" if provider is not None else None,
                "model_calls_artifact": "real_model_calls.json",
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000),
            },
        }
        if case_id == "workspace":
            workspace_root = Path(os.environ["AGENTIC_RESOLVER_WORKSPACE_ROOT"])
            program_path = workspace_root / "summarize.py"
            report_path = workspace_root / "report.json"
            evidence["workspace_artifacts"] = {
                "summarize.py": program_path.read_text(encoding="utf-8") if program_path.is_file() else None,
                "report.json": json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None,
            }
        checks, failures = _evaluate_case(case, evidence)
        _write_json(artifact_dir / "evidence.json", evidence)
        _write_json(
            artifact_dir / "behavior_review.json",
            {
                "schema_version": "dsh_behavior_review.v1",
                "case_id": case_id,
                "input_and_output": evidence,
                "technical_checks": checks,
                "review_questions": case.behavior_rubric,
                "character_review_decision": None,
            },
        )
    except Exception as exc:  # noqa: BLE001 - preserve live evidence.
        error = {
            "error_class": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failures.append(f"case_exception:{type(exc).__name__}")
        _write_json(artifact_dir / "case_exception.json", error)
        try:
            interaction_ids = [row["request"]["interaction_id"] for row in response["interactions"]]
            _write_json(artifact_dir / "failure_evidence.json", {
                "response": response,
                "mongo": await diagnostics.snapshot(channel_id, interaction_ids),
                "adapter_deliveries": adapter.deliveries if adapter else [],
            })
        except Exception as snapshot_error:  # noqa: BLE001 - retain the original failure.
            _write_json(artifact_dir / "failure_snapshot_error.json", {
                "error_class": type(snapshot_error).__name__, "error": str(snapshot_error),
            })
    finally:
        if server is not None and brain_task is not None:
            try:
                await _stop_brain(server, brain_task)
                cleanup["brain_stopped"] = True
            except Exception as exc:  # noqa: BLE001 - continue cleanup.
                cleanup["errors"].append(f"brain:{type(exc).__name__}:{exc}")
        if sidecar is not None:
            try:
                await asyncio.to_thread(sidecar.stop)
                cleanup["sidecar_stopped"] = True
            except Exception as exc:  # noqa: BLE001 - continue cleanup.
                cleanup["errors"].append(f"sidecar:{type(exc).__name__}:{exc}")
        if provider is not None:
            try:
                await asyncio.to_thread(provider.close)
            except (OSError, RuntimeError) as exc:
                cleanup["errors"].append(f"provider:{type(exc).__name__}:{exc}")
        if adapter is not None:
            try:
                adapter.stop()
                cleanup["adapter_stopped"] = True
            except Exception as exc:  # noqa: BLE001 - continue cleanup.
                cleanup["errors"].append(f"adapter:{type(exc).__name__}:{exc}")
        try:
            from kazusa_ai_chatbot.db import close_db
            await close_db()
            await diagnostics.drop()
            cleanup["database_dropped"] = True
        except Exception as exc:  # noqa: BLE001 - report cleanup failure.
            cleanup["errors"].append(f"database:{type(exc).__name__}:{exc}")
        diagnostics.close()
        LLInterface.ainvoke = original_ainvoke
        _write_json(artifact_dir / "cleanup.json", cleanup)
    if cleanup["errors"]:
        failures.append("cleanup_incomplete")
    status = "passed" if not failures else "failed"
    _write_json(
        artifact_dir / "case_result.json",
        {
            "schema_version": "dsh_behavior_case_result.v1",
            "case_id": case_id,
            "technical_status": status,
            "checks": checks,
            "failures": failures,
            "error": error,
            "cleanup": cleanup,
            "processes": recorder.processes,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        },
    )
    return 0 if status == "passed" else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    case = BehaviorCase(**json.loads(args.case_file.read_text(encoding="utf-8")))
    return asyncio.run(_execute_case(case, args.artifact_dir))


if __name__ == "__main__":
    raise SystemExit(_main())
