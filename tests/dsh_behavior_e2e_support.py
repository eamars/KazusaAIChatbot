"""Isolated real-model support for the three DSH behavior contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import httpx
import pytest

from experiments.dsh_runtime_probe import (
    ProbeRecorder,
    SidecarProcess,
    start_configured_sidecar,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
ARTIFACT_ROOT = PROJECT_ROOT / "test_artifacts" / "dsh_behavior_e2e"
DATABASE_PREFIX = "_test_kazusa_dsh_behavior_"
CHILD_TIMEOUT_SECONDS = 30 * 60
HTTP_TIMEOUT_SECONDS = 10 * 60


@dataclass(frozen=True, slots=True)
class BehaviorCase:
    """One user-observable behavior contract and its local evidence."""

    case_id: str
    workspace_files: Mapping[str, str]


CASES = {
    "foreground": BehaviorCase(
        case_id="foreground",
        workspace_files={
            "rollout/status_note.txt": (
                "The rollout owner is Mira. The checksum review must finish "
                "before rollout begins."
            ),
        },
    ),
    "deferred": BehaviorCase(
        case_id="deferred",
        workspace_files={
            "handover/incident_note.txt": (
                "The cache alert is stable. Rowan owns the follow-up. The "
                "next review happens after the morning metrics arrive."
            ),
        },
    ),
    "internal": BehaviorCase(case_id="internal", workspace_files={}),
}


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


async def run_live_behavior_case(case_id: str, tmp_path: Path) -> None:
    """Run one real-model contract in a guarded child process."""

    _require_live_configuration()
    case = CASES[case_id]
    artifact_dir = _artifact_directory(case_id)
    case_root = tmp_path / case_id
    environment = _case_environment(case, case_root)
    command = [
        str(PYTHON_EXECUTABLE),
        str(Path(__file__).resolve()),
        "--execute-case",
        case_id,
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
    from kazusa_ai_chatbot.db import _client as client_module
    from kazusa_ai_chatbot.db import db_bootstrap
    from kazusa_ai_chatbot.db.character_identity_growth import (
        ensure_seed_identity,
    )

    database_name = os.environ["MONGODB_DB_NAME"]
    if not database_name.startswith(DATABASE_PREFIX):
        raise RuntimeError("database name is outside the behavior-test prefix")
    client_module.MONGODB_DB_NAME = database_name
    client_module._assert_guarded_database_name()
    seed = load_character_profile_seed(
        PROJECT_ROOT / "personalities" / "example.json",
    )
    await client_module.close_db()
    await db_bootstrap()
    await ensure_seed_identity(
        character_id=CHARACTER_GLOBAL_USER_ID,
        seed=seed,
    )
    database = await client_module.get_db()
    await database.command("ping")
    _write_json(
        artifact_dir / "database_preparation.json",
        {"database_name": database_name, "status": "prepared"},
    )


async def _drop_database() -> None:
    from motor.motor_asyncio import AsyncIOMotorClient

    from kazusa_ai_chatbot.db import _client as client_module

    database_name = os.environ["MONGODB_DB_NAME"]
    if not database_name.startswith(DATABASE_PREFIX):
        raise RuntimeError("database cleanup target is outside the test prefix")
    if os.environ["KAZUSA_EPHEMERAL_TEST_DATABASE_NAME"] != database_name:
        raise RuntimeError("database cleanup target does not match its guard")
    await client_module.close_db()
    client = AsyncIOMotorClient(client_module.MONGODB_URI)
    try:
        await client.drop_database(database_name)
    finally:
        client.close()


async def _read_collection(name: str) -> list[dict[str, Any]]:
    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    rows = await database[name].find({}, {"_id": 0}).to_list(length=None)
    return [dict(row) for row in rows if isinstance(row, Mapping)]


async def _seed_user(global_user_id: str, platform_user_id: str) -> None:
    from kazusa_ai_chatbot.db import create_user_profile
    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    existing = await database.user_profiles.find_one(
        {"global_user_id": global_user_id},
        {"_id": 1},
    )
    if existing is None:
        await create_user_profile({
            "global_user_id": global_user_id,
            "display_name": "DSH Behavior User",
            "platform_accounts": [{
                "platform": "debug",
                "platform_user_id": platform_user_id,
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/chat",
            headers={
                "X-Kazusa-Control-Console": "debug-v1",
                "X-Kazusa-Control-Console-Auth": os.environ[
                    "KAZUSA_CONTROL_BRAIN_SHARED_SECRET"
                ],
            },
            json=_chat_request(case_id, text),
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise TypeError("chat response returned a non-object")
    return dict(payload)


async def _post_internal_interaction() -> dict[str, Any]:
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
        "kind": "question",
        "resolution_thread_id": f"thread-{uuid4().hex}",
        "segment_id": f"segment-{uuid4().hex}",
        "activation_id": f"activation-{uuid4().hex}",
        "lease_epoch": 1,
        "dsh_call_id": f"call-{uuid4().hex}",
        "tool_name": None,
        "operation_id": f"operation-{uuid4().hex}",
        "operation_payload_digest": "sha256:behavior-operation",
        "arguments_digest": "sha256:behavior-arguments",
        "transient_detail": (
            "The user asked whether a local rollout note is enough to claim "
            "the rollout is safe. Answer only if the available context "
            "supports it; otherwise reject the unsupported conclusion."
        ),
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
    return dict(payload)


async def _wait_for_deferred_delivery(
    adapter: CallbackAdapter,
) -> None:
    deadline = time.monotonic() + HTTP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        jobs = await _read_collection("background_work_jobs")
        if (
            len(adapter.deliveries) == 1
            and any(row.get("delivery_state") == "delivered" for row in jobs)
        ):
            return
        await asyncio.sleep(0.5)
    raise RuntimeError("deferred DSH delivery did not settle")


async def _capture_evidence(
    *,
    response: Mapping[str, object],
    adapter: CallbackAdapter,
) -> dict[str, object]:
    collection_names = (
        "dsh_task_bindings",
        "accepted_tasks",
        "background_work_jobs",
        "dsh_interactions",
        "llm_trace_runs",
        "llm_trace_steps",
        "event_log_events",
        "conversation_history",
    )
    mongo = {
        name: await _read_collection(name)
        for name in collection_names
    }
    return {
        "response": dict(response),
        "adapter": {
            "capability_payloads": adapter.capabilities,
            "delivery_payloads": adapter.deliveries,
        },
        "mongo": mongo,
    }


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


def _evaluate_case(
    case_id: str,
    evidence: Mapping[str, object],
) -> tuple[dict[str, bool], list[str]]:
    mongo = evidence["mongo"]
    if not isinstance(mongo, Mapping):
        raise TypeError("captured Mongo evidence is invalid")
    response = evidence["response"]
    if not isinstance(response, Mapping):
        raise TypeError("captured response is invalid")
    bindings = [
        row for row in mongo["dsh_task_bindings"]
        if isinstance(row, Mapping)
    ]
    results = _latest_task_results(evidence)
    runtime_failures = [
        row for row in mongo["event_log_events"]
        if isinstance(row, Mapping)
        and (
            row.get("event_family") == "runtime_error"
            or (
                row.get("event_family") == "pipeline_turn"
                and row.get("status") == "failed"
            )
        )
    ]
    checks: dict[str, bool] = {"no_runtime_failure": not runtime_failures}
    if case_id == "foreground":
        rendered = json.dumps(results, ensure_ascii=False).lower()
        checks.update({
            "one_task_binding": len(bindings) == 1,
            "resolved_task_result": (
                len(results) == 1 and results[0].get("status") == "resolved"
            ),
            "grounded_evidence": (
                len(results) == 1
                and bool(results[0].get("evidence"))
                and "mira" in rendered
            ),
            "visible_character_surface": bool(response.get("messages")),
            "trace_retained": bool(response.get("trace_id")),
        })
    elif case_id == "deferred":
        jobs = [
            row for row in mongo["background_work_jobs"]
            if isinstance(row, Mapping)
        ]
        deliveries = evidence["adapter"]
        delivery_rows = (
            deliveries.get("delivery_payloads", [])
            if isinstance(deliveries, Mapping)
            else []
        )
        checks.update({
            "one_task_binding": len(bindings) == 1,
            "one_background_job": len(jobs) == 1,
            "background_job_delivered": (
                len(jobs) == 1 and jobs[0].get("delivery_state") == "delivered"
            ),
            "one_visible_delivery": len(delivery_rows) == 1,
            "grounded_terminal_result": (
                len(results) == 1
                and results[0].get("status") == "resolved"
                and bool(results[0].get("evidence"))
            ),
        })
    elif case_id == "internal":
        interactions = [
            row for row in mongo["dsh_interactions"]
            if isinstance(row, Mapping)
        ]
        decision = response.get("decision")
        checks.update({
            "one_durable_interaction": len(interactions) == 1,
            "semantic_decision_returned": decision in {
                "answer", "allow_once", "reject",
            },
            "decision_reason_retained": bool(response.get("reason")),
            "decision_matches_persistence": (
                len(interactions) == 1
                and interactions[0].get("decision_state") == decision
            ),
            "cognition_trace_retained": bool(mongo["llm_trace_runs"]),
        })
    else:
        raise ValueError(f"unsupported behavior case: {case_id}")
    failures = [name for name, passed in checks.items() if not passed]
    return checks, failures


async def _stop_brain(server: Any, task: asyncio.Task[Any]) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=45)
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _execute_case(case_id: str, artifact_dir: Path) -> int:
    import uvicorn

    from kazusa_ai_chatbot import service

    started_at = time.perf_counter()
    recorder = ProbeRecorder(
        probe_name=f"live-{case_id}",
        artifact_dir=artifact_dir,
    )
    adapter = CallbackAdapter(shared_secret=f"adapter-{uuid4().hex}")
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
        sidecar = await asyncio.to_thread(
            start_configured_sidecar,
            recorder,
            name="dsh-sidecar",
            environment=os.environ.copy(),
        )
        readiness = await _wait_for_runtime_readiness()
        _write_json(artifact_dir / "readiness.json", readiness)
        adapter.start()
        registration = await _register_adapter(adapter)
        _write_json(artifact_dir / "adapter_registration.json", registration)
        if case_id == "foreground":
            response = await _post_chat(
                case_id,
                "Please read rollout/status_note.txt and tell me who owns "
                "the rollout and what must happen first.",
            )
        elif case_id == "deferred":
            response = await _post_chat(
                case_id,
                "Handle this in the background: read "
                "handover/incident_note.txt and send me a brief summary "
                "when the work is finished.",
            )
            await _wait_for_deferred_delivery(adapter)
        else:
            response = await _post_internal_interaction()
        evidence = await _capture_evidence(
            response=response,
            adapter=adapter,
        )
        checks, failures = _evaluate_case(case_id, evidence)
        _write_json(artifact_dir / "evidence.json", evidence)
        _write_json(
            artifact_dir / "behavior_review.json",
            {
                "schema_version": "dsh_behavior_review.v1",
                "case_id": case_id,
                "input_and_output": evidence,
                "technical_checks": checks,
                "review_questions": [
                    "Was the DSH judgment grounded in retained evidence?",
                    "Did the visible response remain character-owned?",
                    "Did the result avoid invented success or permission?",
                    "Was recurrence and delivery coherent for the user?",
                ],
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
        try:
            adapter.stop()
            cleanup["adapter_stopped"] = True
        except Exception as exc:  # noqa: BLE001 - continue cleanup.
            cleanup["errors"].append(f"adapter:{type(exc).__name__}:{exc}")
        try:
            await _drop_database()
            cleanup["database_dropped"] = True
        except Exception as exc:  # noqa: BLE001 - report cleanup failure.
            cleanup["errors"].append(f"database:{type(exc).__name__}:{exc}")
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
    parser.add_argument("--execute-case", choices=tuple(CASES), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    return asyncio.run(_execute_case(args.execute_case, args.artifact_dir))


if __name__ == "__main__":
    raise SystemExit(_main())
