"""Probe real DSH process, RPC, SQLite, Brain-task, and Mongo boundaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ENTRY = (
    PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
)
RPC_TOKEN = "dsh-probe-rpc-token"
SEMANTIC_SECRET = "dsh-probe-semantic-secret"
WORKSPACE_ROOT = PROJECT_ROOT.resolve()
PROBE_NAMES = (
    "sidecar-lifecycle",
    "brain-task-lifecycle",
    "transport-loss",
)
PROCESS_EXIT_TIMEOUT_SECONDS = 15.0


class ProbeFailure(RuntimeError):
    """Raised when an observed runtime result violates a probe contract."""


class ProbeBlocked(RuntimeError):
    """Raised when an external prerequisite is unavailable."""


def _utc_now() -> str:
    """Return one canonical UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    """Serialize stable JSON for digests and artifacts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _free_port() -> int:
    """Reserve and release one loopback port for a child process."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _tested_revision() -> dict[str, object]:
    """Record the commit and a content-aware dirty-worktree digest."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    digest = sha256(status + diff)
    entries = status.decode("utf-8", errors="surrogateescape").split("\0")
    for entry in entries:
        if not entry.startswith("?? "):
            continue
        path = PROJECT_ROOT / entry[3:]
        if path.is_file():
            digest.update(entry[3:].encode("utf-8"))
            digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_state_digest": f"sha256:{digest.hexdigest()}",
    }


@dataclass
class ProbeRecorder:
    """Accumulate machine-readable evidence for one probe run."""

    probe_name: str
    artifact_dir: Path
    started_at: str = field(default_factory=_utc_now)
    observations: list[dict[str, object]] = field(default_factory=list)
    processes: list[dict[str, object]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    cleanup: list[dict[str, object]] = field(default_factory=list)

    def observe(self, kind: str, **evidence: object) -> None:
        """Append one bounded observation row."""

        self.observations.append({"kind": kind, **evidence})

    def result(self, status: str, error: str = "") -> dict[str, object]:
        """Build the stable result artifact."""

        result: dict[str, object] = {
            "schema_version": "dsh_runtime_probe_result.v1",
            "probe_name": self.probe_name,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "tested_revision": _tested_revision(),
            "status": status,
            "observations": list(self.observations),
            "processes": list(self.processes),
            "artifacts": list(self.artifacts),
            "cleanup": list(self.cleanup),
        }
        if error:
            result["error"] = error
        return result


class _FakeOpenAi:
    """Minimal deterministic OpenAI-compatible SSE endpoint."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                request_body = self.rfile.read(length) if length else b""
                try:
                    parsed_request = json.loads(request_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parsed_request = None
                if isinstance(parsed_request, dict):
                    owner.requests.append(parsed_request)
                step = owner.script[min(owner.calls, len(owner.script) - 1)]
                owner.calls += 1
                if step.get("wait") is True:
                    time.sleep(2.0)
                rows = step.get("calls")
                if not isinstance(rows, list):
                    rows = [step] if isinstance(step.get("name"), str) else []
                calls = [
                    row
                    for row in rows
                    if isinstance(row, dict)
                    and isinstance(row.get("name"), str)
                ]
                if (
                    not calls
                    and step.get("wait") is not True
                    and step.get("text") is None
                ):
                    calls = [{
                        "name": "submit_resolution",
                        "arguments": {
                            "status": "resolved",
                            "summary": "done",
                            "findings": [{
                                "claim": "The bounded probe objective is resolved.",
                            }],
                            "completed_subgoals": [],
                            "remaining_needs": [],
                            "clarification_request": None,
                            "approval_request": None,
                            "artifact_refs": ["probe-artifact:bounded-result"],
                            "warnings": [],
                        },
                    }]
                if calls:
                    tool_calls = []
                    for index, call in enumerate(calls):
                        tool_calls.append({
                            "index": index,
                            "id": f"call-{owner.calls}-{index}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(
                                    call.get("arguments", {}),
                                    separators=(",", ":"),
                                ),
                            },
                        })
                    delta: dict[str, object] = {
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    }
                    finish = "tool_calls"
                else:
                    delta = {
                        "role": "assistant",
                        "content": str(step.get("text", "waiting")),
                    }
                    finish = "stop"
                body = "".join([
                    "data: "
                    + json.dumps({
                        "id": f"response-{owner.calls}",
                        "choices": [{
                            "delta": delta,
                            "finish_reason": None,
                        }],
                    })
                    + "\n\n",
                    "data: "
                    + json.dumps({
                        "id": f"response-{owner.calls}",
                        "choices": [{
                            "delta": {},
                            "finish_reason": finish,
                        }],
                    })
                    + "\n\n",
                    "data: [DONE]\n\n",
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    owner.aborted_responses += 1

            def log_message(self, *_args: object) -> None:
                return

        self.script = script or [{}]
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.aborted_responses = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = __import__("threading").Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        """Return the loopback OpenAI-compatible base URL."""

        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        """Stop the owned deterministic provider thread."""

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class _FakeBrainHealth:
    """Loopback Brain readiness owner used by the real sidecar process."""

    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/runtime/dsh/bridge-health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({
                    "schema_version": "dsh_brain_bridge_health.v1",
                    "status": "ready",
                    "configured": True,
                    "durable_store": True,
                    "cognition_judge": True,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = __import__("threading").Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        """Return the loopback Brain health base URL."""

        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        """Stop the owned Brain health thread."""

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


@dataclass
class SidecarProcess:
    """Owned real sidecar plus deterministic loopback dependencies."""

    process: subprocess.Popen[str]
    url: str
    provider: _FakeOpenAi | None
    brain: _FakeBrainHealth | None
    data_root: Path
    recorder: ProbeRecorder
    name: str
    process_row: dict[str, object]
    stopped: bool = False

    @property
    def route_digest(self) -> str:
        """Return the configured model-route digest."""

        if self.provider is None:
            raise ProbeFailure(
                "configured sidecar does not own its model provider",
            )
        return route_digest(self.provider.base_url)

    def runtime_environment(self) -> dict[str, str]:
        """Return the parent-process environment for public runtime creation."""

        if self.provider is None or self.brain is None:
            raise ProbeFailure(
                "configured sidecar does not own loopback dependencies",
            )
        return {
            "KAZUSA_DSH_SIDECAR_URL": self.url,
            "KAZUSA_DSH_RPC_TOKEN": RPC_TOKEN,
            "KAZUSA_DSH_DATA_ROOT": str(self.data_root.resolve()),
            "AGENTIC_RESOLVER_WORKSPACE_ROOT": str(WORKSPACE_ROOT),
            "AGENTIC_RESOLVER_LLM_BASE_URL": self.provider.base_url,
            "AGENTIC_RESOLVER_LLM_API_KEY": "resolver-probe-key",
            "AGENTIC_RESOLVER_LLM_MODEL": "qwen27b-5090",
            "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS": "50176",
            "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS": "8192",
            "AGENTIC_RESOLVER_LLM_THINKING_ENABLED": "true",
            "KAZUSA_DSH_BRAIN_URL": self.brain.base_url,
            "KAZUSA_DSH_BRAIN_SHARED_SECRET": "brain-probe-secret",
            "KAZUSA_DSH_TOOL_GATEWAY_SECRET": SEMANTIC_SECRET,
            "KAZUSA_DSH_PYTHON_EXECUTABLE": str(Path(sys.executable).resolve()),
            "NODE_ENV": "test",
        }

    def wait_for_exit(
        self,
        *,
        timeout_seconds: float = PROCESS_EXIT_TIMEOUT_SECONDS,
    ) -> int:
        """Wait for an expected owned-process exit within the probe bound."""

        try:
            return self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ProbeFailure(
                f"{self.name} did not exit within {timeout_seconds:g} seconds",
            ) from exc

    def stop(self) -> None:
        """Stop owned processes and retain their logs."""

        if self.stopped:
            return
        self.stopped = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        stdout, stderr = self.process.communicate()
        self.process_row["exit_code"] = self.process.returncode
        stdout_path = self.recorder.artifact_dir / f"{self.name}.stdout.log"
        stderr_path = self.recorder.artifact_dir / f"{self.name}.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        self.recorder.artifacts.extend([
            str(stdout_path.resolve()),
            str(stderr_path.resolve()),
        ])
        if self.provider is not None:
            self.provider.close()
        if self.brain is not None:
            self.brain.close()
        self.recorder.cleanup.append({
            "owner": self.name,
            "status": "stopped",
            "pid": self.process.pid,
        })


def route_digest(base_url: str) -> str:
    """Return the canonical digest for the probe's deterministic route."""
    descriptor = {
        "route_name": "kazusa-agentic-resolver",
        "base_url": base_url,
        "model": "qwen27b-5090",
        "context_window_tokens": 50176,
        "max_completion_tokens": 8192,
        "thinking_enabled": True,
        "supports_developer_role": False,
        "max_tokens_field": "max_completion_tokens",
        "thinking_format": "qwen-chat-template",
        "chat_template_kwargs_enable_thinking": True,
        "reasoning_effort": "high",
        "output_mode": "text",
        "compatibility_epoch": "qwen-openai-completions-v1",
        "credential_reference": "AGENTIC_RESOLVER_LLM_API_KEY",
    }
    digest = sha256(_canonical_json(descriptor).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def rpc_call(
    url: str,
    method: str,
    params: dict[str, Any],
    *,
    token: str = RPC_TOKEN,
) -> dict[str, Any]:
    body = json.dumps({
        "jsonrpc": "2.0",
        "id": f"rpc-{time.time_ns()}",
        "method": method,
        "params": {
            "protocol_version": "kazusa.dsh-resolution-rpc.v2",
            **params,
        },
    }).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=5) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ProbeFailure(f"RPC {method} returned a non-object")
    return value


def start_sidecar(
    data_root: Path,
    recorder: ProbeRecorder,
    *,
    name: str,
    script: list[dict[str, Any]] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> SidecarProcess:
    """Start one real sidecar and require authenticated readiness."""

    if not SIDECAR_ENTRY.is_file():
        raise ProbeBlocked(f"built sidecar entry is unavailable: {SIDECAR_ENTRY}")
    port = _free_port()
    url = f"http://127.0.0.1:{port}/rpc"
    provider = _FakeOpenAi(script or [])
    brain = _FakeBrainHealth()
    data_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "KAZUSA_DSH_SIDECAR_URL": url,
        "KAZUSA_DSH_RPC_TOKEN": RPC_TOKEN,
        "KAZUSA_DSH_DATA_ROOT": str(data_root.resolve()),
        "AGENTIC_RESOLVER_WORKSPACE_ROOT": str(WORKSPACE_ROOT),
        "AGENTIC_RESOLVER_LLM_BASE_URL": provider.base_url,
        "AGENTIC_RESOLVER_LLM_API_KEY": "resolver-probe-key",
        "AGENTIC_RESOLVER_LLM_MODEL": "qwen27b-5090",
        "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS": "50176",
        "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS": "8192",
        "AGENTIC_RESOLVER_LLM_THINKING_ENABLED": "true",
        "KAZUSA_DSH_BRAIN_URL": brain.base_url,
        "KAZUSA_DSH_BRAIN_SHARED_SECRET": "brain-probe-secret",
        "KAZUSA_DSH_TOOL_GATEWAY_SECRET": SEMANTIC_SECRET,
        "KAZUSA_DSH_PYTHON_EXECUTABLE": str(Path(sys.executable).resolve()),
        "NODE_ENV": "test",
    })
    environment.pop("KAZUSA_DSH_CAPABILITY_TOKEN", None)
    environment.update(dict(extra_env or {}))
    process = subprocess.Popen(
        ["node", str(SIDECAR_ENTRY)],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    process_row: dict[str, object] = {
        "name": name,
        "pid": process.pid,
        "exit_code": None,
    }
    recorder.processes.append(process_row)
    harness = SidecarProcess(
        process=process,
        url=url,
        provider=provider,
        brain=brain,
        data_root=data_root,
        recorder=recorder,
        name=name,
        process_row=process_row,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            harness.stop()
            raise ProbeFailure(f"sidecar {name} exited before readiness")
        try:
            health = rpc_call(url, "system.health", {}).get("result")
            if isinstance(health, dict) and health.get("status") == "ready":
                return harness
        except OSError:
            pass
        time.sleep(0.05)
    harness.stop()
    raise ProbeFailure(f"sidecar {name} did not become ready")


def start_configured_sidecar(
    recorder: ProbeRecorder,
    *,
    name: str,
    environment: Mapping[str, str],
) -> SidecarProcess:
    """Start a sidecar against externally configured Brain and model owners."""

    if not SIDECAR_ENTRY.is_file():
        raise ProbeBlocked(f"built sidecar entry is unavailable: {SIDECAR_ENTRY}")
    url = environment.get("KAZUSA_DSH_SIDECAR_URL", "").strip()
    token = environment.get("KAZUSA_DSH_RPC_TOKEN", "").strip()
    data_root_value = environment.get("KAZUSA_DSH_DATA_ROOT", "").strip()
    if not url or not token or not data_root_value:
        raise ProbeFailure(
            "configured sidecar requires URL, RPC token, and data root",
        )
    data_root = Path(data_root_value).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["node", str(SIDECAR_ENTRY)],
        cwd=PROJECT_ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    process_row: dict[str, object] = {
        "name": name,
        "pid": process.pid,
        "exit_code": None,
    }
    recorder.processes.append(process_row)
    harness = SidecarProcess(
        process=process,
        url=url,
        provider=None,
        brain=None,
        data_root=data_root,
        recorder=recorder,
        name=name,
        process_row=process_row,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            harness.stop()
            raise ProbeFailure(f"sidecar {name} exited before readiness")
        try:
            health = rpc_call(
                url,
                "system.health",
                {},
                token=token,
            ).get("result")
            if isinstance(health, dict) and health.get("status") == "ready":
                return harness
        except OSError:
            pass
        time.sleep(0.05)
    harness.stop()
    raise ProbeFailure(f"sidecar {name} did not become ready")


def build_intake(
    harness: SidecarProcess,
    operation_id: str,
    thread_id: str,
    segment_id: str,
    *,
    activation_id: str | None = None,
    lease_epoch: int = 1,
    mode: str = "start",
) -> dict[str, Any]:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        SemanticActivationAuthorityV1,
        activation_id_for,
        issue_activation_token,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import (
        semantic_catalog_digest,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    service_scope = {
        "platform": "debug",
        "platform_channel_id": "dsh-runtime-probe",
        "global_user_id": "probe-user",
    }
    workspace_root = str(WORKSPACE_ROOT).replace("\\", "/")
    chosen_activation_id = activation_id or activation_id_for(
        thread_id,
        segment_id,
        lease_epoch,
    )
    clock = datetime.now(UTC)
    issued_at = clock.isoformat().replace("+00:00", "Z")
    expires_at = (clock + timedelta(minutes=5)).isoformat().replace(
        "+00:00",
        "Z",
    )
    authority = SemanticActivationAuthorityV1(
        activation_id=chosen_activation_id,
        lease_epoch=lease_epoch,
        resolution_thread_id=thread_id,
        segment_id=segment_id,
        brain_conversation_ref="chat:debug:dsh-runtime-probe",
        service_scope=service_scope,
        scope_fingerprint=content_digest(service_scope),
        audience_fingerprint="sha256:probe-audience",
        workspace_root=workspace_root,
        route_digest=harness.route_digest,
        catalog_digest=semantic_catalog_digest(),
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest=harness.route_digest,
        workspace_fingerprint=content_digest({"workspace_root": workspace_root}),
        issued_reference_digest=content_digest({
            "resolution_thread_id": thread_id,
            "segment_id": segment_id,
            "operation_id": operation_id,
        }),
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer="dsh-runtime-probe",
        issued_at=issued_at,
        expires_at=expires_at,
        token_id=f"token-{thread_id}-{segment_id}-{lease_epoch}",
        nonce=f"nonce-{thread_id}-{segment_id}-{lease_epoch}",
    )
    return {
        "schema_version": "dsh_resolution_intake.v2",
        "mode": mode,
        "request_id": f"req-{operation_id}",
        "operation_id": operation_id,
        "operation_payload_digest": f"sha256:{operation_id}",
        "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "brain_conversation_ref": "chat:debug:dsh-runtime-probe",
        "workspace_root": workspace_root,
        "route_digest": harness.route_digest,
        "semantic_tool_authority": {
            "catalog_digest": semantic_catalog_digest(),
            "token": issue_activation_token(
                authority,
                secret=SEMANTIC_SECRET.encode("utf-8"),
                now=issued_at,
            ),
        },
        "interaction_authority": {
            "issuer": "dsh-runtime-probe",
            "scope_fingerprint": authority.scope_fingerprint,
            "audience_fingerprint": authority.audience_fingerprint,
        },
        "model_input": {"objective": "finish", "facts": []},
    }


def open_resolution(
    harness: SidecarProcess,
    operation_id: str,
    thread_id: str,
    segment_id: str,
) -> dict[str, Any]:
    """Open one authenticated resolution through HTTP RPC."""

    from kazusa_ai_chatbot.dsh_tool_gateway.authority import activation_id_for

    activation_id = activation_id_for(thread_id, segment_id, 1)
    return rpc_call(
        harness.url,
        "resolution.open",
        {
            "operation_id": operation_id,
            "operation_payload_digest": f"sha256:{operation_id}",
            "activation_id": activation_id,
            "lease_epoch": 1,
            "intake": build_intake(
                harness,
                operation_id,
                thread_id,
                segment_id,
                activation_id=activation_id,
            ),
        },
    )


def continue_resolution(
    harness: SidecarProcess,
    operation_id: str,
    thread_id: str,
    segment_id: str,
    *,
    lease_epoch: int,
) -> dict[str, Any]:
    """Continue one checkpointed resolution through HTTP RPC."""

    from kazusa_ai_chatbot.dsh_tool_gateway.authority import activation_id_for

    activation_id = activation_id_for(thread_id, segment_id, lease_epoch)
    return rpc_call(
        harness.url,
        "resolution.continue",
        {
            "operation_id": operation_id,
            "operation_payload_digest": f"sha256:{operation_id}",
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "intake": build_intake(
                harness,
                operation_id,
                thread_id,
                segment_id,
                activation_id=activation_id,
                lease_epoch=lease_epoch,
                mode="continue",
            ),
        },
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def _sidecar_lifecycle_probe(recorder: ProbeRecorder) -> None:
    data_root = recorder.artifact_dir / "sidecar-data"
    semantic = start_sidecar(
        data_root,
        recorder,
        name="sidecar-semantic",
        script=[
            {
                "name": "kazusa_recall_active_context",
                "arguments": {"kinds": ["history"], "max_results": 1},
            },
            {},
        ],
    )
    try:
        health = rpc_call(semantic.url, "system.health", {})["result"]
        resolved = open_resolution(
            semantic,
            "op-probe-semantic",
            "thread-probe-semantic",
            "segment-probe-semantic",
        )["result"]
        rendered_requests = json.dumps(
            semantic.provider.requests,
            ensure_ascii=False,
        )
        _require(health["status"] == "ready", "sidecar health is not ready")
        _require(health["loopback"] is True, "sidecar is not loopback-bound")
        _require(health["dsh_runtime"] is True, "DSH runtime is not mounted")
        _require(
            resolved["exhaust"]["kind"] == "terminal",
            "semantic resolution did not terminate",
        )
        _require(
            "kazusa_semantic_capability_result.v1" in rendered_requests,
            "semantic worker result did not return to the real sidecar",
        )
        recorder.observe(
            "authenticated_boot_and_semantic_worker",
            profile=health["profile"],
            store_path=health["store_path"],
            disposition=resolved["disposition"],
            provider_calls=semantic.provider.calls,
        )
    finally:
        semantic.stop()

    checkpointed = start_sidecar(
        data_root,
        recorder,
        name="sidecar-checkpoint",
        script=[{"wait": True}],
    )
    checkpoint_response: dict[str, Any]
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                open_resolution,
                checkpointed,
                "op-probe-checkpoint",
                "thread-probe-checkpoint",
                "segment-probe-checkpoint",
            )
            time.sleep(0.2)
            from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
                activation_id_for,
            )

            checkpoint = rpc_call(
                checkpointed.url,
                "resolution.request_checkpoint",
                {
                    "operation_id": "op-probe-checkpoint-request",
                    "operation_payload_digest": "sha256:checkpoint-request",
                    "resolution_thread_id": "thread-probe-checkpoint",
                    "segment_id": "segment-probe-checkpoint",
                    "activation_id": activation_id_for(
                        "thread-probe-checkpoint",
                        "segment-probe-checkpoint",
                        1,
                    ),
                    "lease_epoch": 1,
                },
            )["result"]
            checkpoint_response = pending.result(timeout=5)["result"]
        _require(
            checkpoint["disposition"] == "checkpointed",
            "checkpoint request did not commit",
        )
    finally:
        checkpointed.stop()

    resumed = start_sidecar(
        data_root,
        recorder,
        name="sidecar-resume",
    )
    try:
        inspected = rpc_call(
            resumed.url,
            "resolution.inspect",
            {
                "operation_id": "op-probe-checkpoint",
                "operation_payload_digest": "sha256:op-probe-checkpoint",
            },
        )["result"]
        continuation = continue_resolution(
            resumed,
            "op-probe-continue",
            "thread-probe-checkpoint",
            "segment-probe-checkpoint",
            lease_epoch=2,
        )["result"]
        _require(
            inspected["disposition"] == "checkpointed",
            "restart did not recover the checkpoint",
        )
        _require(
            continuation["disposition"] == "terminal",
            "cold continuation did not terminate",
        )
        _require(
            continuation["session_id"] == checkpoint_response["session_id"],
            "cold continuation changed the DSH session",
        )
        recorder.observe(
            "sqlite_checkpoint_restart",
            session_id=continuation["session_id"],
            recovered_disposition=inspected["disposition"],
            final_disposition=continuation["disposition"],
        )
    finally:
        resumed.stop()

    replay_root = recorder.artifact_dir / "terminal-replay-data"
    crashing = start_sidecar(
        replay_root,
        recorder,
        name="sidecar-terminal-commit-crash",
        extra_env={"KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT": "1"},
    )
    try:
        try:
            open_resolution(
                crashing,
                "op-probe-replay",
                "thread-probe-replay",
                "segment-probe-replay",
            )
        except OSError:
            pass
        else:
            raise ProbeFailure("terminal-commit crash returned an HTTP response")
        crashing.wait_for_exit()
    finally:
        crashing.stop()

    replaying = start_sidecar(
        replay_root,
        recorder,
        name="sidecar-terminal-replay",
    )
    try:
        inspected = rpc_call(
            replaying.url,
            "resolution.inspect",
            {
                "operation_id": "op-probe-replay",
                "operation_payload_digest": "sha256:op-probe-replay",
            },
        )["result"]
        replayed = open_resolution(
            replaying,
            "op-probe-replay",
            "thread-probe-replay",
            "segment-probe-replay",
        )["result"]
        _require(
            inspected["disposition"] == "terminal",
            "terminal commit was unavailable after restart",
        )
        _require(
            replayed["exhaust"] == inspected["exhaust"],
            "terminal replay returned a different exhaust",
        )
        recorder.observe(
            "terminal_commit_response_loss_replay",
            session_id=inspected["session_id"],
            exact_replay=True,
        )
    finally:
        replaying.stop()


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily apply exact environment values and restore the caller."""

    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _guarded_database_environment(database_name: str) -> dict[str, str]:
    _require(
        database_name.startswith("_test_kazusa_"),
        "probe database lacks the required test prefix",
    )
    return {
        "MONGODB_DB_NAME": database_name,
        "KAZUSA_TEST_DB_GUARD": "1",
        "KAZUSA_EPHEMERAL_TEST_DATABASE_GUARD": "1",
        "KAZUSA_EPHEMERAL_TEST_DATABASE_NAME": database_name,
    }


def _task_request_and_context() -> tuple[dict[str, object], dict[str, object]]:
    from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref

    continuation_ref = build_goal_continuation_ref(
        source_episode_id="episode-dsh-runtime-probe",
        source_message_id="message-dsh-runtime-probe",
        branch_id="task_resolution",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal-dsh-runtime-probe",
        },
    )
    scene_context = {
        "channel_scope": "private",
        "character_role": "Probe Character",
        "current_user_role": "Probe User",
        "semantic_scene": "A bounded DSH runtime probe scene.",
        "public_group_scene": "",
        "conversation_continuity": "The turn contains one probe goal.",
        "semantic_temporal_context": "The probe is active now.",
    }
    request = {
        "capability": "task_resolution_request",
        "semantic_goal": "Resolve one bounded public probe question.",
        "reason": "The runtime probe requires one terminal result.",
        "evidence_handles": [],
        "start_in_background": False,
        "goal_continuation_ref": continuation_ref,
    }
    context = {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Probe Character",
        "platform": "debug",
        "channel_id": "debug:dsh-runtime-probe",
        "channel_type": "private",
        "requester_global_user_id": "probe-user",
        "requester_platform_user_id": "probe-platform-user",
        "requester_display_name": "Probe User",
        "source_message_id": "message-dsh-runtime-probe",
        "source_platform_bot_id": "probe-bot",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "",
        "brain_conversation_ref": "chat:debug:dsh-runtime-probe",
        "scene_context": scene_context,
        "goal_continuation_ref": continuation_ref,
        "local_time_context": {"local_time": "2026-09-04 12:00"},
        "prompt_message_context": {"text": "Resolve the probe question."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A bounded probe persona.",
        "conversation_summary": "A bounded probe conversation.",
        "current_timestamp_utc": _utc_now(),
        "active_turn_platform_message_ids": ["message-dsh-runtime-probe"],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }
    return request, context


async def _discover_only_binding(database_name: str) -> dict[str, object]:
    from motor.motor_asyncio import AsyncIOMotorClient

    from kazusa_ai_chatbot.config import MONGODB_URI

    client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    try:
        rows = await client[database_name]["dsh_task_bindings"].find({}).to_list(
            length=2,
        )
    finally:
        client.close()
    _require(len(rows) == 1, f"expected one DSH binding, found {len(rows)}")
    row = dict(rows[0])
    row.pop("_id", None)
    return row


async def _drop_guarded_database(
    database_name: str,
    recorder: ProbeRecorder,
) -> None:
    from motor.motor_asyncio import AsyncIOMotorClient

    from kazusa_ai_chatbot.config import MONGODB_URI
    from kazusa_ai_chatbot.db._client import close_db

    expected = os.environ.get("KAZUSA_EPHEMERAL_TEST_DATABASE_NAME")
    if (
        not database_name.startswith("_test_kazusa_")
        or expected != database_name
    ):
        raise ProbeFailure("refusing to drop an unguarded database")
    await close_db()
    client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    try:
        await client.drop_database(database_name)
    finally:
        client.close()
    recorder.cleanup.append({
        "owner": database_name,
        "status": "dropped",
    })


def _mongo_prerequisite_failure(exception: BaseException) -> bool:
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

    current: BaseException | None = exception
    while current is not None:
        if isinstance(current, (ConnectionFailure, ServerSelectionTimeoutError)):
            return True
        current = current.__cause__
    return False


async def _brain_task_lifecycle_async(recorder: ProbeRecorder) -> None:
    database_name = f"_test_kazusa_dsh_probe_{uuid4().hex}"
    data_root = recorder.artifact_dir / "brain-task-data"
    sidecar = start_sidecar(
        data_root,
        recorder,
        name="sidecar-brain-task",
        script=[
            {
                "name": "kazusa_recall_active_context",
                "arguments": {"kinds": ["history"], "max_results": 1},
            },
            {},
        ],
    )
    environment = {
        **sidecar.runtime_environment(),
        **_guarded_database_environment(database_name),
    }
    with _environment(environment):
        try:
            from agentic_resolver.persistence import (
                MongoResolutionThreadRepository,
            )
            from agentic_resolver.runtime import AgenticResolverRuntime
            from kazusa_ai_chatbot.db import task_resolution_sessions
            from kazusa_ai_chatbot.task_resolution.service import (
                resolve_task_inline,
            )

            runtime = AgenticResolverRuntime.from_environment(
                data_root=data_root.resolve(),
            )
            readiness = await runtime.readiness()
            request, context = _task_request_and_context()
            result = await resolve_task_inline(
                request,
                context,
                inline_budget_seconds=10.0,
                runtime=runtime,
                binding_store=task_resolution_sessions,
            )
            discovered = await _discover_only_binding(database_name)
            session_id = str(discovered["task_session_id"])
            binding = await task_resolution_sessions.find_binding_by_session(
                task_session_id=session_id,
            )
            _require(binding is not None, "public binding owner found no row")
            thread_id = str(binding["resolution_thread_id"])
            thread = await MongoResolutionThreadRepository().get_thread(thread_id)
            _require(thread is not None, "Mongo resolution thread is unavailable")
            _require(result["status"] == "resolved", "task did not resolve inline")
            _require(binding["state"] == "consumed_inline", "binding is not closed")
            recorder.observe(
                "brain_task_lifecycle",
                readiness_status=readiness["status"],
                task_status=result["status"],
                task_session_id=session_id,
                binding_state=binding["state"],
                resolution_thread_id=thread_id,
                thread_state=thread.state,
                database_name=database_name,
            )
        except Exception as exc:
            if _mongo_prerequisite_failure(exc):
                raise ProbeBlocked("guarded MongoDB is unavailable") from exc
            raise
        finally:
            try:
                await _drop_guarded_database(database_name, recorder)
            finally:
                sidecar.stop()


def _brain_task_lifecycle_probe(recorder: ProbeRecorder) -> None:
    asyncio.run(_brain_task_lifecycle_async(recorder))


def _capability_state(context: Mapping[str, object]) -> dict[str, object]:
    """Build the public resolver capability state from task context fields."""

    return {
        "character_profile": {
            "name": context["character_name"],
            "global_user_id": "probe-character",
        },
        "platform": context["platform"],
        "platform_channel_id": context["channel_id"],
        "channel_type": context["channel_type"],
        "global_user_id": context["requester_global_user_id"],
        "platform_user_id": context["requester_platform_user_id"],
        "platform_message_id": context["source_message_id"],
        "platform_bot_id": context["source_platform_bot_id"],
        "user_name": context["requester_display_name"],
        "storage_timestamp_utc": context["current_timestamp_utc"],
        "local_time_context": context["local_time_context"],
        "prompt_message_context": context["prompt_message_context"],
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "cognition_scene_context": context["scene_context"],
        "decontextualized_input": "Resolve the probe question.",
        "active_turn_platform_message_ids": context[
            "active_turn_platform_message_ids"
        ],
        "active_turn_conversation_row_ids": [],
        "cognitive_episode": {
            "episode_id": context["brain_conversation_ref"],
            "trigger_source": "user_message",
        },
    }


async def _transport_loss_async(recorder: ProbeRecorder) -> None:
    database_name = f"_test_kazusa_dsh_probe_{uuid4().hex}"
    data_root = recorder.artifact_dir / "transport-loss-data"
    sidecar = start_sidecar(data_root, recorder, name="sidecar-transport-loss")
    runtime_configured = False
    environment = {
        **sidecar.runtime_environment(),
        **_guarded_database_environment(database_name),
    }
    with _environment(environment):
        try:
            from agentic_resolver.runtime import AgenticResolverRuntime
            from kazusa_ai_chatbot import accepted_task as accepted_task_lifecycle
            from kazusa_ai_chatbot.background_work import jobs as background_work_jobs
            from kazusa_ai_chatbot.cognition_resolver.capabilities import (
                execute_resolver_capability_request,
            )
            from kazusa_ai_chatbot.db import task_resolution_sessions
            from kazusa_ai_chatbot.task_resolution.service import (
                configure_task_resolution_runtime,
            )

            runtime = AgenticResolverRuntime.from_environment(
                data_root=data_root.resolve(),
            )
            readiness = await runtime.readiness()
            configure_task_resolution_runtime(
                runtime,
                binding_store=task_resolution_sessions,
                accepted_task_store=accepted_task_lifecycle,
                background_queue=background_work_jobs,
            )
            runtime_configured = True
            _request, context = _task_request_and_context()
            capability_request = {
                "schema_version": "resolver_capability_request.v1",
                "capability_kind": "task_resolution_request",
                "objective": "Resolve one bounded public probe question.",
                "reason": "The transport-loss probe requires DSH.",
                "priority": "now",
                "goal_continuation_ref": context["goal_continuation_ref"],
            }
            sidecar.stop()
            observation = await execute_resolver_capability_request(
                capability_request,
                _capability_state(context),
            )
            binding = await _discover_only_binding(database_name)
            evidence_state = observation["task_resolution_evidence_state"]
            _require(readiness["status"] == "ready", "readiness did not pass")
            _require(observation["status"] == "failed", "failure was not typed")
            _require(
                evidence_state["state"] == "blocked",
                "failure evidence is not blocked",
            )
            _require(binding["state"] == "faulted", "binding did not fault")
            recorder.observe(
                "transport_loss",
                readiness_before_loss=readiness["status"],
                observation_status=observation["status"],
                evidence_state=evidence_state["state"],
                prompt_safe_summary=observation["prompt_safe_summary"],
                binding_state=binding["state"],
                database_name=database_name,
            )
        except Exception as exc:
            if _mongo_prerequisite_failure(exc):
                raise ProbeBlocked("guarded MongoDB is unavailable") from exc
            raise
        finally:
            try:
                if runtime_configured:
                    configure_task_resolution_runtime(None)
            finally:
                try:
                    await _drop_guarded_database(database_name, recorder)
                finally:
                    sidecar.stop()


def _transport_loss_probe(recorder: ProbeRecorder) -> None:
    asyncio.run(_transport_loss_async(recorder))


def run_probe(probe_name: str, artifact_dir: Path) -> dict[str, object]:
    """Run one named probe and always write its final result artifact."""

    if probe_name not in PROBE_NAMES:
        raise ValueError(f"unsupported probe: {probe_name}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ValueError("artifact directory must be new or empty")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    recorder = ProbeRecorder(probe_name=probe_name, artifact_dir=artifact_dir)
    status = "passed"
    error = ""
    try:
        {
            "sidecar-lifecycle": _sidecar_lifecycle_probe,
            "brain-task-lifecycle": _brain_task_lifecycle_probe,
            "transport-loss": _transport_loss_probe,
        }[probe_name](recorder)
    except ProbeBlocked as exc:
        status = "blocked"
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - serialize unexpected probe failure.
        status = "failed"
        error = f"{exc.__class__.__name__}: {exc}"
    result = recorder.result(status, error)
    result_path = artifact_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    recorder.artifacts.append(str(result_path.resolve()))
    result["artifacts"] = list(recorder.artifacts)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_name", choices=PROBE_NAMES)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the selected probe and emit its machine-readable result."""

    args = _parse_args()
    result = run_probe(args.probe_name, args.artifact_dir.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return {"passed": 0, "failed": 1, "blocked": 2}[str(result["status"])]


if __name__ == "__main__":
    raise SystemExit(main())
