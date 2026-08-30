"""Black-box tests for the independent long-lived Node sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from agentic_resolver.controller import ResolutionController
from agentic_resolver.persistence import InMemoryResolutionThreadRepository
from agentic_resolver.rpc import DSHRpcClient
from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    SemanticActivationAuthorityV1,
    activation_id_for,
    issue_activation_token,
)
from kazusa_ai_chatbot.dsh_tool_gateway.catalog import (
    SEMANTIC_TOOL_NAMES,
    semantic_catalog_digest,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ENTRY = PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
TOKEN = "sidecar-test-token"
WORKSPACE_ROOT = PROJECT_ROOT.resolve()
_ROUTE_BASES: dict[str, str] = {}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _FakeOpenAi:
    """Minimal OpenAI-compatible SSE endpoint used by the real Standard host."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                request_body = b""
                if length:
                    request_body = self.rfile.read(length)
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
                    row for row in rows
                    if isinstance(row, dict) and isinstance(row.get("name"), str)
                ]
                if not calls and step.get("wait") is not True and step.get("text") is None:
                    calls = [{
                        "name": "submit_resolution",
                        "arguments": {
                            "status": "resolved",
                            "summary": "done",
                            "findings": [],
                            "completed_subgoals": [],
                            "remaining_needs": [],
                            "clarification_request": None,
                            "approval_request": None,
                            "artifact_refs": [],
                            "warnings": [],
                        },
                    }]
                if calls:
                    tool_calls = []
                    for index, call in enumerate(calls):
                        arguments = call.get("arguments", {})
                        tool_calls.append({
                            "index": index,
                            "id": f"call-{owner.calls}-{index}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(arguments, separators=(",", ":")),
                            },
                        })
                    delta: dict[str, object] = {
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    }
                    finish = "tool_calls"
                else:
                    delta = {"role": "assistant", "content": str(step.get("text", "waiting"))}
                    finish = "stop"
                body = "".join([
                    f'data: {json.dumps({"id": f"response-{owner.calls}", "choices": [{"delta": delta, "finish_reason": None}]})}\n\n',
                    f'data: {json.dumps({"id": f"response-{owner.calls}", "choices": [{"delta": {}, "finish_reason": finish}]})}\n\n',
                    "data: [DONE]\n\n",
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        self.script = script or [{}]
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = __import__("threading").Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class _FakeBrainHealth:
    """Loopback Brain readiness owner used by the real sidecar process."""

    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/runtime/dsh/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps({
                    "schema_version": "dsh_brain_interaction_health.v1",
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
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


def _route_digest(base_url: str) -> str:
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
    return f"sha256:{sha256(_canonical_json(descriptor).encode()).hexdigest()}"


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rpc(url: str, method: str, params: dict[str, Any], *, token: str = TOKEN) -> dict[str, Any]:
    body = json.dumps({"jsonrpc": "2.0", "id": f"rpc-{time.time_ns()}", "method": method, "params": {"protocol_version": "kazusa.dsh-resolution-rpc.v2", **params}}).encode()
    request = Request(url, data=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _start(
    tmp_path: Path,
    script: list[dict[str, Any]] | None = None,
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], str]:
    port = _port()
    url = f"http://127.0.0.1:{port}/rpc"
    fake_openai = _FakeOpenAi(script or [])
    fake_brain = _FakeBrainHealth()
    env = os.environ.copy()
    env.update({
        "KAZUSA_DSH_SIDECAR_URL": url,
        "KAZUSA_DSH_RPC_TOKEN": TOKEN,
        "KAZUSA_DSH_DATA_ROOT": str(tmp_path.resolve()),
        "AGENTIC_RESOLVER_WORKSPACE_ROOT": str(WORKSPACE_ROOT),
        "AGENTIC_RESOLVER_LLM_BASE_URL": fake_openai.base_url,
        "AGENTIC_RESOLVER_LLM_API_KEY": "resolver-test-key",
        "AGENTIC_RESOLVER_LLM_MODEL": "qwen27b-5090",
        "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS": "50176",
        "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS": "8192",
        "AGENTIC_RESOLVER_LLM_THINKING_ENABLED": "true",
        "KAZUSA_DSH_BRAIN_URL": fake_brain.base_url,
        "KAZUSA_DSH_BRAIN_SHARED_SECRET": "brain-test-secret",
        "KAZUSA_DSH_TOOL_GATEWAY_SECRET": "semantic-test-secret",
        "KAZUSA_DSH_PYTHON_EXECUTABLE": str(Path(sys.executable).resolve()),
        "NODE_ENV": "test",
    })
    env.pop("KAZUSA_DSH_CAPABILITY_TOKEN", None)
    env.update(extra_env or {})
    process = subprocess.Popen(
        ["node", str(SIDECAR_ENTRY)], cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            fake_openai.close()
            fake_brain.close()
            raise AssertionError(f"sidecar exited: {stdout}\n{stderr}")
        try:
            health = _rpc(url, "system.health", {})
            result = health.get("result")
            if not isinstance(result, dict) or result.get("status") != "ready":
                time.sleep(0.05)
                continue
            process._fake_openai = fake_openai  # type: ignore[attr-defined]
            process._fake_brain = fake_brain  # type: ignore[attr-defined]
            _ROUTE_BASES[url] = fake_openai.base_url
            return process, url
        except OSError:
            time.sleep(0.05)
    process.terminate()
    fake_openai.close()
    fake_brain.close()
    raise AssertionError("sidecar did not become healthy")


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    fake_openai = getattr(process, "_fake_openai", None)
    if fake_openai is not None:
        fake_openai.close()
    fake_brain = getattr(process, "_fake_brain", None)
    if fake_brain is not None:
        fake_brain.close()
    for route_url, base_url in list(_ROUTE_BASES.items()):
        if base_url == getattr(fake_openai, "base_url", None):
            _ROUTE_BASES.pop(route_url, None)


def _intake(
    operation_id: str,
    thread_id: str,
    segment_id: str,
    *,
    route_digest: str,
    activation_id: str | None = None,
    lease_epoch: int = 1,
    mode: str = "start",
) -> dict[str, Any]:
    service_scope = {
        "platform": "debug",
        "platform_channel_id": "sidecar-process",
        "global_user_id": "user",
    }
    workspace_root = str(WORKSPACE_ROOT).replace("\\", "/")
    chosen_activation_id = activation_id or activation_id_for(
        thread_id, segment_id, lease_epoch
    )
    clock = datetime.now(UTC)
    issued_at = clock.isoformat().replace("+00:00", "Z")
    expires_at = (clock + timedelta(minutes=5)).isoformat().replace(
        "+00:00", "Z"
    )
    authority = SemanticActivationAuthorityV1(
        activation_id=chosen_activation_id,
        lease_epoch=lease_epoch,
        resolution_thread_id=thread_id,
        segment_id=segment_id,
        brain_conversation_ref="chat:debug:sidecar-test",
        service_scope=service_scope,
        scope_fingerprint=content_digest(service_scope),
        audience_fingerprint="sha256:audience",
        workspace_root=workspace_root,
        route_digest=route_digest,
        catalog_digest=semantic_catalog_digest(),
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest=route_digest,
        workspace_fingerprint=content_digest({"workspace_root": workspace_root}),
        issued_reference_digest=content_digest({
            "resolution_thread_id": thread_id,
            "segment_id": segment_id,
            "operation_id": operation_id,
        }),
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer="dsh-sidecar-test",
        issued_at=issued_at,
        expires_at=expires_at,
        token_id=f"token-{thread_id}-{segment_id}-{lease_epoch}",
        nonce=f"nonce-{thread_id}-{segment_id}-{lease_epoch}",
    )
    return {
        "schema_version": "dsh_resolution_intake.v2", "mode": mode,
        "request_id": f"req-{operation_id}", "operation_id": operation_id,
        "operation_payload_digest": f"sha256:{operation_id}", "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "brain_conversation_ref": "chat:debug:sidecar-test",
        "workspace_root": workspace_root,
        "route_digest": route_digest,
        "semantic_tool_authority": {
            "catalog_digest": semantic_catalog_digest(),
            "token": issue_activation_token(
                authority,
                secret=b"semantic-test-secret",
                now=issued_at,
            ),
        },
        "interaction_authority": {
            "issuer": "dsh-sidecar-test",
            "scope_fingerprint": authority.scope_fingerprint,
            "audience_fingerprint": authority.audience_fingerprint,
        },
        "model_input": {"objective": "finish", "facts": []},
    }


def _open(url: str, operation_id: str, thread_id: str, segment_id: str) -> dict[str, Any]:
    base_url = _ROUTE_BASES[url]
    activation_id = activation_id_for(thread_id, segment_id, 1)
    return _rpc(url, "resolution.open", {"operation_id": operation_id, "operation_payload_digest": f"sha256:{operation_id}", "activation_id": activation_id, "lease_epoch": 1, "intake": _intake(operation_id, thread_id, segment_id, route_digest=_route_digest(base_url), activation_id=activation_id)})


def _continue(
    url: str,
    operation_id: str,
    thread_id: str,
    segment_id: str,
    *,
    lease_epoch: int,
) -> dict[str, Any]:
    base_url = _ROUTE_BASES[url]
    activation_id = activation_id_for(thread_id, segment_id, lease_epoch)
    intake = _intake(
        operation_id,
        thread_id,
        segment_id,
        route_digest=_route_digest(base_url),
        activation_id=activation_id,
        lease_epoch=lease_epoch,
        mode="continue",
    )
    return _rpc(
        url,
        "resolution.continue",
        {
            "operation_id": operation_id,
            "operation_payload_digest": f"sha256:{operation_id}",
            "activation_id": activation_id,
            "lease_epoch": lease_epoch,
            "intake": intake,
        },
    )


def test_standalone_runtime_uses_one_long_lived_sidecar_across_two_resolves(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        assert _open(url, "op_1", "res_1", "seg_1")["result"]["exhaust"]["kind"] == "terminal"
        assert _open(url, "op_2", "res_2", "seg_2")["result"]["exhaust"]["kind"] == "terminal"
        assert process.poll() is None
    finally:
        _stop(process)


def test_real_sidecar_worker_accepts_python_semantic_call_contract(
    tmp_path: Path,
) -> None:
    """The TypeScript gateway and Python worker share one signed call schema."""

    process, url = _start(
        tmp_path,
        [
            {
                "name": "kazusa_recall_active_context",
                "arguments": {"kinds": ["history"], "max_results": 1},
            },
            {},
        ],
    )
    try:
        response = _open(url, "op_semantic", "res_semantic", "seg_semantic")
        assert response["result"]["exhaust"]["kind"] == "terminal"
        fake_openai = process._fake_openai
        rendered = json.dumps(fake_openai.requests, ensure_ascii=False)
        assert "kazusa_semantic_capability_result.v1" in rendered
        assert "SEMANTIC_AUTHORITY_INVALID" not in rendered
    finally:
        _stop(process)


def test_sidecar_requires_loopback_auth_data_root_model_and_versioned_store_path(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        health = _rpc(url, "system.health", {})["result"]
        assert health["store_path"].endswith("dsh/0.1.1-rc.2/sessions.sqlite")
        assert health["loopback"] is True
        assert health["dsh_runtime"] is True
    finally:
        _stop(process)


def test_sidecar_profile_factory_and_dependency_graph_fail_closed(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        assert _rpc(url, "system.health", {})["result"]["profile"] == "kazusa-resolver-standard-v2"
    finally:
        _stop(process)

    probe = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            (
                "import { assertCompatibleDependencyGraph } from "
                "'./sidecars/dsh_resolution/dist/src/profile.js';"
                "assertCompatibleDependencyGraph("
                "{'@deepseek-ai/dsh-session':'incompatible'});"
            ),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0
    assert "incompatible DSH dependency" in probe.stderr


def test_brain_and_sidecar_run_independently_and_sidecar_stop_does_not_stop_brain(tmp_path: Path) -> None:
    process, _ = _start(tmp_path)
    ready = tmp_path / "brain-ready"
    brain = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time; "
                "import kazusa_ai_chatbot.brain_service; "
                f"open({str(ready)!r}, 'w', encoding='utf-8').write('ready'); "
                "time.sleep(30)"
            ),
        ],
        cwd=PROJECT_ROOT,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ready.exists():
            if brain.poll() is not None:
                raise AssertionError("brain process exited before readiness")
            time.sleep(0.05)
        assert ready.read_text(encoding="utf-8") == "ready"
        _stop(process)
        assert brain.poll() is None
    finally:
        if process.poll() is None:
            _stop(process)
        brain.terminate()
        brain.wait(timeout=5)


def test_sidecar_restart_preserves_checkpoint_and_cold_resumes(tmp_path: Path) -> None:
    process, url = _start(tmp_path, [{"wait": True}])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(_open, url, "op_1", "res_1", "seg_1")
            time.sleep(0.2)
            params = {"operation_id": "op_checkpoint", "operation_payload_digest": "sha256:checkpoint", "resolution_thread_id": "res_1", "segment_id": "seg_1", "activation_id": activation_id_for("res_1", "seg_1", 1), "lease_epoch": 1}
            assert _rpc(url, "resolution.request_checkpoint", params)["result"]["disposition"] == "checkpointed"
            response = pending.result(timeout=5)
        assert response["result"]["disposition"] in {"checkpointed", "admitted_active"}
    finally:
        _stop(process)
    restarted, restarted_url = _start(tmp_path)
    try:
        inspected = _rpc(restarted_url, "resolution.inspect", {"operation_id": "op_1", "operation_payload_digest": "sha256:op_1"})["result"]
        assert inspected["disposition"] == "checkpointed"
        assert inspected["session_id"] == response["result"]["session_id"]
        resumed = _continue(
            restarted_url,
            "op_2",
            "res_1",
            "seg_1",
            lease_epoch=2,
        )["result"]
        assert resumed["disposition"] == "terminal"
        assert resumed["session_id"] == inspected["session_id"]
    finally:
        _stop(restarted)


def test_scope_audience_profile_release_store_model_catalog_and_policy_mismatch_rotates(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        health = _rpc(url, "system.health", {})["result"]
        health_catalog = health["catalog"]
        native_catalog_digest = str(health_catalog["native_catalog_digest"])
        fields = {
            "scope_fingerprint": "sha256:old-scope",
            "audience_fingerprint": "sha256:old-audience",
            "resolver_profile_version": "old-profile",
            "dsh_release": "old-release",
            "session_store_epoch": "old-store",
            "route_digest": "sha256:old-route",
            "standard_catalog_digest": "sha256:old-catalog",
            "semantic_catalog_digest": "sha256:old-semantic-catalog",
            "policy_epoch": "old-policy",
            "interaction_id": "old-interaction",
        }
        for index, (field, old_value) in enumerate(fields.items()):
            base_url = _ROUTE_BASES[url]
            intake = _intake(
                f"op_rotate_{index}",
                f"res_rotate_{index}",
                f"seg_rotate_{index}",
                route_digest=_route_digest(base_url),
            )
            parsed = __import__(
                "agentic_resolver.contracts",
                fromlist=["DSHResolutionIntakeV2"],
            ).DSHResolutionIntakeV2.from_mapping(intake)
            initial_segment_id = f"seg_rotate_{index}_initial"
            initial_parsed = __import__(
                "dataclasses",
                fromlist=["replace"],
            ).replace(parsed, segment_id=initial_segment_id)
            repository = InMemoryResolutionThreadRepository()
            segment = ResolutionController._segment(initial_parsed, {
                "route_digest": parsed.route_digest,
                    "native_catalog_digest": native_catalog_digest,
                "semantic_catalog_digest": semantic_catalog_digest(),
                "published_catalog_digest": "sha256:published-catalog",
                "profile_version": "kazusa-resolver-standard-v2",
                "dsh_release": "0.1.1-rc.2",
                "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-standard-v2",
                "policy_epoch": "dsh-standard-policy-v2",
                "workspace_root": str(WORKSPACE_ROOT).replace("\\", "/"),
            })
            segment[field] = old_value
            repository.create_thread_v2(
                resolution_thread_id=str(intake["resolution_thread_id"]),
                brain_conversation_ref=str(intake["brain_conversation_ref"]),
                root_goal_ref="goal-ref",
                priority="now",
                workspace_root=str(intake["workspace_root"]),
                workspace_fingerprint=str(segment["workspace_fingerprint"]),
                route_digest=str(segment["route_digest"]),
                profile_version=str(segment["resolver_profile_version"]),
                standard_catalog_digest=str(segment["standard_catalog_digest"]),
                semantic_catalog_digest=str(segment["semantic_catalog_digest"]),
                scope_fingerprint=str(segment["scope_fingerprint"]),
                audience_fingerprint=str(segment["audience_fingerprint"]),
                policy_epoch=str(segment["policy_epoch"]),
                interaction_id=str(segment["interaction_id"]),
                segment=segment,
                now="2026-08-28T00:00:00Z",
            )
            repository._threads[str(intake["resolution_thread_id"])]["segments"][0] = segment
            client = DSHRpcClient(url, TOKEN)
            controller = ResolutionController(
                repository,
                client,
                owner_id="process-test",
                semantic_authority_secret=b"semantic-test-secret",
            )
            intake["mode"] = "continue"
            result = asyncio.run(controller.continue_resolution(intake))
            assert result["segment_id"] != segment["segment_id"]
            record = repository.get_thread(str(intake["resolution_thread_id"]))
            assert record is not None
            assert record.segments[-1]["rotation_reason"] == f"{field}_mismatch"
    finally:
        _stop(process)


def test_second_http_request_checkpoints_and_cancels_pending_execution(tmp_path: Path) -> None:
    process, url = _start(tmp_path, [{"wait": True}])
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(_open, url, "op_1", "res_1", "seg_1")
            time.sleep(0.2)
            params = {"operation_id": "op_checkpoint", "operation_payload_digest": "sha256:checkpoint", "resolution_thread_id": "res_1", "segment_id": "seg_1", "activation_id": activation_id_for("res_1", "seg_1", 1), "lease_epoch": 1}
            assert _rpc(url, "resolution.request_checkpoint", params)["result"]["disposition"] == "checkpointed"
            assert pending.result(timeout=5)["result"]["disposition"] == "checkpointed"
    finally:
        _stop(process)


def test_zero_call_and_multi_call_steps_execute_no_tool_body_and_exhaust_correction_budget(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        result = _open(url, "op_standard", "res_standard", "seg_standard")["result"]
        assert result["exhaust"]["kind"] == "terminal"
        diagnostics = _rpc(url, "system.health", {})["result"]["diagnostics"]
        assert diagnostics["terminal_tool_executions"] == 1
        assert diagnostics["correction_attempts"] == 0
    finally:
        _stop(process)


def test_kill_after_terminal_commit_before_http_response_replays_exact_exhaust(tmp_path: Path) -> None:
    process, url = _start(
        tmp_path,
        extra_env={"KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT": "1"},
    )
    try:
        with pytest.raises(OSError):
            _open(url, "op_1", "res_1", "seg_1")
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            _stop(process)

    restarted, restarted_url = _start(tmp_path)
    try:
        inspected = _rpc(
            restarted_url,
            "resolution.inspect",
            {
                "operation_id": "op_1",
                "operation_payload_digest": "sha256:op_1",
            },
        )["result"]
        assert inspected["disposition"] == "terminal"
        replayed = _open(
            restarted_url,
            "op_1",
            "res_1",
            "seg_1",
        )["result"]
        assert replayed["exhaust"] == inspected["exhaust"]
    finally:
        _stop(restarted)


def test_missing_or_invalid_terminal_receipt_never_returns_terminal_exhaust(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        exhaust = _open(url, "op_1", "res_1", "seg_1")["result"]["exhaust"]
        assert exhaust["kind"] == "terminal"
        assert exhaust["identity"]["route_digest"] == _route_digest(
            _ROUTE_BASES[url]
        )
    finally:
        _stop(process)


def test_terminal_exhaust_contains_only_validated_submit_resolution_and_evidence_refs(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        exhaust = _open(url, "op_1", "res_1", "seg_1")["result"]["exhaust"]
        assert set(exhaust) <= {"kind", "terminal", "evidence", "identity", "usage", "last_committed_seq"}
    finally:
        _stop(process)


def test_bad_rpc_authentication_version_and_schema_fail_closed(tmp_path: Path) -> None:
    process, url = _start(tmp_path)
    try:
        with pytest.raises(HTTPError) as denied:
            _rpc(url, "system.health", {}, token="wrong")
        assert denied.value.code == 401
    finally:
        _stop(process)


def test_v2_cold_restart_rebuilds_standard_session_evidence_pending_interaction_and_terminal(
    tmp_path: Path,
) -> None:
    """A V2 sidecar restart rebuilds durable session and interaction state."""

    process, url = _start(tmp_path)
    try:
        health = _rpc(url, "system.health", {})["result"]
        assert health["protocol_version"] == "kazusa.dsh-resolution-rpc.v2"
        assert health["profile"] == "kazusa-resolver-standard-v2"
        assert health["store_epoch"] == "dsh-sqlite-0.1.1-rc.2-standard-v2"
    finally:
        _stop(process)

    restarted, restarted_url = _start(tmp_path)
    try:
        health = _rpc(restarted_url, "system.health", {})["result"]
        assert health["protocol_version"] == "kazusa.dsh-resolution-rpc.v2"
        assert health["profile"] == "kazusa-resolver-standard-v2"
        assert health["store_epoch"] == "dsh-sqlite-0.1.1-rc.2-standard-v2"
        assert health["readiness"] == {
            "route": "ready",
            "standard": "ready",
            "semantic_worker": "ready",
            "web": "ready",
            "brain": "ready",
        }
        assert health["diagnostics"]["live_activations"] == 0
    finally:
        _stop(restarted)


def test_default_dsh_web_provider_uses_only_native_names_and_imports_no_kazusa_webagent() -> None:
    """The default web capability remains the installed DSH provider."""

    base_patch = (
        PROJECT_ROOT
        / "sidecars"
        / "dsh_resolution"
        / "node_modules"
        / "@deepseek-ai"
        / "dsh-base"
        / "cordis.patch.yml"
    ).read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-web" in base_patch
    assert "@deepseek-ai/dsh-web-search-deepseek" in base_patch
    sidecar_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "sidecars" / "dsh_resolution" / "src").glob("*.ts")
    ).lower()
    for forbidden in ("kazusa_webagent", "searxng", "url-reader", "url_reader"):
        assert forbidden not in sidecar_source


def test_standard_pwsh_reads_edits_runs_fixture_test_and_imports_no_kazusa_coding_agent(
    tmp_path: Path,
) -> None:
    """Native PowerShell and workspace tests remain outside Kazusa coding code."""

    fixture = tmp_path / "dsh-standard-coding"
    shutil.copytree(
        PROJECT_ROOT / "tests" / "fixtures" / "dsh_standard_coding",
        fixture,
    )
    coding_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "sidecars" / "dsh_resolution" / "src").glob("*.ts")
    )
    assert "coding_agent" not in coding_source
    assert "kazusa_ai_chatbot.coding_agent" not in coding_source
    probe = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-ChildItem -LiteralPath $env:DSH_FIXTURE | Select-Object -ExpandProperty Name",
        ],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "DSH_FIXTURE": str(fixture.resolve())},
    )
    assert probe.returncode == 0, probe.stderr
    assert "calculator.py" in probe.stdout
    assert "test_calculator.py" in probe.stdout


def test_plan3_public_media_tool_is_advertised_with_matching_fourteen_tool_digest(
    tmp_path: Path,
) -> None:
    """A healthy sidecar publishes the additive fourteen-tool digest."""

    process, url = _start(tmp_path)
    try:
        health = _rpc(url, "system.health", {})["result"]
        catalog = health["catalog"]
        assert semantic_catalog_digest().startswith("sha256:")
        assert len(SEMANTIC_TOOL_NAMES) == 14
        assert catalog["semantic_catalog_digest"] == semantic_catalog_digest()
        assert catalog["published_catalog_digest"].startswith("sha256:")
    finally:
        _stop(process)


def test_plan3_public_media_tool_forwards_only_url_and_question() -> None:
    """The executable sidecar gateway forwards only URL and question inputs."""

    sidecar_root = PROJECT_ROOT / "sidecars" / "dsh_resolution"
    node_script = r'''
import { createSemanticGateway } from "./dist/src/semantic_gateway.js";
import {
  issueActivationToken,
  scopeFingerprint,
  workspaceFingerprint,
} from "./dist/src/contracts.js";

const serviceScope = {
  platform: "debug",
  platform_channel_id: "channel-1",
  global_user_id: "user-1",
};
const workspaceRoot = "C:/workspace/project";
const authority = {
  schema_version: "kazusa_semantic_tool_authority.v1",
  resolution_thread_id: "thread-v2",
  segment_id: "segment-v2",
  activation_id: "activation-v2",
  lease_epoch: 1,
  brain_conversation_ref: "chat:debug:one",
  service_scope: serviceScope,
  scope_fingerprint: scopeFingerprint(serviceScope),
  audience_fingerprint: "sha256:audience",
  workspace_root: workspaceRoot,
  route_digest: "sha256:route",
  catalog_digest: "sha256:catalog",
  profile_version: "kazusa-resolver-standard-v2",
  model_route_digest: "sha256:route",
  workspace_fingerprint: workspaceFingerprint(workspaceRoot),
  issued_reference_digest: "sha256:issued",
  policy_epoch: "dsh-standard-policy-v2",
  interaction_issuer: "dsh-sidecar-test",
  issued_at: "2026-08-30T00:00:00.000Z",
  expires_at: "2026-08-30T00:05:00.000Z",
  token_id: "token-1",
  nonce: "nonce-1",
};
const frames = [];
const gateway = createSemanticGateway({
  secret: "gateway-secret",
  authority,
  authorityToken: issueActivationToken(authority, "gateway-secret"),
  call: async (frame) => {
    frames.push(frame);
    return {
      schema_version: "kazusa_semantic_capability_result.v1",
      status: "ok",
      entities: [{ status: "answered" }],
      page: { has_more: false, next_page_ref: null },
      evidence: [],
      mutation: null,
      error: null,
    };
  },
  persistEvidence: async () => {},
  now: () => new Date("2026-08-30T00:01:00.000Z"),
});
await gateway.invoke("kazusa_inspect_public_media", {
  public_media_url: "https://example.test/image.png",
  question: "What is visible?",
});
let rejected = false;
try {
  await gateway.invoke("kazusa_inspect_public_media", {
    public_media_url: "https://example.test/image.png",
    question: "What is visible?",
    capability_token: "must-be-rejected",
  });
} catch {
  rejected = true;
}
process.stdout.write(JSON.stringify({ frames, rejected }));
'''
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=sidecar_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["rejected"] is True
    assert payload["frames"][0]["arguments"] == {
        "public_media_url": "https://example.test/image.png",
        "question": "What is visible?",
    }
    assert "capability_token" not in payload["frames"][0]["arguments"]
    assert "authority" not in payload["frames"][0]["arguments"]
