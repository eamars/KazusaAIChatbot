"""Explicit real-local-model coverage for the standalone DSH V2 resolver."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from subprocess import Popen
from urllib.request import Request, urlopen

import pytest

from agentic_resolver import AgenticResolverRuntime
from kazusa_ai_chatbot.db import resolution_threads
from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    SemanticActivationAuthorityV1,
    activation_id_for,
    issue_activation_token,
)
from kazusa_ai_chatbot.dsh_tool_gateway.catalog import semantic_catalog_digest
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ENTRY = PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v2"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _route_digest(environment: dict[str, str]) -> str:
    thinking = environment["AGENTIC_RESOLVER_LLM_THINKING_ENABLED"] == "true"
    descriptor = {
        "route_name": "kazusa-agentic-resolver",
        "base_url": environment["AGENTIC_RESOLVER_LLM_BASE_URL"],
        "model": environment["AGENTIC_RESOLVER_LLM_MODEL"],
        "context_window_tokens": int(
            environment["AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS"]
        ),
        "max_completion_tokens": int(
            environment["AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS"]
        ),
        "thinking_enabled": thinking,
        "supports_developer_role": False,
        "max_tokens_field": "max_completion_tokens",
        "thinking_format": "qwen-chat-template",
        "chat_template_kwargs_enable_thinking": thinking,
        "reasoning_effort": "high" if thinking else "off",
        "output_mode": "text",
        "compatibility_epoch": "qwen-openai-completions-v1",
        "credential_reference": "AGENTIC_RESOLVER_LLM_API_KEY",
    }
    return f"sha256:{sha256(_canonical_json(descriptor).encode()).hexdigest()}"


def _require_live_backend() -> None:
    """Require every explicit V2 route, host, and bridge setting."""

    if os.environ.get("KAZUSA_RUN_LIVE_LLM") != "1":
        pytest.skip("set KAZUSA_RUN_LIVE_LLM=1 for real-local-model coverage")
    required = (
        "AGENTIC_RESOLVER_LLM_API_KEY",
        "AGENTIC_RESOLVER_LLM_BASE_URL",
        "AGENTIC_RESOLVER_LLM_MODEL",
        "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS",
        "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS",
        "AGENTIC_RESOLVER_LLM_THINKING_ENABLED",
        "AGENTIC_RESOLVER_WORKSPACE_ROOT",
        "KAZUSA_DSH_SIDECAR_URL",
        "KAZUSA_DSH_RPC_TOKEN",
        "KAZUSA_DSH_BRAIN_URL",
        "KAZUSA_DSH_BRAIN_SHARED_SECRET",
        "KAZUSA_DSH_TOOL_GATEWAY_SECRET",
        "KAZUSA_DSH_PYTHON_EXECUTABLE",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"live V2 configuration is missing: {', '.join(missing)}")


def _rpc(
    endpoint: str,
    token: str,
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    request = Request(
        endpoint,
        data=json.dumps({
            "jsonrpc": "2.0",
            "id": f"live-{time.time_ns()}",
            "method": method,
            "params": {"protocol_version": RPC_PROTOCOL_VERSION, **params},
        }).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    timeout = None if method in {"resolution.open", "resolution.continue"} else 10
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    assert isinstance(value, dict)
    return value


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _intake(
    environment: dict[str, str],
    *,
    operation_id: str,
    thread_id: str,
    segment_id: str,
    objective: str,
) -> dict[str, object]:
    payload = {
        "method": "resolution.open",
        "operation_id": operation_id,
        "objective": objective,
        "thread_id": thread_id,
        "segment_id": segment_id,
    }
    payload_digest = f"sha256:{sha256(_canonical_json(payload).encode()).hexdigest()}"
    workspace_root = environment["AGENTIC_RESOLVER_WORKSPACE_ROOT"].replace(
        "\\", "/"
    )
    route_digest = _route_digest(environment)
    service_scope = {
        "platform": "debug",
        "platform_channel_id": "live-channel",
        "global_user_id": "live-user",
    }
    audience = {"kind": "live", "operation": operation_id}
    issued = datetime.now(UTC).replace(microsecond=0)
    authority = SemanticActivationAuthorityV1(
        activation_id=activation_id_for(thread_id, segment_id, 1),
        lease_epoch=1,
        resolution_thread_id=thread_id,
        segment_id=segment_id,
        brain_conversation_ref=f"chat:live:{operation_id}",
        service_scope=service_scope,
        scope_fingerprint=content_digest(service_scope),
        audience_fingerprint=content_digest(audience),
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
        interaction_issuer="dsh-live-test",
        issued_at=issued.isoformat().replace("+00:00", "Z"),
        expires_at=(issued + timedelta(minutes=5)).isoformat().replace(
            "+00:00", "Z"
        ),
        token_id=f"tok-{operation_id}",
        nonce=f"nonce-{operation_id}",
    )
    return {
        "schema_version": "dsh_resolution_intake.v2",
        "mode": "start",
        "request_id": f"request-{operation_id}",
        "operation_id": operation_id,
        "operation_payload_digest": payload_digest,
        "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "brain_conversation_ref": f"chat:live:{operation_id}",
        "workspace_root": environment["AGENTIC_RESOLVER_WORKSPACE_ROOT"],
        "route_digest": route_digest,
        "model_input": {"objective": objective, "facts": []},
        "semantic_tool_authority": {
            "catalog_digest": authority.catalog_digest,
            "token": issue_activation_token(
                authority,
                secret=environment["KAZUSA_DSH_TOOL_GATEWAY_SECRET"].encode(),
                now=authority.issued_at,
            ),
        },
        "interaction_authority": {
            "issuer": "dsh-live-test",
            "scope_fingerprint": authority.scope_fingerprint,
            "audience_fingerprint": authority.audience_fingerprint,
        },
    }


async def _start_sidecar(
    tmp_path: Path,
) -> tuple[Popen[str], str, str, dict[str, str]]:
    environment = os.environ.copy()
    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}/rpc"
    token = environment["KAZUSA_DSH_RPC_TOKEN"]
    environment.update({
        "KAZUSA_DSH_SIDECAR_URL": endpoint,
        "KAZUSA_DSH_DATA_ROOT": str(tmp_path.resolve()),
    })
    process = await asyncio.create_subprocess_exec(
        "node",
        str(SIDECAR_ENTRY),
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(f"sidecar exited: {stdout}\n{stderr}")
        try:
            health = await asyncio.to_thread(_rpc, endpoint, token, "system.health", {})
            result = health.get("result")
            if isinstance(result, dict) and result.get("status") == "ready":
                return process, endpoint, token, environment
        except OSError:
            pass
        await asyncio.sleep(0.1)
    process.terminate()
    await process.wait()
    raise AssertionError("live V2 sidecar did not become healthy")


async def _stop_sidecar(process: Popen[str]) -> None:
    if process.returncode is None:
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_standalone_sidecar_resolution_reaches_submit_resolution(
    tmp_path: Path,
) -> None:
    """The actual Standard sidecar reaches one validated V2 terminal action."""

    _require_live_backend()
    process, endpoint, token, environment = await _start_sidecar(tmp_path)
    operation_id = f"live-submit-{time.time_ns()}"
    intake = _intake(
        environment,
        operation_id=operation_id,
        thread_id=f"thread-{operation_id}",
        segment_id=f"segment-{operation_id}",
        objective="Return a resolved terminal stating that two plus two is four.",
    )
    try:
        result = await asyncio.to_thread(
            _rpc,
            endpoint,
            token,
            "resolution.open",
            {
                "operation_id": operation_id,
                "operation_payload_digest": intake["operation_payload_digest"],
                "activation_id": activation_id_for(
                    f"thread-{operation_id}", f"segment-{operation_id}", 1
                ),
                "lease_epoch": 1,
                "intake": intake,
            },
        )
        payload = result["result"]
        assert isinstance(payload, dict)
        assert payload["disposition"] == "terminal"
        exhaust = payload["exhaust"]
        assert isinstance(exhaust, dict)
        assert exhaust["kind"] == "terminal"
        terminal = exhaust["terminal"]
        assert isinstance(terminal, dict)
        assert terminal["status"] in {"resolved", "partial"}
        assert "four" in str(terminal["summary"]).lower() or "4" in str(terminal["summary"])
        inspection = await asyncio.to_thread(
            _rpc,
            endpoint,
            token,
            "resolution.inspect",
            {
                "operation_id": operation_id,
                "operation_payload_digest": intake["operation_payload_digest"],
            },
        )
        assert inspection["result"] == payload
    finally:
        await _stop_sidecar(process)


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_qwen27b_v2_resolution_round_trip_preserves_thread_and_terminal_contract(
    tmp_path: Path,
) -> None:
    """The configured local Qwen route preserves one V2 thread and terminal."""

    _require_live_backend()
    runtime = AgenticResolverRuntime.from_environment(data_root=tmp_path.resolve())
    authority = runtime.new_runtime_authority(
        objective_ref="v2-round-trip",
        brain_conversation_ref="chat:live:v2-round-trip",
        service_scope={
            "platform": "debug",
            "platform_channel_id": "live",
            "global_user_id": "live-user",
        },
        audience={"kind": "operator", "operation": "v2-round-trip"},
        interaction_issuer="kazusa-brain",
    )
    intake = AgenticResolverRuntime.build_intake(
        authority,
        objective="Return one grounded terminal resolution.",
        facts=["The resolver must use the V2 intake."],
    )
    result = await runtime.resolve(intake.to_dict())
    assert result.kind == "terminal"
    assert result.terminal is not None
    assert result.terminal.status in {"resolved", "partial"}
    assert result.identity is not None
    assert result.identity["resolution_thread_id"] == intake.resolution_thread_id
    assert result.identity["segment_id"] == intake.segment_id
    thread = await resolution_threads.get_thread(intake.resolution_thread_id)
    assert thread is not None
    assert thread["schema_version"] == "resolution_thread_store.v2"
    assert thread["brain_conversation_ref"] == intake.brain_conversation_ref
