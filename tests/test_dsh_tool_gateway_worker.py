"""Semantic dispatcher and worker framing tests."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def test_dispatch_exposes_only_approved_semantic_services_and_routes_no_standard_or_graph_capability() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import SEMANTIC_TOOL_NAMES
    from kazusa_ai_chatbot.dsh_tool_gateway.dispatch import SemanticCapabilityDispatcher

    assert len(SEMANTIC_TOOL_NAMES) == 13
    assert all(name.startswith("kazusa_") for name in SEMANTIC_TOOL_NAMES)
    assert not any(name in {"read_file", "shell", "submit_resolution"} for name in SEMANTIC_TOOL_NAMES)
    assert SemanticCapabilityDispatcher


@pytest.mark.asyncio
async def test_worker_replays_committed_idempotent_results_and_preserves_uncertain_mutation_state() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.worker import (
        SemanticWorker,
        SQLiteSemanticOutcomeOwner,
    )

    calls = []

    async def handler(call):
        calls.append(call.call_id)
        return {"value": "committed"}

    owner_path = Path.cwd() / ".dsh-debug" / "worker-outcomes-test.sqlite"
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    if owner_path.exists():
        owner_path.unlink()
    first_worker = SemanticWorker(
        handler=handler,
        outcome_owner=SQLiteSemanticOutcomeOwner(owner_path),
    )
    first = await first_worker.handle_mapping({
        "call_id": "call-1",
        "payload": {
            "operation": "kazusa_remember_information",
            "idempotency_key": "idem-1",
            "arguments": {"information": "stable"},
        },
    })
    second_worker = SemanticWorker(
        handler=handler,
        outcome_owner=SQLiteSemanticOutcomeOwner(owner_path),
    )
    second = await second_worker.handle_mapping({
        "call_id": "call-2",
        "payload": {
            "operation": "kazusa_remember_information",
            "idempotency_key": "idem-1",
            "arguments": {"information": "stable"},
        },
    })
    assert first == second
    assert calls == ["call-1"]

    async def uncertain_handler(call):
        calls.append(call.call_id)
        raise RuntimeError("simulated worker loss")

    uncertain_worker = SemanticWorker(
        handler=uncertain_handler,
        outcome_owner=SQLiteSemanticOutcomeOwner(owner_path),
    )
    uncertain = await uncertain_worker.handle_mapping({
        "call_id": "call-3",
        "payload": {
            "operation": "kazusa_revise_memory",
            "idempotency_key": "idem-uncertain",
            "arguments": {"memory_ref": "ref"},
        },
    })
    assert uncertain["status"] == "unavailable"
    replay_after_uncertain = await SemanticWorker(
        handler=handler,
        outcome_owner=SQLiteSemanticOutcomeOwner(owner_path),
    ).handle_mapping({
        "call_id": "call-4",
        "payload": {
            "operation": "kazusa_revise_memory",
            "idempotency_key": "idem-uncertain",
            "arguments": {"memory_ref": "ref"},
        },
    })
    assert replay_after_uncertain["error"]["code"] == "OPERATION_OUTCOME_UNCERTAIN"


@pytest.mark.asyncio
async def test_worker_authenticates_before_replay_lookup_and_denies_tampered_committed_retry(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        authenticate_semantic_call,
        issue_semantic_call,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.worker import (
        SemanticWorker,
        SQLiteSemanticOutcomeOwner,
        _parse_signed_call,
    )
    from tests.test_dsh_tool_gateway_authority import _authority

    secret = b"worker-replay-secret"
    authority = _authority()
    call = issue_semantic_call(
        authority,
        operation="kazusa_search_memories",
        arguments={"query": "document MongoDB", "max_results": 1},
        secret=secret,
        call_id="signed-call-1",
        now=authority.issued_at,
    )
    owner = SQLiteSemanticOutcomeOwner(tmp_path / "outcomes.sqlite")
    executions: list[str] = []

    def authenticate(payload):
        return authenticate_semantic_call(
            _parse_signed_call(payload),
            secret=secret,
            now=authority.issued_at,
        )

    def consume(payload):
        signed = authenticate(payload)
        owner.consume(signed.authority, signed.call_id)

    async def handler(worker_call):
        executions.append(worker_call.call_id)
        return {"value": "durable-result"}

    worker = SemanticWorker(
        handler=handler,
        outcome_owner=owner,
        preflight=authenticate,
        replay_consumer=consume,
    )
    frame = {"call_id": call.call_id, "payload": call.to_dict()}
    first = await worker.handle_mapping(frame)
    replay = await worker.handle_mapping(frame)
    assert first == {"value": "durable-result"}
    assert replay == first
    assert executions == [call.call_id]

    tampered_payload = call.to_dict()
    tampered_payload["signature"] = "0" * len(call.signature)
    tampered = await worker.handle_mapping({
        "call_id": call.call_id,
        "payload": tampered_payload,
    })
    assert tampered["status"] == "denied"
    assert tampered["error"]["code"] == "SEMANTIC_AUTHORITY_INVALID"
    assert executions == [call.call_id]

    missing_call = issue_semantic_call(
        authority,
        operation="kazusa_search_memories",
        arguments={"query": "missing outcome", "max_results": 1},
        secret=secret,
        call_id="signed-call-missing-outcome",
        now=authority.issued_at,
    )
    owner.consume(missing_call.authority, missing_call.call_id)
    duplicate_missing = await worker.handle_mapping({
        "call_id": missing_call.call_id,
        "payload": missing_call.to_dict(),
    })
    assert duplicate_missing["status"] == "denied"
    assert duplicate_missing["error"]["code"] == "SEMANTIC_AUTHORITY_REPLAYED"
    assert executions == [call.call_id]


def test_stdio_worker_health_control_frame_is_bounded_and_hidden_from_semantic_catalog(
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import SEMANTIC_TOOL_NAMES

    outcome_path = tmp_path / "health-outcomes.sqlite"
    request = json.dumps(
        {"control": "health", "request_id": "health-test"},
        separators=(",", ":"),
    ).encode("utf-8")
    environment = {
        "PYTHONPATH": str(Path.cwd()),
        "PYTHONUNBUFFERED": "1",
        "KAZUSA_DSH_TOOL_GATEWAY_SECRET": "health-secret",
        "KAZUSA_DSH_SEMANTIC_OUTCOME_PATH": str(outcome_path),
    }
    process = subprocess.Popen(
        [
            str(Path("venv") / "Scripts" / "python.exe"),
            "-u",
            "-m",
            "kazusa_ai_chatbot.dsh_tool_gateway.worker",
        ],
        cwd=Path.cwd(),
        env={**os.environ, **environment},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(struct.pack(">I", len(request)) + request)
        process.stdin.flush()
        header = process.stdout.read(4)
        assert len(header) == 4
        length = struct.unpack(">I", header)[0]
        assert 0 < length <= 32 * 1024
        response = json.loads(process.stdout.read(length).decode("utf-8"))
        assert response == {
            "control": "health",
            "request_id": "health-test",
            "schema_version": "kazusa_semantic_worker_health.v1",
            "status": "ready",
            "protocol": "length-prefixed-json",
        }
        assert "health" not in SEMANTIC_TOOL_NAMES
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_direct_python_worker_process_round_trip_preserves_authority_result_and_evidence() -> None:
    """Signed semantic frames cross the direct Python worker boundary."""
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        SemanticActivationAuthorityV1,
        activation_id_for,
        issue_semantic_call,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.catalog import semantic_catalog_digest
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    secret = b"real-worker-process-secret"
    now = datetime.now(UTC)
    issued_at = (now - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    expires_at = (now + timedelta(minutes=4)).isoformat().replace("+00:00", "Z")
    scope = {
        "platform": "debug",
        "platform_channel_id": "worker-channel",
        "global_user_id": "worker-user",
    }
    workspace = str(Path.cwd().resolve()).replace("\\", "/")
    authority = SemanticActivationAuthorityV1(
        activation_id=activation_id_for("worker-thread", "worker-segment", 1),
        lease_epoch=1,
        resolution_thread_id="worker-thread",
        segment_id="worker-segment",
        brain_conversation_ref="brain-worker",
        service_scope=scope,
        scope_fingerprint=content_digest(scope),
        audience_fingerprint="sha256:worker-audience",
        workspace_root=workspace,
        route_digest="sha256:worker-route",
        catalog_digest=semantic_catalog_digest(),
        profile_version="kazusa-resolver-standard-v2",
        model_route_digest="sha256:worker-route",
        workspace_fingerprint=content_digest({"workspace_root": workspace}),
        issued_reference_digest="sha256:worker-refs",
        policy_epoch="dsh-standard-policy-v2",
        interaction_issuer="brain-worker",
        issued_at=issued_at,
        expires_at=expires_at,
        token_id="worker-token",
        nonce="worker-nonce",
    )
    call = issue_semantic_call(
        authority,
        operation="kazusa_recall_active_context",
        arguments={"kinds": ["commitments"], "max_results": 1},
        secret=secret,
        call_id="worker-call-1",
        now=now.isoformat().replace("+00:00", "Z"),
    )
    frame = json.dumps(
        {"call_id": call.call_id, "payload": call.to_dict()},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    data_root = Path.cwd() / ".dsh-debug"
    data_root.mkdir(parents=True, exist_ok=True)
    outcome_path = data_root / "real-worker-outcomes.sqlite"
    if outcome_path.exists():
        outcome_path.unlink()
    environment = {
        "PYTHONPATH": str(Path.cwd()),
        "PYTHONUNBUFFERED": "1",
        "KAZUSA_DSH_TOOL_GATEWAY_SECRET": secret.decode("ascii"),
        "KAZUSA_DSH_SEMANTIC_OUTCOME_PATH": str(outcome_path),
    }

    def run_once() -> dict:
        process = subprocess.Popen(
            [
                str(Path("venv") / "Scripts" / "python.exe"),
                "-u",
                "-m",
                "kazusa_ai_chatbot.dsh_tool_gateway.worker",
            ],
            cwd=Path.cwd(),
            env={**os.environ, **environment},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(struct.pack(">I", len(frame)) + frame)
            process.stdin.flush()
            header = process.stdout.read(4)
            assert len(header) == 4
            length = struct.unpack(">I", header)[0]
            payload = process.stdout.read(length)
            assert len(payload) == length
            return json.loads(payload.decode("utf-8"))
        finally:
            process.terminate()
            process.wait(timeout=5)

    first = run_once()
    second = run_once()
    assert first == second
    assert first["schema_version"] == "kazusa_semantic_capability_result.v1"
    assert first["status"] == "empty"


def test_real_sidecar_worker_round_trip_preserves_authority_result_and_evidence(
    tmp_path: Path,
) -> None:
    """The compiled Node sidecar drives a signed call through the real worker."""
    from urllib.error import HTTPError

    from kazusa_ai_chatbot.dsh_tool_gateway.authority import activation_id_for
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest
    from tests.test_agentic_resolver_sidecar_process import (
        _ROUTE_BASES,
        _intake,
        _open,
        _route_digest,
        _rpc,
        _start,
        _stop,
    )

    terminal = {
        "status": "resolved",
        "summary": "semantic worker completed",
        "findings": [],
        "completed_subgoals": ["semantic worker round trip"],
        "remaining_needs": [],
        "clarification_request": None,
        "approval_request": None,
        "artifact_refs": [],
        "warnings": [],
    }
    process, url = _start(
        tmp_path,
        [
            {
                "name": "kazusa_recall_active_context",
                "arguments": {"kinds": ["history"], "max_results": 1},
            },
            {"name": "submit_resolution", "arguments": terminal},
        ],
    )
    try:
        first = _open(url, "op-worker-sidecar", "thread-worker-sidecar", "segment-worker-sidecar")
        result = first["result"]
        assert result["disposition"] == "terminal"
        assert "exhaust" in result, result
        fake_openai = process._fake_openai  # type: ignore[attr-defined]
        assert fake_openai.calls >= 2
        assert len(fake_openai.requests) >= 2
        assert "kazusa_semantic_capability_result.v1" in json.dumps(
            fake_openai.requests[1],
            ensure_ascii=False,
        )
        exhaust = result["exhaust"]
        assert exhaust["kind"] == "terminal"
        assert isinstance(exhaust["evidence"], list)
        identity = exhaust["identity"]
        assert identity["resolution_thread_id"] == "thread-worker-sidecar"
        assert identity["segment_id"] == "segment-worker-sidecar"
        assert identity["scope_fingerprint"] == content_digest({
            "platform": "debug",
            "platform_channel_id": "sidecar-process",
            "global_user_id": "user",
        })

        tampered = _intake(
            "op-worker-sidecar-tampered",
            "thread-worker-sidecar-tampered",
            "segment-worker-sidecar-tampered",
            route_digest=_route_digest(_ROUTE_BASES[url]),
        )
        semantic = tampered["semantic_tool_authority"]
        token = str(semantic["token"])
        semantic["token"] = f"{token[:-1]}{'0' if token[-1] != '0' else '1'}"
        with pytest.raises(HTTPError):
            _rpc(url, "resolution.open", {
                "operation_id": "op-worker-sidecar-tampered",
                "operation_payload_digest": "sha256:op-worker-sidecar-tampered",
                "activation_id": activation_id_for(
                    "thread-worker-sidecar-tampered",
                    "segment-worker-sidecar-tampered",
                    1,
                ),
                "lease_epoch": 1,
                "intake": tampered,
            })
    finally:
        _stop(process)
