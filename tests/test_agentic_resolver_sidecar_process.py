"""Black-box tests for the independent long-lived Node sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from agentic_resolver.contracts import DSHResolutionRuntimeV1
from agentic_resolver.controller import ResolutionController
from agentic_resolver.persistence import InMemoryResolutionThreadRepository
from agentic_resolver.rpc import DSHRpcClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ENTRY = PROJECT_ROOT / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
TOKEN = "sidecar-test-token"


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _rpc(url: str, method: str, params: dict[str, Any], *, token: str = TOKEN) -> dict[str, Any]:
    body = json.dumps({"jsonrpc": "2.0", "id": f"rpc-{time.time_ns()}", "method": method, "params": {"protocol_version": "kazusa.dsh-resolution-rpc.v1", **params}}).encode()
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
    env = os.environ.copy()
    env.update({
        "KAZUSA_DSH_SIDECAR_URL": url,
        "KAZUSA_DSH_RPC_TOKEN": TOKEN,
        "KAZUSA_DSH_DATA_ROOT": str(tmp_path.resolve()),
        "KAZUSA_DSH_MODEL": "test-model",
        "NODE_ENV": "test",
        "KAZUSA_DSH_TEST_MODEL_SCRIPT": json.dumps(script or [{"name": "submit_resolution", "arguments": {"status": "resolved", "summary": "done", "findings": [], "completed_subgoals": [], "remaining_needs": [], "clarification_request": None, "approval_request": None, "artifact_refs": [], "warnings": []}}]),
    })
    env.update(extra_env or {})
    process = subprocess.Popen(
        ["node", str(SIDECAR_ENTRY)], cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"sidecar exited: {stdout}\n{stderr}")
        try:
            _rpc(url, "system.health", {})
            return process, url
        except OSError:
            time.sleep(0.05)
    process.terminate()
    raise AssertionError("sidecar did not become healthy")


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _intake(operation_id: str, thread_id: str, segment_id: str) -> dict[str, Any]:
    return {
        "schema_version": "dsh_resolution_intake.v1", "mode": "start",
        "runtime": {
            "request_id": f"req-{operation_id}", "operation_id": operation_id,
            "operation_payload_digest": f"sha256:{operation_id}", "resolution_thread_id": thread_id,
            "segment_id": segment_id, "priority": "now",
            "soft_deadline_at": "2026-08-28T00:00:10Z", "hard_deadline_at": "2026-08-28T00:00:30Z",
            "max_model_steps": 3, "max_tool_calls": 3, "max_tool_bytes": 4096,
            "capability_token": "opaque", "scope_fingerprint": "sha256:scope",
            "audience_fingerprint": "sha256:audience", "resolver_profile_version": "kazusa-resolver-v1",
            "dsh_release": "0.1.1-rc.2", "session_store_epoch": "dsh-sqlite-0.1.1-rc.2-v1",
            "model_route": "test-model", "tool_catalog_digest": "sha256:catalog", "policy_epoch": "test-1",
        },
        "model_input": {"objective": "finish", "constraints": [], "success_criteria": [], "known_facts": [], "uncertainty": [], "literal_inputs": [], "continuation_delta": None, "prior_resolution_refs": [], "requested_evidence_quality": "normal", "notes": []},
    }


def _open(url: str, operation_id: str, thread_id: str, segment_id: str) -> dict[str, Any]:
    return _rpc(url, "resolution.open", {"operation_id": operation_id, "operation_payload_digest": f"sha256:{operation_id}", "activation_id": f"act-{operation_id}", "lease_epoch": 1, "intake": _intake(operation_id, thread_id, segment_id)})


def _continue(
    url: str,
    operation_id: str,
    thread_id: str,
    segment_id: str,
    *,
    lease_epoch: int,
) -> dict[str, Any]:
    intake = _intake(operation_id, thread_id, segment_id)
    intake["mode"] = "continue"
    return _rpc(
        url,
        "resolution.continue",
        {
            "operation_id": operation_id,
            "operation_payload_digest": f"sha256:{operation_id}",
            "activation_id": f"act-{operation_id}",
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
        assert _rpc(url, "system.health", {})["result"]["profile"] == "kazusa-resolver-v1"
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
            params = {"operation_id": "op_checkpoint", "operation_payload_digest": "sha256:checkpoint", "resolution_thread_id": "res_1", "segment_id": "seg_1", "activation_id": "act-op_1", "lease_epoch": 1}
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
        fields = {
            "scope_fingerprint": "sha256:old-scope",
            "audience_fingerprint": "sha256:old-audience",
            "resolver_profile_version": "old-profile",
            "dsh_release": "old-release",
            "session_store_epoch": "old-store",
            "model_route": "old-model",
            "tool_catalog_digest": "sha256:old-catalog",
            "policy_epoch": "old-policy",
        }
        for index, (field, old_value) in enumerate(fields.items()):
            intake = _intake(
                f"op_rotate_{index}",
                f"res_rotate_{index}",
                f"seg_rotate_{index}",
            )
            runtime = intake["runtime"]
            assert isinstance(runtime, dict)
            controller_runtime = deepcopy(runtime)
            repository = InMemoryResolutionThreadRepository()
            segment = ResolutionController._segment(
                DSHResolutionRuntimeV1.from_mapping(controller_runtime)
            )
            segment[field] = old_value
            repository.create_thread(
                str(runtime["resolution_thread_id"]),
                "brain-ref",
                "goal-ref",
                "now",
                str(segment["scope_fingerprint"]),
                str(segment["audience_fingerprint"]),
                segment,
                "2026-08-28T00:00:00Z",
            )
            client = DSHRpcClient(url, TOKEN)
            controller = ResolutionController(
                repository,
                client,
                owner_id="process-test",
            )
            intake["mode"] = "continue"
            intake["runtime"] = controller_runtime
            result = asyncio.run(controller.continue_resolution(intake))
            assert result["segment_id"] != segment["segment_id"]
            record = repository.get_thread(str(runtime["resolution_thread_id"]))
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
            params = {"operation_id": "op_checkpoint", "operation_payload_digest": "sha256:checkpoint", "resolution_thread_id": "res_1", "segment_id": "seg_1", "activation_id": "act-op_1", "lease_epoch": 1}
            assert _rpc(url, "resolution.request_checkpoint", params)["result"]["disposition"] == "checkpointed"
            assert pending.result(timeout=5)["result"]["disposition"] == "checkpointed"
    finally:
        _stop(process)


def test_zero_call_and_multi_call_steps_execute_no_tool_body_and_exhaust_correction_budget(tmp_path: Path) -> None:
    scripts = [
        [{"text": "prose"}, {"text": "still prose"}, {}],
        [
            {"calls": [{"name": "submit_resolution"}, {"name": "other"}]},
            {"calls": [{"name": "submit_resolution"}, {"name": "other"}]},
            {"calls": [{"name": "submit_resolution"}, {"name": "other"}]},
        ],
    ]
    for index, script in enumerate(scripts):
        data_root = tmp_path / str(index)
        process, url = _start(data_root, script)
        try:
            result = _open(
                url,
                f"op_invalid_{index}",
                f"res_invalid_{index}",
                f"seg_invalid_{index}",
            )["result"]
            assert result["exhaust"]["fault"]["code"] == (
                "RESOLVER_ACTION_CONTRACT_EXHAUSTED"
            )
            diagnostics = _rpc(url, "system.health", {})["result"][
                "diagnostics"
            ]
            assert diagnostics["terminal_tool_executions"] == 0
            assert diagnostics["correction_attempts"] == 2
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
    process, url = _start(
        tmp_path,
        extra_env={"KAZUSA_DSH_TEST_CORRUPT_TERMINAL_RECEIPT": "1"},
    )
    try:
        assert _open(url, "op_1", "res_1", "seg_1")["result"]["exhaust"]["kind"] == "runtime_fault"
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
