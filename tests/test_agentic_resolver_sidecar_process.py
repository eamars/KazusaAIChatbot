"""Focused black-box tests for the built DSH sidecar process."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import pytest

from experiments.dsh_runtime_probe import (
    RPC_TOKEN,
    ProbeRecorder,
    continue_resolution,
    open_resolution,
    rpc_call,
    start_sidecar,
)
from kazusa_ai_chatbot.dsh_tool_gateway.authority import activation_id_for


def _recorder(tmp_path: Path, name: str) -> ProbeRecorder:
    """Create one process owner rooted in the pytest temporary directory."""

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return ProbeRecorder(probe_name=name, artifact_dir=artifact_dir)


def test_standalone_runtime_uses_one_long_lived_sidecar_across_two_resolves(
    tmp_path: Path,
) -> None:
    """One healthy process serves independent resolution sessions."""

    recorder = _recorder(tmp_path, "long-lived-sidecar")
    sidecar = start_sidecar(tmp_path / "data", recorder, name="sidecar")
    try:
        first = open_resolution(sidecar, "op-1", "thread-1", "segment-1")
        second = open_resolution(sidecar, "op-2", "thread-2", "segment-2")
        assert first["result"]["exhaust"]["kind"] == "terminal"
        assert second["result"]["exhaust"]["kind"] == "terminal"
        assert sidecar.process.poll() is None
    finally:
        sidecar.stop()


def test_real_sidecar_worker_accepts_python_semantic_call_contract(
    tmp_path: Path,
) -> None:
    """The TypeScript gateway and Python worker share one signed call schema."""

    recorder = _recorder(tmp_path, "semantic-worker")
    sidecar = start_sidecar(
        tmp_path / "data",
        recorder,
        name="sidecar",
        script=[
            {
                "name": "kazusa_recall_active_context",
                "arguments": {"kinds": ["history"], "max_results": 1},
            },
            {},
        ],
    )
    try:
        response = open_resolution(
            sidecar,
            "op-semantic",
            "thread-semantic",
            "segment-semantic",
        )
        rendered = json.dumps(sidecar.provider.requests, ensure_ascii=False)
        assert response["result"]["exhaust"]["kind"] == "terminal"
        assert "kazusa_semantic_capability_result.v1" in rendered
        assert "SEMANTIC_AUTHORITY_INVALID" not in rendered
    finally:
        sidecar.stop()


def test_sidecar_requires_loopback_auth_data_root_model_and_versioned_store_path(
    tmp_path: Path,
) -> None:
    """Authenticated readiness exposes the public runtime identity."""

    recorder = _recorder(tmp_path, "sidecar-readiness")
    sidecar = start_sidecar(tmp_path / "data", recorder, name="sidecar")
    try:
        health = rpc_call(sidecar.url, "system.health", {})["result"]
        assert health["store_path"].endswith(
            "dsh/0.1.1-rc.2/sessions.sqlite",
        )
        assert health["loopback"] is True
        assert health["dsh_runtime"] is True
        assert health["profile"] == "kazusa-resolver-standard-v2"
    finally:
        sidecar.stop()


def test_sidecar_restart_preserves_checkpoint_and_cold_resumes(
    tmp_path: Path,
) -> None:
    """A committed checkpoint resumes in the same persisted DSH session."""

    recorder = _recorder(tmp_path, "checkpoint-restart")
    data_root = tmp_path / "data"
    first = start_sidecar(
        data_root,
        recorder,
        name="sidecar-before-restart",
        script=[{"wait": True}],
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                open_resolution,
                first,
                "op-checkpoint",
                "thread-checkpoint",
                "segment-checkpoint",
            )
            time.sleep(0.2)
            checkpoint = rpc_call(
                first.url,
                "resolution.request_checkpoint",
                {
                    "operation_id": "op-checkpoint-request",
                    "operation_payload_digest": "sha256:checkpoint-request",
                    "resolution_thread_id": "thread-checkpoint",
                    "segment_id": "segment-checkpoint",
                    "activation_id": activation_id_for(
                        "thread-checkpoint",
                        "segment-checkpoint",
                        1,
                    ),
                    "lease_epoch": 1,
                },
            )["result"]
            opened = pending.result(timeout=5)["result"]
        assert checkpoint["disposition"] == "checkpointed"
    finally:
        first.stop()

    resumed = start_sidecar(
        data_root,
        recorder,
        name="sidecar-after-restart",
    )
    try:
        inspected = rpc_call(
            resumed.url,
            "resolution.inspect",
            {
                "operation_id": "op-checkpoint",
                "operation_payload_digest": "sha256:op-checkpoint",
            },
        )["result"]
        terminal = continue_resolution(
            resumed,
            "op-continue",
            "thread-checkpoint",
            "segment-checkpoint",
            lease_epoch=2,
        )["result"]
        assert inspected["disposition"] == "checkpointed"
        assert terminal["disposition"] == "terminal"
        assert terminal["session_id"] == opened["session_id"]
    finally:
        resumed.stop()


def test_kill_after_terminal_commit_before_http_response_replays_exact_exhaust(
    tmp_path: Path,
) -> None:
    """A response-loss restart replays the exact committed terminal exhaust."""

    recorder = _recorder(tmp_path, "terminal-replay")
    data_root = tmp_path / "data"
    crashing = start_sidecar(
        data_root,
        recorder,
        name="sidecar-crashing",
        extra_env={"KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT": "1"},
    )
    try:
        with pytest.raises(OSError):
            open_resolution(
                crashing,
                "op-replay",
                "thread-replay",
                "segment-replay",
            )
        crashing.wait_for_exit()
    finally:
        crashing.stop()

    replaying = start_sidecar(
        data_root,
        recorder,
        name="sidecar-replaying",
    )
    try:
        inspected = rpc_call(
            replaying.url,
            "resolution.inspect",
            {
                "operation_id": "op-replay",
                "operation_payload_digest": "sha256:op-replay",
            },
        )["result"]
        replayed = open_resolution(
            replaying,
            "op-replay",
            "thread-replay",
            "segment-replay",
        )["result"]
        assert inspected["disposition"] == "terminal"
        assert replayed["exhaust"] == inspected["exhaust"]
    finally:
        replaying.stop()


def test_bad_rpc_authentication_fails_closed(tmp_path: Path) -> None:
    """A request with the wrong loopback token receives no health result."""

    recorder = _recorder(tmp_path, "rpc-auth")
    sidecar = start_sidecar(tmp_path / "data", recorder, name="sidecar")
    try:
        with pytest.raises(HTTPError) as denied:
            rpc_call(
                sidecar.url,
                "system.health",
                {},
                token=RPC_TOKEN + "-wrong",
            )
        assert denied.value.code == 401
    finally:
        sidecar.stop()
