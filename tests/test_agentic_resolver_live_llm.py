"""One explicit real DSH sidecar and live model resolution case."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from agentic_resolver import AgenticResolverRuntime
from kazusa_ai_chatbot.db import resolution_threads


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
            "params": {
                "protocol_version": "kazusa.dsh-resolution-rpc.v1",
                **params,
            },
        }).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=5) as response:
        value = json.loads(response.read())
    assert isinstance(value, dict)
    return value


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_standalone_sidecar_resolution_reaches_submit_resolution(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/rpc"
    token = "live-dsh-resolution-token"
    environment = os.environ.copy()
    environment.update({
        "KAZUSA_DSH_SIDECAR_URL": endpoint,
        "KAZUSA_DSH_RPC_TOKEN": token,
        "KAZUSA_DSH_DATA_ROOT": str(tmp_path.resolve()),
        "KAZUSA_DSH_MODEL": "deepseek-v4-flash",
    })
    entry = Path(__file__).parents[1] / "sidecars" / "dsh_resolution" / "dist" / "src" / "main.js"
    process = await asyncio.create_subprocess_exec(
        "node", str(entry),
        cwd=Path(__file__).parents[1],
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        request = Request(
            endpoint,
            data=json.dumps({"jsonrpc": "2.0", "id": "health", "method": "system.health", "params": {"protocol_version": "kazusa.dsh-resolution-rpc.v1"}}).encode(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            await asyncio.to_thread(urlopen, request, timeout=2)
            break
        except OSError:
            if process.returncode is not None:
                stdout, stderr = await process.communicate()
                raise AssertionError(f"sidecar exited: {stdout}\n{stderr}")
            await asyncio.sleep(0.1)
    else:
        process.terminate()
        raise AssertionError("live sidecar did not become healthy")

    previous_endpoint = os.environ.get("KAZUSA_DSH_SIDECAR_URL")
    previous_token = os.environ.get("KAZUSA_DSH_RPC_TOKEN")
    previous_model = os.environ.get("KAZUSA_DSH_MODEL")
    os.environ["KAZUSA_DSH_SIDECAR_URL"] = endpoint
    os.environ["KAZUSA_DSH_RPC_TOKEN"] = token
    os.environ["KAZUSA_DSH_MODEL"] = "deepseek-v4-flash"
    runtime = AgenticResolverRuntime.from_environment(data_root=tmp_path)
    intake = {
        "schema_version": "dsh_resolution_intake.v1",
        "mode": "start",
        "runtime": runtime.new_runtime_authority(
            objective_ref="live-plan-1",
            scope={"kind": "live-test"},
            audience={"kind": "operator"},
        ),
        "model_input": {
            "objective": "Return a resolved submit_resolution stating that two plus two is four.",
            "constraints": ["Use exactly one submit_resolution action."],
            "success_criteria": ["The terminal summary states the answer four."],
            "known_facts": ["2 + 2 = 4"],
            "uncertainty": [],
            "literal_inputs": [],
            "continuation_delta": None,
            "prior_resolution_refs": [],
            "requested_evidence_quality": "normal",
            "notes": [],
        },
    }
    try:
        exhaust = await runtime.resolve(intake)
        assert exhaust.kind == "terminal"
        assert exhaust.terminal.status == "resolved"
        assert "four" in exhaust.terminal.summary.lower() or "4" in exhaust.terminal.summary
        authority = intake["runtime"]
        assert isinstance(authority, dict)
        inspection_frame = await asyncio.to_thread(
            _rpc,
            endpoint,
            token,
            "resolution.inspect",
            {
                "operation_id": authority["operation_id"],
                "operation_payload_digest": authority[
                    "operation_payload_digest"
                ],
            },
        )
        inspection = inspection_frame["result"]
        assert isinstance(inspection, dict)
        assert inspection["disposition"] == "terminal"
        assert inspection["exhaust"] == exhaust.to_dict()
        assert inspection["dsh_message_source_id"].startswith(
            "kazusa-operation:"
        )
        assert exhaust.last_committed_seq > 0

        health_frame = await asyncio.to_thread(
            _rpc, endpoint, token, "system.health", {}
        )
        health = health_frame["result"]
        assert isinstance(health, dict)
        diagnostics = health["diagnostics"]
        assert isinstance(diagnostics, dict)
        assert diagnostics["live_activations"] == 0

        thread = await resolution_threads.get_thread(
            str(authority["resolution_thread_id"])
        )
        assert thread is not None
        assert thread["current_lease"] is None
        operation = next(
            item
            for item in thread["operations"]
            if item["operation_id"] == authority["operation_id"]
        )
        assert operation["disposition"] == "terminal"
        assert operation["last_committed_seq"] == exhaust.last_committed_seq
        segment = next(
            item
            for item in thread["segments"]
            if item["segment_id"] == thread["current_segment_id"]
        )
        assert segment["last_committed_seq"] == exhaust.last_committed_seq
        assert segment["state"] == "terminal"
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=10)
        if previous_endpoint is None:
            os.environ.pop("KAZUSA_DSH_SIDECAR_URL", None)
        else:
            os.environ["KAZUSA_DSH_SIDECAR_URL"] = previous_endpoint
        if previous_token is None:
            os.environ.pop("KAZUSA_DSH_RPC_TOKEN", None)
        else:
            os.environ["KAZUSA_DSH_RPC_TOKEN"] = previous_token
        if previous_model is None:
            os.environ.pop("KAZUSA_DSH_MODEL", None)
        else:
            os.environ["KAZUSA_DSH_MODEL"] = previous_model
