"""Shared sidecar launch, readiness, HTTP, and owned-resource cleanup."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ENTRY = PROJECT_ROOT / 'sidecars' / 'dsh_resolution' / 'dist' / 'src' / 'main.js'
RPC_TOKEN = 'dsh-probe-rpc-token'
PROCESS_EXIT_TIMEOUT_SECONDS = 15.0

class ProbeFailure(RuntimeError):
    """Raised when an observed runtime result violates a probe contract."""


class ProbeBlocked(RuntimeError):
    """Raised when an external prerequisite is unavailable."""


def _free_port() -> int:
    """Reserve and release one loopback port for a child process."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass
class SidecarProcess:
    """Owned sidecar process and optional loopback dependency resources."""

    process: subprocess.Popen[str]
    url: str
    provider: Any
    brain: Any
    data_root: Path
    recorder: Any
    name: str
    process_row: dict[str, object]
    stopped: bool = False

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
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        exc.add_note(f"RPC {method}: {detail}")
        raise
    with response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ProbeFailure(f"RPC {method} returned a non-object")
    return value


def start_configured_sidecar(
    recorder: Any,
    *,
    name: str,
    environment: Mapping[str, str],
    require_ready: bool = True,
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
            if isinstance(health, dict) and (health.get("status") == "ready" or not require_ready):
                return harness
        except OSError:
            pass
        time.sleep(0.05)
    harness.stop()
    raise ProbeFailure(f"sidecar {name} did not become ready")

