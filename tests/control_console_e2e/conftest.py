from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from control_console.auth import hash_operator_token

from browser_harness import (
    DEFAULT_E2E_OPERATOR_TOKEN,
    BrowserSession,
    E2EConsoleConfig,
    E2EConsoleProcess,
    write_summary,
)


@pytest.fixture
def e2e_artifact_dir(tmp_path: Path) -> Path:
    """Return the artifact directory for one E2E test."""

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


@pytest.fixture
def e2e_summary_writer(
    e2e_artifact_dir: Path,
) -> Callable[..., Path]:
    """Return a helper that writes concise JSON test summaries."""

    def write_e2e_summary(
        *,
        name: str,
        conclusion: str,
        details: dict,
    ) -> Path:
        summary_path = write_summary(
            artifact_dir=e2e_artifact_dir,
            name=name,
            conclusion=conclusion,
            details=details,
        )
        return summary_path

    return write_e2e_summary


@pytest.fixture
def e2e_console(
    tmp_path: Path,
    unused_tcp_port: int,
    e2e_artifact_dir: Path,
) -> Callable[[], E2EConsoleProcess]:
    """Return a factory for isolated control-console processes."""

    def build_console(
        *,
        brain_base_url: str = "http://127.0.0.1:9",
        service_registry_path: Path | None = None,
        sse_interval_seconds: float = 2.0,
    ) -> E2EConsoleProcess:
        config = E2EConsoleConfig(
            port=unused_tcp_port,
            state_dir=tmp_path / "console_state",
            operator_token_hash=hash_operator_token(DEFAULT_E2E_OPERATOR_TOKEN),
            brain_base_url=brain_base_url,
            artifact_dir=e2e_artifact_dir,
            service_registry_path=service_registry_path,
            sse_interval_seconds=sse_interval_seconds,
        )
        console = E2EConsoleProcess(config)
        return console

    return build_console


@pytest.fixture
def e2e_browser_page(
    e2e_artifact_dir: Path,
) -> Callable[[str], object]:
    """Return a helper that opens the console in a browser page."""

    sessions: list[BrowserSession] = []

    def open_console_page(base_url: str) -> object:
        session = BrowserSession(artifact_dir=e2e_artifact_dir)
        page = session.__enter__()
        sessions.append(session)
        page.goto(base_url, wait_until="domcontentloaded")
        return page

    yield open_console_page

    for session in reversed(sessions):
        session.__exit__(None, None, None)
