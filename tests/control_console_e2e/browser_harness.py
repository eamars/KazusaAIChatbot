from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


CONSOLE_STARTUP_TIMEOUT_SECONDS = 15.0
CONSOLE_SHUTDOWN_TIMEOUT_SECONDS = 5.0
HTTP_TIMEOUT_SECONDS = 5.0
DEFAULT_E2E_OPERATOR_TOKEN = "secret"

CHROME_CANDIDATE_PATHS = (
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
)


@dataclass(frozen=True)
class E2EConsoleConfig:
    """Settings required to launch an isolated control-console process."""

    port: int
    state_dir: Path
    operator_token_hash: str
    brain_base_url: str
    artifact_dir: Path
    service_registry_path: Path | None = None
    sse_interval_seconds: float = 2.0


class E2EConsoleProcess(AbstractContextManager["E2EConsoleProcess"]):
    """Manage one isolated control-console process for E2E tests."""

    def __init__(self, config: E2EConsoleConfig) -> None:
        """Create a console process manager without starting the process."""

        self.config = config
        self.base_url = f"http://127.0.0.1:{config.port}"
        self.stdout_path = config.artifact_dir / "console.stdout.log"
        self.stderr_path = config.artifact_dir / "console.stderr.log"
        self.launcher_path = config.artifact_dir / "console_launcher.py"
        self._process: subprocess.Popen[str] | None = None
        self._stdout_file: TextIO | None = None
        self._stderr_file: TextIO | None = None

    def __enter__(self) -> "E2EConsoleProcess":
        """Start the console and wait until its TCP port accepts requests."""

        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_launcher()
        self._stdout_file = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_file = self.stderr_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, str(self.launcher_path)],
            cwd=Path.cwd(),
            stdout=self._stdout_file,
            stderr=self._stderr_file,
            text=True,
        )
        self._wait_until_listening()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Stop the console process and leave logs in the artifact directory."""

        del exc_type, exc, traceback
        process = self._process
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=CONSOLE_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=CONSOLE_SHUTDOWN_TIMEOUT_SECONDS)
        if self._stdout_file is not None:
            self._stdout_file.close()
            self._stdout_file = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict | None = None,
    ) -> dict:
        """Send one JSON request to the isolated console."""

        data = None
        headers = {"accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=HTTP_TIMEOUT_SECONDS,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            raise AssertionError(
                f"request failed: {method} {path} status={exc.code} body={body}"
            ) from exc
        payload_dict = json.loads(body)
        return payload_dict

    def _write_launcher(self) -> None:
        """Write a launcher that bypasses environment-backed settings."""

        launcher_source = (
            "from pathlib import Path\n"
            "import uvicorn\n"
            "from control_console.app import create_app\n"
            "from control_console.settings import ControlConsoleSettings\n"
            f"settings = ControlConsoleSettings(\n"
            f"    host='127.0.0.1',\n"
            f"    port={self.config.port},\n"
            f"    operator_token_hash={self.config.operator_token_hash!r},\n"
            f"    state_dir=Path({str(self.config.state_dir)!r}),\n"
            f"    service_registry_path={self._launcher_path_literal(self.config.service_registry_path)},\n"
            f"    brain_base_url={self.config.brain_base_url!r},\n"
            f"    sse_interval_seconds={self.config.sse_interval_seconds!r},\n"
            f")\n"
            "app = create_app(settings=settings)\n"
            "uvicorn.run(\n"
            "    app,\n"
            "    host='127.0.0.1',\n"
            f"    port={self.config.port},\n"
            "    timeout_graceful_shutdown=1,\n"
            ")\n"
        )
        self.launcher_path.write_text(launcher_source, encoding="utf-8")

    @staticmethod
    def _launcher_path_literal(path: Path | None) -> str:
        """Return a launcher-safe literal for an optional path."""

        if path is None:
            return "None"
        return f"Path({str(path)!r})"

    def _wait_until_listening(self) -> None:
        """Wait for the console port or raise with captured stderr preview."""

        deadline = time.monotonic() + CONSOLE_STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                break
            if _can_connect("127.0.0.1", self.config.port):
                return
            time.sleep(0.1)
        stderr_preview = _file_preview(self.stderr_path)
        raise RuntimeError(
            f"console did not start on port {self.config.port}: {stderr_preview}"
        )


class BrowserSession(AbstractContextManager[Any]):
    """Manage one headless browser page for console E2E tests."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        viewport_width: int = 1600,
        viewport_height: int = 900,
    ) -> None:
        """Create a browser session without launching Chrome yet."""

        self.artifact_dir = artifact_dir
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.console_messages: list[str] = []
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    def __enter__(self) -> Any:
        """Launch the existing Chrome binary and return a Playwright page."""

        from playwright.sync_api import sync_playwright

        chrome_path = find_chrome_executable()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            executable_path=str(chrome_path),
            headless=True,
        )
        self._context = self._browser.new_context(
            viewport={
                "width": self.viewport_width,
                "height": self.viewport_height,
            }
        )
        self._page = self._context.new_page()
        self._page.on("console", self._record_console_message)
        self._page.on("pageerror", self._record_page_error)
        return self._page

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the browser and write captured console diagnostics."""

        del exc_type, exc, traceback
        if self.console_messages:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            console_path = self.artifact_dir / "browser.console.log"
            console_path.write_text(
                "\n".join(self.console_messages),
                encoding="utf-8",
            )
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def _record_console_message(self, message: Any) -> None:
        """Record browser warnings and errors for test diagnostics."""

        if message.type not in {"warning", "error"}:
            return
        self.console_messages.append(f"{message.type}: {message.text}")

    def _record_page_error(self, error: Any) -> None:
        """Record uncaught page errors for test diagnostics."""

        self.console_messages.append(f"pageerror: {error}")


def find_chrome_executable() -> Path:
    """Return a locally installed Chrome-compatible browser executable."""

    for candidate_path in CHROME_CANDIDATE_PATHS:
        if candidate_path.exists():
            return candidate_path
    raise RuntimeError("Chrome or Edge executable was not found for E2E tests")


def write_summary(
    *,
    artifact_dir: Path,
    name: str,
    conclusion: str,
    details: dict,
) -> Path:
    """Write one concise JSON result summary for a test slice."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / f"{name}.summary.json"
    summary = {
        "name": name,
        "conclusion": conclusion,
        "details": details,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _can_connect(host: str, port: int) -> bool:
    """Return whether a TCP connection to the host and port succeeds."""

    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _file_preview(path: Path, *, max_chars: int = 1000) -> str:
    """Return a bounded file preview for startup diagnostics."""

    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    preview = text[-max_chars:]
    return preview
