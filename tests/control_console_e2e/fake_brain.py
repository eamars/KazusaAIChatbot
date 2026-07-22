from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
import json
import time


class FakeBrainServer(AbstractContextManager["FakeBrainServer"]):
    """Threaded fake brain HTTP server for console E2E tests."""

    def __init__(self, port: int) -> None:
        """Create a fake brain bound to an explicit test port."""

        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._lock = Lock()
        self._graph = graph_snapshot(status="not_reported", run_id="not-reported")
        self._chat_requests: list[dict[str, Any]] = []
        self._chat_status_code = 200
        self._chat_delay_seconds = 0.0
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> "FakeBrainServer":
        """Start the fake brain server in a background thread."""

        handler_class = self._handler_class()
        self._server = QuietThreadingHTTPServer(
            ("127.0.0.1", self.port),
            handler_class,
        )
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Stop the fake brain server."""

        del exc_type, exc, traceback
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def set_graph(self, graph: dict[str, Any]) -> None:
        """Replace the latest graph returned by the fake brain."""

        with self._lock:
            self._graph = graph

    def latest_graph(self) -> dict[str, Any]:
        """Return a copy of the latest graph."""

        with self._lock:
            graph = dict(self._graph)
        return graph

    def chat_requests(self) -> list[dict[str, Any]]:
        """Return recorded brain chat request payloads."""

        with self._lock:
            requests = list(self._chat_requests)
        return requests

    def record_chat_request(self, payload: dict[str, Any]) -> None:
        """Record one chat request payload."""

        with self._lock:
            self._chat_requests.append(payload)

    def set_chat_status_code(self, status_code: int) -> None:
        """Set the HTTP status returned by `/chat`."""

        with self._lock:
            self._chat_status_code = status_code

    def set_chat_delay_seconds(self, delay_seconds: float) -> None:
        """Delay `/chat` responses to expose in-flight browser UI state."""

        with self._lock:
            self._chat_delay_seconds = delay_seconds

    def chat_status_code(self) -> int:
        """Return the current `/chat` status code."""

        with self._lock:
            status_code = self._chat_status_code
        return status_code

    def chat_delay_seconds(self) -> float:
        """Return the current `/chat` response delay."""

        with self._lock:
            delay_seconds = self._chat_delay_seconds
        return delay_seconds

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        """Build a request handler bound to this fake brain."""

        owner = self

        class FakeBrainHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                """Handle fake brain GET endpoints."""

                routes: dict[str, Callable[[], dict[str, Any]]] = {
                    "/health": owner._health_payload,
                    "/ops/runtime-status": owner._runtime_payload,
                    "/ops/latest-cognition-graph": owner._latest_graph_payload,
                }
                payload_factory = routes.get(self.path)
                if payload_factory is None:
                    self.send_error(404)
                    return
                _write_json(self, payload_factory())

            def do_POST(self) -> None:
                """Handle fake brain chat endpoint."""

                if self.path != "/chat":
                    self.send_error(404)
                    return
                status_code = owner.chat_status_code()
                if status_code != 200:
                    self.send_error(status_code)
                    return
                delay_seconds = owner.chat_delay_seconds()
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                content_length = int(self.headers.get("content-length", "0"))
                payload: dict[str, Any] = {}
                if content_length:
                    body = self.rfile.read(content_length)
                    payload = json.loads(body.decode("utf-8"))
                owner.record_chat_request(payload)
                graph = graph_snapshot(status="completed", run_id="debug-run-1")
                owner.set_graph(graph)
                _write_json(
                    self,
                    {
                        "delivery_tracking_id": "debug-run-1",
                        "messages": [{"text": "fake brain reply"}],
                        "cognition_graph": graph,
                    },
                )

            def log_message(self, format: str, *args: Any) -> None:
                """Silence default request logging."""

                del format, args

        return FakeBrainHandler

    def _health_payload(self) -> dict[str, Any]:
        """Return a healthy brain payload."""

        return {
            "status": "healthy",
            "cache2": {
                "status": "healthy",
                "hit_rate": 1.0,
            },
        }

    def _runtime_payload(self) -> dict[str, Any]:
        """Return a minimal runtime-status payload."""

        return {
            "status": "running",
            "workers": {
                "self_cognition": "idle",
            },
        }

    def _latest_graph_payload(self) -> dict[str, Any]:
        """Return latest cognition graph payload."""

        return {
            "cognition_graph": self.latest_graph(),
        }


def graph_snapshot(
    *,
    status: str,
    run_id: str,
    trigger_source: str = "user_message",
    input_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Return a source-bearing cognition graph snapshot with parallel branches."""

    active_input_sources = input_sources or ["dialog_text"]

    if status == "not_reported":
        return {
            "status": "not_reported",
            "run_id": run_id,
            "trigger_source": "not_reported",
            "input_sources": [],
            "nodes": [],
            "edges": [],
        }

    node_status = "running" if status == "running" else "completed"
    if status == "failed":
        node_status = "failed"
    long_text = "input-start\n" + ("semantic-detail-" * 90) + "input-end <&> \"quoted\""
    return {
        "status": status,
        "run_id": run_id,
        "trigger_source": trigger_source,
        "input_sources": active_input_sources,
        "nodes": [
            {
                "id": "input.message",
                "label": "Queued turn",
                "stage": "L1",
                "lane": "input",
                "column": 1,
                "branch": "source",
                "status": "completed",
                "detail": {
                    "input": long_text,
                    "reply_context": {
                        "reply_excerpt": "earlier message context",
                    },
                },
            },
            {
                "id": "l2.reasoning",
                "label": "Reasoning",
                "stage": "L2",
                "lane": "cognition",
                "column": 2,
                "branch": "judgment",
                "status": node_status,
                "detail": {
                    "internal_monologue": "weigh intent, memory, and scene pressure",
                    "logical_stance": "respond only if grounded",
                    "character_intent": "provide useful information",
                    "judgment_note": "the current request is grounded",
                },
            },
            {
                "id": "memory.lookup",
                "label": "Memory",
                "stage": "Memory",
                "lane": "memory",
                "column": 2,
                "branch": "parallel",
                "status": node_status,
                "detail": {
                    "retrieval_answer": "parallel memory evidence lookup",
                    "memory_evidence": [
                        {"fact": "the operator wants useful detail"},
                    ],
                },
            },
            {
                "id": "decision.reply",
                "label": "Decision",
                "stage": "Decision",
                "lane": "decision",
                "column": 3,
                "branch": "join",
                "status": node_status,
                "detail": {
                    "decision": "produce bounded reply",
                    "reasoning": "the request has a concrete operator goal",
                },
            },
            {
                "id": "l3.visual_directives",
                "label": "Visual directive",
                "stage": "L3",
                "lane": "surface",
                "column": 4,
                "branch": "visual",
                "status": node_status,
                "detail": {
                    "facial_expression": ["focused"],
                    "body_language": ["hands relaxed"],
                    "gaze_direction": ["toward the screen"],
                    "visual_vibe": ["attentive", "attentive"],
                },
            },
            {
                "id": "l3.surface",
                "label": "Visible surface",
                "stage": "L3",
                "lane": "surface",
                "column": 4,
                "branch": "dialog",
                "status": node_status,
                "detail": {
                    "messages": [
                        "first visible line\nsecond visible line",
                        "final visible message <&> \"safe\"",
                    ],
                },
            },
        ],
        "edges": [
            {"source": "input.message", "target": "l2.reasoning", "kind": "fork"},
            {"source": "input.message", "target": "memory.lookup", "kind": "fork"},
            {"source": "l2.reasoning", "target": "decision.reply", "kind": "join"},
            {"source": "memory.lookup", "target": "decision.reply", "kind": "join"},
            {"source": "decision.reply", "target": "l3.visual_directives", "kind": "fork"},
            {"source": "decision.reply", "target": "l3.surface", "kind": "fork"},
        ],
    }


def write_conflict_brain_registry(
    *,
    path: Path,
    fake_brain_base_url: str,
    python_executable: str,
) -> Path:
    """Write a registry where the fake brain appears unmanaged but available."""

    registry = {
        "services": [
            {
                "id": "brain",
                "display_name": "Brain service",
                "kind": "backend",
                "command": [
                    python_executable,
                    "tests/control_console_e2e/fake_services.py",
                    "--name",
                    "brain",
                ],
                "cwd": ".",
                "health_url": f"{fake_brain_base_url}/health",
            }
        ]
    }
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that ignores client disconnects during E2E teardown."""

    def handle_error(self, request, client_address) -> None:
        """Suppress expected disconnect noise and keep other errors visible."""

        del request, client_address


def _write_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
    """Write one JSON HTTP response."""

    body = json.dumps(payload).encode("utf-8")
    handler.send_response(200)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
