from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from typing_extensions import Self


class FakeBrainServer(AbstractContextManager["FakeBrainServer"]):
    """Threaded fake brain HTTP server for console E2E tests."""

    def __init__(self, port: int) -> None:
        """Create a fake brain bound to an explicit test port."""

        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._lock = Lock()
        self._graph = graph_snapshot(status="not_reported", run_id="not-reported")
        self._self_graph = graph_snapshot(
            status="not_reported",
            run_id="self-not-reported",
            run_kind="self_cognition",
        )
        self._chat_graph: dict[str, Any] | None = None
        self._chat_requests: list[dict[str, Any]] = []
        self._chat_status_code = 200
        self._chat_delay_seconds = 0.0
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    def __enter__(self) -> Self:
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

    def set_graph(self, graph: dict[str, Any] | None) -> None:
        """Replace the latest observation returned by the fake brain."""

        with self._lock:
            self._graph = graph

    def set_self_graph(self, graph: dict[str, Any] | None) -> None:
        """Replace the latest self-cognition observation returned by the fake brain."""

        with self._lock:
            self._self_graph = graph

    def set_chat_graph(self, graph: dict[str, Any] | None) -> None:
        """Set the observation returned by the next debug chat request."""

        with self._lock:
            self._chat_graph = graph

    def latest_graph(self) -> dict[str, Any] | None:
        """Return a copy of the latest observation."""

        with self._lock:
            graph = None if self._graph is None else dict(self._graph)
        return graph

    def latest_self_graph(self) -> dict[str, Any] | None:
        """Return a copy of the latest self-cognition observation."""

        with self._lock:
            graph = None if self._self_graph is None else dict(self._self_graph)
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
                with owner._lock:
                    graph = owner._chat_graph
                if graph is None:
                    graph = graph_snapshot(
                        status="completed",
                        run_id="debug-run-1",
                        llm_trace_id="llm-trace-debug-1",
                        cognition_invocation_id=(
                            "cognition-invocation-debug-1"
                        ),
                    )
                owner.set_graph(graph)
                _write_json(
                    self,
                    {
                        "delivery_tracking_id": "debug-run-1",
                        "trace_id": "llm-trace-debug-1",
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
            "self_cognition_graph": self.latest_self_graph(),
        }


def graph_snapshot(
    *,
    status: str,
    run_id: str,
    llm_trace_id: str = "",
    cognition_invocation_id: str = "",
    source_calendar_run_id: str = "",
    run_kind: str = "",
) -> dict[str, Any] | None:
    """Return a canonical Brain observation fixture."""

    run_kind = run_kind or (
        "self_cognition" if run_id.startswith("self-") else "live_turn"
    )

    llm_trace_id = llm_trace_id or f"llm-trace-{run_id}"
    cognition_invocation_id = (
        cognition_invocation_id or f"cognition-invocation-{run_id}"
    )
    if not source_calendar_run_id and run_id.startswith("self-"):
        source_calendar_run_id = f"calendar-run-{run_id}"
    section_ids = [
        "input.turn",
        "decision.response",
        "cognition.appraisals",
        "cognition.goal",
        "cognition.response_plan",
        "cognition.affect",
        "reasoning.subjective",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
    ]
    if run_kind == "self_cognition":
        section_ids.extend(
            ["self.source", "self.route", "self.consolidation"]
        )
        section_ids.remove("input.turn")
        section_ids.remove("decision.response")

    labels = {
        "input.turn": "Queued turn",
        "decision.response": "Response decision",
        "cognition.appraisals": "Semantic appraisals",
        "cognition.goal": "Character goal",
        "cognition.response_plan": "Response plan",
        "cognition.affect": "Affect projection",
        "reasoning.subjective": "Subjective reasoning",
        "reasoning.context_consumption": "Context consumption",
        "evidence.memory": "Memory evidence",
        "evidence.shared_memory_prewarm": "Shared-memory prewarm",
        "context.conversation_progress": "Conversation progress",
        "context.public_group_scene": "Public group scene",
        "action.requests": "Action requests",
        "action.results": "Action results",
        "action.continuation": "Action continuation",
        "surface.visual_directives": "Visual directives",
        "surface.visible_messages": "Visible messages",
        "self.source": "Self-cognition source",
        "self.route": "Self-cognition route",
        "self.consolidation": "Self-cognition consolidation",
    }
    section_categories = {
        "input.turn": "input",
        "decision.response": "decision",
        "cognition.appraisals": "appraisal",
        "cognition.goal": "goal",
        "cognition.response_plan": "response",
        "cognition.affect": "affect",
        "reasoning.subjective": "reasoning",
        "reasoning.context_consumption": "context",
        "evidence.memory": "memory",
        "evidence.shared_memory_prewarm": "prewarm",
        "context.conversation_progress": "progress",
        "context.public_group_scene": "group_scene",
        "action.requests": "action",
        "action.results": "action",
        "action.continuation": "continuation",
        "surface.visual_directives": "visual",
        "surface.visible_messages": "dialog",
        "self.source": "source",
        "self.route": "route",
        "self.consolidation": "continuity",
    }
    record_sections = {
        "cognition.appraisals",
        "cognition.affect",
        "reasoning.context_consumption",
        "evidence.memory",
        "evidence.shared_memory_prewarm",
        "context.conversation_progress",
        "context.public_group_scene",
        "action.requests",
        "action.results",
        "action.continuation",
        "surface.visual_directives",
        "surface.visible_messages",
    }
    sections: list[dict[str, Any]] = []
    for section_id in section_ids:
        presentation = "records" if section_id in record_sections else "fields"
        sections.append(
            {
                "section_id": section_id,
                "label": labels[section_id],
                "category": section_categories[section_id],
                "presentation": presentation,
                "status": (
                    "not_reported" if status == "not_reported" else "completed"
                ),
                "summary": (
                    ""
                    if status == "not_reported"
                    else f"{labels[section_id]} is available."
                ),
                "fields": [],
                "records": [],
                "reported_record_count": 0,
                "displayed_record_count": 0,
                "truncated": False,
            }
        )

    reasoning_section = next(
        section
        for section in sections
        if section["section_id"] == "reasoning.subjective"
    )
    if status != "not_reported":
        reasoning_section["fields"] = [
            {
                "key": "private_monologue",
                "label": "Private monologue",
                "value": "weigh the bounded operator request",
            },
            {
                "key": "logical_stance",
                "label": "Logical stance",
                "value": "respond with grounded detail",
            },
            {
                "key": "character_intent",
                "label": "Character intent",
                "value": "provide useful information",
            },
        ]

    visible_section = next(
        section
        for section in sections
        if section["section_id"] == "surface.visible_messages"
    )
    if status != "not_reported":
        visible_section["records"] = [
            {
                "key": "item_01",
                "label": "Message",
                "summary": "first visible line",
                "fields": [
                    {"key": "position", "label": "Position", "value": 1},
                    {
                        "key": "text",
                        "label": "Text",
                        "value": "first visible line",
                    },
                ],
            },
            {
                "key": "item_02",
                "label": "Message",
                "summary": "final visible message",
                "fields": [
                    {"key": "position", "label": "Position", "value": 2},
                    {
                        "key": "text",
                        "label": "Text",
                        "value": "final visible message",
                    },
                ],
            },
        ]
        visible_section["reported_record_count"] = 2
        visible_section["displayed_record_count"] = 2

    prewarm_section = next(
        section
        for section in sections
        if section["section_id"] == "evidence.shared_memory_prewarm"
    )
    if status != "not_reported":
        prewarm_section["fields"] = [
            {
                "key": "attempted",
                "label": "Attempted",
                "value": True,
            },
            {
                "key": "reason_code",
                "label": "Reason code",
                "value": "shared_memory_merged",
            },
            {
                "key": "retrieved_count",
                "label": "Retrieved count",
                "value": 1,
            },
            {
                "key": "merged_count",
                "label": "Merged count",
                "value": 1,
            },
        ]
        prewarm_section["records"] = [
            {
                "key": "item_01",
                "label": "Shared memory",
                "summary": "operator context",
                "fields": [
                    {
                        "key": "source_kind",
                        "label": "Source kind",
                        "value": "shared_memory",
                    },
                    {
                        "key": "content",
                        "label": "Content",
                        "value": "operator context",
                    },
                ],
            }
        ]
        prewarm_section["reported_record_count"] = 1
        prewarm_section["displayed_record_count"] = 1

    for section in sections:
        for index, record in enumerate(section["records"], 1):
            record["key"] = f"item_{index:02d}"

    node_catalog = [
        ("input.turn", "Queued turn", "Input", "input", 1, "input", ["input.turn"]),
        (
            "decision.response",
            "Response decision",
            "Decision",
            "gate",
            2,
            "decision",
            ["decision.response"],
        ),
        (
            "cognition.meaning",
            "Meaning appraisal",
            "Cognition",
            "cognition",
            3,
            "appraisal",
            ["cognition.appraisals"],
        ),
        (
            "cognition.goal",
            "Character goal",
            "Cognition",
            "cognition",
            3,
            "goal",
            ["cognition.goal"],
        ),
        (
            "cognition.response",
            "Response plan",
            "Cognition",
            "cognition",
            3,
            "response",
            ["cognition.response_plan"],
        ),
        (
            "cognition.affect",
            "Affect projection",
            "Cognition",
            "cognition",
            3,
            "affect",
            ["cognition.affect"],
        ),
        (
            "reasoning.context",
            "Reasoning and context",
            "Reasoning",
            "cognition",
            3,
            "reasoning",
            ["reasoning.subjective", "reasoning.context_consumption"],
        ),
        (
            "evidence.memory",
            "Memory and context",
            "Evidence",
            "memory",
            3,
            "memory",
            [
                "evidence.shared_memory_prewarm",
                "evidence.memory",
                "context.conversation_progress",
                "context.public_group_scene",
            ],
        ),
        (
            "action.results",
            "Actions",
            "Actions",
            "action",
            3,
            "action",
            ["action.requests", "action.results", "action.continuation"],
        ),
        (
            "surface.visual",
            "Visual directive",
            "Surface",
            "surface",
            4,
            "visual",
            ["surface.visual_directives"],
        ),
        (
            "surface.visible",
            "Visible surface",
            "Surface",
            "surface",
            4,
            "dialog",
            ["surface.visible_messages"],
        ),
    ]
    if run_kind == "self_cognition":
        node_catalog = [
            (
                "self.source",
                "Source case",
                "Input",
                "input",
                1,
                "source",
                ["self.source"],
            ),
            (
                "cognition.meaning",
                "Meaning appraisal",
                "Cognition",
                "cognition",
                2,
                "appraisal",
                ["cognition.appraisals"],
            ),
            (
                "cognition.goal",
                "Character goal",
                "Cognition",
                "cognition",
                2,
                "goal",
                ["cognition.goal"],
            ),
            (
                "cognition.response",
                "Response plan",
                "Cognition",
                "cognition",
                2,
                "response",
                ["cognition.response_plan"],
            ),
            (
                "cognition.affect",
                "Affect projection",
                "Cognition",
                "cognition",
                2,
                "affect",
                ["cognition.affect"],
            ),
            (
                "reasoning.context",
                "Reasoning and context",
                "Reasoning",
                "cognition",
                2,
                "reasoning",
                ["reasoning.subjective", "reasoning.context_consumption"],
            ),
            (
                "evidence.memory",
                "Memory and context",
                "Evidence",
                "memory",
                2,
                "memory",
                [
                    "evidence.shared_memory_prewarm",
                    "evidence.memory",
                    "context.conversation_progress",
                    "context.public_group_scene",
                ],
            ),
            (
                "self.route",
                "Route decision",
                "Decision",
                "decision",
                3,
                "route",
                ["self.route"],
            ),
            (
                "action.results",
                "Actions",
                "Actions",
                "action",
                4,
                "action",
                ["action.requests", "action.results", "action.continuation"],
            ),
            (
                "surface.visual",
                "Visual directive",
                "Surface",
                "surface",
                4,
                "visual",
                ["surface.visual_directives"],
            ),
            (
                "surface.visible",
                "Visible surface",
                "Surface",
                "surface",
                4,
                "dialog",
                ["surface.visible_messages"],
            ),
            (
                "self.consolidation",
                "Consolidation",
                "Continuity",
                "memory",
                5,
                "continuity",
                ["self.consolidation"],
            ),
        ]
    nodes = []
    section_by_id = {section["section_id"]: section for section in sections}
    status_priority = (
        "failed",
        "partial",
        "completed",
        "empty",
        "skipped",
        "not_reported",
    )
    for node_id, label, stage, lane, column, category, refs in node_catalog:
        referenced = [section_by_id[ref] for ref in refs]
        node_status = next(
            (
                status
                for status in status_priority
                if any(section["status"] == status for section in referenced)
            ),
            "not_reported",
        )
        nodes.append(
            {
                "node_id": node_id,
                "label": label,
                "stage": stage,
                "lane": lane,
                "column": column,
                "category": category,
                "status": node_status,
                "summary": referenced[0]["summary"][:180] or node_status,
                "section_refs": refs,
            }
        )
    sequence = [
        "input.turn",
        "decision.response",
        "cognition.meaning",
        "cognition.goal",
        "cognition.response",
        "action.results",
    ]
    if run_kind == "self_cognition":
        sequence = [
            "self.source",
            "cognition.meaning",
            "cognition.goal",
            "cognition.response",
            "self.route",
            "action.results",
            "self.consolidation",
        ]
    edges = [
        {
            "source": source,
            "target": target,
            "kind": "sequence",
            "label": "",
        }
        for source, target in pairwise(sequence)
        if source in {node["node_id"] for node in nodes}
        and target in {node["node_id"] for node in nodes}
    ]
    edges.extend(
        {
            "source": source,
            "target": target,
            "kind": "reference",
            "label": "",
        }
        for source, target in (
            ("evidence.memory", "cognition.meaning"),
            ("cognition.response", "cognition.affect"),
            ("cognition.response", "reasoning.context"),
            ("cognition.response", "surface.visual"),
            ("reasoning.context", "surface.visual"),
            ("evidence.memory", "surface.visual"),
            ("action.results", "surface.visual"),
            ("cognition.response", "surface.visible"),
            ("reasoning.context", "surface.visible"),
            ("evidence.memory", "surface.visible"),
            ("action.results", "surface.visible"),
            ("self.route", "surface.visual"),
            ("self.route", "surface.visible"),
            ("surface.visual", "self.consolidation"),
            ("surface.visible", "self.consolidation"),
        )
        if source in {node["node_id"] for node in nodes}
        and target in {node["node_id"] for node in nodes}
    )
    return {
        "schema_version": "cognition_run_observation.v1",
        "run_kind": run_kind,
        "status": (
            status
            if status in {"completed", "failed", "partial"}
            else "completed"
        ),
        "generated_at": "2026-08-26T00:00:00Z",
        "correlation": {
            "run_id": run_id,
            "llm_trace_id": llm_trace_id,
            "cognition_invocation_id": cognition_invocation_id,
            "source_calendar_run_id": source_calendar_run_id or None,
        },
        "sections": sections,
        "nodes": nodes,
        "edges": edges,
        "disclosure": {
            "policy": "approved_cognition_observation.v1",
            "excluded": [
                "prompt",
                "raw_model_output",
                "embedding",
                "raw_message",
                "message_envelope",
                "database_identifier",
                "adapter_identifier",
                "action_parameter",
                "handler_metadata",
                "worker_error_text",
            ],
        },
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
