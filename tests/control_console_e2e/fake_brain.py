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
        self._self_graph = graph_snapshot(
            status="not_reported",
            run_id="self-not-reported",
        )
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

    def set_self_graph(self, graph: dict[str, Any]) -> None:
        """Replace the latest self-cognition graph returned by the fake brain."""

        with self._lock:
            self._self_graph = graph

    def latest_graph(self) -> dict[str, Any]:
        """Return a copy of the latest graph."""

        with self._lock:
            graph = dict(self._graph)
        return graph

    def latest_self_graph(self) -> dict[str, Any]:
        """Return a copy of the latest self-cognition graph."""

        with self._lock:
            graph = dict(self._self_graph)
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
            "self_cognition_graph": self.latest_self_graph(),
        }


def graph_snapshot(*, status: str, run_id: str) -> dict[str, Any]:
    """Return a cognition graph snapshot with parallel branches."""

    if status == "not_reported":
        return {
            "status": "not_reported",
            "run_id": run_id,
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


def native_v2_graph_snapshot(*, status: str, run_id: str) -> dict[str, Any]:
    """Return a graph containing native V2 branch and affect semantics."""

    graph = graph_snapshot(status=status, run_id=run_id)
    if status == 'not_reported':
        return graph
    node_status = 'running' if status == 'running' else 'completed'
    if status == 'failed':
        branch_one_status = 'failed'
        branch_two_status = 'completed'
    else:
        branch_one_status = node_status
        branch_two_status = node_status
    graph['nodes'].extend([
        {
            'id': 'v2.parallel',
            'label': 'Parallel cognition',
            'stage': 'V2',
            'lane': 'cognition',
            'column': 3,
            'branch': 'parallel',
            'status': node_status,
            'detail': {
                'parallel_execution': {
                    'selected_question_count': 2,
                    'dispatched_question_count': 2,
                    'selected_branch_count': 2,
                    'dispatched_branch_count': 2,
                    'completed_branch_count': 1 if status == 'failed' else 2,
                    'failed_branch_count': 1 if status == 'failed' else 0,
                    'maximum_concurrency': 2,
                    'overlap_ms': 42,
                    'dependency_wait_ms': 0,
                    'total_ms': 188,
                },
                'branch_results': [
                    {
                        'branch_index': 1,
                        'selection': 'primary',
                        'intention': '保护重要关系中的边界',
                    },
                    {
                        'branch_index': 2,
                        'selection': 'suppressed',
                        'intention': '立即反击',
                    },
                ],
            },
        },
        {
            'id': 'v2.appraisal',
            'label': 'Appraisal results',
            'stage': 'V2',
            'lane': 'cognition',
            'column': 3,
            'branch': 'appraisal',
            'status': node_status,
            'detail': {
                'appraisal_results': [
                    {
                        'question_kind': 'relationship_social',
                        'semantic_question': '这次行为怎样改变了关系中的安全感？',
                        'status': 'completed',
                        'explanation': '持续贬低削弱了关系安全感。',
                        'propositions': [
                            {
                                'proposition_kind': 'relationship_shift',
                                'semantic_value': '亲近关系中的信任受到伤害。',
                            },
                        ],
                        'deltas': [
                            {'delta': -20, 'reason': '关系安全感下降。'},
                        ],
                    },
                ],
            },
        },
        {
            'id': 'v2.branch.1',
            'label': 'Goal branch 1',
            'stage': 'V2',
            'lane': 'cognition',
            'column': 4,
            'branch': 'branch-1',
            'status': branch_one_status,
            'detail': {
                'phase': 'preliminary',
                'branch_index': 1,
                'goal_kind': 'bond_protection',
                'status': 'failed' if status == 'failed' else 'completed',
                'selection': 'primary',
                'intention': '保护重要关系中的边界',
                'desired_outcome': '让伤害被看见',
                'concrete_detail': '说明这句话造成的伤害',
                'reason': '重要关系中的持续贬低需要回应。',
                'private_monologue': '我不想假装没受伤。',
                'expected_consequences': ['边界变得清楚'],
                'confidence': '高',
            },
        },
        {
            'id': 'v2.branch.2',
            'label': 'Goal branch 2',
            'stage': 'V2',
            'lane': 'cognition',
            'column': 4,
            'branch': 'branch-2',
            'status': branch_two_status,
            'detail': {
                'phase': 'preliminary',
                'branch_index': 2,
                'goal_kind': 'autonomy_boundary',
                'status': 'completed',
                'selection': 'suppressed',
                'intention': '立即反击',
                'desired_outcome': '结束当前攻击',
                'concrete_detail': '用更强硬的话顶回去',
                'reason': '被冒犯会自然地产生反击冲动。',
                'private_monologue': '我很想马上反击，但这会让关系更糟。',
                'expected_consequences': ['冲突可能进一步升级'],
                'confidence': '中',
            },
        },
        {
            'id': 'v2.collapse',
            'label': 'Workspace collapse',
            'stage': 'V2',
            'lane': 'decision',
            'column': 5,
            'branch': 'collapse',
            'status': node_status,
            'detail': {
                'collapse': {
                    'primary_branch_index': 1,
                    'supporting_branch_indices': [],
                    'suppressed_branch_indices': [2],
                    'selection_reason': '主目标保留了受伤事实，反击目标被压下。',
                },
                'selected_intention': {
                    'route': 'speech',
                    'intention': '回应当前关系中的受伤感受',
                    'reason': '当前事件足以支持直接回应',
                },
                'selected_bid_reason': '她先承认这次伤害，再决定如何回应。',
                'private_monologue': '我确实被这句话刺痛了，但我想先把感受说清楚。',
                'goal_resolution': 'answerable_now',
            },
        },
        {
            'id': 'v2.affect',
            'label': 'Affect projection',
            'stage': 'V2',
            'lane': 'cognition',
            'column': 5,
            'branch': 'affect',
            'status': node_status,
            'detail': {
                'affect_projection': [
                    {
                        'emotion': '悲伤',
                        'phase': '激活',
                        'intensity': '高',
                        'trend': '上升',
                        'cause_summary': '关系伤害带来失落。',
                    },
                ],
            },
        },
    ])
    graph['edges'].extend([
        {'source': 'l2.reasoning', 'target': 'v2.parallel', 'kind': 'fork'},
        {'source': 'v2.parallel', 'target': 'v2.appraisal', 'kind': 'fork'},
        {'source': 'v2.parallel', 'target': 'v2.branch.1', 'kind': 'fork'},
        {'source': 'v2.parallel', 'target': 'v2.branch.2', 'kind': 'fork'},
        {'source': 'v2.appraisal', 'target': 'v2.collapse', 'kind': 'join'},
        {'source': 'v2.branch.1', 'target': 'v2.collapse', 'kind': 'join'},
        {'source': 'v2.branch.2', 'target': 'v2.collapse', 'kind': 'join'},
        {'source': 'v2.collapse', 'target': 'v2.affect', 'kind': 'sequence'},
        {'source': 'v2.affect', 'target': 'l3.surface', 'kind': 'join'},
    ])
    return graph


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
