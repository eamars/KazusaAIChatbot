"""Deterministic end-to-end contracts for V3 chain observability."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pymongo.errors import PyMongoError

from control_console.kazusa_client import KazusaClient
from kazusa_ai_chatbot import db as db_facade
from kazusa_ai_chatbot import event_logging, service
from kazusa_ai_chatbot import llm_tracing as tracing
from kazusa_ai_chatbot.db import cognition_chain_runs as chain_runs
from kazusa_ai_chatbot.event_logging import recording
from kazusa_ai_chatbot.llm_tracing import chain_transcript


class _Cursor:
    """Minimal async cursor for the public exact-correlation DB helper."""

    def __init__(
        self,
        rows: list[dict[str, object]],
        query: dict[str, object],
    ) -> None:
        self._rows = [
            row
            for row in rows
            if row.get("run_id") == query.get("run_id")
            and row.get("llm_trace_id") == query.get("llm_trace_id")
        ]

    def sort(self, key: object, direction: object) -> _Cursor:
        """Apply the bounded completion-time ordering used by the helper."""

        del key, direction
        self._rows.sort(
            key=lambda row: str(row.get("completed_at", "")),
            reverse=True,
        )
        return self

    def limit(self, count: int) -> _Cursor:
        """Keep the cursor API compatible with the one-row DB read."""

        self._rows = self._rows[:count]
        return self

    async def to_list(self, length: int) -> list[dict[str, object]]:
        """Return the bounded rows requested by the DB helper."""

        return [dict(row) for row in self._rows[:length]]


class _Collection:
    """In-memory collection that records exact dual-key reads."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.find_queries: list[dict[str, object]] = []

    async def find_one(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> dict[str, object] | None:
        """Find an idempotent row by its immutable chain-run id."""

        del projection
        for row in self.rows:
            if row.get("chain_run_id") == query.get("chain_run_id"):
                return dict(row)
        return None

    async def insert_one(self, document: dict[str, object]) -> None:
        """Store one validated chain-run document."""

        self.rows.append(dict(document))

    def find(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> _Cursor:
        """Record and execute the exact run/trace intersection query."""

        del projection
        self.find_queries.append(dict(query))
        return _Cursor(self.rows, query)


class _Database:
    """Minimal database facade for the public chain-run helpers."""

    def __init__(self, collection: _Collection) -> None:
        self.collection = collection

    def __getitem__(self, name: str) -> _Collection:
        """Return the one collection used by the chain-run owner."""

        del name
        return self.collection


class _FailingCollection:
    """Collection failure used to prove best-effort persistence."""

    async def find_one(
        self,
        query: dict[str, object],
        projection: dict[str, int] | None = None,
    ) -> None:
        """Raise a database error from the public upsert path."""

        del query, projection
        raise PyMongoError("in-memory write failure")


class _FailingDatabase:
    """Database facade returning the failing collection."""

    def __getitem__(self, name: str) -> _FailingCollection:
        """Return the failing chain-run collection."""

        del name
        return _FailingCollection()


def _chain_run_document(
    *,
    chain_run_id: str,
    run_id: str,
    llm_trace_id: str,
    cognition_invocation_id: str,
) -> dict[str, object]:
    """Build one complete sanitized chain-run document."""

    return {
        "schema_version": "cognition_chain_run.v1",
        "chain_run_id": chain_run_id,
        "engine": "v3",
        "run_id": run_id,
        "llm_trace_id": llm_trace_id,
        "cognition_invocation_id": cognition_invocation_id,
        "source_kind": "live",
        "chain_model_name": "chain-model",
        "sidecar_model_name": "sidecar-model",
        "subconscious_enabled": False,
        "appraisal_group_count": 2,
        "started_at": "2026-08-20T00:00:00Z",
        "completed_at": "2026-08-20T00:00:01Z",
        "terminal_disposition": "complete",
        "steps": [{"step_id": "A1", "status": "accepted"}],
        "step_count": 1,
        "ledger": {"active_total_ceiling_tokens": 50000},
        "sidecar": {"l1_stream_count": 0},
        "session_events": [],
        "degradation_markers": [],
        "warning_codes": [],
        "expires_at": "2026-08-22T00:00:00Z",
    }


def _latest_graph(*, run_id: str, llm_trace_id: str) -> dict[str, object]:
    """Build one bounded service graph carrying its correlation keys."""

    return {
        "run_id": run_id,
        "llm_trace_id": llm_trace_id,
        "cognition_invocation_id": "invocation-1",
        "status": "completed",
        "nodes": [],
        "edges": [],
    }


@pytest.mark.asyncio
async def test_protected_and_sanitized_records_share_exact_service_console_correlation(
    monkeypatch,
) -> None:
    """Protected content stays private while exact paired projections survive."""

    run_id = "run-1"
    trace_id = "trace-1"
    invocation_id = "invocation-1"
    raw_prompt = "private prompt content"
    raw_output = "private output content"
    private_metadata = "private metadata content"

    protected_rows: list[dict[str, object]] = []

    async def insert_trace_step(document: dict[str, object]) -> str:
        protected_rows.append(document)
        return str(document["step_id"])

    monkeypatch.setattr(tracing, "LLM_TRACE_CAPTURE_MODE", "full")
    monkeypatch.setattr(
        tracing.db_llm_tracing,
        "insert_trace_step",
        insert_trace_step,
    )
    trace_result = await chain_transcript.record_cognition_chain_transcript(
        trace_id=trace_id,
        run_id=run_id,
        messages=[
            SystemMessage(content=private_metadata),
            HumanMessage(content=raw_prompt),
            AIMessage(content=raw_output),
        ],
        steps=[
            {
                "step_id": "A1",
                "status": "accepted",
                "candidate": private_metadata,
            },
        ],
        terminal_disposition="complete",
    )

    assert trace_result["status"] == "recorded"
    assert protected_rows[0]["raw_messages"][1]["content"] == raw_prompt
    assert protected_rows[0]["steps"][0]["candidate"] == private_metadata

    event_rows: list[dict[str, object]] = []

    async def write_event(document: dict[str, object]) -> str:
        event_rows.append(document)
        return str(document["event_id"])

    monkeypatch.setattr(recording.repository, "write_event", write_event)
    event_result = await event_logging.record_cognition_chain_event(
        run_id=run_id,
        cognition_invocation_id=invocation_id,
        terminal_disposition="complete",
        chain_model_name="chain-model",
        sidecar_model_name="sidecar-model",
        step_count=1,
        duration_ms=100,
    )

    assert event_result["status"] == "recorded"
    assert raw_prompt not in repr(event_rows[0])
    assert raw_output not in repr(event_rows[0])
    assert private_metadata not in repr(event_rows[0])
    assert "raw_prompt" not in repr(event_rows[0])

    collection = _Collection()
    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_Database(collection)),
    )
    chain_document = _chain_run_document(
        chain_run_id="cogchain-1",
        run_id=run_id,
        llm_trace_id=trace_id,
        cognition_invocation_id=invocation_id,
    )
    assert await db_facade.save_cognition_chain_run(chain_document) is True
    assert await db_facade.get_cognition_chain_run(
        run_id=run_id,
        llm_trace_id=trace_id,
    ) == chain_document

    monkeypatch.setattr(
        service,
        "_latest_cognition_graph",
        _latest_graph(run_id=run_id, llm_trace_id=trace_id),
    )
    monkeypatch.setattr(service, "_latest_self_cognition_graph", None)
    service_response = await service.ops_latest_cognition_graph()
    assert service_response.cognition_chain_run == chain_document
    service_payload = service_response.model_dump(mode="json")
    assert raw_prompt not in repr(service_payload)
    assert raw_output not in repr(service_payload)
    assert private_metadata not in repr(service_payload)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(200, json=service_payload)

    client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )
    live_graph, live_chain, self_graph, self_chain = (
        await client.get_latest_cognition_graph_with_chain_runs()
    )
    assert live_graph.run_id == run_id
    assert live_graph.llm_trace_id == trace_id
    assert live_chain.status == "completed"
    assert live_chain.chain_run_id == "cogchain-1"
    assert self_graph.status == "not_reported"
    assert self_chain.status == "not_reported"
    console_payload = live_chain.model_dump(mode="json")
    assert raw_prompt not in repr(console_payload)
    assert raw_output not in repr(console_payload)
    assert private_metadata not in repr(console_payload)
    assert "steps" not in console_payload

    other_document = _chain_run_document(
        chain_run_id="cogchain-other",
        run_id="run-other",
        llm_trace_id="trace-other",
        cognition_invocation_id="invocation-other",
    )
    assert await db_facade.save_cognition_chain_run(other_document) is True

    for mismatched_graph in (
        _latest_graph(run_id=run_id, llm_trace_id="trace-other"),
        _latest_graph(run_id="run-other", llm_trace_id=trace_id),
    ):
        monkeypatch.setattr(service, "_latest_cognition_graph", mismatched_graph)
        mismatch_response = await service.ops_latest_cognition_graph()
        assert mismatch_response.cognition_chain_run is None

    assert collection.find_queries[-2:] == [
        {"run_id": run_id, "llm_trace_id": "trace-other"},
        {"run_id": "run-other", "llm_trace_id": trace_id},
    ]
    mismatch_payload = mismatch_response.model_dump(mode="json")

    async def mismatch_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ops/latest-cognition-graph"
        return httpx.Response(200, json=mismatch_payload)

    mismatch_client = KazusaClient(
        base_url="http://brain.local",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(mismatch_handler),
    )
    _, mismatch_chain, _, _ = (
        await mismatch_client.get_latest_cognition_graph_with_chain_runs()
    )
    assert mismatch_chain.status == "not_reported"

    async def failing_trace_write(document: dict[str, object]) -> str:
        del document
        raise RuntimeError("trace backend unavailable")

    monkeypatch.setattr(
        tracing.db_llm_tracing,
        "insert_trace_step",
        failing_trace_write,
    )
    failed_trace = await chain_transcript.record_cognition_chain_transcript(
        trace_id=trace_id,
        run_id=run_id,
        messages=[HumanMessage(content=raw_prompt)],
        steps=[],
        terminal_disposition="complete",
    )
    assert failed_trace["status"] == "failed"

    async def failing_event_write(document: dict[str, object]) -> str:
        del document
        raise RuntimeError("event backend unavailable")

    monkeypatch.setattr(recording.repository, "write_event", failing_event_write)
    failed_event = await event_logging.record_cognition_chain_event(
        run_id=run_id,
        cognition_invocation_id=invocation_id,
        terminal_disposition="complete",
    )
    assert failed_event["status"] == "failed"

    monkeypatch.setattr(
        chain_runs,
        "get_db",
        AsyncMock(return_value=_FailingDatabase()),
    )
    assert await db_facade.save_cognition_chain_run(chain_document) is False
