"""Real LLM checks for observation-driven RAG query refinement."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.db import build_memory_doc, close_db, save_memory
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as rag2_module
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_persistent_memory_keyword
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_ORIGINAL_QUERY = 'CB1 和 CM4 现在性能到底谁更强？'
_MEMORY_SLOT = (
    'Memory-evidence: retrieve durable evidence about CB1 and CM4 '
    'performance comparison'
)
_POLICY_CONTENT = (
    'RAG2 live validation policy: CB1 与 CM4 的具体性能对比没有持久结论。'
    '涉及最新设备、驱动、价格和跑分时，记忆很容易过期，'
    '应使用实时检索或说明知识可能不是最新。'
)
_DISALLOWED_REFINED_QUERY_PREFIXES = (
    'Live-context:',
    'Conversation-evidence:',
    'Memory-evidence:',
    'Person-context:',
    'Recall:',
    'Web-evidence:',
)


class _DummyResponse:
    """Small response object for deterministic LLM test doubles."""

    def __init__(self, content: str) -> None:
        """Store model-compatible content."""
        self.content = content


class _CapturingAsyncLLM:
    """Capture real LLM calls while preserving the wrapped LLM behavior."""

    def __init__(self, wrapped_llm) -> None:
        """Store the wrapped LLM and initialize captured calls."""
        self._wrapped_llm = wrapped_llm
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages):
        """Invoke the wrapped LLM and store raw prompt/response text."""
        response = await self._wrapped_llm.ainvoke(messages)
        prompt_parts = []
        for message in messages:
            prompt_parts.append(
                {
                    "type": type(message).__name__,
                    "content": str(message.content),
                }
            )
        self.calls.append(
            {
                "messages": prompt_parts,
                "raw_output": str(response.content),
            }
        )
        return response


class _SequencedInitializerLLM:
    """Initializer fake that still flows through rag_initializer and Cache2."""

    def __init__(self, slot_batches: list[list[str]]) -> None:
        """Store ordered slot batches returned by the fake initializer model."""
        self.slot_batches = slot_batches
        self.payloads: list[dict] = []

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Capture initializer payloads and return the next slot batch."""
        payload = json.loads(messages[1].content)
        self.payloads.append(payload)
        batch_index = len(self.payloads) - 1
        if batch_index < len(self.slot_batches):
            slots = self.slot_batches[batch_index]
        else:
            slots = []
        response = _DummyResponse(json.dumps({"unknown_slots": slots}))
        return response


class _SummaryLLM:
    """Evaluator summarizer fake that preserves selected summaries."""

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Extract selected_summary from the evaluator payload."""
        payload = json.loads(messages[1].content)
        raw_result = payload["raw_result"]
        summary = ""
        if isinstance(raw_result, dict):
            summary = str(raw_result.get("selected_summary", ""))
        response = _DummyResponse(summary)
        return response


class _FinalizerLLM:
    """Finalizer fake that echoes the latest resolved summary."""

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Return a compact final answer from resolved facts."""
        payload = json.loads(messages[1].content)
        summary = ""
        for fact in payload["known_facts"]:
            if fact.get("resolved"):
                summary = str(fact.get("summary", ""))
        response = _DummyResponse(f'final: {summary}')
        return response


async def _noop_async(*args, **kwargs) -> None:
    """Accept cache persistence hooks without touching persistent Cache2."""
    del args, kwargs


async def _skip_if_rag_subagent_llm_unavailable() -> None:
    """Skip when the configured RAG subagent endpoint is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{RAG_SUBAGENT_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f'RAG subagent LLM endpoint is unavailable: {exc}'
        )

    if response.status_code >= 500:
        pytest.skip(
            f'RAG subagent LLM endpoint returned server error '
            f'{response.status_code}'
        )


def _observation_payload(
    *,
    original_query: str,
    current_slot: str,
    observation_candidates: list[dict[str, object]],
    source_policy: str = '',
    user_resolution_hints: list[str] | None = None,
) -> dict[str, object]:
    """Build the refiner payload used by unresolved retrieval observations."""
    payload = {
        "original_query": original_query,
        "current_slot": current_slot,
        "agent": "memory_evidence_agent",
        "resolved": False,
        "source_policy": source_policy,
        "missing_context": ["memory_evidence"],
        "conflicts": [],
        "observation_candidates": observation_candidates,
        "source_hints": [],
        "user_resolution_hints": list(user_resolution_hints or []),
        "known_facts": [],
        "pending_slots": [],
    }
    return payload


async def _assess_with_trace(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Run the real continuation refiner and write an inspectable trace."""
    await _skip_if_rag_subagent_llm_unavailable()
    capturing_llm = _CapturingAsyncLLM(rag2_module._continuation_assessor_llm)
    monkeypatch.setattr(
        rag2_module,
        "_continuation_assessor_llm",
        capturing_llm,
    )

    original_query = str(payload["original_query"])
    decision = await rag2_module._assess_continuation(
        observation_payload=payload,
        original_query=original_query,
        previous_refined_queries=[],
        continuation_count=0,
    )
    trace_path = write_llm_trace(
        "rag_phase4_continuation_live_llm",
        case_id,
        {
            "model": RAG_SUBAGENT_LLM_MODEL,
            "payload": payload,
            "llm_calls": capturing_llm.calls,
            "decision": decision,
            "judgment": "manual_review_required_for_refined_query_quality",
        },
    )
    logger.info(
        f'RAG_PHASE4_CONTINUATION_LIVE case={case_id} trace={trace_path} '
        f'should_continue={decision["should_continue"]} '
        f'refined_query={decision["refined_query"]!r}'
    )
    return decision


def _payload_source_text(payload: dict[str, object]) -> str:
    """Return visible source text that may constrain the refined query."""
    parts = [
        str(payload.get("original_query", "")),
        str(payload.get("current_slot", "")),
        str(payload.get("source_policy", "")),
    ]
    for raw_candidate in payload.get("observation_candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        parts.append(str(raw_candidate.get("content", "")))
        parts.append(str(raw_candidate.get("source", "")))

    text = "\n".join(parts)
    return text


def _assert_refined_query_continues(
    decision: dict[str, object],
    *,
    original_query: str,
    source_text: str,
) -> None:
    """Assert the refiner produced a natural-language continuation query."""
    assert decision["should_continue"] is True
    refined_query = str(decision["refined_query"])
    assert refined_query
    assert refined_query.strip() != original_query.strip()
    assert not refined_query.lstrip().startswith(_DISALLOWED_REFINED_QUERY_PREFIXES)

    source_years = set(re.findall(r"\b20\d{2}\b", source_text))
    refined_years = set(re.findall(r"\b20\d{2}\b", refined_query))
    assert refined_years <= source_years


def _assert_refiner_stops(decision: dict[str, object]) -> None:
    """Assert the refiner stopped without carrying a re-entry query."""
    assert decision["should_continue"] is False
    assert decision["refined_query"] == ""


async def _seed_policy_memory() -> dict[str, str]:
    """Insert one cleanup-safe shared memory row for live DB retrieval."""
    suffix = uuid4().hex
    memory_unit_id = f'pytest-rag2-phase4-continuation-{suffix}'
    memory_name = f'rag2_phase4_cb1_cm4_source_policy_{suffix}'
    timestamp = datetime.now(timezone.utc).isoformat()
    doc = build_memory_doc(
        memory_name=memory_name,
        content=_POLICY_CONTENT,
        source_global_user_id='',
        memory_type='fact',
        source_kind='seeded_manual',
        confidence_note='temporary live-test source policy row',
        status='active',
        expiry_timestamp=None,
    )
    doc["memory_unit_id"] = memory_unit_id
    doc["lineage_id"] = memory_unit_id
    await save_memory(doc, timestamp)
    row = {
        "memory_unit_id": memory_unit_id,
        "memory_name": memory_name,
        "content": _POLICY_CONTENT,
    }
    return row


async def _delete_policy_memory(memory_unit_id: str) -> None:
    """Remove one temporary shared memory row from the real DB."""
    db = await get_db()
    await db.memory.delete_one({"memory_unit_id": memory_unit_id})


async def _retrieve_seeded_policy_row(memory_name: str) -> list[dict]:
    """Read the temporary source-policy memory through the real tool."""
    rows = await search_persistent_memory_keyword.ainvoke(
        {"keyword": memory_name, "top_k": 1}
    )
    assert isinstance(rows, list)
    assert rows
    first_row = rows[0]
    assert isinstance(first_row, dict)
    assert first_row.get("memory_name") == memory_name
    return rows


async def test_live_refiner_continues_from_stale_memory_direction(
    monkeypatch,
) -> None:
    """A stale-memory observation should become a refined fresh-query input."""
    payload = _observation_payload(
        original_query=_ORIGINAL_QUERY,
        current_slot=_MEMORY_SLOT,
        observation_candidates=[
            {
                "content": (
                    '涉及最新设备、驱动、价格和跑分时，记忆很容易过期，'
                    '应使用实时检索或说明知识可能不是最新。'
                ),
                "source": "memory",
            }
        ],
        source_policy='semantic durable memory evidence',
    )

    decision = await _assess_with_trace(
        monkeypatch,
        case_id="stale_memory_to_refined_query",
        payload=payload,
    )

    _assert_refined_query_continues(
        decision,
        original_query=_ORIGINAL_QUERY,
        source_text=_payload_source_text(payload),
    )


async def test_live_refiner_stops_on_missing_user_constraints(
    monkeypatch,
) -> None:
    """Missing user-purpose constraints should stop, not plan a new slot."""
    payload = _observation_payload(
        original_query='我该买哪个开发板？',
        current_slot='Memory-evidence: retrieve durable recommendation guidance',
        observation_candidates=[
            {
                "content": (
                    '比较设备、模型、软件、路线或方案时，最好先明确用途、'
                    '预算、约束和偏好；不同目标下最优答案可能完全不同。'
                ),
                "source": "memory",
            }
        ],
        source_policy='semantic durable memory evidence',
    )

    decision = await _assess_with_trace(
        monkeypatch,
        case_id="missing_constraints_stop",
        payload=payload,
    )

    _assert_refiner_stops(decision)


async def test_live_refiner_stops_on_irrelevant_observation(
    monkeypatch,
) -> None:
    """Irrelevant memory by-products must not repair the original query."""
    payload = _observation_payload(
        original_query=_ORIGINAL_QUERY,
        current_slot=_MEMORY_SLOT,
        observation_candidates=[
            {
                "content": (
                    '三一综合学园与格黑娜学园长期不和，'
                    '两校矛盾是伊甸园条约篇的重要背景。'
                ),
                "source": "memory",
            }
        ],
        source_policy='semantic durable memory evidence',
    )

    decision = await _assess_with_trace(
        monkeypatch,
        case_id="irrelevant_memory_stop",
        payload=payload,
    )

    _assert_refiner_stops(decision)


@pytest.mark.live_db
async def test_live_db_memory_byproduct_produces_refined_query(
    monkeypatch,
) -> None:
    """A real DB memory by-product should drive a real refined query."""
    await _skip_if_rag_subagent_llm_unavailable()
    seed = await _seed_policy_memory()
    try:
        rows = await _retrieve_seeded_policy_row(seed["memory_name"])
        payload = _observation_payload(
            original_query=_ORIGINAL_QUERY,
            current_slot=_MEMORY_SLOT,
            observation_candidates=rows,
            source_policy='semantic durable memory evidence',
        )
        decision = await _assess_with_trace(
            monkeypatch,
            case_id="real_db_memory_byproduct_to_refined_query",
            payload=payload,
        )

        _assert_refined_query_continues(
            decision,
            original_query=_ORIGINAL_QUERY,
            source_text=_payload_source_text(payload),
        )
    finally:
        await _delete_policy_memory(seed["memory_unit_id"])
        await close_db()


@pytest.mark.live_db
async def test_live_db_supervisor_reenters_initializer_with_cache2_active(
    monkeypatch,
) -> None:
    """The supervisor should run refined query through the Cache2 initializer."""
    await _skip_if_rag_subagent_llm_unavailable()
    seed = await _seed_policy_memory()
    runtime = RAGCache2Runtime(max_entries=10)
    refined_slot = 'Live-context: answer current CB1 and CM4 performance evidence'
    initializer_llm = _SequencedInitializerLLM(
        [
            [_MEMORY_SLOT],
            [refined_slot],
        ]
    )
    followup_tasks: list[str] = []
    capturing_llm = _CapturingAsyncLLM(rag2_module._continuation_assessor_llm)

    async def _memory_observation_agent(
        task: str,
        context: dict,
        max_attempts: int = 1,
    ) -> dict:
        """Return unresolved observation rows loaded from the real DB."""
        del task, context, max_attempts
        rows = await _retrieve_seeded_policy_row(seed["memory_name"])
        result = {
            "resolved": False,
            "result": {
                "capability": "memory_evidence",
                "primary_worker": "persistent_memory_search_agent",
                "supporting_workers": [],
                "source_policy": "semantic durable memory evidence",
                "selected_summary": "",
                "resolved_refs": [],
                "projection_payload": {"memory_rows": []},
                "worker_payloads": {},
                "evidence": [],
                "missing_context": ["memory_evidence"],
                "conflicts": [],
                "observation_candidates": rows,
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "live_test_uncached",
            },
        }
        return result

    async def _fresh_followup_agent(
        task: str,
        context: dict,
        max_attempts: int = 1,
    ) -> dict:
        """Record that the initializer-produced follow-up slot was executed."""
        del context, max_attempts
        followup_tasks.append(task)
        result = {
            "resolved": True,
            "result": {
                "capability": "live_context",
                "primary_worker": "web_search_agent2",
                "supporting_workers": [],
                "source_policy": "fresh external retrieval",
                "selected_summary": f'fresh follow-up executed: {task}',
                "resolved_refs": [],
                "projection_payload": {
                    "external_text": f'fresh follow-up executed: {task}',
                    "url": "https://example.test/rag-phase4",
                },
                "worker_payloads": {},
                "evidence": [task],
                "missing_context": [],
                "conflicts": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "live_test_uncached",
            },
        }
        return result

    memory_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["memory_evidence_agent"]
    )
    memory_entry["agent"] = _memory_observation_agent
    live_entry = dict(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY["live_context_agent"]
    )
    live_entry["agent"] = _fresh_followup_agent

    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "memory_evidence_agent",
        memory_entry,
    )
    monkeypatch.setitem(
        rag2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "live_context_agent",
        live_entry,
    )
    monkeypatch.setattr(rag2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(rag2_module, "_initializer_llm", initializer_llm)
    monkeypatch.setattr(rag2_module, "_continuation_assessor_llm", capturing_llm)
    monkeypatch.setattr(rag2_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(rag2_module, "_finalizer_llm", _FinalizerLLM())
    monkeypatch.setattr(rag2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(rag2_module, "record_initializer_hit", _noop_async)

    try:
        result = await rag2_module.call_rag_supervisor(
            original_query=_ORIGINAL_QUERY,
            character_name='the active character',
            context={
                "platform": "qq",
                "platform_channel_id": "rag-phase4-live-db",
                "global_user_id": "user-1",
                "user_name": "Live Tester",
                "current_timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt_message_context": {
                    "body_text": _ORIGINAL_QUERY,
                    "mentions": [],
                    "attachments": [],
                    "addressed_to_global_user_ids": [],
                    "broadcast": False,
                },
            },
        )
        trace_path = write_llm_trace(
            "rag_phase4_continuation_live_llm",
            "real_db_supervisor_refined_query_cache2",
            {
                "model": RAG_SUBAGENT_LLM_MODEL,
                "seed_memory_name": seed["memory_name"],
                "continuation_llm_calls": capturing_llm.calls,
                "initializer_payloads": initializer_llm.payloads,
                "followup_tasks": followup_tasks,
                "result": result,
                "judgment": (
                    "supervisor should feed the refined query through "
                    "rag_initializer with Cache2 active"
                ),
            },
        )
        logger.info(
            f'RAG_PHASE4_CONTINUATION_LIVE '
            f'case=real_db_supervisor_refined_query_cache2 trace={trace_path} '
            f'followup_tasks={followup_tasks}'
        )

        known_facts = result["known_facts"]
        assert len(known_facts) >= 2
        first_fact = known_facts[0]
        continuation = first_fact["continuation"]
        assert first_fact["resolved"] is False
        assert continuation["should_continue"] is True
        assert continuation["refined_query"]
        assert len(initializer_llm.payloads) == 2
        assert initializer_llm.payloads[1]["original_query"] == (
            continuation["refined_query"]
        )
        assert followup_tasks == [refined_slot]
        assert known_facts[1]["resolved"] is True
    finally:
        await _delete_policy_memory(seed["memory_unit_id"])
        await close_db()
