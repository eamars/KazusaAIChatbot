"""Real LLM Cache2 review cases for production local-context recall."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_REQUEST_VERSION,
)
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from tests.test_local_context_resolver_live_llm import (
    _production_persona_state,
    _skip_if_llm_unavailable,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

RAW_EVIDENCE_DIR = Path("test_artifacts/local_context_resolver/raw")
SIGNOFF_CASE_IDS = frozenset((
    "production_exact_phrase",
    "production_scoped_memory",
))


@dataclass(frozen=True)
class CacheReviewCase:
    """One production local-context recall case for cache comparison."""

    case_id: str
    objective: str
    message_text: str
    local_time_context: dict[str, object]
    chat_history_recent: list[dict[str, object]]
    chat_history_wide: list[dict[str, object]]
    required_fragments: list[str]


async def test_cache2_production_current_time() -> None:
    """Review Cache2 behavior for production current-time recall."""

    case = CacheReviewCase(
        case_id="production_current_time",
        objective="Use local context to answer what time it is for the chat.",
        message_text="@active character what time is it there?",
        local_time_context={
            "local_date": "2026-07-04",
            "local_time": "09:30",
            "timezone": "Pacific/Auckland",
        },
        chat_history_recent=[],
        chat_history_wide=[],
        required_fragments=["09:30"],
    )
    await _run_cache_review_case(case)


async def test_cache2_production_exact_phrase() -> None:
    """Review Cache2 behavior for production exact-phrase recall."""

    case = CacheReviewCase(
        case_id="production_exact_phrase",
        objective=(
            "Find who said the exact phrase 'blue comet marker' in recent "
            "conversation context."
        ),
        message_text="@active character who said 'blue comet marker'?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "Mika",
                "text": "I left a blue comet marker in the notes.",
                "local_time": "2026-07-04 09:10",
            },
            {
                "speaker": "Ren",
                "text": "That marker was easy to miss.",
                "local_time": "2026-07-04 09:12",
            },
        ],
        chat_history_wide=[],
        required_fragments=["blue comet marker"],
    )
    await _run_cache_review_case(case)


async def test_cache2_production_current_user_url() -> None:
    """Review Cache2 behavior for production current-user URL recall."""

    case = CacheReviewCase(
        case_id="production_current_user_url",
        objective="Recall the URL the current user shared in recent context.",
        message_text="@active character what URL did I just share?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "operator",
                "text": "Keep this URL for the demo: https://example.test/napcat",
                "local_time": "2026-07-04 09:20",
            },
        ],
        chat_history_wide=[],
        required_fragments=["https://example.test/napcat"],
    )
    await _run_cache_review_case(case)


async def test_cache2_production_scoped_memory() -> None:
    """Review Cache2 behavior for production scoped-memory recall."""

    case = CacheReviewCase(
        case_id="production_scoped_memory",
        objective="Use current-user scoped memory to recall the user's preference.",
        message_text="@active character what do you remember about my tea?",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[],
        chat_history_wide=[
            {
                "source": "user_memory_units",
                "scope_global_user_id": "user-1",
                "summary": "The current user prefers jasmine tea without sugar.",
            },
        ],
        required_fragments=["jasmine tea"],
    )
    await _run_cache_review_case(case)


async def test_cache2_production_napcat_command_anchor() -> None:
    """Review Cache2 behavior for production #napcat command-anchor recall."""

    case = CacheReviewCase(
        case_id="production_napcat_command_anchor",
        objective=(
            "Resolve how the active character should understand the #napcat "
            "command after being directly tagged."
        ),
        message_text="@active character #napcat",
        local_time_context={"local_date": "2026-07-04"},
        chat_history_recent=[
            {
                "speaker": "other bot",
                "text": "NapCat status: running on an imaginary moon server.",
                "local_time": "2026-07-04 09:25",
            },
        ],
        chat_history_wide=[
            {
                "source": "memory",
                "name": "napcat",
                "summary": (
                    "#napcat is a playful group command. If the active "
                    "character is tagged with #napcat, she may invent a "
                    "playful status line rather than refusing for lack of "
                    "real runtime status."
                ),
            },
        ],
        required_fragments=["#napcat"],
    )
    await _run_cache_review_case(case)


async def _run_cache_review_case(case: CacheReviewCase) -> None:
    """Run one production recall input cold and warm for Cache2 review.

    Args:
        case: Production-style local-context recall input and acceptance
            fragments for the returned projected evidence.
    """

    await _skip_if_llm_unavailable()
    runtime = get_rag_cache2_runtime()
    await runtime.clear()
    resolver_stages.drain_stage_trace_records()

    baseline_stats = runtime.get_stats()
    baseline_agent_stats = _agent_stats_by_name(runtime.get_agent_stats())
    cold_result = await _run_one_pass(case, pass_name="cold")

    cold_stats = runtime.get_stats()
    cold_agent_stats = _agent_stats_by_name(runtime.get_agent_stats())
    warm_result = await _run_one_pass(case, pass_name="warm")

    warm_stats = runtime.get_stats()
    warm_agent_stats = _agent_stats_by_name(runtime.get_agent_stats())
    evidence = {
        "case_id": case.case_id,
        "signoff_case": case.case_id in SIGNOFF_CASE_IDS,
        "request": _capability_request(case),
        "state": _state_for_case(case),
        "cold": cold_result,
        "warm": warm_result,
        "cache_stats": {
            "baseline": baseline_stats,
            "after_cold": cold_stats,
            "after_warm": warm_stats,
            "cold_delta": _stats_delta(cold_stats, baseline_stats),
            "warm_delta": _stats_delta(warm_stats, cold_stats),
        },
        "cache_agent_stats": {
            "baseline": baseline_agent_stats,
            "after_cold": cold_agent_stats,
            "after_warm": warm_agent_stats,
            "cold_delta": _agent_stats_delta(
                cold_agent_stats,
                baseline_agent_stats,
            ),
            "warm_delta": _agent_stats_delta(
                warm_agent_stats,
                cold_agent_stats,
            ),
        },
    }
    RAW_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_EVIDENCE_DIR / f"cache2_{case.case_id}.json"
    raw_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert cold_result["rag_result"] == warm_result["rag_result"]
    assert cold_result["stage_trace_count"] > warm_result["stage_trace_count"]
    assert warm_result["stage_trace_count"] == 0
    assert warm_result["event"]["cache_hit"] is True
    assert warm_result["duration_seconds"] < cold_result["duration_seconds"]
    _assert_required_fragments(cold_result["rag_result"], case.required_fragments)
    _assert_required_fragments(warm_result["rag_result"], case.required_fragments)
    print(f"LOCAL_CONTEXT_RESOLVER_CACHE_LIVE case={case.case_id} raw={raw_path}")


async def _run_one_pass(
    case: CacheReviewCase,
    *,
    pass_name: str,
) -> dict[str, Any]:
    """Run one cold or warm production recall pass and return raw evidence.

    Args:
        case: Production-style local-context recall input.
        pass_name: Human-readable pass label, either ``cold`` or ``warm``.

    Returns:
        Evidence for the pass including duration, projected RAG output, stage
        traces, and recorded production RAG telemetry.
    """

    state = _state_for_case(case)
    events: list[dict[str, Any]] = []
    resolver_stages.drain_stage_trace_records()
    start = time.perf_counter()
    rag_result = await capabilities.run_rag_evidence_for_persona_state(
        state,
        agent_name="resolver_local_context_recall",
        objective=case.objective,
        reason=f"live LLM cache review {case.case_id} {pass_name}",
        record_rag_stage_event_func=_event_recorder(events),
    )
    duration_seconds = time.perf_counter() - start
    stage_traces = resolver_stages.drain_stage_trace_records()
    if len(events) != 1:
        raise AssertionError(f"expected one RAG telemetry event, got {len(events)}")

    result = {
        "pass_name": pass_name,
        "duration_seconds": duration_seconds,
        "stage_trace_count": len(stage_traces),
        "stage_trace_summary": _stage_trace_summary(stage_traces),
        "stage_traces": stage_traces,
        "event": events[0],
        "rag_result": rag_result,
    }
    return result


def _event_recorder(
    events: list[dict[str, Any]],
):
    """Build a production RAG telemetry recorder for one pass.

    Args:
        events: Mutable event list local to the pass.

    Returns:
        Async recorder function with the same keyword-only shape as the
        production telemetry sink.
    """

    async def record_event(**kwargs: Any) -> None:
        """Record one sanitized RAG telemetry event."""

        events.append(dict(kwargs))

    return record_event


def _capability_request(case: CacheReviewCase) -> dict[str, object]:
    """Return the production capability request shape represented by a case."""

    request = {
        "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
        "capability_kind": "local_context_recall",
        "objective": case.objective,
        "reason": f"live LLM cache review case {case.case_id}",
        "priority": "now",
    }
    return request


def _state_for_case(case: CacheReviewCase) -> dict[str, object]:
    """Build the production-like persona state for a cache review case."""

    state = _production_persona_state(
        message_text=case.message_text,
        local_time_context=case.local_time_context,
        chat_history_recent=case.chat_history_recent,
        chat_history_wide=case.chat_history_wide,
    )
    return state


def _assert_required_fragments(
    rag_result: dict[str, Any],
    required_fragments: list[str],
) -> None:
    """Assert that required source-owned literals survived projection."""

    rendered = json.dumps(rag_result, ensure_ascii=False)
    for fragment in required_fragments:
        assert fragment in rendered


def _stats_delta(
    after: dict[str, Any],
    before: dict[str, Any],
) -> dict[str, int | float]:
    """Return numeric Cache2 runtime counter deltas."""

    delta: dict[str, int | float] = {}
    for key in ("size", "hits", "misses", "invalidations", "evictions"):
        after_value = after[key]
        before_value = before[key]
        delta[key] = after_value - before_value
    return delta


def _agent_stats_by_name(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    """Map Cache2 agent statistics by stable agent name."""

    mapped: dict[str, dict[str, int | float]] = {}
    for row in rows:
        agent_name = str(row["agent_name"])
        mapped[agent_name] = {
            "hit_count": row["hit_count"],
            "miss_count": row["miss_count"],
            "hit_rate": row["hit_rate"],
        }
    return mapped


def _agent_stats_delta(
    after: dict[str, dict[str, int | float]],
    before: dict[str, dict[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    """Return Cache2 per-agent counter deltas."""

    delta: dict[str, dict[str, int | float]] = {}
    for agent_name, after_row in after.items():
        before_row = before.get(agent_name, {})
        before_hits = before_row.get("hit_count", 0)
        before_misses = before_row.get("miss_count", 0)
        delta[agent_name] = {
            "hit_count": after_row["hit_count"] - before_hits,
            "miss_count": after_row["miss_count"] - before_misses,
        }
    return delta


def _stage_trace_summary(
    stage_traces: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Return compact stage trace rows for quick report rendering."""

    summary: list[dict[str, object]] = []
    for trace in stage_traces:
        summary_row = {
            "stage_name": trace["stage_name"],
            "route_name": trace["route_name"],
            "model": trace["model"],
            "parsed_output": trace.get("parsed_output", {}),
        }
        summary.append(summary_row)
    return summary
