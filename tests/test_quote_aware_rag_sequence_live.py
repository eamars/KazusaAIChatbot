"""Live LLM evals for quote-aware RAG sequencing."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import httpx
import pytest
import pytest_asyncio

from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, db_bootstrap, get_character_profile
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.rag import quote_aware_sequence as quote_module
from kazusa_ai_chatbot.time_boundary import (
    build_turn_clock_from_storage_utc,
    storage_utc_now_iso,
)
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


async def _skip_if_llm_unavailable() -> None:
    """Skip the live eval when the configured RAG planner endpoint is down."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{RAG_PLANNER_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"RAG planner LLM endpoint is unavailable: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"RAG planner LLM endpoint returned server error "
            f"{response.status_code}: {RAG_PLANNER_LLM_BASE_URL}"
        )


def _character_name_for_supervisor(profile: dict[str, Any]) -> str:
    """Choose the compact display name used by live RAG prompts."""
    raw_name = str(profile.get("name", "")).strip()
    if "千纱" in raw_name:
        return_value = "千纱"
    else:
        return_value = raw_name
    return return_value


def _build_context(fresh_query: str, reply_excerpt: str) -> dict[str, Any]:
    """Build one live RAG context with pass-local cache fields."""
    storage_timestamp_utc = storage_utc_now_iso()
    turn_clock = build_turn_clock_from_storage_utc(storage_timestamp_utc)
    context = {
        "platform": "qq",
        "platform_channel_id": "quote-aware-live",
        "current_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "global_user_id": "",
        "user_name": "",
        "prompt_message_context": {
            "body_text": fresh_query,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "reply_context": {
            "reply_excerpt": reply_excerpt,
        },
    }
    return context


def _pass_name_for_query(query: str, fresh_query: str) -> str:
    """Infer the wrapper pass name from the model-facing query text."""
    if query.startswith("Research factual content"):
        return_value = "quote_grounding"
    elif query.startswith("Resolve the current user message"):
        return_value = "combined_retry"
    elif query == fresh_query or query.startswith("Known evidence from the quoted"):
        return_value = "fresh_after_quote"
    else:
        return_value = "unknown_pass"
    return return_value


@pytest_asyncio.fixture(scope="module")
async def live_quote_aware_env():
    """Prepare live LLM, DB, and MCP resources for one-at-a-time evals."""
    await _skip_if_llm_unavailable()
    await db_bootstrap()
    character_profile = await get_character_profile()
    if not character_profile.get("name"):
        pytest.fail("Character profile is missing from MongoDB.")

    try:
        await mcp_manager.start()
    except Exception as exc:
        logger.exception(f"MCP manager failed to start for live quote eval: {exc}")

    yield {
        "character_name": _character_name_for_supervisor(character_profile),
    }

    try:
        await mcp_manager.stop()
    except Exception as exc:
        logger.exception(f"MCP manager failed to stop for live quote eval: {exc}")

    await close_db()


async def _run_live_quote_aware_case(
    monkeypatch: pytest.MonkeyPatch,
    live_quote_aware_env: dict[str, Any],
    *,
    case_id: str,
    fresh_query: str,
    reply_excerpt: str,
    note: str,
) -> dict[str, Any]:
    """Run one live quote-aware RAG case and write a trace artifact."""
    passes: list[dict[str, Any]] = []
    real_call_rag_supervisor = quote_module.call_rag_supervisor

    async def _capturing_rag_supervisor(
        *,
        original_query: str,
        character_name: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        result = await real_call_rag_supervisor(
            original_query=original_query,
            character_name=character_name,
            context=context,
        )
        passes.append(
            {
                "pass_name": _pass_name_for_query(original_query, fresh_query),
                "query": original_query,
                "context_body_text": context["prompt_message_context"]["body_text"],
                "unknown_slots": result["unknown_slots"],
                "known_facts": result["known_facts"],
                "answer": result["answer"],
                "loop_count": result["loop_count"],
            }
        )
        return result

    monkeypatch.setattr(
        quote_module,
        "call_rag_supervisor",
        _capturing_rag_supervisor,
    )

    result = await quote_module.call_quote_aware_rag_supervisor(
        fresh_query=fresh_query,
        reply_context={"reply_excerpt": reply_excerpt},
        character_name=live_quote_aware_env["character_name"],
        context=_build_context(fresh_query, reply_excerpt),
    )
    pass_path = [rag_pass["pass_name"] for rag_pass in passes]
    trace_path = write_llm_trace(
        "quote_aware_rag_sequence_live",
        case_id,
        {
            "fresh_query": fresh_query,
            "reply_excerpt": reply_excerpt,
            "note": note,
            "pass_path": pass_path,
            "passes": passes,
            "result": result,
            "judgment": (
                "manual_review_required_for_quote_prefix_slot_quality; "
                "MCP or web-search miss is an environment blocker only"
            ),
        },
    )

    logger.info(
        f"QUOTE_AWARE_RAG_LIVE case={case_id} trace={trace_path} "
        f"pass_path={pass_path} answer={result['answer']!r} "
        f"unknown_slots={json.dumps(result['unknown_slots'], ensure_ascii=False)} "
        f"known_facts={json.dumps(result['known_facts'], ensure_ascii=False, default=str)}"
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"answer", "known_facts", "unknown_slots", "loop_count"}
    assert len(passes) >= 2
    assert pass_path[0] == "quote_grounding"
    for rag_pass in passes:
        assert rag_pass["context_body_text"] == rag_pass["query"]

    return result


async def test_quote_aware_rag_live_quote_hit(
    monkeypatch,
    live_quote_aware_env: dict[str, Any],
) -> None:
    """Quoted entity anchors should be searched before the fresh follow-up."""
    result = await _run_live_quote_aware_case(
        monkeypatch,
        live_quote_aware_env,
        case_id="quote_hit",
        fresh_query='这句话里的 1.5T 皮卡是什么意思？',
        reply_excerpt='比亚迪 Shark 6 是一款搭载 1.5T 插电混动系统的皮卡。',
        note="Quote should anchor BYD Shark 6 before explaining 1.5T pickup.",
    )

    assert isinstance(result["answer"], str)


async def test_quote_aware_rag_live_quote_miss(
    monkeypatch,
    live_quote_aware_env: dict[str, Any],
) -> None:
    """A quote miss should not prevent a self-contained fresh query."""
    result = await _run_live_quote_aware_case(
        monkeypatch,
        live_quote_aware_env,
        case_id="quote_miss",
        fresh_query='1.5T 皮卡是什么意思？',
        reply_excerpt='月亮饼干引擎会让皮卡飞起来。',
        note=(
            "Quote can miss; fresh query is self-contained and should run "
            "without relying on quote facts."
        ),
    )

    assert isinstance(result["known_facts"], list)


async def test_quote_aware_rag_live_combined_retry(
    monkeypatch,
    live_quote_aware_env: dict[str, Any],
) -> None:
    """Vague fresh text after quote and fresh misses should retry once."""
    result = await _run_live_quote_aware_case(
        monkeypatch,
        live_quote_aware_env,
        case_id="combined_retry",
        fresh_query='这个是真的吗？',
        reply_excerpt='玄铁推进器 9999Z 有 12345 牛米。',
        note=(
            "Both first passes are expected to miss or return no substantive "
            "facts, causing exactly one combined retry."
        ),
    )

    assert isinstance(result["unknown_slots"], list)
