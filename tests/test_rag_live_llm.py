"""Phase 7 — Live-LLM integration tests for the RAG resolution pipeline.

These tests exercise the real planner and resolution nodes against the
configured LLM endpoint.  They do NOT mock LLMs — the purpose is to
verify actual dispatcher performance with the user's (potentially weak) LLM.

Tests are marked ``live_llm`` and skipped when the endpoint is unreachable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pytest

from kazusa_ai_chatbot.config import LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_resolution import (
    continuation_resolver,
    entity_grounder,
    rag_planner,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor import (
    rag_supervisor_evaluator,
)

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm


# ── Shared helpers ────────────────────────────────────────────────


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {LLM_BASE_URL}")
    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    await _skip_if_llm_unavailable()


def _make_rag_state(
    decontexualized_input: str,
    *,
    user_name: str = "TestUser",
    global_user_id: str = "test-user-id-001",
    channel_topic: str = "",
    chat_history_recent: list[dict] | None = None,
    continuation_context: dict | None = None,
    retrieval_plan: dict | None = None,
    resolved_entities: list[dict] | None = None,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "discord",
        "platform_channel_id": "test-channel-001",
        "platform_message_id": "test-msg-001",
        "decontexualized_input": decontexualized_input,
        "channel_topic": channel_topic,
        "input_context_to_timestamp": datetime.now(timezone.utc).isoformat(),
        "chat_history_recent": chat_history_recent or [],
        "user_name": user_name,
        "global_user_id": global_user_id,
        "platform_bot_id": "test-bot-001",
        "character_profile": {"name": "千纱"},
        "user_profile": {"affinity": 700},
        "input_embedding": [],
        "depth": "DEEP",
        "depth_confidence": 0.9,
        "cache_hit": False,
        "trigger_dispatchers": [],
        "rag_metadata": {},
        "continuation_context": continuation_context or {},
        "retrieval_plan": retrieval_plan or {},
        "resolved_entities": resolved_entities or [],
        "retrieval_ledger": {},
        "external_rag_next_action": "end",
        "external_rag_task": "",
        "external_rag_context": {},
        "external_rag_expected_response": "",
        "external_rag_results": [],
        "external_rag_is_empty_result": False,
        "input_context_next_action": "end",
        "input_context_task": "",
        "input_context_context": {},
        "input_context_expected_response": "",
        "input_context_results": [],
        "input_context_is_empty_result": False,
        "channel_recent_entity_results": "",
        "third_party_profile_results": "",
        "entity_knowledge_results": "",
        "entity_resolution_notes": "",
    }


def _log_result(label: str, result: dict) -> None:
    logger.info("%s => %r", label, result)


# ══════════════════════════════════════════════════════════════════
# Phase 1 — Continuation Resolver
# ══════════════════════════════════════════════════════════════════


async def test_continuation_resolver_independent_query(ensure_live_llm) -> None:
    """S8-like: a self-contained question needs no continuation resolution."""
    state = _make_rag_state(
        "你最近心情怎么样？",
        chat_history_recent=[
            {"role": "user", "content": "早上好", "display_name": "TestUser"},
            {"role": "assistant", "content": "嗯……早上好。"},
        ],
    )
    result = await continuation_resolver(state)
    _log_result("continuation_resolver.independent", result)

    ctx = result["continuation_context"]
    assert isinstance(ctx, dict)
    assert "needs_context_resolution" in ctx
    assert ctx["needs_context_resolution"] is False, (
        f"Independent query should NOT need continuation resolution: {ctx}"
    )


async def test_continuation_resolver_bare_fragment(ensure_live_llm) -> None:
    """S7: bare continuation fragment must be resolved."""
    state = _make_rag_state(
        "那后来呢？",
        chat_history_recent=[
            {"role": "user", "content": "昨天啾啾说要来找你玩", "display_name": "TestUser"},
            {"role": "assistant", "content": "是吗，她说了什么时候来？"},
            {"role": "user", "content": "好像是周末吧", "display_name": "TestUser"},
        ],
        channel_topic="闲聊",
    )
    result = await continuation_resolver(state)
    _log_result("continuation_resolver.fragment", result)

    ctx = result["continuation_context"]
    assert ctx["needs_context_resolution"] is True, (
        f"Bare fragment should need resolution: {ctx}"
    )
    assert ctx["resolved_task"].strip(), f"resolved_task should be non-empty: {ctx}"
    assert ctx["confidence"] >= 0.3, f"Confidence too low: {ctx}"


# ══════════════════════════════════════════════════════════════════
# Phase 1 — RAG Planner
# ══════════════════════════════════════════════════════════════════


async def test_rag_planner_s1_explicit_third_party(ensure_live_llm) -> None:
    """S1: explicit named third-party → CASCADED with entity."""
    state = _make_rag_state(
        "啾啾之前跟你说过什么有趣的？",
        user_name="EAMARS",
        chat_history_recent=[
            {"role": "user", "content": "啾啾最近怎么样", "display_name": "EAMARS", "global_user_id": "eamars-id"},
            {"role": "assistant", "content": "她前几天来过"},
            {"role": "user", "content": "那她说了什么？", "display_name": "EAMARS", "global_user_id": "eamars-id"},
        ],
        continuation_context={
            "needs_context_resolution": False,
            "resolved_task": "啾啾之前跟你说过什么有趣的？",
            "confidence": 1.0,
        },
    )
    result = await rag_planner(state)
    _log_result("rag_planner.s1", result)

    plan = result["retrieval_plan"]
    assert plan["retrieval_mode"] != "NONE", f"Should not be NONE for third-party query: {plan}"

    entities = plan.get("entities", [])
    surface_forms = [e.get("surface_form", "") for e in entities]
    assert any("啾" in sf for sf in surface_forms), (
        f"Should identify 啾啾 as entity: {entities}"
    )


async def test_rag_planner_s8_pure_current_user(ensure_live_llm) -> None:
    """S8: pure current-user mood query → NONE or CURRENT_USER_STABLE, no entities."""
    state = _make_rag_state(
        "你最近心情怎么样？",
        continuation_context={
            "needs_context_resolution": False,
            "resolved_task": "你最近心情怎么样？",
            "confidence": 1.0,
        },
    )
    result = await rag_planner(state)
    _log_result("rag_planner.s8", result)

    plan = result["retrieval_plan"]
    assert plan["retrieval_mode"] in ("NONE", "CURRENT_USER_STABLE"), (
        f"Mood query should not trigger complex retrieval: {plan}"
    )


async def test_rag_planner_s10_external_knowledge(ensure_live_llm) -> None:
    """S10: pure external knowledge → EXTERNAL_KNOWLEDGE."""
    state = _make_rag_state(
        "你知道量子纠缠是什么意思吗？",
        continuation_context={
            "needs_context_resolution": False,
            "resolved_task": "你知道量子纠缠是什么意思吗？",
            "confidence": 1.0,
        },
    )
    result = await rag_planner(state)
    _log_result("rag_planner.s10", result)

    plan = result["retrieval_plan"]
    active = plan.get("active_sources", [])
    assert "EXTERNAL_KNOWLEDGE" in active, (
        f"Should include EXTERNAL_KNOWLEDGE for quantum entanglement: {plan}"
    )


async def test_rag_planner_s9_stable_profile(ensure_live_llm) -> None:
    """S9: stable profile recall → THIRD_PARTY_PROFILE, not CHANNEL_RECENT_ENTITY."""
    state = _make_rag_state(
        "你对啾啾的整体印象是什么？",
        chat_history_recent=[
            {"role": "user", "content": "聊聊啾啾吧", "display_name": "EAMARS", "global_user_id": "eamars-id"},
        ],
        continuation_context={
            "needs_context_resolution": False,
            "resolved_task": "你对啾啾的整体印象是什么？",
            "confidence": 1.0,
        },
    )
    result = await rag_planner(state)
    _log_result("rag_planner.s9", result)

    plan = result["retrieval_plan"]
    active = plan.get("active_sources", [])
    assert "THIRD_PARTY_PROFILE" in active, (
        f"Should include THIRD_PARTY_PROFILE for stable impression: {plan}"
    )


# ══════════════════════════════════════════════════════════════════
# Phase 1 — Entity Grounder
# ══════════════════════════════════════════════════════════════════


async def test_entity_grounder_exact_match(ensure_live_llm) -> None:
    """Entity grounder resolves known display name from recent history."""
    state = _make_rag_state(
        "啾啾说了什么",
        chat_history_recent=[
            {"role": "user", "content": "hello", "display_name": "啾啾", "global_user_id": "jiujiu-id-001"},
            {"role": "user", "content": "hi", "display_name": "EAMARS", "global_user_id": "eamars-id"},
        ],
        retrieval_plan={
            "retrieval_mode": "CASCADED",
            "active_sources": ["CHANNEL_RECENT_ENTITY", "THIRD_PARTY_PROFILE"],
            "entities": [
                {"surface_form": "啾啾", "entity_type": "person", "resolution_confidence": 0.5}
            ],
        },
    )
    result = await entity_grounder(state)
    _log_result("entity_grounder.exact", result)

    resolved = result["resolved_entities"]
    assert len(resolved) == 1
    assert resolved[0]["resolved_global_user_id"] == "jiujiu-id-001"
    assert resolved[0]["resolution_confidence"] >= 0.9


async def test_entity_grounder_unresolved(ensure_live_llm) -> None:
    """Unresolved entity when no participants match."""
    state = _make_rag_state(
        "小李最近怎么样",
        chat_history_recent=[
            {"role": "user", "content": "hello", "display_name": "EAMARS", "global_user_id": "eamars-id"},
        ],
        retrieval_plan={
            "retrieval_mode": "CASCADED",
            "active_sources": ["CHANNEL_RECENT_ENTITY"],
            "entities": [
                {"surface_form": "小李", "entity_type": "person", "resolution_confidence": 0.3}
            ],
        },
    )
    result = await entity_grounder(state)
    _log_result("entity_grounder.unresolved", result)

    resolved = result["resolved_entities"]
    assert len(resolved) == 1
    assert resolved[0]["resolved_global_user_id"] == ""
    assert "unresolved" in result["entity_resolution_notes"].lower()


# ══════════════════════════════════════════════════════════════════
# Phase 5 — RAG Supervisor Evaluator
# ══════════════════════════════════════════════════════════════════


async def test_evaluator_sufficient_simple_query(ensure_live_llm) -> None:
    """Simple query with no entities → evaluator returns sufficient."""
    state = _make_rag_state(
        "你最近心情怎么样？",
        retrieval_plan={
            "retrieval_mode": "NONE",
            "active_sources": [],
            "entities": [],
        },
    )
    result = await rag_supervisor_evaluator(state)
    _log_result("evaluator.sufficient", result)

    assert result["needs_repair"] is False
    assert result["evaluation"]["verdict"] == "sufficient"


async def test_evaluator_respects_max_repair_cap(ensure_live_llm) -> None:
    """Evaluator must return sufficient when repair_pass >= 1."""
    state = _make_rag_state(
        "她之前说的那个你还记得吗？",
        retrieval_plan={
            "retrieval_mode": "CASCADED",
            "active_sources": ["CHANNEL_RECENT_ENTITY"],
            "entities": [{"surface_form": "她", "entity_type": "person"}],
        },
    )
    state["rag_metadata"] = {"repair_pass": 1}
    result = await rag_supervisor_evaluator(state)
    _log_result("evaluator.capped", result)

    assert result["needs_repair"] is False, (
        f"Evaluator must respect max repair cap: {result}"
    )


async def test_evaluator_cascaded_with_results(ensure_live_llm) -> None:
    """Cascaded query with actual results → evaluator checks coverage via LLM."""
    state = _make_rag_state(
        "啾啾之前跟你说过什么有趣的？",
        retrieval_plan={
            "retrieval_mode": "CASCADED",
            "active_sources": ["CHANNEL_RECENT_ENTITY", "THIRD_PARTY_PROFILE"],
            "entities": [
                {"surface_form": "啾啾", "entity_type": "person", "resolution_confidence": 0.95}
            ],
        },
        resolved_entities=[
            {
                "surface_form": "啾啾",
                "entity_type": "person",
                "resolved_global_user_id": "jiujiu-id-001",
                "resolution_confidence": 0.95,
                "resolution_method": "exact_display_name",
            },
        ],
    )
    state["channel_recent_entity_results"] = "啾啾在2天前说了一些关于周末计划的话"
    state["third_party_profile_results"] = "啾啾是一个活泼的朋友"

    result = await rag_supervisor_evaluator(state)
    _log_result("evaluator.cascaded_with_results", result)

    eval_data = result["evaluation"]
    assert eval_data["verdict"] in ("sufficient", "needs_repair")
    assert isinstance(eval_data.get("coverage_score", 0), (int, float))


# ══════════════════════════════════════════════════════════════════
# End-to-end resolution chain
# ══════════════════════════════════════════════════════════════════


async def test_full_resolution_chain_s1(ensure_live_llm) -> None:
    """S1 end-to-end: continuation → planner → grounder for explicit third-party."""
    state = _make_rag_state(
        "啾啾之前跟你说过什么有趣的？",
        user_name="EAMARS",
        chat_history_recent=[
            {"role": "user", "content": "hello", "display_name": "啾啾", "global_user_id": "jiujiu-id-001"},
            {"role": "user", "content": "那她说了什么？", "display_name": "EAMARS", "global_user_id": "eamars-id"},
        ],
    )

    # Step 1: Continuation resolver
    cr_result = await continuation_resolver(state)
    _log_result("chain.s1.continuation", cr_result)
    state.update(cr_result)

    # Step 2: Planner
    planner_result = await rag_planner(state)
    _log_result("chain.s1.planner", planner_result)
    state.update(planner_result)

    plan = planner_result["retrieval_plan"]
    assert plan["retrieval_mode"] != "NONE", f"S1 should trigger retrieval: {plan}"

    entities = plan.get("entities", [])
    assert entities, f"S1 should identify entities: {plan}"

    # Step 3: Entity grounder
    grounder_result = await entity_grounder(state)
    _log_result("chain.s1.grounder", grounder_result)

    resolved = grounder_result["resolved_entities"]
    assert resolved, f"S1 grounder should produce resolved entities: {grounder_result}"

    jiujiu_resolved = [e for e in resolved if "啾" in e.get("surface_form", "")]
    assert jiujiu_resolved, f"啾啾 should be among resolved entities: {resolved}"
    assert jiujiu_resolved[0]["resolved_global_user_id"] == "jiujiu-id-001", (
        f"啾啾 should resolve to jiujiu-id-001: {jiujiu_resolved}"
    )


async def test_full_resolution_chain_s7_continuation(ensure_live_llm) -> None:
    """S7: bare continuation must produce a resolved_task that replaces raw input."""
    state = _make_rag_state(
        "那后来呢？",
        chat_history_recent=[
            {"role": "user", "content": "昨天啾啾来找你说了什么", "display_name": "TestUser"},
            {"role": "assistant", "content": "她说想周末一起去看电影"},
            {"role": "user", "content": "那后来呢？", "display_name": "TestUser"},
        ],
        channel_topic="闲聊",
    )

    cr_result = await continuation_resolver(state)
    _log_result("chain.s7.continuation", cr_result)
    state.update(cr_result)

    ctx = cr_result["continuation_context"]
    assert ctx["needs_context_resolution"] is True, f"S7 fragment needs resolution: {ctx}"
    assert ctx["resolved_task"] != "那后来呢？", (
        f"resolved_task should not be the raw fragment: {ctx}"
    )

    planner_result = await rag_planner(state)
    _log_result("chain.s7.planner", planner_result)

    plan = planner_result["retrieval_plan"]
    assert plan["retrieval_mode"] != "NONE", f"S7 resolved query should trigger retrieval: {plan}"
