# -*- coding: utf-8 -*-
"""E2E probe for Breaking Changes B1–B5.

Tests the renamed RAG keys, depth classifier recalibration, and removal of
legacy user_facts symbols end-to-end without going through the HTTP service.

Run:
    python probe_breaking_changes.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("kazusa_ai_chatbot.nodes.persona_supervisor2_rag").setLevel(logging.INFO)

from kazusa_ai_chatbot.db import (
    db_bootstrap,
    get_character_profile,
    get_user_profile,
    resolve_global_user_id,
    close_db,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import call_rag_subgraph, _get_rag_cache
from kazusa_ai_chatbot.rag.depth_classifier import DEEP, SHALLOW, InputDepthClassifier
from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot.config import SCHEDULED_TASKS_ENABLED

_TEST_PLATFORM = "probe"
_TEST_PLATFORM_USER_ID = "probe_breaking_001"
_TEST_DISPLAY_NAME = "BreakingProbeUser"

_DEEP_MESSAGE = (
    "你还记得吗？我上次告诉你我在新西兰做软件工程师，"
    "最近压力特别大，你当时说会帮我想办法的。你现在还记得吗？"
)
_SHALLOW_MESSAGE = "你好，今天天气真不错！"

_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"
_GREY  = "\033[90m"


def _ok(msg: str) -> str:
    return f"{_GREEN}✓{_RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}✗{_RESET}  {msg}"


def _header(msg: str) -> str:
    return f"\n{_BOLD}{_CYAN}{'─'*70}{_RESET}\n{_BOLD}  {msg}{_RESET}\n{'─'*70}"


def _sub(msg: str) -> str:
    return f"  {_GREY}{msg}{_RESET}"


_checks: list[tuple[bool, str]] = []


def check(condition: bool, label: str) -> None:
    _checks.append((condition, label))
    print(_ok(label) if condition else _fail(label))


async def run_probe() -> None:
    print(_header("Breaking Changes B1–B5 E2E Probe"))

    # ── 1. Bootstrap ──────────────────────────────────────────────────
    print("\n[1] Bootstrapping …")
    await db_bootstrap()
    character_profile = await get_character_profile()
    check(bool(character_profile.get("name")), "character profile loaded")

    try:
        await mcp_manager.start()
    except Exception:
        print(_sub("MCP unavailable — tool calls skipped"))

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()

    rag_cache = await _get_rag_cache()

    # ── 2. Resolve test user ──────────────────────────────────────────
    print("\n[2] Resolving test user …")
    global_user_id = await resolve_global_user_id(
        platform=_TEST_PLATFORM,
        platform_user_id=_TEST_PLATFORM_USER_ID,
        display_name=_TEST_DISPLAY_NAME,
    )
    check(bool(global_user_id), f"test user resolved ({global_user_id})")
    user_profile = await get_user_profile(global_user_id)

    # ── 3. B4: Depth classifier — no affinity override ────────────────
    print("\n[3] B4: Depth classifier recalibration …")
    classifier = InputDepthClassifier()

    # Low affinity (was formerly forced DEEP) — should now classify by content
    low_affinity_result = await classifier.classify(
        user_input=_SHALLOW_MESSAGE,
        user_topic="greeting",
        affinity=200,
    )
    check(
        "affinity" not in low_affinity_result["reasoning"].lower() or low_affinity_result["depth"] == SHALLOW,
        f"low-affinity greeting not forced DEEP (got {low_affinity_result['depth']}, "
        f"reasoning: {low_affinity_result['reasoning'][:60]})",
    )
    print(_sub(f"  affinity=200 greeting → depth={low_affinity_result['depth']}, "
               f"dispatchers={low_affinity_result['trigger_dispatchers']}"))

    # SHALLOW dispatchers list must be empty (no user_rag dispatcher)
    check(
        low_affinity_result["trigger_dispatchers"] == [] or low_affinity_result["depth"] == DEEP,
        "SHALLOW result has empty trigger_dispatchers (B4: no user_rag dispatcher)",
    )

    # DEEP dispatchers must use new names
    deep_result = await classifier.classify(
        user_input=_DEEP_MESSAGE,
        user_topic="relationship",
        affinity=700,
    )
    print(_sub(f"  deep message → depth={deep_result['depth']}, "
               f"dispatchers={deep_result['trigger_dispatchers']}"))
    check(
        deep_result["depth"] == DEEP,
        f"memory-recall message classified DEEP (got {deep_result['depth']})",
    )
    check(
        "user_rag" not in deep_result["trigger_dispatchers"],
        "DEEP dispatchers do not contain deprecated 'user_rag'",
    )
    check(
        "input_context_rag" in deep_result["trigger_dispatchers"],
        "DEEP dispatchers contain 'input_context_rag' (B3)",
    )

    # ── 4. B1+B2+B3: call_rag_subgraph returns new keys ──────────────
    print("\n[4] B1+B2+B3: RAG subgraph research_facts keys …")
    timestamp = datetime.now(timezone.utc).isoformat()
    rag_state = {
        "timestamp": timestamp,
        "platform": _TEST_PLATFORM,
        "platform_user_id": _TEST_PLATFORM_USER_ID,
        "global_user_id": global_user_id,
        "user_name": _TEST_DISPLAY_NAME,
        "user_input": _DEEP_MESSAGE,
        "user_multimedia_input": [],
        "user_profile": user_profile,
        "platform_bot_id": "probe_bot",
        "bot_name": character_profile.get("name", "Bot"),
        "character_profile": character_profile,
        "platform_channel_id": "probe_channel",
        "channel_name": "probe",
        "chat_history": [],
        "should_respond": True,
        "reason_to_respond": "probe",
        "use_reply_feature": False,
        "channel_topic": "probe",
        "user_topic": "关系回忆",
        "decontexualized_input": _DEEP_MESSAGE,
        "debug_modes": {},
    }

    rag_result = await call_rag_subgraph(rag_state)
    research_facts = rag_result.get("research_facts") or {}
    research_metadata = (rag_result.get("research_metadata") or [{}])[0]

    print(_sub(f"  research_facts keys: {list(research_facts.keys())}"))
    print(_sub(f"  depth: {research_metadata.get('depth')}, cache_hit: {research_metadata.get('cache_hit')}"))
    print(_sub(f"  sources_used: {research_metadata.get('rag_sources_used')}"))

    # B1: user_image present, user_rag_finalized absent
    check("user_image" in research_facts, "B1: research_facts has 'user_image' key")
    check("user_rag_finalized" not in research_facts, "B1: 'user_rag_finalized' removed from research_facts")

    # B2: character_image present
    check("character_image" in research_facts, "B2: research_facts has 'character_image' key")

    # B3: input_context_results present, internal_rag_results absent
    check("input_context_results" in research_facts, "B3: research_facts has 'input_context_results' key")
    check("internal_rag_results" not in research_facts, "B3: 'internal_rag_results' removed from research_facts")

    # confidence_scores use new names
    conf_scores = research_metadata.get("confidence_scores") or {}
    check("user_rag" not in conf_scores, "B1: confidence_scores no longer has 'user_rag'")
    check("input_context_rag" in conf_scores or research_metadata.get("cache_hit"), "B3: confidence_scores has 'input_context_rag'")

    if research_facts.get("user_image"):
        print(_sub(f"  user_image preview: {research_facts['user_image'][:120]}"))
    if research_facts.get("input_context_results"):
        print(_sub(f"  input_context_results preview: {research_facts['input_context_results'][:120]}"))

    # ── 5. B5: Legacy symbols absent from db module ───────────────────
    print("\n[5] B5: Legacy symbols removed from kazusa_ai_chatbot.db …")
    import kazusa_ai_chatbot.db as db_module
    for sym in ("get_user_facts", "upsert_user_facts", "overwrite_user_facts",
                "enable_user_facts_vector_index", "search_users_by_facts"):
        check(
            not hasattr(db_module, sym),
            f"B5: '{sym}' no longer exported from db module",
        )

    # ── 6. Full pipeline smoke test ───────────────────────────────────
    print("\n[6] Full pipeline smoke test …")
    im_state = {
        "timestamp": timestamp,
        "platform": _TEST_PLATFORM,
        "platform_user_id": _TEST_PLATFORM_USER_ID,
        "global_user_id": global_user_id,
        "user_name": _TEST_DISPLAY_NAME,
        "user_input": _DEEP_MESSAGE,
        "user_multimedia_input": [],
        "user_profile": user_profile,
        "platform_bot_id": "probe_bot",
        "bot_name": character_profile.get("name", "Bot"),
        "character_profile": character_profile,
        "platform_channel_id": "probe_channel",
        "channel_name": "probe",
        "chat_history": [],
        "should_respond": True,
        "reason_to_respond": "probe",
        "use_reply_feature": False,
        "channel_topic": "probe",
        "user_topic": "关系回忆",
        "debug_modes": {},
    }

    pipeline_result = await persona_supervisor2(im_state)
    final_dialog = pipeline_result.get("final_dialog") or []
    check(bool(final_dialog), "full pipeline produced a dialog response")
    if final_dialog:
        print(_sub(f"  response: {json.dumps(final_dialog, ensure_ascii=False)[:200]}"))

    # ── Summary ───────────────────────────────────────────────────────
    print(_header("Summary"))
    passed = sum(1 for ok, _ in _checks if ok)
    total = len(_checks)
    print(f"  {passed}/{total} checks passed\n")
    for ok, label in _checks:
        print(_ok(label) if ok else _fail(label))

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()
    await rag_cache.shutdown()
    await mcp_manager.stop()
    await close_db()


if __name__ == "__main__":
    asyncio.run(run_probe())
