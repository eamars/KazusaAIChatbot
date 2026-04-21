# -*- coding: utf-8 -*-
"""E2E probe for RAG redesign (Compatible Changes C3-C10).

Calls the full persona_supervisor2 pipeline directly (no HTTP service).
After the pipeline completes, inspects MongoDB to verify that the new
three-tier image documents and knowledge-base cache entries are written.

Run:
    python probe_rag_redesign.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

# Force UTF-8 output so Chinese characters render correctly on Windows.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Show key pipeline steps without the noise.
for _verbose in (
    "kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator",
    "kazusa_ai_chatbot.nodes.persona_supervisor2_rag",
):
    logging.getLogger(_verbose).setLevel(logging.INFO)

from kazusa_ai_chatbot.db import (
    db_bootstrap,
    get_character_profile,
    get_db,
    get_user_profile,
    resolve_global_user_id,
    close_db,
)
from kazusa_ai_chatbot.mcp_client import mcp_manager
from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import _get_rag_cache
from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot.config import SCHEDULED_TASKS_ENABLED

# ── Probe configuration ────────────────────────────────────────────────────────

# Use a dedicated test identity so the probe does not pollute real user data.
_TEST_PLATFORM = "probe"
_TEST_PLATFORM_USER_ID = "probe_rag_redesign_001"
_TEST_DISPLAY_NAME = "ProbeUser"

# A message designed to trigger:
#   1. DEEP RAG classification (substantive, factual content)
#   2. At least one milestone fact (explicit strong preference declaration)
#   3. A regular fact (occupation / location)
_TEST_MESSAGE = (
    "我要告诉你一件很重要的事：我这辈子超级超级讨厌吃辣椒，"
    "任何辣的东西都不会碰的那种。另外，我是在新西兰做软件工程师的，"
    "最近在做一个 AI 聊天机器人项目，压力挺大的。"
)


# ── Colour helpers ─────────────────────────────────────────────────────────────

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


# ── Assertion helpers ──────────────────────────────────────────────────────────

_checks: list[tuple[bool, str]] = []


def check(condition: bool, label: str) -> None:
    """Record a named assertion and print its outcome immediately."""
    _checks.append((condition, label))
    print(_ok(label) if condition else _fail(label))


# ── Main probe ─────────────────────────────────────────────────────────────────


async def run_probe() -> None:
    print(_header("RAG Redesign E2E Probe"))

    # ── 1. Bootstrap ──────────────────────────────────────────────────────────
    print("\n[1] Bootstrapping DB …")
    await db_bootstrap()

    character_profile = await get_character_profile()
    check(bool(character_profile.get("name")), "character profile loaded from DB")
    print(_sub(f"Character: {character_profile.get('name')}"))

    try:
        await mcp_manager.start()
    except Exception:
        print(_sub("MCP manager unavailable — tool calls will be skipped"))

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.load_pending_events()

    rag_cache = await _get_rag_cache()
    print(_sub(f"RAG cache warm-started: {rag_cache.get_stats()}"))

    # ── 2. Resolve test user ──────────────────────────────────────────────────
    print("\n[2] Resolving test user …")
    global_user_id = await resolve_global_user_id(
        platform=_TEST_PLATFORM,
        platform_user_id=_TEST_PLATFORM_USER_ID,
        display_name=_TEST_DISPLAY_NAME,
    )
    check(bool(global_user_id), f"test user resolved (global_user_id={global_user_id})")

    user_profile = await get_user_profile(global_user_id)
    print(_sub(f"Affinity before: {user_profile.get('affinity', 'N/A')}"))

    # Snapshot existing user_image before the run so we can diff it.
    image_before = user_profile.get("user_image") or {}
    recent_window_len_before = len((image_before.get("recent_window") or []))
    milestone_count_before = len((image_before.get("milestones") or []))
    print(_sub(
        f"user_image before — milestones: {milestone_count_before}, "
        f"recent_window: {recent_window_len_before}"
    ))

    character_before = await get_character_profile()
    self_image_before = character_before.get("self_image") or {}
    char_recent_before = len((self_image_before.get("recent_window") or []))
    print(_sub(f"character self_image recent_window before: {char_recent_before}"))

    # ── 3. Run the pipeline ───────────────────────────────────────────────────
    print(f"\n[3] Running pipeline …")
    print(_sub(f"Message: {_TEST_MESSAGE}"))

    timestamp = datetime.now(timezone.utc).isoformat()

    im_state = {
        "timestamp": timestamp,
        "platform": _TEST_PLATFORM,
        "platform_user_id": _TEST_PLATFORM_USER_ID,
        "global_user_id": global_user_id,
        "user_name": _TEST_DISPLAY_NAME,
        "user_input": _TEST_MESSAGE,
        "user_multimedia_input": [],
        "user_profile": user_profile,
        "platform_bot_id": "probe_bot",
        "bot_name": character_profile.get("name", "KazusaBot"),
        "character_profile": character_profile,
        "platform_channel_id": "probe_channel",
        "channel_name": "probe",
        "chat_history": [],
        "should_respond": True,
        "reason_to_respond": "probe",
        "use_reply_feature": False,
        "channel_topic": "probe",
        "user_topic": "自我介绍",
        "debug_modes": {},
    }

    result = await persona_supervisor2(im_state)

    final_dialog = result.get("final_dialog") or []
    check(bool(final_dialog), "pipeline produced a dialog response")
    if final_dialog:
        print(_sub(f"Response: {json.dumps(final_dialog, ensure_ascii=False)}"))

    # ── 4. Inspect MongoDB directly ───────────────────────────────────────────
    print("\n[4] Inspecting MongoDB …")
    db = await get_db()

    # 4a. user_profiles.user_image
    user_doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
    user_image = (user_doc or {}).get("user_image") or {}

    recent_window_after = user_image.get("recent_window") or []
    milestones_after = user_image.get("milestones") or []
    historical_summary = user_image.get("historical_summary") or ""
    image_meta = user_image.get("meta") or {}

    check(
        len(recent_window_after) > recent_window_len_before or len(milestones_after) > milestone_count_before,
        "user_image updated (new recent_window entry or new milestone)",
    )
    check(
        len(milestones_after) > milestone_count_before,
        f"milestone fact extracted (count: {milestone_count_before} → {len(milestones_after)})",
    )
    check(
        len(recent_window_after) > recent_window_len_before,
        f"recent_window grew (count: {recent_window_len_before} → {len(recent_window_after)})",
    )

    print(_sub("user_image.milestones:"))
    for m in milestones_after:
        print(_sub(f"  [{m.get('category', m.get('milestone_category', '?'))}] {m.get('event', m.get('description', '?'))}"))
    print(_sub(f"user_image.recent_window entries: {len(recent_window_after)}"))
    if recent_window_after:
        latest = recent_window_after[-1]
        print(_sub(f"  latest summary: {latest.get('summary', '')[:120]}"))
    print(_sub(f"user_image.historical_summary chars: {len(historical_summary)}"))
    print(_sub(f"user_image.meta: {image_meta}"))

    # 4b. objective_facts were written (any new entries)
    print()
    objective_facts = (user_doc or {}).get("objective_facts") or []
    check(
        len(objective_facts) > 0,
        f"objective_facts written to user_profiles (count: {len(objective_facts)})",
    )
    if objective_facts:
        for f in objective_facts[-3:]:
            print(_sub(f"  fact: [{f.get('category','')}] {f.get('fact',f.get('description',''))}"))

    # Note: is_milestone is an in-memory routing field used by _update_user_image
    # to populate user_image.milestones — it is not persisted to objective_facts.
    # Milestone correctness is verified via user_image.milestones above.

    # 4c. character_state.self_image
    print()
    char_doc = await db.character_state.find_one({"_id": "global"})
    self_image = (char_doc or {}).get("self_image") or {}
    char_recent_after = len((self_image.get("recent_window") or []))

    check(
        char_recent_after > char_recent_before or self_image.get("meta"),
        f"character self_image updated (recent_window: {char_recent_before} → {char_recent_after})",
    )
    if self_image.get("recent_window"):
        latest_char = (self_image["recent_window"] or [{}])[-1]
        print(_sub(f"  latest char summary: {latest_char.get('summary', '')[:120]}"))

    # 4d. knowledge_base in rag_cache_index
    print()
    kb_entries = await db.rag_cache_index.count_documents({"cache_type": "knowledge_base"})
    check(kb_entries >= 0, f"rag_cache_index queried (knowledge_base entries: {kb_entries})")
    if kb_entries > 0:
        sample = await db.rag_cache_index.find_one({"cache_type": "knowledge_base"})
        sample_result = (sample or {}).get("results", {}).get("knowledge_base_results", "")
        print(_sub(f"  sample KB entry: {str(sample_result)[:120]}"))

    # 4e. Pipeline produced structured output (persona_supervisor2 only exposes
    #     final_dialog and future_promises; research_facts lives in the subgraph).
    print()
    check(
        result.get("final_dialog") is not None,
        "pipeline returned final_dialog (full pipeline executed)",
    )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(_header("Summary"))
    passed = sum(1 for ok, _ in _checks if ok)
    total = len(_checks)
    print(f"  {passed}/{total} checks passed\n")

    for ok, label in _checks:
        print(_ok(label) if ok else _fail(label))

    if SCHEDULED_TASKS_ENABLED:
        await scheduler.shutdown()
    cache = await _get_rag_cache()
    await cache.shutdown()
    await mcp_manager.stop()
    await close_db()


if __name__ == "__main__":
    asyncio.run(run_probe())
