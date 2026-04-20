"""E2E probe: internal_rag_dispatcher → memory_retriever_agent → DB results.

Run:  python probe_memory_retriever.py
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import internal_rag_dispatcher
from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent

TARGET_USER_ID = "165ca534-ebb9-428a-bfc4-0a7a1f7fe5b7"
TARGET_USER_NAME = "希灵"

_BASE_STATE = {
    "character_profile": {"name": "杏山千纱"},
    "platform_bot_id": "3096776418",
    "global_user_id": TARGET_USER_ID,
    "user_name": TARGET_USER_NAME,
    "user_profile": {},
    "input_embedding": [],
    "depth": "DEEP",
    "depth_confidence": 1.0,
    "cache_hit": False,
    "trigger_dispatchers": [],
    "rag_metadata": {},
    "external_rag_next_action": "end",
    "timestamp": "2026-04-20T20:25:36+08:00",
}

SCENARIOS = [
    {
        "label": "Q1 — 第三方人物 (啾啾)",
        "user_input": "你还记得啾啾么？",
        "user_topic": "询问关于'啾啾'的记忆",
    },
    {
        "label": "Q2 — 第三方人物 (Glitch)",
        "user_input": "你觉得 Glitch 这个人怎么样？",
        "user_topic": "询问对 Glitch 的印象",
    },
    {
        "label": "Q3 — 未完成约定",
        "user_input": "你上次说要帮我查一件事，查了吗？",
        "user_topic": "询问之前的约定进展",
    },
    {
        "label": "Q4 — 一般常识 (应路由到 end)",
        "user_input": "天空为什么是蓝色的？",
        "user_topic": "科学问题",
    },
    {
        "label": "Q5 — 近期对话 (游戏/情绪)",
        "user_input": "最近我们聊了什么有趣的事？",
        "user_topic": "询问近期对话",
    },
    {
        "label": "Q6 — 用户自身记忆查询 (含bot @)",
        "user_input": "<@3096776418>  千纱再想想有没有关于我的记忆？",
        "user_topic": "你还记得我吗？有没有关于我的记忆？",
    },
]


def _wrap(text: str, width: int = 66, indent: str = "    ") -> str:
    if not text:
        return f"{indent}(empty)"
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, width=width, initial_indent=indent, subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


async def run_scenario(scenario: dict) -> None:
    label = scenario["label"]
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")
    print(f"  Input : {scenario['user_input']}")

    state = {**_BASE_STATE, "decontexualized_input": scenario["user_input"], "user_topic": scenario["user_topic"]}

    # ── Stage 1: Dispatcher ──────────────────────────────────────────────────
    dispatch_result = await internal_rag_dispatcher(state)
    next_action = dispatch_result.get("internal_rag_next_action", "end")
    task = dispatch_result.get("internal_rag_task", "")
    context = dispatch_result.get("internal_rag_context", {})
    expected = dispatch_result.get("internal_rag_expected_response", "")
    reasoning = dispatch_result.get("internal_rag_dispatcher_reasoning", "")

    print(f"\n  [Dispatcher]")
    print(f"  next_action : {next_action}")
    print(f"  reasoning   : {reasoning}")
    print(f"  task        : {task}")
    print(f"  expected    : {expected}")

    if next_action != "memory_retriever_agent":
        print(f"\n  → Routed to END (no retrieval needed)")
        return

    # Inject user identity into context (mirrors what call_memory_retriever_agent_internal_rag does)
    context["target_user_name"] = TARGET_USER_NAME
    context["target_global_user_id"] = TARGET_USER_ID

    # ── Stage 2: Memory Retriever Agent ─────────────────────────────────────
    print(f"\n  [Memory Retriever]")
    try:
        rag_result = await memory_retriever_agent(
            task=task,
            context=context,
            expected_response=expected,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    status = rag_result.get("status", "?")
    reason = rag_result.get("reason", "")
    response = rag_result.get("response", "")
    tool_used = rag_result.get("knowledge_metadata", {}).get("tool", "n/a")

    status_sym = {"complete": "✓", "partial": "~", "incomplete": "✗", "error": "✗"}.get(status, "?")
    print(f"  status  : {status_sym} {status.upper()}")
    print(f"  tool    : {tool_used}")
    print(f"  reason  : {reason}")
    print(f"\n  Response:")
    print(_wrap(response))


async def main() -> None:
    print(f"E2E Internal RAG Probe  ({len(SCENARIOS)} scenarios)")
    print(f"User: {TARGET_USER_NAME} / {TARGET_USER_ID}\n")

    for scenario in SCENARIOS:
        try:
            await run_scenario(scenario)
        except Exception as exc:
            print(f"\n  FATAL: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
