"""Thin wrapper agent for retrieving a user's hydrated profile image bundle."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.agents.user_image_retriever_agent import user_image_retriever_agent
from kazusa_ai_chatbot.db import get_text_embedding
from kazusa_ai_chatbot.rag.depth_classifier import DEEP
from kazusa_ai_chatbot.utils import parse_llm_json_output


def _walk_for_global_user_id(value: Any) -> str:
    """Recursively scan nested values for the first non-empty ``global_user_id``.

    Args:
        value: Arbitrary nested structure built from dicts, lists, or strings.

    Returns:
        The first non-empty ``global_user_id`` found, otherwise an empty string.
    """
    if isinstance(value, dict):
        candidate = str(value.get("global_user_id", "")).strip()
        if candidate:
            return candidate
        for nested in value.values():
            resolved = _walk_for_global_user_id(nested)
            if resolved:
                return resolved
        return ""

    if isinstance(value, list):
        for item in value:
            resolved = _walk_for_global_user_id(item)
            if resolved:
                return resolved
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.startswith("{") or stripped.startswith("["):
            parsed = parse_llm_json_output(stripped)
            if parsed:
                return _walk_for_global_user_id(parsed)
        return ""

    return ""


def _extract_global_user_id_from_known_facts(context: dict) -> str:
    """Extract a resolved ``global_user_id`` from ``context["known_facts"]``.

    Args:
        context: Agent execution context assembled by the outer-loop supervisor.

    Returns:
        The first resolved ``global_user_id`` found in prior slot facts, or an
        empty string when no such identifier is available.
    """
    known_facts = context.get("known_facts", [])
    return _walk_for_global_user_id(known_facts)


async def user_profile_agent(
    task: str,
    context: dict,
    max_attempts: int = 3,
) -> dict:
    """Retrieve a hydrated user profile bundle for a previously resolved user.

    Args:
        task: Slot description from the outer-loop supervisor.
        context: Runtime hints plus ``known_facts`` from prior slots.
        max_attempts: Unused compatibility parameter kept for the shared agent
            contract.

    Returns:
        A standard supervisor2 agent result dict. ``result`` contains the
        serialized hydrated profile returned by ``user_image_retriever_agent``.
    """
    del max_attempts

    global_user_id = _extract_global_user_id_from_known_facts(context)
    if not global_user_id:
        return {
            "resolved": False,
            "result": "No resolved global_user_id found in context['known_facts'].",
            "attempts": 1,
        }

    input_embedding = await get_text_embedding(task)
    hydrated_profile, _memory_blocks = await user_image_retriever_agent(
        global_user_id,
        user_profile=context.get("user_profile"),
        input_embedding=input_embedding,
        depth=DEEP,
    )

    return {
        "resolved": bool(hydrated_profile),
        "result": hydrated_profile,
        "attempts": 1,
    }


async def test_main() -> None:
    """Run a manual smoke check for the thin user-profile wrapper agent."""
    result = await user_profile_agent(
        task="读取这个用户的画像信息",
        context={
            "known_facts": [
                {
                    "slot": "人物指代: 解析'他'指的是谁",
                    "resolved": True,
                    "result": json.dumps(
                        {"display_name": "小钳子", "global_user_id": "263c883d-aeff-4e0b-a758-6f69186ae8ec"},
                        ensure_ascii=False,
                    ),
                }
            ]
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_main())
