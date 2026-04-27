"""RAG helper agent: retrieve a hydrated user profile bundle."""

from __future__ import annotations

import json
from typing import Any

from kazusa_ai_chatbot.agents.user_image_retriever_agent import user_image_retriever_agent
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import get_character_profile, get_text_embedding
from kazusa_ai_chatbot.rag.cache2_events import CacheDependency
from kazusa_ai_chatbot.rag.cache2_policy import (
    USER_PROFILE_CACHE_NAME,
    build_character_profile_cache_key,
    build_user_profile_cache_key,
    build_user_profile_dependencies,
)
from kazusa_ai_chatbot.rag.depth_classifier import DEEP
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
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


def _extract_global_user_id_from_known_facts(context: dict[str, Any]) -> str:
    """Extract a resolved ``global_user_id`` from ``context["known_facts"]``.

    Args:
        context: Agent execution context assembled by the outer-loop supervisor.

    Returns:
        The first resolved ``global_user_id`` found in prior slot facts, or an
        empty string when no such identifier is available.
    """
    known_facts = context.get("known_facts", [])
    return _walk_for_global_user_id(known_facts)


def _public_character_profile(character_profile: dict[str, Any]) -> dict[str, Any]:
    """Select character profile fields that are safe and useful for RAG.

    Args:
        character_profile: Full character profile/state document from the DB.

    Returns:
        Public character facts plus the character self-image when present.
    """
    public_fields = (
        "name",
        "description",
        "gender",
        "age",
        "birthday",
        "backstory",
        "self_image",
    )
    public_profile: dict[str, Any] = {}
    for key in public_fields:
        value = character_profile.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        public_profile[key] = value
    return public_profile


class UserProfileAgent(BaseRAGHelperAgent):
    """RAG helper agent that retrieves a hydrated user profile bundle.

    Reads the ``global_user_id`` resolved by a prior slot from ``known_facts``
    and returns the full user-image profile bundle.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="user_profile_agent",
            cache_name=USER_PROFILE_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Retrieve a hydrated user profile bundle for a previously resolved user.

        Args:
            task: Slot description from the outer-loop supervisor.
            context: Runtime hints plus ``known_facts`` from prior slots.
            max_attempts: Unused; kept for interface compatibility.

        Returns:
            Dict with resolved (bool), result (hydrated profile or error string),
            and attempts count.
        """
        del max_attempts

        global_user_id = _extract_global_user_id_from_known_facts(context)
        if not global_user_id:
            return self.with_cache_status(
                {
                    "resolved": False,
                    "result": "No resolved global_user_id found in context['known_facts'].",
                    "attempts": 1,
                },
                hit=False,
                reason="skipped_missing_global_user_id",
            )

        # Dynamically switch the profile to fetch, as character's profile may include sensitive runtime data which is
        #   not supposed to be disclosed to LLM. 
        if global_user_id == CHARACTER_GLOBAL_USER_ID:
            cache_key = build_character_profile_cache_key(global_user_id)
        else:
            cache_key = build_user_profile_cache_key(global_user_id)

        cached = await self.read_cache(cache_key)
        if cached is not None:
            return self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )

        if global_user_id == CHARACTER_GLOBAL_USER_ID:
            character_profile = await get_character_profile()
            hydrated_profile = _public_character_profile(dict(character_profile))
            dependencies = [CacheDependency(source="character_state")]
            metadata = {"profile_source": "character_state"}
        else:
            input_embedding = await get_text_embedding(task)
            hydrated_profile, _memory_blocks = await user_image_retriever_agent(
                global_user_id,
                user_profile=context.get("user_profile"),
                input_embedding=input_embedding,
                depth=DEEP,
            )
            dependencies = build_user_profile_dependencies(global_user_id)
            metadata = {"profile_source": "user_profile"}

        if hydrated_profile and isinstance(hydrated_profile, dict):
            await self.write_cache(
                cache_key=cache_key,
                result=hydrated_profile,
                dependencies=dependencies,
                metadata=metadata,
            )

        return self.with_cache_status(
            {
                "resolved": bool(hydrated_profile),
                "result": hydrated_profile,
                "attempts": 1,
            },
            hit=False,
            reason="miss_stored" if hydrated_profile else "miss_unresolved",
            cache_key=cache_key,
        )


async def _test_main() -> None:
    """Run a manual smoke check for UserProfileAgent."""
    agent = UserProfileAgent()
    result = await agent.run(
        task="读取这个用户的画像信息",
        context={
            "known_facts": [
                {
                    "slot": "人物指代",
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

    result = await agent.run(
        task="读取这个用户的画像信息",
        context={
            "known_facts": [
                {
                    "slot": "人物指代",
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

    asyncio.run(_test_main())
