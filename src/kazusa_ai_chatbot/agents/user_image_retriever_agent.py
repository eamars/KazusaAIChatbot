from __future__ import annotations

import hashlib
import json
import logging

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import PROFILE_MEMORY_BUDGET
from kazusa_ai_chatbot.db import build_user_profile_recall_bundle, hydrate_user_profile_with_memory_blocks
from kazusa_ai_chatbot.rag.cache import RAGCache
from kazusa_ai_chatbot.rag.depth_classifier import DEEP

logger = logging.getLogger(__name__)

_USER_PROFILE_MEMORIES_CACHE_TYPE = "user_profile_memories"
_USER_PROFILE_MEMORIES_KEY_VERSION = "v1"


def _user_profile_memories_cache_key(
    global_user_id: str,
    input_embedding: list[float],
    depth: str,
) -> str:
    if depth == DEEP and input_embedding:
        topic_hash = hashlib.sha256(
            json.dumps(input_embedding, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    else:
        topic_hash = "shallow"
    raw = f"{_USER_PROFILE_MEMORIES_CACHE_TYPE}|{_USER_PROFILE_MEMORIES_KEY_VERSION}|{global_user_id}|{topic_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strip_memory_embeddings(blocks: dict) -> dict:
    stripped = dict(blocks)
    stripped["memories"] = [
        {k: v for k, v in mem.items() if k != "embedding"}
        for mem in blocks.get("memories") or []
    ]
    return stripped


async def user_image_retriever_agent(
    global_user_id: str,
    *,
    user_profile: dict | None = None,
    input_embedding: list[float],
    depth: str,
    cache: RAGCache | None = None,
    budget: int = PROFILE_MEMORY_BUDGET,
) -> tuple[dict, dict]:
    memory_blocks: dict | None = None
    cache_key = _user_profile_memories_cache_key(global_user_id, input_embedding, depth)
    if cache is not None and global_user_id:
        try:
            hit = await cache.retrieve_if_similar_by_key(cache_key)
        except PyMongoError:
            logger.exception("user_profile_memories cache probe failed for %s", global_user_id)
        else:
            if hit is not None:
                memory_blocks = hit.get("results") or None

    if memory_blocks is None:
        _, memory_blocks = await build_user_profile_recall_bundle(
            global_user_id,
            user_profile=user_profile,
            topic_embedding=input_embedding,
            include_semantic=depth == DEEP,
            budget=budget,
        )
        if cache is not None and global_user_id:
            try:
                await cache.store_by_key(
                    cache_key=cache_key,
                    results=_strip_memory_embeddings(memory_blocks),
                    cache_type=_USER_PROFILE_MEMORIES_CACHE_TYPE,
                    global_user_id=global_user_id,
                    metadata={"depth": depth},
                )
            except PyMongoError:
                logger.exception("user_profile_memories cache store failed for %s", global_user_id)

    hydrated = hydrate_user_profile_with_memory_blocks(user_profile or {}, memory_blocks)
    return hydrated, memory_blocks
