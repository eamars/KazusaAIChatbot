"""Stage 1 / RAG subgraph with semantic cache + depth-based dispatch.

The subgraph has five wrapper phases on top of the existing dispatchers:

* **Phase 0 — Input analysis**: embed ``decontexualized_input`` and initialise
  the unified metadata bundle that accumulates through the whole RAG pass.
* **Phase 1 — Cache check**: probe ``RAGCache`` for ``objective_user_facts``,
  ``character_diary`` and ``external_knowledge`` matches. A strong hit short-
  circuits the rest of the pipeline.
* **Phase 2 — Depth classification**: ``InputDepthClassifier`` selects
  ``SHALLOW`` (no dispatchers; images from profile) or ``DEEP`` (input_context + optional external).
* **Phase 3 — Conditional dispatch**: START edges fan out only to the
  dispatchers permitted by the chosen depth (and the existing affinity gate
  for external).
* **Phase 4 — Cache storage**: the produced results are written back into
  the cache with a TTL matching their semantic type.

Downstream nodes keep reading ``research_facts`` in the same dict shape; the
new ``research_metadata`` key carries the trace from all five phases.
"""

from langgraph.graph import StateGraph, START, END

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_schema import (  # noqa: F401
    RAGState,
    _build_image_context,
)
from kazusa_ai_chatbot.config import (
    AFFINITY_DEFAULT,
    AFFINITY_MIN,
    AFFINITY_MAX,
    RAG_CACHE_MAX_SIZE,
    RAG_CACHE_SIMILARITY_THRESHOLD,
    RAG_CACHE_TTL_SECONDS,
)
from kazusa_ai_chatbot.db import get_text_embedding
from kazusa_ai_chatbot.rag.cache import RAGCache
from kazusa_ai_chatbot.rag.depth_classifier import (
    DEEP,
    InputDepthClassifier,
)
from kazusa_ai_chatbot.utils import log_preview

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any


logger = logging.getLogger(__name__)


# ── Stage-3 tuning constants ───────────────────────────────────────
# The thresholds below live here until Stage 5 moves them into config.py.

CACHE_HIT_THRESHOLD = 0.82            # similarity required to serve from cache
EXTERNAL_AFFINITY_SKIP_PERCENT = 40   # skip external_rag below this affinity %
KNOWLEDGE_BASE_PROBE_THRESHOLD = 0.72 # similarity to include knowledge_base results


# ── Lazy singletons ────────────────────────────────────────────────
# One RAGCache and one InputDepthClassifier per process. Both are safe to
# share across concurrent invocations.

_rag_cache: RAGCache | None = None
_depth_classifier: InputDepthClassifier | None = None


async def _get_rag_cache() -> RAGCache:
    """Return the process-wide ``RAGCache``, warm-starting from Mongo on first use.

    Construction parameters (similarity threshold, max size, per-type TTLs) are
    sourced from ``kazusa_ai_chatbot.config`` so they can be overridden via env
    vars without touching code.
    """
    global _rag_cache
    if _rag_cache is None:
        cache = RAGCache(
            max_size=RAG_CACHE_MAX_SIZE,
            similarity_threshold=RAG_CACHE_SIMILARITY_THRESHOLD,
            default_ttl_seconds=RAG_CACHE_TTL_SECONDS,
        )
        await cache.start()
        _rag_cache = cache
    return _rag_cache


def _get_depth_classifier() -> InputDepthClassifier:
    """Return the process-wide ``InputDepthClassifier``."""
    global _depth_classifier
    if _depth_classifier is None:
        _depth_classifier = InputDepthClassifier()
    return _depth_classifier


# ── Phase 6 — re-exports from decomposed submodules ──────────
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_resolution import (  # noqa: E402, F401
    continuation_resolver,
    entity_grounder,
    rag_planner,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_executors import (  # noqa: E402, F401
    _should_run_input_context,
    _should_run_tier2,
    _should_run_tier3,
    call_memory_retriever_agent_input_context_rag,
    call_web_search_agent,
    channel_recent_entity_rag,
    external_rag_dispatcher,
    input_context_rag_dispatcher,
    third_party_profile_rag,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor import (  # noqa: E402, F401
    rag_supervisor_evaluator,
)





async def _rag_noop(_: RAGState) -> dict:
    return {}


def _input_context_to_timestamp(chat_history_recent: list[dict], current_timestamp: str) -> str:
    """Return the automatic upper timestamp bound for input-context retrieval.

    Args:
        chat_history_recent: Immediate recent-history window already passed directly
            to downstream cognition.
        current_timestamp: Timestamp of the current in-flight turn.

    Returns:
        The earliest timestamp in ``chat_history_recent`` when present, otherwise
        the current turn timestamp. Conversation retrieval uses this as its
        automatic ``to_timestamp`` bound to avoid re-fetching the short-term
        window or the just-written current message.
    """
    if chat_history_recent:
        earliest_recent_timestamp = chat_history_recent[0].get("timestamp", "")
        if earliest_recent_timestamp:
            return earliest_recent_timestamp
    return current_timestamp


# ── Phase helpers ──────────────────────────────────────────────────

def _result_confidence(value: Any, *, is_empty_result: bool = False) -> float:
    """Estimate how informative a dispatcher's result is.

    A proxy for confidence until dispatchers emit one explicitly — used only
    for metadata and cache-storage heuristics.

    Args:
        value: A dispatcher result; may be ``str``, ``list`` of strings, or ``None``.
        is_empty_result: Explicit emptiness flag emitted by the retrieval agent.

    Returns:
        ``0.0`` when empty / missing or when the retrieval agent explicitly
        marks the result empty, otherwise a length-based proxy capped at ``1.0``.
    """
    if is_empty_result:
        return 0.0
    if value is None:
        return 0.0
    if isinstance(value, list):
        text = "\n".join(str(v) for v in value if v)
    else:
        text = str(value)
    if not text.strip():
        return 0.0
    return min(1.0, len(text.strip()) / 600.0 + 0.2)


def _normalize_retrieval_output(value: Any) -> str:
    """Normalize cached or live retrieval payloads into a single string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    return str(value)


def _build_character_profile_results(character_profile: dict) -> str:
    """Render the safe public subset of ``character_profile`` into RAG text.

    Args:
        character_profile: Full runtime character config, which may contain
            private steering and control fields not suitable for self-answers.

    Returns:
        A formatted text block containing only the allowlisted public
        self-knowledge fields that the character may naturally disclose
        in-world. Returns an empty string when nothing public is available.
    """
    labels = {
        "name": "姓名",
        "description": "角色描述",
        "gender": "性别",
        "age": "年龄",
        "birthday": "生日",
        "backstory": "背景故事",
    }
    lines: list[str] = []
    for key in ("name", "description", "gender", "age", "birthday", "backstory"):
        value = character_profile.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        lines.append(f"- {labels[key]}: {value}")
    if not lines:
        return ""
    return "### 角色公开资料\n" + "\n".join(lines)


def _merge_objective_facts(
    user_objective_facts_text: str,
    character_profile_results: str,
) -> str:
    """Merge user objective facts with the character's public fact block.

    Args:
        user_objective_facts_text: Existing objective-facts text derived from
            the user profile.
        character_profile_results: Public character facts rendered as RAG text.

    Returns:
        A single objective-facts text payload that preserves both sources
        without introducing a new downstream fact channel.
    """
    chunks = [
        chunk.strip()
        for chunk in (user_objective_facts_text, character_profile_results)
        if chunk and chunk.strip()
    ]
    return "\n\n".join(chunks)


async def _probe_knowledge_base(
    cache: RAGCache,
    embedding: list[float],
) -> str:
    """Probe the global knowledge_base cache and return any matching distillation.

    This is a supplementary probe for DEEP passes only — it never short-circuits
    the dispatcher pipeline; it only enriches ``research_facts`` with cached
    cross-session topic knowledge.

    Args:
        cache: Process-wide ``RAGCache`` instance.
        embedding: Query embedding of ``decontexualized_input``.

    Returns:
        The ``knowledge_base_results`` string from the best matching cache entry,
        or an empty string if no entry meets the similarity threshold.
    """
    hit = await cache.retrieve_if_similar(
        embedding=embedding,
        cache_type="knowledge_base",
        global_user_id="",
        threshold=KNOWLEDGE_BASE_PROBE_THRESHOLD,
    )
    if hit is None:
        return ""
    results = hit.get("results") or {}
    kb_text = results.get("knowledge_base_results", "")
    logger.debug(
        "Knowledge-base cache hit: similarity=%.3f preview=%s",
        float(hit.get("similarity", 0.0)),
        log_preview(kb_text),
    )
    return kb_text


async def _probe_cache(
    cache: RAGCache,
    embedding: list[float],
    global_user_id: str,
) -> tuple[dict | None, list[dict]]:
    """Probe the three user-scoped cache types plus the global external one.

    Args:
        cache: Process-wide ``RAGCache`` instance.
        embedding: Query embedding of ``decontexualized_input``.
        global_user_id: Current user's internal UUID.

    Returns:
        ``(cached_payload, probe_trace)``. ``cached_payload`` is the best hit's
        ``results`` dict if any probe scored at or above the cache threshold,
        else ``None``. ``probe_trace`` is a per-cache-type record of the probes
        performed (hit/miss + similarity) for the metadata bundle.
    """
    probes = [
        ("objective_user_facts", global_user_id),
        ("character_diary", global_user_id),
        ("external_knowledge", ""),
    ]
    trace: list[dict] = []
    best: tuple[float, dict, str] | None = None
    for cache_type, owner in probes:
        hit = await cache.retrieve_if_similar(
            embedding=embedding,
            cache_type=cache_type,
            global_user_id=owner,
        )
        if hit is None:
            trace.append({"cache_type": cache_type, "hit": False})
            continue
        sim = float(hit["similarity"])
        trace.append({"cache_type": cache_type, "hit": True, "similarity": sim})
        if best is None or sim > best[0]:
            best = (sim, hit["results"], cache_type)
    if best is None:
        return None, trace
    sim, results, cache_type = best
    logger.debug("RAG cache hit: cache_type=%s similarity=%.3f", cache_type, sim)
    return results, trace


# ── Phase 8 — Cache key construction ─────────────────────────────


def _build_cache_key(resolution_result: dict) -> str:
    """Build a stable hash key from resolution outputs.

    The key captures every dimension that makes two queries semantically
    distinct: resolved task text, resolved entity IDs, active retrieval
    sources, and the time-scope lookback window.

    Args:
        resolution_result: The RAGState after the resolution subgraph has run,
            containing ``continuation_context``, ``resolved_entities``,
            and ``retrieval_plan``.

    Returns:
        A hex-digest string suitable as a cache lookup key.
    """
    continuation = resolution_result.get("continuation_context") or {}
    entities = resolution_result.get("resolved_entities") or []
    plan = resolution_result.get("retrieval_plan") or {}

    key_dict = {
        "resolved_task": continuation.get("resolved_task", ""),
        "entity_ids": sorted(
            e.get("resolved_global_user_id") or e.get("surface_form", "")
            for e in entities
        ),
        "active_sources": sorted(plan.get("active_sources", [])),
        "lookback_hours": plan.get("time_scope", {}).get("lookback_hours", 72),
    }
    raw = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Phase 8 — Resolution + Retrieval subgraph builders ──────────


def _build_resolution_graph():
    """Compile the resolution subgraph (cheap, always runs).

    Topology:
      START → continuation_resolver → rag_planner → entity_grounder → END

    Returns:
        A compiled langgraph ``StateGraph`` for resolution.
    """
    builder = StateGraph(RAGState)
    builder.add_node("continuation_resolver", continuation_resolver)
    builder.add_node("rag_planner", rag_planner)
    builder.add_node("entity_grounder", entity_grounder)
    builder.add_edge(START, "continuation_resolver")
    builder.add_edge("continuation_resolver", "rag_planner")
    builder.add_edge("rag_planner", "entity_grounder")
    builder.add_edge("entity_grounder", END)
    return builder.compile()


def _build_retrieval_graph(depth: str, affinity_percent: float):
    """Compile the retrieval subgraph (expensive, only on cache miss).

    Topology (DEEP):
      START → input_context_rag_dispatcher
        → [Tier 1: call_memory_retriever_agent_input_context_rag]
        → tier_1_join → tier_2_gate
            "run"  → channel_recent_entity_rag → third_party_profile_rag
            "skip" → END
        → tier_3_gate
            "run"  → external_rag_dispatcher → call_web_search_agent
            "skip" → END

    Topology (SHALLOW):
      START → tier_2_gate
            "run"  → channel_recent_entity_rag → third_party_profile_rag → END
            "skip" → END

    Args:
        depth: ``"SHALLOW"`` or ``"DEEP"`` from the classifier.
        affinity_percent: User affinity normalised into a 0–100 band.

    Returns:
        A compiled langgraph ``StateGraph`` for retrieval.
    """
    builder = StateGraph(RAGState)

    # Tier 1: existing dispatchers
    builder.add_node("input_context_rag_dispatcher", input_context_rag_dispatcher)
    builder.add_node("call_memory_retriever_agent_input_context_rag", call_memory_retriever_agent_input_context_rag)
    builder.add_node("external_rag_dispatcher", external_rag_dispatcher)
    builder.add_node("call_web_search_agent", call_web_search_agent)

    # Tier 2: third-party backends
    builder.add_node("channel_recent_entity_rag", channel_recent_entity_rag)
    builder.add_node("third_party_profile_rag", third_party_profile_rag)

    # Joining / noop nodes
    builder.add_node("tier_1_join", _rag_noop)
    builder.add_node("rag_noop", _rag_noop)

    if depth == DEEP:
        builder.add_conditional_edges(
            START,
            _should_run_input_context,
            {"run": "input_context_rag_dispatcher", "skip": "tier_1_join"},
        )

        builder.add_conditional_edges(
            "input_context_rag_dispatcher",
            lambda state: state.get("input_context_next_action", "end"),
            {"memory_retriever_agent": "call_memory_retriever_agent_input_context_rag", "end": "tier_1_join"},
        )
        builder.add_edge("call_memory_retriever_agent_input_context_rag", "tier_1_join")

        builder.add_conditional_edges(
            "tier_1_join",
            _should_run_tier2,
            {
                "run": "channel_recent_entity_rag",
                "skip": "rag_noop",
            },
        )
        builder.add_conditional_edges(
            "rag_noop",
            _should_run_tier3,
            {
                "run": "external_rag_dispatcher",
                "skip": END,
            },
        )

        builder.add_edge("channel_recent_entity_rag", "third_party_profile_rag")
        builder.add_conditional_edges(
            "third_party_profile_rag",
            _should_run_tier3,
            {
                "run": "external_rag_dispatcher",
                "skip": END,
            },
        )

        builder.add_conditional_edges(
            "external_rag_dispatcher",
            lambda state: state.get("external_rag_next_action", "end"),
            {"web_search_agent": "call_web_search_agent", "end": END},
        )
        builder.add_edge("call_web_search_agent", END)

    else:
        # SHALLOW: tier 2 gate runs first
        builder.add_edge(START, "tier_1_join")
        builder.add_conditional_edges(
            "tier_1_join",
            _should_run_tier2,
            {
                "run": "channel_recent_entity_rag",
                "skip": "rag_noop",
            },
        )
        builder.add_edge("channel_recent_entity_rag", "third_party_profile_rag")
        builder.add_edge("third_party_profile_rag", END)
        builder.add_edge("rag_noop", END)

    return builder.compile()


def _build_rag_graph(depth: str, affinity_percent: float):
    """Compile a full tiered dispatcher graph (resolution + retrieval).

    Kept for backward compatibility. Phase 8 uses the split subgraphs
    (``_build_resolution_graph`` + ``_build_retrieval_graph``) directly.
    """
    builder = StateGraph(RAGState)

    # Pre-Tier-1: sequential resolution nodes
    builder.add_node("continuation_resolver", continuation_resolver)
    builder.add_node("rag_planner", rag_planner)
    builder.add_node("entity_grounder", entity_grounder)

    # Tier 1: existing dispatchers
    builder.add_node("input_context_rag_dispatcher", input_context_rag_dispatcher)
    builder.add_node("call_memory_retriever_agent_input_context_rag", call_memory_retriever_agent_input_context_rag)
    builder.add_node("external_rag_dispatcher", external_rag_dispatcher)
    builder.add_node("call_web_search_agent", call_web_search_agent)

    # Tier 2: third-party backends
    builder.add_node("channel_recent_entity_rag", channel_recent_entity_rag)
    builder.add_node("third_party_profile_rag", third_party_profile_rag)

    # Joining / noop nodes
    builder.add_node("tier_1_join", _rag_noop)
    builder.add_node("rag_noop", _rag_noop)

    # --- Edges ---
    # Pre-Tier-1: sequential chain
    builder.add_edge(START, "continuation_resolver")
    builder.add_edge("continuation_resolver", "rag_planner")
    builder.add_edge("rag_planner", "entity_grounder")

    if depth == DEEP:
        builder.add_edge("entity_grounder", "input_context_rag_dispatcher")

        builder.add_conditional_edges(
            "input_context_rag_dispatcher",
            lambda state: state.get("input_context_next_action", "end"),
            {"memory_retriever_agent": "call_memory_retriever_agent_input_context_rag", "end": "tier_1_join"},
        )
        builder.add_edge("call_memory_retriever_agent_input_context_rag", "tier_1_join")

        builder.add_conditional_edges(
            "tier_1_join",
            _should_run_tier2,
            {
                "run": "channel_recent_entity_rag",
                "skip": "rag_noop",
            },
        )
        builder.add_conditional_edges(
            "rag_noop",
            _should_run_tier3,
            {
                "run": "external_rag_dispatcher",
                "skip": END,
            },
        )

        builder.add_edge("channel_recent_entity_rag", "third_party_profile_rag")
        builder.add_conditional_edges(
            "third_party_profile_rag",
            _should_run_tier3,
            {
                "run": "external_rag_dispatcher",
                "skip": END,
            },
        )

        builder.add_conditional_edges(
            "external_rag_dispatcher",
            lambda state: state.get("external_rag_next_action", "end"),
            {"web_search_agent": "call_web_search_agent", "end": END},
        )
        builder.add_edge("call_web_search_agent", END)

    else:
        builder.add_conditional_edges(
            "entity_grounder",
            _should_run_tier2,
            {
                "run": "channel_recent_entity_rag",
                "skip": "rag_noop",
            },
        )
        builder.add_edge("channel_recent_entity_rag", "third_party_profile_rag")
        builder.add_edge("third_party_profile_rag", END)
        builder.add_edge("rag_noop", END)

    return builder.compile()


async def _store_results_in_cache(
    cache: RAGCache,
    embedding: list[float],
    research_facts: dict,
) -> None:
    """Write the produced RAG results back into the cache.

    Stores external_rag_results as global ``external_knowledge`` so it can
    be invalidated independently of per-user data.

    Args:
        cache: Process-wide cache instance.
        embedding: Query embedding of the current input.
        research_facts: Dict returned to the caller — branch payloads.
    """
    external = research_facts.get("external_rag_results")
    if external:
        await cache.store(
            embedding=embedding,
            results={
                "external_rag_results": external,
                "external_rag_is_empty_result": research_facts.get(
                    "external_rag_is_empty_result",
                    False,
                ),
            },
            cache_type="external_knowledge",
            global_user_id="",
            metadata={"source": "rag_subgraph"},
        )


async def call_rag_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """Execute the RAG pipeline with Phase 8 boundary cache.

    Architecture (Phase 8):
      0. Embed, classify depth, initialise metadata.
      1. Probe the legacy embedding-similarity cache (short-circuits everything).
      2. Run the **resolution subgraph** (cheap: 2 LLM calls + deterministic lookup).
      3. Build a **structured cache key** from resolution outputs.
      4. Probe the **boundary cache** with the structured key — on hit, skip retrieval.
      5. On miss, run the **retrieval subgraph** (expensive: DB queries, LLM calls, web).
      6. Store results in both the boundary cache and the legacy embedding cache.

    Args:
        state: The ``GlobalPersonaState`` carrying the current user, character,
            and pre-computed decontextualised input.

    Returns:
        A partial-state dict with ``research_facts`` (the per-branch payload
        dict consumed by downstream cognition nodes) and ``research_metadata``
        (the unified metadata bundle documenting this pass).
    """
    user_name = state["user_name"]
    global_user_id = state["global_user_id"]
    decontexualized_input = state["decontexualized_input"]
    user_profile = state["user_profile"]
    affinity_score = user_profile.get("affinity", AFFINITY_DEFAULT)
    affinity_percent = ((affinity_score - AFFINITY_MIN) / max(1, AFFINITY_MAX - AFFINITY_MIN)) * 100
    input_context_to_timestamp = _input_context_to_timestamp(
        state.get("chat_history_recent") or [],
        state["timestamp"],
    )

    user_image_context = _build_image_context(user_profile.get("user_image") or {})
    user_objective_facts_text = "\n".join(
        str(item.get("fact", item.get("description", "")))
        for item in (user_profile.get("objective_facts") or [])
        if str(item.get("fact", item.get("description", ""))).strip()
    )
    character_image_context = _build_image_context(
        state["character_profile"].get("self_image") or {}
    )
    character_profile_results = _build_character_profile_results(
        state["character_profile"]
    )
    objective_facts_text = _merge_objective_facts(
        user_objective_facts_text,
        character_profile_results,
    )

    logger.debug(
        "RAG input: user=%s global_user=%s channel=%s affinity=%s recent_history=%d input=%s",
        user_name,
        global_user_id,
        state["platform_channel_id"] or "<dm>",
        affinity_score,
        len(state.get("chat_history_recent") or []),
        log_preview(decontexualized_input),
    )

    input_embedding = await get_text_embedding(decontexualized_input)
    metadata: dict = {
        "embedding_dim": len(input_embedding),
        "depth": None,
        "depth_confidence": 0.0,
        "depth_reasoning": "",
        "cache_hit": False,
        "boundary_cache_hit": False,
        "cache_probe": [],
        "trigger_dispatchers": [],
        "rag_sources_used": [],
        "confidence_scores": {},
        "response_confidence": 0.0,
    }

    cache = await _get_rag_cache()

    # ── Legacy embedding-similarity cache probe ───────────────
    cached, probe_trace = await _probe_cache(cache, input_embedding, global_user_id)
    metadata["cache_probe"] = probe_trace
    if cached is not None:
        cached_input_context = _normalize_retrieval_output(cached.get("input_context_results", ""))
        cached_external = _normalize_retrieval_output(cached.get("external_rag_results", ""))
        cached_input_context_is_empty = bool(cached.get("input_context_is_empty_result", False))
        cached_external_is_empty = bool(cached.get("external_rag_is_empty_result", False))
        cached_input_context_conf = _result_confidence(
            cached_input_context,
            is_empty_result=cached_input_context_is_empty,
        )
        cached_external_conf = _result_confidence(
            cached_external,
            is_empty_result=cached_external_is_empty,
        )
        sources_used = []
        if cached_input_context_conf > 0.0:
            sources_used.append("input_context_rag")
        if cached_external_conf > 0.0:
            sources_used.append("external_rag")
        metadata["cache_hit"] = True
        metadata["rag_sources_used"] = ["cache"] if not sources_used else ["cache", *sources_used]
        metadata["confidence_scores"] = {
            "input_context_rag": cached_input_context_conf,
            "external_rag": cached_external_conf,
        }
        metadata["response_confidence"] = max([cached_input_context_conf, cached_external_conf] + [0.0])
        research_facts = {
            "input_context_results": cached_input_context,
            "external_rag_results": cached_external,
            "objective_facts": objective_facts_text,
            "user_image": user_image_context,
            "character_image": character_image_context,
            "input_context_is_empty_result": cached_input_context_is_empty,
            "external_rag_is_empty_result": cached_external_is_empty,
        }
        logger.info(
            "RAG summary: user=%s global_user=%s cache_hit=%s depth=%s sources=%s input_context=%s external=%s input=%s",
            user_name,
            global_user_id,
            True,
            metadata.get("depth"),
            metadata.get("rag_sources_used", []),
            log_preview(research_facts["input_context_results"], max_length=140),
            log_preview(research_facts["external_rag_results"], max_length=140),
            log_preview(decontexualized_input, max_length=160),
        )
        return {
            "research_facts": research_facts,
            "research_metadata": [metadata],
        }

    # ── Depth classification ──────────────────────────────────
    classifier = _get_depth_classifier()
    depth_result = await classifier.classify(
        user_input=decontexualized_input,
        user_topic=state.get("channel_topic", ""),
        affinity=affinity_score,
        input_embedding=input_embedding,
    )
    depth = depth_result["depth"]
    metadata["depth"] = depth
    metadata["depth_confidence"] = depth_result["confidence"]
    metadata["depth_reasoning"] = depth_result["reasoning"]
    metadata["trigger_dispatchers"] = list(depth_result["trigger_dispatchers"])

    knowledge_base_results = ""
    if depth == DEEP:
        knowledge_base_results = await _probe_knowledge_base(cache, input_embedding)

    # ── Phase A: Resolution subgraph (cheap, always runs) ─────
    initial_state: RAGState = {
        "timestamp": state["timestamp"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "platform_message_id": state.get("platform_message_id", ""),
        "decontexualized_input": decontexualized_input,
        "channel_topic": state.get("channel_topic", ""),
        "input_context_to_timestamp": input_context_to_timestamp,
        "chat_history_recent": state.get("chat_history_recent") or [],
        "user_name": user_name,
        "global_user_id": global_user_id,
        "platform_bot_id": state["platform_bot_id"],
        "character_profile": state["character_profile"],
        "user_profile": user_profile,
        "input_embedding": input_embedding,
        "depth": depth,
        "depth_confidence": depth_result["confidence"],
        "cache_hit": False,
        "trigger_dispatchers": list(depth_result["trigger_dispatchers"]),
        "rag_metadata": metadata,
        # Phase 1 fields — initialised empty, populated by resolution nodes
        "continuation_context": {},
        "retrieval_plan": {},
        "resolved_entities": [],
        "retrieval_ledger": {},
        "channel_recent_entity_results": "",
        "third_party_profile_results": "",
        "entity_knowledge_results": "",
        "entity_resolution_notes": "",
    }

    resolution_graph = _build_resolution_graph()
    resolution_result = await resolution_graph.ainvoke(initial_state)

    # Phase 1 metadata enrichment (from resolution)
    metadata["retrieval_plan"] = resolution_result.get("retrieval_plan") or {}
    metadata["resolved_entities"] = resolution_result.get("resolved_entities") or []
    metadata["continuation_context"] = resolution_result.get("continuation_context") or {}

    # ── Phase 8: Boundary cache — probe with structured key ───
    boundary_cache_key = _build_cache_key(resolution_result)
    boundary_hit = await cache.retrieve_if_similar_by_key(boundary_cache_key)
    if boundary_hit is not None:
        metadata["boundary_cache_hit"] = True
        metadata["cache_hit"] = True
        cached_results = boundary_hit["results"]

        input_context_results = _normalize_retrieval_output(cached_results.get("input_context_results", ""))
        external_rag_results = _normalize_retrieval_output(cached_results.get("external_rag_results", ""))
        input_context_is_empty = bool(cached_results.get("input_context_is_empty_result", False))
        external_is_empty = bool(cached_results.get("external_rag_is_empty_result", False))
        channel_recent_entity_results = str(cached_results.get("channel_recent_entity_results") or "")
        third_party_profile_results = str(cached_results.get("third_party_profile_results") or "")
        entity_knowledge_results = ""
        entity_resolution_notes = str(cached_results.get("entity_resolution_notes") or "")

        input_context_conf = _result_confidence(input_context_results, is_empty_result=input_context_is_empty)
        external_conf = _result_confidence(external_rag_results, is_empty_result=external_is_empty)

        sources_used = ["boundary_cache"]
        if input_context_conf > 0.0:
            sources_used.append("input_context_rag")
        if external_conf > 0.0:
            sources_used.append("external_rag")
        if channel_recent_entity_results:
            sources_used.append("channel_recent_entity")
        if third_party_profile_results:
            sources_used.append("third_party_profile")
        metadata["rag_sources_used"] = sources_used
        metadata["confidence_scores"] = {
            "input_context_rag": input_context_conf,
            "external_rag": external_conf,
        }
        metadata["response_confidence"] = max([input_context_conf, external_conf] + [0.0])

        research_facts = {
            "input_context_results": input_context_results,
            "external_rag_results": external_rag_results,
            "objective_facts": objective_facts_text,
            "user_image": user_image_context,
            "character_image": character_image_context,
            "knowledge_base_results": knowledge_base_results,
            "input_context_is_empty_result": input_context_is_empty,
            "external_rag_is_empty_result": external_is_empty,
            "third_party_profile_results": third_party_profile_results,
            "channel_recent_entity_results": channel_recent_entity_results,
            "entity_knowledge_results": entity_knowledge_results,
            "entity_resolution_notes": entity_resolution_notes,
        }
        logger.info(
            "RAG summary: user=%s global_user=%s boundary_cache_hit=%s depth=%s sources=%s input=%s",
            user_name,
            global_user_id,
            True,
            depth,
            sources_used,
            log_preview(decontexualized_input, max_length=160),
        )
        return {
            "research_facts": research_facts,
            "research_metadata": [metadata],
        }

    # ── Phase B: Retrieval subgraph (expensive, only on miss) ─
    retrieval_graph = _build_retrieval_graph(depth, affinity_percent)
    result = await retrieval_graph.ainvoke(resolution_result)

    # ── Phase 5: Bounded evaluator + optional repair pass ────
    metadata["repair_pass"] = 0
    eval_result = await rag_supervisor_evaluator(result)
    metadata["evaluation"] = eval_result.get("evaluation", {})

    if eval_result.get("needs_repair") and eval_result.get("repair_entities"):
        metadata["repair_pass"] = 1
        repair_entities = eval_result["repair_entities"]
        logger.info(
            "RAG evaluator requested repair pass for entities: %s",
            repair_entities,
        )

        repair_state = dict(result)

        # Repair runs a fresh resolution step scoped only to the newly revealed
        # entities, matching the Phase-5 design contract.
        repair_plan = dict(repair_state.get("retrieval_plan") or {})
        repair_sources = eval_result.get("repair_sources", [])
        existing_sources = set(repair_plan.get("active_sources", []))
        for src in repair_sources:
            existing_sources.add(src)
        repair_plan["active_sources"] = sorted(existing_sources)
        repair_plan["entities"] = [
            {
                "surface_form": entity,
                "entity_type": "unknown",
                "resolution_confidence": 0.0,
            }
            for entity in repair_entities
        ]
        repair_state["retrieval_plan"] = repair_plan

        grounded_repair = await entity_grounder(repair_state)
        repair_state["resolved_entities"] = grounded_repair["resolved_entities"]
        repair_state["entity_resolution_notes"] = grounded_repair["entity_resolution_notes"]

        repair_ledger = dict(repair_state.get("retrieval_ledger") or {})
        for re_ent in repair_entities:
            keys_to_remove = [k for k in repair_ledger if re_ent.lower() in k.lower()]
            for k in keys_to_remove:
                del repair_ledger[k]
        repair_state["retrieval_ledger"] = repair_ledger

        # Run repair retrieval
        repair_graph = _build_retrieval_graph(depth, affinity_percent)
        repair_result = await repair_graph.ainvoke(repair_state)

        # Merge repair results — append, don't replace
        for key in ("channel_recent_entity_results", "third_party_profile_results", "entity_resolution_notes"):
            existing_val = str(result.get(key) or "")
            repair_val = str(repair_result.get(key) or "")
            if repair_val and repair_val not in existing_val:
                result[key] = (existing_val + "\n\n" + repair_val).strip() if existing_val else repair_val

        # Merge ledger
        merged_ledger = dict(result.get("retrieval_ledger") or {})
        merged_ledger.update(repair_result.get("retrieval_ledger") or {})
        result["retrieval_ledger"] = merged_ledger

        logger.info("RAG repair pass completed, merged results")

    input_context_results = _normalize_retrieval_output(result.get("input_context_results"))
    external_rag_results = _normalize_retrieval_output(result.get("external_rag_results"))
    input_context_is_empty_result = bool(result.get("input_context_is_empty_result", False))
    external_rag_is_empty_result = bool(result.get("external_rag_is_empty_result", False))
    channel_recent_entity_results = str(result.get("channel_recent_entity_results") or "")
    third_party_profile_results = str(result.get("third_party_profile_results") or "")
    entity_knowledge_results = ""
    entity_resolution_notes = str(result.get("entity_resolution_notes") or "")

    input_context_conf = _result_confidence(
        input_context_results,
        is_empty_result=input_context_is_empty_result,
    )
    external_conf = _result_confidence(
        external_rag_results,
        is_empty_result=external_rag_is_empty_result,
    )
    metadata["confidence_scores"] = {
        "input_context_rag": input_context_conf,
        "external_rag": external_conf,
    }
    sources_used = []
    if input_context_conf > 0.0:
        sources_used.append("input_context_rag")
    if external_conf > 0.0:
        sources_used.append("external_rag")
    if channel_recent_entity_results:
        sources_used.append("channel_recent_entity")
    if third_party_profile_results:
        sources_used.append("third_party_profile")
    metadata["rag_sources_used"] = sources_used
    metadata["response_confidence"] = max([input_context_conf, external_conf] + [0.0])
    metadata["retrieval_ledger"] = result.get("retrieval_ledger") or {}

    logger.debug(
        "RAG metadata: depth=%s confidence=%.3f cache_probe=%s trigger_dispatchers=%s response_confidence=%.3f sources=%s",
        depth,
        depth_result["confidence"],
        metadata.get("cache_probe", []),
        metadata.get("trigger_dispatchers", []),
        metadata.get("response_confidence", 0.0),
        metadata.get("rag_sources_used", []),
    )

    research_facts = {
        "input_context_results": input_context_results,
        "external_rag_results": external_rag_results,
        "objective_facts": objective_facts_text,
        "user_image": user_image_context,
        "character_image": character_image_context,
        "knowledge_base_results": knowledge_base_results,
        "input_context_is_empty_result": input_context_is_empty_result,
        "external_rag_is_empty_result": external_rag_is_empty_result,
        # Phase 4 — new research_facts keys
        "third_party_profile_results": third_party_profile_results,
        "channel_recent_entity_results": channel_recent_entity_results,
        "entity_knowledge_results": entity_knowledge_results,
        "entity_resolution_notes": entity_resolution_notes,
    }

    logger.info(
        "RAG summary: user=%s global_user=%s cache_hit=%s depth=%s depth_conf=%.2f sources=%s kb_hit=%s input_context=%s external=%s tp_profile=%s ch_entity=%s input=%s",
        user_name,
        global_user_id,
        False,
        depth,
        depth_result["confidence"],
        sources_used,
        bool(knowledge_base_results),
        log_preview(input_context_results, max_length=140),
        log_preview(external_rag_results, max_length=140),
        log_preview(third_party_profile_results, max_length=100),
        log_preview(channel_recent_entity_results, max_length=100),
        log_preview(decontexualized_input, max_length=160),
    )

    # ── Phase 8: Store in both caches ─────────────────────────
    await _store_results_in_cache(
        cache, input_embedding, research_facts,
    )

    # Boundary cache: store retrieval results keyed on resolution hash
    retrieval_payload = {
        "input_context_results": input_context_results,
        "external_rag_results": external_rag_results,
        "input_context_is_empty_result": input_context_is_empty_result,
        "external_rag_is_empty_result": external_rag_is_empty_result,
        "third_party_profile_results": third_party_profile_results,
        "channel_recent_entity_results": channel_recent_entity_results,
        "entity_knowledge_results": entity_knowledge_results,
        "entity_resolution_notes": entity_resolution_notes,
    }
    await cache.store_by_key(
        cache_key=boundary_cache_key,
        results=retrieval_payload,
        cache_type="boundary_cache",
        global_user_id=global_user_id,
        metadata={"depth": depth, "sources_used": sources_used},
    )

    return {
        "research_facts": research_facts,
        "research_metadata": [metadata],
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.mcp_client import mcp_manager
    from kazusa_ai_chatbot.db import get_character_profile

    # Connect to MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    state: GlobalPersonaState = {
        "decontexualized_input": "千纱晚上要记得奖励我哦♥",
        "channel_topic": "闲聊",
        "platform_bot_id": "1485169644888395817",
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "character_profile": await get_character_profile(),
        "user_profile": {"affinity": 950},
    }

    result = await call_rag_subgraph(state)
    print(f"RAG SubGraph: {result}")


    await mcp_manager.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
