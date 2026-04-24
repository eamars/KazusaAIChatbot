"""RAGState TypedDict and shared helpers for the RAG pipeline.

Extracted during Phase 6 decomposition to break circular imports between
the orchestrator, resolution, and executor submodules.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class RAGState(TypedDict):
    # Inputs
    timestamp: str
    platform: str
    platform_channel_id: str
    platform_message_id: str
    decontexualized_input: str
    channel_topic: str
    input_context_to_timestamp: str
    chat_history_recent: list[dict]

    # Input facts
    user_name: str
    global_user_id: str
    platform_bot_id: str
    character_profile: dict
    user_profile: dict

    # Stage-3 metadata thread (carried through every phase)
    input_embedding: list[float]
    depth: str                         # "SHALLOW" | "DEEP"
    depth_confidence: float
    cache_hit: bool
    trigger_dispatchers: list[str]
    rag_metadata: dict

    # Phase 1 — continuation_resolver output
    continuation_context: dict

    # Phase 1 — rag_planner output
    retrieval_plan: dict

    # Phase 1 — entity_grounder output
    resolved_entities: list[dict]

    # Phase 2 — retrieval ledger (prevents duplicate subject fetches)
    retrieval_ledger: dict

    # External RAG Dispatcher output
    external_rag_next_action: str
    external_rag_task: str
    external_rag_context: dict
    external_rag_expected_response: str

    # External RAG output
    external_rag_results: Annotated[list[str], operator.add]
    external_rag_is_empty_result: bool

    # Input-Context RAG dispatcher output (was: Internal RAG)
    input_context_next_action: str
    input_context_task: str
    input_context_context: dict
    input_context_expected_response: str

    # Input-Context RAG output
    input_context_results: Annotated[list[str], operator.add]
    input_context_is_empty_result: bool

    # Phase 2 — channel_recent_entity output
    channel_recent_entity_results: str

    # Phase 2 — third_party_profile output
    third_party_profile_results: str

    # Phase 3 — entity knowledge (durable entity/topic memory)
    entity_knowledge_results: str

    # Phase 1 — entity resolution notes (surface for downstream prompts)
    entity_resolution_notes: str


def _build_image_context(image_doc: dict) -> dict:
    """Build structured three-tier image context for downstream JSON prompts.

    Args:
        image_doc: Three-tier image dict with keys ``milestones``,
            ``recent_window``, ``historical_summary``, and ``meta``.

    Returns:
        A nested dict with ``milestones``, ``historical_summary``, and
        ``recent_observations``, or an empty dict if the image doc is empty.
    """
    if not image_doc:
        return {}

    milestones: list[dict[str, Any]] = []
    for milestone in image_doc.get("milestones") or []:
        category = str(
            milestone.get("category", milestone.get("milestone_category", ""))
        ).strip()
        event = str(milestone.get("event", milestone.get("description", ""))).strip()
        if not event:
            continue
        superseded_by = milestone.get("superseded_by")
        if isinstance(superseded_by, str):
            superseded_by = superseded_by.strip() or None
        milestones.append(
            {
                "event": event,
                "category": category,
                "superseded_by": superseded_by,
            }
        )

    recent_observations: list[str] = []
    for observation in image_doc.get("recent_window") or []:
        if isinstance(observation, dict):
            summary = str(observation.get("summary") or "").strip()
        else:
            summary = str(observation).strip()
        if summary:
            recent_observations.append(summary)

    return {
        "milestones": milestones,
        "historical_summary": str(image_doc.get("historical_summary") or "").strip(),
        "recent_observations": recent_observations,
    }
