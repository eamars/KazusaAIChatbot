"""Reflection-triggered cognition dry-run contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.db import CharacterProfileDoc, UserProfileDoc
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.reflection_cycle.context import PromotedReflectionContext
from kazusa_ai_chatbot.time_boundary import LocalTimeContextDoc

ReflectionDryRunOutputMode = Literal["think_only", "preview", "silent"]
ReflectionDryRunStatus = Literal[
    "completed",
    "skipped_busy",
    "skipped_empty_context",
]

_REFLECTION_DRY_RUN_OUTPUT_MODES = frozenset(
    ("think_only", "preview", "silent"),
)
_REFLECTION_DRY_RUN_PROMPT_KEYS = [
    "l1_subconscious.reflection_signal_reflection_artifact",
    "l2a_conscious_framing.reflection_signal_reflection_artifact",
    "l2b_boundary_appraisal.reflection_signal_reflection_artifact",
    "l2c1_judgment_synthesis.reflection_signal_reflection_artifact",
    "l2c2_social_context_appraisal.reflection_signal_reflection_artifact",
    "l2d_action_selection.reflection_signal_reflection_artifact",
]
_REFLECTION_DRY_RUN_INPUT_TEXT = (
    "Reflection dry run over promoted reflection artifact."
)


class ReflectionCognitionDryRunError(ValueError):
    """Raised when reflection dry-run inputs violate the local contract."""


class ReflectionCognitionDryRunAudit(TypedDict):
    """Audit-only summary returned by reflection cognition dry runs."""

    status: ReflectionDryRunStatus
    skip_reason: str
    cognition_called: bool
    episode_id: str
    trigger_source: Literal["reflection_signal"]
    input_sources: list[Literal["reflection_artifact"]]
    output_mode: ReflectionDryRunOutputMode
    prompt_variant: Literal["reflection_signal_reflection_artifact"]
    prompt_keys: list[str]
    cognition_output_keys: list[str]


def build_reflection_signal_cognitive_episode(
    *,
    promoted_reflection_context: PromotedReflectionContext,
    storage_timestamp_utc: str,
    local_time_context: LocalTimeContextDoc,
    output_mode: ReflectionDryRunOutputMode = "think_only",
) -> CognitiveEpisode:
    """Build a source-neutral cognitive episode for promoted reflection.

    Args:
        promoted_reflection_context: Gated reflection context allowed into
            model-facing cognition prompts.
        storage_timestamp_utc: Storage UTC timestamp for the dry-run event.
        local_time_context: Configured-local time context for the event.
        output_mode: Audit-only cognition output mode.

    Returns:
        Valid `CognitiveEpisode` representing the promoted reflection artifact.

    Raises:
        ReflectionCognitionDryRunError: If the reflection context or output
            mode violates the dry-run contract.
    """
    if output_mode not in _REFLECTION_DRY_RUN_OUTPUT_MODES:
        raise ReflectionCognitionDryRunError(
            "reflection output_mode is not supported",
        )
    if _promoted_context_is_empty(promoted_reflection_context):
        raise ReflectionCognitionDryRunError(
            "promoted reflection context is empty",
        )

    reflection_artifact = _canonical_reflection_context(
        promoted_reflection_context,
    )
    digest = hashlib.sha256(reflection_artifact.encode("utf-8")).hexdigest()
    episode_id = f"reflection:dry_run:{digest[:16]}"
    episode: CognitiveEpisode = {
        "episode_id": episode_id,
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": output_mode,
        "percepts": [
            {
                "percept_id": "reflection:artifact:promoted_context",
                "input_source": "reflection_artifact",
                "content": reflection_artifact,
                "visibility": "model_visible",
                "metadata": {"source": "promoted_reflection_context"},
            },
        ],
        "target_scope": {
            "platform": "reflection_cycle",
            "platform_channel_id": "reflection_dry_run",
            "channel_type": "reflection_dry_run",
            "current_platform_user_id": "reflection_cycle",
            "current_global_user_id": "reflection_cycle",
            "current_display_name": "reflection_cycle",
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "reflection_cycle",
            "platform_message_id": "reflection:dry_run",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {"think_only": True, "no_remember": True},
        },
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
    }
    validate_cognitive_episode(episode)
    return episode


async def run_reflection_cognition_dry_run(
    *,
    promoted_reflection_context: PromotedReflectionContext,
    character_profile: CharacterProfileDoc,
    user_profile: UserProfileDoc,
    storage_timestamp_utc: str,
    local_time_context: LocalTimeContextDoc,
    is_primary_interaction_busy: Callable[[], bool],
    call_cognition_subgraph_func: Callable[
        [GlobalPersonaState],
        Awaitable[GlobalPersonaState],
    ],
    output_mode: ReflectionDryRunOutputMode = "think_only",
) -> ReflectionCognitionDryRunAudit:
    """Run shared cognition over promoted reflection in audit-only mode.

    Args:
        promoted_reflection_context: Gated reflection context allowed into
            model-facing cognition prompts.
        character_profile: Current character profile snapshot for cognition.
        user_profile: Reflection-cycle placeholder user profile.
        storage_timestamp_utc: Storage UTC timestamp for the dry-run event.
        local_time_context: Configured-local time context for the event.
        is_primary_interaction_busy: Probe for live interaction load.
        call_cognition_subgraph_func: Injected cognition callable.
        output_mode: Audit-only cognition output mode.

    Returns:
        Audit-only summary of the dry-run outcome.

    Raises:
        ReflectionCognitionDryRunError: If the output mode violates the
            dry-run contract.
    """
    if output_mode not in _REFLECTION_DRY_RUN_OUTPUT_MODES:
        raise ReflectionCognitionDryRunError(
            "reflection output_mode is not supported",
        )

    if is_primary_interaction_busy():
        audit: ReflectionCognitionDryRunAudit = {
            "status": "skipped_busy",
            "skip_reason": "primary_interaction_busy",
            "cognition_called": False,
            "episode_id": "",
            "trigger_source": "reflection_signal",
            "input_sources": ["reflection_artifact"],
            "output_mode": output_mode,
            "prompt_variant": "reflection_signal_reflection_artifact",
            "prompt_keys": [],
            "cognition_output_keys": [],
        }
        return audit

    if _promoted_context_is_empty(promoted_reflection_context):
        audit = {
            "status": "skipped_empty_context",
            "skip_reason": "promoted_reflection_context_empty",
            "cognition_called": False,
            "episode_id": "",
            "trigger_source": "reflection_signal",
            "input_sources": ["reflection_artifact"],
            "output_mode": output_mode,
            "prompt_variant": "reflection_signal_reflection_artifact",
            "prompt_keys": [],
            "cognition_output_keys": [],
        }
        return audit

    episode = build_reflection_signal_cognitive_episode(
        promoted_reflection_context=promoted_reflection_context,
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=local_time_context,
        output_mode=output_mode,
    )
    dry_run_state: GlobalPersonaState = {
        "character_profile": character_profile,
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "user_input": _REFLECTION_DRY_RUN_INPUT_TEXT,
        "prompt_message_context": {
            "body_text": "",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": "reflection_cycle",
        "platform_channel_id": "reflection_dry_run",
        "channel_type": "reflection_dry_run",
        "platform_message_id": "reflection:dry_run",
        "platform_user_id": "reflection_cycle",
        "global_user_id": "reflection_cycle",
        "user_name": "reflection_cycle",
        "user_profile": user_profile,
        "platform_bot_id": "reflection_cycle",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "promoted_reflection_context": promoted_reflection_context,
        "debug_modes": {"think_only": True, "no_remember": True},
        "should_respond": False,
        "decontexualized_input": _REFLECTION_DRY_RUN_INPUT_TEXT,
        "referents": [],
        "rag_result": {
            "answer": "",
            "user_image": {
                "user_memory_context": empty_user_memory_context(),
            },
            "user_memory_unit_candidates": [],
            "character_image": {},
            "third_party_profiles": [],
            "memory_evidence": [],
            "recall_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "supervisor_trace": {
                "loop_count": 0,
                "unknown_slots": [],
                "dispatched": [],
            },
        },
        "internal_monologue": "",
        "action_directives": {},
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "final_dialog": [],
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
    }
    cognition_result = await call_cognition_subgraph_func(dry_run_state)
    cognition_output_keys = sorted(cognition_result)
    audit = {
        "status": "completed",
        "skip_reason": "",
        "cognition_called": True,
        "episode_id": episode["episode_id"],
        "trigger_source": "reflection_signal",
        "input_sources": ["reflection_artifact"],
        "output_mode": output_mode,
        "prompt_variant": "reflection_signal_reflection_artifact",
        "prompt_keys": list(_REFLECTION_DRY_RUN_PROMPT_KEYS),
        "cognition_output_keys": cognition_output_keys,
    }
    return audit


def _promoted_context_is_empty(context: PromotedReflectionContext) -> bool:
    """Return whether promoted context has no reflection content lanes.

    Args:
        context: Promoted reflection context to inspect.

    Returns:
        True when both promoted reflection content lanes are absent or empty.
    """
    promoted_lore = context.get("promoted_lore", [])
    promoted_self_guidance = context.get("promoted_self_guidance", [])
    is_empty = not promoted_lore and not promoted_self_guidance
    return is_empty


def _canonical_reflection_context(
    context: PromotedReflectionContext,
) -> str:
    """Render promoted reflection context using stable JSON bytes.

    Args:
        context: Promoted reflection context to render.

    Returns:
        Canonical JSON string used in the reflection percept and id digest.
    """
    rendered = json.dumps(context, sort_keys=True, ensure_ascii=False)
    return rendered
