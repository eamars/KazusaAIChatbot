"""Internal-thought cognition dry-run contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, TypedDict

from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.config import COGNITION_VISUAL_DIRECTIVES_ENABLED
from kazusa_ai_chatbot.db import CharacterProfileDoc, UserProfileDoc
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.time_context import TimeContextDoc

InternalThoughtDryRunOutputMode = Literal["think_only", "preview", "silent"]
InternalThoughtDryRunStatus = Literal[
    "completed",
    "skipped_busy",
    "skipped_empty_residue",
]

_INTERNAL_THOUGHT_DRY_RUN_OUTPUT_MODES = frozenset(
    ("think_only", "preview", "silent"),
)
_INTERNAL_THOUGHT_DRY_RUN_PROMPT_KEYS = [
    "l1_subconscious.internal_thought_internal_monologue",
    "l2a_consciousness.internal_thought_internal_monologue",
    "l2b_boundary_core.internal_thought_internal_monologue",
    "l2c_judgment_core.internal_thought_internal_monologue",
    "l2d_action_initializer.internal_thought_internal_monologue",
]
_INTERNAL_THOUGHT_DRY_RUN_PROMPT_KEYS_WITHOUT_VISUAL = list(
    _INTERNAL_THOUGHT_DRY_RUN_PROMPT_KEYS
)
_INTERNAL_MONOLOGUE_MAX_CHARACTERS = 4000
_ACTION_LATCH_TEXT_MAX_CHARACTERS = 1000
_INTERNAL_THOUGHT_INPUT_TEXT = (
    "Internal thought dry run over private cognition residue."
)


class InternalThoughtResidue(TypedDict):
    """Private cognition residue admitted to an internal-thought dry run."""

    residue_id: str
    internal_monologue: str
    source: Literal["runtime_internal_thought"]


class InternalActionLatch(TypedDict):
    """Audit-only action candidate attached to private cognition residue."""

    latch_id: str
    action_text: str
    latch_reason: str
    status: Literal["audit_only"]


class PublicSceneResidue(TypedDict):
    """Contract for future public-scene residue that is not produced here."""

    scene_residue_id: str
    source_episode_id: str
    summary: str
    visibility: Literal["public_scene_candidate"]
    merge_status: Literal["not_merged_stage_08"]


class InternalThoughtCognitionDryRunAudit(TypedDict):
    """Audit-only summary returned by internal-thought cognition dry runs."""

    status: InternalThoughtDryRunStatus
    skip_reason: str
    cognition_called: bool
    episode_id: str
    residue_id: str
    action_latch_id: str
    trigger_source: Literal["internal_thought"]
    input_sources: list[Literal["internal_monologue"]]
    output_mode: InternalThoughtDryRunOutputMode
    prompt_variant: Literal["internal_thought_internal_monologue"]
    prompt_keys: list[str]
    cognition_output_keys: list[str]


class InternalThoughtCognitionDryRunError(ValueError):
    """Raised when internal-thought dry-run inputs violate the local contract."""


def build_internal_thought_cognitive_episode(
    *,
    residue: InternalThoughtResidue,
    timestamp: str,
    time_context: TimeContextDoc,
    action_latch: InternalActionLatch | None = None,
    output_mode: InternalThoughtDryRunOutputMode = "think_only",
    visual_directives_enabled: bool = True,
) -> CognitiveEpisode:
    """Build a source-neutral episode for private internal-thought residue.

    Args:
        residue: Private runtime-generated thought admitted to cognition.
        timestamp: UTC timestamp for the dry-run event.
        time_context: Character-local time context associated with the event.
        action_latch: Optional audit-only action candidate.
        output_mode: Audit-only cognition output mode.
        visual_directives_enabled: Whether this run may generate visual
            directives when global config also allows it.

    Returns:
        Valid `CognitiveEpisode` representing the internal-thought residue.

    Raises:
        InternalThoughtCognitionDryRunError: If residue, action latch, or
            output mode violates the dry-run contract.
    """
    if output_mode not in _INTERNAL_THOUGHT_DRY_RUN_OUTPUT_MODES:
        raise InternalThoughtCognitionDryRunError(
            "internal thought output_mode is not supported",
        )

    _validate_residue(residue)
    if action_latch is not None:
        _validate_action_latch(action_latch)

    percept_content = _canonical_percept_content(
        residue=residue,
        action_latch=action_latch,
    )
    digest = hashlib.sha256(percept_content.encode("utf-8")).hexdigest()
    episode_id = f"internal_thought:dry_run:{digest[:16]}"
    debug_modes = {"think_only": True, "no_remember": True}
    effective_visual_directives_enabled = (
        COGNITION_VISUAL_DIRECTIVES_ENABLED and visual_directives_enabled
    )
    if not effective_visual_directives_enabled:
        debug_modes["no_visual_directives"] = True

    episode: CognitiveEpisode = {
        "episode_id": episode_id,
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": output_mode,
        "percepts": [
            {
                "percept_id": "internal_thought:percept:internal_monologue",
                "input_source": "internal_monologue",
                "content": percept_content,
                "visibility": "model_visible",
                "metadata": {"source": "runtime_internal_thought"},
            },
        ],
        "target_scope": {
            "platform": "internal_thought",
            "platform_channel_id": "internal_thought_dry_run",
            "channel_type": "internal_thought_dry_run",
            "current_platform_user_id": "internal_thought",
            "current_global_user_id": "internal_thought",
            "current_display_name": "internal_thought",
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "internal_thought",
            "platform_message_id": "internal_thought:dry_run",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": debug_modes,
        },
        "timestamp": timestamp,
        "time_context": time_context,
    }
    validate_cognitive_episode(episode)
    return episode


async def run_internal_thought_cognition_dry_run(
    *,
    residue: InternalThoughtResidue,
    character_profile: CharacterProfileDoc,
    user_profile: UserProfileDoc,
    timestamp: str,
    time_context: TimeContextDoc,
    is_primary_interaction_busy: Callable[[], bool],
    call_cognition_subgraph_func: Callable[
        [GlobalPersonaState],
        Awaitable[GlobalPersonaState],
    ],
    action_latch: InternalActionLatch | None = None,
    output_mode: InternalThoughtDryRunOutputMode = "think_only",
    visual_directives_enabled: bool = True,
) -> InternalThoughtCognitionDryRunAudit:
    """Run shared cognition over private thought in audit-only mode.

    Args:
        residue: Private runtime-generated thought admitted to cognition.
        character_profile: Current character profile snapshot for cognition.
        user_profile: Dry-run placeholder user profile.
        timestamp: UTC timestamp for the dry-run event.
        time_context: Character-local time context associated with the event.
        is_primary_interaction_busy: Probe for live interaction load.
        call_cognition_subgraph_func: Injected cognition callable.
        action_latch: Optional audit-only action candidate.
        output_mode: Audit-only cognition output mode.
        visual_directives_enabled: Whether this run may generate visual
            directives when global config also allows it.

    Returns:
        Audit-only summary of the dry-run outcome.

    Raises:
        InternalThoughtCognitionDryRunError: If the output mode, residue, or
            action latch violates the dry-run contract.
    """
    if output_mode not in _INTERNAL_THOUGHT_DRY_RUN_OUTPUT_MODES:
        raise InternalThoughtCognitionDryRunError(
            "internal thought output_mode is not supported",
        )

    residue_id = residue["residue_id"]
    action_latch_id = ""
    if action_latch is not None:
        action_latch_id = action_latch["latch_id"]

    if is_primary_interaction_busy():
        audit: InternalThoughtCognitionDryRunAudit = {
            "status": "skipped_busy",
            "skip_reason": "primary_interaction_busy",
            "cognition_called": False,
            "episode_id": "",
            "residue_id": residue_id,
            "action_latch_id": action_latch_id,
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
            "output_mode": output_mode,
            "prompt_variant": "internal_thought_internal_monologue",
            "prompt_keys": [],
            "cognition_output_keys": [],
        }
        return audit

    if "internal_monologue" in residue and residue["internal_monologue"] == "":
        audit = {
            "status": "skipped_empty_residue",
            "skip_reason": "internal_thought_residue_empty",
            "cognition_called": False,
            "episode_id": "",
            "residue_id": residue_id,
            "action_latch_id": action_latch_id,
            "trigger_source": "internal_thought",
            "input_sources": ["internal_monologue"],
            "output_mode": output_mode,
            "prompt_variant": "internal_thought_internal_monologue",
            "prompt_keys": [],
            "cognition_output_keys": [],
        }
        return audit

    episode = build_internal_thought_cognitive_episode(
        residue=residue,
        timestamp=timestamp,
        time_context=time_context,
        action_latch=action_latch,
        output_mode=output_mode,
        visual_directives_enabled=visual_directives_enabled,
    )
    debug_modes = dict(episode["origin_metadata"]["debug_modes"])
    dry_run_state: GlobalPersonaState = {
        "character_profile": character_profile,
        "timestamp": timestamp,
        "time_context": time_context,
        "user_input": _INTERNAL_THOUGHT_INPUT_TEXT,
        "prompt_message_context": {
            "body_text": "",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": "internal_thought",
        "platform_channel_id": "internal_thought_dry_run",
        "channel_type": "internal_thought_dry_run",
        "platform_message_id": "internal_thought:dry_run",
        "platform_user_id": "internal_thought",
        "global_user_id": "internal_thought",
        "user_name": "internal_thought",
        "user_profile": user_profile,
        "platform_bot_id": "internal_thought",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "promoted_reflection_context": {},
        "debug_modes": debug_modes,
        "should_respond": False,
        "decontexualized_input": _INTERNAL_THOUGHT_INPUT_TEXT,
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
    if debug_modes.get("no_visual_directives"):
        prompt_keys = list(_INTERNAL_THOUGHT_DRY_RUN_PROMPT_KEYS_WITHOUT_VISUAL)
    else:
        prompt_keys = list(_INTERNAL_THOUGHT_DRY_RUN_PROMPT_KEYS)

    audit = {
        "status": "completed",
        "skip_reason": "",
        "cognition_called": True,
        "episode_id": episode["episode_id"],
        "residue_id": residue_id,
        "action_latch_id": action_latch_id,
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": output_mode,
        "prompt_variant": "internal_thought_internal_monologue",
        "prompt_keys": prompt_keys,
        "cognition_output_keys": cognition_output_keys,
    }
    return audit


def _validate_residue(residue: InternalThoughtResidue) -> None:
    """Validate private residue before it enters an episode.

    Args:
        residue: Private thought residue supplied to the builder.

    Raises:
        InternalThoughtCognitionDryRunError: If the residue violates the
            internal-thought dry-run contract.
    """
    _require_non_empty_string(residue, "residue_id", "residue")
    internal_monologue = _require_non_empty_string(
        residue,
        "internal_monologue",
        "residue",
    )
    if len(internal_monologue) > _INTERNAL_MONOLOGUE_MAX_CHARACTERS:
        raise InternalThoughtCognitionDryRunError(
            "residue.internal_monologue exceeds 4000 characters",
        )
    source = _require_non_empty_string(residue, "source", "residue")
    if source != "runtime_internal_thought":
        raise InternalThoughtCognitionDryRunError(
            "residue.source is not supported",
        )


def _validate_action_latch(action_latch: InternalActionLatch) -> None:
    """Validate audit-only action latch before it enters an episode.

    Args:
        action_latch: Optional action candidate supplied to the builder.

    Raises:
        InternalThoughtCognitionDryRunError: If the latch violates the
            audit-only action contract.
    """
    _require_non_empty_string(action_latch, "latch_id", "action_latch")
    action_text = _require_non_empty_string(
        action_latch,
        "action_text",
        "action_latch",
    )
    if len(action_text) > _ACTION_LATCH_TEXT_MAX_CHARACTERS:
        raise InternalThoughtCognitionDryRunError(
            "action_latch.action_text exceeds 1000 characters",
        )
    latch_reason = _require_non_empty_string(
        action_latch,
        "latch_reason",
        "action_latch",
    )
    if len(latch_reason) > _ACTION_LATCH_TEXT_MAX_CHARACTERS:
        raise InternalThoughtCognitionDryRunError(
            "action_latch.latch_reason exceeds 1000 characters",
        )
    status = _require_non_empty_string(action_latch, "status", "action_latch")
    if status != "audit_only":
        raise InternalThoughtCognitionDryRunError(
            "action_latch.status is not supported",
        )


def _require_non_empty_string(
    mapping: Mapping[str, object],
    field_name: str,
    container_name: str,
) -> str:
    """Read a required non-empty string from a local dry-run contract.

    Args:
        mapping: Contract mapping to read.
        field_name: Required field name.
        container_name: Name used in error messages.

    Returns:
        Non-empty string value from the mapping.

    Raises:
        InternalThoughtCognitionDryRunError: If the field is absent or empty.
    """
    if field_name not in mapping:
        raise InternalThoughtCognitionDryRunError(
            f"{container_name}.{field_name} is required",
        )
    value = mapping[field_name]
    if not isinstance(value, str) or value == "":
        raise InternalThoughtCognitionDryRunError(
            f"{container_name}.{field_name} is empty",
        )
    return_value = value
    return return_value


def _canonical_percept_content(
    *,
    residue: InternalThoughtResidue,
    action_latch: InternalActionLatch | None,
) -> str:
    """Render internal-thought source content using stable JSON bytes.

    Args:
        residue: Private thought residue supplied to the builder.
        action_latch: Optional audit-only action candidate.

    Returns:
        Canonical JSON string used in the internal-thought percept and id
        digest.
    """
    payload = {
        "residue": residue,
        "action_latch": action_latch or {},
    }
    rendered = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return rendered
