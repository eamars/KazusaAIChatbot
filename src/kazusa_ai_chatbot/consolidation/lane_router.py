"""Coarse consolidation lane routing and auditable lane pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    CONSOLIDATION_LLM_MODEL,
    CONSOLIDATION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.consolidation.persistence import db_writer
from kazusa_ai_chatbot.consolidation.reflection import (
    character_state_reviewer,
    global_state_updater,
    relationship_profile_reviewer,
    relationship_recorder,
)
from kazusa_ai_chatbot.consolidation.character_self_guidance import (
    character_self_guidance_specialist,
)
from kazusa_ai_chatbot.consolidation.metadata import (
    finalize_consolidation_metadata,
)
from kazusa_ai_chatbot.consolidation.source_policy import (
    ASSISTANT_ACCEPTANCE_SOURCE_KIND,
    build_consolidation_source_views,
    source_refs_from_views,
    validate_lane_source_policy,
)
from kazusa_ai_chatbot.consolidation.target import (
    CHARACTER_TARGET_ALIAS,
    GROUP_CHANNEL_TARGET_ALIAS,
    INTERNAL_TARGET_ALIAS,
    USER_TARGET_ALIAS,
    ConsolidationTargetPlan,
    ConsolidationTargetValidationError,
    validate_write_intent,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_tool_result_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

CONSOLIDATION_LANE_NAMES = (
    "character_state",
    "relationship_profile",
    "user_memory_units",
    "active_commitment",
    "character_self_guidance",
    "interaction_style_image",
    "shared_memory_promotion",
)

_ROUTER_TASK_KEYS = frozenset(("lane", "reason", "source_keys"))
_FORBIDDEN_ROUTER_TASK_KEYS = frozenset(
    ("target_id", "write_lane", "payload", "fact")
)
_MAX_ROUTER_TASKS = 4

_LANE_DESCRIPTIONS = {
    "character_state": "Update the active character's durable mood, vibe, or self-state.",
    "relationship_profile": "Update the relationship profile for the current real user.",
    "user_memory_units": "Store durable facts, patterns, shifts, or milestones about the current real user.",
    "active_commitment": "Store an accepted promise or ongoing rule for the current user specifically.",
    "character_self_guidance": "Store accepted general future behavior guidance owned by the character.",
    "interaction_style_image": "Update user or group interaction-style image overlays.",
    "shared_memory_promotion": "Admit promoted reflection evidence into shared memory only.",
}

_ROUTER_PROMPT = '''\
You route one completed episode into coarse consolidation lane tasks.

The human payload contains:
- target_plan: deterministic information about eligible durable targets.
- lane_roster: the only lane names available for this episode.
- source_views: prompt-safe evidence rows with source_key values.

Choose zero to four lane tasks from lane_roster. A task means the episode has
a durable update that should be inspected by that lane's specialist. Return
only lane names, short reasons, and source_keys from source_views. Persistence
details, memory text, target identifiers, timestamps, and cache behavior are
owned by later deterministic stages.

# Decision Procedure
1. Read the source_views and decide whether the episode contains a durable
   memory update after the completed response.
2. Identify the owner and scope of that update: current user, relationship,
   active character, group/channel style, or approved reflection promotion.
3. Select the matching lane from lane_roster. If no roster lane owns the
   durable update, return an empty lane_tasks list.
4. For accepted future behavior rules, include both the request source and the
   final-dialog acceptance source when both source_keys are available.
5. When the durable subject is a user fact or preference and the character
   only acknowledges, remembers, respects, or accommodates it, route the
   user-owned lane. Treat the character's accommodation as support for that
   user memory rather than a separate character behavior rule.
6. When the durable subject is feedback about the current relationship or a
   local repair of the recent interaction, route the relationship-owned lane.
   Acknowledging discomfort or promising to be more careful in that local
   repair is evidence for relationship_profile unless the final dialog accepts
   a standalone general future behavior rule.
7. Keep the router coarse. The selected specialist will write or reject the
   actual memory candidate.

# Lane Ownership
- relationship_profile: relationship feedback for the current real user,
  including trust, comfort, disappointment, repair, tension, or how the recent
  character behavior landed with the user.
- user_memory_units: durable information about the current real user, such as
  personal facts, preferences, habits, recent shifts, milestones, or updates to
  recalled user memory.
- active_commitment: accepted future behavior scoped to the current user, such
  as a promise, reminder, address rule, or ongoing interaction rule for that
  user.
- character_self_guidance: accepted future behavior guidance where the future
  behavior itself is owned by the active character generally across future
  social situations.
- character_state: accepted or evidenced character self-continuity, durable
  mood, vibe, identity, trait, or self-description.
- interaction_style_image: user-style or group/channel interaction norms when
  the target plan and source role make that style target available.
- shared_memory_promotion: approved reflection or shared-memory promotion
  evidence with privacy review.

# Skip Criteria
Return an empty lane_tasks list for one-turn roleplay or temporary behavior,
ordinary world lore in chat, third-party facts that are not about the current
user, and proposed future behavior that the final dialog leaves unaccepted or
only situational.

# Output Format
Return only valid JSON:
{
  "lane_tasks": [
    {
      "lane": "one lane from the roster",
      "reason": "short semantic reason",
      "source_keys": ["source_key"]
    }
  ]
}
'''

_lane_router_llm = LLInterface()
_lane_router_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CONSOLIDATION_LLM",
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
    model=CONSOLIDATION_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CONSOLIDATION_LLM_THINKING_ENABLED,
    ),
)


def build_lane_roster(
    target_plan: ConsolidationTargetPlan,
) -> list[dict[str, str]]:
    """Build router-visible lane roster from deterministic write lanes.

    Args:
        target_plan: Deterministic target plan attached before routing.

    Returns:
        Roster rows containing only currently possible lane names and
        descriptions.
    """

    write_lanes = set()
    target_kinds = set()
    for target in target_plan["targets"]:
        write_lanes.update(target["write_lanes"])
        target_kinds.add(target["target_kind"])

    reflection_origin = target_plan["origin_kind"].startswith("reflection")
    roster: list[dict[str, str]] = []
    if reflection_origin:
        if (
            "user_style_image" in write_lanes
            or "group_channel_style_image" in write_lanes
        ):
            roster.append(_roster_entry("interaction_style_image"))
        roster.append(_roster_entry("shared_memory_promotion"))
        return roster

    if "character_state" in write_lanes:
        roster.append(_roster_entry("character_state"))
    if "relationship_insight" in write_lanes or "affinity" in write_lanes:
        roster.append(_roster_entry("relationship_profile"))
    if "user_memory_units" in write_lanes:
        roster.append(_roster_entry("user_memory_units"))
        roster.append(_roster_entry("active_commitment"))
    if "character_self_guidance" in write_lanes:
        roster.append(_roster_entry("character_self_guidance"))
    if "group_channel_style_image" in write_lanes:
        roster.append(_roster_entry("interaction_style_image"))
    if "internal" in target_kinds:
        roster.append(_roster_entry("shared_memory_promotion"))

    return roster


def validate_lane_router_output(
    output: Mapping[str, Any],
    roster: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate that router output contains only coarse lane tasks.

    Args:
        output: Parsed router JSON.
        roster: Lane roster built from the target plan.

    Returns:
        Normalized router output with validated lane tasks.

    Raises:
        ValueError: If the output contains unknown lanes, non-roster lanes,
            persistence fields, memory payload fields, or malformed task rows.
    """

    lane_tasks = output.get("lane_tasks")
    if not isinstance(lane_tasks, list):
        raise ValueError("lane_tasks must be a list")

    roster_lanes = {
        text_or_empty(entry.get("lane"))
        for entry in roster
        if isinstance(entry, Mapping)
    }
    validated_tasks: list[dict[str, Any]] = []
    for raw_task in lane_tasks[:_MAX_ROUTER_TASKS]:
        if not isinstance(raw_task, Mapping):
            raise ValueError("lane task must be an object")
        task_keys = set(raw_task)
        if task_keys & _FORBIDDEN_ROUTER_TASK_KEYS:
            raise ValueError("router task contains persistence or memory fields")
        if task_keys != _ROUTER_TASK_KEYS:
            raise ValueError("router task must contain only lane, reason, source_keys")

        lane = text_or_empty(raw_task.get("lane"))
        if lane not in CONSOLIDATION_LANE_NAMES:
            raise ValueError(f"unknown consolidation lane: {lane!r}")
        if lane not in roster_lanes:
            raise ValueError(f"lane is not in target roster: {lane!r}")

        reason = text_or_empty(raw_task.get("reason"))
        raw_source_keys = raw_task.get("source_keys")
        if not isinstance(raw_source_keys, list):
            raise ValueError("source_keys must be a list")
        source_keys = [
            source_key.strip()
            for source_key in raw_source_keys
            if isinstance(source_key, str) and source_key.strip()
        ]
        validated_tasks.append(
            {
                "lane": lane,
                "reason": reason,
                "source_keys": source_keys,
            }
        )

    validated_output = {"lane_tasks": validated_tasks}
    return validated_output


async def call_lane_router_llm(
    state: Mapping[str, Any],
    *,
    source_views: list[dict[str, Any]],
    roster: list[dict[str, str]],
) -> dict[str, Any]:
    """Call the background LLM that chooses coarse consolidation lanes.

    Args:
        state: Consolidator state carrying the target plan and turn metadata.
        source_views: Transient source-view rows built from the current state.
        roster: Deterministically pruned lane roster.

    Returns:
        Parsed JSON object returned by the router LLM.
    """

    target_plan = state["consolidation_target_plan"]
    payload = {
        "target_plan": project_tool_result_for_llm(target_plan),
        "lane_roster": roster,
        "source_views": source_views,
    }
    system_prompt = SystemMessage(content=_ROUTER_PROMPT)
    human_message = HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    response = await _lane_router_llm.ainvoke(
        [system_prompt, human_message],
        config=_lane_router_llm_config,
    )
    parsed_output = parse_llm_json_output(response.content)
    return parsed_output


async def run_consolidation_lane_pipeline(
    state: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run source-view, lane-router, source-policy, and persistence handling.

    Args:
        state: Consolidator state after target planning.
        dry_run: When true, return write intents without persistence.

    Returns:
        Auditable packet containing mode, source views, router tasks,
        per-lane results, write intents, and the final working state.
    """

    source_views = build_consolidation_source_views(state)
    source_views_by_key = _source_views_by_key(source_views)
    target_plan = state["consolidation_target_plan"]
    roster = build_lane_roster(target_plan)
    router_output = await call_lane_router_llm(
        state,
        source_views=source_views,
        roster=roster,
    )
    router_validation_error = ""
    try:
        validated_router_output = validate_lane_router_output(
            router_output,
            roster,
        )
    except ValueError as exc:
        logger.warning(f"lane router output dropped: {exc}")
        router_tasks = []
        router_validation_error = str(exc)
    else:
        router_tasks = validated_router_output["lane_tasks"]

    lane_results: list[dict[str, Any]] = []
    write_intents: list[dict[str, Any]] = []
    accepted_lanes: list[str] = []
    accepted_user_memory_refs: list[dict[str, Any]] = []
    accepted_self_guidance_refs: list[dict[str, Any]] = []

    for task in router_tasks:
        selected_views = _selected_source_views(task, source_views_by_key)
        selected_views = _complete_required_source_views(
            task["lane"],
            selected_views,
            source_views_by_key,
        )
        selected_source_keys = [
            text_or_empty(source_view.get("source_key"))
            for source_view in selected_views
            if text_or_empty(source_view.get("source_key"))
        ]
        source_policy = validate_lane_source_policy(
            task["lane"],
            selected_views,
            privacy_review=_privacy_review_for_state_or_views(
                state,
                selected_views,
            ),
        )
        lane_result = {
            "lane": task["lane"],
            "reason": task["reason"],
            "source_policy": source_policy,
            "source_keys": selected_source_keys,
        }
        if not source_policy["accepted"]:
            lane_result["status"] = "rejected"
            lane_results.append(lane_result)
            continue

        source_refs = source_refs_from_views(selected_views)
        write_intent = _write_intent_for_lane(
            task["lane"],
            target_plan,
            source_refs,
        )
        if write_intent is not None:
            write_intents.append(write_intent)
        accepted_lanes.append(task["lane"])
        if task["lane"] in {"user_memory_units", "active_commitment"}:
            accepted_user_memory_refs.extend(source_refs)
        if task["lane"] == "character_self_guidance":
            accepted_self_guidance_refs.extend(source_refs)
        lane_result["status"] = "accepted"
        lane_result["write_intent"] = write_intent
        lane_results.append(lane_result)

    working_state = dict(state)
    working_state["enabled_consolidation_write_lanes"] = accepted_lanes
    working_state["user_memory_unit_source_refs"] = accepted_user_memory_refs
    working_state["character_self_guidance_source_refs"] = accepted_self_guidance_refs
    _ensure_writer_defaults(working_state)

    if not dry_run and accepted_lanes:
        await _run_lane_specialists(working_state, accepted_lanes)
        if accepted_lanes:
            writer_result = await db_writer(working_state)
            working_state.update(writer_result)
        else:
            metadata = dict(working_state.get("metadata", {}) or {})
            metadata["write_success"] = {}
            working_state["metadata"] = metadata
    else:
        metadata = dict(working_state.get("metadata", {}) or {})
        metadata["write_success"] = {}
        working_state["metadata"] = metadata

    metadata = dict(working_state.get("metadata", {}) or {})
    metadata["lane_pipeline"] = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "write_intent_count": len(write_intents),
    }
    if router_validation_error:
        metadata["lane_pipeline"]["router_validation_error"] = (
            router_validation_error
        )
    working_state["metadata"] = finalize_consolidation_metadata(metadata)

    packet = {
        "mode": "dry_run" if dry_run else "apply",
        "accepted_lanes": accepted_lanes,
        "source_views": source_views,
        "router_tasks": router_tasks,
        "lane_results": lane_results,
        "write_intents": write_intents,
        "state": working_state,
    }
    return packet


def _roster_entry(lane: str) -> dict[str, str]:
    """Build one router-visible roster row."""

    roster_entry = {
        "lane": lane,
        "description": _LANE_DESCRIPTIONS[lane],
    }
    return roster_entry


def _source_views_by_key(
    source_views: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index source views by source key."""

    views_by_key: dict[str, dict[str, Any]] = {}
    for source_view in source_views:
        source_key = text_or_empty(source_view.get("source_key"))
        if source_key:
            views_by_key[source_key] = source_view
    return views_by_key


def _selected_source_views(
    task: Mapping[str, Any],
    source_views_by_key: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve router-selected source keys to source-view rows."""

    selected_views: list[dict[str, Any]] = []
    raw_source_keys = task.get("source_keys")
    if not isinstance(raw_source_keys, list):
        return selected_views
    for source_key in raw_source_keys:
        clean_source_key = text_or_empty(source_key)
        if not clean_source_key:
            continue
        source_view = source_views_by_key.get(clean_source_key)
        if source_view is not None:
            selected_views.append(source_view)
    return selected_views


def _complete_required_source_views(
    lane: str,
    selected_views: list[dict[str, Any]],
    source_views_by_key: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach structurally required provenance for accepted-rule lanes."""

    if lane not in {"active_commitment", "character_self_guidance"}:
        return selected_views

    completed_views = list(selected_views)
    selected_keys = {
        text_or_empty(source_view.get("source_key"))
        for source_view in selected_views
    }
    for required_key in (
        "current_turn_user_message",
        ASSISTANT_ACCEPTANCE_SOURCE_KIND,
    ):
        if required_key in selected_keys:
            continue
        source_view = source_views_by_key.get(required_key)
        if source_view is not None:
            completed_views.append(source_view)
            selected_keys.add(required_key)
    return completed_views


def _privacy_review_for_state_or_views(
    state: Mapping[str, Any],
    source_views: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return optional privacy-review payload from state or selected views."""

    privacy_review = state.get("privacy_review")
    if isinstance(privacy_review, dict):
        return privacy_review
    for source_view in source_views:
        privacy_review = source_view.get("privacy_review")
        if isinstance(privacy_review, dict):
            return privacy_review
    return_value = None
    return return_value


def _write_intent_for_lane(
    lane: str,
    target_plan: ConsolidationTargetPlan,
    source_refs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build and validate one lane-level write intent."""

    target_alias, write_lane = _target_alias_and_write_lane(lane, target_plan)
    if not target_alias or not write_lane:
        return_value = None
        return return_value

    intent = {
        "target_alias": target_alias,
        "write_lane": write_lane,
        "payload": {"source_refs": source_refs},
    }
    try:
        validated_intent = validate_write_intent(intent, target_plan)
    except ConsolidationTargetValidationError as exc:
        logger.debug(f"lane write intent denied: {lane}: {exc}")
        return_value = None
        return return_value
    return_value = validated_intent
    return return_value


def _target_alias_and_write_lane(
    lane: str,
    target_plan: ConsolidationTargetPlan,
) -> tuple[str, str]:
    """Map consolidation lane names to existing target-plan write lanes."""

    if lane == "character_state":
        return_value = (CHARACTER_TARGET_ALIAS, "character_state")
        return return_value
    if lane == "relationship_profile":
        return_value = (USER_TARGET_ALIAS, "relationship_insight")
        return return_value
    if lane in {"user_memory_units", "active_commitment"}:
        return_value = (USER_TARGET_ALIAS, "user_memory_units")
        return return_value
    if lane == "character_self_guidance":
        return_value = (CHARACTER_TARGET_ALIAS, "character_self_guidance")
        return return_value
    if lane == "interaction_style_image":
        for target in target_plan["targets"]:
            if (
                target["target_alias"] == GROUP_CHANNEL_TARGET_ALIAS
                and "group_channel_style_image" in target["write_lanes"]
            ):
                return_value = (
                    GROUP_CHANNEL_TARGET_ALIAS,
                    "group_channel_style_image",
                )
                return return_value
        return_value = (USER_TARGET_ALIAS, "user_style_image")
        return return_value
    if lane == "shared_memory_promotion":
        return_value = (INTERNAL_TARGET_ALIAS, "shared_memory_promotion")
        return return_value
    return_value = ("", "")
    return return_value


def _ensure_writer_defaults(working_state: dict[str, Any]) -> None:
    """Populate writer state defaults produced by omitted lane specialists."""

    working_state.setdefault("mood", "")
    working_state.setdefault("global_vibe", "")
    working_state.setdefault("reflection_summary", "")
    working_state.setdefault("subjective_appraisals", [])
    working_state.setdefault("affinity_delta", 0)
    working_state.setdefault("last_relationship_insight", "")
    working_state.setdefault("new_facts", [])
    working_state.setdefault("future_promises", [])
    working_state.setdefault("character_self_guidance", {})
    working_state.setdefault("group_channel_style_image", {})
    working_state.setdefault("metadata", {})


async def _run_lane_specialists(
    working_state: dict[str, Any],
    accepted_lanes: list[str],
) -> None:
    """Run existing lane-local specialists before persistence."""

    accepted_lane_set = set(accepted_lanes)
    if "character_state" in accepted_lane_set:
        character_candidate = await global_state_updater(working_state)
        character_patch = await character_state_reviewer(
            working_state,
            character_candidate,
        )
        if _character_state_patch_has_content(character_patch):
            working_state.update(character_patch)
        else:
            _disable_accepted_lane(
                working_state,
                accepted_lanes,
                "character_state",
            )
    if "relationship_profile" in accepted_lane_set:
        relationship_candidate = await relationship_recorder(working_state)
        relationship_patch = await relationship_profile_reviewer(
            working_state,
            relationship_candidate,
        )
        if _relationship_profile_patch_has_content(relationship_patch):
            working_state.update(relationship_patch)
        else:
            _disable_accepted_lane(
                working_state,
                accepted_lanes,
                "relationship_profile",
            )
    if "character_self_guidance" in accepted_lane_set:
        self_guidance_patch = await character_self_guidance_specialist(
            working_state
        )
        working_state.update(self_guidance_patch)


def _character_state_patch_has_content(patch: Mapping[str, Any]) -> bool:
    """Return whether a reviewed character-state patch can be persisted."""

    return_value = any(
        text_or_empty(patch.get(field_name))
        for field_name in ("mood", "global_vibe", "reflection_summary")
    )
    return return_value


def _relationship_profile_patch_has_content(patch: Mapping[str, Any]) -> bool:
    """Return whether a reviewed relationship patch can be persisted."""

    raw_appraisals = patch.get("subjective_appraisals")
    has_appraisals = isinstance(raw_appraisals, list) and bool(raw_appraisals)
    has_delta = patch.get("affinity_delta", 0) not in (0, None, "")
    has_insight = bool(text_or_empty(patch.get("last_relationship_insight")))
    return_value = has_appraisals or has_delta or has_insight
    return return_value


def _disable_accepted_lane(
    working_state: dict[str, Any],
    accepted_lanes: list[str],
    lane: str,
) -> None:
    """Remove a reviewer-rejected lane from the persistence allow-list."""

    while lane in accepted_lanes:
        accepted_lanes.remove(lane)

    enabled_lanes = working_state.get("enabled_consolidation_write_lanes")
    if isinstance(enabled_lanes, list):
        working_state["enabled_consolidation_write_lanes"] = [
            enabled_lane
            for enabled_lane in enabled_lanes
            if enabled_lane != lane
        ]

    metadata = dict(working_state.get("metadata", {}) or {})
    rejected_lanes = list(metadata.get("review_rejected_lanes", []) or [])
    rejected_lanes.append(lane)
    metadata["review_rejected_lanes"] = rejected_lanes
    working_state["metadata"] = metadata
