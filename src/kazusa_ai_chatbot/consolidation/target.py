"""Deterministic consolidation target planning and write validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict


TargetKind = Literal["user", "group_channel", "character", "internal"]
WriteLane = Literal[
    "relationship_insight",
    "user_memory_units",
    "affinity",
    "user_style_image",
    "group_channel_style_image",
    "character_state",
    "character_self_image",
    "character_self_guidance",
    "shared_memory_promotion",
    "audit",
]


class ConsolidationTargetValidationError(ValueError):
    """Raised when target routing would permit an invalid durable write."""


class ConsolidationTarget(TypedDict):
    """One deterministic durable consolidation target."""

    target_alias: str
    target_kind: TargetKind
    target_id: dict[str, str]
    write_lanes: list[str]


class ConsolidationTargetPlan(TypedDict):
    """Deterministic target contract attached before durable writes."""

    origin_kind: str
    targets: list[ConsolidationTarget]


class ConsolidationWriteIntent(TypedDict):
    """Validated internal write intent for one target lane."""

    target_alias: str
    write_lane: str
    payload: dict


USER_TARGET_ALIAS = "current_user"
GROUP_CHANNEL_TARGET_ALIAS = "group_channel"
CHARACTER_TARGET_ALIAS = "character"
INTERNAL_TARGET_ALIAS = "internal"

USER_WRITE_LANES = [
    "relationship_insight",
    "user_memory_units",
    "affinity",
    "user_style_image",
]
GROUP_CHANNEL_WRITE_LANES = ["group_channel_style_image"]
CHARACTER_WRITE_LANES = [
    "character_state",
    "character_self_image",
    "character_self_guidance",
]
INTERNAL_WRITE_LANES = ["audit", "shared_memory_promotion"]

SYNTHETIC_USER_IDS = frozenset(
    (
        "self_cognition",
        "internal_thought",
        "reflection_signal",
        "group_chat_review",
        "scheduled_future_cognition",
        "orchestrator",
        "system",
        "internal",
        "character",
        "group",
        "group_channel",
    )
)


def build_consolidation_target_plan(
    global_state: Mapping[str, Any],
) -> ConsolidationTargetPlan:
    """Build deterministic consolidation targets from trusted runtime state.

    Args:
        global_state: Completed persona or self-cognition state before the
            consolidation subgraph performs durable writes.

    Returns:
        Target plan containing only deterministic target ids and write lanes.
    """

    origin_kind = _origin_kind(global_state)
    targets: list[ConsolidationTarget] = []

    platform = _state_text(global_state, "platform")
    platform_channel_id = _state_text(global_state, "platform_channel_id")
    channel_type = _state_text(global_state, "channel_type")

    if channel_type == "group" and platform and platform_channel_id:
        targets.append(
            {
                "target_alias": GROUP_CHANNEL_TARGET_ALIAS,
                "target_kind": "group_channel",
                "target_id": {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                },
                "write_lanes": list(GROUP_CHANNEL_WRITE_LANES),
            }
        )

    global_user_id = _current_global_user_id(global_state)
    if _is_real_user_id(global_user_id):
        _validate_runtime_user_profile(global_state, global_user_id)
        targets.append(
            {
                "target_alias": USER_TARGET_ALIAS,
                "target_kind": "user",
                "target_id": {
                    "global_user_id": global_user_id,
                    "validated": "true",
                },
                "write_lanes": list(USER_WRITE_LANES),
            }
        )

    if _has_character_target(origin_kind, targets):
        targets.append(
            {
                "target_alias": CHARACTER_TARGET_ALIAS,
                "target_kind": "character",
                "target_id": {"character_id": _character_target_id(global_state)},
                "write_lanes": list(CHARACTER_WRITE_LANES),
            }
        )

    if origin_kind.startswith("reflection"):
        targets.append(
            {
                "target_alias": INTERNAL_TARGET_ALIAS,
                "target_kind": "internal",
                "target_id": {"scope": "internal"},
                "write_lanes": list(INTERNAL_WRITE_LANES),
            }
        )
    elif not targets:
        targets.append(
            {
                "target_alias": INTERNAL_TARGET_ALIAS,
                "target_kind": "internal",
                "target_id": {"scope": "internal"},
                "write_lanes": list(INTERNAL_WRITE_LANES),
            }
        )

    target_plan: ConsolidationTargetPlan = {
        "origin_kind": origin_kind,
        "targets": targets,
    }
    return target_plan


def validate_write_intent(
    intent: Mapping[str, Any],
    target_plan: ConsolidationTargetPlan,
) -> ConsolidationWriteIntent:
    """Validate an internal write intent before persistence code runs.

    Args:
        intent: Internal write request projected by deterministic code.
        target_plan: Deterministic target plan for the current consolidation.

    Returns:
        The validated intent with a dictionary payload.

    Raises:
        ConsolidationTargetValidationError: If the target alias, write lane,
            user identity, or user-profile validation marker is invalid.
    """

    target_alias = _intent_text(intent, "target_alias")
    write_lane = _intent_text(intent, "write_lane")
    payload = intent.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    targets_by_alias = _targets_by_alias(target_plan)
    target = _target_for_alias(target_alias, targets_by_alias)
    if write_lane not in target["write_lanes"]:
        raise ConsolidationTargetValidationError(
            f"write lane {write_lane!r} is not allowed for {target_alias!r}"
        )
    if target["target_kind"] == "user":
        _validate_user_target(target)

    validated_intent: ConsolidationWriteIntent = {
        "target_alias": target_alias,
        "write_lane": write_lane,
        "payload": dict(payload),
    }
    return validated_intent


def _origin_kind(global_state: Mapping[str, Any]) -> str:
    """Read source provenance without using it for write permission."""

    origin = global_state.get("consolidation_origin")
    if isinstance(origin, Mapping):
        trigger_source = origin.get("trigger_source")
        if isinstance(trigger_source, str) and trigger_source.strip():
            return_value = trigger_source.strip()
            return return_value

    episode = global_state.get("cognitive_episode")
    if isinstance(episode, Mapping):
        trigger_source = episode.get("trigger_source")
        if isinstance(trigger_source, str) and trigger_source.strip():
            return_value = trigger_source.strip()
            return return_value

    return_value = _state_text(global_state, "origin_kind") or "unknown"
    return return_value


def _current_global_user_id(global_state: Mapping[str, Any]) -> str:
    """Read the current user id from trusted state or episode scope."""

    global_user_id = _state_text(global_state, "global_user_id")
    if global_user_id:
        return global_user_id

    episode = global_state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = ""
        return return_value

    target_scope = episode.get("target_scope")
    if not isinstance(target_scope, Mapping):
        return_value = ""
        return return_value

    current_global_user_id = target_scope.get("current_global_user_id")
    if isinstance(current_global_user_id, str):
        return_value = current_global_user_id.strip()
    else:
        return_value = ""
    return return_value


def _is_real_user_id(global_user_id: str) -> bool:
    """Return whether a user id is non-empty and not a source label."""

    if not global_user_id:
        return_value = False
        return return_value
    normalized = global_user_id.strip().casefold()
    return_value = normalized not in SYNTHETIC_USER_IDS
    return return_value


def _validate_runtime_user_profile(
    global_state: Mapping[str, Any],
    global_user_id: str,
) -> None:
    """Require a real user target to carry the expected profile shape.

    Args:
        global_state: Runtime state used to build the target plan.
        global_user_id: Real user id selected for the user target.

    Raises:
        ConsolidationTargetValidationError: If the profile shape is malformed.
    """

    user_profile = global_state.get("user_profile")
    if not isinstance(user_profile, Mapping):
        raise ConsolidationTargetValidationError(
            f"user target {global_user_id!r} missing user_profile"
        )
    if "affinity" not in user_profile:
        raise ConsolidationTargetValidationError(
            f"user target {global_user_id!r} missing user_profile.affinity"
        )
    profile_global_user_id = user_profile.get("global_user_id")
    if profile_global_user_id != global_user_id:
        raise ConsolidationTargetValidationError(
            f"user target {global_user_id!r} has mismatched user_profile.global_user_id"
        )


def _has_character_target(
    origin_kind: str,
    targets: list[ConsolidationTarget],
) -> bool:
    """Return whether character-owned state is eligible for this episode."""

    if origin_kind.startswith("reflection"):
        return_value = False
        return return_value
    if origin_kind == "user_message":
        return_value = True
        return return_value
    has_group_channel_target = False
    for target in targets:
        if target["target_kind"] == "group_channel":
            has_group_channel_target = True
            break
    if origin_kind == "internal_thought" and not has_group_channel_target:
        return_value = True
        return return_value
    has_no_durable_target = not targets
    return has_no_durable_target


def _character_target_id(global_state: Mapping[str, Any]) -> str:
    """Return a stable character target id without using user identity."""

    character_profile = global_state.get("character_profile")
    if isinstance(character_profile, Mapping):
        global_user_id = character_profile.get("global_user_id")
        if isinstance(global_user_id, str) and global_user_id.strip():
            return_value = global_user_id.strip()
            return return_value
        name = character_profile.get("name")
        if isinstance(name, str) and name.strip():
            return_value = name.strip()
            return return_value

    return_value = "active_character"
    return return_value


def _targets_by_alias(
    target_plan: ConsolidationTargetPlan,
) -> dict[str, ConsolidationTarget]:
    """Index target rows by alias."""

    targets_by_alias: dict[str, ConsolidationTarget] = {}
    for target in target_plan["targets"]:
        target_alias = target["target_alias"]
        if target_alias in targets_by_alias:
            raise ConsolidationTargetValidationError(
                f"duplicate target alias {target_alias!r}"
            )
        targets_by_alias[target_alias] = target
    return targets_by_alias


def _target_for_alias(
    target_alias: str,
    targets_by_alias: Mapping[str, ConsolidationTarget],
) -> ConsolidationTarget:
    """Return the target for an alias or raise a validation error."""

    if not target_alias:
        raise ConsolidationTargetValidationError(
            "target_alias: expected non-empty string"
        )
    target = targets_by_alias.get(target_alias)
    if target is None:
        raise ConsolidationTargetValidationError(
            f"unknown consolidation target alias {target_alias!r}"
        )
    return target


def _validate_user_target(target: ConsolidationTarget) -> None:
    """Reject synthetic or unvalidated user target rows."""

    target_id = target["target_id"]
    global_user_id = target_id.get("global_user_id", "").strip()
    if not _is_real_user_id(global_user_id):
        raise ConsolidationTargetValidationError(
            f"user target {target['target_alias']!r} has invalid user id"
        )
    if target_id.get("validated") != "true":
        raise ConsolidationTargetValidationError(
            f"user target {target['target_alias']!r} is not validated"
        )


def _intent_text(intent: Mapping[str, Any], field_name: str) -> str:
    """Read one required write-intent string field."""

    value = intent.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ConsolidationTargetValidationError(
            f"{field_name}: expected non-empty string"
        )
    return_value = value.strip()
    return return_value


def _state_text(global_state: Mapping[str, Any], field_name: str) -> str:
    """Read one optional string field from runtime state."""

    value = global_state.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value
