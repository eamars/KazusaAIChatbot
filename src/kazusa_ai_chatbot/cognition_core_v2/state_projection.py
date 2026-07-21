"""Semantic projection from native V2 state into bounded model context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


RAW_STATE_KEYS = frozenset({
    "entity_id",
    "owner_user_id",
    "created_at",
    "updated_at",
    "started_at",
    "last_reinforced_at",
    "primary_root",
    "root_refs",
    "evidence_refs",
    "state_scope",
    "schema_version",
    "scope",
    "kind",
})


@dataclass(frozen=True)
class PromptProjectionV2:
    """Hold prompt-safe values and private handle bindings separately."""

    payload: dict[str, Any]
    handle_to_ref: dict[str, dict[str, str]]


def project_numeric_band(value: int, *, signed: bool = False) -> str:
    """Translate a bounded scalar into the frozen semantic band vocabulary."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("projection value must be an integer")
    if signed:
        if not -100 <= value <= 100:
            raise ValueError("signed projection value is out of range")
        if value <= -61:
            return "强烈负向"
        if value <= -21:
            return "负向"
        if value <= 20:
            return "中性或混合"
        if value <= 60:
            return "正向"
        return "强烈正向"
    if not 0 <= value <= 100:
        raise ValueError("unsigned projection value is out of range")
    if value == 0:
        return "无"
    if value <= 20:
        return "极低"
    if value <= 40:
        return "低"
    if value <= 60:
        return "中等"
    if value <= 80:
        return "高"
    return "极高"


def project_duration(started_at: str, now: str) -> str:
    """Translate elapsed UTC time into the frozen semantic duration labels."""

    elapsed = _parse_utc(now) - _parse_utc(started_at)
    seconds = max(0.0, elapsed.total_seconds())
    if seconds < 10 * 60:
        return "即时"
    if seconds < 2 * 3600:
        return "近期"
    if seconds < 24 * 3600:
        return "较早"
    if seconds < 7 * 24 * 3600:
        return "最近几天内"
    return "较久以前"


def project_relationship_context(
    relationship: Mapping[str, Any],
) -> dict[str, Any]:
    """Project native relationship axes into qualitative prompt context."""

    axes: dict[str, str] = {}
    for field_name in (
        "familiarity",
        "positive_regard",
        "trust",
        "attachment",
        "desired_closeness",
        "perceived_closeness",
        "care",
        "boundary_safety",
        "exclusivity",
        "unresolved_injury",
        "salience",
    ):
        value = relationship.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool):
            axes[field_name] = project_numeric_band(
                value,
                signed=field_name in {
                    "positive_regard",
                    "trust",
                    "boundary_safety",
                },
            )
    return {
        "relationship_summary": "当前关系背景",
        "axes": axes,
    }


def project_trend(previous: int, current: int) -> str:
    """Return direction using the fixed four-point change rule."""

    difference = current - previous
    if difference >= 4:
        return "上升"
    if difference <= -4:
        return "下降"
    return "稳定"


def project_state_for_prompt(
    state: Mapping[str, Any],
    *,
    character_constraints: Mapping[str, Any],
    relationship_context: Mapping[str, Any] | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
) -> PromptProjectionV2:
    """Project all prompt-visible state into semantic descriptors.

    Persistent ids and raw scalar values stay in ``handle_to_ref`` for
    deterministic mapping and are never included in ``payload``.
    """

    handle_to_ref: dict[str, dict[str, str]] = {}
    payload: dict[str, Any] = {
        "goals": [],
        "threats": [],
        "events": [],
        "knowledge_gaps": [],
        "affect": [],
        "causal_candidates": [],
        "evidence": [],
        "roles": {
            "当前角色": "当前角色",
            "当前用户": "当前用户",
        },
        "character_constraints": _project_constraints(character_constraints),
    }
    for field_name, prompt_name, prefix in (
        ("goals", "goals", "g"),
        ("threats", "threats", "t"),
        ("active_events", "events", "e"),
        ("knowledge_gaps", "knowledge_gaps", "k"),
    ):
        for index, entity in enumerate(state[field_name], start=1):
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": _kind_for_field(field_name),
                "entity_id": entity["entity_id"],
            }
            payload[prompt_name].append(
                _project_entity(handle, entity, state["updated_at"])
            )
    relationship = state.get("relationship")
    if relationship is None:
        relationship = relationship_context
    if isinstance(relationship, Mapping):
        handle_to_ref["r1"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": relationship["relationship_id"],
        }
        payload["relationship"] = _project_relationship(relationship)
    for activation in state["affect_activations"]:
        payload["affect"].append(_project_activation(activation, state))
    for index, drive_id in enumerate(state.get("drives", {}), start=1):
        handle_to_ref[f"d{index}"] = {
            "scope": state["state_scope"],
            "kind": "drive",
            "entity_id": drive_id,
        }
    for index, standard in enumerate(state.get("standards", []), start=1):
        handle_to_ref[f"s{index}"] = {
            "scope": state["state_scope"],
            "kind": "standard",
            "entity_id": standard["standard_id"],
        }
    if isinstance(state.get("meaning_state"), Mapping):
        handle_to_ref["m1"] = {
            "scope": state["state_scope"],
            "kind": "meaning",
            "entity_id": "meaning:character",
        }
    owner_user_id = state.get("owner_user_id")
    handle_to_ref["self"] = {
        "scope": "character",
        "kind": "meaning",
        "entity_id": "meaning:character",
    }
    if isinstance(owner_user_id, str) and owner_user_id:
        handle_to_ref["current_user"] = {
            "scope": "user",
            "kind": "relationship",
            "entity_id": f"relationship:user:{owner_user_id}",
        }
    for index, row in enumerate(evidence, start=1):
        evidence_handle = row.get("evidence_handle")
        if not isinstance(evidence_handle, str):
            continue
        evidence_ref = row.get("evidence_ref")
        if isinstance(evidence_ref, Mapping):
            payload["evidence"].append({
                "handle": evidence_handle,
                "source_kind": evidence_ref.get("source_kind", "unknown"),
                "semantic_summary": row.get(
                    "semantic_text",
                    evidence_ref.get("semantic_summary", ""),
                ),
            })
        for kind, prefix, description in (
            ("event", "ce", "当前事件"),
            ("threat", "ct", "可能的当前威胁"),
            ("knowledge_gap", "ck", "可能的当前知识缺口"),
        ):
            handle = f"{prefix}{index}"
            handle_to_ref[handle] = {
                "scope": state["state_scope"],
                "kind": kind,
                "entity_id": f"candidate:{kind}:{evidence_handle}",
            }
            payload["causal_candidates"].append({
                "handle": handle,
                "candidate_kind": kind,
                "evidence_handle": evidence_handle,
                "description": description,
                "lifecycle": "候选，等待有依据的评估",
            })
    validate_prompt_projection(payload)
    return PromptProjectionV2(payload=payload, handle_to_ref=handle_to_ref)


def validate_prompt_projection(payload: Mapping[str, Any]) -> None:
    """Reject raw state fields or private sentinel values in model payloads."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in RAW_STATE_KEYS:
                    raise ValueError(f"raw state key leaked into prompt: {key}")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)


def _project_entity(
    handle: str,
    entity: Mapping[str, Any],
    now: str,
) -> dict[str, Any]:
    """Project one causal entity without ids, timestamps, or raw axes."""

    result: dict[str, Any] = {
        "handle": handle,
        "description": entity["description"],
        "lifecycle": _lifecycle_label(entity["status"]),
        "salience": project_numeric_band(entity["salience"]),
        "duration": project_duration(entity["created_at"], now),
        "causal_roles": _project_roles(entity.get("role_refs", [])),
    }
    for field_name, signed in (
        ("importance", False),
        ("progress", False),
        ("obstruction", False),
        ("urgency", False),
        ("residual_pressure", False),
        ("harm", False),
        ("responsibility", False),
        ("uncertainty", False),
        ("relevance", False),
        ("trust", False),
        ("attachment", False),
        ("positive_regard", True),
    ):
        if field_name in entity:
            result[field_name] = project_numeric_band(
                entity[field_name],
                signed=signed,
            )
    return result


def _project_roles(value: Any) -> list[str]:
    """Project structured causal roles into semantic relationship phrases."""

    if not isinstance(value, list):
        return []
    role_labels = {
        "actor": "行动者",
        "experiencer": "体验者",
        "target": "对象",
        "object": "客体",
        "affected_goal": "受影响目标",
        "affected_relationship": "受影响关系",
    }
    labels: list[str] = []
    for role in value:
        if not isinstance(role, Mapping):
            continue
        role_name = role.get("role")
        if isinstance(role_name, str) and role_name.strip():
            role_label = role_labels.get(role_name.strip(), "语义")
            labels.append(f"{role_label}角色在因果上具有相关性")
    return labels


def _project_relationship(relationship: Mapping[str, Any]) -> dict[str, Any]:
    """Project relationship axes into semantic labels."""
    return {
        "handle": "r1",
        **project_relationship_context(relationship),
    }


def _project_constraints(constraints: Mapping[str, Any]) -> dict[str, Any]:
    """Project character constraints separately from mutable user state."""

    standard_descriptions = {
        "be truthful": "保持诚实",
        "avoid causing needless harm": "避免造成不必要的伤害",
        "respect personal boundaries": "尊重个人边界",
        "honor accepted commitments": "履行已经接受的承诺",
        "protect dignity and autonomy": "保护尊严与自主性",
    }

    drives = {
        drive_id: {
            "importance": project_numeric_band(row["importance"]),
            "pressure": project_numeric_band(row["pressure"]),
        }
        for drive_id, row in constraints["drives"].items()
    }
    standards = [
        {
            "description": standard_descriptions.get(
                row["description"],
                row["description"],
            ),
            "importance": project_numeric_band(row["importance"]),
        }
        for row in constraints["standards"]
    ]
    meaning = {
        field_name: project_numeric_band(constraints["meaning_state"][field_name])
        for field_name in (
            "purpose_coherence",
            "agency",
            "identity_continuity",
            "salience",
        )
    }
    return {"drives": drives, "standards": standards, "meaning_state": meaning}


def _project_activation(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, str]:
    """Project activation lifecycle controls into natural language."""

    return {
        "emotion": activation["emotion_id"],
        "phase": (
            "原因仍然存在"
            if activation["cause_status"] == "active"
            else "该感受在问题解决后逐渐减弱"
        ),
        "intensity": project_numeric_band(activation["score"]),
        "trend": _trend_label(activation["trend"]),
        "cause_summary": _activation_cause_summary(activation, state),
    }


def _activation_cause_summary(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    """Describe the actual primary cause without exposing its identifier."""

    root = activation.get("primary_root")
    if not isinstance(root, Mapping):
        return "有依据的原因仍在当前语境中"
    fields = {
        "goal": "goals",
        "threat": "threats",
        "event": "active_events",
        "knowledge_gap": "knowledge_gaps",
    }
    field_name = fields.get(root.get("kind"))
    if field_name is not None:
        for entity in state.get(field_name, []):
            if (
                isinstance(entity, Mapping)
                and entity.get("entity_id") == root.get("entity_id")
            ):
                description = entity.get("description")
                if isinstance(description, str) and description.strip():
                    return description[:500]
    if root.get("kind") == "relationship":
        return "当前关系带来激活情绪的社会压力"
    if root.get("kind") == "meaning":
        return "目标感和能动性持续偏低"
    return "有依据的原因仍在当前语境中"


def _kind_for_field(field_name: str) -> str:
    """Return the canonical singular entity kind."""

    return {
        "goals": "goal",
        "threats": "threat",
        "active_events": "event",
        "knowledge_gaps": "knowledge_gap",
    }[field_name]


def _lifecycle_label(status: str) -> str:
    """Translate deterministic status into a model-facing descriptor."""

    return {
        "pursuing": "进行中",
        "blocked": "受阻，等待解决",
        "satisfied": "已完成",
        "failed": "失败，等待恢复",
        "abandoned": "已放下",
        "active": "活跃且未解决",
        "resolved": "已解决",
        "replaced": "已被替代",
        "open": "开放且不确定",
        "reduced": "部分减弱但仍不确定",
    }.get(status, status)


def _trend_label(value: str) -> str:
    """Translate a persisted activation trend for model-facing context."""

    return {
        "rising": "上升",
        "stable": "稳定",
        "falling": "下降",
    }.get(value, value)


def _parse_utc(value: str) -> datetime:
    """Parse a required UTC Z timestamp."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("projection timestamp must end in Z")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("projection timestamp must be timezone aware")
    return parsed.astimezone(timezone.utc)
