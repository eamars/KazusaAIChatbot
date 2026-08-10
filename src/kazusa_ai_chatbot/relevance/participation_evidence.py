"""Bounded provenance contracts for semantic relevance admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Literal, TypedDict


_MAX_ASSESSMENT_REFS = 3
_MAX_CHARACTER_STATE_ITEMS = 6
_MAX_STATE_SUMMARY_CHARS = 160
_MAX_STATE_EVIDENCE_CHARS = 1400
_MIN_ENTITY_SALIENCE = 25
_MIN_AFFECT_SCORE = 40
_MIN_DRIVE_PRESSURE = 61
_MIN_MEANING_SALIENCE = 61

_ENTITY_KIND_ORDER = {
    "threat": 0,
    "goal": 1,
    "knowledge_gap": 2,
    "event": 3,
    "affect": 4,
    "drive": 5,
    "meaning": 6,
}
_ENTITY_FIELDS = {
    "goal": "goals",
    "threat": "threats",
    "event": "active_events",
    "knowledge_gap": "knowledge_gaps",
}
_ACTIVE_ENTITY_STATUSES = {
    "goal": {"pursuing", "blocked"},
    "threat": {"active"},
    "event": {"active"},
    "knowledge_gap": {"open", "reduced"},
}
_DRIVE_SUMMARIES = {
    "autonomy": "维护自主决定空间",
    "connection": "维持与他人的真实连接",
    "safety": "保护自己和相关参与者的安全",
    "competence": "恢复有效行动和解决问题的能力",
    "care": "照顾可能受到影响的人",
    "integrity": "维护诚实、尊严与行为一致性",
    "exploration": "追查当前未知信息并形成理解",
    "meaning": "恢复目标感与行动意义",
}
_CHARACTER_RECIPIENT_KINDS = {
    "private_scope",
    "typed_character_target",
    "typed_character_reply",
    "canonical_name_span",
    "open_turn",
    "bot_continuity",
    "history_context",
}
_INTERACTION_ADMISSION_KINDS = {
    "private_scope",
    "typed_character_target",
    "typed_character_reply",
    "typed_broadcast",
    "canonical_name_span",
    "current_message",
    "open_turn",
    "bot_continuity",
    "history_context",
}
_ASSESSMENT_FIELDS = {
    "recipient_relation",
    "admission_basis",
    "interaction_evidence_refs",
    "character_state_refs",
}


InteractionEvidenceKind = Literal[
    "private_scope",
    "typed_character_target",
    "typed_character_reply",
    "typed_broadcast",
    "typed_other_target",
    "typed_other_reply",
    "typed_unknown_reply",
    "canonical_name_span",
    "current_message",
    "open_turn",
    "bot_continuity",
    "history_context",
]
CharacterStateEvidenceKind = Literal[
    "goal",
    "threat",
    "event",
    "knowledge_gap",
    "affect",
    "drive",
    "meaning",
]
RecipientRelation = Literal[
    "character",
    "group",
    "current_author",
    "other_participant",
    "participant_1",
    "participant_2",
    "participant_3",
    "participant_4",
    "participant_5",
    "participant_6",
    "participant_7",
    "participant_8",
    "unknown",
]
AdmissionBasis = Literal[
    "interaction_relevance",
    "character_state_salience",
    "none",
]


class InteractionEvidenceItem(TypedDict):
    """One model-visible interaction provenance item."""

    ref: str
    kind: InteractionEvidenceKind
    summary: str


class CharacterStateEvidenceItem(TypedDict):
    """One prompt-safe candidate from active native character state."""

    ref: str
    kind: CharacterStateEvidenceKind
    summary: str
    attention: Literal["active", "pressured"]


class ParticipationAssessment(TypedDict):
    """Model judgment supported by references from the final payload."""

    recipient_relation: RecipientRelation
    admission_basis: AdmissionBasis
    interaction_evidence_refs: list[str]
    character_state_refs: list[str]


def _clip_text(value: object, limit: int) -> str:
    """Clip one semantic field while preserving its head and tail."""

    if not isinstance(value, str) or limit <= 0:
        return_value = ""
        return return_value
    clean_value = value.strip()
    if len(clean_value) <= limit:
        return_value = clean_value
        return return_value
    head_length = max(1, (limit - 3) // 2)
    tail_length = max(1, limit - 3 - head_length)
    return_value = (
        clean_value[:head_length]
        + "..."
        + clean_value[-tail_length:]
    )
    return return_value


def _text_list(value: object) -> list[str]:
    """Return non-empty strings from one optional external sequence."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return_value: list[str] = []
        return return_value
    return_value = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return return_value


def _append_interaction_item(
    evidence: list[InteractionEvidenceItem],
    *,
    ref: str,
    kind: InteractionEvidenceKind,
    summary: str,
) -> None:
    """Append one unique evidence item to the ordered catalog."""

    if any(item["ref"] == ref for item in evidence):
        return
    evidence.append({
        "ref": ref,
        "kind": kind,
        "summary": _clip_text(summary, 240),
    })


def _open_turn_summary(item: Mapping[str, Any]) -> str:
    """Render one supplied open-turn slot without operational identifiers."""

    parts = [
        f"author={_clip_text(item.get('author_relation'), 40) or 'unknown'}",
        f"intent={_clip_text(item.get('latest_intent'), 100)}",
        f"target={_clip_text(item.get('target_summary'), 40) or 'none'}",
        f"reply={_clip_text(item.get('reply_summary'), 40) or 'none'}",
    ]
    summary = "; ".join(parts)
    return summary


def _history_summary(item: Mapping[str, Any]) -> str:
    """Render one stable-handle history row as structural discourse evidence."""

    parts = [
        f"speaker={_clip_text(item.get('speaker_relation'), 40) or 'unknown'}",
        f"target={_clip_text(item.get('target_summary'), 60) or 'none'}",
        f"reply={_clip_text(item.get('reply_summary'), 40) or 'none'}",
        f"relation={_clip_text(item.get('turn_relation'), 40) or 'unknown'}",
        f"body={_clip_text(item.get('body_text'), 120)}",
    ]
    summary = "; ".join(parts)
    return summary


def build_interaction_evidence(
    *,
    conversation_scope: str,
    active_character_name: str,
    current_message: Mapping[str, Any],
    open_turns: Sequence[Mapping[str, Any]],
    latest_bot_continuity: str,
    history: Sequence[Mapping[str, Any]],
) -> list[InteractionEvidenceItem]:
    """Build bounded interaction provenance without judging relevance.

    Args:
        conversation_scope: Canonical ``group`` or ``private`` scope.
        active_character_name: Runtime character name used only for a complete
            contiguous name-span candidate.
        current_message: Typed current-message projection.
        open_turns: Prompt-visible open-turn slots.
        latest_bot_continuity: Bounded recent character continuity.
        history: Prompt-visible history rows with stable participant handles.

    Returns:
        Ordered evidence items that the relevance model may cite.
    """

    if conversation_scope not in {"group", "private"}:
        raise ValueError("interaction conversation_scope is invalid")
    if not isinstance(active_character_name, str) or not (
        active_character_name.strip()
    ):
        raise ValueError("interaction active_character_name is required")
    if not isinstance(current_message, Mapping):
        raise ValueError("interaction current_message must be a mapping")

    evidence: list[InteractionEvidenceItem] = []
    if conversation_scope == "private":
        _append_interaction_item(
            evidence,
            ref="scope_private",
            kind="private_scope",
            summary="private conversation with the active character",
        )

    target_labels = _text_list(
        current_message.get("semantic_target_labels")
    )
    if "character" in target_labels:
        _append_interaction_item(
            evidence,
            ref="target_character",
            kind="typed_character_target",
            summary="typed target identifies the active character",
        )
    if "broadcast" in target_labels:
        _append_interaction_item(
            evidence,
            ref="target_broadcast",
            kind="typed_broadcast",
            summary="typed target identifies the whole group",
        )
    if "other_participant" in target_labels:
        _append_interaction_item(
            evidence,
            ref="target_other",
            kind="typed_other_target",
            summary="typed target identifies another participant",
        )

    reply_targets = _text_list(
        current_message.get("reply_target_labels")
    )
    reply_target = current_message.get("reply_target_label")
    if isinstance(reply_target, str) and reply_target:
        reply_targets.append(reply_target)
    if "character" in reply_targets:
        _append_interaction_item(
            evidence,
            ref="reply_character",
            kind="typed_character_reply",
            summary="typed reply target identifies the active character",
        )
    if "other_participant" in reply_targets:
        _append_interaction_item(
            evidence,
            ref="reply_other",
            kind="typed_other_reply",
            summary="typed reply target identifies another participant",
        )
    if "unknown_participant" in reply_targets:
        _append_interaction_item(
            evidence,
            ref="reply_unknown",
            kind="typed_unknown_reply",
            summary="typed reply target is unresolved",
        )

    body_text = current_message.get("body_text")
    visible_body = body_text if isinstance(body_text, str) else ""
    character_name = active_character_name.strip()
    if character_name in visible_body:
        _append_interaction_item(
            evidence,
            ref="name_1",
            kind="canonical_name_span",
            summary=character_name,
        )
    _append_interaction_item(
        evidence,
        ref="message_1",
        kind="current_message",
        summary=visible_body,
    )

    for index, item in enumerate(list(open_turns)[:3], start=1):
        if not isinstance(item, Mapping):
            continue
        expected_ref = f"open_{index}"
        slot = item.get("slot")
        ref = slot if slot == expected_ref else expected_ref
        _append_interaction_item(
            evidence,
            ref=ref,
            kind="open_turn",
            summary=_open_turn_summary(item),
        )

    if isinstance(latest_bot_continuity, str) and (
        latest_bot_continuity.strip()
    ):
        _append_interaction_item(
            evidence,
            ref="continuity_1",
            kind="bot_continuity",
            summary=latest_bot_continuity,
        )

    for index, item in enumerate(list(history)[-10:], start=1):
        if not isinstance(item, Mapping):
            continue
        _append_interaction_item(
            evidence,
            ref=f"history_{index}",
            kind="history_context",
            summary=_history_summary(item),
        )

    return_value = evidence
    return return_value


def _integer(value: object) -> int | None:
    """Return a native scalar only when it is an integer axis value."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return_value = value
    return return_value


def _active_root_description(
    activation: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    """Resolve an active affect root to its semantic description."""

    root = activation.get("primary_root")
    if not isinstance(root, Mapping):
        return_value = ""
        return return_value
    kind = root.get("kind")
    entity_id = root.get("entity_id")
    if kind not in _ENTITY_FIELDS or not isinstance(entity_id, str):
        return_value = ""
        return return_value
    entities = state.get(_ENTITY_FIELDS[kind])
    if not isinstance(entities, Sequence) or isinstance(
        entities,
        (str, bytes),
    ):
        return_value = ""
        return return_value
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        if entity.get("entity_id") != entity_id:
            continue
        if entity.get("status") not in _ACTIVE_ENTITY_STATUSES[kind]:
            return_value = ""
            return return_value
        description = _clip_text(entity.get("description"), 140)
        return_value = description
        return return_value
    return_value = ""
    return return_value


def _entity_candidates(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect lifecycle-qualified causal entities for prompt projection."""

    candidates: list[dict[str, Any]] = []
    source_order = 0
    for kind in ("goal", "threat", "event", "knowledge_gap"):
        entities = state.get(_ENTITY_FIELDS[kind])
        if not isinstance(entities, Sequence) or isinstance(
            entities,
            (str, bytes),
        ):
            continue
        for entity in entities:
            source_order += 1
            if not isinstance(entity, Mapping):
                continue
            if entity.get("status") not in _ACTIVE_ENTITY_STATUSES[kind]:
                continue
            salience = _integer(entity.get("salience"))
            if kind == "threat":
                residual_pressure = _integer(
                    entity.get("residual_pressure")
                )
                available_strengths = [
                    value
                    for value in (salience, residual_pressure)
                    if value is not None
                ]
                if not available_strengths:
                    continue
                strength = max(available_strengths)
            else:
                if salience is None:
                    continue
                strength = salience
            if strength < _MIN_ENTITY_SALIENCE:
                continue
            summary = _clip_text(
                entity.get("description"),
                _MAX_STATE_SUMMARY_CHARS,
            )
            if not summary:
                continue
            candidates.append({
                "kind": kind,
                "summary": summary,
                "attention": "active",
                "strength": strength,
                "source_order": source_order,
            })
    return_value = candidates
    return return_value


def _affect_candidates(
    state: Mapping[str, Any],
    *,
    source_order: int,
) -> list[dict[str, Any]]:
    """Collect active affect whose primary cause remains resolvable."""

    activations = state.get("affect_activations")
    if not isinstance(activations, Sequence) or isinstance(
        activations,
        (str, bytes),
    ):
        return_value: list[dict[str, Any]] = []
        return return_value
    candidates: list[dict[str, Any]] = []
    for activation in activations:
        source_order += 1
        if not isinstance(activation, Mapping):
            continue
        if (
            activation.get("phase") != "active"
            or activation.get("cause_status") != "active"
        ):
            continue
        score = _integer(activation.get("score"))
        if score is None or score < _MIN_AFFECT_SCORE:
            continue
        root_description = _active_root_description(activation, state)
        emotion_id = _clip_text(activation.get("emotion_id"), 40)
        if not root_description or not emotion_id:
            continue
        summary = _clip_text(
            f"{emotion_id}：{root_description}",
            _MAX_STATE_SUMMARY_CHARS,
        )
        candidates.append({
            "kind": "affect",
            "summary": summary,
            "attention": "active",
            "strength": score,
            "source_order": source_order,
        })
    return_value = candidates
    return return_value


def _drive_and_meaning_candidates(
    state: Mapping[str, Any],
    *,
    source_order: int,
) -> list[dict[str, Any]]:
    """Collect pressured drive and meaning-state semantic candidates."""

    candidates: list[dict[str, Any]] = []
    drives = state.get("drives")
    if isinstance(drives, Mapping):
        for drive_id, drive in drives.items():
            source_order += 1
            if not isinstance(drive_id, str) or not isinstance(
                drive,
                Mapping,
            ):
                continue
            pressure = _integer(drive.get("pressure"))
            summary = _DRIVE_SUMMARIES.get(drive_id)
            if (
                pressure is None
                or pressure < _MIN_DRIVE_PRESSURE
                or summary is None
            ):
                continue
            candidates.append({
                "kind": "drive",
                "summary": summary,
                "attention": "pressured",
                "strength": pressure,
                "source_order": source_order,
            })

    meaning_state = state.get("meaning_state")
    if isinstance(meaning_state, Mapping):
        source_order += 1
        salience = _integer(meaning_state.get("salience"))
        if salience is not None and salience >= _MIN_MEANING_SALIENCE:
            candidates.append({
                "kind": "meaning",
                "summary": "目标感、能动性或身份连续性当前需要关注",
                "attention": "pressured",
                "strength": salience,
                "source_order": source_order,
            })
    return_value = candidates
    return return_value


def _fit_character_state_evidence(
    evidence: list[CharacterStateEvidenceItem],
) -> list[CharacterStateEvidenceItem]:
    """Fit state evidence under the exact rendered character budget."""

    while (
        len(json.dumps(evidence, ensure_ascii=False))
        > _MAX_STATE_EVIDENCE_CHARS
    ):
        longest_index = max(
            range(len(evidence)),
            key=lambda index: len(evidence[index]["summary"]),
        )
        longest_summary = evidence[longest_index]["summary"]
        if len(longest_summary) > 40:
            evidence[longest_index]["summary"] = _clip_text(
                longest_summary,
                len(longest_summary) - 10,
            )
            continue
        evidence.pop()
        if not evidence:
            break
    for index, item in enumerate(evidence, start=1):
        item["ref"] = f"state_{index}"
    return_value = evidence
    return return_value


def project_character_state_evidence(
    state: Mapping[str, Any] | None,
) -> list[CharacterStateEvidenceItem]:
    """Project active native character state without ids or numeric telemetry.

    Args:
        state: Process-local ``cognition_state.v2`` character snapshot, or
            ``None`` when no snapshot is available.

    Returns:
        At most six semantic candidates under the 1,400-character budget.
    """

    if state is None or not state:
        return_value: list[CharacterStateEvidenceItem] = []
        return return_value
    if not isinstance(state, Mapping):
        raise ValueError("character cognition state must be a mapping")
    if state.get("schema_version") != "cognition_state.v2":
        raise ValueError("character cognition state schema is invalid")
    if state.get("state_scope") != "character":
        raise ValueError("character cognition state scope is invalid")

    candidates = _entity_candidates(state)
    source_order = len(candidates)
    candidates.extend(
        _affect_candidates(state, source_order=source_order)
    )
    source_order += len(candidates)
    candidates.extend(
        _drive_and_meaning_candidates(
            state,
            source_order=source_order,
        )
    )
    candidates.sort(key=lambda item: (
        -item["strength"],
        _ENTITY_KIND_ORDER[item["kind"]],
        item["source_order"],
    ))

    evidence: list[CharacterStateEvidenceItem] = []
    for index, candidate in enumerate(
        candidates[:_MAX_CHARACTER_STATE_ITEMS],
        start=1,
    ):
        evidence.append({
            "ref": f"state_{index}",
            "kind": candidate["kind"],
            "summary": candidate["summary"],
            "attention": candidate["attention"],
        })
    return_value = _fit_character_state_evidence(evidence)
    return return_value


def _normalized_refs(
    value: object,
    *,
    available: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Discard unusable model refs and retain a bounded unique list."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    return_value = []
    for ref in value:
        if (
            not isinstance(ref, str)
            or not ref
            or ref not in available
            or ref in return_value
        ):
            continue
        return_value.append(ref)
        if len(return_value) == _MAX_ASSESSMENT_REFS:
            break
    return return_value


def _evidence_map(
    evidence: Sequence[Mapping[str, Any]],
    *,
    catalog_name: str,
) -> dict[str, Mapping[str, Any]]:
    """Index a final prompt-visible catalog and reject duplicate refs."""

    available: dict[str, Mapping[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError(f"{catalog_name} item must be an object")
        ref = item.get("ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{catalog_name} ref is invalid")
        if ref in available:
            raise ValueError(f"{catalog_name} contains duplicate refs")
        available[ref] = item
    return_value = available
    return return_value


def _history_ref_has_relation(
    item: Mapping[str, Any],
    relation: str,
) -> bool:
    """Check a structural history summary for one projected relation token."""

    if item.get("kind") != "history_context":
        return_value = False
        return return_value
    summary = item.get("summary")
    if not isinstance(summary, str):
        return_value = False
        return return_value
    relation_tokens = {
        f"speaker={relation}",
        f"target={relation}",
        f"reply={relation}",
    }
    return_value = any(token in summary for token in relation_tokens)
    return return_value


def _validate_recipient_grounding(
    recipient: str,
    interaction_refs: list[str],
    interaction_map: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require structural provenance for a model-claimed recipient."""

    cited_items = [interaction_map[ref] for ref in interaction_refs]
    cited_kinds = {item.get("kind") for item in cited_items}
    if recipient == "character":
        character_grounded = bool(
            cited_kinds & _CHARACTER_RECIPIENT_KINDS
        ) and any(
            item.get("kind") != "history_context"
            or _history_ref_has_relation(item, "character")
            for item in cited_items
            if item.get("kind") in _CHARACTER_RECIPIENT_KINDS
        )
        if not character_grounded:
            raise ValueError(
                "character recipient requires supplied grounding evidence"
            )
    elif recipient == "group":
        if not cited_kinds & {"typed_broadcast", "current_message"}:
            raise ValueError("group recipient requires supplied evidence")
    elif recipient.startswith("participant_"):
        if not any(
            _history_ref_has_relation(item, recipient)
            for item in cited_items
        ):
            raise ValueError(
                "participant recipient requires matching history evidence"
            )
    elif recipient == "current_author":
        if not any(
            item.get("kind") == "private_scope"
            or _history_ref_has_relation(item, "current_author")
            for item in cited_items
        ):
            raise ValueError(
                "current-author recipient requires supplied evidence"
            )
    elif recipient == "other_participant":
        if not any(
            item.get("kind") in {
                "typed_other_target",
                "typed_other_reply",
                "current_message",
            }
            or _history_ref_has_relation(item, "other_participant")
            for item in cited_items
        ):
            raise ValueError(
                "other recipient requires supplied evidence"
            )


def _validate_interaction_basis(
    recipient: str,
    interaction_refs: list[str],
    interaction_map: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require one cited item that can support character participation."""

    cited_items = [interaction_map[ref] for ref in interaction_refs]
    cited_kinds = {item.get("kind") for item in cited_items}
    if not cited_kinds & _INTERACTION_ADMISSION_KINDS:
        raise ValueError("interaction relevance requires positive evidence")
    if recipient == "group":
        if not cited_kinds & {"typed_broadcast", "current_message"}:
            raise ValueError("group interaction relevance lacks evidence")
        return
    if recipient == "character":
        _validate_recipient_grounding(
            recipient,
            interaction_refs,
            interaction_map,
        )
        return
    continuity_kinds = {
        "private_scope",
        "open_turn",
        "bot_continuity",
        "history_context",
    }
    if not cited_kinds & continuity_kinds:
        raise ValueError(
            "non-character interaction relevance requires continuity"
        )
    history_items = [
        item
        for item in cited_items
        if item.get("kind") == "history_context"
    ]
    if (
        cited_kinds == {"history_context"}
        and not any(
            _history_ref_has_relation(item, "character")
            for item in history_items
        )
    ):
        raise ValueError(
            "history interaction relevance lacks character continuity"
        )


def validate_participation_assessment(
    raw: Mapping[str, Any],
    *,
    interaction_evidence: Sequence[Mapping[str, Any]],
    character_state_evidence: Sequence[Mapping[str, Any]],
    stage: str,
    action: str,
    append_target: str,
    use_reply_feature: bool,
) -> ParticipationAssessment:
    """Validate evidence citations and action consistency for one LLM result.

    Args:
        raw: Parsed model object containing the internal assessment fields.
        interaction_evidence: Exact final interaction catalog shown to the
            model.
        character_state_evidence: Exact final active-state catalog.
        stage: ``frontline`` or ``settled``.
        action: Validated public action for the owning stage.
        append_target: Frontline open slot or ``none``.
        use_reply_feature: Settled native-reply request.

    Returns:
        The exact validated internal participation assessment.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("participation assessment must be an object")
    if not _ASSESSMENT_FIELDS.issubset(raw):
        raise ValueError("participation assessment fields are incomplete")
    if stage not in {"frontline", "settled"}:
        raise ValueError("participation assessment stage is invalid")
    valid_actions = (
        {"discard", "start", "append"}
        if stage == "frontline"
        else {"ignore", "proceed", "wait"}
    )
    if action not in valid_actions:
        raise ValueError("participation assessment action is invalid")
    if not isinstance(append_target, str):
        raise ValueError("participation append_target must be a string")
    if not isinstance(use_reply_feature, bool):
        raise ValueError("participation use_reply_feature must be bool")

    recipients = {
        "character",
        "group",
        "current_author",
        "other_participant",
        "participant_1",
        "participant_2",
        "participant_3",
        "participant_4",
        "participant_5",
        "participant_6",
        "participant_7",
        "participant_8",
        "unknown",
    }
    recipient = raw.get("recipient_relation")
    if recipient not in recipients:
        raise ValueError("participation recipient_relation is invalid")
    basis = raw.get("admission_basis")
    if basis not in {
        "interaction_relevance",
        "character_state_salience",
        "none",
    }:
        raise ValueError("participation admission_basis is invalid")

    interaction_map = _evidence_map(
        interaction_evidence,
        catalog_name="interaction evidence",
    )
    state_map = _evidence_map(
        character_state_evidence,
        catalog_name="character-state evidence",
    )
    interaction_refs = _normalized_refs(
        raw.get("interaction_evidence_refs"),
        available=interaction_map,
    )
    state_refs = _normalized_refs(
        raw.get("character_state_refs"),
        available=state_map,
    )

    admission_actions = {"start", "append", "proceed", "wait"}
    if action in admission_actions and basis == "none":
        raise ValueError("admission action requires a participation basis")
    if action == "discard" and basis != "none":
        raise ValueError("frontline discard requires no admission basis")
    if basis == "none" and state_refs:
        raise ValueError("no admission basis cannot cite character state")
    if basis == "interaction_relevance":
        if state_refs:
            raise ValueError(
                "interaction relevance cannot cite character state"
            )
        _validate_interaction_basis(
            recipient,
            interaction_refs,
            interaction_map,
        )
    elif basis == "character_state_salience":
        if not state_refs:
            raise ValueError(
                "character-state salience requires state evidence"
            )

    if recipient != "unknown":
        _validate_recipient_grounding(
            recipient,
            interaction_refs,
            interaction_map,
        )
    if action == "append":
        append_item = interaction_map.get(append_target)
        if (
            append_target not in interaction_refs
            or append_item is None
            or append_item.get("kind") != "open_turn"
        ):
            raise ValueError(
                "frontline append requires its supplied open-turn ref"
            )
    elif append_target != "none":
        raise ValueError("non-append action must use none append_target")

    if action != "proceed" and use_reply_feature:
        raise ValueError("non-proceed action cannot request a reply anchor")
    if use_reply_feature:
        if recipient != "character":
            raise ValueError(
                "reply anchor requires a character recipient"
            )
        _validate_recipient_grounding(
            "character",
            interaction_refs,
            interaction_map,
        )

    assessment: ParticipationAssessment = {
        "recipient_relation": recipient,
        "admission_basis": basis,
        "interaction_evidence_refs": interaction_refs,
        "character_state_refs": state_refs,
    }
    return_value = assessment
    return return_value
