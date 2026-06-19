"""Primary-participant context hydration for group-review self-cognition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from kazusa_ai_chatbot.db import (
    build_user_engagement_relevance_context,
    get_user_profile,
)
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import build_affinity_block, text_or_empty

SOURCE_NAME = "group_review_participant_context"
CONTEXT_SHAPE = "single_flow_focus"
THREAD_REFERENCE_SOURCE_NAME = "group_review_thread_reference"
THREAD_REFERENCE_CONTEXT_SHAPE = "bounded_second_person_reference_warnings"
THREAD_REFERENCE_GUIDANCE = (
    "二人称归属按同一行明确地址和可见线程读取；"
    "缺少同一行当前角色指向时，保留为侧线/未定对象。"
)
THREAD_REFERENCE_ROW_LIMIT = 3
THREAD_REFERENCE_ADJACENT_LOOKBACK_ROWS = 2
CONVERSATION_HELPER_ATTEMPTS = 1
CONVERSATION_LOOKBACK_HOURS = 72
CONVERSATION_EVIDENCE_LIMIT = 3
CONVERSATION_EVIDENCE_CHAR_LIMIT = 240
VISIBLE_SAMPLE_LIMIT = 3
VISIBLE_SAMPLE_CHAR_LIMIT = 160
ENGAGEMENT_GUIDELINES_LIMIT = 3
_SECOND_PERSON_MARKERS = ("你", "您")

_ROLE_ORDER = {
    "direct_cue": 0,
    "reply_to_character": 1,
    "latest_speaker": 2,
    "topic_author": 3,
    "dominant_speaker": 4,
}
_FIT_ORDER = ("high", "medium", "low")
_GUIDANCE = {
    "group_pile_on": (
        "Address the shared pile-on or accusation as one beat; do not answer "
        "each participant one by one."
    ),
    "direct_reply": (
        "Reply to the primary target or current thread only; do not fan out."
    ),
    "continue_visible_thread": (
        "Follow the current discussion flow without widening the reply."
    ),
    "ambient_observation": (
        "Use participant context only as background; speak only if the "
        "visible group flow itself gives enough reason."
    ),
}


@dataclass
class _ParticipantCandidate:
    """Internal candidate assembled from user-authored group-window rows."""

    key: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    roles: set[str] = field(default_factory=set)
    global_user_id: str = ""
    platform_user_id: str = ""
    display_name: str = ""
    latest_timestamp: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def add_row(self, row: dict[str, Any]) -> None:
        """Append a source row and update latest visible identity fields."""

        self.rows.append(row)
        global_user_id = text_or_empty(row.get("global_user_id"))
        platform_user_id = text_or_empty(row.get("platform_user_id"))
        display_name = text_or_empty(row.get("display_name"))
        if global_user_id:
            self.global_user_id = global_user_id
        if platform_user_id:
            self.platform_user_id = platform_user_id
        if display_name:
            self.display_name = display_name

        row_timestamp = _row_timestamp(row)
        if row_timestamp >= self.latest_timestamp:
            self.latest_timestamp = row_timestamp

    @property
    def reply_target_fit(self) -> str:
        """Return deterministic target-fit tier from candidate roles."""

        if "direct_cue" in self.roles or "reply_to_character" in self.roles:
            return_value = "high"
            return return_value
        medium_roles = {"latest_speaker", "topic_author", "dominant_speaker"}
        if self.roles.intersection(medium_roles):
            return_value = "medium"
            return return_value
        return_value = "low"
        return return_value

    @property
    def score(self) -> int:
        """Return a deterministic tie-break score within the same fit tier."""

        score = len(self.roles) + len(self.rows)
        return score


async def build_group_review_participant_context(
    *,
    participant_rows: list[dict[str, Any]],
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    window_start_utc: str,
    current_timestamp_utc: str,
    conversation_agent: ConversationEvidenceAgent | None = None,
) -> dict[str, Any] | None:
    """Build bounded, flow-focused participant context for group review.

    Args:
        participant_rows: Internal source rows from one group activity window.
        target_scope: Group-scoped self-cognition target metadata.
        character_profile: Active character profile snapshot used only for
            direct-cue and reply-to-character detection.
        window_start_utc: Activity-window start; conversation lookback ends
            before this timestamp.
        current_timestamp_utc: Current collection timestamp for helper context.
        conversation_agent: Optional test seam for the conversation-evidence
            helper. Production uses ``ConversationEvidenceAgent`` directly.

    Returns:
        Prompt-facing participant context, or ``None`` when the window contains
        no user-authored participant rows.
    """

    user_rows = _user_rows(participant_rows)
    if not user_rows:
        return_value = None
        return return_value

    candidates = _build_candidates(
        user_rows,
        character_profile=character_profile,
    )
    if not candidates:
        context = _context_payload(
            focus_mode="ambient_observation",
            primary_reply_target={},
            background_flow=_background_flow(
                primary=None,
                candidate_count=0,
                high_fit_count=0,
            ),
        )
        return context

    primary = _select_primary(candidates)
    high_fit_count = _high_fit_count(candidates)
    focus_mode = _focus_mode(primary, high_fit_count)
    hydration = await _hydrate_primary(
        primary,
        target_scope=target_scope,
        character_profile=character_profile,
        window_start_utc=window_start_utc,
        current_timestamp_utc=current_timestamp_utc,
        conversation_agent=conversation_agent,
    )
    primary_reply_target = _primary_reply_target(primary, hydration)
    background_flow = _background_flow(
        primary=primary,
        candidate_count=len(candidates),
        high_fit_count=high_fit_count,
    )
    context = _context_payload(
        focus_mode=focus_mode,
        primary_reply_target=primary_reply_target,
        background_flow=background_flow,
    )
    return context


def build_group_review_thread_reference_context(
    participant_rows: list[dict[str, Any]],
    character_profile: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build bounded warnings for ambiguous second-person group rows.

    Args:
        participant_rows: Prompt-safe activity-window participant rows.
        character_profile: Active character profile used only to recognize
            direct row-level address.

    Returns:
        Prompt-facing thread-reference context, or ``None`` when no ambiguous
        second-person row is visible.
    """

    user_rows = _user_rows(participant_rows)
    ambiguous_rows: list[dict[str, str]] = []
    for index, row in enumerate(user_rows):
        body_text = text_or_empty(row.get("body_text"))
        if not _contains_second_person(body_text):
            continue
        if _row_has_direct_cue(row, character_profile=character_profile):
            continue
        if _row_replies_to_character(row, character_profile=character_profile):
            continue

        sample = _clip_text(body_text, VISIBLE_SAMPLE_CHAR_LIMIT)
        if not sample:
            continue
        warning = {
            "speaker": text_or_empty(row.get("display_name")) or "visible speaker",
            "sample": sample,
            "referent_status": "ambiguous_or_side_thread",
            "basis": _thread_reference_basis(
                user_rows=user_rows,
                row_index=index,
            ),
        }
        ambiguous_rows.append(warning)
        if len(ambiguous_rows) >= THREAD_REFERENCE_ROW_LIMIT:
            break

    if not ambiguous_rows:
        return_value = None
        return return_value

    context = {
        "source": THREAD_REFERENCE_SOURCE_NAME,
        "context_shape": THREAD_REFERENCE_CONTEXT_SHAPE,
        "guidance": THREAD_REFERENCE_GUIDANCE,
        "ambiguous_second_person_rows": ambiguous_rows,
    }
    return context


def _user_rows(
    participant_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return user-authored participant rows in source order."""

    rows = [
        dict(row)
        for row in participant_rows
        if text_or_empty(row.get("role")) == "user"
    ]
    return rows


def _contains_second_person(text: str) -> bool:
    """Return whether visible text contains a second-person marker."""

    contains_marker = any(marker in text for marker in _SECOND_PERSON_MARKERS)
    return contains_marker


def _thread_reference_basis(
    *,
    user_rows: list[dict[str, Any]],
    row_index: int,
) -> str:
    """Return semantic basis text for an ambiguous second-person row."""

    basis_parts = ["same row has no direct active-character address"]
    if _adjacent_flow_points_elsewhere(
        user_rows=user_rows,
        row_index=row_index,
    ):
        basis_parts.append(
            "adjacent visible flow points to another participant thread"
        )
    basis = "; ".join(basis_parts)
    return basis


def _adjacent_flow_points_elsewhere(
    *,
    user_rows: list[dict[str, Any]],
    row_index: int,
) -> bool:
    """Return whether nearby visible rows suggest another participant thread."""

    current_speaker = text_or_empty(user_rows[row_index].get("display_name"))
    prior_start = max(0, row_index - THREAD_REFERENCE_ADJACENT_LOOKBACK_ROWS)
    prior_rows = user_rows[prior_start:row_index]
    for prior_row in prior_rows:
        prior_speaker = text_or_empty(prior_row.get("display_name"))
        if prior_speaker and prior_speaker != current_speaker:
            return_value = True
            return return_value

    current_text = text_or_empty(user_rows[row_index].get("body_text"))
    for prior_row in prior_rows:
        prior_text = text_or_empty(prior_row.get("body_text"))
        if current_speaker and current_speaker in prior_text:
            return_value = True
            return return_value
        prior_speaker = text_or_empty(prior_row.get("display_name"))
        if prior_speaker and prior_speaker in current_text:
            return_value = True
            return return_value

    return_value = False
    return return_value


def _build_candidates(
    user_rows: list[dict[str, Any]],
    *,
    character_profile: Mapping[str, Any],
) -> list[_ParticipantCandidate]:
    """Group user rows into deterministic participant candidates."""

    candidates_by_key: dict[str, _ParticipantCandidate] = {}
    row_keys: dict[int, str] = {}
    for index, row in enumerate(user_rows):
        key = _candidate_key(row)
        if not key:
            continue
        if key not in candidates_by_key:
            candidates_by_key[key] = _ParticipantCandidate(key=key)
        candidate = candidates_by_key[key]
        candidate.add_row(row)
        row_keys[index] = key
        if _row_has_direct_cue(row, character_profile=character_profile):
            candidate.roles.add("direct_cue")
        if _row_replies_to_character(row, character_profile=character_profile):
            candidate.roles.add("reply_to_character")

    if not candidates_by_key:
        candidates: list[_ParticipantCandidate] = []
        return candidates

    latest_key = _latest_row_key(user_rows, row_keys)
    if latest_key:
        candidates_by_key[latest_key].roles.add("latest_speaker")

    for candidate in candidates_by_key.values():
        if len(candidate.rows) > 1:
            candidate.roles.add("topic_author")
        if len(candidate.rows) >= 3:
            candidate.roles.add("dominant_speaker")

    candidates = list(candidates_by_key.values())
    return candidates


def _candidate_key(row: Mapping[str, Any]) -> str:
    """Return the first stable participant grouping key from a row."""

    for field_name in ("global_user_id", "platform_user_id", "display_name"):
        value = text_or_empty(row.get(field_name))
        if value:
            key = f"{field_name}:{value}"
            return key
    return_value = ""
    return return_value


def _row_has_direct_cue(
    row: Mapping[str, Any],
    *,
    character_profile: Mapping[str, Any],
) -> bool:
    """Return whether source metadata or text points at the character."""

    target_global_user_id = text_or_empty(character_profile.get("global_user_id"))
    platform_bot_id = text_or_empty(character_profile.get("platform_bot_id"))
    character_name = text_or_empty(character_profile.get("name"))

    addressed_ids = row.get("addressed_to_global_user_ids")
    if isinstance(addressed_ids, list):
        for addressed_id in addressed_ids:
            if text_or_empty(addressed_id) == target_global_user_id:
                return_value = True
                return return_value

    mentions = row.get("mentions")
    if _mentions_character(
        mentions,
        target_global_user_id=target_global_user_id,
        platform_bot_id=platform_bot_id,
        character_name=character_name,
    ):
        return_value = True
        return return_value

    if row.get("is_directed_at_character") is True:
        return_value = True
        return return_value

    body_text = text_or_empty(row.get("body_text"))
    if character_name and character_name.casefold() in body_text.casefold():
        return_value = True
        return return_value

    return_value = False
    return return_value


def _mentions_character(
    value: object,
    *,
    target_global_user_id: str,
    platform_bot_id: str,
    character_name: str,
) -> bool:
    """Return whether a mention row targets the active character."""

    if not isinstance(value, list):
        return_value = False
        return return_value

    for item in value:
        if not isinstance(item, Mapping):
            continue
        mention_global_user_id = text_or_empty(item.get("global_user_id"))
        if target_global_user_id and mention_global_user_id == target_global_user_id:
            return_value = True
            return return_value
        mention_platform_user_id = text_or_empty(item.get("platform_user_id"))
        if platform_bot_id and mention_platform_user_id == platform_bot_id:
            return_value = True
            return return_value
        mention_display_name = text_or_empty(item.get("display_name"))
        names_match = (
            character_name
            and mention_display_name.casefold() == character_name.casefold()
        )
        if names_match:
            return_value = True
            return return_value

    return_value = False
    return return_value


def _row_replies_to_character(
    row: Mapping[str, Any],
    *,
    character_profile: Mapping[str, Any],
) -> bool:
    """Return whether native reply metadata targets the character account."""

    reply_context = row.get("reply_context")
    if not isinstance(reply_context, Mapping):
        return_value = False
        return return_value

    if reply_context.get("reply_to_current_bot") is True:
        return_value = True
        return return_value

    platform_bot_id = text_or_empty(character_profile.get("platform_bot_id"))
    reply_platform_user_id = text_or_empty(
        reply_context.get("reply_to_platform_user_id"),
    )
    if platform_bot_id and reply_platform_user_id == platform_bot_id:
        return_value = True
        return return_value

    target_global_user_id = text_or_empty(character_profile.get("global_user_id"))
    reply_global_user_id = text_or_empty(
        reply_context.get("reply_to_global_user_id"),
    )
    if target_global_user_id and reply_global_user_id == target_global_user_id:
        return_value = True
        return return_value

    return_value = False
    return return_value


def _latest_row_key(
    user_rows: list[dict[str, Any]],
    row_keys: dict[int, str],
) -> str:
    """Return the candidate key for the latest grouped user row."""

    latest_key = ""
    latest_timestamp = datetime.min.replace(tzinfo=timezone.utc)
    latest_index = -1
    for index, row in enumerate(user_rows):
        key = row_keys.get(index)
        if not key:
            continue
        row_timestamp = _row_timestamp(row)
        is_later = row_timestamp > latest_timestamp
        is_same_later_index = (
            row_timestamp == latest_timestamp
            and index > latest_index
        )
        if is_later or is_same_later_index:
            latest_key = key
            latest_timestamp = row_timestamp
            latest_index = index
    return latest_key


def _row_timestamp(row: Mapping[str, Any]) -> datetime:
    """Parse a row timestamp for deterministic ordering."""

    timestamp_text = text_or_empty(row.get("timestamp"))
    if not timestamp_text:
        return_value = datetime.min.replace(tzinfo=timezone.utc)
        return return_value
    try:
        timestamp = parse_storage_utc_datetime(timestamp_text)
    except ValueError:
        return_value = datetime.min.replace(tzinfo=timezone.utc)
        return return_value
    return timestamp


def _select_primary(
    candidates: list[_ParticipantCandidate],
) -> _ParticipantCandidate:
    """Select one primary candidate by fit tier, recency, then score."""

    for fit in _FIT_ORDER:
        tier_candidates = [
            candidate
            for candidate in candidates
            if candidate.reply_target_fit == fit
        ]
        if tier_candidates:
            primary = max(
                tier_candidates,
                key=lambda candidate: (
                    candidate.latest_timestamp,
                    candidate.score,
                    candidate.key,
                ),
            )
            return primary

    primary = max(
        candidates,
        key=lambda candidate: (
            candidate.latest_timestamp,
            candidate.score,
            candidate.key,
        ),
    )
    return primary


def _high_fit_count(candidates: list[_ParticipantCandidate]) -> int:
    """Return how many candidates carry high reply-target fit."""

    count = sum(
        1
        for candidate in candidates
        if candidate.reply_target_fit == "high"
    )
    return count


def _focus_mode(primary: _ParticipantCandidate, high_fit_count: int) -> str:
    """Return the prompt-facing focus mode for the selected group beat."""

    if high_fit_count >= 2:
        return_value = "group_pile_on"
        return return_value
    if primary.reply_target_fit == "high":
        return_value = "direct_reply"
        return return_value
    if primary.reply_target_fit == "medium":
        return_value = "continue_visible_thread"
        return return_value
    return_value = "ambient_observation"
    return return_value


async def _hydrate_primary(
    primary: _ParticipantCandidate,
    *,
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    window_start_utc: str,
    current_timestamp_utc: str,
    conversation_agent: ConversationEvidenceAgent | None,
) -> dict[str, Any]:
    """Hydrate one primary participant when a global user id is available."""

    hydration = _empty_hydration()
    if not primary.global_user_id:
        return hydration

    profile = await get_user_profile(primary.global_user_id)
    if not isinstance(profile, dict):
        profile = {}

    engagement_context = await build_user_engagement_relevance_context(
        primary.global_user_id,
    )
    if not isinstance(engagement_context, dict):
        engagement_context = {}

    evidence = await _conversation_evidence(
        primary,
        target_scope=target_scope,
        character_profile=character_profile,
        window_start_utc=window_start_utc,
        current_timestamp_utc=current_timestamp_utc,
        conversation_agent=conversation_agent,
    )
    hydration = _hydration_from_sources(
        profile,
        engagement_context=engagement_context,
        conversation_evidence=evidence,
    )
    return hydration


def _empty_hydration() -> dict[str, Any]:
    """Return visible-only primary context defaults."""

    hydration = {
        "relationship_label": "unknown",
        "relationship_band": "unknown",
        "last_relationship_insight": "",
        "engagement_guidelines": [],
        "nearby_conversation_evidence": [],
    }
    return hydration


def _hydration_from_sources(
    profile: dict[str, Any],
    *,
    engagement_context: dict[str, Any],
    conversation_evidence: list[str],
) -> dict[str, Any]:
    """Project DB and helper results into prompt-facing semantic fields."""

    affinity = _profile_affinity(profile)
    if affinity is None:
        relationship_label = "unknown"
        relationship_band = "unknown"
    else:
        affinity_block = build_affinity_block(affinity)
        relationship_label = text_or_empty(affinity_block.get("level"))
        relationship_band = _relationship_band(affinity)

    raw_guidelines = engagement_context.get("engagement_guidelines")
    engagement_guidelines = _string_items(
        raw_guidelines,
        limit=ENGAGEMENT_GUIDELINES_LIMIT,
        char_limit=CONVERSATION_EVIDENCE_CHAR_LIMIT,
    )
    hydration = {
        "relationship_label": relationship_label,
        "relationship_band": relationship_band,
        "last_relationship_insight": text_or_empty(
            profile.get("last_relationship_insight"),
        ),
        "engagement_guidelines": engagement_guidelines,
        "nearby_conversation_evidence": conversation_evidence,
    }
    return hydration


def _profile_affinity(profile: Mapping[str, Any]) -> int | None:
    """Return affinity only when the profile carries a valid integer score."""

    affinity = profile.get("affinity")
    if isinstance(affinity, bool):
        return_value = None
        return return_value
    if not isinstance(affinity, int):
        return_value = None
        return return_value
    return affinity


def _relationship_band(affinity: int) -> str:
    """Map affinity into a coarse prompt-facing relationship band."""

    if affinity >= 600:
        return_value = "positive"
        return return_value
    if affinity <= 400:
        return_value = "negative"
        return return_value
    return_value = "neutral"
    return return_value


async def _conversation_evidence(
    primary: _ParticipantCandidate,
    *,
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    window_start_utc: str,
    current_timestamp_utc: str,
    conversation_agent: ConversationEvidenceAgent | None,
) -> list[str]:
    """Call the conversation-evidence helper once for the primary user."""

    agent = conversation_agent
    if agent is None:
        agent = ConversationEvidenceAgent()

    lookback_start_utc = _lookback_start_utc(window_start_utc)
    task = (
        "Conversation-filter: recent messages from speaker=current_user "
        f"from timestamp {lookback_start_utc} to timestamp {window_start_utc}; "
        f"limit {CONVERSATION_EVIDENCE_LIMIT}; same group channel."
    )
    context = _conversation_helper_context(
        primary,
        target_scope=target_scope,
        character_profile=character_profile,
        lookback_start_utc=lookback_start_utc,
        window_start_utc=window_start_utc,
        current_timestamp_utc=current_timestamp_utc,
    )
    raw_result = await agent.run(
        task,
        context,
        max_attempts=CONVERSATION_HELPER_ATTEMPTS,
    )
    evidence = _extract_evidence(raw_result)
    return evidence


def _lookback_start_utc(window_start_utc: str) -> str:
    """Return the closed lookback lower bound for helper retrieval."""

    window_start = parse_storage_utc_datetime(window_start_utc)
    lookback_start = window_start - timedelta(hours=CONVERSATION_LOOKBACK_HOURS)
    lookback_start_utc = normalize_storage_utc_iso(lookback_start.isoformat())
    return lookback_start_utc


def _conversation_helper_context(
    primary: _ParticipantCandidate,
    *,
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    lookback_start_utc: str,
    window_start_utc: str,
    current_timestamp_utc: str,
) -> dict[str, Any]:
    """Build bounded helper context without delivery or action metadata."""

    context = {
        "platform": text_or_empty(target_scope.get("platform")),
        "platform_channel_id": text_or_empty(
            target_scope.get("platform_channel_id"),
        ),
        "channel_type": text_or_empty(target_scope.get("channel_type")),
        "global_user_id": primary.global_user_id,
        "platform_user_id": primary.platform_user_id,
        "display_name": primary.display_name,
        "current_timestamp_utc": current_timestamp_utc,
        "from_timestamp_utc": lookback_start_utc,
        "to_timestamp_utc": window_start_utc,
        "limit": CONVERSATION_EVIDENCE_LIMIT,
        "character_profile": {
            "global_user_id": text_or_empty(
                character_profile.get("global_user_id"),
            ),
            "name": text_or_empty(character_profile.get("name")),
        },
        "active_turn_platform_message_ids": _active_turn_platform_message_ids(
            primary.rows,
        ),
        "active_turn_conversation_row_ids": [],
    }
    return context


def _active_turn_platform_message_ids(
    rows: list[dict[str, Any]],
) -> list[str]:
    """Return platform ids from the active group-review window."""

    message_ids = [
        message_id
        for row in rows
        if (message_id := text_or_empty(row.get("platform_message_id")))
    ]
    return message_ids


def _extract_evidence(raw_result: object) -> list[str]:
    """Project the helper result to capped evidence strings only."""

    if not isinstance(raw_result, Mapping):
        evidence: list[str] = []
        return evidence

    result_payload = raw_result.get("result")
    if not isinstance(result_payload, Mapping):
        evidence = []
        return evidence

    raw_evidence = result_payload.get("evidence")
    evidence = _string_items(
        raw_evidence,
        limit=CONVERSATION_EVIDENCE_LIMIT,
        char_limit=CONVERSATION_EVIDENCE_CHAR_LIMIT,
    )
    if evidence:
        return evidence

    selected_summary = text_or_empty(result_payload.get("selected_summary"))
    summary_lines = selected_summary.splitlines()
    evidence = _string_items(
        summary_lines,
        limit=CONVERSATION_EVIDENCE_LIMIT,
        char_limit=CONVERSATION_EVIDENCE_CHAR_LIMIT,
    )
    return evidence


def _primary_reply_target(
    primary: _ParticipantCandidate,
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Project the selected participant into prompt-facing context."""

    primary_target = {
        "display_name": primary.display_name or "visible speaker",
        "reply_target_fit": primary.reply_target_fit,
        "role_in_window": _ordered_roles(primary.roles),
        "relationship_label": hydration["relationship_label"],
        "relationship_band": hydration["relationship_band"],
        "last_relationship_insight": hydration["last_relationship_insight"],
        "engagement_guidelines": hydration["engagement_guidelines"],
        "nearby_conversation_evidence": (
            hydration["nearby_conversation_evidence"]
        ),
        "visible_samples": _visible_samples(primary.rows),
    }
    return primary_target


def _ordered_roles(roles: set[str]) -> list[str]:
    """Return candidate roles in stable prompt-facing order."""

    ordered = sorted(
        roles,
        key=lambda role: _ROLE_ORDER.get(role, len(_ROLE_ORDER)),
    )
    return ordered


def _visible_samples(rows: list[dict[str, Any]]) -> list[str]:
    """Return the latest visible text samples for the primary participant."""

    samples = [
        sample
        for row in rows[-VISIBLE_SAMPLE_LIMIT:]
        if (
            sample := _clip_text(
                text_or_empty(row.get("body_text")),
                VISIBLE_SAMPLE_CHAR_LIMIT,
            )
        )
    ]
    return samples


def _background_flow(
    *,
    primary: _ParticipantCandidate | None,
    candidate_count: int,
    high_fit_count: int,
) -> dict[str, str]:
    """Build aggregate background-flow context without a participant roster."""

    participant_count_label = _participant_count_label(candidate_count)
    if high_fit_count >= 2:
        mode = "multi_person_pile_on"
        summary = (
            "Multiple visible speakers are aiming the same beat at the "
            "character; treat it as one shared group moment."
        )
    elif primary is None:
        mode = "ambient_group"
        summary = (
            "User-authored rows are visible, but no stable primary speaker "
            "could be selected."
        )
    elif candidate_count <= 1:
        mode = "none"
        summary = "The visible window centers on one speaker."
    elif primary.reply_target_fit == "low":
        mode = "ambient_group"
        summary = (
            "Several speakers are visible, but no strong reply target stands "
            "out from the flow."
        )
    else:
        mode = "side_thread"
        summary = (
            "Other visible speakers form background flow; keep attention on "
            "the selected current beat."
        )
    background_flow = {
        "mode": mode,
        "summary": summary,
        "participant_count_label": participant_count_label,
    }
    return background_flow


def _participant_count_label(candidate_count: int) -> str:
    """Map participant count to a prompt-facing coarse label."""

    if candidate_count <= 1:
        return_value = "single"
        return return_value
    if candidate_count <= 4:
        return_value = "few"
        return return_value
    return_value = "many"
    return return_value


def _context_payload(
    *,
    focus_mode: str,
    primary_reply_target: dict[str, Any],
    background_flow: dict[str, str],
) -> dict[str, Any]:
    """Build the final prompt-facing participant context payload."""

    context = {
        "source": SOURCE_NAME,
        "context_shape": CONTEXT_SHAPE,
        "focus_mode": focus_mode,
        "guidance": _GUIDANCE[focus_mode],
        "primary_reply_target": primary_reply_target,
        "background_flow": background_flow,
    }
    return context


def _string_items(
    value: object,
    *,
    limit: int,
    char_limit: int,
) -> list[str]:
    """Return capped non-empty strings from a list-like value."""

    if not isinstance(value, list):
        items: list[str] = []
        return items

    items = []
    for item in value:
        text = text_or_empty(item)
        if not text:
            continue
        clipped = _clip_text(text, char_limit)
        items.append(clipped)
        if len(items) >= limit:
            break
    return items


def _clip_text(text: str, limit: int) -> str:
    """Clip one prompt-facing text field to a fixed character budget."""

    if len(text) <= limit:
        return_value = text
        return return_value
    clipped = f"{text[:limit - 3].rstrip()}..."
    return clipped
