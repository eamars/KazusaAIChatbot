"""Production reflection-run repository helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.config import CHARACTER_TIME_ZONE
from kazusa_ai_chatbot.db import reflection_cycle as reflection_store
from kazusa_ai_chatbot.db.schemas import (
    CharacterReflectionRunDoc,
    ReflectionMessageRefDoc,
    ReflectionScopeDoc,
)
from kazusa_ai_chatbot.reflection_cycle.models import (
    DailySynthesisResult,
    READONLY_REFLECTION_PROMPT_VERSION,
    REFLECTION_RUN_KIND_DAILY_CHANNEL,
    REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
    REFLECTION_RUN_KIND_HOURLY,
    REFLECTION_STATUS_DRY_RUN,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    REFLECTION_STATUS_SUCCEEDED,
    REFLECTION_TERMINAL_STATUSES,
    ReflectionLLMResult,
    ReflectionScopeInput,
)


_HOURLY_SCOPE_SUFFIX_RE = re.compile(r"_[0-9]{8}T[0-9]{2}Z$")


def now_iso() -> str:
    """Return the current UTC timestamp for reflection-run documents."""

    current_time = datetime.now(timezone.utc).isoformat()
    return current_time


def deterministic_run_id(parts: list[str]) -> str:
    """Build a stable reflection-run id from ordered logical parts."""

    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    run_id = f"reflection_run_{digest[:32]}"
    return run_id


def hourly_run_id(*, scope_ref: str, hour_start: str) -> str:
    """Return the deterministic id for one hourly reflection slot."""

    run_id = deterministic_run_id([
        REFLECTION_RUN_KIND_HOURLY,
        scope_ref,
        hour_start,
        READONLY_REFLECTION_PROMPT_VERSION,
    ])
    return run_id


def channel_scope_ref_for_hourly(scope_ref: str) -> str:
    """Return the parent channel scope ref for an hourly scope ref."""

    suffix_match = _HOURLY_SCOPE_SUFFIX_RE.search(scope_ref)
    if suffix_match is None:
        return_value = scope_ref
        return return_value
    return_value = scope_ref[:suffix_match.start()]
    return return_value


def daily_channel_run_id(*, scope_ref: str, character_local_date: str) -> str:
    """Return the deterministic id for one per-channel daily reflection."""

    run_id = deterministic_run_id([
        REFLECTION_RUN_KIND_DAILY_CHANNEL,
        scope_ref,
        character_local_date,
        READONLY_REFLECTION_PROMPT_VERSION,
    ])
    return run_id


def daily_global_promotion_run_id(
    *,
    character_local_date: str,
    prompt_version: str,
) -> str:
    """Return the deterministic id for one daily global promotion."""

    run_id = deterministic_run_id([
        REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
        character_local_date,
        prompt_version,
    ])
    return run_id


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    normalized = parsed.astimezone(timezone.utc)
    return normalized


def hour_start_for_scope(scope: ReflectionScopeInput) -> datetime:
    """Return the UTC hour start for an hourly scope."""

    timestamp = parse_timestamp(scope.first_timestamp)
    hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
    return hour_start


def character_local_date_for_utc(
    value: datetime,
    *,
    time_zone: str = CHARACTER_TIME_ZONE,
) -> str:
    """Return the character-local calendar date for a UTC timestamp."""

    local_time = value.astimezone(ZoneInfo(time_zone))
    local_date = local_time.date().isoformat()
    return local_date


def scope_doc(scope: ReflectionScopeInput) -> ReflectionScopeDoc:
    """Project a runtime scope into persistence metadata."""

    doc: ReflectionScopeDoc = {
        "scope_ref": scope.scope_ref,
        "platform": scope.platform,
        "platform_channel_id": scope.platform_channel_id,
        "channel_type": scope.channel_type,
    }
    return doc


def source_message_refs(scope: ReflectionScopeInput) -> list[ReflectionMessageRefDoc]:
    """Build persistence-only source-message references from scope rows."""

    refs: list[ReflectionMessageRefDoc] = []
    for message in scope.messages:
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        message_ref: ReflectionMessageRefDoc = {
            "platform": str(message.get("platform", scope.platform) or scope.platform),
            "platform_channel_id": str(
                message.get(
                    "platform_channel_id",
                    scope.platform_channel_id,
                )
                or scope.platform_channel_id
            ),
            "channel_type": str(
                message.get("channel_type", scope.channel_type) or scope.channel_type
            ),
            "role": role,
            "timestamp": str(message.get("timestamp", "")),
        }
        conversation_history_id = str(
            message.get("conversation_history_id", "") or ""
        ).strip()
        if conversation_history_id:
            message_ref["conversation_history_id"] = conversation_history_id
        refs.append(message_ref)
    return refs


def active_character_utterances(
    scope: ReflectionScopeInput,
    *,
    max_items: int = 3,
    max_chars: int = 180,
) -> list[str]:
    """Return short assistant-authored snippets for promotion evidence cards."""

    utterances: list[str] = []
    for message in scope.messages:
        if message.get("role") != "assistant":
            continue
        text = " ".join(str(message.get("body_text", "") or "").split())
        if not text:
            continue
        if len(text) > max_chars:
            text = f"{text[:max_chars - 3]}..."
        utterances.append(text)
        if len(utterances) >= max_items:
            break
    return utterances


def build_hourly_run_document(
    *,
    scope: ReflectionScopeInput,
    result: ReflectionLLMResult | None,
    status: str,
    attempt_count: int,
    error: str = "",
) -> CharacterReflectionRunDoc:
    """Build one hourly reflection-run document."""

    hour_start_dt = hour_start_for_scope(scope)
    hour_end_dt = hour_start_dt + timedelta(hours=1)
    timestamp = now_iso()
    parsed_output: dict[str, Any] = {}
    validation_warnings: list[str] = []
    if result is not None:
        parsed_output = dict(result.parsed_output)
        validation_warnings = list(result.validation_warnings)
    parsed_output["hourly_scope_ref"] = scope.scope_ref
    parsed_output["active_character_utterances"] = active_character_utterances(
        scope,
    )
    channel_scope_ref = channel_scope_ref_for_hourly(scope.scope_ref)
    run_id = hourly_run_id(
        scope_ref=channel_scope_ref,
        hour_start=hour_start_dt.isoformat(),
    )
    channel_scope = ReflectionScopeInput(
        scope_ref=channel_scope_ref,
        platform=scope.platform,
        platform_channel_id=scope.platform_channel_id,
        channel_type=scope.channel_type,
        assistant_message_count=scope.assistant_message_count,
        user_message_count=scope.user_message_count,
        total_message_count=scope.total_message_count,
        first_timestamp=scope.first_timestamp,
        last_timestamp=scope.last_timestamp,
        messages=[],
    )
    document: CharacterReflectionRunDoc = {
        "_id": run_id,
        "run_id": run_id,
        "run_kind": REFLECTION_RUN_KIND_HOURLY,
        "status": status,
        "prompt_version": READONLY_REFLECTION_PROMPT_VERSION,
        "attempt_count": attempt_count,
        "scope": scope_doc(channel_scope),
        "window_start": scope.first_timestamp,
        "window_end": scope.last_timestamp,
        "hour_start": hour_start_dt.isoformat(),
        "hour_end": hour_end_dt.isoformat(),
        "character_local_date": character_local_date_for_utc(hour_start_dt),
        "source_message_refs": source_message_refs(scope),
        "source_reflection_run_ids": [],
        "output": parsed_output,
        "promotion_decisions": [],
        "validation_warnings": validation_warnings,
        "error": error,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return document


def build_daily_channel_run_document(
    *,
    channel_scope: ReflectionScopeInput,
    hourly_docs: list[CharacterReflectionRunDoc],
    result: DailySynthesisResult | None,
    character_local_date: str,
    status: str,
    attempt_count: int,
    error: str = "",
) -> CharacterReflectionRunDoc:
    """Build one per-channel daily reflection-run document."""

    timestamp = now_iso()
    output: dict[str, Any] = {}
    validation_warnings: list[str] = []
    if result is not None:
        output = dict(result.parsed_output)
        validation_warnings = list(result.validation_warnings)
    source_run_ids = [str(document["run_id"]) for document in hourly_docs]
    run_id = daily_channel_run_id(
        scope_ref=channel_scope.scope_ref,
        character_local_date=character_local_date,
    )
    document: CharacterReflectionRunDoc = {
        "_id": run_id,
        "run_id": run_id,
        "run_kind": REFLECTION_RUN_KIND_DAILY_CHANNEL,
        "status": status,
        "prompt_version": READONLY_REFLECTION_PROMPT_VERSION,
        "attempt_count": attempt_count,
        "scope": scope_doc(channel_scope),
        "window_start": _lead_value(hourly_docs, "hour_start"),
        "window_end": _last_value(hourly_docs, "hour_end"),
        "character_local_date": character_local_date,
        "source_message_refs": [],
        "source_reflection_run_ids": source_run_ids,
        "output": output,
        "promotion_decisions": [],
        "validation_warnings": validation_warnings,
        "error": error,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return document


def build_global_promotion_run_document(
    *,
    character_local_date: str,
    prompt_version: str,
    source_run_ids: list[str],
    output: dict[str, Any],
    promotion_decisions: list[dict[str, Any]],
    status: str,
    attempt_count: int,
    validation_warnings: list[str],
    error: str = "",
) -> CharacterReflectionRunDoc:
    """Build one global promotion reflection-run document."""

    timestamp = now_iso()
    run_id = daily_global_promotion_run_id(
        character_local_date=character_local_date,
        prompt_version=prompt_version,
    )
    document: CharacterReflectionRunDoc = {
        "_id": run_id,
        "run_id": run_id,
        "run_kind": REFLECTION_RUN_KIND_DAILY_GLOBAL_PROMOTION,
        "status": status,
        "prompt_version": prompt_version,
        "attempt_count": attempt_count,
        "scope": {
            "scope_ref": "daily_global",
            "platform": "system",
            "platform_channel_id": "global",
            "channel_type": "system",
        },
        "character_local_date": character_local_date,
        "source_message_refs": [],
        "source_reflection_run_ids": source_run_ids,
        "output": output,
        "promotion_decisions": promotion_decisions,
        "validation_warnings": validation_warnings,
        "error": error,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    return document


async def upsert_run(document: CharacterReflectionRunDoc) -> None:
    """Persist one reflection-run document through the DB interface."""

    await reflection_store.upsert_reflection_run(document)


async def existing_run_ids(run_ids: list[str]) -> set[str]:
    """Return persisted run ids from the reflection-run DB interface."""

    existing = await reflection_store.list_existing_run_ids(run_ids)
    return existing


async def hourly_runs_for_channel_day(
    *,
    scope_ref: str,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]:
    """Load hourly runs for one scope and character-local date."""

    docs = await reflection_store.list_hourly_runs_for_channel_day(
        scope_ref=scope_ref,
        character_local_date=character_local_date,
    )
    return docs


async def daily_channel_runs(
    *,
    character_local_date: str,
) -> list[CharacterReflectionRunDoc]:
    """Load daily channel runs for one character-local date."""

    docs = await reflection_store.list_daily_channel_runs(
        character_local_date=character_local_date,
    )
    return docs


async def reflection_run_by_id(
    run_id: str,
) -> CharacterReflectionRunDoc | None:
    """Load one reflection run by id through the DB interface."""

    doc = await reflection_store.find_reflection_run_by_id(run_id)
    return doc


def is_terminal_run(document: CharacterReflectionRunDoc) -> bool:
    """Return whether a reflection run is terminal for scheduling."""

    status = str(document["status"])
    return_value = status in REFLECTION_TERMINAL_STATUSES
    return return_value


def status_for_result(*, dry_run: bool, failed: bool = False) -> str:
    """Return the persisted status for a worker unit of work."""

    if dry_run:
        return_value = REFLECTION_STATUS_DRY_RUN
    elif failed:
        return_value = REFLECTION_STATUS_FAILED
    else:
        return_value = REFLECTION_STATUS_SUCCEEDED
    return return_value


def skipped_status() -> str:
    """Return the skipped status value for reflection run documents."""

    return REFLECTION_STATUS_SKIPPED


def _lead_value(documents: list[CharacterReflectionRunDoc], field_name: str) -> str:
    """Return the first non-empty field value from ordered run documents."""

    for document in documents:
        value = str(document.get(field_name, "") or "")
        if value:
            return value
    return_value = ""
    return return_value


def _last_value(documents: list[CharacterReflectionRunDoc], field_name: str) -> str:
    """Return the last non-empty field value from ordered run documents."""

    for document in reversed(documents):
        value = str(document.get(field_name, "") or "")
        if value:
            return value
    return_value = ""
    return return_value
