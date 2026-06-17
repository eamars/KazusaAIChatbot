"""Read-only repository adapters for console lookup pages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import PyMongoError

from control_console.redaction import redact_mapping
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.db.errors import DatabaseOperationError

AsyncHelper = Callable[..., Awaitable[Any]]
STYLE_GUIDELINE_FIELDS = (
    "speech_guidelines",
    "social_guidelines",
    "pacing_guidelines",
    "engagement_guidelines",
)
REPOSITORY_HELPER_ERRORS = (
    DatabaseOperationError,
    ImportError,
    KeyError,
    PyMongoError,
    ValueError,
)
APPLICATION_IDENTITY_TIMEOUT_SECONDS = 1.0
APPLICATION_IDENTITY_ERRORS = (*REPOSITORY_HELPER_ERRORS, TimeoutError)


class ControlConsoleRepository:
    """Read-only domain lookup facade with safe unavailable fallbacks."""

    def __init__(
        self,
        *,
        get_character_profile: AsyncHelper | None = None,
        get_character_runtime_state: AsyncHelper | None = None,
        list_growth_traits: AsyncHelper | None = None,
        query_user_memory_units: AsyncHelper | None = None,
        search_user_memory_units_by_keyword: AsyncHelper | None = None,
        build_interaction_style_context: AsyncHelper | None = None,
        list_due_calendar_runs: AsyncHelper | None = None,
    ) -> None:
        """Create a read-only repository facade."""

        self._get_character_profile = get_character_profile
        self._get_character_runtime_state = get_character_runtime_state
        self._list_growth_traits = list_growth_traits
        self._query_user_memory_units = query_user_memory_units
        self._search_user_memory_units_by_keyword = search_user_memory_units_by_keyword
        self._build_interaction_style_context = build_interaction_style_context
        self._list_due_calendar_runs = list_due_calendar_runs

    async def application_identity(self) -> dict[str, Any]:
        """Return the active character name for the browser shell."""

        try:
            helper = self._get_character_profile
            if helper is None:
                from kazusa_ai_chatbot.db.character import get_character_profile

                helper = get_character_profile
            profile = await asyncio.wait_for(
                helper(),
                timeout=APPLICATION_IDENTITY_TIMEOUT_SECONDS,
            )
        except APPLICATION_IDENTITY_ERRORS as exc:
            identity = _not_connected_identity(
                status="unavailable",
                reason=str(exc),
            )
            return identity

        if not isinstance(profile, dict):
            identity = _not_connected_identity(
                status="unavailable",
                reason="character profile helper returned invalid data",
            )
            return identity

        character_name = str(profile.get("name", "")).strip()
        if not character_name:
            identity = _not_connected_identity(
                status="empty",
                reason="character profile is missing name",
            )
            return identity

        identity = {
            "status": "available",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "character_name": character_name[:120],
            "source": "character_state",
        }
        return identity

    async def latest_character_status(self) -> dict[str, Any]:
        """Return a bounded character-status summary."""

        try:
            helper = self._get_character_runtime_state
            if helper is None:
                from kazusa_ai_chatbot.db.character import (
                    get_character_runtime_state,
                )

                helper = get_character_runtime_state
            runtime_state = await helper()
        except REPOSITORY_HELPER_ERRORS as exc:
            summary = _unavailable_summary(
                area="character_status",
                reason=str(exc),
            )
            return summary

        if not runtime_state:
            return _empty_summary(area="character_status")

        status = {
            "status": "available",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": redact_mapping({
                key: runtime_state.get(key)
                for key in (
                    "mood",
                    "global_vibe",
                    "reflection_summary",
                    "updated_at",
                )
                if key in runtime_state
            }),
        }
        return status

    async def global_growth_summary(self) -> dict[str, Any]:
        """Return a bounded global-growth summary."""

        try:
            helper = self._list_growth_traits
            if helper is None:
                from kazusa_ai_chatbot.db.global_character_growth import (
                    list_prompt_visible_growth_traits,
                )

                helper = list_prompt_visible_growth_traits
            traits = await helper(limit=12)
        except REPOSITORY_HELPER_ERRORS as exc:
            summary = _unavailable_summary(
                area="global_growth",
                reason=str(exc),
                items=[],
            )
            return summary

        summary = {
            "status": "available" if traits else "empty",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": [
                redact_mapping({
                    "trait_id": trait.get("trait_id", ""),
                    "growth_axis": trait.get("growth_axis", ""),
                    "status": trait.get("status", ""),
                    "maturity_band": trait.get("maturity_band", ""),
                    "updated_at": trait.get("updated_at", ""),
                })
                for trait in list(traits)[:12]
                if isinstance(trait, dict)
            ],
        }
        return summary

    async def lookup_memory(
        self,
        *,
        global_user_id: str,
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return a bounded redacted memory lookup page."""

        clean_global_user_id = global_user_id.strip()
        clean_query = query.strip()
        if not clean_global_user_id:
            page = _lookup_page(
                status="needs_input",
                items=[],
                reason="global_user_id is required for scoped memory lookup",
            )
            return page

        query_helper = self._query_user_memory_units
        keyword_helper = self._search_user_memory_units_by_keyword
        page = {
            "status": "unavailable",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "next_cursor": None,
            "reason": "memory repository helper is unavailable",
            "redaction": _lookup_redaction(),
        }
        try:
            if query_helper is None or keyword_helper is None:
                from kazusa_ai_chatbot.db.user_memory_units import (
                    query_user_memory_units,
                    search_user_memory_units_by_keyword,
                )

                query_helper = query_user_memory_units
                keyword_helper = search_user_memory_units_by_keyword

            if clean_query:
                documents = await keyword_helper(
                    clean_global_user_id,
                    clean_query,
                    limit=limit,
                )
            else:
                documents = await query_helper(
                    clean_global_user_id,
                    limit=limit,
                )
        except REPOSITORY_HELPER_ERRORS as exc:
            page["reason"] = str(exc)[:160]
            return page

        items = [
            _project_memory_unit(document)
            for document in list(documents)[:limit]
            if isinstance(document, dict)
        ]
        page = _lookup_page(
            status="available" if items else "empty",
            items=items,
            reason="no memory units matched the lookup" if not items else "",
        )
        return page

    async def empty_lookup(self, *, namespace: str) -> dict[str, Any]:
        """Return a bounded empty lookup for a not-yet-wired helper."""

        page = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
            "next_cursor": None,
            "redaction": {
                "namespace": namespace,
                "embeddings": "excluded",
                "model_inputs": "excluded",
                "raw_messages": "excluded",
            },
        }
        return page

    async def lookup_due_calendar_runs(
        self,
        *,
        current_timestamp_utc: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return due calendar-run state without scheduler payload internals."""

        helper = self._list_due_calendar_runs
        page = _calendar_lookup_page(
            status="unavailable",
            items=[],
            reason="calendar run helper is unavailable",
        )
        try:
            if helper is None:
                from kazusa_ai_chatbot.calendar_scheduler.repository import (
                    list_due_calendar_runs,
                )

                helper = list_due_calendar_runs

            documents = await helper(
                current_timestamp_utc=current_timestamp_utc,
                trigger_kinds=sorted(calendar_models.CALENDAR_TRIGGER_KINDS),
                max_attempts=calendar_models.DEFAULT_RUN_MAX_ATTEMPTS,
                limit=limit,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            page["reason"] = str(exc)[:160]
            return page

        items = [
            _project_calendar_run(document)
            for document in list(documents)[:limit]
            if isinstance(document, dict)
        ]
        page = _calendar_lookup_page(
            status="available" if items else "empty",
            items=items,
            reason="no due calendar runs matched the lookup" if not items else "",
        )
        return page

    async def lookup_interaction_style(
        self,
        *,
        global_user_id: str,
        platform: str,
        platform_channel_id: str,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Return scoped interaction-style guidance for operator inspection."""

        clean_global_user_id = global_user_id.strip()
        clean_platform = platform.strip()
        clean_platform_channel_id = platform_channel_id.strip()
        if not clean_global_user_id and not clean_platform_channel_id:
            page = _style_lookup_page(
                status="needs_input",
                items=[],
                reason=(
                    "global_user_id is required for private style lookup; "
                    "platform and group id are required for group style lookup"
                ),
            )
            return page
        if clean_platform_channel_id and not clean_platform:
            page = _style_lookup_page(
                status="needs_input",
                items=[],
                reason="platform is required when group id is provided",
            )
            return page

        channel_type = "group" if clean_platform_channel_id else "private"
        helper = self._build_interaction_style_context
        page = _style_lookup_page(
            status="unavailable",
            items=[],
            reason="interaction-style helper is unavailable",
        )
        try:
            if helper is None:
                from kazusa_ai_chatbot.db.interaction_style_images import (
                    build_interaction_style_context,
                )

                helper = build_interaction_style_context

            context = await helper(
                global_user_id=clean_global_user_id,
                channel_type=channel_type,
                platform=clean_platform,
                platform_channel_id=clean_platform_channel_id,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            page["reason"] = str(exc)[:160]
            return page

        items = _project_interaction_style_context(context, limit=limit)
        page = _style_lookup_page(
            status="available" if items else "empty",
            items=items,
            reason="no interaction-style guidance matched the lookup" if not items else "",
        )
        return page


def _unavailable_summary(
    *,
    area: str,
    reason: str,
    items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a bounded unavailable domain summary."""

    summary = {
        "status": "unavailable",
        "area": area,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "reason": str(reason)[:160],
    }
    return summary


def _empty_summary(*, area: str) -> dict[str, Any]:
    """Build a bounded empty domain summary."""

    summary = {
        "status": "empty",
        "area": area,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }
    return summary


def _not_connected_identity(*, status: str, reason: str) -> dict[str, Any]:
    """Build the safe browser fallback for missing character identity."""

    identity = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "character_name": "not connected",
        "source": "character_state",
        "reason": str(reason)[:160],
    }
    return identity


def _project_memory_unit(document: dict[str, Any]) -> dict[str, Any]:
    """Project one memory-unit document into a browser-safe row."""

    allowed_fields = (
        "unit_id",
        "unit_type",
        "status",
        "fact",
        "relationship_signal",
        "subjective_appraisal",
        "due_at",
        "last_seen_at",
        "updated_at",
    )
    row = {
        field: document[field]
        for field in allowed_fields
        if field in document and document[field] not in (None, "")
    }
    projected_row = redact_mapping(row)
    return projected_row


def _lookup_page(
    *,
    status: str,
    items: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Build a bounded lookup page with shared redaction metadata."""

    page = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "next_cursor": None,
        "reason": reason,
        "redaction": _lookup_redaction(),
    }
    return page


def _lookup_redaction() -> dict[str, str]:
    """Return the static lookup redaction contract."""

    redaction = {
        "embeddings": "excluded",
        "model_inputs": "excluded",
        "raw_messages": "excluded",
    }
    return redaction


def _project_interaction_style_context(
    context: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Project prompt-facing style context into redacted table rows."""

    application_order = context.get("application_order", [])
    if not isinstance(application_order, list):
        application_order = []

    rows: list[dict[str, Any]] = []
    for scope in application_order:
        overlay = context.get(scope)
        if not isinstance(scope, str) or not isinstance(overlay, dict):
            continue
        confidence = str(overlay.get("confidence", ""))
        for field in STYLE_GUIDELINE_FIELDS:
            guidelines = overlay.get(field, [])
            if not isinstance(guidelines, list) or not guidelines:
                continue
            row = redact_mapping({
                "scope": scope,
                "field": field,
                "guidelines": [str(item) for item in guidelines[:limit]],
                "confidence": confidence,
            })
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def _style_lookup_page(
    *,
    status: str,
    items: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Build a bounded interaction-style lookup page."""

    page = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "next_cursor": None,
        "reason": reason,
        "redaction": {
            "source_run_ids": "excluded",
            "model_inputs": "excluded",
            "raw_reflections": "excluded",
        },
    }
    return page


def _project_calendar_run(document: dict[str, Any]) -> dict[str, Any]:
    """Project one calendar-run document into a browser-safe row."""

    allowed_fields = (
        "run_id",
        "schedule_id",
        "trigger_kind",
        "status",
        "due_at",
        "attempt_count",
        "max_attempts",
        "lease_owner",
        "lease_expires_at",
        "completed_at",
        "failed_at",
        "skipped_at",
        "result_summary",
        "failure_summary",
        "period_start_utc",
        "slot_index",
        "offset_seconds",
        "updated_at",
    )
    row = {
        field: document[field]
        for field in allowed_fields
        if field in document and document[field] not in (None, "")
    }
    projected_row = redact_mapping(row)
    return projected_row


def _calendar_lookup_page(
    *,
    status: str,
    items: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Build a bounded due-run lookup page."""

    page = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
        "next_cursor": None,
        "reason": reason,
        "redaction": {
            "payload": "excluded",
            "source_scope": "excluded",
            "idempotency_keys": "excluded",
            "raw_messages": "excluded",
        },
    }
    return page
