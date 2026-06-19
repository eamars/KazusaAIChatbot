"""Read-only repository adapters for console lookup pages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import PyMongoError

from control_console.redaction import redact_mapping
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.db.character import (
    get_character_profile as default_get_character_profile,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.users import (
    find_user_profile_by_identifier as default_find_user_profile_by_identifier,
)

AsyncHelper = Callable[..., Awaitable[Any]]
STYLE_GUIDELINE_FIELDS = (
    "speech_guidelines",
    "social_guidelines",
    "pacing_guidelines",
    "engagement_guidelines",
)
CHARACTER_PROFILE_FIELDS = (
    "name",
    "description",
    "gender",
    "age",
    "birthday",
    "personality_brief",
    "updated_at",
)
SELF_IMAGE_FIELDS = (
    "summary",
    "current_self_concept",
    "historical_summary",
    "recent_window",
    "milestones",
    "updated_at",
)
USER_PROFILE_FIELDS = (
    "updated_at",
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
        find_user_profile_by_identifier: AsyncHelper | None = None,
    ) -> None:
        """Create a read-only repository facade."""

        self._get_character_profile = get_character_profile
        self._get_character_runtime_state = get_character_runtime_state
        self._list_growth_traits = list_growth_traits
        self._query_user_memory_units = query_user_memory_units
        self._search_user_memory_units_by_keyword = search_user_memory_units_by_keyword
        self._build_interaction_style_context = build_interaction_style_context
        self._list_due_calendar_runs = list_due_calendar_runs
        self._find_user_profile_by_identifier = find_user_profile_by_identifier

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

    async def character_entity(self, *, limit: int = 25) -> dict[str, Any]:
        """Return the owner-oriented character inspection envelope."""

        profile_panel = _entity_panel(
            status="unavailable",
            items=[],
            reason="character profile helper is unavailable",
        )
        self_image_panel = _entity_panel(
            status="unavailable",
            items=[],
            reason="character profile helper is unavailable",
        )
        identity: dict[str, Any] = {}
        try:
            helper = self._get_character_profile or default_get_character_profile
            profile = await helper()
        except APPLICATION_IDENTITY_ERRORS as exc:
            reason = str(exc)
            profile_panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=reason,
            )
            self_image_panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=reason,
            )
        else:
            if isinstance(profile, dict):
                profile_items = _project_character_profile(profile)
                self_image_items = _project_self_image(profile.get("self_image"))
                identity = {
                    "character_name": str(profile.get("name", "")).strip()[:120],
                }
                profile_panel = _entity_panel(
                    status="available" if profile_items else "empty",
                    items=profile_items[:limit],
                    reason=(
                        ""
                        if profile_items
                        else "character profile has no browser-safe fields"
                    ),
                )
                self_image_panel = _entity_panel(
                    status="available" if self_image_items else "empty",
                    items=self_image_items[:limit],
                    reason=(
                        ""
                        if self_image_items
                        else "character self-image is not available"
                    ),
                )
            else:
                profile_panel = _entity_panel(
                    status="unavailable",
                    items=[],
                    reason="character profile helper returned invalid data",
                )
                self_image_panel = _entity_panel(
                    status="unavailable",
                    items=[],
                    reason="character profile helper returned invalid data",
                )

        state_summary = await self.latest_character_status()
        state_summary_payload = state_summary.get("summary", {})
        if isinstance(state_summary_payload, dict):
            state_row = {
                key: state_summary_payload[key]
                for key in ("mood", "global_vibe", "updated_at")
                if key in state_summary_payload
            }
        else:
            state_row = {}
        state_items = _project_key_value_items(state_row)
        state_panel = _entity_panel(
            status="available" if state_items else state_summary.get(
                "status",
                "empty",
            ),
            items=state_items[:limit],
            reason=str(state_summary.get("reason", "")),
        )

        growth_summary = await self.global_growth_summary()
        growth_items = [
            item
            for item in growth_summary.get("items", [])
            if isinstance(item, dict)
        ]
        growth_panel = _entity_panel(
            status="available" if growth_items else growth_summary.get(
                "status",
                "empty",
            ),
            items=growth_items[:limit],
            reason=str(growth_summary.get("reason", "")),
        )
        learning_items = _project_learning_items(state_summary)

        panels = {
            "profile": profile_panel,
            "self_image": self_image_panel,
            "state": state_panel,
            "growth": growth_panel,
            "memory": _entity_panel(
                status="empty",
                items=[],
                reason=(
                    "shared and character memory search is not exposed by this "
                    "read-only console surface"
                ),
            ),
            "learning": _entity_panel(
                status="available" if learning_items else "empty",
                items=learning_items[:limit],
                reason=(
                    ""
                    if learning_items
                    else "no promoted background-learning summary is available"
                ),
            ),
        }
        envelope = _owner_entity_envelope(
            owner="character",
            identity=identity,
            panels=panels,
        )
        return envelope

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
                    list_active_growth_traits,
                )

                helper = list_active_growth_traits
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
                    "trait_name": trait.get("trait_name", ""),
                    "guidance": trait.get("guidance", ""),
                    "strength": trait.get("strength", ""),
                    "status": trait.get("status", ""),
                    "maturity_band": trait.get("maturity_band", ""),
                    "evidence_count": trait.get("evidence_count", ""),
                    "first_observed_date": trait.get("first_observed_date", ""),
                    "last_observed_date": trait.get("last_observed_date", ""),
                    "updated_at": trait.get("updated_at", ""),
                })
                for trait in list(traits)[:12]
                if isinstance(trait, dict)
            ],
        }
        return summary

    async def _resolve_platform_user_identity(
        self,
        *,
        platform: str,
        platform_user_id: str,
    ) -> dict[str, Any]:
        """Resolve an operator-facing platform account.

        Returns:
            A resolution envelope containing safe browser identity metadata and
            the canonical user id needed by repository helpers.
        """

        clean_platform = platform.strip()
        clean_platform_user_id = platform_user_id.strip()
        if not clean_platform and not clean_platform_user_id:
            resolution = _platform_user_resolution(
                status="needs_input",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason="platform and platform user id are required",
            )
            return resolution
        if not clean_platform:
            resolution = _platform_user_resolution(
                status="needs_input",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason="platform is required when platform user id is provided",
            )
            return resolution
        if not clean_platform_user_id:
            resolution = _platform_user_resolution(
                status="needs_input",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason="platform user id is required when platform is provided",
            )
            return resolution

        helper = (
            self._find_user_profile_by_identifier
            or default_find_user_profile_by_identifier
        )
        try:
            profile = await helper(
                identifier=clean_platform_user_id,
                platform=clean_platform,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            resolution = _platform_user_resolution(
                status="unavailable",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason=str(exc)[:160],
            )
            return resolution

        if not isinstance(profile, dict):
            resolution = _platform_user_resolution(
                status="empty",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason="no user profile matched the platform account",
            )
            return resolution

        global_user_id = str(profile.get("global_user_id", "")).strip()
        if not global_user_id:
            resolution = _platform_user_resolution(
                status="unavailable",
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
                reason="matched user profile is missing canonical identity",
            )
            return resolution

        display_name = _display_name_for_platform_account(
            profile,
            platform=clean_platform,
            platform_user_id=clean_platform_user_id,
        )
        resolution = _platform_user_resolution(
            status="resolved",
            platform=clean_platform,
            platform_user_id=clean_platform_user_id,
            reason="",
            display_name=display_name,
            global_user_id=global_user_id,
            profile=profile,
        )
        return resolution

    async def lookup_memory(
        self,
        *,
        platform: str,
        platform_user_id: str,
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return a bounded redacted memory lookup page."""

        clean_query = query.strip()
        resolution = await self._resolve_platform_user_identity(
            platform=platform,
            platform_user_id=platform_user_id,
        )
        if resolution["status"] != "resolved":
            page = _lookup_page(
                status=resolution["status"],
                items=[],
                reason=resolution["reason"],
                identity=resolution["identity"],
            )
            return page
        clean_global_user_id = resolution["global_user_id"]

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
            identity=resolution["identity"],
        )
        return page

    async def lookup_user_entity(
        self,
        *,
        platform: str,
        platform_user_id: str,
        query: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return the owner-oriented user inspection envelope."""

        resolution = await self._resolve_platform_user_identity(
            platform=platform,
            platform_user_id=platform_user_id,
        )
        if resolution["status"] != "resolved":
            panels = {
                "profile": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
                "relationship": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
                "memory": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
                "style": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
            }
            envelope = _owner_entity_envelope(
                owner="user",
                identity=resolution["identity"],
                panels=panels,
                status=resolution["status"],
            )
            return envelope

        profile = resolution.get("profile")
        if not isinstance(profile, dict):
            profile = {}
        identity = dict(resolution["identity"])
        profile_items = _project_user_profile(
            profile,
            identity=identity,
        )
        relationship_items = _project_relationship_items(profile)
        memory = await self.lookup_memory(
            platform=platform,
            platform_user_id=platform_user_id,
            query=query,
            limit=limit,
        )
        style = await self.lookup_interaction_style(
            platform=platform,
            platform_user_id=platform_user_id,
            platform_channel_id="",
            limit=limit,
        )
        panels = {
            "profile": _entity_panel(
                status="available" if profile_items else "empty",
                items=profile_items[:limit],
                reason=(
                    ""
                    if profile_items
                    else "user profile has no browser-safe fields"
                ),
            ),
            "relationship": _entity_panel(
                status="available" if relationship_items else "empty",
                items=relationship_items[:limit],
                reason=(
                    ""
                    if relationship_items
                    else "relationship summary is not available"
                ),
            ),
            "memory": _lookup_panel_from_page(memory),
            "style": _lookup_panel_from_page(style),
        }
        envelope = _owner_entity_envelope(
            owner="user",
            identity=identity,
            panels=panels,
        )
        return envelope

    async def lookup_group_entity(
        self,
        *,
        platform: str,
        group_id: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return the owner-oriented group inspection envelope."""

        clean_platform = platform.strip()
        clean_group_id = group_id.strip()
        identity = {
            "platform": clean_platform,
            "group_id": clean_group_id,
        }
        if not clean_platform and not clean_group_id:
            status_value = "needs_input"
            reason = "platform and group id are required"
        elif not clean_platform:
            status_value = "needs_input"
            reason = "platform is required when group id is provided"
        elif not clean_group_id:
            status_value = "needs_input"
            reason = "group id is required when platform is provided"
        else:
            status_value = ""
            reason = ""

        if status_value:
            panels = {
                "style": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "progress": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "guidance": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
            }
            envelope = _owner_entity_envelope(
                owner="group",
                identity=identity,
                panels=panels,
                status=status_value,
            )
            return envelope

        style = await self.lookup_interaction_style(
            platform=clean_platform,
            platform_user_id="",
            platform_channel_id=clean_group_id,
            limit=limit,
        )
        panels = {
            "style": _lookup_panel_from_page(style),
            "progress": _entity_panel(
                status="empty",
                items=[],
                reason=(
                    "group conversation-progress summaries are not exposed by "
                    "this read-only console surface"
                ),
            ),
            "guidance": _entity_panel(
                status="empty",
                items=[],
                reason=(
                    "reflection-derived group guidance is not available in a "
                    "browser-safe projection"
                ),
            ),
        }
        envelope = _owner_entity_envelope(
            owner="group",
            identity=identity,
            panels=panels,
        )
        return envelope

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
        platform: str,
        platform_user_id: str,
        platform_channel_id: str,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Return scoped interaction-style guidance for operator inspection."""

        clean_platform = platform.strip()
        clean_platform_user_id = platform_user_id.strip()
        clean_platform_channel_id = platform_channel_id.strip()
        if not clean_platform_user_id and not clean_platform_channel_id:
            page = _style_lookup_page(
                status="needs_input",
                items=[],
                reason=(
                    "platform and platform user id are required for private "
                    "style lookup; platform and group id are required for "
                    "group style lookup"
                ),
            )
            return page
        if (clean_platform_user_id or clean_platform_channel_id) and not clean_platform:
            page = _style_lookup_page(
                status="needs_input",
                items=[],
                reason="platform is required for user or group style lookup",
            )
            return page

        identity: dict[str, Any] | None = None
        clean_global_user_id = ""
        if clean_platform_user_id:
            resolution = await self._resolve_platform_user_identity(
                platform=clean_platform,
                platform_user_id=clean_platform_user_id,
            )
            identity = resolution["identity"]
            if resolution["status"] != "resolved":
                page = _style_lookup_page(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                    identity=identity,
                )
                return page
            clean_global_user_id = resolution["global_user_id"]

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
            identity=identity,
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


def _display_name_for_platform_account(
    profile: dict[str, Any],
    *,
    platform: str,
    platform_user_id: str,
) -> str:
    """Return the display name for the matched platform account.

    Returns:
        The matching account display name, or an empty string when unavailable.
    """

    accounts = profile.get("platform_accounts")
    if not isinstance(accounts, list):
        return_value = ""
        return return_value

    for account in accounts:
        if not isinstance(account, dict):
            continue
        account_platform = str(account.get("platform", "")).strip()
        account_platform_user_id = str(account.get("platform_user_id", "")).strip()
        if account_platform != platform or account_platform_user_id != platform_user_id:
            continue
        return_value = str(account.get("display_name", "")).strip()
        return return_value

    return_value = ""
    return return_value


def _owner_entity_envelope(
    *,
    owner: str,
    identity: dict[str, Any],
    panels: dict[str, dict[str, Any]],
    status: str | None = None,
) -> dict[str, Any]:
    """Build a browser-safe owner inspection envelope."""

    status_value = status or _combined_panel_status(panels)
    envelope = {
        "status": status_value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owner": owner,
        "identity": redact_mapping(identity),
        "panels": panels,
        "redaction": _owner_entity_redaction(),
    }
    return envelope


def _combined_panel_status(panels: dict[str, dict[str, Any]]) -> str:
    """Return one top-level status from child panel states."""

    statuses = [
        str(panel.get("status", "empty"))
        for panel in panels.values()
        if isinstance(panel, dict)
    ]
    if "available" in statuses:
        status_value = "available"
    elif "needs_input" in statuses:
        status_value = "needs_input"
    elif "unavailable" in statuses:
        status_value = "unavailable"
    else:
        status_value = "empty"
    return status_value


def _entity_panel(
    *,
    status: str,
    items: list[dict[str, Any]],
    reason: str = "",
) -> dict[str, Any]:
    """Build one owner-page panel from bounded table rows."""

    panel = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [redact_mapping(item) for item in items if isinstance(item, dict)],
        "reason": str(reason)[:160],
    }
    return panel


def _lookup_panel_from_page(page: dict[str, Any]) -> dict[str, Any]:
    """Convert an existing lookup page into an owner-envelope panel."""

    raw_items = page.get("items", [])
    items = raw_items if isinstance(raw_items, list) else []
    panel = _entity_panel(
        status=str(page.get("status", "unavailable")),
        items=[item for item in items if isinstance(item, dict)],
        reason=str(page.get("reason", "")),
    )
    return panel


def _owner_entity_redaction() -> dict[str, str]:
    """Return the owner-envelope redaction contract."""

    redaction = {
        "model_inputs": "excluded",
        "raw_messages": "excluded",
        "raw_reflections": "excluded",
        "internal_global_ids": "excluded",
        "vector_fields": "excluded",
    }
    return redaction


def _project_character_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Project character profile fields that are safe for the console."""

    row = {
        field: profile[field]
        for field in CHARACTER_PROFILE_FIELDS
        if field in profile and profile[field] not in (None, "")
    }
    items = [redact_mapping(row)] if row else []
    return items


def _project_self_image(value: Any) -> list[dict[str, Any]]:
    """Project the character self-image field into table rows."""

    if isinstance(value, str) and value.strip():
        items = [{"summary": value.strip()}]
        return items
    if not isinstance(value, dict):
        items: list[dict[str, Any]] = []
        return items

    row: dict[str, Any] = {}
    for field in SELF_IMAGE_FIELDS:
        if field not in value:
            continue
        field_value = value[field]
        if field_value in (None, "", [], {}):
            continue
        row[field] = field_value

    meta = value.get("meta")
    if isinstance(meta, dict):
        last_updated = meta.get("last_updated")
        if last_updated not in (None, ""):
            row["last_updated"] = last_updated
        synthesis_count = meta.get("synthesis_count")
        if synthesis_count not in (None, ""):
            row["synthesis_count"] = synthesis_count

    items = [redact_mapping(row)] if row else []
    return items


def _project_key_value_items(value: Any) -> list[dict[str, Any]]:
    """Project a mapping into key/value rows."""

    if not isinstance(value, dict):
        items: list[dict[str, Any]] = []
        return items

    items = [
        redact_mapping({"key": key, "value": item})
        for key, item in value.items()
        if item not in (None, "")
    ]
    return items


def _project_learning_items(state_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Project promoted learning summaries from safe character state fields."""

    summary = state_summary.get("summary", {})
    if not isinstance(summary, dict):
        items: list[dict[str, Any]] = []
        return items

    reflection_summary = summary.get("reflection_summary")
    if not reflection_summary:
        items = []
        return items

    items = [redact_mapping({
        "source": "character_state.reflection_summary",
        "summary": reflection_summary,
    })]
    return items


def _project_user_profile(
    profile: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project a user profile without internal canonical identifiers."""

    row = {
        field: profile[field]
        for field in USER_PROFILE_FIELDS
        if field in profile and profile[field] not in (None, "")
    }
    for field in ("platform", "platform_user_id", "display_name"):
        value = identity.get(field)
        if value not in (None, ""):
            row[field] = value

    items = [redact_mapping(row)] if row else []
    return items


def _project_relationship_items(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Project relationship-oriented user profile fields."""

    rows: list[dict[str, Any]] = []
    relationship_summary = profile.get("relationship_summary")
    if relationship_summary:
        rows.append({
            "key": "relationship_summary",
            "value": relationship_summary,
        })
    last_relationship_insight = profile.get("last_relationship_insight")
    if last_relationship_insight:
        rows.append({
            "key": "last_relationship_insight",
            "value": last_relationship_insight,
        })
    relationship_status = profile.get("relationship_status")
    if relationship_status:
        rows.append({
            "key": "relationship_status",
            "value": relationship_status,
        })
    affinity = profile.get("affinity")
    if affinity not in (None, ""):
        rows.append({
            "key": "affinity",
            "value": affinity,
        })
    items = [redact_mapping(row) for row in rows]
    return items


def _platform_user_resolution(
    *,
    status: str,
    platform: str,
    platform_user_id: str,
    reason: str,
    display_name: str = "",
    global_user_id: str = "",
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an identity-resolution result for lookup pages.

    Returns:
        A dictionary with browser-safe identity metadata plus the canonical user
        id reserved for internal repository calls.
    """

    resolution = {
        "status": status,
        "reason": reason,
        "global_user_id": global_user_id,
        "identity": {
            "platform": platform,
            "platform_user_id": platform_user_id,
            "display_name": display_name,
            "resolution_status": status,
        },
    }
    if profile is not None:
        resolution["profile"] = profile
    return resolution


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
    identity: dict[str, Any] | None = None,
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
    if identity is not None:
        page["identity"] = identity
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
    identity: dict[str, Any] | None = None,
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
    if identity is not None:
        page["identity"] = identity
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
