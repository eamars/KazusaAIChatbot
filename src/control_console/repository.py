"""Read-only repository adapters for console lookup pages."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import PyMongoError

from control_console.redaction import (
    redact_character_operational_state_view,
    redact_latest_context_consumption,
    redact_mapping,
    redact_operational_relationship_context,
)
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.character_identity_growth.projection import (
    project_candidate_for_console,
    project_growth_health_for_console,
    project_growth_run_for_console,
    project_identity_for_console,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_character_operational_state,
    project_numeric_band,
    project_operational_relationship_context,
    project_relationship_context,
)
from kazusa_ai_chatbot.conversation_progress import (
    ConversationProgressScope,
    load_progress_context as default_load_progress_context,
)
from kazusa_ai_chatbot.db import background_work_jobs as background_work_job_store
from kazusa_ai_chatbot.db import character_identity_growth as identity_store
from kazusa_ai_chatbot.db.character import (
    get_character_profile as default_get_character_profile,
    get_character_runtime_state as default_get_character_runtime_state,
)
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_interaction_style_context as default_build_interaction_style_context,
)
from kazusa_ai_chatbot.db.conversation import (
    list_recent_group_summaries as default_list_recent_group_summaries,
)
from kazusa_ai_chatbot.db.self_cognition import (
    list_group_review_windows as default_list_group_review_windows,
)
from kazusa_ai_chatbot.db.user_memory_units import (
    query_user_memory_units as default_query_user_memory_units,
    search_user_memory_units_by_keyword as default_search_user_memory_units_by_keyword,
)
from kazusa_ai_chatbot.db.users import (
    find_user_profile_by_identifier as default_find_user_profile_by_identifier,
    get_user_profile as default_get_user_profile,
    list_recent_user_profiles as default_list_recent_user_profiles,
)
from kazusa_ai_chatbot.internal_monologue_residue import (
    load_residue_context as default_load_residue_context,
)
from kazusa_ai_chatbot.rag.recall.collectors.calendar_runs import (
    CalendarRunCollector,
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
    "backstory",
    "personality_brief",
    "boundary_profile",
    "linguistic_texture_profile",
    "visual_characterization",
    "updated_at",
)
SELF_IMAGE_FIELDS = (
    "self_concept",
    "current_growth_edges",
)
CHARACTER_COGNITION_FIELDS = (
    "drives",
    "standards",
    "meaning_state",
    "goals",
    "threats",
    "active_events",
    "knowledge_gaps",
    "affect_activations",
    "updated_at",
)
USER_COGNITION_FIELDS = (
    "goals",
    "threats",
    "active_events",
    "knowledge_gaps",
    "affect_activations",
    "updated_at",
)
RELATIONSHIP_AXES = (
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
)
SIGNED_RELATIONSHIP_AXES = frozenset({
    "positive_regard",
    "trust",
    "boundary_safety",
})
PRIVATE_STATE_KEYS = frozenset({
    "schema_version",
    "state_scope",
    "owner_user_id",
    "relationship_id",
    "other_user_id",
    "evidence_refs",
    "entity_id",
    "standard_id",
    "activation_id",
    "emotion_id",
    "root_refs",
    "source_refs",
    "role_refs",
})
SAFE_WORKER_EVENT_FIELDS = (
    "processed_count",
    "succeeded_count",
    "failed_count",
    "skipped_count",
    "deferred",
    "defer_reason",
    "run_kind",
    "worker_name",
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
MAX_AUDIT_READ_LIMIT = 100
AUDIT_SUCCESS_EVENTS = frozenset({
    "service_started",
    "service_stopped",
    "service_config_applied",
    "brain_model_route_applied",
})
AUDIT_FAILURE_EVENTS = frozenset({
    "auth_failed",
    "service_crashed",
    "debug_chat_unavailable",
    "service_config_apply_failed",
    "service_config_reset_failed",
    "brain_model_route_apply_failed",
    "brain_model_route_reset_failed",
})
AUDIT_ACTION_PREFIXES = (
    ("brain_model_route_apply", "model route apply"),
    ("brain_model_route_reset", "model route reset"),
    ("service_config_apply", "service config apply"),
    ("service_config_reset", "service config reset"),
    ("service_restart", "service restart"),
    ("service_start", "service start"),
    ("service_stop", "service stop"),
    ("service_crashed", "service crash"),
    ("debug_chat", "debug chat"),
    ("auth", "authentication"),
)
AUDIT_VIEW_EVENT_LABELS = {
    "audit_view": "Audit",
    "brain_model_route_models_view": "Services",
    "brain_model_routes_view": "Services",
    "event_view": "Event monitor",
}
AUDIT_LOOKUP_VIEW_LABELS = {
    "background": "Background work",
    "calendar": "Calendar",
    "entity.character": "Character",
    "entity.group": "Groups",
    "entity.groups": "Groups",
    "entity.user": "Users",
    "entity.users": "Users",
    "memory": "Memory lookup",
    "style": "Style lookup",
}


class ControlConsoleRepository:
    """Read-only domain lookup facade with safe unavailable fallbacks."""

    def __init__(
        self,
        *,
        get_character_profile: AsyncHelper | None = None,
        get_character_runtime_state: AsyncHelper | None = None,
        list_identity_revisions: AsyncHelper | None = None,
        list_identity_growth_candidates: AsyncHelper | None = None,
        list_recent_identity_growth_runs: AsyncHelper | None = None,
        build_identity_growth_health: AsyncHelper | None = None,
        query_user_memory_units: AsyncHelper | None = None,
        search_user_memory_units_by_keyword: AsyncHelper | None = None,
        build_interaction_style_context: AsyncHelper | None = None,
        find_user_profile_by_identifier: AsyncHelper | None = None,
        get_character_user_profile: AsyncHelper | None = None,
        collect_calendar_pending_runs: AsyncHelper | None = None,
        list_calendar_schedules: AsyncHelper | None = None,
        list_recent_calendar_runs: AsyncHelper | None = None,
        find_deliverable_background_work_jobs: AsyncHelper | None = None,
        list_recent_background_work_jobs: AsyncHelper | None = None,
        load_progress_context: AsyncHelper | None = None,
        load_residue_context: AsyncHelper | None = None,
        list_recent_user_profiles: AsyncHelper | None = None,
        list_recent_group_summaries: AsyncHelper | None = None,
        list_group_review_windows: AsyncHelper | None = None,
    ) -> None:
        """Create a read-only repository facade."""

        self._get_character_profile = get_character_profile
        self._get_character_runtime_state = get_character_runtime_state
        self._list_identity_revisions = list_identity_revisions
        self._list_identity_growth_candidates = (
            list_identity_growth_candidates
        )
        self._list_recent_identity_growth_runs = (
            list_recent_identity_growth_runs
        )
        self._build_identity_growth_health = build_identity_growth_health
        self._query_user_memory_units = query_user_memory_units
        self._search_user_memory_units_by_keyword = search_user_memory_units_by_keyword
        self._build_interaction_style_context = build_interaction_style_context
        self._find_user_profile_by_identifier = find_user_profile_by_identifier
        self._get_character_user_profile = get_character_user_profile
        self._collect_calendar_pending_runs = collect_calendar_pending_runs
        self._list_calendar_schedules = list_calendar_schedules
        self._list_recent_calendar_runs = list_recent_calendar_runs
        self._find_deliverable_background_work_jobs = (
            find_deliverable_background_work_jobs
        )
        self._list_recent_background_work_jobs = list_recent_background_work_jobs
        self._load_progress_context = load_progress_context
        self._load_residue_context = load_residue_context
        self._list_recent_user_profiles = list_recent_user_profiles
        self._list_recent_group_summaries = list_recent_group_summaries
        self._list_group_review_windows = list_group_review_windows

    async def application_identity(self) -> dict[str, Any]:
        """Return the active character name for the browser shell."""

        try:
            helper = self._get_character_profile or default_get_character_profile
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
            "source": "character_identity_revisions",
        }
        return identity

    async def character_entity(
        self,
        *,
        current_timestamp_utc: str | None = None,
        latest_context_consumption: Mapping[str, Any] | None = None,
        include_operational_context: bool = False,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Return native profile, cognition, growth, and continuity state."""

        timestamp = current_timestamp_utc or datetime.now(timezone.utc).isoformat()
        identity: dict[str, Any] = {}
        profile: dict[str, Any] = {}
        profile_panel = _entity_panel(
            status="unavailable",
            items=[],
            reason="character profile helper is unavailable",
        )
        profile_helper = self._get_character_profile or default_get_character_profile
        try:
            loaded_profile = await profile_helper()
        except APPLICATION_IDENTITY_ERRORS as exc:
            profile_panel["reason"] = str(exc)[:160]
        else:
            if isinstance(loaded_profile, dict):
                profile = loaded_profile
                profile_items = _project_character_profile(profile)
                profile_panel = _entity_panel(
                    status="available" if profile_items else "empty",
                    items=profile_items,
                    reason=(
                        ""
                        if profile_items
                        else "character profile has no browser-safe fields"
                    ),
                )
                identity = {
                    "character_name": str(profile.get("name", "")).strip()[:120],
                }
            else:
                profile_panel["reason"] = (
                    "character profile helper returned invalid data"
                )

        runtime_panel = _entity_panel(
            status="unavailable",
            items=[],
            reason="character runtime-state helper is unavailable",
        )
        runtime_cognition_state: Any = None
        self_image_panel = _entity_panel(
            status="unavailable",
            items=[],
            reason="character identity profile is unavailable",
        )
        runtime_helper = (
            self._get_character_runtime_state
            or default_get_character_runtime_state
        )
        try:
            runtime_state = await runtime_helper()
        except REPOSITORY_HELPER_ERRORS as exc:
            reason = str(exc)[:160]
            runtime_panel["reason"] = reason
        else:
            if isinstance(runtime_state, dict):
                cognition_state = runtime_state.get("cognition_state")
                runtime_cognition_state = cognition_state
                cognition_items = _project_cognition_state_items(
                    cognition_state,
                    fields=CHARACTER_COGNITION_FIELDS,
                )
                runtime_panel = _entity_panel(
                    status="available" if cognition_items else "empty",
                    items=cognition_items,
                    reason=(
                        ""
                        if cognition_items
                        else "character cognition state is empty"
                    ),
                )
            else:
                runtime_panel["reason"] = (
                    "character runtime-state helper returned invalid data"
                )

        if profile_panel["status"] != "unavailable":
            self_image_items = _project_self_image(profile.get("self_image"))
            self_image_panel = _entity_panel(
                status="available" if self_image_items else "empty",
                items=self_image_items[:limit],
                reason=(
                    ""
                    if self_image_items
                    else "latest identity has no self-image"
                ),
            )
        character_id = _character_id_from_profile(profile)
        panels = {
            "profile": profile_panel,
            "cognition_state": runtime_panel,
        }
        if include_operational_context:
            panels["operational_posture"] = (
                _project_character_operational_posture(
                    runtime_cognition_state,
                    effective_at=timestamp,
                    latest_context_consumption=latest_context_consumption,
                )
            )
        panels.update({
            "self_image": self_image_panel,
            "growth": await self._character_identity_growth_panel(
                character_id=character_id,
                limit=limit,
            ),
            "carry_over": await self._character_identity_lineage_panel(
                character_id=character_id,
                limit=limit,
            ),
        })
        envelope = _owner_entity_envelope(
            owner="character",
            identity=identity,
            panels=panels,
            required_panel_names=("profile", "cognition_state"),
        )
        return envelope

    async def _character_identity_growth_panel(
        self,
        *,
        character_id: str,
        limit: int,
    ) -> dict[str, Any]:
        """Combine redacted candidate and routed-run outcomes."""

        candidate_helper = (
            self._list_identity_growth_candidates
            or identity_store.list_identity_growth_candidates
        )
        run_helper = (
            self._list_recent_identity_growth_runs
            or identity_store.list_recent_identity_growth_runs
        )
        source_panels: dict[str, dict[str, Any]] = {}
        items: list[dict[str, Any]] = []
        reasons: list[str] = []
        try:
            candidates = await candidate_helper(
                character_id=character_id,
                limit=limit,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            source_panels["candidates"] = {"status": "unavailable"}
            reasons.append(f"identity candidates unavailable: {exc}")
        else:
            candidate_items = [
                project_candidate_for_console(candidate)
                for candidate in list(candidates)[:limit]
                if isinstance(candidate, dict)
            ]
            items.extend(candidate_items)
            source_panels["candidates"] = {
                "status": "available" if candidate_items else "empty"
            }

        try:
            runs = await run_helper(
                character_id=character_id,
                limit=limit,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            source_panels["runs"] = {"status": "unavailable"}
            reasons.append(f"identity growth runs unavailable: {exc}")
        else:
            run_items = [
                project_growth_run_for_console(run)
                for run in list(runs)[:limit]
                if isinstance(run, dict)
            ]
            items.extend(run_items)
            source_panels["runs"] = {
                "status": "available" if run_items else "empty"
            }

        status_value = _combined_panel_status(source_panels)
        reason = "; ".join(reasons)
        if status_value == "empty":
            reason = "no identity candidates or routed growth runs"
        return _entity_panel(
            status=status_value,
            items=items,
            reason=reason,
        )

    async def _character_identity_lineage_panel(
        self,
        *,
        character_id: str,
        limit: int,
    ) -> dict[str, Any]:
        """Combine public health with immutable redacted revision history."""

        revision_helper = (
            self._list_identity_revisions
            or identity_store.list_identity_revisions
        )
        health_helper = (
            self._build_identity_growth_health
            or identity_store.build_identity_growth_health
        )
        source_panels: dict[str, dict[str, Any]] = {}
        items: list[dict[str, Any]] = []
        reasons: list[str] = []
        latest_revision_number: int | None = None

        try:
            health = await health_helper(character_id=character_id)
        except REPOSITORY_HELPER_ERRORS as exc:
            source_panels["health"] = {"status": "unavailable"}
            reasons.append(f"identity health unavailable: {exc}")
        else:
            if not isinstance(health, dict):
                source_panels["health"] = {"status": "unavailable"}
                reasons.append("identity health helper returned invalid data")
            else:
                health_item = project_growth_health_for_console(health)
                latest_revision_number = int(
                    health_item["latest_revision_number"]
                )
                items.append(health_item)
                source_panels["health"] = {"status": "available"}

        try:
            revisions = await revision_helper(
                character_id=character_id,
                limit=limit,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            source_panels["revisions"] = {"status": "unavailable"}
            reasons.append(f"identity revisions unavailable: {exc}")
        else:
            revision_items = [
                project_identity_for_console(revision)
                for revision in list(revisions)[:limit]
                if isinstance(revision, dict)
            ]
            if latest_revision_number is None and revision_items:
                latest_revision_number = max(
                    int(item["revision_number"])
                    for item in revision_items
                )
            for revision_item in revision_items:
                revision_item["is_current"] = (
                    revision_item["revision_number"]
                    == latest_revision_number
                )
            items.extend(revision_items)
            source_panels["revisions"] = {
                "status": "available" if revision_items else "empty"
            }

        status_value = _combined_panel_status(source_panels)
        reason = "; ".join(reasons)
        if status_value == "empty":
            reason = "no identity revision continuity is available"
        return _entity_panel(
            status=status_value,
            items=items,
            reason=reason,
        )

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
                query_helper = default_query_user_memory_units
                keyword_helper = default_search_user_memory_units_by_keyword

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
        platform_channel_id: str = "",
        channel_type: str = "",
        query: str,
        current_timestamp_utc: str | None = None,
        include_operational_context: bool = False,
        limit: int,
    ) -> dict[str, Any]:
        """Return native V2 state for one platform-facing user account."""

        timestamp = current_timestamp_utc or datetime.now(timezone.utc).isoformat()
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
                "cognition_state": _entity_panel(
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
                "conversation_progress": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
                "carry_over": _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                ),
            }
            if include_operational_context:
                panels["relationship_operational"] = _entity_panel(
                    status=resolution["status"],
                    items=[],
                    reason=resolution["reason"],
                )
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
        cognition_state = profile.get("cognition_state")
        relationship_panel = _project_relationship_panel(cognition_state)
        relationship_operational_panel = (
            _project_relationship_operational_panel(
                cognition_state,
                effective_at=timestamp,
            )
            if include_operational_context
            else None
        )
        cognition_items = _project_cognition_state_items(
            cognition_state,
            fields=USER_COGNITION_FIELDS,
        )
        memory = await self.lookup_memory(
            platform=platform,
            platform_user_id=platform_user_id,
            query=query,
            limit=limit,
        )
        style_channel_id = (
            platform_channel_id
            if channel_type.strip().lower() == "group"
            else ""
        )
        style = await self.lookup_interaction_style(
            platform=platform,
            platform_user_id=platform_user_id,
            platform_channel_id=style_channel_id,
            limit=limit,
        )
        thread_scope_reason = _missing_scope_reason(
            (
                ("channel id", platform_channel_id),
                ("channel type", channel_type),
            ),
            purpose="user-thread carry-over",
        )
        if thread_scope_reason:
            carry_over_panel = _entity_panel(
                status="needs_input",
                items=[],
                reason=thread_scope_reason,
            )
        else:
            carry_over_panel = await self._residue_panel(
                trigger_scope={
                    "character_id": await self._active_character_id(),
                    "platform": platform.strip(),
                    "platform_channel_id": platform_channel_id.strip(),
                    "channel_type": channel_type.strip(),
                    "global_user_id": resolution["global_user_id"],
                },
                current_timestamp_utc=timestamp,
                empty_reason="no current user-thread carry-over is loaded",
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
            "relationship": relationship_panel,
            "cognition_state": _entity_panel(
                status="available" if cognition_items else "empty",
                items=cognition_items,
                reason=(
                    ""
                    if cognition_items
                    else "user cognition state is empty"
                ),
            ),
            "memory": _lookup_panel_from_page(memory),
            "style": _lookup_panel_from_page(style),
            "conversation_progress": await (
                self._conversation_progress_panel(
                    platform=platform,
                    platform_channel_id=platform_channel_id,
                    channel_type=channel_type,
                    global_user_id=resolution["global_user_id"],
                    current_timestamp_utc=timestamp,
                )
            ),
            "carry_over": carry_over_panel,
        }
        if relationship_operational_panel is not None:
            panels["relationship_operational"] = relationship_operational_panel
        required_panel_names = (
            "profile",
            "relationship",
            "cognition_state",
        )
        if include_operational_context:
            required_panel_names = (
                "profile",
                "relationship",
                "relationship_operational",
                "cognition_state",
            )
        envelope = _owner_entity_envelope(
            owner="user",
            identity=identity,
            panels=panels,
            required_panel_names=required_panel_names,
        )
        return envelope

    async def list_user_entities(self, *, limit: int) -> dict[str, Any]:
        """Return a bounded directory of safe platform-facing user accounts."""

        helper = (
            self._list_recent_user_profiles
            or default_list_recent_user_profiles
        )
        try:
            profiles = await helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            page = _directory_page(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return page

        items = [
            _project_user_directory_item(profile)
            for profile in list(profiles)[:limit]
            if isinstance(profile, dict)
        ]
        items = [item for item in items if item]
        page = _directory_page(
            status="available" if items else "empty",
            items=items,
            reason="no recent user profiles are available" if not items else "",
        )
        return page

    async def lookup_group_entity(
        self,
        *,
        platform: str,
        group_id: str,
        participant_platform_user_id: str = "",
        current_timestamp_utc: str | None = None,
        limit: int,
    ) -> dict[str, Any]:
        """Return sourced activity and continuity for one group scope."""

        timestamp = current_timestamp_utc or datetime.now(timezone.utc).isoformat()
        clean_platform = platform.strip()
        clean_group_id = group_id.strip()
        clean_participant_platform_user_id = participant_platform_user_id.strip()
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
                "activity": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "review": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "style": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "carry_over": _entity_panel(
                    status=status_value,
                    items=[],
                    reason=reason,
                ),
                "participant_progress": _entity_panel(
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

        activity_panel = await self._group_activity_panel(
            platform=clean_platform,
            group_id=clean_group_id,
        )
        review_panel = await self._group_review_panel(
            platform=clean_platform,
            group_id=clean_group_id,
        )
        style = await self.lookup_interaction_style(
            platform=clean_platform,
            platform_user_id="",
            platform_channel_id=clean_group_id,
            limit=limit,
        )
        panels = {
            "activity": activity_panel,
            "review": review_panel,
            "style": _lookup_panel_from_page(style),
            "carry_over": await self._residue_panel(
                trigger_scope={
                    "character_id": await self._active_character_id(),
                    "platform": clean_platform,
                    "platform_channel_id": clean_group_id,
                    "channel_type": "group",
                    "global_user_id": "",
                },
                current_timestamp_utc=timestamp,
                empty_reason="no group-scene carry-over is loaded",
            ),
            "participant_progress": await (
                self._participant_progress_panel(
                    platform=clean_platform,
                    group_id=clean_group_id,
                    participant_platform_user_id=clean_participant_platform_user_id,
                    current_timestamp_utc=timestamp,
                )
            ),
        }
        envelope = _owner_entity_envelope(
            owner="group",
            identity=identity,
            panels=panels,
            required_panel_names=("activity", "review"),
        )
        return envelope

    async def list_group_entities(self, *, limit: int) -> dict[str, Any]:
        """Return a bounded group directory sourced from conversation metadata."""

        helper = (
            self._list_recent_group_summaries
            or default_list_recent_group_summaries
        )
        try:
            summaries = await helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            page = _directory_page(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return page

        items = [
            _project_group_summary(summary)
            for summary in list(summaries)[:limit]
            if isinstance(summary, dict)
        ]
        page = _directory_page(
            status="available" if items else "empty",
            items=items,
            reason="no recent group activity is available" if not items else "",
        )
        return page

    async def _group_activity_panel(
        self,
        *,
        platform: str,
        group_id: str,
    ) -> dict[str, Any]:
        """Return current bounded activity aggregates for one group."""

        helper = (
            self._list_recent_group_summaries
            or default_list_recent_group_summaries
        )
        try:
            summaries = await helper(
                limit=1,
                platform=platform,
                platform_channel_id=group_id,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_group_summary(summary)
            for summary in list(summaries)[:1]
            if isinstance(summary, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no group activity matched this scope" if not items else "",
        )
        return panel

    async def _group_review_panel(
        self,
        *,
        platform: str,
        group_id: str,
    ) -> dict[str, Any]:
        """Return the latest terminal self-cognition review for one group."""

        helper = (
            self._list_group_review_windows
            or default_list_group_review_windows
        )
        try:
            reviews = await helper(
                platform=platform,
                platform_channel_id=group_id,
                limit=1,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_group_review(review)
            for review in list(reviews)[:1]
            if isinstance(review, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no terminal group reviews matched this scope" if not items else "",
        )
        return panel

    async def lookup_calendar(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        platform_user_id: str,
        channel_type: str,
        current_timestamp_utc: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return schedule state, recent outcomes, and scoped cognition visibility."""

        cognition_panel = await self._calendar_pending_runs_panel(
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            channel_type=channel_type,
            current_timestamp_utc=current_timestamp_utc,
        )
        schedules_panel = await self._calendar_schedules_panel(limit=limit)
        runs_panel = await self._recent_calendar_runs_panel(limit=limit)
        schedules = schedules_panel.get("items", [])
        if not isinstance(schedules, list):
            schedules = []
        runs = runs_panel.get("items", [])
        if not isinstance(runs, list):
            runs = []
        active_schedules = [
            schedule
            for schedule in schedules
            if isinstance(schedule, dict) and schedule.get("status") == "active"
        ]
        summary = {
            "active_schedules": len(active_schedules),
            "upcoming": sum(
                1
                for schedule in active_schedules
                if str(schedule.get("next_run_at", "")) > current_timestamp_utc
            ),
            "overdue": sum(
                1
                for schedule in active_schedules
                if (
                    str(schedule.get("next_run_at", ""))
                    and str(schedule.get("next_run_at", ""))
                    <= current_timestamp_utc
                )
            ),
            "running": sum(
                1
                for run in runs
                if isinstance(run, dict) and run.get("status") == "running"
            ),
            "completed": sum(
                1
                for run in runs
                if isinstance(run, dict) and run.get("status") == "completed"
            ),
            "failed": sum(
                1
                for run in runs
                if isinstance(run, dict) and run.get("status") == "failed"
            ),
            "skipped": sum(
                1
                for run in runs
                if isinstance(run, dict) and run.get("status") == "skipped"
            ),
        }
        source_status = _combined_panel_status({
            "schedules": schedules_panel,
            "runs": runs_panel,
        })
        summary_panel = _entity_panel(
            status=source_status if source_status != "empty" else "available",
            items=[summary],
            reason=(
                "calendar summary is partial because one source is unavailable"
                if source_status == "partial"
                else ""
            ),
        )
        panels = {
            "summary": summary_panel,
            "schedules": schedules_panel,
            "runs": runs_panel,
            "cognition_visibility": cognition_panel,
        }
        page = _panel_lookup_page(
            namespace="calendar",
            panels=panels,
            required_panel_names=("summary", "schedules", "runs"),
        )
        return page

    def audit_page(
        self,
        *,
        events: list[dict[str, Any]],
        limit: int,
        category: str = "",
        event_type: str = "",
        service_id: str = "",
        operator_id: str = "",
        outcome: str = "",
        request_id: str = "",
        since: str = "",
    ) -> dict[str, Any]:
        """Collapse bounded local audit rows into actions and view counts."""

        bounded_events = [
            event
            for event in events[:MAX_AUDIT_READ_LIMIT]
            if isinstance(event, dict)
        ]
        view_counts: dict[str, int] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in bounded_events:
            current_event_type = str(event.get("event_type", ""))
            if current_event_type.endswith("_view"):
                view_label = _audit_view_label(event)
                view_counts[view_label] = (
                    view_counts.get(view_label, 0) + 1
                )
                continue
            current_request_id = str(event.get("request_id", "")).strip()
            if not current_request_id:
                continue
            grouped.setdefault(current_request_id, []).append(event)

        actions = [
            _project_audit_action(group_events)
            for group_events in grouped.values()
        ]
        actions = [
            action
            for action in actions
            if _audit_action_matches(
                action,
                category=category,
                event_type=event_type,
                service_id=service_id,
                operator_id=operator_id,
                outcome=outcome,
                request_id=request_id,
                since=since,
            )
        ]
        actions.sort(
            key=lambda action: str(action.get("created_at", "")),
            reverse=True,
        )
        actions = actions[:limit]
        outcome_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        for action in actions:
            action_outcome = str(action.get("outcome", ""))
            action_name = str(action.get("action", ""))
            outcome_counts[action_outcome] = (
                outcome_counts.get(action_outcome, 0) + 1
            )
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        page = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "actions": actions,
            "view_summary": [
                {"view": key, "count": value}
                for key, value in sorted(view_counts.items())
            ],
            "facets": {
                "outcomes": outcome_counts,
                "actions": action_counts,
            },
            "next_cursor": None,
        }
        return page

    async def _recent_calendar_runs_panel(
        self,
        *,
        limit: int,
    ) -> dict[str, Any]:
        """Return bounded recent calendar execution outcomes."""

        helper = (
            self._list_recent_calendar_runs
            or calendar_repository.list_recent_calendar_runs
        )
        try:
            documents = await helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_calendar_run(document)
            for document in list(documents)[:limit]
            if isinstance(document, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no recent calendar runs are available" if not items else "",
        )
        return panel

    async def _calendar_pending_runs_panel(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        platform_user_id: str,
        channel_type: str,
        current_timestamp_utc: str,
    ) -> dict[str, Any]:
        """Return source-scoped pending calendar candidates from Recall."""

        clean_platform = platform.strip()
        clean_platform_channel_id = platform_channel_id.strip()
        clean_platform_user_id = platform_user_id.strip()
        clean_channel_type = channel_type.strip()
        if not (
            clean_platform
            and clean_platform_channel_id
            and clean_platform_user_id
            and clean_channel_type
        ):
            panel = _entity_panel(
                status="needs_input",
                items=[],
                reason=(
                    "platform, channel id, platform user id, and channel type "
                    "are required for calendar cognition visibility"
                ),
            )
            return panel

        resolution = await self._resolve_platform_user_identity(
            platform=clean_platform,
            platform_user_id=clean_platform_user_id,
        )
        if resolution["status"] != "resolved":
            panel = _entity_panel(
                status=resolution["status"],
                items=[],
                reason=resolution["reason"],
            )
            return panel

        context = {
            "platform": clean_platform,
            "platform_channel_id": clean_platform_channel_id,
            "global_user_id": resolution["global_user_id"],
            "current_timestamp_utc": current_timestamp_utc,
        }
        try:
            if self._collect_calendar_pending_runs is None:
                collector = CalendarRunCollector()
                candidates = await collector.collect(context)
            else:
                candidates = await self._collect_calendar_pending_runs(context)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_calendar_candidate(candidate)
            for candidate in list(candidates)
            if isinstance(candidate, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no pending calendar recall candidates matched the scope"
            if not items
            else "",
        )
        return panel

    async def _calendar_schedules_panel(self, *, limit: int) -> dict[str, Any]:
        """Return bounded schedule-definition backing rows."""

        helper = (
            self._list_calendar_schedules
            or calendar_repository.list_calendar_schedules_for_inspection
        )
        try:
            schedules = await helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_calendar_schedule(schedule)
            for schedule in list(schedules)[:limit]
            if isinstance(schedule, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no active or paused schedule definitions matched the lookup"
            if not items
            else "",
        )
        return panel

    async def lookup_background_work(
        self,
        *,
        worker_event_rows: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Return job state and aggregated worker outcomes without tick noise."""

        jobs_panel = await self._background_job_queue_panel(limit=limit)
        delivery_panel = await self._background_delivery_panel(limit=limit)
        job_items = jobs_panel.get("items", [])
        if not isinstance(job_items, list):
            job_items = []
        summary = {
            "queued": sum(
                1
                for item in job_items
                if isinstance(item, dict) and item.get("status") == "queued"
            ),
            "running": sum(
                1
                for item in job_items
                if (
                    isinstance(item, dict)
                    and item.get("status") in {"in_progress", "delivery_in_progress"}
                )
            ),
            "completed": sum(
                1
                for item in job_items
                if (
                    isinstance(item, dict)
                    and item.get("status") in {"completed", "delivered"}
                )
            ),
            "failed": sum(
                1
                for item in job_items
                if (
                    isinstance(item, dict)
                    and item.get("status") in {"failed", "delivery_failed"}
                )
            ),
            "delivery_ready": sum(
                1
                for item in job_items
                if (
                    isinstance(item, dict)
                    and item.get("delivery_state") == "ready"
                )
            ),
            "deferred": sum(
                1
                for item in job_items
                if isinstance(item, dict) and item.get("status") == "deferred"
            ),
        }
        worker_activity = _worker_activity_panel(
            worker_event_rows,
            limit=limit,
        )
        errors = _worker_error_panel(worker_event_rows, limit=limit)
        summary_status = _combined_panel_status({
            "jobs": jobs_panel,
            "worker_activity": worker_activity,
        })
        if summary_status == "empty":
            summary_status = "available"
        panels = {
            "summary": _entity_panel(
                status=summary_status,
                items=[summary],
                reason=(
                    "background summary is partial because one source is unavailable"
                    if summary_status == "partial"
                    else ""
                ),
            ),
            "jobs": jobs_panel,
            "worker_activity": worker_activity,
            "errors": errors,
            "delivery_detail": delivery_panel,
        }
        page = _panel_lookup_page(
            namespace="background",
            panels=panels,
            required_panel_names=("summary", "jobs", "worker_activity"),
        )
        return page

    async def _background_delivery_panel(
        self,
        *,
        limit: int,
    ) -> dict[str, Any]:
        """Return safe delivery state for result-ready jobs."""

        job_helper = (
            self._find_deliverable_background_work_jobs
            or background_work_job_store.find_deliverable_background_work_jobs
        )
        try:
            jobs = await job_helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_background_job(job)
            for job in list(jobs)[:limit]
            if isinstance(job, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason=(
                "no background-work jobs are ready for delivery"
                if not items
                else ""
            ),
        )
        return panel

    async def _background_job_queue_panel(
        self,
        *,
        limit: int,
    ) -> dict[str, Any]:
        """Return bounded background-work job queue rows."""

        helper = (
            self._list_recent_background_work_jobs
            or background_work_job_store.list_recent_background_work_jobs
        )
        try:
            jobs = await helper(limit=limit)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        items = [
            _project_background_job(job)
            for job in list(jobs)[:limit]
            if isinstance(job, dict)
        ]
        panel = _entity_panel(
            status="available" if items else "empty",
            items=items,
            reason="no background-work jobs matched the lookup" if not items else "",
        )
        return panel

    async def _conversation_progress_panel(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        channel_type: str,
        global_user_id: str,
        current_timestamp_utc: str,
    ) -> dict[str, Any]:
        """Return one exact conversation-progress prompt projection."""

        clean_platform = platform.strip()
        clean_platform_channel_id = platform_channel_id.strip()
        clean_channel_type = channel_type.strip()
        clean_global_user_id = global_user_id.strip()
        scope_reason = _missing_scope_reason(
            (
                ("platform", clean_platform),
                ("channel id", clean_platform_channel_id),
                ("channel type", clean_channel_type),
                ("user identity", clean_global_user_id),
            ),
            purpose="conversation progress",
        )
        if scope_reason:
            panel = _entity_panel(
                status="needs_input",
                items=[],
                reason=scope_reason,
            )
            return panel

        helper = self._load_progress_context or default_load_progress_context
        scope = ConversationProgressScope(
            platform=clean_platform,
            platform_channel_id=clean_platform_channel_id,
            global_user_id=clean_global_user_id,
        )
        platform_bot_id = await self._active_character_platform_user_id(
            platform=clean_platform,
        )
        if not platform_bot_id:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=(
                    "active character has no registered platform account "
                    f"for {clean_platform}"
                ),
            )
            return panel
        try:
            result = await helper(
                scope=scope,
                current_timestamp_utc=current_timestamp_utc,
                platform_bot_id=platform_bot_id,
                active_turn_conversation_row_ids=[],
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        prompt_doc = result.get("conversation_progress")
        if not isinstance(prompt_doc, dict):
            prompt_doc = {}
        status_value = str(prompt_doc.get("status", "empty")) if prompt_doc else "empty"
        if status_value not in ("empty", "unavailable", "needs_input"):
            status_value = "available"
        source = str(result.get("source", ""))
        items = []
        if prompt_doc:
            items.append({
                "source": source,
                "state": _project_safe_state_value(prompt_doc),
            })
        panel = _entity_panel(
            status=status_value,
            items=items,
            reason="" if prompt_doc else "no conversation-progress prompt context is loaded",
        )
        return panel

    async def _participant_progress_panel(
        self,
        *,
        platform: str,
        group_id: str,
        participant_platform_user_id: str,
        current_timestamp_utc: str,
    ) -> dict[str, Any]:
        """Return participant progress only for an explicit group user id."""

        if not participant_platform_user_id:
            panel = _entity_panel(
                status="needs_input",
                items=[],
                reason="participant platform user id is required",
            )
            return panel

        resolution = await self._resolve_platform_user_identity(
            platform=platform,
            platform_user_id=participant_platform_user_id,
        )
        if resolution["status"] != "resolved":
            panel = _entity_panel(
                status=resolution["status"],
                items=[],
                reason=resolution["reason"],
            )
            return panel

        panel = await self._conversation_progress_panel(
            platform=platform,
            platform_channel_id=group_id,
            channel_type="group",
            global_user_id=resolution["global_user_id"],
            current_timestamp_utc=current_timestamp_utc,
        )
        return panel

    async def _residue_panel(
        self,
        *,
        trigger_scope: dict[str, str],
        current_timestamp_utc: str,
        empty_reason: str,
    ) -> dict[str, Any]:
        """Return current internal-monologue carry-over context."""

        helper = self._load_residue_context or default_load_residue_context
        helper_kwargs: dict[str, Any] = {
            "trigger_scope": trigger_scope,
            "current_timestamp_utc": current_timestamp_utc,
        }
        if self._load_residue_context is None:
            helper_kwargs["record_telemetry"] = False
        try:
            result = await helper(**helper_kwargs)
        except REPOSITORY_HELPER_ERRORS as exc:
            panel = _entity_panel(
                status="unavailable",
                items=[],
                reason=str(exc),
            )
            return panel

        content = str(result.get("internal_monologue_residue_context", ""))
        status_value = str(result.get("status", "empty"))
        if status_value == "loaded":
            status_value = "available"
        items = []
        if content:
            items.append({"context": content})
        panel = _entity_panel(
            status=status_value,
            items=items,
            reason="" if content else empty_reason,
        )
        return panel

    async def _active_character_id(self) -> str:
        """Return the active character id used for residue trigger scopes."""

        helper = self._get_character_profile or default_get_character_profile
        try:
            profile = await helper()
        except APPLICATION_IDENTITY_ERRORS:
            profile = {}
        if isinstance(profile, dict):
            character_id = _character_id_from_profile(profile)
            return character_id
        character_id = _character_id_from_profile({})
        return character_id

    async def _active_character_platform_user_id(self, *, platform: str) -> str:
        """Return the active character's native account id for a platform.

        Args:
            platform: Platform namespace whose character account is required.

        Returns:
            The registered native account id, or an empty string when the
            character identity cannot be resolved for the platform.
        """

        profile_helper = (
            self._get_character_profile or default_get_character_profile
        )
        user_profile_helper = (
            self._get_character_user_profile or default_get_user_profile
        )
        try:
            character_profile = await profile_helper()
            if not isinstance(character_profile, dict):
                return_value = ""
                return return_value
            character_id = _character_id_from_profile(character_profile)
            user_profile = await user_profile_helper(character_id)
        except APPLICATION_IDENTITY_ERRORS:
            return_value = ""
            return return_value

        if not isinstance(user_profile, dict):
            return_value = ""
            return return_value
        accounts = user_profile.get("platform_accounts")
        if not isinstance(accounts, list):
            return_value = ""
            return return_value
        for account in accounts:
            if not isinstance(account, dict):
                continue
            if str(account.get("platform", "")).strip() != platform:
                continue
            platform_user_id = str(
                account.get("platform_user_id", "")
            ).strip()
            if platform_user_id:
                return_value = platform_user_id
                return return_value

        return_value = ""
        return return_value

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
                helper = default_build_interaction_style_context

            context = await helper(
                global_user_id=clean_global_user_id,
                channel_type=channel_type,
                platform=clean_platform,
                platform_channel_id=clean_platform_channel_id,
            )
        except REPOSITORY_HELPER_ERRORS as exc:
            page["reason"] = str(exc)[:160]
            return page

        if not isinstance(context, Mapping):
            page["reason"] = "interaction-style helper returned invalid data"
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
        "source": "character_identity_revisions",
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


def _missing_scope_reason(
    requirements: tuple[tuple[str, str], ...],
    *,
    purpose: str,
) -> str:
    """Describe only the operator inputs absent from one scoped read."""

    missing_fields = [
        label
        for label, value in requirements
        if not str(value).strip()
    ]
    if not missing_fields:
        return ""
    if len(missing_fields) == 1:
        missing_text = missing_fields[0]
    else:
        missing_text = (
            ", ".join(missing_fields[:-1])
            + f" and {missing_fields[-1]}"
        )
    verb = "is" if len(missing_fields) == 1 else "are"
    reason = f"{missing_text} {verb} required"
    return f"{reason} for {purpose}"


def _owner_entity_envelope(
    *,
    owner: str,
    identity: dict[str, Any],
    panels: dict[str, dict[str, Any]],
    status: str | None = None,
    required_panel_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a browser-safe owner inspection envelope."""

    status_panels = _required_status_panels(
        panels,
        required_panel_names=required_panel_names,
    )
    status_value = status or _combined_panel_status(status_panels)
    envelope = {
        "status": status_value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "owner": owner,
        "identity": redact_mapping(identity),
        "panels": panels,
        "redaction": _owner_entity_redaction(),
    }
    return envelope


def _required_status_panels(
    panels: dict[str, dict[str, Any]],
    *,
    required_panel_names: tuple[str, ...] | None,
) -> dict[str, dict[str, Any]]:
    """Select the panels that own a page's top-level availability."""

    if required_panel_names is None:
        return panels
    return {
        panel_name: panels[panel_name]
        for panel_name in required_panel_names
    }


def _combined_panel_status(panels: dict[str, dict[str, Any]]) -> str:
    """Return one top-level status from child panel states."""

    statuses = [
        str(panel.get("status", "empty"))
        for panel in panels.values()
        if isinstance(panel, dict)
    ]
    has_success = any(
        status in {"available", "empty", "needs_input"}
        for status in statuses
    )
    if "partial" in statuses:
        status_value = "partial"
    elif "unavailable" in statuses and has_success:
        status_value = "partial"
    elif statuses and all(status == "unavailable" for status in statuses):
        status_value = "unavailable"
    elif "available" in statuses:
        status_value = "available"
    elif "needs_input" in statuses:
        status_value = "needs_input"
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


def _worker_activity_panel(
    worker_event_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    """Aggregate repetitive worker ticks into one row per worker."""

    rows = [
        row
        for row in worker_event_rows[:limit]
        if (
            isinstance(row, dict)
            and row.get("event_type") != "event_log.unavailable"
        )
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda item: str(item.get("created_at", "")),
    ):
        worker_name = str(row.get("worker_name", "")).strip()
        if not worker_name:
            worker_name = str(row.get("component", "background_work"))
        aggregate = grouped.setdefault(worker_name, {
            "worker_name": worker_name,
            "event_count": 0,
            "last_status": "",
            "last_created_at": "",
            "processed_count": 0,
            "succeeded_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "deferred_count": 0,
        })
        aggregate["event_count"] += 1
        for field in (
            "processed_count",
            "succeeded_count",
            "failed_count",
            "skipped_count",
        ):
            value = row.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                aggregate[field] += value
        if row.get("deferred") is True:
            aggregate["deferred_count"] += 1
        created_at = str(row.get("created_at", ""))
        if created_at >= str(aggregate["last_created_at"]):
            aggregate["last_created_at"] = created_at
            aggregate["last_status"] = str(row.get("status", ""))
            defer_reason = str(row.get("defer_reason", ""))
            if defer_reason:
                aggregate["defer_reason"] = defer_reason

    items = sorted(
        grouped.values(),
        key=lambda item: str(item["last_created_at"]),
        reverse=True,
    )[:limit]
    has_unavailable_sentinel = any(
        isinstance(row, dict)
        and row.get("event_type") == "event_log.unavailable"
        for row in worker_event_rows[:limit]
    )
    if has_unavailable_sentinel and items:
        status_value = "partial"
        reason = "some background worker telemetry is unavailable"
    elif has_unavailable_sentinel:
        status_value = "unavailable"
        reason = "background worker telemetry is unavailable"
    else:
        status_value = "available" if items else "empty"
        reason = "no background worker activity is available" if not items else ""
    panel = _entity_panel(
        status=status_value,
        items=items,
        reason=reason,
    )
    return panel


def _worker_error_panel(
    worker_event_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    """Return bounded worker failures separately from routine activity."""

    items = [
        _project_worker_event(row)
        for row in worker_event_rows
        if (
            isinstance(row, dict)
            and (
                row.get("status") in {"failed", "unavailable"}
                or row.get("level") == "error"
            )
        )
    ][:limit]
    panel = _entity_panel(
        status="available" if items else "empty",
        items=items,
        reason="no recent background worker errors" if not items else "",
    )
    return panel


def _panel_lookup_page(
    *,
    namespace: str,
    panels: dict[str, dict[str, Any]],
    required_panel_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a lookup response from panel envelopes."""

    status_panels = _required_status_panels(
        panels,
        required_panel_names=required_panel_names,
    )
    status_value = _combined_panel_status(status_panels)
    page = {
        "status": status_value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
        "next_cursor": None,
        "reason": "",
        "namespace": namespace,
        "panels": panels,
        "redaction": {
            "prompt_view": "production projections only",
            "raw_documents": "excluded",
            "internal_global_ids": "excluded",
            "dedupe_tokens": "excluded",
        },
    }
    return page


def _directory_page(
    *,
    status: str,
    items: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Build a bounded safe owner-directory response."""

    page = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [
            redact_mapping(item)
            for item in items
            if isinstance(item, dict)
        ],
        "next_cursor": None,
        "reason": str(reason)[:160],
        "redaction": _owner_entity_redaction(),
    }
    return page


def _audit_view_label(event: dict[str, Any]) -> str:
    """Return one human owner-page label for a recorded view."""

    event_type = str(event.get("event_type", "")).strip()
    if event_type == "lookup_view":
        target = event.get("target")
        if isinstance(target, dict):
            namespace = str(target.get("namespace", "")).strip()
            if namespace in AUDIT_LOOKUP_VIEW_LABELS:
                return AUDIT_LOOKUP_VIEW_LABELS[namespace]
            if namespace:
                return namespace.replace(".", " ").replace("_", " ").title()
    if event_type in AUDIT_VIEW_EVENT_LABELS:
        return AUDIT_VIEW_EVENT_LABELS[event_type]
    return event_type.replace("_", " ").strip().title() or "Console view"


def _project_audit_action(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Project one exact-request audit group into a human action."""

    ordered_events = sorted(
        events,
        key=lambda event: str(event.get("created_at", "")),
    )
    latest_event = ordered_events[-1]
    event_types = [
        str(event.get("event_type", ""))
        for event in ordered_events
    ]
    action_name = _audit_action_name(event_types)
    outcome = _audit_action_outcome(ordered_events)
    service_id = next(
        (
            str(event.get("service_id", "")).strip()
            for event in reversed(ordered_events)
            if str(event.get("service_id", "")).strip()
        ),
        "",
    )
    target = latest_event.get("target")
    if not isinstance(target, dict):
        target = {}
    target_label = _audit_target_label(
        service_id=service_id,
        target=target,
    )
    reason = next(
        (
            str(event.get("reason", "")).strip()
            for event in reversed(ordered_events)
            if str(event.get("reason", "")).strip()
        ),
        "",
    )
    action = {
        "request_id": str(latest_event.get("request_id", "")),
        "action": action_name,
        "category": action_name.split(" ", maxsplit=1)[0],
        "target_label": target_label,
        "outcome": outcome,
        "event_count": len(ordered_events),
        "operator_id": str(latest_event.get("operator_id", "")),
        "service_id": service_id,
        "created_at": str(latest_event.get("created_at", "")),
        "reason": reason,
        "event_types": event_types,
    }
    projected_action = redact_mapping(action)
    return projected_action


def _audit_action_name(event_types: list[str]) -> str:
    """Map audit event families to stable human action labels."""

    for prefix, label in AUDIT_ACTION_PREFIXES:
        if any(event_type.startswith(prefix) for event_type in event_types):
            return label
    if event_types:
        label = event_types[-1].replace("_", " ").strip()
        return label
    return "operator action"


def _audit_action_outcome(events: list[dict[str, Any]]) -> str:
    """Derive action outcome from explicit terminal events and new state."""

    event_types = {
        str(event.get("event_type", ""))
        for event in events
    }
    if event_types & AUDIT_FAILURE_EVENTS:
        return "failed"
    if event_types & AUDIT_SUCCESS_EVENTS:
        return "succeeded"
    if any(
        event_type.endswith("_failed")
        for event_type in event_types
    ):
        return "failed"
    if any(
        isinstance(event.get("new_state"), dict)
        and bool(event["new_state"])
        for event in events
    ):
        return "succeeded"
    if any(
        event_type.endswith("_requested")
        for event_type in event_types
    ):
        return "requested"
    return "recorded"


def _audit_target_label(*, service_id: str, target: dict[str, Any]) -> str:
    """Humanize whitelisted target fields without dumping an object."""

    if service_id:
        return service_id
    for field in ("service_id", "route_key", "namespace", "scope"):
        value = str(target.get(field, "")).strip()
        if value:
            return value
    return "control console"


def _audit_action_matches(
    action: dict[str, Any],
    *,
    category: str,
    event_type: str,
    service_id: str,
    operator_id: str,
    outcome: str,
    request_id: str,
    since: str,
) -> bool:
    """Apply bounded exact audit filters to one collapsed action."""

    if category and action.get("category") != category:
        return False
    event_types = action.get("event_types")
    if (
        event_type
        and (
            not isinstance(event_types, list)
            or event_type not in event_types
        )
    ):
        return False
    if service_id and action.get("service_id") != service_id:
        return False
    if operator_id and action.get("operator_id") != operator_id:
        return False
    if outcome and action.get("outcome") != outcome:
        return False
    if request_id and action.get("request_id") != request_id:
        return False
    if since and str(action.get("created_at", "")) < since:
        return False
    return True


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


def _character_id_from_profile(profile: dict[str, Any]) -> str:
    """Return the active character id with the service-compatible fallback."""

    character_id = str(profile.get("global_user_id", "")).strip()
    if not character_id:
        character_id = "00000000-0000-4000-8000-000000000001"
    return character_id


def _project_calendar_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project one Recall calendar candidate into browser-safe fields."""

    allowed_fields = (
        "source",
        "claim",
        "temporal_scope",
        "lifecycle_status",
        "evidence_time",
        "authority",
    )
    row = {
        field: candidate[field]
        for field in allowed_fields
        if field in candidate and candidate[field] not in (None, "")
    }
    projected_row = redact_mapping(row)
    return projected_row


def _project_calendar_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    """Project one schedule definition without source ids or payload internals."""

    source_scope = schedule.get("source_scope")
    if not isinstance(source_scope, dict):
        source_scope = {}
    row = {
        "trigger_kind": schedule.get("trigger_kind", ""),
        "status": schedule.get("status", ""),
        "start_at": schedule.get("start_at", ""),
        "next_run_at": schedule.get("next_run_at", ""),
        "source_platform": source_scope.get("source_platform", ""),
        "source_channel_type": source_scope.get("source_channel_type", ""),
        "recurrence": schedule.get("recurrence", {}),
        "timezone": schedule.get("timezone", ""),
        "updated_at": schedule.get("updated_at", ""),
    }
    projected_row = redact_mapping({
        key: value
        for key, value in row.items()
        if value not in (None, "", {})
    })
    return projected_row


def _project_background_job(job: dict[str, Any]) -> dict[str, Any]:
    """Project one background-work job without task payload internals."""

    allowed_fields = (
        "status",
        "delivery_state",
        "worker",
        "created_at",
        "updated_at",
        "completed_at",
        "delivery_attempt_count",
        "result_summary",
        "failure_summary",
        "artifact_char_count",
        "source_platform",
        "source_channel_type",
        "requester_display_name",
    )
    row = {
        field: job[field]
        for field in allowed_fields
        if field in job and job[field] not in (None, "")
    }
    if (
        row.get("result_summary")
        and row.get("failure_summary") == row.get("result_summary")
    ):
        row.pop("failure_summary", None)
    projected_row = redact_mapping(row)
    return projected_row


def _project_global_growth_run(row: dict[str, Any]) -> dict[str, Any]:
    """Project semantic growth outcomes without run or source identities."""

    allowed_fields = (
        "status",
        "completed_at",
        "summary",
        "accepted_candidates",
        "trait_updates",
        "shadow_projection",
        "failure_summary",
    )
    projected: dict[str, Any] = {
        "kind": "recent_outcome",
        **{
            field: row[field]
            for field in allowed_fields
            if field in row and row[field] not in (None, "")
        },
    }
    projected_row = redact_mapping(_project_safe_state_value(projected))
    shadow_projection = row.get("shadow_projection")
    if isinstance(shadow_projection, dict):
        prompt_visible_now = shadow_projection.get("prompt_visible_now")
        projected_shadow = projected_row.get("shadow_projection")
        if (
            isinstance(prompt_visible_now, bool)
            and isinstance(projected_shadow, dict)
        ):
            projected_shadow["prompt_visible_now"] = prompt_visible_now
    return projected_row


def _project_growth_trait(row: dict[str, Any]) -> dict[str, Any]:
    """Project one active growth trait into operator-meaningful fields."""

    allowed_fields = (
        "growth_axis",
        "trait_name",
        "guidance",
        "strength",
        "status",
        "maturity_band",
        "evidence_count",
        "first_observed_date",
        "last_observed_date",
        "updated_at",
    )
    projected = {
        "kind": "active_trait",
        **{
            field: row[field]
            for field in allowed_fields
            if field in row and row[field] not in (None, "")
        },
    }
    projected_row = redact_mapping(projected)
    return projected_row


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
        if "updated_at" not in row and last_updated not in (None, ""):
            row["updated_at"] = last_updated

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


def _project_safe_state_value(value: Any) -> Any:
    """Remove private handles and identifiers from nested semantic state."""

    if isinstance(value, dict):
        projected: dict[str, Any] = {}
        for key, nested_value in value.items():
            normalized_key = str(key)
            if (
                normalized_key in PRIVATE_STATE_KEYS
                or normalized_key.endswith("_id")
                or normalized_key.endswith("_ids")
                or normalized_key.endswith("_refs")
            ):
                continue
            projected[normalized_key] = _project_safe_state_value(nested_value)
        return projected
    if isinstance(value, list):
        return [
            _project_safe_state_value(item)
            for item in value[:50]
        ]
    return value


def _project_cognition_state_items(
    value: Any,
    *,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Project selected native cognition fields into safe key/value rows."""

    if not isinstance(value, dict):
        items: list[dict[str, Any]] = []
        return items

    items = [
        redact_mapping({
            "key": field,
            "value": _project_safe_state_value(value[field]),
        })
        for field in fields
        if field in value and value[field] not in (None, "")
    ]
    return items


def _project_user_profile(
    profile: dict[str, Any],
    *,
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project a user profile without internal canonical identifiers."""

    accounts = profile.get("platform_accounts")
    if not isinstance(accounts, list):
        accounts = []
    aliases = profile.get("suspected_aliases")
    if not isinstance(aliases, list):
        aliases = []
    cognition_state = profile.get("cognition_state")
    if not isinstance(cognition_state, dict):
        cognition_state = {}
    row: dict[str, Any] = {
        "accounts": [
            {
                "platform": str(account.get("platform", "")),
                "platform_user_id": str(account.get("platform_user_id", "")),
                "display_name": str(account.get("display_name", "")),
            }
            for account in accounts
            if isinstance(account, dict)
        ],
        "account_count": len(accounts),
        "alias_count": len(aliases),
    }
    updated_at = cognition_state.get("updated_at")
    if updated_at:
        row["updated_at"] = updated_at
    for field in ("platform", "platform_user_id", "display_name"):
        value = identity.get(field)
        if value not in (None, ""):
            row[field] = value

    items = [redact_mapping(row)] if row else []
    return items


def _project_relationship_panel(cognition_state: Any) -> dict[str, Any]:
    """Project exact native relationship axes with canonical semantic bands."""

    if not isinstance(cognition_state, dict):
        panel = _entity_panel(
            status="empty",
            items=[],
            reason="user cognition state is not available",
        )
        return panel
    relationship = cognition_state.get("relationship")
    if not isinstance(relationship, dict):
        panel = _entity_panel(
            status="empty",
            items=[],
            reason="native V2 relationship state is not available",
        )
        return panel

    items: list[dict[str, Any]] = []
    for axis in RELATIONSHIP_AXES:
        value = relationship.get(axis)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        items.append({
            "axis": axis,
            "value": value,
            "band": project_numeric_band(
                value,
                signed=axis in SIGNED_RELATIONSHIP_AXES,
            ),
        })
    evidence_refs = relationship.get("evidence_refs")
    evidence_count = len(evidence_refs) if isinstance(evidence_refs, list) else 0
    panel = _entity_panel(
        status="available" if items else "empty",
        items=items,
        reason="native V2 relationship axes are empty" if not items else "",
    )
    panel["evidence_count"] = evidence_count
    panel["updated_at"] = str(relationship.get("updated_at", ""))
    return panel


def _project_character_operational_posture(
    cognition_state: Any,
    *,
    effective_at: str,
    latest_context_consumption: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project persisted/effective posture and exact latest consumption.

    The console-supplied effective timestamp may carry a UTC +00:00 suffix;
    it is converted to the terminal Z form required by the native parser.
    """

    if not isinstance(cognition_state, Mapping):
        return _entity_panel(
            status="empty",
            items=[],
            reason="native character operational state is not available",
        )
    if cognition_state.get("state_scope") != "character":
        return _entity_panel(
            status="empty",
            items=[],
            reason="runtime cognition state is not character-scoped",
        )
    source_updated_at = cognition_state.get("updated_at")
    if not isinstance(source_updated_at, str) or not source_updated_at.strip():
        return _entity_panel(
            status="unavailable",
            items=[],
            reason="character operational state version is unavailable",
        )
    native_effective_at = (
        f"{effective_at[:-6]}Z"
        if effective_at.endswith("+00:00")
        else effective_at
    )
    try:
        persisted_view = project_character_operational_state(
            cognition_state,
            effective_at=source_updated_at,
        )
        effective_view = project_character_operational_state(
            cognition_state,
            effective_at=native_effective_at,
        )
    except (KeyError, ValueError):
        return _entity_panel(
            status="unavailable",
            items=[],
            reason="character operational state projection is unavailable",
        )
    persisted = redact_character_operational_state_view(persisted_view)
    effective = redact_character_operational_state_view(effective_view)
    if not persisted or not effective:
        return _entity_panel(
            status="unavailable",
            items=[],
            reason="character operational state projection was redacted",
        )
    latest_source = (
        latest_context_consumption
        if isinstance(latest_context_consumption, Mapping)
        else {"status": "not_reported"}
    )
    latest = redact_latest_context_consumption(latest_source)
    item = {
        "persisted": persisted,
        "effective": effective,
        "fading_changed": (
            persisted["affect"] != effective["affect"]
            or persisted["pressures"] != effective["pressures"]
        ),
        "latest_context": latest,
    }
    return _entity_panel(status="available", items=[item], reason="")


def _project_relationship_operational_panel(
    cognition_state: Any,
    *,
    effective_at: str,
) -> dict[str, Any]:
    """Project current-user causal relationship context without identifiers.

    The console-supplied effective timestamp may carry a UTC +00:00 suffix;
    it is converted to the terminal Z form required by the native parser.
    """

    if not isinstance(cognition_state, Mapping):
        return _entity_panel(
            status="empty",
            items=[],
            reason="user cognition state is not available",
        )
    if cognition_state.get("state_scope") != "user":
        return _entity_panel(
            status="empty",
            items=[],
            reason="native V2 relationship state is not user-scoped",
        )
    native_effective_at = (
        f"{effective_at[:-6]}Z"
        if effective_at.endswith("+00:00")
        else effective_at
    )
    try:
        relationship_context = project_relationship_context(
            cognition_state,
            effective_at=native_effective_at,
        )
        public_context = project_operational_relationship_context(
            relationship_context,
        )
    except (KeyError, ValueError):
        return _entity_panel(
            status="unavailable",
            items=[],
            reason="native V2 relationship projection is unavailable",
        )
    projected = redact_operational_relationship_context(public_context)
    if not projected:
        return _entity_panel(
            status="unavailable",
            items=[],
            reason="native V2 relationship projection was redacted",
        )
    return _entity_panel(status="available", items=[projected], reason="")


def _project_user_directory_item(profile: dict[str, Any]) -> dict[str, Any]:
    """Project one user directory row without canonical or alias identities."""

    accounts = profile.get("accounts")
    if not isinstance(accounts, list):
        accounts = profile.get("platform_accounts")
    if not isinstance(accounts, list):
        accounts = []
    safe_accounts = [
        {
            "platform": str(account.get("platform", "")),
            "platform_user_id": str(account.get("platform_user_id", "")),
            "display_name": str(account.get("display_name", "")),
        }
        for account in accounts
        if isinstance(account, dict)
    ]
    cognition_state = profile.get("cognition_state")
    if not isinstance(cognition_state, dict):
        cognition_state = {}
    aliases = profile.get("suspected_aliases")
    alias_count = profile.get("alias_count")
    if not isinstance(alias_count, int):
        alias_count = len(aliases) if isinstance(aliases, list) else 0
    updated_at = profile.get("updated_at") or cognition_state.get("updated_at", "")
    display_name = ""
    if safe_accounts:
        display_name = str(safe_accounts[0].get("display_name", ""))
    row = {
        "display_name": display_name,
        "accounts": safe_accounts,
        "account_count": len(safe_accounts),
        "alias_count": alias_count,
        "updated_at": str(updated_at),
    }
    return row


def _project_group_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Project one conversation-owned group activity summary."""

    channel_name_value = summary.get("channel_name")
    channel_name = (
        str(channel_name_value).strip()
        if channel_name_value is not None
        else ""
    )
    row = {
        "platform": str(summary.get("platform", "")),
        "group_id": str(summary.get("platform_channel_id", "")),
        "channel_name": channel_name,
        "last_activity_at": str(summary.get("last_activity_at", "")),
        "message_count": int(summary.get("message_count", 0)),
        "participant_count": int(summary.get("participant_count", 0)),
    }
    projected_row = redact_mapping(row)
    return projected_row


def _project_group_review(review: dict[str, Any]) -> dict[str, Any]:
    """Project one group-review ledger row without internal case identities."""

    allowed_fields = (
        "window_start",
        "window_end",
        "status",
        "reviewed_at",
        "skip_reason",
    )
    row = {
        field: review[field]
        for field in allowed_fields
        if field in review and review[field] not in (None, "")
    }
    projected_row = redact_mapping(row)
    return projected_row


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
    context: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Project role-labelled or legacy style context into redacted rows."""

    if context.get("schema_version") == "interaction_style_turn_snapshot.v1":
        return _project_role_labelled_interaction_style_context(
            context,
            limit=limit,
        )

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


def _project_role_labelled_interaction_style_context(
    context: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Expose the exact relevance, cognition, and surface projections."""

    rows: list[dict[str, Any]] = []
    for consumer_role in ("relevance", "cognition", "surface"):
        role_projection = context.get(consumer_role)
        if not isinstance(role_projection, Mapping):
            continue
        for source_name in ("user", "group_channel"):
            source_projection = role_projection.get(source_name)
            if not isinstance(source_projection, Mapping):
                continue
            row = _project_interaction_style_role_row(
                consumer_role=consumer_role,
                source_name=source_name,
                source_projection=source_projection,
            )
            if not row:
                continue
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def _project_interaction_style_role_row(
    *,
    consumer_role: str,
    source_name: str,
    source_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one declared source-role pair without image provenance."""

    status = source_projection.get("status")
    if not isinstance(status, str) or status not in {
        "active",
        "empty",
        "missing",
        "failed",
    }:
        return {}
    revision = source_projection.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int):
        return {}
    confidence = ""
    guidance_source: Mapping[str, Any] = source_projection
    if consumer_role == "surface":
        overlay = source_projection.get("overlay")
        if not isinstance(overlay, Mapping):
            return {}
        guidance_source = overlay
    confidence_value = guidance_source.get("confidence")
    if isinstance(confidence_value, str):
        confidence = confidence_value
    fields = (
        ("engagement_guidelines",)
        if consumer_role == "relevance"
        else STYLE_GUIDELINE_FIELDS
        if consumer_role == "surface"
        else ("social_guidelines", "engagement_guidelines")
    )
    guidance = {
        field_name: [item for item in field_value[:8] if isinstance(item, str)]
        for field_name in fields
        if isinstance((field_value := guidance_source.get(field_name)), list)
    }
    row = redact_mapping({
        "consumer_role": consumer_role,
        "source": source_name,
        "status": status,
        "revision": revision,
        "confidence": confidence,
        "guidance": guidance,
    })
    return row


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
        "trigger_kind",
        "status",
        "due_at",
        "completed_at",
        "failed_at",
        "skipped_at",
        "updated_at",
    )
    row = {
        field: document[field]
        for field in allowed_fields
        if field in document and document[field] not in (None, "")
    }
    result_summary = document.get("result_summary")
    if isinstance(result_summary, dict):
        projected_summary = _project_run_summary(result_summary)
        if projected_summary:
            row["result_summary"] = projected_summary
    failure_summary = document.get("failure_summary")
    if isinstance(failure_summary, dict):
        projected_summary = _project_run_summary(failure_summary)
        if projected_summary:
            row["failure_summary"] = projected_summary
    projected_row = redact_mapping(row)
    return projected_row


def _project_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep nonduplicated semantic run outcomes and nonzero counters."""

    allowed_fields = (
        "processed_count",
        "succeeded_count",
        "failed_count",
        "skipped_count",
        "deferred",
        "defer_reason",
        "reason",
        "retryable",
        "error",
    )
    projected: dict[str, Any] = {}
    for field in allowed_fields:
        if field not in summary:
            continue
        value = summary[field]
        if value in (None, ""):
            continue
        if field.endswith("_count") and value == 0:
            continue
        if field == "deferred" and value is False:
            continue
        projected[field] = value
    projected_row = redact_mapping(projected)
    return projected_row


def _project_worker_event(row: dict[str, Any]) -> dict[str, Any]:
    """Project one worker error without correlation or run identifiers."""

    allowed_fields = (
        "event_type",
        "component",
        "level",
        "status",
        "created_at",
        "error_class",
        "message",
        *SAFE_WORKER_EVENT_FIELDS,
    )
    projected = {
        field: row[field]
        for field in allowed_fields
        if field in row and row[field] not in (None, "")
    }
    projected_row = redact_mapping(projected)
    return projected_row
