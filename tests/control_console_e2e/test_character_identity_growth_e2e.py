"""Guarded browser proof for character identity growth observability."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import pytest
from pymongo import MongoClient

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.character_profile import (
    load_character_profile_seed,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db._client import (
    IDENTITY_GROWTH_DATABASE_GUARD_ENV,
    IDENTITY_GROWTH_TEST_DATABASE_ENV,
    close_db,
    get_db,
)
from kazusa_ai_chatbot.db.character import (
    ensure_operational_character_state,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    CANDIDATE_SCHEMA_VERSION,
    GROWTH_COLLECTION_NAMES,
    RUN_SCHEMA_VERSION,
    claim_identity_revision_consumption,
    complete_growth_run_post_commit,
    ensure_character_identity_growth_indexes,
    ensure_seed_identity,
    insert_growth_candidate,
    insert_growth_run,
    promote_ready_candidate,
)


_RUN_FLAG = "KAZUSA_RUN_IDENTITY_GROWTH_CONSOLE_E2E"
_ARTIFACT_DIRECTORY = Path("test_artifacts/character_identity_growth")
_CHARACTER_ROUTE = "**/api/entities/character*"
_SELF_CONCEPT_MARKER = "Browser proof current self-concept."
_GROWTH_EDGE_MARKER = "Browser proof current growth edge."
_INTERNAL_VALUE_MARKERS = (
    "browser-candidate-",
    "browser-correlation-",
    "browser-evidence-",
    "browser-root-",
    "browser-run-",
)
_FORBIDDEN_PUBLIC_FIELDS = (
    "candidate_id",
    "character_id",
    "correlation_id",
    "effective_identity",
    "evidence_ref_id",
    "new_value",
    "old_value",
    "promotion_run_id",
    "raw_output",
    "replacement_text",
    "root_episode_id",
)
_HEALTH_REASONS = {
    "healthy_idle": "not_routed",
    "waiting_for_evidence": "candidate_emerging",
    "semantic_rejection": "review_rejected",
    "promotion_ready": "candidate_ready",
    "awaiting_consumption": "awaiting_first_consumption",
    "healthy_active": "revision_consumed",
    "pipeline_error": "proposal_contract_failed",
    "consumption_error": "revision_consumption_mismatch",
}


pytestmark = [
    pytest.mark.live_db,
    pytest.mark.skipif(
        os.environ.get(_RUN_FLAG) != "1",
        reason="identity-growth console E2E is an explicit live-DB gate",
    ),
]


@pytest.fixture
def clean_identity_growth_console_data():
    """Create and remove one validated identity-growth browser dataset."""

    _require_guarded_database()
    try:
        asyncio.run(_prepare_identity_growth_data())
        yield
    finally:
        _drop_identity_growth_data_sync()


def test_character_identity_growth_console_protocol(
    e2e_console,
    e2e_browser_page,
    clean_identity_growth_console_data,
) -> None:
    """Exercise lineage, health, pace, redaction, and responsive states."""

    del clean_identity_growth_console_data
    _ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    network_record: dict[str, Any] = {
        "browser_runner": (
            "existing Playwright/Chrome harness; in-app Browser unavailable"
        ),
        "database": os.environ[IDENTITY_GROWTH_TEST_DATABASE_ENV],
        "health_states_rendered": [],
    }

    with e2e_console(use_live_project_db=True) as console:
        page = e2e_browser_page(console.base_url)
        _login(page)

        character_payload = _open_character_and_capture(page)
        _assert_redacted_character_network(character_payload)
        network_record["character"] = _character_network_summary(
            character_payload
        )
        _assert_character_identity_render(page)
        _assert_revision_keyboard_access(page)

        for state, reason in _HEALTH_REASONS.items():
            _render_health_state(
                page,
                source_payload=character_payload,
                state=state,
                reason=reason,
            )
            network_record["health_states_rendered"].append(state)

        _verify_loading_empty_and_error_states(
            page,
            source_payload=character_payload,
        )
        _restore_character_payload(page, character_payload)

        scope_record = _verify_scoped_residue_pages(page)
        network_record["scoped_residue"] = scope_record

        config_record = _verify_pace_config_and_restart(page)
        network_record["pace_config"] = config_record

        _open_page(page, "character", "Character")
        page.wait_for_function(
            """() => (
              document.querySelector('#character-self-image-table')
              ?.textContent?.includes('Browser proof current self-concept.')
            )"""
        )
        desktop_path = (
            _ARTIFACT_DIRECTORY
            / "checkpoint_g_character_identity_desktop.png"
        )
        page.set_viewport_size({"width": 1600, "height": 1000})
        page.screenshot(path=str(desktop_path), full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(100)
        assert page.evaluate(
            "() => document.documentElement.scrollWidth "
            "<= document.documentElement.clientWidth + 1"
        )
        narrow_path = (
            _ARTIFACT_DIRECTORY
            / "checkpoint_g_character_identity_narrow.png"
        )
        page.screenshot(path=str(narrow_path), full_page=True)

        console_messages = list(
            getattr(page, "kazusa_console_messages", [])
        )
        assert console_messages == []
        network_record["screenshots"] = {
            "desktop": str(desktop_path),
            "narrow": str(narrow_path),
        }
        network_record["browser_console_messages"] = console_messages

    network_path = (
        _ARTIFACT_DIRECTORY / "checkpoint_g_console_network_redaction.json"
    )
    network_path.write_text(
        json.dumps(network_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    assert network_path.exists()


async def _prepare_identity_growth_data() -> None:
    """Seed current/prior revisions and every candidate/run display state."""

    await _drop_identity_growth_data(close_client=False)
    await ensure_character_identity_growth_indexes()
    await ensure_operational_character_state()
    _repository_root = Path(__file__).resolve().parents[2]
    seed = load_character_profile_seed(
        _repository_root / "personalities" / "example.json",
    )
    await ensure_seed_identity(
        character_id=CHARACTER_GLOBAL_USER_ID,
        seed=seed,
    )

    promotion_changes = _promotion_changes(seed)
    promotion_candidate = _candidate(
        candidate_id="browser-candidate-promoted",
        base_revision_number=0,
        status="ready",
        changes=promotion_changes,
        root_numbers=(1, 2, 3),
        scope_kinds=("private", "group", "private"),
        updated_at="2026-07-28T10:03:00+00:00",
    )
    await insert_growth_candidate(promotion_candidate)
    await insert_growth_run(
        _run(
            run_id="browser-run-promoted",
            base_revision_number=0,
            disposition="candidate_updated",
            lifecycle_state="in_progress",
            reason_code="candidate_ready",
            candidate_id="browser-candidate-promoted",
            root_numbers=(1, 2, 3),
            started_at="2026-07-28T10:04:00+00:00",
            completed_at=None,
        )
    )
    revision = await promote_ready_candidate(
        character_id=CHARACTER_GLOBAL_USER_ID,
        candidate_id="browser-candidate-promoted",
        run_id="browser-run-promoted",
        now=datetime(2026, 7, 28, 10, 5, tzinfo=timezone.utc),
    )
    await complete_growth_run_post_commit(
        run_id="browser-run-promoted",
        character_id=CHARACTER_GLOBAL_USER_ID,
        revision_number=1,
    )
    cognition = project_identity_for_cognition(revision)
    surface = project_identity_for_surface(revision)
    consumers = projected_identity_consumer_kinds(cognition)
    digest = identity_projection_digest(
        revision_number=1,
        cognition_context=cognition,
        surface_context=surface,
    )
    await claim_identity_revision_consumption(
        character_id=CHARACTER_GLOBAL_USER_ID,
        episode_id="browser-consumption-episode",
        correlation_id="browser-consumption-correlation",
        loaded_revision_number=1,
        consumer_kinds=consumers,
        projection_digest=digest,
    )

    current_candidates = (
        _candidate(
            candidate_id="browser-candidate-emerging",
            base_revision_number=1,
            status="emerging",
            changes=[_text_patch("description", "Browser proof candidate.")],
            root_numbers=(10,),
            scope_kinds=("private",),
            updated_at="2026-07-28T10:10:00+00:00",
        ),
        _candidate(
            candidate_id="browser-candidate-ready",
            base_revision_number=1,
            status="ready",
            changes=[_text_patch("backstory", "Browser proof candidate.")],
            root_numbers=(20, 21, 22),
            scope_kinds=("group", "private", "group"),
            updated_at="2026-07-28T10:20:00+00:00",
        ),
        _candidate(
            candidate_id="browser-candidate-rejected",
            base_revision_number=1,
            status="rejected",
            changes=[
                _text_patch(
                    "visual_characterization",
                    "Browser proof candidate.",
                )
            ],
            root_numbers=(30,),
            scope_kinds=("group",),
            updated_at="2026-07-28T10:30:00+00:00",
            rejection_reason="review_rejected",
        ),
    )
    for candidate in current_candidates:
        await insert_growth_candidate(candidate)

    recent_runs = (
        _run(
            run_id="browser-run-no-change",
            base_revision_number=1,
            disposition="no_change",
            lifecycle_state="complete",
            reason_code="proposal_no_change",
            candidate_id=None,
            root_numbers=(40,),
            started_at="2026-07-28T10:40:00+00:00",
            completed_at="2026-07-28T10:40:10+00:00",
        ),
        _run(
            run_id="browser-run-rejected",
            base_revision_number=1,
            disposition="rejected",
            lifecycle_state="complete",
            reason_code="review_rejected",
            candidate_id="browser-candidate-rejected",
            root_numbers=(41,),
            started_at="2026-07-28T10:41:00+00:00",
            completed_at="2026-07-28T10:41:10+00:00",
        ),
        _run(
            run_id="browser-run-failed",
            base_revision_number=1,
            disposition="failed",
            lifecycle_state="failed",
            reason_code="proposal_contract_failed",
            candidate_id=None,
            root_numbers=(42,),
            started_at="2026-07-28T10:42:00+00:00",
            completed_at="2026-07-28T10:42:10+00:00",
            validation_errors=("proposal_contract_failed",),
        ),
    )
    for run in recent_runs:
        await insert_growth_run(run)
    await close_db()


async def _drop_identity_growth_data(
    *,
    close_client: bool = True,
) -> None:
    """Remove only test-owned identity ledgers and operational state."""

    database = await get_db()
    for collection_name in (*GROWTH_COLLECTION_NAMES, "character_state"):
        await database.drop_collection(collection_name)
    if close_client:
        await close_db()


def _drop_identity_growth_data_sync() -> None:
    """Synchronously remove the exact test-owned collections at teardown."""

    _require_guarded_database()
    client = MongoClient(os.environ["MONGODB_URI"])
    try:
        database = client[os.environ["MONGODB_DB_NAME"]]
        for collection_name in (*GROWTH_COLLECTION_NAMES, "character_state"):
            database.drop_collection(collection_name)
    finally:
        client.close()


def _promotion_changes(
    seed: dict[str, object],
) -> list[dict[str, object]]:
    """Build one change for every tagged diff category."""

    age = int(seed["age"])
    replacement_age = age + 1 if age < 199 else age - 1
    boundary = dict(seed["boundary_profile"])
    integrity = float(boundary["self_integrity"])
    replacement_band = "very_low" if integrity >= 0.5 else "very_high"
    strategy = str(boundary["compliance_strategy"])
    replacement_strategy = next(
        value
        for value in sorted(
            models.ENUM_VALUES_BY_PATH[
                "boundary_profile.compliance_strategy"
            ]
        )
        if value != strategy
    )
    return [
        _text_patch("self_image.self_concept", _SELF_CONCEPT_MARKER),
        {
            "path": "age",
            "value_kind": "integer",
            "replacement_integer": replacement_age,
        },
        {
            "path": "boundary_profile.self_integrity",
            "value_kind": "semantic_band",
            "replacement_band": replacement_band,
        },
        {
            "path": "boundary_profile.compliance_strategy",
            "value_kind": "closed_enum",
            "replacement_enum": replacement_strategy,
        },
        {
            "path": "self_image.current_growth_edges",
            "value_kind": "text_list",
            "replacement_items": [_GROWTH_EDGE_MARKER],
        },
    ]


def _text_patch(path: str, value: str) -> dict[str, object]:
    """Build one strict text identity patch."""

    return {
        "path": path,
        "value_kind": "text",
        "replacement_text": value,
    }


def _evidence_ref(
    number: int,
    *,
    scope_kind: str,
) -> dict[str, object]:
    """Build one generic repository-owned root reference."""

    day = 27 if number % 3 else 28
    return {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": f"browser-evidence-{number}",
        "root_episode_id": f"browser-root-{number}",
        "correlation_id": f"browser-correlation-{number}",
        "source_kind": "settled_episode",
        "derived_reflection_run_ids": [],
        "character_local_date": f"2026-07-{day:02d}",
        "scope_kind": scope_kind,
        "captured_at": f"2026-07-{day:02d}T10:{number % 60:02d}:00+00:00",
    }


def _candidate(
    *,
    candidate_id: str,
    base_revision_number: int,
    status: str,
    changes: list[dict[str, object]],
    root_numbers: tuple[int, ...],
    scope_kinds: tuple[str, ...],
    updated_at: str,
    rejection_reason: str | None = None,
) -> dict[str, object]:
    """Build one validated candidate for browser-safe projection."""

    evidence_refs = [
        _evidence_ref(number, scope_kind=scope_kind)
        for number, scope_kind in zip(root_numbers, scope_kinds, strict=True)
    ]
    roots = sorted({
        str(row["root_episode_id"])
        for row in evidence_refs
    })
    local_dates = sorted({
        str(row["character_local_date"])
        for row in evidence_refs
    })
    scopes = sorted({
        str(row["scope_kind"])
        for row in evidence_refs
    })
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "character_id": CHARACTER_GLOBAL_USER_ID,
        "base_revision_number": base_revision_number,
        "status": status,
        "change_kind": "inferred_growth",
        "proposed_changes": changes,
        "semantic_summary": "Browser proof reviewed identity change.",
        "evidence_refs": evidence_refs,
        "distinct_episode_count": len(roots),
        "distinct_local_dates": local_dates,
        "source_scope_kinds": scopes,
        "claimed_root_episode_ids": roots,
        "newest_root_captured_at": max(
            str(row["captured_at"])
            for row in evidence_refs
        ),
        "reversal_of_paths": [],
        "fresh_post_revision_root_count": 0,
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
        "privacy_review": "low",
        "promoted_revision_number": None,
        "rejection_reason": rejection_reason,
        "created_at": updated_at,
        "updated_at": updated_at,
    }


def _run(
    *,
    run_id: str,
    base_revision_number: int,
    disposition: str,
    lifecycle_state: str,
    reason_code: str,
    candidate_id: str | None,
    root_numbers: tuple[int, ...],
    started_at: str,
    completed_at: str | None,
    validation_errors: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build one sanitized growth run for browser-safe projection."""

    roots = sorted(f"browser-root-{number}" for number in root_numbers)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": "episode",
        "character_id": CHARACTER_GLOBAL_USER_ID,
        "base_revision_number": base_revision_number,
        "correlation_id": f"browser-correlation-{run_id}",
        "root_episode_ids": roots,
        "source_evidence_count": len(roots),
        "attempt_count_by_stage": {"proposal": 1, "review": 1},
        "lifecycle_state": lifecycle_state,
        "disposition": disposition,
        "proposal_reason_code": reason_code,
        "review_reason_code": reason_code,
        "policy_reason_code": reason_code,
        "persistence_reason_code": reason_code,
        "candidate_id": candidate_id,
        "promoted_revision_number": None,
        "validation_error_codes": sorted(validation_errors),
        "first_consumption": None,
        "post_commit_attempt_count": 0,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _require_guarded_database() -> None:
    """Require the exact explicitly authorized test database."""

    database_name = os.environ.get(
        IDENTITY_GROWTH_TEST_DATABASE_ENV,
        "",
    ).strip()
    if os.environ.get(IDENTITY_GROWTH_DATABASE_GUARD_ENV) != "1":
        raise AssertionError("identity-growth database guard is required")
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        raise AssertionError("live test database guard is required")
    if not database_name:
        raise AssertionError("identity-growth test database is required")
    if os.environ.get("MONGODB_DB_NAME") != database_name:
        raise AssertionError("MongoDB name must match the guarded database")


def _login(page) -> None:
    """Authenticate the browser as the isolated E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("body[data-auth-state='authenticated']")


def _open_page(page, page_name: str, expected_heading: str) -> None:
    """Open one page and verify its visible heading."""

    page.locator(f"[data-page-link='{page_name}']").click()
    active_page = page.locator(f"[data-page='{page_name}']")
    active_page.wait_for(state="visible")
    assert active_page.locator("h2").first.inner_text() == expected_heading


def _open_character_and_capture(page) -> dict[str, Any]:
    """Open Character and return its authenticated real-DB payload."""

    with page.expect_response(
        lambda response: "/api/entities/character" in response.url
    ) as response_info:
        _open_page(page, "character", "Character")
    payload = response_info.value.json()
    assert payload["owner"] == "character"
    assert payload["status"] == "available"
    return payload


def _assert_character_identity_render(page) -> None:
    """Verify current identity, history, diffs, candidates, and reasons."""

    self_image_text = page.locator(
        "#character-self-image-table"
    ).inner_text()
    assert _SELF_CONCEPT_MARKER in self_image_text
    assert _GROWTH_EDGE_MARKER in self_image_text

    revision_cards = page.locator(
        "#character-carry-over-table details.identity-revision-card"
    )
    assert revision_cards.count() == 2
    lineage_text = page.locator(
        "#character-carry-over-table"
    ).inner_text().lower()
    assert "current" in lineage_text
    assert "revision 0" in lineage_text
    for value_kind in (
        "text",
        "integer",
        "semantic band",
        "closed enum",
        "text list",
    ):
        assert value_kind in lineage_text

    growth_text = page.locator(
        "#character-growth-table"
    ).inner_text().lower()
    for candidate_state in ("promoted", "emerging", "ready", "rejected"):
        assert candidate_state in growth_text
    for reason in (
        "proposal no change",
        "review rejected",
        "proposal contract failed",
    ):
        assert reason in growth_text
    assert "roots" in growth_text
    assert "local dates" in growth_text


def _assert_revision_keyboard_access(page) -> None:
    """Toggle the older revision through its native keyboard control."""

    older_revision = page.locator(
        "#character-carry-over-table details.identity-revision-card"
    ).nth(1)
    assert older_revision.get_attribute("open") is None
    older_revision.locator("summary").focus()
    page.keyboard.press("Enter")
    page.wait_for_function(
        """() => (
          document.querySelectorAll(
            '#character-carry-over-table details.identity-revision-card'
          )[1]?.open === true
        )"""
    )


def _render_health_state(
    page,
    *,
    source_payload: dict[str, Any],
    state: str,
    reason: str,
) -> None:
    """Render one declared health state from a redacted API-shaped payload."""

    payload = deepcopy(source_payload)
    health = _health_item(payload)
    health["state"] = state
    health["latest_reason_code"] = reason

    def fulfill(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    page.route(_CHARACTER_ROUTE, fulfill)
    try:
        page.locator("[data-page-link='character']").click()
        expected = state.replace("_", " ")
        page.wait_for_function(
            """(label) => (
              document.querySelector(
                '#character-carry-over-table .identity-health-card .badge'
              )?.textContent?.trim() === label
            )""",
            arg=expected,
        )
    finally:
        page.unroute(_CHARACTER_ROUTE, fulfill)


def _verify_loading_empty_and_error_states(
    page,
    *,
    source_payload: dict[str, Any],
) -> None:
    """Verify explicit loading, empty, and HTTP-error browser states."""

    page.evaluate("renderCharacterLoadingState()")
    assert page.locator("#character-status").inner_text() == "loading"
    assert "Loading character identity" in page.locator(
        "#character-self-image-table"
    ).inner_text()

    empty_payload = deepcopy(source_payload)
    empty_reasons = {
        "self_image": "no self image data",
        "growth": "no identity growth data",
        "carry_over": "no identity lineage data",
    }
    for panel_name, reason in empty_reasons.items():
        empty_payload["panels"][panel_name] = {
            "status": "empty",
            "items": [],
            "reason": reason,
        }

    def fulfill_empty(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(empty_payload),
        )

    page.route(_CHARACTER_ROUTE, fulfill_empty)
    try:
        page.locator("[data-page-link='character']").click()
        page.wait_for_function(
            """() => (
              document.querySelector('#character-self-image-table')
              ?.textContent?.includes('no self image data')
            )"""
        )
        assert "no identity lineage data" in page.locator(
            "#character-carry-over-table"
        ).inner_text()
    finally:
        page.unroute(_CHARACTER_ROUTE, fulfill_empty)

    def fulfill_error(route) -> None:
        route.fulfill(
            status=200,
            content_type="text/plain",
            body="malformed test response",
        )

    page.route(_CHARACTER_ROUTE, fulfill_error)
    try:
        page.locator("[data-page-link='character']").click()
        page.wait_for_function(
            """() => (
              document.querySelector('#character-status')
              ?.textContent?.trim() === 'unavailable'
            )"""
        )
        assert "Character identity could not be loaded" in page.locator(
            "#character-carry-over-table"
        ).inner_text()
    finally:
        page.unroute(_CHARACTER_ROUTE, fulfill_error)


def _restore_character_payload(
    page,
    payload: dict[str, Any],
) -> None:
    """Restore the real redacted Character response after state probes."""

    def fulfill(route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(payload),
        )

    page.route(_CHARACTER_ROUTE, fulfill)
    try:
        page.locator("[data-page-link='character']").click()
        page.wait_for_function(
            """() => (
              document.querySelector('#character-self-image-table')
              ?.textContent?.includes('Browser proof current self-concept.')
            )"""
        )
    finally:
        page.unroute(_CHARACTER_ROUTE, fulfill)


def _verify_scoped_residue_pages(page) -> dict[str, Any]:
    """Verify User/Group carry-over remains owner-scoped."""

    result: dict[str, Any] = {}
    with page.expect_response(
        lambda response: (
            "/api/entities/users?" in response.url
            and "/api/entities/users/" not in response.url
        )
    ) as users_info:
        _open_page(page, "users", "Users")
    users = users_info.value.json()
    assert users["status"] == "available"
    assert users["items"]
    account = users["items"][0]["accounts"][0]
    platform = account["platform"]
    platform_user_id = account["platform_user_id"]
    page.locator("#user-platform-channel-id").fill(platform_user_id)
    page.locator("#user-channel-type").select_option("private")
    with page.expect_response(
        lambda response: (
            f"/api/entities/users/{platform}/{platform_user_id}"
            in response.url
        )
    ) as user_info:
        page.locator("#user-directory-table button").first.click()
    user_payload = user_info.value.json()
    user_carry = user_payload["panels"]["carry_over"]
    assert user_carry["status"] in {"available", "empty"}
    assert "identity lineage" not in page.locator(
        "#user-carry-over-table"
    ).inner_text().lower()
    result["user"] = {
        "status": user_carry["status"],
        "item_count": len(user_carry["items"]),
    }

    with page.expect_response(
        lambda response: (
            "/api/entities/groups?" in response.url
            and "/api/entities/groups/" not in response.url
        )
    ) as groups_info:
        _open_page(page, "groups", "Groups")
    groups = groups_info.value.json()
    assert groups["status"] == "available"
    assert groups["items"]
    group = groups["items"][0]
    group_platform = group["platform"]
    group_id = group["group_id"]
    with page.expect_response(
        lambda response: (
            f"/api/entities/groups/{group_platform}/{group_id}"
            in response.url
        )
    ) as group_info:
        page.locator("#group-directory-table button").first.click()
    group_payload = group_info.value.json()
    group_carry = group_payload["panels"]["carry_over"]
    assert group_carry["status"] in {"available", "empty"}
    assert "identity lineage" not in page.locator(
        "#group-carry-over-table"
    ).inner_text().lower()
    result["group"] = {
        "status": group_carry["status"],
        "item_count": len(group_carry["items"]),
    }
    return result


def _verify_pace_config_and_restart(page) -> dict[str, Any]:
    """Verify all pace fields, invalid values, and restart application."""

    _open_page(page, "services", "Services")
    _click_service_action(page, service_id="brain", action="start")
    _wait_for_service_state(page, service_id="brain", state="running")

    with page.expect_response(
        lambda response: "/api/services/brain/config" in response.url
        and response.request.method == "GET"
    ) as config_info:
        page.locator("[data-config-service='brain']").first.click()
    config = config_info.value.json()
    fields = {field["key"]: field for field in config["fields"]}
    expected = {
        "character_identity_growth_enabled": {},
        "character_identity_growth_inferred_min_episodes": {
            "min_value": 2,
            "max_value": 8,
        },
        "character_identity_growth_inferred_min_local_dates": {
            "min_value": 1,
            "max_value": 7,
        },
        (
            "character_identity_growth_"
            "max_inferred_promotions_per_local_day"
        ): {
            "min_value": 0,
            "max_value": 3,
        },
        "character_identity_growth_prompt_char_budget": {
            "min_value": 8000,
            "max_value": 30000,
        },
    }
    for field_key, validation in expected.items():
        field = fields[field_key]
        assert field["validation"] == validation
        assert field["restart_required"] is True
        assert page.locator(
            f"[data-config-input='{field_key}']"
        ).count() == 1
    assert page.locator(
        "#service-config-restart-note"
    ).inner_text() == "Apply and restart"

    invalid_bound = _authenticated_json_request(
        page,
        method="PUT",
        path="/api/services/brain/config",
        payload={
            "reason": "browser bound validation",
            "values": {
                "character_identity_growth_inferred_min_episodes": 1,
            },
        },
    )
    assert invalid_bound["status"] == 422
    assert "at least 2" in invalid_bound["body"]["detail"]["message"]

    invalid_cross_field = _authenticated_json_request(
        page,
        method="PUT",
        path="/api/services/brain/config",
        payload={
            "reason": "browser cross-field validation",
            "values": {
                "character_identity_growth_inferred_min_episodes": 2,
                "character_identity_growth_inferred_min_local_dates": 3,
            },
        },
    )
    assert invalid_cross_field["status"] == 422
    assert "cannot exceed" in (
        invalid_cross_field["body"]["detail"]["message"]
    )

    pace_input = page.locator(
        "[data-config-input="
        "'character_identity_growth_inferred_min_episodes']"
    )
    pace_input.fill("4")
    with page.expect_response(
        lambda response: "/api/services/brain/config" in response.url
        and response.request.method == "PUT"
    ) as apply_info:
        page.locator("#service-config-apply").click()
    applied = apply_info.value.json()
    assert applied["restart"] == {
        "attempted": True,
        "succeeded": True,
        "reason": "config apply requires restart",
    }
    _wait_for_service_state(page, service_id="brain", state="running")
    page.wait_for_function(
        """() => (
          document.querySelector('#service-config-state')
          ?.textContent?.trim() === 'override active'
          && document.querySelector('#service-config-apply')?.disabled === false
        )"""
    )

    with page.expect_response(
        lambda response: "/api/services/brain/config/reset" in response.url
    ) as reset_info:
        page.locator("#service-config-reset").click()
    reset_response = reset_info.value
    assert reset_response.status == 200
    reset = reset_response.json()
    assert reset["restart"]["attempted"] is True
    assert reset["restart"]["succeeded"] is True
    _wait_for_service_state(page, service_id="brain", state="running")
    page.wait_for_function(
        """() => (
          document.querySelector('#service-config-state')
          ?.textContent?.trim() === 'default'
          && document.querySelector('#service-config-reset')?.disabled === false
        )"""
    )
    page.locator("#service-config-close").click()
    _click_service_action(page, service_id="brain", action="stop")
    _wait_for_service_state(page, service_id="brain", state="stopped")

    return {
        "field_count": len(expected),
        "fields": sorted(expected),
        "bound_rejection_status": invalid_bound["status"],
        "cross_field_rejection_status": invalid_cross_field["status"],
        "restart_attempted": applied["restart"]["attempted"],
        "restart_succeeded": applied["restart"]["succeeded"],
        "reset_restart_succeeded": reset["restart"]["succeeded"],
    }


def _authenticated_json_request(
    page,
    *,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send one cookie-sharing authenticated browser-context request."""

    base_url = page.url.rstrip("/")
    request_context = page.context.request
    session_response = request_context.get(
        f"{base_url}/api/auth/session"
    )
    assert session_response.status == 200
    session = session_response.json()
    response = request_context.fetch(
        f"{base_url}{path}",
        method=method,
        headers={
            "content-type": "application/json",
            session["csrf_header_name"]: session["csrf_token"],
        },
        data=json.dumps(payload),
    )
    return {
        "status": response.status,
        "body": response.json(),
    }


def _click_service_action(
    page,
    *,
    service_id: str,
    action: str,
) -> None:
    """Click one service action and await its API response."""

    with page.expect_response(
        lambda response: (
            f"/api/services/{service_id}/{action}" in response.url
        )
    ) as response_info:
        page.locator(
            f"[data-service='{service_id}'][data-action='{action}']"
        ).click()
    assert response_info.value.status == 200


def _wait_for_service_state(
    page,
    *,
    service_id: str,
    state: str,
) -> None:
    """Wait for one service status badge."""

    page.wait_for_function(
        """({serviceId, expectedState}) => (
          document.querySelector(
            `[data-service-card="${serviceId}"] `
            + '[data-service-status-badge]'
          )?.textContent?.trim() === expectedState
        )""",
        arg={"serviceId": service_id, "expectedState": state},
        timeout=30000,
    )


def _health_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the public health item from a Character payload."""

    for item in payload["panels"]["carry_over"]["items"]:
        if item.get("kind") == "identity_growth_health":
            return item
    raise AssertionError("Character payload has no identity health item")


def _assert_redacted_character_network(
    payload: dict[str, Any],
) -> None:
    """Assert growth and lineage network surfaces contain no raw handles."""

    public_panels = {
        "growth": payload["panels"]["growth"],
        "carry_over": payload["panels"]["carry_over"],
    }
    serialized = json.dumps(
        public_panels,
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    for forbidden_field in _FORBIDDEN_PUBLIC_FIELDS:
        assert f'"{forbidden_field}"' not in serialized
    for marker in _INTERNAL_VALUE_MARKERS:
        assert marker not in serialized
    revisions = [
        item
        for item in public_panels["carry_over"]["items"]
        if item.get("kind") == "identity_revision"
    ]
    assert revisions
    for revision in revisions:
        for diff in revision["change_diff"]:
            assert set(diff) == {"path", "value_kind", "change"}


def _character_network_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return a content-free summary of the redacted Character payload."""

    growth_items = payload["panels"]["growth"]["items"]
    carry_items = payload["panels"]["carry_over"]["items"]
    revisions = [
        item
        for item in carry_items
        if item.get("kind") == "identity_revision"
    ]
    candidates = [
        item
        for item in growth_items
        if item.get("kind") == "identity_candidate"
    ]
    return {
        "status": payload["status"],
        "panel_statuses": {
            name: panel["status"]
            for name, panel in payload["panels"].items()
        },
        "revision_numbers": [
            revision["revision_number"]
            for revision in revisions
        ],
        "diff_value_kinds": sorted({
            diff["value_kind"]
            for revision in revisions
            for diff in revision["change_diff"]
        }),
        "candidate_states": sorted({
            candidate["status"]
            for candidate in candidates
        }),
        "scope_kinds": sorted({
            scope
            for revision in revisions
            for scope in revision["source_scope_kinds"]
        }),
        "health_state": _health_item(payload)["state"],
        "redaction_assertion": "raw handles and before/after values absent",
    }
