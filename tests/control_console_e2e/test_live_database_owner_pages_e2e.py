from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import export_collection_rows


pytestmark = pytest.mark.skipif(
    os.environ.get("KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E") != "1",
    reason="live DB owner-page E2E is opt-in because it reads configured MongoDB",
)


@dataclass(frozen=True)
class LiveOwnerCandidates:
    """Platform-facing identifiers selected from real read-only DB records."""

    platform: str
    platform_user_id: str
    group_platform: str
    group_id: str
    user_id_mask: str
    group_id_mask: str


def test_live_database_owner_pages_render_human_readable_data(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
    e2e_artifact_dir,
) -> None:
    """Validate real DB-backed owner pages render readable, redacted data."""

    candidates = asyncio.run(_discover_live_owner_candidates())
    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)

        with page.expect_response(
            lambda response: "/api/entities/character" in response.url
        ) as character_response_info:
            _open_page(page, "character", "Character")
        character_payload = character_response_info.value.json()
        _assert_owner_payload(
            payload=character_payload,
            owner="character",
            required_populated_panels=("profile", "self_image"),
        )
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#character-profile-table",
                "#character-state-table",
                "#character-self-image-table",
                "#character-growth-table",
                "#character-memory-table",
                "#character-learning-table",
            ),
        )
        character_screenshot_path = (
            e2e_artifact_dir / "live_db_owner_character.png"
        )
        page.screenshot(path=str(character_screenshot_path), full_page=True)

        _open_page(page, "users", "Users")
        page.locator("#user-platform").select_option(candidates.platform)
        page.locator("#user-platform-user-id").fill(candidates.platform_user_id)
        with page.expect_response(
            lambda response: "/api/entities/user" in response.url
        ) as user_response_info:
            page.locator("#refresh-users").click()
        user_payload = user_response_info.value.json()
        _assert_owner_payload(
            payload=user_payload,
            owner="user",
            required_populated_panels=("profile",),
        )
        _assert_any_panel_populated(
            user_payload,
            panel_names=("relationship", "memory", "style"),
        )
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#user-profile-table",
                "#user-relationship-table",
                "#user-memory-table",
                "#user-style-table",
            ),
        )
        user_screenshot_path = e2e_artifact_dir / "live_db_owner_user.png"
        page.screenshot(path=str(user_screenshot_path), full_page=True)

        _open_page(page, "groups", "Groups")
        page.locator("#group-platform").select_option(candidates.group_platform)
        page.locator("#group-id").fill(candidates.group_id)
        with page.expect_response(
            lambda response: "/api/entities/group" in response.url
        ) as group_response_info:
            page.locator("#refresh-groups").click()
        group_payload = group_response_info.value.json()
        _assert_owner_payload(
            payload=group_payload,
            owner="group",
            required_populated_panels=("style",),
        )
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#group-style-table",
                "#group-progress-table",
                "#group-guidance-table",
            ),
        )

        group_screenshot_path = e2e_artifact_dir / "live_db_owner_group.png"
        page.screenshot(path=str(group_screenshot_path), full_page=True)
        console_messages = list(getattr(page, "kazusa_console_messages", []))
        assert console_messages == []
        summary = e2e_summary_writer(
            name="live_database_owner_pages",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "tested_samples": {
                    "user": {
                        "platform": candidates.platform,
                        "platform_user_id_masked": candidates.user_id_mask,
                    },
                    "group": {
                        "platform": candidates.group_platform,
                        "group_id_masked": candidates.group_id_mask,
                    },
                },
                "panel_counts": {
                    "character": _panel_item_counts(character_payload),
                    "user": _panel_item_counts(user_payload),
                    "group": _panel_item_counts(group_payload),
                },
                "screenshots": {
                    "character": str(character_screenshot_path),
                    "user": str(user_screenshot_path),
                    "group": str(group_screenshot_path),
                },
                "redaction": "no visible global_user_id, embeddings, prompts, or raw object placeholders",
            },
        )

    assert summary.exists()


async def _discover_live_owner_candidates() -> LiveOwnerCandidates:
    """Find real platform-facing user and group samples without DB mutation."""

    projection = {"_id": 0, "embedding": 0}
    try:
        profiles = await export_collection_rows(
            collection_name="user_profiles",
            filter_doc={"platform_accounts.0": {"$exists": True}},
            projection=projection,
            sort_doc={},
            limit=200,
        )
        memory_rows = await export_collection_rows(
            collection_name="user_memory_units",
            filter_doc={},
            projection=projection,
            sort_doc={"updated_at": -1},
            limit=500,
        )
        style_rows = await export_collection_rows(
            collection_name="interaction_style_images",
            filter_doc={},
            projection=projection,
            sort_doc={"updated_at": -1},
            limit=500,
        )
    finally:
        await close_db()

    user_candidate = _select_user_candidate(
        profiles=profiles,
        memory_rows=memory_rows,
        style_rows=style_rows,
    )
    group_candidate = _select_group_candidate(style_rows)
    candidates = LiveOwnerCandidates(
        platform=user_candidate["platform"],
        platform_user_id=user_candidate["platform_user_id"],
        group_platform=group_candidate["platform"],
        group_id=group_candidate["group_id"],
        user_id_mask=_mask_identifier(user_candidate["platform_user_id"]),
        group_id_mask=_mask_identifier(group_candidate["group_id"]),
    )
    return candidates


def _select_user_candidate(
    *,
    profiles: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Select a real user that is likely to populate more than profile rows."""

    memory_owner_ids = {
        str(row.get("global_user_id", "")).strip()
        for row in memory_rows
        if str(row.get("global_user_id", "")).strip()
    }
    style_owner_ids = {
        str(row.get("global_user_id", "")).strip()
        for row in style_rows
        if row.get("scope_type") == "user"
        and str(row.get("global_user_id", "")).strip()
    }
    candidates: list[tuple[int, dict[str, str]]] = []
    for profile in profiles:
        global_user_id = str(profile.get("global_user_id", "")).strip()
        account = _first_platform_account(profile)
        if account is None:
            continue
        score = 0
        if global_user_id in memory_owner_ids:
            score += 2
        if global_user_id in style_owner_ids:
            score += 1
        candidates.append((score, account))

    assert candidates, "real DB has no user profile with a platform account"
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_candidate = candidates[0][1]
    return selected_candidate


def _first_platform_account(profile: dict[str, Any]) -> dict[str, str] | None:
    """Return the first browser-lookup-safe platform account from a profile."""

    accounts = profile.get("platform_accounts")
    if not isinstance(accounts, list):
        return None
    for account in accounts:
        if not isinstance(account, dict):
            continue
        platform = str(account.get("platform", "")).strip()
        platform_user_id = str(account.get("platform_user_id", "")).strip()
        if platform and platform_user_id:
            result = {
                "platform": platform,
                "platform_user_id": platform_user_id,
            }
            return result
    return None


def _select_group_candidate(style_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Select a real group-channel style row for group owner validation."""

    for row in style_rows:
        if row.get("scope_type") != "group_channel":
            continue
        platform = str(row.get("platform", "")).strip()
        group_id = str(row.get("platform_channel_id", "")).strip()
        if platform and group_id:
            candidate = {
                "platform": platform,
                "group_id": group_id,
            }
            return candidate
    raise AssertionError("real DB has no group-channel interaction style row")


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("body[data-auth-state='authenticated']")
    page.wait_for_selector("#overview-grid .metric")


def _open_page(page, page_name: str, expected_heading: str) -> None:
    """Open a sidebar page and assert the active page heading."""

    page.locator(f"[data-page-link='{page_name}']").click()
    active_page = page.locator(f"[data-page='{page_name}']")
    active_page.evaluate(
        "element => { if (!element.classList.contains('active')) throw new Error('page not active'); }"
    )
    heading = active_page.locator("h2").first.inner_text()
    assert heading == expected_heading


def _assert_owner_payload(
    *,
    payload: dict[str, Any],
    owner: str,
    required_populated_panels: tuple[str, ...],
) -> None:
    """Assert one owner endpoint returned real, browser-safe panel data."""

    assert payload["owner"] == owner
    assert payload["status"] in {"available", "empty", "partial"}
    panels = payload["panels"]
    for panel_name in required_populated_panels:
        panel = panels[panel_name]
        items = panel["items"]
        assert panel["status"] == "available", panel.get("reason", "")
        assert isinstance(items, list)
        assert items
    redaction_text = repr(payload.get("redaction", {})).lower()
    assert "excluded" in redaction_text


def _assert_any_panel_populated(
    payload: dict[str, Any],
    *,
    panel_names: tuple[str, ...],
) -> None:
    """Assert at least one optional real-data panel is populated."""

    panels = payload["panels"]
    populated = [
        panel_name
        for panel_name in panel_names
        if panels[panel_name]["status"] == "available"
        and panels[panel_name]["items"]
    ]
    assert populated, f"no optional real-data panels populated: {panel_names}"


def _assert_owner_tables_readable(page, *, selectors: tuple[str, ...]) -> None:
    """Assert rendered owner tables are readable and do not leak internals."""

    forbidden_fragments = (
        "[object Object]",
        "object Object",
        "undefined",
        "global_user_id",
        "embedding",
        "model_input",
        "raw_reflection",
        "raw_wire_text",
    )
    for selector in selectors:
        text = page.locator(selector).inner_text()
        assert text.strip(), f"{selector} rendered no text"
        for fragment in forbidden_fragments:
            assert fragment not in text, f"{selector} contains {fragment!r}"


def _panel_item_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Return panel item counts for redacted validation summaries."""

    counts: dict[str, int] = {}
    panels = payload.get("panels", {})
    for panel_name, panel in panels.items():
        if not isinstance(panel, dict):
            counts[panel_name] = 0
            continue
        items = panel.get("items", [])
        if isinstance(items, list):
            counts[panel_name] = len(items)
        else:
            counts[panel_name] = 0
    return counts


def _mask_identifier(value: str) -> str:
    """Return a stable display mask for a real platform identifier."""

    text = str(value)
    if len(text) <= 4:
        masked = "*" * len(text)
        return masked
    masked = f"{text[:2]}***{text[-2:]}"
    return masked
