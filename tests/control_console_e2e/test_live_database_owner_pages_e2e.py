from __future__ import annotations

import os
from typing import Any

import pytest

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


pytestmark = pytest.mark.skipif(
    os.environ.get("KAZUSA_RUN_CONTROL_CONSOLE_LIVE_DB_E2E") != "1",
    reason="live DB owner-page E2E is opt-in because it reads configured MongoDB",
)

def test_live_database_owner_pages_render_human_readable_data(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
    e2e_artifact_dir,
) -> None:
    """Validate real DB-backed owner pages render readable, redacted data."""

    with e2e_console(use_live_project_db=True) as console:
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
            required_populated_panels=(
                "profile",
                "cognition_state",
            ),
        )
        self_image_panel = character_payload["panels"]["self_image"]
        assert self_image_panel["status"] in {"available", "empty"}
        if self_image_panel["status"] == "empty":
            assert self_image_panel["reason"] == (
                "character self-image is not available"
            )
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#character-profile-table",
                "#character-cognition-state-table",
                "#character-self-image-table",
                "#character-growth-table",
                "#character-carry-over-table",
            ),
        )
        character_screenshot_path = (
            e2e_artifact_dir / "live_db_owner_character.png"
        )
        page.screenshot(path=str(character_screenshot_path), full_page=True)

        with page.expect_response(
            lambda response: (
                "/api/entities/users?" in response.url
                and "/api/entities/users/" not in response.url
            )
        ) as user_directory_response_info:
            _open_page(page, "users", "Users")
        user_directory = user_directory_response_info.value.json()
        assert user_directory["status"] == "available"
        assert user_directory["items"]
        user_account = user_directory["items"][0]["accounts"][0]
        user_platform = user_account["platform"]
        user_platform_user_id = user_account["platform_user_id"]
        with page.expect_response(
            lambda response: (
                f"/api/entities/users/{user_platform}/"
                f"{user_platform_user_id}" in response.url
            )
        ) as user_response_info:
            page.locator("#user-directory-table button").first.click()
        user_payload = user_response_info.value.json()
        _assert_owner_payload(
            payload=user_payload,
            owner="user",
            required_populated_panels=("profile", "relationship"),
        )
        relationship = user_payload["panels"]["relationship"]
        assert len(relationship["items"]) == 11
        assert all(
            set(item) == {"axis", "value", "band"}
            for item in relationship["items"]
        )
        assert relationship["evidence_count"] >= 0
        assert relationship["updated_at"]
        assert user_payload["panels"]["style"]["status"] == "empty"
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#user-profile-table",
                "#user-relationship-table",
                "#user-cognition-state-table",
                "#user-memory-table",
                "#user-style-table",
                "#user-conversation-progress-table",
                "#user-carry-over-table",
            ),
        )
        user_screenshot_path = e2e_artifact_dir / "live_db_owner_user.png"
        page.screenshot(path=str(user_screenshot_path), full_page=True)

        with page.expect_response(
            lambda response: (
                "/api/entities/groups?" in response.url
                and "/api/entities/groups/" not in response.url
            )
        ) as group_directory_response_info:
            _open_page(page, "groups", "Groups")
        group_directory = group_directory_response_info.value.json()
        assert group_directory["status"] == "available"
        assert group_directory["items"]
        group_item = group_directory["items"][0]
        group_platform = group_item["platform"]
        group_id = group_item["group_id"]
        with page.expect_response(
            lambda response: (
                f"/api/entities/groups/{group_platform}/{group_id}"
                in response.url
            )
        ) as group_response_info:
            page.locator("#group-directory-table button").first.click()
        group_payload = group_response_info.value.json()
        _assert_owner_payload(
            payload=group_payload,
            owner="group",
            required_populated_panels=("activity",),
        )
        assert group_payload["panels"]["style"]["status"] == "empty"
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#group-activity-table",
                "#group-review-table",
                "#group-style-table",
                "#group-carry-over-table",
                "#group-participant-progress-table",
            ),
        )

        group_screenshot_path = e2e_artifact_dir / "live_db_owner_group.png"
        page.screenshot(path=str(group_screenshot_path), full_page=True)

        with page.expect_response(
            lambda response: "/api/lookups/calendar" in response.url
        ) as calendar_response_info:
            _open_page(page, "calendar", "Calendar")
        calendar_payload = calendar_response_info.value.json()
        schedules_panel = calendar_payload["panels"]["schedules"]
        assert schedules_panel["status"] in {"available", "empty"}
        if schedules_panel["status"] == "available":
            assert schedules_panel["items"]
        else:
            assert schedules_panel["reason"] == "no schedules are configured"
        runs_panel = calendar_payload["panels"]["runs"]
        assert runs_panel["status"] == "available"
        assert any(
            row.get("status") == "completed"
            for row in runs_panel["items"]
        )
        _assert_owner_tables_readable(
            page,
            selectors=(
                "#calendar-summary-table",
                "#calendar-schedules-table",
                "#calendar-runs-table",
                "#calendar-cognition-visibility-table",
            ),
        )
        calendar_screenshot_path = (
            e2e_artifact_dir / "live_db_owner_calendar.png"
        )
        page.screenshot(path=str(calendar_screenshot_path), full_page=True)

        visible_owner_text = page.locator("main").text_content()
        for forbidden in (
            "[object Object]",
            "panel_contract",
            "projection_owner",
            "scope_order",
            "scope_summary",
            "affinity",
            "relationship_summary",
        ):
            assert forbidden not in visible_owner_text
        console_messages = list(getattr(page, "kazusa_console_messages", []))
        assert console_messages == []
        summary = e2e_summary_writer(
            name="live_database_owner_pages",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "tested_samples": {
                    "user": {
                        "platform": user_platform,
                        "platform_user_id_masked": _mask_identifier(
                            user_platform_user_id,
                        ),
                    },
                    "group": {
                        "platform": group_platform,
                        "group_id_masked": _mask_identifier(group_id),
                    },
                },
                "panel_counts": {
                    "character": _panel_item_counts(character_payload),
                    "user": _panel_item_counts(user_payload),
                    "group": _panel_item_counts(group_payload),
                    "calendar": _panel_item_counts(calendar_payload),
                },
                "screenshots": {
                    "character": str(character_screenshot_path),
                    "user": str(user_screenshot_path),
                    "group": str(group_screenshot_path),
                    "calendar": str(calendar_screenshot_path),
                },
                "redaction": "no visible global_user_id, embeddings, prompts, or raw object placeholders",
            },
        )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_function(
        """() => (
          document.querySelector('#overview-service-status')?.textContent
          !== 'not loaded'
        )"""
    )


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
