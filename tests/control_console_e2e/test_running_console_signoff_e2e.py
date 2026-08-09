from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote
import os

import pytest

from browser_harness import BrowserSession, write_summary


RUN_REVIEW_ENV = "KAZUSA_RUN_CONTROL_CONSOLE_REVIEW_E2E"
REVIEW_URL_ENV = "KAZUSA_CONTROL_CONSOLE_REVIEW_URL"
REVIEW_TOKEN_ENV = "KAZUSA_CONTROL_CONSOLE_REVIEW_TOKEN"
REVIEW_ARTIFACT_ENV = "KAZUSA_CONTROL_CONSOLE_REVIEW_ARTIFACT_DIR"

PAGE_HEADINGS = {
    "overview": "Overview",
    "services": "Services",
    "logs": "Live logs",
    "debug": "Debug chat",
    "events": "Event monitor",
    "character": "Character",
    "users": "Users",
    "groups": "Groups",
    "calendar": "Calendar",
    "background": "Background work",
    "health": "Health/cache",
    "audit": "Audit",
}
VISIBLE_FORBIDDEN_TEXT = (
    "[object Object]",
    "panel_contract",
    "projection_owner",
    "scope_order",
    "scope_summary",
    "Growth Runs Audit",
    "Prompt View",
    "Operational Backing",
    "Event stream",
    "Model inputs excluded",
    "Background work state",
)
V1_USER_FIELDS = (
    "affinity",
    "relationship_summary",
    "last_relationship_insight",
    "relationship_status",
)
RELATIONSHIP_AXES = {
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
}
CHARACTER_COGNITION_FIELDS = {
    "drives",
    "standards",
    "meaning_state",
    "goals",
    "threats",
    "active_events",
    "knowledge_gaps",
    "affect_activations",
    "updated_at",
}
USER_COGNITION_FIELDS = {
    "goals",
    "threats",
    "active_events",
    "knowledge_gaps",
    "affect_activations",
    "updated_at",
}
OWNER_PRIVATE_FIELDS = {
    "owner_user_id",
    "relationship_id",
    "other_user_id",
    "evidence_refs",
    "embedding",
    "model_input",
    "raw_reflection",
    "raw_wire_text",
}

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_REVIEW_ENV) != "1",
    reason="running-console signoff is opt-in and targets an explicit URL",
)


def test_running_console_information_contract_matrix() -> None:
    """Capture all twelve pages from the exact reviewed local process."""

    base_url = _required_environment(REVIEW_URL_ENV).rstrip("/")
    operator_token = _required_environment(REVIEW_TOKEN_ENV)
    artifact_dir = Path(_required_environment(REVIEW_ARTIFACT_ENV))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    browser_session = BrowserSession(artifact_dir=artifact_dir)
    with browser_session as page:
        page.goto(base_url, wait_until="domcontentloaded")
        _login(page, operator_token)
        response_failures: list[str] = []
        request_failures: list[str] = []
        page.on(
            "response",
            lambda response: _record_response_failure(
                response,
                response_failures,
            ),
        )
        page.on(
            "requestfailed",
            lambda request: _record_request_failure(
                request,
                request_failures,
            ),
        )

        evidence = _inspect_all_pages(
            page=page,
            artifact_dir=artifact_dir,
        )
        console_messages = list(
            getattr(page, "kazusa_console_messages", []),
        )
        assert console_messages == []
        assert response_failures == []
        assert request_failures == []

    summary_path = write_summary(
        artifact_dir=artifact_dir,
        name="running_console_information_contract",
        conclusion="pass",
        details={
            "console_url": base_url,
            "browser_backend": "regular Playwright fallback",
            "pages": evidence,
            "browser_console_messages": 0,
            "failed_http_responses": 0,
            "failed_browser_requests": 0,
            "database_access": "bounded read-only console endpoints",
            "llm_calls": 0,
        },
    )
    assert summary_path.exists()


def _inspect_all_pages(*, page: Any, artifact_dir: Path) -> dict[str, Any]:
    """Inspect page contracts, render state, and bounded API summaries."""

    evidence: dict[str, Any] = {}
    bootstrap = _api_get(page, "/api/bootstrap")

    _open_page(page, "overview")
    _assert_nonempty(
        page,
        (
            "#overview-service-summary-table",
            "#overview-readiness-table",
            "#overview-failures-table",
            "#overview-changes-table",
            "#overview-cognition-graph",
        ),
    )
    overview = _api_get(page, "/api/overview")
    evidence["overview"] = {
        "data_source": "/api/overview",
        "panels": _panel_summary(overview),
        "cognition_status": page.locator(
            "#overview-cognition-status",
        ).inner_text(),
    }
    _capture_page(page, artifact_dir, "overview")

    _open_page(page, "services")
    page.wait_for_selector("#service-grid .service-card")
    routes = _api_get(page, "/api/services/brain/model-routes")
    service_details = page.locator("#service-grid details")
    if service_details.count():
        service_details.first.evaluate("element => { element.open = true; }")
    current_route_values = sum(
        bool(route.get("effective", {}).get("model"))
        for route in routes.get("routes", [])
    )
    assert current_route_values == len(routes.get("routes", []))
    selected_route = page.locator(".brain-route-tile.selected")
    assert selected_route.locator("code").count() == 1
    assert selected_route.locator("code").inner_text().strip()
    assert selected_route.locator(".brain-route-meta .badge").count() == 3
    assert selected_route.evaluate(
        "element => getComputedStyle(element).boxShadow"
    ) == "none"
    runtime_box = page.locator(".brain-runtime-panel").bounding_box()
    routes_box = page.locator(".brain-routes-panel").bounding_box()
    assert runtime_box is not None
    assert routes_box is not None
    assert runtime_box["height"] < routes_box["height"]
    assert abs(runtime_box["x"] - routes_box["x"]) < 1
    assert abs(runtime_box["width"] - routes_box["width"]) < 1
    evidence["services"] = {
        "data_sources": (
            "/api/bootstrap",
            "/api/services/brain/model-routes",
        ),
        "service_count": len(bootstrap.get("services", [])),
        "service_states": _count_values(
            service.get("actual_state", "unknown")
            for service in bootstrap.get("services", [])
        ),
        "route_count": len(routes.get("routes", [])),
        "current_route_values": current_route_values,
    }
    _capture_page(page, artifact_dir, "services")
    page.screenshot(
        path=str(artifact_dir / "services-expanded-detail.png"),
        full_page=True,
    )

    _open_page(page, "logs")
    page.wait_for_timeout(750)
    _assert_nonempty(page, ("#log-table", "#log-stream-status"))
    evidence["logs"] = {
        "data_source": "/api/logs/stream",
        "stream_status": page.locator("#log-stream-status").inner_text(),
        "rendered_rows": page.locator("#log-table tr").count(),
    }
    _capture_page(page, artifact_dir, "logs")

    _open_page(page, "debug")
    debug_page_text = page.locator("[data-page='debug']").inner_text()
    assert "current browser session" in debug_page_text.lower()
    evidence["debug"] = {
        "data_source": "/api/debug-chat",
        "history_scope": "current browser session",
        "brain_status": page.locator("#debug-brain-status").inner_text(),
        "send_enabled": page.locator("#debug-send").is_enabled(),
        "cognition_status": page.locator(
            "#debug-cognition-status",
        ).inner_text(),
        "live_request_sent": False,
    }
    _capture_page(page, artifact_dir, "debug")

    _open_page(page, "events")
    with page.expect_response(
        lambda response: "/api/events?" in response.url,
    ):
        page.locator("#refresh-events").click()
    events = _api_get(page, "/api/events?source=all&limit=25")
    event_items = events.get("items", [])
    assert event_items
    assert len(event_items) <= 25
    assert page.locator("#event-table .table-row").count() <= 25
    assert all(item.get("source") != "process" for item in event_items)
    assert all(item.get("source") == "kazusa" for item in event_items)
    assert {
        str(item.get("event_type"))
        for item in event_items
    }.isdisjoint({"tick", "load_residue_context"})
    assert page.locator("#event-source option[value='console']").count() == 0
    _assert_nonempty(
        page,
        (
            "#event-table",
            "#event-severity-facets",
            "#event-status-facets",
            "#event-component-facets",
        ),
    )
    event_details = page.locator("#event-table details")
    if event_details.count():
        event_details.first.evaluate("element => { element.open = true; }")
    evidence["events"] = {
        "data_source": "/api/events",
        "item_count": len(event_items),
        "sources": sorted(
            {
                str(item.get("source"))
                for item in event_items
                if item.get("source")
            }
        ),
        "facets": sorted(events.get("facets", {})),
    }
    _capture_page(page, artifact_dir, "events")
    if event_details.count():
        page.screenshot(
            path=str(artifact_dir / "events-expanded-detail.png"),
            full_page=True,
        )

    _open_page(page, "character")
    _wait_status_loaded(page, "#character-status")
    character = _api_get(page, "/api/entities/character?limit=25")
    _assert_owner_private_fields_absent(character)
    character_fields = _key_value_names(
        character["panels"]["cognition_state"].get("items", []),
    )
    assert CHARACTER_COGNITION_FIELDS <= character_fields
    _assert_nonempty(
        page,
        (
            "#character-profile-table",
            "#character-cognition-state-table",
            "#character-self-image-table",
            "#character-growth-table",
            "#character-carry-over-table",
        ),
    )
    evidence["character"] = {
        "data_source": "/api/entities/character",
        "panels": _panel_summary(character),
        "v2_cognition_fields": sorted(character_fields),
    }
    _capture_page(page, artifact_dir, "character")

    _open_page(page, "users")
    _wait_status_loaded(page, "#user-directory-status")
    users = _api_get(page, "/api/entities/users?limit=25")
    assert users.get("items")
    user_account = users["items"][0]["accounts"][0]
    user_platform = str(user_account["platform"])
    user_id = str(user_account["platform_user_id"])
    user_path = (
        f"/api/entities/users/{quote(user_platform, safe='')}/"
        f"{quote(user_id, safe='')}?memory_limit=50"
    )
    with page.expect_response(
        lambda response: "/api/entities/users/" in response.url,
    ):
        page.locator("#user-directory-table button").first.click()
    user = _api_get(page, user_path)
    _assert_owner_private_fields_absent(user)
    relationship_items = user["panels"]["relationship"].get("items", [])
    assert {
        str(item.get("axis"))
        for item in relationship_items
    } == RELATIONSHIP_AXES
    assert all(
        set(item) == {"axis", "value", "band"}
        for item in relationship_items
    )
    user_cognition_fields = _key_value_names(
        user["panels"]["cognition_state"].get("items", []),
    )
    assert USER_COGNITION_FIELDS <= user_cognition_fields
    _assert_nonempty(
        page,
        (
            "#user-directory-table",
            "#user-profile-table",
            "#user-relationship-table",
            "#user-cognition-state-table",
            "#user-memory-table",
            "#user-style-table",
            "#user-conversation-progress-table",
            "#user-carry-over-table",
        ),
    )
    evidence["users"] = {
        "directory_source": "/api/entities/users",
        "detail_source": "/api/entities/users/{platform}/{platform_user_id}",
        "directory_count": len(users["items"]),
        "selected_platform": user_platform,
        "selected_id_masked": _mask_identifier(user_id),
        "panels": _panel_summary(user),
        "relationship_axes": sorted(RELATIONSHIP_AXES),
        "relationship_evidence_count": user["panels"]["relationship"].get(
            "evidence_count",
            0,
        ),
        "relationship_updated_at_present": bool(
            user["panels"]["relationship"].get("updated_at"),
        ),
        "v2_cognition_fields": sorted(user_cognition_fields),
    }
    _capture_page(page, artifact_dir, "users")

    _open_page(page, "groups")
    _wait_status_loaded(page, "#group-directory-status")
    groups = _api_get(page, "/api/entities/groups?limit=25")
    assert groups.get("items")
    group_item, group = _group_with_review(
        page,
        groups["items"],
    )
    group_platform = str(group_item["platform"])
    group_id = str(group_item["group_id"])
    group_button = _group_directory_button(
        page,
        platform=group_platform,
        group_id=group_id,
    )
    with page.expect_response(
        lambda response: (
            f"/api/entities/groups/{quote(group_platform, safe='')}/"
            f"{quote(group_id, safe='')}" in response.url
        ),
    ):
        group_button.click()
    _assert_owner_private_fields_absent(group)
    assert group["panels"]["review"]["status"] == "available"
    assert group["panels"]["review"].get("items")
    assert len(group["panels"]["review"]["items"]) == 1
    directory_labels = page.locator(
        "#group-directory-table tr td:first-child",
    ).all_inner_texts()
    assert directory_labels
    assert "None" not in directory_labels
    _assert_nonempty(
        page,
        (
            "#group-directory-table",
            "#group-activity-table",
            "#group-review-table",
            "#group-style-table",
            "#group-carry-over-table",
            "#group-participant-progress-table",
        ),
    )
    evidence["groups"] = {
        "directory_source": "/api/entities/groups",
        "detail_source": "/api/entities/groups/{platform}/{group_id}",
        "directory_count": len(groups["items"]),
        "selected_platform": group_platform,
        "selected_id_masked": _mask_identifier(group_id),
        "panels": _panel_summary(group),
        "review_count": len(group["panels"]["review"]["items"]),
    }
    _capture_page(page, artifact_dir, "groups")
    page.locator("#group-review-table").screenshot(
        path=str(artifact_dir / "groups-review-detail.png"),
    )

    _open_page(page, "calendar")
    _wait_status_loaded(page, "#calendar-status")
    calendar = _api_get(page, "/api/lookups/calendar?limit=12")
    _assert_keys_absent(
        calendar.get("panels", {}),
        {"lease_owner", "schedule_id", "run_id"},
    )
    _assert_nonempty(
        page,
        (
            "#calendar-summary-table",
            "#calendar-schedules-table",
            "#calendar-runs-table",
            "#calendar-cognition-visibility-table",
        ),
    )
    assert page.locator("#calendar-runs-table .record-card").count() <= 6
    assert "prompt view" not in page.locator(
        "[data-page='calendar']",
    ).inner_text().lower()
    for run in calendar["panels"]["runs"].get("items", []):
        result_summary = run.get("result_summary", {})
        if not isinstance(result_summary, dict):
            continue
        assert "status" not in result_summary
        assert "run_kind" not in result_summary
        assert all(
            value != 0
            for key, value in result_summary.items()
            if key.endswith("_count")
        )
    evidence["calendar"] = {
        "data_source": "/api/lookups/calendar",
        "panels": _panel_summary(calendar),
        "completed_runs": sum(
            item.get("status") == "completed"
            for item in calendar["panels"]["runs"].get("items", [])
        ),
    }
    _capture_page(page, artifact_dir, "calendar")

    _open_page(page, "background")
    _wait_status_loaded(page, "#background-status")
    background = _api_get(page, "/api/lookups/background-work?limit=25")
    _assert_keys_absent(
        background.get("panels", {}),
        {
            "prompt",
            "prompt_text",
            "model_input",
            "raw_payload",
            "lease_owner",
        },
    )
    _assert_nonempty(
        page,
        (
            "#background-summary-table",
            "#background-jobs-table",
            "#background-worker-table",
            "#background-errors-table",
            "#background-delivery-table",
        ),
    )
    for job in background["panels"]["jobs"].get("items", []):
        result_summary = job.get("result_summary")
        failure_summary = job.get("failure_summary")
        assert not (
            result_summary
            and failure_summary
            and result_summary == failure_summary
        )
    evidence["background"] = {
        "data_source": "/api/lookups/background-work",
        "panels": _panel_summary(background),
    }
    _capture_page(page, artifact_dir, "background")

    _open_page(page, "health")
    _wait_status_loaded(page, "#health-status")
    health = _api_get(page, "/api/health")
    _assert_nonempty(
        page,
        (
            "#health-readiness-table",
            "#health-workers-table",
            "#health-cache-table",
        ),
    )
    evidence["health"] = {
        "data_source": "/api/health",
        "panels": _panel_summary(health),
    }
    _capture_page(page, artifact_dir, "health")

    _open_page(page, "audit")
    with page.expect_response(
        lambda response: "/api/audit?" in response.url,
    ):
        page.locator("#refresh-audit").click()
    audit = _api_get(page, "/api/audit?limit=100")
    actions = audit.get("actions", audit.get("items", []))
    assert all(isinstance(item.get("target_label"), str) for item in actions)
    assert all(item.get("outcome") for item in actions)
    assert all(
        "_" not in str(item.get("view", ""))
        for item in audit.get("view_summary", [])
    )
    _assert_nonempty(
        page,
        (
            "#audit-table",
            "#audit-view-summary",
            "#audit-outcome-facets",
        ),
    )
    evidence["audit"] = {
        "data_source": "/api/audit",
        "action_count": len(actions),
        "view_summary_count": len(audit.get("view_summary", [])),
        "facets": sorted(audit.get("facets", {})),
    }
    _capture_page(page, artifact_dir, "audit")

    return evidence


def _login(page: Any, token: str) -> None:
    """Authenticate and wait for the initial overview projection."""

    page.locator("#token").fill(token)
    page.locator("#login").click()
    page.wait_for_function(
        """() => (
          document.querySelector('#overview-service-status')?.textContent
          !== 'not loaded'
        )""",
    )


def _open_page(page: Any, page_name: str) -> None:
    """Open one page and verify the expected active heading."""

    page.locator(f"[data-page-link='{page_name}']").click()
    active_page = page.locator(f"[data-page='{page_name}']")
    active_page.wait_for(state="visible")
    assert active_page.locator("h2").first.inner_text() == PAGE_HEADINGS[
        page_name
    ]


def _capture_page(page: Any, artifact_dir: Path, page_name: str) -> None:
    """Check visible safety rules and capture one full-page screenshot."""

    active_page = page.locator(f"[data-page='{page_name}']")
    visible_text = active_page.inner_text()
    for forbidden_text in VISIBLE_FORBIDDEN_TEXT:
        assert forbidden_text not in visible_text
    if page_name == "users":
        for forbidden_field in V1_USER_FIELDS:
            assert forbidden_field not in visible_text
    if page_name not in {"events", "audit"}:
        assert page.locator("#ui-notice").is_hidden()
    page.screenshot(
        path=str(artifact_dir / f"{page_name}.png"),
        full_page=True,
    )


def _wait_status_loaded(page: Any, selector: str) -> None:
    """Wait until one page-level status leaves its initial placeholder."""

    page.wait_for_function(
        """selector => {
          const text = document.querySelector(selector)?.textContent;
          return Boolean(text && text !== 'not loaded');
        }""",
        arg=selector,
    )


def _assert_nonempty(page: Any, selectors: Iterable[str]) -> None:
    """Require each selected semantic region to render readable text."""

    for selector in selectors:
        text = page.locator(selector).inner_text()
        assert text.strip(), f"{selector} rendered no readable text"


def _api_get(page: Any, path: str) -> dict[str, Any]:
    """Fetch one authenticated JSON endpoint through the browser session."""

    result = page.evaluate(
        """async path => {
          const response = await fetch(path, {
            headers: {Accept: 'application/json'},
          });
          return {
            status: response.status,
            payload: await response.json(),
          };
        }""",
        path,
    )
    assert result["status"] == 200, f"{path} returned {result['status']}"
    payload = result["payload"]
    assert isinstance(payload, dict)
    return payload


def _panel_summary(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return status and row counts without retaining live payload values."""

    summary: dict[str, dict[str, Any]] = {}
    for panel_name, panel in payload.get("panels", {}).items():
        if not isinstance(panel, dict):
            summary[str(panel_name)] = {
                "status": "invalid",
                "item_count": 0,
            }
            continue
        items = panel.get("items", [])
        summary[str(panel_name)] = {
            "status": panel.get("status", "not_reported"),
            "item_count": len(items) if isinstance(items, list) else 0,
        }
    return summary


def _key_value_names(items: list[dict[str, Any]]) -> set[str]:
    """Return semantic keys from a key/value panel."""

    return {
        str(item.get("key"))
        for item in items
        if item.get("key") is not None
    }


def _group_with_review(
    page: Any,
    group_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one bounded live group sample with sourced review state."""

    for group_item in group_items:
        platform = str(group_item["platform"])
        group_id = str(group_item["group_id"])
        path = (
            f"/api/entities/groups/{quote(platform, safe='')}/"
            f"{quote(group_id, safe='')}?limit=25"
        )
        payload = _api_get(page, path)
        review_panel = payload.get("panels", {}).get("review", {})
        if (
            review_panel.get("status") == "available"
            and review_panel.get("items")
        ):
            return group_item, payload
    raise AssertionError("no live directory group exposed available review state")


def _group_directory_button(
    page: Any,
    *,
    platform: str,
    group_id: str,
) -> Any:
    """Return the directory button for one exact group source identity."""

    for button in page.locator("#group-directory-table button").all():
        if (
            button.get_attribute("data-group-platform") == platform
            and button.get_attribute("data-group-id") == group_id
        ):
            return button
    raise AssertionError("review-backed group is missing from the directory")


def _assert_owner_private_fields_absent(payload: dict[str, Any]) -> None:
    """Assert owner panels omit private source identifiers and raw content."""

    _assert_keys_absent(payload.get("panels", {}), OWNER_PRIVATE_FIELDS)


def _assert_keys_absent(value: Any, forbidden: set[str]) -> None:
    """Walk one response subtree and reject forbidden dictionary keys."""

    if isinstance(value, dict):
        conflicting_keys = forbidden.intersection(value)
        assert not conflicting_keys, f"private keys crossed boundary: {conflicting_keys}"
        for nested_value in value.values():
            _assert_keys_absent(nested_value, forbidden)
        return
    if isinstance(value, list):
        for nested_value in value:
            _assert_keys_absent(nested_value, forbidden)


def _count_values(values: Iterable[str]) -> dict[str, int]:
    """Count bounded categorical values for the evidence summary."""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _record_response_failure(response: Any, failures: list[str]) -> None:
    """Record authenticated HTTP failures for final browser assertions."""

    if response.status < 400:
        return
    failures.append(f"{response.status} {response.request.method} {response.url}")


def _record_request_failure(request: Any, failures: list[str]) -> None:
    """Record failed requests while allowing deliberate stream cancellation."""

    failure = str(request.failure)
    if "/api/logs/stream?" in request.url and "ERR_ABORTED" in failure:
        return
    failures.append(f"{request.method} {request.url}: {failure}")


def _mask_identifier(value: str) -> str:
    """Mask a real identifier while preserving bounded review usefulness."""

    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def _required_environment(name: str) -> str:
    """Return one required opt-in value or fail with a precise reason."""

    value = os.environ.get(name, "").strip()
    assert value, f"{name} is required when {RUN_REVIEW_ENV}=1"
    return value
