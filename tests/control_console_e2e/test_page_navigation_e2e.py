from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_each_sidebar_page_has_connected_or_explicitly_gated_state(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify each sidebar page activates and exposes connected/gated content."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)

        _open_page(page, "overview", "Overview")
        assert page.locator("#overview-grid .metric").count() > 0
        assert page.locator("#overview-cognition-graph").inner_text().strip()

        _open_page(page, "services", "Services")
        assert page.locator("#service-grid .service-card").count() >= 3

        _open_page(page, "debug", "Debug chat")
        assert page.locator("#debug-send").is_disabled()
        assert "Start or connect the brain service" in page.locator(
            "textarea[name='message_text']"
        ).get_attribute("placeholder")

        _open_page(page, "events", "Event monitor")
        with page.expect_response(lambda response: "/api/events" in response.url):
            page.locator("#refresh-events").click()
        assert page.locator("#event-table").inner_text().strip()

        with page.expect_response(
            lambda response: "/api/character/status" in response.url
        ):
            _open_page(page, "character", "Character")
        assert page.locator("#character-state-table").inner_text().strip()
        assert page.locator("#character-growth-table").inner_text().strip()

        _open_page(page, "memory", "Memory")
        assert page.locator("#memory-status").inner_text() == "needs input"
        page.locator("#memory-global-user-id").fill("e2e-user")
        with page.expect_response(lambda response: "/api/lookups/memory" in response.url):
            page.locator("#refresh-memory").click()
        assert page.locator("#memory-table").inner_text().strip()

        _open_page(page, "style", "Interaction style")
        assert page.locator("#style-status").inner_text() == "needs input"
        page.locator("#style-platform").fill("debug")
        page.locator("#style-channel-id").fill("e2e-group")
        with page.expect_response(lambda response: "/api/lookups/style" in response.url):
            page.locator("#refresh-style").click()
        assert page.locator("#style-table").inner_text().strip()

        with page.expect_response(
            lambda response: "/api/lookups/calendar" in response.url
        ):
            _open_page(page, "calendar", "Calendar")
        assert page.locator("#calendar-status").inner_text() != "not loaded"
        assert page.locator("#calendar-table").inner_text().strip()

        with page.expect_response(
            lambda response: "/api/lookups/background" in response.url
        ):
            _open_page(page, "background", "Background work")
        assert page.locator("#background-status").inner_text() != "not loaded"
        assert page.locator("#background-table").inner_text().strip()

        _open_page(page, "health", "Health/cache")
        assert page.locator("#health-brain-status").inner_text() != "locked"
        assert page.locator("#health-runtime-table").inner_text().strip()

        with page.expect_response(lambda response: "/api/audit" in response.url):
            _open_page(page, "audit", "Audit")
        assert page.locator("#audit-table").inner_text().strip()

        summary = e2e_summary_writer(
            name="page_navigation_connected_states",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "pages": [
                    "overview",
                    "services",
                    "debug",
                    "events",
                    "character",
                    "memory",
                    "style",
                    "calendar",
                    "background",
                    "health",
                    "audit",
                ],
            },
        )

    assert summary.exists()


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
    assert active_page.locator("h2").first.inner_text() == expected_heading
