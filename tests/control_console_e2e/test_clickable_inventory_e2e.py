from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_logged_out_and_logged_in_shell_controls_are_responsive(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify shell controls respond before and after operator login."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        page.wait_for_selector("body[data-auth-state='locked']")

        nav_count = page.locator("[data-page-link]").count()
        disabled_nav_count = page.locator("[data-page-link]:disabled").count()
        assert nav_count > 0
        assert disabled_nav_count == nav_count

        page.locator("[data-theme-choice='dark']").click()
        assert page.locator("body").get_attribute("data-theme") == "dark"
        page.locator("[data-theme-choice='bright']").click()
        assert page.locator("body").get_attribute("data-theme") == "bright"

        page.locator("#login").click()
        page.wait_for_selector("#ui-notice[data-tone='danger']")
        assert "422" in page.locator("#ui-notice").inner_text()

        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("body[data-auth-state='authenticated']")

        assert page.locator("#login-form").is_hidden()
        assert page.locator("[data-page-link]:not(:disabled)").count() == nav_count
        page.locator("[data-page-link='services']").click()
        assert page.locator("[data-page='services']").evaluate(
            "element => element.classList.contains('active')"
        )

        summary = e2e_summary_writer(
            name="clickable_inventory_shell",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "nav_count": nav_count,
                "checked_controls": [
                    "theme toggle",
                    "empty login notice",
                    "valid login",
                    "services nav",
                ],
            },
        )

    assert summary.exists()


def test_operator_actions_use_in_page_feedback_not_blocking_alerts(
    e2e_console,
    e2e_browser_page,
) -> None:
    """Visible operator errors and loading states should stay in the page."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        page.wait_for_selector("body[data-auth-state='locked']")
        dialogs = []
        page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))

        page.locator("#login").click()

        page.wait_for_selector("#ui-notice[data-tone='danger']")
        assert dialogs == []
        assert "422" in page.locator("#ui-notice").inner_text()

        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("body[data-auth-state='authenticated']")
        page.locator("[data-page-link='events']").click()
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/events')) {
                  return new Promise((resolve) => {
                    setTimeout(() => resolve(originalFetch(input, init)), 650);
                  });
                }
                return originalFetch(input, init);
              };
            }"""
        )

        page.locator("#refresh-events").click()

        page.wait_for_selector("#refresh-events:disabled")
        assert "Loading events" in page.locator("#ui-notice").inner_text()
        page.wait_for_selector("#refresh-events:not(:disabled)", timeout=5000)

        refresh_cases = [
            {
                "page": "memory",
                "button": "#refresh-memory",
                "endpoint": "/api/lookups/memory",
                "loading": "Loading memory",
                "setup": """() => {
                  document.querySelector('#memory-platform').value = 'qq';
                  document.querySelector('#memory-platform-user-id').value = 'e2e-user';
                }""",
            },
            {
                "page": "style",
                "button": "#refresh-style",
                "endpoint": "/api/lookups/style",
                "loading": "Loading interaction style",
                "setup": """() => {
                  document.querySelector('#style-platform').value = 'debug';
                  document.querySelector('#style-platform-user-id').value = 'e2e-user';
                  document.querySelector('#style-channel-id').value = 'e2e-group';
                }""",
            },
            {
                "page": "calendar",
                "button": "#refresh-calendar",
                "endpoint": "/api/lookups/calendar",
                "loading": "Loading calendar",
                "setup": "() => {}",
            },
            {
                "page": "background",
                "button": "#refresh-background",
                "endpoint": "/api/lookups/background",
                "loading": "Loading background work",
                "setup": "() => {}",
            },
        ]
        for case in refresh_cases:
            page.locator(f"[data-page-link='{case['page']}']").click()
            page.evaluate(case["setup"])
            page.evaluate(
                """(endpoint) => {
                  const originalFetch = window.fetch.bind(window);
                  window.fetch = (input, init) => {
                    const url = String(input);
                    if (url.includes(endpoint)) {
                      return new Promise((resolve) => {
                        setTimeout(() => resolve(originalFetch(input, init)), 650);
                      });
                    }
                    return originalFetch(input, init);
                  };
                }""",
                case["endpoint"],
            )

            page.locator(case["button"]).click()

            page.wait_for_selector(f"{case['button']}:disabled")
            assert case["loading"] in page.locator("#ui-notice").inner_text()
            page.wait_for_selector(f"{case['button']}:not(:disabled)", timeout=5000)
