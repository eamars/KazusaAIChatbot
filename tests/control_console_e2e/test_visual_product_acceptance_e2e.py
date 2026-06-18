from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_desktop_visual_acceptance_for_cards_buttons_and_branding(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify desktop visual invariants that matter for human review."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        page.set_viewport_size({"width": 1600, "height": 900})
        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("#overview-grid .metric")
        assert page.locator("#login-form").is_hidden()
        assert "Kazusa" not in page.locator("#brand-name").inner_text()

        pages = [
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
        ]
        for page_name in pages:
            page.locator(f"[data-page-link='{page_name}']").click()
            page.wait_for_timeout(100)
            card_overflows = page.evaluate(
                """() => Array.from(document.querySelectorAll(
                    '[data-page].active .card, [data-page].active .metric'
                  ))
                  .filter((element) => element.scrollWidth > element.clientWidth + 1)
                  .map((element) => element.textContent.trim().slice(0, 80))"""
            )
            button_overflows = page.evaluate(
                """() => Array.from(document.querySelectorAll(
                    '[data-page].active button, .topbar button'
                  ))
                  .filter((element) => element.scrollWidth > element.clientWidth + 1)
                  .map((element) => element.textContent.trim())"""
            )
            assert card_overflows == []
            assert button_overflows == []

        assert getattr(page, "kazusa_console_messages", None) == []

        summary = e2e_summary_writer(
            name="visual_product_acceptance",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "viewport": "1600x900",
                "checked_pages": pages,
                "checked_invariants": [
                    "token field hidden after login",
                    "database fallback brand does not say Kazusa",
                    "active page cards do not horizontally overflow",
                    "visible buttons do not clip labels",
                ],
            },
        )

    assert summary.exists()
