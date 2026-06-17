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

        with page.expect_event("dialog") as dialog_info:
            page.locator("#login").click()
        dialog = dialog_info.value
        assert "422" in dialog.message
        dialog.accept()

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
                    "empty login alert",
                    "valid login",
                    "services nav",
                ],
            },
        )

    assert summary.exists()
