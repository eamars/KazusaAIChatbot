from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_auth_session_refresh_and_csrf_are_visible_in_browser(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify browser auth, refresh persistence, and CSRF failure behavior."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        page.wait_for_selector("body[data-auth-state='locked']")

        bootstrap_status = page.evaluate(
            "() => fetch('/api/bootstrap').then((response) => response.status)"
        )
        assert bootstrap_status == 401

        with page.expect_event("dialog") as dialog_info:
            page.locator("#token").fill("bad-token")
            page.locator("#login").click()
        dialog = dialog_info.value
        assert "401" in dialog.message
        dialog.accept()
        page.wait_for_selector("body[data-auth-state='locked']")

        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("body[data-auth-state='authenticated']")
        assert page.locator("#login-form").is_hidden()

        session_before_refresh = page.evaluate(
            "() => fetch('/api/auth/session').then((response) => response.json())"
        )
        assert session_before_refresh["authenticated"] is True
        csrf_header_name = session_before_refresh["csrf_header_name"]

        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector("body[data-auth-state='authenticated']")
        assert page.locator("#login-form").is_hidden()
        session_after_refresh = page.evaluate(
            "() => fetch('/api/auth/session').then((response) => response.json())"
        )
        assert session_after_refresh["authenticated"] is True

        invalid_csrf_status = page.evaluate(
            """(csrfHeaderName) => fetch('/api/debug-chat', {
                method: 'POST',
                headers: {
                  'content-type': 'application/json',
                  [csrfHeaderName]: 'invalid-csrf-token'
                },
                body: JSON.stringify({
                  channel_id: 'debug',
                  user_id: 'operator',
                  user_display_name: 'Operator',
                  message_text: 'csrf probe',
                  debug_modes: ['no_remember']
                })
              }).then((response) => response.status)""",
            csrf_header_name,
        )
        assert invalid_csrf_status == 403

        page.context.clear_cookies()
        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector("body[data-auth-state='locked']")
        assert page.locator("#login-form").is_visible()

        summary = e2e_summary_writer(
            name="auth_session_refresh_csrf",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked_controls": [
                    "locked bootstrap",
                    "bad login alert",
                    "valid login",
                    "refresh session restore",
                    "invalid csrf rejection",
                    "missing session relock",
                ],
            },
        )

    assert summary.exists()
