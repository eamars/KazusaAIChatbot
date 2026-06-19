from __future__ import annotations

from pathlib import Path
import sys

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import FakeBrainServer, write_conflict_brain_registry


def test_debug_chat_error_paths_are_visible_and_actionable(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify stopped and failing brain debug-chat paths are visible."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.locator("[data-page-link='debug']").click()
        assert page.locator("#debug-send").is_disabled()
        assert "Start or connect the brain service" in page.locator(
            "textarea[name='message_text']"
        ).get_attribute("placeholder")
        unavailable = page.evaluate(
            """() => fetch('/api/debug-chat', {
                method: 'POST',
                headers: {
                  'content-type': 'application/json',
                  [window.__kazusaE2E.csrfHeaderName]: window.__kazusaE2E.csrfToken
                },
                body: JSON.stringify({
                  channel_id: 'debug',
                  user_id: 'operator',
                  user_display_name: 'Operator',
                  message_text: 'brain unavailable probe',
                  debug_modes: ['no_remember']
                })
              }).then((response) => response.json())"""
        )
        assert unavailable["brain_available"] is False
        assert unavailable["error"]["code"] == "brain_unavailable"

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        fake_brain.set_chat_status_code(500)
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )
        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page.goto(console.base_url, wait_until="domcontentloaded")
            _login(page)
            page.locator("[data-page-link='debug']").click()
            page.locator("textarea[name='message_text']").fill("500 probe")
            with page.expect_response(
                lambda response: "/api/debug-chat" in response.url
            ):
                page.locator("#debug-send").click()
            page.wait_for_selector("#chat-history .message:nth-child(2)")
            assert "brain_unavailable" in page.locator("#chat-history").inner_text()
            assert page.locator("#debug-cognition-status").inner_text() == (
                "not reported"
            )

            summary = e2e_summary_writer(
                name="debug_chat_error_paths",
                conclusion="pass",
                details={
                    "stopped_brain": "send disabled plus API brain_unavailable",
                    "brain_500": "history row with brain_unavailable and graph not reported",
                },
            )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate and expose CSRF metadata for browser-side fetch probes."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("#overview-grid .metric")
    page.evaluate(
        """() => fetch('/api/auth/session')
          .then((response) => response.json())
          .then((session) => {
            window.__kazusaE2E = {
              csrfHeaderName: session.csrf_header_name,
              csrfToken: session.csrf_token
            };
          })"""
    )
