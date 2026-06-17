from __future__ import annotations

import os
import urllib.request

import pytest

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


@pytest.mark.skipif(
    os.environ.get("KAZUSA_RUN_REAL_CONTROL_CONSOLE_E2E") != "1",
    reason="real service lifecycle E2E is opt-in because it starts local services",
)
def test_real_brain_and_debug_adapter_start_stop_from_ui(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Start real brain/debug services from the web UI and verify endpoints."""

    with e2e_console(brain_base_url="http://127.0.0.1:8000") as console:
        page = e2e_browser_page(console.base_url)
        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("#service-grid .service-card", state="attached")
        page.locator("[data-page-link='services']").click()

        _click_service_action(page, service_id="brain", action="start")
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='brain'] .badge\")?.textContent === 'running'",
            timeout=90000,
        )
        assert _http_status("http://127.0.0.1:8000/health") == 200

        _click_service_action(page, service_id="adapter.napcat", action="start")
        napcat_state = _wait_for_service_terminal_or_running(
            page,
            service_id="adapter.napcat",
            timeout=60000,
        )
        if napcat_state == "running":
            _click_service_action(page, service_id="adapter.napcat", action="stop")
            page.wait_for_function(
                "() => document.querySelector(\"[data-service-card='adapter.napcat'] .badge\")?.textContent === 'stopped'",
                timeout=60000,
            )
        else:
            assert napcat_state in {"crashed", "unavailable", "conflict"}
            assert page.locator(
                "[data-service-card='adapter.napcat']"
            ).inner_text().strip()

        _click_service_action(page, service_id="adapter.debug", action="start")
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='adapter.debug'] .badge\")?.textContent === 'running'",
            timeout=60000,
        )
        assert _http_status("http://127.0.0.1:8080/api/health") == 200

        page.locator("[data-page-link='debug']").click()
        page.locator("input[name='debug_mode'][value='listen_only']").check()
        page.locator("textarea[name='message_text']").fill("real service smoke test")
        with page.expect_response(lambda response: "/api/debug-chat" in response.url):
            page.locator("#debug-send").click()
        page.wait_for_selector("#chat-history .message:nth-child(2)", timeout=130000)
        assert page.locator("#chat-history").inner_text().strip()

        page.locator("[data-page-link='services']").click()
        _click_service_action(page, service_id="adapter.debug", action="stop")
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='adapter.debug'] .badge\")?.textContent === 'stopped'",
            timeout=60000,
        )
        _click_service_action(page, service_id="brain", action="stop")
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='brain'] .badge\")?.textContent === 'stopped'",
            timeout=60000,
        )

        summary = e2e_summary_writer(
            name="real_service_lifecycle",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "brain_health": "http://127.0.0.1:8000/health",
                "debug_adapter_health": "http://127.0.0.1:8080/api/health",
                "napcat_state": napcat_state,
                "checked_paths": [
                    "brain start",
                    "NapCat adapter start or visible failure",
                    "debug adapter start",
                    "web debug chat send",
                    "debug adapter stop",
                    "brain stop",
                ],
            },
        )

    assert summary.exists()


def _click_service_action(page, *, service_id: str, action: str) -> None:
    """Click one service action and wait for the console API response."""

    with page.expect_response(
        lambda response: f"/api/services/{service_id}/{action}" in response.url,
        timeout=90000,
    ):
        page.locator(
            f"[data-service='{service_id}'][data-action='{action}']"
        ).click()


def _wait_for_service_terminal_or_running(
    page,
    *,
    service_id: str,
    timeout: int,
) -> str:
    """Wait for a service to report a non-transient state."""

    page.wait_for_function(
        """({serviceId}) => {
          const text = document.querySelector(
            `[data-service-card="${serviceId}"] .badge`
          )?.textContent;
          return ['running', 'crashed', 'unavailable', 'conflict'].includes(text);
        }""",
        arg={"serviceId": service_id},
        timeout=timeout,
    )
    state = page.locator(f"[data-service-card='{service_id}'] .badge").inner_text()
    return state


def _http_status(url: str) -> int:
    """Return the HTTP status for a local endpoint."""

    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status
