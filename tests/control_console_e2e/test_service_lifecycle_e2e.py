from __future__ import annotations

from pathlib import Path
import json
import sys

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_service_cards_start_stop_and_dependency_states(
    tmp_path: Path,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify service lifecycle controls through the browser UI."""

    registry_path = _write_fake_registry(tmp_path)

    with e2e_console(service_registry_path=registry_path) as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.locator("[data-page-link='services']").click()

        assert _service_badge(page, "brain") == "stopped"
        assert _button(page, "adapter.debug", "start").is_disabled()
        assert _button(page, "adapter.napcat", "start").is_disabled()
        assert _button(page, "brain", "start").is_enabled()
        assert _button(page, "brain", "stop").is_disabled()

        with page.expect_response(lambda response: "/api/services/brain/start" in response.url):
            _button(page, "brain", "start").click()
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='brain'] [data-service-status-badge]\")?.textContent === 'running'"
        )
        assert _button(page, "brain", "start").is_disabled()
        assert _button(page, "brain", "stop").is_enabled()
        assert _button(page, "adapter.debug", "start").is_enabled()
        assert _button(page, "adapter.napcat", "start").is_enabled()

        with page.expect_response(
            lambda response: "/api/services/adapter.debug/start" in response.url
        ):
            _button(page, "adapter.debug", "start").click()
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='adapter.debug'] [data-service-status-badge]\")?.textContent === 'running'"
        )
        assert _button(page, "adapter.debug", "stop").is_enabled()

        with page.expect_response(lambda response: "/api/services/brain/stop" in response.url):
            _button(page, "brain", "stop").click()
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='brain'] [data-service-status-badge]\")?.textContent === 'stopped'"
        )
        assert _service_badge(page, "adapter.debug") == "stopped"
        assert _button(page, "adapter.debug", "start").is_disabled()

        summary = e2e_summary_writer(
            name="service_lifecycle_fake_registry",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "registry": str(registry_path),
                "services": ["brain", "adapter.debug", "adapter.napcat"],
                "checked_controls": [
                    "dependency disabled state",
                    "brain start",
                    "adapter start after dependency",
                    "dependent stop before brain stop",
                    "button mutual exclusion",
                ],
            },
        )

    assert summary.exists()


def test_service_action_click_shows_operator_feedback_while_pending(
    tmp_path: Path,
    e2e_console,
    e2e_browser_page,
) -> None:
    """Lifecycle clicks should give immediate in-page feedback while pending."""

    registry_path = _write_fake_registry(tmp_path)

    with e2e_console(service_registry_path=registry_path) as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.locator("[data-page-link='services']").click()
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/services/brain/start')) {
                  return new Promise((resolve) => {
                    setTimeout(() => resolve(originalFetch(input, init)), 650);
                  });
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _button(page, "brain", "start").click()

        page.wait_for_selector("[data-service='brain'][data-action='start']:disabled")
        assert "Starting Brain service" in page.locator("#ui-notice").inner_text()
        page.wait_for_function(
            "() => document.querySelector(\"[data-service-card='brain'] [data-service-status-badge]\")?.textContent === 'running'",
            timeout=10000,
        )
        page.wait_for_selector("#ui-notice[data-tone='success']")
        assert "Brain service started" in page.locator("#ui-notice").inner_text()


def _write_fake_registry(tmp_path: Path) -> Path:
    """Write a deterministic service registry for fake lifecycle processes."""

    fake_service_path = Path("tests/control_console_e2e/fake_services.py")
    services = [
        {
            "id": "brain",
            "display_name": "Brain service",
            "kind": "backend",
            "command": [sys.executable, str(fake_service_path), "--name", "brain"],
            "cwd": ".",
        },
        {
            "id": "adapter.debug",
            "display_name": "Debug adapter",
            "kind": "adapter",
            "command": [
                sys.executable,
                str(fake_service_path),
                "--name",
                "adapter.debug",
            ],
            "cwd": ".",
            "dependencies": ["brain"],
        },
        {
            "id": "adapter.napcat",
            "display_name": "NapCat QQ adapter",
            "kind": "adapter",
            "command": [
                sys.executable,
                str(fake_service_path),
                "--name",
                "adapter.napcat",
            ],
            "cwd": ".",
            "dependencies": ["brain"],
        },
    ]
    registry_path = tmp_path / "fake_services.json"
    registry_path.write_text(
        json.dumps({"services": services}, indent=2),
        encoding="utf-8",
    )
    return registry_path


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("#service-grid .service-card", state="attached")


def _button(page, service_id: str, action: str):
    """Return one service action button."""

    return page.locator(
        f"[data-service='{service_id}'][data-action='{action}']"
    )


def _service_badge(page, service_id: str) -> str:
    """Return the visible service badge text."""

    return page.locator(
        f"[data-service-card='{service_id}'] [data-service-status-badge]"
    ).inner_text()
