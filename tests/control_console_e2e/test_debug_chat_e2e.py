from __future__ import annotations

import sys
from pathlib import Path

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import FakeBrainServer, write_conflict_brain_registry


def test_debug_chat_sends_to_brain_and_updates_history_and_graph(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify debug chat UI talks to brain and renders the returned graph."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
            sse_interval_seconds=60.0,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            page.locator("[data-page-link='debug']").click()
            assert page.locator("#debug-send").is_enabled()

            _send_debug_message(
                page,
                mode="visible_reply",
                message="visible reply probe",
            )
            _send_debug_message(
                page,
                mode="think_only",
                message="think only probe",
            )
            _send_debug_message(
                page,
                mode="listen_only",
                message="listen only probe",
            )

            assert page.locator("#chat-history .message").count() == 7
            chat_text = page.locator("#chat-history").inner_text()
            assert "visible reply probe" in chat_text
            assert "think only probe" in chat_text
            assert "listen only probe" in chat_text
            assert "fake brain reply" in chat_text
            assert "trace llm-trace-debug-1" in chat_text
            assert "tracking debug-run-1" in chat_text
            assert page.locator("#debug-cognition-status").inner_text() == "completed"
            run_summary = page.locator(
                "#debug-cognition-graph .graph-run-summary"
            )
            assert "Current debug cognition" in run_summary.inner_text()
            assert "debug-run-1" not in run_summary.inner_text()
            run_reference = run_summary.locator(".graph-run-reference")
            run_reference.locator("summary").click()
            assert "debug-run-1" in run_reference.inner_text()
            assert page.locator(
                "#debug-cognition-graph .graph-node[data-node-id]"
            ).count() == 11
            assert page.locator(
                "#debug-cognition-graph [data-edge-kind='sequence']"
            ).count() >= 1
            assert page.locator(
                "#debug-cognition-graph [data-edge-kind='reference']"
            ).count() >= 1
            debug_graph = page.locator("#debug-cognition-graph")
            debug_graph.locator(
                ".graph-node[data-node-id='surface.visual']"
            ).click()
            inspector = debug_graph.locator(
                "[aria-label='Cognition node detail']"
            )
            inspector_text = inspector.inner_text()
            assert "Visual directives" in inspector_text
            assert "Fields" not in inspector_text
            assert "summary" not in inspector_text.casefold()
            assert "status" not in inspector_text.casefold()
            assert "stage" not in inspector_text.casefold()
            assert "lane" not in inspector_text.casefold()
            assert "branch" not in inspector_text.casefold()

            requests = fake_brain.chat_requests()
            assert len(requests) == 3
            assert requests[0]["debug_modes"] == {
                "listen_only": False,
                "think_only": False,
                "no_remember": True,
            }
            assert requests[1]["debug_modes"] == {
                "listen_only": False,
                "think_only": True,
                "no_remember": True,
            }
            assert requests[2]["debug_modes"] == {
                "listen_only": True,
                "think_only": False,
                "no_remember": True,
            }

            summary = e2e_summary_writer(
                name="debug_chat_fake_brain",
                conclusion="pass",
                details={
                    "console_url": console.base_url,
                    "fake_brain": fake_brain.base_url,
                    "checked_modes": [
                        "visible_reply",
                        "think_only",
                        "listen_only",
                        "no_remember",
                    ],
                    "checked_outputs": [
                        "chat history",
                        "debug cognition graph",
                        "brain /chat payload",
                    ],
                },
            )

    assert summary.exists()


def test_debug_chat_click_shows_live_running_state_before_response(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
) -> None:
    """A slow debug cognition run should immediately update visible UI state."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        fake_brain.set_chat_delay_seconds(1.0)
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            page.locator("[data-page-link='debug']").click()
            page.locator("input[name='debug_mode'][value='visible_reply']").check()
            page.locator("textarea[name='message_text']").fill(
                "slow visible reply probe"
            )

            page.locator("#debug-send").click()

            page.wait_for_function(
                "() => document.querySelector('#debug-cognition-status')?.textContent === 'running'",
                timeout=1000,
            )
            assert page.locator("#debug-send").is_disabled()
            assert page.locator("#ui-notice").is_visible()
            assert page.locator("#debug-cognition-graph [data-node-id]").count() == 0
            assert "slow visible reply probe" in page.locator(
                "#chat-history",
            ).inner_text()

            page.wait_for_function(
                "() => document.querySelector('#debug-cognition-status')?.textContent === 'completed'",
                timeout=5000,
            )
            assert page.locator("#debug-send").is_enabled()
            assert page.locator("#ui-notice").is_hidden()
            assert "fake brain reply" in page.locator("#chat-history").inner_text()


def test_debug_chat_abort_renders_network_failure_without_graph(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
) -> None:
    """An aborted debug request should render a safe, recoverable failure."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            page.locator("[data-page-link='debug']").click()
            page.locator("input[name='debug_mode'][value='visible_reply']").check()
            page.locator("textarea[name='message_text']").fill(
                "aborted network probe"
            )
            page.route(
                "**/api/debug-chat",
                lambda route: route.abort("aborted"),
            )
            page.locator("#debug-send").click()

            page.wait_for_function(
                """() => [...document.querySelectorAll('#chat-history .message')]
                  .some(message => message.textContent.includes('debug request failed'))"""
            )
            assert page.locator("#debug-cognition-status").inner_text() == (
                "unavailable"
            )
            failure_graph = page.locator("#debug-cognition-graph")
            assert "debug_request_failed" in failure_graph.inner_text()
            assert failure_graph.locator("[data-node-id]").count() == 0
            assert page.locator("#debug-send").is_enabled()

            for availability, reason in (
                ("unavailable", "brain_unavailable"),
                ("invalid", "observation_protocol_error"),
                ("not_reported", "not_reported"),
            ):
                page.evaluate(
                    """view => renderDebugCognitionGraph(view)""",
                    {
                        "view_kind": "debug_latest",
                        "availability": availability,
                        "reason_code": reason,
                        "observation": None,
                    },
                )
                assert page.locator("#debug-cognition-status").inner_text() == (
                    availability.replace("_", " ")
                )
                assert reason in failure_graph.inner_text()
                assert failure_graph.locator("[data-node-id]").count() == 0

            assert page.kazusa_console_messages == []


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


def _send_debug_message(page, *, mode: str, message: str) -> None:
    """Send one debug message through the UI."""

    page.locator(f"input[name='debug_mode'][value='{mode}']").check()
    page.locator("textarea[name='message_text']").fill(message)
    with page.expect_response(lambda response: "/api/debug-chat" in response.url):
        page.locator("#debug-send").click()
    page.wait_for_function(
        "() => document.querySelector('#debug-cognition-status')?.textContent === 'completed'"
    )
