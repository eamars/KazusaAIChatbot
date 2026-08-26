from __future__ import annotations

import sys
from pathlib import Path

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import (
    FakeBrainServer,
    graph_snapshot,
    write_conflict_brain_registry,
)


def test_stage3_fresh_database_graph_and_debug_handoff(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify Stage 3 settlement/lifecycle telemetry is usable in the console."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "stage3_brain_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )
        fake_brain.set_graph(_stage3_graph_snapshot())

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
            sse_interval_seconds=0.2,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)

            assert page.locator("#overview-cognition-status").inner_text() == (
                "completed"
            )
            graph = page.locator("#overview-cognition-graph")
            assert graph.locator(".graph-node[data-node-id]").count() == 11
            assert graph.locator(
                ".graph-node[data-node-id='reasoning.context']"
            ).count() == 1
            assert graph.locator(
                ".graph-node[data-node-id='surface.visible']"
            ).count() == 1
            assert graph.locator(
                ".graph-node[data-node-id='reasoning.context']"
            ).get_attribute(
                "title"
            ) == "Subjective reasoning is available."

            graph.locator(
                ".graph-node[data-node-id='reasoning.context']"
            ).click()
            inspector_text = graph.locator("[aria-label='Cognition node detail']").inner_text()
            assert "Context consumption" in inspector_text
            assert "prompts" not in graph.inner_text().lower()
            assert "embeddings" not in graph.inner_text().lower()
            assert "raw messages" not in graph.inner_text().lower()
            assert "message envelopes" not in graph.inner_text().lower()

            page.locator("[data-page-link='debug']").click()
            page.wait_for_selector("#debug-send")
            assert page.locator("#debug-send").is_enabled()
            page.locator("textarea[name='message_text']").fill(
                "stage3 browser handoff probe"
            )
            with page.expect_response(
                lambda response: "/api/debug-chat" in response.url
            ):
                page.locator("#debug-send").click()
            page.wait_for_function(
                "() => document.querySelector('#debug-cognition-status')?.textContent === 'completed'"
            )
            assert "fake brain reply" in page.locator("#chat-history").inner_text()
            assert page.locator(
                "#debug-cognition-graph .graph-node[data-node-id]"
            ).count() == 11
            assert len(getattr(page, "kazusa_console_messages", [])) == 0
            requests = fake_brain.chat_requests()
            assert len(requests) == 1
            assert requests[0]["message_envelope"]["body_text"] == (
                "stage3 browser handoff probe"
            )

            summary = e2e_summary_writer(
                name="stage3_fresh_database_console",
                conclusion="pass",
                details={
                    "console_url": console.base_url,
                    "fake_brain": fake_brain.base_url,
                    "graph_nodes": 7,
                    "checked_paths": [
                        "fresh Stage 3 settlement graph",
                        "lifecycle node inspection",
                        "protected-field redaction",
                        "debug-chat handoff",
                        "browser console and page errors",
                    ],
                },
            )

    assert summary.exists()


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


def _stage3_graph_snapshot() -> dict:
    """Return canonical Stage 3 observation telemetry for browser inspection."""

    snapshot = graph_snapshot(
        status="completed",
        run_id="stage3-fresh-database-browser-proof",
        llm_trace_id="llm-trace-stage3",
        cognition_invocation_id="cognition-invocation-stage3",
    )
    assert snapshot is not None
    return snapshot
