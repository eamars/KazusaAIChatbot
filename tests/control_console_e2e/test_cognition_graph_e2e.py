from __future__ import annotations

from pathlib import Path
import sys

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import FakeBrainServer, graph_snapshot, write_conflict_brain_registry


def test_overview_cognition_graph_updates_from_latest_brain_run(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify Overview graph states, refresh, page switch, and SSE update."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )
        fake_brain.set_graph(graph_snapshot(status="not_reported", run_id="run-none"))

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
            sse_interval_seconds=0.2,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            _assert_graph_status(page, "not reported")

            fake_brain.set_graph(graph_snapshot(status="running", run_id="run-live"))
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("body[data-auth-state='authenticated']")
            _assert_graph_status(page, "running")
            assert page.locator("#overview-cognition-graph .graph-node").count() == 4
            assert "run-live" in page.locator(
                "#overview-cognition-graph .graph-legend"
            ).inner_text()
            assert "internal_monologue" in page.locator(
                "[data-node-id='l2.reasoning'] .node-detail"
            ).inner_text()

            page.locator("[data-page-link='services']").click()
            fake_brain.set_graph(
                graph_snapshot(status="completed", run_id="run-complete")
            )
            page.locator("[data-page-link='overview']").click()
            page.wait_for_function(
                "() => document.querySelector('#overview-cognition-status')?.textContent === 'completed'"
            )
            assert "run-complete" in page.locator(
                "#overview-cognition-graph .graph-legend"
            ).inner_text()

            fake_brain.set_graph(graph_snapshot(status="failed", run_id="run-failed"))
            page.wait_for_function(
                "() => document.querySelector('#overview-cognition-status')?.textContent === 'failed'"
            )

            fake_brain.set_graph({
                "status": "invalid-status",
                "run_id": "run-invalid",
                "nodes": [],
                "edges": [],
            })
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("body[data-auth-state='authenticated']")
            _assert_graph_status(page, "not reported")

            summary = e2e_summary_writer(
                name="overview_cognition_graph_states",
                conclusion="pass",
                details={
                    "console_url": console.base_url,
                    "fake_brain": fake_brain.base_url,
                    "states": [
                        "not_reported",
                        "running",
                        "completed",
                        "failed",
                        "invalid_payload",
                    ],
                    "checked_paths": [
                        "initial bootstrap",
                        "browser refresh",
                        "page switch",
                        "SSE invalidation",
                        "parallel branch nodes",
                        "reasoning hover detail content",
                    ],
                },
            )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("#overview-grid .metric")


def _assert_graph_status(page, expected_status: str) -> None:
    """Assert the visible overview graph status."""

    page.wait_for_function(
        """(expectedStatus) => (
          document.querySelector('#overview-cognition-status')?.textContent
          === expectedStatus
        )""",
        arg=expected_status,
    )
