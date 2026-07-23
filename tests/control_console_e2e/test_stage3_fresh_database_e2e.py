from __future__ import annotations

from pathlib import Path
import sys

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import FakeBrainServer, write_conflict_brain_registry


def test_stage3_fresh_database_graph_and_debug_handoff(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_artifact_dir: Path,
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
            assert graph.locator(".graph-node").count() == 7
            assert graph.locator("[data-node-id='settlement.trace']").count() == 1
            assert graph.locator("[data-node-id='lifecycle.record']").count() == 1
            assert graph.locator("[data-node-id='settlement.trace']").get_attribute(
                "title"
            ) == "one canonical episode trace settled"

            graph.locator("[data-node-id='settlement.trace']").click()
            inspector_text = graph.locator(".graph-inspector").inner_text()
            assert "one canonical episode trace settled" in inspector_text
            assert "prompts" not in graph.inner_text().lower()
            assert "embeddings" not in graph.inner_text().lower()
            assert "raw messages" not in graph.inner_text().lower()
            assert "message envelopes" not in graph.inner_text().lower()

            screenshot_path = e2e_artifact_dir / "stage3_settlement_graph.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            assert screenshot_path.exists()

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
            assert page.locator("#debug-cognition-graph .graph-node").count() == 6
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
                        "browser screenshot",
                        "debug-chat handoff",
                        "browser console and page errors",
                    ],
                    "screenshot": str(screenshot_path),
                },
            )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("#overview-grid .metric")


def _stage3_graph_snapshot() -> dict:
    """Return bounded Stage 3 graph telemetry for browser inspection."""

    nodes = [
        {
            "id": "input.user_message",
            "label": "User message",
            "stage": "Input",
            "lane": "input",
            "column": 1,
            "branch": "source",
            "status": "completed",
            "detail": {"summary": "user_message source admitted"},
        },
        {
            "id": "l2.cognition",
            "label": "Cognition",
            "stage": "L2",
            "lane": "cognition",
            "column": 2,
            "branch": "judgment",
            "status": "completed",
            "detail": {"reasoning": "grounded character judgment completed"},
        },
        {
            "id": "l2.actions",
            "label": "Actions",
            "stage": "L2",
            "lane": "action",
            "column": 3,
            "branch": "action",
            "status": "completed",
            "detail": {"summary": "action specs and results projected"},
        },
        {
            "id": "l3.surface",
            "label": "Visible surface",
            "stage": "L3",
            "lane": "surface",
            "column": 4,
            "branch": "dialog",
            "status": "completed",
            "detail": {"summary": "one visible surface returned"},
        },
        {
            "id": "settlement.trace",
            "label": "Episode trace",
            "stage": "Settlement",
            "lane": "settlement",
            "column": 5,
            "branch": "audit",
            "status": "completed",
            "detail": {"summary": "one canonical episode trace settled"},
        },
        {
            "id": "lifecycle.record",
            "label": "Lifecycle record",
            "stage": "Persistence",
            "lane": "persistence",
            "column": 6,
            "branch": "audit",
            "status": "completed",
            "detail": {"summary": "one post-turn lifecycle record persisted"},
        },
        {
            "id": "delivery.correlation",
            "label": "Delivery correlation",
            "stage": "Delivery",
            "lane": "delivery",
            "column": 7,
            "branch": "audit",
            "status": "completed",
            "detail": {"summary": "delivery correlation recorded"},
        },
    ]
    edges = [
        {"source": "input.user_message", "target": "l2.cognition", "kind": "sequence"},
        {"source": "l2.cognition", "target": "l2.actions", "kind": "fork"},
        {"source": "l2.cognition", "target": "l3.surface", "kind": "fork"},
        {"source": "l2.actions", "target": "settlement.trace", "kind": "join"},
        {"source": "l3.surface", "target": "settlement.trace", "kind": "join"},
        {"source": "settlement.trace", "target": "lifecycle.record", "kind": "sequence"},
        {"source": "lifecycle.record", "target": "delivery.correlation", "kind": "sequence"},
    ]
    return {
        "status": "completed",
        "run_id": "stage3-fresh-database-browser-proof",
        "nodes": nodes,
        "edges": edges,
        "redaction": {
            "detail": "bounded Stage 3 operator graph only",
            "excluded": [
                "prompts",
                "embeddings",
                "raw messages",
                "message envelopes",
            ],
        },
    }
