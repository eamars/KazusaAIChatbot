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
            assert page.locator(
                "#overview-self-cognition-graph .graph-node"
            ).count() == 0
            assert page.locator(
                "#overview-self-cognition-card"
            ).is_hidden()

            fake_brain.set_graph(graph_snapshot(status="running", run_id="run-live"))
            fake_brain.set_self_graph(
                graph_snapshot(status="completed", run_id="self-run-complete")
            )
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("body[data-auth-state='authenticated']")
            _assert_graph_status(page, "running")
            assert page.locator("#overview-cognition-graph .graph-node").count() == 6
            assert page.locator("#overview-cognition-graph .graph-stage-group").count() == 4
            assert "run-live" in page.locator(
                "#overview-cognition-graph .graph-run-summary"
            ).inner_text()
            assert page.locator("#overview-cognition-graph .graph-edge-layer").count() == 0
            assert page.locator("#overview-cognition-graph .node-detail").count() == 0
            assert page.locator(
                "#overview-self-cognition-graph .graph-node"
            ).count() == 6
            assert page.locator(
                "#overview-self-cognition-card"
            ).is_visible()
            assert page.locator(
                "#overview-cognition-graph .graph-latest-event"
            ).count() == 0
            page.locator(
                "#overview-self-cognition-graph [data-node-id='l3.visual_directives']"
            ).click()
            assert "focused" in page.locator(
                "#overview-self-cognition-graph .graph-inspector"
            ).inner_text()
            assert page.locator(
                "#overview-cognition-graph [data-node-id='l2.reasoning']"
            ).get_attribute(
                "title"
            ) == "weigh intent, memory, and scene pressure"
            assert page.locator(
                "#overview-cognition-graph .graph-node.is-current"
            ).count() == 1
            assert page.evaluate(
                """() => {
                  const stage = document.querySelector(
                    '#overview-cognition-graph .cognition-graph-stage'
                  );
                  return stage.scrollWidth <= stage.clientWidth + 1;
                }"""
            )

            page.locator(
                "#overview-cognition-graph [data-node-id='l2.reasoning']"
            ).click()
            assert "weigh intent, memory, and scene pressure" in page.locator(
                "#overview-cognition-graph .graph-inspector"
            ).inner_text()
            reasoning_detail = page.locator("#overview-cognition-graph .graph-inspector")
            assert "summary" not in reasoning_detail.inner_text().lower()
            assert "stage" not in reasoning_detail.inner_text().lower()
            assert "lane" not in reasoning_detail.inner_text().lower()
            assert "branch" not in reasoning_detail.inner_text().lower()

            page.locator(
                "#overview-cognition-graph [data-node-id='l3.visual_directives']"
            ).click()
            visual_detail = page.locator("#overview-cognition-graph .graph-inspector")
            assert "focused" in visual_detail.inner_text()
            assert "attentive" in visual_detail.inner_text()

            page.locator(
                "#overview-cognition-graph [data-node-id='input.message']"
            ).click()
            input_detail = page.locator("#overview-cognition-graph .graph-inspector")
            input_text = input_detail.inner_text()
            assert "input-start" in input_text
            assert "input-end" in input_text
            assert page.locator(
                "#overview-cognition-graph .graph-inspector-rows"
            ).evaluate("element => element.scrollHeight >= element.clientHeight")

            page.locator("[data-page-link='services']").click()
            fake_brain.set_graph(
                graph_snapshot(status="completed", run_id="run-complete")
            )
            page.locator("[data-page-link='overview']").click()
            page.wait_for_function(
                "() => document.querySelector('#overview-cognition-status')?.textContent === 'completed'"
            )
            assert "run-complete" in page.locator(
                "#overview-cognition-graph .graph-run-summary"
            ).inner_text()
            assert "Final node detail" in page.locator(
                "#overview-cognition-graph .graph-inspector"
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
                        "semantic visual and surface nodes",
                        "compact stage groups",
                        "stable inspector detail",
                        "graph stage no horizontal overflow",
                    ],
                },
            )

    assert summary.exists()


def test_cognition_graph_option_a_state_treatment(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify selected, running, completed, terminated, failed, and pending cues."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "brain_conflict_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )
        fake_brain.set_graph(_option_a_state_graph())

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
            sse_interval_seconds=0.2,
        ) as console:
            page = e2e_browser_page(console.base_url)
            page.set_viewport_size({"width": 1600, "height": 900})
            _login(page)
            _assert_graph_status(page, "running")

            page.locator("[data-node-id='decision.reply']").click()
            selected = page.locator("[data-node-id='decision.reply']")
            assert selected.get_attribute("aria-pressed") == "true"
            assert selected.locator(".node-selected-badge").count() == 0
            selected_style = selected.evaluate(
                """(element) => {
                  const activeProbe = document.createElement("span");
                  activeProbe.style.background = "var(--nav-active-bg)";
                  document.body.appendChild(activeProbe);
                  const activeBackground =
                    getComputedStyle(activeProbe).backgroundColor;
                  activeProbe.remove();
                  const style = getComputedStyle(element);
                  return {
                    activeBackground,
                    backgroundColor: style.backgroundColor,
                    outlineStyle: style.outlineStyle,
                    boxShadow: style.boxShadow,
                  };
                }"""
            )
            assert selected_style["backgroundColor"] == selected_style["activeBackground"]
            assert selected_style["outlineStyle"] in {"none", ""}
            assert selected_style["boxShadow"] != "none"

            running = page.locator("[data-node-id='l2.reasoning']")
            assert "is-current" in (running.get_attribute("class") or "")
            assert running.locator(".node-status").inner_text() == "running"
            running_status_class = (
                running.locator(".node-status").get_attribute("class") or ""
            )
            assert "warn" in running_status_class

            completed = page.locator("[data-node-id='input.message']")
            assert completed.locator(".node-status").inner_text() == "completed"
            assert "success" in (
                completed.locator(".node-status").get_attribute("class") or ""
            )

            terminated = page.locator("[data-node-id='surface.skipped']")
            assert terminated.locator(".node-status").inner_text() == "terminated"
            assert "is-terminal" in (terminated.get_attribute("class") or "")
            assert page.locator(
                "#overview-cognition-graph .graph-stage-group.status-terminated"
            ).count() == 1
            assert page.locator(
                "#overview-cognition-graph .graph-connector.status-terminated"
            ).count() >= 1

            failed = page.locator("[data-node-id='tool.failed']")
            assert failed.locator(".node-status").inner_text() == "failed"
            failed_status_class = (
                failed.locator(".node-status").get_attribute("class") or ""
            )
            assert "danger" in failed_status_class

            pending = page.locator("[data-node-id='surface.pending']")
            assert pending.locator(".node-status").inner_text() == "pending"
            assert "pending" in (pending.get_attribute("class") or "")

            summary = e2e_summary_writer(
                name="cognition_graph_option_a_state_treatment",
                conclusion="pass",
                details={
                    "console_url": console.base_url,
                    "fake_brain": fake_brain.base_url,
                    "checked_states": [
                        "selected",
                        "running",
                        "completed",
                        "terminated",
                        "failed",
                        "pending",
                    ],
                    "checked_paths": [
                        "selected node chip and primary ring",
                        "running node warning badge/current marker",
                        "completed node success badge",
                        "skipped node rendered as terminated",
                        "terminated stage and connector",
                        "failed and pending node badges",
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


def _option_a_state_graph() -> dict:
    """Return one graph containing every visual state Option A must distinguish."""

    return {
        "status": "running",
        "run_id": "option-a-state-treatment",
        "nodes": [
            {
                "id": "input.message",
                "label": "Queued turn",
                "stage": "L1",
                "lane": "input",
                "column": 1,
                "branch": "source",
                "status": "completed",
                "detail": {"input": "Accepted by the brain input queue."},
            },
            {
                "id": "decision.reply",
                "label": "Response decision",
                "stage": "L1",
                "lane": "gate",
                "column": 2,
                "branch": "decision",
                "status": "completed",
                "detail": {
                    "reasoning": "Noise without platform-level bot address metadata.",
                    "decision": "false",
                },
            },
            {
                "id": "l2.reasoning",
                "label": "Reasoning",
                "stage": "L2",
                "lane": "cognition",
                "column": 3,
                "branch": "live",
                "status": "running",
                "detail": {
                    "internal_monologue": "Checking whether a reply is grounded.",
                },
            },
            {
                "id": "surface.skipped",
                "label": "Visible surface",
                "stage": "L3",
                "lane": "surface",
                "column": 4,
                "branch": "terminated",
                "status": "skipped",
                "detail": {"messages": []},
            },
            {
                "id": "l3.visual_directives",
                "label": "Visual directive",
                "stage": "L3",
                "lane": "surface",
                "column": 4,
                "branch": "visual",
                "status": "skipped",
                "detail": {
                    "facial_expression": [],
                    "body_language": [],
                    "gaze_direction": [],
                    "visual_vibe": [],
                    "empty_state": "Visual directives disabled for this run.",
                },
            },
            {
                "id": "tool.failed",
                "label": "Tool read",
                "stage": "Tool",
                "lane": "action",
                "column": 5,
                "branch": "error",
                "status": "failed",
                "detail": {"blocker": "Bounded lookup failed."},
            },
            {
                "id": "surface.pending",
                "label": "Reply delivery",
                "stage": "L3",
                "lane": "surface",
                "column": 6,
                "branch": "waiting",
                "status": "pending",
                "detail": {"summary": "Waiting for upstream decision."},
            },
        ],
        "edges": [
            {"source": "input.message", "target": "decision.reply", "kind": "sequence"},
            {"source": "decision.reply", "target": "l2.reasoning", "kind": "fork"},
            {
                "source": "decision.reply",
                "target": "surface.skipped",
                "kind": "sequence",
            },
            {"source": "l2.reasoning", "target": "tool.failed", "kind": "reference"},
            {"source": "tool.failed", "target": "surface.pending", "kind": "sequence"},
        ],
    }
