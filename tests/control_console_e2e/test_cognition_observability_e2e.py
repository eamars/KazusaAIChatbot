from __future__ import annotations

import re
import sys
from copy import deepcopy
from pathlib import Path

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN
from fake_brain import FakeBrainServer, graph_snapshot, write_conflict_brain_registry

LIVE_SECTION_IDS = [
    "input.turn",
    "decision.response",
    "cognition.appraisals",
    "cognition.goal",
    "cognition.response_plan",
    "cognition.affect",
    "reasoning.subjective",
    "reasoning.context_consumption",
    "evidence.memory",
    "evidence.shared_memory_prewarm",
    "context.conversation_progress",
    "context.public_group_scene",
    "action.requests",
    "action.results",
    "action.continuation",
    "surface.visual_directives",
    "surface.visible_messages",
]
SELF_SECTION_IDS = [
    "cognition.appraisals",
    "cognition.goal",
    "cognition.response_plan",
    "cognition.affect",
    "reasoning.subjective",
    "reasoning.context_consumption",
    "evidence.memory",
    "evidence.shared_memory_prewarm",
    "context.conversation_progress",
    "context.public_group_scene",
    "action.requests",
    "action.results",
    "action.continuation",
    "surface.visual_directives",
    "surface.visible_messages",
    "self.source",
    "self.route",
    "self.consolidation",
]
LIVE_NODE_SECTION_REFS = [
    ("input.turn", ["input.turn"]),
    ("decision.response", ["decision.response"]),
    ("cognition.meaning", ["cognition.appraisals"]),
    ("cognition.goal", ["cognition.goal"]),
    ("cognition.response", ["cognition.response_plan"]),
    ("cognition.affect", ["cognition.affect"]),
    (
        "reasoning.context",
        ["reasoning.subjective", "reasoning.context_consumption"],
    ),
    (
        "evidence.memory",
        [
            "evidence.shared_memory_prewarm",
            "evidence.memory",
            "context.conversation_progress",
            "context.public_group_scene",
            "producer.extra",
        ],
    ),
    (
        "action.results",
        ["action.requests", "action.results", "action.continuation"],
    ),
    ("surface.visual", ["surface.visual_directives"]),
    ("surface.visible", ["surface.visible_messages"]),
]
SELF_NODE_SECTION_REFS = [
    ("self.source", ["self.source"]),
    ("cognition.meaning", ["cognition.appraisals"]),
    ("cognition.goal", ["cognition.goal"]),
    ("cognition.response", ["cognition.response_plan"]),
    ("cognition.affect", ["cognition.affect"]),
    (
        "reasoning.context",
        ["reasoning.subjective", "reasoning.context_consumption"],
    ),
    (
        "evidence.memory",
        [
            "evidence.shared_memory_prewarm",
            "evidence.memory",
            "context.conversation_progress",
            "context.public_group_scene",
            "producer.extra",
        ],
    ),
    ("self.route", ["self.route"]),
    (
        "action.results",
        ["action.requests", "action.results", "action.continuation"],
    ),
    ("surface.visual", ["surface.visual_directives"]),
    ("surface.visible", ["surface.visible_messages"]),
    ("self.consolidation", ["self.consolidation"]),
]
UNKNOWN_PRODUCER_VALUE = (
    "生产者扩展内容\n第二行内容 🚀 "
    "<script>window.__producer_extra_executed = true</script> "
    "<img src=x onerror=\"window.__producer_extra_executed = true\"> "
    "& <b>标记</b>"
)


def test_live_debug_and_self_views_share_observation_section_layout(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
) -> None:
    """Live and self views should render the same shared section contracts."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        not_reported_observation = graph_snapshot(
            status="not_reported",
            run_id="not-reported-fixture",
        )
        assert not_reported_observation is not None
        assert all(
            section["status"] == "not_reported"
            for section in not_reported_observation["sections"]
        )
        live_observation = graph_snapshot(
            status="completed",
            run_id="layout-run-shared",
            llm_trace_id="live-layout-trace",
            cognition_invocation_id="live-layout-invocation",
        )
        self_observation = graph_snapshot(
            status="completed",
            run_id="layout-run-shared",
            llm_trace_id="self-layout-trace",
            cognition_invocation_id="self-layout-invocation",
            run_kind="self_cognition",
        )
        assert live_observation is not None
        assert self_observation is not None
        live_observation = _add_unknown_producer_section(live_observation)
        self_observation = _add_unknown_producer_section(self_observation)
        debug_observation = graph_snapshot(
            status="completed",
            run_id="layout-run-shared",
            llm_trace_id="debug-layout-trace",
            cognition_invocation_id="debug-layout-invocation",
        )
        assert debug_observation is not None
        debug_observation = _add_unknown_producer_section(debug_observation)
        fake_brain.set_graph(live_observation)
        fake_brain.set_self_graph(self_observation)
        fake_brain.set_chat_graph(debug_observation)
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "layout_brain_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page = e2e_browser_page(console.base_url)
            page.set_viewport_size({"width": 1440, "height": 1000})
            _login(page)

            live_graph = page.locator("#overview-cognition-graph")
            self_graph = page.locator("#overview-self-cognition-graph")
            assert live_graph.locator(".graph-node[data-node-id]").count() == 11
            assert self_graph.locator(".graph-node[data-node-id]").count() == 12
            _assert_graph_geometry(
                page,
                live_graph,
                expected_group_count=4,
                expected_columns=_observed_columns(live_observation),
            )
            _assert_graph_geometry(
                page,
                self_graph,
                expected_group_count=5,
                expected_columns=_observed_columns(self_observation),
            )
            cognition_card_widths = page.evaluate(
                """() => [
                  document.querySelector('#overview-cognition-graph')
                    ?.closest('.cognition-card')?.getBoundingClientRect().width || 0,
                  document.querySelector('#overview-self-cognition-graph')
                    ?.closest('.cognition-card')?.getBoundingClientRect().width || 0,
                ]"""
            )
            assert max(cognition_card_widths) - min(cognition_card_widths) <= 1
            page.evaluate(
                "() => { window.__producer_extra_executed = false; }"
            )

            assert [
                section["section_id"]
                for section in live_observation["sections"]
            ] == [*LIVE_SECTION_IDS, "producer.extra"]
            assert [
                section["section_id"]
                for section in self_observation["sections"]
            ] == [*SELF_SECTION_IDS, "producer.extra"]

            _send_debug_message(page)
            debug_graph = page.locator("#debug-cognition-graph")
            debug_graph.locator(
                ".graph-node[data-node-id='cognition.affect']"
            ).click()
            _assert_graph_geometry(
                page,
                debug_graph,
                expected_group_count=4,
                expected_columns=_observed_columns(debug_observation),
            )
            page.locator("[data-page-link='overview']").click()
            assert live_graph.locator(
                ".cognition-graph-shell"
            ).get_attribute("data-graph-selected-node-id") == (
                "surface.visible"
            )
            assert self_graph.locator(
                ".cognition-graph-shell"
            ).get_attribute("data-graph-selected-node-id") == (
                "self.consolidation"
            )

            _assert_exact_node_sections(live_graph, LIVE_NODE_SECTION_REFS)
            _assert_exact_node_sections(self_graph, SELF_NODE_SECTION_REFS)
            _assert_unknown_producer_section(live_graph, page)
            _assert_unknown_producer_section(self_graph, page)

            _send_debug_message(page)
            debug_graph = page.locator("#debug-cognition-graph")
            assert debug_graph.locator(".graph-node[data-node-id]").count() == 11
            assert [
                section["section_id"]
                for section in debug_observation["sections"]
            ] == [*LIVE_SECTION_IDS, "producer.extra"]
            _assert_exact_node_sections(debug_graph, LIVE_NODE_SECTION_REFS)
            _assert_unknown_producer_section(debug_graph, page)

            page.evaluate(
                """() => document.querySelectorAll('[data-page]').forEach(page => {
                  page.classList.toggle('active', page.dataset.page === 'overview');
                })"""
            )
            panel_live_view = {
                "view_kind": "overview_latest",
                "availability": "unavailable",
                "reason_code": "panel_live_unavailable",
                "generated_at": "2026-08-27T00:00:00Z",
                "observation": None,
            }
            panel_self_view = {
                "view_kind": "self_latest",
                "availability": "invalid",
                "reason_code": "panel_self_invalid",
                "generated_at": "2026-08-27T00:00:00Z",
                "observation": None,
            }
            page.evaluate(
                """payload => renderOverview(payload)""",
                {
                    "overview": {
                        "latest_cognition_observation": {
                            "view_kind": "overview_latest",
                            "availability": "available",
                            "reason_code": "top_level_live",
                            "generated_at": "2026-08-27T00:00:00Z",
                            "observation": live_observation,
                        },
                        "latest_self_cognition_observation": {
                            "view_kind": "self_latest",
                            "availability": "available",
                            "reason_code": "top_level_self",
                            "generated_at": "2026-08-27T00:00:00Z",
                            "observation": self_observation,
                        },
                        "panels": {
                            "cognition_observations": {
                                "items": [
                                    {
                                        "observation_kind": "conversation",
                                        "view": panel_live_view,
                                    },
                                    {
                                        "observation_kind": "self_cognition",
                                        "view": panel_self_view,
                                    },
                                ],
                            },
                        },
                    },
                },
            )
            assert page.locator("#overview-cognition-status").inner_text() == (
                "unavailable"
            )
            assert "panel_live_unavailable" in page.locator(
                "#overview-cognition-graph"
            ).inner_text()
            assert page.locator(
                "#overview-cognition-graph [data-node-id]"
            ).count() == 0
            self_card = page.locator("#overview-self-cognition-card")
            assert self_card.is_visible()
            assert page.locator("#overview-self-cognition-status").inner_text() == (
                "invalid"
            )
            assert "panel_self_invalid" in page.locator(
                "#overview-self-cognition-graph"
            ).inner_text()
            assert page.locator(
                "#overview-self-cognition-graph [data-node-id]"
            ).count() == 0
            assert page.kazusa_console_messages == []


def test_prewarm_and_context_sources_render_status_counts_and_omissions(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
) -> None:
    """The renderer should expose prewarm/context dispositions and counts."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        observation = graph_snapshot(
            status="completed",
            run_id="prewarm-render-run",
            llm_trace_id="prewarm-render-trace",
            cognition_invocation_id="prewarm-render-invocation",
        )
        assert observation is not None
        observation["status"] = "partial"
        prewarm = next(
            section
            for section in observation["sections"]
            if section["section_id"] == "evidence.shared_memory_prewarm"
        )
        prewarm["reported_record_count"] = 3
        prewarm["displayed_record_count"] = 1
        prewarm["truncated"] = True
        context = next(
            section
            for section in observation["sections"]
            if section["section_id"] == "reasoning.context_consumption"
        )
        context["status"] = "partial"
        context["summary"] = "One context source was unavailable."
        context["fields"] = [
            {
                "key": "overall_status",
                "label": "Overall status",
                "value": "partial",
            },
            {
                "key": "consumer_count",
                "label": "Consumer count",
                "value": 1,
            },
        ]
        context["records"] = [
            {
                "key": "item_01",
                "label": "Context",
                "summary": "style source",
                "fields": [
                    {"key": "stage", "label": "Stage", "value": "surface"},
                    {
                        "key": "source_kind",
                        "label": "Source kind",
                        "value": "style",
                    },
                    {"key": "status", "label": "Status", "value": "partial"},
                ],
            }
        ]
        context["reported_record_count"] = 2
        context["displayed_record_count"] = 1
        context["truncated"] = True
        progress = next(
            section
            for section in observation["sections"]
            if section["section_id"] == "context.conversation_progress"
        )
        progress["status"] = "not_reported"
        progress["summary"] = ""
        progress["fields"] = []
        progress["records"] = []
        progress["reported_record_count"] = 0
        progress["displayed_record_count"] = 0
        progress["truncated"] = False
        group = next(
            section
            for section in observation["sections"]
            if section["section_id"] == "context.public_group_scene"
        )
        group["fields"] = [{
            "key": "visible_participants",
            "label": "Visible participants",
            "value": ["first participant", "<b>second & third</b>"],
        }]
        visual = next(
            section
            for section in observation["sections"]
            if section["section_id"] == "surface.visual_directives"
        )
        visual["status"] = "skipped"
        visual["summary"] = ""
        visual["fields"] = []
        visual["records"] = []
        visual["reported_record_count"] = 0
        visual["displayed_record_count"] = 0
        visual["truncated"] = False
        for node in observation["nodes"]:
            if node["node_id"] == "reasoning.context":
                node["status"] = "partial"
            if node["node_id"] == "surface.visual":
                node["status"] = "skipped"
                node["summary"] = "skipped"
        fake_brain.set_graph(observation)
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "prewarm_brain_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            graph = page.locator("#overview-cognition-graph")
            graph.locator(
                ".graph-node[data-node-id='evidence.memory']"
            ).click()

            prewarm_section = graph.locator(
                "[data-section-id='evidence.shared_memory_prewarm']"
            )
            prewarm_text = prewarm_section.inner_text()
            assert "Shared-memory prewarm" in prewarm_text
            assert "shared_memory_merged" in prewarm_text
            _assert_labeled_value(prewarm_section, "Retrieved count", "1")
            _assert_labeled_value(prewarm_section, "Merged count", "1")
            counts_text = " ".join(
                prewarm_section.locator(".observation-counts").inner_text().split()
            )
            assert re.search(r"displayed\D+1", counts_text, re.IGNORECASE)
            assert re.search(r"reported\D+3", counts_text, re.IGNORECASE)
            assert "omitted" in prewarm_text.casefold()

            context_section = graph.locator(
                "[data-section-id='context.conversation_progress']"
            )
            assert "not reported" in context_section.inner_text().casefold()
            group_section = graph.locator(
                "[data-section-id='context.public_group_scene']"
            )
            assert group_section.locator(
                ".semantic-list li"
            ).all_text_contents() == [
                "first participant",
                "<b>second & third</b>",
            ]
            assert group_section.locator("b").count() == 0
            graph.locator(
                ".graph-node[data-node-id='reasoning.context']"
            ).click()
            consumption_section = graph.locator(
                "[data-section-id='reasoning.context_consumption']"
            )
            assert "partial" in consumption_section.inner_text().casefold()
            assert "omitted" in consumption_section.inner_text().casefold()
            graph.locator(
                ".graph-node[data-node-id='surface.visual']"
            ).click()
            visual_section = graph.locator(
                "[data-section-id='surface.visual_directives']"
            )
            assert visual_section.locator(
                ".observation-section-header .badge"
            ).inner_text() == "skipped"
            assert page.kazusa_console_messages == []


def test_canonical_sequence_and_reference_edges_render(
    tmp_path: Path,
    unused_tcp_port_factory,
    e2e_console,
    e2e_browser_page,
) -> None:
    """Only canonical sequence/reference relationships should be rendered."""

    brain_port = unused_tcp_port_factory()
    with FakeBrainServer(brain_port) as fake_brain:
        observation = graph_snapshot(
            status="completed",
            run_id="edge-render-run",
            llm_trace_id="edge-render-trace",
            cognition_invocation_id="edge-render-invocation",
        )
        assert observation is not None
        assert {edge["kind"] for edge in observation["edges"]} == {
            "sequence",
            "reference",
        }
        fake_brain.set_graph(deepcopy(observation))
        registry_path = write_conflict_brain_registry(
            path=tmp_path / "edge_brain_registry.json",
            fake_brain_base_url=fake_brain.base_url,
            python_executable=sys.executable,
        )

        with e2e_console(
            brain_base_url=fake_brain.base_url,
            service_registry_path=registry_path,
        ) as console:
            page = e2e_browser_page(console.base_url)
            _login(page)
            graph = page.locator("#overview-cognition-graph")
            assert graph.locator("[data-edge-kind='sequence']").count() >= 1
            assert graph.locator("[data-edge-kind='reference']").count() >= 1
            assert graph.locator("[data-edge-kind='fork']").count() == 0
            assert graph.locator("[data-edge-kind='join']").count() == 0
            assert page.kazusa_console_messages == []


def _login(page) -> None:
    """Authenticate the browser page as the deterministic E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_function(
        """() => (
          document.querySelector('#overview-service-status')?.textContent
          !== 'not loaded'
        )"""
    )


def _selected_section_ids(graph) -> list[str]:
    """Return the producer section ids currently shown in one inspector."""

    return graph.locator("[data-section-id]").evaluate_all(
        "elements => elements.map(element => element.dataset.sectionId)"
    )


def _assert_graph_geometry(
    page,
    graph,
    *,
    expected_group_count: int,
    expected_columns: list[int],
) -> None:
    """Verify the bounded producer-column graph geometry at desktop width."""

    metrics = graph.evaluate(
        """graph => {
          const rail = graph.querySelector('.graph-stage-rail');
          const stage = graph.querySelector('.cognition-graph-stage');
          const groups = [...(rail?.querySelectorAll(':scope > .graph-stage-group') || [])];
          const columns = groups.map(group => Number(
            group.querySelector('.graph-stage-header span')?.textContent
              .replace('Step ', '')
          ));
          const stacks = groups
            .map(group => group.querySelector('.graph-branch-stack'))
            .filter(Boolean);
          const nodes = [...(graph.querySelectorAll('.graph-node') || [])];
          const card = graph.closest('.card');
          const widths = groups.map(group => group.getBoundingClientRect().width);
          return {
            groupCount: groups.length,
            columns,
            groupWidths: widths,
            railGap: rail ? getComputedStyle(rail).columnGap : '',
            railOverflow: rail ? rail.scrollWidth - rail.clientWidth : 0,
            stageHeight: stage?.getBoundingClientRect().height || 0,
            branchGaps: stacks.map(stack => getComputedStyle(stack).gap),
            nestedStageWrappers: graph.querySelectorAll(
              '.graph-stage-group .graph-stage-group'
            ).length,
            nodeParentClasses: nodes.map(node => node.parentElement?.className || ''),
            cardWidth: card?.getBoundingClientRect().width || 0,
          };
        }"""
    )
    assert metrics["groupCount"] == expected_group_count
    assert metrics["columns"] == expected_columns
    assert metrics["groupWidths"]
    assert max(metrics["groupWidths"]) - min(metrics["groupWidths"]) <= 1
    assert metrics["railGap"] == "12px"
    assert metrics["railOverflow"] <= 1
    assert metrics["stageHeight"] > 0
    assert metrics["branchGaps"]
    assert set(metrics["branchGaps"]) == {"8px"}
    assert metrics["nestedStageWrappers"] == 0
    assert metrics["nodeParentClasses"]
    assert set(metrics["nodeParentClasses"]) == {"graph-branch-stack"}
    assert metrics["cardWidth"] > 0
    document_widths = page.evaluate(
        """() => ({
          scrollWidth: document.documentElement.scrollWidth,
          clientWidth: document.documentElement.clientWidth,
        })"""
    )
    assert document_widths["scrollWidth"] == document_widths["clientWidth"]


def _observed_columns(observation: dict) -> list[int]:
    """Return sorted producer columns represented by the observation nodes."""

    return sorted({max(1, int(node["column"])) for node in observation["nodes"]})


def _add_unknown_producer_section(observation: dict) -> dict:
    """Attach one producer-approved section to an existing memory node."""

    observation["sections"].append({
        "section_id": "producer.extra",
        "label": "Producer extension",
        "category": "producer",
        "presentation": "fields",
        "status": "completed",
        "summary": "Producer extension is available.",
        "fields": [{
            "key": "detail",
            "label": "Producer detail",
            "value": UNKNOWN_PRODUCER_VALUE,
        }],
        "records": [],
        "reported_record_count": 0,
        "displayed_record_count": 0,
        "truncated": False,
    })
    evidence_node = next(
        node
        for node in observation["nodes"]
        if node["node_id"] == "evidence.memory"
    )
    evidence_node["section_refs"].append("producer.extra")
    return observation


def _assert_exact_node_sections(graph, expected_refs) -> None:
    """Check every node's producer-ordered section references in the browser."""

    for node_id, expected in expected_refs:
        graph.locator(f".graph-node[data-node-id='{node_id}']").click()
        assert _selected_section_ids(graph) == expected


def _assert_unknown_producer_section(graph, page) -> None:
    """Check generic text rendering and the absence of HTML execution."""

    graph.locator(".graph-node[data-node-id='evidence.memory']").click()
    section = graph.locator("[data-section-id='producer.extra']")
    assert section.count() == 1
    assert UNKNOWN_PRODUCER_VALUE in section.text_content()
    assert section.locator("script, img, b").count() == 0
    assert page.evaluate(
        "() => window.__producer_extra_executed === true"
    ) is False


def _assert_labeled_value(section, label: str, expected: str) -> None:
    """Assert one exact value through its visible producer label."""

    label_node = section.get_by_text(label, exact=True)
    assert label_node.count() == 1
    row_text = " ".join(label_node.locator("xpath=..").inner_text().split())
    assert row_text in {f"{label} {expected}", f"{label}: {expected}"}


def _send_debug_message(page) -> None:
    """Send one deterministic debug request and await its completed graph."""

    page.locator("[data-page-link='debug']").click()
    page.locator("input[name='debug_mode'][value='visible_reply']").check()
    page.locator("textarea[name='message_text']").fill("layout debug probe")
    with page.expect_response(lambda response: "/api/debug-chat" in response.url):
        page.locator("#debug-send").click()
    page.wait_for_function(
        "() => document.querySelector('#debug-cognition-status')?.textContent === 'completed'"
    )
