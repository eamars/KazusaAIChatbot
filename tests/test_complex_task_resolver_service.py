"""Standalone service tests for the complex-task resolver."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.complex_task_resolver import (
    COMPLEX_TASK_GRAPH_VERSION,
    COMPLEX_TASK_NODE_VERSION,
    COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
    COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
    COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
    project_complex_task_packet,
    resolve_complex_task,
    validate_complex_task_resolver_context,
    validate_complex_task_resolver_request,
)
from kazusa_ai_chatbot.complex_task_resolver import service as resolver_service
from kazusa_ai_chatbot.complex_task_resolver.algorithmic import AlgorithmicSubagent
from kazusa_ai_chatbot.complex_task_resolver.subagents import (
    UnavailableEvidenceSubagent,
)


class _StageInvoker:
    """Return queued structured responses and retain payloads for inspection."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def __call__(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("unexpected stage invocation")
        response = self._responses.pop(0)
        return response


def _install_stage_invokers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    planner: _StageInvoker,
    node_resolver: _StageInvoker,
    collapse: _StageInvoker,
    synthesizer: _StageInvoker,
    limits: dict[str, int],
    planned_graph: dict[str, object] | None = None,
) -> dict[str, object]:
    """Patch internal stage handlers and return public resolver options."""

    if planned_graph is not None:
        async def plan_graph(
            request: dict[str, object],
            context: dict[str, object],
            options: dict[str, object],
            trace_summary: dict[str, object],
        ) -> dict[str, object]:
            del request, context, options, trace_summary
            return resolver_service.validate_complex_task_graph(planned_graph)

        monkeypatch.setattr(
            resolver_service,
            "_plan_graph",
            plan_graph,
        )
    monkeypatch.setattr(
        resolver_service,
        "_plan_stage_handler",
        planner,
    )
    monkeypatch.setattr(
        resolver_service,
        "_node_stage_handler",
        node_resolver,
    )
    monkeypatch.setattr(
        resolver_service,
        "_collapse_stage_handler",
        collapse,
    )
    monkeypatch.setattr(
        resolver_service,
        "_synthesizer_stage_handler",
        synthesizer,
    )
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        _deterministic_subagents,
    )
    options = {
        "schema_version": COMPLEX_TASK_RESOLVER_OPTIONS_VERSION,
        "limits": limits,
    }
    return options


def _deterministic_subagents() -> dict[str, object]:
    """Return resolver-local subagents suitable for deterministic tests."""

    subagents = {
        "algorithmic": AlgorithmicSubagent(),
        "evidence": UnavailableEvidenceSubagent(
            "web_search unavailable in deterministic service test"
        ),
    }
    return subagents


class _ContextRecordingEvidenceSubagent:
    """Resolved evidence stub that records the context it received."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    async def run(
        self,
        request: dict[str, object],
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> dict[str, object]:
        del request
        self.contexts.append(context)
        return {
            "schema_version": "complex_task_subagent_result.v1",
            "resolved": True,
            "status": "resolved",
            "result": {"summary": "Evidence prose returned without handles."},
            "attempts": max_attempts,
            "cache": {"enabled": False},
            "trace": {"context_received": True},
            "unresolved_items": [],
        }


class _SequencedEvidenceSubagent:
    """Return queued evidence envelopes for graph-control tests."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        self._results = list(results)
        self.requests: list[dict[str, object]] = []

    async def run(
        self,
        request: dict[str, object],
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> dict[str, object]:
        del context
        self.requests.append(request)
        if not self._results:
            raise AssertionError("unexpected evidence subagent call")
        result = dict(self._results.pop(0))
        result["attempts"] = max_attempts
        return result


def test_semantic_tasks_normalize_planner_work_type_aliases() -> None:
    """Accept planner semantic work labels that map to graph node kinds."""

    tasks = resolver_service._semantic_tasks({
        "tasks": [
            {
                "objective": "Calculate total cost.",
                "kind": "arithmetic",
            },
            {
                "objective": "Find public price evidence.",
                "kind": "public_evidence",
            },
            {
                "objective": "Analyze the calculated distribution.",
                "kind": "analysis",
            },
        ],
    })

    assert tasks == [
        {
            "objective": "Calculate total cost.",
            "kind": "algorithmic_task",
        },
        {
            "objective": "Find public price evidence.",
            "kind": "evidence_need",
        },
        {
            "objective": "Analyze the calculated distribution.",
            "kind": "synthesis",
        },
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_runs_one_active_node_and_projects_packet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve one pending algorithmic node through injected runtime stages."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare two options using sourced weighted scores.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "The user is testing the resolver module.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "math_1",
            "subagent": "algorithmic",
            "action": "evaluate_expression",
            "objective": "Calculate the supplied weighted score expression.",
            "payload": {
                "expression": (
                    "Decimal('0.70') * Decimal('0.6') "
                    "+ Decimal('0.95') * Decimal('0.4')"
                ),
                "label": "option_b_score",
                "input_values": [
                    {
                        "label": "weight_quality",
                        "value": "0.70",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Operand values available: 0.70, 0.6, "
                            "0.95, 0.4, 45, 6."
                        ),
                    },
                    {
                        "label": "quality_score",
                        "value": "0.6",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Operand values available: 0.70, 0.6, "
                            "0.95, 0.4, 45, 6."
                        ),
                    },
                    {
                        "label": "weight_cost",
                        "value": "0.95",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Operand values available: 0.70, 0.6, "
                            "0.95, 0.4, 45, 6."
                        ),
                    },
                    {
                        "label": "cost_score",
                        "value": "0.4",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Operand values available: 0.70, 0.6, "
                            "0.95, 0.4, 45, 6."
                        ),
                    },
                ],
                "formula_constants": [],
            },
            "constraints": {},
        },
        "node_update": {
            "status": "resolved",
            "result_summary": "Weighted score calculation completed.",
            "answer_text": "option_b has the higher weighted score.",
            "source_backed_facts": ["option_b scored 0.8000."],
            "assumptions_or_inferences": ["Weights were supplied by input."],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "option_b has the higher weighted score.",
        "knowledge_we_know_so_far": [
            "option_b scored 0.8000.",
            "Weights were supplied by input.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_subagent_attempts": 1},
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    projection = project_complex_task_packet(packet)

    assert packet["investigation_summary"] == (
        "option_b has the higher weighted score."
    )
    assert packet["knowledge_we_know_so_far"] == [
        "option_b scored 0.8000.",
        "Weights were supplied by input.",
        (
            "option_b_score: Decimal('0.70') * Decimal('0.6') "
            "+ Decimal('0.95') * Decimal('0.4') = 0.800"
        ),
    ]
    assert packet["graph"]["nodes"]["math_1"]["status"] == "resolved"
    assert packet["trace_summary"]["iterations"] == 1
    assert packet["trace_summary"]["subagent_calls"] == 1
    subagent_call_log = packet["trace_summary"]["subagent_call_log"]
    assert len(subagent_call_log) == 1
    assert subagent_call_log[0]["subagent"] == "algorithmic"
    assert subagent_call_log[0]["node_id"] == "math_1"
    assert subagent_call_log[0]["action"] == "evaluate_expression"
    assert subagent_call_log[0]["resolved"] is True
    assert subagent_call_log[0]["status"] == "resolved"
    assert subagent_call_log[0]["result"]["result_str"] == "0.800"
    stage_io_log = packet["trace_summary"]["stage_io_log"]
    assert stage_io_log[0]["stage"] == "active_node_resolver"
    assert stage_io_log[-1]["stage"] == "bottom_up_synthesis"
    assert projection["knowledge_we_know_so_far"] == [
        "option_b scored 0.8000.",
        "Weights were supplied by input.",
        (
            "option_b_score: Decimal('0.70') * Decimal('0.6') "
            "+ Decimal('0.95') * Decimal('0.4') = 0.800"
        ),
    ]
    assert "graph" not in projection
    assert "math_1" not in str(projection)
    assert "expected_final_answer" not in str(node_resolver.calls[0])


@pytest.mark.asyncio
async def test_resolve_complex_task_records_bounded_collapse_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply collapse decisions only from a bounded graph candidate response."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve duplicate documentation branches.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Branch A resolved to the same docs.",
            "answer_text": "Both branches found the same documentation.",
            "source_backed_facts": ["Both branches cite the same official docs."],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": True,
            "target_node_id": "branch_b",
            "reason": "same documentation result",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The duplicate branches collapse to one answer.",
        "knowledge_we_know_so_far": [
            "Both branches cite the same official docs.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 2, "max_subagent_attempts": 1},
        planned_graph=_graph_with_duplicate_branch(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["trace_summary"]["collapse_count"] == 1
    assert packet["graph"]["nodes"]["branch_a"]["status"] == "collapsed"
    assert packet["graph"]["nodes"]["branch_a"]["collapsed_into"] == "branch_b"


@pytest.mark.asyncio
async def test_resolve_complex_task_maps_semantic_collapse_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build collapse graph events from local-LLM-friendly collapse output."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve duplicate documentation branches.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Branch A resolved to the same docs.",
            "answer_text": "Both branches found the same documentation.",
            "source_backed_facts": ["Both branches cite the same official docs."],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": True,
            "target_node_id": "branch_b",
            "reason": "same documentation result",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The duplicate branches collapse to one answer.",
        "knowledge_we_know_so_far": [
            "Both branches cite the same official docs.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 2, "max_subagent_attempts": 1},
        planned_graph=_graph_with_duplicate_branch(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["trace_summary"]["collapse_count"] == 1
    assert packet["graph"]["nodes"]["branch_a"]["status"] == "collapsed"
    assert packet["graph"]["nodes"]["branch_a"]["collapsed_into"] == "branch_b"


@pytest.mark.asyncio
async def test_resolve_complex_task_maps_semantic_decomposition_to_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the strict graph from local-LLM-friendly planner output."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare agent harnesses across five dimensions.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Evidence is available as a compact summary.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Collect source-backed facts for each harness.",
                "kind": "evidence_need",
            },
            {
                "objective": "Compare the harnesses across the fixed dimensions.",
                "kind": "subtask",
            },
        ],
    }])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Source facts collected.",
            "answer_text": "Source facts collected.",
            "source_backed_facts": ["Source set collected."],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Source-backed comparison can proceed.",
        "knowledge_we_know_so_far": ["Source set collected."],
        "knowledge_still_lacking": ["Comparison branch still pending."],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
    )

    packet = await resolve_complex_task(request, context, options)
    graph = packet["graph"]

    assert graph["root_node_id"] == "root"
    assert graph["active_node_id"] == "task_1"
    assert graph["nodes"]["root"]["children"] == ["task_1", "task_2"]
    assert graph["nodes"]["task_1"]["node_kind"] == "evidence_need"
    assert graph["nodes"]["task_1"]["status"] == "blocked"
    assert graph["nodes"]["task_2"]["status"] == "pending"
    assert packet["trace_summary"]["subagent_call_log"][0]["subagent"] == "evidence"


@pytest.mark.asyncio
async def test_resolve_complex_task_groups_mixed_prerequisites_under_trunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a recursive trunk for mixed evidence and arithmetic work."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare products using current facts and energy math.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Collect current product facts.",
                "kind": "evidence_need",
            },
            {
                "objective": "Calculate total energy requirement.",
                "kind": "algorithmic_task",
            },
            {
                "objective": "Synthesize a recommendation.",
                "kind": "synthesis",
            },
        ],
    }])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "task_1_1",
            "subagent": "evidence",
            "action": "collect_evidence",
            "objective": "Collect current product facts.",
            "payload": {},
            "constraints": {},
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence unavailable.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_nodes": 8, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)
    graph = packet["graph"]

    assert graph["nodes"]["task_1"]["node_kind"] == "subtask"
    assert graph["nodes"]["task_1"]["status"] == "expanded"
    assert graph["nodes"]["task_1"]["children"] == ["task_1_1", "task_1_2"]
    assert graph["nodes"]["task_1_1"]["depth"] == 2
    assert graph["nodes"]["task_1_2"]["depth"] == 2
    assert graph["nodes"]["task_2"]["node_kind"] == "synthesis"


@pytest.mark.asyncio
async def test_active_node_loop_projects_prior_attempt_before_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry one active node with prior attempt state in the next prompt."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare two products with ambiguous names.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Resolve ambiguous product references.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([
        {
            "node_attempt": {
                "action": "disambiguate_entity",
                "status": "partial",
                "result_summary": "R9700 may refer to more than one product.",
                "blockers": ["ambiguous entity"],
                "next_action": "expand into entity-resolution and evidence tasks",
            },
        },
        {
            "node_expansion": {
                "children": [
                    {
                        "objective": "Resolve which R9700 device is meant.",
                        "kind": "evidence_need",
                    },
                    {
                        "objective": "Collect comparison evidence.",
                        "kind": "evidence_need",
                    },
                ],
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Ambiguous product comparison was decomposed.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_node_attempts": 2},
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]

    assert node["status"] == "expanded"
    assert node["children"] == ["task_1_1", "task_1_2"]
    assert node["attempts"][0]["action"] == "disambiguate_entity"
    assert node["attempts"][1]["action"] == "expand_node"
    assert node_resolver.calls[1]["active_node"]["recent_attempts"] == [{
        "action": "disambiguate_entity",
        "result_summary": "R9700 may refer to more than one product.",
        "blockers": ["ambiguous entity"],
        "next_action": "expand into entity-resolution and evidence tasks",
    }]
    assert packet["trace_summary"]["node_attempt_count"] == 2


@pytest.mark.asyncio
async def test_active_node_loop_blocks_after_attempt_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when node attempts never produce an applicable result."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve a blocked source.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Fetch a blocked source.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([
        {
            "node_attempt": {
                "action": "refine_search",
                "status": "partial",
                "result_summary": "The first source was blocked.",
                "blockers": ["source access blocked"],
                "next_action": "try another source",
            },
        },
        {
            "node_attempt": {
                "action": "refine_search",
                "status": "partial",
                "result_summary": "Alternate source was also blocked.",
                "blockers": ["alternate source blocked"],
                "next_action": "block with dependency reason",
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The source could not be verified.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_node_attempts": 2},
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]

    assert node["status"] == "blocked"
    assert node["knowledge_still_lacking"] == [
        "structured terminal output or usable evidence for this node",
        (
            "node-resolution loop exhausted: source access blocked; "
            "alternate source blocked"
        ),
    ]
    assert node["evidence_boundary_notes"] == (
        ["Node resolution stopped at the configured attempt limit."]
    )
    assert [attempt["action"] for attempt in node["attempts"]] == [
        "refine_search",
        "refine_search",
    ]
    assert packet["trace_summary"]["node_attempt_count"] == 2


@pytest.mark.asyncio
async def test_active_node_loop_retries_same_objective_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry generic nodes when expansion repeats the active objective."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Draft a standalone resolver implementation plan.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Define the interface contracts and data structures.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([
        {
            "node_expansion": {
                "children": [{
                    "objective": (
                        "Define the interface contracts and data structures."
                    ),
                    "kind": "subtask",
                }],
            },
        },
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": (
                    "Interface contracts should define request, context, "
                    "options, packet projection, graph nodes, and audit traces."
                ),
                "knowledge_we_know_so_far": [
                    (
                        "The resolver entrypoint accepts semantic objective, "
                        "context, and limits, then returns a graph-backed "
                        "knowledge projection."
                    ),
                ],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The design node resolved after retry.",
        "knowledge_we_know_so_far": [
            "The resolver has a standalone semantic IO contract.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_node_attempts": 2},
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]

    assert node["status"] == "resolved"
    assert node["children"] == []
    assert len(node_resolver.calls) == 2
    assert node_resolver.calls[1]["active_node"]["recent_attempts"] == [{
        "action": "expand_node",
        "result_summary": (
            "The proposed expansion repeated the active node objective."
        ),
        "blockers": ["same-objective expansion"],
        "next_action": (
            "Resolve the node directly or split it into narrower executable "
            "children."
        ),
    }]
    assert [attempt["status"] for attempt in node["attempts"]] == [
        "blocked",
        "resolved",
    ]
    assert packet["trace_summary"]["node_attempt_count"] == 2


@pytest.mark.asyncio
async def test_resolve_complex_task_expands_active_node_and_traverses_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expand a complicated node into bounded children and execute a leaf."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Build an emergency power plan.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Break down energy requirements.",
                "kind": "subtask",
            },
        ],
    }])
    node_resolver = _StageInvoker([
        {
            "node_expansion": {
                "children": [
                    {
                        "objective": "Calculate CPAP energy from 45 W for 6 h.",
                        "kind": "algorithmic_task",
                    },
                ],
            },
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "task_1_1",
                "subagent": "algorithmic",
                "action": "evaluate_expression",
                "objective": "Calculate CPAP energy.",
                "payload": {
                    "expression": "45 * 6",
                    "label": "cpap_watt_hours",
                    "input_values": [
                        {
                            "label": "cpap_power_w",
                            "value": "45",
                            "source_node_id": "task_1_1",
                            "source_text": (
                                "Calculate CPAP energy from 45 W for 6 h."
                            ),
                        },
                        {
                            "label": "cpap_hours",
                            "value": "6",
                            "source_node_id": "task_1_1",
                            "source_text": (
                                "Calculate CPAP energy from 45 W for 6 h."
                            ),
                        },
                    ],
                    "formula_constants": [],
                },
                "constraints": {},
            },
        },
    ])
    collapse = _StageInvoker([
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
    ])
    synthesizer = _StageInvoker([{
        "investigation_summary": "CPAP needs 270 Wh.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_subagent_attempts": 1},
    )

    packet = await resolve_complex_task(request, context, options)
    graph = packet["graph"]

    assert graph["nodes"]["task_1"]["status"] == "expanded"
    assert graph["nodes"]["task_1"]["children"] == ["task_1_1"]
    assert graph["nodes"]["task_1_1"]["parent_id"] == "task_1"
    assert graph["nodes"]["task_1_1"]["depth"] == 2
    assert graph["nodes"]["task_1_1"]["node_kind"] == "algorithmic_task"
    assert graph["nodes"]["task_1_1"]["status"] == "resolved"
    assert packet["trace_summary"]["subagent_calls"] == 1


@pytest.mark.asyncio
async def test_owned_evidence_node_self_expansion_uses_subagent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run evidence subagent when an evidence node expands into itself."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare local LLM GPU options.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_evidence_node()
    graph["nodes"]["evidence"]["objective"] = (
        "Identify three realistic local LLM GPU options."
    )
    node_resolver = _StageInvoker([{
        "node_expansion": {
            "children": [{
                "objective": "Identify three realistic local LLM GPU options.",
                "kind": "evidence_need",
            }],
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence source was unavailable.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            "web_search unavailable in deterministic service test",
        ],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["evidence"]
    subagent_call = packet["trace_summary"]["subagent_call_log"][0]

    assert node["children"] == []
    assert node["status"] == "blocked"
    assert subagent_call["subagent"] == "evidence"
    assert subagent_call["action"] == "collect_evidence"
    assert subagent_call["objective"] == (
        "Identify three realistic local LLM GPU options."
    )


@pytest.mark.asyncio
async def test_evidence_node_records_missing_private_artifact_without_subagent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve LLM-owned private-artifact gaps without public retrieval."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Review a diff against the internal L2d workflow plan.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    root = resolver_service._make_graph_node(
        node_id="root",
        parent_id=None,
        depth=0,
        objective="Review a diff against the internal L2d workflow plan.",
        node_kind="root",
        status="expanded",
        children=["task_1"],
    )
    task_1 = resolver_service._make_graph_node(
        node_id="task_1",
        parent_id="root",
        depth=1,
        objective="Retrieve the internal diff and L2d workflow plan.",
        node_kind="evidence_need",
        status="pending",
        children=[],
    )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "task_1",
        "nodes": {
            "root": root,
            "task_1": task_1,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    node_resolver = _StageInvoker([{
        "decision": "record_knowledge",
        "investigation_summary": (
            "The node needs caller-supplied private project artifacts."
        ),
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            "the exact diff content",
            "the internal L2d workflow plan",
        ],
        "recommended_next_iteration": [
            "Provide the diff and workflow plan as caller-supplied evidence.",
        ],
        "evidence_boundary_notes": [
            "This branch depends on private or caller-supplied artifacts.",
        ],
        "completion": "not_answerable",
        "continuation_tasks": [],
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The review is blocked by missing artifacts.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            "the exact diff content",
            "the internal L2d workflow plan",
        ],
        "recommended_next_iteration": [
            "Provide the diff and workflow plan as caller-supplied evidence.",
        ],
        "evidence_boundary_notes": [
            "Private project artifacts are outside public evidence retrieval.",
        ],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=_StageInvoker([]),
        synthesizer=synthesizer,
        limits={"max_iterations": 1},
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]

    assert packet["trace_summary"]["subagent_call_log"] == []
    assert node["status"] == "cannot_answer"
    assert node["knowledge_still_lacking"] == [
        "the exact diff content",
        "the internal L2d workflow plan",
    ]
    assert node["evidence_boundary_notes"] == [
        "This branch depends on private or caller-supplied artifacts.",
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_preserves_semantic_algorithmic_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep arithmetic-like planner hints visible for subagent ownership."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare agent harnesses across five dimensions.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Evidence is available as a compact summary.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Compare the harnesses using a common framework.",
                "kind": "algorithmic_task",
            },
        ],
    }])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "result_summary": "Comparison branch resolved.",
                "answer_text": "Comparison branch resolved.",
                "source_backed_facts": [],
                "assumptions_or_inferences": [],
                "cannot_answer_reason": None,
            },
        },
        {
            "node_update": {
                "status": "blocked",
                "result_summary": "",
                "answer_text": "",
                "source_backed_facts": [],
                "assumptions_or_inferences": [],
                "cannot_answer_reason": "No expression was provided.",
            },
        },
    ])
    collapse = _StageInvoker([])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The comparison is complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["graph"]["nodes"]["task_1"]["node_kind"] == "algorithmic_task"
    assert packet["graph"]["nodes"]["task_1"]["status"] == "blocked"


@pytest.mark.asyncio
async def test_resolve_complex_task_repairs_algorithmic_node_to_subagent_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a subagent-owned node repair prose into typed calculator IO."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate required watt hours from supplied numbers.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "result_summary": "LLM says the answer is 270 Wh.",
                "answer_text": "45 W * 6 h = 270 Wh.",
                "source_backed_facts": [],
                "assumptions_or_inferences": [],
                "cannot_answer_reason": None,
            },
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "math_1",
                "subagent": "algorithmic",
                "action": "evaluate_expression",
                "objective": "Calculate watt hours.",
                "payload": {
                    "expression": "45 * 6",
                    "label": "required_watt_hours",
                    "input_values": [
                        {
                            "label": "power_w",
                            "value": "45",
                            "source_node_id": "math_1",
                            "source_text": (
                                "Operand values available: 0.70, 0.6, "
                                "0.95, 0.4, 45, 6."
                            ),
                        },
                        {
                            "label": "hours",
                            "value": "6",
                            "source_node_id": "math_1",
                            "source_text": (
                                "Operand values available: 0.70, 0.6, "
                                "0.95, 0.4, 45, 6."
                            ),
                        },
                    ],
                    "formula_constants": [],
                },
                "constraints": {},
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]

    assert node["status"] == "resolved"
    assert node["investigation_summary"] == "required_watt_hours: 45 * 6 = 270"
    assert node["knowledge_we_know_so_far"] == [
        "required_watt_hours: 45 * 6 = 270",
    ]
    assert node_resolver.calls[1]["stage"] == "subagent_request_repair"
    assert packet["trace_summary"]["subagent_call_log"][0]["action"] == (
        "evaluate_expression"
    )
    assert packet["trace_summary"]["subagent_call_log"][0]["resolved"] is True


@pytest.mark.asyncio
async def test_resolve_complex_task_repairs_algorithmic_self_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repair algorithmic self-expansion into typed calculator IO."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate required watt hours from supplied numbers.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([
        {
            "node_expansion": {
                "children": [{
                    "objective": "Objective for math_1",
                    "kind": "algorithmic_task",
                }],
            },
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "math_1",
                "subagent": "algorithmic",
                "action": "evaluate_expression",
                "objective": "Calculate watt hours.",
                "payload": {
                    "expression": "45 * 6",
                    "label": "required_watt_hours",
                    "input_values": [
                        {
                            "label": "power_w",
                            "value": "45",
                            "source_node_id": "math_1",
                            "source_text": (
                                "Operand values available: 0.70, 0.6, "
                                "0.95, 0.4, 45, 6."
                            ),
                        },
                        {
                            "label": "hours",
                            "value": "6",
                            "source_node_id": "math_1",
                            "source_text": (
                                "Operand values available: 0.70, 0.6, "
                                "0.95, 0.4, 45, 6."
                            ),
                        },
                    ],
                    "formula_constants": [],
                },
                "constraints": {},
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]

    assert node["children"] == []
    assert node["status"] == "resolved"
    assert node_resolver.calls[1]["stage"] == "subagent_request_repair"
    assert packet["trace_summary"]["subagent_call_log"][0]["action"] == (
        "evaluate_expression"
    )
    assert packet["trace_summary"]["subagent_call_log"][0]["resolved"] is True


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_algorithmic_hidden_operand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject deterministic arithmetic when any operand lacks graph provenance."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate required watt hours.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "math_1",
            "subagent": "algorithmic",
            "action": "evaluate_expression",
            "objective": "Calculate watt hours.",
            "payload": {
                "expression": "45 * 6",
                "label": "required_watt_hours",
                "input_values": [
                    {
                        "label": "power_w",
                        "value": "45",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Operand values available: 0.70, 0.6, "
                            "0.95, 0.4, 45, 6."
                        ),
                    },
                ],
                "formula_constants": [],
            },
            "constraints": {},
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation did not complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            "algorithmic payload: numeric literals missing operand provenance: 6"
        ],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]
    subagent_call = packet["trace_summary"]["subagent_call_log"][0]

    assert node["status"] == "blocked"
    assert node["knowledge_still_lacking"] == [
        "algorithmic payload: numeric literals missing operand provenance: 6"
    ]
    assert subagent_call["resolved"] is False
    assert subagent_call["status"] == "invalid"
    assert subagent_call["unresolved_items"] == [
        "algorithmic payload: numeric literals missing operand provenance: 6"
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_algorithmic_recommendation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject calculator operands sourced only from semantic recommendations."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate from sourced numbers only.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_math_node()
    graph["nodes"]["math_1"]["knowledge_we_know_so_far"] = []
    graph["nodes"]["math_1"]["recommended_next_iteration"] = [
        "Try calculating 45 * 6 if those operands become sourced."
    ]
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "math_1",
            "subagent": "algorithmic",
            "action": "evaluate_expression",
            "objective": "Calculate watt hours.",
            "payload": {
                "expression": "45 * 6",
                "label": "required_watt_hours",
                "input_values": [
                    {
                        "label": "power_w",
                        "value": "45",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Try calculating 45 * 6 if those operands "
                            "become sourced."
                        ),
                    },
                    {
                        "label": "hours",
                        "value": "6",
                        "source_node_id": "math_1",
                        "source_text": (
                            "Try calculating 45 * 6 if those operands "
                            "become sourced."
                        ),
                    },
                ],
                "formula_constants": [],
            },
            "constraints": {},
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation did not complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            (
                "algorithmic input_values[0].source_text: not found in "
                "source node projection"
            )
        ],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]
    subagent_call = packet["trace_summary"]["subagent_call_log"][0]

    assert node["status"] == "blocked"
    assert subagent_call["resolved"] is False
    assert subagent_call["status"] == "invalid"
    assert subagent_call["unresolved_items"] == [
        (
            "algorithmic input_values[0].source_text: not found in "
            "source node projection"
        )
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_retries_invalid_calculator_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use validation feedback as a bounded local repair attempt."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate sourced watt hours.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_math_node()
    source_text = "Operand values available: 0.70, 0.6, 0.95, 0.4, 45, 6."
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "math_1",
                "subagent": "algorithmic",
                "action": "evaluate_expression",
                "objective": "Calculate watt hours.",
                "payload": {
                    "expression": "45 * 6",
                    "label": "required_watt_hours",
                },
                "constraints": {},
            },
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "math_1",
                "subagent": "algorithmic",
                "action": "evaluate_expression",
                "objective": "Calculate watt hours.",
                "payload": {
                    "expression": "45 * 6",
                    "label": "required_watt_hours",
                    "input_values": [
                        {
                            "label": "power_w",
                            "value": "45",
                            "source_node_id": "math_1",
                            "source_text": source_text,
                        },
                        {
                            "label": "duration_h",
                            "value": "6",
                            "source_node_id": "math_1",
                            "source_text": source_text,
                        },
                    ],
                    "formula_constants": [],
                },
                "constraints": {},
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation completed after repair.",
        "knowledge_we_know_so_far": [
            "required_watt_hours: 45 * 6 = 270"
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 2,
            "max_subagent_attempts": 1,
        },
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]
    attempt_log = packet["trace_summary"]["node_attempt_log"]
    subagent_calls = packet["trace_summary"]["subagent_call_log"]

    assert len(node_resolver.calls) == 2
    assert node["status"] == "resolved"
    assert "required_watt_hours" in node["knowledge_we_know_so_far"][0]
    assert attempt_log[0]["action"] == "revise_calculation_request"
    assert attempt_log[0]["status"] == "blocked"
    assert attempt_log[1]["status"] == "resolved"
    assert [call["status"] for call in subagent_calls] == [
        "invalid",
        "resolved",
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_owned_subagent_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent typed graph nodes from calling the wrong resolver subagent."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate required watt hours.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "math_1",
            "subagent": "evidence",
            "action": "collect_evidence",
            "objective": "Retrieve source material for a math node.",
            "payload": {"query": "source material"},
            "constraints": {},
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The node requested the wrong subagent.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [
            "subagent request: algorithmic_task nodes require algorithmic subagent"
        ],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]
    subagent_call = packet["trace_summary"]["subagent_call_log"][0]

    assert node["status"] == "blocked"
    assert subagent_call["subagent"] == "evidence"
    assert subagent_call["resolved"] is False
    assert subagent_call["unresolved_items"] == [
        "subagent request: algorithmic_task nodes require algorithmic subagent"
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_over_depth_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject over-depth node expansion locally instead of failing the packet."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve a bounded branch.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_math_node()
    graph["max_depth"] = 1
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_expansion": {
            "children": [{
                "objective": "Attempt a deeper child.",
                "kind": "subtask",
            }],
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The branch hit graph depth limits.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": ["node_expansion: exceeds max_depth"],
        "recommended_next_iteration": [
            "Resolve within the current graph depth."
        ],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]

    assert "failure_reason" not in packet["trace_summary"]
    assert node["status"] == "blocked"
    assert node["children"] == []
    assert node["knowledge_still_lacking"] == [
        "node_expansion: exceeds max_depth"
    ]


@pytest.mark.asyncio
async def test_evidence_node_uses_subagent_when_expansion_exceeds_graph_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve bounded evidence nodes directly when no child capacity remains."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Find two portable power stations in New Zealand.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_evidence_node()
    graph["max_nodes"] = 3
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_expansion": {
            "children": [
                {
                    "objective": "Find available NZ products.",
                    "kind": "evidence_need",
                },
                {
                    "objective": "Extract weight and price.",
                    "kind": "evidence_need",
                },
            ],
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence collected at graph boundary.",
        "knowledge_we_know_so_far": [
            "Found two NZ portable power station candidates.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    evidence_subagent = _SequencedEvidenceSubagent([{
        "schema_version": "complex_task_subagent_result.v1",
        "resolved": True,
        "status": "resolved",
        "result": {
            "summary": "Found two NZ portable power station candidates.",
        },
        "cache": {"enabled": False},
        "trace": {"fallback_reason": "graph boundary"},
        "unresolved_items": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=graph,
    )
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        lambda: {
            "algorithmic": AlgorithmicSubagent(),
            "evidence": evidence_subagent,
        },
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["evidence"]
    subagent_call = packet["trace_summary"]["subagent_call_log"][0]

    assert node["status"] == "resolved"
    assert node["children"] == []
    assert evidence_subagent.requests[0]["subagent"] == "evidence"
    assert subagent_call["resolved"] is True


@pytest.mark.asyncio
async def test_public_evidence_call_budget_blocks_third_backend_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep WebAgent3/RAG evidence calls under the hard per-run cap."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Collect three independent public evidence facts.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([
        _evidence_subagent_response("evidence_a"),
        _evidence_subagent_response("evidence_b"),
        _evidence_subagent_response("evidence_c"),
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no duplicate",
        },
    }] * 3)
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence branches were bounded.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_nodes": 4, "max_depth": 1},
        planned_graph=_graph_with_three_evidence_nodes(),
    )
    sequenced_subagent = _SequencedEvidenceSubagent([
        _resolved_evidence_subagent_result("Evidence A resolved."),
        _resolved_evidence_subagent_result("Evidence B resolved."),
    ])
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        lambda: {
            "algorithmic": AlgorithmicSubagent(),
            "evidence": sequenced_subagent,
        },
    )

    packet = await resolve_complex_task(request, context, options)

    graph = packet["graph"]
    trace_summary = packet["trace_summary"]
    assert len(sequenced_subagent.requests) == 2
    assert trace_summary["evidence_calls"] == 2
    assert trace_summary["subagent_calls"] == 3
    assert graph["nodes"]["evidence_a"]["status"] == "resolved"
    assert graph["nodes"]["evidence_b"]["status"] == "resolved"
    assert graph["nodes"]["evidence_c"]["status"] == "blocked"
    assert graph["nodes"]["evidence_c"]["knowledge_still_lacking"] == [
        "public evidence for: Objective for evidence_c",
    ]
    assert graph["nodes"]["evidence_c"]["evidence_boundary_notes"] == [
        "Resolver-local subagent could not complete this node.",
        (
            "The resolver stopped public evidence retrieval at its configured "
            "per-run cap."
        ),
    ]
    assert trace_summary["subagent_call_log"][-1]["status"] == "partial"
    assert (
        trace_summary["subagent_call_log"][-1]["trace"]["reason"]
        == "evidence_call_budget_exhausted"
    )


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_algorithmic_node_direct_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LLM prose from resolving deterministic arithmetic nodes."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Calculate required watt hours.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "result_summary": "LLM says the answer is 270 Wh.",
                "answer_text": "45 W * 6 h = 270 Wh.",
                "source_backed_facts": [],
                "assumptions_or_inferences": [],
                "cannot_answer_reason": None,
            },
        },
        {
            "node_update": {
                "status": "blocked",
                "result_summary": "",
                "answer_text": "",
                "source_backed_facts": [],
                "assumptions_or_inferences": [],
                "cannot_answer_reason": "No expression was provided.",
            },
        },
    ])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Calculation complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={
            "max_iterations": 1,
            "max_node_attempts": 1,
            "max_subagent_attempts": 1,
        },
        planned_graph=_graph_with_math_node(),
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["math_1"]

    assert node["status"] == "blocked"
    assert node["knowledge_still_lacking"] == [
        "unsupported algorithmic action: missing_expression"
    ]
    assert packet["trace_summary"]["subagent_call_log"][0]["subagent"] == (
        "algorithmic"
    )
    assert packet["knowledge_still_lacking"] == [
        "unresolved branch: Objective for math_1"
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_blocks_synthesis_with_unresolved_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent synthesis from resolving while earlier prerequisites are blocked."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Recommend a product after evidence and calculation.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Recommendation complete.",
            "answer_text": "Recommendation complete.",
            "source_backed_facts": ["Recommendation fact."],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Recommendation complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
        planned_graph=_graph_with_blocked_prerequisite_and_pending_synthesis(),
    )

    packet = await resolve_complex_task(request, context, options)
    synthesis = packet["graph"]["nodes"]["synthesis"]

    assert synthesis["status"] == "blocked"
    assert synthesis["knowledge_still_lacking"] == [
        "synthesis requires resolved prerequisite nodes: "
        "Objective for evidence"
    ]
    assert packet["knowledge_still_lacking"] == [
        "unresolved branch: Objective for evidence",
        "unresolved branch: Objective for synthesis",
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_fails_closed_on_malformed_subagent_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep malformed LLM subagent requests inside node failure handling."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Verify current portable power station options.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "evidence",
            "subagent": "evidence",
            "action": "search",
            "objective": "Verify current portable power station options.",
            "payload": {"queries": ["portable power station NZ"]},
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence unavailable.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
        planned_graph=_graph_with_evidence_node(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["graph"]["nodes"]["evidence"]["status"] == "blocked"
    assert packet["knowledge_still_lacking"] == [
        "unresolved branch: Objective for evidence"
    ]
    assert packet["trace_summary"]["subagent_call_log"][0]["subagent"] == (
        "evidence"
    )


@pytest.mark.asyncio
async def test_resolve_complex_task_records_unresolved_graph_knowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unresolved graph state visible as semantic knowledge gaps."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare agent harnesses across five dimensions.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Evidence is available as a compact summary.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Collect source-backed facts.",
                "kind": "evidence_need",
            },
            {
                "objective": "Synthesize comparison.",
                "kind": "synthesis",
            },
        ],
    }])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Source facts collected.",
            "answer_text": "Source facts collected.",
            "source_backed_facts": ["Source set collected."],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "not semantically duplicate",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The comparison is complete.",
        "knowledge_we_know_so_far": ["Source set collected."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["knowledge_still_lacking"] == [
        "unresolved branch: Collect source-backed facts.",
        "unresolved branch: Synthesize comparison.",
    ]
    assert packet["knowledge_we_know_so_far"] == ["Source set collected."]
    assert packet["investigation_summary"] == "The comparison is complete."
    assert packet["evidence_boundary_notes"][0] == (
        "Resolver-local subagent could not complete this node."
    )


@pytest.mark.asyncio
async def test_resolve_complex_task_does_not_collapse_synthesis_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthesis pending instead of collapsing it into evidence output."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare agent harnesses across five dimensions.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Evidence is available as a compact summary.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Synthesis branch resolved.",
            "answer_text": "Synthesis branch resolved.",
            "source_backed_facts": [],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The comparison is complete.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
        planned_graph=_graph_with_pending_synthesis(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["trace_summary"]["collapse_count"] == 0
    assert packet["graph"]["nodes"]["synthesis"]["status"] == "resolved"
    assert collapse.calls == []


@pytest.mark.asyncio
async def test_resolve_complex_task_records_node_evidence_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep node-authored source boundaries in the final semantic packet."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare current agent harness evidence.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Evidence is available only as prose.",
        "persona_context_summary": "Return factual structure, not dialog.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([])
    node_resolver = _StageInvoker([])
    collapse = _StageInvoker([])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The comparison is complete.",
        "knowledge_we_know_so_far": [
            "Harness feature claims were collected.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
        planned_graph=_graph_with_resolved_untraced_evidence(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["investigation_summary"] == "The comparison is complete."
    assert packet["knowledge_we_know_so_far"] == [
        "Harness feature claims were collected."
    ]
    assert packet["evidence_boundary_notes"] == [
        "Evidence was collected without structured source handles.",
    ]
    assert packet["graph"]["nodes"]["evidence"]["knowledge_we_know_so_far"] == [
        "Harness feature claims were collected."
    ]
    assert synthesizer.calls[0]["node_evidence_boundary_notes"] == [
        "Evidence was collected without structured source handles.",
    ]


@pytest.mark.asyncio
async def test_resolve_complex_task_ignores_extra_node_update_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail the graph when an LLM node update has harmless extras."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve a node update with extra LLM fields.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "investigation_summary": "Extra fields did not break node merge.",
            "knowledge_we_know_so_far": [
                "Extra fields did not break node merge.",
            ],
            "knowledge_still_lacking": [],
            "evidence_boundary_notes": [],
            "result_summary": "Legacy extra field ignored.",
            "answer_text": "Legacy extra field ignored.",
            "source_backed_facts": [],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
            "recommended_next_iteration": [
                "This field belongs to synthesis, not node_update.",
            ],
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "The node update was resolved.",
        "knowledge_we_know_so_far": [
            "Extra fields did not break node merge.",
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=_StageInvoker([{
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "not duplicate",
            },
        }]),
        synthesizer=synthesizer,
        limits={"max_iterations": 1},
        planned_graph=_graph_with_duplicate_branch(),
    )

    packet = await resolve_complex_task(request, context, options)
    branch = packet["graph"]["nodes"]["branch_a"]

    assert branch["status"] == "resolved"
    assert branch["investigation_summary"] == (
        "Extra fields did not break node merge."
    )
    assert branch["knowledge_we_know_so_far"] == [
        "Extra fields did not break node merge.",
    ]
    assert branch["recommended_next_iteration"] == [
        "This field belongs to synthesis, not node_update.",
    ]
    assert "failure_stage" not in packet["trace_summary"]


@pytest.mark.asyncio
async def test_resolve_complex_task_drops_non_string_node_projection_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop malformed semantic rows from LLM node updates without failing."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve malformed semantic rows.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "investigation_summary": "Mixed rows were normalized.",
            "knowledge_we_know_so_far": [
                {"claim": "object row from local LLM"},
                "String row survives.",
            ],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Malformed rows did not fail the packet.",
        "knowledge_we_know_so_far": ["String row survives."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=_StageInvoker([{
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "not duplicate",
            },
        }]),
        synthesizer=synthesizer,
        limits={"max_iterations": 1},
        planned_graph=_graph_with_duplicate_branch(),
    )

    packet = await resolve_complex_task(request, context, options)
    branch = packet["graph"]["nodes"]["branch_a"]

    assert branch["status"] == "resolved"
    assert branch["knowledge_we_know_so_far"] == ["String row survives."]
    assert "failure_stage" not in packet["trace_summary"]


@pytest.mark.asyncio
async def test_resolve_complex_task_returns_failed_packet_on_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a failed packet when an internal stage emits invalid structure."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve a malformed internal planner response.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Task missing a required kind.",
        }],
    }])
    node_resolver = _StageInvoker([])
    collapse = _StageInvoker([])
    synthesizer = _StageInvoker([])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1},
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["investigation_summary"].startswith(
        "The resolver failed before completing public answer research"
    )
    assert packet["root_question"] == request["objective"]
    assert packet["graph"]["nodes"]["root"]["status"] == "cannot_answer"
    assert packet["trace_summary"]["failure_stage"] == "internal_resolution"
    assert "planner task kind" in packet["knowledge_still_lacking"][0]


@pytest.mark.asyncio
async def test_resolve_complex_task_rejects_planner_graph_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject full graph output from the planner LLM stage."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve a planner shortcut response.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "graph": _graph_with_math_node(),
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=_StageInvoker([]),
        collapse=_StageInvoker([]),
        synthesizer=_StageInvoker([]),
        limits={"max_iterations": 1},
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["trace_summary"]["failure_stage"] == "internal_resolution"
    assert "planner response: missing tasks" in (
        packet["knowledge_still_lacking"][0]
    )


@pytest.mark.asyncio
async def test_resolve_complex_task_rejects_legacy_collapse_update_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject collapse-stage node update shortcuts."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve duplicate documentation branches.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "result_summary": "Branch A resolved to the same docs.",
            "answer_text": "Both branches found the same documentation.",
            "source_backed_facts": [],
            "assumptions_or_inferences": [],
            "cannot_answer_reason": None,
        },
    }])
    collapse = _StageInvoker([{
        "node_updates": {
            "branch_a": {
                "status": "collapsed",
                "collapsed_into": "branch_b",
            },
        },
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=_StageInvoker([]),
        limits={"max_iterations": 1},
        planned_graph=_graph_with_duplicate_branch(),
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["trace_summary"]["failure_stage"] == "internal_resolution"
    assert "collapse response: missing collapse_decision" in (
        packet["knowledge_still_lacking"][0]
    )


@pytest.mark.asyncio
async def test_evidence_subagent_receives_public_resolver_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass validated public context through resolver-local subagent IO."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Collect current source evidence.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "Use current review context.",
        "persona_context_summary": "Return factual structure.",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    recording_subagent = _ContextRecordingEvidenceSubagent()
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Collect current source evidence.",
            "kind": "evidence_need",
        }],
    }])
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "task_1",
            "subagent": "evidence",
            "action": "collect_evidence",
            "objective": "Collect current source evidence.",
            "payload": {"query": "current source evidence"},
            "constraints": {},
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Evidence was collected.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=_StageInvoker([]),
        synthesizer=synthesizer,
        limits={"max_iterations": 1},
    )
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        lambda: {
            "algorithmic": AlgorithmicSubagent(),
            "evidence": recording_subagent,
        },
    )

    packet = await resolve_complex_task(request, context, options)

    assert recording_subagent.contexts[0]["time_context"] == {
        "current_date": "2026-06-30",
    }
    assert recording_subagent.contexts[0]["root_question"] == request["objective"]
    evidence_node = packet["graph"]["nodes"]["task_1"]
    assert evidence_node["investigation_summary"] == (
        "Evidence prose returned without handles."
    )
    assert evidence_node["knowledge_we_know_so_far"] == [
        "Evidence prose returned without handles."
    ]
    assert packet["knowledge_we_know_so_far"] == [
        "Evidence prose returned without handles."
    ]
    assert (
        "Evidence subagent returned prose summary; preserve embedded caveats "
        "instead of treating every sentence as confirmed knowledge."
    ) in evidence_node["evidence_boundary_notes"]
    assert packet["evidence_boundary_notes"] == [
        "Resolver-local subagent produced bounded output.",
        (
            "Evidence subagent returned prose summary; preserve embedded "
            "caveats instead of treating every sentence as confirmed knowledge."
        ),
    ]


@pytest.mark.asyncio
async def test_active_node_followup_tasks_create_bounded_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create child nodes only from structured active-node follow-up tasks."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare two public benchmark claims.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Resolve benchmark claim coverage.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": (
                    "The parent knows a narrower source is needed."
                ),
                "knowledge_we_know_so_far": [],
                "knowledge_still_lacking": [
                    "source-backed throughput evidence"
                ],
                "recommended_next_iteration": [
                    "Search public benchmark evidence."
                ],
                "evidence_boundary_notes": [],
            },
            "followup_tasks": [{
                "schema_version": "complex_task_followup_task.v1",
                "objective": (
                    "Find source-backed throughput evidence for the benchmark."
                ),
                "kind": "subtask",
                "reason": "The parent node lacks source-backed throughput.",
            }],
        },
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": "Follow-up evidence was bounded.",
                "knowledge_we_know_so_far": [
                    "No public source was available in the test context."
                ],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
        },
    ])
    collapse = _StageInvoker([
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
    ])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Follow-up child was executed.",
        "knowledge_we_know_so_far": [
            "No public source was available in the test context."
        ],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_nodes": 8, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)
    graph = packet["graph"]
    parent = graph["nodes"]["task_1"]
    child_id = parent["children"][0]
    child = graph["nodes"][child_id]

    assert parent["status"] == "expanded"
    assert child["parent_id"] == "task_1"
    assert child["node_kind"] == "subtask"
    assert child["status"] == "resolved"
    assert graph["traversal_order"][-2:] == ["task_1", child_id]
    followup_event = packet["trace_summary"]["followup_event_log"][0]
    assert followup_event["event"] == "created"
    assert followup_event["source_stage"] == "active_node_resolver"
    assert followup_event["parent_node_id"] == "task_1"
    assert followup_event["child_node_id"] == child_id
    assert followup_event["objective"] == (
        "Find source-backed throughput evidence for the benchmark."
    )
    assert followup_event["kind"] == "subtask"


@pytest.mark.asyncio
async def test_synthesis_followup_tasks_continue_before_final_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run synthesis-level follow-up nodes before returning the final packet."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Compare public inference options.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Collect initial comparison evidence.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": "Initial evidence was incomplete.",
                "knowledge_we_know_so_far": ["Initial public fact found."],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
        },
        {
            "node_update": {
                "status": "resolved",
                "investigation_summary": "Synthesis follow-up was resolved.",
                "knowledge_we_know_so_far": [
                    "Follow-up public fact found."
                ],
                "knowledge_still_lacking": [],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
        },
    ])
    collapse = _StageInvoker([
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no match",
            },
        },
    ])
    synthesizer = _StageInvoker([
        {
            "investigation_summary": "Initial synthesis found a missing branch.",
            "knowledge_we_know_so_far": ["Initial public fact found."],
            "knowledge_still_lacking": ["Need one narrower evidence target."],
            "recommended_next_iteration": [
                "Search the narrower public evidence target."
            ],
            "followup_tasks": [{
                "schema_version": "complex_task_followup_task.v1",
                "objective": "Resolve the narrower public evidence target.",
                "kind": "subtask",
                "reason": "Required before final bottom-up synthesis.",
            }],
            "evidence_boundary_notes": [],
        },
        {
            "investigation_summary": "Final synthesis used the follow-up node.",
            "knowledge_we_know_so_far": [
                "Initial public fact found.",
                "Follow-up public fact found.",
            ],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
    ])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_nodes": 8, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)
    graph = packet["graph"]
    root = graph["nodes"]["root"]
    followup_id = root["children"][1]

    assert root["children"] == ["task_1", followup_id]
    assert graph["nodes"][followup_id]["node_kind"] == "subtask"
    assert graph["nodes"][followup_id]["status"] == "resolved"
    assert packet["investigation_summary"] == (
        "Final synthesis used the follow-up node."
    )
    assert packet["trace_summary"]["iterations"] == 2
    followup_event = packet["trace_summary"]["followup_event_log"][-1]
    assert followup_event["event"] == "created"
    assert followup_event["source_stage"] == "bottom_up_synthesis"
    assert followup_event["parent_node_id"] == "root"
    assert followup_event["child_node_id"] == followup_id
    assert followup_event["objective"] == (
        "Resolve the narrower public evidence target."
    )
    assert followup_event["kind"] == "subtask"


@pytest.mark.asyncio
async def test_synthesis_dependency_followup_tasks_are_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute synthesis follow-ups even when earlier prerequisites failed."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Find current versions and recommend conservative targets.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {
                "objective": "Find current version facts.",
                "kind": "evidence_need",
            },
            {
                "objective": "Synthesize conservative targets.",
                "kind": "synthesis",
            },
        ],
    }])
    node_resolver = _StageInvoker([
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "task_1",
                "subagent": "evidence",
                "action": "collect_versions",
                "objective": "Find current version facts.",
                "payload": {"query": "current version facts"},
                "constraints": {},
            },
        },
        {
            "node_update": {
                "status": "blocked",
                "investigation_summary": (
                    "The synthesis node needs narrower evidence."
                ),
                "knowledge_we_know_so_far": [],
                "knowledge_still_lacking": ["current version facts"],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
            "followup_tasks": [{
                "schema_version": "complex_task_followup_task.v1",
                "objective": "Retrieve product-specific version facts.",
                "kind": "evidence_need",
                "reason": "The first evidence branch failed too broadly.",
            }],
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "task_2_1",
                "subagent": "evidence",
                "action": "collect_versions",
                "objective": "Retrieve product-specific version facts.",
                "payload": {"query": "product-specific version facts"},
                "constraints": {},
            },
        },
    ])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Follow-up evidence was attempted.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=_StageInvoker([]),
        synthesizer=synthesizer,
        limits={
            "max_iterations": 3,
            "max_nodes": 5,
            "max_depth": 2,
        },
    )
    sequenced_subagent = _SequencedEvidenceSubagent([
        {
            "schema_version": "complex_task_subagent_result.v1",
            "resolved": False,
            "status": "partial",
            "result": {"summary": "Broad retrieval did not find facts."},
            "cache": {"enabled": False},
            "trace": {"result": "broad_failure"},
            "unresolved_items": ["current version facts"],
        },
        {
            "schema_version": "complex_task_subagent_result.v1",
            "resolved": True,
            "status": "resolved",
            "result": {"summary": "Product-specific facts were found."},
            "cache": {"enabled": False},
            "trace": {"result": "followup_success"},
            "unresolved_items": [],
        },
    ])
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        lambda: {
            "algorithmic": AlgorithmicSubagent(),
            "evidence": sequenced_subagent,
        },
    )

    packet = await resolve_complex_task(request, context, options)

    graph = packet["graph"]
    assert graph["nodes"]["task_2"]["status"] == "expanded"
    assert graph["nodes"]["task_2"]["children"] == ["task_2_1"]
    assert graph["nodes"]["task_2_1"]["status"] == "resolved"
    assert len(sequenced_subagent.requests) == 2
    followup_events = packet["trace_summary"]["followup_event_log"]
    assert followup_events[0]["event"] == "created"
    assert followup_events[0]["source_key"] == "task_2"
    assert followup_events[0]["child_node_id"] == "task_2_1"


@pytest.mark.asyncio
async def test_packet_preserves_resolved_subagent_trace_from_blocked_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep resolved subagent facts even if graph dependencies block the node."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Estimate call savings from semantic collapse.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    graph = _graph_with_blocked_prerequisite_and_pending_synthesis()
    graph["nodes"]["synthesis"]["objective"] = (
        "Calculate total calls saved from 30 cases, 8 nodes each, and 35% "
        "saved duplicated calls. Operand values available: 30, 8, 0.35."
    )
    node_resolver = _StageInvoker([{
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": "synthesis",
            "subagent": "algorithmic",
            "action": "evaluate_expression",
            "objective": "Calculate total calls saved.",
            "payload": {
                "expression": "30 * 8 * 0.35",
                "label": "total_calls_saved",
                "input_values": [
                    {
                        "label": "case_count",
                        "value": "30",
                        "source_node_id": "synthesis",
                        "source_text": (
                            "Operand values available: 30, 8, 0.35."
                        ),
                    },
                    {
                        "label": "nodes_per_case",
                        "value": "8",
                        "source_node_id": "synthesis",
                        "source_text": (
                            "Operand values available: 30, 8, 0.35."
                        ),
                    },
                    {
                        "label": "savings_rate",
                        "value": "0.35",
                        "source_node_id": "synthesis",
                        "source_text": (
                            "Operand values available: 30, 8, 0.35."
                        ),
                    },
                ],
                "formula_constants": [],
            },
            "constraints": {},
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": (
            "Synthesis was blocked by an unresolved prerequisite."
        ),
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": ["prerequisite branch"],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=_StageInvoker([]),
        node_resolver=node_resolver,
        collapse=_StageInvoker([]),
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_subagent_attempts": 1},
        planned_graph=graph,
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["graph"]["nodes"]["synthesis"]["status"] == "blocked"
    assert packet["trace_summary"]["subagent_call_log"][0]["resolved"] is True
    assert packet["knowledge_we_know_so_far"] == [
        "total_calls_saved: 30 * 8 * 0.35 = 84.0",
    ]


@pytest.mark.asyncio
async def test_default_traversal_budget_rejects_fifth_followup_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep default traversal bounded to the hard Phase 1 iteration cap."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Find three current facts and repair one missing branch.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [
            {"objective": "Resolve fact A.", "kind": "subtask"},
            {"objective": "Resolve fact B.", "kind": "subtask"},
            {"objective": "Resolve fact C.", "kind": "subtask"},
            {"objective": "Synthesize the facts.", "kind": "synthesis"},
        ],
    }])
    resolved_update = {
        "node_update": {
            "status": "resolved",
            "investigation_summary": "One initial branch resolved.",
            "knowledge_we_know_so_far": ["one branch fact"],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
    }
    node_resolver = _StageInvoker([
        resolved_update,
        resolved_update,
        resolved_update,
        {
            "node_update": {
                "status": "blocked",
                "investigation_summary": "Synthesis needs a narrower repair.",
                "knowledge_we_know_so_far": [],
                "knowledge_still_lacking": ["narrower repair fact"],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [],
            },
            "followup_tasks": [{
                "schema_version": "complex_task_followup_task.v1",
                "objective": "Resolve the narrower repair fact.",
                "kind": "evidence_need",
                "reason": "The initial trunk left one narrow gap.",
            }],
        },
        {
            "subagent_request": {
                "schema_version": "complex_task_subagent_request.v1",
                "node_id": "task_4_1",
                "subagent": "evidence",
                "action": "retrieve_source_facts",
                "objective": "Resolve the narrower repair fact.",
                "payload": {"query": "narrower repair fact"},
                "constraints": {},
            },
        },
    ])
    collapse = _StageInvoker([
        {
            "collapse_decision": {
                "should_collapse": False,
                "target_node_id": "",
                "reason": "no duplicate",
            },
        },
    ] * 5)
    synthesizer = _StageInvoker([{
        "investigation_summary": "Initial facts and repair fact were resolved.",
        "knowledge_we_know_so_far": ["narrower repair fact"],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_nodes": 6, "max_depth": 2},
    )
    sequenced_subagent = _SequencedEvidenceSubagent([{
        "schema_version": "complex_task_subagent_result.v1",
        "resolved": True,
        "status": "resolved",
        "result": {"summary": "Narrower repair fact resolved."},
        "cache": {"enabled": False},
        "trace": {"result": "followup_success"},
        "unresolved_items": [],
    }])
    monkeypatch.setattr(
        resolver_service,
        "_internal_subagents",
        lambda: {
            "algorithmic": AlgorithmicSubagent(),
            "evidence": sequenced_subagent,
        },
    )

    packet = await resolve_complex_task(request, context, options)

    graph = packet["graph"]
    assert "task_4_1" not in graph["nodes"]
    assert graph["nodes"]["task_4"]["children"] == []
    assert graph["nodes"]["task_4"]["status"] == "blocked"
    assert packet["trace_summary"]["iterations"] == 4
    assert packet["trace_summary"]["subagent_calls"] == 0
    assert packet["trace_summary"]["followup_created_count"] == 0
    assert packet["trace_summary"]["followup_rejected_count"] == 1
    assert "follow-up task not created" in "\n".join(
        packet["knowledge_still_lacking"]
    )


@pytest.mark.asyncio
async def test_recommendation_prose_does_not_create_followup_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep semantic recommendations from becoming executable control flow."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Inspect a recommendation-only result.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Summarize available evidence.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "investigation_summary": "Only a prose recommendation exists.",
            "knowledge_we_know_so_far": ["Known fact."],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [
                "Search public evidence for a narrower benchmark."
            ],
            "evidence_boundary_notes": [],
        },
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Recommendation stayed semantic.",
        "knowledge_we_know_so_far": ["Known fact."],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [
            "Search public evidence for a narrower benchmark."
        ],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 3, "max_nodes": 8, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)

    assert packet["graph"]["nodes"]["root"]["children"] == ["task_1"]
    assert "followup_event_log" in packet["trace_summary"]
    assert packet["trace_summary"]["followup_event_log"] == []
    assert packet["recommended_next_iteration"] == [
        "Search public evidence for a narrower benchmark."
    ]


@pytest.mark.asyncio
async def test_followup_limit_rejection_preserves_semantic_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject over-limit follow-up tasks without dropping the missing work."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve one bounded branch.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Resolve branch before hitting node cap.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "investigation_summary": "A follow-up would exceed graph limits.",
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        "followup_tasks": [{
            "schema_version": "complex_task_followup_task.v1",
            "objective": "Resolve rejected child task.",
            "kind": "subtask",
            "reason": "Needed but graph node cap is already exhausted.",
        }],
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Rejected follow-up was preserved.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 2, "max_nodes": 2, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]

    assert node["children"] == []
    assert node["knowledge_still_lacking"] == [
        "follow-up task not created: Resolve rejected child task."
    ]
    assert node["evidence_boundary_notes"] == [
        "Resolver follow-up task rejected: followup_tasks: exceeds max_nodes"
    ]
    followup_event = packet["trace_summary"]["followup_event_log"][0]
    assert followup_event["event"] == "rejected"
    assert followup_event["source_stage"] == "active_node_resolver"
    assert followup_event["parent_node_id"] == "task_1"
    assert followup_event["objective"] == "Resolve rejected child task."
    assert followup_event["kind"] == "subtask"
    assert followup_event["reason"] == "followup_tasks: exceeds max_nodes"


@pytest.mark.asyncio
async def test_active_followup_at_iteration_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not create active follow-up children that cannot be traversed."""

    request = validate_complex_task_resolver_request({
        "schema_version": COMPLEX_TASK_RESOLVER_REQUEST_VERSION,
        "objective": "Resolve follow-up at traversal cap.",
        "reason": "service test",
        "source": "test",
        "priority": "review",
    })
    context = validate_complex_task_resolver_context({
        "schema_version": COMPLEX_TASK_RESOLVER_CONTEXT_VERSION,
        "conversation_summary": "",
        "persona_context_summary": "",
        "time_context": {"current_date": "2026-06-30"},
        "available_evidence": [],
    })
    planner = _StageInvoker([{
        "tasks": [{
            "objective": "Create a child too late.",
            "kind": "subtask",
        }],
    }])
    node_resolver = _StageInvoker([{
        "node_update": {
            "status": "resolved",
            "investigation_summary": "A follow-up was requested too late.",
            "knowledge_we_know_so_far": [],
            "knowledge_still_lacking": [],
            "recommended_next_iteration": [],
            "evidence_boundary_notes": [],
        },
        "followup_tasks": [{
            "schema_version": "complex_task_followup_task.v1",
            "objective": "Resolve child after traversal cap.",
            "kind": "subtask",
            "reason": "No traversal iteration remains for this child.",
        }],
    }])
    collapse = _StageInvoker([{
        "collapse_decision": {
            "should_collapse": False,
            "target_node_id": "",
            "reason": "no match",
        },
    }])
    synthesizer = _StageInvoker([{
        "investigation_summary": "Late follow-up was rejected.",
        "knowledge_we_know_so_far": [],
        "knowledge_still_lacking": [],
        "recommended_next_iteration": [],
        "evidence_boundary_notes": [],
    }])
    options = _install_stage_invokers(
        monkeypatch,
        planner=planner,
        node_resolver=node_resolver,
        collapse=collapse,
        synthesizer=synthesizer,
        limits={"max_iterations": 1, "max_nodes": 8, "max_depth": 3},
    )

    packet = await resolve_complex_task(request, context, options)
    node = packet["graph"]["nodes"]["task_1"]
    followup_event = packet["trace_summary"]["followup_event_log"][0]

    assert node["children"] == []
    assert node["knowledge_still_lacking"] == [
        "follow-up task not created: Resolve child after traversal cap."
    ]
    assert followup_event["event"] == "rejected"
    assert followup_event["reason"] == (
        "active-node follow-up rejected at max_iterations"
    )


def _graph_with_math_node() -> dict[str, object]:
    """Build a root graph with one pending algorithmic subtask."""

    root = _node(
        node_id="root",
        parent_id=None,
        depth=0,
        node_kind="root",
        status="expanded",
        children=["math_1"],
    )
    math_node = _node(
        node_id="math_1",
        parent_id="root",
        depth=1,
        node_kind="algorithmic_task",
        status="pending",
        children=[],
        knowledge_we_know_so_far=[
            "Operand values available: 0.70, 0.6, 0.95, 0.4, 45, 6."
        ],
    )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "math_1",
        "nodes": {
            "root": root,
            "math_1": math_node,
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _graph_with_resolved_untraced_evidence() -> dict[str, object]:
    """Build a graph with resolved evidence facts but no source handles."""

    evidence = _node(
        node_id="evidence",
        parent_id="root",
        depth=1,
        node_kind="evidence_need",
        status="resolved",
        children=[],
        investigation_summary="Evidence collected.",
        knowledge_we_know_so_far=[
            "Harness feature claims were collected.",
        ],
        evidence_boundary_notes=[
            "Evidence was collected without structured source handles.",
        ],
    )
    synthesis = _node(
        node_id="synthesis",
        parent_id="root",
        depth=1,
        node_kind="synthesis",
        status="resolved",
        children=[],
        investigation_summary="Comparison synthesized.",
    )
    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "synthesis",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["evidence", "synthesis"],
            ),
            "evidence": evidence,
            "synthesis": synthesis,
        },
        "traversal_order": ["root", "evidence", "synthesis"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _graph_with_three_evidence_nodes() -> dict[str, object]:
    """Build a root graph with three pending public evidence leaves."""

    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "evidence_a",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["evidence_a", "evidence_b", "evidence_c"],
            ),
            "evidence_a": _node(
                node_id="evidence_a",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="pending",
                children=[],
            ),
            "evidence_b": _node(
                node_id="evidence_b",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="pending",
                children=[],
            ),
            "evidence_c": _node(
                node_id="evidence_c",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="pending",
                children=[],
            ),
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 4,
        "max_depth": 1,
    }
    return graph


def _graph_with_evidence_node() -> dict[str, object]:
    """Build a root graph with one pending evidence subtask."""

    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "evidence",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["evidence"],
            ),
            "evidence": _node(
                node_id="evidence",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="pending",
                children=[],
            ),
        },
        "traversal_order": ["root"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _evidence_subagent_response(node_id: str) -> dict[str, object]:
    """Build an active-node response that requests public evidence."""

    response = {
        "subagent_request": {
            "schema_version": "complex_task_subagent_request.v1",
            "node_id": node_id,
            "subagent": "evidence",
            "action": "retrieve_source_facts",
            "objective": f"Objective for {node_id}",
            "payload": {"query": f"query for {node_id}"},
            "constraints": {},
        },
    }
    return response


def _resolved_evidence_subagent_result(summary: str) -> dict[str, object]:
    """Build a resolved evidence result for sequenced subagent tests."""

    result = {
        "schema_version": "complex_task_subagent_result.v1",
        "resolved": True,
        "status": "resolved",
        "result": {"summary": summary},
        "cache": {"enabled": False},
        "trace": {"result": summary},
        "unresolved_items": [],
    }
    return result


def _graph_with_blocked_prerequisite_and_pending_synthesis() -> dict[str, object]:
    """Build a graph where synthesis depends on a blocked prerequisite."""

    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "synthesis",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["evidence", "synthesis"],
            ),
            "evidence": _node(
                node_id="evidence",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="blocked",
                children=[],
            ),
            "synthesis": _node(
                node_id="synthesis",
                parent_id="root",
                depth=1,
                node_kind="synthesis",
                status="pending",
                children=[],
            ),
        },
        "traversal_order": ["root", "evidence"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _graph_with_pending_synthesis() -> dict[str, object]:
    """Build a graph where synthesis is the active node."""

    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "synthesis",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["evidence", "synthesis"],
            ),
            "evidence": _node(
                node_id="evidence",
                parent_id="root",
                depth=1,
                node_kind="evidence_need",
                status="resolved",
                children=[],
                investigation_summary="Evidence branch resolved.",
                knowledge_we_know_so_far=["Evidence branch resolved."],
            ),
            "synthesis": _node(
                node_id="synthesis",
                parent_id="root",
                depth=1,
                node_kind="synthesis",
                status="pending",
                children=[],
            ),
        },
        "traversal_order": ["root", "evidence"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _graph_with_duplicate_branch() -> dict[str, object]:
    """Build a graph with one pending branch and one resolved candidate."""

    graph = {
        "schema_version": COMPLEX_TASK_GRAPH_VERSION,
        "root_node_id": "root",
        "active_node_id": "branch_a",
        "nodes": {
            "root": _node(
                node_id="root",
                parent_id=None,
                depth=0,
                node_kind="root",
                status="expanded",
                children=["branch_a", "branch_b"],
            ),
            "branch_a": _node(
                node_id="branch_a",
                parent_id="root",
                depth=1,
                node_kind="subtask",
                status="pending",
                children=[],
            ),
            "branch_b": _node(
                node_id="branch_b",
                parent_id="root",
                depth=1,
                node_kind="subtask",
                status="resolved",
                children=[],
                investigation_summary="Official docs already resolved.",
                knowledge_we_know_so_far=["Official docs located."],
            ),
        },
        "traversal_order": ["root", "branch_b"],
        "collapse_events": [],
        "max_nodes": 8,
        "max_depth": 3,
    }
    return graph


def _node(
    *,
    node_id: str,
    parent_id: str | None,
    depth: int,
    node_kind: str,
    status: str,
    children: list[str],
    investigation_summary: str = "",
    knowledge_we_know_so_far: list[str] | None = None,
    knowledge_still_lacking: list[str] | None = None,
    recommended_next_iteration: list[str] | None = None,
    evidence_boundary_notes: list[str] | None = None,
) -> dict[str, object]:
    """Build a valid graph node for service tests."""

    known_rows = knowledge_we_know_so_far or []
    lacking_rows = knowledge_still_lacking or []
    next_rows = recommended_next_iteration or []
    boundary_rows = evidence_boundary_notes or []
    node = {
        "schema_version": COMPLEX_TASK_NODE_VERSION,
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "objective": f"Objective for {node_id}",
        "node_kind": node_kind,
        "status": status,
        "children": children,
        "investigation_summary": investigation_summary,
        "knowledge_we_know_so_far": known_rows,
        "knowledge_still_lacking": lacking_rows,
        "recommended_next_iteration": next_rows,
        "evidence_boundary_notes": boundary_rows,
        "evidence_refs": [],
        "source_observation_ids": [],
        "collapsed_into": None,
        "attempts": [],
    }
    return node
