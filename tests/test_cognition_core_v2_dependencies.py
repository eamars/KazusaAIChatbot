"""Checkpoint E dependency, branch, overlap, and collapse tests."""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import select_route
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
    MAX_GOAL_BRANCHES,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import (
    DependencyGraphError,
    build_dependency_graph,
    build_dependency_levels,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    execute_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids


def _bid(branch_id: str, route: str = "speech") -> dict[str, object]:
    """Build one complete bid for workspace and route tests."""

    return {
        "branch_id": branch_id,
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": f"intention from {branch_id}",
        "desired_outcome": "grounded outcome",
        "concrete_detail": "bounded detail",
        "reason": "typed evidence supports this branch",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["preserve continuity"],
        "confidence": "high",
        "requested_route": route,
    }


def test_dependency_levels_release_internal_ready_branches() -> None:
    """Independent branches overlap and dependents wait for internal results."""

    branches = [
        BranchDefinition("meaning", (), ("reflect",)),
        BranchDefinition("safety", (), ("protect",)),
        BranchDefinition("repair", ("meaning",), ("repair",)),
    ]

    assert build_dependency_levels(branches) == (
        ("meaning", "safety"),
        ("repair",),
    )


def test_dependency_graph_accepts_declared_question_dependencies() -> None:
    """Question dependencies become ready only when their family completes."""

    branch = BranchDefinition(
        "relationship_connection",
        ("q:relationship_social",),
        ("connect",),
    )
    graph = build_dependency_graph(
        [branch],
        external_dependencies={"q:relationship_social"},
    )

    assert graph.ready_branch_ids(set(), set(), set()) == []
    assert graph.ready_branch_ids(
        set(),
        set(),
        set(),
        {"q:relationship_social"},
    ) == ["relationship_connection"]


def test_dependency_graph_rejects_an_undeclared_dependency() -> None:
    """Unknown refs fail before execution begins."""

    branches = [
        BranchDefinition("repair", ("missing",), ("repair",)),
    ]

    with pytest.raises(DependencyGraphError):
        build_dependency_levels(branches)


def test_fourteen_branch_registry_and_jealousy_dependency_are_frozen() -> None:
    """All approved branches remain available with explicit family ownership."""

    assert MAX_GOAL_BRANCHES == 14
    assert len(DEFAULT_BRANCH_DEFINITIONS) == 14
    assert DEFAULT_BRANCH_DEFINITIONS["trust_verification"].dependencies == (
        "q:relationship_social",
        "q:goal_threat_outcome",
    )
    assert DEFAULT_BRANCH_DEFINITIONS["bond_protection"].dependencies == (
        "q:relationship_social",
        "q:goal_threat_outcome",
    )


@pytest.mark.asyncio
async def test_independent_branches_overlap_without_code_call_cap() -> None:
    """All dependency-ready calls start together and no code cap is present."""

    definitions = [
        BranchDefinition("first", (), ("reflect",)),
        BranchDefinition("second", (), ("protect",)),
        BranchDefinition("dependent", ("first",), ("repair",)),
    ]
    graph = build_dependency_graph(definitions)
    independent_started: set[str] = set()
    simultaneous_start = asyncio.Event()
    dependent_started_after: list[set[str]] = []

    async def handler(definition: BranchDefinition) -> dict[str, str]:
        if definition.branch_id in {"first", "second"}:
            independent_started.add(definition.branch_id)
            if len(independent_started) == 2:
                simultaneous_start.set()
            await simultaneous_start.wait()
        else:
            dependent_started_after.append(set(independent_started))
        await asyncio.sleep(0)
        return {"branch_id": definition.branch_id}

    execution = await asyncio.wait_for(
        execute_dependency_graph(graph, handler),
        timeout=0.5,
    )

    assert execution.maximum_concurrency == 2
    assert set(execution.results) == {"first", "second", "dependent"}
    assert dependent_started_after == [{"first", "second"}]
    source = inspect.getsource(execute_dependency_graph)
    assert "Semaphore" not in source
    assert "concurrency_cap" not in source


@pytest.mark.asyncio
async def test_branch_failure_isolated_from_successful_slots() -> None:
    """A failed branch warns and its dependent is skipped without losing siblings."""

    definitions = [
        BranchDefinition("successful", (), ("reflect",)),
        BranchDefinition("failing", (), ("protect",)),
        BranchDefinition("blocked", ("failing",), ("repair",)),
    ]
    graph = build_dependency_graph(definitions)

    async def handler(definition: BranchDefinition) -> dict[str, str]:
        if definition.branch_id == "failing":
            raise RuntimeError("patched branch failure")
        return {"branch_id": definition.branch_id}

    execution = await execute_dependency_graph(graph, handler)

    assert set(execution.results) == {"successful"}
    assert "blocked" in execution.failed_branch_ids
    assert any("patched branch failure" in warning for warning in execution.warnings)


@pytest.mark.asyncio
async def test_collapse_copies_complete_bids_from_handle_partition() -> None:
    """Collapse output selects handles; code copies the complete internal bids."""

    class _LLM:
        async def ainvoke(self, messages: list[object], *, config: object) -> SimpleNamespace:
            del messages, config
            return SimpleNamespace(
                content=json.dumps({
                    "primary_bid_handle": "b1",
                    "supporting_bid_handles": ["b2"],
                    "suppressed_bid_handles": [],
                })
            )

    services = SimpleNamespace(
        llm=_LLM(),
        collapse_config=object(),
    )

    result = await collapse_bids([_bid("first"), _bid("second")], services)

    assert result["primary_bid"]["branch_id"] == "first"
    assert result["supporting_bids"][0]["reason"] == _bid("second")["reason"]


@pytest.mark.asyncio
async def test_route_selection_validates_action_availability() -> None:
    """Route-only selection cannot invent an unavailable executable action."""

    class _LLM:
        async def ainvoke(self, messages: list[object], *, config: object) -> object:
            del messages, config
            return type("Response", (), {
                "content": json.dumps({
                    "selected_bid_handle": "b1",
                    "route": "action",
                    "action_handle": "a1",
                })
            })()

    services = type("Services", (), {
        "llm": _LLM(),
        "action_selection_config": object(),
    })()
    with pytest.raises(CognitionExecutionError):
        await select_route(
            {
                **_bid("safety", route="action"),
                "requested_action_kind": "protect",
            },
            [],
            [],
            [],
            services,
        )
