"""Patched orchestration tests for V2 branch dependencies and workspace guards."""

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    validate_cognition_chain_output,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    BranchResult,
    EmotionActivation,
    EmotionDefinition,
    LocalMotivationalState,
    WorkspaceResult,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import (
    DependencyGraphError,
    build_dependency_graph,
    build_dependency_levels,
)
from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    project_v1_output,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    execute_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_workspace


BRANCH_EXECUTION_TIMEOUT_SECONDS = 0.5


def test_dependency_levels_only_release_declared_ready_branches() -> None:
    """Release independent branches together and wait for declared inputs only."""

    branches = [
        BranchDefinition("meaning", (), (), ("reflect",)),
        BranchDefinition("safety", (), (), ("protect",)),
        BranchDefinition("repair", (), ("meaning",), ("repair",)),
    ]

    levels = build_dependency_levels(branches)

    assert levels == (("meaning", "safety"), ("repair",))


def test_dependency_graph_rejects_a_missing_dependency() -> None:
    """Fail closed instead of inventing readiness for an unknown branch."""

    branches = [
        BranchDefinition("repair", (), ("missing",), ("repair",)),
    ]

    with pytest.raises(DependencyGraphError):
        build_dependency_levels(branches)


def test_workspace_cannot_select_a_suppressed_bid() -> None:
    """Allow output projection to use only bids admitted by the workspace."""

    bids = [
        BranchResult(
            branch_id="ordinary",
            action_bid={"decision": "visible_reply", "detail": "answer"},
            perceived_meaning="direct question",
            desired_outcome="answer",
            confidence="high",
        ),
        BranchResult(
            branch_id="suppressed",
            action_bid={"decision": "visible_reply", "detail": "overreach"},
            perceived_meaning="unrelated",
            desired_outcome="overreach",
            confidence="low",
        ),
    ]

    result = collapse_workspace(
        bids=bids,
        admitted_bid_ids={"ordinary"},
        selected_bid_id="suppressed",
    )

    assert result.selected_bid_id is None
    assert result.suppressed_bid_ids == ("suppressed",)


@pytest.mark.asyncio
async def test_independent_branches_overlap_and_dependencies_wait() -> None:
    """Run independent branches concurrently before releasing a dependent one."""

    definitions = [
        BranchDefinition("first", (), (), ("reflect",)),
        BranchDefinition("second", (), (), ("protect",)),
        BranchDefinition("dependent", (), ("first",), ("repair",)),
    ]
    graph = build_dependency_graph(definitions)
    independent_started: set[str] = set()
    simultaneous_start = asyncio.Event()
    dependent_started_after: list[set[str]] = []

    async def handler(definition: BranchDefinition) -> BranchResult:
        """Hold independent work until both handlers have started."""

        if definition.branch_id in {"first", "second"}:
            independent_started.add(definition.branch_id)
            if len(independent_started) == 2:
                simultaneous_start.set()
            await simultaneous_start.wait()
        else:
            dependent_started_after.append(set(independent_started))
        result = BranchResult(
            branch_id=definition.branch_id,
            action_bid={"decision": "visible_reply", "detail": "grounded"},
            perceived_meaning="patched branch",
            desired_outcome="complete test",
            confidence="high",
        )
        return result

    execution = await asyncio.wait_for(
        execute_dependency_graph(graph, handler, concurrency_cap=2),
        timeout=BRANCH_EXECUTION_TIMEOUT_SECONDS,
    )

    assert execution.maximum_concurrency == 2
    assert set(execution.results) == {"first", "second", "dependent"}
    assert dependent_started_after == [{"first", "second"}]
    assert max(
        execution.started_at["first"],
        execution.started_at["second"],
    ) < min(
        execution.ended_at["first"],
        execution.ended_at["second"],
    )
    assert execution.started_at["dependent"] >= max(
        execution.ended_at["first"],
        execution.ended_at["second"],
    )


@pytest.mark.asyncio
async def test_branch_failure_warns_and_preserves_successful_branch_results() -> None:
    """Retain successful branch bids while skipping dependents of a failure."""

    definitions = [
        BranchDefinition("successful", (), (), ("reflect",)),
        BranchDefinition("failing", (), (), ("protect",)),
        BranchDefinition("blocked", (), ("failing",), ("repair",)),
    ]
    graph = build_dependency_graph(definitions)

    async def handler(definition: BranchDefinition) -> BranchResult:
        """Simulate one branch failure without mutating shared state."""

        if definition.branch_id == "failing":
            raise RuntimeError("patched branch failure")
        result = BranchResult(
            branch_id=definition.branch_id,
            action_bid={"decision": "visible_reply", "detail": "grounded"},
            perceived_meaning="successful patched branch",
            desired_outcome="preserve valid result",
            confidence="high",
        )
        return result

    execution = await execute_dependency_graph(graph, handler)

    assert set(execution.results) == {"successful"}
    assert any("failing failed: patched branch failure" in warning for warning in execution.warnings)
    assert any(
        "blocked skipped because a dependency failed" in warning
        for warning in execution.warnings
    )


def test_output_projection_validates_a_v1_compatible_workspace_result() -> None:
    """Project authoritative state and one admitted bid through the V1 schema."""

    activations = {
        "fear": EmotionActivation(
            emotion_id="fear",
            activation=1.0,
            trend="beginning",
            causal_source_refs=("credible_threat",),
        ),
    }
    workspace = WorkspaceResult(
        selected_bid_id="safety",
        public_intention="address the credible threat",
        internal_summary="preserve safety while responding",
        suppressed_bid_ids=(),
    )

    output = project_v1_output(
        activations,
        workspace,
        [{"decision": "visible_reply", "detail": "address safety"}],
        ["one branch unavailable"],
    )
    validated_output = validate_cognition_chain_output(output)

    assert validated_output["cognition_residue"]["emotional_appraisal"] == "fear"
    assert validated_output["chain_trace"]["selected_actions_summary"] == "safety"


def test_test_only_emotion_and_branch_use_existing_extension_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove registry extension without reducer, scheduler, workspace, or facade edits."""

    from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
        DEFAULT_BRANCH_DEFINITIONS,
        select_preliminary_branches,
    )
    from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
        EMOTION_DEFINITIONS,
    )
    from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
        derive_emotion_activations,
    )

    test_definition = EmotionDefinition(
        emotion_id="test_resolve",
        causal_inputs=("test_resolution",),
        begin_guard="test root present",
        sustain_rule="test root remains",
        fade_rule="test root absent",
        action_tendencies=("resolve",),
    )
    monkeypatch.setitem(EMOTION_DEFINITIONS, "test_resolve", test_definition)
    activations = derive_emotion_activations(
        LocalMotivationalState(),
        {"test_resolution": 0.7},
    )
    extension_definitions = dict(DEFAULT_BRANCH_DEFINITIONS)
    extension_definitions["test_resolution_branch"] = BranchDefinition(
        branch_id="test_resolution_branch",
        activating_emotions=("test_resolve",),
        dependencies=(),
        action_tendencies=("resolve",),
    )

    selected_branches = select_preliminary_branches(
        activations,
        extension_definitions,
    )

    assert activations["test_resolve"].activation == 0.7
    assert [branch.branch_id for branch in selected_branches] == [
        "test_resolution_branch",
    ]
