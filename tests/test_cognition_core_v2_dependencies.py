"""Checkpoint E dependency, branch, overlap, and collapse tests."""

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
from kazusa_ai_chatbot.cognition_core_v2.facade import (
    _raise_for_failed_required_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    BranchFailure,
    ParallelExecutionResult,
    execute_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
    run_goal_cognition,
    validate_goal_bid_draft,
)
from kazusa_ai_chatbot.cognition_core_v2 import goal_cognition as goal_module
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids


def _bid(branch_id: str) -> dict[str, object]:
    """Build one complete motive bid for workspace tests."""

    return {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": f"goal:{branch_id}",
        },
        "intention": f"intention from {branch_id}",
        "desired_outcome": "grounded outcome",
        "concrete_detail": "bounded detail",
        "reason": "typed evidence supports this branch",
        "private_monologue": "I should preserve the grounded current motive.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["preserve continuity"],
        "confidence": "high",
    }


def _current_event() -> list[dict[str, str]]:
    """Build one typed current-event projection for workspace tests."""

    return [{
        "handle": "e1",
        "source_kind": "episode",
        "semantic_text": "The current user asks for current hardware prices.",
    }]


def _goal_contexts(*branch_ids: str) -> dict[str, dict[str, object]]:
    """Build bounded persistent-goal provenance keyed by bid goal ref."""

    return {
        f"goal:{branch_id}": {
            "goal_handle": f"goal:{branch_id}",
            "goal_kind": branch_id,
            "description": f"persistent matter for {branch_id}",
            "status": "pursuing",
            "salience": 10,
            "importance": 70,
            "progress": 20,
            "obstruction": 0,
            "urgency": 10,
        }
        for branch_id in branch_ids
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
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
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
        workspace_collapse_config=object(),
    )

    result = await collapse_bids(
        [_bid("first"), _bid("second")],
        services,
        current_event=_current_event(),
        goal_context_by_ref=_goal_contexts("first", "second"),
    )

    assert result["primary_bid"]["branch_id"] == "first"
    assert result["supporting_bids"][0]["reason"] == _bid("second")["reason"]


@pytest.mark.asyncio
async def test_collapse_assigns_handles_in_frozen_registry_order() -> None:
    """Input completion order cannot change branch-to-handle assignment."""

    captured: dict[str, object] = {}

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured.update(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps({
                "primary_bid_handle": "b1",
                "supporting_bid_handles": ["b2"],
                "suppressed_bid_handles": [],
            }))

    result = await collapse_bids(
        [_bid("social_care"), _bid("autonomy_boundary")],
        SimpleNamespace(llm=_LLM(), workspace_collapse_config=object()),
        current_event=_current_event(),
        goal_context_by_ref=_goal_contexts(
            "social_care",
            "autonomy_boundary",
        ),
    )

    assert captured["current_event"] == _current_event()
    assert captured["bids"]["b1"]["branch_id"] == "autonomy_boundary"
    assert captured["bids"]["b1"]["persistent_goal"] == (
        _goal_contexts("autonomy_boundary")["goal:autonomy_boundary"]
    )
    assert captured["bids"]["b1"]["intention"] == (
        "intention from autonomy_boundary"
    )
    assert result["primary_branch_id"] == "autonomy_boundary"


@pytest.mark.asyncio
async def test_collapse_suppresses_an_unrelated_persistent_goal_bid() -> None:
    """The model-authored partition can exclude a stale persistent motive."""

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            response = {
                "primary_bid_handle": "b1",
                "supporting_bid_handles": [],
                "suppressed_bid_handles": ["b2"],
            }
            return SimpleNamespace(content=json.dumps(response))

    result = await collapse_bids(
        [_bid("ordinary_response"), _bid("autonomy_boundary")],
        SimpleNamespace(llm=_LLM(), workspace_collapse_config=object()),
        current_event=_current_event(),
        goal_context_by_ref=_goal_contexts("autonomy_boundary"),
    )

    assert result["primary_branch_id"] == "ordinary_response"
    assert result["supporting_branch_ids"] == []
    assert result["suppressed_branch_ids"] == ["autonomy_boundary"]


def test_goal_bid_rejects_route_and_capability_authority() -> None:
    """Goal branches cannot pre-empt the semantic action planner."""

    draft = {
        "intention": "perform the permitted action",
        "desired_outcome": "complete the bounded work",
        "concrete_detail": "use the declared action only",
        "reason": "the evidence supports execution",
        "private_monologue": "I should use only the declared action.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the bounded work completes"],
        "confidence": "high",
        "requested_route": "action",
    }
    with pytest.raises(ValueError, match="fields are not exact"):
        validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles=set(),
        )


def test_goal_bid_evidence_limit_matches_the_nine_handle_prompt_packet() -> None:
    """All nine projected evidence handles remain valid as one goal bid."""

    evidence_handles = [f"e{index}" for index in range(1, 10)]
    draft = {
        "intention": "continue the active conversation",
        "desired_outcome": "preserve every relevant constraint",
        "concrete_detail": "use the complete projected evidence packet",
        "reason": "all projected evidence remains relevant",
        "private_monologue": "I should account for the full packet.",
        "target_role_handles": [],
        "evidence_handles": evidence_handles,
        "expected_consequences": ["the continuation stays consistent"],
        "confidence": "high",
    }

    validated = validate_goal_bid_draft(
        draft,
        evidence_handles=set(evidence_handles),
        role_handles=set(),
    )

    assert validated["evidence_handles"] == evidence_handles
    with pytest.raises(ValueError, match="evidence handles are invalid"):
        validate_goal_bid_draft(
            {
                **draft,
                "evidence_handles": [*evidence_handles, "e10"],
            },
            evidence_handles={*evidence_handles, "e10"},
            role_handles=set(),
        )


@pytest.mark.asyncio
async def test_goal_bid_gets_one_bounded_schema_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repairable extra authority field receives one LLM correction."""

    valid = {
        "intention": "respond to the direct greeting",
        "desired_outcome": "continue the addressed conversation",
        "concrete_detail": "acknowledge the participant's greeting",
        "reason": "the participant directly addressed the character",
        "private_monologue": "I want to answer their greeting warmly.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the conversation continues"],
        "confidence": "high",
        "relational_willingness": {
            "schema_version": "relational_willingness.v1",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "reason": '当前回合证据不涉及关系许可判断',
            "evidence_handles": ["e1"],
        },
    }
    responses = [
        {**valid, "requested_route": "speech"},
        valid,
    ]

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            return SimpleNamespace(
                content=json.dumps(responses[len(self.messages) - 1]),
            )

    trace_recorder = AsyncMock()
    monkeypatch.setattr(
        goal_module.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        goal_module.llm_tracing,
        "current_trace_id",
        lambda: "trace-1",
    )
    llm = _LLM()
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "g1"},
        {"_role_bindings": {}, "role_summaries": {}},
        [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-1",
                "occurred_at": "2026-07-15T00:00:00Z",
                "semantic_summary": "the participant greeted the character",
            },
            "semantic_text": "the participant greeted the character",
            "visible_to": ["q:event_agency"],
        }],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=SimpleNamespace(
                route_name="COGNITION_LLM_GOAL_ORDINARY_RESPONSE",
                model="test-model",
            ),
        ),
    )

    assert bid["branch_id"] == "ordinary_response"
    assert len(llm.messages) == 2
    assert "修复" in str(llm.messages[1][0].content)
    assert "requested_route" not in GOAL_COGNITION_PROMPT
    assert "action_handle" not in GOAL_COGNITION_PROMPT
    assert "resolver_handle" not in GOAL_COGNITION_PROMPT
    assert [
        call.kwargs["stage_name"]
        for call in trace_recorder.await_args_list
    ] == [
        "goal_cognition.ordinary_response.initial",
        "goal_cognition.ordinary_response.repair_1",
    ]
    assert trace_recorder.await_args_list[0].kwargs["parse_status"] == (
        "contract_error"
    )


@pytest.mark.asyncio
async def test_goal_bid_repair_replays_grounding_after_handle_failures() -> None:
    """Every ordinary repair receives exact grounding and schema feedback."""

    valid = {
        "intention": "respond to the current relationship-sensitive request",
        "desired_outcome": "keep the response aligned with current boundaries",
        "concrete_detail": "answer from current evidence and relationship state",
        "reason": "the current request requires a relational judgment",
        "private_monologue": "I should answer according to my current boundary.",
        "target_role_handles": ["r1"],
        "evidence_handles": ["e1", "e5"],
        "expected_consequences": ["the current user receives a clear stance"],
        "confidence": "high",
        "relational_willingness": {
            "schema_version": "relational_willingness.v1",
            "applicability": "relationship_sensitive",
            "stance": "deflect",
            "reason": '当前请求需要结合当前关系与角色边界作出判断',
            "evidence_handles": ["e1", "e5"],
        },
    }
    invalid_source_handles = {
        **valid,
        "evidence_handles": ["e1", "ev3", "ev21"],
        "relational_willingness": {
            **valid["relational_willingness"],
            "evidence_handles": ["e5", "r1"],
        },
    }
    missing_episode_coverage = {
        **valid,
        "relational_willingness": {
            **valid["relational_willingness"],
            "evidence_handles": ["e5"],
        },
    }
    responses = [
        invalid_source_handles,
        missing_episode_coverage,
        valid,
    ]

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            response = responses[len(self.messages) - 1]
            return SimpleNamespace(content=json.dumps(response))

    llm = _LLM()
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "g1"},
        {
            "current_event": [{
                "handle": "e1",
                "source_kind": "episode",
                "semantic_text": "the current user made a direct request",
            }],
            "relationship": {"relationship_summary": "current relationship"},
            "_role_bindings": {"r1": {"role": "current_user"}},
            "role_summaries": {"r1": "the current user"},
        },
        [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-1",
                "occurred_at": "2026-07-15T00:00:00Z",
                "semantic_summary": "the current user made a direct request",
            },
            "semantic_text": "the current user made a direct request",
            "visible_to": ["q:event_agency"],
        }, {
            "evidence_handle": "e5",
            "evidence_ref": {
                "source_kind": "conversation_evidence",
                "source_id": "history-1",
                "occurred_at": "2026-07-14T00:00:00Z",
                "semantic_summary": "earlier relationship context",
            },
            "semantic_text": "earlier relationship context",
            "visible_to": ["q:event_agency"],
        }],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert bid["relational_willingness"]["evidence_handles"] == [
        "e1",
        "e5",
    ]
    assert len(llm.messages) == 3
    first_repair = json.loads(str(llm.messages[1][1].content))
    second_repair = json.loads(str(llm.messages[2][1].content))
    for message_set, repair_payload in zip(
        llm.messages[1:],
        (first_repair, second_repair),
        strict=True,
    ):
        assert (
            len(str(message_set[0].content))
            + len(str(message_set[1].content))
            <= goal_module.GOAL_COGNITION_PROMPT_CAP
        )
        assert repair_payload["evidence"] == [{
            "handle": "e1",
            "semantic_text": "the current user made a direct request",
            "source_kind": "episode",
        }, {
            "handle": "e5",
            "semantic_text": "earlier relationship context",
            "source_kind": "conversation_evidence",
        }]
        feedback = repair_payload["repair_feedback"]
        assert feedback["allowed_evidence_handles"] == ["e1", "e5"]
        assert feedback["current_episode_evidence_handles"] == ["e1"]
        assert feedback["allowed_role_handles"] == ["r1"]
        relational_contract = feedback["relational_willingness_contract"]
        assert relational_contract["required_fields"] == [
            "schema_version",
            "applicability",
            "stance",
            "reason",
            "evidence_handles",
        ]
        assert relational_contract["current_episode_evidence_handles"] == [
            "e1"
        ]
    assert "ev3" in first_repair["repair_feedback"]["invalid_draft"]
    assert first_repair["repair_feedback"]["validation_error"] == (
        "evidence handles are not permitted"
    )
    assert second_repair["repair_feedback"]["validation_error"] == (
        "relational willingness must cite current episode evidence"
    )


@pytest.mark.asyncio
async def test_goal_bid_schema_exhaustion_is_typed_after_three_attempts() -> None:
    """A required branch requests graph retry after its local attempt cap."""

    invalid = {
        "intention": "respond",
        "desired_outcome": "continue",
        "concrete_detail": "acknowledge the greeting",
        "reason": "the participant directly addressed the character",
        "private_monologue": "I want to answer.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the conversation continues"],
        "confidence": "high",
        "requested_route": "speech",
    }

    class _LLM:
        def __init__(self) -> None:
            self.call_count = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.call_count += 1
            return SimpleNamespace(content=json.dumps(invalid))

    llm = _LLM()
    with pytest.raises(CognitionExecutionError) as error_info:
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {"scope": "user", "kind": "goal", "entity_id": "g1"},
            {"_role_bindings": {}, "role_summaries": {}},
            [{
                "evidence_handle": "e1",
                "evidence_ref": {
                    "source_kind": "episode",
                    "source_id": "episode-1",
                    "occurred_at": "2026-07-15T00:00:00Z",
                    "semantic_summary": "direct greeting",
                },
                "semantic_text": "direct greeting",
                "visible_to": ["q:event_agency"],
            }],
            SimpleNamespace(
                llm=llm,
                goal_ordinary_response_config=object(),
            ),
        )

    assert error_info.value.safe_checkpoint == "pre_state_commit"
    assert error_info.value.retryable is True
    assert error_info.value.attempt_count == 3
    assert llm.call_count == 3


@pytest.mark.asyncio
async def test_required_selection_regenerates_with_the_same_producer() -> None:
    """Retry only structural production without a semantic evaluator."""

    selected = {
        "selection_kind": "choice",
        "selection": "当前角色选择让当前用户继续抱紧她。",
        "reason": "当前输入把选择权交给当前角色。",
        "private_monologue": "我现在直接作出自己的选择。",
        "target_role_handles": [],
        "evidence_handles": ["e1", "e2"],
        "expected_consequences": ["当前用户得到一个明确选择。"],
        "confidence": "high",
        "relational_willingness": {
            "schema_version": "relational_willingness.v1",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "reason": '当前回合证据不涉及关系许可判断',
            "evidence_handles": ["e1"],
        },
    }
    responses = [
        {"selection": ""},
        {**selected, "unknown_field": "invalid"},
        selected,
    ]

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            return SimpleNamespace(content=json.dumps(
                responses[len(self.messages) - 1],
                ensure_ascii=False,
            ))

    llm = _LLM()
    semantic_text = json.dumps({
        "role_explicit_content": (
            "当前用户要求当前角色说出当前角色希望当前用户执行的下一步动作。"
        ),
        "response_operation": {
            "operation": "当前角色选择并告诉当前用户下一步动作",
            "response_owner_role": "当前角色",
            "selection_owner_role": "当前角色",
            "selection_required": True,
            "embedded_actor_role": "当前用户",
            "embedded_target_role": "当前角色",
        },
    }, ensure_ascii=False)
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-15T00:00:00Z",
            "semantic_summary": semantic_text,
        },
        "semantic_text": semantic_text,
        "visible_to": ["q:event_agency"],
    }, {
        "evidence_handle": "e2",
        "evidence_ref": {
            "source_kind": "conversation_evidence",
            "source_id": "conversation-progress-event:completed-event",
            "occurred_at": "2026-07-15T00:00:00Z",
            "semantic_summary": "此前选择已经完成。",
        },
        "semantic_text": "此前选择已经完成。",
        "visible_to": ["q:event_agency"],
    }]

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "g1"},
        {"_role_bindings": {}, "role_summaries": {}},
        evidence,
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.messages) == 3
    assert (
        llm.messages[0][0].content
        == goal_module.REQUIRED_SELECTION_GOAL_PROMPT
    )
    assert all(
        message_set[0].content
        == goal_module.REQUIRED_SELECTION_GOAL_REPAIR_PROMPT
        for message_set in llm.messages[1:]
    )
    repair_payload = json.loads(str(llm.messages[1][1].content))
    repair_feedback = repair_payload["repair_feedback"]
    assert repair_feedback["required_evidence_handles"] == ["e1"]
    assert repair_feedback["allowed_evidence_handles"] == ["e1", "e2"]
    assert repair_feedback["current_episode_evidence_handles"] == ["e1"]
    assert "selection_kind" in repair_feedback["required_top_level_fields"]
    assert "selection goal draft fields are not exact" in (
        repair_feedback["validation_error"]
    )
    assert bid["intention"] == selected["selection"]
    assert bid["desired_outcome"] == selected["selection"]
    assert bid["concrete_detail"] == selected["selection"]


@pytest.mark.asyncio
async def test_required_selection_regeneration_excludes_optional_conversation(
) -> None:
    """Keep optional conversation handles out of mandatory retry feedback."""

    valid_selection = {
        'selection_kind': 'choice',
        'selection': '当前角色选择让当前用户陪她去散步。',
        'reason': '当前角色根据关系和此刻感受作出具体选择。',
        'private_monologue': '我现在直接说出自己的选择。',
        'target_role_handles': [],
        'evidence_handles': ['e1', 'e2'],
        'expected_consequences': ['当前用户得到一个明确选择。'],
        'confidence': 'high',
        'relational_willingness': {
            'schema_version': 'relational_willingness.v1',
            'applicability': 'not_relationship_sensitive',
            'stance': 'not_applicable',
            'reason': '当前回合证据不涉及关系许可判断',
            'evidence_handles': ['e1'],
        },
    }
    invalid_selection = {
        **valid_selection,
        'unknown_field': 'invalid',
    }

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            response = (
                invalid_selection
                if len(self.messages) == 1
                else valid_selection
            )
            return SimpleNamespace(content=json.dumps(
                response,
                ensure_ascii=False,
            ))

    llm = _LLM()
    semantic_text = json.dumps({
        'role_explicit_content': '当前用户要求当前角色亲口说出自己的选择。',
        'response_operation': {
            'operation': '当前角色选择并告诉当前用户下一步',
            'response_owner_role': '当前角色',
            'selection_owner_role': '当前角色',
            'selection_required': True,
            'embedded_actor_role': '当前用户',
            'embedded_target_role': '当前角色',
        },
    }, ensure_ascii=False)
    evidence = [{
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode-1',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:event_agency'],
    }, {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-history:prior-turn',
            'occurred_at': '2026-07-29T23:59:00Z',
            'semantic_summary': '此前聊过昨晚发生的事情。',
        },
        'semantic_text': '此前聊过昨晚发生的事情。',
        'visible_to': ['q:event_agency'],
    }]

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {'_role_bindings': {}, 'role_summaries': {}},
        evidence,
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.messages) == 2
    assert (
        llm.messages[1][0].content
        == goal_module.REQUIRED_SELECTION_GOAL_REPAIR_PROMPT
    )
    regeneration_payload = json.loads(str(llm.messages[1][1].content))
    regeneration_feedback = regeneration_payload["repair_feedback"]
    assert regeneration_feedback["required_evidence_handles"] == ["e1"]
    assert regeneration_feedback["allowed_evidence_handles"] == [
        "e1",
        "e2",
    ]
    assert regeneration_feedback["current_episode_evidence_handles"] == [
        "e1"
    ]
    assert "selection goal draft fields are not exact" in (
        regeneration_feedback["validation_error"]
    )
    assert "conversation_evidence_relations" not in (
        str(regeneration_payload)
    )
    assert bid['intention'] == valid_selection['selection']


@pytest.mark.asyncio
async def test_required_selection_repair_replays_grounding_after_handle_failures(
) -> None:
    """Every selection repair receives exact grounding and schema feedback."""

    selected = {
        'selection_kind': 'choice',
        'selection': 'The character makes the current choice directly.',
        'reason': 'The current operation gives the character the choice.',
        'private_monologue': 'I should make this choice from the current facts.',
        'target_role_handles': ['r1'],
        'evidence_handles': ['e1', 'e5'],
        'expected_consequences': ['The current choice is communicated clearly.'],
        'confidence': 'high',
        'relational_willingness': {
            'schema_version': 'relational_willingness.v1',
            'applicability': 'relationship_sensitive',
            'stance': 'deflect',
            'reason': '当前选择需要结合关系和角色边界判断。',
            'evidence_handles': ['e1', 'e5'],
        },
    }
    invalid_source_handles = {
        **selected,
        'evidence_handles': ['e1', 'r1'],
        'relational_willingness': {
            **selected['relational_willingness'],
            'evidence_handles': ['e5', 'r1'],
        },
    }
    missing_episode_coverage = {
        **selected,
        'relational_willingness': {
            **selected['relational_willingness'],
            'evidence_handles': ['e5'],
        },
    }
    responses = [
        invalid_source_handles,
        missing_episode_coverage,
        selected,
    ]

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            response = responses[len(self.messages) - 1]
            return SimpleNamespace(content=json.dumps(
                response,
                ensure_ascii=False,
            ))

    llm = _LLM()
    semantic_text = json.dumps({
        'role_explicit_content': 'The character must make a choice.',
        'response_operation': {
            'operation': 'The character makes the current choice.',
            'response_owner_role': 'current character',
            'selection_owner_role': 'current character',
            'selection_required': True,
            'embedded_actor_role': 'current user',
            'embedded_target_role': 'current character',
        },
    })
    evidence = [{
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode-1',
            'occurred_at': '2026-07-15T00:00:00Z',
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:event_agency'],
    }, {
        'evidence_handle': 'e5',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'history-1',
            'occurred_at': '2026-07-14T00:00:00Z',
            'semantic_summary': 'Earlier relationship context.',
        },
        'semantic_text': 'Earlier relationship context.',
        'visible_to': ['q:event_agency'],
    }]

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {
            '_role_bindings': {
                'r1': {
                    'role': 'target',
                    'entity_kind': 'relationship',
                    'entity_id': 'relationship:u1',
                },
            },
            'role_summaries': {'r1': 'The current relationship.'},
        },
        evidence,
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.messages) == 3
    assert bid['evidence_handles'] == ['e1', 'e5']
    assert bid['relational_willingness']['evidence_handles'] == [
        'e1',
        'e5',
    ]
    for message_set in llm.messages[1:]:
        assert (
            message_set[0].content
            == goal_module.REQUIRED_SELECTION_GOAL_REPAIR_PROMPT
        )
        repair_payload = json.loads(str(message_set[1].content))
        feedback = repair_payload['repair_feedback']
        assert repair_payload['required_selection_operations'][0][
            'evidence_handle'
        ] == 'e1'
        assert feedback['allowed_evidence_handles'] == ['e1', 'e5']
        assert feedback['required_evidence_handles'] == ['e1']
        assert feedback['current_episode_evidence_handles'] == ['e1']
        assert feedback['allowed_role_handles'] == ['r1']
        assert feedback['role_handles_forbidden_in_evidence_handles'] == [
            'r1'
        ]
        assert feedback['required_top_level_fields'][-1] == (
            'relational_willingness'
        )
        relational_contract = feedback['relational_willingness_contract']
        assert relational_contract['current_episode_evidence_handles'] == [
            'e1'
        ]
    first_repair = json.loads(str(llm.messages[1][1].content))
    second_repair = json.loads(str(llm.messages[2][1].content))
    assert 'r1' in first_repair['repair_feedback']['invalid_draft']
    assert first_repair['repair_feedback']['validation_error'] == (
        'evidence handles are not permitted'
    )
    assert second_repair['repair_feedback']['validation_error'] == (
        'relational willingness must cite current episode evidence'
    )


@pytest.mark.asyncio
async def test_active_selection_repair_uses_the_same_grounding_contract(
) -> None:
    """Active required selections use producer repair without relation output."""

    valid = {
        'selection_kind': 'choice',
        'selection': 'The character chooses the grounded next step.',
        'reason': 'The current operation requires a concrete choice.',
        'private_monologue': 'I should choose from the current evidence.',
        'target_role_handles': ['r1'],
        'evidence_handles': ['e1'],
        'expected_consequences': ['The active goal receives a clear choice.'],
        'confidence': 'high',
    }
    invalid = {
        **valid,
        'evidence_handles': ['r1'],
    }

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            response = invalid if len(self.messages) == 1 else valid
            return SimpleNamespace(content=json.dumps(response))

    llm = _LLM()
    semantic_text = json.dumps({
        'role_explicit_content': 'The character must choose.',
        'response_operation': {
            'operation': 'The character chooses the next step.',
            'response_owner_role': 'current character',
            'selection_owner_role': 'current character',
            'selection_required': True,
            'embedded_actor_role': 'current user',
            'embedded_target_role': 'current character',
        },
    })
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['autonomy_boundary'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {
            '_role_bindings': {
                'r1': {
                    'role': 'target',
                    'entity_kind': 'relationship',
                    'entity_id': 'relationship:u1',
                },
            },
            'role_summaries': {'r1': 'The current relationship.'},
        },
        [{
            'evidence_handle': 'e1',
            'evidence_ref': {
                'source_kind': 'episode',
                'source_id': 'episode-1',
                'occurred_at': '2026-07-15T00:00:00Z',
                'semantic_summary': semantic_text,
            },
            'semantic_text': semantic_text,
            'visible_to': ['q:event_agency'],
        }],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.messages) == 2
    assert 'relational_willingness' not in bid
    assert (
        llm.messages[1][0].content
        == goal_module.REQUIRED_SELECTION_GOAL_REPAIR_PROMPT
    )
    repair_payload = json.loads(str(llm.messages[1][1].content))
    feedback = repair_payload['repair_feedback']
    assert 'relational_willingness' not in feedback['required_top_level_fields']
    assert 'relational_willingness_contract' not in feedback
    assert feedback['required_evidence_handles'] == ['e1']
    assert feedback['role_handles_forbidden_in_evidence_handles'] == ['r1']


@pytest.mark.asyncio
async def test_required_selection_structure_exhaustion_is_typed() -> None:
    """Fail before state commit after bounded producer-only attempts."""

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"selection": ""}',
    ))
    semantic_text = json.dumps({
        "role_explicit_content": "当前角色必须作出一个选择。",
        "response_operation": {
            "operation": "当前角色选择一个动作",
            "response_owner_role": "当前角色",
            "selection_owner_role": "当前角色",
            "selection_required": True,
            "embedded_actor_role": "当前角色",
            "embedded_target_role": "当前用户",
        },
    }, ensure_ascii=False)

    with pytest.raises(CognitionExecutionError) as error_info:
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {"scope": "user", "kind": "goal", "entity_id": "g1"},
            {"_role_bindings": {}, "role_summaries": {}},
            [{
                "evidence_handle": "e1",
                "evidence_ref": {
                    "source_kind": "episode",
                    "source_id": "episode-1",
                    "occurred_at": "2026-07-15T00:00:00Z",
                    "semantic_summary": semantic_text,
                },
                "semantic_text": semantic_text,
                "visible_to": ["q:event_agency"],
            }],
            SimpleNamespace(
                llm=llm,
                goal_ordinary_response_config=object(),
            ),
        )

    assert llm.ainvoke.await_count == 3
    assert error_info.value.error_code == "goal_bid_structure_exhausted"
    assert error_info.value.stage == "goal_cognition"
    assert error_info.value.safe_checkpoint == "pre_state_commit"


@pytest.mark.asyncio
async def test_required_selection_invalid_evidence_degrades_after_exhaustion(
) -> None:
    """Continue with valid evidence after repeated invalid-handle output."""

    selected = {
        "selection_kind": "choice",
        "selection": "The character accepts the current question.",
        "reason": "The current input directly asks for the character's answer.",
        "private_monologue": "I can answer the question directly.",
        "target_role_handles": ["current_user"],
        "evidence_handles": ["e1", "r1"],
        "expected_consequences": ["The conversation continues."],
        "confidence": "high",
        "relational_willingness": {
            "schema_version": "relational_willingness.v1",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "reason": '当前回合证据不涉及关系许可判断',
            "evidence_handles": ["e1"],
        },
    }

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.calls += 1
            return SimpleNamespace(
                content=json.dumps(selected, ensure_ascii=False),
            )

    llm = _LLM()
    semantic_text = json.dumps({
        "role_explicit_content": "The character must answer.",
        "response_operation": {
            "operation": "The character answers the current question.",
            "response_owner_role": "current character",
            "selection_owner_role": "current character",
            "selection_required": True,
            "embedded_actor_role": "current user",
            "embedded_target_role": "current character",
        },
    })

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "g1"},
        {
            "_role_bindings": {
                "current_user": {
                    "role": "target",
                    "entity_kind": "user",
                    "entity_id": "u1",
                },
                "r1": {
                    "role": "target",
                    "entity_kind": "relationship",
                    "entity_id": "relationship:u1",
                },
            },
            "role_summaries": {
                "current_user": "The current user.",
                "r1": "The current relationship.",
            },
        },
        [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode-1",
                "occurred_at": "2026-07-15T00:00:00Z",
                "semantic_summary": semantic_text,
            },
            "semantic_text": semantic_text,
            "visible_to": ["q:event_agency"],
        }],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert llm.calls == 3
    assert bid["evidence_handles"] == ["e1"]
    assert bid["target_roles"] == [{
        "role": "target",
        "entity_kind": "user",
        "entity_id": "u1",
    }]


def test_required_branch_failure_cannot_collapse_to_silence() -> None:
    """A required cognition failure remains an execution failure."""

    original_error = ValueError("selection goal structure remains invalid")
    execution = ParallelExecutionResult(
        failed_branch_ids={"ordinary_response"},
        failure_records={
            "ordinary_response": BranchFailure(
                branch_id="ordinary_response",
                error_code="goal_bid_structure_exhausted",
                stage="goal_cognition",
                attempt_count=3,
                safe_checkpoint="pre_state_commit",
                retryable=True,
                exception_class="ValueError",
                exception=original_error,
            ),
        },
    )

    with pytest.raises(
        CognitionExecutionError,
        match="required cognition",
    ) as raised:
        _raise_for_failed_required_branches(
            execution,
            [DEFAULT_BRANCH_DEFINITIONS["ordinary_response"]],
        )

    assert raised.value.error_code == "goal_bid_structure_exhausted"
    assert raised.value.branch_id == "ordinary_response"
    assert raised.value.stage == "goal_cognition"
    assert raised.value.attempt_count == 3
    assert raised.value.safe_checkpoint == "pre_state_commit"
    assert raised.value.retryable is True
    assert raised.value.__cause__ is original_error


def test_required_selection_producer_demands_one_actual_selection() -> None:
    """Keep the authoritative choice inside the producing cognition call."""

    prompt = goal_module.REQUIRED_SELECTION_GOAL_PROMPT

    assert '`selection` 是唯一' in prompt
    assert '权威选择内容' in prompt
    assert "不得只说以后决定" in prompt
    assert '"selection": ""' in prompt


def test_required_selection_producer_selects_relevant_progress_evidence(
) -> None:
    """Keep progress relevance with the producing semantic owner."""

    prompt = goal_module.REQUIRED_SELECTION_GOAL_PROMPT

    assert '`conversation_progress_evidence`' in prompt
    assert '引用其中会实质约束本轮选择的' in prompt
    assert '不引用与当前选择无关的历史' in prompt
    assert '`completed`' in prompt
    assert '`rejected`' in prompt
    assert '`superseded`' in prompt
    assert '只有当前输入明确要求重开' in prompt
    assert '`supporting_evidence` 只提供可选支持' in prompt
    assert '`semantic_context` 中出现的 handle' in prompt
    assert 'conversation_evidence_relations' not in prompt
    assert not hasattr(goal_module, 'REQUIRED_SELECTION_VERIFIER_PROMPT')
    assert hasattr(goal_module, 'REQUIRED_SELECTION_GOAL_REPAIR_PROMPT')
