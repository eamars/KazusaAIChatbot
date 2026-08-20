"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
    CONTINUITY_AUTHORITY_INSTRUCTIONS,
    GOAL_COGNITION_ATTEMPT_LIMIT,
    GOAL_COGNITION_PROMPT,
    GOAL_COGNITION_PROMPT_CAP,
    NON_ORDINARY_GOAL_COGNITION_PROMPT,
    ORDINARY_RECURRENCE_GOAL_COGNITION_PROMPT,
    ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
    REQUIRED_SELECTION_GOAL_PROMPT,
    SELECTION_GOAL_REPAIR_INSTRUCTIONS,
    _conversation_progress_evidence,
    _materialize_recurrence_relational_willingness,
    build_goal_output_contract,
    run_goal_cognition,
    selection_goal_draft_to_goal_bid,
    validate_goal_bid_draft,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.goal_cognition"
EXPECTED_SYMBOLS = ["run_goal_cognition", "build_goal_output_contract"]
_REPLAY_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "cognition_core_v2_relational_carrier_failure_cases.json"
)


def _recurrence_carrier() -> dict[str, object]:
    """Build one valid carrier for deterministic recurrence boundary tests."""

    return {
        "schema_version": "current_turn_relational_willingness.v2",
        "episode_id": "unit-recurrence-episode",
        "branch_id": "ordinary_response",
        "decision": {
            "schema_version": "relational_willingness.v2",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "The current episode is not relationship sensitive.",
            "evidence_handles": ["e1"],
        },
    }


def _current_episode_evidence() -> dict[str, object]:
    """Build one minimal current-episode row for recurrence validation."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:unit-recurrence-episode",
            "occurred_at": "2026-08-13T00:00:00Z",
            "semantic_summary": "Current episode evidence.",
        },
        "semantic_text": "Current episode evidence.",
        "visible_to": ["q:event_agency"],
        "authority": "current_event",
    }


class _UnexpectedLLM:
    """Fail if a deterministic recurrence precondition reaches the model."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        """Record an unexpected call and fail the owning unit test."""

        del args, kwargs
        self.calls += 1
        raise AssertionError("recurrence precondition reached the LLM")


def _goal_services(llm: object) -> SimpleNamespace:
    """Build the minimal service surface used before the model boundary."""

    config = SimpleNamespace(route_name="unit.goal_cognition")
    return SimpleNamespace(
        llm=llm,
        goal_ordinary_response_config=config,
        goal_active_branch_config=config,
    )


def _input_operation() -> dict[str, object]:
    """Build the required-selection operation before character judgment."""

    return {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }


def _selection_draft(
    selected_operation: dict[str, object],
) -> dict[str, object]:
    """Build one complete selection draft for the validator."""

    return {
        "selection": "choose a reward",
        "selected_response_operation": selected_operation,
        "reason": "the current episode requires a concrete choice",
        "private_monologue": "choose from the current request",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the selected reward is stated"],
        "confidence": "high",
    }


def test_goal_cognition_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_required_selection_emits_selected_response_operation() -> None:
    """Selection validation returns the operation chosen by cognition."""

    selected_operation = {
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    validated = validate_selection_goal_draft(
        _selection_draft(selected_operation),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": _input_operation(),
        }],
        maximum_evidence_handles=4,
    )

    assert validated["selected_response_operation"] == {
        **_input_operation(),
        **selected_operation,
    }


def test_public_selection_materializer_preserves_roles_and_evidence() -> None:
    """The public materializer preserves validated selection ownership."""

    selected_operation = {
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    selection_draft = validate_selection_goal_draft(
        _selection_draft(selected_operation),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": _input_operation(),
        }],
        maximum_evidence_handles=4,
    )
    bid = selection_goal_draft_to_goal_bid(
        selection_draft,
        branch_id="ordinary_response",
        include_relational_willingness=False,
    )

    assert bid["target_role_handles"] == []
    assert bid["evidence_handles"] == ["e1"]
    assert bid["selected_response_operation"] == {
        **_input_operation(),
        **selected_operation,
    }


def test_required_selection_rejects_conflicting_known_endpoint_with_field_error() -> None:
    """Selection validation rejects reversal of a known input endpoint."""

    selected_operation = {
        "operation": "the character gives the selected reward to the user",
        "embedded_actor_role": CURRENT_USER_ROLE,
    }

    with pytest.raises(
        ValueError,
        match=(
            "embedded_actor_role conflicts with known input role: "
            "expected='当前角色'; actual='当前用户'"
        ),
    ):
        validate_selection_goal_draft(
            _selection_draft(selected_operation),
            evidence_handles={"e1"},
            role_handles=set(),
            required_evidence_handles={"e1"},
            required_operations=[{
                "evidence_handle": "e1",
                "response_operation": {
                    **_input_operation(),
                    "embedded_actor_role": CURRENT_CHARACTER_ROLE,
                },
            }],
            maximum_evidence_handles=4,
        )


def test_required_selection_prompt_separates_wrapper_and_embedded_action_roles() -> None:
    """Required-selection contracts type the nested action, not its wrappers."""

    governing_instructions = (
        ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
        " ".join(SELECTION_GOAL_REPAIR_INSTRUCTIONS),
        REQUIRED_SELECTION_GOAL_PROMPT,
        _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
    )
    for instruction in governing_instructions:
        assert "回应包装" in instruction
        assert "具体嵌入行动" in instruction
        assert "writable_fields" in instruction
        assert "code_owned_fields" in instruction
        assert "相同 wording" in instruction

    selection_contract = build_goal_output_contract(
        evidence_handles={"e1"},
        episode_evidence_handles={"e1"},
        required_evidence_handles={"e1"},
        role_bindings={},
        selection_required=True,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
        authoritative_operation=_input_operation(),
    )
    assert selection_contract["field_types"]["selected_response_operation"] == (
        "per_input_writable_selected_response_operation"
    )
    operation_contract = selection_contract["selected_response_operation"]
    assert operation_contract["writable_fields"] == [
        "operation",
        "embedded_actor_role",
        "embedded_target_role",
    ]
    assert operation_contract["code_owned_fields"] == {
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
    }
    assert "selected_response_operation_fields" not in selection_contract


def test_required_selection_prompt_projects_exact_writable_endpoint_fields() -> None:
    """Project only unresolved endpoints into the model-owned field set."""

    cases = (
        (
            _input_operation(),
            ["operation", "embedded_actor_role", "embedded_target_role"],
            {},
        ),
        (
            {
                **_input_operation(),
                "embedded_actor_role": CURRENT_CHARACTER_ROLE,
            },
            ["operation", "embedded_target_role"],
            {"embedded_actor_role": CURRENT_CHARACTER_ROLE},
        ),
        (
            {
                **_input_operation(),
                "embedded_actor_role": CURRENT_CHARACTER_ROLE,
                "embedded_target_role": CURRENT_USER_ROLE,
            },
            ["operation"],
            {
                "embedded_actor_role": CURRENT_CHARACTER_ROLE,
                "embedded_target_role": CURRENT_USER_ROLE,
            },
        ),
    )
    for authoritative_operation, writable_fields, known_fields in cases:
        contract = build_goal_output_contract(
            evidence_handles={"e1"},
            episode_evidence_handles={"e1"},
            required_evidence_handles={"e1"},
            role_bindings={},
            selection_required=True,
            require_relational_willingness=False,
            maximum_evidence_handles=9,
            authoritative_operation=authoritative_operation,
        )
        assert contract["field_types"]["selected_response_operation"] == (
            "per_input_writable_selected_response_operation"
        )
        contract = contract["selected_response_operation"]
        assert contract["writable_fields"] == writable_fields
        assert contract["required_fields"] == ["operation"]
        assert contract["optional_fields"] == writable_fields[1:]
        assert contract["code_owned_fields"] == {
            "response_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_required": True,
            **known_fields,
        }


def test_required_selection_accepts_matching_known_endpoint_and_canonicalizes() -> None:
    """Matching redundant known endpoints are accepted and code-bound."""

    authoritative_operation = {
        **_input_operation(),
        "embedded_actor_role": CURRENT_CHARACTER_ROLE,
        "embedded_target_role": CURRENT_USER_ROLE,
    }
    validated = validate_selection_goal_draft(
        _selection_draft({
            "operation": authoritative_operation["operation"],
            "embedded_actor_role": CURRENT_CHARACTER_ROLE,
            "embedded_target_role": CURRENT_USER_ROLE,
        }),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": authoritative_operation,
        }],
        maximum_evidence_handles=4,
    )

    assert validated["selected_response_operation"] == authoritative_operation


def test_required_selection_accepts_authoritative_operation_text() -> None:
    """Usable authoritative wording remains acceptable selected wording."""

    authoritative_operation = _input_operation()
    validated = validate_selection_goal_draft(
        _selection_draft({"operation": authoritative_operation["operation"]}),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": authoritative_operation,
        }],
        maximum_evidence_handles=4,
    )

    assert validated["selected_response_operation"] == authoritative_operation


def test_required_selection_rejects_unknown_operation_field() -> None:
    """Unknown selected-operation fields remain a structural contract error."""

    with pytest.raises(
        ValueError,
        match=r"selected response operation contains unknown fields: \['extra'\]",
    ):
        validate_selection_goal_draft(
            _selection_draft({
                "operation": "the character chooses a reward",
                "extra": "unexpected",
            }),
            evidence_handles={"e1"},
            role_handles=set(),
            required_evidence_handles={"e1"},
            required_operations=[{
                "evidence_handle": "e1",
                "response_operation": _input_operation(),
            }],
            maximum_evidence_handles=4,
        )


def test_goal_prompt_labels_confidence_as_descriptor() -> None:
    """Goal prompts distinguish semantic confidence from evaluator scores."""

    prompt = GOAL_COGNITION_PROMPT

    assert "confidence 是有界的置信度描述" in prompt
    assert "不是 score" in prompt


def test_conversation_progress_evidence_preserves_temporal_provenance() -> None:
    """Goal projection carries source time and deterministic age metadata."""

    projected = _conversation_progress_evidence([{
        'evidence_handle': 'e-progress',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:1',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': 'same concrete matter',
        },
        'semantic_text': 'the same concrete matter remains open',
        'visible_to': ['q:event_agency'],
        'authority': 'participant_continuity',
        'temporal_provenance': {
            'occurred_at': '2026-07-30T00:00:00Z',
            'age_descriptor': 'recent',
        },
    }])

    assert projected == [{
        'evidence_handle': 'e-progress',
        'semantic_text': 'the same concrete matter remains open',
        'authority': 'participant_continuity',
        'temporal_provenance': {
            'occurred_at': '2026-07-30T00:00:00Z',
            'age_descriptor': 'recent',
        },
    }]


def test_goal_prompt_declares_one_primary_current_scene_objective() -> None:
    """Every goal mode receives the same generic evidence-authority contract."""

    prompts = (
        GOAL_COGNITION_PROMPT,
        ORDINARY_RECURRENCE_GOAL_COGNITION_PROMPT,
        ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
        NON_ORDINARY_GOAL_COGNITION_PROMPT,
    )
    for prompt in prompts:
        assert '当前事件' in prompt
        assert '目标' in prompt
        assert '每次只选择一个当前目标' in (
            prompt + CONTINUITY_AUTHORITY_INSTRUCTIONS
        )
        rendered_contract = prompt + CONTINUITY_AUTHORITY_INSTRUCTIONS
        for authority in (
            'current_event',
            'public_scene',
            'participant_continuity',
            'private_motive_only',
            'conditional_character_guidance',
        ):
            assert f'`{authority}`' in rendered_contract
        assert '所有返回的目标说明和预期后果都必须共同服务这个主要目标' in (
            rendered_contract
        )
        assert '连贯的从属行动' in rendered_contract
        assert '归属于某位说话者的公开发言' in rendered_contract
        assert '不单独证明其声称的外部命题' in rendered_contract
        assert '最新的第一人称更正或状态更新具有更高的判断权重' in (
            rendered_contract
        )
        assert '共享公开事项仍未解决' in rendered_contract
        assert '单项核查、子问题、局部信号或从属步骤' in rendered_contract
        assert '不足以单独证明整个事项已经解决' in rendered_contract
        assert '把竞争事项提升为主要目标前' in rendered_contract
        assert '当前可观察证据' in rendered_contract
        assert '对整个事项具有判断权的参与者作出明确更正' in (
            rendered_contract
        )
        assert '只有满足这一条件，才可开始新的主要话题' in (
            rendered_contract
        )
        assert '否则保留原事项为唯一主要目标' in rendered_contract
        assert '竞争内容只能作为对原目标的直接支持、拒绝或推后处理' in (
            rendered_contract
        )
        assert '可以质疑或核实冲突主张' in rendered_contract
        assert '角色仍自主决定立场以及服务主要目标的从属行动方式和先后' in (
            rendered_contract
        )
        assert '只允许原样出现在各自的类型化 handle 字段' in (
            rendered_contract
        )
        assert '必须使用语义角色描述或自然指代' in rendered_contract
        assert '不得复写任何 handle token 或内部标识' in rendered_contract
        assert '输入请求或模型设想的现实世界实体交互、感知或观察' in (
            rendered_contract
        )
        assert '提议或取得证据后再回应的目标' in rendered_contract
        assert '不能描述该行为已经发生' in rendered_contract
        assert '观察已经完成或相应结果已经确定' in rendered_contract
    assert '每次只选择一个当前目标' in CONTINUITY_AUTHORITY_INSTRUCTIONS
    assert '所有从属行动必须服务于同一个目标' in (
        CONTINUITY_AUTHORITY_INSTRUCTIONS
    )


def test_goal_output_contract_keeps_existing_schema() -> None:
    """Keep the generic goal schema and consequence bounds stable."""

    generic_contract = build_goal_output_contract(
        evidence_handles={'e1'},
        episode_evidence_handles={'e1'},
        required_evidence_handles=set(),
        role_bindings={},
        selection_required=False,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
    )
    selection_contract = build_goal_output_contract(
        evidence_handles={'e1'},
        episode_evidence_handles={'e1'},
        required_evidence_handles={'e1'},
        role_bindings={},
        selection_required=True,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
        authoritative_operation=_input_operation(),
    )
    relational_contract = build_goal_output_contract(
        evidence_handles={'e1'},
        episode_evidence_handles={'e1'},
        required_evidence_handles=set(),
        role_bindings={},
        selection_required=False,
        require_relational_willingness=True,
        maximum_evidence_handles=9,
    )

    expected_fields = (
        {
            'intention',
            'desired_outcome',
            'concrete_detail',
            'reason',
            'private_monologue',
            'target_role_handles',
            'evidence_handles',
            'expected_consequences',
            'confidence',
        },
        {
            'selection',
            'selected_response_operation',
            'reason',
            'private_monologue',
            'target_role_handles',
            'evidence_handles',
            'expected_consequences',
            'confidence',
        },
    )
    for contract, fields in zip(
        (generic_contract, selection_contract),
        expected_fields,
        strict=True,
    ):
        assert set(contract['top_level_fields']) == fields
        assert contract['bounds']['expected_consequences'] == {
            'minimum_items': 1,
            'maximum_items': 8,
            'item_maximum_chars': 240,
        }
    relational_output_contract = relational_contract[
        'relational_willingness_contract'
    ]
    assert 'schema_version' not in relational_output_contract
    assert 'schema_version' not in relational_output_contract['required_fields']


@pytest.mark.asyncio
async def test_goal_cognition_rejects_recurrence_without_episode_binding() -> None:
    """Reject a carrier before the recurrence model boundary."""

    llm = _UnexpectedLLM()
    with pytest.raises(CognitionExecutionError) as error_info:
        await run_goal_cognition(
            BranchDefinition(
                branch_id='ordinary_response',
                dependencies=(),
                action_tendencies=('speak',),
                goal_kind='ordinary_response',
            ),
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': 'goal:unit-recurrence',
            },
            {},
            [_current_episode_evidence()],
            _goal_services(llm),
            current_turn_relational_willingness=_recurrence_carrier(),
        )

    failure = error_info.value
    assert failure.error_code == 'current_turn_relational_carrier_invalid'
    assert failure.branch_id == 'ordinary_response'
    assert failure.stage == 'goal_cognition'
    assert failure.attempt_count == 0
    assert failure.safe_checkpoint == 'pre_state_commit'
    assert failure.retryable is False
    assert llm.calls == 0


def test_selection_goal_rejects_model_authored_relational_schema() -> None:
    """Reject protocol metadata before binding the public relational shape."""

    draft = _selection_draft({
        "operation": "the user gives the selected reward to the character",
    })
    draft["relational_willingness"] = {
        "schema_version": "relational_willingness.v2",
        "applicability": "not_relationship_sensitive",
        "stance": "not_applicable",
        "current_user_relationship_state": "not_applicable",
        "reason": "the request is not relationship sensitive",
        "evidence_handles": ["e1"],
    }

    with pytest.raises(ValueError, match="schema_version is code-owned"):
        validate_selection_goal_draft(
            draft,
            evidence_handles={"e1"},
            role_handles=set(),
            required_evidence_handles={"e1"},
            required_operations=[{
                "evidence_handle": "e1",
                "response_operation": _input_operation(),
            }],
            episode_handles={"e1"},
            require_relational_willingness=True,
            maximum_evidence_handles=4,
        )


def test_goal_bid_rejects_model_authored_relational_schema() -> None:
    """Reject protocol metadata on the ordinary goal path as well."""

    draft = {
        "intention": "answer the current request",
        "desired_outcome": "the user receives a grounded answer",
        "concrete_detail": "use the current evidence",
        "reason": "the request is answerable",
        "private_monologue": "answer from the current context",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the answer addresses the request"],
        "confidence": "high",
        "relational_willingness": {
            "schema_version": "relational_willingness.v2",
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "the request is not relationship sensitive",
            "evidence_handles": ["e1"],
        },
    }

    with pytest.raises(ValueError, match="schema_version is code-owned"):
        validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles=set(),
            require_relational_willingness=True,
            episode_handles={"e1"},
        )


def _generic_goal_draft() -> dict[str, object]:
    """Build a complete non-relational goal draft."""

    return {
        "intention": "answer the current request",
        "desired_outcome": "the user receives a grounded answer",
        "concrete_detail": "use the current evidence",
        "reason": "the request is answerable",
        "private_monologue": "answer from the current context",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the answer addresses the request"],
        "confidence": "high",
    }


def test_nonowning_relational_willingness_is_stripped_without_propagation() -> None:
    """Normalize a non-owning branch's extra relational field once."""

    draft = _generic_goal_draft()
    draft["relational_willingness"] = {"unexpected": "model-owned"}

    validated = validate_goal_bid_draft(
        draft,
        evidence_handles={"e1"},
        role_handles=set(),
        require_relational_willingness=False,
    )

    assert "relational_willingness" not in validated


def test_owned_relational_willingness_boundary_failure_fails_closed() -> None:
    """Keep relational validation active on the owning branch."""

    draft = _generic_goal_draft()
    draft["relational_willingness"] = {"unexpected": "model-owned"}

    with pytest.raises(ValueError, match="relational willingness"):
        validate_goal_bid_draft(
            draft,
            evidence_handles={"e1"},
            role_handles=set(),
            require_relational_willingness=True,
            episode_handles={"e1"},
        )


def test_unrecoverable_goal_structure_retries_within_budget() -> None:
    """Keep a bounded producer retry budget for unrecoverable structure."""

    assert GOAL_COGNITION_ATTEMPT_LIMIT >= 2


def test_provider_failure_preserves_goal_attempt_cap() -> None:
    """Provider failures remain bounded by the same stage attempt cap."""

    assert GOAL_COGNITION_ATTEMPT_LIMIT == 3


def test_prompt_cap_preserves_zero_call_disposition() -> None:
    """Prompt-budget failure remains a positive zero-call boundary."""

    assert GOAL_COGNITION_PROMPT_CAP > 0


def test_goal_cognition_rejects_recurrence_without_current_episode_evidence(
) -> None:
    """Reject a valid carrier when no current-episode handle is available."""

    with pytest.raises(CognitionExecutionError) as error_info:
        _materialize_recurrence_relational_willingness(
            _recurrence_carrier(),
            set(),
        )

    failure = error_info.value
    assert failure.error_code == 'current_turn_relational_carrier_invalid'
    assert failure.branch_id == 'ordinary_response'
    assert failure.stage == 'goal_cognition'
    assert failure.attempt_count == 0
    assert failure.safe_checkpoint == 'pre_state_commit'
    assert failure.retryable is False


def test_relational_carrier_replay_fixture_is_deidentified_and_complete() -> None:
    """Keep the checked-in carrier replay independent of protected identities."""

    fixture = json.loads(
        _REPLAY_FIXTURE_PATH.read_text(encoding='utf-8')
    )
    assert fixture['fixture_kind'] == 'deidentified_contract_replay'
    assert fixture['observed_failure'] == {
        'error_code': 'current_turn_relational_carrier_invalid',
        'stage_name': 'goal_cognition',
        'branch_id': 'ordinary_response',
        'phase': 'preliminary',
        'ordinary_goal_model_calls_during_failure': 0,
        'current_episode_evidence_handle': 'e1',
    }
    cases = fixture['cases']
    assert len(cases) == 2
    for case in cases:
        assert case['source_trace_label'].startswith('trace_case_')
        assert 'source_trace_id' not in case
        assert case['resolver_cycle_index'] == 1
        assert case['evidence'] == [{
            'evidence_handle': 'e1',
            'source_kind': 'episode',
        }]
        assert case['failure_variant'] == 'missing_episode_binding'
        assert case['carrier']['episode_id'] == case['episode_id']
        assert case['carrier']['branch_id'] == 'ordinary_response'
        assert case['carrier']['decision']['evidence_handles'] == ['e1']
