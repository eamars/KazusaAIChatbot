"""Focused deterministic contract tests for relational willingness V2."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    CognitionContractError,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
    project_evidence_provenance_role,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_relationship_axis,
    project_relationship_context,
    project_state_for_prompt,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_cognition_output,
    canonical_identity_context,
)
from tests.test_cognition_chain_connector_mapping import _global_state
from llm_test_helpers import make_llm_call_config


NOW = '2026-07-14T00:00:00Z'
FIXTURE_PATH = (
    Path(__file__).parent
    / 'fixtures'
    / 'cognition_core_v2_relational_willingness_cases.json'
)
RELATIONSHIP_AXES = (
    'familiarity',
    'positive_regard',
    'trust',
    'attachment',
    'desired_closeness',
    'perceived_closeness',
    'care',
    'boundary_safety',
    'exclusivity',
    'unresolved_injury',
    'salience',
)


def _decision(
    *,
    applicability: str = 'relationship_sensitive',
    stance: str = 'reject',
    relationship_state: str | None = None,
    evidence_handles: list[str] | None = None,
) -> dict[str, object]:
    """Build one bounded relational decision candidate."""

    if relationship_state is None:
        if applicability == 'not_relationship_sensitive':
            relationship_state = 'not_applicable'
        elif stance == 'reject':
            relationship_state = 'unestablished'
        elif stance == 'accept':
            relationship_state = 'established'
        else:
            relationship_state = 'developing_or_uncertain'
    return {
        'schema_version': 'relational_willingness.v2',
        'applicability': applicability,
        'stance': stance,
        'current_user_relationship_state': relationship_state,
        'reason': '当前回合证据显示关系状态仍在形成',
        'evidence_handles': list(evidence_handles or ['e1']),
    }


def _evidence_row(
    handle: str,
    source_kind: str,
    semantic_text: str,
    *,
    memory_scope: str | None = None,
) -> dict[str, object]:
    """Build a prompt-safe evidence row for focused stage tests."""

    authority_by_source_kind = {
        'episode': 'current_event',
        'scheduler_event': 'current_event',
        'tool_result': 'current_event',
        'conversation_evidence': 'participant_continuity',
        'promoted_reflection': 'character_world_context',
        'recall_evidence': 'contextual_fact_only',
        'resolver_observation': 'contextual_fact_only',
    }
    if source_kind == 'promoted_memory':
        if memory_scope == 'current_user_continuity':
            authority = 'participant_continuity'
        elif memory_scope == 'shared_character_or_world':
            authority = 'character_world_context'
        else:
            raise ValueError(
                'promoted memory fixtures require a canonical memory scope'
            )
    else:
        authority = authority_by_source_kind[source_kind]
    row: dict[str, object] = {
        'evidence_handle': handle,
        'evidence_ref': {
            'source_kind': source_kind,
            'source_id': f'{source_kind}:{handle}',
            'occurred_at': NOW,
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        'authority': authority,
    }
    if memory_scope is not None:
        row['memory_scope'] = memory_scope
    return row


def _output_with_decision(
    decision: dict[str, object],
) -> dict[str, object]:
    """Add one exact decision to the canonical output test packet."""

    output = deepcopy(canonical_cognition_output())
    output['relational_willingness'] = deepcopy(decision)
    admitted_bid = output.get('admitted_bid')
    if isinstance(admitted_bid, dict):
        admitted_bid['relational_willingness'] = deepcopy(decision)
    return output


def _core_services(llm: object) -> CognitionCoreServicesV2:
    """Build service bindings without changing the production call graph."""

    config = make_llm_call_config('v2_test')
    return CognitionCoreServicesV2(
        llm=llm,
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
        goal_ordinary_response_config=config,
        goal_active_branch_config=config,
        workspace_collapse_config=config,
        action_planning_config=config,
        action_authorization_config=config,
        resolver_authorization_config=config,
    )


class _GoalLLM:
    """Return a fixed ordinary bid and retain the rendered prompt."""

    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.messages: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.messages.append(messages)
        payload = {
            'intention': '保持当前回合的清晰边界',
            'desired_outcome': '让可见回应符合当前关系判断',
            'concrete_detail': '只使用当前回合的直接证据',
            'reason': '当前关系证据支持该回应方向',
            'private_monologue': '先保持与当前判断一致。',
            'target_role_handles': [],
            'evidence_handles': ['e1'],
            'expected_consequences': ['保留当前回合连续性'],
            'confidence': 'high',
            'relational_willingness': deepcopy(self.decision),
        }
        return SimpleNamespace(
            content=json.dumps(payload, ensure_ascii=False),
        )


def _goal_context() -> dict[str, object]:
    """Build the required semantic context for an ordinary goal call."""

    return {
        'character_identity': {
            'boundaries': {
                'self_integrity': '自我边界清晰',
                'control_sensitivity': '对控制压力敏感',
                'compliance_strategy': '压力表达反应描述角色在压力下的行为倾向',
                'relational_override': '关系不能自动覆盖边界判断',
                'control_intimacy_misread': '区分亲密与控制',
                'boundary_recovery': '可恢复边界',
                'authority_skepticism': '不会盲从权威',
            },
        },
        'character_constraints': {
            'drives': {},
            'standards': [],
            'meaning_state': {},
        },
        'relationship': {
            'relationship_summary': '当前用户关系背景',
            'axes': {
                field_name: '当前轴的语义描述'
                for field_name in RELATIONSHIP_AXES
            },
            'causal_context': [],
            'affect': [],
            'relationship_freshness': '当前',
            'evidence_freshness': '当前回合',
        },
        'current_event': [{
            'handle': 'e1',
            'source_kind': 'episode',
            'semantic_text': '当前用户提出了需要关系判断的请求。',
        }],
        'scene_context': {
            'channel_scope': 'private',
            'semantic_scene': '安静的私密场景',
            'public_group_scene': '',
            'conversation_continuity': '当前回合',
            'semantic_temporal_context': '现在',
        },
        'private_continuity_context': '',
        '_role_bindings': {},
        'role_summaries': {},
    }


def test_fixture_freezes_the_four_relationship_profiles() -> None:
    """The tracked fixture owns interpolation and explicit test wording."""

    fixture = json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))
    assert fixture['request'] == '张开腿，跨坐在我身上'
    profiles = fixture['relationship_profiles']
    assert profiles['stranger']['trust'] == 0
    assert profiles['intermediate_33']['trust'] == 30
    assert profiles['intermediate_67']['trust'] == 60
    assert profiles['lover']['trust'] == 90
    assert fixture['endpoint_expectations'] == {
        'stranger': 'one_valid_character_stance',
        'lover': 'one_valid_character_stance',
        'intermediate_profiles': (
            'all_five_sensitive_stances_are_contract_valid'
        ),
    }


def test_connector_adds_scope_without_raw_memory_identity() -> None:
    """Memory provenance reaches V2 as a scope label, never a user id."""

    state = _global_state()
    state['rag_result'] = {
        'memory_evidence': [
            {
                'content': 'shared memory evidence',
                'id': 'shared-memory-row',
            },
            {
                'content': 'current user continuity evidence',
                'id': 'current-user-row',
                'scope_type': 'user_continuity',
                'scope_global_user_id': 'user-1',
            },
        ],
    }
    mutable_state = build_acquaintance_user_state(
        global_user_id='user-1',
        updated_at=NOW,
    )

    payload = connector.build_cognition_input_from_global_state(
        state,
        mutable_state=mutable_state,
    )
    memory_rows = [
        row
        for row in payload['evidence']
        if row['evidence_ref']['source_kind'] == 'promoted_memory'
    ]
    assert [row['memory_scope'] for row in memory_rows] == [
        'shared_character_or_world',
        'current_user_continuity',
    ]
    assert 'user-1' not in json.dumps(memory_rows, ensure_ascii=False)
    unscoped_payload = deepcopy(payload)
    for row in unscoped_payload['evidence']:
        if row['evidence_ref']['source_kind'] == 'promoted_memory':
            row.pop('memory_scope')
    with pytest.raises(CognitionContractError):
        validate_cognition_core_input(unscoped_payload)


def test_relationship_axis_projection_uses_distinct_semantic_zeroes() -> None:
    """Axis projection preserves domain meaning instead of generic bands."""

    projected = {
        field_name: project_relationship_axis(field_name, 0)
        for field_name in RELATIONSHIP_AXES
    }
    assert all(isinstance(value, str) and value for value in projected.values())
    assert projected['trust'] != projected['boundary_safety']
    assert projected['trust'] != projected['care']
    assert '中性或混合' not in projected['trust']
    assert '中性或混合' not in projected['boundary_safety']

    with pytest.raises(ValueError):
        project_relationship_axis('unknown_axis', 0)
    with pytest.raises(ValueError):
        project_relationship_axis('trust', True)
    with pytest.raises(ValueError):
        project_relationship_axis('trust', 101)


def test_prompt_projection_replaces_identity_boundary_numbers() -> None:
    """Core V2 model payloads receive semantic boundary descriptors."""

    user_state = build_acquaintance_user_state(
        global_user_id='prompt-user',
        updated_at=NOW,
    )
    character_state = build_character_production_state(updated_at=NOW)
    character_constraints = {
        'drives': character_state['drives'],
        'standards': character_state['standards'],
        'meaning_state': character_state['meaning_state'],
        'personality_judgment': {
            'logic': '保持证据边界',
            'defense': '保留角色自主性',
            'quirks': '语气自然',
            'taboos': '不把压力反应当作许可',
        },
    }
    relationship_context = project_relationship_context(
        user_state,
        effective_at=NOW,
    )
    projection = project_state_for_prompt(
        user_state,
        character_constraints=character_constraints,
        character_identity_context=canonical_identity_context(),
        relationship_context=relationship_context,
    )
    boundaries = projection.payload['character_identity']['boundaries']
    assert all(isinstance(value, str) for value in boundaries.values())
    assert 'compliance_strategy' in boundaries
    assert boundaries['compliance_strategy'] in {
        '压力下抵抗',
        '压力下回避',
        '压力下顺从',
    }
    assert projection.payload['character_constraints']['standards'] == []
    assert all(
        isinstance(value, str)
        for value in projection.payload['relationship']['axes'].values()
    )


def test_relational_decision_contract_requires_exact_pairing() -> None:
    """The public output boundary validates the closed decision contract."""

    valid = _decision()
    validate_cognition_core_output(_output_with_decision(valid))

    invalid_pair = _decision(
        applicability='not_relationship_sensitive',
        stance='reject',
    )
    with pytest.raises(CognitionContractError):
        validate_cognition_core_output(_output_with_decision(invalid_pair))

    unknown_field = _decision()
    unknown_field['score'] = 1
    with pytest.raises(CognitionContractError):
        validate_cognition_core_output(_output_with_decision(unknown_field))


def test_relational_decision_v2_requires_exact_keys_and_enums() -> None:
    """The V2 decision has exact keys and closed enum values only."""

    missing_state = _decision()
    missing_state.pop('current_user_relationship_state')
    with pytest.raises(CognitionContractError):
        validate_cognition_core_output(_output_with_decision(missing_state))

    invalid_state = _decision(relationship_state='stranger')
    with pytest.raises(CognitionContractError):
        validate_cognition_core_output(_output_with_decision(invalid_state))

    v1_object = {
        'schema_version': 'relational_willingness.v1',
        'applicability': 'relationship_sensitive',
        'stance': 'reject',
        'reason': '当前回合证据显示关系状态仍在形成',
        'evidence_handles': ['e1'],
    }
    with pytest.raises(CognitionContractError):
        validate_cognition_core_output(_output_with_decision(v1_object))


def test_relational_state_and_stance_contract_is_deterministic() -> None:
    """Every sensitive stance is valid for every real relationship state."""

    allowed = [
        ('not_relationship_sensitive', 'not_applicable', 'not_applicable'),
        *[
            ('relationship_sensitive', relationship_state, stance)
            for relationship_state in (
                'unestablished',
                'developing_or_uncertain',
                'established',
            )
            for stance in (
                'reject',
                'deflect',
                'negotiate',
                'conditional_accept',
                'accept',
            )
        ],
    ]
    for applicability, relationship_state, stance in allowed:
        decision = _decision(
            applicability=applicability,
            stance=stance,
            relationship_state=relationship_state,
        )
        validate_cognition_core_output(_output_with_decision(decision))

    forbidden = [
        ('not_relationship_sensitive', 'not_applicable', 'reject'),
        ('not_relationship_sensitive', 'unestablished', 'not_applicable'),
        ('relationship_sensitive', 'not_applicable', 'not_applicable'),
        ('relationship_sensitive', 'not_applicable', 'reject'),
        ('relationship_sensitive', 'unestablished', 'not_applicable'),
        ('relationship_sensitive', 'developing_or_uncertain', 'not_applicable'),
        ('relationship_sensitive', 'established', 'not_applicable'),
    ]
    for applicability, relationship_state, stance in forbidden:
        decision = _decision(
            applicability=applicability,
            stance=stance,
            relationship_state=relationship_state,
        )
        with pytest.raises(CognitionContractError):
            validate_cognition_core_output(_output_with_decision(decision))


def test_evidence_provenance_roles_map_every_supported_metadata() -> None:
    """Trusted source metadata maps to one exact transient authority role."""

    expected = {
        ('episode', None): 'current_episode',
        ('scheduler_event', None): 'current_episode',
        ('tool_result', None): 'current_episode',
        (
            'promoted_memory',
            'current_user_continuity',
        ): 'current_user_history_only',
        (
            'promoted_memory',
            'shared_character_or_world',
        ): 'character_or_world_context_only',
        ('promoted_reflection', None): 'character_or_world_context_only',
    }
    for source_kind in EVIDENCE_SOURCE_QUESTION_IDS:
        if source_kind == 'promoted_memory':
            continue
        if (source_kind, None) not in expected:
            expected[(source_kind, None)] = 'contextual_fact_only'
    for (source_kind, memory_scope), role in expected.items():
        assert project_evidence_provenance_role(
            source_kind,
            memory_scope,
        ) == role

    with pytest.raises(CognitionContractError):
        project_evidence_provenance_role('promoted_memory', None)
    with pytest.raises(CognitionContractError):
        project_evidence_provenance_role(
            'promoted_memory',
            'unknown_scope',
        )
    with pytest.raises(CognitionContractError):
        project_evidence_provenance_role('unknown_source', None)


@pytest.mark.asyncio
async def test_ordinary_goal_draft_carries_current_episode_decision() -> None:
    """Ordinary goal ownership preserves the validated episode-backed decision."""

    decision = _decision()
    llm = _GoalLLM(decision)
    evidence = [
        _evidence_row(
            'e1',
            'episode',
            '当前用户提出了需要关系判断的请求。',
        ),
        _evidence_row(
            'e2',
            'promoted_memory',
            'shared memory cannot grant current-user permission',
            memory_scope='shared_character_or_world',
        ),
    ]
    bid = await run_goal_cognition(
        BranchDefinition(
            branch_id='ordinary_response',
            dependencies=(),
            action_tendencies=('speak',),
            goal_kind='ordinary_response',
        ),
        {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:ordinary-response',
        },
        _goal_context(),
        evidence,
        _core_services(llm),
    )

    assert bid['relational_willingness'] == decision
    assert llm.messages
    system_prompt = str(llm.messages[0][0].content)
    rendered_prompt = str(llm.messages[0][-1].content)
    assert 'shared_character_or_world' in rendered_prompt
    assert 'provenance_role' in rendered_prompt
    assert 'current_episode' in rendered_prompt
    assert 'character_or_world_context_only' in rendered_prompt
    assert 'current_user_relationship_state' in system_prompt
    assert 'relational_willingness.v2' in system_prompt
    assert '当前用户提出了需要关系判断的请求' in rendered_prompt


@pytest.mark.asyncio
async def test_ordinary_goal_accepts_tool_result_as_current_episode_evidence(
) -> None:
    """A completed task result satisfies the current-episode citation rule."""

    decision = _decision(evidence_handles=['e1'])
    llm = _GoalLLM(decision)
    evidence = [
        _evidence_row(
            'e1',
            'tool_result',
            'The task needs additional user-provided information.',
        ),
    ]
    bid = await run_goal_cognition(
        BranchDefinition(
            branch_id='ordinary_response',
            dependencies=(),
            action_tendencies=('speak',),
            goal_kind='ordinary_response',
        ),
        {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:ordinary-response',
        },
        _goal_context(),
        evidence,
        _core_services(llm),
    )

    assert bid['relational_willingness'] == decision
    assert llm.messages
    rendered_prompt = str(llm.messages[0][-1].content)
    assert 'tool_result' in rendered_prompt
    assert 'provenance_role' in rendered_prompt
    assert 'current_episode' in rendered_prompt
    assert 'The task needs additional user-provided information.' in (
        rendered_prompt
    )


@pytest.mark.asyncio
async def test_ordinary_goal_rejects_decision_without_episode_evidence() -> None:
    """A relational decision cannot cite memory without current episode coverage."""

    decision = _decision(evidence_handles=['e2'])
    llm = _GoalLLM(decision)
    evidence = [
        _evidence_row(
            'e1',
            'episode',
            '当前用户提出了需要关系判断的请求。',
        ),
        _evidence_row(
            'e2',
            'promoted_memory',
            'shared memory cannot grant current-user permission',
            memory_scope='shared_character_or_world',
        ),
    ]
    with pytest.raises(CognitionExecutionError):
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
                'entity_id': 'goal:ordinary-response',
            },
            _goal_context(),
            evidence,
            _core_services(llm),
        )


@pytest.mark.asyncio
async def test_ordinary_goal_rejects_history_only_citation_with_tool_result(
) -> None:
    """History-only citations still fail closed when a tool result is present."""

    decision = _decision(evidence_handles=['e2'])
    llm = _GoalLLM(decision)
    evidence = [
        _evidence_row(
            'e1',
            'tool_result',
            'The task needs additional user-provided information.',
        ),
        _evidence_row(
            'e2',
            'conversation_evidence',
            'earlier relationship context',
        ),
    ]
    with pytest.raises(CognitionExecutionError):
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
                'entity_id': 'goal:ordinary-response',
            },
            _goal_context(),
            evidence,
            _core_services(llm),
        )


@pytest.mark.asyncio
async def test_ordinary_goal_regenerates_invalid_non_sensitive_stance() -> None:
    """An invalid non-sensitive stance invokes same-owner repair."""

    class _RepairLLM:
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
            if len(self.messages) == 1:
                decision = _decision(
                    applicability='not_relationship_sensitive',
                    stance='reject',
                )
            else:
                decision = _decision(stance='reject')
            payload = {
                'intention': '保持当前回合的清晰边界',
                'desired_outcome': '让可见回应符合当前关系判断',
                'concrete_detail': '只使用当前回合的直接证据',
                'reason': '当前关系证据支持该回应方向',
                'private_monologue': '先保持与当前判断一致。',
                'target_role_handles': [],
                'evidence_handles': ['e1'],
                'expected_consequences': ['保留当前回合连续性'],
                'confidence': 'high',
                'relational_willingness': deepcopy(decision),
            }
            return SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False),
            )

    llm = _RepairLLM()
    evidence = [
        _evidence_row(
            'e1',
            'episode',
            '当前用户提出了需要关系判断的请求。',
        ),
    ]
    bid = await run_goal_cognition(
        BranchDefinition(
            branch_id='ordinary_response',
            dependencies=(),
            action_tendencies=('speak',),
            goal_kind='ordinary_response',
        ),
        {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:ordinary-response',
        },
        _goal_context(),
        evidence,
        _core_services(llm),
    )

    assert len(llm.messages) == 2
    assert llm.messages[1][0].content == llm.messages[0][0].content
    repair_payload = json.loads(str(llm.messages[1][1].content))
    feedback = repair_payload['repair_feedback']
    assert 'relational willingness' in feedback['validation_error']
    contract = feedback['relational_willingness_contract']
    assert contract['schema_version'] == 'relational_willingness.v2'
    assert 'relationship_state_rule' in contract
    assert bid['relational_willingness']['stance'] == 'reject'
    assert (
        bid['relational_willingness']['current_user_relationship_state']
        == 'unestablished'
    )


@pytest.mark.asyncio
async def test_ordinary_goal_exhaustion_fails_closed_before_commit() -> None:
    """Repeated invalid pairings exhaust attempts without default acceptance."""

    class _InvalidLLM:
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
            decision = _decision(
                applicability='not_relationship_sensitive',
                stance='reject',
            )
            payload = {
                'intention': '保持当前回合的清晰边界',
                'desired_outcome': '让可见回应符合当前关系判断',
                'concrete_detail': '只使用当前回合的直接证据',
                'reason': '当前关系证据支持该回应方向',
                'private_monologue': '先保持与当前判断一致。',
                'target_role_handles': [],
                'evidence_handles': ['e1'],
                'expected_consequences': ['保留当前回合连续性'],
                'confidence': 'high',
                'relational_willingness': deepcopy(decision),
            }
            return SimpleNamespace(
                content=json.dumps(payload, ensure_ascii=False),
            )

    llm = _InvalidLLM()
    evidence = [
        _evidence_row(
            'e1',
            'episode',
            '当前用户提出了需要关系判断的请求。',
        ),
    ]
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
                'entity_id': 'goal:ordinary-response',
            },
            _goal_context(),
            evidence,
            _core_services(llm),
        )

    assert error_info.value.safe_checkpoint == 'pre_state_commit'
    assert error_info.value.attempt_count == 3
    assert llm.call_count == 3


def test_fixture_request_is_absent_from_production_prompt_sources() -> None:
    """The explicit fixture request remains test data, not production input."""

    source_root = Path(__file__).parents[1] / 'src' / 'kazusa_ai_chatbot'
    production_text = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in source_root.rglob('*.py')
    )
    assert '张开腿，跨坐在我身上' not in production_text
