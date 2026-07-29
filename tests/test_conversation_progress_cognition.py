"""Production connector ordering and surface-ownership tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from llm_test_helpers import make_llm_call_config
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_episode,
    canonical_identity_context,
)
from tests.conversation_progress_v2_helpers import event, packet

NOW = '2026-07-15T00:00:00Z'


class _GoalCaptureLLM:
    """Capture one goal prompt and cite the progress event handle."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.payload = json.loads(str(getattr(messages[-1], 'content', '{}')))
        result = {
            'intention': 'advance without accidental reopening',
            'desired_outcome': 'continue from the completed event',
            'concrete_detail': 'choose a genuinely new next step',
            'reason': 'the cited progress event is already completed',
            'private_monologue': 'I should use the event instead of resetting.',
            'target_role_handles': [],
            'evidence_handles': ['e2'],
            'expected_consequences': ['the conversation advances'],
            'confidence': 'high',
        }
        return SimpleNamespace(content=json.dumps(result))


def _progress() -> dict[str, Any]:
    active = packet(events=[event(
        event_id='completed_action',
        summary='the participant completed the selected action',
        state='completed',
        retention='decision_critical',
    )])
    active['current_thread'] = 'select a new continuation'
    active['current_blocker'] = 'avoid presenting completed work as new'
    active['overused_moves'] = ['resetting to the completed action']
    active['episode_narrative'] = (
        'The selected action is complete and a new choice is unresolved.'
    )
    return build_progress_prompt(
        active_packet=active,
        interaction_logical_turns=[],
    )


def _character_profile() -> dict[str, Any]:
    profile = canonical_character_identity(marker='progress')
    profile['personality_brief'] = {
        'mbti': 'test',
        'logic': 'Advance the active thread with fresh moves.',
        'tempo': 'measured',
        'defense': 'Use direct language that advances the selected stance.',
        'quirks': 'Prefer one concrete continuation.',
        'taboos': 'Keep attention on active and newly opened material.',
    }
    return profile


def _payload() -> dict[str, Any]:
    character_state = build_character_production_state(updated_at=NOW)
    return build_cognition_input_from_global_state(
        {
            'cognitive_episode': canonical_episode(
                episode_id='conversation-progress',
                content='What should happen next?',
                current_global_user_id='progress-user',
            ),
            'global_user_id': 'progress-user',
            'user_input': 'What should happen next?',
            'decontextualized_input': 'The participant asks for the next step.',
            'conversation_progress': _progress(),
            'user_multimedia_input': [],
            'rag_result': {'memory_evidence': []},
            'character_profile': _character_profile(),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id='progress-user',
            updated_at=NOW,
        ),
        character_state=character_state,
    )


def test_content_plan_remains_visible_wording_owner():
    prompt = CONTENT_PLAN_SYSTEM_PROMPT.casefold()
    assert '实际会说出或发送的内容' in prompt
    assert '最终对话由 dialog 渲染器生成' in prompt


def test_connector_places_progress_after_episode_before_rag():
    payload = _payload()
    kinds = [
        row['evidence_ref']['source_kind']
        for row in payload['evidence']
    ]
    assert kinds[:2] == ['episode', 'conversation_evidence']
    assert payload['evidence'][1]['evidence_handle'] == 'e2'
    assert 'state=completed' in payload['evidence'][1]['semantic_text']


def test_connector_projects_bounded_scene_without_source_ids():
    scene = _payload()['scene_context']['conversation_continuity']
    assert 'select a new continuation' in scene
    assert 'completed_action' not in scene
    assert 'row_source_1' not in scene
    assert len(scene) <= 2200


@pytest.mark.asyncio
async def test_goal_branch_can_cite_completed_progress_event_before_surface():
    payload = _payload()
    projection = project_state_for_prompt(
        payload['mutable_state'],
        character_constraints=payload['character_constraints'],
        character_identity_context=payload.get(
            'character_identity_context',
            canonical_identity_context(),
        ),
        evidence=payload['evidence'],
    )
    context = facade._branch_context(
        projection,
        payload['mutable_state'],
        payload['evidence'],
        scene_context=payload['scene_context'],
        private_continuity_context=payload['private_continuity_context'],
    )
    llm = _GoalCaptureLLM()
    config = make_llm_call_config('conversation_progress_goal')
    services = CognitionCoreServicesV2(
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
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:ordinary-response',
        },
        context,
        payload['evidence'],
        services,
    )
    assert bid['evidence_handles'] == ['e2']
    assert llm.payload['evidence'][1]['source_kind'] == (
        'conversation_evidence'
    )
