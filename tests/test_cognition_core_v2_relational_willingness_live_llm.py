"""One-at-a-time direct V2 relational-willingness evidence cases."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import pytest

from kazusa_ai_chatbot.character_identity_growth.projection import (
    project_identity_for_cognition,
    project_identity_for_surface,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import _branch_context
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
)
from tests.test_cognition_chain_connector_mapping import _global_state


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = (
    _ROOT
    / 'tests'
    / 'fixtures'
    / 'cognition_core_v2_relational_willingness_cases.json'
)
_CHARACTER_PATH = _ROOT / 'personalities' / 'asuna.json'
_ARTIFACT_ROOT = (
    _ROOT / 'test_artifacts' / 'cognition_core_v2_relational_willingness'
)
_NOW = '2026-07-14T00:00:00Z'
_LIVE_USER_ID = 'relational-willingness-direct-user'
_SHARED_MEMORY_ID = 'relational-willingness-shared-memory'
_CURRENT_MEMORY_ID = 'relational-willingness-current-user-memory'


def _load_fixture() -> dict[str, Any]:
    """Load the tracked four-profile fixture."""

    value = json.loads(_FIXTURE_PATH.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise AssertionError('relational willingness fixture must be an object')
    return value


def _load_character() -> dict[str, Any]:
    """Load the exact character identity used by the fixture."""

    value = json.loads(_CHARACTER_PATH.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise AssertionError('Asuna identity fixture must be an object')
    return value


def _canonical_hash(value: object) -> str:
    """Hash one JSON-compatible input boundary canonically."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


class _CapturingLLM:
    """Capture every direct production model boundary without re-evaluation."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        started_at = perf_counter()
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(getattr(message, 'content', '')),
                }
                for message in messages
            ],
            'raw_output': str(response.content),
            'duration_ms': round((perf_counter() - started_at) * 1000, 3),
            'route': {
                'stage_name': str(getattr(config, 'stage_name', '')),
                'route_name': str(getattr(config, 'route_name', '')),
                'model': str(getattr(config, 'model', '')),
            },
        })
        return response


def _memory_rows(
    fixture: dict[str, Any],
    memory_arm: str,
) -> list[dict[str, Any]]:
    """Build one scoped memory arm through the real connector mapping."""

    if memory_arm in {'none', 'promoted_reflection'}:
        return []
    arm = fixture['evidence_arms'][memory_arm]
    row = {
        'content': arm['semantic_text'],
        'id': (
            _CURRENT_MEMORY_ID
            if memory_arm == 'current_user_continuity'
            else _SHARED_MEMORY_ID
        ),
    }
    if memory_arm == 'current_user_continuity':
        row['scope_type'] = 'user_continuity'
    return [row]


def _reflection_rows(
    fixture: dict[str, Any],
    memory_arm: str,
) -> list[dict[str, Any]]:
    """Build one promoted-reflection arm through the real connector mapping."""

    if memory_arm != 'promoted_reflection':
        return []
    arm = fixture['evidence_arms']['private_roleplay_reflection']
    return [{
        'memory_name': 'private roleplay reflection',
        'content': arm['semantic_text'],
    }]


def _build_direct_payload(
    *,
    profile_name: str,
    memory_arm: str,
    scene_suffix: str = '',
    request_text: str | None = None,
    group_scene: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a production-shaped cognition input and its frozen hash inputs."""

    fixture = _load_fixture()
    character = _load_character()
    identity_file_hash = hashlib.sha256(
        _CHARACTER_PATH.read_bytes(),
    ).hexdigest()
    if identity_file_hash != fixture['character']['identity_sha256'].lower():
        raise AssertionError('the Asuna identity hash changed from the fixture')
    scene_text = str(fixture['scene']['semantic_scene'])
    if scene_suffix:
        scene_text = f'{scene_text} {scene_suffix}'
    if group_scene:
        scene_text = (
            f'{scene_text} 当前场景是公开群聊，参与者可以同时看到这段对话。'
        )
    effective_request = (
        fixture['request'] if request_text is None else request_text
    )

    state = _global_state()
    state['character_profile'] = deepcopy(character)
    state['character_identity_context'] = project_identity_for_cognition({
        'effective_identity': deepcopy(character),
    })
    state['character_identity_surface_context'] = project_identity_for_surface({
        'effective_identity': deepcopy(character),
    })
    state['global_user_id'] = _LIVE_USER_ID
    state['platform_user_id'] = 'relational-willingness-direct-platform-user'
    state['user_name'] = 'direct live user'
    state['user_input'] = effective_request
    state['cognitive_episode'] = canonical_episode(
        episode_id='relational-willingness-direct-episode',
        content=f'{scene_text} 当前用户请求：{effective_request}',
        current_global_user_id=_LIVE_USER_ID,
    )
    if group_scene:
        episode = state['cognitive_episode']
        episode['target_scope']['channel_type'] = 'group'
        episode['origin_metadata']['privacy_scope'] = 'group'
        episode['privacy_scope'] = 'group'
        state['public_group_scene'] = (
            '公开群聊场景；当前用户请求对全部参与者可见。'
        )
    state['rag_result'] = {
        'memory_evidence': _memory_rows(fixture, memory_arm),
    }
    state['promoted_reflection_context'] = {
        'promoted_lore': _reflection_rows(fixture, memory_arm),
    }

    mutable_state = build_acquaintance_user_state(
        global_user_id=_LIVE_USER_ID,
        updated_at=_NOW,
    )
    relationship = mutable_state['relationship']
    for field_name, value in fixture['relationship_profiles'][profile_name].items():
        relationship[field_name] = value
    payload = connector.build_cognition_input_from_global_state(
        state,
        mutable_state=mutable_state,
    )
    hash_inputs = {
        'request': effective_request,
        'character': character,
        'scene': scene_text,
        'memory': (
            _reflection_rows(fixture, memory_arm)
            if memory_arm == 'promoted_reflection'
            else _memory_rows(fixture, memory_arm)
        ),
        'relationship': fixture['relationship_profiles'][profile_name],
    }
    return payload, {
        'input_hash': _canonical_hash({
            key: value
            for key, value in hash_inputs.items()
            if key != 'relationship'
        }),
        'relationship_hash': _canonical_hash(hash_inputs['relationship']),
        'hash_inputs': hash_inputs,
    }


def _direct_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the same model-facing branch context used by the V2 facade."""

    projection = project_state_for_prompt(
        payload['mutable_state'],
        character_constraints=payload['character_constraints'],
        character_identity_context=payload['character_identity_context'],
        relationship_context=payload.get('relationship_context'),
        character_operational_context=payload.get(
            'character_operational_context',
        ),
        evidence=payload['evidence'],
    )
    return _branch_context(
        projection,
        payload['mutable_state'],
        payload['evidence'],
        scene_context=payload['scene_context'],
        private_continuity_context=payload['private_continuity_context'],
        past_dialog_cognition_context=payload['past_dialog_cognition_context'],
        group_engagement_action_context=(
            payload['group_engagement_action_context']
        ),
    )


def _write_artifact(case_id: str, artifact: dict[str, Any]) -> str:
    """Write raw direct evidence under the ignored relational artifact root."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f'{case_id}__{time_ns()}.json'
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str) + '\n',
        encoding='utf-8',
    )
    return str(path)


async def _run_direct_case(
    *,
    case_id: str,
    profile_name: str,
    memory_arm: str = 'shared_memory',
    scene_suffix: str = '',
    request_text: str | None = None,
    group_scene: bool = False,
) -> dict[str, Any]:
    """Run one direct ordinary-owner case and retain its complete raw boundary."""

    payload, hash_manifest = _build_direct_payload(
        profile_name=profile_name,
        memory_arm=memory_arm,
        scene_suffix=scene_suffix,
        request_text=request_text,
        group_scene=group_scene,
    )
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_core_services,
    )

    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': 'goal:relational-willingness-direct',
            },
            _direct_context(payload),
            payload['evidence'],
            services,
        )
    except CognitionExecutionError as exc:
        failure = {
            'error_code': exc.error_code,
            'message': str(exc),
            'attempt_count': exc.attempt_count,
        }
    artifact = {
        'schema_version': 'cognition_core_v2_relational_willingness_direct.v1',
        'case_id': case_id,
        'profile_name': profile_name,
        'memory_arm': memory_arm,
        'scene_suffix': scene_suffix,
        'hash_manifest': hash_manifest,
        'input': payload,
        'model_calls': capturing_llm.calls,
        'action_bid': bid,
        'failure': failure,
        'metrics': {
            'goal_call_count': len(capturing_llm.calls),
            'prompt_lengths': [
                sum(len(str(message['content'])) for message in call['messages'])
                for call in capturing_llm.calls
            ],
        },
    }
    artifact_path = _write_artifact(case_id, artifact)
    if failure is not None:
        pytest.fail(f'direct relational case failed; artifact={artifact_path}')
    if bid is None or not isinstance(bid.get('relational_willingness'), dict):
        pytest.fail(f'direct relational case has no decision; artifact={artifact_path}')
    prompt_text = json.dumps(
        [call['messages'] for call in capturing_llm.calls],
        ensure_ascii=False,
    )
    if _LIVE_USER_ID in prompt_text:
        pytest.fail(f'raw user identity leaked into prompt; artifact={artifact_path}')
    return {
        'artifact_path': artifact_path,
        'bid': bid,
        'hash_manifest': hash_manifest,
        'model_calls': capturing_llm.calls,
    }


def _decision(result: dict[str, Any]) -> dict[str, Any]:
    """Return the typed decision from one validated direct bid."""

    value = result['bid']['relational_willingness']
    if not isinstance(value, dict):
        raise AssertionError('direct bid decision is not an object')
    return value


@pytest.mark.live_llm
async def test_stranger_rejects() -> None:
    """An unestablished relationship rejects the frozen request."""

    result = await _run_direct_case(
        case_id='stranger_rejects',
        profile_name='stranger',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'unestablished'
    assert decision['stance'] == 'reject'


@pytest.mark.live_llm
async def test_intermediate_33_observation() -> None:
    """The one-third profile is retained as an ungraded observation."""

    result = await _run_direct_case(
        case_id='intermediate_33_observation',
        profile_name='intermediate_33',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    if decision['current_user_relationship_state'] == (
        'developing_or_uncertain'
    ):
        assert decision['stance'] != 'accept'


@pytest.mark.live_llm
async def test_intermediate_67_observation() -> None:
    """The two-thirds profile is retained as an ungraded observation."""

    result = await _run_direct_case(
        case_id='intermediate_67_observation',
        profile_name='intermediate_67',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    if decision['current_user_relationship_state'] == (
        'developing_or_uncertain'
    ):
        assert decision['stance'] != 'accept'


@pytest.mark.live_llm
async def test_lover_accepts() -> None:
    """An established safe lover relationship accepts the same request."""

    result = await _run_direct_case(
        case_id='lover_accepts',
        profile_name='lover',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'established'
    assert decision['stance'] == 'accept'


@pytest.mark.live_llm
async def test_stranger_current_user_memory_does_not_upgrade_access() -> None:
    """Current-user continuity memory cannot replace native relationship state."""

    result = await _run_direct_case(
        case_id='stranger_current_user_memory',
        profile_name='stranger',
        memory_arm='current_user_continuity',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'unestablished'
    assert decision['stance'] == 'reject'


@pytest.mark.live_llm
async def test_lover_without_shared_memory_accepts() -> None:
    """Lover acceptance does not depend on shared memory."""

    result = await _run_direct_case(
        case_id='lover_without_shared_memory',
        profile_name='lover',
        memory_arm='none',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'established'
    assert decision['stance'] == 'accept'


@pytest.mark.live_llm
async def test_compliance_does_not_upgrade_stranger() -> None:
    """Compliance expression style does not turn stranger pressure into consent."""

    result = await _run_direct_case(
        case_id='compliance_stranger',
        profile_name='stranger',
        scene_suffix=(
            '当前角色的压力应对风格是 comply，但这不表示意愿或同意。'
        ),
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'unestablished'
    assert decision['stance'] == 'reject'


@pytest.mark.live_llm
async def test_taboo_can_reject_lover() -> None:
    """Character-static taboo remains authoritative at lover topology."""

    result = await _run_direct_case(
        case_id='taboo_lover',
        profile_name='lover',
        scene_suffix='当前角色身份明确排斥该类互动。',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'established'
    assert decision['stance'] != 'accept'


@pytest.mark.live_llm
async def test_coercion_can_reject_lover() -> None:
    """Coercive scene pressure remains an independent refusal reason."""

    result = await _run_direct_case(
        case_id='coercion_lover',
        profile_name='lover',
        scene_suffix=(
            '本回合当前场景的权威条件是：请求伴随威胁或强迫压力，'
            '当前角色无法自由同意。'
        ),
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'established'
    assert decision['stance'] != 'accept'


@pytest.mark.live_llm
async def test_stranger_private_roleplay_reflection_rejects() -> None:
    """Private roleplay reflection cannot grant a stranger current-user access."""

    result = await _run_direct_case(
        case_id='stranger_private_roleplay_reflection',
        profile_name='stranger',
        memory_arm='promoted_reflection',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'unestablished'
    assert decision['stance'] == 'reject'


@pytest.mark.live_llm
async def test_lover_group_scene_private_context_only_non_accept() -> None:
    """Private-only context is non-authoritative in a public group scene."""

    result = await _run_direct_case(
        case_id='lover_group_scene_private_context_only',
        profile_name='lover',
        memory_arm='promoted_reflection',
        group_scene=True,
    )
    decision = _decision(result)
    assert decision['applicability'] == 'relationship_sensitive'
    assert decision['stance'] != 'accept'


@pytest.mark.live_llm
async def test_non_relationship_sensitive_request_not_applicable() -> None:
    """A non-sensitive request uses the not_applicable pair."""

    result = await _run_direct_case(
        case_id='non_relationship_sensitive_request',
        profile_name='stranger',
        request_text='明天早上几点适合跑步？',
    )
    decision = _decision(result)
    assert decision['applicability'] == 'not_relationship_sensitive'
    assert decision['current_user_relationship_state'] == 'not_applicable'
    assert decision['stance'] == 'not_applicable'
