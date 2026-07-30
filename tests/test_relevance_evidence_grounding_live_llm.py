"""Real-model gates for evidence-grounded relevance admission."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_character_production_state,
)
from kazusa_ai_chatbot.config import (
    RELEVANCE_AGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.relevance.participation_evidence import (
    validate_participation_assessment,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.test_relevance_turn_settlement_live_llm import (
    _CapturingLLM,
    ensure_relevance_live_llms,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]
frontline_module = import_module(
    'kazusa_ai_chatbot.relevance.frontline_relevance_agent',
)
settled_module = import_module(
    'kazusa_ai_chatbot.relevance.persona_relevance_agent',
)
_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / 'test_artifacts'
    / 'relevance_evidence_grounding'
)
_NOW = '2026-07-30T00:00:00Z'
_CHARACTER_NAME = '一之濑明日奈'
_INCIDENT = '直接找你一换一是吧'


def _character_state(description: str) -> dict[str, Any]:
    """Build one active goal whose text can intersect the current message."""

    state = build_character_production_state(updated_at=_NOW)
    state['goals'].append({
        'entity_id': 'goal:redacted-live-evidence',
        'description': description,
        'salience': 80,
        'role_refs': [],
        'evidence_refs': [],
        'created_at': _NOW,
        'updated_at': _NOW,
        'goal_kind': 'safety',
        'status': 'pursuing',
        'importance': 80,
        'progress': 10,
        'obstruction': 40,
        'expected_success': 60,
        'controllability': 60,
        'recoverability': 70,
        'urgency': 75,
    })
    return state


def _frontline_state(
    body_text: str,
    *,
    targets: list[str] | None = None,
    reply_target: str = 'none',
    character_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded frontline state for the captured admission boundary."""

    return {
        'conversation_scope': 'group',
        'active_character_name': _CHARACTER_NAME,
        'current_message': {
            'body_text': body_text,
            'semantic_target_labels': targets or [],
            'reply_target_label': reply_target,
            'media_labels': [],
        },
        'open_turns': [],
        'recent_preludes': [],
        'latest_bot_continuity': '',
        'character_cognition_state': (
            character_state
            or build_character_production_state(updated_at=_NOW)
        ),
    }


def _settled_state(
    body_text: str,
    *,
    targets: list[str] | None = None,
    reply_target: str = 'none',
    character_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a bounded settled state for the same admission boundary."""

    return {
        'conversation_scope': 'group',
        'active_character_name': _CHARACTER_NAME,
        'assembled_fragments': [{
            'body_text': body_text,
            'semantic_target_labels': targets or [],
            'reply_target_label': reply_target,
            'media_labels': [],
        }],
        'media_descriptions': [],
        'fresh_history': [],
        'scene_context': 'A group conversation.',
        'relationship_context': 'The author is a group participant.',
        'group_attention': 'low_noise',
        'bot_continuity': '',
        'character_cognition_state': (
            character_state
            or build_character_production_state(updated_at=_NOW)
        ),
        'observation_status': 'observation_complete',
    }


def _assessment(raw_output: dict[str, Any]) -> dict[str, Any]:
    """Select the model-only assessment fields retained in raw evidence."""

    keys = (
        'recipient_relation',
        'admission_basis',
        'interaction_evidence_refs',
        'character_state_refs',
    )
    return {
        key: raw_output.get(key)
        for key in keys
    }


def _validated_assessment(
    *,
    raw_output: dict[str, Any],
    model_payload: dict[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Mirror the production assessment boundary for artifact reporting."""

    action_key = (
        'intake_action'
        if stage == 'frontline'
        else 'response_action'
    )
    try:
        assessment = validate_participation_assessment(
            raw_output,
            interaction_evidence=model_payload.get(
                'interaction_evidence',
                [],
            ),
            character_state_evidence=model_payload.get(
                'character_state_evidence',
                [],
            ),
            stage=stage,
            action=str(raw_output.get(action_key, '')),
            append_target=str(raw_output.get('append_target', 'none')),
            use_reply_feature=bool(
                raw_output.get('use_reply_feature', False),
            ),
        )
    except ValueError as exc:
        assessment = {
            'recipient_relation': 'unknown',
            'admission_basis': 'none',
            'interaction_evidence_refs': [],
            'character_state_refs': [],
        }
        validation = {
            'status': 'invalid_fail_closed',
            'reason': str(exc),
        }
        return assessment, validation
    validation = {
        'status': 'valid',
        'reason': '',
    }
    return assessment, validation


def _write_artifact(
    *,
    case_id: str,
    stage: str,
    state: dict[str, Any],
    messages: list[Any],
    raw_response_text: str,
    public_decision: dict[str, Any],
    expected_action: str,
    duration_ms: int,
    input_cap: int,
    completion_cap: int,
) -> None:
    """Write raw structured evidence for parent-authored quality review."""

    rendered_input = ''.join(str(message.content) for message in messages)
    model_payload = json.loads(str(messages[1].content))
    raw_output = parse_llm_json_output(
        raw_response_text,
        deterministic_only=True,
    )
    validated_assessment, assessment_validation = _validated_assessment(
        raw_output=raw_output,
        model_payload=model_payload,
        stage=stage,
    )
    actual_action = str(
        public_decision.get('intake_action')
        or public_decision.get('response_action')
        or ''
    )
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        'case_id': case_id,
        'stage': stage,
        'redacted_semantic_input': state,
        'rendered_input_chars': len(rendered_input),
        'rendered_input_sha256': hashlib.sha256(
            rendered_input.encode('utf-8'),
        ).hexdigest(),
        'route': 'RELEVANCE_AGENT_LLM',
        'model': RELEVANCE_AGENT_LLM_MODEL,
        'configured_limits': {
            'input_chars': input_cap,
            'completion_tokens': completion_cap,
            'thinking_enabled': False,
        },
        'raw_output': raw_response_text,
        'parsed_public_decision': public_decision,
        'model_participation_assessment': _assessment(raw_output),
        'validated_participation_assessment': validated_assessment,
        'participation_validation': assessment_validation,
        'available_refs': {
            'interaction': model_payload.get('interaction_evidence', []),
            'character_state': model_payload.get(
                'character_state_evidence',
                [],
            ),
        },
        'cited_refs': {
            'interaction': raw_output.get(
                'interaction_evidence_refs',
                [],
            ),
            'character_state': raw_output.get(
                'character_state_refs',
                [],
            ),
        },
        'expected_outcome': expected_action,
        'automated_judgment': {
            'actual_action': actual_action,
            'passed': actual_action == expected_action,
        },
        'duration_ms': duration_ms,
    }
    path = _ARTIFACT_DIR / f'{case_id}.json'
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


async def _run_frontline(
    case_id: str,
    state: dict[str, Any],
    expected_action: str,
) -> dict[str, Any]:
    """Run one frontline case and persist its raw structured evidence."""

    messages = frontline_module.build_frontline_messages(state)
    captured_llm = _CapturingLLM(
        frontline_module._frontline_relevance_agent_llm,
    )
    started_at = perf_counter()
    with patch.object(
        frontline_module,
        '_frontline_relevance_agent_llm',
        captured_llm,
    ):
        decision = await frontline_module.frontline_relevance_agent(state)
    duration_ms = int((perf_counter() - started_at) * 1000)
    _write_artifact(
        case_id=case_id,
        stage='frontline',
        state=state,
        messages=messages,
        raw_response_text=captured_llm.raw_response_text,
        public_decision=decision,
        expected_action=expected_action,
        duration_ms=duration_ms,
        input_cap=frontline_module.FRONTLINE_RELEVANCE_MAX_INPUT_CHARS,
        completion_cap=(
            frontline_module.FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS
        ),
    )
    return decision


async def _run_settled(
    case_id: str,
    state: dict[str, Any],
    expected_action: str,
) -> dict[str, Any]:
    """Run one settled case and persist its raw structured evidence."""

    messages = settled_module.build_settled_relevance_messages(
        state,
        observation_status='observation_complete',
    )
    captured_llm = _CapturingLLM(settled_module._relevance_agent_llm)
    started_at = perf_counter()
    with patch.object(
        settled_module,
        '_relevance_agent_llm',
        captured_llm,
    ):
        decision = await settled_module.relevance_agent(state)
    duration_ms = int((perf_counter() - started_at) * 1000)
    _write_artifact(
        case_id=case_id,
        stage='settled',
        state=state,
        messages=messages,
        raw_response_text=captured_llm.raw_response_text,
        public_decision=decision,
        expected_action=expected_action,
        duration_ms=duration_ms,
        input_cap=settled_module.SETTLED_RELEVANCE_MAX_INPUT_CHARS,
        completion_cap=(
            settled_module.SETTLED_RELEVANCE_MAX_COMPLETION_TOKENS
        ),
    )
    return decision


async def test_live_incident_frontline_discards(
    ensure_relevance_live_llms,
) -> None:
    """The captured pronoun and incidental glyph stay outside cognition."""

    del ensure_relevance_live_llms
    decision = await _run_frontline(
        'incident_frontline_discards',
        _frontline_state(_INCIDENT),
        'discard',
    )
    assert decision['intake_action'] == 'discard'


async def test_live_incident_settled_ignores(
    ensure_relevance_live_llms,
) -> None:
    """The same unsupported incident remains silent at settlement."""

    del ensure_relevance_live_llms
    decision = await _run_settled(
        'incident_settled_ignores',
        _settled_state(_INCIDENT),
        'ignore',
    )
    assert decision['response_action'] == 'ignore'


async def test_live_canonical_name_frontline_starts(
    ensure_relevance_live_llms,
) -> None:
    """A complete natural name address remains interaction-relevant."""

    del ensure_relevance_live_llms
    decision = await _run_frontline(
        'canonical_name_frontline_starts',
        _frontline_state('一之濑明日奈，你怎么看这次一换一？'),
        'start',
    )
    assert decision['intake_action'] == 'start'


async def test_live_canonical_name_settled_proceeds(
    ensure_relevance_live_llms,
) -> None:
    """A complete natural name address still enters cognition."""

    del ensure_relevance_live_llms
    decision = await _run_settled(
        'canonical_name_settled_proceeds',
        _settled_state('一之濑明日奈，你怎么看这次一换一？'),
        'proceed',
    )
    assert decision['response_action'] == 'proceed'


async def test_live_state_salience_frontline_starts_for_other_recipient(
    ensure_relevance_live_llms,
) -> None:
    """Concrete active-state salience may admit other-recipient speech."""

    del ensure_relevance_live_llms
    state = _character_state('阻止群里的一换一挑战伤害参与者')
    decision = await _run_frontline(
        'state_salience_frontline_starts_for_other_recipient',
        _frontline_state(
            '小林，今晚的一换一挑战会把你带进危险区域',
            targets=['other_participant'],
            reply_target='other_participant',
            character_state=state,
        ),
        'start',
    )
    assert decision['intake_action'] == 'start'


async def test_live_state_salience_settled_proceeds_without_reply_anchor(
    ensure_relevance_live_llms,
) -> None:
    """State-grounded speech preserves the other recipient and no reply."""

    del ensure_relevance_live_llms
    state = _character_state('阻止群里的一换一挑战伤害参与者')
    decision = await _run_settled(
        'state_salience_settled_proceeds_without_reply_anchor',
        _settled_state(
            '小林，今晚的一换一挑战会把你带进危险区域',
            targets=['other_participant'],
            reply_target='other_participant',
            character_state=state,
        ),
        'proceed',
    )
    assert decision['response_action'] == 'proceed'
    assert decision['use_reply_feature'] is False


async def test_live_unmatched_state_frontline_discards(
    ensure_relevance_live_llms,
) -> None:
    """Unrelated active state supplies no reason to enter a side exchange."""

    del ensure_relevance_live_llms
    state = _character_state('修好自己房间里损坏的旧相机')
    decision = await _run_frontline(
        'unmatched_state_frontline_discards',
        _frontline_state(
            '小林，今晚的一换一挑战会把你带进危险区域',
            targets=['other_participant'],
            reply_target='other_participant',
            character_state=state,
        ),
        'discard',
    )
    assert decision['intake_action'] == 'discard'


async def test_live_unmatched_state_settled_ignores(
    ensure_relevance_live_llms,
) -> None:
    """Unrelated active state cannot admit another participant's message."""

    del ensure_relevance_live_llms
    state = _character_state('修好自己房间里损坏的旧相机')
    decision = await _run_settled(
        'unmatched_state_settled_ignores',
        _settled_state(
            '小林，今晚的一换一挑战会把你带进危险区域',
            targets=['other_participant'],
            reply_target='other_participant',
            character_state=state,
        ),
        'ignore',
    )
    assert decision['response_action'] == 'ignore'
