"""Deterministic participation-evidence contract tests."""

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import (
    build_character_production_state,
)
from kazusa_ai_chatbot.relevance.participation_evidence import (
    build_interaction_evidence,
    project_character_state_evidence,
    validate_participation_assessment,
)


_NOW = "2026-07-30T00:00:00Z"
_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "relevance"
    / "a72b0182de174ec0b0ff533891c2e294.json"
)


def _incident_fixture() -> dict:
    """Load the redacted captured-failure decision slice."""

    fixture_text = _FIXTURE_PATH.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)
    return fixture


def _active_goal_state() -> dict:
    """Build one valid active character goal for salience projection."""

    state = build_character_production_state(updated_at=_NOW)
    state["goals"].append({
        "entity_id": "goal:verify-current-challenge",
        "description": "确认群里正在讨论的一换一挑战是否会伤害同伴",
        "salience": 80,
        "role_refs": [],
        "evidence_refs": [],
        "created_at": _NOW,
        "updated_at": _NOW,
        "goal_kind": "safety",
        "status": "pursuing",
        "importance": 80,
        "progress": 10,
        "obstruction": 40,
        "expected_success": 60,
        "controllability": 60,
        "recoverability": 70,
        "urgency": 75,
    })
    return state


def test_complete_character_name_is_the_only_name_span_candidate() -> None:
    """Expose full-name provenance without classifying address semantics."""

    fixture = _incident_fixture()
    incident_evidence = build_interaction_evidence(
        conversation_scope="group",
        active_character_name=fixture["active_character_name"],
        current_message=fixture["current_message"],
        open_turns=[],
        latest_bot_continuity="",
        history=[],
    )
    incident_kinds = {
        item["kind"] for item in incident_evidence
    }

    assert "canonical_name_span" not in incident_kinds

    named_message = dict(fixture["current_message"])
    named_message["body_text"] = "一之濑明日奈，你觉得要一换一吗？"
    named_evidence = build_interaction_evidence(
        conversation_scope="group",
        active_character_name=fixture["active_character_name"],
        current_message=named_message,
        open_turns=[],
        latest_bot_continuity="",
        history=[],
    )
    name_rows = [
        item
        for item in named_evidence
        if item["kind"] == "canonical_name_span"
    ]

    assert name_rows == [{
        "ref": "name_1",
        "kind": "canonical_name_span",
        "summary": "一之濑明日奈",
    }]


@pytest.mark.parametrize(
    'body_text',
    [
        '一',
        '一换一',
        '你',
        '你你你',
        '直接找你一换一是吧',
        '直接找你二换二是吧',
    ],
)
def test_pronouns_numerals_and_name_fragments_create_no_name_span(
    body_text: str,
) -> None:
    """Partial glyphs and pronouns never become canonical-name provenance."""

    evidence = build_interaction_evidence(
        conversation_scope='group',
        active_character_name='一之濑明日奈',
        current_message={
            'body_text': body_text,
            'semantic_target_labels': [],
            'reply_target_label': 'none',
        },
        open_turns=[],
        latest_bot_continuity='',
        history=[],
    )

    assert all(item['kind'] != 'canonical_name_span' for item in evidence)


def test_interaction_evidence_projects_typed_and_continuity_provenance() -> None:
    """Expose protocol facts and supplied continuity without semantic routing."""

    evidence = build_interaction_evidence(
        conversation_scope='private',
        active_character_name='一之濑明日奈',
        current_message={
            'body_text': '继续说刚才那个',
            'semantic_target_labels': ['character', 'broadcast'],
            'reply_target_label': 'character',
        },
        open_turns=[{
            'slot': 'open_1',
            'author_relation': 'same_author',
            'latest_intent': '未说完的问题',
            'target_summary': 'character',
        }],
        latest_bot_continuity='角色刚回答了同一话题',
        history=[],
    )

    assert {
        (item['ref'], item['kind'])
        for item in evidence
    } >= {
        ('scope_private', 'private_scope'),
        ('target_character', 'typed_character_target'),
        ('target_broadcast', 'typed_broadcast'),
        ('reply_character', 'typed_character_reply'),
        ('message_1', 'current_message'),
        ('open_1', 'open_turn'),
        ('continuity_1', 'bot_continuity'),
    }


def test_character_state_projection_is_bounded_and_hides_telemetry() -> None:
    """Project active semantic state without raw ids or scalar values."""

    state = _active_goal_state()
    evidence = project_character_state_evidence(state)

    assert evidence == [{
        "ref": "state_1",
        "kind": "goal",
        "summary": "确认群里正在讨论的一换一挑战是否会伤害同伴",
        "attention": "active",
    }]
    rendered = json.dumps(evidence, ensure_ascii=False)
    assert "goal:verify-current-challenge" not in rendered
    assert '"salience"' not in rendered
    assert '"80"' not in rendered
    assert len(rendered) <= 1400


def test_character_state_projection_applies_thresholds_order_and_cap() -> None:
    """Only active pressure candidates survive in descending source strength."""

    state = build_character_production_state(updated_at=_NOW)
    state['threats'] = [{
        'description': '阻止迫近的一换一危险',
        'status': 'active',
        'salience': 10,
        'residual_pressure': 95,
    }]
    state['goals'] = [{
        'description': '确认挑战规则',
        'status': 'pursuing',
        'salience': 75,
    }]
    state['active_events'] = [{
        'description': '群里正在组织挑战',
        'status': 'active',
        'salience': 80,
    }]
    state['knowledge_gaps'] = [{
        'description': '谁会因挑战受到伤害',
        'status': 'open',
        'salience': 85,
    }]
    state['drives']['autonomy']['pressure'] = 90
    state['drives']['care']['pressure'] = 65
    state['meaning_state']['salience'] = 70

    evidence = project_character_state_evidence(state)

    assert [item['kind'] for item in evidence] == [
        'threat',
        'drive',
        'knowledge_gap',
        'event',
        'goal',
        'meaning',
    ]
    assert [item['ref'] for item in evidence] == [
        'state_1',
        'state_2',
        'state_3',
        'state_4',
        'state_5',
        'state_6',
    ]
    assert len(evidence) == 6


def test_active_threat_can_qualify_from_residual_pressure_alone() -> None:
    """Residual pressure independently keeps an active threat in evidence."""

    state = build_character_production_state(updated_at=_NOW)
    state['threats'] = [{
        'description': '一换一风险仍有余压',
        'status': 'active',
        'residual_pressure': 25,
    }]

    evidence = project_character_state_evidence(state)

    assert evidence == [{
        'ref': 'state_1',
        'kind': 'threat',
        'summary': '一换一风险仍有余压',
        'attention': 'active',
    }]


def test_character_state_projection_excludes_inactive_and_low_pressure() -> None:
    """Lifecycle and threshold failures do not become semantic candidates."""

    state = build_character_production_state(updated_at=_NOW)
    state['goals'] = [{
        'description': '低显著目标',
        'status': 'pursuing',
        'salience': 24,
    }]
    state['threats'] = [{
        'description': '已经解决的威胁',
        'status': 'resolved',
        'salience': 100,
        'residual_pressure': 100,
    }]
    state['active_events'] = [{
        'description': '已结束事件',
        'status': 'resolved',
        'salience': 100,
    }]
    state['knowledge_gaps'] = [{
        'description': '已解决问题',
        'status': 'resolved',
        'salience': 100,
    }]
    state['drives']['autonomy']['pressure'] = 60
    state['meaning_state']['salience'] = 60

    assert project_character_state_evidence(state) == []


def test_interaction_admission_rejects_message_only_character_claim() -> None:
    """Reject direct-character admission without an allowed grounding ref."""

    interaction_evidence = [{
        "ref": "message_1",
        "kind": "current_message",
        "summary": "直接找你一换一是吧",
    }]
    raw = {
        "recipient_relation": "character",
        "admission_basis": "interaction_relevance",
        "interaction_evidence_refs": ["message_1"],
        "character_state_refs": [],
    }

    with pytest.raises(ValueError, match="character recipient"):
        validate_participation_assessment(
            raw,
            interaction_evidence=interaction_evidence,
            character_state_evidence=[],
            stage="frontline",
            action="start",
            append_target="none",
            use_reply_feature=False,
        )


def test_state_salience_preserves_other_recipient_without_reply_anchor() -> None:
    """Allow state-grounded speech while retaining the actual recipient."""

    state_evidence = project_character_state_evidence(_active_goal_state())
    raw = {
        "recipient_relation": "other_participant",
        "admission_basis": "character_state_salience",
        "interaction_evidence_refs": ["message_1"],
        "character_state_refs": ["state_1"],
    }
    assessment = validate_participation_assessment(
        raw,
        interaction_evidence=[{
            "ref": "message_1",
            "kind": "current_message",
            "summary": "直接找你一换一是吧",
        }],
        character_state_evidence=state_evidence,
        stage="settled",
        action="proceed",
        append_target="none",
        use_reply_feature=False,
    )

    assert assessment == raw


def test_evidence_refs_discard_bad_entries_and_truncate() -> None:
    """Keep the first three usable unique refs from model evidence."""

    interaction_evidence = [
        {
            "ref": "target_character",
            "kind": "typed_character_target",
            "summary": "typed target identifies the character",
        },
        {
            "ref": "reply_character",
            "kind": "typed_character_reply",
            "summary": "typed reply identifies the character",
        },
        {
            "ref": "name_1",
            "kind": "canonical_name_span",
            "summary": "一之濑明日奈",
        },
        {
            "ref": "message_1",
            "kind": "current_message",
            "summary": "我说了有奖励么？",
        },
    ]
    raw = {
        "recipient_relation": "character",
        "admission_basis": "interaction_relevance",
        "interaction_evidence_refs": [
            7,
            "",
            "invented",
            "target_character",
            "target_character",
            "reply_character",
            "name_1",
            "message_1",
        ],
        "character_state_refs": "not-a-list",
    }

    assessment = validate_participation_assessment(
        raw,
        interaction_evidence=interaction_evidence,
        character_state_evidence=[],
        stage="frontline",
        action="start",
        append_target="none",
        use_reply_feature=False,
    )

    assert assessment["interaction_evidence_refs"] == [
        "target_character",
        "reply_character",
        "name_1",
    ]
    assert assessment["character_state_refs"] == []


def test_assessment_without_usable_grounding_fails_closed() -> None:
    """A normalized empty list cannot invent participation grounding."""

    raw = {
        "recipient_relation": "character",
        "admission_basis": "interaction_relevance",
        "interaction_evidence_refs": ["invented"],
        "character_state_refs": [],
    }

    with pytest.raises(ValueError, match="positive evidence"):
        validate_participation_assessment(
            raw,
            interaction_evidence=[],
            character_state_evidence=[],
            stage="frontline",
            action="start",
            append_target="none",
            use_reply_feature=False,
        )


def test_wrong_kind_and_incomplete_assessments_fail_closed() -> None:
    """Structural validation cannot reinterpret unsupported model claims."""

    interaction_evidence = [{
        'ref': 'target_other',
        'kind': 'typed_other_target',
        'summary': 'typed target identifies another participant',
    }]
    wrong_kind = {
        'recipient_relation': 'character',
        'admission_basis': 'interaction_relevance',
        'interaction_evidence_refs': ['target_other'],
        'character_state_refs': [],
    }
    incomplete = {
        'recipient_relation': 'unknown',
        'admission_basis': 'none',
        'interaction_evidence_refs': [],
    }

    for raw in (wrong_kind, incomplete):
        with pytest.raises(ValueError):
            validate_participation_assessment(
                raw,
                interaction_evidence=interaction_evidence,
                character_state_evidence=[],
                stage='frontline',
                action='start',
                append_target='none',
                use_reply_feature=False,
            )


def test_append_requires_the_visible_cited_open_turn() -> None:
    """An append action cannot use an omitted or uncited candidate slot."""

    interaction_evidence = [{
        'ref': 'open_1',
        'kind': 'open_turn',
        'summary': 'author=same_author; target=character',
    }]
    assessment = {
        'recipient_relation': 'character',
        'admission_basis': 'interaction_relevance',
        'interaction_evidence_refs': ['open_1'],
        'character_state_refs': [],
    }

    validated = validate_participation_assessment(
        assessment,
        interaction_evidence=interaction_evidence,
        character_state_evidence=[],
        stage='frontline',
        action='append',
        append_target='open_1',
        use_reply_feature=False,
    )

    assert validated == assessment

    with pytest.raises(ValueError):
        validate_participation_assessment(
            {
                **assessment,
                'interaction_evidence_refs': [],
            },
            interaction_evidence=interaction_evidence,
            character_state_evidence=[],
            stage='frontline',
            action='append',
            append_target='open_1',
            use_reply_feature=False,
        )
