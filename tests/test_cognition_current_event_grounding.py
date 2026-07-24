"""V2 current-event grounding ownership tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tests.cognition_core_v2_test_helpers import canonical_user_message_episode
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
)
from llm_test_helpers import make_llm_call_config


STORAGE_TIMESTAMP_UTC = "2026-07-15T00:00:00+00:00"
V2_TIMESTAMP = "2026-07-15T00:00:00Z"
CURRENT_EVENT_TEXT = (
    "The current participant says the character won and offers dessert."
)


class _CapturingAppraisalLLM:
    """Capture one semantic-appraisal payload and return an empty judgment."""

    def __init__(self) -> None:
        self.human_payload = ""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.human_payload = str(getattr(messages[-1], "content", ""))
        payload = json.loads(self.human_payload)
        question = payload["question"]
        result = {
            "question_id": question["question_id"],
            "selected_evidence_handles": ["e1"],
            "selected_role_handles": [],
            "propositions": [],
            "deltas": [],
            "explanation": "The current event is grounded by visible evidence.",
        }
        return SimpleNamespace(content=json.dumps(result))


def _episode() -> dict[str, Any]:
    """Build a current message with sensitive operational provenance."""

    return canonical_user_message_episode(
        episode_id="episode:secret-channel-id:secret-message-id",
        percept_id="percept:secret-message-id",
        storage_timestamp_utc=STORAGE_TIMESTAMP_UTC,
        local_time_context=local_time_context_from_storage_utc(
            STORAGE_TIMESTAMP_UTC,
        ),
        user_input=CURRENT_EVENT_TEXT,
        platform="private-platform-id",
        platform_channel_id="secret-channel-id",
        channel_type="group",
        platform_message_id="secret-message-id",
        platform_user_id="secret-platform-user-id",
        global_user_id="secret-global-user-id",
        user_name="Visible Participant",
        active_turn_platform_message_ids=["secret-message-id"],
        active_turn_conversation_row_ids=["secret-storage-row-id"],
        debug_modes={},
        target_addressed_user_ids=["secret-character-id"],
        target_broadcast=False,
    )


def _connector_payload() -> dict[str, Any]:
    """Build an exact V2 input from the current typed episode."""

    character_state = build_character_production_state(
        updated_at=V2_TIMESTAMP,
    )
    return build_cognition_input_from_global_state(
        {
            "cognitive_episode": _episode(),
            "global_user_id": "secret-global-user-id",
            "user_input": "fallback should not replace the episode percept",
            "decontextualized_input": "fallback semantic rewrite",
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id="secret-global-user-id",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=character_state,
    )


def test_current_event_is_canonical_episode_evidence() -> None:
    """Visible current-event content and typed provenance remain separate."""

    payload = _connector_payload()
    evidence = payload["evidence"][0]

    assert evidence["semantic_text"] == CURRENT_EVENT_TEXT
    assert evidence["evidence_ref"] == {
        "source_kind": "episode",
        "source_id": "percept:secret-message-id",
        "occurred_at": V2_TIMESTAMP,
        "semantic_summary": CURRENT_EVENT_TEXT,
    }
    for secret in (
        "secret-channel-id",
        "secret-platform-user-id",
        "secret-global-user-id",
        "secret-storage-row-id",
    ):
        assert secret not in evidence["semantic_text"]


def test_rag_recall_evidence_is_canonical_cognition_evidence() -> None:
    """RAG recall facts remain available to downstream cognition."""

    recalled_fact = "我把门禁卡放进书桌右边第二个抽屉了。"
    payload = build_cognition_input_from_global_state(
        {
            "cognitive_episode": _episode(),
            "global_user_id": "secret-global-user-id",
            "user_input": CURRENT_EVENT_TEXT,
            "decontextualized_input": CURRENT_EVENT_TEXT,
            "user_multimedia_input": [],
            "rag_result": {
                "memory_evidence": [],
                "conversation_evidence": [
                    "用户曾在此前对话中说明门禁卡的存放地点。"
                ],
                "recall_evidence": [{
                    "content": recalled_fact,
                    "speaker": "当前用户",
                    "type": "explicit_statement",
                }],
            },
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id="secret-global-user-id",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=build_character_production_state(
            updated_at=V2_TIMESTAMP,
        ),
    )

    evidence = payload["evidence"]
    assert len(evidence) == 3
    conversation = evidence[1]
    assert conversation["evidence_ref"]["source_kind"] == (
        "conversation_evidence"
    )
    assert conversation["semantic_text"] == (
        "用户曾在此前对话中说明门禁卡的存放地点。"
    )
    recall = evidence[2]
    assert recall["evidence_ref"]["source_kind"] == "recall_evidence"
    assert recall["semantic_text"] == recalled_fact
    assert recall["evidence_ref"]["semantic_summary"] == recalled_fact
    assert recall["evidence_ref"]["occurred_at"] == V2_TIMESTAMP


def test_rag_conversation_mapping_is_canonical_cognition_evidence() -> None:
    """Conversation resolver mappings remain available to cognition."""

    recalled_fact = "我把门禁卡放进书桌右边第二个抽屉了。"
    payload = build_cognition_input_from_global_state(
        {
            "cognitive_episode": _episode(),
            "global_user_id": "secret-global-user-id",
            "user_input": CURRENT_EVENT_TEXT,
            "decontextualized_input": CURRENT_EVENT_TEXT,
            "user_multimedia_input": [],
            "rag_result": {
                "memory_evidence": [],
                "conversation_evidence": [{
                    "role": "user",
                    "content": recalled_fact,
                    "metadata": {"type": "direct_statement"},
                }],
                "recall_evidence": [],
            },
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id="secret-global-user-id",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=build_character_production_state(
            updated_at=V2_TIMESTAMP,
        ),
    )

    evidence = payload["evidence"]
    assert len(evidence) == 2
    conversation = evidence[1]
    assert conversation["evidence_ref"]["source_kind"] == (
        "conversation_evidence"
    )
    assert conversation["semantic_text"] == recalled_fact


def test_role_explicit_current_event_is_forwarded_without_reinterpretation() -> None:
    """Cognition should consume the upstream role meaning as current evidence."""

    role_explicit_content = (
        "当前用户要求当前角色说出当前角色希望当前用户下一步做什么。"
    )
    response_operation = {
        "operation": "当前角色为当前用户选择一个动作",
        "response_owner_role": "当前角色",
        "selection_owner_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }
    episode = _episode()
    episode["percepts"][0]["content"]["role_explicit_content"] = (
        role_explicit_content
    )
    episode["percepts"][0]["content"]["response_operation"] = (
        response_operation
    )
    payload = build_cognition_input_from_global_state(
        {
            "cognitive_episode": episode,
            "global_user_id": "secret-global-user-id",
            "user_input": CURRENT_EVENT_TEXT,
            "decontextualized_input": CURRENT_EVENT_TEXT,
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id="secret-global-user-id",
            updated_at=V2_TIMESTAMP,
        ),
        character_state=build_character_production_state(
            updated_at=V2_TIMESTAMP,
        ),
    )

    expected_meaning = json.dumps(
        {
            "response_operation": response_operation,
            "role_explicit_content": role_explicit_content,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert payload["evidence"][0]["semantic_text"] == expected_meaning
    assert payload["scene_context"]["semantic_scene"] == expected_meaning
    assert payload["scene_context"]["character_role"].startswith("当前角色")
    assert payload["scene_context"]["current_user_role"].startswith("当前用户")


@pytest.mark.asyncio
async def test_appraisal_prompt_excludes_current_event_provenance_ids() -> None:
    """Operational provenance is bound to handles outside model payloads."""

    payload = _connector_payload()
    mutable_state = payload["mutable_state"]
    constraints = payload["character_constraints"]
    evidence = payload["evidence"]
    projection = project_state_for_prompt(
        mutable_state,
        character_constraints=constraints,
        evidence=evidence,
    )
    questions = plan_semantic_questions(
        evidence,
        mutable_state,
        projection.handle_to_ref,
    )
    llm = _CapturingAppraisalLLM()
    config = make_llm_call_config("current_event_appraisal")
    services = CognitionCoreServicesV2(
        llm=llm,
        appraisal_config=config,
        goal_cognition_config=config,
        collapse_config=config,
        action_selection_config=config,
    )

    await appraise_semantic_question(
        questions[0],
        evidence,
        projection,
        services,
    )

    assert CURRENT_EVENT_TEXT in llm.human_payload
    for secret in (
        "secret-channel-id",
        "secret-message-id",
        "secret-platform-user-id",
        "secret-global-user-id",
        "secret-storage-row-id",
        "target_scope",
        "origin_metadata",
    ):
        assert secret not in llm.human_payload


def test_reducer_records_current_event_ref_from_evidence_provenance() -> None:
    """A grounded event root carries typed evidence and native state scope."""

    payload = _connector_payload()
    evidence = payload["evidence"]
    comparison_results: list[dict[str, Any]] = []
    updated = apply_semantic_appraisals(
        payload["mutable_state"],
        [{
            "question_id": "q:event_agency",
            "selected_evidence_handles": ["e1"],
            "selected_role_handles": [],
            "propositions": [{
                "proposition_kind": "intentionality",
                "subject_handle": "ce1",
                "evidence_handles": ["e1"],
                "role_assignments": [],
                "semantic_value": CURRENT_EVENT_TEXT,
            }],
            "deltas": [{
                "target_path": "active_events.ce1.intentionality",
                "delta": 30,
                "evidence_handles": ["e1"],
                "reason": "The visible event is presented as intentional.",
            }],
            "explanation": "The visible current event supports an event root.",
        }],
        evidence,
        {
            "ce1": {
                "scope": "user",
                "kind": "event",
                "entity_id": "candidate:event:e1",
            },
        },
        comparison_results,
    )

    event = updated["active_events"][0]
    assert comparison_results[0]["current_event_ref"] == {
        "scope": "user",
        "kind": "event",
        "entity_id": event["entity_id"],
    }
    assert comparison_results[0]["evidence_refs"] == [
        evidence[0]["evidence_ref"]
    ]
