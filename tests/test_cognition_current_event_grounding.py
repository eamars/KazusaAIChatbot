"""V2 current-event grounding ownership tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    build_text_chat_cognitive_episode,
)
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

    return build_text_chat_cognitive_episode(
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
            "decontexualized_input": "fallback semantic rewrite",
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


@pytest.mark.asyncio
async def test_appraisal_prompt_excludes_current_event_provenance_ids() -> None:
    """Operational provenance is bound to handles outside model payloads."""

    payload = _connector_payload()
    mutable_state = payload["mutable_state"]
    constraints = payload["character_constraints"]
    evidence = payload["evidence"]
    questions = plan_semantic_questions(
        evidence,
        mutable_state,
        constraints,
    )
    projection = project_state_for_prompt(
        mutable_state,
        character_constraints=constraints,
        evidence=evidence,
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
