"""Deterministic checks for self-cognition source-packet framing."""

from __future__ import annotations

from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.self_cognition.projection import (
    build_source_packet,
    render_source_packet_text,
)


def test_self_cognition_framing_presents_chat_data_without_action_guidance() -> None:
    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "case-past-due",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "contact_is_socially_available",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "673225019",
        },
        "source_refs": [
            {
                "source_kind": "future_promise",
                "source_id": "promise-001",
                "due_at": "2026-05-10T00:00:00+00:00",
                "summary": "The user expected a follow-up by this time.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "Reminder was expected before this timestamp.",
                "timestamp": "2026-05-09T23:50:00+00:00",
            }
        ],
    }

    source_packet = build_source_packet(case)
    rendered_text = render_source_packet_text(source_packet)

    assert "来源位置：我和对方私聊窗口的最近可见内容。" in rendered_text
    assert (
        "出现原因：我在这段私聊里，需要接上这段对话的时间线和约定。"
        in rendered_text
    )
    assert "There is no new user message to answer" not in rendered_text
    assert "Prefer silence" not in rendered_text
    assert "Silence is allowed" not in rendered_text
    assert "proactive message candidate" not in rendered_text
    assert "visible `speak` action" not in rendered_text
    assert "shared action-spec contract" not in rendered_text
    assert "当前自检" not in rendered_text
    assert "自然路线" not in rendered_text
    assert "# 来源状态" in rendered_text
    assert "- 约定状态: 已过期" in rendered_text
    assert "contact_is_socially_available" not in rendered_text
    assert "- idle_local_datetime: 2026-05-10 12:30" in rendered_text
    assert "- last_evidence_local_datetime: 2026-05-10 12:00" in rendered_text
    assert "idle_timestamp" not in rendered_text
    assert "last_evidence_timestamp" not in rendered_text
    assert "+00:00" not in rendered_text

def test_topic_followup_source_packet_does_not_preload_retrieval() -> None:
    case = {
        "case_name": models.CASE_TOPIC_RAG_FOLLOWUP,
        "case_id": "case-rag-followup",
        "idle_timestamp_utc": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp_utc": "2026-05-10T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_BOUNDED_FOLLOWUP_TOPIC,
        "semantic_due_state": models.DUE_STATE_FUTURE_DUE,
        "actionability": "followup_topic_needs_retrieval",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "54369546",
            "channel_type": "group",
            "user_id": "user-001",
        },
        "user_profile": {
            "display_name": "Test User",
        },
        "character_profile": {
            "name": "Test Character",
        },
        "visible_context": [
            {
                "role": "user",
                "text": "GraphRAG 能不能补上小数据场景？",
                "timestamp": "2026-05-10T00:00:00+00:00",
            }
        ],
    }

    source_packet = build_source_packet(case)
    rendered_text = render_source_packet_text(source_packet)

    assert "GraphRAG 能不能补上小数据场景？" in rendered_text
    assert "# 检索补充" not in rendered_text
    assert "rag_evidence" not in source_packet
