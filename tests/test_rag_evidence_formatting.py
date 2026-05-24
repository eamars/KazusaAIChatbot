"""Tests for cognition-ready RAG evidence formatting."""

from __future__ import annotations

from kazusa_ai_chatbot.rag.evidence_formatting import (
    format_evidence_block,
    sanitize_public_rag_evidence_text,
)
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm_seconds


def test_format_evidence_block_orders_conclusion_context_uncertainty() -> None:
    block = format_evidence_block(
        conclusion="The pickup agreement is 9:30.",
        evidence_items=[
            "User at 2026-05-02 00:30:00: Please pick me up at 9:30.",
            "Kazusa at 2026-05-02 00:31:00: I will remember 9:30.",
        ],
        uncertainty="No later cancellation was found.",
    )

    assert block == (
        "结论：The pickup agreement is 9:30.\n"
        "上下文：\n"
        "- User at 2026-05-02 00:30:00: Please pick me up at 9:30.\n"
        "- Kazusa at 2026-05-02 00:31:00: I will remember 9:30.\n"
        "不确定性：No later cancellation was found."
    )


def test_format_evidence_block_uses_empty_uncertainty_when_clear() -> None:
    block = format_evidence_block(
        conclusion="The remembered preference is tea.",
        evidence_items=["user_memory_units: User prefers tea."],
    )

    assert block.endswith("\n不确定性：无")


def test_format_evidence_block_does_not_emit_blank_sections() -> None:
    block = format_evidence_block(
        conclusion="No matching evidence was found.",
        evidence_items=[],
        uncertainty="The search returned no direct or nearby evidence.",
    )

    assert block == (
        "结论：No matching evidence was found.\n"
        "不确定性：The search returned no direct or nearby evidence."
    )
    assert "上下文：\n不确定性" not in block
    assert "\n\n" not in block


def test_sanitize_public_rag_evidence_text_renders_internal_source_labels() -> None:
    text = (
        "recall:user_memory_units at 2026-05-22T10:16:47.993342+00:00 "
        "confirmed durable_commitment."
    )

    clean_text = sanitize_public_rag_evidence_text(text)

    assert "recall:" not in clean_text
    assert "user_memory_units" not in clean_text
    assert "durable_commitment" not in clean_text
    assert "2026-05-22T10:16:47.993342+00:00" not in clean_text
    assert '召回候选' in clean_text
    assert '用户记忆' in clean_text
    assert "2026-05-22 22:16" in clean_text


def test_sanitize_public_rag_evidence_text_renders_person_context_labels() -> None:
    text = (
        "person_context_agent resolved via user_lookup_agent with "
        "global_user_id 263c883d-aeff-4e0b-a758-6f69186ae8ec."
    )

    clean_text = sanitize_public_rag_evidence_text(text)

    assert "person_context_agent" not in clean_text
    assert "user_lookup_agent" not in clean_text
    assert "global_user_id" not in clean_text
    assert "263c883d-aeff-4e0b-a758-6f69186ae8ec" not in clean_text
    assert '人物上下文' in clean_text
    assert '用户识别' in clean_text


def test_sanitize_public_rag_evidence_text_removes_conversation_source_ids() -> None:
    text = (
        "conversation:platform_message_id:1195502528: "
        "蚝爹油: 这是你第一次来我家么？"
    )

    clean_text = sanitize_public_rag_evidence_text(text)

    assert "conversation:" not in clean_text
    assert "platform_message_id" not in clean_text
    assert "1195502528" not in clean_text
    assert '对话候选' in clean_text
    assert "蚝爹油: 这是你第一次来我家么？" in clean_text


def test_sanitize_public_rag_evidence_text_removes_ocr_source_id_labels() -> None:
    text = (
        "image description includes global_user_id 为 "
        "d815be2-d6dd-41b7-9026-aa01ea4367a2 and account details."
    )

    clean_text = sanitize_public_rag_evidence_text(text)

    assert "global_user_id" not in clean_text
    assert "d815be2-d6dd-41b7-9026-aa01ea4367a2" not in clean_text
    assert "[来源标识已省略]" in clean_text


def test_sanitize_public_rag_evidence_text_removes_readable_message_ids() -> None:
    text = "消息 ID 529487488 直接识别了肉桂皮、小豆蔻和八角。"

    clean_text = sanitize_public_rag_evidence_text(text)

    assert "消息 ID" not in clean_text
    assert "529487488" not in clean_text
    assert "消息记录" in clean_text
    assert "直接识别" in clean_text


def test_format_storage_utc_for_llm_seconds_projects_configured_local_time() -> None:
    assert (
        format_storage_utc_for_llm_seconds("2026-05-01T12:34:56.789000+00:00")
        == "2026-05-02 00:34:56"
    )


def test_format_storage_utc_for_llm_seconds_rejects_ambiguous_time() -> None:
    assert format_storage_utc_for_llm_seconds("2026-05-02 00:34:56") == ""
    assert format_storage_utc_for_llm_seconds("2026-05-02") == ""
