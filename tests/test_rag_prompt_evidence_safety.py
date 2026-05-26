"""Tests for prompt-facing RAG evidence leak checks."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.rag.evidence_formatting import (
    ensure_public_rag_evidence_prompt_safe,
)


def _safe_rag_result() -> dict[str, object]:
    return {
        "answer": '结论：the relevant preference is tea.',
        "memory_evidence": [
            {
                "summary": '结论：User prefers tea.',
                "content": (
                    '上下文：\n'
                    "- user_memory_units at 2026-05-02 00:34:56: User prefers tea.\n"
                    '不确定性：无'
                ),
            }
        ],
        "recall_evidence": [
            {
                "selected_summary": '结论：Pickup agreement is 9:30.',
                "evidence_summary": (
                    '上下文：\n'
                    "- conversation_progress: pickup at 9:30.\n"
                    '不确定性：无'
                ),
            }
        ],
        "third_party_profiles": ["Night resolved to a public display profile."],
        "conversation_evidence": [
            (
                '结论：User sent an image.\n'
                '上下文：\n'
                "- User at 2026-05-02 00:35:01: <image>a clean chart</image>\n"
                '不确定性：无'
            )
        ],
        "external_evidence": [
            {
                "summary": '结论：Weather context was available.',
                "content": '上下文：\n- External source: light rain.\n不确定性：无',
                "url": "https://weather.example/current",
            }
        ],
        "supervisor_trace": {
            "dispatched": [
                {
                    "source_ref": {
                        "conversation_row_id": "507f1f77bcf86cd799439011",
                        "platform_message_id": "qq-message-1",
                        "storage_timestamp_utc": "2026-05-01T12:34:56.789000+00:00",
                    }
                }
            ]
        },
    }


def test_public_rag_result_evidence_allows_trace_ids_only() -> None:
    ensure_public_rag_evidence_prompt_safe(_safe_rag_result())


def test_public_rag_result_evidence_allows_external_url_query_marker() -> None:
    rag_result = _safe_rag_result()
    rag_result["external_evidence"] = [
        {
            "summary": '结论：External context was available.',
            "content": '上下文：\n- External source was read.\n不确定性：无',
            "url": (
                "https://example.test/redirect?"
                "url=https%3A%2F%2Ftarget.example%2Fpage"
            ),
        }
    ]

    ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_allows_external_url_uuid_path() -> None:
    rag_result = _safe_rag_result()
    rag_result["external_evidence"] = [
        {
            "summary": '结论：External context was available.',
            "content": '上下文：\n- External source was read.\n不确定性：无',
            "url": (
                "https://example.test/resource/"
                "123e4567-e89b-12d3-a456-426614174000"
            ),
        }
    ]

    ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_malformed_external_url() -> None:
    rag_result = _safe_rag_result()
    rag_result["external_evidence"] = [
        {
            "summary": '结论：External context was available.',
            "content": '上下文：\n- External source was read.\n不确定性：无',
            "url": "http://[broken",
        }
    ]

    with pytest.raises(ValueError, match=r"external_evidence\[0\]\.url"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_raw_cq_wire_text_urls_ids_and_embeddings() -> None:
    rag_result = _safe_rag_result()
    rag_result["memory_evidence"] = [
        {
            "summary": '结论：leaked raw row.',
            "content": '上下文：\n- [CQ:image,file=abc]\n不确定性：无',
            "conversation_row_id": "row-1",
        }
    ]
    rag_result["recall_evidence"] = [
        {
            "selected_summary": '结论：leaked vector payload.',
            "embedding": [0.1, 0.2],
        }
    ]
    rag_result["conversation_evidence"] = [
        '结论：leaked url.\n上下文：\n- url=https://cdn.example/image.png\n不确定性：无'
    ]

    with pytest.raises(ValueError, match="prompt-facing RAG evidence"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_source_uuid_text() -> None:
    rag_result = _safe_rag_result()
    rag_result["conversation_evidence"] = [
        (
            '结论：Tester global_user_id: '
            "123e4567-e89b-12d3-a456-426614174000 sent the chart.\n"
            '上下文：\n'
            "- Tester: chart sent.\n"
            '不确定性：无'
        )
    ]

    with pytest.raises(ValueError, match="prompt-facing RAG evidence"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_third_party_profile_uuid_text() -> None:
    rag_result = _safe_rag_result()
    rag_result["third_party_profiles"] = [
        "Night | 123e4567-e89b-12d3-a456-426614174000"
    ]

    with pytest.raises(ValueError, match="prompt-facing RAG evidence"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_readable_message_id_text() -> None:
    rag_result = _safe_rag_result()
    rag_result["answer"] = "在消息 ID 529487488 中，用户识别了五种香料。"

    with pytest.raises(ValueError, match="prompt-facing RAG evidence"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_allows_source_refs_inside_supervisor_trace() -> None:
    rag_result = _safe_rag_result()
    rag_result["supervisor_trace"] = {
        "raw_refs": [
            {
                "_id": "507f1f77bcf86cd799439011",
                "conversation_row_id": "row-1",
                "platform_message_id": "message-1",
                "embedding": [0.1],
                "raw_wire_text": "[CQ:image,file=abc]",
                "url": "https://cdn.example/image.png",
            }
        ]
    }

    ensure_public_rag_evidence_prompt_safe(rag_result)
