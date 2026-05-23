"""Tests for prompt-facing RAG evidence leak checks."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.rag.evidence_formatting import (
    ensure_public_rag_evidence_prompt_safe,
)


def _safe_rag_result() -> dict[str, object]:
    return {
        "answer": "Conclusion: the relevant preference is tea.",
        "memory_evidence": [
            {
                "summary": "Conclusion: User prefers tea.",
                "content": (
                    "Evidence summary:\n"
                    "- user_memory_units at 2026-05-02 00:34:56: User prefers tea.\n"
                    "Uncertainty: none"
                ),
            }
        ],
        "recall_evidence": [
            {
                "selected_summary": "Conclusion: Pickup agreement is 9:30.",
                "evidence_summary": (
                    "Evidence summary:\n"
                    "- conversation_progress: pickup at 9:30.\n"
                    "Uncertainty: none"
                ),
            }
        ],
        "third_party_profiles": ["Night resolved to a public display profile."],
        "conversation_evidence": [
            (
                "Conclusion: User sent an image.\n"
                "Evidence summary:\n"
                "- User at 2026-05-02 00:35:01: <image>a clean chart</image>\n"
                "Uncertainty: none"
            )
        ],
        "external_evidence": [
            {
                "summary": "Conclusion: Weather context was available.",
                "content": "Evidence summary:\n- External source: light rain.\nUncertainty: none",
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


def test_public_rag_result_evidence_rejects_raw_cq_wire_text_urls_ids_and_embeddings() -> None:
    rag_result = _safe_rag_result()
    rag_result["memory_evidence"] = [
        {
            "summary": "Conclusion: leaked raw row.",
            "content": "Evidence summary:\n- [CQ:image,file=abc]\nUncertainty: none",
            "conversation_row_id": "row-1",
        }
    ]
    rag_result["recall_evidence"] = [
        {
            "selected_summary": "Conclusion: leaked vector payload.",
            "embedding": [0.1, 0.2],
        }
    ]
    rag_result["conversation_evidence"] = [
        "Conclusion: leaked url.\nEvidence summary:\n- url=https://cdn.example/image.png\nUncertainty: none"
    ]

    with pytest.raises(ValueError, match="prompt-facing RAG evidence"):
        ensure_public_rag_evidence_prompt_safe(rag_result)


def test_public_rag_result_evidence_rejects_source_uuid_text() -> None:
    rag_result = _safe_rag_result()
    rag_result["conversation_evidence"] = [
        (
            "Conclusion: Tester global_user_id: "
            "123e4567-e89b-12d3-a456-426614174000 sent the chart.\n"
            "Evidence summary:\n"
            "- Tester: chart sent.\n"
            "Uncertainty: none"
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
