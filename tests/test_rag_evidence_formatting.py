"""Tests for cognition-ready RAG evidence formatting."""

from __future__ import annotations

from kazusa_ai_chatbot.rag.evidence_formatting import format_evidence_block
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm_seconds


def test_format_evidence_block_orders_conclusion_evidence_uncertainty() -> None:
    block = format_evidence_block(
        conclusion="The pickup agreement is 9:30.",
        evidence_items=[
            "User at 2026-05-02 00:30:00: Please pick me up at 9:30.",
            "Kazusa at 2026-05-02 00:31:00: I will remember 9:30.",
        ],
        uncertainty="No later cancellation was found.",
    )

    assert block == (
        "Conclusion: The pickup agreement is 9:30.\n"
        "Evidence summary:\n"
        "- User at 2026-05-02 00:30:00: Please pick me up at 9:30.\n"
        "- Kazusa at 2026-05-02 00:31:00: I will remember 9:30.\n"
        "Uncertainty: No later cancellation was found."
    )


def test_format_evidence_block_uses_empty_uncertainty_when_clear() -> None:
    block = format_evidence_block(
        conclusion="The remembered preference is tea.",
        evidence_items=["user_memory_units: User prefers tea."],
    )

    assert block.endswith("\nUncertainty: none")


def test_format_evidence_block_does_not_emit_blank_sections() -> None:
    block = format_evidence_block(
        conclusion="No matching evidence was found.",
        evidence_items=[],
        uncertainty="The search returned no direct or nearby evidence.",
    )

    assert block == (
        "Conclusion: No matching evidence was found.\n"
        "Uncertainty: The search returned no direct or nearby evidence."
    )
    assert "Evidence summary:\nUncertainty" not in block
    assert "\n\n" not in block


def test_format_storage_utc_for_llm_seconds_projects_configured_local_time() -> None:
    assert (
        format_storage_utc_for_llm_seconds("2026-05-01T12:34:56.789000+00:00")
        == "2026-05-02 00:34:56"
    )


def test_format_storage_utc_for_llm_seconds_rejects_ambiguous_time() -> None:
    assert format_storage_utc_for_llm_seconds("2026-05-02 00:34:56") == ""
    assert format_storage_utc_for_llm_seconds("2026-05-02") == ""
