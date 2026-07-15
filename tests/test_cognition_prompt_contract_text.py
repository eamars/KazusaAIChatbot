"""Deterministic prompt-text contracts for the canonical V2 stages."""

from __future__ import annotations

from kazusa_ai_chatbot.cognition_core_v2.action_selection import ROUTE_PROMPT
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    SURFACE_STAGE_PROMPTS,
    SURFACE_STAGE_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import COLLAPSE_PROMPT


def test_semantic_appraisal_prompt_limits_model_authority() -> None:
    """Keep state, lifecycle, and persistence outside appraisal authority."""

    for required_text in (
        "bounded evidence",
        "permitted prompt-local handles",
        "propositions",
        "deltas",
    ):
        assert required_text in SEMANTIC_APPRAISAL_PROMPT
    for forbidden_text in (
        "emotion_id",
        "activation_id",
        "replacement_state",
        "persistence route",
    ):
        assert forbidden_text not in SEMANTIC_APPRAISAL_PROMPT


def test_goal_prompt_requires_complete_grounded_bid() -> None:
    """Keep goal cognition handle-grounded and final-wording free."""

    for required_text in (
        "independent goal cognition branch",
        "evidence handles",
        "complete bid",
        "Do not write final dialogue",
    ):
        assert required_text in GOAL_COGNITION_PROMPT


def test_collapse_and_route_prompts_select_handles_only() -> None:
    """Prevent collapse and routing from inventing bid content."""

    assert "prompt-local partition" in COLLAPSE_PROMPT
    assert "Do not rewrite bid content" in COLLAPSE_PROMPT
    assert "Select only a route" in ROUTE_PROMPT
    assert "Do not rewrite bid content" in ROUTE_PROMPT


def test_surface_prompts_leave_final_dialogue_to_dialog() -> None:
    """Keep all four surface stages semantic and non-rendering."""

    assert set(SURFACE_STAGE_PROMPTS) == {
        "style",
        "content_plan",
        "preference",
        "visual",
    }
    assert all(
        "without writing dialogue" in prompt.lower()
        for prompt in SURFACE_STAGE_PROMPTS.values()
    )


def test_generated_semantic_prompts_preserve_language_policy() -> None:
    """Model-authored semantic prose retains the project language contract."""

    for prompt in (
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        SURFACE_STAGE_SYSTEM_PROMPT,
    ):
        normalized = " ".join(prompt.split())
        assert "Simplified Chinese" in normalized
        assert "quoted user text" in normalized
        assert "proper nouns" in normalized
        assert "schema or enum tokens" in normalized


def test_semantic_prompts_preserve_typed_source_ownership() -> None:
    """Internal evidence cannot be reclassified as current user speech."""

    for prompt in (
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        SURFACE_STAGE_SYSTEM_PROMPT,
    ):
        lowered = " ".join(prompt.casefold().split())
        assert "character-owned" in lowered
        assert "live user speech" in lowered
        assert "operational metadata" in lowered


def test_v2_prompts_do_not_restore_operational_or_scalar_gates() -> None:
    """Character judgment stays semantic and free of retired gate vocabulary."""

    prompts = "\n".join((
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        COLLAPSE_PROMPT,
        ROUTE_PROMPT,
        SURFACE_STAGE_SYSTEM_PROMPT,
        *SURFACE_STAGE_PROMPTS.values(),
    )).casefold()
    for forbidden in (
        "relationship score threshold",
        "mood gate",
        "vibe gate",
        "tool cost",
        "willingness score",
        "kazusa",
    ):
        assert forbidden not in prompts
