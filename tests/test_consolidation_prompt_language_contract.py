"""Deterministic language-contract tests for consolidation prompts."""

from __future__ import annotations

import re

import pytest

from kazusa_ai_chatbot.consolidation import (
    character_self_guidance,
    lane_router,
    memory_units,
)
from kazusa_ai_chatbot.reflection_cycle import promotion

_PROMPT_SPECS = {
    "lane_router": (lane_router._ROUTER_PROMPT, {}),
    "lane_descriptions": (
        "\n".join(lane_router._LANE_DESCRIPTIONS.values()),
        {},
    ),
    "self_guidance_specialist": (
        character_self_guidance._SPECIALIST_PROMPT,
        {"character_name": "测试角色"},
    ),
    "self_guidance_reviewer": (
        character_self_guidance._REVIEW_PROMPT,
        {"character_name": "测试角色"},
    ),
    "memory_extractor": (
        memory_units._EXTRACTOR_PROMPT,
        {"character_name": "测试角色"},
    ),
    "memory_merge_judge": (memory_units._MERGE_JUDGE_PROMPT, {}),
    "memory_rewrite": (
        memory_units._REWRITE_PROMPT,
        {"character_name": "测试角色"},
    ),
    "memory_stability": (memory_units._STABILITY_PROMPT, {}),
    "reflection_promotion": (
        promotion.GLOBAL_PROMOTION_SYSTEM_PROMPT,
        {"character_name": "测试角色"},
    ),
    "reflection_promotion_reviewer": (
        promotion.GLOBAL_PROMOTION_REVIEW_SYSTEM_PROMPT,
        {},
    ),
}

_CONTRACT_TOKENS = {
    "CHALLENGE",
    "CONFIRM",
    "DIVERGE",
    "HumanMessage",
    "REFUSE",
    "TENTATIVE",
    "API",
    "ASCII",
    "ID",
    "JSON",
    "LLM",
    "MongoDB",
    "URL",
    "UUID",
    "active_character",
    "active_commitment",
    "active_hour_summaries",
    "active_hour_slots",
    "active_unit",
    "absent",
    "accept",
    "acceptable",
    "agreed",
    "action",
    "affects_identity_or_boundaries",
    "assistant",
    "allowed_unit_types",
    "assistant_final_dialog",
    "boundary_assessment",
    "authority",
    "candidate",
    "candidate_clusters",
    "blocked",
    "candidate_id",
    "captured_at",
    "character_agreement",
    "character_cognition_summary",
    "character_identity_growth",
    "character_operational_state_task",
    "character_self_guidance",
    "character_local_date",
    "channel_daily_syntheses",
    "channel_type",
    "cross_hour_topics",
    "cluster_id",
    "confidence",
    "conversation_quality_feedback",
    "conversation_quality_patterns",
    "conversation",
    "current_turn_user_message",
    "daily_run_id",
    "daily_global_promotion",
    "day_summary",
    "decision",
    "decontextualized_event",
    "decontextualized_input",
    "defense_rule",
    "due_at",
    "due_time",
    "evidence_card_id",
    "evidence_refs",
    "evidence_cards",
    "episode_id",
    "episode_trace",
    "evaluation_mode",
    "existing_last_seen_at",
    "existing_memory_unit_id",
    "existing_unit",
    "existing_unit_id",
    "existing_updated_at",
    "fact",
    "future_promises_evidence",
    "global_applicability",
    "global_user_id",
    "identity_evidence",
    "identity_or_boundaries",
    "input_sources",
    "interaction_style_image",
    "interaction_subtext",
    "lane",
    "lane_roster",
    "lane_tasks",
    "lineage_id",
    "lore",
    "max_lore",
    "max_self_guidance",
    "max_total_decisions",
    "memory_name",
    "memory_unit",
    "memory_type",
    "memory_unit_write_contract",
    "memory_units",
    "merge",
    "merge_result",
    "milestone",
    "new_candidate",
    "new_facts_evidence",
    "new_memory_unit",
    "no_action",
    "objective_fact",
    "operational",
    "output_mode",
    "participant_ref",
    "participant_observations",
    "private_detail_risk",
    "privacy_review",
    "privacy_risk_labels",
    "promotion_decisions",
    "promotion_limits",
    "promoter",
    "promote_new",
    "recent",
    "recent_shift",
    "recent_examples",
    "recency",
    "reason",
    "relationship_signal",
    "reject",
    "rewrite",
    "scope_ref",
    "group",
    "private",
    "system",
    "scope_type",
    "self_cognition",
    "self_guidance",
    "session_spread",
    "signal_strength",
    "source",
    "source_global_user_id",
    "source_key",
    "source_keys",
    "source_reflection_run_ids",
    "source_views",
    "source_role",
    "source_privacy_notes",
    "user_style_signal",
    "group_channel_style_image",
    "write_lanes",
    "stable",
    "stable_pattern",
    "stability_evidence",
    "status",
    "subjective_appraisal",
    "subjective_appraisal_evidence",
    "supersede",
    "supports",
    "target_specific_meaning_removed",
    "timestamp",
    "tool_result",
    "trigger_source",
    "unit_id",
    "unit_type",
    "user_details_removed",
    "user_memory_units",
    "user_name",
    "visible_self_expression_summary",
    "window",
    "write",
    "chat",
    "chat_history_recent",
    "content",
    "consolidation_origin",
    "create",
    "current_turn_timestamp",
    "dedup_key",
    "distinct_day_count",
    "distinct_message_ref_count",
    "emotional_appraisal",
    "enabled_lanes",
    "evolve",
    "existing_unit_count",
    "false",
    "final_dialog",
    "finalization",
    "internal_monologue",
    "internal_thought",
    "many_observations",
    "message_id",
    "multiple_days_or_sessions",
    "new_evidence_ref_count",
    "null",
    "occurrence_count",
    "occurrence_count_label",
    "revise",
    "scheduled_tick",
    "several_observations",
    "shared_memory_promotion",
    "single_day_or_session",
    "single_observation",
    "spread_label",
    "target_plan",
    "timestamps",
    "true",
    "two_observations",
    "unknown",
    "unknown_session_spread",
    "updated_at",
    "user_message",
    "YYYY-MM-DD",
    "character_intent",
    "global",
    "high",
    "logical_stance",
    "low",
    "medium",
    "rag_user_memory_unit_candidates",
    "scoped",
    "source_refs",
    "reviews",
    "unreviewed",
    "automated_llm",
    "ReflectionPromotionDecision",
    "reflection_cycle",
    "reflection_promoted",
    "reflection_run_id",
    "source_utterance",
    "sanitized_content",
    "sanitized_memory_name",
    "sanitized_observation",
    "selected_candidate_id",
    "spoken",
    "needs_human_review",
    "review_questions",
    "reviewer",
    "validation_warning_labels",
    "verdict",
}

_CAPTURED_CASE_MARKERS = {
    "case_01",
    "case_07",
    "case_11",
    "case_13",
    "case_26",
    "test_live_",
    "candidate-memory-boundary-repeat",
    "candidate-stability-repeat",
}


def _rendered_prompts() -> list[tuple[str, str]]:
    """Return every owned prompt after applying its runtime placeholders."""

    rendered: list[tuple[str, str]] = []
    for name, (prompt, values) in _PROMPT_SPECS.items():
        formats_at_runtime = name in {
            "memory_extractor",
            "memory_rewrite",
            "reflection_promotion",
        }
        rendered_prompt = (
            prompt.format(**values)
            if formats_at_runtime
            else prompt
        )
        rendered.append((name, rendered_prompt))
    return rendered


_RENDERED_PROMPTS = _rendered_prompts()


def _non_contractual_english_tokens(prompt: str) -> set[str]:
    """Find English prose tokens outside the explicit wire-contract allowlist."""

    tokens = set(re.findall(r"[A-Za-z][A-Za-z_-]{2,}", prompt))
    return {
        token
        for token in tokens
        if token not in _CONTRACT_TOKENS
        and token.lower() not in {item.lower() for item in _CONTRACT_TOKENS}
    }


@pytest.mark.parametrize(
    "name,prompt",
    _RENDERED_PROMPTS,
    ids=[name for name, _ in _RENDERED_PROMPTS],
)
def test_consolidation_prompts_use_chinese_for_runtime_prose(
    name: str,
    prompt: str,
) -> None:
    """Reject human-readable English while allowing stable contract tokens."""

    del name
    assert _non_contractual_english_tokens(prompt) == set()


@pytest.mark.parametrize(
    "name,prompt",
    _RENDERED_PROMPTS,
    ids=[name for name, _ in _RENDERED_PROMPTS],
)
def test_consolidation_prompt_rendering_has_no_unresolved_placeholders(
    name: str,
    prompt: str,
) -> None:
    """Ensure each prompt remains renderable with its runtime contract values."""

    del name
    assert "{character_name}" not in prompt


def test_consolidation_prompts_have_no_captured_case_or_test_identifiers() -> None:
    """Keep prompt localization independent from live fixtures and test names."""

    combined = "\n".join(prompt for _, prompt in _RENDERED_PROMPTS)

    for marker in _CAPTURED_CASE_MARKERS:
        assert marker not in combined


def test_consolidation_prompt_language_contract_is_complete() -> None:
    """Apply the complete language and placeholder contract in one node."""

    for name, prompt in _RENDERED_PROMPTS:
        assert _non_contractual_english_tokens(prompt) == set(), name
        assert "{character_name}" not in prompt, name

    combined = "\n".join(prompt for _, prompt in _RENDERED_PROMPTS)
    assert all(marker not in combined for marker in _CAPTURED_CASE_MARKERS)


def test_reflection_prompt_remains_outside_the_localization_surface() -> None:
    """The already-localized reflection prompt is governed read-only here."""

    from kazusa_ai_chatbot.consolidation import reflection

    reflection_prompts = (
        reflection._CHARACTER_STATE_REVIEW_PROMPT,
        reflection._RELATIONSHIP_PROFILE_REVIEW_PROMPT,
    )
    assert all("cognition" in prompt for prompt in reflection_prompts)
    assert all("episode" not in prompt for prompt in reflection_prompts)
