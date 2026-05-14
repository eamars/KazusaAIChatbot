"""Prompt contract tests for the global character growth LLM."""

from __future__ import annotations

import inspect
import json

from kazusa_ai_chatbot.global_character_growth import llm


def test_candidate_generation_prompt_uses_required_chinese_sections() -> None:
    """The prompt should be a coherent Chinese contract, not an appended block."""

    prompt = llm.GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT

    required_sections = [
        '# 语言政策',
        '# 核心过滤器',
        '# 运行规则',
        '# 任务目标',
        '# 思考路径',
        '# 输入格式',
        '# 输出格式',
    ]
    for section in required_sections:
        assert section in prompt

    assert prompt.index('# 语言政策') < prompt.index('# 核心过滤器')
    assert prompt.index('# 核心过滤器') < prompt.index('# 运行规则')
    assert prompt.index('# 运行规则') < prompt.index('# 任务目标')
    assert prompt.index('# 任务目标') < prompt.index('# 思考路径')
    assert prompt.index('# 思考路径') < prompt.index('# 输入格式')
    assert prompt.index('# 输入格式') < prompt.index('# 输出格式')
    assert '附加规则' not in prompt
    assert '额外规则' not in prompt


def test_candidate_generation_prompt_covers_rejection_policy() -> None:
    """Industrial scope controls should be represented inside the prompt."""

    prompt = llm.GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT

    for text in (
        '技术',
        '产品',
        '食物',
        '茶',
        '烹饪',
        '地点',
        '爱好',
        '用户专属',
        '群聊',
        '隐私',
        '全局人格成长',
    ):
        assert text in prompt


def test_candidate_generation_prompt_has_schema_keys_and_no_character_name() -> None:
    """Reusable prompt text must not hard-code a concrete character."""

    prompt = llm.GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT

    for key in (
        'candidate_deltas',
        'candidate_action',
        'growth_axis',
        'source_card_ids',
        'scope_assessment',
        'private_detail_risk',
        'support_level',
    ):
        assert key in prompt
    assert '杏山' not in prompt
    assert '千纱' not in prompt
    assert '角色' not in prompt
    assert '{character_name}' in prompt


def test_candidate_generation_prompt_avoids_unrelated_internal_terms() -> None:
    """Growth candidate evaluation should avoid architecture and storage leakage."""

    prompt = llm.GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT

    for forbidden in (
        'L2',
        'L3',
        'Dialog',
        'MongoDB',
        '数据库',
        '写库',
        '潜意识层',
    ):
        assert forbidden not in prompt


def test_build_candidate_prompt_renders_payload_separately() -> None:
    """Prompt rendering should keep the schema in system text and data in human text."""

    payload = {
        "evaluation_mode": "global_character_growth_v1",
        "prompt_version": "global_character_growth_candidate_v1",
        "memory_cards": [],
        "current_global_growth_traits": [],
        "candidate_limits": {
            "max_candidates": 4,
            "max_source_cards_per_candidate": 8,
        },
        "allowed_growth_axes": ["clarity"],
    }

    rendered = llm.build_candidate_generation_prompt(
        payload=payload,
        character_name="Test Character",
    )

    assert "Test Character" in rendered.system_prompt
    assert "Test Character" not in llm.GLOBAL_CHARACTER_GROWTH_CANDIDATE_PROMPT
    assert json.loads(rendered.human_prompt) == payload


def test_candidate_prompt_char_count_matches_rendered_prompts() -> None:
    """Prompt size checks should measure the final rendered prompt pair."""

    payload = {
        "evaluation_mode": "global_character_growth_v1",
        "prompt_version": "global_character_growth_candidate_v1",
        "memory_cards": [],
        "current_global_growth_traits": [],
        "candidate_limits": {
            "max_candidates": 4,
            "max_source_cards_per_candidate": 8,
        },
        "allowed_growth_axes": ["clarity"],
    }

    rendered = llm.build_candidate_generation_prompt(
        payload=payload,
        character_name="Test Character",
    )
    count = llm.count_candidate_generation_prompt_chars(
        payload=payload,
        character_name="Test Character",
    )

    assert count == len(rendered.system_prompt) + len(rendered.human_prompt)


def test_generate_growth_candidates_default_character_label_is_chinese() -> None:
    """Default prompt rendering should not inject an English role label."""

    signature = inspect.signature(llm.generate_growth_candidates)

    assert signature.parameters["character_name"].default == "当前主体"


def test_validate_candidate_response_shape_rejects_missing_array() -> None:
    """LLM output without candidate_deltas should fail validation cleanly."""

    warnings = llm.validate_llm_candidate_response_shape({"summary": "none"})

    assert warnings == ["candidate_deltas must be a list"]
