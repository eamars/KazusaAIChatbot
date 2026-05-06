"""Prompt contract tests for memory-writing LLM stages."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_images as images_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_memory_units as memory_units_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_reflection as reflection_module,
)
from kazusa_ai_chatbot.reflection_cycle import promotion as promotion_module
from scripts import sanitize_memory_writer_perspective as migration_module
from tests.test_reflection_cycle_stage1c_promotion import _promotion_payload


CHARACTER_NAME = "杏山千纱 (Kyōyama Kazusa)"


def test_memory_unit_prompts_render_third_person_contract() -> None:
    """Extractor and rewrite prompts should expose the durable memory contract."""

    extractor_prompt = memory_units_module._EXTRACTOR_PROMPT.format(
        character_name=CHARACTER_NAME,
    )
    rewrite_prompt = memory_units_module._REWRITE_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    for prompt in [extractor_prompt, rewrite_prompt]:
        _assert_third_person_contract(prompt)
        assert 'active_character' not in prompt
    assert 'speaker_kind' not in extractor_prompt
    assert 'speaker_name' in extractor_prompt
    assert '规范名称是一个不可拆分的完整字符串' in extractor_prompt


def test_reflection_prompts_render_third_person_contract() -> None:
    """Reflection prompts should not ask durable summaries to use first person."""

    prompts = [
        reflection_module._GLOBAL_STATE_UPDATER_PROMPT.format(
            character_name=CHARACTER_NAME,
        ),
        reflection_module._RELATIONSHIP_RECORDER_PROMPT.format(
            character_name=CHARACTER_NAME,
            user_name='测试用户',
            character_mbti='ISTJ',
        ),
    ]

    for prompt in prompts:
        _assert_third_person_contract(prompt)
    assert '以杏山千纱 (Kyōyama Kazusa)的第一人称' not in prompts[0]
    assert '描述“我”如何理解' not in prompts[1]


def test_character_image_prompts_render_third_person_contract() -> None:
    """Self-image prompts should require the profile-derived character name."""

    prompts = [
        images_module._CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT.format(
            character_name=CHARACTER_NAME,
        ),
        images_module._CHARACTER_IMAGE_COMPRESS_PROMPT.format(
            character_name=CHARACTER_NAME,
        ),
    ]

    for prompt in prompts:
        _assert_third_person_contract(prompt)


def test_reflection_promotion_prompt_renders_third_person_contract() -> None:
    """Promotion prompt should render the profile-derived character name."""

    prompt = promotion_module.build_global_promotion_prompt(
        _promotion_payload(),
        character_name=CHARACTER_NAME,
    )

    _assert_third_person_contract(prompt.system_prompt)
    assert 'evaluation_mode' in prompt.human_prompt


def test_migration_prompt_renders_third_person_contract() -> None:
    """Offline migration prompt should render the same profile-name contract."""

    prompt = migration_module.MIGRATION_REWRITE_SYSTEM_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    _assert_third_person_contract(prompt)
    assert '如果你无法确信某个字段里的' in prompt
    assert '"status": "ready|unchanged|blocked"' in prompt
    assert 'Markdown 反引号' in prompt
    assert '规范名称是一个不可拆分的完整字符串' in prompt
    assert '用户用亲昵称呼请求' in prompt


def _assert_third_person_contract(prompt: str) -> None:
    """Assert one rendered prompt includes the agreed memory perspective."""

    assert '# 记忆视角契约' in prompt
    assert '第三人称视角' in prompt
    assert CHARACTER_NAME in prompt
    assert '可写入记忆文本的唯一名称' in prompt
    assert '不要缩写、截断、翻译' in prompt
    assert '改写' in prompt
    assert '逐字复制完整字符串' in prompt
    assert '宁可省略主语' in prompt
    assert '短名' in prompt
    assert '不要用“我”指代' in prompt
    assert '不是指向' in prompt
    assert '角色' not in prompt
    assert '当前角色' not in prompt
    assert 'active character' not in prompt
    assert 'current character' not in prompt
    assert 'character_profile' not in prompt
    assert '{character_name}' not in prompt
