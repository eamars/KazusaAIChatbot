"""Prompt and documentation boundary tests for internal monologue residue."""

from __future__ import annotations

import pytest

from pathlib import Path

from kazusa_ai_chatbot.internal_monologue_residue import recorder


_ROOT = Path(__file__).resolve().parents[1]


def test_recorder_system_prompt_is_chinese_first_person_and_not_appended() -> None:
    """Recorder prompt is a coherent Chinese prompt, not an appended patch."""

    prompt = recorder.render_recorder_system_prompt(
        character_name='杏山千纱',
        ambient_condition='群聊环境',
    )

    assert '杏山千纱' in prompt
    assert '群聊环境' in prompt
    assert '# 语言政策' in prompt
    assert '# 核心任务' in prompt
    assert '# 证据身份' in prompt
    assert '# 生成步骤' in prompt
    assert '# 私念视角契约' in prompt
    assert '# 输出格式' in prompt
    assert '# 输入格式' not in prompt
    assert '# Input Format' not in prompt
    assert '# Output Format' not in prompt
    assert '补充约束' not in prompt
    assert '追加规则' not in prompt
    assert 'system message' not in prompt.lower()
    assert '不要用我的名字称呼我自己' in prompt
    assert '对方、那个人、某人、他或她' in prompt
    assert 'source_reliability_notes' in prompt
    assert '来源可靠性限制' in prompt
    assert '侧线/未定对象' in prompt
    assert '来源优先级' in prompt
    assert prompt.count('# 输出格式') == 1


def test_internal_monologue_residue_experiments_are_removed() -> None:
    """Production implementation must not leave residue POC scripts behind."""

    experiment_dir = _ROOT / 'experiments'
    matches = sorted(experiment_dir.glob('internal_monologue_residue*'))
    pycache_dir = experiment_dir / '__pycache__'
    if pycache_dir.exists():
        matches.extend(sorted(pycache_dir.glob('internal_monologue_residue*')))

    assert matches == []
