import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake


class _StaticJsonLLM:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    async def ainvoke(self, messages, *, config):
        self.calls.append((messages, config))
        content = json.dumps(self.payload)
        return SimpleNamespace(content=content)


def test_visible_source_spans_keep_cjk_punctuation_out_of_github_url() -> None:
    spans = source_intake.build_visible_source_spans(
        '@杏山千纱 千纱千纱，能不能帮我分析一下这个项目'
        'https://github.com/sdyzjx/open-yachiyo，辛苦你了'
    )

    assert 'https://github.com/sdyzjx/open-yachiyo' in spans
    assert all('，辛苦你了' not in span for span in spans)


def test_normalize_source_intake_output_bounds_unknown_values() -> None:
    result = source_intake.normalize_source_intake_output(
        {
            'task_source_mode': 'invented_mode',
            'source_mentions': [
                {
                    'raw_text': 'https://github.com/owner/repo',
                    'role': 'new_role',
                    'family_hint': 'new_family',
                },
                {'raw_text': ''},
                {'raw_text': 'x' * 900},
            ],
        }
    )

    assert result.task_source_mode == 'unclear'
    assert result.source_mentions[0].role == 'unknown'
    assert result.source_mentions[0].family_hint == 'unknown'
    assert len(result.source_mentions) == 2
    assert len(result.source_mentions[1].raw_text) == 512


def test_normalize_source_intake_output_accepts_inline_source_fields() -> None:
    result = source_intake.normalize_source_intake_output(
        {
            'task_source_mode': 'inline_bundle',
            'source_mentions': [
                {
                    'raw_text': 'def solve():\n    return 1',
                    'role': 'primary_code_source',
                    'family_hint': 'inline_code',
                    'language_hint': 'python',
                    'filename_hint': 'solver.py',
                },
                {
                    'raw_text': 'Traceback (most recent call last):',
                    'role': 'supporting_context',
                    'family_hint': 'log_or_trace',
                },
            ],
        }
    )

    assert result.task_source_mode == 'inline_bundle'
    assert len(result.source_mentions) == 2
    assert result.source_mentions[0].family_hint == 'inline_code'
    assert result.source_mentions[0].language_hint == 'python'
    assert result.source_mentions[0].filename_hint == 'solver.py'
    assert result.source_mentions[1].family_hint == 'log_or_trace'


@pytest.mark.asyncio
async def test_run_source_intake_uses_pm_llm_and_visible_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _StaticJsonLLM(
        {
            'task_source_mode': 'single_primary',
            'source_mentions': [
                {
                    'raw_text': 'https://github.com/owner/repo',
                    'role': 'primary_code_source',
                    'family_hint': 'github_repository',
                }
            ],
        }
    )
    monkeypatch.setattr(source_intake, '_source_intake_llm', llm)

    result = await source_intake.run_source_intake(
        'Please inspect https://github.com/owner/repo.'
    )

    assert result.task_source_mode == 'single_primary'
    assert result.source_mentions[0].raw_text == 'https://github.com/owner/repo'
    assert llm.calls
    _, config = llm.calls[0]
    assert config.route_name == 'CODING_AGENT_PM_LLM'


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_source_intake_extracts_captured_github_task() -> None:
    result = await source_intake.run_source_intake(
        '@杏山千纱 千纱千纱，能不能帮我分析一下这个项目'
        'https://github.com/sdyzjx/open-yachiyo'
    )

    assert result.task_source_mode == 'single_primary'
    assert result.source_mentions
    assert result.source_mentions[0].raw_text == (
        'https://github.com/sdyzjx/open-yachiyo'
    )
    assert result.source_mentions[0].role == 'primary_code_source'


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_source_intake_marks_unsupported_web_url() -> None:
    result = await source_intake.run_source_intake(
        'Analyze this page https://zh.moegirl.org.cn/'
        '%E6%9D%8F%E5%B1%B1%E5%8D%83%E7%BA%B1'
    )

    assert result.task_source_mode in {'single_primary', 'unclear'}
    assert result.source_mentions
    assert result.source_mentions[0].raw_text.startswith(
        'https://zh.moegirl.org.cn/'
    )


@pytest.mark.live_llm
@pytest.mark.asyncio
async def test_live_source_intake_marks_multiple_repos_mode() -> None:
    result = await source_intake.run_source_intake(
        'Compare https://github.com/owner/one with https://github.com/owner/two.'
    )

    assert result.task_source_mode in {'compare_sources', 'unclear'}
    assert len(result.source_mentions) == 2
