import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching.managed_inline import (
    InlineSourceBundle,
)
from kazusa_ai_chatbot.coding_agent.code_fetching import source_resolver
from kazusa_ai_chatbot.coding_agent.code_fetching.source_intake import (
    SourceIntakeResult,
    SourceMention,
)


def _intake(
    mode: str,
    mentions: list[tuple[str, str, str] | tuple[str, str, str, str, str]],
) -> SourceIntakeResult:
    return SourceIntakeResult(
        task_source_mode=mode,
        source_mentions=tuple(
            _mention_from_tuple(mention)
            for mention in mentions
        ),
    )


def _mention_from_tuple(
    mention: tuple[str, str, str] | tuple[str, str, str, str, str],
) -> SourceMention:
    raw_text, role, family = mention[:3]
    language_hint = ''
    filename_hint = ''
    if len(mention) == 5:
        language_hint = mention[3]
        filename_hint = mention[4]
    return SourceMention(
        raw_text=raw_text,
        role=role,
        family_hint=family,
        language_hint=language_hint,
        filename_hint=filename_hint,
    )


def test_resolves_fenced_inline_code_as_inline_bundle() -> None:
    task_text = (
        'Can you review this Z3 fragment?\n'
        '```python\n'
        'from z3 import Int, Solver\n'
        'x = Int("x")\n'
        'solver = Solver()\n'
        'solver.add(x > 3)\n'
        '```\n'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'inline_bundle',
            [
                (
                    'x = Int("x")',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'solver.py',
                )
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert result.issue_code is None
    assert isinstance(result.source, InlineSourceBundle)
    assert len(result.source.fragments) == 1
    assert result.source.fragments[0].content == (
        'from z3 import Int, Solver\n'
        'x = Int("x")\n'
        'solver = Solver()\n'
        'solver.add(x > 3)\n'
    )
    assert result.source.fragments[0].language_hint == 'python'
    assert result.source.fragments[0].filename_hint == 'solver.py'


def test_resolves_multiple_inline_fragments_as_one_bundle() -> None:
    task_text = (
        'Review both files:\n'
        '```python\n'
        'def parse(value):\n'
        '    return value.strip()\n'
        '```\n'
        '```python\n'
        'def test_parse():\n'
        '    assert parse(" x ") == "x"\n'
        '```\n'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'inline_bundle',
            [
                (
                    'def parse(value):',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'parser.py',
                ),
                (
                    'def test_parse():',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'test_parser.py',
                ),
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert isinstance(result.source, InlineSourceBundle)
    assert [item.filename_hint for item in result.source.fragments] == [
        'parser.py',
        'test_parser.py',
    ]


def test_mixed_github_and_inline_primary_sources_need_clarification() -> None:
    task_text = (
        'Analyze https://github.com/owner/repo and this snippet too:\n'
        '```python\n'
        'def local_only():\n'
        '    return 1\n'
        '```'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'mixed_primary_with_context',
            [
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'def local_only():',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    '',
                ),
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'mixed_primary_sources'
    assert result.source is None


def test_log_only_supporting_context_does_not_become_primary_source() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'The run failed with this log:\n'
                'Traceback (most recent call last):\n'
                '  File "main.py", line 1, in <module>'
            )
        },
        _intake(
            'unclear',
            [
                (
                    'Traceback (most recent call last):',
                    'supporting_context',
                    'log_or_trace',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'supporting_context_only'
    assert result.source is None


def test_explicit_invalid_source_is_authoritative_over_inline_code() -> None:
    task_text = (
        'If the link is bad, use this code:\n'
        '```python\n'
        'print("fallback")\n'
        '```'
    )

    result = source_resolver.resolve_source_request(
        {
            'source_url': 'https://github.com/not a valid/repo',
            'question': task_text,
        },
        _intake(
            'inline_bundle',
            [
                (
                    'print("fallback")',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    '',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'malformed_source'
    assert result.source is None


def test_inline_source_secret_like_content_needs_redaction() -> None:
    task_text = (
        'Review this file:\n'
        '```python\n'
        'API_KEY = "abc123"\n'
        'def call():\n'
        '    return API_KEY\n'
        '```'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'inline_bundle',
            [
                (
                    'API_KEY = "abc123"',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'client.py',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'inline_source_unsafe_content'
    assert result.source is None


def test_truncated_inline_source_needs_complete_code_text() -> None:
    task_text = (
        'This code is truncated but can you analyze it?\n'
        '```python\n'
        'def process(value):\n'
        '    if value:\n'
        '        return value\n'
        '    # rest omitted\n'
        '```'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'inline_bundle',
            [
                (
                    'def process(value):',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'process.py',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'inline_source_incomplete'
    assert result.source is None


def test_inline_source_too_many_fragments_needs_narrowing() -> None:
    blocks: list[str] = []
    mentions: list[tuple[str, str, str, str, str]] = []
    for index in range(9):
        blocks.append(
            '```python\n'
            f'def fragment_{index}():\n'
            f'    return {index}\n'
            '```'
        )
        mentions.append(
            (
                f'def fragment_{index}():',
                'primary_code_source',
                'inline_code',
                'python',
                f'fragment_{index}.py',
            )
        )
    task_text = 'Review these fragments:\n' + '\n'.join(blocks)

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake('inline_bundle', mentions),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'inline_source_too_many_fragments'
    assert result.source is None


def test_image_only_source_needs_text_source() -> None:
    result = source_resolver.resolve_source_request(
        {'question': 'Can you analyze the code in the attached screenshot?'},
        _intake(
            'unclear',
            [
                (
                    'attached screenshot',
                    'primary_code_source',
                    'attachment',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'image_only_source'
    assert result.source is None


def test_reference_only_unsupported_source_does_not_block_inline_primary() -> None:
    task_text = (
        'Review this snippet; the docs are only background '
        'https://docs.example.com/api.\n'
        '```python\n'
        'def load(value):\n'
        '    return value or "default"\n'
        '```'
    )

    result = source_resolver.resolve_source_request(
        {'question': task_text},
        _intake(
            'mixed_primary_with_context',
            [
                (
                    'https://docs.example.com/api',
                    'reference_only',
                    'documentation_url',
                ),
                (
                    'def load(value):',
                    'primary_code_source',
                    'inline_code',
                    'python',
                    'loader.py',
                ),
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert isinstance(result.source, InlineSourceBundle)
    assert result.limitations


def test_trusted_inline_sources_resolve_without_source_intake() -> None:
    result = source_resolver.resolve_source_request(
        {
            'inline_sources': [
                {
                    'content': 'def answer():\n    return 42\n',
                    'filename_hint': 'answer.py',
                    'language_hint': 'python',
                    'source_label': 'user pasted code',
                }
            ]
        },
        None,
    )

    assert result.status == 'succeeded'
    assert isinstance(result.source, InlineSourceBundle)
    assert result.source.fragments[0].content == 'def answer():\n    return 42\n'
    assert result.source.fragments[0].filename_hint == 'answer.py'


def test_resolves_llm_extracted_github_url_from_cjk_chat_message() -> None:
    request = {
        'question': (
            '@杏山千纱 千纱千纱，能不能帮我分析一下这个项目'
            'https://github.com/sdyzjx/open-yachiyo'
        )
    }
    result = source_resolver.resolve_source_request(
        request,
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/sdyzjx/open-yachiyo',
                    'primary_code_source',
                    'github_repository',
                )
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert result.issue_code is None
    assert result.source is not None
    assert result.source.owner == 'sdyzjx'
    assert result.source.repo == 'open-yachiyo'


def test_rejects_unanchored_llm_source_instead_of_fetching_it() -> None:
    result = source_resolver.resolve_source_request(
        {'question': 'Can you analyze this project?'},
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/invented/repo',
                    'primary_code_source',
                    'github_repository',
                )
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'source_not_visible_in_request'
    assert result.source is None
    assert result.retry_feedback


@pytest.mark.asyncio
async def test_select_source_retries_source_intake_with_resolver_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    task_text = 'Please inspect https://github.com/owner/repo.'

    async def fake_run_source_intake(
        task_text_arg: str,
        *,
        retry_feedback: list[str] | None = None,
    ) -> SourceIntakeResult:
        calls.append(
            {
                'task_text': task_text_arg,
                'retry_feedback': retry_feedback,
            }
        )
        if retry_feedback is None:
            return _intake(
                'single_primary',
                [
                    (
                        'https://github.com/invented/repo',
                        'primary_code_source',
                        'github_repository',
                    )
                ],
            )

        assert retry_feedback
        assert retry_feedback[0].startswith('Use only source text visible')
        return _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                )
            ],
        )

    monkeypatch.setattr(
        source_resolver.source_intake,
        'run_source_intake',
        fake_run_source_intake,
    )

    trace_summary: list[str] = []
    result = await source_resolver.select_source_for_request(
        {'question': task_text},
        trace_summary,
    )

    assert result.status == 'succeeded'
    assert result.source is not None
    assert result.source.owner == 'owner'
    assert result.source.repo == 'repo'
    assert len(calls) == 2
    assert calls[0]['retry_feedback'] is None
    assert calls[1]['retry_feedback']
    assert 'source_intake:retried_once' in result.trace_summary
    assert trace_summary == list(result.trace_summary)


@pytest.mark.asyncio
async def test_select_source_retries_compare_mode_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []
    task_text = (
        'Look at these and summarize the design: '
        'https://github.com/owner/one https://github.com/owner/two'
    )

    async def fake_run_source_intake(
        task_text_arg: str,
        *,
        retry_feedback: list[str] | None = None,
    ) -> SourceIntakeResult:
        calls.append(
            {
                'task_text': task_text_arg,
                'retry_feedback': retry_feedback,
            }
        )
        mentions = [
            (
                'https://github.com/owner/one',
                'primary_code_source',
                'github_repository',
            ),
            (
                'https://github.com/owner/two',
                'primary_code_source',
                'github_repository',
            ),
        ]
        if retry_feedback is None:
            return _intake('compare_sources', mentions)

        assert retry_feedback
        assert retry_feedback[0].startswith('Use compare_sources only')
        return _intake('unclear', mentions)

    monkeypatch.setattr(
        source_resolver.source_intake,
        'run_source_intake',
        fake_run_source_intake,
    )

    trace_summary: list[str] = []
    result = await source_resolver.select_source_for_request(
        {'question': task_text},
        trace_summary,
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'ambiguous_primary_source'
    assert result.source is None
    assert len(calls) == 2
    assert calls[0]['retry_feedback'] is None
    assert calls[1]['retry_feedback']
    assert 'source_intake:retried_once' in result.trace_summary


def test_rejects_supported_url_syntax_with_unsupported_provider() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Analyze https://zh.moegirl.org.cn/'
                '%E6%9D%8F%E5%B1%B1%E5%8D%83%E7%BA%B1'
            )
        },
        _intake(
            'single_primary',
            [
                (
                    'https://zh.moegirl.org.cn/'
                    '%E6%9D%8F%E5%B1%B1%E5%8D%83%E7%BA%B1',
                    'primary_code_source',
                    'web_page',
                )
            ],
        ),
    )

    assert result.status == 'rejected'
    assert result.issue_code == 'unsupported_provider'
    assert result.source is None


def test_bare_package_name_is_not_treated_as_code_source() -> None:
    result = source_resolver.resolve_source_request(
        {'question': 'Can you analyze react?'},
        _intake('source_free', []),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'no_source_found'


def test_github_issue_is_rejected_when_it_is_the_primary_target() -> None:
    result = source_resolver.resolve_source_request(
        {'question': 'Analyze https://github.com/owner/repo/issues/123'},
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo/issues/123',
                    'primary_code_source',
                    'github_issue',
                )
            ],
        ),
    )

    assert result.status == 'rejected'
    assert result.issue_code == 'unsupported_source_family'
    assert result.source is None


def test_github_issue_can_scope_a_repository_analysis_when_reference_only() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Use https://github.com/owner/repo/issues/123 as context and '
                'analyze the repository.'
            )
        },
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo/issues/123',
                    'reference_only',
                    'github_issue',
                ),
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                ),
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert result.source is not None
    assert result.source.owner == 'owner'
    assert result.source.repo == 'repo'
    assert result.limitations


def test_rejects_explicit_multi_repo_compare_request() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Compare https://github.com/owner/one with '
                'https://github.com/owner/two.'
            )
        },
        _intake(
            'compare_sources',
            [
                (
                    'https://github.com/owner/one',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'https://github.com/owner/two',
                    'primary_code_source',
                    'github_repository',
                ),
            ],
        ),
    )

    assert result.status == 'rejected'
    assert result.issue_code == 'unsupported_multi_source'


def test_asks_for_input_when_multiple_primary_repos_are_ambiguous() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Look at https://github.com/owner/one and '
                'https://github.com/owner/two.'
            )
        },
        _intake(
            'unclear',
            [
                (
                    'https://github.com/owner/one',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'https://github.com/owner/two',
                    'primary_code_source',
                    'github_repository',
                ),
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'ambiguous_primary_source'


def test_prefers_nested_scope_inside_the_same_repository() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Analyze https://github.com/owner/repo and focus on '
                'https://github.com/owner/repo/tree/main/src/app.'
            )
        },
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'https://github.com/owner/repo/tree/main/src/app',
                    'scope_modifier',
                    'github_directory',
                ),
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert result.source is not None
    assert result.source.source_kind == 'directory'
    assert result.source.repo_relative_path == 'src/app'


def test_conflicting_same_repo_file_scopes_need_user_choice() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Check https://github.com/owner/repo/blob/main/a.py and '
                'https://github.com/owner/repo/blob/main/b.py.'
            )
        },
        _intake(
            'unclear',
            [
                (
                    'https://github.com/owner/repo/blob/main/a.py',
                    'primary_code_source',
                    'github_file',
                ),
                (
                    'https://github.com/owner/repo/blob/main/b.py',
                    'primary_code_source',
                    'github_file',
                ),
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'ambiguous_primary_source'


def test_required_supporting_web_doc_blocks_fetching() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Analyze https://github.com/owner/repo using required spec '
                'https://docs.example.com/spec.'
            )
        },
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'https://docs.example.com/spec',
                    'supporting_context',
                    'documentation_url',
                ),
            ],
        ),
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'required_supporting_source_unsupported'
    assert result.source is None


def test_reference_only_unsupported_doc_preserves_limitation_and_fetches() -> None:
    result = source_resolver.resolve_source_request(
        {
            'question': (
                'Analyze https://github.com/owner/repo; optional context is '
                'https://docs.example.com/spec.'
            )
        },
        _intake(
            'single_primary',
            [
                (
                    'https://github.com/owner/repo',
                    'primary_code_source',
                    'github_repository',
                ),
                (
                    'https://docs.example.com/spec',
                    'reference_only',
                    'documentation_url',
                ),
            ],
        ),
    )

    assert result.status == 'succeeded'
    assert result.source is not None
    assert result.limitations


def test_explicit_invalid_source_is_authoritative_over_question_url() -> None:
    result = source_resolver.resolve_source_request(
        {
            'source_url': 'https://github.com/not a valid/repo',
            'question': 'Actually analyze https://github.com/owner/real.',
        },
        None,
    )

    assert result.status == 'needs_user_input'
    assert result.issue_code == 'malformed_source'
    assert result.source is None


def test_raw_local_path_in_chat_is_rejected_as_unsupported_source_family() -> None:
    result = source_resolver.resolve_source_request(
        {'question': 'Analyze C:\\workspace\\repo\\src\\app.py'},
        _intake(
            'single_primary',
            [
                (
                    'C:\\workspace\\repo\\src\\app.py',
                    'primary_code_source',
                    'local_path',
                )
            ],
        ),
    )

    assert result.status == 'rejected'
    assert result.issue_code == 'unsupported_source_family'
    assert result.source is None
