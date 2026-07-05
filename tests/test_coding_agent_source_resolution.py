import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import source_resolver
from kazusa_ai_chatbot.coding_agent.code_fetching.source_intake import (
    SourceIntakeResult,
    SourceMention,
)


def _intake(
    mode: str,
    mentions: list[tuple[str, str, str]],
) -> SourceIntakeResult:
    return SourceIntakeResult(
        task_source_mode=mode,
        source_mentions=tuple(
            SourceMention(raw_text=raw_text, role=role, family_hint=family)
            for raw_text, role, family in mentions
        ),
    )


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
