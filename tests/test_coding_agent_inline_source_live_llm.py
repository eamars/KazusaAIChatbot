"""Real LLM checks for inline source intake and deterministic resolution."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kazusa_ai_chatbot.coding_agent.code_fetching import managed_inline
from kazusa_ai_chatbot.coding_agent.code_fetching import source_intake
from kazusa_ai_chatbot.coding_agent.code_fetching import source_resolver
from kazusa_ai_chatbot.coding_agent.code_fetching.managed_inline import (
    InlineSourceBundle,
)
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_TEST_NAME = "coding_agent_inline_source_live_llm"
_EXPLICIT_SOURCE_FIELDS = ("source_url", "repo_url", "repo_hint")


async def test_inline_source_single_fenced_python_live_llm(tmp_path: Path) -> None:
    await _run_live_case(
        tmp_path,
        case_id="single_fenced_python",
        task_text=(
            "Please review this function for correctness.\n"
            "```python\n"
            "def clamp(value, low, high):\n"
            "    if value < low:\n"
            "        return low\n"
            "    if value > high:\n"
            "        return high\n"
            "    return value\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_unfenced_python_anchor_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="unfenced_python_anchor",
        task_text=(
            "Analyze this pasted source fragment: def normalize(value):\n"
            "    text = value.strip()\n"
            "    return text.lower()\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_multiple_fragments_one_task_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="multiple_fragments_one_task",
        task_text=(
            "Review these two related files together.\n"
            "```python\n"
            "def parse_row(row):\n"
            "    return row.split(',')\n"
            "```\n"
            "```python\n"
            "def test_parse_row():\n"
            "    assert parse_row('a,b') == ['a', 'b']\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_filename_hint_live_llm(tmp_path: Path) -> None:
    await _run_live_case(
        tmp_path,
        case_id="filename_hint",
        task_text=(
            "File `validator.py`:\n"
            "```python\n"
            "def is_valid(name):\n"
            "    return bool(name and name.strip())\n"
            "```\n"
            "Can you inspect the validation behavior?"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_diff_review_live_llm(tmp_path: Path) -> None:
    await _run_live_case(
        tmp_path,
        case_id="diff_review",
        task_text=(
            "Please review this patch.\n"
            "```diff\n"
            "-timeout = 1\n"
            "+timeout = max(1, configured_timeout)\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_code_plus_stack_trace_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="code_plus_stack_trace",
        task_text=(
            "Find why this code fails.\n"
            "```python\n"
            "def first(items):\n"
            "    return items[0]\n"
            "```\n"
            "Traceback (most recent call last): IndexError: list index out of range"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_code_plus_requirements_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="code_plus_requirements",
        task_text=(
            "Requirement: empty input should return an empty list.\n"
            "```python\n"
            "def split_tags(value):\n"
            "    return value.split(',')\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_cjk_prompt_python_code_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="cjk_prompt_python_code",
        task_text=(
            '千纱，帮我分析一下这段代码为什么边界处理不稳定：\n'
            '```python\n'
            'def pick(values):\n'
            '    if len(values) == 1:\n'
            '        return values[1]\n'
            '    return values[0]\n'
            '```\n'
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_markdown_language_fence_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="markdown_language_fence",
        task_text=(
            "The source is in this Python fence.\n"
            "```python\n"
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.value = 0\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_production_z3_replay_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="production_z3_replay",
        task_text=(
            '关于你发的 Z3 代码，帮我分析一下失败原因。\n'
            '```python\n'
            'from z3 import Int, Solver\n'
            'x = Int("x")\n'
            'solver = Solver()\n'
            'solver.add(x > 3)\n'
            'solver.add(x < 2)\n'
            'print(solver.check())\n'
            '```\n'
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_no_source_asks_for_source_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="no_source_asks_for_source",
        task_text="Can you analyze why my project fails?",
        expected_status="needs_user_input",
        expected_issue="no_source_found",
        expect_inline=False,
    )


async def test_inline_source_image_only_needs_text_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="image_only_needs_text",
        task_text="Please analyze the code in the attached screenshot.",
        expected_status="needs_user_input",
        expected_issue="image_only_source",
        expect_inline=False,
    )


async def test_inline_source_truncated_code_needs_resend_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="truncated_code_needs_resend",
        task_text=(
            "This pasted code is truncated here; can you still analyze it?\n"
            "```python\n"
            "def build(items):\n"
            "    result = []\n"
            "    # rest omitted\n"
            "```\n"
        ),
        expected_status="needs_user_input",
        expected_issue="inline_source_incomplete",
        expect_inline=False,
    )


async def test_inline_source_oversized_needs_narrowing_live_llm(
    tmp_path: Path,
) -> None:
    large_body = "x = 1\n" * 2500
    await _run_live_case(
        tmp_path,
        case_id="oversized_needs_narrowing",
        task_text=(
            "Review this oversized pasted file.\n"
            "```python\n"
            f"{large_body}"
            "```\n"
        ),
        expected_status="needs_user_input",
        expected_issue="inline_source_too_large",
        expect_inline=False,
    )


async def test_inline_source_too_many_fragments_needs_narrowing_live_llm(
    tmp_path: Path,
) -> None:
    blocks = [
        "```python\n"
        f"def fragment_{index}():\n"
        f"    return {index}\n"
        "```"
        for index in range(9)
    ]
    await _run_live_case(
        tmp_path,
        case_id="too_many_fragments_needs_narrowing",
        task_text="Review all of these independent snippets.\n" + "\n".join(blocks),
        expected_status="needs_user_input",
        expected_issue="inline_source_too_many_fragments",
        expect_inline=False,
    )


async def test_inline_source_secret_like_content_needs_redaction_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="secret_like_content_needs_redaction",
        task_text=(
            "Review this pasted client code.\n"
            "```python\n"
            "API_KEY = 'abc123'\n"
            "def call():\n"
            "    return API_KEY\n"
            "```\n"
        ),
        expected_status="needs_user_input",
        expected_issue="inline_source_unsafe_content",
        expect_inline=False,
    )


async def test_inline_source_mixed_primary_sources_needs_clarification_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="mixed_primary_sources_needs_clarification",
        task_text=(
            "Analyze https://github.com/owner/repo and this pasted code too.\n"
            "```python\n"
            "def local_only():\n"
            "    return 1\n"
            "```\n"
        ),
        expected_status="needs_user_input",
        expected_issue="mixed_primary_sources",
        expect_inline=False,
    )


async def test_inline_source_explicit_invalid_repo_stays_authoritative_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="explicit_invalid_repo_stays_authoritative",
        task_text=(
            "If that repository is bad, use this pasted code.\n"
            "```python\n"
            "print('fallback')\n"
            "```\n"
        ),
        expected_status="needs_user_input",
        expected_issue="malformed_source",
        expect_inline=False,
        request_fields={"source_url": "https://github.com/not a valid/repo"},
    )


async def test_inline_source_unsupported_reference_only_does_not_block_inline_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="unsupported_reference_only_does_not_block_inline",
        task_text=(
            "Review this snippet; this docs URL is optional context "
            "https://docs.example.com/api.\n"
            "```python\n"
            "def load(value):\n"
            "    return value or 'default'\n"
            "```\n"
        ),
        expected_status="succeeded",
        expected_issue=None,
        expect_inline=True,
    )


async def test_inline_source_log_only_supporting_context_no_primary_live_llm(
    tmp_path: Path,
) -> None:
    await _run_live_case(
        tmp_path,
        case_id="log_only_supporting_context_no_primary",
        task_text=(
            "I only have this run log, can you analyze the code?\n"
            "Traceback (most recent call last):\n"
            "  File \"main.py\", line 1, in <module>\n"
            "IndexError: list index out of range\n"
        ),
        expected_status="needs_user_input",
        expected_issue="supporting_context_only",
        expect_inline=False,
    )


async def _run_live_case(
    tmp_path: Path,
    *,
    case_id: str,
    task_text: str,
    expected_status: str,
    expected_issue: str | None,
    expect_inline: bool,
    request_fields: dict[str, object] | None = None,
) -> None:
    request = {} if request_fields is None else dict(request_fields)
    request["question"] = task_text
    request["workspace_root"] = str(tmp_path / case_id)

    first_intake = None
    retry_intake = None
    if any(request.get(field_name) for field_name in _EXPLICIT_SOURCE_FIELDS):
        resolution = source_resolver.resolve_source_request(request, None)
    else:
        first_intake = await source_intake.run_source_intake(task_text)
        resolution = source_resolver.resolve_source_request(request, first_intake)
        if resolution.retry_feedback:
            retry_intake = await source_intake.run_source_intake(
                task_text,
                retry_feedback=list(resolution.retry_feedback),
            )
            resolution = source_resolver.resolve_source_request(
                request,
                retry_intake,
            )

    materialized = None
    if isinstance(resolution.source, InlineSourceBundle):
        repository, source_scope = (
            managed_inline.materialize_inline_source_bundle(
                resolution.source,
                str(tmp_path / case_id),
            )
        )
        materialized = {
            "repository": _public_repository(repository),
            "source_scope": source_scope,
            "fragment_count": len(resolution.source.fragments),
            "fragment_hashes": [
                _fragment_hash(fragment.content)
                for fragment in resolution.source.fragments
            ],
        }

    trace_path = write_llm_trace(
        _TEST_NAME,
        case_id,
        {
            "task_text": task_text,
            "request_fields": request_fields or {},
            "first_intake": (
                None if first_intake is None
                else source_intake.source_intake_result_to_dict(first_intake)
            ),
            "retry_intake": (
                None if retry_intake is None
                else source_intake.source_intake_result_to_dict(retry_intake)
            ),
            "resolution": source_resolver.source_resolution_to_dict(
                resolution,
            ),
            "materialized": materialized,
            "expected": {
                "status": expected_status,
                "issue_code": expected_issue,
                "expect_inline": expect_inline,
            },
            "human_review_note": (
                "Inspect source-intake roles, selected anchors, resolver "
                "status, and materialized public metadata."
            ),
        },
    )

    assert trace_path.exists()
    assert resolution.status == expected_status
    assert resolution.issue_code == expected_issue
    if expect_inline:
        assert isinstance(resolution.source, InlineSourceBundle)
        assert materialized is not None
        assert materialized["fragment_count"] >= 1
    else:
        assert resolution.source is None


def _public_repository(repository: dict[str, object]) -> dict[str, object]:
    public_repository = {
        key: value
        for key, value in repository.items()
        if key not in {"local_root", "workspace_root", "cache_key"}
    }
    return public_repository


def _fragment_hash(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest
