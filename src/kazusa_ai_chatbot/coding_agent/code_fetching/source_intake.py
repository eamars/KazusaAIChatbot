"""LLM source-intake contract for coding-agent fetch requests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CODING_AGENT_PM_LLM_API_KEY,
    CODING_AGENT_PM_LLM_BASE_URL,
    CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PM_LLM_MODEL,
    CODING_AGENT_PM_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

SOURCE_INTAKE_PROMPT = '''\
You are the source-intake specialist inside a coding agent.

Your job is to read the user-visible task text and extract source mentions that
the coding agent should resolve before code reading. Extract only text that is
visible in the task. Do not invent repositories, URLs, packages, paths, or
source roles.

The resolver currently supports one public GitHub code source:
- github.com repository URLs
- github.com tree directory URLs
- github.com blob file URLs
- raw.githubusercontent.com file URLs
- owner/repo GitHub shorthand when it is visibly written as a source
- inline code or diff text pasted in the task

Other visible source forms must still be extracted when they are the user's
target, but label their family accurately. Examples include package references,
documentation URLs, generic web pages, GitHub issues or pull requests, local
paths, stack traces, logs, attachments, and unknown source-like text.

Choose task_source_mode:
- single_primary: one primary code source is intended.
- inline_bundle: one or more inline code fragments form the source for one task.
- mixed_primary_with_context: inline code and another code source are both
  presented as possible primary targets, or one primary source has supporting
  context.
- compare_sources: the user asks to compare or jointly analyze multiple code
  sources.
- source_free: the task has no source to fetch.
- unclear: source intent or primary source choice is unclear.

Use source_free only for source-independent coding tasks, such as asking for a
new script, general programming explanation, or standalone code proposal. If
the task refers to this project, this repository, this codebase, or existing
source but does not provide a concrete source, choose unclear.

Use compare_sources only for an explicit comparison or joint multi-codebase
analysis request. Do not choose compare_sources merely because multiple URLs,
mirrors, short links, docs, or alternatives are visible. For multiple mentions
that appear to describe one intended target, choose single_primary when the
primary target is clear, otherwise choose unclear.

Example:
Task: Review this mirror if possible: https://gitlab.example/group/project,
and this short link https://short.example/project.
Correct task_source_mode: single_primary.
Reason: these are alternative visible sources for one intended review target,
not a request to compare two codebases.

Example:
Task: Look at these and summarize the design: https://github.com/a/one
https://github.com/b/two.
Correct task_source_mode: unclear.
Reason: more than one repository is visible, but the user did not explicitly
ask to compare them or choose a primary repository.

Example:
Task: Analyze these two files: https://github.com/a/repo/blob/main/a.py and
https://github.com/a/repo/blob/main/b.py.
Correct task_source_mode: single_primary.
Reason: both mentions are scopes inside one codebase. This is not a
multi-repository comparison request.

Choose each source mention role:
- primary_code_source: the source to fetch as code.
- scope_modifier: narrows or modifies another source in the same repository.
- supporting_context: required external material needed to perform the task.
- reference_only: optional background, optional context, or an unsupported clue
  that should not become the primary fetched content by itself.
- unknown: visible source-like text whose role is unclear.

For pasted code blocks, inline snippets, or diffs, set family_hint to
inline_code or inline_diff and return a short exact anchor visible inside the
fragment. Do not copy long code into the JSON. When a language fence or safe
filename hint is visible, include language_hint and filename_hint. Stack traces,
logs, and command output are supporting_context unless the user explicitly asks
to analyze that log as the source artifact.

For GitHub issues, pull requests, and discussions, distinguish the requested
target carefully. If the user asks to analyze the issue, thread, bug report, or
PR content itself, use primary_code_source with the GitHub thread family. If the
user says to use the repository behind the issue, asks for a project-level or
repository-level summary, or says not to use the thread content, use
reference_only with the GitHub thread family so the resolver can derive the
repository identity.

Choose each source mention family_hint:
- github_repository
- github_directory
- github_file
- raw_github_file
- github_issue
- github_pull
- github_discussion
- package_reference
- documentation_url
- web_page
- local_path
- inline_code
- inline_diff
- log_or_trace
- attachment
- unknown

Return strict JSON:
{
  "task_source_mode": "single_primary | inline_bundle | mixed_primary_with_context | compare_sources | source_free | unclear",
  "source_mentions": [
    {
      "raw_text": "exact visible source text",
      "role": "one allowed role",
      "family_hint": "one allowed family hint",
      "language_hint": "short optional language name",
      "filename_hint": "short optional safe filename"
    }
  ]
}
'''

_SOURCE_INTAKE_TIMEOUT_SECONDS = 300
_MAX_TASK_TEXT_CHARS = 12000
_MAX_VISIBLE_SPANS = 40
_MAX_SPAN_CHARS = 512
_MAX_HINT_CHARS = 80
_MAX_RETRY_FEEDBACK_ITEMS = 8
_MAX_RETRY_FEEDBACK_CHARS = 240
_HTTP_CANDIDATE_RE = re.compile(r"https?://[^\s<>\[\]]+")
_FENCED_CODE_BLOCK_RE = re.compile(
    r"```[A-Za-z0-9_+.#-]*[^\n]*\n.*?(?:\n```|```)",
    re.DOTALL,
)
_GITHUB_REPO_RE = re.compile(
    r"https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+(?:\.git)?)"
)
_OWNER_REPO_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
    r"(?![A-Za-z0-9_.-])"
)
_PACKAGE_SOURCE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:npm|pypi|cargo|crate|go|gomod):[A-Za-z0-9_.@/+-]+"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}\"'"
_SOURCE_MODES = {
    "single_primary",
    "inline_bundle",
    "mixed_primary_with_context",
    "compare_sources",
    "source_free",
    "unclear",
}
_SOURCE_ROLES = {
    "primary_code_source",
    "scope_modifier",
    "supporting_context",
    "reference_only",
    "unknown",
}
_SOURCE_FAMILIES = {
    "github_repository",
    "github_directory",
    "github_file",
    "raw_github_file",
    "github_issue",
    "github_pull",
    "github_discussion",
    "package_reference",
    "documentation_url",
    "web_page",
    "local_path",
    "inline_code",
    "inline_diff",
    "log_or_trace",
    "attachment",
    "unknown",
}


@dataclass(frozen=True)
class SourceMention:
    """One source-like span and the LLM's semantic role for it."""

    raw_text: str
    role: str
    family_hint: str
    language_hint: str = ""
    filename_hint: str = ""


@dataclass(frozen=True)
class SourceIntakeResult:
    """Normalized source-intake LLM output."""

    task_source_mode: str
    source_mentions: tuple[SourceMention, ...]


_source_intake_llm = LLInterface()
_source_intake_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.0,
    top_p=0.3,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=_SOURCE_INTAKE_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def run_source_intake(
    task_text: str,
    *,
    retry_feedback: list[str] | None = None,
) -> SourceIntakeResult:
    """Ask the PM route to extract source mentions from task text."""

    payload = {
        "task_text": _bounded_text(task_text, _MAX_TASK_TEXT_CHARS),
        "visible_source_spans": build_visible_source_spans(task_text),
        "retry_feedback": _bounded_retry_feedback(retry_feedback),
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    messages = [
        SystemMessage(content=SOURCE_INTAKE_PROMPT),
        HumanMessage(content=payload_text),
    ]
    response = await _source_intake_llm.ainvoke(
        messages,
        config=_source_intake_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    result = normalize_source_intake_output(parsed)
    return result


def build_visible_source_spans(task_text: str) -> list[str]:
    """Return visible source-like spans without assigning source semantics."""

    spans: list[str] = []
    for regex in (
        _FENCED_CODE_BLOCK_RE,
        _HTTP_CANDIDATE_RE,
        _GITHUB_REPO_RE,
        _OWNER_REPO_RE,
        _PACKAGE_SOURCE_RE,
        _WINDOWS_PATH_RE,
    ):
        for match in regex.finditer(task_text):
            span = _clean_visible_span(match.group(0))
            if span:
                _append_unique_visible_span(spans, span)
            if len(spans) >= _MAX_VISIBLE_SPANS:
                return spans

    return spans


def normalize_source_intake_output(parsed: object) -> SourceIntakeResult:
    """Normalize LLM JSON into the bounded source-intake contract."""

    if not isinstance(parsed, dict):
        result = SourceIntakeResult(
            task_source_mode="unclear",
            source_mentions=(),
        )
        return result

    raw_mode = parsed.get("task_source_mode")
    task_source_mode = raw_mode if raw_mode in _SOURCE_MODES else "unclear"
    source_mentions = _normalize_source_mentions(parsed.get("source_mentions"))
    result = SourceIntakeResult(
        task_source_mode=task_source_mode,
        source_mentions=tuple(source_mentions),
    )
    return result


def source_intake_result_to_dict(
    result: SourceIntakeResult,
) -> dict[str, Any]:
    """Serialize a normalized intake result for trace artifacts."""

    return {
        "task_source_mode": result.task_source_mode,
        "source_mentions": [
            {
                "raw_text": mention.raw_text,
                "role": mention.role,
                "family_hint": mention.family_hint,
                "language_hint": mention.language_hint,
                "filename_hint": mention.filename_hint,
            }
            for mention in result.source_mentions
        ],
    }


def _normalize_source_mentions(raw_mentions: object) -> list[SourceMention]:
    if not isinstance(raw_mentions, list):
        return []

    mentions: list[SourceMention] = []
    for raw_mention in raw_mentions:
        if not isinstance(raw_mention, dict):
            continue
        raw_text = _bounded_text(raw_mention.get("raw_text"), _MAX_SPAN_CHARS)
        if not raw_text:
            continue
        raw_role = raw_mention.get("role")
        role = raw_role if raw_role in _SOURCE_ROLES else "unknown"
        raw_family = raw_mention.get("family_hint")
        family_hint = raw_family if raw_family in _SOURCE_FAMILIES else "unknown"
        language_hint = _bounded_text(
            raw_mention.get("language_hint"),
            _MAX_HINT_CHARS,
        )
        filename_hint = _bounded_text(
            raw_mention.get("filename_hint"),
            _MAX_HINT_CHARS,
        )
        mention = SourceMention(
            raw_text=raw_text,
            role=role,
            family_hint=family_hint,
            language_hint=language_hint,
            filename_hint=filename_hint,
        )
        mentions.append(mention)
        if len(mentions) >= _MAX_VISIBLE_SPANS:
            break

    return mentions


def _bounded_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    bounded = value.strip()[:max_chars]
    return bounded


def _bounded_retry_feedback(
    retry_feedback: list[str] | None,
) -> list[str]:
    if retry_feedback is None:
        return []

    bounded_feedback: list[str] = []
    for item in retry_feedback[:_MAX_RETRY_FEEDBACK_ITEMS]:
        bounded_item = _bounded_text(item, _MAX_RETRY_FEEDBACK_CHARS)
        if bounded_item:
            bounded_feedback.append(bounded_item)

    return bounded_feedback


def _clean_visible_span(span: str) -> str:
    cleaned_span = span.strip().rstrip(_TRAILING_URL_PUNCTUATION)
    github_prefix = _github_repo_prefix(cleaned_span)
    if github_prefix and _github_prefix_exhausts_url_path(
        cleaned_span,
        github_prefix,
    ):
        return github_prefix
    return cleaned_span[:_MAX_SPAN_CHARS]


def _github_repo_prefix(span: str) -> str:
    match = _GITHUB_REPO_RE.match(span)
    if match is None:
        return ""
    return match.group(0).rstrip(".")


def _github_prefix_exhausts_url_path(span: str, prefix: str) -> bool:
    if len(span) == len(prefix):
        return True
    next_char = span[len(prefix)]
    return next_char != "/"


def _append_unique_visible_span(spans: list[str], span: str) -> None:
    if span in spans:
        return
    spans.append(span)
