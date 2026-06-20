"""Public data contracts for code reading."""

from typing import TypedDict

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
    ResultStatus,
)


class CodeEvidenceRow(TypedDict):
    """Bounded source evidence used for answer synthesis."""

    path: str
    line_start: int
    line_end: int
    symbol_or_topic: str
    excerpt: str
    reason: str


class CodeReadingRequest(TypedDict, total=False):
    """Request accepted by `code_reading.run` after Phase 0 succeeds."""

    question: str
    repository: CodeRepositoryRef
    source_scope: CodeSourceScope
    preferred_language: str
    max_answer_chars: int


class CodeReadingResult(TypedDict):
    """Read-only answer result from bounded local evidence."""

    status: ResultStatus
    answer_text: str
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
