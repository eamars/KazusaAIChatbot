"""Public and internal data contracts for code reading."""

from typing import Literal, NotRequired, TypedDict

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
    """Request accepted by `code_reading.run` after source fetching succeeds."""

    question: str
    repository: CodeRepositoryRef
    source_scope: CodeSourceScope
    preferred_language: str
    max_answer_chars: int
    read_only_context_handoff: bool


class CodeReadingResult(TypedDict):
    """Read-only answer result from bounded local evidence."""

    status: ResultStatus
    answer_text: str
    evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]


ReadingPMStatus = Literal[
    "need_programmers",
    "sufficient",
    "needs_user_input",
    "overloaded",
]
ReadingIntent = Literal[
    "architecture_overview",
    "pipeline_or_data_flow",
    "control_or_feedback_flow",
    "api_or_interface_contract",
    "symbol_behavior",
    "state_lifecycle",
    "dependency_usage",
    "configuration_behavior",
    "error_handling",
    "test_coverage",
    "docs_to_code_consistency",
    "insufficient_evidence",
    "unsupported_request",
]
ReadingTaskScopeKind = Literal["file", "directory", "symbol", "search"]
ReadingProgrammerStatus = Literal["succeeded", "blocked", "no_evidence"]


class ReadingTaskScope(TypedDict):
    """Bounded repository scope selected for one programmer worker."""

    kind: ReadingTaskScopeKind
    values: list[str]


class ReadingProgrammerTask(TypedDict):
    """One bounded programmer mission selected by the reading PM."""

    assignment_id: str
    role: str
    scope: ReadingTaskScope
    questions: list[str]
    required_slots: list[str]


class ReadingProgrammerFact(TypedDict):
    """A source-backed fact extracted by a programmer worker."""

    kind: str
    summary: str
    evidence_refs: list[str]


class ReadingCandidateNextHop(TypedDict):
    """Generic follow-up scope discovered from bounded evidence."""

    reason: str
    scope: ReadingTaskScope


class ReadingProgrammerReport(TypedDict):
    """Compressed memory artifact returned by one programmer worker."""

    assignment_id: str
    status: ReadingProgrammerStatus
    files_read: list[str]
    facts: list[ReadingProgrammerFact]
    evidence: list[CodeEvidenceRow]
    open_questions: list[str]
    discovered_symbols: list[str]
    candidate_next_hops: list[ReadingCandidateNextHop]


class ReadingPMInput(TypedDict):
    """Compact model-facing input for the reading product manager."""

    question: str
    repository_summary: dict[str, object]
    source_scope: dict[str, object]
    repo_map_summary: dict[str, object]
    previous_reports: list[ReadingProgrammerReport]
    review_mode: NotRequired[str]


class ReadingPMDecision(TypedDict):
    """The only decision shape returned by the reading product manager."""

    status: ReadingPMStatus
    intent: ReadingIntent
    required_slots: list[str]
    assignments: list[ReadingProgrammerTask]
    missing_slots: list[str]


class ReadingSupervisorState(TypedDict):
    """Bounded supervisor state for one code-reading request."""

    request: CodeReadingRequest
    repository_summary: dict[str, object]
    source_scope: CodeSourceScope
    repo_map_summary: dict[str, object]
    pm_decisions: list[ReadingPMDecision]
    programmer_reports: list[ReadingProgrammerReport]
    selected_evidence: list[CodeEvidenceRow]
    limitations: list[str]
    trace_summary: list[str]
    wave_count: int
