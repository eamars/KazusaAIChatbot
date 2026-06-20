"""LLM-backed reading product-manager decisions."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    AssignmentScopeKind,
    PMDecision,
    PMInput,
    ProgrammerAssignment,
    ProgrammerReport,
    ReadingIntent,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_LLM_API_KEY,
    CODING_AGENT_LLM_BASE_URL,
    CODING_AGENT_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_LLM_MODEL,
    CODING_AGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

READING_INTENTS: tuple[ReadingIntent, ...] = (
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
)
PM_STATUSES = (
    "need_programmers",
    "sufficient",
    "needs_user_input",
    "overloaded",
)
ASSIGNMENT_SCOPE_KINDS: tuple[AssignmentScopeKind, ...] = (
    "file",
    "directory",
    "symbol",
    "search",
)
MAX_ASSIGNMENTS_PER_DECISION = 3
MAX_ASSIGNMENT_VALUES = 6
MAX_ASSIGNMENT_QUESTIONS = 3
MAX_REQUIRED_SLOTS = 8
MAX_TEXT_FIELD_CHARS = 300


READING_PM_PROMPT = '''\
You are the product manager for a read-only code-reading agent.
Your job is generic semantic decomposition only. You do not know concrete code
facts until programmer workers read bounded source evidence.

# Decision Rules
- Choose one generic intent from the allowed list.
- If the question is too broad for at most three bounded workers, return
  overloaded with missing_slots explaining the narrower scope needed.
- If the request asks to rewrite, edit, modify, patch, implement, or otherwise
  change code, return needs_user_input with intent unsupported_request and no
  assignments.
- If the question needs clarification before bounded reading, return
  needs_user_input.
- If previous programmer reports already contain enough source-backed facts,
  return sufficient and no assignments.
- Otherwise return need_programmers with one to three bounded assignments.
- For control_or_feedback_flow questions, include bounded work for both the
  control calculation and the caller or output application when those files or
  symbols are visible in the repository map.
- Each assignment must use exactly one scope kind: file, directory, symbol, or
  search.
- File and directory values must be repo-relative paths visible in the
  repository map.
- Symbol and search values must be short generic source terms from the user's
  question or repository map, not phrases chosen to force a known answer.
- Do not include concrete code facts unless they are visible in the current
  input, and do not include repository-specific shortcuts, domain-specific
  shortcuts, file-system roots, cache keys, API keys, or raw provider details.

# Allowed Intents
architecture_overview
pipeline_or_data_flow
control_or_feedback_flow
api_or_interface_contract
symbol_behavior
state_lifecycle
dependency_usage
configuration_behavior
error_handling
test_coverage
docs_to_code_consistency
insufficient_evidence
unsupported_request

# Output Format
Return strict JSON:
{
  "status": "need_programmers | sufficient | needs_user_input | overloaded",
  "intent": "one allowed intent",
  "required_slots": ["short evidence slot names"],
  "assignments": [
    {
      "assignment_id": "stable short id",
      "role": "generic worker role",
      "scope": {
        "kind": "file | directory | symbol | search",
        "values": ["repo-relative paths or source terms"]
      },
      "questions": ["bounded local source question"],
      "required_slots": ["slots this worker should fill"]
    }
  ],
  "missing_slots": ["slots still missing or clarification needed"]
}
'''

_reading_pm_llm = LLInterface()
_reading_pm_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_LLM",
    base_url=CODING_AGENT_LLM_BASE_URL,
    api_key=CODING_AGENT_LLM_API_KEY,
    model=CODING_AGENT_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_LLM_THINKING_ENABLED,
    ),
)


def decide_reading_work(
    pm_input: PMInput,
    *,
    trace: dict[str, object] | None = None,
) -> PMDecision:
    """Ask the reading PM to choose bounded programmer work.

    Args:
        pm_input: Compact user question, repository map, and previous report
            memory.
        trace: Optional internal diagnostic dictionary populated with safe route
            and model metadata plus raw and parsed model output.

    Returns:
        A schema-validated PM decision using only the simplified contract.
    """

    payload = _pm_payload(pm_input)
    response = _reading_pm_llm.invoke([
        SystemMessage(content=READING_PM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_reading_pm_llm_config)
    parsed = parse_llm_json_output(response.content)
    decision = normalize_pm_decision(parsed)
    _fill_trace(
        trace,
        raw_output=response.content,
        parsed_output=parsed,
        normalized_output=decision,
    )
    return decision


def normalize_pm_decision(parsed: object) -> PMDecision:
    """Normalize and validate PM JSON into the simplified decision contract."""

    if not isinstance(parsed, dict):
        decision = _needs_user_input_decision(
            "Reading PM returned malformed output."
        )
        return decision

    status = _bounded_text(parsed.get("status"))
    if status not in PM_STATUSES:
        status = "needs_user_input"

    intent = _bounded_text(parsed.get("intent"))
    if intent not in READING_INTENTS:
        intent = "insufficient_evidence"

    required_slots = _string_list(parsed.get("required_slots"), MAX_REQUIRED_SLOTS)
    missing_slots = _string_list(parsed.get("missing_slots"), MAX_REQUIRED_SLOTS)
    assignments = _assignments_from_parsed(parsed.get("assignments"))

    if status == "need_programmers" and not assignments:
        status = "needs_user_input"
        missing_slots.append("No bounded programmer assignments were provided.")
    if status != "need_programmers":
        assignments = []
    if len(assignments) > MAX_ASSIGNMENTS_PER_DECISION:
        status = "overloaded"
        assignments = []
        missing_slots.append("Too many programmer assignments were requested.")

    decision: PMDecision = {
        "status": status,
        "intent": intent,
        "required_slots": required_slots,
        "assignments": assignments,
        "missing_slots": missing_slots,
    }
    return decision


def validate_programmer_assignment(
    assignment: ProgrammerAssignment,
    source_scope: CodeSourceScope,
) -> None:
    """Reject PM assignment contracts that do not define a bounded read."""

    scope = assignment.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("Programmer assignment must include a scope.")

    scope_kind = scope.get("kind")
    if scope_kind not in ASSIGNMENT_SCOPE_KINDS:
        raise ValueError("Programmer assignment scope kind must be bounded.")

    values = scope.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("Programmer assignment must include scope values.")
    if not assignment.get("questions"):
        raise ValueError("Programmer assignment must include local questions.")
    if not assignment.get("required_slots"):
        raise ValueError("Programmer assignment must include required slots.")

    if scope_kind in ("file", "directory"):
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Programmer assignment has an invalid path.")
            _validate_assignment_path(value, source_scope)


def selected_evidence_from_reports(
    programmer_reports: list[ProgrammerReport],
) -> list[CodeEvidenceRow]:
    """Flatten report evidence while preserving report-memory ordering."""

    evidence: list[CodeEvidenceRow] = []
    seen: set[tuple[str, int, str]] = set()
    for report in programmer_reports:
        for row in report["evidence"]:
            key = (row["path"], row["line_start"], row["symbol_or_topic"])
            if key in seen:
                continue
            seen.add(key)
            evidence.append(row)
    return evidence


def _validate_assignment_path(
    value: str,
    source_scope: CodeSourceScope,
) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Programmer assignment path escapes repository.")

    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        return

    scope_path = PurePosixPath(scoped_path.replace("\\", "/"))
    if scope_path.is_absolute() or ".." in scope_path.parts:
        raise ValueError("Programmer assignment source scope is invalid.")
    if source_scope["kind"] == "file" and path != scope_path:
        raise ValueError("Programmer assignment must stay inside source scope.")
    if (
        source_scope["kind"] != "file"
        and path != scope_path
        and scope_path not in path.parents
    ):
        raise ValueError("Programmer assignment must stay inside source scope.")


def _pm_payload(pm_input: PMInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": pm_input["question"],
        "repository_summary": pm_input["repository_summary"],
        "source_scope": pm_input["source_scope"],
        "repo_map_summary": pm_input["repo_map_summary"],
        "previous_reports": _compact_reports(pm_input["previous_reports"]),
    }
    return payload


def _compact_reports(
    reports: list[ProgrammerReport],
) -> list[dict[str, object]]:
    compact_reports: list[dict[str, object]] = []
    for report in reports:
        compact_report = {
            "assignment_id": report["assignment_id"],
            "status": report["status"],
            "files_read": report["files_read"],
            "facts": report["facts"],
            "open_questions": report["open_questions"],
            "evidence_refs": [
                _evidence_ref(row)
                for row in report["evidence"]
            ],
        }
        compact_reports.append(compact_report)
    return compact_reports


def _assignments_from_parsed(parsed: object) -> list[ProgrammerAssignment]:
    if not isinstance(parsed, list):
        return []

    assignments: list[ProgrammerAssignment] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        assignment = _assignment_from_dict(item, index=index)
        if assignment is not None:
            assignments.append(assignment)
        if len(assignments) >= MAX_ASSIGNMENTS_PER_DECISION + 1:
            break
    return assignments


def _assignment_from_dict(
    item: dict[str, Any],
    *,
    index: int,
) -> ProgrammerAssignment | None:
    scope = item.get("scope")
    if not isinstance(scope, dict):
        return None

    scope_kind = _bounded_text(scope.get("kind"))
    if scope_kind not in ASSIGNMENT_SCOPE_KINDS:
        return None

    values = _string_list(scope.get("values"), MAX_ASSIGNMENT_VALUES)
    if not values:
        return None

    assignment_id = _bounded_text(item.get("assignment_id"))
    if not assignment_id:
        assignment_id = f"assignment-{index}"
    role = _bounded_text(item.get("role"))
    if not role:
        role = "source reader"
    questions = _string_list(item.get("questions"), MAX_ASSIGNMENT_QUESTIONS)
    if not questions:
        questions = ["Read bounded source evidence for the requested slot."]
    required_slots = _string_list(item.get("required_slots"), MAX_REQUIRED_SLOTS)

    assignment: ProgrammerAssignment = {
        "assignment_id": assignment_id,
        "role": role,
        "scope": {
            "kind": scope_kind,
            "values": values,
        },
        "questions": questions,
        "required_slots": required_slots,
    }
    return assignment


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    strings: list[str] = []
    for item in value:
        text = _bounded_text(item)
        if not text or text in strings:
            continue
        strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    if len(text) > MAX_TEXT_FIELD_CHARS:
        text = text[:MAX_TEXT_FIELD_CHARS].rstrip()
    return text


def _needs_user_input_decision(reason: str) -> PMDecision:
    decision: PMDecision = {
        "status": "needs_user_input",
        "intent": "insufficient_evidence",
        "required_slots": [],
        "assignments": [],
        "missing_slots": [reason],
    }
    return decision


def _evidence_ref(row: dict[str, object]) -> str:
    ref = f"{row['path']}:{row['line_start']}-{row['line_end']}"
    return ref


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: dict[str, object],
    normalized_output: PMDecision,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _reading_pm_llm_config.route_name
    trace["model"] = _reading_pm_llm_config.model
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
