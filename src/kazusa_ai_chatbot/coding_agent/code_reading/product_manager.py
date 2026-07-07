"""LLM-backed reading product-manager decisions."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PM_TARGET_INPUT_TOKEN_CAP,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    ReadingTaskScopeKind,
    ReadingPMDecision,
    ReadingPMInput,
    ReadingProgrammerTask,
    ReadingProgrammerReport,
    ReadingIntent,
)
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
ASSIGNMENT_SCOPE_KINDS: tuple[ReadingTaskScopeKind, ...] = (
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
MAX_PM_DECISION_ATTEMPTS = 2
MAX_PM_QUESTION_CHARS = 12000
MAX_PM_REPO_MAP_FILES = 60
MAX_PM_REPO_MAP_DIRECTORIES = 18
MAX_PM_SOURCE_CLASS_ITEMS = 18
MAX_PM_ITEM_SYMBOLS = 8
MAX_PM_ITEM_IMPORTS = 8
MAX_PM_ITEM_EXCERPT_CHARS = 160
MAX_PM_TOP_SYMBOLS = 28
MAX_PM_PREVIOUS_REPORTS = 6
MAX_PM_REPORT_FILES_READ = 6
MAX_PM_REPORT_FACTS = 6
MAX_PM_REPORT_OPEN_QUESTIONS = 4
MAX_PM_REPORT_SYMBOLS = 8
MAX_PM_REPORT_NEXT_HOPS = 4
MAX_PM_REPORT_EVIDENCE_REFS = 10
READING_PM_LLM_CALL_TIMEOUT_SECONDS = 300


READING_PM_PROMPT = '''\
You are the product manager for a read-only code-reading agent.
Your job is generic semantic decomposition only. You do not know concrete code
facts until programmer workers read bounded source evidence.

# Decision Rules
- Choose one generic intent from the allowed list.
- If the question is too broad for at most three bounded workers, return
  overloaded with missing_slots explaining the narrower scope needed.
- Judge broadness from the user's requested scope, not from how small or easy
  the current repository map looks. Requests to explain, inspect, summarize, or
  analyze everything, the whole repository, every file, all behavior, or the
  entire project are unbounded unless the user names a specific workflow,
  subsystem, API, symbol, state transition, or bounded evidence question.
- For unbounded whole-project requests, do not create representative
  architecture-overview assignments. Return overloaded or needs_user_input and
  ask for a narrower target instead.
- A high-level architecture or responsibility-boundary question is bounded
  when the user names the product, subsystem, workflow, interface, or symbol to
  explain. For that case, choose representative entry points, interfaces, and
  core orchestration slices rather than treating every repository file or every
  behavior as required evidence.
- If the request asks to rewrite, edit, modify, patch, implement, or otherwise
  change code, return needs_user_input with intent unsupported_request and no
  assignments.
- If the request is explicitly a read-only repository evidence survey for a
  future code-writing workflow, gather current-code evidence only. Do not reject
  merely because the requested outcome describes a possible future change.
- For future code-writing evidence surveys, do not require source evidence for
  files, tests, docs, or APIs that are meant to be created by the future patch
  and do not exist in the current repository. Use existing related source,
  interface, test, and documentation patterns as evidence; record missing
  future artifacts as limitations instead of blocking the survey.
- If the question needs clarification before bounded reading, return
  needs_user_input.
- If previous programmer reports already contain enough source-backed facts,
  return sufficient and no assignments.
- Treat no_evidence reports and unresolved open_questions as missing evidence,
  not sufficiency, unless other successful reports directly answer the same
  required slot from source-backed facts.
- If previous reports cover upstream modality or input handling plus a
  downstream component's generic plan/state consumption, treat that handoff as
  sufficient for downstream output construction unless the user explicitly asks
  for modality-specific downstream code.
- Do not repeat assignments that already returned supported facts unless the
  remaining missing slot is specific and materially different.
- Prefer concrete symbols and next-hop scopes discovered in successful reports
  over guessed method or class names. Do not invent likely private method names
  or framework internals; if a symbol is not visible in repository metadata or
  previous reports, use a search scope with generic source terms instead.
- When a previous report found relevant files but left an open question, prefer
  a follow-up file scope using exact previous files_read paths or exact
  candidate_next_hops. Do not switch to a guessed path.
- Do not use file or directory scope for a guessed path. File and directory
  values must be exact paths visible in repo_map_summary, previous files_read,
  or previous candidate_next_hops. If unsure, use search with short source
  terms instead of a path-like guess.
- If the payload review_mode is final_no_more_programmers, do not return
  need_programmers and do not include assignments. Returning need_programmers
  in this mode is invalid. Return sufficient when previous reports support a
  coherent source-backed answer chain with only minor limitations. Return
  needs_user_input when any user-requested chain segment remains represented
  only by no_evidence, blocked reports, or unresolved open_questions.
- Otherwise return need_programmers with one to three bounded assignments.
- There is a total budget of six programmer reports across all waves. Count
  previous_reports before assigning more work. Do not request more assignments
  than the remaining report budget; choose the most decisive missing slots
  first.
- Use repo_map_summary source classes, defined symbols, imported modules, and
  previous report candidate_next_hops as generic navigation hints when choosing
  bounded assignments.
- For pipeline_or_data_flow questions, identify each user-named event in the
  requested chain. Include bounded work for the first ingress, every named
  middle transition or dispatcher stage, internal representation or decision
  logic, and downstream output or response construction when those scopes are
  visible or searchable. Do not mark sufficient if a named middle transition
  remains ungrounded.
- Preserve evidence-owned component boundaries. If programmer reports show
  that a requested effect belongs to a different component, client, adapter,
  browser/desktop app, worker, or external service, treat that boundary as the
  source-backed answer for that segment. Record any absent same-component path
  as a limitation instead of a missing slot. Do not add server-side,
  client-side, worker-side, or adapter-side ownership constraints unless the
  user explicitly asks for that ownership or source evidence establishes it.
- For control_or_feedback_flow questions, include bounded work for both the
  control calculation and the caller or output application when those files or
  symbols are visible in the repository map.
- Each assignment must use exactly one scope kind: file, directory, symbol, or
  search.
- File and directory values must be repo-relative paths visible in the
  repository map.
- Symbol and search values must be short generic source terms from the user's
  question or repository map, not phrases chosen to force a known answer.
- When the user question names concrete symbols, events, methods, routes,
  services, fields, or constants, preserve those identifiers in the relevant
  assignment questions or values so the programmer can find exact source
  evidence.
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

READING_PM_ASSIGNMENT_CAP_RETRY_PROMPT = '''\
Your previous reading plan requested more than three programmer assignments.
Return strict JSON again with at most three assignments. Choose the most
decisive bounded source slices first, keep each assignment narrow, and do not
return overloaded merely because extra lower-priority slices could be useful.
'''

_reading_pm_llm = LLInterface()
_reading_pm_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=READING_PM_LLM_CALL_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


def decide_reading_work(
    pm_input: ReadingPMInput,
    *,
    trace: dict[str, object] | None = None,
) -> ReadingPMDecision:
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
    payload_text = json.dumps(payload, ensure_ascii=False)
    prompt_input_chars = len(READING_PM_PROMPT) + len(payload_text)
    context_budget = prompt_budget_metadata(
        system_prompt=READING_PM_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PM_TARGET_INPUT_TOKEN_CAP,
    )
    if context_budget["over_hard_cap"]:
        decision = _overloaded_decision(
            "Reading PM prompt exceeded the context budget."
        )
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_output=decision,
            attempts=[],
            prompt_input_chars=prompt_input_chars,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return decision
    attempts: list[dict[str, object]] = []
    decision: ReadingPMDecision | None = None
    raw_output = ""
    parsed: object = {}
    for attempt_index in range(MAX_PM_DECISION_ATTEMPTS):
        messages = [
            SystemMessage(content=READING_PM_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index:
            retry_message = HumanMessage(
                content=READING_PM_ASSIGNMENT_CAP_RETRY_PROMPT,
            )
            messages.append(retry_message)
        timed_out = False
        try:
            response = _reading_pm_llm.invoke(
                messages,
                config=_reading_pm_llm_config,
            )
            raw_output = response.content
            parsed = parse_llm_json_output(raw_output)
        except TimeoutError:
            timed_out = True
            raw_output = ""
            parsed = {}
        decision = normalize_pm_decision(parsed)
        if timed_out:
            decision = _needs_user_input_decision(
                "Reading PM LLM call timed out."
            )
        attempts.append({
            "attempt": attempt_index + 1,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "normalized_output": decision,
            "timed_out": timed_out,
        })
        if timed_out or not _should_retry_pm_decision(decision):
            break

    if decision is None:
        decision = _needs_user_input_decision(
            "Reading PM returned malformed output."
        )
    _fill_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed if isinstance(parsed, dict) else {},
        normalized_output=decision,
        attempts=attempts,
        prompt_input_chars=prompt_input_chars,
        context_budget=context_budget,
        blocked_before_invoke=False,
    )
    return decision


def normalize_pm_decision(parsed: object) -> ReadingPMDecision:
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

    decision: ReadingPMDecision = {
        "status": status,
        "intent": intent,
        "required_slots": required_slots,
        "assignments": assignments,
        "missing_slots": missing_slots,
    }
    return decision


def validate_programmer_assignment(
    assignment: ReadingProgrammerTask,
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
    programmer_reports: list[ReadingProgrammerReport],
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


def _pm_payload(pm_input: ReadingPMInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": _bounded_multiline_text(
            pm_input["question"],
            MAX_PM_QUESTION_CHARS,
        ),
        "repository_summary": pm_input["repository_summary"],
        "source_scope": pm_input["source_scope"],
        "repo_map_summary": _compact_repo_map_summary(
            pm_input["repo_map_summary"],
        ),
        "previous_reports": _compact_reports(pm_input["previous_reports"]),
    }
    review_mode = pm_input.get("review_mode")
    if review_mode:
        payload["review_mode"] = review_mode
    return payload


def _compact_repo_map_summary(
    repo_map_summary: dict[str, object],
) -> dict[str, object]:
    compact_summary: dict[str, object] = {
        "source_scope_kind": repo_map_summary.get("source_scope_kind"),
        "source_scope_path": repo_map_summary.get("source_scope_path"),
        "total_safe_files": repo_map_summary.get("total_safe_files"),
        "files": _string_list(
            repo_map_summary.get("files"),
            MAX_PM_REPO_MAP_FILES,
        ),
        "top_directories": _string_list(
            repo_map_summary.get("top_directories"),
            MAX_PM_REPO_MAP_DIRECTORIES,
        ),
        "source_classes": _compact_source_classes(
            repo_map_summary.get("source_classes"),
        ),
        "top_symbols": _compact_top_symbols(
            repo_map_summary.get("top_symbols"),
        ),
    }
    return compact_summary


def _compact_source_classes(value: object) -> dict[str, list[dict[str, object]]]:
    if not isinstance(value, dict):
        return {}

    compact_classes: dict[str, list[dict[str, object]]] = {}
    remaining_items = MAX_PM_SOURCE_CLASS_ITEMS
    for source_class, items in value.items():
        if remaining_items <= 0:
            break
        if not isinstance(source_class, str) or not isinstance(items, list):
            continue
        compact_items: list[dict[str, object]] = []
        for item in items:
            if remaining_items <= 0:
                break
            if not isinstance(item, dict):
                continue
            compact_items.append(_compact_source_class_item(item))
            remaining_items -= 1
        if compact_items:
            compact_classes[source_class] = compact_items
    return compact_classes


def _compact_source_class_item(item: dict[str, object]) -> dict[str, object]:
    compact_item: dict[str, object] = {
        "path": _bounded_text(item.get("path")),
        "source_class": _bounded_text(item.get("source_class")),
        "defined_symbols": _string_list(
            item.get("defined_symbols"),
            MAX_PM_ITEM_SYMBOLS,
        ),
        "imported_modules": _string_list(
            item.get("imported_modules"),
            MAX_PM_ITEM_IMPORTS,
        ),
        "summary_excerpt": _bounded_multiline_text(
            item.get("summary_excerpt"),
            MAX_PM_ITEM_EXCERPT_CHARS,
        ),
    }
    return compact_item


def _compact_top_symbols(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    compact_symbols: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        compact_symbols.append({
            "symbol": _bounded_text(item.get("symbol")),
            "path": _bounded_text(item.get("path")),
            "source_class": _bounded_text(item.get("source_class")),
        })
        if len(compact_symbols) >= MAX_PM_TOP_SYMBOLS:
            break
    return compact_symbols


def _compact_reports(
    reports: list[ReadingProgrammerReport],
) -> list[dict[str, object]]:
    compact_reports: list[dict[str, object]] = []
    for report in reports[:MAX_PM_PREVIOUS_REPORTS]:
        compact_report = {
            "assignment_id": report["assignment_id"],
            "status": report["status"],
            "files_read": _string_list(
                report["files_read"],
                MAX_PM_REPORT_FILES_READ,
            ),
            "facts": report["facts"][:MAX_PM_REPORT_FACTS],
            "open_questions": _string_list(
                report["open_questions"],
                MAX_PM_REPORT_OPEN_QUESTIONS,
            ),
            "discovered_symbols": _string_list(
                report["discovered_symbols"],
                MAX_PM_REPORT_SYMBOLS,
            ),
            "candidate_next_hops": _compact_next_hops(
                report["candidate_next_hops"],
            ),
            "evidence_refs": [
                _evidence_ref(row)
                for row in report["evidence"][:MAX_PM_REPORT_EVIDENCE_REFS]
            ],
        }
        compact_reports.append(compact_report)
    return compact_reports


def _compact_next_hops(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    next_hops: list[dict[str, object]] = []
    for item in value[:MAX_PM_REPORT_NEXT_HOPS]:
        if not isinstance(item, dict):
            continue
        next_hops.append({
            "path": _bounded_text(item.get("path")),
            "reason": _bounded_text(item.get("reason")),
        })
    return next_hops


def _assignments_from_parsed(parsed: object) -> list[ReadingProgrammerTask]:
    if not isinstance(parsed, list):
        return []

    assignments: list[ReadingProgrammerTask] = []
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
) -> ReadingProgrammerTask | None:
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

    assignment: ReadingProgrammerTask = {
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


def _bounded_multiline_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _needs_user_input_decision(reason: str) -> ReadingPMDecision:
    decision: ReadingPMDecision = {
        "status": "needs_user_input",
        "intent": "insufficient_evidence",
        "required_slots": [],
        "assignments": [],
        "missing_slots": [reason],
    }
    return decision


def _overloaded_decision(reason: str) -> ReadingPMDecision:
    decision: ReadingPMDecision = {
        "status": "overloaded",
        "intent": "insufficient_evidence",
        "required_slots": [],
        "assignments": [],
        "missing_slots": [reason],
    }
    return decision


def _should_retry_pm_decision(decision: ReadingPMDecision) -> bool:
    if decision["status"] != "overloaded":
        return False
    return "Too many programmer assignments were requested." in decision[
        "missing_slots"
    ]


def _evidence_ref(row: dict[str, object]) -> str:
    ref = f"{row['path']}:{row['line_start']}-{row['line_end']}"
    return ref


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: object,
    normalized_output: ReadingPMDecision,
    attempts: list[dict[str, object]] | None = None,
    prompt_input_chars: int,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _reading_pm_llm_config.route_name
    trace["model"] = _reading_pm_llm_config.model
    trace["prompt_input_chars"] = prompt_input_chars
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    if attempts is not None:
        trace["attempts"] = attempts
