"""Programmer worker for bounded code-reading assignments."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.evidence import (
    EvidenceBundle,
    collect_assignment_evidence,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    ReadingProgrammerTask,
    ReadingProgrammerFact,
    ReadingProgrammerReport,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PROGRAMMER_LLM_MODEL,
    CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

MAX_FACTS_PER_REPORT = 8
MAX_OPEN_QUESTIONS = 5
MAX_TEXT_FIELD_CHARS = 500
MAX_DISCOVERED_SYMBOLS = 12
MAX_CANDIDATE_NEXT_HOPS = 6
PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS = 300
_PYTHON_DEFINITION_RE = re.compile(
    r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
)
_PYTHON_FROM_IMPORT_RE = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+",
)
_PYTHON_IMPORT_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)",
)


PROGRAMMER_REPORT_PROMPT = '''\
You are a programmer worker for a read-only code-reading agent.
You receive one bounded assignment and selected source excerpts. Report only
facts supported by those excerpts. Do not infer from outside knowledge.

# Report Rules
- Use only the provided evidence rows.
- Every fact must cite one or more evidence_refs exactly as provided.
- If the evidence is empty, return status no_evidence with no facts.
- If the assignment cannot be answered from the evidence, return blocked or
  no_evidence and explain the open question.
- When evidence rows contain branch conditions, filters, return values,
  queueing calls, dispatch calls, or other decision/call-site logic, summarize
  that logic and its outcome. Do not stop at definitions or settings
  declarations when decision rows are present.
- Prefer implementation decision and call-site evidence over docs, tests, or
  settings declarations for behavior questions.
- If evidence supports a generic handoff, plan, state, or consumer contract
  relevant to the assignment but does not mention the original input modality
  or product wording, report the supported generic fact and keep the missing
  modality-specific proof as an open question.
- Do not include local roots, workspace roots, cache keys, API keys, raw
  provider details, or full files.

# Output Format
Return strict JSON:
{
  "status": "succeeded | blocked | no_evidence",
  "facts": [
    {
      "kind": "short fact type",
      "summary": "source-backed fact summary",
      "evidence_refs": ["exact provided evidence ref"]
    }
  ],
  "open_questions": ["missing source-backed question"]
}
'''

_programmer_llm = LLInterface()
_programmer_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PROGRAMMER_LLM",
    base_url=CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    api_key=CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    model=CODING_AGENT_PROGRAMMER_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=PROGRAMMER_LLM_CALL_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
    ),
)


def run_programmer_assignment(
    repository: CodeRepositoryRef,
    assignment: ReadingProgrammerTask,
    source_scope: CodeSourceScope,
    *,
    max_files: int,
    max_excerpt_chars: int,
    trace: dict[str, object] | None = None,
) -> ReadingProgrammerReport:
    """Run one bounded reading assignment and return report memory.

    Args:
        repository: Successful Phase 0 repository source contract.
        assignment: Validated programmer assignment.
        source_scope: Enclosing Phase 0 source scope.
        max_files: Supervisor-owned file cap for this worker.
        max_excerpt_chars: Supervisor-owned excerpt cap for this worker.
        trace: Optional internal diagnostic dictionary populated with safe route
            and model metadata plus raw and parsed model output.

    Returns:
        A structured report containing only repo-relative files and evidence.
    """

    repo_root = Path(repository["local_root"])
    bundle = collect_assignment_evidence(
        repo_root=repo_root,
        source_scope=source_scope,
        assignment=assignment,
        max_files=max_files,
        max_excerpt_chars=max_excerpt_chars,
    )
    payload = _programmer_payload(
        assignment=assignment,
        bundle=bundle,
    )
    raw_output = ""
    parsed: object = {}
    timed_out = False
    try:
        response = _programmer_llm.invoke([
            SystemMessage(content=PROGRAMMER_REPORT_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ], config=_programmer_llm_config)
        raw_output = response.content
        parsed = parse_llm_json_output(raw_output)
    except TimeoutError:
        timed_out = True
    report = normalize_programmer_report(
        parsed,
        assignment=assignment,
        bundle=bundle,
    )
    if timed_out:
        report = _blocked_report(
            assignment=assignment,
            bundle=bundle,
            reason="Programmer LLM call timed out.",
        )
    _fill_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=report,
        bundle=bundle,
        timed_out=timed_out,
    )
    return report


def normalize_programmer_report(
    parsed: object,
    *,
    assignment: ReadingProgrammerTask,
    bundle: EvidenceBundle,
) -> ReadingProgrammerReport:
    """Normalize programmer JSON into the simplified report contract."""

    evidence_refs = {_evidence_ref(row) for row in bundle.rows}
    if not isinstance(parsed, dict):
        report = _blocked_report(
            assignment=assignment,
            bundle=bundle,
            reason="Programmer returned malformed output.",
        )
        return report

    status = _bounded_text(parsed.get("status"))
    if status not in ("succeeded", "blocked", "no_evidence"):
        status = "blocked"
    facts = _facts_from_parsed(
        parsed.get("facts"),
        evidence_refs=evidence_refs,
    )
    open_questions = _string_list(parsed.get("open_questions"), MAX_OPEN_QUESTIONS)

    if not bundle.rows:
        status = "no_evidence"
        facts = []
        if not open_questions:
            open_questions = ["No bounded evidence matched the assignment."]
    elif facts and status != "succeeded":
        status = "succeeded"
    elif status == "no_evidence":
        status = "blocked"
        if not open_questions:
            open_questions = ["Evidence was found, but no supported facts were returned."]
    elif status == "succeeded" and not facts:
        status = "blocked"
        if not open_questions:
            open_questions = ["Evidence was found, but no supported facts were returned."]

    discovered_symbols = _discovered_symbols(bundle)
    report: ReadingProgrammerReport = {
        "assignment_id": assignment["assignment_id"],
        "status": status,
        "files_read": bundle.files_read,
        "facts": facts,
        "evidence": bundle.rows,
        "open_questions": open_questions,
        "discovered_symbols": discovered_symbols,
        "candidate_next_hops": _candidate_next_hops(
            bundle=bundle,
            assignment=assignment,
            discovered_symbols=discovered_symbols,
        ),
    }
    return report


def _programmer_payload(
    *,
    assignment: ReadingProgrammerTask,
    bundle: EvidenceBundle,
) -> dict[str, object]:
    payload = {
        "assignment": assignment,
        "evidence_rows": [
            {
                "evidence_ref": _evidence_ref(row),
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "symbol_or_topic": row["symbol_or_topic"],
                "excerpt": row["excerpt"],
                "reason": row["reason"],
            }
            for row in bundle.rows
        ],
        "limitations": bundle.limitations,
    }
    return payload


def _facts_from_parsed(
    parsed: object,
    *,
    evidence_refs: set[str],
) -> list[ReadingProgrammerFact]:
    if not isinstance(parsed, list):
        return []

    facts: list[ReadingProgrammerFact] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        refs = [
            ref for ref in _string_list(item.get("evidence_refs"), 6)
            if ref in evidence_refs
        ]
        if not refs:
            continue
        summary = _bounded_text(item.get("summary"))
        if not summary:
            continue
        fact: ReadingProgrammerFact = {
            "kind": _bounded_text(item.get("kind")) or "source_fact",
            "summary": summary,
            "evidence_refs": refs,
        }
        facts.append(fact)
        if len(facts) >= MAX_FACTS_PER_REPORT:
            break
    return facts


def _blocked_report(
    *,
    assignment: ReadingProgrammerTask,
    bundle: EvidenceBundle,
    reason: str,
) -> ReadingProgrammerReport:
    discovered_symbols = _discovered_symbols(bundle)
    report: ReadingProgrammerReport = {
        "assignment_id": assignment["assignment_id"],
        "status": "blocked",
        "files_read": bundle.files_read,
        "facts": [],
        "evidence": bundle.rows,
        "open_questions": [reason],
        "discovered_symbols": discovered_symbols,
        "candidate_next_hops": _candidate_next_hops(
            bundle=bundle,
            assignment=assignment,
            discovered_symbols=discovered_symbols,
        ),
    }
    return report


def _discovered_symbols(bundle: EvidenceBundle) -> list[str]:
    symbols: list[str] = []
    for row in bundle.rows:
        for line in row["excerpt"].splitlines():
            match = _PYTHON_DEFINITION_RE.match(line)
            if match is None:
                continue
            _append_unique(symbols, match.group(1))
            if len(symbols) >= MAX_DISCOVERED_SYMBOLS:
                return symbols
    return symbols


def _candidate_next_hops(
    *,
    bundle: EvidenceBundle,
    assignment: ReadingProgrammerTask,
    discovered_symbols: list[str],
) -> list[dict[str, object]]:
    hops: list[dict[str, object]] = []
    if discovered_symbols:
        _append_next_hop(
            hops,
            reason="Bounded evidence defines related symbols.",
            kind="symbol",
            values=discovered_symbols[:4],
        )

    imported_modules = _imported_modules_from_rows(bundle.rows)
    if imported_modules:
        _append_next_hop(
            hops,
            reason="Bounded evidence imports related modules.",
            kind="search",
            values=imported_modules[:4],
        )

    for path_text in bundle.files_read:
        directory = PurePosixPath(path_text).parent.as_posix()
        if directory == ".":
            continue
        if _assignment_already_targets_directory(assignment, directory):
            continue
        _append_next_hop(
            hops,
            reason="Adjacent files in the same directory may hold callers or collaborators.",
            kind="directory",
            values=[directory],
        )
        if len(hops) >= MAX_CANDIDATE_NEXT_HOPS:
            break
    return hops


def _imported_modules_from_rows(rows: list[CodeEvidenceRow]) -> list[str]:
    modules: list[str] = []
    for row in rows:
        for line in row["excerpt"].splitlines():
            from_match = _PYTHON_FROM_IMPORT_RE.match(line)
            if from_match is not None:
                _append_unique(modules, _bounded_search_term(from_match.group(1)))
                continue

            import_match = _PYTHON_IMPORT_RE.match(line)
            if import_match is None:
                continue
            for module_name in import_match.group(1).split(","):
                clean_module = module_name.strip().split(" as ", maxsplit=1)[0]
                _append_unique(modules, _bounded_search_term(clean_module))
        if len(modules) >= MAX_DISCOVERED_SYMBOLS:
            break
    return modules[:MAX_DISCOVERED_SYMBOLS]


def _append_next_hop(
    hops: list[dict[str, object]],
    *,
    reason: str,
    kind: str,
    values: list[str],
) -> None:
    if len(hops) >= MAX_CANDIDATE_NEXT_HOPS:
        return
    clean_values = [
        _bounded_search_term(value)
        for value in values
        if _bounded_search_term(value)
    ]
    if not clean_values:
        return
    hop = {
        "reason": reason,
        "scope": {
            "kind": kind,
            "values": clean_values,
        },
    }
    if hop in hops:
        return
    hops.append(hop)


def _assignment_already_targets_directory(
    assignment: ReadingProgrammerTask,
    directory: str,
) -> bool:
    scope = assignment["scope"]
    if scope["kind"] != "directory":
        return False
    return directory in scope["values"]


def _bounded_search_term(value: str) -> str:
    text = " ".join(value.strip().split())
    if len(text) > 120:
        text = text[:120].rstrip()
    return text


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


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _evidence_ref(row: CodeEvidenceRow) -> str:
    ref = f"{row['path']}:{row['line_start']}-{row['line_end']}"
    return ref


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: object,
    normalized_output: ReadingProgrammerReport,
    bundle: EvidenceBundle,
    timed_out: bool,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _programmer_llm_config.route_name
    trace["model"] = _programmer_llm_config.model
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    trace["evidence_count"] = len(bundle.rows)
    trace["files_read"] = bundle.files_read
    trace["timed_out"] = timed_out
