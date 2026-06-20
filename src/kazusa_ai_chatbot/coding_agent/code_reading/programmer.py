"""Programmer worker for bounded code-reading assignments."""

from __future__ import annotations

import json
from pathlib import Path
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
    ProgrammerAssignment,
    ProgrammerFact,
    ProgrammerReport,
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

MAX_FACTS_PER_REPORT = 8
MAX_OPEN_QUESTIONS = 5
MAX_TEXT_FIELD_CHARS = 500


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


def run_programmer_assignment(
    repository: CodeRepositoryRef,
    assignment: ProgrammerAssignment,
    source_scope: CodeSourceScope,
    *,
    max_files: int,
    max_excerpt_chars: int,
    trace: dict[str, object] | None = None,
) -> ProgrammerReport:
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
    response = _programmer_llm.invoke([
        SystemMessage(content=PROGRAMMER_REPORT_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_programmer_llm_config)
    parsed = parse_llm_json_output(response.content)
    report = normalize_programmer_report(
        parsed,
        assignment=assignment,
        bundle=bundle,
    )
    _fill_trace(
        trace,
        raw_output=response.content,
        parsed_output=parsed,
        normalized_output=report,
        bundle=bundle,
    )
    return report


def normalize_programmer_report(
    parsed: object,
    *,
    assignment: ProgrammerAssignment,
    bundle: EvidenceBundle,
) -> ProgrammerReport:
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
    elif status == "succeeded" and not facts:
        status = "blocked"
        if not open_questions:
            open_questions = ["Evidence was found, but no supported facts were returned."]

    report: ProgrammerReport = {
        "assignment_id": assignment["assignment_id"],
        "status": status,
        "files_read": bundle.files_read,
        "facts": facts,
        "evidence": bundle.rows,
        "open_questions": open_questions,
    }
    return report


def _programmer_payload(
    *,
    assignment: ProgrammerAssignment,
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
) -> list[ProgrammerFact]:
    if not isinstance(parsed, list):
        return []

    facts: list[ProgrammerFact] = []
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
        fact: ProgrammerFact = {
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
    assignment: ProgrammerAssignment,
    bundle: EvidenceBundle,
    reason: str,
) -> ProgrammerReport:
    report: ProgrammerReport = {
        "assignment_id": assignment["assignment_id"],
        "status": "blocked",
        "files_read": bundle.files_read,
        "facts": [],
        "evidence": bundle.rows,
        "open_questions": [reason],
    }
    return report


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


def _evidence_ref(row: CodeEvidenceRow) -> str:
    ref = f"{row['path']}:{row['line_start']}-{row['line_end']}"
    return ref


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: dict[str, Any],
    normalized_output: ProgrammerReport,
    bundle: EvidenceBundle,
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
