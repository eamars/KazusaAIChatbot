"""Evidence-grounded answer synthesis for code reading."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    PMDecision,
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

_BACKTICKED_CODE_RE = re.compile(r"`([^`]+)`")
_PATH_TOKEN_RE = re.compile(
    r"\b[A-Za-z0-9_./-]+\.(?:py|md|toml|yaml|yml|json|txt|ini|cfg)\b"
)
_CODE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_COMMON_CODE_WORDS = {
    "API",
    "JSON",
    "LLM",
    "PM",
}
MAX_TEXT_FIELD_CHARS = 6000


SYNTHESIS_PROMPT = '''\
You synthesize the final answer for a read-only code-reading request.
Use only the PM decision, programmer reports, selected evidence rows,
limitations, and public repository metadata provided in the user payload.

# Synthesis Rules
- Do not invent concrete identifiers.
- Mention a concrete file, symbol, function, class, constant, route, field, or
  module only when it appears in selected evidence or public repository
  metadata.
- Prefer exact code identifiers from selected evidence when naming the key
  variables, functions, classes, fields, or output calls that answer the
  question.
- Do not name an algorithm, architecture pattern, framework, or design pattern
  unless that name appears in selected evidence. Describe the visible formula
  or call flow instead.
- Do not infer the term PID from words such as integral, derivative, error, or
  feedback unless the selected evidence itself contains PID.
- Preserve limitations and missing proof.
- Do not expose local roots, workspace roots, cache keys, API keys, raw provider
  objects, or full source files.
- Keep the answer in the user's requested language when that is clear.

# Output Format
Return strict JSON:
{
  "answer_text": "grounded answer text",
  "limitations": ["extra limitations or missing proof"]
}
'''

_synthesis_llm = LLInterface()
_synthesis_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_LLM",
    base_url=CODING_AGENT_LLM_BASE_URL,
    api_key=CODING_AGENT_LLM_API_KEY,
    model=CODING_AGENT_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=CODING_AGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_LLM_THINKING_ENABLED,
    ),
)


def synthesize_from_programmer_reports(
    *,
    question: str,
    pm_decision: PMDecision,
    programmer_reports: list[ProgrammerReport],
    evidence: list[CodeEvidenceRow],
    limitations: list[str],
    repository_summary: dict[str, object],
    preferred_language: str | None,
    max_answer_chars: int,
    trace: dict[str, object] | None = None,
) -> tuple[str, list[str]]:
    """Create the final PM answer from report memory and selected evidence.

    Args:
        question: User-visible question.
        pm_decision: PM decision that declared the evidence sufficient.
        programmer_reports: Structured report memory from bounded workers.
        evidence: Selected repo-relative evidence rows from those reports.
        limitations: Existing deterministic and PM limitations.
        repository_summary: Public-safe repository metadata.
        preferred_language: Optional caller language hint.
        max_answer_chars: Public answer cap.
        trace: Optional internal diagnostic dictionary populated with safe route
            and model metadata plus raw and parsed model output.

    Returns:
        The public answer text and combined limitations after grounding checks.
    """

    if not evidence:
        answer = "I found no bounded local source evidence for this question."
        result_limitations = [
            *limitations,
            "No selected evidence was available for synthesis.",
        ]
        return answer, result_limitations

    payload = _synthesis_payload(
        question=question,
        pm_decision=pm_decision,
        programmer_reports=programmer_reports,
        evidence=evidence,
        limitations=limitations,
        repository_summary=repository_summary,
        preferred_language=preferred_language,
        max_answer_chars=max_answer_chars,
    )
    response = _synthesis_llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_synthesis_llm_config)
    parsed = parse_llm_json_output(response.content)
    answer_text, parsed_limitations = normalize_synthesis_output(
        parsed,
        max_answer_chars=max_answer_chars,
    )
    combined_limitations = _dedupe_strings([*limitations, *parsed_limitations])
    grounding_violations = ungrounded_code_terms(
        answer_text,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    if grounding_violations:
        combined_limitations.append(
            "Synthesis included ungrounded code terms: "
            + ", ".join(grounding_violations[:8])
        )
        answer_text = (
            "I cannot provide a grounded answer from the selected evidence "
            "without inventing concrete code identifiers."
        )

    _fill_trace(
        trace,
        raw_output=response.content,
        parsed_output=parsed,
        normalized_answer=answer_text,
        limitations=combined_limitations,
    )
    return answer_text, combined_limitations


def normalize_synthesis_output(
    parsed: object,
    *,
    max_answer_chars: int,
) -> tuple[str, list[str]]:
    """Normalize synthesis JSON into answer text and limitations."""

    if not isinstance(parsed, dict):
        answer = "The synthesis model returned malformed output."
        limitations = ["Synthesis model returned malformed output."]
        return answer, limitations

    answer_text = _bounded_text(parsed.get("answer_text"), max_answer_chars)
    if not answer_text:
        answer_text = "The selected evidence did not support a final answer."
    limitations = _string_list(parsed.get("limitations"), 8)
    return answer_text, limitations


def ungrounded_code_terms(
    answer_text: str,
    *,
    evidence: list[CodeEvidenceRow],
    repository_summary: dict[str, object],
) -> list[str]:
    """Return concrete code terms in the answer that are not in evidence."""

    grounded_text = _grounded_text(
        evidence=evidence,
        repository_summary=repository_summary,
    )
    candidates = _code_term_candidates(answer_text)
    ungrounded: list[str] = []
    for candidate in candidates:
        if candidate in _COMMON_CODE_WORDS:
            continue
        if _candidate_is_grounded(candidate, grounded_text):
            continue
        ungrounded.append(candidate)
    return ungrounded


def _candidate_is_grounded(candidate: str, grounded_text: str) -> bool:
    lowered = candidate.casefold()
    if lowered in grounded_text:
        return True
    if "." not in candidate:
        return False
    parts = [part.casefold() for part in candidate.split(".") if part]
    if not parts:
        return False
    return all(part in grounded_text for part in parts)


def _synthesis_payload(
    *,
    question: str,
    pm_decision: PMDecision,
    programmer_reports: list[ProgrammerReport],
    evidence: list[CodeEvidenceRow],
    limitations: list[str],
    repository_summary: dict[str, object],
    preferred_language: str | None,
    max_answer_chars: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": question,
        "preferred_language": preferred_language,
        "max_answer_chars": max_answer_chars,
        "repository_summary": repository_summary,
        "pm_decision": {
            "status": pm_decision["status"],
            "intent": pm_decision["intent"],
            "required_slots": pm_decision["required_slots"],
            "missing_slots": pm_decision["missing_slots"],
        },
        "programmer_reports": _compact_reports(programmer_reports),
        "selected_evidence": [
            {
                "evidence_ref": _evidence_ref(row),
                "path": row["path"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "symbol_or_topic": row["symbol_or_topic"],
                "excerpt": row["excerpt"],
                "reason": row["reason"],
            }
            for row in evidence
        ],
        "limitations": limitations,
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


def _code_term_candidates(answer_text: str) -> list[str]:
    terms: list[str] = []
    for match in _BACKTICKED_CODE_RE.finditer(answer_text):
        _append_unique(terms, match.group(1).strip())
    for match in _PATH_TOKEN_RE.finditer(answer_text):
        _append_unique(terms, match.group(0).strip())
    for match in _CODE_IDENTIFIER_RE.finditer(answer_text):
        token = match.group(0)
        if "_" in token or any(char.isupper() for char in token[1:]):
            _append_unique(terms, token)
    return terms


def _grounded_text(
    *,
    evidence: list[CodeEvidenceRow],
    repository_summary: dict[str, object],
) -> str:
    parts: list[str] = []
    for row in evidence:
        parts.extend([
            row["path"],
            row["symbol_or_topic"],
            row["excerpt"],
            row["reason"],
        ])
    for value in repository_summary.values():
        if isinstance(value, str):
            parts.append(value)
    text = "\n".join(parts).casefold()
    return text


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    strings: list[str] = []
    for item in value:
        text = _bounded_text(item, MAX_TEXT_FIELD_CHARS)
        if not text or text in strings:
            continue
        strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped


def _bounded_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
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
    parsed_output: dict[str, Any],
    normalized_answer: str,
    limitations: list[str],
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _synthesis_llm_config.route_name
    trace["model"] = _synthesis_llm_config.model
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = {
        "answer_text": normalized_answer,
        "limitations": limitations,
    }
