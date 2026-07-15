"""Evidence-grounded answer synthesis for code reading."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    SYNTHESIS_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_reading.models import (
    CodeEvidenceRow,
    ReadingPMDecision,
    ReadingProgrammerReport,
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

_BACKTICKED_CODE_RE = re.compile(r"`([^`]+)`")
_PATH_TOKEN_RE = re.compile(
    r"\b[A-Za-z0-9_./-]+\.(?:py|md|toml|yaml|yml|json|txt|ini|cfg)\b"
)
_CODE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CJK_TEXT_RE = re.compile(r"[\u3400-\u9fff]")
_COMMON_CODE_WORDS = {
    "AGENTS",
    "API",
    "JSON",
    "LLM",
    "PM",
    "README",
}
MAX_TEXT_FIELD_CHARS = 6000
MAX_SYNTHESIS_QUESTION_CHARS = 8000
MAX_SYNTHESIS_REPORTS = 6
MAX_SYNTHESIS_FILES_READ = 8
MAX_SYNTHESIS_FACTS = 8
MAX_SYNTHESIS_OPEN_QUESTIONS = 6
MAX_SYNTHESIS_DISCOVERED_SYMBOLS = 8
MAX_SYNTHESIS_NEXT_HOPS = 6
MAX_SYNTHESIS_EVIDENCE_REFS = 12
MAX_SYNTHESIS_EVIDENCE_ROWS = 18
MAX_SYNTHESIS_EVIDENCE_EXCERPT_CHARS = 450
MAX_LIMITATIONS_IN_ANSWER = 3
MAX_EVIDENCE_ANCHORS = 24
SYNTHESIS_LLM_CALL_TIMEOUT_SECONDS = 300
UNGROUNDED_CODE_TERMS_LIMITATION = "Synthesis included ungrounded code terms."
PUBLIC_SYNTHESIS_DIAGNOSTIC_LIMITATIONS = {
    "Reading synthesis LLM call timed out.",
    "Reading synthesis prompt exceeded the context budget.",
    "Synthesis model omitted answer_text.",
    "Synthesis model returned answer outside JSON schema.",
    "Synthesis model returned malformed output.",
}


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
- Preserve limitations and missing proof.
- Structure pipeline or data-flow answers around the user-requested chain
  segments. Cover each segment only with source-backed facts from reports or
  selected evidence.
- If reports list a user-requested segment as open, no_evidence, or missing,
  do not imply that segment is answered. State the missing segment as a
  limitation instead.
- Do not expose local roots, workspace roots, cache keys, API keys, raw provider
  objects, or full source files.
- Keep the answer in the user's requested language when that is clear.

# Output Format
Return only strict JSON. Do not write prose before or after the JSON object.
Always put the final answer in `answer_text`. If the evidence is partial, put
the best grounded partial answer in `answer_text` and list the missing proof in
`limitations`.
{
  "answer_text": "grounded answer text",
  "limitations": ["extra limitations or missing proof"]
}
'''

SYNTHESIS_OUTPUT_FORMAT = '''\
{
  "answer_text": "grounded answer text",
  "limitations": ["extra limitations or missing proof"]
}
'''

_synthesis_llm = LLInterface()
_synthesis_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=SYNTHESIS_LLM_CALL_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


def synthesize_from_programmer_reports(
    *,
    question: str,
    pm_decision: ReadingPMDecision,
    programmer_reports: list[ReadingProgrammerReport],
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
    payload_text = json.dumps(payload, ensure_ascii=False)
    prompt_input_chars = len(SYNTHESIS_PROMPT) + len(payload_text)
    context_budget = prompt_budget_metadata(
        system_prompt=SYNTHESIS_PROMPT,
        payload_text=payload_text,
        target_input_tokens=SYNTHESIS_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        answer_text = (
            "I cannot synthesize a grounded code-reading answer because the "
            "selected evidence exceeded the context budget."
        )
        combined_limitations = _dedupe_strings([
            *limitations,
            "Reading synthesis prompt exceeded the context budget.",
        ])
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_answer=answer_text,
            limitations=combined_limitations,
            prompt_input_chars=prompt_input_chars,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return answer_text, combined_limitations
    raw_output = ""
    parsed: object = {}
    timed_out = False
    try:
        response = _synthesis_llm.invoke([
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=payload_text),
        ], config=_synthesis_llm_config)
        raw_output = response.content
        parsed = parse_llm_json_output(
            raw_output,
            expected_output_format=SYNTHESIS_OUTPUT_FORMAT,
        )
    except TimeoutError:
        timed_out = True
    answer_text, parsed_limitations = normalize_synthesis_output(
        parsed,
        raw_output=raw_output,
        max_answer_chars=max_answer_chars,
    )
    if timed_out:
        answer_text = (
            "I cannot synthesize a grounded code-reading answer because the "
            "reading synthesis LLM call timed out."
        )
        parsed_limitations = ["Reading synthesis LLM call timed out."]
    combined_limitations = _dedupe_strings([*limitations, *parsed_limitations])
    public_limitations = _public_grounded_limitations(
        combined_limitations,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    answer_text, repaired_grounding = _repair_ungrounded_answer_terms(
        answer_text,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    if repaired_grounding:
        public_limitations = _dedupe_strings([
            *public_limitations,
            UNGROUNDED_CODE_TERMS_LIMITATION,
        ])
    answer_text = _answer_with_required_limitations(
        answer_text,
        public_limitations,
        max_answer_chars=max_answer_chars,
    )
    answer_text, repaired_limitations = _repair_ungrounded_answer_terms(
        answer_text,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    if repaired_limitations:
        public_limitations = _dedupe_strings([
            *public_limitations,
            UNGROUNDED_CODE_TERMS_LIMITATION,
        ])
    answer_text = _append_missing_evidence_anchors(
        answer_text,
        evidence=evidence,
        max_answer_chars=max_answer_chars,
    )

    _fill_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_answer=answer_text,
        limitations=public_limitations,
        prompt_input_chars=prompt_input_chars,
        context_budget=context_budget,
        blocked_before_invoke=False,
        timed_out=timed_out,
    )
    return answer_text, public_limitations


def normalize_synthesis_output(
    parsed: object,
    *,
    raw_output: str = "",
    max_answer_chars: int,
) -> tuple[str, list[str]]:
    """Normalize synthesis JSON into answer text and limitations."""

    if not isinstance(parsed, dict):
        fallback_answer = _raw_answer_fallback(raw_output, max_answer_chars)
        if fallback_answer:
            limitations = ["Synthesis model returned answer outside JSON schema."]
            return fallback_answer, limitations
        answer = "The synthesis model returned malformed output."
        limitations = ["Synthesis model returned malformed output."]
        return answer, limitations

    answer_text = _bounded_text(parsed.get("answer_text"), max_answer_chars)
    if not answer_text:
        answer_text = _raw_answer_fallback(raw_output, max_answer_chars)
        if answer_text:
            limitations = _string_list(parsed.get("limitations"), 8)
            limitations.append("Synthesis model omitted answer_text.")
            return answer_text, limitations
        answer_text = "The selected evidence did not support a final answer."
    limitations = _string_list(parsed.get("limitations"), 8)
    return answer_text, limitations


def ungrounded_code_terms(
    answer_text: str,
    *,
    question: str = "",
    evidence: list[CodeEvidenceRow],
    repository_summary: dict[str, object],
) -> list[str]:
    """Return concrete code terms in the answer that are not in evidence.

    The user question can mention an identifier that the selected evidence
    never proves. Treat selected evidence and public repository metadata as
    grounding sources; the question text is task context, not proof.
    """

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
    pm_decision: ReadingPMDecision,
    programmer_reports: list[ReadingProgrammerReport],
    evidence: list[CodeEvidenceRow],
    limitations: list[str],
    repository_summary: dict[str, object],
    preferred_language: str | None,
    max_answer_chars: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": _bounded_text(question, MAX_SYNTHESIS_QUESTION_CHARS),
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
        "selected_evidence": _compact_evidence_rows(evidence),
        "limitations": limitations,
    }
    return payload


def _compact_reports(
    reports: list[ReadingProgrammerReport],
) -> list[dict[str, object]]:
    compact_reports: list[dict[str, object]] = []
    for report in reports[:MAX_SYNTHESIS_REPORTS]:
        compact_report = {
            "assignment_id": report["assignment_id"],
            "status": report["status"],
            "files_read": _string_list(
                report["files_read"],
                MAX_SYNTHESIS_FILES_READ,
            ),
            "facts": _compact_facts(report["facts"]),
            "open_questions": _string_list(
                report["open_questions"],
                MAX_SYNTHESIS_OPEN_QUESTIONS,
            ),
            "discovered_symbols": _string_list(
                report["discovered_symbols"],
                MAX_SYNTHESIS_DISCOVERED_SYMBOLS,
            ),
            "candidate_next_hops": _compact_next_hops(
                report["candidate_next_hops"],
            ),
            "evidence_refs": [
                _evidence_ref(row)
                for row in report["evidence"][:MAX_SYNTHESIS_EVIDENCE_REFS]
            ],
        }
        compact_reports.append(compact_report)
    return compact_reports


def _compact_facts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    facts: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = {
            "kind": _bounded_text(item.get("kind"), MAX_TEXT_FIELD_CHARS),
            "summary": _bounded_text(item.get("summary"), MAX_TEXT_FIELD_CHARS),
            "evidence_refs": _string_list(
                item.get("evidence_refs"),
                MAX_SYNTHESIS_EVIDENCE_REFS,
            ),
        }
        facts.append(fact)
        if len(facts) >= MAX_SYNTHESIS_FACTS:
            break
    return facts


def _compact_next_hops(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    hops: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        hop = {
            "reason": _bounded_text(item.get("reason"), MAX_TEXT_FIELD_CHARS),
            "scope": item.get("scope"),
        }
        hops.append(hop)
        if len(hops) >= MAX_SYNTHESIS_NEXT_HOPS:
            break
    return hops


def _compact_evidence_rows(
    evidence: list[CodeEvidenceRow],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in evidence[:MAX_SYNTHESIS_EVIDENCE_ROWS]:
        compact_row = {
            "evidence_ref": _evidence_ref(row),
            "path": row["path"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "symbol_or_topic": row["symbol_or_topic"],
            "excerpt": _bounded_text(
                row["excerpt"],
                MAX_SYNTHESIS_EVIDENCE_EXCERPT_CHARS,
            ),
            "reason": row["reason"],
        }
        rows.append(compact_row)
    return rows


def _code_term_candidates(answer_text: str) -> list[str]:
    terms: list[str] = []
    for match in _BACKTICKED_CODE_RE.finditer(answer_text):
        candidate = _clean_code_candidate(match.group(1))
        if _is_backticked_code_candidate(candidate):
            _append_unique(terms, candidate)
    for match in _PATH_TOKEN_RE.finditer(answer_text):
        _append_unique(terms, _clean_code_candidate(match.group(0)))
    for match in _CODE_IDENTIFIER_RE.finditer(answer_text):
        token = match.group(0)
        if "_" in token or any(char.isupper() for char in token[1:]):
            _append_unique(terms, _clean_code_candidate(token))
    return terms


def _is_backticked_code_candidate(value: str) -> bool:
    """Return whether backticked text is a single code-like token."""

    if not value:
        return False
    if _CJK_TEXT_RE.search(value):
        return False
    if any(char.isspace() for char in value):
        return False
    if any(char in value for char in "()[]{}<>"):
        return False
    if "->" in value:
        return False
    if _PATH_TOKEN_RE.fullmatch(value):
        return True
    if "/" in value and "." not in value and not value.startswith("/"):
        return False
    if "_" in value or "." in value or "/" in value:
        return True
    return any(char.isupper() for char in value[1:])


def _remove_ungrounded_code_terms(
    answer_text: str,
    grounding_violations: list[str],
) -> str:
    repaired = answer_text
    for violation in grounding_violations:
        repaired = re.sub(
            rf"\s*\([^)]*`{re.escape(violation)}`[^)]*\)",
            "",
            repaired,
        )
        repaired = repaired.replace(f"`{violation}`", "")
        repaired = re.sub(
            rf"(?<![\w.]){re.escape(violation)}(?![\w.])",
            "",
            repaired,
        )
    repaired = re.sub(r"\s*\(\s*\)", "", repaired)
    repaired = re.sub(r"\s*\uff08\s*\uff09", "", repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    return repaired.strip()


def _clean_code_candidate(value: str) -> str:
    text = value.strip()
    text = text.strip("'\"")
    return text


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


def _public_grounded_limitations(
    limitations: list[str],
    *,
    evidence: list[CodeEvidenceRow],
    repository_summary: dict[str, object],
) -> list[str]:
    """Return limitations safe to repeat in public answer text."""

    public_limitations: list[str] = []
    for limitation in limitations:
        if limitation in PUBLIC_SYNTHESIS_DIAGNOSTIC_LIMITATIONS:
            public_limitations.append(limitation)
            continue
        grounding_violations = ungrounded_code_terms(
            limitation,
            evidence=evidence,
            repository_summary=repository_summary,
        )
        if grounding_violations:
            continue
        public_limitations.append(limitation)
    return public_limitations


def _repair_ungrounded_answer_terms(
    answer_text: str,
    *,
    evidence: list[CodeEvidenceRow],
    repository_summary: dict[str, object],
) -> tuple[str, bool]:
    """Remove ungrounded code identifiers from public answer text."""

    grounding_violations = ungrounded_code_terms(
        answer_text,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    if not grounding_violations:
        return answer_text, False

    repaired_answer = _remove_ungrounded_code_terms(
        answer_text,
        grounding_violations,
    )
    remaining_violations = ungrounded_code_terms(
        repaired_answer,
        evidence=evidence,
        repository_summary=repository_summary,
    )
    if repaired_answer and not remaining_violations:
        return repaired_answer, True

    fallback_answer = (
        "I cannot provide a grounded answer from the selected evidence "
        "without inventing concrete code identifiers."
    )
    return fallback_answer, True


def _append_missing_evidence_anchors(
    answer_text: str,
    *,
    evidence: list[CodeEvidenceRow],
    max_answer_chars: int,
) -> str:
    """Preserve bounded source identifiers omitted by synthesis prose.

    The model owns the explanation, while this projection keeps selected
    evidence identifiers visible when the model summarizes around them. Every
    appended value is taken directly from a selected evidence row and the
    result remains inside the caller's public answer limit.
    """

    anchors = _evidence_anchors(evidence)
    missing = [
        anchor
        for anchor in anchors
        if anchor.casefold() not in answer_text.casefold()
    ]
    if not missing:
        return answer_text

    prefix = "Evidence anchors: "
    selected: list[str] = []
    for anchor in missing:
        candidate = prefix + ", ".join([
            *_format_evidence_anchors(selected),
            _format_evidence_anchor(anchor),
        ])
        if len(answer_text) + 2 + len(candidate) > max_answer_chars:
            break
        selected.append(anchor)
    if not selected:
        return answer_text
    return f"{answer_text}\n\n{prefix}{', '.join(_format_evidence_anchors(selected))}"


def _evidence_anchors(evidence: list[CodeEvidenceRow]) -> list[str]:
    """Collect bounded code and markup anchors directly from evidence rows."""

    anchors: list[str] = []
    for row in evidence:
        for value in (row["path"], row["symbol_or_topic"]):
            if _is_evidence_anchor(value):
                _append_unique(anchors, value)
        excerpt = row["excerpt"]
        for match in _PATH_TOKEN_RE.finditer(excerpt):
            _append_unique(anchors, _clean_code_candidate(match.group(0)))
        for match in _CODE_IDENTIFIER_RE.finditer(excerpt):
            token = _clean_code_candidate(match.group(0))
            if "_" in token or any(char.isupper() for char in token[1:]):
                _append_unique(anchors, token)
        for match in re.finditer(r"<[A-Za-z][^>]*>", excerpt):
            _append_unique(anchors, match.group(0))
        if len(anchors) >= MAX_EVIDENCE_ANCHORS:
            break
    return anchors[:MAX_EVIDENCE_ANCHORS]


def _is_evidence_anchor(value: str) -> bool:
    """Return whether a row field is useful as a public source anchor."""

    return bool(
        value
        and (
            _PATH_TOKEN_RE.fullmatch(value) is not None
            or "_" in value
            or any(char.isupper() for char in value[1:])
        )
    )


def _format_evidence_anchor(value: str) -> str:
    """Format a source anchor without changing its grounded value."""

    if value.startswith("<") and value.endswith(">"):
        return value
    return f"`{value}`"


def _format_evidence_anchors(values: list[str]) -> list[str]:
    """Format a list of evidence anchors for the public answer."""

    return [_format_evidence_anchor(value) for value in values]


def _answer_with_required_limitations(
    answer_text: str,
    limitations: list[str],
    *,
    max_answer_chars: int,
) -> str:
    """Append required limitations that the model left out of answer text."""

    if not limitations:
        return answer_text

    missing_limitations = _missing_visible_limitations(
        answer_text,
        limitations[:MAX_LIMITATIONS_IN_ANSWER],
    )
    if not missing_limitations:
        return answer_text

    limitation_text = "Limitations: " + "; ".join(missing_limitations)
    if answer_text:
        updated_answer = f"{answer_text}\n\n{limitation_text}"
    else:
        updated_answer = limitation_text
    if len(updated_answer) > max_answer_chars:
        updated_answer = updated_answer[:max_answer_chars].rstrip()
    return updated_answer


def _missing_visible_limitations(
    answer_text: str,
    limitations: list[str],
) -> list[str]:
    answer_key = _visibility_key(answer_text)
    missing_limitations: list[str] = []
    for limitation in limitations:
        normalized_limitation = _trim_sentence_punctuation(limitation)
        limitation_key = _visibility_key(normalized_limitation)
        if limitation_key and limitation_key not in answer_key:
            missing_limitations.append(normalized_limitation)
    return missing_limitations


def _trim_sentence_punctuation(value: str) -> str:
    trimmed_value = value.strip().rstrip(".;")
    return trimmed_value


def _visibility_key(value: str) -> str:
    key = re.sub(r"\s+", " ", value.casefold()).strip().rstrip(".;")
    return key


def _bounded_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _raw_answer_fallback(raw_output: str, max_answer_chars: int) -> str:
    """Keep a prose synthesis answer when only JSON wrapping failed."""

    answer_text = _bounded_text(raw_output, max_answer_chars)
    stripped_answer = answer_text.lstrip()
    if not stripped_answer:
        return ""
    if stripped_answer.startswith(("{", "[", "```")):
        return ""
    return answer_text


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
    normalized_answer: str,
    limitations: list[str],
    prompt_input_chars: int,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    timed_out: bool = False,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _synthesis_llm_config.route_name
    trace["model"] = _synthesis_llm_config.model
    trace["prompt_input_chars"] = prompt_input_chars
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["timed_out"] = timed_out
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = {
        "answer_text": normalized_answer,
        "limitations": limitations,
    }
