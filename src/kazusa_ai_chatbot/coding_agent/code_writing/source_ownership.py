"""LLM-backed source ownership decisions for existing repository writing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_reading.models import CodeEvidenceRow
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    SourceOwnerCandidate,
    SourceOwnershipDecision,
    SourceOwnershipDecisionStatus,
    SourceOwnershipResolution,
    SourceOwnershipResolutionStatus,
    WritingFileDemand,
    WritingMode,
    WritingReadingEvidenceRequest,
)
from kazusa_ai_chatbot.coding_agent.context_budget import (
    PM_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
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

SOURCE_OWNERSHIP_STATUSES: tuple[SourceOwnershipDecisionStatus, ...] = (
    "accepted",
    "need_reading",
    "needs_pm_repair",
)
SOURCE_OWNERSHIP_RESOLUTION_STATUSES: tuple[
    SourceOwnershipResolutionStatus,
    ...
] = (
    "accepted",
    "need_reading",
    "repair_required",
)
SOURCE_OWNERSHIP_FILE_KINDS = {"existing", "docs", "config"}
MAX_DEMANDS = 5
MAX_CANDIDATES = 14
MAX_LIST_ITEMS = 8
MAX_TEXT_CHARS = 700
MAX_QUESTION_CHARS = 5000
MAX_EVIDENCE_ROWS_PER_CANDIDATE = 4
MAX_EVIDENCE_EXCERPT_CHARS = 500
SOURCE_OWNERSHIP_LLM_CALL_TIMEOUT_SECONDS = 300


SOURCE_OWNERSHIP_PROMPT = '''\
You are the Source Ownership PM for a code-writing agent.
You receive semantic file demands and source candidate files gathered by the
read-only code reader. Decide which candidate file owns each demand, or request
more source reading.

# Rules
- Choose an owned_path only from candidate_paths in the input.
- Do not create paths, rename paths, write code, write imports, or plan patch
  operations.
- Select a file as owner only when the candidate evidence shows it owns the
  requested component, endpoint, model, documentation page, test area, or
  configuration surface.
- A file that only imports, calls, or uses a component is supporting context,
  not the owner of that component.
- Put supporting files in read_only_paths only when they are candidate paths.
- If no candidate proves ownership, return need_reading with the missing source
  facts to collect.
- If the demand is too vague for ownership even with the candidates, return
  needs_pm_repair and explain what semantic demand must be fixed.
- Keep reasons short and based on the visible candidate evidence.

# Output Format
Return strict JSON:
{
  "decisions": [
    {
      "demand_id": "same demand_id from input",
      "status": "accepted | need_reading | needs_pm_repair",
      "owned_path": "candidate path when accepted, otherwise empty",
      "read_only_paths": ["supporting candidate path"],
      "reason": "short source-grounded reason",
      "evidence_refs": ["candidate evidence ref"],
      "required_slots": ["source facts still needed when not accepted"]
    }
  ]
}
'''

SOURCE_OWNERSHIP_RETRY_PROMPT = '''\
Your previous source ownership response was empty or not valid JSON.
Return one strict JSON object only, matching the required output format from
the system instructions. Do not include markdown, commentary, or code fences.
'''

_source_ownership_llm = LLInterface()
_source_ownership_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.source_ownership",
    route_name="CODING_AGENT_PM_LLM",
    base_url=CODING_AGENT_PM_LLM_BASE_URL,
    api_key=CODING_AGENT_PM_LLM_API_KEY,
    model=CODING_AGENT_PM_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PM_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def decide_source_ownership(
    *,
    question: str,
    mode: WritingMode,
    file_demands: list[WritingFileDemand],
    owner_candidates: list[SourceOwnerCandidate],
    reading_evidence: list[CodeEvidenceRow],
    trace: dict[str, object] | None = None,
) -> SourceOwnershipResolution:
    """Choose existing-source owners from bounded source candidates.

    Args:
        question: User-visible writing request.
        mode: Current writing mode selected by the supervisor.
        file_demands: PM-authored semantic file demands.
        owner_candidates: Source-reader candidate files allowed for selection.
        reading_evidence: Source evidence rows used only for bounded candidate
            context.
        trace: Optional debug trace sink.

    Returns:
        Accepted owner decisions, or a source-reading/PM-repair request before
        file mechanics run.
    """

    required_demands = _demands_requiring_ownership(mode, file_demands)
    if not required_demands:
        resolution = _accepted_resolution([])
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_output=resolution,
            payload={},
            context_budget={},
            blocked_before_invoke=True,
        )
        return resolution

    payload = _source_ownership_payload(
        question=question,
        demands=required_demands,
        owner_candidates=owner_candidates,
        reading_evidence=reading_evidence,
    )
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    context_budget = prompt_budget_metadata(
        system_prompt=SOURCE_OWNERSHIP_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PM_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        resolution = _need_reading_resolution(
            demands=required_demands,
            reason="Source ownership prompt exceeded the context budget.",
        )
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_output=resolution,
            payload=payload,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return resolution

    raw_output, parsed_output, attempts = await _invoke_source_ownership_json(
        payload_text,
    )
    resolution = normalize_source_ownership_resolution(
        parsed_output,
        required_demands=required_demands,
        owner_candidates=owner_candidates,
    )
    _fill_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed_output,
        normalized_output=resolution,
        payload=payload,
        context_budget=context_budget,
        blocked_before_invoke=False,
        attempts=attempts,
    )
    return resolution


async def _invoke_source_ownership_json(
    payload_text: str,
) -> tuple[str, object, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    raw_output = ""
    parsed_output: object = {}
    for attempt_index in range(2):
        messages = [
            SystemMessage(content=SOURCE_OWNERSHIP_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index > 0:
            messages.append(HumanMessage(content=SOURCE_OWNERSHIP_RETRY_PROMPT))
        timed_out = False
        try:
            response = await asyncio.wait_for(
                _source_ownership_llm.ainvoke(
                    messages,
                    config=_source_ownership_llm_config,
                ),
                timeout=SOURCE_OWNERSHIP_LLM_CALL_TIMEOUT_SECONDS,
            )
            raw_output = response.content
            parsed_output = parse_llm_json_output(raw_output)
        except asyncio.TimeoutError:
            timed_out = True
            raw_output = ""
            parsed_output = {}
        attempts.append({
            "attempt": attempt_index + 1,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "timed_out": timed_out,
        })
        if parsed_output or timed_out:
            break
    result = raw_output, parsed_output, attempts
    return result


def normalize_source_ownership_resolution(
    parsed_output: object,
    *,
    required_demands: list[WritingFileDemand],
    owner_candidates: list[SourceOwnerCandidate],
) -> SourceOwnershipResolution:
    """Normalize LLM JSON into source ownership decisions."""

    demand_ids = {
        demand["demand_id"]
        for demand in required_demands
        if isinstance(demand.get("demand_id"), str) and demand["demand_id"]
    }
    candidate_paths = {
        candidate["path"]
        for candidate in owner_candidates
        if isinstance(candidate.get("path"), str) and candidate["path"]
    }
    if not isinstance(parsed_output, dict):
        resolution = _need_reading_resolution(
            demands=required_demands,
            reason="Source ownership PM returned malformed output.",
        )
        return resolution

    parsed_decisions = parsed_output.get("decisions")
    if not isinstance(parsed_decisions, list):
        resolution = _need_reading_resolution(
            demands=required_demands,
            reason="Source ownership PM did not return decisions.",
        )
        return resolution

    decisions: list[SourceOwnershipDecision] = []
    seen_demand_ids: set[str] = set()
    for item in parsed_decisions:
        if not isinstance(item, dict):
            continue
        demand_id = _bounded_text(item.get("demand_id"), MAX_TEXT_CHARS)
        if demand_id not in demand_ids or demand_id in seen_demand_ids:
            continue
        seen_demand_ids.add(demand_id)
        decision = _normalized_decision(
            item,
            demand_id=demand_id,
            candidate_paths=candidate_paths,
        )
        decisions.append(decision)

    for demand in required_demands:
        demand_id = _bounded_text(demand.get("demand_id"), MAX_TEXT_CHARS)
        if not demand_id or demand_id in seen_demand_ids:
            continue
        decision = _missing_decision(demand_id)
        decisions.append(decision)

    resolution = _resolution_from_decisions(decisions)
    return resolution


def _normalized_decision(
    item: dict[str, object],
    *,
    demand_id: str,
    candidate_paths: set[str],
) -> SourceOwnershipDecision:
    status = _bounded_text(item.get("status"), MAX_TEXT_CHARS)
    if status not in SOURCE_OWNERSHIP_STATUSES:
        status = "need_reading"

    owned_path = _bounded_text(item.get("owned_path"), MAX_TEXT_CHARS)
    if owned_path not in candidate_paths:
        owned_path = ""
        if status == "accepted":
            status = "need_reading"

    read_only_paths = [
        path
        for path in _string_list(item.get("read_only_paths"), MAX_LIST_ITEMS)
        if path in candidate_paths and path != owned_path
    ]
    reason = _bounded_text(item.get("reason"), MAX_TEXT_CHARS)
    if not reason:
        reason = "Source ownership was not proven."
    evidence_refs = _string_list(item.get("evidence_refs"), MAX_LIST_ITEMS)
    required_slots = _string_list(item.get("required_slots"), MAX_LIST_ITEMS)
    if status == "accepted" and not owned_path:
        status = "need_reading"
    if status != "accepted" and not required_slots:
        required_slots = [f"Current source owner for {demand_id}."]

    decision: SourceOwnershipDecision = {
        "demand_id": demand_id,
        "status": status,
        "owned_path": owned_path,
        "read_only_paths": read_only_paths,
        "reason": reason,
        "evidence_refs": evidence_refs,
        "required_slots": required_slots,
    }
    return decision


def _resolution_from_decisions(
    decisions: list[SourceOwnershipDecision],
) -> SourceOwnershipResolution:
    reading_requests: list[WritingReadingEvidenceRequest] = []
    errors: list[str] = []
    status: SourceOwnershipResolutionStatus = "accepted"
    for decision in decisions:
        if decision["status"] == "accepted":
            continue
        if decision["status"] == "need_reading":
            status = "need_reading"
            reading_requests.append(_reading_request_from_decision(decision))
        if decision["status"] == "needs_pm_repair" and status != "need_reading":
            status = "repair_required"
        errors.append(
            f"{decision['demand_id']}: {decision['reason']}"
        )

    resolution: SourceOwnershipResolution = {
        "status": status,
        "decisions": decisions,
        "errors": errors[:MAX_LIST_ITEMS],
        "repair_feedback": _repair_feedback(errors),
        "reading_requests": reading_requests[:MAX_LIST_ITEMS],
    }
    return resolution


def _reading_request_from_decision(
    decision: SourceOwnershipDecision,
) -> WritingReadingEvidenceRequest:
    required_slots = decision["required_slots"][:MAX_LIST_ITEMS]
    if not required_slots:
        required_slots = [f"Current source owner for {decision['demand_id']}."]
    request: WritingReadingEvidenceRequest = {
        "request_id": f"owner-{decision['demand_id']}",
        "task": "Find the current source file that owns the requested work.",
        "reason": decision["reason"],
        "required_slots": required_slots,
    }
    return request


def _need_reading_resolution(
    *,
    demands: list[WritingFileDemand],
    reason: str,
) -> SourceOwnershipResolution:
    decisions = []
    for demand in demands:
        demand_id = _bounded_text(demand.get("demand_id"), MAX_TEXT_CHARS)
        if not demand_id:
            continue
        decision: SourceOwnershipDecision = {
            "demand_id": demand_id,
            "status": "need_reading",
            "owned_path": "",
            "read_only_paths": [],
            "reason": reason,
            "evidence_refs": [],
            "required_slots": [f"Current source owner for {demand_id}."],
        }
        decisions.append(decision)
    resolution = _resolution_from_decisions(decisions)
    return resolution


def _accepted_resolution(
    decisions: list[SourceOwnershipDecision],
) -> SourceOwnershipResolution:
    resolution: SourceOwnershipResolution = {
        "status": "accepted",
        "decisions": decisions,
        "errors": [],
        "repair_feedback": [],
        "reading_requests": [],
    }
    return resolution


def _missing_decision(demand_id: str) -> SourceOwnershipDecision:
    decision: SourceOwnershipDecision = {
        "demand_id": demand_id,
        "status": "need_reading",
        "owned_path": "",
        "read_only_paths": [],
        "reason": "Source ownership PM did not decide this demand.",
        "evidence_refs": [],
        "required_slots": [f"Current source owner for {demand_id}."],
    }
    return decision


def _source_ownership_payload(
    *,
    question: str,
    demands: list[WritingFileDemand],
    owner_candidates: list[SourceOwnerCandidate],
    reading_evidence: list[CodeEvidenceRow],
) -> dict[str, object]:
    evidence_by_path = _evidence_by_path(reading_evidence)
    payload = {
        "question": _bounded_multiline_text(question, MAX_QUESTION_CHARS),
        "file_demands": [
            _compact_demand(demand)
            for demand in demands[:MAX_DEMANDS]
        ],
        "candidate_paths": [
            _compact_candidate(candidate, evidence_by_path)
            for candidate in owner_candidates[:MAX_CANDIDATES]
        ],
    }
    return payload


def _compact_demand(demand: WritingFileDemand) -> dict[str, object]:
    compact = {
        "demand_id": _bounded_text(demand.get("demand_id"), MAX_TEXT_CHARS),
        "role": _bounded_text(demand.get("role"), MAX_TEXT_CHARS),
        "purpose": _bounded_text(demand.get("purpose"), MAX_TEXT_CHARS),
        "file_kind": _bounded_text(demand.get("file_kind"), MAX_TEXT_CHARS),
        "interface_contract": _bounded_object(demand.get("interface_contract")),
        "integration_contract": _bounded_object(
            demand.get("integration_contract"),
        ),
        "change_goal": _bounded_text(demand.get("change_goal"), MAX_TEXT_CHARS),
        "work_instructions": _string_list(
            demand.get("work_instructions"),
            MAX_LIST_ITEMS,
        ),
        "required_slots": _string_list(demand.get("required_slots"), MAX_LIST_ITEMS),
        "validation_expectations": _string_list(
            demand.get("validation_expectations"),
            MAX_LIST_ITEMS,
        ),
    }
    return compact


def _compact_candidate(
    candidate: SourceOwnerCandidate,
    evidence_by_path: dict[str, list[CodeEvidenceRow]],
) -> dict[str, object]:
    path = candidate["path"]
    compact = {
        "path": path,
        "role": candidate["role"],
        "symbols": candidate["symbols"][:MAX_LIST_ITEMS],
        "feature_markers": candidate["feature_markers"][:MAX_LIST_ITEMS],
        "reasons": candidate["reasons"][:MAX_LIST_ITEMS],
        "evidence_refs": candidate["evidence_refs"][:MAX_LIST_ITEMS],
        "evidence": [
            _compact_evidence_row(row)
            for row in evidence_by_path.get(path, [])[:MAX_EVIDENCE_ROWS_PER_CANDIDATE]
        ],
    }
    return compact


def _compact_evidence_row(row: CodeEvidenceRow) -> dict[str, object]:
    compact = {
        "ref": f"{row['path']}:{row['line_start']}-{row['line_end']}",
        "topic": _bounded_text(row.get("symbol_or_topic"), MAX_TEXT_CHARS),
        "excerpt": _bounded_multiline_text(
            row.get("excerpt"),
            MAX_EVIDENCE_EXCERPT_CHARS,
        ),
        "reason": _bounded_text(row.get("reason"), MAX_TEXT_CHARS),
    }
    return compact


def _evidence_by_path(
    reading_evidence: list[CodeEvidenceRow],
) -> dict[str, list[CodeEvidenceRow]]:
    evidence_by_path: dict[str, list[CodeEvidenceRow]] = {}
    for row in reading_evidence:
        path = row["path"]
        evidence_by_path.setdefault(path, []).append(row)
    return evidence_by_path


def _demands_requiring_ownership(
    mode: WritingMode,
    file_demands: list[WritingFileDemand],
) -> list[WritingFileDemand]:
    if mode != "edit_existing_repository":
        return []
    demands = [
        demand
        for demand in file_demands
        if _bounded_text(demand.get("file_kind"), MAX_TEXT_CHARS)
        in SOURCE_OWNERSHIP_FILE_KINDS
    ]
    return demands[:MAX_DEMANDS]


def _bounded_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    bounded: dict[str, object] = {}
    for raw_key, raw_item in value.items():
        if not isinstance(raw_key, str):
            continue
        key = _bounded_text(raw_key, MAX_TEXT_CHARS)
        if isinstance(raw_item, list):
            bounded[key] = _string_list(raw_item, MAX_LIST_ITEMS)
        elif isinstance(raw_item, dict):
            bounded[key] = _bounded_object(raw_item)
        else:
            bounded[key] = _bounded_text(raw_item, MAX_TEXT_CHARS)
    return bounded


def _string_list(value: object, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = _bounded_text(item, MAX_TEXT_CHARS)
        if not text or text in texts:
            continue
        texts.append(text)
        if len(texts) >= max_items:
            break
    return texts


def _bounded_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.strip().split())
    return text[:max_chars].rstrip()


def _bounded_multiline_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text[:max_chars].rstrip()


def _repair_feedback(errors: list[str]) -> list[str]:
    if not errors:
        return []
    feedback = [
        "Correct source ownership before File PM dispatch.",
        *errors[:MAX_LIST_ITEMS],
    ]
    return feedback


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: object,
    normalized_output: SourceOwnershipResolution,
    payload: dict[str, object],
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    if trace is None:
        return
    trace.update({
        "effective_route": "CODING_AGENT_PM_LLM",
        "model": CODING_AGENT_PM_LLM_MODEL,
        "thinking_enabled": CODING_AGENT_PM_LLM_THINKING_ENABLED,
        "payload": payload,
        "context_budget": context_budget,
        "blocked_before_invoke": blocked_before_invoke,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "normalized_output": normalized_output,
    })
    if attempts is not None:
        trace["attempts"] = attempts


__all__ = [
    "decide_source_ownership",
    "normalize_source_ownership_resolution",
]
