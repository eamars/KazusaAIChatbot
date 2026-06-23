"""LLM-backed product-manager decisions for code writing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PM_TARGET_INPUT_TOKEN_CAP,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    WritingExternalEvidenceRequest,
    WritingFileDemand,
    WritingFileKind,
    WritingMode,
    WritingPMDecision,
    WritingPMInput,
    WritingPMStatus,
    WritingReadingEvidenceRequest,
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

PM_STATUSES: tuple[WritingPMStatus, ...] = (
    "need_reading",
    "need_file_pms",
    "ready_to_write",
    "needs_user_input",
    "overloaded",
    "rejected",
)
WRITING_MODES: tuple[WritingMode, ...] = (
    "edit_existing_repository",
    "create_new_project",
)
MAX_ASSIGNMENTS_PER_DECISION = 5
MAX_ASSIGNMENT_VALUES = 8
MAX_WORK_INSTRUCTIONS = 4
MAX_REQUIRED_SLOTS = 8
MAX_READING_EVIDENCE_REQUESTS = 3
MAX_EXTERNAL_EVIDENCE_REQUESTS = 3
FILE_KINDS: tuple[WritingFileKind, ...] = (
    "existing",
    "new",
    "test",
    "docs",
    "config",
    "support",
)
MAX_PM_OWNER_CANDIDATES = 10
MAX_PM_OWNER_EVIDENCE_REFS = 12
MAX_PM_OWNER_EXCEPTION_TYPES = 8
MAX_PM_OWNER_FEATURE_MARKERS = 12
MAX_PM_OWNER_REASONS = 4
MAX_PM_OWNER_SYMBOLS = 8
MAX_PM_READING_REPORTS = 4
MAX_PM_READING_FACTS = 8
MAX_PM_QUESTION_CHARS = 6000
MAX_TEXT_FIELD_CHARS = 500
PM_LLM_CALL_TIMEOUT_SECONDS = 300


WRITING_PM_PROMPT = '''\
You are the top-level writing product manager for a code-writing agent that
proposes patches. You split the request into small semantic file demands and
decide when source reading, public evidence, or user input is needed. You do
not choose filesystem paths, line ranges, imports, functions, code, tests, or
patches.

# Inputs You Use
- question: the user request.
- mode: edit_existing_repository or create_new_project.
- repository_summary: public repository metadata for existing-source work.
- reading_reports: scope-free reading status rows. Source paths, source owner
  candidates, line refs, excerpts, and current-file context are reserved for
  File Agent and File PM.
- source_reading_state: whether source reading is absent, sufficient, or
  exhausted.
- previous_writing_reports: programmer reports from this writing request.
- patch_check_result: validation result for a previous patch attempt, if any.
- file_resolution_feedback: file-planning feedback to revise semantic work
  orders, if any.
- file_plan_feedback: resolved file-plan feedback to revise semantic work
  orders, if any.
- external_evidence: public evidence gathered for this request.

# Decision Procedure
1. Start from mode.
   - Return need_file_pms whenever you return one or more file_demands.
   - Return ready_to_write only when previous_writing_reports already contain
     sufficient completed implementation artifacts and no new file_demands are
     needed.
2. For edit_existing_repository:
   - If source reading is absent and current-source understanding is needed,
     return need_reading with one to three semantic source evidence requests.
   - If source reading is sufficient or the user request already describes the
     required behavior, return semantic file demands.
   - Do not infer or output source owners, concrete paths, related paths,
     read-only paths, current-file context, or import lines.
3. For create_new_project:
   - Describe the semantic modules, scripts, tests, docs, config, and support
     files needed for the new project.
   - Do not choose concrete paths. File Agent owns file names and locations.
4. Build one to five file_demands:
   - Each demand has one semantic owner role, purpose, file_kind,
     interface_contract, integration_contract, change_goal,
     work_instructions, required_slots, and validation_expectations.
   - Each interface_contract has a component and at least one input, output,
     and invariant. Each integration_contract has provides_to_pm and
     consumes_from.
   - If runtime behavior and tests are both required, assign the relevant code
     and tests together only when one worker can safely own the interface; use
     companion demands when separate workers should own separate files.
   - Keep shared inputs, outputs, data shapes, producer demand ids, consumer
     demand ids, and validation rules explicit so File PMs do not invent
     incompatible interfaces.
   - Do not produce exact import lines. Return cross_module_imports as an empty
     object; File Agent and File PM will derive imports after paths are known.
5. Use previous_writing_reports and patch_check_result:
   - If prior programmer reports are sufficient and validation is absent or
     succeeded, return ready_to_write with no file demands.
   - If validation failed, assign a complete corrected replacement for the
     affected work, request missing source evidence, or ask for user input.
6. If file_resolution_feedback or file_plan_feedback is present:
   - Keep the same user goal and correct the file demands.
   - Fix semantic purpose, file_kind, interface_contract,
     integration_contract, change_goal, validation_expectations, and demand
     separation.
   - Do not write code, file content, tests, paths, imports, diffs, or patch
     operations.
7. Request external evidence only for public or third-party facts not available
   in source evidence. Return rejected or needs_user_input for requests to run
   commands, install packages, apply patches, inspect secrets, or use private
   credentials.

# Output Format
Return strict JSON:
{
  "status": "need_reading | need_file_pms | ready_to_write | needs_user_input | overloaded | rejected",
  "mode": "edit_existing_repository | create_new_project",
  "intent": "short generic implementation intent",
  "reading_requests": [
    {
      "request_id": "stable short id",
      "task": "read-only source evidence task",
      "reason": "why this evidence is needed before code writing",
      "required_slots": ["specific current-source facts to collect"]
    }
  ],
  "file_demands": [
    {
      "demand_id": "stable short id",
      "role": "generic worker role",
      "purpose": "what this file or file group must provide",
      "file_kind": "existing | new | test | docs | config | support",
      "interface_contract": {
        "component": "small component name",
        "inputs": ["inputs this worker must consume"],
        "outputs": ["outputs this worker must provide"],
        "invariants": ["interface rules to preserve"]
      },
      "integration_contract": {
        "provides_to_pm": ["deliverables PM will reconcile"],
        "consumes_from": ["other file contracts this file needs"]
      },
      "change_goal": "one sentence scoped implementation goal",
      "work_instructions": ["bounded downstream work instruction"],
      "required_slots": ["slots this worker must satisfy"],
      "validation_expectations": ["structural checks or tests to satisfy"]
    }
  ],
  "cross_module_imports": {},
  "missing_slots": ["clarification or missing evidence"],
  "external_evidence_requests": [
    {
      "request_id": "stable short id",
      "task": "public evidence task for the web helper",
      "reason": "why this evidence is needed for the patch"
    }
  ]
}
'''

WRITING_PM_JSON_RETRY_PROMPT = '''\
Your previous product-manager response was empty or not valid JSON.
Return one strict JSON object only, matching the required output format from
the system instructions. Do not include markdown, commentary, or code fences.
'''

WRITING_PM_COMPACT_RECOVERY_PROMPT = '''\
You are the product manager for a code-writing agent. You do not write code.
Choose one JSON decision for the provided payload.

Rules:
- Use the provided mode.
- If you return any file_demands, status must be need_file_pms.
- Use ready_to_write only when previous_writing_reports already contain
  sufficient completed implementation artifacts.
- For edit_existing_repository, use the user request and source_reading_state.
  Return need_reading only when current-source understanding is needed and
  source reading is absent or incomplete.
- For create_new_project, describe semantic modules, tests, docs, config, and
  support files. Do not choose concrete paths.
- Keep at most five file_demands with separate responsibilities, clear file
  purpose, required_slots, interface_contract, and integration_contract.
- Do not output source owners, concrete paths, related paths, read-only paths,
  current-file context, import lines, code, or patch operations.
- Return cross_module_imports as an empty object.
- Fill every interface_contract component, inputs, outputs, and invariants
  field with at least one useful value.
- If patch_check_result failed, assign a complete corrected replacement,
  request missing source evidence, or return needs_user_input/overloaded.
- If file_resolution_feedback or file_plan_feedback is present, correct the
  file demands for the same user goal before returning need_file_pms.
- Return rejected or needs_user_input for command execution, package
  installation, real patch application, secrets, or private credentials.
- Return strict JSON only, with no markdown or commentary.

Schema:
{
  "status": "need_reading | need_file_pms | ready_to_write | needs_user_input | overloaded | rejected",
  "mode": "edit_existing_repository | create_new_project",
  "intent": "short implementation intent",
  "reading_requests": [
    {
      "request_id": "stable id",
      "task": "read-only source evidence task",
      "reason": "why the evidence is required",
      "required_slots": ["source facts to collect"]
    }
  ],
  "file_demands": [
    {
      "demand_id": "stable id",
      "role": "limited worker role",
      "purpose": "what this file demand must provide",
      "file_kind": "existing | new | test | docs | config | support",
      "interface_contract": {
        "component": "small component name",
        "inputs": ["inputs"],
        "outputs": ["outputs"],
        "invariants": ["rules"]
      },
      "integration_contract": {
        "provides_to_pm": ["deliverables"],
        "consumes_from": ["other file contracts"]
      },
      "change_goal": "scoped implementation goal",
      "work_instructions": ["bounded downstream work instruction"],
      "required_slots": ["interface or file requirements"],
      "validation_expectations": ["checks to satisfy"]
    }
  ],
  "cross_module_imports": {},
  "missing_slots": ["missing facts or clarification"],
  "external_evidence_requests": [
    {
      "request_id": "stable id",
      "task": "public evidence task",
      "reason": "why the evidence is required"
    }
  ]
}
'''

_writing_pm_llm = LLInterface()
_writing_pm_llm_config = LLMCallConfig(
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
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def decide_writing_work(
    pm_input: WritingPMInput,
    *,
    trace: dict[str, object] | None = None,
) -> WritingPMDecision:
    """Ask the writing PM to choose bounded code-writing work.

    Args:
        pm_input: Compact user request, mode, repository summary, scope-free
            reading status, and prior writing reports.
        trace: Optional internal diagnostic dictionary populated with safe route
            and model metadata plus raw and parsed model output.

    Returns:
        A schema-normalized PM decision for limited writing work.
    """

    payload = _pm_payload(pm_input)
    payload_text = json.dumps(payload, ensure_ascii=False)
    context_budget = prompt_budget_metadata(
        system_prompt=WRITING_PM_PROMPT,
        payload_text=payload_text,
        target_input_tokens=PM_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
    )
    if context_budget["over_hard_cap"]:
        decision = _overloaded_decision(
            mode=pm_input["mode"],
            reason="Writing PM prompt exceeded the context budget.",
        )
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_output=decision,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return decision

    raw_output, parsed, attempts = await _invoke_pm_json(payload_text)
    decision = normalize_writing_pm_decision(parsed, mode=pm_input["mode"])
    _fill_trace(
        trace,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=decision,
        context_budget=context_budget,
        blocked_before_invoke=False,
        attempts=attempts,
    )
    return decision


async def _invoke_pm_json(
    payload_text: str,
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    raw_output = ""
    parsed: dict[str, object] = {}
    use_compact_recovery = False
    for attempt_index in range(3):
        prompt_kind = "compact_recovery" if use_compact_recovery else "full"
        if use_compact_recovery:
            messages = [
                SystemMessage(content=WRITING_PM_COMPACT_RECOVERY_PROMPT),
                HumanMessage(content=payload_text),
            ]
        else:
            messages = [
                SystemMessage(content=WRITING_PM_PROMPT),
                HumanMessage(content=payload_text),
            ]
        if attempt_index == 1 and not use_compact_recovery:
            prompt_kind = "full_json_retry"
            messages.append(HumanMessage(content=WRITING_PM_JSON_RETRY_PROMPT))
        timed_out = False
        try:
            response = await asyncio.wait_for(
                _writing_pm_llm.ainvoke(
                    messages,
                    config=_writing_pm_llm_config,
                ),
                timeout=PM_LLM_CALL_TIMEOUT_SECONDS,
            )
            raw_output = response.content
            parsed = parse_llm_json_output(raw_output)
        except asyncio.TimeoutError:
            timed_out = True
            raw_output = ""
            parsed = {}
        attempts.append({
            "attempt": attempt_index + 1,
            "prompt_kind": prompt_kind,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "timed_out": timed_out,
        })
        if parsed:
            break
        if use_compact_recovery:
            break
        if timed_out or not raw_output.strip():
            use_compact_recovery = True
    return raw_output, parsed, attempts


def normalize_writing_pm_decision(
    parsed: object,
    *,
    mode: WritingMode,
) -> WritingPMDecision:
    """Normalize and validate PM JSON into the writing decision contract."""

    if not isinstance(parsed, dict):
        decision = _needs_user_input_decision(
            mode=mode,
            reason="Writing PM returned malformed output.",
        )
        return decision

    status = _bounded_text(parsed.get("status"))
    if status not in PM_STATUSES:
        status = "needs_user_input"

    parsed_mode = _bounded_text(parsed.get("mode"))
    if parsed_mode not in WRITING_MODES:
        parsed_mode = mode
    if parsed_mode != mode:
        parsed_mode = mode

    file_demands = _file_demands_from_parsed(parsed.get("file_demands"))
    cross_module_imports = _cross_module_imports_from_parsed(
        parsed.get("cross_module_imports")
    )
    missing_slots = _string_list(parsed.get("missing_slots"), MAX_REQUIRED_SLOTS)
    reading_requests = _reading_requests_from_parsed(parsed.get("reading_requests"))
    external_requests = _external_requests_from_parsed(
        parsed.get("external_evidence_requests")
    )

    if status == "need_reading" and parsed_mode != "edit_existing_repository":
        status = "needs_user_input"
        missing_slots.append("Repository source is required for source reading.")
    if status == "need_reading" and not reading_requests:
        reading_requests = [_fallback_reading_request(missing_slots)]
    if status == "need_reading":
        external_requests = []
    if status == "need_file_pms" and not file_demands:
        status = "needs_user_input"
        missing_slots.append("No semantic file demands were provided.")
    if status != "need_file_pms":
        file_demands = []
        cross_module_imports = {}
    if status != "need_reading":
        reading_requests = []
    if len(file_demands) > MAX_ASSIGNMENTS_PER_DECISION:
        status = "overloaded"
        file_demands = []
        missing_slots.append("Too many file demands were requested.")

    decision: WritingPMDecision = {
        "status": status,
        "mode": parsed_mode,
        "intent": _bounded_text(parsed.get("intent")) or "code_change",
        "file_demands": file_demands,
        "file_contracts": [],
        "cross_module_imports": cross_module_imports,
        "missing_slots": missing_slots,
        "reading_requests": reading_requests,
        "external_evidence_requests": external_requests,
    }
    return decision


def _pm_payload(pm_input: WritingPMInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": _bounded_multiline_text(
            pm_input["question"],
            MAX_PM_QUESTION_CHARS,
        ),
        "mode": pm_input["mode"],
        "repository_summary": pm_input["repository_summary"],
        "reading_reports": _scope_free_reading_reports(
            pm_input["reading_reports"]
        ),
        "source_reading_state": pm_input.get("supervisor_evidence_state", {}),
        "previous_writing_reports": _compact_writing_reports(
            pm_input["previous_writing_reports"]
        ),
    }
    external_evidence = pm_input.get("external_evidence")
    if external_evidence:
        payload["external_evidence"] = external_evidence
    validation_feedback = pm_input.get("validation_feedback")
    if validation_feedback is not None:
        payload["patch_check_result"] = validation_feedback
    file_resolution_feedback = pm_input.get("file_resolution_feedback")
    if file_resolution_feedback is not None:
        payload["file_resolution_feedback"] = file_resolution_feedback
    file_plan_feedback = pm_input.get("file_plan_feedback")
    if file_plan_feedback is not None:
        payload["file_plan_feedback"] = file_plan_feedback
    return payload


def _scope_free_reading_reports(
    reading_reports: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return reading status for PM without source scope or path details."""

    reports: list[dict[str, object]] = []
    for report in reading_reports[:MAX_PM_READING_REPORTS]:
        if not isinstance(report, dict):
            continue

        scope_free_report: dict[str, object] = {
            "status": _bounded_text(report.get("status")),
            "scope_details": "reserved_for_file_agent_and_file_pm",
        }
        facts = _string_list(report.get("facts"), MAX_PM_READING_FACTS)
        if facts:
            scope_free_report["facts"] = facts
        limitations = _string_list(report.get("limitations"), MAX_REQUIRED_SLOTS)
        if limitations:
            scope_free_report["limitations_present"] = True
        reports.append(scope_free_report)
    return reports


def _compact_owner_candidates(
    owner_candidates: list[SourceOwnerCandidate],
) -> list[dict[str, object]]:
    compact_candidates: list[dict[str, object]] = []
    for candidate in owner_candidates[:MAX_PM_OWNER_CANDIDATES]:
        compact_candidate = {
            "path": candidate["path"],
            "role": candidate["role"],
            "line_start": candidate["line_start"],
            "line_end": candidate["line_end"],
            "symbols": _string_list(
                candidate["symbols"],
                MAX_PM_OWNER_SYMBOLS,
            ),
            "exception_types": _string_list(
                candidate["exception_types"],
                MAX_PM_OWNER_EXCEPTION_TYPES,
            ),
            "feature_markers": _string_list(
                candidate["feature_markers"],
                MAX_PM_OWNER_FEATURE_MARKERS,
            ),
            "reasons": _string_list(
                candidate["reasons"],
                MAX_PM_OWNER_REASONS,
            ),
            "evidence_refs": _string_list(
                candidate["evidence_refs"],
                MAX_PM_OWNER_EVIDENCE_REFS,
            ),
        }
        compact_candidates.append(compact_candidate)
    return compact_candidates


def _file_demands_from_parsed(parsed: object) -> list[WritingFileDemand]:
    if not isinstance(parsed, list):
        return []

    demands: list[WritingFileDemand] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        demand = _file_demand_from_dict(item, index=index)
        if demand is not None:
            demands.append(demand)
        if len(demands) >= MAX_ASSIGNMENTS_PER_DECISION + 1:
            break
    return demands


def _file_demand_from_dict(
    item: dict[str, Any],
    *,
    index: int,
) -> WritingFileDemand | None:
    purpose = _bounded_text(item.get("purpose"))
    work_instructions = _string_list(
        item.get("work_instructions"),
        MAX_WORK_INSTRUCTIONS,
    )
    required_slots = _string_list(item.get("required_slots"), MAX_REQUIRED_SLOTS)
    if not purpose and not work_instructions and not required_slots:
        return None

    demand_id = _bounded_text(item.get("demand_id"))
    if not demand_id:
        demand_id = f"file-{index}"
    role = _bounded_text(item.get("role")) or "file programmer"
    file_kind = _bounded_text(item.get("file_kind"))
    if file_kind not in FILE_KINDS:
        file_kind = "support"
    if not work_instructions:
        work_instructions = [purpose or "Implement this file demand."]
    if not required_slots:
        required_slots = [purpose or "file deliverable"]

    demand: WritingFileDemand = {
        "demand_id": demand_id,
        "role": role,
        "purpose": purpose,
        "file_kind": file_kind,
        "preferred_path": _bounded_text(item.get("preferred_path")),
        "preferred_name": _bounded_text(item.get("preferred_name")),
        "placement_hint": _bounded_text(item.get("placement_hint")),
        "related_paths": _string_list(
            item.get("related_paths"),
            MAX_ASSIGNMENT_VALUES,
        ),
        "read_only_paths": _string_list(
            item.get("read_only_paths"),
            MAX_ASSIGNMENT_VALUES,
        ),
        "interface_contract": _bounded_object_dict(
            item.get("interface_contract"),
        ),
        "integration_contract": _bounded_object_dict(
            item.get("integration_contract"),
        ),
        "change_goal": _bounded_text(item.get("change_goal")),
        "work_instructions": work_instructions,
        "required_slots": required_slots,
        "validation_expectations": _string_list(
            item.get("validation_expectations"),
            MAX_REQUIRED_SLOTS,
        ),
        "forbidden_paths": _string_list(
            item.get("forbidden_paths"),
            MAX_ASSIGNMENT_VALUES,
        ),
    }
    return demand


def _external_requests_from_parsed(
    parsed: object,
) -> list[WritingExternalEvidenceRequest]:
    if not isinstance(parsed, list):
        return []

    requests: list[WritingExternalEvidenceRequest] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        task = _bounded_text(item.get("task"))
        if not task:
            continue
        request_id = _bounded_text(item.get("request_id"))
        if not request_id:
            request_id = f"external-{index}"
        request: WritingExternalEvidenceRequest = {
            "request_id": request_id,
            "task": task,
            "reason": _bounded_text(item.get("reason")),
        }
        requests.append(request)
        if len(requests) >= MAX_EXTERNAL_EVIDENCE_REQUESTS:
            break
    return requests


def _cross_module_imports_from_parsed(parsed: object) -> dict[str, list[str]]:
    if not isinstance(parsed, dict):
        return {}

    imports_by_demand: dict[str, list[str]] = {}
    for raw_key, raw_value in parsed.items():
        demand_id = _bounded_text(raw_key)
        if not demand_id:
            continue
        import_lines = _string_list(raw_value, MAX_ASSIGNMENT_VALUES)
        if not import_lines:
            continue
        imports_by_demand[demand_id] = import_lines
        if len(imports_by_demand) >= MAX_ASSIGNMENTS_PER_DECISION:
            break
    return imports_by_demand


def _reading_requests_from_parsed(
    parsed: object,
) -> list[WritingReadingEvidenceRequest]:
    if not isinstance(parsed, list):
        return []

    requests: list[WritingReadingEvidenceRequest] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        task = _bounded_text(item.get("task"))
        reason = _bounded_text(item.get("reason"))
        if not task or not reason:
            continue
        request_id = _bounded_text(item.get("request_id"))
        if not request_id:
            request_id = f"read-{index}"
        required_slots = _string_list(
            item.get("required_slots"),
            MAX_REQUIRED_SLOTS,
        )
        if not required_slots:
            required_slots = ["Current source owners and validation evidence."]
        request: WritingReadingEvidenceRequest = {
            "request_id": request_id,
            "task": task,
            "reason": reason,
            "required_slots": required_slots,
        }
        requests.append(request)
        if len(requests) >= MAX_READING_EVIDENCE_REQUESTS:
            break
    return requests


def _fallback_reading_request(
    missing_slots: list[str],
) -> WritingReadingEvidenceRequest:
    required_slots = list(missing_slots)
    if not required_slots:
        required_slots = [
            "Current source owners for the requested behavior.",
            "Tests or validation paths that cover the current behavior.",
        ]
    request: WritingReadingEvidenceRequest = {
        "request_id": "source-evidence",
        "task": (
            "Collect current repository evidence needed before limited "
            "implementation planning."
        ),
        "reason": "The writing PM requires source evidence before assigning work.",
        "required_slots": required_slots[:MAX_REQUIRED_SLOTS],
    }
    return request


def _compact_writing_reports(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    compact_reports: list[dict[str, object]] = []
    for report in reports:
        compact_report = {
            "assignment_id": report["assignment_id"],
            "file_contract_id": report.get("file_contract_id", ""),
            "file_label": report.get("file_label", ""),
            "edit_mode": report.get("edit_mode", ""),
            "status": report["status"],
            "facts": report["facts"],
            "files_considered": report["files_considered"],
            "open_questions": report["open_questions"],
            "code_artifact_chars": len(report.get("code_artifact", "")),
            "created_files": report.get("created_files", []),
            "changed_files": report.get("changed_files", []),
        }
        compact_reports.append(compact_report)
    return compact_reports


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


def _bounded_object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}

    bounded: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = _bounded_text(raw_key)
        if not key:
            continue
        if isinstance(raw_value, list):
            bounded[key] = _string_list(raw_value, MAX_REQUIRED_SLOTS)
            continue
        if isinstance(raw_value, dict):
            bounded[key] = _bounded_object_dict(raw_value)
            continue
        bounded[key] = _bounded_text(raw_value)
    return bounded


def _needs_user_input_decision(
    *,
    mode: WritingMode,
    reason: str,
) -> WritingPMDecision:
    decision: WritingPMDecision = {
        "status": "needs_user_input",
        "mode": mode,
        "intent": "code_change",
        "file_demands": [],
        "file_contracts": [],
        "cross_module_imports": {},
        "missing_slots": [reason],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    return decision


def _overloaded_decision(
    *,
    mode: WritingMode,
    reason: str,
) -> WritingPMDecision:
    decision: WritingPMDecision = {
        "status": "overloaded",
        "mode": mode,
        "intent": "code_change",
        "file_demands": [],
        "file_contracts": [],
        "cross_module_imports": {},
        "missing_slots": [reason],
        "reading_requests": [],
        "external_evidence_requests": [],
    }
    return decision


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: dict[str, object],
    normalized_output: WritingPMDecision,
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
    attempts: list[dict[str, object]] | None = None,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _writing_pm_llm_config.route_name
    trace["model"] = _writing_pm_llm_config.model
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    if attempts is not None:
        trace["attempts"] = attempts
