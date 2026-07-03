"""LLM-backed product-manager lifecycle decisions for code writing."""

from __future__ import annotations

import ast
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
    WritingBlocker,
    WritingChildPMTask,
    WritingCompletionReport,
    WritingDomain,
    WritingInformationRequest,
    WritingPMDecision,
    WritingPMInput,
    WritingPMStatus,
    WritingProgrammerTask,
    WritingRepairInstruction,
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
    "request_information",
    "create_child_pm",
    "create_programmer_task",
    "complete",
    "blocked",
)
PM_DOMAINS: tuple[WritingDomain, ...] = ("writing",)
ACTION_PAYLOAD_KEYS: dict[WritingPMStatus, str] = {
    "request_information": "information_request",
    "create_child_pm": "child_pm_task",
    "create_programmer_task": "programmer_task",
    "repair_child": "repair_instruction",
    "complete": "completion_report",
    "blocked": "blocker",
}
MAX_LIST_ITEMS = 16
MAX_TEXT_FIELD_CHARS = 900
MAX_REASON_CHARS = 1200
MAX_PM_PAYLOAD_CHARS = 12000
PM_LLM_CALL_TIMEOUT_SECONDS = 300


WRITING_PM_PROMPT = '''\
You are the product-manager role inside a code-writing agent.

Your job is to manage one assigned work item. You own only your direct
children. A direct child can be another product manager or one programmer.
Use context_limits as workflow limits. pm_depth is your current level.
remaining_child_pm_depth is how many more child-product-manager levels you may
create from this PM. If remaining_child_pm_depth is 0, choose
create_programmer_task, request_information, complete, or blocked instead of
create_child_pm.

# Decision Rules
1. First decide whether the next useful child can be one programmer. Choose
   create_programmer_task when one artifact can be specified with a clear
   purpose, required behavior, provided interface, consumed interface, imports,
   and output format.
   For a test artifact, the required behavior is clear only when the facts
   already state the exact visible behavior that tests will assert. If tests
   would compare rows, text, records, files, or reported errors and those
   visible details are not stated, choose request_information.
2. For a small multi-artifact work item, create programmer tasks one artifact
   at a time when the next artifact contract is clear from available facts and
   direct child reports.
   This stage currently writes Python artifacts. Describe interfaces with
   normal Python type hints. Every input and output item must have an agreed
   name at PM level. For a list, name the list and its element type. For a dict
   or record, name the record type and define its fields with Python value
   types. Prefer TypedDict, dataclass, or NamedTuple for records that another
   artifact will consume. Later consumed interfaces must repeat the same names
   and type shapes.
   When output values are discovered while processing input, define one direct
   function that returns a named result record after processing. Put discovered
   values such as field names, rows, counts, and errors in that result record.
   Prefer completed result records over lazy callbacks or generators when later
   artifacts must consume the discovered values together.
   For new artifacts, if the user asks for a tool to read an input format but
   does not name an existing standard, source file, or repository contract, the
   generated artifacts may define a simple local format and document it in
   README and tests. Do not request external evidence just to discover a public
   standard for an input format the new tool can own.
3. Choose create_child_pm when the assigned work item is still too broad for a
   safe one-artifact programmer contract and a smaller owner should coordinate
   a cohesive sub-area before programmer work starts.
4. Choose request_information when the next child instruction depends on facts
   the supervisor can obtain from existing source, generated artifacts,
   provided evidence, or public external evidence. Use this when a dependent
   artifact needs actual generated behavior that is not present in available
   facts or direct child reports. Before assigning tests, documentation, or a
   command wrapper that asserts a generated artifact's output, check that direct
   child reports include the observable output shape the child must assert. For
   Python text or tabular outputs, this includes header-row presence, field
   order, row shape, return fields, and error-reporting surface. Phrases like
   "write records" are not enough for exact-row tests because they do not say
   whether header rows are part of the output. If those facts are absent or
   ambiguous, request information instead of asking the next child to guess.
   When supervisor-provided facts are present and a programmer task depends on
   them, put the relevant request_id values in consumed_fact_ids.
5. Choose complete when your assigned work item is satisfied and your report
   can be sent upward.
6. Choose blocked when the work cannot proceed from the available facts, when
   required user requirements are missing, or when the missing facts cannot be
   obtained from workspace artifacts, existing source, provided evidence, or
   public external evidence.

Do not manage descendants owned by a child PM. Do not ask a programmer to
infer peer output. Do not claim that commands, tests, patch apply, package
installation, or real workspace mutation happened.
Write output as a role decision. Do not cite prompt rule numbers, prompt
section names, or hidden instruction wording.

# Boundary Rules
- Child PM reports may ask for artifact identifiers, artifact purpose,
  provided interfaces, consumed interfaces, and open risks.
- Child PM reports must not ask for command results, test execution results,
  package installation results, patch-apply results, or real workspace mutation
  results.
- File paths and path reservation are file-mechanics facts. Ask for artifact
  labels or reserved paths only when those facts are already provided.

# Output Format
Return strict JSON with exactly one action payload populated.
All top-level fields shown below must be present. Use null for every action
payload that is not selected.
{
  "status": "request_information | create_child_pm | create_programmer_task | complete | blocked",
  "reason": "short reason",
  "information_request": null,
  "child_pm_task": null,
  "programmer_task": null,
  "completion_report": null,
  "blocker": null
}

For information_request:
{
  "request_id": "short id",
  "needed_facts": ["facts needed before the next child instruction"],
  "target_artifacts": ["artifact labels or paths to inspect"],
  "reason_for_next_instruction": "why these facts are needed"
}

For child_pm_task:
{
  "child_pm_id": "short id",
  "domain": "writing",
  "goal": "what the child PM owns",
  "scope": "direct scope for the child PM",
  "constraints": ["constraints"],
  "expected_report": ["facts the child PM must report upward"]
}

For programmer_task:
{
  "task_id": "short id",
  "artifact_purpose": "one artifact purpose",
  "required_behavior": ["required behavior"],
  "provided_interfaces": ["interfaces this artifact provides"],
  "consumed_interfaces": ["interfaces this artifact consumes"],
  "consumed_fact_ids": ["supervisor fact request_id values used by this task"],
  "imports": ["required imports"],
  "output_format": "expected artifact format"
}

For completion_report:
{
  "pm_id": "this PM id",
  "status": "complete",
  "provided_facts": ["facts produced"],
  "created_artifacts": [{"artifact_id": "id", "purpose": "purpose"}],
  "consumed_facts": ["facts used"],
  "open_risks": ["remaining risks"],
  "next_dependency_needs": ["facts needed by later work"]
}

For blocker:
{
  "summary": "specific blocker",
  "missing_facts": ["facts that are missing"],
  "why_information_request_is_not_enough": "why a narrower request cannot unblock it"
}
'''

WRITING_PM_RETRY_PROMPT = '''\
Your previous response was empty or not valid JSON.
Return exactly one strict JSON object matching the required output format.
Do not include markdown, commentary, code, paths, diffs, or command output.
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
    """Ask one writing PM to choose its next lifecycle action.

    Args:
        pm_input: One PM work packet with direct child reports and approved
            facts only.
        trace: Optional diagnostic dictionary for live LLM review.

    Returns:
        A normalized PM lifecycle decision.
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
        decision = _blocked_decision(
            "Writing PM prompt exceeded the context budget.",
        )
        _fill_trace(
            trace,
            system_prompt=WRITING_PM_PROMPT,
            human_payload=payload,
            human_payload_text=payload_text,
            raw_output="",
            parsed_output={},
            normalized_output=decision,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return decision

    raw_output, parsed, attempts = await _invoke_pm_json(payload_text)
    decision = normalize_writing_pm_decision(parsed)
    _fill_trace(
        trace,
        system_prompt=WRITING_PM_PROMPT,
        human_payload=payload,
        human_payload_text=payload_text,
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
    for attempt_index in range(2):
        messages = [
            SystemMessage(content=WRITING_PM_PROMPT),
            HumanMessage(content=payload_text),
        ]
        if attempt_index > 0:
            messages.append(HumanMessage(content=WRITING_PM_RETRY_PROMPT))
        message_trace = []
        for message_index, message in enumerate(messages):
            role = "system" if message_index == 0 else "human"
            message_trace.append({"role": role, "content": message.content})
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
            "messages": message_trace,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "timed_out": timed_out,
        })
        if parsed or timed_out:
            break
    return raw_output, parsed, attempts


def normalize_writing_pm_decision(parsed: object) -> WritingPMDecision:
    """Normalize PM JSON into the lifecycle action contract."""

    if not isinstance(parsed, dict):
        decision = _blocked_decision("Writing PM returned malformed output.")
        return decision

    status = _bounded_text(parsed.get("status"))
    if status not in PM_STATUSES:
        decision = _blocked_decision("Writing PM returned unsupported status.")
        return decision

    decision = _empty_decision(status=status)
    decision["reason"] = (
        _bounded_multiline_text(parsed.get("reason"), MAX_REASON_CHARS)
        or "PM selected a lifecycle action."
    )
    payload_key = ACTION_PAYLOAD_KEYS[status]
    raw_payload = parsed.get(payload_key)
    normalized_payload = _normalize_payload(
        status=status,
        payload=raw_payload,
    )
    if normalized_payload is None:
        decision = _blocked_decision(
            f"Writing PM did not provide valid {payload_key}.",
        )
        return decision

    decision[payload_key] = normalized_payload
    return decision


def _normalize_payload(
    *,
    status: WritingPMStatus,
    payload: object,
) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    if status == "request_information":
        return _information_request(payload)
    if status == "create_child_pm":
        return _child_pm_task(payload)
    if status == "create_programmer_task":
        return _programmer_task(payload)
    if status == "repair_child":
        return _repair_instruction(payload)
    if status == "complete":
        return _completion_report(payload)
    return _blocker(payload)


def _information_request(
    payload: dict[str, object],
) -> WritingInformationRequest | None:
    needed_facts = _string_list(payload.get("needed_facts"), MAX_LIST_ITEMS)
    request_id = _bounded_text(payload.get("request_id")) or "info_request"
    if not needed_facts:
        return None
    request: WritingInformationRequest = {
        "request_id": request_id,
        "needed_facts": needed_facts,
        "target_artifacts": _string_list(
            payload.get("target_artifacts"),
            MAX_LIST_ITEMS,
        ),
        "reason_for_next_instruction": _bounded_multiline_text(
            payload.get("reason_for_next_instruction"),
            MAX_REASON_CHARS,
        ),
    }
    return request


def _child_pm_task(payload: dict[str, object]) -> WritingChildPMTask | None:
    goal = _bounded_multiline_text(payload.get("goal"), MAX_REASON_CHARS)
    scope = _bounded_multiline_text(payload.get("scope"), MAX_REASON_CHARS)
    if not goal or not scope:
        return None
    domain = _bounded_text(payload.get("domain"))
    if domain not in PM_DOMAINS:
        domain = "writing"
    task: WritingChildPMTask = {
        "child_pm_id": (
            _bounded_text(payload.get("child_pm_id")) or "child_pm"
        ),
        "domain": domain,
        "goal": goal,
        "scope": scope,
        "constraints": _string_list(payload.get("constraints"), MAX_LIST_ITEMS),
        "expected_report": _string_list(
            payload.get("expected_report"),
            MAX_LIST_ITEMS,
        ),
    }
    return task


def _programmer_task(payload: dict[str, object]) -> WritingProgrammerTask | None:
    purpose = _bounded_multiline_text(
        payload.get("artifact_purpose"),
        MAX_REASON_CHARS,
    )
    required_behavior = _string_list(
        payload.get("required_behavior"),
        MAX_LIST_ITEMS,
    )
    if not purpose or not required_behavior:
        return None
    task: WritingProgrammerTask = {
        "task_id": _bounded_text(payload.get("task_id")) or "programmer_task",
        "artifact_purpose": purpose,
        "required_behavior": required_behavior,
        "provided_interfaces": _string_list(
            payload.get("provided_interfaces"),
            MAX_LIST_ITEMS,
        ),
        "consumed_interfaces": _string_list(
            payload.get("consumed_interfaces"),
            MAX_LIST_ITEMS,
        ),
        "consumed_fact_ids": _string_list(
            payload.get("consumed_fact_ids"),
            MAX_LIST_ITEMS,
        ),
        "imports": _import_list(payload.get("imports")),
        "output_format": (
            _bounded_text(payload.get("output_format")) or "python"
        ),
    }
    return task


def _repair_instruction(
    payload: dict[str, object],
) -> WritingRepairInstruction | None:
    child_id = _bounded_text(payload.get("child_id"))
    feedback = _bounded_multiline_text(payload.get("feedback"), MAX_REASON_CHARS)
    expected = _bounded_multiline_text(
        payload.get("expected_correction"),
        MAX_REASON_CHARS,
    )
    if not child_id or not feedback or not expected:
        return None
    instruction: WritingRepairInstruction = {
        "child_id": child_id,
        "feedback": feedback,
        "expected_correction": expected,
    }
    return instruction


def _completion_report(
    payload: dict[str, object],
) -> WritingCompletionReport | None:
    pm_id = _bounded_text(payload.get("pm_id"))
    if not pm_id:
        return None
    report: WritingCompletionReport = {
        "pm_id": pm_id,
        "status": "complete",
        "provided_facts": _string_list(
            payload.get("provided_facts"),
            MAX_LIST_ITEMS,
        ),
        "created_artifacts": _dict_list(
            payload.get("created_artifacts"),
            MAX_LIST_ITEMS,
        ),
        "consumed_facts": _string_list(
            payload.get("consumed_facts"),
            MAX_LIST_ITEMS,
        ),
        "open_risks": _string_list(payload.get("open_risks"), MAX_LIST_ITEMS),
        "next_dependency_needs": _string_list(
            payload.get("next_dependency_needs"),
            MAX_LIST_ITEMS,
        ),
    }
    return report


def _blocker(payload: dict[str, object]) -> WritingBlocker | None:
    summary = _bounded_multiline_text(payload.get("summary"), MAX_REASON_CHARS)
    why_not = _bounded_multiline_text(
        payload.get("why_information_request_is_not_enough"),
        MAX_REASON_CHARS,
    )
    if not summary:
        return None
    blocker: WritingBlocker = {
        "summary": summary,
        "missing_facts": _string_list(payload.get("missing_facts"), MAX_LIST_ITEMS),
        "why_information_request_is_not_enough": why_not,
    }
    return blocker


def _pm_payload(pm_input: WritingPMInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "pm_id": _bounded_text(pm_input["pm_id"]),
        "domain": pm_input["domain"],
        "work_item": {
            "goal": _bounded_multiline_text(
                pm_input["work_item"]["goal"],
                MAX_PM_PAYLOAD_CHARS,
            ),
            "scope": _bounded_multiline_text(
                pm_input["work_item"]["scope"],
                MAX_REASON_CHARS,
            ),
            "constraints": pm_input["work_item"]["constraints"],
            "expected_result": _bounded_multiline_text(
                pm_input["work_item"]["expected_result"],
                MAX_REASON_CHARS,
            ),
        },
        "available_facts": pm_input["available_facts"],
        "direct_child_reports": pm_input["direct_child_reports"],
        "child_feedback": pm_input["child_feedback"],
        "context_limits": pm_input["context_limits"],
    }
    return payload


def _empty_decision(*, status: WritingPMStatus) -> WritingPMDecision:
    decision: WritingPMDecision = {
        "status": status,
        "reason": "",
        "information_request": None,
        "child_pm_task": None,
        "programmer_task": None,
        "repair_instruction": None,
        "completion_report": None,
        "blocker": None,
    }
    return decision


def _blocked_decision(reason: str) -> WritingPMDecision:
    decision = _empty_decision(status="blocked")
    decision["reason"] = reason
    decision["blocker"] = {
        "summary": reason,
        "missing_facts": [],
        "why_information_request_is_not_enough": (
            "The PM output could not be accepted as a valid lifecycle action."
        ),
    }
    return decision


def _dict_list(value: object, limit: int) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    rows: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        row: dict[str, object] = {}
        for raw_key, raw_value in item.items():
            key = _bounded_text(raw_key)
            if not key:
                continue
            if isinstance(raw_value, list):
                row[key] = _string_list(raw_value, MAX_LIST_ITEMS)
            else:
                row[key] = _bounded_multiline_text(
                    raw_value,
                    MAX_TEXT_FIELD_CHARS,
                )
        if row:
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    strings: list[str] = []
    for item in value:
        text = _bounded_multiline_text(item, MAX_TEXT_FIELD_CHARS)
        if not text or text in strings:
            continue
        strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def _import_list(value: object) -> list[str]:
    imports: list[str] = []
    for text in _string_list(value, MAX_LIST_ITEMS):
        if not _is_import_statement(text):
            continue
        imports.append(text)
    return imports


def _is_import_statement(text: str) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if len(tree.body) != 1:
        return False
    statement = tree.body[0]
    return isinstance(statement, (ast.Import, ast.ImportFrom))


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


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    system_prompt: str,
    human_payload: dict[str, object],
    human_payload_text: str,
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
    trace["thinking_enabled"] = CODING_AGENT_PM_LLM_THINKING_ENABLED
    trace["system_prompt"] = system_prompt
    trace["human_payload"] = human_payload
    trace["human_payload_text"] = human_payload_text
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    if attempts is not None:
        trace["attempts"] = attempts
