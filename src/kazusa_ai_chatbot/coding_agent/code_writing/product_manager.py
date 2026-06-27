"""LLM-backed product-manager decisions for new-artifact code writing."""

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
    ExternalEvidenceSummary,
    WritingArtifactItem,
    WritingContentFormat,
    WritingExternalEvidenceRequest,
    WritingFileKind,
    WritingPMDecision,
    WritingPMInput,
    WritingPMStatus,
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
    "need_programmers",
    "need_external_evidence",
    "sufficient",
    "rejected",
)
FILE_KINDS: tuple[WritingFileKind, ...] = (
    "source",
    "test",
    "docs",
    "config",
    "data",
)
CONTENT_FORMATS: tuple[WritingContentFormat, ...] = (
    "python",
    "markdown",
    "text",
    "json",
    "csv",
)
MAX_ARTIFACT_ITEMS = 8
MAX_EXTERNAL_EVIDENCE_REQUESTS = 3
MAX_LIMITATIONS = 8
MAX_REQUIRED_BEHAVIOR = 12
MAX_INTERFACES = 12
MAX_IMPORTS = 16
MAX_TEXT_FIELD_CHARS = 700
MAX_QUESTION_CHARS = 6000
PM_LLM_CALL_TIMEOUT_SECONDS = 300


WRITING_PM_PROMPT = '''\
You are the product manager for new-file code writing.
You receive one source-free coding request and decide which new artifacts are
needed. You do not write code and you do not edit existing source files.

# Inputs You Use
- question: the requested new script, file set, docs, tests, or small project.
- mode: create_new_project for Phase 2.
- external_evidence: public evidence already gathered for the request.
- acceptance_criteria: preserved user-visible requirements from the original
  request.
- previous_artifacts: generated artifacts from earlier programmer calls.
- validation_feedback: structural validation result from a previous package.
- alignment_feedback: semantic artifact-alignment result from a previous
  package.
- reservation_feedback: file reservation feedback from File Agent.

# Work Rules
1. Return need_programmers when new artifacts must be written.
2. Return need_external_evidence only when current public facts are required
   before artifact contracts can be written.
3. Return sufficient only when previous_artifacts already satisfy the request.
4. Return rejected for requests that require changing existing source files,
   applying patches, installing packages, running commands, credentials,
   private data, or unsafe file paths.
5. Each artifact item describes one new file or document. Do not include file
   paths. File Agent owns path reservation.
6. If the user names a file, place that exact name in preferred_name.
7. Use simple names and plain language. Keep interfaces and behavior explicit
   enough that one programmer can write the artifact without seeing peers.
8. Imports must be valid import statements, such as "import json" or
   "from collections import Counter". Use an empty list when no exact import is
   required. Programmers may add standard library imports if needed.
9. When validation_feedback is present, revise the artifact contracts so the
   next generated package resolves the reported mismatch. Keep all artifacts
   still needed for a complete package.
10. When alignment_feedback is present, revise the artifact contracts so the
    next generated package satisfies the preserved acceptance criteria. Keep
    the repair focused on the reported user-visible mismatch.
11. When acceptance_criteria are present, every criterion must be represented
    by one or more artifact contracts. Do not leave a criterion only in
    feature_goal or documentation when source or tests must implement it.
12. When a request asks for a command-line or externally runnable tool, assign
    an artifact contract that implements the runnable entry point and assign
    tests or documentation that exercise or explain that same entry point.
13. When one artifact consumes another artifact's interface, make that
    interface concrete. State the callable name, arguments, return shape or
    side effect, and any important empty/error behavior. The consuming artifact
    must repeat the complete provider contract, including return shape or side
    effect, instead of replacing it with a vague summary. Avoid contracts such
    as "returns a dictionary" unless the dictionary shape is also described.
14. When source and tests depend on an input file, data record, config object,
    or command-line arguments, define the shared input shape in the source and
    test contracts. Do not leave the input format only for README or docs.
    For file parsing, state the line or record format. For returned record
    dictionaries, state the required keys and value meaning. Examples may
    illustrate the contract, but they do not replace explicit required fields.

# Output Format
Return strict JSON:
{
  "status": "need_programmers | need_external_evidence | sufficient | rejected",
  "feature_goal": "short goal for the requested new artifact set",
  "artifact_items": [
    {
      "artifact_id": "stable short id",
      "file_label": "human-readable file label",
      "file_kind": "source | test | docs | config | data",
      "content_format": "python | markdown | text | json | csv",
      "purpose": "what this artifact must provide",
      "imports": ["valid required import statements"],
      "provided_interfaces": [{"name": "...", "kind": "...", "contract": "..."}],
      "consumed_interfaces": [{"name": "...", "provider": "...", "contract": "..."}],
      "required_behavior": ["specific behavior this artifact must satisfy"],
      "preferred_name": "user-specified filename or omit"
    }
  ],
  "selected_artifacts": [],
  "external_evidence_requests": [
    {
      "request_id": "stable short id",
      "task": "public evidence task",
      "reason": "why the evidence is needed"
    }
  ],
  "limitations": ["missing facts or rejected scope"]
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
    """Ask the writing PM to choose new artifacts or evidence needs.

    Args:
        pm_input: Compact source-free request, previous artifacts, and feedback.
        trace: Optional internal diagnostic dictionary populated with route,
            model, prompt input, raw output, parsed output, and prompt-budget
            metadata.

    Returns:
        A schema-normalized PM decision.
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
        decision = _rejected_decision(
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
        message_trace = [
            {
                "role": "system",
                "content": message.content,
            }
            for message in messages
        ]
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
    """Normalize PM JSON into the new-artifact decision contract."""

    if not isinstance(parsed, dict):
        decision = _rejected_decision("Writing PM returned malformed output.")
        return decision

    status = _bounded_text(parsed.get("status"))
    if status not in PM_STATUSES:
        status = "rejected"

    artifact_items = _artifact_items_from_parsed(parsed.get("artifact_items"))
    external_requests = _external_requests_from_parsed(
        parsed.get("external_evidence_requests"),
    )
    limitations = _string_list(parsed.get("limitations"), MAX_LIMITATIONS)

    if status == "need_programmers" and not artifact_items:
        status = "rejected"
        limitations.append("No artifact contracts were provided.")
    if status == "need_external_evidence" and not external_requests:
        status = "rejected"
        limitations.append("No external evidence requests were provided.")
    if status != "need_programmers":
        artifact_items = []
    if status != "need_external_evidence":
        external_requests = []
    if len(artifact_items) > MAX_ARTIFACT_ITEMS:
        status = "rejected"
        artifact_items = []
        limitations.append("Too many artifact contracts were requested.")

    decision: WritingPMDecision = {
        "status": status,
        "feature_goal": (
            _bounded_text(parsed.get("feature_goal")) or "new artifact request"
        ),
        "artifact_items": artifact_items,
        "selected_artifacts": _dict_list(parsed.get("selected_artifacts")),
        "external_evidence_requests": external_requests,
        "limitations": _dedupe_strings(limitations),
    }
    return decision


def _pm_payload(pm_input: WritingPMInput) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": _bounded_multiline_text(
            pm_input["question"],
            MAX_QUESTION_CHARS,
        ),
        "mode": pm_input["mode"],
        "external_evidence": _compact_external_evidence(
            pm_input["external_evidence"],
        ),
        "previous_artifacts": pm_input["previous_artifacts"],
    }
    acceptance_criteria = pm_input.get("acceptance_criteria")
    if acceptance_criteria is not None:
        payload["acceptance_criteria"] = acceptance_criteria
    validation_feedback = pm_input.get("validation_feedback")
    if validation_feedback is not None:
        payload["validation_feedback"] = validation_feedback
    alignment_feedback = pm_input.get("alignment_feedback")
    if alignment_feedback is not None:
        payload["alignment_feedback"] = alignment_feedback
    reservation_feedback = pm_input.get("reservation_feedback")
    if reservation_feedback is not None:
        payload["reservation_feedback"] = reservation_feedback
    return payload


def _compact_external_evidence(
    evidence: list[ExternalEvidenceSummary],
) -> list[dict[str, object]]:
    compact_rows: list[dict[str, object]] = []
    for row in evidence:
        compact_rows.append({
            "request_id": row["request_id"],
            "task": _bounded_text(row["task"]),
            "resolved": row["resolved"],
            "result": _bounded_multiline_text(row["result"], MAX_TEXT_FIELD_CHARS),
            "limitation": _bounded_text(row.get("limitation")),
        })
    return compact_rows


def _artifact_items_from_parsed(parsed: object) -> list[WritingArtifactItem]:
    if not isinstance(parsed, list):
        return []

    items: list[WritingArtifactItem] = []
    for index, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        artifact = _artifact_item_from_dict(item, index=index)
        if artifact is not None:
            items.append(artifact)
        if len(items) >= MAX_ARTIFACT_ITEMS + 1:
            break
    return items


def _artifact_item_from_dict(
    item: dict[str, Any],
    *,
    index: int,
) -> WritingArtifactItem | None:
    purpose = _bounded_text(item.get("purpose"))
    required_behavior = _string_list(
        item.get("required_behavior"),
        MAX_REQUIRED_BEHAVIOR,
    )
    if not purpose and not required_behavior:
        return None

    artifact_id = _bounded_text(item.get("artifact_id")) or f"artifact-{index}"
    file_kind = _bounded_text(item.get("file_kind"))
    if file_kind not in FILE_KINDS:
        file_kind = "source"
    content_format = _bounded_text(item.get("content_format"))
    if content_format not in CONTENT_FORMATS:
        content_format = _default_content_format(file_kind)

    if not required_behavior:
        required_behavior = [purpose]

    artifact: WritingArtifactItem = {
        "artifact_id": artifact_id,
        "file_label": _bounded_text(item.get("file_label")) or artifact_id,
        "file_kind": file_kind,
        "content_format": content_format,
        "purpose": purpose or artifact_id,
        "imports": _import_list(item.get("imports")),
        "provided_interfaces": _dict_list(
            item.get("provided_interfaces"),
            MAX_INTERFACES,
        ),
        "consumed_interfaces": _dict_list(
            item.get("consumed_interfaces"),
            MAX_INTERFACES,
        ),
        "required_behavior": required_behavior,
    }
    preferred_name = _bounded_text(item.get("preferred_name"))
    if preferred_name:
        artifact["preferred_name"] = preferred_name
    return artifact


def _default_content_format(file_kind: str) -> WritingContentFormat:
    if file_kind == "docs":
        return "markdown"
    if file_kind == "data":
        return "csv"
    if file_kind == "config":
        return "text"
    return "python"


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
        request_id = _bounded_text(item.get("request_id")) or f"external-{index}"
        request: WritingExternalEvidenceRequest = {
            "request_id": request_id,
            "task": task,
            "reason": _bounded_text(item.get("reason")),
        }
        requests.append(request)
        if len(requests) >= MAX_EXTERNAL_EVIDENCE_REQUESTS:
            break
    return requests


def _dict_list(value: object, limit: int = MAX_INTERFACES) -> list[dict[str, object]]:
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
                row[key] = _string_list(raw_value, MAX_REQUIRED_BEHAVIOR)
            else:
                row[key] = _bounded_text(raw_value)
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
        text = _bounded_text(item)
        if not text or text in strings:
            continue
        strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def _import_list(value: object) -> list[str]:
    imports: list[str] = []
    for text in _string_list(value, MAX_IMPORTS):
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


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped


def _rejected_decision(reason: str) -> WritingPMDecision:
    decision: WritingPMDecision = {
        "status": "rejected",
        "feature_goal": "new artifact request",
        "artifact_items": [],
        "selected_artifacts": [],
        "external_evidence_requests": [],
        "limitations": [reason],
    }
    return decision


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
