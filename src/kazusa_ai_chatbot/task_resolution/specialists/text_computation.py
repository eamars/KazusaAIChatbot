"""Bounded supplied-text and deterministic-computation specialist."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.complex_task_resolver.algorithmic import AlgorithmicSubagent
from kazusa_ai_chatbot.complex_task_resolver.contracts import (
    COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_LLM_API_KEY,
    BACKGROUND_WORK_LLM_BASE_URL,
    BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    BACKGROUND_WORK_LLM_MODEL,
    BACKGROUND_WORK_LLM_THINKING_ENABLED,
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionExecutionContextV1,
    TaskSpecialistRequestV1,
    TaskSpecialistResultV1,
)
from kazusa_ai_chatbot.task_resolution.specialists import (
    _caller_supplied_expression,
    _prompt_message_text,
    _require_handler_coding_objective_mode,
    _specialist_evidence,
    _specialist_result,
    _validated_handler_inputs,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


SPECIALIST = "text_computation"
MAX_TEXT_SOURCE_CHARS = 8000

TextComputationTaskType = Literal[
    "coding_snippet",
    "text_rewrite",
    "summary",
    "unsupported",
    "needs_user_input",
]


class TextComputationTaskRouterDecision(TypedDict):
    """Specialist-owned text task classification result."""

    task_type: TextComputationTaskType
    reason: str


class TextComputationGeneratorResult(TypedDict):
    """Specialist-owned generated artifact result."""

    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    artifact_text: str
    failure_summary: str
    result_summary: str


TEXT_COMPUTATION_TASK_ROUTER_PROMPT = '''\
Classify one supplied-text task into one task type.
Choose only the task type. Do not produce artifact text, code, rewrites,
summaries, low-level tool arguments, files, adapter data, persistence actions,
or delivery decisions.

# Task Types
- coding_snippet: produce a bounded code text snippet only.
- text_rewrite: rewrite or polish supplied text.
- summary: summarize supplied text or context.
- needs_user_input: required source text or constraints are missing.
- unsupported: the task asks for web research, repository work, attachments,
  filesystem, packages, shell, database, adapter, or side-effect work.

# Output Format
Return exactly this JSON object:
{"task_type": "one listed type", "reason": "short classification reason"}
'''

TEXT_COMPUTATION_GENERATOR_PROMPT = '''\
Generate one bounded text artifact for a validated supplied-text task.
The task type is already selected. Do not change it, choose specialists,
perform research, use tools, access files, mutate persistence, or deliver
adapter text.

# Rules
- For coding_snippet, return only a bounded code text snippet.
- For text_rewrite, return only the rewritten text.
- For summary, return only the summary.
- For missing source text or unsupported tasks, return a non-success status
  with empty artifact_text.
- Keep artifact_text within max_output_chars.

# Output Format
Return exactly this JSON object:
{"status": "succeeded | failed | needs_user_input | rejected", "artifact_text": "text when succeeded", "failure_summary": "reason when not succeeded", "result_summary": "short safe result summary"}
'''

_task_router_llm = LLInterface()
_generator_llm = LLInterface()
_task_router_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="BACKGROUND_WORK_LLM",
    base_url=BACKGROUND_WORK_LLM_BASE_URL,
    api_key=BACKGROUND_WORK_LLM_API_KEY,
    model=BACKGROUND_WORK_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=BACKGROUND_WORK_LLM_THINKING_ENABLED),
)
_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="BACKGROUND_WORK_LLM",
    base_url=BACKGROUND_WORK_LLM_BASE_URL,
    api_key=BACKGROUND_WORK_LLM_API_KEY,
    model=BACKGROUND_WORK_LLM_MODEL,
    temperature=0.2,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=BACKGROUND_WORK_LLM_THINKING_ENABLED),
)


async def resolve_with_text_computation(
    request: dict[str, object],
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskSpecialistResultV1:
    """Resolve a supplied-text task or explicit numeric expression in place."""

    task_request, context = _validated_handler_inputs(request, execution_context)
    _require_handler_coding_objective_mode(
        task_request,
        specialist=SPECIALIST,
    )
    expression = _caller_supplied_expression(context)
    if expression:
        return await _resolve_expression(task_request, expression)

    source_text = _prompt_message_text(context)
    max_output_chars = min(
        context["max_output_chars"],
        BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    )
    decision = await _route_text_task(
        task=task_request["objective"],
        source_text=source_text,
        max_output_chars=max_output_chars,
    )
    if decision["task_type"] == "unsupported":
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(task_request),
            reason="The text/computation specialist refused this task domain.",
        )
    if decision["task_type"] == "needs_user_input":
        return _specialist_result(
            specialist=SPECIALIST,
            status="needs_user_input",
            remaining_needs=["Provide the source text or required constraints."],
            reason="The text/computation specialist needs supplied text.",
        )
    generated = await _generate_text_artifact(
        task_type=decision["task_type"],
        task=task_request["objective"],
        source_text=source_text,
        max_output_chars=max_output_chars,
    )
    return _generated_result(task_request, context, generated)


async def _resolve_expression(
    request: TaskSpecialistRequestV1,
    expression: str,
) -> TaskSpecialistResultV1:
    """Run the existing deterministic evaluator for structured caller input."""

    calculation_request = {
        "schema_version": COMPLEX_TASK_SUBAGENT_REQUEST_VERSION,
        "node_id": request["task_node_id"],
        "subagent": "algorithmic",
        "action": "evaluate_expression",
        "objective": request["objective"],
        "payload": {
            "expression": expression,
            "label": "task-resolution result",
        },
        "constraints": {},
    }
    calculation = await AlgorithmicSubagent().run(
        calculation_request,
        {},
        max_attempts=1,
    )
    if calculation["status"] != "resolved":
        return _specialist_result(
            specialist=SPECIALIST,
            status="needs_user_input",
            remaining_needs=["Provide one valid caller-supplied numeric expression."],
            reason="The supplied numeric expression is not valid for calculation.",
        )
    calculation_result = calculation["result"]
    display = calculation_result.get("display")
    if not isinstance(display, str) or not display.strip():
        return _specialist_result(
            specialist=SPECIALIST,
            status="failed",
            remaining_needs=_remaining_needs(request),
            reason="The deterministic calculation did not return a safe result.",
        )
    evidence = _specialist_evidence(
        request=request,
        specialist=SPECIALIST,
        summary=display,
        provenance_refs=[f"caller_expression:{request['task_node_id']}"],
    )
    return _specialist_result(
        specialist=SPECIALIST,
        status="resolved",
        evidence=[evidence],
        completed_subgoals=[request["objective"]],
        reason="The caller-supplied expression was evaluated deterministically.",
    )


async def _route_text_task(
    *,
    task: str,
    source_text: str,
    max_output_chars: int,
) -> TextComputationTaskRouterDecision:
    """Run the first bounded LLM call that owns text-task classification."""

    payload = {
        "task": task,
        "source_text": source_text[:MAX_TEXT_SOURCE_CHARS],
        "max_output_chars": max_output_chars,
    }
    response = await _task_router_llm.ainvoke(
        [
            SystemMessage(content=TEXT_COMPUTATION_TASK_ROUTER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_task_router_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    return _normalize_task_router_result(parsed)


async def _generate_text_artifact(
    *,
    task_type: TextComputationTaskType,
    task: str,
    source_text: str,
    max_output_chars: int,
) -> TextComputationGeneratorResult:
    """Run the second bounded LLM call for an already-classified text task."""

    payload = {
        "task_type": task_type,
        "task": task,
        "source_text": source_text[:MAX_TEXT_SOURCE_CHARS],
        "max_output_chars": max_output_chars,
    }
    response = await _generator_llm.ainvoke(
        [
            SystemMessage(content=TEXT_COMPUTATION_GENERATOR_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_generator_llm_config,
    )
    raw_output = response.content
    parsed = parse_llm_json_output(raw_output)
    return _normalize_generator_result(
        parsed,
        task_type=task_type,
        raw_output=raw_output,
        max_output_chars=max_output_chars,
    )


def _generated_result(
    request: TaskSpecialistRequestV1,
    context: TaskResolutionExecutionContextV1,
    generated: TextComputationGeneratorResult,
) -> TaskSpecialistResultV1:
    """Map a bounded generator result into the canonical specialist outcome."""

    if generated["status"] == "succeeded" and generated["artifact_text"]:
        evidence = _specialist_evidence(
            request=request,
            specialist=SPECIALIST,
            summary=generated["artifact_text"],
            provenance_refs=[f"caller_text:{context['source_message_id']}"],
        )
        return _specialist_result(
            specialist=SPECIALIST,
            status="resolved",
            evidence=[evidence],
            completed_subgoals=[request["objective"]],
            reason="The supplied-text artifact was generated.",
        )
    if generated["status"] == "needs_user_input":
        return _specialist_result(
            specialist=SPECIALIST,
            status="needs_user_input",
            remaining_needs=["Provide the source text or required constraints."],
            reason="The supplied-text task needs additional user input.",
        )
    if generated["status"] == "rejected":
        return _specialist_result(
            specialist=SPECIALIST,
            status="incompatible",
            remaining_needs=_remaining_needs(request),
            reason="The supplied-text task is outside this specialist's scope.",
        )
    return _specialist_result(
        specialist=SPECIALIST,
        status="failed",
        remaining_needs=_remaining_needs(request),
        reason="The supplied-text artifact could not be generated.",
    )


def _normalize_task_router_result(
    value: object,
) -> TextComputationTaskRouterDecision:
    """Normalize the first LLM response into the closed task-type contract."""

    if not isinstance(value, dict):
        return {
            "task_type": "unsupported",
            "reason": "The text task router returned invalid output.",
        }
    task_type = value.get("task_type")
    if task_type not in {
        "coding_snippet",
        "text_rewrite",
        "summary",
        "unsupported",
        "needs_user_input",
    }:
        task_type = "unsupported"
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "The text task router did not provide a reason."
    return {
        "task_type": task_type,
        "reason": reason.strip()[:600],
    }


def _normalize_generator_result(
    value: object,
    *,
    task_type: TextComputationTaskType,
    raw_output: str,
    max_output_chars: int,
) -> TextComputationGeneratorResult:
    """Normalize the second LLM response into a bounded artifact envelope."""

    if not isinstance(value, dict) or not value:
        direct_artifact = _direct_artifact_text(
            task_type=task_type,
            raw_output=raw_output,
            max_output_chars=max_output_chars,
        )
        if direct_artifact:
            return {
                "status": "succeeded",
                "artifact_text": direct_artifact,
                "failure_summary": "",
                "result_summary": "Generated a bounded supplied-text artifact.",
            }
        return {
            "status": "failed",
            "artifact_text": "",
            "failure_summary": "The text generator returned invalid output.",
            "result_summary": "Text artifact generation failed.",
        }
    status = value.get("status")
    if status not in {"succeeded", "failed", "needs_user_input", "rejected"}:
        status = "failed"
    artifact_text = _optional_text(value.get("artifact_text"), max_output_chars)
    failure_summary = _optional_text(value.get("failure_summary"), 600)
    result_summary = _optional_text(value.get("result_summary"), 1200)
    if status != "succeeded":
        artifact_text = ""
    if status == "succeeded" and not artifact_text:
        status = "failed"
    return {
        "status": status,
        "artifact_text": artifact_text,
        "failure_summary": failure_summary,
        "result_summary": result_summary,
    }


def _direct_artifact_text(
    *,
    task_type: TextComputationTaskType,
    raw_output: str,
    max_output_chars: int,
) -> str:
    """Accept bounded direct output only for an already-approved text type."""

    if task_type not in {"coding_snippet", "text_rewrite", "summary"}:
        return ""
    text = raw_output.strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    return _optional_text(text, max_output_chars)


def _optional_text(value: object, maximum: int) -> str:
    """Return one clipped string while accepting an empty optional field."""

    if not isinstance(value, str):
        return ""
    return value.strip()[:maximum]


def _remaining_needs(request: TaskSpecialistRequestV1) -> list[str]:
    """Retain canonical unresolved needs after a specialist refusal or failure."""

    if request["remaining_needs"]:
        return list(request["remaining_needs"])
    return [request["objective"]]
