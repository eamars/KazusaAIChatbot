"""Text-artifact worker with separate task routing and generation stages."""

from __future__ import annotations

import json
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkResult,
    BackgroundWorkWorkerDecision,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_LLM_API_KEY,
    BACKGROUND_WORK_LLM_BASE_URL,
    BACKGROUND_WORK_LLM_MODEL,
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    BACKGROUND_WORK_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
WORKER = "text_artifact"
DESCRIPTION = (
    "Bounded text artifacts including compact code snippets, text rewrites, "
    "and summaries."
)

TextArtifactTaskType = Literal[
    "coding_snippet",
    "text_rewrite",
    "summary",
    "unsupported",
    "needs_user_input",
]


class TextArtifactTaskRouterDecision(TypedDict):
    """Worker-local task classification output."""

    task_type: TextArtifactTaskType
    reason: str


class TextArtifactGeneratorResult(TypedDict):
    """Worker-local artifact generation output."""

    status: Literal["succeeded", "failed", "needs_user_input", "rejected"]
    artifact_text: str
    failure_summary: str
    result_summary: str


TEXT_ARTIFACT_TASK_ROUTER_PROMPT = '''\
You classify a routed text-artifact task into one task type.
Choose only the task type. Do not produce artifact text, code, rewrites,
summaries, cleaned task briefs, tool arguments, files, adapter data, delivery
decisions, or job ids.

# Task Types
- coding_snippet: produce a bounded code text snippet only.
- text_rewrite: rewrite or polish provided text.
- summary: summarize provided text or context.
- needs_user_input: required source text or constraints are missing.
- unsupported: the task asks for execution, files, shell, packages, web
  research, images, attachments, or repository mutation.

# Output Format
Return strict JSON:
{
  "task_type": "coding_snippet | text_rewrite | summary | unsupported | needs_user_input",
  "reason": "short classification reason"
}
'''

_llm_interface = LLInterface()
_text_artifact_task_router_llm = LLInterface()
_text_artifact_generator_llm = LLInterface()
_text_artifact_task_router_llm_config = LLMCallConfig(
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
    thinking=LLMThinkingConfig(
        enabled=BACKGROUND_WORK_LLM_THINKING_ENABLED,
    ),
)


async def _route_text_artifact_task(
    *,
    task: str,
    source_summary: str,
    max_output_chars: int,
) -> TextArtifactTaskRouterDecision:
    """Classify one routed text-artifact task before generation."""

    payload = build_text_artifact_task_router_payload(
        task=task,
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    response = await _text_artifact_task_router_llm.ainvoke([
        SystemMessage(content=TEXT_ARTIFACT_TASK_ROUTER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_text_artifact_task_router_llm_config)
    parsed = parse_llm_json_output(response.content)
    decision = normalize_text_artifact_task_router_output(parsed)
    return decision


def build_text_artifact_task_router_payload(
    *,
    task: str,
    source_summary: str,
    max_output_chars: int,
) -> dict[str, object]:
    """Build the prompt payload for text-artifact task classification."""

    payload: dict[str, object] = {
        "task": task,
        "source_summary": source_summary,
        "max_output_chars": max_output_chars,
    }
    return payload


def normalize_text_artifact_task_router_output(
    parsed: object,
) -> TextArtifactTaskRouterDecision:
    """Normalize worker-local task routing without artifact fields."""

    if not isinstance(parsed, dict):
        result: TextArtifactTaskRouterDecision = {
            "task_type": "unsupported",
            "reason": "Text-artifact task router returned malformed output.",
        }
        return result

    task_type = _bounded_text(parsed.get("task_type"))
    if task_type not in (
        "coding_snippet",
        "text_rewrite",
        "summary",
        "unsupported",
        "needs_user_input",
    ):
        task_type = "unsupported"
    reason = _bounded_text(parsed.get("reason"))
    if not reason:
        reason = "No task-router reason was provided."
    decision: TextArtifactTaskRouterDecision = {
        "task_type": task_type,
        "reason": reason,
    }
    return decision


TEXT_ARTIFACT_GENERATOR_PROMPT = '''\
You generate one bounded text artifact for a validated text-artifact task.
The task type has already been selected by another stage. Do not change the
task type, choose workers, dispatch providers, deliver adapter text, call
cognition, or mutate persistence.

# Generation Rules
- For coding_snippet, return code text only when the request is bounded.
- For text_rewrite, return the rewritten text only.
- For summary, return the summary only.
- If the validated task is unsupported or missing required source text, return
  a failed, rejected, or needs_user_input status with no artifact_text.
- Keep artifact_text within max_output_chars.

# Output Format
Return strict JSON:
{
  "status": "succeeded | failed | needs_user_input | rejected",
  "artifact_text": "artifact text when succeeded, otherwise empty",
  "failure_summary": "failure reason when not succeeded, otherwise empty",
  "result_summary": "short prompt-safe summary of the result"
}
'''

_text_artifact_generator_llm_config = LLMCallConfig(
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
    thinking=LLMThinkingConfig(
        enabled=BACKGROUND_WORK_LLM_THINKING_ENABLED,
    ),
)


async def _generate_text_artifact(
    *,
    task_decision: TextArtifactTaskRouterDecision,
    task: str,
    source_summary: str,
    max_output_chars: int,
) -> TextArtifactGeneratorResult:
    """Generate one artifact from a validated worker-local task decision."""

    payload = build_text_artifact_generator_payload(
        task_type=task_decision["task_type"],
        task=task,
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    response = await _text_artifact_generator_llm.ainvoke([
        SystemMessage(content=TEXT_ARTIFACT_GENERATOR_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_text_artifact_generator_llm_config)
    parsed = parse_llm_json_output(response.content)
    result = normalize_text_artifact_generator_output(
        parsed,
        max_output_chars=max_output_chars,
    )
    return result


def build_text_artifact_generator_payload(
    *,
    task_type: str,
    task: str,
    source_summary: str,
    max_output_chars: int,
) -> dict[str, object]:
    """Build the prompt payload for artifact generation."""

    payload: dict[str, object] = {
        "task_type": task_type,
        "task": task,
        "source_summary": source_summary,
        "max_output_chars": max_output_chars,
    }
    return payload


def normalize_text_artifact_generator_output(
    parsed: object,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> TextArtifactGeneratorResult:
    """Normalize artifact generation without task-type selection fields."""

    if not isinstance(parsed, dict):
        result: TextArtifactGeneratorResult = {
            "status": "failed",
            "artifact_text": "",
            "failure_summary": "Text-artifact generator returned malformed output.",
            "result_summary": "Text artifact generation failed.",
        }
        return result

    status = _bounded_text(parsed.get("status"))
    if status not in ("succeeded", "failed", "needs_user_input", "rejected"):
        status = "failed"
    artifact_text = _bounded_text(
        parsed.get("artifact_text"),
        limit=max_output_chars,
    )
    failure_summary = _bounded_text(parsed.get("failure_summary"))
    result_summary = _bounded_text(parsed.get("result_summary"))
    if status != "succeeded":
        artifact_text = ""
        if not failure_summary:
            failure_summary = "Text artifact generation did not succeed."
    if status == "succeeded" and not result_summary:
        result_summary = "Text artifact generated."
    result: TextArtifactGeneratorResult = {
        "status": status,
        "artifact_text": artifact_text,
        "failure_summary": failure_summary,
        "result_summary": result_summary,
    }
    return result


async def execute(
    decision: BackgroundWorkWorkerDecision,
    *,
    max_output_chars: int = BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
) -> BackgroundWorkResult:
    """Run the text-artifact worker after background-work routing."""

    task = _bounded_text(decision.get("task_brief"))
    source_summary = _bounded_text(decision.get("source_summary"))
    if not task:
        task = source_summary
    if not source_summary:
        source_summary = task
    task_decision = await _route_text_artifact_task(
        task=task,
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    if task_decision["task_type"] in ("unsupported", "needs_user_input"):
        status = (
            "needs_user_input"
            if task_decision["task_type"] == "needs_user_input"
            else "rejected"
        )
        result: BackgroundWorkResult = {
            "status": status,
            "worker": WORKER,
            "artifact_text": "",
            "failure_summary": task_decision["reason"],
            "result_summary": task_decision["reason"],
            "worker_metadata": {"task_type": task_decision["task_type"]},
        }
        return result

    generator_result = await _generate_text_artifact(
        task_decision=task_decision,
        task=task,
        source_summary=source_summary,
        max_output_chars=max_output_chars,
    )
    result = {
        "status": generator_result["status"],
        "worker": WORKER,
        "artifact_text": generator_result["artifact_text"],
        "failure_summary": generator_result["failure_summary"],
        "result_summary": generator_result["result_summary"],
        "worker_metadata": {"task_type": task_decision["task_type"]},
    }
    return result


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    """Return stripped external LLM text capped to a local bound."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()[:limit]
    return return_value
