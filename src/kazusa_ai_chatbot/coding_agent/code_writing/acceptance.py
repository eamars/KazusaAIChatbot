"""LLM-backed acceptance and artifact-alignment checks for code writing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    PM_TARGET_INPUT_TOKEN_CAP,
    collect_artifact_ids,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    GeneratedArtifact,
    PatchValidationSummary,
    WritingAcceptanceCriterion,
    WritingAcceptanceResult,
    WritingAlignmentResult,
    WritingPMDecision,
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

ACCEPTANCE_LLM_CALL_TIMEOUT_SECONDS = 300
MAX_ACCEPTANCE_CRITERIA = 12
MAX_REASONS = 8
MAX_TEXT_FIELD_CHARS = 700
MAX_ARTIFACT_CONTENT_CHARS = 24000


ACCEPTANCE_CRITERIA_PROMPT = '''\
You preserve the user's requested outcome for new-file code writing.
You receive the original user request before any artifact planning.

# Work Rules
1. Extract user-visible requirements that final artifacts must satisfy.
2. Preserve external behavior, input/output behavior, documentation needs,
   config behavior, tests, data files, and project-shape requirements.
3. Keep each criterion concrete enough that another reviewer can inspect final
   artifacts against it.
4. Preserve the requested user-facing access method as its own criterion when
   the request asks for a tool, script, command-line program, web page,
   package, library, data file, or document set.
5. Do not plan files, write code, choose paths, run commands, or reject safe
   requests merely because implementation details are not fixed.
6. If the request is unsafe, private, requires credentials, or requires
   changing existing source files, return fail with a limitation.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "acceptance_criteria": [
    {
      "criterion_id": "short stable id",
      "requirement": "one user-visible requirement",
      "evidence_needed": "what final artifacts must show"
    }
  ],
  "limitations": ["blocking limitation, or empty list"]
}
'''

ARTIFACT_ALIGNMENT_PROMPT = '''\
You judge whether generated new-file artifacts satisfy the preserved user
requirements.

# Inputs You Use
- user_request: the original request.
- acceptance_criteria: preserved user-visible requirements.
- pm_decision: artifact contracts chosen by the writing PM.
- generated_artifacts: final generated artifact contents and paths.
- validation: structural patch validation result.

# Review Rules
1. Judge final artifacts against acceptance_criteria and user_request.
2. A README or final answer can explain behavior, but it does not prove the
   behavior exists unless source or data artifacts implement it.
3. If a requirement asks for an externally usable behavior, the generated
   source artifact must expose a concrete way to use that behavior.
4. The requested user-facing access method is a required behavior. Do not
   treat a missing access method as a minor improvement when it is part of the
   request.
5. An orchestration helper function is not enough by itself when the request
   asks for a user-runnable artifact. The generated artifacts must show how the
   user can invoke that behavior through the requested access method.
6. If tests are requested, generated tests must exercise the requested
   behavior, not only helper functions that avoid the main behavior.
7. If feedback_for_pm names a required repair for a user-visible requirement,
   the status must be fail and the repair must also appear in blockers.
8. Do not require one exact implementation when another implementation
   satisfies the same requirement.
9. Do not run code, infer hidden files, assume package installation, or accept
   behavior that is only promised outside the generated artifacts.
10. Fail when an important user-visible requirement is missing, contradicted,
   or present only as documentation without implementation.

# Output Format
Return strict JSON:
{
  "status": "pass | fail",
  "confidence": 0,
  "request_satisfied": true,
  "reasons": ["short reason"],
  "blockers": ["blocking mismatch, or empty list"],
  "feedback_for_pm": "concise repair instruction for the writing PM"
}
'''

_acceptance_llm = LLInterface()
_acceptance_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.acceptance",
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


async def derive_acceptance_criteria(
    *,
    question: str,
    trace: dict[str, object] | None = None,
) -> WritingAcceptanceResult:
    """Extract preserved user-visible acceptance criteria."""

    payload = {"user_request": _bounded_multiline_text(question, 6000)}
    raw_output, parsed, context_budget, blocked = await _invoke_json_stage(
        system_prompt=ACCEPTANCE_CRITERIA_PROMPT,
        payload=payload,
    )
    if blocked:
        result = _acceptance_failed("Acceptance prompt exceeded context budget.")
    else:
        result = _normalize_acceptance_result(parsed)
    _fill_trace(
        trace,
        system_prompt=ACCEPTANCE_CRITERIA_PROMPT,
        human_payload=payload,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=result,
        context_budget=context_budget,
        blocked_before_invoke=blocked,
    )
    return result


async def evaluate_artifact_alignment(
    *,
    question: str,
    acceptance_criteria: list[WritingAcceptanceCriterion],
    pm_decision: WritingPMDecision,
    generated_artifacts: list[GeneratedArtifact],
    validation: PatchValidationSummary,
    trace: dict[str, object] | None = None,
) -> WritingAlignmentResult:
    """Judge final generated artifacts against preserved criteria."""

    payload = {
        "user_request": _bounded_multiline_text(question, 6000),
        "acceptance_criteria": acceptance_criteria,
        "pm_decision": _compact_pm_decision(pm_decision),
        "generated_artifacts": _compact_generated_artifacts(generated_artifacts),
        "validation": validation,
    }
    raw_output, parsed, context_budget, blocked = await _invoke_json_stage(
        system_prompt=ARTIFACT_ALIGNMENT_PROMPT,
        payload=payload,
    )
    if blocked:
        result = _alignment_failed("Alignment prompt exceeded context budget.")
    else:
        result = _normalize_alignment_result(parsed)
    _fill_trace(
        trace,
        system_prompt=ARTIFACT_ALIGNMENT_PROMPT,
        human_payload=payload,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=result,
        context_budget=context_budget,
        blocked_before_invoke=blocked,
    )
    return result


async def _invoke_json_stage(
    *,
    system_prompt: str,
    payload: dict[str, object],
) -> tuple[str, dict[str, object], dict[str, object], bool]:
    payload_text = json.dumps(payload, ensure_ascii=False)
    context_budget = prompt_budget_metadata(
        system_prompt=system_prompt,
        payload_text=payload_text,
        target_input_tokens=PM_TARGET_INPUT_TOKEN_CAP,
        artifact_ids=collect_artifact_ids(payload),
    )
    if context_budget["over_hard_cap"]:
        return "", {}, context_budget, True

    try:
        response = await asyncio.wait_for(
            _acceptance_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=payload_text),
                ],
                config=_acceptance_llm_config,
            ),
            timeout=ACCEPTANCE_LLM_CALL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return "", {}, context_budget, False

    raw_output = response.content
    parsed = parse_llm_json_output(raw_output)
    return raw_output, parsed, context_budget, False


def _normalize_acceptance_result(parsed: object) -> WritingAcceptanceResult:
    if not isinstance(parsed, dict):
        return _acceptance_failed("Acceptance owner returned malformed output.")

    status = _bounded_text(parsed.get("status"))
    criteria = _criteria_from_parsed(parsed.get("acceptance_criteria"))
    limitations = _string_list(parsed.get("limitations"), MAX_REASONS)
    if status != "pass" or not criteria:
        return {
            "status": "fail",
            "acceptance_criteria": criteria,
            "limitations": limitations or [
                "Acceptance owner did not preserve usable criteria.",
            ],
        }
    return {
        "status": "pass",
        "acceptance_criteria": criteria,
        "limitations": limitations,
    }


def _normalize_alignment_result(parsed: object) -> WritingAlignmentResult:
    if not isinstance(parsed, dict):
        return _alignment_failed("Alignment owner returned malformed output.")

    status = _bounded_text(parsed.get("status"))
    confidence = _confidence(parsed.get("confidence"))
    request_satisfied = parsed.get("request_satisfied") is True
    blockers = _string_list(parsed.get("blockers"), MAX_REASONS)
    reasons = _string_list(parsed.get("reasons"), MAX_REASONS)
    feedback = _bounded_multiline_text(parsed.get("feedback_for_pm"), 1200)
    if status != "pass" or not request_satisfied or blockers:
        return {
            "status": "fail",
            "confidence": confidence,
            "request_satisfied": False,
            "reasons": reasons,
            "blockers": blockers or [
                "Generated artifacts do not satisfy preserved requirements.",
            ],
            "feedback_for_pm": feedback or (
                "Revise artifact contracts so generated artifacts satisfy the "
                "preserved user-visible requirements."
            ),
        }
    return {
        "status": "pass",
        "confidence": confidence,
        "request_satisfied": True,
        "reasons": reasons,
        "blockers": [],
        "feedback_for_pm": feedback,
    }


def _criteria_from_parsed(value: object) -> list[WritingAcceptanceCriterion]:
    if not isinstance(value, list):
        return []

    criteria: list[WritingAcceptanceCriterion] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        requirement = _bounded_multiline_text(item.get("requirement"), 1000)
        evidence_needed = _bounded_multiline_text(item.get("evidence_needed"), 1000)
        if not requirement or not evidence_needed:
            continue
        criterion: WritingAcceptanceCriterion = {
            "criterion_id": (
                _bounded_text(item.get("criterion_id")) or f"criterion_{index}"
            ),
            "requirement": requirement,
            "evidence_needed": evidence_needed,
        }
        criteria.append(criterion)
        if len(criteria) >= MAX_ACCEPTANCE_CRITERIA:
            break
    return criteria


def _compact_pm_decision(decision: WritingPMDecision) -> dict[str, object]:
    compact_decision = {
        "status": decision["status"],
        "reason": decision["reason"],
        "completion_report": decision["completion_report"],
        "blocker": decision["blocker"],
    }
    return compact_decision


def _compact_generated_artifacts(
    artifacts: list[GeneratedArtifact],
) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    remaining_chars = MAX_ARTIFACT_CONTENT_CHARS
    for artifact in artifacts:
        content = artifact["content"]
        if len(content) > remaining_chars:
            content = content[:remaining_chars].rstrip()
        remaining_chars = max(remaining_chars - len(content), 0)
        compact.append({
            "artifact_id": artifact["artifact_id"],
            "file_label": artifact["file_label"],
            "file_kind": artifact["file_kind"],
            "content_format": artifact["content_format"],
            "path": artifact["path"],
            "purpose": artifact["purpose"],
            "content": content,
        })
        if remaining_chars <= 0:
            break
    return compact


def _acceptance_failed(reason: str) -> WritingAcceptanceResult:
    return {
        "status": "fail",
        "acceptance_criteria": [],
        "limitations": [reason],
    }


def _alignment_failed(reason: str) -> WritingAlignmentResult:
    return {
        "status": "fail",
        "confidence": 0,
        "request_satisfied": False,
        "reasons": [],
        "blockers": [reason],
        "feedback_for_pm": reason,
    }


def _confidence(value: object) -> int:
    if not isinstance(value, int | float):
        return 0
    confidence = float(value)
    if 0 <= confidence <= 1:
        confidence *= 100
    confidence = max(0, min(100, confidence))
    return int(round(confidence))


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
    raw_output: str,
    parsed_output: dict[str, object],
    normalized_output: dict[str, object],
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _acceptance_llm_config.route_name
    trace["model"] = _acceptance_llm_config.model
    trace["thinking_enabled"] = CODING_AGENT_PM_LLM_THINKING_ENABLED
    trace["system_prompt"] = system_prompt
    trace["human_payload"] = human_payload
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = normalized_output
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
