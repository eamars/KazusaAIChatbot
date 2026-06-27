"""Final answer synthesis for new-artifact patch proposals."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.context_budget import (
    SYNTHESIS_TARGET_INPUT_TOKEN_CAP,
    collect_artifact_ids,
    collect_selected_evidence_refs,
    prompt_budget_metadata,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ExternalEvidenceSummary,
    GeneratedArtifact,
    PatchArtifact,
    PatchValidationSummary,
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

DEFAULT_MAX_ANSWER_CHARS = 4000
MAX_TEXT_FIELD_CHARS = 6000
MAX_LIMITATIONS_IN_ANSWER = 3
NO_EXECUTION_LIMITATION = "Generated artifacts were not executed in the target project."
NO_LIMITATION_CLAIM_RE = re.compile(
    r"[^.!?\n]*(?:no reported limitations|no limitations|no missing information)"
    r"[^.!?\n]*[.!?]?",
    flags=re.IGNORECASE,
)


WRITING_SYNTHESIS_PROMPT = '''\
You synthesize the final user-facing answer for a new-artifact code-writing
proposal. Use only the PM decision, generated artifact manifest, review
package summary, external evidence summaries, and limitations provided in the
payload.

# Synthesis Rules
- State that the output is a patch proposal, not an applied mutation.
- Summarize the generated artifacts and review-package materialization status.
- Treat the package status as evidence that artifacts were materialized for
  review only. Do not describe it as generated-code validation.
- Generated tests are proposal artifacts, not executed proof.
- Mention limitations and missing information.
- If limitations are present, do not say there are no limitations.
- Mention external evidence only when it is present.
- Do not claim that tests, package installation, Docker, or target project
  commands were run.
- Do not expose local roots, storage roots, cache keys, API keys, raw provider
  objects, or full source files.
- Keep the answer in the user's requested language when that is clear.

# Output Format
Return strict JSON:
{
  "answer_text": "final answer text",
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
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def synthesize_patch_proposal(
    *,
    question: str,
    pm_decision: WritingPMDecision,
    generated_artifacts: list[GeneratedArtifact],
    patch_artifacts: list[PatchArtifact],
    validation: PatchValidationSummary,
    external_evidence: list[ExternalEvidenceSummary],
    limitations: list[str],
    preferred_language: str | None,
    max_answer_chars: int,
    trace: dict[str, object] | None = None,
) -> tuple[str, list[str]]:
    """Create the final answer from generated artifacts and validation."""

    if not patch_artifacts:
        answer = "No patch proposal artifact was produced."
        result_limitations = [
            *limitations,
            "No patch artifacts were available for synthesis.",
        ]
        return answer, result_limitations

    capability_limitations = _dedupe_strings([
        *limitations,
        NO_EXECUTION_LIMITATION,
    ])
    payload = _synthesis_payload(
        question=question,
        pm_decision=pm_decision,
        generated_artifacts=generated_artifacts,
        patch_artifacts=patch_artifacts,
        validation=validation,
        external_evidence=external_evidence,
        limitations=capability_limitations,
        preferred_language=preferred_language,
        max_answer_chars=max_answer_chars,
    )
    payload_text = json.dumps(payload, ensure_ascii=False)
    context_budget = prompt_budget_metadata(
        system_prompt=WRITING_SYNTHESIS_PROMPT,
        payload_text=payload_text,
        target_input_tokens=SYNTHESIS_TARGET_INPUT_TOKEN_CAP,
        selected_evidence_refs=collect_selected_evidence_refs(payload),
        artifact_ids=collect_artifact_ids(payload),
    )
    if context_budget["over_hard_cap"]:
        answer_text = (
            "Patch artifacts were produced, but synthesis exceeded the context "
            "budget."
        )
        combined_limitations = _dedupe_strings([
            *capability_limitations,
            "Synthesis prompt exceeded the context budget.",
        ])
        _fill_trace(
            trace,
            raw_output="",
            parsed_output={},
            normalized_answer=answer_text,
            limitations=combined_limitations,
            context_budget=context_budget,
            blocked_before_invoke=True,
        )
        return answer_text, combined_limitations

    response = await _synthesis_llm.ainvoke([
        SystemMessage(content=WRITING_SYNTHESIS_PROMPT),
        HumanMessage(content=payload_text),
    ], config=_synthesis_llm_config)
    parsed = parse_llm_json_output(response.content)
    answer_text, parsed_limitations = normalize_synthesis_output(
        parsed,
        max_answer_chars=max_answer_chars,
    )
    combined_limitations = _dedupe_strings([
        *capability_limitations,
        *parsed_limitations,
    ])
    answer_text = _answer_with_required_limitations(
        answer_text,
        combined_limitations,
        max_answer_chars=max_answer_chars,
    )
    _fill_trace(
        trace,
        raw_output=response.content,
        parsed_output=parsed,
        normalized_answer=answer_text,
        limitations=combined_limitations,
        context_budget=context_budget,
        blocked_before_invoke=False,
    )
    return answer_text, combined_limitations


def _answer_with_required_limitations(
    answer_text: str,
    limitations: list[str],
    *,
    max_answer_chars: int,
) -> str:
    if not limitations:
        return answer_text

    cleaned_answer = NO_LIMITATION_CLAIM_RE.sub("", answer_text).strip()
    missing_limitations = _missing_visible_limitations(
        cleaned_answer,
        limitations[:MAX_LIMITATIONS_IN_ANSWER],
    )
    if not missing_limitations:
        updated_answer = cleaned_answer
    else:
        limitation_text = "Limitations: " + "; ".join(missing_limitations)
        if cleaned_answer:
            updated_answer = f"{cleaned_answer}\n\n{limitation_text}"
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


def _visibility_key(value: str) -> str:
    key = re.sub(r"\s+", " ", value.casefold()).strip().rstrip(".;")
    return key


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
        answer_text = (
            "Patch artifacts were produced, but no final summary was returned."
        )
    limitations = _string_list(parsed.get("limitations"), 8)
    return answer_text, limitations


def _synthesis_payload(
    *,
    question: str,
    pm_decision: WritingPMDecision,
    generated_artifacts: list[GeneratedArtifact],
    patch_artifacts: list[PatchArtifact],
    validation: PatchValidationSummary,
    external_evidence: list[ExternalEvidenceSummary],
    limitations: list[str],
    preferred_language: str | None,
    max_answer_chars: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "question": question,
        "preferred_language": preferred_language,
        "max_answer_chars": max_answer_chars,
        "pm_decision": {
            "status": pm_decision["status"],
            "reason": pm_decision["reason"],
            "completion_report": pm_decision["completion_report"],
            "blocker": pm_decision["blocker"],
        },
        "generated_artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "file_label": artifact["file_label"],
                "file_kind": artifact["file_kind"],
                "content_format": artifact["content_format"],
                "path": artifact["path"],
                "content_chars": len(artifact["content"]),
                "purpose": artifact["purpose"],
            }
            for artifact in generated_artifacts
        ],
        "patch_artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "base": artifact["base"],
                "files": artifact["files"],
                "summary": artifact["summary"],
            }
            for artifact in patch_artifacts
        ],
        "validation": validation,
        "external_evidence": external_evidence,
        "limitations": limitations,
    }
    return payload


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
    seen_keys: set[str] = set()
    for value in values:
        key = _dedupe_key(value)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(value)
    return deduped


def _dedupe_key(value: str) -> str:
    normalized = _visibility_key(value)
    if normalized.startswith("the "):
        normalized = normalized.removeprefix("the ")
    return normalized


def _trim_sentence_punctuation(value: str) -> str:
    trimmed = value.strip().rstrip(".;")
    return trimmed


def _bounded_text(value: object, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _fill_trace(
    trace: dict[str, object] | None,
    *,
    raw_output: str,
    parsed_output: dict[str, Any],
    normalized_answer: str,
    limitations: list[str],
    context_budget: dict[str, object],
    blocked_before_invoke: bool,
) -> None:
    if trace is None:
        return

    trace["effective_route"] = _synthesis_llm_config.route_name
    trace["model"] = _synthesis_llm_config.model
    trace["thinking_enabled"] = CODING_AGENT_PM_LLM_THINKING_ENABLED
    trace["context_budget"] = context_budget
    trace["blocked_before_invoke"] = blocked_before_invoke
    trace["raw_output"] = raw_output
    trace["parsed_output"] = parsed_output
    trace["normalized_output"] = {
        "answer_text": normalized_answer,
        "limitations": limitations,
    }
