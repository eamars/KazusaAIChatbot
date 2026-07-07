"""LLM-backed programmer role for existing-source modification artifacts."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    ModificationArtifact,
    normalize_modification_artifact,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    CODING_AGENT_PROGRAMMER_LLM_MODEL,
    CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

MODIFYING_PROGRAMMER_PROMPT = '''\
You are the existing-source modification programmer inside a coding agent.

You receive a user request, bounded source evidence, and current file text.
Return one strict JSON object only. Do not include Markdown, prose before JSON,
raw unified diffs, or command output. Do not claim that tests or commands were
run.

Allowed operation_kind values:
- replace
- insert_before
- insert_after
- replace_file_small

For replace, insert_before, and insert_after, exact_anchor must be copied
exactly from the current file text. Use replace_file_small only for small files
where a full replacement is clearer than a section anchor.
Exact means byte-for-byte substring: preserve indentation, trailing newlines,
blank lines between functions, comments, and decorators. If the edit spans
multiple adjacent functions and exact blank-line copying is uncertain, use
replace_file_small for that file instead of a shortened anchor.

Every succeeded artifact must:
- target one path from file_contexts
- include at least one evidence_ids value copied from evidence[*].evidence_id
- include replacement_or_insert_content as the final inserted/replacement text,
  not a diff hunk
- avoid adding broad exception handlers such as bare except, except Exception,
  or except BaseException; preserve existing error handling or catch a specific
  observed exception type only
- use status "blocked" with blocker text only when the requested change cannot
  be localized to the provided files

Return strict JSON:
{
  "artifacts": [
    {
      "artifact_id": "short id",
      "status": "succeeded | blocked",
      "task_id": "short task id",
      "target_path": "repo-relative path",
      "evidence_ids": ["ev-1"],
      "operation_kind": "replace | insert_before | insert_after | replace_file_small",
      "exact_anchor": "exact source text anchor or empty for replace_file_small",
      "replacement_or_insert_content": "new text",
      "operation_summary": "short summary",
      "risk_notes": ["risk or uncertainty"],
      "tests_or_docs_to_update": ["related path"]
    }
  ],
  "answer_text": "short proposal summary",
  "limitations": ["limitation"]
}
'''

_modifying_programmer_llm = LLInterface()
_modifying_programmer_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="CODING_AGENT_PROGRAMMER_LLM",
    base_url=CODING_AGENT_PROGRAMMER_LLM_BASE_URL,
    api_key=CODING_AGENT_PROGRAMMER_LLM_API_KEY,
    model=CODING_AGENT_PROGRAMMER_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=CODING_AGENT_PROGRAMMER_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    timeout_seconds=300,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PROGRAMMER_LLM_THINKING_ENABLED,
    ),
)


async def run_modifying_programmer(
    payload: dict[str, object],
) -> dict[str, object]:
    """Generate structured existing-file modification artifacts."""

    payload_text = json.dumps(payload, ensure_ascii=False)
    messages = [
        SystemMessage(content=MODIFYING_PROGRAMMER_PROMPT),
        HumanMessage(content=payload_text),
    ]
    response = await _modifying_programmer_llm.ainvoke(
        messages,
        config=_modifying_programmer_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    result = normalize_modifying_programmer_response(parsed)
    result["raw_output"] = response.content
    return result


def normalize_modifying_programmer_response(
    parsed: object,
) -> dict[str, object]:
    """Normalize raw programmer JSON into validated artifacts."""

    if not isinstance(parsed, dict):
        result = {
            "artifacts": [],
            "answer_text": "",
            "limitations": ["Modifying programmer did not return JSON object."],
        }
        return result

    raw_artifacts = parsed.get("artifacts")
    artifacts: list[ModificationArtifact] = []
    if isinstance(raw_artifacts, list):
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, dict):
                continue
            artifact = normalize_modification_artifact(raw_artifact)
            artifacts.append(artifact)

    answer_text = parsed.get("answer_text")
    if not isinstance(answer_text, str):
        answer_text = ""
    limitations = _string_list(parsed.get("limitations"))
    result = {
        "artifacts": artifacts,
        "answer_text": answer_text,
        "limitations": limitations,
    }
    return result


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        strings.append(text)
    return strings
