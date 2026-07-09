"""LLM-backed product-manager role for existing-source modification."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    ModifyingPMDecision,
    normalize_modifying_pm_decision,
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

MODIFYING_PM_LLM_TIMEOUT_SECONDS = 300

MODIFYING_PM_PROMPT = '''\
You are the product-manager role inside an existing-source coding agent.

You receive a user request, source-reading evidence, and a deterministic file
plan. Your job is to choose the next lifecycle action for review-only source
modification. You may ask one programmer to produce structured modification
artifacts for existing files and new source files that belong in the same
existing-source patch. You do not write code yourself.

# Decision Rules
1. Prefer create_programmer_task when the next useful change can be bounded to
   explicit target_paths from file_plan.owned_path_candidates plus any focused
   companion tests or documentation from file_plan.test_or_doc_path_candidates.
2. The first programmer task must include at least one runtime source owner
   path when file_plan.owned_path_candidates is not empty. Do not start with
   tests or README only.
3. For multi-file behavior, include all owner files that must change together
   when splitting them would create inconsistent runtime behavior. Put focused
   tests and docs in target_paths when the user explicitly asked for them or
   when stale assertions/docs are likely.
   When file_plan.test_or_doc_path_candidates is not empty, the default
   source-change task should include the focused companion tests or docs unless
   the user request is strictly source-only. Do not omit companion tests or docs
   merely because a source-only patch can be generated.
   If the user request or requirements say not to modify tests, docs, README,
   or provided verification files, put those files in read_only_paths and keep
   them out of programmer_task.target_paths.
   If the runtime change needs a new helper, module, adapter, or source owner
   plus an existing caller edit, include both the new repo-relative source path
   and the existing caller/source path in one programmer_task.target_paths list.
   Use expected_operations containing create_file for the new path and replace,
   insert_before, insert_after, or replace_file_small for existing paths.
4. Use read_only_paths for evidence files the programmer may inspect but should
   not change in this task.
5. Choose repair_child only for structural feedback from the supervisor such as
   handoff_validation, parser_validation, patch_validation, review_materialization,
   contract_validation, or execution_verification. For execution_verification,
   use only structured failure summaries, failed paths, required owner paths,
   and protected verification paths. Never use raw command output or raw
   executed test output as repair feedback. Include required_source_owner_paths
   in programmer_task.target_paths. Add source collaborator or caller paths,
   such as CLI wiring files, when the failure is in integration handoff. Treat
   protected_verification_paths and failed_paths as read-only evidence; they
   must not appear in programmer_task.target_paths unless they are also
   required_source_owner_paths. When handoff_validation supplies
   allowed_source_target_paths, choose programmer_task.target_paths only from
   that list. README, docs, and tests are read-only context during
   execution_verification unless they are explicitly listed in
   allowed_source_target_paths.
6. Choose complete only when previous_programmer_reports show enough produced
   artifacts for the assigned user request.
7. Choose blocked when the request cannot be localized to the available source
   evidence and file plan. Choose request_information when a narrower source
   reading request could unblock the next programmer task.

Do not claim that commands, tests, patch application, package installation,
network calls, or repository mutation happened. Do not emit raw diffs.

# Output Format
Return strict JSON with these top-level fields:
{
  "status": "request_information | create_programmer_task | repair_child | complete | blocked",
  "reason": "short reason",
  "owned_paths": ["repo-relative paths this PM owns for the next task"],
  "read_only_paths": ["repo-relative read-only context paths"],
  "required_evidence_ids": ["evidence ids used"],
  "programmer_task": null,
  "repair_instruction": null,
  "blocker": null
}

For create_programmer_task, programmer_task must be:
{
  "task_id": "short id",
  "target_paths": ["repo-relative source, test, or doc paths"],
  "change_goal": "specific local change goal",
  "required_behavior": ["observable behavior requirements"],
  "forbidden_changes": ["things the programmer must avoid"],
  "consumed_interfaces": ["interfaces or contracts consumed"],
  "expected_operations": ["create_file | replace | insert_before | insert_after | replace_file_small"],
  "acceptance_checks": ["review checks for the produced artifacts"],
  "local_risks": ["localized risks"]
}

For repair_child, repair_instruction must be:
{
  "child_id": "task id or child id",
  "feedback_source": "handoff_validation | parser_validation | patch_validation | review_materialization | contract_validation | execution_verification",
  "feedback": "contract problem to fix",
  "expected_correction": "correction expected from the child"
}

For blocked, blocker must be:
{
  "summary": "specific blocker",
  "missing_facts": ["missing facts"],
  "why_information_request_is_not_enough": "why a narrower request cannot unblock it"
}
'''

MODIFYING_PM_RETRY_PROMPT = '''\
Your previous response was empty or not valid JSON.
Return exactly one strict JSON object matching the required output format.
Do not include markdown, commentary, code, diffs, paths outside the payload, or
command output.
'''

_modifying_pm_llm = LLInterface()
_modifying_pm_llm_config = LLMCallConfig(
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
    timeout_seconds=MODIFYING_PM_LLM_TIMEOUT_SECONDS,
    thinking=LLMThinkingConfig(
        enabled=CODING_AGENT_PM_LLM_THINKING_ENABLED,
    ),
)


async def run_modifying_pm(payload: dict[str, object]) -> ModifyingPMDecision:
    """Ask the modifying PM to choose one lifecycle action."""

    payload_text = json.dumps(payload, ensure_ascii=False)
    messages = [
        SystemMessage(content=MODIFYING_PM_PROMPT),
        HumanMessage(content=payload_text),
    ]
    response = await _modifying_pm_llm.ainvoke(
        messages,
        config=_modifying_pm_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict):
        retry_messages = [
            SystemMessage(content=MODIFYING_PM_PROMPT),
            HumanMessage(content=payload_text),
            HumanMessage(content=MODIFYING_PM_RETRY_PROMPT),
        ]
        response = await _modifying_pm_llm.ainvoke(
            retry_messages,
            config=_modifying_pm_llm_config,
        )
        parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict):
        parsed = {
            "status": "blocked",
            "reason": "The modifying PM did not return a JSON object.",
            "owned_paths": [],
            "read_only_paths": [],
            "required_evidence_ids": [],
            "blocker": {
                "summary": "The modifying PM did not return a JSON object.",
                "missing_facts": [],
                "why_information_request_is_not_enough": (
                    "The failure is an invalid role response, not missing source evidence."
                ),
            },
        }
    decision = normalize_modifying_pm_decision(parsed)
    decision["raw_output"] = response.content
    return decision


__all__ = [
    "MODIFYING_PM_PROMPT",
    "normalize_modifying_pm_decision",
    "run_modifying_pm",
]
