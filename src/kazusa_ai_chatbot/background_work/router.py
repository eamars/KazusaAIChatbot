"""Route-only LLM stage for claimed background-work jobs."""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.background_work.models import (
    BackgroundWorkRouterDecision,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_LLM_API_KEY,
    BACKGROUND_WORK_LLM_BASE_URL,
    BACKGROUND_WORK_LLM_MODEL,
    BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    BACKGROUND_WORK_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
BACKGROUND_WORK_ROUTER_PROMPT = '''\
You route queued background work to one enabled worker.
Choose only whether the task can execute and which worker owns it.
Do not produce worker-facing task briefs, worker-local task types, tool
arguments, files, adapter data, delivery decisions, job ids, or final text.

# Decision Procedure
- Enabled workers are listed in the payload. Choose one worker name exactly.
- Use action="execute" only when an enabled worker clearly owns the task.
- Use action="needs_user_input" when the task is too ambiguous to start.
- Use action="reject" when the task asks for unsupported execution.
- Use action="stop" when no worker should run.

# Output Format
Return strict JSON with exactly these semantic fields:
{
  "action": "execute | reject | needs_user_input | stop",
  "worker": "enabled worker name | none",
  "reason": "short reason for the route"
}
'''

_llm_interface = LLInterface()
_background_work_router_llm = LLInterface()
_background_work_router_llm_config = LLMCallConfig(
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


async def route_background_work(
    *,
    task_brief: str,
    source_summary: str,
    worker_descriptions: dict[str, str],
    max_output_chars: int,
) -> BackgroundWorkRouterDecision:
    """Route one claimed job to a worker without generating worker params."""

    payload = build_background_work_router_payload(
        task_brief=task_brief,
        source_summary=source_summary,
        worker_descriptions=worker_descriptions,
        max_output_chars=max_output_chars,
    )
    response = await _background_work_router_llm.ainvoke([
        SystemMessage(content=BACKGROUND_WORK_ROUTER_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ], config=_background_work_router_llm_config)
    parsed = parse_llm_json_output(response.content)
    decision = normalize_background_work_router_output(
        parsed,
        enabled_workers=set(worker_descriptions),
    )
    return decision


def build_background_work_router_payload(
    *,
    task_brief: str,
    source_summary: str,
    worker_descriptions: dict[str, str],
    max_output_chars: int,
) -> dict[str, object]:
    """Build a prompt-safe router payload with no runtime mechanics."""

    workers = [
        {
            "worker": worker,
            "description": description,
        }
        for worker, description in sorted(worker_descriptions.items())
    ]
    payload: dict[str, object] = {
        "task_brief": task_brief,
        "source_summary": source_summary,
        "max_output_chars": max_output_chars,
        "workers": workers,
    }
    return payload


def normalize_background_work_router_output(
    parsed: object,
    *,
    enabled_workers: set[str] | None = None,
) -> BackgroundWorkRouterDecision:
    """Normalize router output to the four allowed route fields."""

    if enabled_workers is None:
        enabled_workers = {"text_artifact"}

    if not isinstance(parsed, dict):
        result: BackgroundWorkRouterDecision = {
            "action": "reject",
            "worker": "none",
            "reason": "Background-work router returned malformed output.",
        }
        return result

    action = _bounded_text(parsed.get("action"))
    if action not in ("execute", "reject", "needs_user_input", "stop"):
        action = "reject"
    worker = _bounded_text(parsed.get("worker"))
    if worker not in enabled_workers:
        worker = "none"
    if action != "execute":
        worker = "none"
    reason = _bounded_text(parsed.get("reason"))
    if not reason:
        reason = "No route reason was provided."
    decision: BackgroundWorkRouterDecision = {
        "action": action,
        "worker": worker,
        "reason": reason,
    }
    return decision


def _bounded_text(value: object, *, limit: int = 4000) -> str:
    """Return stripped external LLM text capped to a local bound."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()[:limit]
    return return_value
