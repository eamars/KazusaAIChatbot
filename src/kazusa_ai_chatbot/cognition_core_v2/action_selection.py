"""V1-compatible route-only action selection for a collapsed V2 intention."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.contracts import CognitionChainServices
from kazusa_ai_chatbot.cognition_core_v2.contracts import WorkspaceResult
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)


ACTION_SELECTION_PROMPT = '''You map one selected character intention onto only the supplied semantic actions.
Choose no action when the intention cannot be supported by an available action.
Do not mutate state, select executable handlers, or write final visible dialogue.

# Generation Procedure
Use the selected workspace intention and summary with the supplied action
affordances. Keep every request within an available capability and state its
semantic reason. For every request, detail must be a concise nonempty semantic
description of what that action should accomplish; it must not be final dialogue.

# Output Format
Return a JSON object with "action_requests" as a list of objects. Each object
has "capability", "decision", "detail", and "reason" string fields. Detail
must contain 1 to 500 non-whitespace characters.
'''

ACTION_DETAIL_CHAR_LIMIT = 500


async def select_semantic_actions(
    workspace: WorkspaceResult,
    input_payload: Mapping[str, object],
    services: CognitionChainServices,
) -> list[dict[str, str]]:
    """Route a selected workspace intention using the existing V1 action config.

    Args:
        workspace: Collapsed V2 intention that may require a semantic action.
        input_payload: Validated V1 input containing caller-owned affordances.
        services: Existing V1 action-selection LLM binding.

    Returns:
        Structurally valid V1 semantic action requests within caller caps.
    """

    if workspace.selected_bid_id is None or not workspace.public_intention:
        return []
    raw_affordances = input_payload["available_actions"]
    runtime_context = input_payload["runtime_context"]
    if not isinstance(raw_affordances, list) or not isinstance(runtime_context, Mapping):
        raise TypeError("validated V1 action inputs must retain their public shapes")
    affordances = [
        affordance
        for affordance in raw_affordances
        if isinstance(affordance, Mapping) and affordance["available"] is True
    ]
    payload = {
        "selected_intention": workspace.public_intention,
        "workspace_summary": workspace.internal_summary,
        "action_affordances": affordances,
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    max_requests = runtime_context["max_action_requests"]
    if not isinstance(max_requests, int):
        raise TypeError("validated V1 maximum action requests must be numeric")
    started_at = time.perf_counter()
    raw_output: str | None = None
    parsed: object | None = None
    try:
        response = await services.llm.ainvoke(
            [
                SystemMessage(content=ACTION_SELECTION_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=services.action_selection_config,
        )
        raw_output = response.content
        parsed = services.parse_json(raw_output)
        action_requests = _validate_action_requests(
            parsed,
            affordances,
            max_requests,
        )
    except Exception as exc:
        ended_at = time.perf_counter()
        capture_validation_stage(
            stage_id="action_selection",
            config=services.action_selection_config,
            system_prompt=ACTION_SELECTION_PROMPT,
            human_payload=payload_text,
            raw_output=raw_output,
            parsed_output=parsed,
            parse_status="failed",
            started_at=started_at,
            ended_at=ended_at,
            error=str(exc),
        )
        raise
    ended_at = time.perf_counter()
    capture_validation_stage(
        stage_id="action_selection",
        config=services.action_selection_config,
        system_prompt=ACTION_SELECTION_PROMPT,
        human_payload=payload_text,
        raw_output=raw_output,
        parsed_output=parsed,
        parse_status="succeeded",
        started_at=started_at,
        ended_at=ended_at,
    )
    return action_requests


def _validate_action_requests(
    parsed: object,
    affordances: list[Mapping[str, object]],
    max_requests: int,
) -> list[dict[str, str]]:
    """Enforce structural V1 action boundaries without semantic reinterpretation."""

    if not isinstance(parsed, Mapping):
        raise ValueError("action selection output must be an object")
    raw_requests = parsed.get("action_requests")
    if not isinstance(raw_requests, list):
        raise ValueError("action selection requests must be a list")
    allowed_capabilities = {
        affordance["capability"]
        for affordance in affordances
        if isinstance(affordance["capability"], str)
    }
    normalized_requests: list[dict[str, str]] = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, Mapping):
            raise ValueError("action request must be an object")
        capability = raw_request.get("capability")
        if capability not in allowed_capabilities:
            raise ValueError("action request uses an unavailable capability")
        request = {"capability": capability}
        for field_name in ("decision", "detail", "reason"):
            value = raw_request.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"action request requires {field_name}")
            if field_name == "detail":
                detail = value.strip()
                if not detail or len(detail) > ACTION_DETAIL_CHAR_LIMIT:
                    raise ValueError(
                        "action request detail must be a concise nonempty "
                        "semantic description"
                    )
                request[field_name] = detail
                continue
            request[field_name] = value
        normalized_requests.append(request)
        if len(normalized_requests) >= max_requests:
            break
    return normalized_requests
