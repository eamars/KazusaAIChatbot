"""Bounded goal-branch cognition that returns action bids, never mutations."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.contracts import CognitionChainServices
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    BranchResult,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)


GOAL_COGNITION_PROMPT = '''You reason about one active character goal from bounded semantic evidence.
Choose a grounded desired outcome and one semantic action bid. Do not mutate
state, select executable handlers, or write final visible dialogue.

# Generation Procedure
Use only the activating state, declared dependencies, and supplied evidence.
Keep the bid within the declared action tendencies. State uncertainty plainly.

# Output Format
Return a JSON object with "perceived_meaning", "desired_outcome", "confidence",
and "action_bid". The action_bid object has "intention", "detail", and
"reason" string fields. The confidence field must be exactly one qualitative
descriptor: "low", "medium", or "high". Do not return a number.
'''

CONFIDENCE_DESCRIPTORS = frozenset(("low", "medium", "high"))


async def run_goal_branch(
    definition: BranchDefinition,
    semantic_state: Mapping[str, str],
    evidence: list[Mapping[str, str]],
    services: CognitionChainServices,
) -> BranchResult:
    """Run one branch against its declared prompt-safe local context.

    Args:
        definition: Activated branch contract with its allowed tendencies.
        semantic_state: Calibrated descriptors without raw numerical state.
        evidence: Prompt-safe evidence scoped to the activated branch.
        services: Existing V1 cognition LLM binding.

    Returns:
        A validated action bid that the workspace may admit or suppress.
    """

    payload = {
        "branch_id": definition.branch_id,
        "action_tendencies": list(definition.action_tendencies),
        "dependencies": list(definition.dependencies),
        "activating_state": dict(semantic_state),
        "evidence": evidence,
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    started_at = time.perf_counter()
    raw_output: str | None = None
    parsed: object | None = None
    try:
        response = await services.llm.ainvoke(
            [
                SystemMessage(content=GOAL_COGNITION_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=services.cognition_config,
        )
        raw_output = response.content
        parsed = services.parse_json(raw_output)
        branch_result = _validate_branch_result(definition.branch_id, parsed)
    except Exception as exc:
        ended_at = time.perf_counter()
        capture_validation_stage(
            stage_id="goal_cognition",
            branch_id=definition.branch_id,
            config=services.cognition_config,
            system_prompt=GOAL_COGNITION_PROMPT,
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
        stage_id="goal_cognition",
        branch_id=definition.branch_id,
        config=services.cognition_config,
        system_prompt=GOAL_COGNITION_PROMPT,
        human_payload=payload_text,
        raw_output=raw_output,
        parsed_output=parsed,
        parse_status="succeeded",
        started_at=started_at,
        ended_at=ended_at,
    )
    return branch_result


def _validate_branch_result(branch_id: str, parsed: object) -> BranchResult:
    """Validate the compact branch result before workspace admission."""

    if not isinstance(parsed, Mapping):
        raise ValueError("goal branch output must be an object")
    action_bid = parsed.get("action_bid")
    if not isinstance(action_bid, Mapping):
        raise ValueError("goal branch output requires an action bid")
    bid_fields = ("intention", "detail", "reason")
    normalized_bid: dict[str, str] = {}
    for field_name in bid_fields:
        value = action_bid.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"goal branch action bid requires {field_name}")
        normalized_bid[field_name] = value
    perceived_meaning = parsed.get("perceived_meaning")
    desired_outcome = parsed.get("desired_outcome")
    confidence = parsed.get("confidence")
    if not isinstance(perceived_meaning, str) or not perceived_meaning:
        raise ValueError("goal branch requires perceived meaning")
    if not isinstance(desired_outcome, str) or not desired_outcome:
        raise ValueError("goal branch requires desired outcome")
    if confidence not in CONFIDENCE_DESCRIPTORS:
        raise ValueError(
            "goal branch confidence must be low, medium, or high"
        )
    result = BranchResult(
        branch_id=branch_id,
        action_bid=normalized_bid,
        perceived_meaning=perceived_meaning,
        desired_outcome=desired_outcome,
        confidence=confidence,
    )
    return result
