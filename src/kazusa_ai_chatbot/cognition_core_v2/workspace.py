"""Admitted-bid validation and bounded workspace collapse for V2."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_chain_core.contracts import CognitionChainServices
from kazusa_ai_chatbot.cognition_core_v2.contracts import BranchResult, WorkspaceResult
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_stage,
)


WORKSPACE_INTEGRATION_PROMPT = '''You reconcile only the supplied, admitted character-goal bids.
Choose one grounded public intention and identify any suppressed bid ids.
Do not invent a bid, mutate state, select executable handlers, or write final dialogue.

# Generation Procedure
Compare the permitted bids and their stated consequences. Prefer a coherent
intention that preserves unresolved motives in the private summary.

# Output Format
Return a JSON object with "selected_bid_id", "public_intention",
"internal_summary", and "suppressed_bid_ids" (a list of strings).
'''


def admit_branch_bids(
    branch_results: Mapping[str, BranchResult],
) -> dict[str, BranchResult]:
    """Admit only results whose mapping key matches their declared branch id."""

    admitted = {
        branch_id: result
        for branch_id, result in branch_results.items()
        if branch_id == result.branch_id and result.action_bid
    }
    return admitted


def collapse_workspace(
    *,
    bids: list[BranchResult],
    admitted_bid_ids: set[str],
    selected_bid_id: str | None,
) -> WorkspaceResult:
    """Fail closed when the selected bid is absent from the admitted bid set.

    Args:
        bids: Completed branch bids before workspace admission filtering.
        admitted_bid_ids: Branch ids permitted to influence the workspace.
        selected_bid_id: Proposed selected bid from deterministic or semantic work.

    Returns:
        A workspace result that contains no selected intention for an invalid bid.
    """

    bid_by_id = {bid.branch_id: bid for bid in bids}
    if selected_bid_id not in admitted_bid_ids:
        suppressed = () if selected_bid_id is None else (selected_bid_id,)
        return WorkspaceResult(
            selected_bid_id=None,
            public_intention="",
            internal_summary="selected bid was not admitted",
            suppressed_bid_ids=suppressed,
        )
    if selected_bid_id not in bid_by_id:
        return WorkspaceResult(
            selected_bid_id=None,
            public_intention="",
            internal_summary="selected bid was unavailable",
            suppressed_bid_ids=(),
        )
    selected_bid = bid_by_id[selected_bid_id]
    intention = selected_bid.action_bid.get("intention")
    if not isinstance(intention, str):
        decision = selected_bid.action_bid.get("decision")
        intention = decision if isinstance(decision, str) else ""
    suppressed = tuple(
        branch_id
        for branch_id in admitted_bid_ids
        if branch_id != selected_bid_id
    )
    result = WorkspaceResult(
        selected_bid_id=selected_bid_id,
        public_intention=intention,
        internal_summary=selected_bid.perceived_meaning,
        suppressed_bid_ids=suppressed,
    )
    return result


async def integrate_workspace(
    admitted_bids: Mapping[str, BranchResult],
    services: CognitionChainServices,
) -> WorkspaceResult:
    """Select a V2 workspace intention from only admitted branch bids.

    Args:
        admitted_bids: Validated branch outputs allowed to influence collapse.
        services: Existing V1 cognition binding for multi-bid reconciliation.

    Returns:
        A selected or no-response workspace result with suppressed bids listed.
    """

    if not admitted_bids:
        empty_result = collapse_workspace(
            bids=[],
            admitted_bid_ids=set(),
            selected_bid_id=None,
        )
        return empty_result
    if len(admitted_bids) == 1:
        selected_bid_id = next(iter(admitted_bids))
        deterministic_result = collapse_workspace(
            bids=list(admitted_bids.values()),
            admitted_bid_ids=set(admitted_bids),
            selected_bid_id=selected_bid_id,
        )
        return deterministic_result
    payload = {
        "admitted_bids": {
            branch_id: {
                "action_bid": dict(result.action_bid),
                "perceived_meaning": result.perceived_meaning,
                "desired_outcome": result.desired_outcome,
                "confidence": result.confidence,
            }
            for branch_id, result in admitted_bids.items()
        },
    }
    payload_text = json.dumps(payload, ensure_ascii=False)
    started_at = time.perf_counter()
    raw_output: str | None = None
    parsed: object | None = None
    try:
        response = await services.llm.ainvoke(
            [
                SystemMessage(content=WORKSPACE_INTEGRATION_PROMPT),
                HumanMessage(content=payload_text),
            ],
            config=services.cognition_config,
        )
        raw_output = response.content
        parsed = services.parse_json(raw_output)
        workspace_result = _validate_workspace_result(parsed, admitted_bids)
    except Exception as exc:
        ended_at = time.perf_counter()
        capture_validation_stage(
            stage_id="workspace_integration",
            config=services.cognition_config,
            system_prompt=WORKSPACE_INTEGRATION_PROMPT,
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
        stage_id="workspace_integration",
        config=services.cognition_config,
        system_prompt=WORKSPACE_INTEGRATION_PROMPT,
        human_payload=payload_text,
        raw_output=raw_output,
        parsed_output=parsed,
        parse_status="succeeded",
        started_at=started_at,
        ended_at=ended_at,
    )
    return workspace_result


def _validate_workspace_result(
    parsed: object,
    admitted_bids: Mapping[str, BranchResult],
) -> WorkspaceResult:
    """Reject workspace selections that do not originate from admitted bids."""

    if not isinstance(parsed, Mapping):
        raise ValueError("workspace output must be an object")
    selected_bid_id = parsed.get("selected_bid_id")
    public_intention = parsed.get("public_intention")
    internal_summary = parsed.get("internal_summary")
    suppressed_bid_ids = parsed.get("suppressed_bid_ids")
    if selected_bid_id not in admitted_bids:
        raise ValueError("workspace selected an unadmitted bid")
    if not isinstance(public_intention, str) or not public_intention:
        raise ValueError("workspace requires a public intention")
    if not isinstance(internal_summary, str) or not internal_summary:
        raise ValueError("workspace requires an internal summary")
    if not isinstance(suppressed_bid_ids, list):
        raise ValueError("workspace suppressed bids must be a list")
    normalized_suppressed = []
    for branch_id in suppressed_bid_ids:
        if branch_id not in admitted_bids or branch_id == selected_bid_id:
            raise ValueError("workspace suppressed bids must be other admitted bids")
        normalized_suppressed.append(branch_id)
    result = WorkspaceResult(
        selected_bid_id=selected_bid_id,
        public_intention=public_intention,
        internal_summary=internal_summary,
        suppressed_bid_ids=tuple(normalized_suppressed),
    )
    return result
