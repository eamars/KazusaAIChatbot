"""Compositional semantic action planning over admitted cognition bids."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
    derive_action_route,
)
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    ActionBidV2,
    CognitionCoreServicesV2,
    CognitionEvidenceV2,
    CognitionExecutionError,
    ResolverAffordanceV2,
    ResolverCapabilityRequestV2,
    SelectedIntentionV2,
    SemanticActionRequestV2,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_PENDING_DECISIONS,
    RESOLVER_GOAL_PROGRESS_VERSION,
    ResolverValidationError,
    validate_resolver_goal_progress,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

ACTION_REQUEST_CAP = 3
ACTION_PLANNING_PROMPT_CAP = 24000
ACTION_PLANNING_REPAIR_OUTPUT_CAP = 4000
ACTION_PLANNING_ATTEMPT_LIMIT = 2
MODEL_TEXT_CAP = 500


ACTION_PLANNING_PROMPT = '''You are the semantic capability-proposal boundary
for one active character. Propose concrete executable action requests or
resolver requests that advance the admitted motives. The primary bid owns the
visible intention. Supporting bids may contribute compatible private actions
or evidence needs. Do not select or restate a route, rewrite bid content,
generate final dialogue, execute a capability, authorize execution, or invent
an unavailable capability. Protocol code derives route after semantic action
authorization.

Produce one semantic proposal object. Action requests and resolver requests
are mutually exclusive. Either array may contain up to three requests.
Immediate visible speech is not a capability request and never appears in this
output. When evidence or a persisted clarification or approval step is needed,
select resolver requests only; resolver recurrence owns the later visible
answer or question.
Select a private action only when its cited admitted bid requires that
capability's durable or out-of-turn effect as part of the desired outcome.
The bid cannot broaden capability eligibility. Current evidence must itself
support the selected capability's declared real effect; generic words such as
task, action, request, analysis, or work do not establish capability fit.
In particular, accepted coding capability requires current evidence that asks
for actual code, repository, or software-engineering work. A drifted bid cannot
convert physical chat or ordinary conversation into coding work.
Never create an action request for the planner's own reasoning, memory recall,
reply preparation, response rehearsal, wording, or thought that completes in
the current cognition turn. Use a resolver for genuinely missing evidence and
use speech without a private action when the admitted response can finish now.
No supplied action capability actuates the character's body or a physical
scene. Never select an action to execute a physical request or to generate,
store, or later present a physical-action description. For a physical chat
request, use speech for the character's verbal stance unless a distinct
supplied capability genuinely provides another explicit non-physical effect.
Respect episode.output_mode. Silence permits no requests. A normal
visible_reply may combine its protocol-owned visible response with up to three
grounded private action requests. A scheduled_action_request permits executable
actions only.

Each request must cite one supplied bid handle and one supplied capability
handle. For an action request, follow the affordance's decision_mode:
- optional: use the affordance's default_decision or an empty string;
- required_text: provide one concrete bounded semantic decision;
- closed: copy one value from allowed_decisions.
When decision_pattern is non-empty, decision must full-match it exactly, with
no prefix, suffix, explanation, or final message text.
semantic_goal states the concrete semantic objective, not execution parameters
or final wording. reason explains why the request advances its cited bid.
Do not emit context_ref. Selecting an action_handle binds its deterministic
context_ref after validation.

Treat character-owned reflection or internal observation as evidence, not live
user speech. Do not copy packet headings, timestamps, schema keys, transport
summaries, or operational metadata into generated prose. Write generated
free-text fields in Simplified Chinese while preserving quoted user text,
proper nouns, code, URLs, capability names, and schema or enum tokens.

resolver_pending_resolution is null unless the resolver context contains an
active pending item and the current evidence supports a decision. When present,
return exactly decision and reason; deterministic code binds the active item.
resolver_goal_progress is null when no goal progress is needed. When present,
return a partial semantic update containing only fields that changed. Protocol
code binds schema_version and original_goal and deterministic code preserves omitted
known checklist fields from current_resolver_goal_progress.

# Output Format
Return exactly one JSON object with exactly these fields:
- action_requests: array of zero to three objects, each with exactly
  bid_handle, action_handle, decision, semantic_goal, and reason
- resolver_requests: array of zero to three objects, each with exactly
  bid_handle, resolver_handle, semantic_goal, and reason
- resolver_pending_resolution: null or an object with exactly decision and
  reason
- resolver_goal_progress: null or a partial semantic update object
When resolver_requests is non-empty, action_requests must be empty. When
action_requests is non-empty, resolver_requests must be empty.
Do not emit any other field.
'''


async def plan_actions(
    *,
    primary_bid: ActionBidV2 | None,
    supporting_bids: Sequence[ActionBidV2],
    episode: Mapping[str, Any],
    evidence: Sequence[CognitionEvidenceV2],
    available_actions: Sequence[ActionAffordanceV2],
    available_resolvers: Sequence[ResolverAffordanceV2],
    resolver_context: str,
    services: CognitionCoreServicesV2,
    current_goal_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select one route and bounded semantic requests from admitted motives.

    Args:
        primary_bid: Workspace-selected motive that owns the visible intention.
        supporting_bids: Other admitted motives that may contribute requests.
        episode: Canonical source envelope used only through semantic fields.
        evidence: Prompt-safe evidence available to admitted motives.
        available_actions: Registry-derived executable action affordances.
        available_resolvers: Registry-derived resolver affordances.
        resolver_context: Bounded prompt-safe resolver recurrence projection.
        services: Injected LLM binding and action-planning configuration.

    Returns:
        Selected intention, semantic requests, and resolver lifecycle decisions.
    """

    if primary_bid is None:
        return_value = _silence_result()
        return return_value

    bids = [primary_bid, *supporting_bids]
    bid_handles = {
        f"b{index}": bid for index, bid in enumerate(bids, start=1)
    }
    planner_actions = [
        affordance
        for affordance in available_actions
        if affordance["permission"] == "allowed"
        and affordance["action_kind"] not in {
            SPEAK_CAPABILITY,
            APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        }
    ]
    action_handles = {
        f"a{index}": affordance
        for index, affordance in enumerate(planner_actions, start=1)
    }
    resolver_handles = {
        f"r{index}": affordance
        for index, affordance in enumerate(
            (
                affordance
                for affordance in available_resolvers
                if affordance["availability"] == "available"
            ),
            start=1,
        )
    }
    prompt_payload = {
        "bids": {
            handle: {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
                "expected_consequences": list(bid["expected_consequences"]),
                "confidence": bid["confidence"],
                "evidence_handles": list(bid["evidence_handles"]),
            }
            for handle, bid in bid_handles.items()
        },
        "episode": {
            "trigger_source": episode.get("trigger_source", ""),
            "input_sources": episode.get("input_sources", []),
            "output_mode": episode.get("output_mode", ""),
            "local_time_context": episode.get("local_time_context", {}),
        },
        "evidence": [
            {
                "handle": row["evidence_handle"],
                "source_kind": row["evidence_ref"]["source_kind"],
                "semantic_text": row["semantic_text"],
            }
            for row in evidence
        ],
        "action_handles": {
            handle: {
                "action_kind": affordance["action_kind"],
                "semantic_capability": affordance["capability"],
                "decision_mode": affordance["decision_mode"],
                "allowed_decisions": list(affordance["allowed_decisions"]),
                "default_decision": affordance["default_decision"],
                "decision_pattern": affordance["decision_pattern"],
            }
            for handle, affordance in action_handles.items()
        },
        "resolver_handles": {
            handle: {
                "capability": affordance["capability"],
                "semantic_capability": affordance["semantic_capability"],
            }
            for handle, affordance in resolver_handles.items()
        },
        "resolver_context": resolver_context,
        "current_resolver_goal_progress": current_goal_progress,
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > ACTION_PLANNING_PROMPT_CAP:
        raise CognitionExecutionError("action-planning prompt exceeds contract cap")

    messages: list[BaseMessage] = [
        SystemMessage(content=ACTION_PLANNING_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    decision = await _invoke_action_planner(
        services=services,
        messages=messages,
        bid_handles=bid_handles,
        action_handles=action_handles,
        resolver_handles=resolver_handles,
        current_goal_progress=current_goal_progress,
    )
    authorized_action_rows = await authorize_action_requests(
        action_requests=decision["action_requests"],
        bid_handles=bid_handles,
        evidence=evidence,
        action_handles=action_handles,
        services=services,
    )
    action_requests = _materialize_action_requests(
        authorized_action_rows,
        bid_handles,
        action_handles,
    )
    resolver_requests = _materialize_resolver_requests(
        decision["resolver_requests"],
        bid_handles,
        resolver_handles,
    )
    route = derive_action_route(
        episode=episode,
        primary_bid=primary_bid,
        action_requests=action_requests,
        resolver_requests=resolver_requests,
    )
    intention: SelectedIntentionV2 = {
        "selected_branch_id": primary_bid["branch_id"],
        "route": route,
        "intention": primary_bid["intention"],
        "target_roles": list(primary_bid["target_roles"]),
        "reason": primary_bid["reason"],
    }
    return_value = {
        "intention": intention,
        "action_requests": action_requests,
        "resolver_requests": resolver_requests,
        "resolver_pending_resolution": decision[
            "resolver_pending_resolution"
        ],
        "resolver_goal_progress": decision["resolver_goal_progress"],
    }
    return return_value


async def _invoke_action_planner(
    *,
    services: CognitionCoreServicesV2,
    messages: list[BaseMessage],
    bid_handles: Mapping[str, ActionBidV2],
    action_handles: Mapping[str, ActionAffordanceV2],
    resolver_handles: Mapping[str, ResolverAffordanceV2],
    current_goal_progress: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Invoke the semantic planner with one bounded contract replacement."""

    current_messages = list(messages)
    for attempt_index in range(ACTION_PLANNING_ATTEMPT_LIMIT):
        started_at = perf_counter()
        response = await services.llm.ainvoke(
            current_messages,
            config=services.action_selection_config,
        )
        response_text = str(response.content)
        parsed: object = {}
        stage_name = (
            "action_planning"
            if attempt_index == 0
            else "action_planning.repair"
        )
        try:
            parsed = parse_llm_json_output(response_text)
            decision = _validate_action_plan_decision(
                parsed,
                bid_handles=bid_handles,
                action_handles=action_handles,
                resolver_handles=resolver_handles,
                current_goal_progress=current_goal_progress,
            )
        except (ResolverValidationError, ValueError) as exc:
            await _record_action_planning_trace(
                services=services,
                messages=current_messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                started_at=started_at,
                stage_name=stage_name,
            )
            if attempt_index + 1 >= ACTION_PLANNING_ATTEMPT_LIMIT:
                raise ValueError(
                    f"action plan is invalid after one replacement: {exc}"
                ) from exc
            current_messages.append(
                _action_planning_repair_message(
                    response_text=response_text,
                    contract_error=str(exc),
                )
            )
            continue

        await _record_action_planning_trace(
            services=services,
            messages=current_messages,
            response_text=response_text,
            parsed_output=decision,
            parse_status="succeeded",
            status="succeeded",
            started_at=started_at,
            stage_name=stage_name,
        )
        return decision

    raise AssertionError("action-planning attempt loop did not terminate")


def _action_planning_repair_message(
    *,
    response_text: str,
    contract_error: str,
) -> HumanMessage:
    """Build one bounded same-owner replacement request."""

    bounded_response = _bounded_repair_output(response_text)
    repair_payload = {
        "repair_instruction": (
            "Return a complete replacement object for the original action "
            "plan. Preserve only grounded semantic choices, satisfy every "
            "exact field and request rule, and emit JSON only."
        ),
        "contract_error": contract_error[:MODEL_TEXT_CAP],
        "invalid_response": bounded_response,
    }
    return HumanMessage(
        content=json.dumps(repair_payload, ensure_ascii=False, sort_keys=True)
    )


def _bounded_repair_output(response_text: str) -> str:
    """Keep both ends of one rejected output within the retry prompt cap."""

    if len(response_text) <= ACTION_PLANNING_REPAIR_OUTPUT_CAP:
        return response_text
    half_cap = ACTION_PLANNING_REPAIR_OUTPUT_CAP // 2
    return_value = (
        response_text[:half_cap]
        + "\n... bounded rejected output ...\n"
        + response_text[-half_cap:]
    )
    return return_value


def _validate_action_plan_decision(
    parsed: object,
    *,
    bid_handles: Mapping[str, ActionBidV2],
    action_handles: Mapping[str, ActionAffordanceV2],
    resolver_handles: Mapping[str, ResolverAffordanceV2],
    current_goal_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate fixed model shape, cardinality, and prompt-local ownership."""

    if not isinstance(parsed, Mapping):
        raise ValueError("action plan must be an object")
    required = {
        "action_requests",
        "resolver_requests",
        "resolver_pending_resolution",
        "resolver_goal_progress",
    }
    if set(parsed) != required:
        raise ValueError("action plan fields are not exact")
    action_requests = parsed["action_requests"]
    resolver_requests = parsed["resolver_requests"]
    if not isinstance(action_requests, list):
        raise ValueError("action requests must be an array")
    if not isinstance(resolver_requests, list):
        raise ValueError("resolver requests must be an array")
    if len(action_requests) > ACTION_REQUEST_CAP:
        raise ValueError("action plan permits at most three action requests")
    if len(resolver_requests) > ACTION_REQUEST_CAP:
        raise ValueError("action plan permits at most three resolver requests")
    if action_requests and resolver_requests:
        raise ValueError("action and resolver requests are mutually exclusive")

    normalized_actions = [
        _validate_action_request_row(row, bid_handles, action_handles)
        for row in action_requests
    ]
    normalized_resolvers = [
        _validate_resolver_request_row(row, bid_handles, resolver_handles)
        for row in resolver_requests
    ]
    pending_resolution = _validate_pending_resolution_choice(
        parsed["resolver_pending_resolution"]
    )
    goal_progress = _validate_goal_progress_choice(
        parsed["resolver_goal_progress"],
        current_goal_progress=current_goal_progress,
    )
    return_value = {
        "action_requests": normalized_actions,
        "resolver_requests": normalized_resolvers,
        "resolver_pending_resolution": pending_resolution,
        "resolver_goal_progress": goal_progress,
    }
    return return_value


def _validate_action_request_row(
    value: object,
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
) -> dict[str, str]:
    """Validate one action row and its registry-derived decision semantics."""

    required = {
        "bid_handle",
        "action_handle",
        "decision",
        "semantic_goal",
        "reason",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("action request fields are not exact")
    bid_handle = value["bid_handle"]
    action_handle = value["action_handle"]
    if bid_handle not in bids:
        raise ValueError("action request bid handle is unavailable")
    if action_handle not in actions:
        raise ValueError("action request action handle is unavailable")
    decision = _bounded_model_text(
        value["decision"],
        "action request decision",
        maximum=200,
        allow_empty=True,
    )
    semantic_goal = _bounded_model_text(
        value["semantic_goal"],
        "action request semantic_goal",
    )
    reason = _bounded_model_text(value["reason"], "action request reason")
    affordance = actions[action_handle]
    mode = affordance["decision_mode"]
    if mode == "required_text" and not decision:
        raise ValueError(
            f"action request {action_handle} requires a concrete decision"
        )
    if mode == "closed" and decision not in affordance["allowed_decisions"]:
        allowed_decisions = affordance["allowed_decisions"]
        raise ValueError(
            f"action request {action_handle} decision must be one of "
            f"{allowed_decisions!r}"
        )
    if mode == "optional" and decision not in {
        "",
        affordance["default_decision"],
    }:
        default_decision = affordance["default_decision"]
        raise ValueError(
            f"action request {action_handle} decision must be empty or "
            f"{default_decision!r}"
        )
    decision_pattern = affordance["decision_pattern"]
    if decision_pattern and re.fullmatch(decision_pattern, decision) is None:
        raise ValueError(
            f"action request {action_handle} decision must full-match "
            f"{decision_pattern!r}"
        )
    return_value = {
        "bid_handle": bid_handle,
        "action_handle": action_handle,
        "decision": decision,
        "semantic_goal": semantic_goal,
        "reason": reason,
    }
    return return_value


def _validate_resolver_request_row(
    value: object,
    bids: Mapping[str, ActionBidV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> dict[str, str]:
    """Validate one resolver row and its admitted-bid provenance."""

    required = {
        "bid_handle",
        "resolver_handle",
        "semantic_goal",
        "reason",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("resolver request fields are not exact")
    bid_handle = value["bid_handle"]
    resolver_handle = value["resolver_handle"]
    if bid_handle not in bids:
        raise ValueError("resolver request bid handle is unavailable")
    if resolver_handle not in resolvers:
        raise ValueError("resolver request resolver handle is unavailable")
    semantic_goal = _bounded_model_text(
        value["semantic_goal"],
        "resolver request semantic_goal",
    )
    reason = _bounded_model_text(value["reason"], "resolver request reason")
    return_value = {
        "bid_handle": bid_handle,
        "resolver_handle": resolver_handle,
        "semantic_goal": semantic_goal,
        "reason": reason,
    }
    return return_value


def _validate_pending_resolution_choice(value: object) -> dict | None:
    """Validate the model-owned semantic choice before active-row binding."""

    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, Mapping) or set(value) != {"decision", "reason"}:
        raise ValueError("pending resolution fields are not exact")
    decision = value["decision"]
    if decision not in ALLOWED_PENDING_DECISIONS:
        raise ValueError("pending resolution decision is invalid")
    reason = _bounded_model_text(value["reason"], "pending resolution reason")
    return_value = {"decision": decision, "reason": reason}
    return return_value


def _validate_goal_progress_choice(
    value: object,
    *,
    current_goal_progress: Mapping[str, Any] | None,
) -> dict | None:
    """Merge one semantic delta into protocol-owned resolver progress."""

    if value is None:
        return_value = None
        return return_value
    if not isinstance(value, Mapping):
        raise ValueError("resolver goal progress must be an object or null")
    if current_goal_progress is None:
        raw_progress = dict(value)
        raw_progress.setdefault(
            "schema_version",
            RESOLVER_GOAL_PROGRESS_VERSION,
        )
        validated = validate_resolver_goal_progress(raw_progress)
        return_value = dict(validated)
        return return_value

    current = dict(validate_resolver_goal_progress(current_goal_progress))
    allowed_fields = set(current)
    if not set(value).issubset(allowed_fields):
        raise ValueError("resolver goal progress update fields are invalid")
    supplied_version = value.get("schema_version")
    if supplied_version not in {None, RESOLVER_GOAL_PROGRESS_VERSION}:
        raise ValueError("resolver goal progress schema_version is invalid")
    supplied_goal = value.get("original_goal")
    if supplied_goal not in {None, current["original_goal"]}:
        raise ValueError("resolver goal progress cannot replace original_goal")
    raw_progress = dict(current)
    raw_progress.update({
        key: item
        for key, item in value.items()
        if key not in {"schema_version", "original_goal"}
    })
    validated = validate_resolver_goal_progress(raw_progress)
    return_value = dict(validated)
    return return_value


def _materialize_action_requests(
    requests: Sequence[Mapping[str, str]],
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
) -> list[SemanticActionRequestV2]:
    """Copy admitted provenance into planner-selected action requests."""

    result: list[SemanticActionRequestV2] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = actions[request["action_handle"]]
        result.append({
            "action_kind": affordance["action_kind"],
            "decision": request["decision"],
            "context_ref": affordance["context_ref"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "target_roles": list(bid["target_roles"]),
            "evidence_handles": list(bid["evidence_handles"]),
        })
    return result


def _materialize_resolver_requests(
    requests: Sequence[Mapping[str, str]],
    bids: Mapping[str, ActionBidV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> list[ResolverCapabilityRequestV2]:
    """Copy admitted evidence provenance into resolver requests."""

    result: list[ResolverCapabilityRequestV2] = []
    for request in requests:
        bid = bids[request["bid_handle"]]
        affordance = resolvers[request["resolver_handle"]]
        result.append({
            "capability": affordance["capability"],
            "semantic_goal": request["semantic_goal"],
            "reason": request["reason"],
            "evidence_handles": list(bid["evidence_handles"]),
        })
    return result


async def _record_action_planning_trace(
    *,
    services: CognitionCoreServicesV2,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started_at: float,
    stage_name: str,
) -> None:
    """Preserve the protected action-planning model boundary."""

    trace_id = llm_tracing.current_trace_id()
    if not trace_id:
        return
    config = services.action_selection_config
    await llm_tracing.record_llm_trace_step(
        trace_id=trace_id,
        stage_name=stage_name,
        route_name=config.route_name,
        model_name=config.model,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        duration_ms=max(0, int((perf_counter() - started_at) * 1000)),
        output_state_fields=[
            "intention",
            "action_requests",
            "resolver_requests",
            "resolver_pending_resolution",
            "resolver_goal_progress",
        ],
    )


def _bounded_model_text(
    value: object,
    label: str,
    *,
    maximum: int = MODEL_TEXT_CAP,
    allow_empty: bool = False,
) -> str:
    """Validate one bounded model-authored semantic string."""

    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    normalized = value.strip()
    if (not allow_empty and not normalized) or len(normalized) > maximum:
        raise ValueError(f"{label} is invalid")
    return normalized


def _silence_result() -> dict[str, Any]:
    """Return deterministic silence when workspace admits no motive."""

    return_value = {
        "intention": {
            "route": "silence",
            "intention": "remain silent",
            "target_roles": [],
            "reason": "no valid admitted bid",
        },
        "action_requests": [],
        "resolver_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }
    return return_value
