"""Dialog execution agent.

Design intent:
- Dialog agent turns the upstream content plan into natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3. If dialog
  needs a fact, answer, conclusion, question, or code block, it must already be
  represented in `text_surface_output_v2.content_plan`.
"""

import time
from typing import Any, NotRequired, TypedDict

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    project_model_visible_percepts,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceOutputV2,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import (
    DIALOG_GENERATOR_LLM_API_KEY,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
    DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    DIALOG_GENERATOR_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.utils import (
    parse_llm_json_output,
    log_list_preview,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import logging
import json


from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
DIALOG_COMPONENT = "nodes.dialog_agent"
DEFAULT_DIALOG_USAGE_MODE = "live_visible_reply"
DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE = (
    "self_cognition_action_candidate_render"
)


class StateContractError(ValueError):
    """Raised when internal graph state violates the dialog contract."""


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _dialog_usage_mode(global_state: GlobalPersonaState) -> str:
    """Describe why the shared dialog graph is being invoked.

    Args:
        global_state: Persona or self-cognition state passed to dialog.

    Returns:
        Stable log label distinguishing visible replies from private renders.
    """

    explicit_mode = global_state.get("dialog_usage_mode")
    if isinstance(explicit_mode, str) and explicit_mode.strip():
        usage_mode = explicit_mode.strip()
        return usage_mode

    debug_modes = global_state["debug_modes"]
    if isinstance(debug_modes, dict) and debug_modes.get("think_only"):
        usage_mode = "debug_think_only"
        return usage_mode

    cognitive_episode = global_state.get("cognitive_episode")
    if isinstance(cognitive_episode, dict):
        trigger_source = cognitive_episode.get("trigger_source")
        output_mode = cognitive_episode.get("output_mode")
        if trigger_source == "internal_thought":
            usage_mode = f"internal_thought_{output_mode or 'unknown'}"
            return usage_mode
        if trigger_source == "reflection_signal":
            usage_mode = f"reflection_{output_mode or 'unknown'}"
            return usage_mode
        if output_mode == "think_only":
            usage_mode = "debug_think_only"
            return usage_mode

    if global_state["should_respond"] is False:
        usage_mode = "private_finalization"
        return usage_mode

    usage_mode = DEFAULT_DIALOG_USAGE_MODE
    return usage_mode


# Define DialogAgent state
class DialogAgentState(TypedDict):
    # A: Core instructions
    internal_monologue: str
    text_surface_output_v2: TextSurfaceOutputV2
    cognitive_episode: CognitiveEpisode

    # B: Social context
    chat_history_wide: list[dict]
    chat_history_recent: list[dict]
    platform_user_id: str
    platform_bot_id: str
    global_user_id: str
    user_name: str
    user_profile: dict

    # D: Character soul
    character_profile: dict

    # Output
    final_dialog: list[str]  # Ordered outbound chat messages.
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    dialog_usage_mode: str
    llm_trace_id: str


_V2_DIALOG_GENERATOR_PROMPT = '''\
You are the character's final text-expression renderer. The upstream
cognition and surface stages have already decided whether the character
speaks, the visible intent, the content, the boundary, the addressee, and the
style. Render that canonical `text_surface_output_v2` as natural chat text.
Use style_guidance only for lexical choice and cadence; it must not introduce,
select, infer, or alter semantic content.

Do not re-evaluate permission, stance, truth, safety, relationship meaning, or
whether to answer. Do not add facts, actions, promises, targets, or questions
that are absent from the supplied surface output.
Produce only words the character could literally type or say. Never narrate
an action, body movement, stage direction, camera or scene direction, or
performance cue. Do not emit visible markup residue, stage-direction
delimiters, or unmatched enclosing punctuation. The character may verbally
discuss, accept, refuse, or negotiate a physical-action topic. Return 1-N
complete text messages in `final_dialog`.

Preserve the requested response operation, content plan, every content
requirement, actor, action, target or beneficiary direction, semantic claim,
condition, time scope, topic, required content beat, visible boundary, and
addressee plan. Preserve source descriptors, attributes, qualifiers,
quantities, polarity, and comparative degree. Non-conflicting elaboration is
allowed, but it must not transform, replace, or compound a supplied attribute
into a different claim. Preserve explicit entity and target specificity.
When repair_context contains typed percept roles, they are authoritative:
speaker_role=current_user owns first-person pronouns, addressee_role=self is
the character, and implicit_imperative_subject_role=self owns an unstated
command subject. Repair any upstream actor/target reversal against those roles.
A rhetorical question cannot substitute for a requested answer, inference,
guess, explanation, acceptance, refusal, or negotiation. It may appear only as
an additional character-voice beat after the requested operation is complete.
Never generalize, euphemize, narrow, broaden, or replace a supplied referent.
Acceptance, refusal, permission, and consent must remain bounded to the exact
source-requested act and scope. Indefinite or unrestricted permission must not
substitute for a specific permission.
Possessive, controlling, exclusive, jealous, tsundere, or other expressive or
relational style may shape the wording of source-grounded current meaning.
Style alone cannot authorize a new semantic claim, literal future rule or
exclusivity condition, obligation, prohibition, commitment, or expectation.
When source meaning is limited to the current occurrence, output
must remain silent about future claims, promises, conditions, expectations,
threats, habits, or rules, including contrastive or teasing additions.
Preserve explicit future content when the source actually supplies or requires
it. Do not turn an answer, inference, explanation, acceptance, refusal, or
negotiation into an ask-back or a different operation. If `repair_context` is
present, use its current_visible_percepts as current-turn grounding and revise
the original dialog only enough to resolve every listed semantic issue while
preserving the surface contract.
permitted_action_results is the only authority that the character brain
executed an action in this cognition chain. A matching result with status
executed may support a completed-effect claim bounded to its exact action_kind,
semantic_result, and target_roles. Express that outcome only as literal speech
and never as narrated enactment. A result with status scheduled or pending may
be acknowledged only in that actual lifecycle state and never as completed.
Failed and unavailable results authorize no success claim. Current percepts
may still ground externally reported or observed events. A user request,
content plan, requirement, or intention without a matching executed result
authorizes only the character's verbal stance, such as acceptance, refusal,
negotiation, teasing, bounded permission, or spoken instruction.

Input JSON:
{{
    "text_surface_output_v2": {{
        "schema_version": "text_surface_output.v2",
        "content_plan": "string",
        "content_requirements": ["string"],
        "visible_boundaries": ["string"],
        "addressee_plan": ["string"],
        "style_guidance": "string",
        "selected_surface_intent": "string",
        "permitted_action_results": [{{
            "action_kind": "string",
            "status": "executed|scheduled|pending|failed|unavailable",
            "semantic_result": "string",
            "target_roles": []
        }}]
    }},
    "user_name": "string"
}}

Return only this JSON object, without Markdown fences:
{{
    "final_dialog": ["complete visible message"]
}}
'''

_V2_DIALOG_COMPLIANCE_PROMPT = '''You are a semantic compliance verifier for
one generated character response. current_visible_percepts are the semantic
authority for the canonical current turn, while permitted_action_results inside
text_surface_output_v2 are the exact authority for execution outcomes. Treat
the remaining text_surface_output_v2 fields and candidate_final_dialog as
proposals to audit against those authorities. Reject unsupported content even
when the surface and candidate agree. Reject any
action or stage narration, body-movement narration, camera or scene direction,
or performance cue; visible output must contain only words the character could
literally type or say. Reject visible markup residue, stage-direction
delimiters, or unmatched enclosing punctuation even when no narrated action
text remains. Physical-action topics may still be discussed verbally.

Reject semantic drift in the current user's requested response operation,
actors, actions, targets or beneficiaries, semantic claims, conditions,
time scope, topic, required content beats, visible boundaries, or addressee
plan. Verify that source descriptors, attributes, qualifiers, quantities,
polarity, and comparative degree remain unchanged. Allow non-conflicting
elaboration, but it cannot create a new constraint, obligation, permission,
prohibition, commitment, expectation, or future stance. Reject any
transformation, replacement, or compounding of a supplied attribute into a
different claim.
Typed role fields on current_visible_percepts are authoritative. For user
dialog, speaker_role=current_user owns first-person pronouns,
addressee_role=self is the character, and
implicit_imperative_subject_role=self owns an unstated command subject. Reject
surface or dialog content that reverses any of those roles.
For a requested answer, inference, guess, explanation, acceptance, refusal, or
negotiation, verify that the candidate actually performs that operation with
the source-defined actors and targets. Reject a candidate that merely restates,
redirects, or asks back the requested operation. A rhetorical question is only
an optional character-voice beat after the operation is complete.
Treat permitted_action_results as the closed execution ledger for actions
performed by the character brain. Allow such a completed-effect claim only
when a matching result has status executed, and keep the claim bounded to its
action_kind, semantic_result, and target_roles. Current visible percepts may
still ground external actions reported or observed as events, but a user's
request cannot prove the character performed it. Scheduled and pending
authorize only their exact lifecycle acknowledgement; failed and unavailable
authorize no success claim. Even with an executed result, reject action or
stage narration: the candidate may state the outcome only in literal spoken
or typed words.
Preserve explicit entity and target specificity.
Never generalize, euphemize, narrow, broaden, or replace a supplied referent.
Acceptance, refusal, permission, and consent must remain bounded to the exact
source-requested act and scope. Indefinite or unrestricted permission must not
substitute for a specific permission.
Perform a claim-by-claim audit before choosing aligned. For every candidate
claim, condition, restriction, obligation, permission, prohibition,
commitment, expectation, and present or future time scope, identify its basis
in current_visible_percepts. Surface and candidate agreement is not evidence.
Possessive, controlling, exclusive, jealous, tsundere, or other expressive or
relational style may color supported meaning, but style alone cannot authorize
a new semantic claim, literal future rule or exclusivity condition.
When source meaning is limited to the current occurrence, output
must remain silent about future claims, promises, conditions, expectations,
threats, habits, or rules, including contrastive or teasing additions.
Explicit future content remains allowed when current_visible_percepts supply
or require it. Judge meaning rather than word overlap or writing style. Do not
rewrite the dialog and do not add new requirements.

Return exactly one JSON object with exactly aligned and issues. aligned must
be a boolean. issues must be a duplicate-free list of zero to eight concise
semantic issue strings, each at most 300 characters. Use an empty issues list
when aligned is true and at least one issue when aligned is false.'''

_dialog_generator_llm = LLInterface()
_dialog_compliance_llm = LLInterface()
_dialog_generator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.65,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=0.25,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)
_dialog_compliance_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.compliance",
    route_name="DIALOG_GENERATOR_LLM",
    base_url=DIALOG_GENERATOR_LLM_BASE_URL,
    api_key=DIALOG_GENERATOR_LLM_API_KEY,
    model=DIALOG_GENERATOR_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=DIALOG_GENERATOR_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=DIALOG_GENERATOR_LLM_THINKING_ENABLED,
    ),
)


async def dialog_generator(state: DialogAgentState) -> DialogAgentState:

    usage_mode = state["dialog_usage_mode"]
    surface_output = state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "dialog state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    surface_output = validate_text_surface_output(surface_output)
    system_prompt = SystemMessage(content=_V2_DIALOG_GENERATOR_PROMPT)
    current_visible_percepts = _current_visible_percepts(
        state["cognitive_episode"]
    )

    msg = {
        "text_surface_output_v2": dict(surface_output),
        "user_name": state["user_name"],
    }

    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))

    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke(
        [system_prompt, human_message],
        config=_dialog_generator_llm_config,
    )

    result = parse_llm_json_output(response.content)
    invalid_fields: list[str] = []
    if isinstance(result, list):
        logger.warning(
            "Dialog generator returned a top-level list; "
            "normalizing it into final_dialog"
        )
        generated_dialog = result
        parsed_keys = ["<top-level-list>"]
        invalid_fields.append("top_level")
    else:
        generated_dialog = result.get("final_dialog", [])
        parsed_keys = list(result.keys())

    if not isinstance(generated_dialog, list):
        logger.warning(
            f"Dialog generator final_dialog is not a list: "
            f"type={type(generated_dialog).__name__}"
        )
        generated_dialog = []
        invalid_fields.append("final_dialog")
    valid_dialog: list[str] = []
    for segment in generated_dialog:
        if not isinstance(segment, str):
            continue
        if segment:
            valid_dialog.append(segment)
    if len(valid_dialog) != len(generated_dialog):
        logger.warning(
            f"Dialog generator dropped invalid messages: "
            f"raw_count={len(generated_dialog)} valid_count={len(valid_dialog)}"
        )
        invalid_fields.append("final_dialog_message")
    generated_dialog = valid_dialog
    parse_status = "succeeded" if not invalid_fields else "warning"
    llm_trace_id = state.get("llm_trace_id", "")
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_generator",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_prompt, human_message],
        response_text=str(response.content),
        parsed_output=result,
        parse_status=parse_status,
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog"],
    )
    repair_issues: list[str] = []
    if generated_dialog:
        verdict = await _verify_dialog_compliance(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=state.get("llm_trace_id", ""),
        )
        if not verdict["aligned"]:
            repair_issues = verdict["issues"]
            repair_payload = {
                **msg,
                "repair_context": {
                    "issues": repair_issues,
                    "original_final_dialog": generated_dialog,
                    "current_visible_percepts": current_visible_percepts,
                },
            }
            repair_message = HumanMessage(
                content=json.dumps(repair_payload, ensure_ascii=False),
            )
            repair_started_at = time.perf_counter()
            repair_response = await _dialog_generator_llm.ainvoke(
                [system_prompt, repair_message],
                config=_dialog_generator_llm_config,
            )
            repair_result = parse_llm_json_output(repair_response.content)
            generated_dialog = _validated_dialog_messages(repair_result)
            await llm_tracing.record_llm_trace_step(
                trace_id=llm_trace_id,
                stage_name="dialog_generator_repair",
                route_name="DIALOG_GENERATOR_LLM",
                model_name=DIALOG_GENERATOR_LLM_MODEL,
                messages=[system_prompt, repair_message],
                response_text=str(repair_response.content),
                parsed_output=repair_result,
                parse_status="succeeded",
                status="succeeded",
                duration_ms=_elapsed_ms(repair_started_at),
                output_state_fields=["final_dialog"],
            )
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_compliance",
                violation_kind="semantic_dialog_misalignment",
                missing_fields=[],
                invalid_fields=repair_issues,
                repair_used=True,
                status="repaired",
                correlation_id=state.get("llm_trace_id", ""),
            )
    generated_dialog_preview = (
        generated_dialog
        if isinstance(generated_dialog, list)
        else []
    )
    logger.debug(
        f"Dialog generator: "
        f"parsed_keys={parsed_keys} "
        f"messages={len(generated_dialog_preview)} "
        f"dialog={log_list_preview(generated_dialog_preview)}"
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_generator",
        route_name="generate",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info" if not invalid_fields else "warning",
        correlation_id=llm_trace_id,
    )
    if invalid_fields:
        await event_logging.record_model_contract_event(
            component=DIALOG_COMPONENT,
            stage_name="dialog_generator",
            violation_kind="invalid_dialog_output",
            missing_fields=[],
            invalid_fields=invalid_fields,
            repair_used=True,
            status="repaired",
            correlation_id=llm_trace_id,
        )

    return_value = {
        "final_dialog": generated_dialog,
    }
    return return_value



async def dialog_agent(
    global_state: GlobalPersonaState
) -> list[str]:
    """
    Dialog agent that renders dialogue from the canonical V2 surface output.
    """
    
    usage_mode = _dialog_usage_mode(global_state)
    surface_output = global_state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            "persona state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    validate_text_surface_output(surface_output)
    content_plan_entry_count = 1
    sub_agent_builder = StateGraph(DialogAgentState)

    sub_agent_builder.add_node("generator", dialog_generator)
    sub_agent_builder.add_edge(START, "generator")
    sub_agent_builder.add_edge("generator", END)
    
    # Compile
    sub_graph = sub_agent_builder.compile()

    # Build initial state
    subState: DialogAgentState = {
        # A
        "internal_monologue": global_state["internal_monologue"],
        "text_surface_output_v2": surface_output,
        "cognitive_episode": global_state["cognitive_episode"],

        # B
        "chat_history_wide": global_state["chat_history_wide"],
        "chat_history_recent": global_state["chat_history_recent"],
        "platform_user_id": global_state["platform_user_id"],
        "platform_bot_id": global_state["platform_bot_id"],
        "global_user_id": global_state["global_user_id"],
        "user_name": global_state["user_name"],
        "user_profile": global_state["user_profile"],

        # D
        "character_profile": global_state["character_profile"],
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": usage_mode,
        "llm_trace_id": global_state.get("llm_trace_id", ""),
    }
    result = await sub_graph.ainvoke(subState)

    # Assemble output.
    final_dialog = result["final_dialog"]

    logger.info(
        f"Dialog output: usage_mode={usage_mode} "
        f"dialog={log_list_preview(final_dialog)}"
    )
    logger.debug(
        f'Dialog metadata: usage_mode={usage_mode} '
        f'messages={len(final_dialog)}'
    )
    quality_status = "passed" if final_dialog else "empty"
    await event_logging.record_dialog_quality_event(
        component=DIALOG_COMPONENT,
        correlation_id="",
        usage_mode=usage_mode,
        quality_status=quality_status,
        retry_count=0,
        failure_codes=[] if final_dialog else ["empty_dialog"],
        content_plan_entry_count=content_plan_entry_count,
        status="succeeded",
    )

    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": [global_state["global_user_id"]] if final_dialog else [],
        "target_broadcast": False,
    }
    return return_value


async def _verify_dialog_compliance(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, str]],
    llm_trace_id: str,
) -> dict[str, Any]:
    """Obtain one bounded semantic verdict for the initial dialog."""

    system_message = SystemMessage(content=_V2_DIALOG_COMPLIANCE_PROMPT)
    human_message = HumanMessage(content=json.dumps({
        "text_surface_output_v2": dict(surface_output),
        "candidate_final_dialog": generated_dialog,
        "current_visible_percepts": current_visible_percepts,
    }, ensure_ascii=False))
    started_at = time.perf_counter()
    response = await _dialog_compliance_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_compliance_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_compliance_verdict(parsed)
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_compliance_verifier",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_compliance_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name="dialog_compliance",
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=0,
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


def _current_visible_percepts(
    episode: CognitiveEpisode,
) -> list[dict[str, str]]:
    """Project current model-visible percepts within the shared prompt bound."""

    percepts = project_model_visible_percepts(episode)
    serialized = json.dumps(percepts, ensure_ascii=False)
    if len(serialized) > 24000:
        raise StateContractError("current visible percepts exceed dialog bounds")
    return percepts


def _validate_compliance_verdict(value: object) -> dict[str, Any]:
    """Validate the exact semantic-verifier shape and bounded issue strings."""

    if not isinstance(value, dict) or set(value) != {"aligned", "issues"}:
        raise StateContractError("dialog compliance fields are not exact")
    aligned = value["aligned"]
    issues = value["issues"]
    if not isinstance(aligned, bool):
        raise StateContractError("dialog compliance aligned must be boolean")
    if not isinstance(issues, list) or len(issues) > 8:
        raise StateContractError("dialog compliance issues are invalid")
    if len(issues) != len(set(issues)):
        raise StateContractError("dialog compliance issues are duplicated")
    if any(
        not isinstance(issue, str)
        or not issue.strip()
        or len(issue) > 300
        for issue in issues
    ):
        raise StateContractError("dialog compliance issue text is invalid")
    if aligned and issues:
        raise StateContractError("aligned dialog cannot contain issues")
    if not aligned and not issues:
        raise StateContractError("misaligned dialog requires issues")
    return {"aligned": aligned, "issues": list(issues)}


def _validated_dialog_messages(value: object) -> list[str]:
    """Validate the single repair result without adding semantic judgment."""

    if not isinstance(value, dict) or set(value) != {"final_dialog"}:
        raise StateContractError("dialog repair fields are not exact")
    messages = value["final_dialog"]
    if not isinstance(messages, list) or not messages:
        raise StateContractError("dialog repair messages are invalid")
    if any(
        not isinstance(message, str) or not message.strip()
        for message in messages
    ):
        raise StateContractError("dialog repair message text is invalid")
    validated_messages = list(messages)
    return validated_messages
