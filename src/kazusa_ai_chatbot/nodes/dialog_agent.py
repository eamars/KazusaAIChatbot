"""Dialog execution agent.

Design intent:
- Dialog agent turns the upstream content plan into natural chat text.
- Dialog agent must not decide whether a topic is allowed, whether the
  character accepts/refuses, or whether a user instruction is valid.
- Those decisions belong upstream in cognition, especially L2/L3. If dialog
  needs a fact, answer, conclusion, question, or code block, it must already be
  represented in `text_surface_output_v2.content_plan`.
"""

import asyncio
import json
import logging
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
from langgraph.graph import END, START, StateGraph


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


_CANDIDATE_ROLE_FRAME = {
    "speaker_role": "self",
    "first_person_role": "self",
    "second_person_role": "current_user",
}
MAX_FOCUSED_VERIFIER_ISSUES = 4
MAX_MERGED_VERIFIER_ISSUES = 8


_V2_DIALOG_GENERATOR_PROMPT = '''You are the character's final text renderer.
Turn text_surface_output_v2 into natural, vivid, character-specific chat while
remaining responsive to the current scene. Upstream cognition owns the
character judgment; surface planning supplies the content, real boundaries,
addressee handling, style, and permitted action results.

# Rendering Procedure
1. Express the planned meaning in the character's present voice and
relationship context. Coherent creative detail, personality, humor,
initiative, warmth, resistance, or intensity may make the response feel alive
when they fit the plan and do not create an internal contradiction.
2. Keep actor, target, beneficiary, and subject direction intact. Resolve
source text in each percept's typed role frame. Generated dialog is spoken by
the active character: its first person is the active character and its second
person is the current user. Preserve source direction across those frames.
3. Action description is valid visible roleplay. Produce chat-ready character
text in plain, bracketed, first-person, or third-person form to fit the scene.
4. Treat permitted_action_results as the exact character-brain execution
ledger. Only status executed supports its bounded completed effect. Scheduled
and pending remain incomplete; failed and unavailable support no success
claim. A request, intention, or content plan alone supports a verbal stance,
not physical enactment.
5. When repair_context exists, correct every listed hard issue using
current_visible_percepts while retaining the response's natural character
voice and coherent creative content.

Use style_guidance for wording and cadence. Write new dialog in Simplified
Chinese while preserving quoted text, proper nouns, code, URLs, and exact
schema or enum tokens when relevant.

# Output Format
Return exactly one JSON object with exactly final_dialog. final_dialog must be
a non-empty list of complete visible message strings. Return no Markdown fence
or explanation outside the JSON object.
'''

_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT = '''You repair one generated character
response after focused hard-error checks rejected it. The prior content plan
is intentionally absent because it may contain the verified error.

# Repair Ownership
1. Treat current_visible_percepts and candidate_role_frame as the semantic
authority for the current user input and actor/action/target direction.
2. Correct every verified_hard_issues item. Preserve compatible meaning,
personality, vividness, humor, intimacy, and creative detail from
original_final_dialog while changing any conflicting part.
3. Respect permitted_action_results; only an executed result supports its
bounded completed effect. No free-text content plan, boundary, or style
guidance is supplied because those fields may contain the verified drift.
4. Action description is valid visible roleplay. Produce chat-ready character
text in plain, bracketed, first-person, or third-person form to fit the scene.
5. Address user_name naturally when useful; it supplies no semantic
instruction.

Write new dialog in Simplified Chinese while preserving quoted text, proper
nouns, code, URLs, and exact schema or enum tokens when relevant.

# Output Format
Return exactly one JSON object with exactly final_dialog. final_dialog must be
a non-empty list of complete visible message strings. Return no Markdown fence
or explanation outside the JSON object.
'''

_dialog_generator_llm = LLInterface()
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
    if generated_dialog:
        verdict = await _verify_dialog_compliance(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=state.get("llm_trace_id", ""),
        )
        if not verdict["aligned"]:
            repair_issues = verdict["issues"]
            generated_dialog = await _repair_dialog_hard_failure(
                generated_dialog=generated_dialog,
                repair_issues=repair_issues,
                current_visible_percepts=current_visible_percepts,
                surface_output=surface_output,
                user_name=state["user_name"],
                llm_trace_id=llm_trace_id,
            )
            repaired_verdict = await _verify_dialog_compliance(
                surface_output=surface_output,
                generated_dialog=generated_dialog,
                current_visible_percepts=current_visible_percepts,
                llm_trace_id=llm_trace_id,
                post_repair=True,
            )
            if not repaired_verdict["aligned"]:
                await event_logging.record_model_contract_event(
                    component=DIALOG_COMPONENT,
                    stage_name="dialog_compliance",
                    violation_kind="semantic_dialog_misalignment",
                    missing_fields=[],
                    invalid_fields=repaired_verdict["issues"],
                    repair_used=True,
                    status="failed",
                    correlation_id=llm_trace_id,
                )
                raise StateContractError(
                    "dialog remains hard-invalid after one repair"
                )
            await event_logging.record_model_contract_event(
                component=DIALOG_COMPONENT,
                stage_name="dialog_compliance",
                violation_kind="semantic_dialog_misalignment",
                missing_fields=[],
                invalid_fields=repair_issues,
                repair_used=True,
                status="repaired",
                correlation_id=llm_trace_id,
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


async def _repair_dialog_hard_failure(
    *,
    generated_dialog: list[str],
    repair_issues: list[str],
    current_visible_percepts: list[dict[str, Any]],
    surface_output: TextSurfaceOutputV2,
    user_name: str,
    llm_trace_id: str,
) -> list[str]:
    """Repair one verified hard error without a drifted content plan."""

    system_message = SystemMessage(
        content=_V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    )
    payload = {
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "current_visible_percepts": current_visible_percepts,
        "original_final_dialog": generated_dialog,
        "permitted_action_results": list(
            surface_output["permitted_action_results"]
        ),
        "user_name": user_name,
        "verified_hard_issues": repair_issues,
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_generator_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_generator_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    repaired_dialog = _validated_dialog_messages(parsed)
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name="dialog_generator_repair",
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["final_dialog"],
    )
    return repaired_dialog


_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT = '''Check semantic fidelity for one
character response by meaning, not wording overlap. current_visible_percepts
contains the current input and typed scene roles; candidate_role_frame defines
candidate pronouns. A percept's role_explicit_content is the upstream
LLM-resolved meaning with literal current_user and self roles. Use it for
nested actor/action/target direction while retaining content as evidence.
When response_operation exists, response_owner_role, selection_owner_role,
selection_required, embedded_actor_role, and embedded_target_role are
authoritative. If selection_required is true, asking another role to choose is
a subject reversal.

Mark aligned false only for:
1. An internal contradiction inside the candidate response.
2. A direct conflict with the current user input.
3. An actor, action, target, beneficiary, or subject reversal. Resolve the
percept roles and candidate_role_frame separately, then compare direction.

A role reversal requires one unambiguous reading established by current
grammar and context. Treat jokes, double entendres, and ellipsis that permit
multiple reasonable role readings as aligned.

Coherent invention, compatible future content, playful conditions, strong
personality, ask-backs, drift, and make-up content are not failures when
coherent with the current input and resolved roles. Add no style requirement.

# Output Format
Return exactly one JSON object with exactly aligned and issues. aligned is a
boolean. issues is a duplicate-free list of zero to four concise hard-failure
strings, each at most 300 characters. Use an empty issues list when aligned is
true and at least one issue when aligned is false.
'''
_dialog_semantic_fidelity_llm = LLInterface()
_dialog_semantic_fidelity_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.semantic_fidelity",
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


async def _verify_dialog_semantic_fidelity(
    *,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check contradiction and resolved semantic-role direction."""

    system_message = SystemMessage(
        content=_V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "current_visible_percepts": current_visible_percepts,
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_semantic_fidelity_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_semantic_fidelity_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_compliance_verdict(
        parsed,
        max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
    )
    trace_stage_name = (
        "dialog_semantic_fidelity_recheck"
        if post_repair
        else "dialog_semantic_fidelity_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_semantic_fidelity_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_semantic_fidelity_recheck"
            if post_repair
            else "dialog_semantic_fidelity"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


_V2_DIALOG_ROLE_DIRECTION_PROMPT = '''Verify only required response and role
direction for one character reply. candidate_role_frame defines the reply's
pronouns. required_role_operations contains typed meanings already resolved by
the upstream decontextualizer. Treat self as the active character and
current_user as the current user.

For every required operation, preserve response_owner_role,
selection_owner_role, embedded_actor_role, and embedded_target_role. When
selection_required is true, selection_owner_role must choose or state the
requested action. Mark aligned false when the reply instead asks or tells
another role to make that required selection, or clearly reverses the embedded
actor and target.

The character may refuse, negotiate, add a condition, or decline to perform an
action without reversing role direction. Treat jokes, double entendres,
ellipsis, and wording with multiple reasonable role readings as aligned.
Ignore style, novelty, intimacy, safety, action execution, and writing quality.

# Output Format
Return exactly one JSON object with exactly aligned and issues. aligned is a
boolean. issues is a duplicate-free list of zero to four concise role-direction
failures, each at most 300 characters. Use an empty issues list when aligned is
true and at least one issue when aligned is false.
'''
_dialog_role_direction_llm = LLInterface()
_dialog_role_direction_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.role_direction",
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


def _required_selection_role_operations(
    current_visible_percepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project only typed operations that require a semantic selection."""

    required_operations: list[dict[str, Any]] = []
    for percept in current_visible_percepts:
        operation = percept.get("response_operation")
        if not isinstance(operation, dict):
            continue
        if operation.get("selection_required") is not True:
            continue
        projected_percept: dict[str, Any] = {
            "input_source": percept.get("input_source", ""),
            "content": percept.get("content", ""),
            "role_explicit_content": percept.get(
                "role_explicit_content",
                "",
            ),
            "response_operation": {
                "operation": operation.get("operation", ""),
                "response_owner_role": operation.get(
                    "response_owner_role",
                    "",
                ),
                "selection_owner_role": operation.get(
                    "selection_owner_role",
                    "",
                ),
                "selection_required": True,
                "embedded_actor_role": operation.get(
                    "embedded_actor_role",
                    "",
                ),
                "embedded_target_role": operation.get(
                    "embedded_target_role",
                    "",
                ),
            },
        }
        required_operations.append(projected_percept)
    return required_operations


async def _verify_dialog_role_direction(
    *,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check nested role direction when typed input requires a selection."""

    required_operations = _required_selection_role_operations(
        current_visible_percepts
    )
    if not required_operations:
        return {"aligned": True, "issues": []}

    system_message = SystemMessage(
        content=_V2_DIALOG_ROLE_DIRECTION_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "candidate_role_frame": dict(_CANDIDATE_ROLE_FRAME),
        "required_role_operations": required_operations,
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_role_direction_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_role_direction_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_compliance_verdict(
        parsed,
        max_issues=MAX_FOCUSED_VERIFIER_ISSUES,
    )
    trace_stage_name = (
        "dialog_role_direction_recheck"
        if post_repair
        else "dialog_role_direction_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_role_direction_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_role_direction_recheck"
            if post_repair
            else "dialog_role_direction"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


_V2_DIALOG_SURFACE_INTEGRITY_PROMPT = '''Check surface integrity using the
candidate response and exact permitted_action_results.

Mark aligned false only for:
1. A claim that the character brain completed a system, tool, platform, or
other capability without a matching executed permitted result. Bound an
executed claim to that result's action_kind, semantic_result, and target_roles.
Scheduled or pending is incomplete; failed or unavailable proves no success.
Physical roleplay, body states, consent/refusal, requests, invitations, and
future, conditional, or hypothetical events are not capability execution.
Action description in plain, bracketed, first-person, or third-person form is
valid roleplay and is not a failure.

Coherent invention, creative language, personality, drift, and make-up content
are not failures. Add no quality or style requirement.

# Output Format
Return one JSON object with exactly aligned and issues. issues is a
duplicate-free list of zero to four objects with exactly kind, evidence, and
explanation. kind is false_execution.
evidence copies one exact non-empty candidate substring. explanation states
the concrete conflict in one sentence. Use no issues when aligned is true and
at least one when false.
'''
_dialog_surface_integrity_llm = LLInterface()
_dialog_surface_integrity_llm_config = LLMCallConfig(
    stage_name=f"{__name__}.surface_integrity",
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


async def _verify_dialog_surface_integrity(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Check literal-speech boundaries and exact action execution truth."""

    system_message = SystemMessage(
        content=_V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )
    payload = {
        "candidate_final_dialog": generated_dialog,
        "permitted_action_results": list(
            surface_output["permitted_action_results"]
        ),
    }
    human_message = HumanMessage(content=json.dumps(
        payload,
        ensure_ascii=False,
    ))
    started_at = time.perf_counter()
    response = await _dialog_surface_integrity_llm.ainvoke(
        [system_message, human_message],
        config=_dialog_surface_integrity_llm_config,
    )
    parsed = parse_llm_json_output(response.content)
    verdict = _validate_surface_compliance_verdict(
        parsed,
        generated_dialog=generated_dialog,
    )
    trace_stage_name = (
        "dialog_surface_integrity_recheck"
        if post_repair
        else "dialog_surface_integrity_verifier"
    )
    await llm_tracing.record_llm_trace_step(
        trace_id=llm_trace_id,
        stage_name=trace_stage_name,
        route_name="DIALOG_GENERATOR_LLM",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        messages=[system_message, human_message],
        response_text=str(response.content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=["dialog_surface_integrity_verdict"],
    )
    await event_logging.record_llm_stage_event(
        component=DIALOG_COMPONENT,
        stage_name=(
            "dialog_surface_integrity_recheck"
            if post_repair
            else "dialog_surface_integrity"
        ),
        route_name="verify",
        model_name=DIALOG_GENERATOR_LLM_MODEL,
        status="succeeded",
        prompt_chars=len(system_message.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status="succeeded",
        retry_count=int(post_repair),
        json_repair_used=False,
        duration_ms=_elapsed_ms(started_at),
        severity="info",
        correlation_id=llm_trace_id,
    )
    return verdict


async def _verify_dialog_compliance(
    *,
    surface_output: TextSurfaceOutputV2,
    generated_dialog: list[str],
    current_visible_percepts: list[dict[str, Any]],
    llm_trace_id: str,
    post_repair: bool = False,
) -> dict[str, Any]:
    """Run the three focused checks and merge bounded verdict shapes."""

    semantic_verdict, role_verdict, surface_verdict = await asyncio.gather(
        _verify_dialog_semantic_fidelity(
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
        _verify_dialog_role_direction(
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
        _verify_dialog_surface_integrity(
            surface_output=surface_output,
            generated_dialog=generated_dialog,
            current_visible_percepts=current_visible_percepts,
            llm_trace_id=llm_trace_id,
            post_repair=post_repair,
        ),
    )
    issues: list[str] = []
    combined_issues = (
        semantic_verdict["issues"]
        + role_verdict["issues"]
        + surface_verdict["issues"]
    )
    for issue in combined_issues:
        if issue not in issues:
            issues.append(issue)
    merged_verdict: dict[str, Any] = {
        "aligned": (
            semantic_verdict["aligned"]
            and role_verdict["aligned"]
            and surface_verdict["aligned"]
        ),
        "issues": issues,
    }
    return _validate_compliance_verdict(
        merged_verdict,
        max_issues=MAX_MERGED_VERIFIER_ISSUES,
    )



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


def _current_visible_percepts(
    episode: CognitiveEpisode,
) -> list[dict[str, Any]]:
    """Project current model-visible percepts within the shared prompt bound."""

    percepts = project_model_visible_percepts(episode)
    serialized = json.dumps(percepts, ensure_ascii=False)
    if len(serialized) > 24000:
        raise StateContractError("current visible percepts exceed dialog bounds")
    return percepts


def _validate_compliance_verdict(
    value: object,
    *,
    max_issues: int,
) -> dict[str, Any]:
    """Validate one exact verdict shape and its caller-owned issue bound."""

    if not isinstance(value, dict) or set(value) != {"aligned", "issues"}:
        raise StateContractError("dialog compliance fields are not exact")
    aligned = value["aligned"]
    issues = value["issues"]
    if not isinstance(aligned, bool):
        raise StateContractError("dialog compliance aligned must be boolean")
    if not isinstance(issues, list) or len(issues) > max_issues:
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


def _validate_surface_compliance_verdict(
    value: object,
    *,
    generated_dialog: list[str],
) -> dict[str, Any]:
    """Validate evidence-bearing surface issues and flatten them for repair."""

    if not isinstance(value, dict) or set(value) != {"aligned", "issues"}:
        raise StateContractError("surface compliance fields are not exact")
    aligned = value["aligned"]
    issues = value["issues"]
    if not isinstance(aligned, bool):
        raise StateContractError("surface compliance aligned must be boolean")
    if (
        not isinstance(issues, list)
        or len(issues) > MAX_FOCUSED_VERIFIER_ISSUES
    ):
        raise StateContractError("surface compliance issues are invalid")
    candidate_text = "\n".join(generated_dialog)
    normalized_rows: list[tuple[str, str, str]] = []
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {
            "kind",
            "evidence",
            "explanation",
        }:
            raise StateContractError("surface issue fields are not exact")
        kind = issue["kind"]
        evidence = issue["evidence"]
        explanation = issue["explanation"]
        if kind not in {
            "false_execution",
        }:
            raise StateContractError("surface issue kind is invalid")
        if (
            not isinstance(evidence, str)
            or not evidence.strip()
            or len(evidence) > 120
            or evidence not in candidate_text
        ):
            raise StateContractError("surface issue evidence is invalid")
        if (
            not isinstance(explanation, str)
            or not explanation.strip()
            or len(explanation) > 140
        ):
            raise StateContractError("surface issue explanation is invalid")
        normalized_rows.append((kind, evidence, explanation))
    if len(normalized_rows) != len(set(normalized_rows)):
        raise StateContractError("surface compliance issues are duplicated")
    if aligned and normalized_rows:
        raise StateContractError("aligned surface cannot contain issues")
    if not aligned and not normalized_rows:
        raise StateContractError("misaligned surface requires issues")
    normalized_issues = [
        f"{kind}: {evidence!r} - {explanation}"
        for kind, evidence, explanation in normalized_rows
    ]
    return {
        "aligned": aligned,
        "issues": normalized_issues,
    }
