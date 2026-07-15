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
from kazusa_ai_chatbot.nodes.linguistic_texture import (
    get_hesitation_density_description,
    get_fragmentation_description,
    get_counter_questioning_description,
    get_softener_density_description,
    get_formalism_avoidance_description,
    get_abstraction_reframing_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
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

Do not re-evaluate permission, stance, truth, safety, relationship meaning, or
whether to answer. Do not add facts, actions, promises, targets, or questions
that are absent from the supplied surface output. Preserve the content plan
and visible boundaries. Use the character profile only to determine wording,
rhythm, and voice. Return 1-N complete text messages in `final_dialog`.

Input JSON:
{{
    "text_surface_output_v2": {{
        "schema_version": "text_surface_output.v2",
        "content_plan": "string",
        "visible_boundaries": ["string"],
        "addressee_plan": ["string"],
        "style_guidance": "string",
        "pacing_guidance": "string",
        "selected_surface_intent": "string"
    }},
    "user_name": "string"
}}

Return only this JSON object, without Markdown fences:
{{
    "final_dialog": ["complete visible message"]
}}
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
    dialog_prompt = _V2_DIALOG_GENERATOR_PROMPT
    ltp = state["character_profile"]["linguistic_texture_profile"]
    system_prompt = SystemMessage(content=dialog_prompt.format(
        character_name=state["character_profile"]["name"],
        character_logic=state["character_profile"]["personality_brief"]["logic"],
        character_tempo=state["character_profile"]["personality_brief"]["tempo"],
        character_defense=state["character_profile"]["personality_brief"]["defense"],
        character_quirks=state["character_profile"]["personality_brief"]["quirks"],
        character_taboos=state["character_profile"]["personality_brief"]["taboos"],
        ltp_hesitation_density=get_hesitation_density_description(ltp["hesitation_density"]),
        ltp_fragmentation=get_fragmentation_description(ltp["fragmentation"]),
        ltp_emotional_leakage=get_emotional_leakage_description(ltp["emotional_leakage"]),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(ltp["rhythmic_bounce"]),
        ltp_direct_assertion=get_direct_assertion_description(ltp["direct_assertion"]),
        ltp_softener_density=get_softener_density_description(ltp["softener_density"]),
        ltp_counter_questioning=get_counter_questioning_description(ltp["counter_questioning"]),
        ltp_formalism_avoidance=get_formalism_avoidance_description(ltp["formalism_avoidance"]),
        ltp_abstraction_reframing=get_abstraction_reframing_description(ltp["abstraction_reframing"]),
        ltp_self_deprecation=get_self_deprecation_description(ltp["self_deprecation"]),
    ))

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
