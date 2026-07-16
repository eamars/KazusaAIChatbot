"""Four bounded V2 text-surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceServicesV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


STYLE_SYSTEM_PROMPT = '''Choose bounded style guidance from the supplied
expression policy, semantic affect, semantic relationship, and interaction
style context. Treat the visible episode, selected intention, primary bid,
supporting bids, and permitted action results as grounding constraints.
Do not write final dialogue. Write newly generated free text in Simplified
Chinese while preserving quoted user text, proper nouns, code, URLs, and
schema or enum tokens when needed. Treat character-owned reflection or
internal observation as evidence, never as live user speech. Do not copy
source-packet headings, timestamps, transport summaries, schema keys, or
operational metadata.

# Output Format
Return exactly one JSON object with exactly style_guidance. Its value must be
one non-empty string of at most 1000 characters.'''


async def run_style_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run the stage-local style prompt and validate its exact field."""

    prompt_text = _surface_prompt_text(payload)
    style_llm = services.llm
    response = await style_llm.ainvoke(
        [
            SystemMessage(content=STYLE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.style_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"style_guidance"}:
        raise ValueError("style stage fields are not exact")
    return _bounded_text(parsed["style_guidance"], "style guidance", 1000)


CONTENT_PLAN_SYSTEM_PROMPT = '''
Plan visible content from the selected semantic intention, primary bid,
supporting bids, permitted details, visible episode,
semantic affect, semantic relationship, and permitted action results. Treat
the expression policy and interaction style context as constraints.
Do not write final dialogue. Write newly generated free text in Simplified
Chinese while preserving quoted user text, proper nouns, code, URLs, and
schema or enum tokens when needed. Treat character-owned reflection or
internal observation as evidence, never as live user speech. Do not copy
source-packet headings, timestamps, transport summaries, schema keys, or
operational metadata.

# Output Format
Return exactly one JSON object with exactly content_plan. Its value must be
one non-empty string of at most 1000 characters.'''


async def run_content_plan_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run the stage-local content prompt and validate its exact field."""

    prompt_text = _surface_prompt_text(payload)
    content_plan_llm = services.llm
    response = await content_plan_llm.ainvoke(
        [
            SystemMessage(content=CONTENT_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.content_plan_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"content_plan"}:
        raise ValueError("content-plan stage fields are not exact")
    return _bounded_text(parsed["content_plan"], "content plan", 1000)


PREFERENCE_SYSTEM_PROMPT = '''Choose preference-sensitive boundaries and an
addressee plan from the selected intention, visible episode, projected bids,
expression policy, semantic affect, semantic relationship, permitted action
results, and interaction style context. Treat all supplied fields as semantic
surface input rather than state authority.
Do not write final dialogue. Write newly generated free text in Simplified
Chinese while preserving quoted user text, proper nouns, code, URLs, and
schema or enum tokens when needed. Treat character-owned reflection or
internal observation as evidence, never as live user speech. Do not copy
source-packet headings, timestamps, transport summaries, schema keys, or
operational metadata.

visible_boundaries must contain only expression or visibility constraints and
permitted-detail limits; never restate user facts as boundaries. addressee_plan
must contain only intended semantic addressee handling.

# Output Format
Return exactly one JSON object with exactly visible_boundaries and
addressee_plan. Each value must be a duplicate-free list containing one to
eight non-empty strings of at most 500 characters each.'''


async def run_preference_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[list[str], list[str]]:
    """Run the stage-local preference prompt and return two distinct lists."""

    prompt_text = _surface_prompt_text(payload)
    preference_llm = services.llm
    response = await preference_llm.ainvoke(
        [
            SystemMessage(content=PREFERENCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.preference_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "visible_boundaries",
        "addressee_plan",
    }:
        raise ValueError("preference stage fields are not exact")
    return (
        _bounded_text_list(parsed["visible_boundaries"], "visible boundaries"),
        _bounded_text_list(parsed["addressee_plan"], "addressee plan"),
    )


VISUAL_SYSTEM_PROMPT = '''Choose pacing and visual-directive guidance from the
selected intention, visible episode, projected bids, expression policy,
semantic affect, semantic relationship, permitted action results, and
interaction style context. Treat every field as a bounded semantic constraint.
Do not write final dialogue. Write newly generated free text in Simplified
Chinese while preserving quoted user text, proper nouns, code, URLs, and
schema or enum tokens when needed. Treat character-owned reflection or
internal observation as evidence, never as live user speech. Do not copy
source-packet headings, timestamps, transport summaries, schema keys, or
operational metadata.

# Output Format
Return exactly one JSON object with exactly pacing_guidance. Its value must be
one non-empty string of at most 1000 characters.'''


async def run_visual_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run the stage-local visual prompt and validate its exact field."""

    prompt_text = _surface_prompt_text(payload)
    visual_llm = services.llm
    response = await visual_llm.ainvoke(
        [
            SystemMessage(content=VISUAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.visual_config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"pacing_guidance"}:
        raise ValueError("visual stage fields are not exact")
    return _bounded_text(parsed["pacing_guidance"], "pacing guidance", 1000)


def _surface_prompt_text(payload: Mapping[str, Any]) -> str:
    """Serialize one already-projected surface packet within the fixed cap."""

    prompt_text = json.dumps(
        {"surface": payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(prompt_text) > 24000:
        raise ValueError("surface stage prompt exceeds the contract cap")
    return prompt_text


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    """Validate one bounded non-empty stage-owned text field."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _bounded_text_list(value: Any, label: str) -> list[str]:
    """Validate one bounded duplicate-free stage-owned text list."""

    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError(f"{label} is invalid")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    for item in value:
        _bounded_text(item, label, 500)
    return list(value)
