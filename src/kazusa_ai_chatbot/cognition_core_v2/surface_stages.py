"""Four bounded V2 text-surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


STYLE_SYSTEM_PROMPT = '''Choose context-appropriate speech style from the
expression policy, semantic affect, semantic relationship, interaction style,
and character_voice_context. Decide only lexical register, sentence shape,
rhythm, hesitation, and punctuation. Express the selected intention and bids
without changing their meaning or adding a content beat. Convert voice traits
into wording guidance for chat-ready character text rather than camera, scene,
or performance instructions.

Treat character-owned reflection and internal observation as context rather
than live user speech. Operational metadata is not wording. Write new free
text in Simplified Chinese while
preserving quoted user text, proper nouns, code, URLs, and schema or enum
tokens. Return style guidance rather than final dialog.

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


CONTENT_PLAN_SYSTEM_PROMPT = '''Plan the visible content that best expresses
the selected character judgment in this current scene. Use the selected
intention, primary and supporting bids, visible episode, semantic affect,
semantic relationship, expression policy, interaction style, and permitted
action results.

# Planning Procedure
1. Answer or engage with the current input in the way chosen by cognition.
Keep the response appropriate to the character's relationship, emotion, and
scene pressure rather than mechanically copying earlier dialog.
2. Choose vivid, character-specific content. Coherent imaginative detail and
playful development are welcome when they do not contradict the current input
or an explicit active constraint and do not reverse actor, target,
beneficiary, or subject roles.
3. Treat typed visible-percept roles as authoritative. For user dialog,
current_user owns first-person pronouns; self is the active character and
direct addressee; self is also an implicit imperative subject.
4. permitted_action_results is the exact character-brain capability ledger.
Only executed supports its bounded completed effect; other statuses support no
completed effect. A request or bid supports a verbal or roleplayed stance, not
capability execution.
5. In-character action description is valid visible roleplay in plain,
bracketed, first-person, or third-person form when it fits the current scene.

Return a concise plan plus one to eight semantic requirements that protect
the chosen meaning, real active boundaries, role direction, and action truth.
Treat character-owned reflection and internal observation as context rather
than live user speech. Operational metadata is not prose. Write new free text
in Simplified Chinese while
preserving quoted user text, proper nouns, code, URLs, and schema or enum
tokens. Keep machine role tokens in structured input only; use natural Chinese
participant descriptions in free text. Do not write final dialog.

# Output Format
Return exactly one JSON object with exactly content_plan and
content_requirements. content_plan must be one non-empty string of at most
1000 characters. content_requirements must be a duplicate-free list of one to
eight non-empty semantic requirement strings of at most 500 characters each.'''


async def run_content_plan_stage(
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> tuple[str, list[str]]:
    """Run the content prompt and return its plan and semantic requirements."""

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
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "content_plan",
        "content_requirements",
    }:
        raise ValueError("content-plan stage fields are not exact")
    content_plan = _bounded_text(parsed["content_plan"], "content plan", 1000)
    content_requirements = _bounded_text_list(
        parsed["content_requirements"],
        "content requirements",
    )
    return content_plan, content_requirements


PREFERENCE_SYSTEM_PROMPT = '''Identify which real visible boundary or
addressee constraint exists, if any, in the selected character judgment and
current scene. Use the selected intention, visible episode, projected bids,
expression policy, semantic affect, semantic relationship, interaction style,
and permitted action results as context.

visible_boundaries contains only active expression limits or permitted-detail
limits. addressee_plan contains only actual semantic addressee handling.
Return an empty list when none exists instead of inventing resistance or an
addressee rule. Treat action-result status exactly: only executed supports its
bounded completed effect; other statuses retain their stated meaning.
Character-owned reflection is context, not live user speech; operational
metadata is not dialog content.

Write new free text in Simplified Chinese while preserving quoted user text,
proper nouns, code, URLs, and schema or enum tokens. Return planning fields
rather than final dialog.

# Output Format
Return exactly one JSON object with exactly visible_boundaries and
addressee_plan. Each value must be a duplicate-free list containing zero to
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
        _bounded_text_list(
            parsed["visible_boundaries"],
            "visible boundaries",
            minimum=0,
        ),
        _bounded_text_list(
            parsed["addressee_plan"],
            "addressee plan",
            minimum=0,
        ),
    )


VISUAL_SYSTEM_PROMPT = '''Create image-generation visual_directives for a
terminal image surface from the selected intention, visible episode, projected
bids, expression policy, semantic affect, semantic relationship, permitted
action results, interaction style context, and character_voice_context. The
directives may include physically visible character traits, pose, expression,
composition, environment, and scene atmosphere that serve the selected
surface intent. They are private image-oriented instructions, not text to send
to the user, dialog guidance, or an instruction to invoke another model or
handler. Do not write final dialogue. Write newly generated free text in
Simplified Chinese while preserving quoted user text, proper nouns, code,
URLs, and schema or enum tokens when needed. Treat character-owned reflection
or internal observation as evidence, never as live user speech. Do not copy
source-packet headings, timestamps, transport summaries, schema keys, or
operational metadata.

# Output Format
Return exactly one JSON object with exactly visual_directives. Its value must
be one non-empty string of at most 1000 characters.'''


async def run_visual_stage(
    payload: Mapping[str, Any],
    services: VisualSurfaceServicesV2,
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
    if not isinstance(parsed, Mapping) or set(parsed) != {"visual_directives"}:
        raise ValueError("visual stage fields are not exact")
    visual_directives = _bounded_text(
        parsed["visual_directives"],
        "visual directives",
        1000,
    )
    return visual_directives


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


def _bounded_text_list(
    value: Any,
    label: str,
    *,
    minimum: int = 1,
) -> list[str]:
    """Validate one duplicate-free text list against its stage cardinality."""

    if not isinstance(value, list) or not minimum <= len(value) <= 8:
        raise ValueError(f"{label} is invalid")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    for item in value:
        _bounded_text(item, label, 500)
    return list(value)
