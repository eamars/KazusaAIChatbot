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


STYLE_SYSTEM_PROMPT = '''Choose bounded speech-safe style guidance from the
expression policy, semantic affect, semantic relationship, and interaction
style context. Use character_voice_context only for character wording and
cadence. Its physical or visual traits must never become narrated action,
stage direction, camera direction, scene direction, or performance cues in
text. Return guidance only for lexical register and wording, sentence length
and shape, rhythm, hesitation, and punctuation.
style_guidance must never suggest any detail, topic, example, image, action,
claim, inference, or content beat to add, even optionally. It must not select
or alter semantic content.
Treat the visible episode, selected intention, primary bid, supporting bids,
and permitted action results as grounding constraints.
Do not write final dialogue. Write newly generated free text in Simplified Chinese while
preserving quoted user text, proper nouns, code, URLs, and schema or enum
tokens when needed. Treat character-owned reflection or internal observation
as evidence, never as live user speech. Do not copy source-packet headings,
timestamps, transport summaries, schema keys, or operational metadata.

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


CONTENT_PLAN_SYSTEM_PROMPT = '''Plan speech-safe visible content from the
selected semantic intention, primary bid, supporting bids, permitted details,
visible episode, semantic affect, semantic relationship, and permitted action
results. Treat the expression policy and interaction style context as
constraints. Preserve the current user's requested response operation,
including whether the response should answer, infer, explain, ask, accept,
refuse, or negotiate. Preserve every actor, action, target or beneficiary,
semantic claim, condition, required content beat, present or future time scope,
topic, and visible limit. Preserve source descriptors, attributes, qualifiers,
quantities, polarity, and comparative degree. Non-conflicting elaboration is
allowed, but it must not transform, replace, or compound a supplied attribute
into a different claim. Preserve explicit entity and target specificity.
Typed role fields on a visible percept are authoritative. For user dialog,
speaker_role=current_user owns first-person pronouns, addressee_role=self is
the active character, and implicit_imperative_subject_role=self owns an
unstated command subject. Never reverse those roles even when an upstream bid
does.
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
it. Do not substitute a different operation, reverse a semantic role, invent
a condition or future rule, or introduce an unrelated topic. Record these
invariants as explicit content requirements for the final renderer.
The text channel has no physical actuator. For a physical request, plan only
the character's literal verbal stance: acceptance, refusal, negotiation,
teasing, bounded permission, or spoken instruction. Never plan or require a
first-person execution claim that the requested movement is happening,
finished, or established as a body position, even if an upstream bid describes
execution. Preserve the character's stance while keeping physical enactment
out of visible text.
A verbal offer or permission remains allowed, but never plan a claim or
presupposition that the requested physical act was performed, completed,
delivered, or received, regardless of grammatical person. Do not treat virtual
or simulated delivery as an exception.
Physical-action topics may be discussed, accepted, refused, or negotiated in
words, while the visible text remains literal speech rather than narrated
execution. Do not write final dialogue. Write newly generated free text in
Simplified Chinese while preserving quoted user text, proper nouns,
code, URLs, and schema or enum tokens when needed. Treat character-owned
reflection or internal observation as evidence, never as live user speech. Do
not copy source-packet headings, timestamps, transport summaries, schema keys,
or operational metadata.

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


def _bounded_text_list(value: Any, label: str) -> list[str]:
    """Validate one bounded duplicate-free stage-owned text list."""

    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError(f"{label} is invalid")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    for item in value:
        _bounded_text(item, label, 500)
    return list(value)
