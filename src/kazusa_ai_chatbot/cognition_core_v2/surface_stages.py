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


SURFACE_STAGE_PROMPTS = {
    "style": (
        "Choose bounded style guidance from the supplied expression policy "
        "without writing dialogue."
    ),
    "content_plan": (
        "Plan visible content from the selected semantic intention without "
        "writing dialogue."
    ),
    "preference": (
        "Choose addressee and preference-sensitive boundaries without writing "
        "dialogue."
    ),
    "visual": "Choose pacing and visual-directive guidance without writing dialogue.",
}

SURFACE_STAGE_SYSTEM_PROMPT = (
    "Return exactly one JSON object with exactly one key, result. The result "
    "value must be one concise semantic surface guidance string of at most "
    "1000 characters. Do not return any other keys, nested objects, dialogue, "
    "or numeric fields. Write newly generated free text in Simplified Chinese, "
    "while preserving quoted user text, proper nouns, code, URLs, and schema "
    "or enum tokens when needed. Treat character-owned reflection or internal "
    "observation as evidence, never as live user speech, and do not copy "
    "source-packet or operational metadata."
)


async def run_surface_stage(
    stage_name: str,
    payload: Mapping[str, Any],
    services: TextSurfaceServicesV2,
) -> str:
    """Run one local stage with a bounded prompt and semantic text result."""

    if stage_name not in SURFACE_STAGE_PROMPTS:
        raise ValueError(f"unknown surface stage: {stage_name}")
    config = getattr(services, f"{stage_name}_config")
    prompt_payload = {
        "stage": stage_name,
        "instruction": SURFACE_STAGE_PROMPTS[stage_name],
        "surface": payload,
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > 24000:
        raise ValueError("surface stage prompt exceeds the contract cap")
    response = await services.llm.ainvoke(
        [
            SystemMessage(content=SURFACE_STAGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=config,
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, Mapping) or set(parsed) != {"result"}:
        raise ValueError(
            f"{stage_name} stage result must contain exactly result"
        )
    result = parsed["result"]
    if not isinstance(result, str):
        raise ValueError(f"{stage_name} stage result must be text")
    return _bounded_result(result)


def _bounded_result(value: str) -> str:
    """Keep stage text bounded before the final surface contract."""

    if not value.strip() or len(value) > 1000:
        raise ValueError("surface stage text is invalid")
    return value
