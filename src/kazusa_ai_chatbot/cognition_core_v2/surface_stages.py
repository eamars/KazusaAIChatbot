"""Four bounded V2 text-surface stage handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceServicesV2,
)


SURFACE_STAGE_PROMPTS = {
    "style": "Choose bounded style guidance from the supplied expression policy.",
    "content_plan": "Plan visible content from the selected semantic intention.",
    "preference": "Choose addressee and preference-sensitive boundaries.",
    "visual": "Choose pacing and visual-directive guidance without writing dialogue.",
}


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
            SystemMessage(content="Return one concise semantic surface plan."),
            HumanMessage(content=prompt_text),
        ],
        config=config,
    )
    parsed = services.parse_json(response.content)
    if isinstance(parsed, str):
        return _bounded_result(parsed)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{stage_name} stage result must be an object")
    for field_name in ("result", "content", stage_name):
        value = parsed.get(field_name)
        if isinstance(value, str):
            return _bounded_result(value)
    raise ValueError(f"{stage_name} stage result has no semantic text")


def _bounded_result(value: str) -> str:
    """Keep stage text bounded before the final surface contract."""

    if not value.strip() or len(value) > 1000:
        raise ValueError("surface stage text is invalid")
    return value
