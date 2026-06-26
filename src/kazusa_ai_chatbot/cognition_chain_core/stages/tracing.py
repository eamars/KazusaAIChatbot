"""Shared best-effort trace helpers for cognition LLM stages."""

import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_chain_core.contracts import LLMStageBinding


async def record_cognition_stage_trace(
    *,
    state: dict[str, Any],
    stage_name: str,
    llm: LLMStageBinding,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    output_state_fields: Sequence[str],
    started_at: float,
) -> None:
    """Record one cognition LLM stage without affecting the live path."""

    await llm_tracing.record_llm_trace_step(
        trace_id=str(state.get("llm_trace_id", "")),
        stage_name=stage_name,
        route_name=llm.config.route_name,
        model_name=llm.config.model,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        output_state_fields=output_state_fields,
    )
