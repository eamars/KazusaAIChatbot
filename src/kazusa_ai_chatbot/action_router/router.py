"""LLM call wrapper for route-only action initialization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_router.contracts import (
    normalize_action_router_output,
)
from kazusa_ai_chatbot.action_router.payload import build_action_router_payload
from kazusa_ai_chatbot.action_router.prompt import ACTION_ROUTER_PROMPT
from kazusa_ai_chatbot.action_spec.models import CapabilitySpecV1
from kazusa_ai_chatbot.utils import parse_llm_json_output


class ActionRouterLLM(Protocol):
    """Minimal async LLM interface required by the action router."""

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> object:
        """Return one model response for the supplied prompt messages."""


def build_action_router_payload_text(
    state: Mapping[str, object],
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> str:
    """Build the serialized prompt-safe action-router human payload."""

    payload = build_action_router_payload(state, capabilities)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=None)
    return payload_text


def build_action_router_messages(
    state: Mapping[str, object],
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> list[BaseMessage]:
    """Build the system and human messages for one action-router call."""

    human_payload = build_action_router_payload_text(state, capabilities)
    messages: list[BaseMessage] = [
        SystemMessage(content=ACTION_ROUTER_PROMPT),
        HumanMessage(content=human_payload),
    ]
    return messages


async def route_action_requests(
    llm: ActionRouterLLM,
    state: Mapping[str, object],
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> dict[str, object]:
    """Call the action-router LLM and normalize its raw route output."""

    messages = build_action_router_messages(state, capabilities)
    response = await llm.ainvoke(messages)
    raw_content = getattr(response, "content", None)
    if not isinstance(raw_content, str):
        raise TypeError("Action router LLM response content must be text")
    raw_parsed = parse_llm_json_output(raw_content)
    parsed = normalize_action_router_output(raw_parsed)
    return parsed
