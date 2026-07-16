"""Compact frontline relevance routing for each active chat message."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.config import (
    FRONTLINE_RELEVANCE_MAX_INPUT_CHARS,
    RELEVANCE_AGENT_LLM_API_KEY,
    RELEVANCE_AGENT_LLM_BASE_URL,
    RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS,
    RELEVANCE_AGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


logger = logging.getLogger(__name__)

FRONTLINE_RELEVANCE_COMPLETION_TOKEN_CAP = 256
FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS = min(
    RELEVANCE_AGENT_LLM_MAX_COMPLETION_TOKENS,
    FRONTLINE_RELEVANCE_COMPLETION_TOKEN_CAP,
)


class FrontlineDecision(TypedDict):
    """Validated semantic action returned by the frontline route."""

    intake_action: Literal["discard", "start", "append"]
    append_target: Literal["none", "open_1", "open_2", "open_3"]
    prelude_targets: list[Literal["prelude_1", "prelude_2"]]
    reason: str


FrontlineState = Mapping[str, Any]


_FRONTLINE_SYSTEM_PROMPT = '''You are the compact frontline intake judge for a character brain.
Answer only the routing question for the current message.

# Decision Contract
- First obey the typed target and reply labels. When every supplied target and
  reply label is other_participant and none is character, the message is not
  character-relevant: discard it regardless of topic interest or open turns;
- discard only clearly unrelated third-party traffic;
- start a bounded candidate when the message is character-relevant but does
  not continue a supplied open turn;
- append only when the message clearly continues exactly one supplied open
  turn.
Append includes a clarification, correction, replacement, delayed attachment,
or completion whose meaning continues a supplied open intent.
Typed target and reply labels are authoritative. The label character means the
active character; other_participant means somebody else. Direct character
address, a reply to the character, and ambiguous short or attachment-only
continuations are admitted conservatively. A new character-relevant topic is
start, even when unrelated to every open turn. Same-author timing alone never
proves append.

# Generation Procedure
1. Check typed target and reply evidence for character relevance.
2. Compare the current intent with only the supplied open turns.
3. If starting a turn, select supplied recent preludes only when they complete
   the current intent.
4. Verify that the action and slot fields obey the output contract.

# Output Format
Return exactly one JSON object and no surrounding text:
{"intake_action":"discard|start|append",
"append_target":"none|open_1|open_2|open_3","prelude_targets":[],
"reason":"at most 80 characters"}
For discard or start, append_target is none. For append, choose one supplied
open slot. prelude_targets may contain up to two supplied prelude_1 or prelude_2
labels only for start; otherwise use an empty list. Never invent a slot.'''

_frontline_relevance_agent_llm = LLInterface()
_frontline_relevance_agent_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RELEVANCE_AGENT_LLM",
    base_url=RELEVANCE_AGENT_LLM_BASE_URL,
    api_key=RELEVANCE_AGENT_LLM_API_KEY,
    model=RELEVANCE_AGENT_LLM_MODEL,
    temperature=0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=FRONTLINE_RELEVANCE_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=False),
)


def _clip_text(value: object, limit: int) -> str:
    """Clip one model-facing text field while retaining both ends."""

    if limit <= 0:
        return_value = ""
        return return_value
    if not isinstance(value, str):
        return_value = ""
        return return_value

    clean_value = value.strip()
    if len(clean_value) <= limit:
        return_value = clean_value
        return return_value

    head_length = max(1, (limit - 3) // 2)
    tail_length = max(1, limit - 3 - head_length)
    return_value = (
        clean_value[:head_length]
        + "..."
        + clean_value[-tail_length:]
    )
    return return_value


def _string_list(value: object, *, limit: int) -> list[str]:
    """Return a bounded list of semantic labels only."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return_value: list[str] = []
        return return_value

    labels = [
        _clip_text(item, 40)
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return_value = labels[:limit]
    return return_value


def _project_current_message(value: object) -> dict[str, Any]:
    """Project current typed message evidence without transport metadata."""

    if not isinstance(value, Mapping):
        return_value = {
            "body_text": "",
            "semantic_target_labels": [],
            "reply_target_label": "none",
            "media_labels": [],
        }
        return return_value

    return_value = {
        "body_text": _clip_text(value.get("body_text"), 2000),
        "semantic_target_labels": _string_list(
            value.get("semantic_target_labels"),
            limit=4,
        ),
        "reply_target_label": _clip_text(
            value.get("reply_target_label"),
            60,
        ) or "none",
        "media_labels": _string_list(value.get("media_labels"), limit=4),
    }
    return return_value


def _project_open_turns(value: object) -> list[dict[str, str]]:
    """Project at most three supplied open-turn slots."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return_value: list[dict[str, str]] = []
        return return_value

    projected: list[dict[str, str]] = []
    for index, item in enumerate(value[:3], start=1):
        if not isinstance(item, Mapping):
            continue
        projected.append({
            "slot": f"open_{index}",
            "author_relation": _clip_text(item.get("author_relation"), 60),
            "latest_intent": _clip_text(item.get("latest_intent"), 180),
            "opening_excerpt": _clip_text(item.get("opening_excerpt"), 180),
            "target_summary": _clip_text(item.get("target_summary"), 80),
            "reply_summary": _clip_text(item.get("reply_summary"), 80),
            "media_summary": _clip_text(item.get("media_summary"), 80),
        })
    return_value = projected
    return return_value


def _project_preludes(value: object) -> list[dict[str, str]]:
    """Project at most two recent silent same-author preludes."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return_value: list[dict[str, str]] = []
        return return_value

    projected: list[dict[str, str]] = []
    for index, item in enumerate(value[:2], start=1):
        if not isinstance(item, Mapping):
            continue
        projected.append({
            "slot": f"prelude_{index}",
            "summary": _clip_text(item.get("summary"), 180),
            "target_summary": _clip_text(item.get("target_summary"), 60),
        })
    return_value = projected
    return return_value


def _project_frontline_state(state: FrontlineState) -> dict[str, Any]:
    """Build the bounded semantic frontline payload."""

    return_value = {
        "current_message": _project_current_message(
            state.get("current_message")
        ),
        "open_turns": _project_open_turns(state.get("open_turns")),
        "recent_preludes": _project_preludes(state.get("recent_preludes")),
        "latest_bot_continuity": _clip_text(
            state.get("latest_bot_continuity"),
            400,
        ),
    }
    return return_value


def build_frontline_messages(
    state: FrontlineState,
) -> tuple[SystemMessage, HumanMessage]:
    """Render the bounded frontline system and semantic human messages."""

    payload = _project_frontline_state(state)
    human_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    available_human_chars = FRONTLINE_RELEVANCE_MAX_INPUT_CHARS - len(
        _FRONTLINE_SYSTEM_PROMPT
    )
    if len(human_content) > available_human_chars:
        payload["latest_bot_continuity"] = ""
        payload["recent_preludes"] = []
        payload["current_message"]["body_text"] = _clip_text(
            payload["current_message"].get("body_text"),
            max(0, available_human_chars // 2),
        )
        human_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if len(human_content) > available_human_chars:
        payload["open_turns"] = payload["open_turns"][:1]
        payload["current_message"]["body_text"] = _clip_text(
            payload["current_message"].get("body_text"),
            400,
        )
        human_content = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if len(human_content) > available_human_chars:
        raise ValueError("frontline semantic projection exceeds its hard cap")

    messages = (
        SystemMessage(content=_FRONTLINE_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    )
    return_value = messages
    return return_value


def validate_frontline_decision(raw: Mapping[str, Any]) -> FrontlineDecision:
    """Validate and bound one frontline model object."""

    if not isinstance(raw, Mapping):
        raise ValueError("frontline output must be an object")

    action = raw.get("intake_action")
    append_target = raw.get("append_target")
    prelude_targets = raw.get("prelude_targets")
    reason = raw.get("reason")
    if action not in {"discard", "start", "append"}:
        raise ValueError("frontline intake_action is invalid")
    if append_target not in {"none", "open_1", "open_2", "open_3"}:
        raise ValueError("frontline append_target is invalid")
    if not isinstance(prelude_targets, list):
        raise ValueError("frontline prelude_targets must be a list")
    if not isinstance(reason, str):
        raise ValueError("frontline reason must be a string")

    clean_preludes: list[Literal["prelude_1", "prelude_2"]] = []
    for target in prelude_targets[:2]:
        if target not in {"prelude_1", "prelude_2"}:
            raise ValueError("frontline prelude target is invalid")
        if target not in clean_preludes:
            clean_preludes.append(target)

    if action in {"discard", "start"} and append_target != "none":
        raise ValueError("frontline non-append action must use none target")
    if action == "append" and append_target == "none":
        raise ValueError("frontline append action needs an open target")
    if action != "start" and clean_preludes:
        raise ValueError("frontline preludes are valid only for start")

    return_value: FrontlineDecision = {
        "intake_action": action,
        "append_target": append_target,
        "prelude_targets": clean_preludes,
        "reason": _clip_text(reason, 80),
    }
    return return_value


def _discard_decision(reason: str) -> FrontlineDecision:
    """Return the structural fail-closed frontline decision."""

    return_value: FrontlineDecision = {
        "intake_action": "discard",
        "append_target": "none",
        "prelude_targets": [],
        "reason": _clip_text(reason, 80),
    }
    return return_value


async def frontline_relevance_agent(state: FrontlineState) -> FrontlineDecision:
    """Run the compact frontline route and return a validated action."""

    messages = build_frontline_messages(state)
    started_at = time.perf_counter()
    response = await _frontline_relevance_agent_llm.ainvoke(
        list(messages),
        config=_frontline_relevance_agent_llm_config,
    )
    parsed_output = parse_llm_json_output(
        str(response.content),
        deterministic_only=True,
    )
    parse_status = "succeeded"
    try:
        decision = validate_frontline_decision(parsed_output)
    except ValueError as exc:
        logger.warning(f"Invalid frontline output: {exc}")
        decision = _discard_decision("invalid frontline output")
        parse_status = "invalid"

    await llm_tracing.record_llm_trace_step(
        trace_id=str(state.get("llm_trace_id", "")),
        stage_name="frontline_relevance_agent",
        route_name="frontline_relevance",
        model_name=RELEVANCE_AGENT_LLM_MODEL,
        messages=list(messages),
        response_text=str(response.content),
        parsed_output=decision,
        parse_status=parse_status,
        status="succeeded",
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        output_state_fields=[
            "intake_action",
            "append_target",
            "prelude_targets",
            "reason",
        ],
    )
    return decision
