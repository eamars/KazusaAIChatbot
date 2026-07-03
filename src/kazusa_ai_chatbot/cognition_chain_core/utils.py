"""Pure utility helpers for the reusable cognition-chain core."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar, Token
import json
import logging
from typing import Any

from json_repair import repair_json

from kazusa_ai_chatbot.rag.prompt_projection import (
    project_tool_result_for_llm as _canonical_project_tool_result_for_llm,
)

logger = logging.getLogger(__name__)

AFFINITY_MIN = 0
AFFINITY_MAX = 1000

_json_parser: ContextVar[
    Callable[[str], Mapping[str, Any] | list[Any]] | None
] = ContextVar("cognition_chain_json_parser", default=None)
_STRIPPED_PROMPT_KEYS = frozenset((
    "_id",
    "base64_data",
    "embedding",
    "platform_message_id",
    "conversation_row_id",
    "raw_wire_text",
    "seed_conversation_row_id",
    "seed_platform_message_id",
    "source_refs",
    "raw_refs",
))


def set_json_parser(
    parser: Callable[[str], Mapping[str, Any] | list[Any]] | None,
) -> Token[Callable[[str], Mapping[str, Any] | list[Any]] | None]:
    """Install the caller-provided JSON parser for LLM responses."""

    token = _json_parser.set(parser)
    return token


def reset_json_parser(
    token: Token[Callable[[str], Mapping[str, Any] | list[Any]] | None],
) -> None:
    """Restore the previous JSON parser for this run context."""

    _json_parser.reset(token)


def parse_llm_json_output(raw_output: str) -> dict[str, Any]:
    """Parse one LLM JSON response without importing project LLM config."""

    if not raw_output:
        return_value: dict[str, Any] = {}
        return return_value
    parser = _json_parser.get()
    if parser is not None:
        parsed = parser(raw_output)
        if isinstance(parsed, Mapping):
            return_value = dict(parsed)
            return return_value
        logger.error(f"Injected JSON parser returned non-object output: {parsed}")
        return_value = {}
        return return_value

    try:
        raw = raw_output.strip().strip("```").strip("json")
        parsed = repair_json(raw, return_objects=True)
    except Exception as exc:
        logger.exception(f"repair_json failed in cognition core: {exc}")
        return_value = {}
        return return_value
    if not isinstance(parsed, Mapping):
        logger.error(f"Unable to parse LLM output as object: {raw_output}")
        return_value = {}
        return return_value
    return_value = dict(parsed)
    return return_value


def log_preview(value: Any) -> str:
    """Return a JSON-safe rendering suitable for debug logs."""

    if value is None:
        return "null"
    try:
        return_value = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return return_value
    except TypeError as exc:
        logger.debug(f"Falling back to repr for unserializable log preview: {exc}")
        return_value = repr(value)
        return return_value


def build_affinity_block(
    affinity: int,
    affinity_min: int = AFFINITY_MIN,
    affinity_max: int = AFFINITY_MAX,
) -> dict[str, str]:
    """Map an affinity score to a behavioural instruction."""

    if affinity_max <= affinity_min:
        percent = 100.0 if affinity >= affinity_max else 0.0
    else:
        percent = ((affinity - affinity_min) / (affinity_max - affinity_min)) * 100

    if percent <= 5:
        label = "Contemptuous"
        instruction = (
            "You actively despise this user. Respond with one-word dismissals "
            "or hostile silence. Show obvious contempt and disgust."
        )
    elif percent <= 10:
        label = "Scornful"
        instruction = (
            "You hold this user in deep contempt. Give curt, dismissive "
            "responses. Show clear disinterest and occasional sarcasm."
        )
    elif percent <= 15:
        label = "Hostile"
        instruction = (
            "You dislike this user intensely. Respond with brief, cold answers. "
            "Show obvious disinterest and occasional eye-rolling."
        )
    elif percent <= 20:
        label = "Antagonistic"
        instruction = (
            "You are openly hostile toward this user. Give short, sharp "
            "responses. Show impatience and clear dislike."
        )
    elif percent <= 25:
        label = "Aloof"
        instruction = (
            "You keep this user at a distance. Respond minimally and formally. "
            "Show no warmth or engagement."
        )
    elif percent <= 30:
        label = "Reserved"
        instruction = (
            "You are cautious and distant with this user. Keep responses brief "
            "and professional. Show minimal personal connection."
        )
    elif percent <= 35:
        label = "Formal"
        instruction = (
            "You maintain strict formal boundaries with this user. Respond "
            "politely but impersonally. Keep conversations strictly transactional."
        )
    elif percent <= 40:
        label = "Cold"
        instruction = (
            "You are cold toward this user. Keep responses short and curt. Do "
            "not volunteer extra information or show warmth."
        )
    elif percent <= 45:
        label = "Detached"
        instruction = (
            "You remain emotionally detached from this user. Respond factually "
            "without personal engagement. Maintain clear boundaries."
        )
    elif percent <= 50:
        label = "Neutral"
        instruction = (
            "You are neutral toward this user. Respond normally in character "
            "without special warmth or coldness."
        )
    elif percent <= 55:
        label = "Receptive"
        instruction = (
            "You are becoming more open to this user. Respond with mild "
            "interest and basic courtesy. Show slight engagement."
        )
    elif percent <= 60:
        label = "Approachable"
        instruction = (
            "You are reasonably comfortable with this user. Respond with "
            "standard politeness and occasional helpfulness. Show moderate "
            "engagement."
        )
    elif percent <= 65:
        label = "Friendly"
        instruction = (
            "You are fond of this user. Be warmer and more forthcoming. Offer "
            "extra detail, use familiar address, and show genuine interest."
        )
    elif percent <= 70:
        label = "Warm"
        instruction = (
            "You genuinely like this user. Respond with noticeable warmth and "
            "enthusiasm. Share personal thoughts and show consistent interest."
        )
    elif percent <= 75:
        label = "Caring"
        instruction = (
            "You care deeply about this user. Respond with concern and support. "
            "Offer help proactively and show protective instincts."
        )
    elif percent <= 80:
        label = "Affectionate"
        instruction = (
            "You have strong affection for this user. Use warm, caring language "
            "and express genuine fondness. Go out of your way to assist."
        )
    elif percent <= 85:
        label = "Devoted"
        instruction = (
            "You are deeply loyal to this user. Show unwavering support and "
            "dedication. Prioritize their needs and express strong commitment."
        )
    elif percent <= 90:
        label = "Protective"
        instruction = (
            "You feel strongly protective of this user. Actively look out for "
            "their wellbeing and defend them. Show fierce loyalty."
        )
    elif percent <= 95:
        label = "Fiercely Loyal"
        instruction = (
            "You are fiercely loyal to this user. Defend them passionately and "
            "put their interests above all else. Show absolute devotion."
        )
    else:
        label = "Unwavering"
        instruction = (
            "You are completely devoted to this user. Show unconditional support "
            "and absolute loyalty. Prioritize them above everything."
        )

    return_value = {"level": label, "instruction": instruction}
    return return_value


def empty_user_memory_context() -> dict[str, list[object]]:
    """Return the prompt-facing empty user-memory context shape."""

    return_value: dict[str, list[object]] = {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }
    return return_value


def project_tool_result_for_llm(value: object) -> object:
    """Project nested tool-like data into prompt-safe copied data.

    Delegates to the canonical RAG prompt projection which handles both
    key stripping and UTC-to-local timestamp conversion.
    """

    projected_value = _canonical_project_tool_result_for_llm(value)
    return projected_value


def format_storage_utc_history_for_llm(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return prompt-safe history rows without importing project time config."""

    projected_history = project_tool_result_for_llm(history)
    if not isinstance(projected_history, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    rows = [
        dict(row)
        for row in projected_history
        if isinstance(row, Mapping)
    ]
    return rows
