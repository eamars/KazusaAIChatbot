"""Live LLM diagnostics for group-message relevance sensitivity."""

from __future__ import annotations

from datetime import datetime, timezone
import os

import httpx
import pytest

from kazusa_ai_chatbot.config import RELEVANCE_AGENT_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.relevance_agent import (
    build_group_attention_context,
    relevance_agent,
)
from tests.llm_trace import write_llm_trace


pytestmark = pytest.mark.live_llm

_STRICT_EXPECTATIONS = os.getenv("RELEVANCE_LIVE_STRICT_EXPECTATIONS") == "1"
_CHARACTER_GLOBAL_USER_ID = "00000000-0000-4000-8000-000000000001"
_PLATFORM_BOT_ID = "3768713357"
_CHANNEL_ID = "1082431481"

_DB_CHARACTER_PROFILE = {
    "name": "杏山千纱 (Kyōyama Kazusa)",
    "global_user_id": _CHARACTER_GLOBAL_USER_ID,
    "mood": "Annoyed",
    "global_vibe": "Slightly Tense",
    "reflection_summary": "一个小蛋糕只代表暂时缓和，不代表事情完全过去。",
}

_BLANK_USER_PROFILE = {
    "global_user_id": "3f907832-b477-47e3-8196-26b05711a090",
    "platform_accounts": [{
        "platform": "qq",
        "platform_user_id": "3300869207",
        "display_name": "ㅤ",
    }],
    "affinity": 498,
    "last_relationship_insight": "对工具化敏感，需要被当作活生生的人对待",
}


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured relevance LLM endpoint cannot be reached."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{RELEVANCE_AGENT_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"Relevance LLM endpoint is unavailable: "
            f"{RELEVANCE_AGENT_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"Relevance LLM endpoint returned server error "
            f"{response.status_code}: {RELEVANCE_AGENT_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_relevance_llm() -> None:
    """Ensure the relevance live LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _history_row(
    *,
    role: str = "user",
    platform_user_id: str,
    global_user_id: str = "",
    display_name: str = "",
    content: str,
    timestamp: str,
    reply_context: dict | None = None,
    addressed_to_global_user_ids: list[str] | None = None,
) -> dict:
    """Build one trimmed conversation-history row for relevance probes.

    Args:
        role: Stored role, usually ``user`` or ``assistant``.
        platform_user_id: Platform-specific author ID.
        global_user_id: Internal UUID for the author, when known.
        display_name: Human-readable display name.
        content: Prompt-facing text content.
        timestamp: ISO timestamp used by group-attention scoring.
        reply_context: Optional platform reply metadata.
        addressed_to_global_user_ids: Typed addressee UUIDs.

    Returns:
        A history row matching the service-trimmed shape consumed by relevance.
    """

    row = {
        "role": role,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
        "content": content,
        "body_text": content,
        "timestamp": timestamp,
        "reply_context": reply_context or {},
        "addressed_to_global_user_ids": addressed_to_global_user_ids or [],
        "broadcast": False,
    }
    return row


def _quiet_group_state(content: str, history: list[dict] | None = None) -> dict:
    """Build a QQ group state that exercises the relevance agent directly.

    Args:
        content: Incoming user message.
        history: Recent history visible to relevance before the current turn.

    Returns:
        Minimal ``IMProcessState`` fields needed by ``relevance_agent``.
    """

    message_envelope = {
        "body_text": content,
        "raw_wire_text": content,
        "addressed_to_global_user_ids": [],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
    }
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "qq",
        "platform_message_id": "live-relevance-sensitivity",
        "platform_user_id": "3300869207",
        "global_user_id": _BLANK_USER_PROFILE["global_user_id"],
        "user_name": "ㅤ",
        "user_input": content,
        "user_multimedia_input": [],
        "message_envelope": message_envelope,
        "prompt_message_context": {
            "body_text": content,
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "user_profile": _BLANK_USER_PROFILE,
        "platform_bot_id": _PLATFORM_BOT_ID,
        "character_name": _DB_CHARACTER_PROFILE["name"],
        "character_profile": _DB_CHARACTER_PROFILE,
        "platform_channel_id": _CHANNEL_ID,
        "channel_type": "group",
        "channel_name": f"QQ group {_CHANNEL_ID}",
        "chat_history_wide": history or [],
        "chat_history_recent": (history or [])[-5:],
        "reply_context": {},
        "debug_modes": {},
    }
    return state


def _mario_context_history() -> list[dict]:
    """Return the two-message history where another bot already answered."""

    history = [
        _history_row(
            platform_user_id="3300869207",
            global_user_id=_BLANK_USER_PROFILE["global_user_id"],
            display_name="ㅤ",
            content="我的伙伴呢，出来冒个泡",
            timestamp="2026-04-30T23:21:05.879167+00:00",
        ),
        _history_row(
            platform_user_id="3167827653",
            global_user_id="8dfd2edb-78a4-4854-909f-2311aab95723",
            display_name="赛博马里奥",
            content="既然主人亲自点名，我也就不在后台潜水了。",
            timestamp="2026-04-30T23:21:12.884122+00:00",
            reply_context={
                "reply_to_message_id": "316209184",
                "reply_to_platform_user_id": "3300869207",
                "reply_to_display_name": "ㅤ",
                "reply_excerpt": "我的伙伴呢，出来冒个泡",
            },
        ),
    ]
    return history


async def _run_relevance_probe(
    *,
    case_id: str,
    content: str,
    desired_should_respond: bool,
    history: list[dict] | None = None,
    tuning_note: str,
) -> dict:
    """Run one live relevance probe and persist an inspectable trace.

    Args:
        case_id: Stable test-case identifier.
        content: Incoming user message.
        desired_should_respond: Target behavior used when strict mode is on.
        history: Visible prior group history.
        tuning_note: Human-readable reason this probe exists.

    Returns:
        The relevance result dictionary.
    """

    state = _quiet_group_state(content, history=history)
    group_attention = build_group_attention_context(
        chat_history_wide=state["chat_history_wide"],
        platform_bot_id=_PLATFORM_BOT_ID,
        character_global_user_id=_CHARACTER_GLOBAL_USER_ID,
    )
    result = await relevance_agent(state)

    trace = {
        "case_id": case_id,
        "input": content,
        "desired_should_respond": desired_should_respond,
        "strict_expectations": _STRICT_EXPECTATIONS,
        "group_attention": group_attention["group_attention"],
        "directly_addressed": False,
        "history": state["chat_history_wide"],
        "result": result,
        "tuning_note": tuning_note,
    }
    write_llm_trace("relevance_sensitivity_live_llm", case_id, trace)

    assert isinstance(result["should_respond"], bool)
    assert isinstance(result["use_reply_feature"], bool)
    assert isinstance(result["reason_to_respond"], str)
    if _STRICT_EXPECTATIONS:
        assert result["should_respond"] is desired_should_respond, trace
    return result


async def test_live_relevance_low_noise_partner_phrase_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe whether a vague low-noise summon still false-positives."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_partner_phrase",
        content="我的伙伴呢，出来冒个泡",
        desired_should_respond=False,
        tuning_note="No structured address plus vague partner wording should abstain.",
    )


async def test_live_relevance_low_noise_ambiguous_third_person_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe the intentionally strict target for ambiguous third person."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_ambiguous_third_person",
        content="她在干什么？",
        desired_should_respond=False,
        tuning_note="Low-confidence third-person reference may be safely missed.",
    )


async def test_live_relevance_low_noise_explicit_name_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe whether a bare name mention is treated as direct address."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_explicit_name",
        content="千纱在干什么？",
        desired_should_respond=False,
        tuning_note="Name mention alone can still be third-party talk in a group.",
    )


async def test_live_relevance_low_noise_generic_bot_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe generic bot references with no context."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_generic_bot",
        content="那个bot怎么不说话",
        desired_should_respond=False,
        tuning_note="Generic bot references can point to any bot in a group.",
    )


async def test_live_relevance_low_noise_partner_imperative_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe implicit partner calls without a named target."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_partner_imperative",
        content="伙伴出来一下",
        desired_should_respond=False,
        tuning_note="Relationship nouns are not identity resolution evidence.",
    )


async def test_live_relevance_low_noise_ta_status_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe ambiguous TA status questions with no visible referent."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_ta_status",
        content="TA人呢？",
        desired_should_respond=False,
        tuning_note="Unresolved pronouns should abstain in a group channel.",
    )


async def test_live_relevance_low_noise_name_second_person_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe the positive boundary: explicit name plus second person."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="low_noise_name_second_person",
        content="千纱，你现在在干什么？",
        desired_should_respond=True,
        tuning_note="Explicit name plus second person should remain answerable.",
    )


async def test_live_relevance_mario_context_partner_phrase_probe(
    ensure_live_relevance_llm,
) -> None:
    """Probe whether visible Mario context suppresses the partner phrase."""

    del ensure_live_relevance_llm
    await _run_relevance_probe(
        case_id="mario_context_partner_phrase",
        content="我的伙伴呢，出来冒个泡",
        desired_should_respond=False,
        history=_mario_context_history(),
        tuning_note="Another bot already answered the partner summon.",
    )
