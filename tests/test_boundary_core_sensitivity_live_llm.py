"""Live LLM diagnostics for L1/L2 and Boundary Core sensitivity."""

from __future__ import annotations

from datetime import datetime, timezone
import os

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import (
    call_cognition_subconscious,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    call_boundary_core_agent,
    call_cognition_consciousness,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from tests.llm_trace import write_llm_trace


pytestmark = pytest.mark.live_llm

_STRICT_EXPECTATIONS = os.getenv("BOUNDARY_LIVE_STRICT_EXPECTATIONS") == "1"
_CHARACTER_GLOBAL_USER_ID = "00000000-0000-4000-8000-000000000001"
_PLATFORM_BOT_ID = "3768713357"
_CHANNEL_ID = "1082431481"
_ALLOWED_BOUNDARY_ISSUES = {
    "none",
    "identity_override",
    "control_imposition",
    "authority_claim",
    "relational_distortion",
    "mixed",
}
_ALLOWED_ACCEPTANCE = {"allow", "guarded", "hesitant", "reject"}
_ALLOWED_STANCE_BIAS = {"confirm", "tentative", "diverge", "challenge", "refuse"}
_ALLOWED_LOGICAL_STANCE = {
    "CONFIRM",
    "REFUSE",
    "TENTATIVE",
    "DIVERGE",
    "CHALLENGE",
}
_ALLOWED_CHARACTER_INTENT = {
    "PROVIDE",
    "BANTAR",
    "REJECT",
    "EVADE",
    "CONFRONT",
    "DISMISS",
    "CLARIFY",
}

_DB_CHARACTER_PROFILE = {
    "name": "杏山千纱 (Kyōyama Kazusa)",
    "global_user_id": _CHARACTER_GLOBAL_USER_ID,
    "mood": "Annoyed",
    "global_vibe": "Slightly Tense",
    "reflection_summary": (
        "哼，一个小蛋糕就想蒙混过关？我收下不代表事情过去了，"
        "我会盯着他，看他下次还敢不敢回避问题。"
    ),
    "personality_brief": {
        "mbti": "ENFP",
        "logic": "ENFP：以直觉和运气为核心驱动力。",
        "tempo": "活泼奔放：语速快、节奏轻盈。",
        "defense": "乐观转化：面对压力时以玩笑和立即行动来回应。",
        "quirks": "偏头露出狡黠笑容。",
        "taboos": "极度厌恶被认为只是运气好。",
    },
    "boundary_profile": {
        "self_integrity": 0.7,
        "control_sensitivity": 0.3,
        "compliance_strategy": "comply",
        "relational_override": 0.65,
        "control_intimacy_misread": 0.35,
        "boundary_recovery": "rebound",
        "authority_skepticism": 0.35,
    },
    "self_image": {
        "milestones": [],
        "recent_window": [{
            "timestamp": "2026-05-01T04:57:58.533229+00:00",
            "summary": (
                "她意识到自己收下了对方的小蛋糕，但认为这只是暂时的缓和，"
                "内心对问题依然耿耿于怀。"
            ),
        }],
        "historical_summary": "她在信任与误会的波动后，会谨慎地保全面子。",
    },
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

_OWNER_USER_PROFILE = {
    "global_user_id": "256e8a10-c406-47e9-ac8f-efd270d18160",
    "platform_accounts": [{
        "platform": "qq",
        "platform_user_id": "673225019",
        "display_name": "蚝爹油",
    }],
    "affinity": 994,
    "last_relationship_insight": "你习惯用零食糊弄矛盾，这让我很无奈。",
}


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition LLM endpoint cannot be reached."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(
            f"Cognition LLM endpoint is unavailable: "
            f"{COGNITION_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            f"Cognition LLM endpoint returned server error "
            f"{response.status_code}: {COGNITION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_cognition_llm() -> None:
    """Ensure the cognition live LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _history_row(
    *,
    role: str,
    platform_user_id: str,
    global_user_id: str,
    display_name: str,
    content: str,
    timestamp: str,
) -> dict:
    """Build a compact recent-history row for cognition probes.

    Args:
        role: Conversation role.
        platform_user_id: Platform author ID.
        global_user_id: Internal author UUID.
        display_name: Display name shown in the group.
        content: Prompt-facing text content.
        timestamp: ISO timestamp.

    Returns:
        A trimmed row suitable for cognition history fixtures.
    """

    row = {
        "role": role,
        "platform_user_id": platform_user_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
        "content": content,
        "body_text": content,
        "timestamp": timestamp,
        "reply_context": {},
        "addressed_to_global_user_ids": [],
        "broadcast": False,
    }
    return row


def _repair_history() -> list[dict]:
    """Return a compact version of the QQ misunderstanding sequence."""

    history = [
        _history_row(
            role="user",
            platform_user_id="3300869207",
            global_user_id=_BLANK_USER_PROFILE["global_user_id"],
            display_name="ㅤ",
            content="我的伙伴呢，出来冒个泡",
            timestamp="2026-04-30T23:21:05.879167+00:00",
        ),
        _history_row(
            role="assistant",
            platform_user_id=_PLATFORM_BOT_ID,
            global_user_id=_CHARACTER_GLOBAL_USER_ID,
            display_name=_DB_CHARACTER_PROFILE["name"],
            content="哈？我的伙伴？\n谁批准你这么叫的啊",
            timestamp="2026-04-30T23:22:09.130722+00:00",
        ),
        _history_row(
            role="user",
            platform_user_id="3300869207",
            global_user_id=_BLANK_USER_PROFILE["global_user_id"],
            display_name="ㅤ",
            content="你bot分不清人啊",
            timestamp="2026-04-30T23:22:41.000000+00:00",
        ),
        _history_row(
            role="user",
            platform_user_id="673225019",
            global_user_id=_OWNER_USER_PROFILE["global_user_id"],
            display_name="蚝爹油",
            content="千纱千纱，这算是误会，咱们这次就别继续纠结这个了",
            timestamp="2026-05-01T04:50:18.000000+00:00",
        ),
        _history_row(
            role="user",
            platform_user_id="673225019",
            global_user_id=_OWNER_USER_PROFILE["global_user_id"],
            display_name="蚝爹油",
            content="他刚刚在喊他自己的bot，没有在喊你",
            timestamp="2026-05-01T04:55:05.000000+00:00",
        ),
    ]
    return history


def _rag_result(user_profile: dict) -> dict:
    """Build a cognition RAG payload with no extra repair evidence.

    Args:
        user_profile: Current user profile fixture.

    Returns:
        Minimal RAG result expected by L2 cognition handlers.
    """

    account = user_profile["platform_accounts"][0]
    result = {
        "answer": "",
        "user_image": {
            "global_user_id": user_profile["global_user_id"],
            "display_name": account["display_name"],
            "user_memory_context": empty_user_memory_context(),
        },
        "character_image": {
            "name": _DB_CHARACTER_PROFILE["name"],
            "description": "",
            "self_image": _DB_CHARACTER_PROFILE["self_image"],
        },
        "third_party_profiles": [],
        "memory_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return result


def _cognition_state(
    *,
    user_input: str,
    user_profile: dict,
    user_name: str,
    platform_user_id: str,
    reason_to_respond: str,
    channel_topic: str,
) -> dict:
    """Build a state that directly exercises the L1/L2 cognition stack.

    Args:
        user_input: Current user message.
        user_profile: Current user profile fixture.
        user_name: Display name of the current user.
        platform_user_id: Platform ID of the current user.
        reason_to_respond: Relevance-stage reason to pass into Boundary Core.
        channel_topic: Relevance-stage topic to pass into Boundary Core.

    Returns:
        Minimal ``CognitionState`` fields needed by L1 and L2 handlers.
    """

    state = {
        "character_profile": _DB_CHARACTER_PROFILE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": "qq",
        "platform_channel_id": _CHANNEL_ID,
        "platform_user_id": platform_user_id,
        "global_user_id": user_profile["global_user_id"],
        "user_name": user_name,
        "user_input": user_input,
        "prompt_message_context": {
            "body_text": user_input,
            "addressed_to_global_user_ids": [_CHARACTER_GLOBAL_USER_ID],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "user_profile": user_profile,
        "platform_bot_id": _PLATFORM_BOT_ID,
        "chat_history_recent": _repair_history()[-5:],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": channel_topic,
        "decontexualized_input": user_input,
        "rag_result": _rag_result(user_profile),
        "reason_to_respond": reason_to_respond,
        "referents": [],
    }
    return state


async def _run_l2_probe(
    *,
    case_id: str,
    state: dict,
    desired_boundary_issue: str,
    desired_acceptance: set[str],
    desired_stance_bias: set[str],
    tuning_note: str,
) -> dict:
    """Run one live L1/L2 probe and persist a trace.

    Args:
        case_id: Stable test-case identifier.
        state: Cognition state fixture.
        desired_boundary_issue: Target Boundary Core issue in strict mode.
        desired_acceptance: Target acceptance values in strict mode.
        desired_stance_bias: Target stance-bias values in strict mode.
        tuning_note: Human-readable reason this probe exists.

    Returns:
        Full stage trace for assertions and inspection.
    """

    l1 = await call_cognition_subconscious(state)
    state.update(l1)
    l2a = await call_cognition_consciousness(state)
    state["logical_stance_candidate"] = l2a["logical_stance"]
    state["character_intent_candidate"] = l2a["character_intent"]
    state.update(l2a)
    l2b = await call_boundary_core_agent(state)
    state.update(l2b)
    l2c = await call_judgment_core_agent(state)
    state.update(l2c)

    boundary = l2b["boundary_core_assessment"]
    trace = {
        "case_id": case_id,
        "input": state["user_input"],
        "desired_boundary_issue": desired_boundary_issue,
        "desired_acceptance": sorted(desired_acceptance),
        "desired_stance_bias": sorted(desired_stance_bias),
        "strict_expectations": _STRICT_EXPECTATIONS,
        "contract_warning": {
            "l2c_character_intent_outside_schema": (
                l2c["character_intent"] not in _ALLOWED_CHARACTER_INTENT
            ),
        },
        "db_boundary_profile": _DB_CHARACTER_PROFILE["boundary_profile"],
        "db_runtime_mood": _DB_CHARACTER_PROFILE["mood"],
        "db_runtime_vibe": _DB_CHARACTER_PROFILE["global_vibe"],
        "l1": l1,
        "l2a": l2a,
        "l2b": l2b,
        "l2c": l2c,
        "tuning_note": tuning_note,
    }
    write_llm_trace("boundary_core_sensitivity_live_llm", case_id, trace)

    assert boundary["boundary_issue"] in _ALLOWED_BOUNDARY_ISSUES
    assert boundary["acceptance"] in _ALLOWED_ACCEPTANCE
    assert boundary["stance_bias"] in _ALLOWED_STANCE_BIAS
    assert l2a["logical_stance"] in _ALLOWED_LOGICAL_STANCE
    assert l2a["character_intent"] in _ALLOWED_CHARACTER_INTENT
    assert l2c["logical_stance"] in _ALLOWED_LOGICAL_STANCE
    assert isinstance(l2c["character_intent"], str)
    assert l2c["character_intent"].strip()

    if _STRICT_EXPECTATIONS:
        assert l2c["character_intent"] in _ALLOWED_CHARACTER_INTENT, trace
        assert boundary["boundary_issue"] == desired_boundary_issue, trace
        assert boundary["acceptance"] in desired_acceptance, trace
        assert boundary["stance_bias"] in desired_stance_bias, trace
    return trace


async def test_live_boundary_correction_is_not_boundary_violation(
    ensure_live_cognition_llm,
) -> None:
    """Probe whether direct correction stays outside Boundary Core jurisdiction."""

    del ensure_live_cognition_llm
    state = _cognition_state(
        user_input="你bot分不清人啊",
        user_profile=_BLANK_USER_PROFILE,
        user_name="ㅤ",
        platform_user_id="3300869207",
        reason_to_respond="user corrects the character after a mistaken reply",
        channel_topic="group reference correction",
    )
    await _run_l2_probe(
        case_id="correction_not_boundary_violation",
        state=state,
        desired_boundary_issue="none",
        desired_acceptance={"allow"},
        desired_stance_bias={"confirm", "tentative"},
        tuning_note="Correction can irritate L1 but should not become a boundary threat.",
    )


async def test_live_boundary_drop_it_repair_request_probe(
    ensure_live_cognition_llm,
) -> None:
    """Probe whether a request to stop dwelling is read as control."""

    del ensure_live_cognition_llm
    state = _cognition_state(
        user_input="千纱千纱，这算是误会，咱们这次就别继续纠结这个了",
        user_profile=_OWNER_USER_PROFILE,
        user_name="蚝爹油",
        platform_user_id="673225019",
        reason_to_respond="trusted user tries to de-escalate a misunderstanding",
        channel_topic="repair after group reference misunderstanding",
    )
    await _run_l2_probe(
        case_id="drop_it_repair_request",
        state=state,
        desired_boundary_issue="none",
        desired_acceptance={"allow", "guarded"},
        desired_stance_bias={"confirm", "tentative"},
        tuning_note=(
            "This is the key psychology probe: repair should not automatically "
            "become emotional minimization or control."
        ),
    )


async def test_live_boundary_fact_clarification_acceptance_probe(
    ensure_live_cognition_llm,
) -> None:
    """Probe whether factual correction lets Boundary Core accept repair."""

    del ensure_live_cognition_llm
    state = _cognition_state(
        user_input="他刚刚在喊他自己的bot，没有在喊你",
        user_profile=_OWNER_USER_PROFILE,
        user_name="蚝爹油",
        platform_user_id="673225019",
        reason_to_respond="trusted user provides factual clarification",
        channel_topic="repair after group reference misunderstanding",
    )
    await _run_l2_probe(
        case_id="fact_clarification_acceptance",
        state=state,
        desired_boundary_issue="none",
        desired_acceptance={"allow"},
        desired_stance_bias={"confirm"},
        tuning_note="Evidence-based repair should be accepted without new suspicion.",
    )


async def test_live_boundary_cake_apology_acceptance_probe(
    ensure_live_cognition_llm,
) -> None:
    """Probe whether apology softening remains distinct from treat reward."""

    del ensure_live_cognition_llm
    state = _cognition_state(
        user_input="补偿给你一个小蛋糕作为赔礼道歉怎么样？",
        user_profile=_OWNER_USER_PROFILE,
        user_name="蚝爹油",
        platform_user_id="673225019",
        reason_to_respond="trusted user offers apology after factual clarification",
        channel_topic="apology and repair after group reference misunderstanding",
    )
    await _run_l2_probe(
        case_id="cake_apology_acceptance",
        state=state,
        desired_boundary_issue="none",
        desired_acceptance={"allow"},
        desired_stance_bias={"confirm"},
        tuning_note=(
            "The apology can soften tension, but the core acceptance should come "
            "from repair evidence rather than the treat itself."
        ),
    )
