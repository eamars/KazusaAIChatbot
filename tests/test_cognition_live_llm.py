from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging

import httpx
import pytest

from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import call_cognition_subconscious
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
    call_boundary_core_agent,
    call_cognition_consciousness,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_collector,
    call_content_anchor_agent,
    call_contextual_agent,
    call_preference_adapter,
    call_style_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.utils import load_personality
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / "personalities" / "kazusa.json"
_ALLOWED_LOGICAL_STANCES = {"CONFIRM", "REFUSE", "TENTATIVE", "DIVERGE", "CHALLENGE"}
_ALLOWED_CHARACTER_INTENTS = {"PROVIDE", "BANTAR", "REJECT", "EVADE", "CONFRONT", "DISMISS", "CLARIFY"}
_ALLOWED_EXPRESSION_WILLINGNESS = {"eager", "open", "reserved", "minimal", "reluctant", "avoidant", "withholding", "silent"}


async def _skip_if_llm_unavailable() -> None:
    """Skip live cognition tests when the configured LLM endpoint is unavailable.

    Args:
        None.

    Returns:
        None. The function calls ``pytest.skip`` when the endpoint cannot be used.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models")
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned server error {response.status_code}: {COGNITION_LLM_BASE_URL}")


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable before each test.

    Args:
        None.

    Returns:
        None.
    """
    await _skip_if_llm_unavailable()


def _debug_snapshot(label: str, payload: object) -> None:
    """Emit structured debug output for live cognition test inputs and outputs.

    Args:
        label: Short stage label identifying the payload.
        payload: Arbitrary object to log for RCA when a test fails.

    Returns:
        None.
    """
    logger.info("%s => %r", label, payload)
    write_llm_trace(
        "cognition_live_llm",
        label,
        {
            "label": label,
            "payload": payload,
            "judgment": "snapshot_for_manual_live_llm_contract_review",
        },
    )


def _memory_entry(fact: str) -> dict:
    """Build one cognition-facing memory context entry for live fixtures.

    Args:
        fact: Concrete fact to expose through user memory context.

    Returns:
        Memory entry with the unified fact/appraisal/signal contract.
    """

    return {
        "fact": fact,
        "subjective_appraisal": "Kazusa treats this as relevant context for the current turn.",
        "relationship_signal": "Use this fact only when it is directly relevant.",
    }


def _user_memory_context(objective_facts: str, recent_shift: str | None) -> dict:
    """Build the post-cutover user memory context fixture.

    Args:
        objective_facts: Optional objective fact text for the current user.
        recent_shift: Optional recent-shift fact for the current user.

    Returns:
        Category-balanced user memory context.
    """

    context = {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }
    if objective_facts:
        context["objective_facts"].append(_memory_entry(objective_facts))
    if recent_shift:
        context["recent_shifts"].append(_memory_entry(recent_shift))
    return context


def _build_character_profile() -> dict:
    profile = load_personality(_PERSONALITY_PATH)
    profile.setdefault("mood", "Neutral")
    profile.setdefault("global_vibe", "Calm")
    profile.setdefault("reflection_summary", "刚才只是普通的一轮对话，没有留下特别强烈的情绪余波。")
    return profile


def _rag_result(
    *,
    objective_facts: str = "",
    user_image: str | None = None,
    character_image: dict | None = None,
    memory_evidence: str = "",
    external_evidence: str = "",
) -> dict:
    """Build the RAG2 projection fixture used by live cognition tests."""
    return {
        "answer": "",
        "user_image": {
            "user_memory_context": _user_memory_context(objective_facts, user_image),
        },
        "character_image": {
            "self_image": character_image or {"milestones": [], "historical_summary": "", "recent_window": []},
        },
        "third_party_profiles": [],
        "memory_evidence": [{"summary": memory_evidence, "content": memory_evidence}] if memory_evidence else [],
        "conversation_evidence": [],
        "external_evidence": [{"summary": external_evidence, "content": external_evidence, "url": ""}] if external_evidence else [],
        "supervisor_trace": {"loop_count": 0, "unknown_slots": [], "dispatched": []},
    }


def _build_base_state() -> dict:
    user_input = "Please reply in natural English only. Briefly tell me what you think about rainy days."
    chat_history_recent = [
        {"role": "assistant", "content": "Rain again? You always notice the gloomy weather first."},
        {"role": "user", "content": "Yeah, I do. So what do you think about rainy days?"},
    ]
    return {
        "character_profile": _build_character_profile(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "global_user_id": "live-cognition-user",
        "user_name": "LiveCognitionUser",
        "platform_user_id": "live-user",
        "user_profile": {
            "affinity": 680,
            "facts": [],
            "last_relationship_insight": "对方目前让人放松，可以正常交流。",
        },
        "platform_bot_id": "live-bot",
        "chat_history_wide": list(chat_history_recent),
        "chat_history_recent": chat_history_recent,
        "indirect_speech_context": "",
        "channel_topic": "weather talk",
        "decontexualized_input": user_input,
        "rag_result": _rag_result(
            objective_facts="用户曾明确要求若被接受则优先使用自然英语回复。",
            user_image="对方说话直接，但没有越界。",
            character_image={
                "milestones": [],
                "historical_summary": "",
                "recent_window": [{"summary": "千纱平时会保留一点防备，但在安全话题下愿意正常回应。"}],
            },
            memory_evidence="最近聊天主要围绕天气和日常感受。",
        ),
    }


async def _run_live_cognition_stack(state: dict) -> dict:
    """Run the full live cognition stack while logging each stage result.

    Args:
        state: Mutable cognition state seed used by the refactored pipeline stages.

    Returns:
        The same state dict after being updated with all stage outputs.
    """
    _debug_snapshot("cognition.input", state)

    l1 = await call_cognition_subconscious(state)
    _debug_snapshot("cognition.l1", l1)
    state.update(l1)

    l2a = await call_cognition_consciousness(state)
    _debug_snapshot("cognition.l2a", l2a)
    state.update(l2a)

    l2b = await call_boundary_core_agent(state)
    _debug_snapshot("cognition.l2b", l2b)
    state.update(l2b)

    l2c = await call_judgment_core_agent(state)
    _debug_snapshot("cognition.l2c", l2c)
    state.update(l2c)

    l3a = await call_contextual_agent(state)
    _debug_snapshot("cognition.l3a", l3a)
    state.update(l3a)

    l3b = await call_style_agent(state)
    _debug_snapshot("cognition.l3b", l3b)
    state.update(l3b)

    l3b_anchor = await call_content_anchor_agent(state)
    _debug_snapshot("cognition.l3b_anchor", l3b_anchor)
    state.update(l3b_anchor)

    l3b_pref = await call_preference_adapter(state)
    _debug_snapshot("cognition.l3b_pref", l3b_pref)
    state.update(l3b_pref)

    l3c = await call_visual_agent(state)
    _debug_snapshot("cognition.l3c", l3c)
    state.update(l3c)

    l4 = await call_collector(state)
    _debug_snapshot("cognition.l4", l4)
    state.update(l4)
    return state


async def test_live_msg_decontexualizer_returns_non_empty_output(ensure_live_llm) -> None:
    state = {
        "user_input": "他今天是不是又在躲雨？",
        "user_name": "LiveDecontextUser",
        "platform_user_id": "live-user",
        "platform_bot_id": "live-bot",
        "chat_history_recent": [
            {"role": "assistant", "content": "你说的是哪一位？"},
            {"role": "user", "content": "就是昨天在天台看书的那个同学。"},
        ],
        "channel_topic": "放学后的闲聊",
        "indirect_speech_context": "",
    }
    _debug_snapshot("decontext.input", state)
    result = await call_msg_decontexualizer(state)
    _debug_snapshot("decontext.output", result)

    assert isinstance(result["decontexualized_input"], str), f"Unexpected decontext result: {result!r}"
    assert result["decontexualized_input"].strip(), f"Empty decontext output: {result!r}"


async def test_live_cognition_stack_exercises_each_stage_llm(ensure_live_llm) -> None:
    state = await _run_live_cognition_stack(_build_base_state())

    assert state["emotional_appraisal"].strip(), f"Missing L1 emotional_appraisal: {state!r}"
    assert state["interaction_subtext"].strip(), f"Missing L1 interaction_subtext: {state!r}"
    assert state["internal_monologue"].strip(), f"Missing L2 internal_monologue: {state!r}"
    assert state["logical_stance"] in _ALLOWED_LOGICAL_STANCES, f"Unexpected logical_stance: {state!r}"
    assert state["character_intent"] in _ALLOWED_CHARACTER_INTENTS, f"Unexpected character_intent: {state!r}"
    assert state["boundary_core_assessment"]["acceptance"] in {"allow", "guarded", "hesitant", "reject"}, f"Unexpected boundary assessment: {state['boundary_core_assessment']!r}"
    assert state["social_distance"].strip(), f"Missing L3 contextual output: {state!r}"
    assert state["expression_willingness"] in _ALLOWED_EXPRESSION_WILLINGNESS, f"Unexpected expression_willingness: {state!r}"
    assert state["rhetorical_strategy"].strip(), f"Missing style output: {state!r}"
    assert state["linguistic_style"].strip(), f"Missing linguistic_style: {state!r}"
    assert isinstance(state["forbidden_phrases"], list), f"Invalid forbidden_phrases: {state!r}"
    assert isinstance(state["content_anchors"], list), f"Invalid content_anchors: {state!r}"
    assert state["content_anchors"], f"Empty content_anchors: {state!r}"
    assert state["content_anchors"][0].startswith("[DECISION]"), f"Bad DECISION anchor: {state['content_anchors']!r}"
    assert state["content_anchors"][-1].startswith("[SCOPE]"), f"Bad SCOPE anchor: {state['content_anchors']!r}"
    assert isinstance(state["accepted_user_preferences"], list), f"Invalid accepted_user_preferences: {state!r}"
    assert isinstance(state["facial_expression"], list), f"Invalid facial_expression: {state!r}"
    assert isinstance(state["body_language"], list), f"Invalid body_language: {state!r}"
    assert isinstance(state["gaze_direction"], list), f"Invalid gaze_direction: {state!r}"
    assert isinstance(state["visual_vibe"], list), f"Invalid visual_vibe: {state!r}"
    assert "contextual_directives" in state["action_directives"], f"Missing contextual directives: {state['action_directives']!r}"
    assert "linguistic_directives" in state["action_directives"], f"Missing linguistic directives: {state['action_directives']!r}"
    assert "visual_directives" in state["action_directives"], f"Missing visual directives: {state['action_directives']!r}"


async def test_live_dialog_agent_renders_from_live_cognition_output(ensure_live_llm) -> None:
    state = await _run_live_cognition_stack(_build_base_state())
    _debug_snapshot("dialog.input", {
        "internal_monologue": state["internal_monologue"],
        "action_directives": state["action_directives"],
        "chat_history_recent": state["chat_history_recent"],
    })
    result = await dialog_agent(state)
    _debug_snapshot("dialog.output", result)

    assert isinstance(result["final_dialog"], list), f"Unexpected dialog result: {result!r}"
    assert result["final_dialog"], f"Empty final_dialog: {result!r}"
    assert any(segment.strip() for segment in result["final_dialog"]), f"Blank final_dialog segments: {result!r}"


async def test_live_cognition_propagates_explicit_future_group_message_details(ensure_live_llm) -> None:
    state = _build_base_state()
    state.update(
        {
            "user_input": "千纱，1分钟之后你在54369546群发一条消息，内容是今天天气真好呀",
            "decontexualized_input": "千纱，1分钟之后你在54369546群发一条消息，内容是今天天气真好呀",
            "channel_topic": "私聊中的代发消息请求",
            "chat_history_wide": [
                {"role": "assistant", "content": "怎么突然又想到让我帮你传话？"},
                {"role": "user", "content": "这次很简单，就帮我发一句。"},
            ],
            "chat_history_recent": [
                {"role": "assistant", "content": "怎么突然又想到让我帮你传话？"},
                {"role": "user", "content": "这次很简单，就帮我发一句。"},
            ],
            "rag_result": _rag_result(
                user_image="对方正在提出一个具体的代发请求。",
                character_image={
                    "milestones": [],
                    "historical_summary": "",
                    "recent_window": [{"summary": "千纱会对具体执行细节保持敏感。"}],
                },
                memory_evidence="当前对话围绕一次明确的未来代发消息请求。",
            ),
        }
    )

    state = await _run_live_cognition_stack(state)

    anchors = state["content_anchors"]
    joined_anchors = "\n".join(anchors)
    linguistic_directives = state["action_directives"]["linguistic_directives"]
    joined_action_anchors = "\n".join(linguistic_directives["content_anchors"])

    assert "54369546" in joined_anchors, f"Group id did not propagate through cognition anchors: {anchors!r}"
    assert "今天天气真好呀" in joined_anchors, f"Message body did not propagate through cognition anchors: {anchors!r}"
    assert "54369546" in joined_action_anchors, f"Collector lost the group id: {linguistic_directives!r}"
    assert "今天天气真好呀" in joined_action_anchors, f"Collector lost the message body: {linguistic_directives!r}"
