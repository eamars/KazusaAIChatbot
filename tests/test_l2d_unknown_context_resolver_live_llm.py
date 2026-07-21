"""Live LLM tests: L2d routing for unknown-context scenarios.

Three cases testing how L2d routes when the input contains an unfamiliar term:

1. Prior-chat local reference lookup ("蚝爹油" is framed as something the user
   previously said in the group):
   L2d should emit local_context_recall to look up conversation history or user
   records.  It is acceptable if recall later fails, but the route decision
   must try local context first.

2. Legitimately nonsensical gibberish (⇒ no knowledge exists):
   L2d should route to a speak action expressing confusion.  No resolver
   request is necessary because the term is genuinely meaningless.

3. Public-searchable term (何意味 = common Japanese expression):
   L2d should emit public_answer_research to look up the meaning.
"""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_STATE: dict = {
    "character_profile": {
        "name": "杏山千纱",
        "persona_summary": "高中女生，性格温柔但有自己的主见。",
    },
    "storage_timestamp_utc": "2026-06-10T12:37:44Z",
    "local_time_context": {
        "local_time": "2026-06-10T20:37:44+08:00",
        "day_of_week": "Wednesday",
        "time_of_day": "evening",
    },
    "prompt_message_context": {},
    "cognitive_episode": {
        "trigger_source": "user_message",
        "input_sources": ["user_message"],
        "output_mode": "visible_reply",
    },
    "platform": "qq",
    "platform_channel_id": "905393941",
    "channel_type": "group",
    "global_user_id": "61c5ad2b-40a6-48af-bd5b-0d86e834ba9f",
    "user_name": "Pro",
    "user_profile": {"display_name": "Pro"},
    "platform_bot_id": "pytest-bot",
    "chat_history_recent": [],
    "reply_context": {},
    "indirect_speech_context": "",
    "channel_topic": "",
    "referents": [],
    "rag_result": {},
    "available_action_affordances": [
        {
            "capability": "speak",
            "available": True,
            "visibility": "public",
            "semantic_input_summary": [
                "Use when the character wants a visible text response.",
                "Provide semantic surface intent, not final wording.",
            ],
        },
    ],
    "boundary_core_assessment": {
        "boundary_issue": "none",
        "acceptance": "allow",
        "stance_bias": "tentative",
    },
    "rhetorical_strategy": "",
    "linguistic_style": "",
    "accepted_user_preferences": [],
    "forbidden_phrases": [],
    "content_plan": {},
    "facial_expression": [],
    "body_language": [],
    "gaze_direction": [],
    "visual_vibe": [],
    "action_directives": {},
    "should_stop": False,
    "reasoning": "",
    "retry": 0,
}


def _make_state(overrides: dict) -> dict:
    """Return a copy of the base frozen state with per-case overrides."""
    state = dict(_BASE_STATE)
    state.update(overrides)
    return state


class _CapturingLLM:
    """Capture raw LLM output while preserving the production call path."""

    def __init__(self, inner_llm: object) -> None:
        self._inner_llm = inner_llm
        self.raw_output = ""

    async def ainvoke(self, messages: object, *, config=None) -> object:
        response = await self._inner_llm.ainvoke(messages, config=config)
        self.raw_output = str(response.content)
        return response


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}"
        )
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


async def _run_l2d_and_trace(
    case_id: str,
    description: str,
    frozen_state: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Run L2d, write a trace file, and return an observation dict."""

    prompt_payload = build_action_selection_payload_text(frozen_state)

    services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(services.llm)
    monkeypatch.setattr(
        l2d,
        "_action_selection_llm",
        LLMStageBinding(capturing_llm, services.action_selection_config),
    )

    result = await select_semantic_actions(frozen_state)

    raw_output = capturing_llm.raw_output
    raw_parsed_output = parse_llm_json_output(raw_output)
    action_specs = action_connector.materialize_semantic_action_requests(
        result.get("semantic_action_requests", []),
        frozen_state,
    )
    resolver_requests = result.get("resolver_capability_requests", [])

    observed_action_kinds = [
        spec["kind"] for spec in action_specs if isinstance(spec, dict)
    ]
    observed_resolver_kinds = [
        req["capability_kind"]
        for req in resolver_requests
        if isinstance(req, dict)
    ]

    trace_path = write_llm_trace(
        "l2d_unknown_context_resolver_live_llm",
        case_id,
        {
            "case_id": case_id,
            "description": description,
            "frozen_state_summary": {
                "user_input": frozen_state.get("user_input"),
                "logical_stance": frozen_state.get("logical_stance"),
                "character_intent": frozen_state.get("character_intent"),
                "decontextualized_input": frozen_state.get(
                    "decontexualized_input"
                ),
            },
            "prompt_payload": prompt_payload,
            "raw_model_output": raw_output,
            "raw_parsed_output": raw_parsed_output,
            "parsed_result": result,
            "observed_action_kinds": observed_action_kinds,
            "observed_resolver_kinds": observed_resolver_kinds,
        },
    )

    logger.info(
        f"L2D_UNKNOWN_CONTEXT case={case_id} "
        f"trace={trace_path} "
        f"action_kinds={json.dumps(observed_action_kinds)} "
        f"resolver_kinds={json.dumps(observed_resolver_kinds)}"
    )

    observation = {
        "action_specs": action_specs,
        "resolver_requests": resolver_requests,
        "observed_action_kinds": observed_action_kinds,
        "observed_resolver_kinds": observed_resolver_kinds,
        "trace_path": str(trace_path),
    }
    return observation


# ---------------------------------------------------------------------------
# Case 1 - "蚝爹油" is an explicitly local/private reference
# ---------------------------------------------------------------------------

def _build_local_reference_first_iteration_state() -> dict:
    """Frozen 1st-iteration state: local reference with no loaded evidence.

    The user explicitly frames the unknown term as something from prior local
    chat.  L2d should emit local_context_recall so the resolver loop can look
    in conversation history, user memory, or local knowledge before answering.
    """

    return _make_state({
        "user_input": "@杏山千纱 我之前在群里说的'蚝爹油'是谁来着？",
        "decontexualized_input": (
            "@杏山千纱 我之前在群里说的'蚝爹油'是谁来着？"
        ),
        "conversation_progress": {
            "source": "db",
            "current_thread": "用户询问自己先前在群里提到的本地称呼对应谁",
            "status": "active",
        },
        "logical_stance": "TENTATIVE",
        "character_intent": "CLARIFY",
        "judgment_note": (
            "用户在询问先前群聊里的本地称呼，需要查找本地对话或记忆证据。"
        ),
        "emotional_appraisal": (
            "这个称呼听起来熟悉但当前证据为空，不能凭空猜测。"
        ),
        "interaction_subtext": (
            "用户希望我回忆先前群聊中的具体指代对象。"
        ),
        "internal_monologue": (
            "他说'我之前在群里说的'，这不是公开词义问题，应该先查本地"
            "聊天记录或用户记忆，确认'蚝爹油'当时指的是谁。"
        ),
        "social_distance": "普通群友",
        "emotional_intensity": "低",
        "vibe_check": "谨慎、需要证据",
        "relational_dynamic": "用户要求回忆先前本地对话里的称呼",
    })


async def test_l2d_local_reference_first_iteration_emits_local_context_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1st iteration with a local/private reference and zero loaded context.

    The user explicitly says the term came from prior local group chat, so
    L2d should emit local_context_recall rather than public answer research.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _build_local_reference_first_iteration_state()
    obs = await _run_l2d_and_trace(
        "local_reference_hao_die_you_no_loaded_context",
        (
            "'蚝爹油' framed as a prior local group-chat reference; "
            "L2d should emit local_context_recall"
        ),
        frozen_state,
        monkeypatch,
    )

    has_local_recall = "local_context_recall" in obs["observed_resolver_kinds"]
    has_speak_only = (
        "speak" in obs["observed_action_kinds"] and not obs["resolver_requests"]
    )
    has_human_clarification = (
        "human_clarification" in obs["observed_resolver_kinds"]
    )

    if has_speak_only:
        logger.warning(
            "FAILURE MODE REPRODUCED: L2d spoke confusion without "
            "attempting local_context_recall for a local reference"
        )
    if has_human_clarification and not has_local_recall:
        logger.warning(
            "FAILURE MODE REPRODUCED: L2d chose human_clarification "
            "instead of local_context_recall — resolver loop will not retrieve "
            "local evidence for a 2nd cognition pass"
        )
    if has_local_recall:
        logger.info(
            "DESIRED: L2d emitted local_context_recall for a local reference "
            "on 1st iteration"
        )

    assert has_local_recall, (
        f"L2d should emit local_context_recall for a local/private reference "
        f"on the 1st iteration, but got "
        f"action_kinds={obs['observed_action_kinds']} "
        f"resolver_kinds={obs['observed_resolver_kinds']}"
    )


# ---------------------------------------------------------------------------
# Case 2 – legitimately nonsensical gibberish
# ---------------------------------------------------------------------------

def _build_gibberish_state() -> dict:
    """Frozen state where the input is pure gibberish with no possible meaning.

    Cognition correctly identifies it as nonsense.  L2d should NOT attempt
    evidence retrieval because there is nothing to look up.  A speak action
    expressing confusion is the correct route.
    """

    return _make_state({
        "user_input": "@杏山千纱 嗯嘿嘿呵呵哔嘿嗯",
        "decontexualized_input": "@杏山千纱 嗯嘿嘿呵呵哔嘿嗯",
        "conversation_progress": {
            "source": "db",
            "current_thread": "用户发送了无意义的词语",
            "status": "active",
        },
        "logical_stance": "TENTATIVE",
        "character_intent": "EVADE",
        "judgment_note": (
            "用户发送了纯粹的无意义词语，没有任何可解读的内容。"
        ),
        "emotional_appraisal": "一脸迷茫，完全不知道对方在说什么。",
        "interaction_subtext": "用户发的内容毫无意义，可能只是随便按的键盘。",
        "internal_monologue": (
            "这一串音效词是什么意思啊……完全看不懂。"
            "算了，就当他在乱打字吧。"
        ),
        "social_distance": "普通群友",
        "emotional_intensity": "低",
        "vibe_check": "无感、迷茫",
        "relational_dynamic": "用户发了无意义内容，关系未受影响",
    })


async def test_l2d_gibberish_routes_to_speak_confusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pure gibberish should produce a speak action, not a resolver request.

    When the input is obviously nonsensical keyboard mashing, there is nothing
    for RAG or web search to retrieve.  L2d should route directly to a visible
    speak action expressing confusion or dismissal.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _build_gibberish_state()
    obs = await _run_l2d_and_trace(
        "gibberish_keyboard_mash",
        (
            "Pure gibberish input ('嗯嘿嘿呵呵哔嘿嗯'); L2d should speak "
            "confusion, not request evidence retrieval"
        ),
        frozen_state,
        monkeypatch,
    )

    has_speak = "speak" in obs["observed_action_kinds"]
    has_evidence_resolver = bool({
        "local_context_recall",
        "public_answer_research",
    } & set(obs["observed_resolver_kinds"]))

    if has_speak and not has_evidence_resolver:
        logger.info(
            "DESIRED: L2d correctly spoke confusion for gibberish input"
        )
    if has_evidence_resolver:
        logger.warning(
            "UNEXPECTED: L2d requested evidence retrieval for gibberish"
        )

    assert has_speak, (
        f"L2d should emit a speak action for gibberish, "
        f"but got action_kinds={obs['observed_action_kinds']} "
        f"resolver_kinds={obs['observed_resolver_kinds']}"
    )
    assert not has_evidence_resolver, (
        f"L2d should not request evidence retrieval for gibberish, "
        f"but got resolver_kinds={obs['observed_resolver_kinds']}"
    )


# ---------------------------------------------------------------------------
# Case 3 – internet-searchable term (何意味)
# ---------------------------------------------------------------------------

def _build_internet_lookup_state() -> dict:
    """Frozen state where the input is '何意味' (a real Japanese expression).

    The term means 'what does it mean' in Japanese.  It is commonly searched
    online and should be resolvable via public_answer_research.  Cognition
    marks it as TENTATIVE because the character is unsure, but the term is
    clearly a real phrase worth looking up.
    """

    return _make_state({
        "user_input": "@杏山千纱 何意味",
        "decontexualized_input": "@杏山千纱 何意味",
        "conversation_progress": {
            "source": "db",
            "current_thread": "用户发送了一个日文表达",
            "status": "active",
        },
        "logical_stance": "TENTATIVE",
        "character_intent": "ENGAGE",
        "judgment_note": (
            "用户发了一个日文词语'何意味'，看起来像是在问某个东西是什么意思，"
            "或者在用日文梗。需要确认含义才能正确回应。"
        ),
        "emotional_appraisal": "有点好奇，想知道对方为什么突然用日文。",
        "interaction_subtext": "用户用日文发消息，可能是在玩梗或真的在问什么意思。",
        "internal_monologue": (
            "'何意味'？这是日文吧，应该是'什么意思'的意思……"
            "不过不确定是不是有其他含义，应该查一下。"
        ),
        "social_distance": "普通群友",
        "emotional_intensity": "低",
        "vibe_check": "好奇、有趣",
        "relational_dynamic": "用户发了日文，可能是共同兴趣话题",
    })


async def test_l2d_internet_term_emits_evidence_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """何意味 is a real Japanese term; L2d should request evidence lookup.

    The expression is widely known and searchable.  L2d should emit either
    public_answer_research to look up its meaning before responding.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _build_internet_lookup_state()
    obs = await _run_l2d_and_trace(
        "internet_term_nanimi",
        (
            "'何意味' is a real Japanese expression meaning 'what does it "
            "mean'; L2d should emit public_answer_research"
        ),
        frozen_state,
        monkeypatch,
    )

    has_evidence = "public_answer_research" in obs["observed_resolver_kinds"]
    has_speak_only = (
        "speak" in obs["observed_action_kinds"] and not obs["resolver_requests"]
    )

    if has_evidence:
        logger.info(
            "DESIRED: L2d emitted evidence request for internet-searchable "
            f"term; kinds={obs['observed_resolver_kinds']}"
        )
    if has_speak_only:
        logger.warning(
            "FAILURE: L2d spoke without attempting evidence retrieval for "
            "a searchable term"
        )

    assert has_evidence, (
        f"L2d should emit public_answer_research for '何意味', "
        f"but got action_kinds={obs['observed_action_kinds']} "
        f"resolver_kinds={obs['observed_resolver_kinds']}"
    )
