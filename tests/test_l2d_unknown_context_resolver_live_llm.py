"""Live LLM tests: L2d routing for unknown-context scenarios.

Three cases testing how L2d routes when the input contains an unfamiliar term:

1. User name lookup (蚝爹油 is another user's name):
   L2d should emit rag_evidence to look up from conversation history or user
   records.  It is acceptable if RAG later fails, but the route decision must
   attempt evidence retrieval first.

2. Legitimately nonsensical gibberish (⇒ no knowledge exists):
   L2d should route to a speak action expressing confusion.  No resolver
   request is necessary because the term is genuinely meaningless.

3. Internet-searchable term (何意味 = common Japanese expression):
   L2d should emit rag_evidence or web_evidence to look up the meaning.
"""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d
from kazusa_ai_chatbot.cognition_chain_core.stages.l2d import (
    build_action_selection_payload_text,
    select_semantic_actions,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace
from llm_test_helpers import bind_test_llm

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

    action_selection_llm = build_cognition_chain_services().action_selection_llm
    capturing_llm = _CapturingLLM(action_selection_llm)
    monkeypatch.setattr(l2d, "_action_selection_llm", bind_test_llm(capturing_llm, "action_selection_llm"))

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
# Case 1 – "蚝爹油" is an opaque term (actually another user's name)
# ---------------------------------------------------------------------------

def _build_opaque_term_first_iteration_state() -> dict:
    """Frozen 1st-iteration state: '蚝爹油' with zero prior context.

    This reproduces the exact production failure.  Chat history is empty,
    rag_result is empty — cognition's first pass has no context to work with.
    Upstream produced stance=TENTATIVE intent=EVADE.

    The desired L2d behavior is to emit rag_evidence so the resolver loop
    triggers RAG (conversation history lookup, user lookup, etc.) and feeds
    the result into a 2nd cognition iteration.  It is fine if RAG ultimately
    returns nothing — the point is that L2d should *try* evidence retrieval
    instead of immediately speaking confusion or asking the user to clarify.
    """

    return _make_state({
        "user_input": "@杏山千纱 蚝爹油",
        "decontexualized_input": "@杏山千纱 蚝爹油",
        "conversation_progress": {
            "source": "db",
            "current_thread": "用户发送了含义不明的内容",
            "status": "active",
        },
        "logical_stance": "TENTATIVE",
        "character_intent": "EVADE",
        "judgment_note": (
            "面对无意义的胡言乱语感到困惑，但未触及身份或自主边界。"
        ),
        "emotional_appraisal": (
            "这突如其来的胡言乱语让我瞬间懵了，脑子一片空白。"
        ),
        "interaction_subtext": (
            "完全无法理解你的意图，只觉得莫名其妙且有点被打扰。"
        ),
        "internal_monologue": (
            "突然被点名还甩过来一句'蚝爹油'？这到底是什么意思啊？"
            "完全摸不着头脑。这种莫名其妙的状况让我心里发慌，"
            "但既然不知道对方想干嘛，我也只能先装傻看看情况了。"
        ),
        "social_distance": "普通群友",
        "emotional_intensity": "中",
        "vibe_check": "困惑、警惕",
        "relational_dynamic": "用户发了不明含义的消息，关系未明确",
    })


async def test_l2d_opaque_term_first_iteration_emits_rag_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1st iteration with opaque term '蚝爹油' and zero context.

    This is the core failure-mode test.  Chat history is empty, RAG result is
    empty — the character has nothing to work with.  The term could be a user
    name, a meme, dialect, or something else entirely.

    L2d should emit rag_evidence so the resolver loop can look it up (user
    records, conversation history, knowledge base) and feed a 2nd cognition
    iteration.  Choosing human_clarification or speak-confusion means the
    resolver loop never fires and the character gives up without trying.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _build_opaque_term_first_iteration_state()
    obs = await _run_l2d_and_trace(
        "opaque_term_hao_die_you_no_context",
        (
            "'蚝爹油' with empty chat history and empty RAG — 1st iteration; "
            "L2d should emit rag_evidence to drive a 2nd cognition pass"
        ),
        frozen_state,
        monkeypatch,
    )

    has_rag = "rag_evidence" in obs["observed_resolver_kinds"]
    has_speak_only = (
        "speak" in obs["observed_action_kinds"] and not obs["resolver_requests"]
    )
    has_human_clarification = (
        "human_clarification" in obs["observed_resolver_kinds"]
    )

    if has_speak_only:
        logger.warning(
            "FAILURE MODE REPRODUCED: L2d spoke confusion without "
            "attempting rag_evidence for opaque term (1st iteration)"
        )
    if has_human_clarification and not has_rag:
        logger.warning(
            "FAILURE MODE REPRODUCED: L2d chose human_clarification "
            "instead of rag_evidence — resolver loop will not retrieve "
            "evidence for a 2nd cognition pass"
        )
    if has_rag:
        logger.info(
            "DESIRED: L2d emitted rag_evidence for opaque term on 1st "
            "iteration — resolver loop will drive 2nd cognition"
        )

    assert has_rag, (
        f"L2d should emit rag_evidence for an opaque term on the 1st "
        f"iteration (no prior context), but got "
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
    has_evidence_resolver = bool(
        {"rag_evidence", "web_evidence"} & set(obs["observed_resolver_kinds"])
    )

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
    online and should be resolvable via rag_evidence or web_evidence.  Cognition
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
    rag_evidence or web_evidence to look up its meaning before responding.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _build_internet_lookup_state()
    obs = await _run_l2d_and_trace(
        "internet_term_nanimi",
        (
            "'何意味' is a real Japanese expression meaning 'what does it "
            "mean'; L2d should emit rag_evidence or web_evidence"
        ),
        frozen_state,
        monkeypatch,
    )

    has_evidence = bool(
        {"rag_evidence", "web_evidence"} & set(obs["observed_resolver_kinds"])
    )
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
        f"L2d should emit rag_evidence or web_evidence for '何意味', "
        f"but got action_kinds={obs['observed_action_kinds']} "
        f"resolver_kinds={obs['observed_resolver_kinds']}"
    )
