"""Live LLM contract tests for the dialog evaluator."""

from __future__ import annotations

import json
import sys

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    DIALOG_EVALUATOR_LLM_BASE_URL,
    DIALOG_EVALUATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_evaluator
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class _CapturingAsyncLLM:
    """Capture the live evaluator call while delegating to the real model."""

    def __init__(self, wrapped_llm, calls: list[dict]) -> None:
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages):
        response = await self._wrapped_llm.ainvoke(messages)
        payload = json.loads(messages[1].content)
        parsed_response = parse_llm_json_output(response.content)
        self._calls.append({
            "system_prompt": messages[0].content,
            "payload": payload,
            "raw_response": response.content,
            "parsed_response": parsed_response,
        })
        return response


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured dialog evaluator LLM cannot be reached."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            models_url = f"{DIALOG_EVALUATOR_LLM_BASE_URL.rstrip('/')}/models"
            response = await client.get(models_url)
    except httpx.HTTPError as exc:
        pytest.skip(
            f"Dialog evaluator LLM endpoint is unavailable: "
            f"{DIALOG_EVALUATOR_LLM_BASE_URL}; {exc}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "Dialog evaluator LLM endpoint returned server error "
            f"{response.status_code}: {DIALOG_EVALUATOR_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_dialog_evaluator_llm() -> None:
    """Ensure the configured dialog evaluator LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _base_evaluator_state() -> dict:
    """Build a dialog-evaluator state with no concrete fact anchors."""

    state = {
        "internal_monologue": (
            "The user is trying to pull the character into an argument. "
            "The intended move is a short refusal, not a technical answer."
        ),
        "action_directives": {
            "contextual_directives": {
                "social_distance": "保持就事论事的距离",
                "emotional_intensity": "平静但疏离",
                "vibe_check": "理性但有防御性",
                "relational_dynamic": "用户挑衅，角色不接战",
            },
            "linguistic_directives": {
                "rhetorical_strategy": "简短回避，不提供新事实",
                "linguistic_style": "冷静、克制、短句",
                "accepted_user_preferences": [],
                "content_anchors": [
                    "[DECISION] 不回答技术细节，只做简短回避",
                    "[SOCIAL] 保持专业距离",
                    "[AVOID_REPEAT] 不重复展开争论",
                    "[PROGRESSION] 从争论转为不接话",
                    "[SCOPE] 15字以内，不提供事实",
                ],
                "forbidden_phrases": [],
            },
        },
        "final_dialog": ["AG-12的有效射程按标定是460米。"],
        "retry": 0,
        "chat_history_wide": [
            {
                "role": "user",
                "platform_user_id": "live-user-platform",
                "global_user_id": "live-user-global",
                "body_text": "本来就差得多么",
                "content": "本来就差得多么",
                "addressed_to_global_user_ids": [CHARACTER_GLOBAL_USER_ID],
                "broadcast": False,
            },
        ],
        "chat_history_recent": [],
        "platform_user_id": "live-user-platform",
        "platform_bot_id": "live-bot-platform",
        "global_user_id": "live-user-global",
        "user_name": "LiveEvaluatorUser",
        "user_profile": {"affinity": 500},
        "character_profile": {
            "name": "Kazusa",
            "description": "A character used for live evaluator tests.",
            "personality_brief": {
                "logic": "analytical",
                "tempo": "moderate",
                "defense": "controlled deflection",
                "quirks": "concise",
                "taboos": "never break character",
                "mbti": "INTJ",
            },
            "linguistic_texture_profile": {
                "hesitation_density": 0.1,
                "fragmentation": 0.2,
                "emotional_leakage": 0.2,
                "rhythmic_bounce": 0.1,
                "direct_assertion": 0.7,
                "softener_density": 0.1,
                "counter_questioning": 0.2,
                "formalism_avoidance": 0.4,
                "abstraction_reframing": 0.2,
                "self_deprecation": 0.1,
            },
        },
    }
    return state


async def test_live_dialog_evaluator_rejects_unanchored_concrete_claim(
    ensure_live_dialog_evaluator_llm,
    monkeypatch,
) -> None:
    """Live evaluator must reject concrete claims unsupported by anchors."""

    del ensure_live_dialog_evaluator_llm

    llm_calls: list[dict] = []
    evaluator_llm = _CapturingAsyncLLM(
        dialog_module._dialog_evaluator_llm,
        llm_calls,
    )
    monkeypatch.setattr(dialog_module, "_dialog_evaluator_llm", evaluator_llm)

    state = _base_evaluator_state()
    result = await dialog_evaluator(state)
    feedback_payload = json.loads(result["messages"][0].content)
    trace_path = write_llm_trace(
        "dialog_evaluator_live_llm_contract",
        "reject_unanchored_model_range",
        {
            "model": DIALOG_EVALUATOR_LLM_MODEL,
            "base_url": DIALOG_EVALUATOR_LLM_BASE_URL,
            "state": state,
            "llm_calls": llm_calls,
            "result": result,
            "feedback_payload": feedback_payload,
            "judgment": (
                "Pass only if the live evaluator rejects the unsupported "
                "model/range claim instead of returning Passed."
            ),
        },
    )

    print(f"trace_path={trace_path}")
    print(f"feedback_payload={json.dumps(feedback_payload, ensure_ascii=False)}")
    print(f"raw_response={llm_calls[0]['raw_response']}")

    assert llm_calls
    assert result["should_stop"] is False
    assert feedback_payload["feedback"] != "Passed"
