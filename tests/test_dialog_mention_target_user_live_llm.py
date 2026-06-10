"""Live LLM contract tests for dialog-owned target mention decisions."""

from __future__ import annotations

from pathlib import Path
import logging

import httpx
import pytest

from kazusa_ai_chatbot.config import DIALOG_GENERATOR_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.utils import load_personality
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm

_ROOT = Path(__file__).resolve().parents[1]
_PERSONALITY_PATH = _ROOT / "personalities" / "kazusa.json"


async def _skip_if_llm_unavailable() -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{DIALOG_GENERATOR_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError:
        pytest.skip(
            f"LLM endpoint is unavailable: {DIALOG_GENERATOR_LLM_BASE_URL}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "LLM endpoint returned server error "
            f"{response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    await _skip_if_llm_unavailable()


def _dialog_state(*, direct_target: bool) -> dict:
    """Build a real dialog-agent state for mention-decision validation.

    Args:
        direct_target: Whether the content plan should aim at the current
            user personally.

    Returns:
        Dialog-agent input using the real character profile.
    """

    if direct_target:
        internal_monologue = (
            "One current target user is owed a light follow-up about an "
            "overdue challenge. The line should feel like a direct nudge."
        )
        linguistic_directives = {
            "rhetorical_strategy": (
                "Make a short teasing nudge toward the current target user."
            ),
            "linguistic_style": (
                "Casual chat fragments, playful but clear."
            ),
            "accepted_user_preferences": [],
            "content_plan": {
                "visible_goal": "Directly nudge the current target user to resume the overdue challenge.",
                "semantic_content": (
                    "The user promised a harder challenge on 2026-05-07 and "
                    "it is now overdue; ask for the challenge in a light "
                    "teasing way."
                ),
                "rendering": "1-2 short chat fragments; direct to the current target user.",
            },
            "forbidden_phrases": [],
        }
        contextual_directives = {
            "social_distance": "close and playful",
            "emotional_intensity": "low excitement",
            "vibe_check": "casual follow-up",
            "relational_dynamic": "the character is nudging one known user",
        }
    else:
        internal_monologue = (
            "The chat atmosphere is loose and lively. The next line should be "
            "a general aside about the mood, not aimed at one person."
        )
        linguistic_directives = {
            "rhetorical_strategy": (
                "Make a brief general remark about the chat atmosphere."
            ),
            "linguistic_style": (
                "Casual chat fragments, lightly amused."
            ),
            "accepted_user_preferences": [],
            "content_plan": {
                "visible_goal": "Make a general comment about the current chat mood.",
                "semantic_content": "The group has been talking about AI character tropes.",
                "voice": "Do not address or challenge one specific user.",
                "rendering": "1 short chat fragment; broadcast-like, not personally aimed.",
            },
            "forbidden_phrases": [],
        }
        contextual_directives = {
            "social_distance": "friendly shared-room distance",
            "emotional_intensity": "low amusement",
            "vibe_check": "casual group banter",
            "relational_dynamic": "the character is making a broad aside",
        }

    state = {
        "character_profile": load_personality(_PERSONALITY_PATH),
        "internal_monologue": internal_monologue,
        "action_directives": {
            "linguistic_directives": linguistic_directives,
            "contextual_directives": contextual_directives,
        },
        "chat_history_wide": [
            {
                "role": "user",
                "body_text": "I still need to prepare that harder challenge.",
                "platform_user_id": "platform-user-1",
                "global_user_id": "global-user-1",
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
            {
                "role": "assistant",
                "body_text": "I will be waiting for it.",
                "platform_user_id": "bot-1",
                "global_user_id": "",
                "addressed_to_global_user_ids": ["global-user-1"],
                "broadcast": False,
            },
        ],
        "chat_history_recent": [],
        "platform_user_id": "platform-user-1",
        "platform_bot_id": "bot-1",
        "global_user_id": "global-user-1",
        "user_name": "Target User",
        "user_profile": {"affinity": 700},
    }
    return state


def _record_trace(case_id: str, input_state: dict, result: dict, judgment: str) -> None:
    """Write and log an inspectable dialog mention live-LLM trace.

    Args:
        case_id: Stable scenario identifier.
        input_state: Dialog input state used for the model call.
        result: Dialog-agent output.
        judgment: Human-readable assessment for this contract check.

    Returns:
        None.
    """

    trace_path = write_llm_trace(
        "dialog_mention_target_user_live_llm",
        case_id,
        {
            "content_plan": input_state["action_directives"][
                "linguistic_directives"
            ]["content_plan"],
            "result": result,
            "judgment": judgment,
        },
    )
    logger.info(f"dialog mention live trace: case={case_id} path={trace_path}")


async def test_live_dialog_mentions_target_for_unanchored_group_self_cognition(
    ensure_live_llm,
) -> None:
    del ensure_live_llm
    state = _dialog_state(direct_target=True)

    result = await dialog_agent(state)

    _record_trace(
        "unanchored_group_self_cognition",
        state,
        result,
        "Expected true because the generated message is semantically aimed at one current target user.",
    )
    assert result["final_dialog"], f"Dialog output was empty: {result!r}"
    assert result["mention_target_user"] is True, result


async def test_live_dialog_does_not_mention_for_general_group_remark(
    ensure_live_llm,
) -> None:
    del ensure_live_llm
    state = _dialog_state(direct_target=False)

    result = await dialog_agent(state)

    _record_trace(
        "general_group_remark",
        state,
        result,
        "Expected false because the generated message is a general remark, not personally aimed.",
    )
    assert result["final_dialog"], f"Dialog output was empty: {result!r}"
    assert result["mention_target_user"] is False, result
