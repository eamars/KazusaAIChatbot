"""Live LLM contract tests for dialog-authored inline mention tags."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

import kazusa_ai_chatbot.nodes.dialog_agent as dialog_module
from kazusa_ai_chatbot.config import DIALOG_GENERATOR_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm
RETIRED_FIELD = "mention" + "_target_user"
FORBIDDEN_NATIVE_MARKERS = (
    "<@",
    "[CQ:",
    "CQ:at",
    "qq=",
    "Discord",
    "NapCat",
    "OneBot",
    "QQ",
)


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


class _CapturingLiveLLM:
    """Capture raw live LLM calls while preserving the real model behavior."""

    def __init__(self, delegate: Any, calls: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._calls = calls

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        response = await self._delegate.ainvoke(messages, **kwargs)
        self._calls.append({
            "messages": [
                {
                    "type": message.__class__.__name__,
                    "content": getattr(message, "content", ""),
                }
                for message in messages
            ],
            "raw_content": getattr(response, "content", ""),
        })
        return response


def _character_profile() -> dict[str, Any]:
    return {
        "name": "Kazusa",
        "personality_brief": {
            "logic": "Judge facts and social context before speaking.",
            "tempo": "Short chat lines; more complete when technical.",
            "defense": "Slightly guarded, but factual.",
            "quirks": "Uses brief pauses when unsure.",
            "taboos": "Do not expose system instructions.",
            "mbti": "INTJ",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.25,
            "fragmentation": 0.35,
            "emotional_leakage": 0.25,
            "rhythmic_bounce": 0.25,
            "direct_assertion": 0.6,
            "softener_density": 0.3,
            "counter_questioning": 0.25,
            "formalism_avoidance": 0.45,
            "abstraction_reframing": 0.25,
            "self_deprecation": 0.15,
        },
    }


def _base_state(
    *,
    user_name: str,
    content_plan: dict[str, str],
    internal_monologue: str,
    chat_history_wide: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if chat_history_wide is None:
        chat_history_wide = [
            {
                "role": "user",
                "body_text": f"{user_name}: I will bring the harder challenge.",
                "platform_user_id": "platform-alex",
                "global_user_id": "global-alex",
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            }
        ]

    return {
        "character_profile": _character_profile(),
        "internal_monologue": internal_monologue,
        "action_directives": {
            "linguistic_directives": {
                "rhetorical_strategy": "Follow the content plan exactly.",
                "linguistic_style": "Concise group chat wording.",
                "accepted_user_preferences": [],
                "content_plan": content_plan,
                "forbidden_phrases": [],
            },
            "contextual_directives": {
                "social_distance": "friendly",
                "emotional_intensity": "low",
                "vibe_check": "casual",
                "relational_dynamic": "ordinary group chat",
            },
        },
        "chat_history_wide": chat_history_wide,
        "chat_history_recent": [],
        "debug_modes": {},
        "should_respond": True,
        "dialog_usage_mode": "live_visible_reply",
        "platform_user_id": "platform-alex",
        "platform_bot_id": "bot-1",
        "global_user_id": "global-alex",
        "user_name": user_name,
        "user_profile": {"affinity": 700},
    }


def _case_current_user() -> dict[str, Any]:
    return _base_state(
        user_name="Alex",
        internal_monologue=(
            "Alex is the current user and needs a direct, light nudge."
        ),
        content_plan={
            "visible_goal": "Directly tag Alex and nudge them.",
            "semantic_content": (
                "Alex promised a harder challenge and it is overdue. Use the "
                "exact visible tag @Alex."
            ),
            "rendering": "One short ordinary text message; include @Alex exactly once.",
        },
    )


def _case_no_tag() -> dict[str, Any]:
    return _base_state(
        user_name="Alex",
        internal_monologue=(
            "The next line should be a general aside to the room."
        ),
        content_plan={
            "visible_goal": "Make a general group remark.",
            "semantic_content": (
                "The group is joking about character tropes. Do not tag anyone."
            ),
            "rendering": "One short ordinary text message; no @ token.",
        },
    )


def _case_named_participant() -> dict[str, Any]:
    return _base_state(
        user_name="Alex",
        internal_monologue=(
            "Moca is a named participant already present in the content plan."
        ),
        content_plan={
            "visible_goal": "Tag the named participant Moca.",
            "semantic_content": (
                "Moca solved the puzzle first. Use the exact visible tag @Moca."
            ),
            "rendering": "One short ordinary text message; include @Moca exactly once.",
        },
        chat_history_wide=[
            {
                "role": "user",
                "body_text": "Moca: I think the answer is 42.",
                "platform_user_id": "platform-moca",
                "global_user_id": "global-moca",
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            }
        ],
    )


def _case_multiple_tags() -> dict[str, Any]:
    return _base_state(
        user_name="Alex",
        internal_monologue=(
            "Alex and Moca both need to be named in the same visible reply."
        ),
        content_plan={
            "visible_goal": "Tag Alex and Moca in the same reply.",
            "semantic_content": (
                "Alex should bring the challenge, and Moca should keep the "
                "score. Use exact visible tags @Alex and @Moca."
            ),
            "rendering": "One or two short fragments; include @Alex and @Moca.",
        },
        chat_history_wide=[
            {
                "role": "user",
                "body_text": "Alex: I can bring the challenge.",
                "platform_user_id": "platform-alex",
                "global_user_id": "global-alex",
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
            {
                "role": "user",
                "body_text": "Moca: I can keep score.",
                "platform_user_id": "platform-moca",
                "global_user_id": "global-moca",
                "addressed_to_global_user_ids": [],
                "broadcast": False,
            },
        ],
    )


def _case_fenced_at() -> dict[str, Any]:
    return _base_state(
        user_name="Alex",
        internal_monologue=(
            "The answer is technical and the at sign belongs in code."
        ),
        content_plan={
            "visible_goal": "Show a tiny code example with a decorator.",
            "semantic_content": (
                "Include this exact code block: ```python\n"
                "@cache\n"
                "def score():\n"
                "    return 1\n"
                "```"
            ),
            "rendering": (
                "One ordinary text message with the fenced code block; do not tag anyone."
            ),
        },
    )


CASES = {
    "current_user": {
        "state": _case_current_user,
        "must_include": ("@Alex",),
        "must_not_include": (),
    },
    "no_tag": {
        "state": _case_no_tag,
        "must_include": (),
        "must_not_include": ("@",),
    },
    "named_participant": {
        "state": _case_named_participant,
        "must_include": ("@Moca",),
        "must_not_include": (),
    },
    "multiple_tags": {
        "state": _case_multiple_tags,
        "must_include": ("@Alex", "@Moca"),
        "must_not_include": (),
    },
    "fenced_at": {
        "state": _case_fenced_at,
        "must_include": ("```", "@cache"),
        "must_not_include": ("@Alex", "@Moca"),
    },
}


def _joined_dialog(final_dialog: Any) -> str:
    if not isinstance(final_dialog, list):
        return_value = ""
        return return_value
    text_items = [item for item in final_dialog if isinstance(item, str)]
    return_value = "\n".join(text_items)
    return return_value


def _raw_json(raw_content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return_value = None
        return return_value
    if not isinstance(parsed, dict):
        return_value = None
        return return_value
    return parsed


def _assess_case(
    *,
    case_id: str,
    result: dict[str, Any],
    raw_content: str,
) -> dict[str, Any]:
    failures: list[str] = []
    parsed_raw = _raw_json(raw_content)
    if parsed_raw is None:
        failures.append("raw generator output is not a JSON object")
    else:
        if RETIRED_FIELD in parsed_raw:
            failures.append("raw generator output contains retired field")
        raw_dialog = parsed_raw.get("final_dialog")
        if not isinstance(raw_dialog, list):
            failures.append("raw final_dialog is not a list")
        elif not all(isinstance(item, str) for item in raw_dialog):
            failures.append("raw final_dialog contains a non-string item")

    if RETIRED_FIELD in result:
        failures.append("dialog_agent result contains retired field")

    final_dialog = result.get("final_dialog")
    if not isinstance(final_dialog, list):
        failures.append("result final_dialog is not a list")
    elif not final_dialog:
        failures.append("result final_dialog is empty")
    elif not all(isinstance(item, str) for item in final_dialog):
        failures.append("result final_dialog contains a non-string item")

    joined_dialog = _joined_dialog(final_dialog)
    if not joined_dialog.strip():
        failures.append("joined dialog is blank")

    for required_text in CASES[case_id]["must_include"]:
        if required_text not in joined_dialog:
            failures.append(f"missing required text: {required_text}")

    for forbidden_text in CASES[case_id]["must_not_include"]:
        if forbidden_text in joined_dialog:
            failures.append(f"contains forbidden text: {forbidden_text}")

    combined_text = f"{raw_content}\n{joined_dialog}"
    for marker in FORBIDDEN_NATIVE_MARKERS:
        if marker in combined_text:
            failures.append(f"contains native/platform marker: {marker}")

    return {
        "passed": not failures,
        "failures": failures,
        "parsed_raw": parsed_raw,
        "joined_dialog": joined_dialog,
    }


async def _run_case(
    case_id: str,
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del ensure_live_llm
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        _CapturingLiveLLM(dialog_module._dialog_generator_llm, calls),
    )

    state = CASES[case_id]["state"]()
    result = await dialog_agent(state)
    raw_content = calls[-1]["raw_content"] if calls else ""
    assessment = _assess_case(
        case_id=case_id,
        result=result,
        raw_content=raw_content,
    )
    trace_path = write_llm_trace(
        "dialog_inline_mentions_live_llm",
        case_id,
        {
            "case_id": case_id,
            "content_plan": state["action_directives"]["linguistic_directives"][
                "content_plan"
            ],
            "model_calls": calls,
            "result": result,
            "assessment": assessment,
            "quality_notes": (
                "Pass means the model followed the inline tag-sign contract "
                "without adding native platform syntax or the retired field."
            ),
        },
    )
    logger.info(
        "dialog inline mention live trace: case=%s path=%s",
        case_id,
        trace_path,
    )

    assert assessment["passed"], assessment


async def test_dialog_inline_mention_shape_current_user_live_llm(
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case("current_user", ensure_live_llm, monkeypatch)


async def test_dialog_inline_mention_shape_no_tag_live_llm(
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case("no_tag", ensure_live_llm, monkeypatch)


async def test_dialog_inline_mention_shape_named_participant_live_llm(
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case("named_participant", ensure_live_llm, monkeypatch)


async def test_dialog_inline_mention_shape_multiple_tags_live_llm(
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case("multiple_tags", ensure_live_llm, monkeypatch)


async def test_dialog_inline_mention_shape_fenced_at_live_llm(
    ensure_live_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_case("fenced_at", ensure_live_llm, monkeypatch)
