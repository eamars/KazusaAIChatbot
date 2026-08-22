"""Live dialog quality gates for task-resolution result surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.background_work.result_source import (
    build_result_ready_episode_from_job,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from tests.cognition_test_helpers import canonical_episode


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ARTIFACT_ROOT = Path("test_artifacts/task_resolution/raw")
_FORBIDDEN_RUNTIME_WORDS = (
    "worker",
    "queue",
    "job id",
    "background_work",
    "后台工作者",
    "队列",
)


class _CapturingDialogLLM:
    """Capture production dialog-model requests and raw responses."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object | None = None,
    ) -> object:
        response = await self.delegate.ainvoke(messages, config=config)
        self.calls.append({
            "prompt_messages": [str(message.content) for message in messages],
            "raw_model_output": str(response.content),
        })
        return response


def _character_profile() -> dict[str, object]:
    """Return a compact character profile for grounded dialog review."""

    return {
        "name": "Kazusa",
        "mood": "Neutral",
        "vibe_check": "Calm and attentive",
        "character_reflection": "Keep factual claims tied to evidence.",
        "personality_brief": {
            "logic": "State verified facts before limitations.",
            "tempo": "Use concise natural sentences.",
            "defense": "Stay direct without sounding mechanical.",
            "quirks": "Use a brief pause only when it helps clarity.",
            "taboos": "Never expose runtime internals.",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.2,
            "fragmentation": 0.2,
            "emotional_leakage": 0.2,
            "rhythmic_bounce": 0.2,
            "direct_assertion": 0.7,
            "softener_density": 0.2,
            "counter_questioning": 0.1,
            "formalism_avoidance": 0.5,
            "abstraction_reframing": 0.2,
            "self_deprecation": 0.0,
        },
    }


def _surface(
    *,
    content_plan: str,
    requirement: str,
    limitation: str,
) -> dict[str, object]:
    """Build one task-result text-surface contract."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": [requirement, limitation],
        "visible_boundaries": [],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "current user",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "plain, warm, and factual",
            "sentence_shape": "two concise sentences",
            "rhythm": "natural conversational rhythm",
            "hesitation": "minimal",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "deliver the grounded task result",
        "permitted_action_results": [],
    }


def _dialog_state(
    *,
    episode: dict[str, object],
    surface: dict[str, object],
) -> dict[str, object]:
    """Build the production dialog node state for one result episode."""

    return {
        "internal_monologue": (
            "The evidence supports a direct answer, and the limitation must "
            "remain visible."
        ),
        "text_surface_output_v2": surface,
        "cognitive_episode": episode,
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "task-resolution-user",
        "platform_bot_id": "task-resolution-bot",
        "global_user_id": "task-resolution-user",
        "user_name": "Test User",
        "user_profile": {},
        "character_profile": _character_profile(),
        "dialog_usage_mode": "live_visible_reply",
    }


def _visible_text(result: dict[str, object]) -> str:
    """Join the validated visible dialog fragments."""

    final_dialog = result.get("final_dialog")
    assert isinstance(final_dialog, list)
    text = "\n".join(
        row.strip()
        for row in final_dialog
        if isinstance(row, str) and row.strip()
    )
    assert text
    return text


def _assert_prompt_safe_dialog(text: str, *, anchor: str) -> None:
    """Require grounded content without runtime implementation vocabulary."""

    assert anchor in text
    lowered = text.casefold()
    assert all(word.casefold() not in lowered for word in _FORBIDDEN_RUNTIME_WORDS)


def _write_artifact(case_id: str, value: dict[str, object]) -> Path:
    """Write one raw persona result for parent-authored review."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"{case_id}.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


async def test_live_inline_result_returns_grounded_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline partial evidence produces grounded dialog and its limitation."""

    capturing_llm = _CapturingDialogLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        capturing_llm,
    )
    episode = canonical_episode(
        episode_id="task-resolution-inline-persona",
        content="What is the current Python 3.14 release status?",
    )
    surface = _surface(
        content_plan=(
            "Tell the user that public evidence confirms Python 3.14 release "
            "information, while noting that the latest status may change."
        ),
        requirement="Mention Python 3.14 and the verified public evidence.",
        limitation="State that the latest release status may change.",
    )

    result = await dialog_module.dialog_generator(
        _dialog_state(episode=episode, surface=surface)
    )
    visible_text = _visible_text(result)
    _assert_prompt_safe_dialog(visible_text, anchor="3.14")
    assert any(
        term in visible_text.casefold()
        for term in ("change", "current", "latest", "变化", "当前", "最新")
    )
    artifact_path = _write_artifact(
        "persona_inline_grounded_dialog",
        {
            "schema_version": "task_resolution_persona_live_case.v1",
            "case_id": "persona_inline_grounded_dialog",
            "task_result_status": "partial",
            "task_evidence": [{
                "summary": "Python 3.14 public release evidence is available.",
                "limitations": ["The latest status may change."],
            }],
            "surface": surface,
            "dialog_model_calls": capturing_llm.calls,
            "dialog_result": result,
            "visible_text": visible_text,
        },
    )
    print(f"TASK_RESOLUTION_PERSONA_ARTIFACT={artifact_path}")


async def test_live_deferred_result_reenters_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable completed job re-enters as evidence and renders naturally."""

    job = {
        "schema_version": "background_work_job.v2",
        "job_id": "job-task-resolution-persona",
        "accepted_task_id": "task-task-resolution-persona",
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": "2026-08-01T00:05:00+00:00",
        "completed_at": "2026-08-01T00:05:00+00:00",
        "source_platform": "debug",
        "source_platform_bot_id": "task-resolution-bot",
        "source_channel_id": "debug:user:task-resolution-user",
        "source_channel_type": "private",
        "source_character_name": "Kazusa",
        "requester_global_user_id": "task-resolution-user",
        "requester_platform_user_id": "task-resolution-user",
        "requester_display_name": "Test User",
        "result_summary": (
            "Pydantic 2.11 migration evidence is ready; one optional plugin "
            "compatibility check remains."
        ),
        "artifact_text": "",
        "failure_summary": "",
        "worker_metadata": {},
    }
    episode = build_result_ready_episode_from_job(job)
    percept = episode["percepts"][0]
    assert percept["content"]["cognition_source"]["source_kind"] == "tool_result"
    assert episode["origin_metadata"]["task_id"] == job["accepted_task_id"]
    assert percept["content"]["semantic_summary"] == job["result_summary"]

    capturing_llm = _CapturingDialogLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        capturing_llm,
    )
    surface = _surface(
        content_plan=(
            "Tell the user the Pydantic 2.11 migration evidence is ready and "
            "that one optional plugin compatibility check remains."
        ),
        requirement="Mention Pydantic 2.11 and the migration evidence.",
        limitation="Mention the remaining optional plugin check.",
    )
    result = await dialog_module.dialog_generator(
        _dialog_state(episode=episode, surface=surface)
    )
    visible_text = _visible_text(result)
    _assert_prompt_safe_dialog(visible_text, anchor="2.11")
    assert any(
        term in visible_text.casefold()
        for term in ("plugin", "check", "插件", "检查")
    )
    artifact_path = _write_artifact(
        "persona_deferred_result_reentry",
        {
            "schema_version": "task_resolution_persona_live_case.v1",
            "case_id": "persona_deferred_result_reentry",
            "durable_job": job,
            "result_ready_episode": episode,
            "surface": surface,
            "dialog_model_calls": capturing_llm.calls,
            "dialog_result": result,
            "visible_text": visible_text,
            "delivery_owner": "outside_background_worker",
        },
    )
    print(f"TASK_RESOLUTION_PERSONA_ARTIFACT={artifact_path}")
