"""Focused tests for past-dialog cognition residual projection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pymongo.errors import PyMongoError


def _candidate(
    *,
    visible_text: str = "I meant that the idea needed time.",
    llm_trace_id: str = "trace-1",
    role: str = "assistant",
    global_user_id: str = "character-1",
    conversation_row_id: str = "row-1",
    source: str = "reply_context",
) -> Any:
    from kazusa_ai_chatbot.past_dialog_cognition.models import (
        PastDialogCognitionCandidate,
    )

    return PastDialogCognitionCandidate(
        visible_text=visible_text,
        llm_trace_id=llm_trace_id,
        created_at="2026-06-01T00:00:00Z",
        source=source,
        role=role,
        global_user_id=global_user_id,
        conversation_row_id=conversation_row_id,
        platform_message_id=f"platform-{conversation_row_id}",
        platform="debug",
        platform_channel_id="channel-1",
    )


@pytest.mark.asyncio
async def test_empty_candidates_return_empty_context() -> None:
    from kazusa_ai_chatbot.past_dialog_cognition.runtime import (
        build_past_dialog_cognition_context,
    )

    result = await build_past_dialog_cognition_context(
        [],
        character_global_user_id="character-1",
    )

    assert result["past_dialog_cognition_context"] == ""
    assert result["candidate_count"] == 0
    assert result["selected_count"] == 0


@pytest.mark.asyncio
async def test_metadata_mode_empty_parsed_output_is_ordinary_forgetting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        assert list(trace_ids) == ["trace-1"]
        assert set(stage_names) == {
            "l2a_conscious_framing",
            "l2c1_judgment_synthesis",
        }
        return [{
            "trace_id": "trace-1",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {},
            "created_at": "2026-06-01T00:00:01Z",
        }]

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [_candidate()],
        character_global_user_id="character-1",
    )

    assert result["past_dialog_cognition_context"] == ""
    assert result["candidate_count"] == 1
    assert result["selected_count"] == 0


@pytest.mark.asyncio
async def test_trace_database_error_is_ordinary_forgetting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        del trace_ids, stage_names
        raise PyMongoError("trace store unavailable")

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [_candidate()],
        character_global_user_id="character-1",
    )

    assert result["past_dialog_cognition_context"] == ""
    assert result["status"] == "lookup_failed"
    assert result["selected_count"] == 0


@pytest.mark.asyncio
async def test_l2a_and_l2c1_fields_are_projected_without_raw_or_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": "trace-1",
                "stage_name": "l2a_conscious_framing",
                "sequence": 1,
                "parsed_output": {
                    "internal_monologue": "She was unsure but wanted to be fair.",
                    "logical_stance": "The idea is possible, but incomplete.",
                    "character_intent": "Answer cautiously.",
                    "dialog_draft": "raw dialog wording must not leak",
                },
                "created_at": "2026-06-01T00:00:01Z",
                "raw_messages": "raw prompt must not leak",
                "raw_response_text": "raw response must not leak",
            },
            {
                "trace_id": "trace-1",
                "stage_name": "l2c1_judgment_synthesis",
                "sequence": 2,
                "parsed_output": {
                    "logical_stance": "Treat it as a tentative proposal.",
                    "character_intent": "Preserve continuity without overstating.",
                    "judgment_note": "The earlier message was exploratory.",
                },
                "created_at": "2026-06-01T00:00:02Z",
            },
        ]

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [_candidate(conversation_row_id="row-secret")],
        character_global_user_id="character-1",
    )

    context = result["past_dialog_cognition_context"]
    assert result["selected_count"] == 1
    assert "I meant that the idea needed time." in context
    assert "She was unsure but wanted to be fair." in context
    assert "Treat it as a tentative proposal." in context
    assert "The earlier message was exploratory." in context
    assert "raw prompt must not leak" not in context
    assert "raw response must not leak" not in context
    assert "raw dialog wording must not leak" not in context
    assert "trace-1" not in context
    assert "row-secret" not in context
    assert "l2a_conscious_framing" not in context
    assert "l2c1_judgment_synthesis" not in context


@pytest.mark.asyncio
async def test_candidates_sharing_trace_project_one_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split dialog rows from one cognition should not duplicate residuals."""

    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    captured_trace_ids: list[str] = []

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        del stage_names
        captured_trace_ids.extend(trace_ids)
        return [{
            "trace_id": "trace-shared",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {
                "internal_monologue": "shared private residual",
            },
            "created_at": "2026-06-01T00:00:01Z",
        }]

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [
            _candidate(
                visible_text="first split message",
                llm_trace_id="trace-shared",
                conversation_row_id="row-1",
                source="conversation_evidence",
            ),
            _candidate(
                visible_text="second split message",
                llm_trace_id="trace-shared",
                conversation_row_id="row-2",
                source="conversation_evidence",
            ),
        ],
        character_global_user_id="character-1",
    )

    context = result["past_dialog_cognition_context"]
    assert captured_trace_ids == ["trace-shared"]
    assert result["candidate_count"] == 2
    assert result["selected_count"] == 1
    assert context.count("shared private residual") == 1


@pytest.mark.asyncio
async def test_candidate_filtering_happens_before_trace_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    captured_trace_ids: list[str] = []

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        captured_trace_ids.extend(trace_ids)
        return [{
            "trace_id": "trace-valid",
            "stage_name": "l2a_conscious_framing",
            "sequence": 1,
            "parsed_output": {
                "internal_monologue": "valid residual",
            },
            "created_at": "2026-06-01T00:00:01Z",
        }]

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [
            _candidate(
                visible_text="valid",
                llm_trace_id="trace-valid",
                conversation_row_id="row-valid",
            ),
            _candidate(
                llm_trace_id="trace-user",
                role="user",
                conversation_row_id="row-user",
            ),
            _candidate(
                llm_trace_id="trace-other",
                global_user_id="someone-else",
                conversation_row_id="row-other",
            ),
            _candidate(
                visible_text="",
                llm_trace_id="trace-empty",
                conversation_row_id="row-empty",
            ),
            _candidate(
                llm_trace_id="",
                conversation_row_id="row-missing-trace",
            ),
        ],
        character_global_user_id="character-1",
    )

    assert captured_trace_ids == ["trace-valid"]
    assert result["candidate_count"] == 5
    assert result["selected_count"] == 1
    assert "valid residual" in result["past_dialog_cognition_context"]


@pytest.mark.asyncio
async def test_max_dialog_and_character_caps_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.past_dialog_cognition import runtime

    async def list_steps(
        trace_ids: Sequence[str],
        *,
        stage_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": trace_id,
                "stage_name": "l2a_conscious_framing",
                "sequence": index,
                "parsed_output": {
                    "internal_monologue": f"residual for {trace_id} " * 20,
                },
                "created_at": f"2026-06-01T00:00:0{index}Z",
            }
            for index, trace_id in enumerate(trace_ids, start=1)
        ]

    monkeypatch.setattr(
        runtime,
        "list_llm_trace_steps_for_trace_ids",
        list_steps,
    )

    result = await runtime.build_past_dialog_cognition_context(
        [
            _candidate(
                visible_text=f"visible {index}",
                llm_trace_id=f"trace-{index}",
                conversation_row_id=f"row-{index}",
            )
            for index in range(1, 5)
        ],
        character_global_user_id="character-1",
        max_dialogs=2,
        context_char_limit=220,
    )

    context = result["past_dialog_cognition_context"]
    assert result["selected_count"] <= 2
    assert len(context) <= 220
    assert "visible 1" in context
    assert "visible 3" not in context
    assert "visible 4" not in context
