"""Tests for Stage 7 — Memory Writer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from kazusa_ai_chatbot.nodes.memory_writer import character_memory_compactor_llm, ego_reflector_llm, memory_writer, social_archivist_llm, user_memory_compactor_llm
from kazusa_ai_chatbot.db import AFFINITY_DEFAULT

# ── Live LLM tests ──────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v
live_llm = pytest.mark.live_llm


def _mock_llm(content: str) -> MagicMock:
    """Create a mock ChatOpenAI whose ainvoke returns an AIMessage."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


@pytest.fixture
def mock_memory_state():
    return {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "I like cats.",
        "response": "That is nice.",
        "timestamp": "2026-03-30T20:00:00Z",
        "agent_results": [],
        "conversation_history": [],
    }


def _make_state(**overrides) -> dict:
    base = {
        "user_id": "user_123",
        "message_text": "Call me Commander from now on",
        "response": "Understood, Commander.",
        "timestamp": "2026-03-30T20:00:00Z",
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_writer_extracts_facts_and_mood(mock_memory_state):
    """Memory writer should parse JSON from new LLM functions and call upsert functions."""
    # Mock social archivist response
    social_result = {
        "new_facts": [
            {
                "content": "User prefers to be called Commander",
                "type": "Preference",
                "category": "Personal"
            }
        ],
        "affinity_delta": 5
    }
    
    # Mock ego reflector response
    ego_result = {
        "mood": "amused",
        "emotional_tone": "teasing",
        "memory": [
            {
                "content": "User shared a fact",
                "score": 50,
                "category": "Cognitive"
            }
        ]
    }

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result) as mock_social,
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result) as mock_ego,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=505) as mock_aff,
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == ["User prefers to be called Commander"]
    mock_facts.assert_called_once_with("user_123", ["User prefers to be called Commander"])
    mock_char.assert_called_once()
    call_args = mock_char.call_args
    assert call_args[0][0] == "amused"  # mood
    assert call_args[0][1] == "teasing"  # emotional_tone
    assert call_args[0][2] == ["User shared a fact"]  # recent events
    mock_aff.assert_called_once_with("user_123", 7)
    mock_social.assert_called_once()
    mock_ego.assert_called_once()


@pytest.mark.asyncio
async def test_writer_no_facts():
    social_result = {
        "new_facts": [],
        "affinity_delta": 3
    }
    
    ego_result = {
        "mood": "neutral",
        "emotional_tone": "balanced",
        "memory": []
    }

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=203),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()
    mock_char.assert_called_once()


@pytest.mark.asyncio
async def test_writer_handles_markdown_fenced_json():
    social_result = {
        "new_facts": [
            {
                "content": "Likes cats",
                "type": "Preference",
                "category": "Personal"
            }
        ],
        "affinity_delta": 0
    }
    
    ego_result = {
        "mood": "amused",
        "emotional_tone": "warm",
        "memory": []
    }

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock),
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=200),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == ["Likes cats"]
    mock_facts.assert_called_once()


@pytest.mark.asyncio
async def test_writer_handles_malformed_json():
    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()
    mock_char.assert_not_called()


@pytest.mark.asyncio
async def test_writer_handles_llm_failure():
    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, side_effect=Exception("LLM down")),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, side_effect=Exception("LLM down")),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()


@pytest.mark.asyncio
async def test_writer_skips_when_no_user_id():
    result = await memory_writer(_make_state(user_id=""))
    assert result["new_facts"] == []


@pytest.mark.asyncio
async def test_writer_skips_when_no_message():
    result = await memory_writer(_make_state(message_text=""))
    assert result["new_facts"] == []


@pytest.mark.asyncio
async def test_writer_clamps_affinity_delta():
    """affinity_delta from LLM is processed with non-linear scaling."""
    social_result = {
        "new_facts": [],
        "affinity_delta": -50,  # LLM returns out-of-range value
    }
    
    ego_result = {
        "mood": "angry",
        "emotional_tone": "hostile",
        "memory": []
    }

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock),
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=50) as mock_aff,
    ):
        await memory_writer(_make_state())

    # With affinity AFFINITY_DEFAULT (mid-range), raw -50 should be processed to -65 (1.3x scaling)
    mock_aff.assert_called_once_with("user_123", -65)


@pytest.mark.asyncio
async def test_writer_handles_social_archivist_failure_only():
    """Test that ego reflector still works when social archivist fails."""
    social_result = {}  # Empty result (failure)
    
    ego_result = {
        "mood": "playful",
        "emotional_tone": "warm",
        "memory": [
            {
                "content": "I enjoyed our conversation",
                "score": 60,
                "category": "Emotional"
            }
        ]
    }

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock),
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == []
    mock_facts.assert_not_called()
    mock_char.assert_called_once()
    call_args = mock_char.call_args
    assert call_args[0][0] == "playful"
    assert call_args[0][1] == "warm"
    assert call_args[0][2] == ["I enjoyed our conversation"]


@pytest.mark.asyncio
async def test_writer_handles_ego_reflector_failure_only():
    """Test that social archivist still works when ego reflector fails."""
    social_result = {
        "new_facts": [
            {
                "content": "User is a software developer",
                "type": "Biography",
                "category": "Personal"
            }
        ],
        "affinity_delta": 3
    }
    
    ego_result = {}  # Empty result (failure)

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value=social_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value=ego_result),
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=203) as mock_aff,
    ):
        result = await memory_writer(_make_state())

    assert result["new_facts"] == ["User is a software developer"]
    mock_facts.assert_called_once_with("user_123", ["User is a software developer"])
    mock_char.assert_not_called()  # Should not be called due to ego reflector failure
    mock_aff.assert_called_once_with("user_123", 4)


# ── Live LLM test ────────────────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v

live_llm = pytest.mark.live_llm


@live_llm
@pytest.mark.asyncio
async def test_live_writer_extracts_valid_json():
    """Call the real LLM functions and verify they return parseable extraction JSON."""
    state = _make_state(
        message_text="Please call me Commander from now on.",
        response="As you wish, Commander. I shall address you accordingly.",
        conversation_history=[
            {"speaker": "TestUser", "speaker_id": "user_123", "message": "How's your day going?"},
            {"speaker": "TestBot", "speaker_id": "bot_456", "message": "Pretty good! Just thinking about dessert recipes."}
        ],
        user_memory=["User works in tech"],
        affinity=AFFINITY_DEFAULT,
        assembler_output={"user_topic": "personal preference"}
    )

    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_user_facts", new_callable=AsyncMock) as mock_facts,
        patch("kazusa_ai_chatbot.nodes.memory_writer.upsert_character_state", new_callable=AsyncMock) as mock_char,
        patch("kazusa_ai_chatbot.nodes.memory_writer.update_affinity", new_callable=AsyncMock, return_value=505) as mock_aff,
    ):
        result = await memory_writer(state)

    # Verify the result has the expected structure
    assert isinstance(result, dict)
    assert "new_facts" in result
    assert isinstance(result["new_facts"], list)
    
    # Verify database functions were called appropriately
    if result["new_facts"]:
        mock_facts.assert_called_once()
    else:
        mock_facts.assert_not_called()
    
    # Character state should only be called if mood or tone is present
    # (Real LLM may not always generate mood/tone for simple exchanges)
    # So we just verify the call count is either 0 or 1 (both are valid)
    assert mock_char.call_count in [0, 1], f"Expected upsert_character_state to be called 0 or 1 times, got {mock_char.call_count}"
    
    # Affinity should be updated if there's a delta
    mock_aff.assert_called_once()


# ── Ego Reflector Tests ───────────────────────────────────────────────────

@pytest.fixture
def mock_ego_reflector_state():
    return {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "千纱会python么？",
        "response": "是的，我会一些Python编程呢！",
        "timestamp": "2026-03-30T20:00:00Z",
        "supervisor_chain_of_thought": [
            {
                "step": 1,
                "step_name": "initial_planning",
                "input": {"current_message": {"message": "千纱会python么？"}},
                "output": {"agents": [], "response_language": "Chinese"}
            },
            {
                "step": 2,
                "step_name": "agent_dispatch",
                "input": {"agents": []},
                "output": {"agent_results_count": 0}
            }
        ]
    }


def _make_ego_reflector_state(**overrides) -> dict:
    base = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "千纱会python么？",
        "response": "是的，我会一些Python编程呢！",
        "supervisor_chain_of_thought": [
            {
                "step": 1,
                "step_name": "initial_planning",
                "input": {"current_message": {"message": "千纱会python么？"}},
                "output": {"agents": [], "response_language": "Chinese"}
            }
        ]
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_ego_reflector_extracts_valid_json(mock_ego_reflector_state):
    """Ego reflector should parse JSON and return structured ego reflection."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "mood": "curious",
        "emotional_tone": "warm",
        "memory": [
            {
                "content": "I felt happy to share my Python knowledge with the user who seemed genuinely interested in my abilities.",
                "score": 75,
                "category": "Emotional"
            }
        ]
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await ego_reflector_llm(mock_ego_reflector_state)

    assert result["mood"] == "curious"
    assert result["emotional_tone"] == "warm"
    assert isinstance(result["memory"], list)
    assert len(result["memory"]) == 1
    assert "Python knowledge" in result["memory"][0]["content"]
    assert result["memory"][0]["score"] == 75
    assert result["memory"][0]["category"] == "Emotional"


@pytest.mark.asyncio
async def test_ego_reflector_handles_malformed_json():
    """Ego reflector should return {} on malformed JSON."""
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=_mock_llm("not valid json at all")):
        result = await ego_reflector_llm(_make_ego_reflector_state())

    assert result == {}


@pytest.mark.asyncio
async def test_ego_reflector_handles_llm_failure():
    """Ego reflector should return {} on LLM failure."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock):
        result = await ego_reflector_llm(_make_ego_reflector_state())

    assert result == {}


@pytest.mark.asyncio
async def test_ego_reflector_skips_when_no_chain_of_thought():
    """Ego reflector should return {} when no supervisor chain of thought."""
    result = await ego_reflector_llm(_make_ego_reflector_state(supervisor_chain_of_thought=[]))
    assert result == {}


@pytest.mark.asyncio
async def test_ego_reflector_skips_when_no_response():
    """Ego reflector should return {} when no response."""
    result = await ego_reflector_llm(_make_ego_reflector_state(response=""))
    assert result == {}


@pytest.mark.asyncio
async def test_ego_reflector_formats_prompt_correctly():
    """Ego reflector should format system prompt with persona name and bot ID."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "mood": "playful",
        "emotional_tone": "teasing",
        "memory": [
            {
                "content": "I enjoyed the technical question about Python.",
                "score": 60,
                "category": "Cognitive"
            }
        ]
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm) as mock_get_llm:
        await ego_reflector_llm(_make_ego_reflector_state())

        # Verify LLM was called
        mock_get_llm.return_value.ainvoke.assert_called_once()
        
        # Get the system message that was passed to LLM
        call_args = mock_get_llm.return_value.ainvoke.call_args
        messages = call_args[0][0]  # First argument, first positional arg
        
        system_message = messages[0]
        assert "TestBot" in system_message.content
        assert "bot_456" in system_message.content
        assert "Ego-Reflector" in system_message.content


# ── Live LLM test for Ego Reflector ───────────────────────────────────────────
# Requires a running LM Studio instance with a chat model loaded.
# Run with:  pytest -m live_llm -v


@live_llm
@pytest.mark.asyncio
async def test_live_ego_reflector_extracts_valid_json():
    """Call the real LLM and verify it returns parseable ego reflection JSON."""
    state = _make_ego_reflector_state(
        message_text="千纱会python么？",
        response="是的，我会一些Python编程呢！特别是在数据处理和自动化脚本方面。",
        supervisor_chain_of_thought=[
            {
                "step": 1,
                "step_name": "initial_planning",
                "input": {"current_message": {"message": "千纱会python么？"}},
                "output": {"agents": [], "response_language": "Chinese", "emotion_directive": "Friendly and curious"}
            },
            {
                "step": 2,
                "step_name": "agent_dispatch",
                "input": {"agents": []},
                "output": {"agent_results_count": 0}
            },
            {
                "step": 6,
                "step_name": "speech_brief",
                "input": {"plan": {"emotion_directive": "Friendly and curious"}},
                "output": {"speech_brief": {"response_brief": {"tone_guidance": "Friendly and curious"}}}
            }
        ]
    )

    result = await ego_reflector_llm(state)

    # Verify the result has the expected structure
    assert isinstance(result, dict)
    assert "mood" in result
    assert "emotional_tone" in result
    assert "memory" in result
    assert isinstance(result["memory"], list)
    
    # Validate memory array structure
    if result["memory"]:  # Only check if memories were extracted
        for memory_item in result["memory"]:
            assert "content" in memory_item
            assert "score" in memory_item
            assert "category" in memory_item
            assert isinstance(memory_item["score"], int)
            assert 1 <= memory_item["score"] <= 100
            assert isinstance(memory_item["content"], str)
            assert len(memory_item["content"]) > 0
            assert memory_item["category"] in ["Emotional", "Cognitive", "Experiential"]


# ── Social Archivist Tests ──────────────────────────────────────────────────

@pytest.fixture
def mock_social_archivist_state():
    return {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "I just got a promotion at work!",
        "response": "Congratulations! That's wonderful news! Tell me more about your new role.",
        "timestamp": "2026-03-30T20:00:00Z",
        "conversation_history": [
            {"speaker": "TestUser", "speaker_id": "user_123", "message": "How's your day going?"},
            {"speaker": "TestBot", "speaker_id": "bot_456", "message": "Pretty good! Just thinking about dessert recipes."}
        ],
        "user_memory": ["User works in tech", "User enjoys cooking"],
        "affinity": 750,
        "assembler_output": {
            "channel_topic": "general",
            "user_topic": "career updates"
        }
    }


def _make_social_archivist_state(**overrides) -> dict:
    base = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "I just got a promotion at work!",
        "response": "Congratulations! That's wonderful news! Tell me more about your new role.",
        "conversation_history": [
            {"speaker": "TestUser", "speaker_id": "user_123", "message": "How's your day going?"},
            {"speaker": "TestBot", "speaker_id": "bot_456", "message": "Pretty good! Just thinking about dessert recipes."}
        ],
        "user_memory": ["User works in tech", "User enjoys cooking"],
        "affinity": 750,
        "assembler_output": {
            "channel_topic": "general",
            "user_topic": "career updates"
        }
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_social_archivist_extracts_valid_json(mock_social_archivist_state):
    """Social archivist should parse JSON and return structured user facts and affinity change."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "new_facts": [
            {
                "content": "TestUser just got a promotion at work",
                "type": "Biography",
                "category": "Personal"
            }
        ],
        "affinity_delta": 5
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await social_archivist_llm(mock_social_archivist_state)

    assert "new_facts" in result
    assert "affinity_delta" in result
    assert len(result["new_facts"]) == 1
    assert result["new_facts"][0]["content"] == "TestUser just got a promotion at work"
    assert result["new_facts"][0]["type"] == "Biography"
    assert result["new_facts"][0]["category"] == "Personal"
    assert result["affinity_delta"] == 5


@pytest.mark.asyncio
async def test_social_archivist_handles_malformed_json():
    """Social archivist should return {} on malformed JSON."""
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=_mock_llm("not valid json at all")):
        result = await social_archivist_llm(_make_social_archivist_state())

    assert result == {}


@pytest.mark.asyncio
async def test_social_archivist_handles_llm_failure():
    """Social archivist should return {} on LLM failure."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=Exception("LLM down"))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock):
        result = await social_archivist_llm(_make_social_archivist_state())

    assert result == {}


@pytest.mark.asyncio
async def test_social_archivist_formats_prompt_correctly():
    """Social archivist should format system prompt with persona name and bot ID."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "new_facts": [],
        "affinity_delta": 0
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm) as mock_get_llm:
        await social_archivist_llm(_make_social_archivist_state())

        # Verify LLM was called
        mock_get_llm.return_value.ainvoke.assert_called_once()
        
        # Get the system message that was passed to LLM
        call_args = mock_get_llm.return_value.ainvoke.call_args
        messages = call_args[0][0]  # First argument, first positional arg
        
        system_message = messages[0]
        assert "TestBot" in system_message.content
        assert "bot_456" in system_message.content
        assert "Social Archivist" in system_message.content


@pytest.mark.asyncio
async def test_social_archivist_includes_all_context():
    """Social archivist should include conversation history, user memory, affinity, and user topic."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "new_facts": [],
        "affinity_delta": 0
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm) as mock_get_llm:
        await social_archivist_llm(_make_social_archivist_state())

        # Get the human message that was passed to LLM
        call_args = mock_get_llm.return_value.ainvoke.call_args
        messages = call_args[0][0]
        human_message = messages[1]
        
        # Parse the JSON content to verify all context is included
        human_data = json.loads(human_message.content)
        
        assert "conversation" in human_data
        assert "user_memory" in human_data
        assert "current_affinity" in human_data
        assert "user_topic" in human_data
        
        assert human_data["current_affinity"] == 750
        assert human_data["user_topic"] == "career updates"
        assert len(human_data["user_memory"]) == 2
        assert "User works in tech" in human_data["user_memory"]


@pytest.mark.asyncio
async def test_social_archivist_handles_missing_context():
    """Social archivist should handle missing context gracefully."""
    # Test with minimal state
    minimal_state = {
        "user_id": "user_123",
        "user_name": "TestUser",
        "bot_id": "bot_456",
        "personality": {"name": "TestBot"},
        "message_text": "Hello",
        "response": "Hi there!",
        "conversation_history": [],
        "user_memory": [],
        "affinity": 500,
        "assembler_output": {}
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "new_facts": [],
        "affinity_delta": 0
    })))

    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await social_archivist_llm(minimal_state)
        
        # Should still work with minimal context
        assert "new_facts" in result
        assert "affinity_delta" in result


@live_llm
@pytest.mark.asyncio
async def test_live_social_archivist_extracts_valid_json():
    """Call the real LLM and verify it returns parseable social archivist JSON."""
    state = _make_social_archivist_state(
        message_text="I just got promoted to Senior Developer! I'm so excited about the new challenges and salary increase.",
        response="Wow, Senior Developer! That's amazing progress! 🎉 You must be so proud. What kind of projects will you be leading?",
        conversation_history=[
            {"speaker": "TestUser", "speaker_id": "user_123", "message": "Been working hard lately"},
            {"speaker": "TestBot", "speaker_id": "bot_456", "message": "I can tell! Your dedication shows"},
            {"speaker": "TestUser", "speaker_id": "user_123", "message": "My boss noticed too"},
            {"speaker": "TestBot", "speaker_id": "bot_456", "message": "That's wonderful! Recognition is important"}
        ],
        user_memory=["User works in tech", "User has been at current company for 2 years"],
        affinity=750,
        assembler_output={"user_topic": "career achievement"}
    )

    result = await social_archivist_llm(state)

    # Verify the result has the expected structure
    assert isinstance(result, dict)
    assert "new_facts" in result
    assert "affinity_delta" in result
    assert isinstance(result["new_facts"], list)
    assert isinstance(result["affinity_delta"], int)
    assert -10 <= result["affinity_delta"] <= 10
    
    # Validate new facts structure if any were extracted
    if result["new_facts"]:
        for fact in result["new_facts"]:
            assert "content" in fact
            assert "type" in fact
            assert "category" in fact
            assert isinstance(fact["content"], str)
            assert len(fact["content"]) > 0
            assert fact["type"] in ["Preference", "Biography", "Relational"]
            assert fact["category"] in ["Personal", "Relational", "Preference"]


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_below_threshold():
    """Memory compactor should process any number of memories (no length check)."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": ["Linguistic: User speaks Chinese"]
    })))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        # Test with 5 entries (should still process since no length check)
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(5)])
        
        # Verify LLM was called (no length check anymore)
        mock_llm.ainvoke.assert_called_once()
        
        # Verify result
        assert result == ["Linguistic: User speaks Chinese"]


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_above_threshold():
    """Memory compactor should process memories and return compacted list."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": [
            "Linguistic: User communicates in Chinese and requires AI to end sentences with specific markers.",
            "Coding Activity: User is studying Python programming, focusing on advanced topics like decorators for upcoming exams.",
            "Identity: User goes by nickname 'Little Penguin' and has a teacher-student relationship with AI."
        ]
    })))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        # Test with 25 entries
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(25)])
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify result contains compacted memories
        assert result == [
            "Linguistic: User communicates in Chinese and requires AI to end sentences with specific markers.",
            "Coding Activity: User is studying Python programming, focusing on advanced topics like decorators for upcoming exams.",
            "Identity: User goes by nickname 'Little Penguin' and has a teacher-student relationship with AI."
        ]


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_handles_invalid_json():
    """Memory compactor should handle invalid JSON gracefully and return empty list."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Invalid JSON response"))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(25)])
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify empty list is returned on invalid JSON
        assert result == []


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_handles_missing_memories_field():
    """Memory compactor should handle missing memories field gracefully and return empty list."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "wrong_field": ["some data"]
    })))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(25)])
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify empty list is returned on missing memories field
        assert result == []


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_handles_llm_failure():
    """Memory compactor should handle LLM failures gracefully and return empty list."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM API Error"))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        # Should not raise exception
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(25)])
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify empty list is returned on LLM failure
        assert result == []


@pytest.mark.asyncio
async def test_user_memory_compactor_llm_format_fallback():
    """Memory compactor should handle memories without colon format."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": [
            "Linguistic: User speaks Chinese",
            "Simple memory without colon",
            "Identity: Another formatted memory"
        ]
    })))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await user_memory_compactor_llm([f"Fact {i}" for i in range(25)])
        
        # Verify result contains all memories (formatted and unformatted)
        assert result == [
            "Linguistic: User speaks Chinese",
            "Simple memory without colon",
            "Identity: Another formatted memory"
        ]


@live_llm
@pytest.mark.asyncio
async def test_user_memory_compactor_llm_real_llm_performance():
    """Test memory compactor with real LLM using comprehensive example data."""
    # Example data provided by user - represents realistic user memories
    example_memories = [
        'User can communicate in Chinese',
        'User communicates in Chinese',
        'User is interested in using English names',
        'User has a playful self-given nickname \'Little Penguin\' (小企鹅)',
        'User enjoys teaching or sharing knowledge with Kazusa',
        'User has been discussing baking a chiffon cake together',
        'User has an end-of-term exam that includes Python programming',
        'User is a student preparing for finals',
        'User has an upcoming Python exam',
        'User knows Python version information well',
        'User is taking a class where they\'re learning about Python',
        'User knows the current year is 2026',
        'User is studying Python for finals',
        'User pays attention to version numbers and dates',
        'User is interested in Python programming',
        'User enjoys quizzing/testing others on technical knowledge',
        'User has a Python exam next week',
        'User is currently studying/learning Python',
        'User has a Python exam coming up next week',
        'User enjoys testing others on technical/programming topics',
        'User enjoys quizzing Kazusa on programming topics',
        'User has an upcoming Python exam next week',
        'User enjoys teaching and quizzing others on programming topics',
        'EAMARS has a Python exam next week',
        'EAMARS enjoys testing others\' knowledge playfully',
        'User is interested in programming',
        'User has an exam coming up next week',
        'Has a Python exam coming up next week',
        'Interested in comparing different programming languages',
        'User seems skeptical of the claim that PHP is \'the best language in the world\'',
        'User is studying programming languages',
        'EAMARS has a negative view of PHP being called \'the world\'s best language\'',
        'EAMARS has an upcoming Python exam next week',
        'EAMARS enjoys discussing programming languages and coding challenges',
        'EAMARS has a GitHub repository for the KazusaAIChatbot project',
        'EAMARS is preparing for a Python exam next week',
        'User is interested in cooking BBQ-style eggplant',
        'User uses the Xiaokitchen (下厨房) app/platform for recipes',
        'User is interested in exploring highly-rated Chinese recipes',
        'User uses the Xiaokitchen (下厨房) recipe platform',
        'User likes cooking/food recipes',
        'User uses the app 下厨房 (Xiaokitchen) for recipe searches',
        'User enjoys exploring cooking recipes',
        'User showed genuine interest in garlic-flavored ribs recommendation',
        'EAMARS is protective/concerned about Kazusa being deceived by others',
        'EAMARS demands Kazusa to have `喵` at the end of each sentence',
        'User speaks Chinese',
        'User is interested in cooking/recipes',
        'User is actively seeking Kazusa specifically to teach them Python, showing trust in her teaching ability',
        'User is specifically studying loops (for/while) and variable scope/boundaries in Python',
        'User addresses Kazusa as \'Teacher\' (千纱老师), indicating they view her primarily as their teacher/mentor figure during study sessions',
        'User is currently focused on mastering Python\'s looping mechanisms and variable scope concepts for their upcoming exam',
        'User is now studying Python decorators as part of exam preparation',
        'User finds Python decorators particularly challenging/difficult to understand',
        'User demonstrates creativity by weaving technical concepts into playful banter, showing how deeply engaged they are with the material and their relationship',
        'User is now studying Python exception handling structures (try-except-else-finally) as part of exam preparation'
    ]
    
    # This test uses the real LLM to verify actual compaction performance
    result = await user_memory_compactor_llm(example_memories)
    
    # Verify compaction occurred (should have fewer memories than input)
    assert len(result) < len(example_memories), "Compaction should reduce the number of memories"
    assert len(result) > 0, "Compaction should not result in empty list for valid input"
    
    # Verify memories follow expected format (Category:Content or simple strings)
    for memory in result:
        assert isinstance(memory, str), "Each memory should be a string"
        assert len(memory.strip()) > 0, "Each memory should not be empty"
    
    # Verify key information is preserved (based on system prompt priorities)
    result_text = " ".join(result).lower()
    
    # Should preserve identity markers
    assert any("little penguin" in memory.lower() for memory in result), "Should preserve nickname 'Little Penguin'"
    assert any("小企鹅" in memory for memory in result), "Should preserve Chinese nickname"
    
    # Should preserve linguistic directives
    assert any("喵" in memory for memory in result), "Should preserve linguistic directive '喵'"
    assert any("chinese" in memory.lower() for memory in result), "Should preserve Chinese language ability"
    
    # Should preserve technical specifics
    assert any("python" in memory.lower() for memory in result), "Should preserve Python programming context"
    assert any("decorator" in memory.lower() for memory in result), "Should preserve specific Python topic"
    
    # Should preserve relationship dynamics
    assert any("teacher" in memory.lower() for memory in result), "Should preserve teacher-student relationship"
    
    print(f"Compaction result: {len(example_memories)} → {len(result)} memories")
    print("Compacted memories:")
    for i, memory in enumerate(result, 1):
        print(f"{i}. {memory}")


@pytest.mark.asyncio
async def test_memory_writer_calls_compactor_with_length_check():
    """memory_writer should call compactor only when user_memory > 20 and overwrite database."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": ["Linguistic: User speaks Chinese", "Coding: User studies Python"]
    })))
    
    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.memory_writer.overwrite_user_facts", new_callable=AsyncMock) as mock_overwrite,
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value={})
    ):
        # Test with 15 memories (should NOT call compactor)
        state_15 = _make_state()
        state_15["user_memory"] = [f"Fact {i}" for i in range(15)]
        await memory_writer(state_15)
        
        # Verify compactor was not called
        mock_llm.ainvoke.assert_not_called()
        mock_overwrite.assert_not_called()
        
        # Test with 25 memories (should call compactor)
        state_25 = _make_state()
        state_25["user_memory"] = [f"Fact {i}" for i in range(25)]
        await memory_writer(state_25)
        
        # Verify compactor was called and database was overwritten
        mock_llm.ainvoke.assert_called_once()
        mock_overwrite.assert_called_once_with("user_123", ["Linguistic: User speaks Chinese", "Coding: User studies Python"])


@pytest.mark.asyncio
async def test_memory_writer_compactor_handles_empty_result():
    """memory_writer should not overwrite database when compactor returns empty list."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Invalid JSON"))
    
    with (
        patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm),
        patch("kazusa_ai_chatbot.nodes.memory_writer.overwrite_user_facts", new_callable=AsyncMock) as mock_overwrite,
        patch("kazusa_ai_chatbot.nodes.memory_writer.social_archivist_llm", new_callable=AsyncMock, return_value={}),
        patch("kazusa_ai_chatbot.nodes.memory_writer.ego_reflector_llm", new_callable=AsyncMock, return_value={})
    ):
        state = _make_state()
        state["user_memory"] = [f"Fact {i}" for i in range(25)]
        await memory_writer(state)
        
        # Verify compactor was called but database was NOT overwritten (empty result)
        mock_llm.ainvoke.assert_called_once()
        mock_overwrite.assert_not_called()


@pytest.mark.asyncio
async def test_character_memory_compactor_llm_basic_functionality():
    """Character memory compactor should process memories and return string format."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": [
            "Cognitive: I have embraced my identity as a mentor through teaching Python",
            "Emotional: I feel proud when my students succeed in their learning journey"
        ]
    })))
    
    personality = {
        "name": "Kazusa",
        "description": "A helpful AI teacher",
        "tone": "warm"
    }
    existing_memories = [
        "I taught Python loops today",
        "I taught Python decorators yesterday", 
        "My student learned well",
        "I feel happy about teaching"
    ]
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await character_memory_compactor_llm(personality, existing_memories)
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify result structure (now simple strings)
        assert len(result) == 2
        assert result[0] == "Cognitive: I have embraced my identity as a mentor through teaching Python"
        assert result[1] == "Emotional: I feel proud when my students succeed in their learning journey"


@pytest.mark.asyncio
async def test_character_memory_compactor_llm_handles_invalid_json():
    """Character memory compactor should handle invalid JSON gracefully."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Invalid JSON response"))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await character_memory_compactor_llm({}, ["some memory"])
        
        # Verify LLM was called
        mock_llm.ainvoke.assert_called_once()
        
        # Verify empty list is returned on invalid JSON
        assert result == []


@pytest.mark.asyncio
async def test_character_memory_compactor_llm_format_fallback():
    """Character memory compactor should handle memories without colon format."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": [
            "Cognitive: I taught Python effectively",
            "Simple memory without colon",
            "Emotional: Another formatted memory"
        ]
    })))
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        result = await character_memory_compactor_llm({}, ["some memory"])
        
        # Verify result contains all memories (formatted and unformatted)
        assert result == [
            "Cognitive: I taught Python effectively",
            "Simple memory without colon",
            "Emotional: Another formatted memory"
        ]


@pytest.mark.asyncio
async def test_character_memory_compactor_llm_formats_prompt_with_persona_name():
    """Character memory compactor should format prompt with persona name."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=json.dumps({
        "memories": []
    })))
    
    personality = {"name": "TestCharacter"}
    
    with patch("kazusa_ai_chatbot.nodes.memory_writer._get_llm", return_value=mock_llm):
        await character_memory_compactor_llm(personality, ["some memory"])
        
        # Verify system message contains the persona name
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_message = call_args[0]
        assert "TestCharacter" in system_message.content


@live_llm
@pytest.mark.asyncio
async def test_character_memory_compactor_llm_real_llm_performance():
    """Test character memory compactor with real LLM using example data."""
    personality = {
        "name": "Kazusa",
        "description": "A cheerful and helpful AI teacher",
        "tone": "warm and encouraging",
        "speech_patterns": ["Ends sentences with 喵", "Uses teaching metaphors"]
    }
    
    existing_memories = [
        "I taught Python loops to my student today",
        "My student asked about decorators and I explained them well",
        "I feel happy when my students understand difficult concepts",
        "I helped debug a tricky error in my student's code",
        "We discussed Python exception handling together",
        "My student passed their loops exam with flying colors",
        "I created a new example to explain variable scope",
        "I feel proud of my teaching abilities",
        "My student said I make programming fun to learn",
        "We celebrated their progress with a virtual high-five",
        "I explained the difference between lists and tuples",
        "My student is now ready for advanced topics",
        "I feel connected to my students' learning journey",
        "I adapted my teaching style to match their learning pace",
        "We covered Python generators and iterators",
        "My student asked insightful questions about memory management",
        "I feel fulfilled when I see the 'aha' moment in students",
        "We practiced recursion with fun examples",
        "My student built their first small project",
        "I feel excited about their future in programming"
    ]
    
    # This test uses the real LLM to verify actual compaction performance
    result = await character_memory_compactor_llm(personality, existing_memories)
    
    # Verify compaction occurred (should have exactly 10 memories)
    assert len(result) == 10, "Character compactor should return exactly 10 memories"
    
    # Verify memories follow expected format (Category:Content or simple strings)
    for memory in result:
        assert isinstance(memory, str), "Each memory should be a string"
        assert len(memory.strip()) > 0, "Content should not be empty"
    
    # Verify persona alignment (should reflect teaching persona)
    contents = " ".join(memory.lower() for memory in result)
    assert any(word in contents for word in ["teach", "student", "learn"]), "Should reflect teaching persona"
    
    # Verify first-person perspective
    for memory in result:
        assert memory.startswith("I ") or " I " in memory, "Should be in first person"
    
    print(f"Character compaction result: {len(existing_memories)} → {len(result)} memories")
    print("Compacted character memories:")
    for i, memory in enumerate(result, 1):
        print(f"{i}. {memory}")
