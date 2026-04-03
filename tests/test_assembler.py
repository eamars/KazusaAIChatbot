"""Tests for Stage 5 — Context Assembler."""

from kazusa_ai_chatbot.nodes.assembler import (
    UNIVERSAL_RULES,
    _build_affinity_block,
    _build_character_state_block,
    _build_memory_block,
    _build_personality_block,
    _build_rag_block,
    assembler,
)


class TestBuildPersonalityBlock:
    def test_empty_personality(self):
        result = _build_personality_block({})
        assert "helpful role-play character" in result

    def test_with_name(self):
        result = _build_personality_block({"name": "Zara"})
        assert "You are Zara." in result

    def test_with_all_fields(self, sample_personality):
        result = _build_personality_block(sample_personality)
        assert "Zara" in result
        assert "sardonic" in result
        assert "Military jargon" in result
        assert "Dawnhollow" in result

    def test_extra_fields_included(self):
        result = _build_personality_block({"name": "Zara", "custom_field": "value"})
        assert "custom_field" in result

    def test_underscore_keys_skipped(self):
        result = _build_personality_block({"name": "Zara", "_reference": {"height": 155}})
        assert "_reference" not in result
        assert "155" not in result

    def test_gender_age_birthday(self):
        result = _build_personality_block(
            {"name": "Kazusa", "gender": "女", "age": 15, "birthday": "8月5日"}
        )
        assert "Gender: 女" in result
        assert "Age: 15" in result
        assert "Birthday: 8月5日" in result


class TestBuildRagBlock:
    def test_empty_results(self):
        assert _build_rag_block([]) == ""

    def test_with_results(self, sample_rag_results):
        result = _build_rag_block(sample_rag_results)
        assert "Relevant world context" in result
        assert "shadow wolves" in result
        assert "Captain Voss" in result
        assert "lore/events" in result


class TestBuildMemoryBlock:
    def test_empty_memory(self):
        assert _build_memory_block([]) == ""

    def test_with_facts(self):
        result = _build_memory_block(["User goes by Commander", "Allied with Northern Faction"])
        assert "About this user" in result
        assert "Commander" in result
        assert "Northern Faction" in result


class TestBuildCharacterStateBlock:
    def test_empty_state(self):
        assert _build_character_state_block({}) == ""

    def test_with_mood_and_tone(self, sample_character_state):
        result = _build_character_state_block(sample_character_state)
        assert "current character state" in result.lower() or "character state" in result.lower()
        assert "alert" in result
        assert "guarded" in result

    def test_with_recent_events(self, sample_character_state):
        result = _build_character_state_block(sample_character_state)
        assert "Shadow wolves" in result

    def test_mood_only(self):
        result = _build_character_state_block({"mood": "happy"})
        assert "happy" in result


class TestBuildAffinityBlock:
    def test_hostile(self):
        result = _build_affinity_block(100)
        assert "Hostile" in result
        assert "contempt" in result or "dismissive" in result

    def test_cold(self):
        result = _build_affinity_block(300)
        assert "Cold" in result

    def test_neutral(self):
        result = _build_affinity_block(500)
        assert "Neutral" in result

    def test_friendly(self):
        result = _build_affinity_block(700)
        assert "Friendly" in result

    def test_devoted(self):
        result = _build_affinity_block(950)
        assert "Devoted" in result

    def test_boundary_values(self):
        assert "Hostile" in _build_affinity_block(200)
        assert "Cold" in _build_affinity_block(400)
        assert "Neutral" in _build_affinity_block(600)
        assert "Friendly" in _build_affinity_block(800)
        assert "Devoted" in _build_affinity_block(801)


class TestAssembler:
    def test_assembler_produces_messages(self, assembled_state):
        result = assembler(assembled_state)
        messages = result["llm_messages"]
        assert len(messages) >= 2  # system + at least current user message
        assert messages[0].type == "system"
        assert messages[-1].type == "human"

    def test_system_prompt_contains_personality(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "Zara" in system

    def test_system_prompt_contains_rag(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "shadow wolves" in system

    def test_system_prompt_contains_user_memory(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "Commander" in system

    def test_system_prompt_contains_character_state(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "alert" in system
        assert "guarded" in system

    def test_system_prompt_contains_affinity(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "Affinity" in system
        assert "Neutral" in system  # default 500 → Neutral

    def test_system_prompt_contains_universal_rules(self, assembled_state):
        result = assembler(assembled_state)
        system = result["llm_messages"][0].content
        assert "[Rules]" in system
        for rule in UNIVERSAL_RULES:
            assert rule in system

    def test_current_message_at_end(self, assembled_state):
        result = assembler(assembled_state)
        last_msg = result["llm_messages"][-1]
        assert last_msg.type == "human"
        assert "northern gate" in last_msg.content

    def test_history_included(self, assembled_state):
        result = assembler(assembled_state)
        messages = result["llm_messages"]
        # system + 2 history messages + current = 4
        assert len(messages) == 4

    def test_user_name_in_message(self, assembled_state):
        result = assembler(assembled_state)
        last_msg = result["llm_messages"][-1]
        assert "TestUser" in last_msg.content

    def test_no_rag_no_memory(self, base_state):
        state = {
            **base_state,
            "rag_results": [],
            "conversation_history": [],
            "user_memory": [],
            "character_state": {},
        }
        result = assembler(state)
        messages = result["llm_messages"]
        assert len(messages) == 2  # system + current user message

    def test_preserves_state_fields(self, assembled_state):
        result = assembler(assembled_state)
        assert result["user_id"] == "user_123"
        assert result["channel_id"] == "chan_456"
