import pytest
from kazusa_ai_chatbot.utils import build_affinity_block
from kazusa_ai_chatbot.config import AFFINITY_DEFAULT


def test_trim_history_dict():
    """Test the trim_history_dict function."""
    from kazusa_ai_chatbot.utils import trim_history_dict
    
    history = [
        {"name": "Alice", "user_id": "user_123", "content": "Hello", "role": "user"},
        {"name": "Bob", "user_id": "user_456", "content": "Hi", "role": "user"},
    ]
    
    trimmed = trim_history_dict(history)
    assert len(trimmed) == 2
    assert trimmed[0]["name"] == "Alice"
    assert trimmed[0]["user_id"] == "user_123"
    assert trimmed[0]["content"] == "Hello"
    assert trimmed[0]["role"] == "user"
    assert trimmed[1]["name"] == "Bob"
    assert trimmed[1]["user_id"] == "user_456"
    assert trimmed[1]["content"] == "Hi"
    assert trimmed[1]["role"] == "user"


class TestBuildAffinityBlock:
    def test_hostile(self):
        result = build_affinity_block(100)
        assert result["level"] == "Scornful"
        assert "contempt" in result["instruction"] or "dismissive" in result["instruction"]

    def test_cold(self):
        result = build_affinity_block(300)
        assert result["level"] == "Reserved"
        assert "brief" in result["instruction"] or "professional" in result["instruction"]

    def test_neutral(self):
        result = build_affinity_block(AFFINITY_DEFAULT)
        assert result["level"] == "Antagonistic"

    def test_friendly(self):
        result = build_affinity_block(700)
        assert result["level"] == "Warm"
        assert "warmth" in result["instruction"] or "enthusiasm" in result["instruction"]

    def test_devoted(self):
        result = build_affinity_block(900)
        assert result["level"] == "Protective"
        assert "protective" in result["instruction"] or "loyalty" in result["instruction"]
