from kazusa_ai_chatbot.config import AFFINITY_DEFAULT
from kazusa_ai_chatbot.utils import build_affinity_block, parse_llm_json_output


def test_trim_history_dict():
    """Test the trim_history_dict function."""
    from kazusa_ai_chatbot.utils import trim_history_dict
    
    history = [
        {
            "display_name": "<user A>",
            "platform_user_id": "user_123",
            "global_user_id": "uuid-1",
            "body_text": "Hello",
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t1",
        },
        {
            "display_name": "<user B>",
            "platform_user_id": "user_456",
            "global_user_id": "uuid-2",
            "body_text": "Hi",
            "addressed_to_global_user_ids": ["character"],
            "mentions": [],
            "broadcast": False,
            "role": "user",
            "timestamp": "t2",
        },
    ]
    
    trimmed = trim_history_dict(history)
    assert len(trimmed) == 2
    assert trimmed[0]["name"] == "<user A>"
    assert trimmed[0]["platform_user_id"] == "user_123"
    assert trimmed[0]["body_text"] == "Hello"
    assert "content" not in trimmed[0]
    assert trimmed[0]["role"] == "user"
    assert trimmed[1]["name"] == "<user B>"
    assert trimmed[1]["platform_user_id"] == "user_456"
    assert trimmed[1]["body_text"] == "Hi"
    assert "content" not in trimmed[1]
    assert trimmed[1]["role"] == "user"


def test_parse_llm_json_output_accepts_markdown_fenced_raw_output():
    """Markdown fences are repaired from raw LLM text without escaped wrapping."""
    raw_output = """```json
{
  "continuity": "related_shift",
  "open_loops": ["follow up"]
}
```"""

    result = parse_llm_json_output(raw_output)

    assert result == {
        "continuity": "related_shift",
        "open_loops": ["follow up"],
    }


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
        assert result["level"] == "Neutral"

    def test_friendly(self):
        result = build_affinity_block(700)
        assert result["level"] == "Warm"
        assert "warmth" in result["instruction"] or "enthusiasm" in result["instruction"]

    def test_devoted(self):
        result = build_affinity_block(900)
        assert result["level"] == "Protective"
        assert "protective" in result["instruction"] or "loyalty" in result["instruction"]
