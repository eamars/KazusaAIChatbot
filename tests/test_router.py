"""Tests for Stage 2 — Rule-Based Router."""

from kazusa_ai_chatbot.nodes.router import router


def test_router_enables_rag_for_question(base_state):
    base_state["message_text"] = "What happened at the northern gate?"
    result = router(base_state)
    assert result["retrieve_rag"] is True
    assert result["retrieve_memory"] is True
    assert result["rag_query"] == base_state["message_text"]


def test_router_enables_rag_for_who_question(base_state):
    base_state["message_text"] = "Who is Captain Voss?"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_router_enables_rag_for_tell_me(base_state):
    base_state["message_text"] = "Tell me about the shadow wolves"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_router_skips_rag_for_casual_greeting(base_state):
    base_state["message_text"] = "Hey"
    result = router(base_state)
    assert result["retrieve_rag"] is False
    assert result["rag_query"] == ""


def test_router_skips_rag_for_short_casual(base_state):
    base_state["message_text"] = "Thanks"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_router_skips_rag_for_ok(base_state):
    base_state["message_text"] = "Ok"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_router_enables_rag_for_long_message(base_state):
    base_state["message_text"] = "I was thinking about going to the market today to buy some supplies"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_router_skips_rag_for_short_non_question(base_state):
    base_state["message_text"] = "Nice work"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_router_always_retrieves_memory(base_state):
    base_state["message_text"] = "Hi"
    result = router(base_state)
    assert result["retrieve_memory"] is True


def test_router_empty_message(base_state):
    base_state["message_text"] = ""
    result = router(base_state)
    assert result["retrieve_rag"] is False
    assert result["retrieve_memory"] is True


def test_router_preserves_state(base_state):
    result = router(base_state)
    assert result["user_id"] == "user_123"
    assert result["channel_id"] == "chan_456"

def test_chinese_input(base_state):
    base_state["message_text"] = "你好"
    result = router(base_state)
    assert result["retrieve_rag"] is True
    assert result["retrieve_memory"] is True


# ── Chinese tests ────────────────────────────────────────────────────


def test_zh_question_shenme(base_state):
    base_state["message_text"] = "北门发生了什么事？"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_shei(base_state):
    base_state["message_text"] = "谁是沃斯队长"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_weishenme(base_state):
    base_state["message_text"] = "为什么北门被攻破了"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_zenme(base_state):
    base_state["message_text"] = "怎么回事"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_nali(base_state):
    base_state["message_text"] = "影狼在哪里"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_mark(base_state):
    base_state["message_text"] = "你还好吗？"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_question_traditional(base_state):
    base_state["message_text"] = "那個人是誰"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_tell_me(base_state):
    base_state["message_text"] = "告诉我关于暗影狼的事"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_long_sentence(base_state):
    base_state["message_text"] = "昨天晚上北门外面好像有奇怪的动静"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_zh_casual_haha(base_state):
    base_state["message_text"] = "哈哈"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_zh_casual_ok(base_state):
    base_state["message_text"] = "好的"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_zh_casual_bye(base_state):
    base_state["message_text"] = "再见"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_zh_casual_en(base_state):
    base_state["message_text"] = "嗯"
    result = router(base_state)
    assert result["retrieve_rag"] is False


# ── Japanese tests ───────────────────────────────────────────────────


def test_ja_question_nani(base_state):
    base_state["message_text"] = "何が起きたの？"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_question_doko(base_state):
    base_state["message_text"] = "北の門はどこですか"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_question_dare(base_state):
    base_state["message_text"] = "ヴォス隊長はだれ"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_question_naze(base_state):
    base_state["message_text"] = "なぜ門が破られたのか"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_question_doushite(base_state):
    base_state["message_text"] = "どうして攻撃された"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_tell_me(base_state):
    base_state["message_text"] = "影狼について教えて"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_question_mark(base_state):
    base_state["message_text"] = "大丈夫？"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_greeting_triggers_rag(base_state):
    base_state["message_text"] = "こんにちは"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_ja_casual_un(base_state):
    base_state["message_text"] = "うん"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_ja_casual_hai(base_state):
    base_state["message_text"] = "はい"
    result = router(base_state)
    assert result["retrieve_rag"] is False


def test_ja_casual_warota(base_state):
    base_state["message_text"] = "ワロタ"
    result = router(base_state)
    assert result["retrieve_rag"] is False


# ── Mixed language / edge cases ──────────────────────────────────────


def test_mixed_en_zh(base_state):
    base_state["message_text"] = "What happened at 北门?"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_mixed_en_ja(base_state):
    base_state["message_text"] = "Tell me about 影狼"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_fullwidth_question_mark(base_state):
    base_state["message_text"] = "门还安全吗？"
    result = router(base_state)
    assert result["retrieve_rag"] is True


def test_cjk_short_non_casual(base_state):
    """Short CJK text that isn't in the casual list should still trigger RAG."""
    base_state["message_text"] = "影狼"
    result = router(base_state)
    assert result["retrieve_rag"] is True
