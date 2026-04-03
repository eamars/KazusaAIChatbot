"""Stage 2 — Rule-Based Router (no LLM).

Decides which retrieval steps are needed based on simple heuristics.
No model call — pure keyword / pattern matching.

Supports English, Chinese (simplified/traditional), and Japanese.
For unrecognised languages the safe default is retrieve_rag=True.
"""

from __future__ import annotations

import re
import unicodedata

from kazusa_ai_chatbot.state import BotState

# ── CJK detection ──────────────────────────────────────────────────
_CJK_RANGES = re.compile(
    r"[\u4e00-\u9fff"         # CJK Unified Ideographs
    r"\u3400-\u4dbf"          # CJK Extension A
    r"\u3040-\u309f"          # Hiragana
    r"\u30a0-\u30ff"          # Katakana
    r"\uac00-\ud7af]",        # Hangul
)

# ── English patterns ───────────────────────────────────────────────
_EN_QUESTION_WORDS = re.compile(
    r"\b(what|where|when|who|why|how|tell me|explain|describe|remind me)\b",
    re.IGNORECASE,
)

_EN_CASUAL = re.compile(
    r"^(hi|hello|hey|yo|sup|thanks|ty|ok|okay|lol|lmao|haha|bye|gn|gm)\b",
    re.IGNORECASE,
)

# ── Chinese patterns ──────────────────────────────────────────────
_ZH_QUESTION_WORDS = re.compile(
    r"(什么|什麼|哪里|哪裡|哪儿|哪個|谁|誰|为什么|為什麼|怎么|怎麼"
    r"|如何|多少|几|幾|告诉我|告訴我|解释|解釋|描述|提醒我"
    r"|在哪|是谁|是誰|吗|嗎|呢|吧|么|嘛)",
)

_ZH_CASUAL = re.compile(
    r"^(嗯|好的|行|哈哈|哈哈哈|拜拜|再见|再見|ok|OK)$",
)

# ── Japanese patterns ──────────────────────────────────────────────
_JA_QUESTION_WORDS = re.compile(
    r"(何|なに|なん|どこ|どれ|どう|いつ|だれ|誰|なぜ|どうして"
    r"|教えて|説明して|描写して|思い出させて"
    r"|ですか|ますか|のか|かな|だろう)",
)

_JA_CASUAL = re.compile(
    r"^(うん|はい|笑|ワロタ|じゃね|バイバイ|おk|おけ)$",
)


def _has_cjk(text: str) -> bool:
    """Return True if the text contains any CJK characters."""
    return bool(_CJK_RANGES.search(text))


def _cjk_char_count(text: str) -> int:
    """Count characters that are CJK ideographs/kana (proxy for 'word' count)."""
    return sum(1 for ch in text if _CJK_RANGES.match(ch))


def _is_question(text: str) -> bool:
    """Check if the text looks like a question in any supported language."""
    if _EN_QUESTION_WORDS.search(text):
        return True
    if text.rstrip().endswith("?") or text.rstrip().endswith("？"):
        return True
    if _ZH_QUESTION_WORDS.search(text):
        return True
    if _JA_QUESTION_WORDS.search(text):
        return True
    return False


def _is_casual(text: str) -> bool:
    """Check if the text is a short casual greeting/filler in any supported language."""
    stripped = text.strip()
    if _EN_CASUAL.match(stripped) and len(stripped.split()) <= 4:
        return True
    if _ZH_CASUAL.match(stripped):
        return True
    if _JA_CASUAL.match(stripped):
        return True
    return False


def _is_long_message(text: str) -> bool:
    """Check if the message is 'long enough' to likely reference world context."""
    # For whitespace-delimited languages
    if len(text.split()) >= 6:
        return True
    # For CJK: ~4+ characters is roughly equivalent to a meaningful sentence
    if _cjk_char_count(text) >= 4:
        return True
    return False


def router(state: BotState) -> BotState:
    """Set retrieval flags based on the incoming message."""
    text = state.get("message_text", "")

    # Default: always retrieve memory (cheap DB lookup)
    retrieve_memory = True

    # Decide whether RAG is needed
    if not text:
        retrieve_rag = False
    elif _is_casual(text):
        retrieve_rag = False
    elif _is_question(text):
        retrieve_rag = True
    elif _is_long_message(text):
        retrieve_rag = True
    elif _has_cjk(text):
        # Non-English text that isn't casual and isn't clearly short
        # — safe default is to retrieve context
        retrieve_rag = True
    else:
        retrieve_rag = False

    # Build a cleaned query for RAG (strip filler words, keep substance)
    rag_query = text if retrieve_rag else ""

    return {
        **state,
        "retrieve_rag": retrieve_rag,
        "retrieve_memory": retrieve_memory,
        "rag_query": rag_query,
    }
