#!/usr/bin/env python
"""
Depth Classifier Keyword Updater — Extract keywords from conversation history.

Usage:
    python update_classifier.py [--sample-size 500] [--output-only]

This script:
1. Queries conversation_history collection in MongoDB
2. Classifies messages as SHALLOW/DEEP based on content patterns
3. Extracts frequent phrases (3-6 words)
4. Deduplicates and balances English/Chinese
5. Updates src/kazusa_ai_chatbot/rag/depth_classifier.py
6. Tests the updated classifier
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Classification signals ──────────────────────────────────────────

SHALLOW_SIGNALS = {
    # English
    "hello": 1.0, "hi": 1.0, "hey": 1.0, "hi there": 1.0,
    "good morning": 1.0, "good night": 1.0, "good evening": 1.0,
    "thanks": 1.0, "thank you": 1.0, "okay": 1.0, "ok": 1.0,
    "yes": 0.8, "no": 0.8, "right": 0.8, "sure": 0.8, "yep": 0.8,
    "what is": 0.7, "how are you": 1.0, "are you": 0.7,
    "what time": 0.9, "what day": 0.9, "what color": 0.8,
    "do you like": 0.7, "is there": 0.6,
    # Chinese
    "你好": 1.0, "你在": 0.9, "你在么": 1.0, "你在吗": 1.0,
    "早上好": 1.0, "晚安": 1.0, "谢谢": 1.0, "好的": 1.0,
    "知道了": 1.0, "明白": 1.0, "好吧": 0.8, "是的": 0.8,
    "现在几点": 0.9, "是吗": 0.7, "对吗": 0.7, "好不好": 0.6,
    "什么颜色": 0.8, "怎么样": 0.6,
}

DEEP_SIGNALS = {
    # English
    "remember": 1.0, "remember when": 1.0, "remember my": 1.0,
    "i told you": 1.0, "you said": 1.0, "you promised": 1.0,
    "you said before": 0.9, "last time": 0.9, "before": 0.7,
    "promised": 1.0, "promised to": 1.0,
    "i visited": 0.9, "i got back": 0.9, "i just got": 0.9,
    "do you remember": 1.0, "do i like": 0.8, "what kind of": 0.7,
    "what did i": 0.9, "what have i": 0.8, "compared to": 0.8,
    "but you": 0.8, "why did": 0.8, "why are you": 0.8,
    "how do you feel": 0.9, "what about me": 0.8,
    "last time we": 0.9, "based on our": 0.8,
    # Chinese
    "你还记得": 1.0, "记得吗": 0.9, "记得": 0.8,
    "我之前": 0.9, "我以前": 0.9, "我去过": 0.9,
    "我告诉你": 1.0, "我说过": 1.0, "我之前告诉": 0.9,
    "你之前": 0.8, "你说过": 0.9, "你答应": 1.0,
    "为什么": 0.8, "你对我": 0.8, "你怎么": 0.6,
    "我的爱好": 1.0, "我最喜欢": 0.9, "关于我": 0.8,
    "上次": 0.8, "之前": 0.6, "以前": 0.6,
    "和...相比": 0.8, "对比": 0.8,
}


# ── Helper functions ────────────────────────────────────────────────

def classify_message(content: str) -> str:
    """Classify a message as SHALLOW, DEEP, or AMBIGUOUS."""
    content_lower = content.lower()

    # Count signal matches
    shallow_score = sum(
        weight for phrase, weight in SHALLOW_SIGNALS.items()
        if phrase in content_lower
    )
    deep_score = sum(
        weight for phrase, weight in DEEP_SIGNALS.items()
        if phrase in content_lower
    )

    if deep_score > shallow_score:
        return "DEEP"
    elif shallow_score > deep_score:
        return "SHALLOW"
    else:
        return "AMBIGUOUS"


def extract_phrases(content: str, phrase_length: int = 3) -> list[str]:
    """Extract 3-6 word phrases from content."""
    # Clean up
    content = re.sub(r"[^\w\s']", " ", content)  # Remove special chars
    words = content.lower().split()
    
    if len(words) < phrase_length:
        return []

    phrases = []
    for i in range(len(words) - phrase_length + 1):
        phrase = " ".join(words[i : i + phrase_length])
        # Skip common stopwords
        if phrase and not all(w in ["a", "the", "is", "to", "in", "of", "and", "or"] for w in words[i : i + phrase_length]):
            phrases.append(phrase)

    return phrases


def deduplicate_and_balance(
    phrases: list[tuple[str, int]],
    min_count: int = 3,
    target_count: int = 35,
) -> list[str]:
    """Deduplicate phrases and ensure English/Chinese balance."""
    # Filter by minimum count
    filtered = [phrase for phrase, count in phrases if count >= min_count]

    # Separate by language
    english = [p for p in filtered if all(ord(c) < 128 for c in p if c.isalpha())]
    chinese = [p for p in filtered if any(ord(c) >= 0x4E00 for c in p)]

    # Balance: 45-55% split
    target_en = int(target_count * 0.5)
    target_zh = target_count - target_en

    english = english[:target_en]
    chinese = chinese[:target_zh]

    return sorted(english) + sorted(chinese)


async def extract_keywords_from_db(
    sample_size: int = 500,
) -> dict[str, list[str]]:
    """Extract keywords from MongoDB conversation history."""
    try:
        from kazusa_ai_chatbot.db import get_db
    except ImportError:
        logger.error("Failed to import kazusa_ai_chatbot.db — ensure environment is set up")
        return {}

    db = await get_db()

    # Query recent conversations
    now = datetime.now(timezone.utc)
    two_weeks_ago = now - timedelta(days=14)

    try:
        conversations = await db["kazusa_bot_core"]["conversation_history"].find(
            {
                "role": "user",
                "timestamp": {"$gte": two_weeks_ago.isoformat()},
            },
            {"content": 1},
        ).to_list(sample_size)
    except Exception as e:
        logger.error(f"Failed to query conversation_history: {e}")
        return {}

    logger.info(f"Extracted {len(conversations)} recent messages")

    # Classify and collect phrases
    shallow_phrases: list[str] = []
    deep_phrases: list[str] = []

    for doc in conversations:
        content = doc.get("content", "").strip()
        if not content or len(content) < 3:
            continue

        depth = classify_message(content)
        phrases = extract_phrases(content)

        if depth == "SHALLOW":
            shallow_phrases.extend(phrases)
        elif depth == "DEEP":
            deep_phrases.extend(phrases)

    # Count frequencies
    shallow_counted = Counter(shallow_phrases).most_common(50)
    deep_counted = Counter(deep_phrases).most_common(60)

    logger.info(f"SHALLOW candidates: {len(shallow_counted)}")
    logger.info(f"DEEP candidates: {len(deep_counted)}")

    # Deduplicate and balance
    shallow_final = deduplicate_and_balance(shallow_counted, min_count=2, target_count=35)
    deep_final = deduplicate_and_balance(deep_counted, min_count=2, target_count=42)

    return {
        "SHALLOW": shallow_final,
        "DEEP": deep_final,
        "sample_size": len(conversations),
        "timestamp": now.isoformat(),
    }


def update_classifier_file(keywords: dict[str, list[str]]) -> bool:
    """Update depth_classifier.py with new keywords."""
    classifier_path = Path(__file__).parent.parent.parent / "src" / "kazusa_ai_chatbot" / "rag" / "depth_classifier.py"

    if not classifier_path.exists():
        logger.error(f"Classifier file not found: {classifier_path}")
        return False

    content = classifier_path.read_text()

    # Generate new keyword lists
    shallow_str = (
        "SHALLOW_KEYWORDS: list[str] = [\n"
        "    # English — simple greetings, direct yes/no, basic facts\n"
    )
    for kw in keywords.get("SHALLOW", []):
        if all(ord(c) < 128 for c in kw if c.isalpha()):
            shallow_str += f'    "{kw}",\n'

    shallow_str += "    # Chinese — simple greetings, direct acknowledgments\n"
    for kw in keywords.get("SHALLOW", []):
        if any(ord(c) >= 0x4E00 for c in kw):
            shallow_str += f'    "{kw}",\n'
    shallow_str += "]\n"

    # Similar for DEEP
    deep_str = (
        "DEEP_KEYWORDS: list[str] = [\n"
        "    # English — past experiences, relationships, reasoning, contradictions\n"
    )
    for kw in keywords.get("DEEP", []):
        if all(ord(c) < 128 for c in kw if c.isalpha()):
            deep_str += f'    "{kw}",\n'

    deep_str += "    # Chinese — past conversations, personal facts, emotional context\n"
    for kw in keywords.get("DEEP", []):
        if any(ord(c) >= 0x4E00 for c in kw):
            deep_str += f'    "{kw}",\n'
    deep_str += "]\n"

    # Replace in file
    import re
    pattern_shallow = r"SHALLOW_KEYWORDS: list\[str\] = \[.*?\]"
    pattern_deep = r"DEEP_KEYWORDS: list\[str\] = \[.*?\]"

    content = re.sub(pattern_shallow, shallow_str, content, flags=re.DOTALL)
    content = re.sub(pattern_deep, deep_str, content, flags=re.DOTALL)

    try:
        classifier_path.write_text(content)
        logger.info(f"✅ Updated {classifier_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to update classifier: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=500, help="Number of messages to analyze")
    parser.add_argument("--output-only", action="store_true", help="Only output, don't update file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("🔍 Extracting keywords from conversation history...")
    keywords = await extract_keywords_from_db(sample_size=args.sample_size)

    if not keywords:
        logger.error("❌ Failed to extract keywords")
        return

    logger.info(f"✅ Extracted keywords:")
    logger.info(f"   SHALLOW: {len(keywords.get('SHALLOW', []))} keywords")
    logger.info(f"   DEEP: {len(keywords.get('DEEP', []))} keywords")

    if args.output_only:
        logger.info("Output-only mode — not updating classifier file")
        print(keywords)
        return

    # Update classifier
    if update_classifier_file(keywords):
        logger.info("✅ Classifier updated successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python -m src.kazusa_ai_chatbot.rag.depth_classifier")
        logger.info("2. Review test results")
        logger.info("3. Commit changes to git")
    else:
        logger.error("❌ Failed to update classifier")


if __name__ == "__main__":
    asyncio.run(main())
