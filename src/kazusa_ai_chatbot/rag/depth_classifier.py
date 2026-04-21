"""Input depth classifier — SHALLOW vs. DEEP.

Two depth layers map naturally to the two pipeline phases:
  * SHALLOW → cache only; user_image + character_image come from the profile
  * DEEP    → input_context_rag_dispatcher + optional external_rag_dispatcher

Classification is embedding-first (fast path, multilingual) with an LLM
fallback for ambiguous inputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.db import get_text_embedding
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output

logger = logging.getLogger(__name__)


# ── Depth constants ────────────────────────────────────────────────

SHALLOW = "SHALLOW"
DEEP = "DEEP"

SHALLOW_DISPATCHERS: list[str] = []
DEEP_DISPATCHERS = ["input_context_rag", "external_rag"]


# ── Thresholds ─────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.60


# ── Keyword sets (Chinese + English, enumerated for embedding centroid) ──

SHALLOW_KEYWORDS: list[str] = [
    # English — simple greetings, direct yes/no, basic facts
    "hello",
    "hi",
    "hey",
    "good morning",
    "good night",
    "thanks",
    "thank you",
    "what is your name",
    "who are you",
    "are you here",
    "you there",
    "what time is it",
    "what day is it",
    "how are you",
    "do you like",
    "what color",
    "what is",
    "yes or no",
    "right",
    "okay",
    # Chinese — simple greetings, direct acknowledgments
    "你好",
    "你在么",
    "你在吗",
    "早上好",
    "晚安",
    "谢谢",
    "好的",
    "明白",
    "知道了",
    "你叫什么",
    "你是谁",
    "你几岁",
    "你喜欢什么颜色",
    "是吗",
    "对吗",
    "好不好",
    "现在几点",
]

DEEP_KEYWORDS: list[str] = [
    # English — past experiences, relationships, reasoning, contradictions
    "remember when",
    "i told you",
    "i told you about",
    "i visited",
    "i got back from",
    "i love the",
    "what did i tell you",
    "do you remember",
    "do i like",
    "what kind of",
    "what have i",
    "compared to",
    "but you said",
    "you promised",
    "why did",
    "why are you",
    "how do you feel",
    "what about me",
    "remember my",
    "last time",
    "before",
    "what is my favorite",
    "do you remember my",
    # Chinese — past conversations, personal facts, emotional context
    "你还记得吗",
    "记得吗",
    "我之前",
    "我以前",
    "我去过",
    "我刚从",
    "我访问过",
    "我喜欢",
    "我喜欢的",
    "我告诉你",
    "我说过",
    "对比一下",
    "但你说",
    "你答应",
    "你承诺",
    "为什么",
    "你怎么感觉",
    "你对我",
    "我的爱好",
    "我最喜欢",
    "关于我",
    "上次",
]


TIME_SENSITIVE_TERMS: tuple[str, ...] = (
    "今天",
    "今日",
    "现在",
    "当前",
    "实时",
    "最新",
    "刚刚",
    "today",
    "current",
    "latest",
    "real-time",
    "right now",
    "now",
)

EXTERNAL_INFO_TERMS: tuple[str, ...] = (
    "天气",
    "新闻",
    "股价",
    "汇率",
    "比分",
    "路况",
    "航班",
    "油价",
    "气温",
    "weather",
    "news",
    "stock",
    "price",
    "traffic",
    "flight",
    "temperature",
)



# ── Module-level lazy state ────────────────────────────────────────

_shallow_centroid: list[float] | None = None
_deep_centroid: list[float] | None = None
_shallow_centroid_norm: float = 0.0
_deep_centroid_norm: float = 0.0
_centroid_lock = asyncio.Lock()

_depth_classifier_llm = None


def _get_llm():
    """Return the module-level LLM instance, initialising it on first call."""
    global _depth_classifier_llm
    if _depth_classifier_llm is None:
        _depth_classifier_llm = get_llm(temperature=0.0, top_p=1.0)
    return _depth_classifier_llm


# ── Vector helpers ─────────────────────────────────────────────────


def _vec_norm(v: list[float]) -> float:
    """Return the L2 norm of vector ``v``."""
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float], *, a_norm: float | None = None, b_norm: float | None = None) -> float:
    """Return cosine similarity in [0, 1] between vectors ``a`` and ``b``.

    Args:
        a: First vector.
        b: Second vector — must have the same length as ``a``.
        a_norm: Pre-computed L2 norm of ``a``; computed here if not provided.
        b_norm: Pre-computed L2 norm of ``b``; computed here if not provided.

    Returns:
        Cosine similarity, or 0.0 if either vector is empty or all-zero.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    if a_norm is None:
        a_norm = _vec_norm(a)
    if b_norm is None:
        b_norm = _vec_norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return dot / (a_norm * b_norm)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return the element-wise mean of a list of equal-length float vectors.

    Args:
        vectors: Non-empty list of vectors that all share the same dimension.

    Returns:
        A single vector whose each element is the mean across all inputs.
        Returns an empty list if ``vectors`` is empty.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    total = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            total[i] += x
    n = len(vectors)
    return [t / n for t in total]


def _looks_time_sensitive_external_query(user_input: str, user_topic: str) -> bool:
    """Return True when the input obviously asks for live external information.

    Args:
        user_input: Raw user message.
        user_topic: Optional topic label derived upstream.

    Returns:
        True when the combined text contains both a time-sensitive signal and an
        external-information signal such as weather or news.
    """
    combined = f"{user_input}\n{user_topic}".lower()
    has_time_signal = any(term in combined for term in TIME_SENSITIVE_TERMS)
    if not has_time_signal:
        return False
    return any(term in combined for term in EXTERNAL_INFO_TERMS)


async def _ensure_centroids() -> None:
    """Pre-compute SHALLOW and DEEP keyword centroids once per process.

    Acquires ``_centroid_lock`` to guard concurrent first-callers. Subsequent
    calls return immediately without acquiring the lock.
    """
    global _shallow_centroid, _deep_centroid, _shallow_centroid_norm, _deep_centroid_norm
    if _shallow_centroid is not None and _deep_centroid is not None:
        return
    async with _centroid_lock:
        if _shallow_centroid is not None and _deep_centroid is not None:
            return
        logger.info("Pre-computing depth classifier keyword centroids (SHALLOW=%d, DEEP=%d)",
                    len(SHALLOW_KEYWORDS), len(DEEP_KEYWORDS))
        shallow_vecs = await asyncio.gather(*(get_text_embedding(k) for k in SHALLOW_KEYWORDS))
        deep_vecs = await asyncio.gather(*(get_text_embedding(k) for k in DEEP_KEYWORDS))
        _shallow_centroid = _mean_vector(shallow_vecs)
        _deep_centroid = _mean_vector(deep_vecs)
        _shallow_centroid_norm = _vec_norm(_shallow_centroid)
        _deep_centroid_norm = _vec_norm(_deep_centroid)


# ── Public classifier ──────────────────────────────────────────────


class InputDepthClassifier:
    """Classify user input into SHALLOW or DEEP retrieval depth.

    Uses an embedding-based fast path (cosine similarity against pre-computed
    keyword centroids) and an LLM fallback for ambiguous inputs.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        """Initialise the classifier with configurable thresholds.

        Args:
            similarity_threshold: Minimum cosine similarity for the fast path
                to commit to a SHALLOW or DEEP label. Inputs below this threshold
                fall through to the LLM.
        """
        self._threshold = similarity_threshold

    async def classify(
        self,
        *,
        user_input: str,
        user_topic: str = "",
        affinity: int = 500,
    ) -> dict[str, Any]:
        """Classify a user message into SHALLOW or DEEP retrieval depth.

        Tries the embedding fast path first. Falls back to LLM on ambiguity or
        embedding failure. Defaults to DEEP if all paths fail.

        Args:
            user_input: The raw user message (Chinese or English).
            user_topic: Optional topic label for the current conversation thread.
            affinity: Current affinity score (0–1000) between user and bot.

        Returns:
            Dict with keys ``depth`` (``"SHALLOW"`` or ``"DEEP"``),
            ``trigger_dispatchers`` (list of dispatcher names), ``confidence``
            (float 0–1), and ``reasoning`` (one-sentence explanation).
        """
        if _looks_time_sensitive_external_query(user_input, user_topic):
            return self._result(
                DEEP,
                confidence=1.0,
                reasoning="rule→DEEP: live external-information query (time-sensitive + external domain)",
            )

        # Fast path: embedding cosine similarity against centroids
        try:
            await _ensure_centroids()
            input_emb = await get_text_embedding(user_input)
        except OpenAIError:
            logger.exception("Fast-path embedding classification failed — falling back to LLM")
        else:
            input_norm = _vec_norm(input_emb)
            sim_shallow = _cosine_similarity(input_emb, _shallow_centroid, a_norm=input_norm, b_norm=_shallow_centroid_norm)
            sim_deep = _cosine_similarity(input_emb, _deep_centroid, a_norm=input_norm, b_norm=_deep_centroid_norm)
            logger.debug("depth_classifier fast-path: sim_shallow=%.3f, sim_deep=%.3f", sim_shallow, sim_deep)

            if sim_shallow >= self._threshold and sim_shallow > sim_deep:
                return self._result(SHALLOW, confidence=sim_shallow,
                                    reasoning=f"embedding→SHALLOW (sim={sim_shallow:.3f})")
            if sim_deep >= self._threshold and sim_deep > sim_shallow:
                return self._result(DEEP, confidence=sim_deep,
                                    reasoning=f"embedding→DEEP (sim={sim_deep:.3f})")

        # Fallback path: LLM
        try:
            depth, reasoning = await self._llm_classify(user_input, user_topic, affinity)
        except OpenAIError:
            logger.exception("LLM fallback failed — defaulting to DEEP")
            return self._result(DEEP, confidence=0.0, reasoning="all classification paths failed; default DEEP")
        return self._result(depth, confidence=0.5, reasoning=f"llm→{depth}: {reasoning}")

    # ── internals ──────────────────────────────────────────────

    @staticmethod
    def _result(depth: str, *, confidence: float, reasoning: str) -> dict[str, Any]:
        """Build the standard classifier result dict.

        Args:
            depth: ``"SHALLOW"`` or ``"DEEP"``.
            confidence: Float in [0, 1] indicating classification certainty.
            reasoning: Human-readable explanation of the decision.

        Returns:
            Dict with ``depth``, ``trigger_dispatchers``, ``confidence``, and
            ``reasoning``.
        """
        dispatchers = SHALLOW_DISPATCHERS if depth == SHALLOW else DEEP_DISPATCHERS
        return {
            "depth": depth,
            "trigger_dispatchers": list(dispatchers),
            "confidence": float(confidence),
            "reasoning": reasoning,
        }


    # ── LLM system prompt (fallback path) ──────────────────────────────
    # Note: affinity threshold is checked deterministically before LLM is called,
    # so the LLM only receives inputs with affinity >= threshold.

    _DEPTH_CLASSIFIER_SYSTEM_PROMPT = """\
You are a depth classifier for a bilingual (Chinese/English) roleplay chatbot.
Your job is to label each user input as either SHALLOW or DEEP.

# Input format
You will receive a JSON object with these fields:
- user_input (string): the user's message, may be in Chinese or English
- user_topic (string): topic category derived from the conversation
- affinity (integer): relationship score 0–1000 between user and bot

# Output format (strict JSON, no extra keys, no markdown fence)
{
    "depth": "SHALLOW" | "DEEP",
    "reasoning": "one sentence explaining why"
}

# Classification rules
- SHALLOW: the input is a simple factual question, greeting, preference check,
or yes/no query that requires no deep memory retrieval — cache or the basic
user profile is sufficient.
- DEEP: the input references past events, emotional context, asks "why" about
behaviour, involves temporal reasoning, or asks about contradictions /
promises — full memory search is required.
- Input language may be Chinese or English — classify based on meaning, not language.
- When in doubt, prefer DEEP (over-retrieval is safer than missed context).
"""
    async def _llm_classify(self, user_input: str, user_topic: str, affinity: int) -> tuple[str, str]:
        """Call the LLM to classify ambiguous inputs.

        Args:
            user_input: The raw user message.
            user_topic: Topic label for context.
            affinity: Current affinity score (already validated to be >= threshold).

        Returns:
            Tuple of (``"SHALLOW"`` or ``"DEEP"``, reasoning string).
        """
        llm = _get_llm()
        system_prompt = SystemMessage(content=self._DEPTH_CLASSIFIER_SYSTEM_PROMPT)
        human_message = HumanMessage(content=json.dumps({
            "user_input": user_input,
            "user_topic": user_topic,
            "affinity": affinity,
        }, ensure_ascii=False))
        response = await llm.ainvoke([system_prompt, human_message])
        parsed = parse_llm_json_output(response.content)
        depth = str(parsed.get("depth", "")).upper().strip()
        reasoning = str(parsed.get("reasoning", "")).strip() or "no reasoning returned"
        if depth not in {SHALLOW, DEEP}:
            return DEEP, f"unparsable LLM output, defaulted DEEP (raw={parsed})"
        return depth, reasoning


# ── Standalone test harness ────────────────────────────────────────


async def test_main() -> None:
    """Run a handful of classification cases against a live embedding + LLM backend."""
    logging.basicConfig(level=logging.INFO)

    classifier = InputDepthClassifier()

    cases = [
        # Expected: SHALLOW (EN greeting / preference)
        {"user_input": "What's your favorite colour?", "user_topic": "chitchat", "affinity": 700},
        # Expected: SHALLOW (ZH preference)
        {"user_input": "你喜欢什么颜色？", "user_topic": "闲聊", "affinity": 700},
        # Expected: DEEP (EN temporal + emotional history)
        {"user_input": "Remember when you promised to help me last week? Why didn't you?",
         "user_topic": "关系", "affinity": 700},
        # Expected: DEEP (ambiguous / nonsense — tests LLM fallback path OR default)
        {"user_input": "xyzabc !@#", "user_topic": "", "affinity": 700},
    ]

    for i, case in enumerate(cases, 1):
        print(f"\n[case {i}] input={case['user_input']!r} affinity={case['affinity']}")
        result = await classifier.classify(**case)
        print(json.dumps(result, indent=2, ensure_ascii=False))


async def test_main2() -> None:
    """Test cases designed to trigger the LLM fallback path.
    
    These inputs are deliberately ambiguous (low embedding similarity) to force
    the classifier to use the LLM instead of the fast embedding path.
    """
    logging.basicConfig(level=logging.INFO)

    classifier = InputDepthClassifier()

    # Cases designed to trigger LLM path (ambiguous / low embedding similarity)
    cases = [
        # Ambiguous: Could be SHALLOW (simple greeting) or DEEP (context-dependent)
        {"user_input": "Is it really you?", "user_topic": "relationship", "affinity": 700},
        # Ambiguous: Generic phrase, unclear intent
        {"user_input": "怎么样", "user_topic": "chitchat", "affinity": 700},
        # Ambiguous: Mixed signals (greeting + emotional)
        {"user_input": "Hello, how have things been since we last talked?",
         "user_topic": "relationship", "affinity": 700},
        # Ambiguous: Deliberate LLM prompt (not in keyword list)
        {"user_input": "Tell me something interesting", "user_topic": "creative", "affinity": 700},
        # Ambiguous: Abstract question
        {"user_input": "What does loyalty mean to you?", "user_topic": "philosophy", "affinity": 700},
        # Ambiguous: Chinese mixed signals
        {"user_input": "你最近怎么样？我们聊好久没有聊天了",
         "user_topic": "relationship", "affinity": 700},
    ]

    print("\n" + "="*70)
    print("TEST_MAIN2: LLM Fallback Path Cases")
    print("="*70)
    
    for i, case in enumerate(cases, 1):
        print(f"\n[case {i}] input={case['user_input']!r} affinity={case['affinity']}")
        print(f"         topic={case['user_topic']!r}")
        result = await classifier.classify(**case)
        # Highlight that LLM path was used (reasoning contains "llm→")
        is_llm_path = "llm→" in result.get("reasoning", "")
        print(f"         [{'LLM PATH ✓' if is_llm_path else 'fast path'}]")
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(test_main2())
