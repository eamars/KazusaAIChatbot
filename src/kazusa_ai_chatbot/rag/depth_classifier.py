"""Input depth classifier — SHALLOW vs. DEEP.

Two depth layers map naturally to the two storage layers:
  * SHALLOW → cache + user_rag only
  * DEEP    → full DB search across all dispatchers

Classification is embedding-first (fast path, multilingual) with an LLM
fallback for ambiguous inputs.  If affinity is low, the classifier always
returns DEEP (the bot should not over-trust shallow heuristics for users it
does not know well yet).
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

SHALLOW_DISPATCHERS = ["user_rag"]
DEEP_DISPATCHERS = ["user_rag", "internal_rag", "external_rag"]


# ── Thresholds ─────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.75
AFFINITY_DEEP_THRESHOLD = 400


# ── Keyword sets (Chinese + English, enumerated for embedding centroid) ──

SHALLOW_KEYWORDS: list[str] = [
    # English
    "what is your name",
    "who are you",
    "how old are you",
    "what is your favorite color",
    "what do you like",
    "do you like",
    "are you",
    "hello",
    "hi there",
    "good morning",
    "good night",
    "thanks",
    "thank you",
    "yes or no",
    "what time is it",
    # Chinese
    "你叫什么名字",
    "你是谁",
    "你几岁",
    "你多大",
    "你喜欢什么颜色",
    "你喜欢什么",
    "你是不是",
    "你好",
    "早上好",
    "晚安",
    "谢谢",
    "好不好",
    "现在几点",
    "什么颜色",
    "是吗",
]

DEEP_KEYWORDS: list[str] = [
    # English
    "why do you always",
    "remember when",
    "last time we",
    "you said before",
    "based on our",
    "how has your feeling changed",
    "what did i tell you",
    "do you remember the promise",
    "why did you",
    "how do you really feel about",
    "compared to before",
    "what have we been through",
    # Chinese
    "你为什么总是",
    "你以前",
    "你说过",
    "上次",
    "为什么",
    "你还记得吗",
    "记得吗",
    "我之前告诉你",
    "你之前说",
    "你对我的感觉有没有变",
    "和以前相比",
    "我们之间",
    "你答应过",
]


# ── LLM system prompt (fallback path) ──────────────────────────────

_DEPTH_CLASSIFIER_SYSTEM_PROMPT = """\
You are a depth classifier for a bilingual (Chinese/English) roleplay chatbot.
Your job is to label each user input as either SHALLOW or DEEP.

# Input format
You will receive a JSON object with these fields:
- user_input (string): the user's message, may be in Chinese or English
- user_topic (string): topic category derived from the conversation
- affinity (integer): relationship score 0–1000 between user and bot

# Output format (strict JSON, no extra keys, no markdown fence)
{{
    "depth": "SHALLOW" | "DEEP",
    "reasoning": "one sentence explaining why"
}}

# Classification rules
- SHALLOW: the input is a simple factual question, greeting, preference check,
  or yes/no query that requires no deep memory retrieval — cache or the basic
  user profile is sufficient.
- DEEP: the input references past events, emotional context, asks "why" about
  behaviour, involves temporal reasoning, or asks about contradictions /
  promises — full memory search is required.
- If `affinity` is strictly less than {affinity_threshold}, output DEEP regardless of content.
- Input language may be Chinese or English — classify based on meaning, not language.
- When in doubt, prefer DEEP (over-retrieval is safer than missed context).
"""


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
    keyword centroids) and an LLM fallback for ambiguous inputs. Low-affinity
    users always get DEEP regardless of input content.
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        affinity_deep_threshold: int = AFFINITY_DEEP_THRESHOLD,
    ) -> None:
        """Initialise the classifier with configurable thresholds.

        Args:
            similarity_threshold: Minimum cosine similarity for the fast path
                to commit to a SHALLOW or DEEP label. Inputs below this threshold
                fall through to the LLM.
            affinity_deep_threshold: Affinity scores strictly below this value
                always produce a DEEP result, bypassing all other checks.
        """
        self._threshold = similarity_threshold
        self._affinity_deep_threshold = affinity_deep_threshold

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
        # Affinity override — always go DEEP if relationship is weak
        if affinity < self._affinity_deep_threshold:
            return self._result(
                DEEP,
                confidence=1.0,
                reasoning=f"affinity {affinity} < {self._affinity_deep_threshold} → DEEP",
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

    async def _llm_classify(self, user_input: str, user_topic: str, affinity: int) -> tuple[str, str]:
        """Call the LLM to classify ambiguous inputs.

        Args:
            user_input: The raw user message.
            user_topic: Topic label for context.
            affinity: Current affinity score.

        Returns:
            Tuple of (``"SHALLOW"`` or ``"DEEP"``, reasoning string).
        """
        llm = _get_llm()
        system_prompt = SystemMessage(content=_DEPTH_CLASSIFIER_SYSTEM_PROMPT.format(
            affinity_threshold=self._affinity_deep_threshold,
        ))
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
        # Expected: DEEP (affinity override)
        {"user_input": "你好", "user_topic": "greeting", "affinity": 300},
    ]

    for i, case in enumerate(cases, 1):
        print(f"\n[case {i}] input={case['user_input']!r} affinity={case['affinity']}")
        result = await classifier.classify(**case)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(test_main())
