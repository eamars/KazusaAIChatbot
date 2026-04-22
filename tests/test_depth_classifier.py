"""Unit tests for ``kazusa_ai_chatbot.rag.depth_classifier``.

Offline tests mock embeddings and LLM. Tests marked ``live_llm`` call real
services and are excluded from the default pytest run.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

import kazusa_ai_chatbot.rag.depth_classifier as dc
from kazusa_ai_chatbot.rag.depth_classifier import (
    DEEP,
    DEEP_DISPATCHERS,
    InputDepthClassifier,
    SHALLOW,
    SHALLOW_DISPATCHERS,
)

logger = logging.getLogger(__name__)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_centroid_cache(monkeypatch, request):
    """Force each test to install its own centroid pair."""
    if request.node.get_closest_marker("live_llm") is not None:
        yield
        return
    monkeypatch.setattr(dc, "_shallow_centroid", None)
    monkeypatch.setattr(dc, "_deep_centroid", None)
    monkeypatch.setattr(dc, "_shallow_centroid_norm", 0.0)
    monkeypatch.setattr(dc, "_deep_centroid_norm", 0.0)
    yield


def _install_fake_centroids(monkeypatch,
                             *,
                             shallow: list[float] = (1.0, 0.0, 0.0),
                             deep: list[float] = (0.0, 1.0, 0.0)) -> None:
    """Bypass the network by pre-setting centroids directly."""
    monkeypatch.setattr(dc, "_shallow_centroid", list(shallow))
    monkeypatch.setattr(dc, "_deep_centroid", list(deep))
    monkeypatch.setattr(dc, "_shallow_centroid_norm", dc._vec_norm(list(shallow)))
    monkeypatch.setattr(dc, "_deep_centroid_norm", dc._vec_norm(list(deep)))


def _patch_embedding(value: list[float]):
    return patch(
        "kazusa_ai_chatbot.rag.depth_classifier.get_text_embedding",
        AsyncMock(return_value=value),
    )


def _log_live_case(case_name: str, *, user_input: str, user_topic: str, result: dict) -> None:
    logger.info(
        "depth_classifier.%s input=%r topic=%r output=%r",
        case_name,
        user_input,
        user_topic,
        result,
    )


# ── Tests ──────────────────────────────────────────────────────────


class TestFastPathShallow:
    async def test_embedding_matches_shallow(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.5)
        with _patch_embedding([1.0, 0.1, 0.0]):
            result = await classifier.classify(
                user_input="what colour do you like",
                user_topic="chitchat",
                affinity=700,
            )
        assert result["depth"] == SHALLOW
        assert result["trigger_dispatchers"] == SHALLOW_DISPATCHERS
        assert result["confidence"] > 0.5


class TestFastPathDeep:
    async def test_embedding_matches_deep(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.5)
        with _patch_embedding([0.1, 1.0, 0.0]):
            result = await classifier.classify(
                user_input="remember when you said you'd help me",
                user_topic="relationship",
                affinity=700,
            )
        assert result["depth"] == DEEP
        assert result["trigger_dispatchers"] == DEEP_DISPATCHERS


class TestFallbackToLLM:
    async def test_ambiguous_triggers_llm(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.95)

        # Ambiguous embedding — below threshold to both centroids
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"depth": "SHALLOW", "reasoning": "simple greeting"}'
        fake_llm.ainvoke = AsyncMock(return_value=fake_response)

        with _patch_embedding([0.5, 0.5, 0.0]), \
             patch("kazusa_ai_chatbot.rag.depth_classifier._get_llm", return_value=fake_llm):
            result = await classifier.classify(
                user_input="xyz abc",
                user_topic="",
                affinity=700,
            )

        assert result["depth"] == SHALLOW
        fake_llm.ainvoke.assert_awaited_once()

    async def test_llm_default_deep_on_unparseable(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.95)

        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = "not json at all"
        fake_llm.ainvoke = AsyncMock(return_value=fake_response)

        with _patch_embedding([0.5, 0.5, 0.0]), \
             patch("kazusa_ai_chatbot.rag.depth_classifier._get_llm", return_value=fake_llm):
            result = await classifier.classify(
                user_input="???",
                user_topic="",
                affinity=700,
            )
        assert result["depth"] == DEEP


class TestFailureFallback:
    async def test_embedding_failure_drops_to_llm(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier()

        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"depth": "DEEP", "reasoning": "fallback"}'
        fake_llm.ainvoke = AsyncMock(return_value=fake_response)

        with patch("kazusa_ai_chatbot.rag.depth_classifier.get_text_embedding",
                   AsyncMock(side_effect=OpenAIError("network down"))), \
             patch("kazusa_ai_chatbot.rag.depth_classifier._get_llm", return_value=fake_llm):
            result = await classifier.classify(
                user_input="hello",
                user_topic="",
                affinity=700,
            )

        assert result["depth"] == DEEP
        fake_llm.ainvoke.assert_awaited_once()

    async def test_everything_fails_defaults_deep(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier()

        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=OpenAIError("llm down"))

        with patch("kazusa_ai_chatbot.rag.depth_classifier.get_text_embedding",
                   AsyncMock(side_effect=OpenAIError("embed down"))), \
             patch("kazusa_ai_chatbot.rag.depth_classifier._get_llm", return_value=fake_llm):
            result = await classifier.classify(
                user_input="?",
                user_topic="",
                affinity=700,
            )

        assert result["depth"] == DEEP


# ── Live LLM integration tests ─────────────────────────────────────
# Run with: pytest -m live_llm


@pytest.mark.live_llm
class TestSearchIntentLive:
    """Verify bug-report inputs are classified DEEP with a real backend.

    Case A & B hit the rule-based early exit (conf=1.0, no LLM call).
    Case C has no explicit search keyword and falls through to the real LLM,
    verifying the updated prompt classifies price/market queries as DEEP.
    """

    async def test_case_a_explicit_search_with_product_price(self):
        """'搜一下现在DDR5内存价格' must be DEEP via search-intent rule."""
        classifier = InputDepthClassifier()
        user_input = "小钳子(@974a5aa4-d67b-4adb-8b0a-eea6cfb9297e): 搜一下现在DDR5内存价格】"
        user_topic = "hardware"
        result = await classifier.classify(
            user_input=user_input,
            user_topic=user_topic,
            affinity=700,
        )
        _log_live_case("case_a", user_input=user_input, user_topic=user_topic, result=result)
        assert result["depth"] == DEEP, f"Expected DEEP, got: {result}"
        assert result["confidence"] == 1.0, "Rule-based exit must produce conf=1.0"
        assert "search" in result["reasoning"].lower() or "搜" in result["reasoning"]

    async def test_case_b_did_you_find_price(self):
        """'搜到ddr5的价格了么' must be DEEP via search-intent rule ('搜到' term)."""
        classifier = InputDepthClassifier()
        user_input = "蚝爹油(@76a37e60-982e-45cb-af28-6d8c6b533297): 千纱搜到ddr5的价格了么？"
        user_topic = "hardware"
        result = await classifier.classify(
            user_input=user_input,
            user_topic=user_topic,
            affinity=700,
        )
        _log_live_case("case_b", user_input=user_input, user_topic=user_topic, result=result)
        assert result["depth"] == DEEP, f"Expected DEEP, got: {result}"
        assert result["confidence"] == 1.0, "Rule-based exit must produce conf=1.0"

    async def test_case_c_price_query_no_explicit_search_keyword_llm_path(self):
        """'DDR5内存现在多少钱' — no explicit search keyword, time-sensitive + no EXTERNAL_INFO match.

        Falls through to the real LLM; verifies the updated prompt classifies
        external price queries as DEEP.
        """
        classifier = InputDepthClassifier()
        user_input = "DDR5内存现在多少钱"
        user_topic = "hardware"
        result = await classifier.classify(
            user_input=user_input,
            user_topic=user_topic,
            affinity=700,
        )
        _log_live_case("case_c", user_input=user_input, user_topic=user_topic, result=result)
        assert result["depth"] == DEEP, f"Expected DEEP, got: {result}"


