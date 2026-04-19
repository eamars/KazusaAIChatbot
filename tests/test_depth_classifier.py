"""Unit tests for ``kazusa_ai_chatbot.rag.depth_classifier``.

Embedding and LLM calls are mocked so the tests run fully offline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import OpenAIError

import kazusa_ai_chatbot.rag.depth_classifier as dc
from kazusa_ai_chatbot.rag.depth_classifier import (
    AFFINITY_DEEP_THRESHOLD,
    DEEP,
    DEEP_DISPATCHERS,
    InputDepthClassifier,
    SHALLOW,
    SHALLOW_DISPATCHERS,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_centroid_cache(monkeypatch):
    """Force each test to install its own centroid pair."""
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


# ── Tests ──────────────────────────────────────────────────────────


class TestAffinityOverride:
    async def test_low_affinity_forces_deep(self):
        classifier = InputDepthClassifier()
        result = await classifier.classify(
            user_input="hi",
            user_topic="greeting",
            affinity=AFFINITY_DEEP_THRESHOLD - 1,
        )
        assert result["depth"] == DEEP
        assert result["trigger_dispatchers"] == DEEP_DISPATCHERS

    async def test_at_threshold_does_not_force_deep(self, monkeypatch):
        """Affinity >= threshold does NOT auto-force DEEP."""
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier()
        # Input embedding matches SHALLOW centroid
        with _patch_embedding([1.0, 0.0, 0.0]):
            result = await classifier.classify(
                user_input="what is your name",
                user_topic="",
                affinity=AFFINITY_DEEP_THRESHOLD,
            )
        assert result["depth"] == SHALLOW


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


class TestDispatcherRouting:
    async def test_shallow_returns_user_rag_only(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.5)
        with _patch_embedding([1.0, 0.1, 0.0]):
            result = await classifier.classify(
                user_input="you like cats?",
                user_topic="", affinity=700,
            )
        assert result["trigger_dispatchers"] == ["user_rag"]

    async def test_deep_returns_all_three(self, monkeypatch):
        _install_fake_centroids(monkeypatch)
        classifier = InputDepthClassifier(similarity_threshold=0.5)
        with _patch_embedding([0.1, 1.0, 0.0]):
            result = await classifier.classify(
                user_input="why did you lie to me last time",
                user_topic="", affinity=700,
            )
        assert set(result["trigger_dispatchers"]) == {"user_rag", "internal_rag", "external_rag"}
