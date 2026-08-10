"""Test helpers for explicit-config LLM fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kazusa_ai_chatbot.llm_interface import LLMCallConfig


@dataclass(frozen=True)
class TestLLMStageBinding:
    """Explicit test-only pairing of an LLM double and its call config."""

    llm: Any
    config: LLMCallConfig


def make_llm_call_config(stage_name: str = "test_stage") -> LLMCallConfig:
    """Build a representative explicit LLM call config for tests."""

    config = LLMCallConfig(
        stage_name=stage_name,
        route_name="TEST_LLM",
        base_url="http://test-llm.invalid/v1",
        api_key="test-api-key",
        model="test-model",
        temperature=0.0,
        top_p=None,
        top_k=None,
        max_completion_tokens=128,
        presence_penalty=None,
    )
    return config


def bind_test_llm(
    llm: Any,
    stage_name: str = "test_stage",
) -> TestLLMStageBinding:
    """Pair a fake LLM with an explicit test call configuration."""

    binding = TestLLMStageBinding(
        llm=llm,
        config=make_llm_call_config(stage_name),
    )
    return binding
