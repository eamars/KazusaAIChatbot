"""Base agent interface and registry.

Every sub-agent (tool agent, speech agent, etc.) implements :class:`BaseAgent`.
The :data:`AGENT_REGISTRY` maps agent names to their concrete classes so the
persona supervisor can dispatch dynamically at runtime.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

from kazusa_ai_chatbot.state import AgentResult, BotState

logger = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Protocol that every sub-agent must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique agent name used in supervisor plans, e.g. ``web_search_agent``."""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """One-line description shown to the supervisor LLM for planning."""

    @abc.abstractmethod
    async def run(
        self,
        state: BotState,
        task: str,
        expected_response: str = "",
    ) -> AgentResult:
        """Execute the agent and return a structured result.

        Parameters
        ----------
        state:
            The full bot state (agents may read but should not mutate it).
        task:
            The complete task for the agent to carry out (includes user intent and any supervisor instructions).
        expected_response:
            A supervisor-authored description of the shape and level of detail the
            agent should return.

        Returns
        -------
        AgentResult
            Always returns a result, even on failure (with ``status="error"``).
        """


# ── Agent registry ──────────────────────────────────────────────────

AGENT_REGISTRY: dict[str, BaseAgent] = {}


def register_agent(agent: BaseAgent) -> None:
    """Register a concrete agent instance in the global registry."""
    AGENT_REGISTRY[agent.name] = agent
    logger.info("Registered agent: %s", agent.name)


def get_agent(name: str) -> BaseAgent | None:
    """Look up an agent by name."""
    return AGENT_REGISTRY.get(name)


def list_agent_descriptions() -> list[dict[str, str]]:
    """Return a list of ``{"name": ..., "description": ...}`` for all agents.

    Used by the supervisor to build its planning prompt.
    """
    return [
        {"name": a.name, "description": a.description}
        for a in AGENT_REGISTRY.values()
    ]
