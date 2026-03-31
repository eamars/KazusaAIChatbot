"""Stage 6a — Persona Supervisor.

LLM-based planner that decides which sub-agents to invoke before the
speech agent generates the final reply.

Flow:
  1. Run the relevance agent to check if the bot should respond.
  2. If relevant, build a planning prompt with the user message + agent catalog.
  3. Call the LLM to get a ``SupervisorPlan`` (agents list + speech directive).
  4. Execute each requested agent sequentially (isolated contexts).
  5. Write ``supervisor_plan`` and ``agent_results`` to state.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base import AGENT_REGISTRY, get_agent, list_agent_descriptions
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from state import AgentResult, BotState, SupervisorPlan

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_PLANNING_SYSTEM = """\
You are a planning assistant for a role-play chatbot. Your job is to decide
which specialist agents (if any) should be called BEFORE the final in-character
reply is generated.

Available agents:
{agent_catalog}

Rules:
- Only request an agent if the user's message clearly needs it.
- Most messages (greetings, casual chat, lore questions) need NO agents — respond with an empty list.
- The speech agent always runs last and is NOT in the list — do not include it.
- Provide a brief speech_directive telling the speech agent how to use the agent results
  (tone, level of detail, whether to apologise if something failed, etc.).
- If no agents are needed, set speech_directive to guide the direct reply.

Respond with ONLY valid JSON (no markdown fences):
{{"agents": ["agent_name", ...], "speech_directive": "..."}}
"""


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


# Agents that are auto-managed by the supervisor and should not appear
# in the planning LLM's catalog (they are not plannable).
_AUTO_AGENTS = frozenset({"relevance_agent"})


def _build_agent_catalog() -> str:
    """Format the agent registry into a short description list.

    Excludes auto-managed agents (e.g. relevance_agent) that the
    supervisor runs unconditionally.
    """
    descriptions = [
        d for d in list_agent_descriptions()
        if d["name"] not in _AUTO_AGENTS
    ]
    if not descriptions:
        return "(none)"
    return "\n".join(
        f"- {d['name']}: {d['description']}" for d in descriptions
    )


def _parse_plan(raw: str) -> SupervisorPlan:
    """Parse the LLM's JSON response into a SupervisorPlan.

    Falls back to an empty plan on parse failure.
    """
    # Strip markdown fences if the LLM wraps them anyway
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
        agents = data.get("agents", [])
        directive = data.get("speech_directive", "Respond directly to the user.")

        # Validate agent names against registry
        valid_agents = [a for a in agents if a in AGENT_REGISTRY]
        if len(valid_agents) != len(agents):
            unknown = set(agents) - set(valid_agents)
            logger.warning("Supervisor requested unknown agents: %s", unknown)

        return SupervisorPlan(
            agents=valid_agents,
            speech_directive=directive,
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Failed to parse supervisor plan: %s", raw[:200])
        return SupervisorPlan(
            agents=[],
            speech_directive="Respond directly to the user.",
        )


def _check_relevance(agent_result: AgentResult) -> bool:
    """Parse the relevance agent's result and return should_respond."""
    try:
        data = json.loads(agent_result["summary"])
        return bool(data.get("should_respond", True))
    except (json.JSONDecodeError, TypeError):
        # Fail-open: if we can't parse, assume we should respond
        return True


async def persona_supervisor(state: BotState) -> dict:
    """Plan which agents to call and execute them.

    Writes ``supervisor_plan`` and ``agent_results`` to state.
    """
    message_text = state.get("message_text", "")
    agent_results: list[AgentResult] = []

    # ── Step 0: Relevance check ──────────────────────────────────────
    relevance_agent = get_agent("relevance_agent")
    if relevance_agent is not None:
        try:
            rel_result = await relevance_agent.run(state, message_text)
            agent_results.append(rel_result)
        except Exception as exc:
            logger.exception("Relevance agent crashed — defaulting to respond")
            rel_result = AgentResult(
                agent="relevance_agent",
                status="error",
                summary=json.dumps({"should_respond": True, "reason": f"Crashed: {exc}"}),
                tool_history=[],
            )
            agent_results.append(rel_result)

        if not _check_relevance(rel_result):
            logger.info("Relevance agent says: do not respond")
            plan = SupervisorPlan(
                agents=[],
                speech_directive="Do not respond. Stay silent.",
            )
            return {
                "supervisor_plan": plan,
                "agent_results": agent_results,
            }

    # ── Step 1: LLM planning call ───────────────────────────────────
    catalog = _build_agent_catalog()
    system_prompt = _PLANNING_SYSTEM.format(agent_catalog=catalog)

    try:
        llm = _get_llm()
        planning_messages = [
            HumanMessage(
                content=f"{system_prompt}\n\n---\n\nUser message: \"{message_text}\""
            ),
        ]
        result = await llm.ainvoke(planning_messages)
        plan = _parse_plan(result.content or "")
    except Exception:
        logger.exception("Supervisor planning LLM call failed")
        plan = SupervisorPlan(
            agents=[],
            speech_directive="Respond directly to the user.",
        )

    logger.info("Supervisor plan: agents=%s", plan["agents"])

    # ── Step 2: Execute agents sequentially ─────────────────────────
    for agent_name in plan["agents"]:
        agent = get_agent(agent_name)
        if agent is None:
            logger.error("Agent '%s' not found in registry", agent_name)
            agent_results.append(AgentResult(
                agent=agent_name,
                status="error",
                summary=f"Agent '{agent_name}' is not available.",
                tool_history=[],
            ))
            continue

        logger.info("Running agent: %s", agent_name)
        try:
            result = await agent.run(state, message_text)
            agent_results.append(result)
        except Exception as exc:
            logger.exception("Agent '%s' crashed", agent_name)
            agent_results.append(AgentResult(
                agent=agent_name,
                status="error",
                summary=f"Agent crashed: {exc}",
                tool_history=[],
            ))

    return {
        "supervisor_plan": plan,
        "agent_results": agent_results,
    }
