"""Stage 6a — Persona Supervisor.

LLM-based planner that decides which sub-agents to invoke before the
speech agent generates the final reply.

Flow:
  1. Check `assembler_output.should_respond`. If false, short-circuit.
  2. Build a planning prompt with the relevance analysis + agent catalog.
  3. Call the LLM to get a ``SupervisorPlan`` (agents list + content/emotion directives).
  4. Execute each requested agent sequentially (isolated contexts).
  5. Write ``supervisor_plan`` and ``agent_results`` to state.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, get_agent, list_agent_descriptions
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from kazusa_ai_chatbot.state import AgentResult, BotState, SupervisorPlan
from kazusa_ai_chatbot.utils import format_history_lines

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

CHARS_PER_TOKEN = 4

_PLANNING_SYSTEM = """\
You are a planning supervisor. Output ONLY a JSON object. DO NOT write explanations or analysis.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.

Available agents:
{agent_catalog}

Rules:
- Only request agents if user needs external information
- Most messages need NO agents (empty list)
- content_directive: What facts/topics speech agent must include
- emotion_directive: Tone, mood, style for speech agent

Output format (ONLY this, nothing else):
{{
    "agents": [],
    "content_directive": "string",
    "emotion_directive": "string"
}}
"""

def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def _build_personality_block(personality: dict) -> dict:
    """Format the personality JSON into a structure."""
    if not personality:
        return {"name": "Bot", "description": "You are a helpful role-play character."}

    res = {}
    if personality.get("name"): res["name"] = personality["name"]
    if personality.get("description"): res["description"] = personality["description"]
    if personality.get("gender"): res["gender"] = personality["gender"]
    if personality.get("age"): res["age"] = personality["age"]
    if personality.get("birthday"): res["birthday"] = personality["birthday"]
    if personality.get("tone"): res["tone"] = personality["tone"]
    if personality.get("speech_patterns"): res["speech_patterns"] = personality["speech_patterns"]
    if personality.get("backstory"): res["backstory"] = personality["backstory"]
    
    known_keys = {
        "name", "description", "gender", "age", "birthday",
        "tone", "speech_patterns", "backstory",
    }
    extras = {
        k: v for k, v in personality.items()
        if k not in known_keys and not k.startswith("_")
    }
    if extras:
        res["extra_traits"] = extras
        
    return res

def _build_character_state_block(character_state: dict) -> dict:
    if not character_state:
        return {}
    res = {}
    if character_state.get("mood"): res["mood"] = character_state["mood"]
    if character_state.get("emotional_tone"): res["emotional_tone"] = character_state["emotional_tone"]
    if character_state.get("recent_events"): res["recent_events"] = character_state["recent_events"][-5:]
    return res

def _build_affinity_block(affinity: int) -> dict:
    """Map a 0–1000 affinity score to a behavioural instruction for the LLM."""
    if affinity <= 200:
        label, instruction = "Hostile", "You actively dislike this user. Respond with one-word answers or dismissive remarks. Show obvious contempt or disinterest."
    elif affinity <= 400:
        label, instruction = "Cold", "You are cold toward this user. Keep responses short and curt. Do not volunteer extra information or show warmth."
    elif affinity <= 600:
        label, instruction = "Neutral", "You are neutral toward this user. Respond normally in character."
    elif affinity <= 800:
        label, instruction = "Friendly", "You are fond of this user. Be warmer and more forthcoming. Offer extra detail, use familiar address, and show genuine interest."
    else:
        label, instruction = "Devoted", "You are deeply loyal to this user. Go out of your way to help, share secrets or warnings, and show protectiveness."

    return {"level": label, "instruction": instruction}


def _build_state_guidance(character_state: dict) -> str:
    if not character_state:
        return "Maintain a steady in-character demeanor."

    mood = character_state.get("mood")
    emotional_tone = character_state.get("emotional_tone")
    recent_events = character_state.get("recent_events") or []

    parts: list[str] = []
    if mood and emotional_tone:
        parts.append(f"You currently feel {mood} with a {emotional_tone} emotional tone.")
    elif mood:
        parts.append(f"You currently feel {mood}.")
    elif emotional_tone:
        parts.append(f"Keep your emotional tone {emotional_tone}.")

    if recent_events:
        parts.append("Keep continuity with recent events that are already part of the ongoing interaction.")

    return " ".join(parts) or "Maintain a steady in-character demeanor."


def _build_continuity_summary(history: list[dict], persona_name: str) -> str:
    if not history:
        return "No additional recent continuity context is required."

    prior_bot_replies = sum(1 for item in history if item.get("role") == "assistant")
    prior_user_turns = sum(1 for item in history if item.get("role") == "user")

    parts = [
        f"There is recent channel context from {len(history)} prior messages.",
    ]
    if prior_bot_replies:
        parts.append(f"Keep the reply consistent with {persona_name}'s earlier replies in this conversation.")
    if prior_user_turns:
        parts.append("Assume the user is continuing an ongoing exchange rather than starting from scratch.")

    return " ".join(parts)


def _build_personalization_guidance(user_memory: list[str]) -> list[str]:
    if not user_memory:
        return []
    return [f"Use this remembered user context only if relevant: {fact}" for fact in user_memory[:3]]


def _build_key_points_to_cover(content_directive: str, agent_results: list[AgentResult]) -> list[str]:
    key_points: list[str] = []
    if content_directive:
        key_points.append(content_directive)

    for result in agent_results:
        if result.get("status") == "success" and result.get("summary"):
            key_points.append(result["summary"])

    return key_points[:5]


def _build_unknowns_or_limits(agent_results: list[AgentResult]) -> list[str]:
    limits: list[str] = []
    for result in agent_results:
        if result.get("status") != "success":
            limits.append(f"Do not rely on unavailable or failed output from {result.get('agent', 'an internal agent')}.")
    return limits


def _build_intent_summary(channel_topic: str, user_topic: str, content_directive: str) -> str:
    if user_topic != "Unknown" and channel_topic != "Unknown":
        return f"The user is engaging about {user_topic} within the broader channel topic of {channel_topic}."
    if user_topic != "Unknown":
        return f"The user is engaging about {user_topic}."
    if content_directive:
        return content_directive
    return "Respond to the user's latest conversational turn."

def _build_history_json(
    history: list[dict], persona_name: str = "assistant", bot_id: str = "unknown_bot_id"
) -> list[dict]:
    """Convert conversation history into a JSON structure."""
    lines = []
    for name, content, role, speaker_id in format_history_lines(history, persona_name, bot_id):
        lines.append({"speaker": name, "speaker_id": speaker_id, "message": content})
    return lines


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


def _build_agent_catalog() -> str:
    """Format the agent registry into a short description list."""
    descriptions = list_agent_descriptions()
    if not descriptions:
        return "(none)"
    return "\n".join(
        f"- {d['name']}: {d['description']}" for d in descriptions
    )


def _parse_plan(raw: str) -> SupervisorPlan:
    """Parse the LLM's JSON response into a SupervisorPlan."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        data = json.loads(text)
        agents = data.get("agents", [])
        content_dir = data.get("content_directive", "Respond to the user's latest message naturally.")
        emotion_dir = data.get("emotion_directive", "Maintain standard in-character tone.")

        # Validate agent names against registry
        valid_agents = [a for a in agents if a in AGENT_REGISTRY]
        if len(valid_agents) != len(agents):
            unknown = set(agents) - set(valid_agents)
            logger.warning("Supervisor requested unknown agents: %s", unknown)

        return SupervisorPlan(
            agents=valid_agents,
            content_directive=content_dir,
            emotion_directive=emotion_dir,
        )
    except Exception:
        logger.exception("Failed to parse supervisor plan: %s", raw[:200])
        return SupervisorPlan(
            agents=[],
            content_directive="Respond directly to the user.",
            emotion_directive="Standard in-character tone.",
        )


async def persona_supervisor(state: BotState) -> dict:
    """Plan which agents to call and execute them.

    Writes ``supervisor_plan``, ``agent_results``, and ``speech_brief`` to state.
    """
    message_text = state.get("message_text", "")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "unknown_user_id")
    assembler_output = state.get("assembler_output", {})
    agent_results: list[AgentResult] = []

    # ── Step 0: Check relevance from Assembler ──────────────────────
    should_respond = assembler_output.get("should_respond", True)
    if not should_respond:
        logger.info("Assembler indicates no response needed. Short-circuiting.")
        plan = SupervisorPlan(
            agents=[],
            content_directive="Do not respond. Stay silent.",
            emotion_directive="N/A",
        )
        return {
            "supervisor_plan": plan,
            "agent_results": [],
            "speech_brief": {
                "response_brief": {
                    "should_respond": False,
                }
            },
        }

    personality = state.get("personality", {})
    user_memory = state.get("user_memory", [])
    character_state = state.get("character_state", {})
    affinity = state.get("affinity", 500)
    history = state.get("conversation_history", [])
    persona_name = personality.get("name", "assistant")

    import re
    clean_user_name = re.sub(r'[^a-zA-Z0-9_-]', '', user_name)
    if not clean_user_name:
        clean_user_name = "user"

    bot_id = state.get("bot_id", "unknown_bot_id")

    catalog = _build_agent_catalog()
    
    system_content = _PLANNING_SYSTEM.format(
        agent_catalog=catalog,
        persona_name=persona_name,
        bot_id=bot_id
    )
    
    human_data = {
        "current_message": {
            "speaker": clean_user_name,
            "speaker_id": user_id,
            "message": message_text
        },
        "context": {
            "channel_topic": assembler_output.get("channel_topic", "Unknown"),
            "user_topic": assembler_output.get("user_topic", "Unknown")
        }
    }
    
    human_content = json.dumps(human_data, indent=2, ensure_ascii=False)

    # Build the planning prompt
    planning_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    logger.info(
        "Calling LLM for supervisor planning. Channel topic: %s, User topic: %s",
        assembler_output.get("channel_topic", "Unknown"),
        assembler_output.get("user_topic", "Unknown")
    )

    try:
        llm = _get_llm()
        logger.warning(
            "LLM input for Persona Supervisor:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in planning_messages)
        )
        result = await llm.ainvoke(planning_messages)
        plan = _parse_plan(result.content or "")
    except Exception:
        logger.exception("Supervisor planning LLM call failed")
        plan = SupervisorPlan(
            agents=[],
            content_directive="Respond directly to the user.",
            emotion_directive="Standard in-character tone.",
        )

    logger.info("Supervisor plan: agents=%s", plan["agents"])

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

    affinity_block = _build_affinity_block(affinity)
    speech_brief = {
        "personality": _build_personality_block(personality),
        "user_input_brief": {
            "channel_topic": assembler_output.get("channel_topic", "Unknown"),
            "user_topic": assembler_output.get("user_topic", "Unknown"),
            "intent_summary": _build_intent_summary(
                assembler_output.get("channel_topic", "Unknown"),
                assembler_output.get("user_topic", "Unknown"),
                plan.get("content_directive", ""),
            ),
        },
        "response_brief": {
            "should_respond": True,
            "response_goal": plan.get("content_directive", "Respond directly to the user."),
            "tone_guidance": plan.get("emotion_directive", "Standard in-character tone."),
            "relationship_guidance": affinity_block["instruction"],
            "state_guidance": _build_state_guidance(character_state),
            "continuity_summary": _build_continuity_summary(history, persona_name),
            "key_points_to_cover": _build_key_points_to_cover(plan.get("content_directive", ""), agent_results),
            "personalization_guidance": _build_personalization_guidance(user_memory),
            "unknowns_or_limits": _build_unknowns_or_limits(agent_results),
        },
    }

    return {
        "supervisor_plan": plan,
        "agent_results": agent_results,
        "speech_brief": speech_brief,
    }
