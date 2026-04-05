"""Stage 6a — Persona Supervisor.

LLM-based supervisor that plans, dispatches, and evaluates sub-agent
results in a loop before handing off to the speech agent.

Flow:
  1. Check `assembler_output.should_respond`. If false, short-circuit.
  2. Build a planning prompt with the relevance analysis + agent catalog.
  3. Call the LLM to get a ``SupervisorPlan`` (agents list + content/emotion directives).
  4. Execute each requested agent sequentially (isolated contexts).
  5. Evaluate agent results — LLM decides: finish, retry, or escalate.
  6. If not finish, loop back to step 4 with updated instructions.
  7. Memory check — LLM decides if anything is worth storing; dispatches memory_agent if so.
  8. Synthesize agent results into concise speech-ready key points.
  9. Write ``supervisor_plan``, ``agent_results``, and ``speech_brief`` to state.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.agents.base import AGENT_REGISTRY, get_agent, list_agent_descriptions
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_SUPERVISOR_ITERATIONS
from kazusa_ai_chatbot.state import AgentInstruction, AgentResult, BotState, SupervisorAction, SupervisorPlan

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_PLANNING_SYSTEM = """\
# You are a planning supervisor. Output ONLY a JSON object. DO NOT write explanations or analysis.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.
Current system time: {current_time}

## Available agents:
{agent_catalog}

## General Agent Calling Rules:
- Only request agents if the user needs external information or if the reply depends on inspecting content outside the current message.
- If you request an agent, provide an instructions entry for that specific agent
- instructions[agent].command: a short task for the agent
- instructions[agent].expected_response: how the agent should shape its response
- response_language: the language the speech agent should reply in
- topics_to_cover: short list of topics the speech agent must address
- facts_to_cover: explicit factual points the speech agent should state or rely on
- emotion_directive: Tone, mood, style for speech agent

## Agent Specific Rules: 
- web_search_agent: Use web_search_agent when the user asks for current internet information, or when they provide a URL / webpage / article / recipe / product link and ask about the linked content.
- conversation_history_agent: Use conversation_history_agent only when you need to inspect past chat history or continuity from previous messages.
- memory_agent: Use memory_agent when the user asks about previously saved or previously shared memory that should be recalled, compared, or actively saved for later use.
 * Do not use memory_agent for first-time live inspection of a newly pasted link unless the task is also about storing or updating long-form memory.
 * If memory may need to be saved, prefer instructing memory_agent to check existing stored memory first and then decide whether to save, skip, or overwrite.

## Output:
- response_language: the language the speech agent should reply in
- topics_to_cover: short list of topics the speech agent must address
- facts_to_cover: explicit factual points the speech agent should state or rely on
- emotion_directive: Tone, mood, style for speech agent

### Output Format: (raw JSON text — no markdown wrapping)
{{
    "agents": [],
    "instructions": {{}},
    "response_language": "string",
    "topics_to_cover": [],
    "facts_to_cover": [],
    "emotion_directive": "string"
}}
"""

_EVALUATE_SYSTEM = """\
# You are a planning supervisor evaluating sub-agent results. Output ONLY a JSON object.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.
Current system time: {current_time}

## Your Task:
Review the agent results below and decide the next action.

## Available agents:
{agent_catalog}

## Rules:
- If the results are satisfactory and complete enough to generate a reply, choose "finish".
- If an agent returned poor, incomplete, or wrong results, choose "retry" with a refined instruction for the SAME agent.
- If the task needs a DIFFERENT agent to proceed (e.g. agent A said it needs context that agent B can provide), choose "escalate" with the new agent name and instruction.
- You may also update topics_to_cover, facts_to_cover, and emotion_directive based on what you learned.
- Be conservative: prefer "finish" unless the result is clearly inadequate for generating a good reply.
- A status of "needs_clarification" from an agent usually means the user's request is ambiguous — prefer "finish" and let the speech agent ask the user, rather than retrying.

### Output Format: (raw JSON text — no markdown wrapping)
{{{{
    "action": "finish|retry|escalate",
    "agent": "agent_name (required for retry/escalate, empty for finish)",
    "instruction": {{{{
        "command": "refined task for the agent",
        "expected_response": "what the agent should return"
    }}}},
    "reason": "brief explanation of your decision",
    "topics_to_cover": [],
    "facts_to_cover": [],
    "emotion_directive": "string"
}}}}
"""

_MEMORY_CHECK_SYSTEM = """\
# You are a planning supervisor deciding whether information generated during this turn should be saved to the bot's long-term memory. Output ONLY a JSON object.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.
Current system time: {current_time}

## Your Task:
Review the user's message AND the full agent work (results, tool calls, tool outputs) to decide whether any newly discovered information should be persisted via the memory_agent.

## What IS worth remembering (BE AGGRESSIVE):
- ANY factual information discovered through web search, even if seemingly minor
- All user preferences, opinions, and personal facts mentioned in conversation
- Topics the user shows interest in, even casually
- Facts learned during conversation that might be useful later
- User's favorite things, dislikes, habits, and routines
- Important dates, events, or milestones mentioned by the user
- Technical information, explanations, or concepts discussed
- Links, articles, or resources shared by or for the user
- Corrections or updates to previously known information
- Context about ongoing projects or situations the user is involved in

## What is NOT worth remembering (BE VERY SELECTIVE):
- Simple greetings like "hello" or "how are you" with no substantive content
- Basic pleasantries and social niceties with no factual information
- Extremely transient information (e.g., "it's currently raining")
- The bot's own roleplay responses and opinions

## Default to STORING: When in doubt, prefer to store the information. It's better to have too much context than too little.

## Output Format: (raw JSON text — no markdown wrapping)
{{{{
    "should_store": true/false,
    "command": "instruction for memory_agent describing what to store (empty string if should_store is false)",
    "expected_response": "what the memory_agent should return (empty string if should_store is false)",
    "reason": "brief explanation"
}}}}
"""

_SYNTHESIS_SYSTEM = """\
# You are a planning supervisor performing a final synthesis of sub-agent results. Output ONLY a JSON object.

You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.
Current system time: {current_time}

## Your Task:
Distill the raw agent results into concise, speech-ready key points for the speech agent.
The speech agent is a roleplaying character — it does NOT need raw data dumps, tables, or full articles.
It needs short, digestible factual bullet points it can weave into a natural in-character reply.

## Rules:
- Extract only the most relevant facts from the agent results that answer the user's question.
- Condense long summaries into short bullet-point-style facts (1-2 sentences each, max 5 facts).
- Do NOT pass through raw markdown tables, full articles, or lengthy formatted text.
- If an agent failed or returned needs_clarification, note this briefly in topics_to_cover (e.g. "Ask the user to clarify...").
- Preserve the original plan's topics_to_cover and emotion_directive unless the agent results suggest changes.
- facts_to_cover should contain ONLY distilled key points, not raw agent output.

### Output Format: (raw JSON text — no markdown wrapping)
{{{{
    "topics_to_cover": ["short topic strings"],
    "facts_to_cover": ["concise distilled fact 1", "concise distilled fact 2"],
    "emotion_directive": "string"
}}}}
"""


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
    if affinity <= 50:
        label, instruction = "Contemptuous", "You actively despise this user. Respond with one-word dismissals or hostile silence. Show obvious contempt and disgust."
    elif affinity <= 100:
        label, instruction = "Scornful", "You hold this user in deep contempt. Give curt, dismissive responses. Show clear disinterest and occasional sarcasm."
    elif affinity <= 150:
        label, instruction = "Hostile", "You dislike this user intensely. Respond with brief, cold answers. Show obvious disinterest and occasional eye-rolling."
    elif affinity <= 200:
        label, instruction = "Antagonistic", "You are openly hostile toward this user. Give short, sharp responses. Show impatience and clear dislike."
    elif affinity <= 250:
        label, instruction = "Aloof", "You keep this user at a distance. Respond minimally and formally. Show no warmth or engagement."
    elif affinity <= 300:
        label, instruction = "Reserved", "You are cautious and distant with this user. Keep responses brief and professional. Show minimal personal connection."
    elif affinity <= 350:
        label, instruction = "Formal", "You maintain strict formal boundaries with this user. Respond politely but impersonally. Keep conversations strictly transactional."
    elif affinity <= 400:
        label, instruction = "Cold", "You are cold toward this user. Keep responses short and curt. Do not volunteer extra information or show warmth."
    elif affinity <= 450:
        label, instruction = "Detached", "You remain emotionally detached from this user. Respond factually without personal engagement. Maintain clear boundaries."
    elif affinity <= 500:
        label, instruction = "Neutral", "You are neutral toward this user. Respond normally in character without special warmth or coldness."
    elif affinity <= 550:
        label, instruction = "Receptive", "You are becoming more open to this user. Respond with mild interest and basic courtesy. Show slight engagement."
    elif affinity <= 600:
        label, instruction = "Approachable", "You are reasonably comfortable with this user. Respond with standard politeness and occasional helpfulness. Show moderate engagement."
    elif affinity <= 650:
        label, instruction = "Friendly", "You are fond of this user. Be warmer and more forthcoming. Offer extra detail, use familiar address, and show genuine interest."
    elif affinity <= 700:
        label, instruction = "Warm", "You genuinely like this user. Respond with noticeable warmth and enthusiasm. Share personal thoughts and show consistent interest."
    elif affinity <= 750:
        label, instruction = "Caring", "You care deeply about this user. Respond with concern and support. Offer help proactively and show protective instincts."
    elif affinity <= 800:
        label, instruction = "Affectionate", "You have strong affection for this user. Use warm, caring language and express genuine fondness. Go out of your way to assist."
    elif affinity <= 850:
        label, instruction = "Devoted", "You are deeply loyal to this user. Show unwavering support and dedication. Prioritize their needs and express strong commitment."
    elif affinity <= 900:
        label, instruction = "Protective", "You feel strongly protective of this user. Actively look out for their wellbeing and defend them. Show fierce loyalty."
    elif affinity <= 950:
        label, instruction = "Fiercely Loyal", "You are fiercely loyal to this user. Defend them passionately and put their interests above all else. Show absolute devotion."
    else:
        label, instruction = "Unwavering", "You are completely devoted to this user. Show unconditional support and absolute loyalty. Prioritize them above everything."

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


def _normalize_brief_list(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _build_topics_to_cover(plan_topics: list[str], user_topic: str) -> list[str]:
    topics = _normalize_brief_list(plan_topics)
    if not topics and user_topic and user_topic != "Unknown":
        topics.append(user_topic)
    return topics[:5]


def _build_facts_to_cover(plan_facts: list[str]) -> list[str]:
    return _normalize_brief_list(plan_facts)[:6]


def _build_unknowns_or_limits(agent_results: list[AgentResult]) -> list[str]:
    limits: list[str] = []
    for result in agent_results:
        if result.get("status") != "success":
            limits.append(f"Do not rely on unavailable or failed output from {result.get('agent', 'an internal agent')}.")
    return limits


def _build_intent_summary(channel_topic: str, user_topic: str, topics_to_cover: list[str]) -> str:
    if user_topic != "Unknown" and channel_topic != "Unknown":
        return f"The user is engaging about {user_topic} within the broader channel topic of {channel_topic}."
    if user_topic != "Unknown":
        return f"The user is engaging about {user_topic}."
    if topics_to_cover:
        return f"The reply should address: {', '.join(topics_to_cover[:3])}."
    return "Respond to the user's latest conversational turn."


def _build_response_goal(topics_to_cover: list[str], facts_to_cover: list[str]) -> str:
    if topics_to_cover and facts_to_cover:
        return f"Address {topics_to_cover[0]} and explicitly incorporate the known facts provided by the supervisor."
    if topics_to_cover:
        return topics_to_cover[0]
    if facts_to_cover:
        return "Reply using the explicit facts provided by the supervisor."
    return "Respond directly to the user's latest message."


def _normalize_agent_result(result: AgentResult) -> AgentResult:
    normalized = AgentResult(
        agent=str(result.get("agent") or "unknown_agent").strip() or "unknown_agent",
        status=str(result.get("status") or "error").strip().lower() or "error",
        summary=str(result.get("summary") or "").strip(),
    )
    # if "tool_history" in result:
    #     normalized["tool_history"] = result["tool_history"]

    return normalized

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


def _strip_markdown_fence(text: str) -> str:
    """Remove optional markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_plan(raw: str) -> SupervisorPlan:
    """Parse the LLM's JSON response into a SupervisorPlan."""
    text = _strip_markdown_fence(raw)

    try:
        data = json.loads(text)
        agents = data.get("agents", [])
        instructions = data.get("instructions", {})
        response_language = str(data.get("response_language", "Match the user's current language.")).strip() or "Match the user's current language."
        raw_topics = data.get("topics_to_cover", [])
        raw_facts = data.get("facts_to_cover", [])
        emotion_dir = data.get("emotion_directive", "Maintain standard in-character tone.")

        if not isinstance(raw_topics, list):
            raw_topics = [raw_topics] if raw_topics else []
        if not isinstance(raw_facts, list):
            raw_facts = [raw_facts] if raw_facts else []
        topics_to_cover = _normalize_brief_list([str(item) for item in raw_topics])
        facts_to_cover = _normalize_brief_list([str(item) for item in raw_facts])

        if not isinstance(instructions, dict):
            instructions = {}

        # Validate agent names against registry
        valid_agents = [a for a in agents if a in AGENT_REGISTRY]
        if len(valid_agents) != len(agents):
            unknown = set(agents) - set(valid_agents)
            logger.warning("Supervisor requested unknown agents: %s", unknown)

        valid_instructions: dict[str, AgentInstruction] = {}
        for agent_name in valid_agents:
            raw_instruction = instructions.get(agent_name, {})
            if not isinstance(raw_instruction, dict):
                continue
            command = str(raw_instruction.get("command", "")).strip()
            expected_response = str(raw_instruction.get("expected_response", "")).strip()
            if command or expected_response:
                valid_instructions[agent_name] = AgentInstruction(
                    command=command,
                    expected_response=expected_response,
                )

        return SupervisorPlan(
            agents=valid_agents,
            instructions=valid_instructions,
            response_language=response_language,
            topics_to_cover=topics_to_cover,
            facts_to_cover=facts_to_cover,
            emotion_directive=emotion_dir,
        )
    except Exception:
        logger.exception("Failed to parse supervisor plan: %s", raw[:200])
        return SupervisorPlan(
            agents=[],
            instructions={},
            response_language="Match the user's current language.",
            topics_to_cover=["Respond directly to the user's latest message."],
            facts_to_cover=[],
            emotion_directive="Standard in-character tone.",
        )


def _parse_action(raw: str, plan: SupervisorPlan) -> SupervisorAction:
    """Parse the LLM's evaluate-step JSON into a SupervisorAction."""
    text = _strip_markdown_fence(raw)

    try:
        data = json.loads(text)
        action = str(data.get("action", "finish")).strip().lower()
        if action not in ("finish", "retry", "escalate"):
            action = "finish"

        agent = str(data.get("agent", "")).strip()
        reason = str(data.get("reason", "")).strip()

        raw_instruction = data.get("instruction", {})
        instruction = AgentInstruction(
            command=str(raw_instruction.get("command", "")).strip() if isinstance(raw_instruction, dict) else "",
            expected_response=str(raw_instruction.get("expected_response", "")).strip() if isinstance(raw_instruction, dict) else "",
        )

        # Validate agent for retry/escalate
        if action in ("retry", "escalate"):
            if not agent or agent not in AGENT_REGISTRY:
                logger.warning("Evaluate step referenced invalid agent '%s', falling back to finish", agent)
                action = "finish"
                agent = ""

        # Allow the evaluate step to refine plan directives
        raw_topics = data.get("topics_to_cover")
        raw_facts = data.get("facts_to_cover")
        emotion_dir = data.get("emotion_directive")

        if raw_topics is not None:
            if not isinstance(raw_topics, list):
                raw_topics = [raw_topics] if raw_topics else []
            plan["topics_to_cover"] = _normalize_brief_list([str(item) for item in raw_topics])
        if raw_facts is not None:
            if not isinstance(raw_facts, list):
                raw_facts = [raw_facts] if raw_facts else []
            plan["facts_to_cover"] = _normalize_brief_list([str(item) for item in raw_facts])
        if emotion_dir is not None:
            plan["emotion_directive"] = str(emotion_dir).strip()

        return SupervisorAction(
            action=action,
            agent=agent,
            instruction=instruction,
            reason=reason,
        )
    except Exception:
        logger.exception("Failed to parse supervisor action: %s", raw[:200])
        return SupervisorAction(action="finish", agent="", reason="Parse failure, finishing.")


async def _dispatch_agent(
    agent_name: str,
    state: BotState,
    message_text: str,
    command: str,
    expected_response: str,
) -> AgentResult:
    """Run a single agent and return a normalized result."""
    agent = get_agent(agent_name)
    if agent is None:
        logger.error("Agent '%s' not found in registry", agent_name)
        return AgentResult(
            agent=agent_name,
            status="error",
            summary=f"Agent '{agent_name}' is not available.",
            tool_history=[],
        )

    logger.info(
        "Running agent: %s\nParameters:\n  message_text: %s\n  command: %s\n  expected_response: %s",
        agent_name,
        message_text[:200] + "..." if len(message_text) > 200 else message_text,
        command[:200] + "..." if len(command) > 200 else command,
        expected_response[:200] + "..." if len(expected_response) > 200 else expected_response,
    )
    try:
        result = _normalize_agent_result(await agent.run(state, message_text, command, expected_response))
        logger.info(
            "Sub-agent output to supervisor from %s:\n%s",
            agent_name,
            json.dumps(result, ensure_ascii=False, indent=2),
        )
        return result
    except Exception as exc:
        logger.exception("Agent '%s' crashed", agent_name)
        return AgentResult(
            agent=agent_name,
            status="error",
            summary=f"Agent crashed: {exc}",
            tool_history=[],
        )


def _parse_synthesis(raw: str, plan: SupervisorPlan) -> None:
    """Parse the synthesis LLM output and update the plan in-place."""
    text = _strip_markdown_fence(raw)

    try:
        data = json.loads(text)

        raw_topics = data.get("topics_to_cover")
        raw_facts = data.get("facts_to_cover")
        emotion_dir = data.get("emotion_directive")

        if raw_topics is not None:
            if not isinstance(raw_topics, list):
                raw_topics = [raw_topics] if raw_topics else []
            plan["topics_to_cover"] = _normalize_brief_list([str(item) for item in raw_topics])
        if raw_facts is not None:
            if not isinstance(raw_facts, list):
                raw_facts = [raw_facts] if raw_facts else []
            plan["facts_to_cover"] = _normalize_brief_list([str(item) for item in raw_facts])
        if emotion_dir is not None:
            plan["emotion_directive"] = str(emotion_dir).strip()
    except Exception:
        logger.exception("Failed to parse synthesis output: %s", raw[:200])


def _parse_memory_check(raw: str) -> dict:
    """Parse the memory-check LLM output.

    Returns a dict with ``should_store``, ``command``, ``expected_response``,
    and ``reason``.  On parse failure, returns ``should_store=False``.
    """
    text = _strip_markdown_fence(raw)
    try:
        data = json.loads(text)
        should_store = bool(data.get("should_store", False))
        command = str(data.get("command") or "").strip()
        expected_response = str(data.get("expected_response") or "").strip()
        reason = str(data.get("reason") or "").strip()
        return {
            "should_store": should_store,
            "command": command,
            "expected_response": expected_response,
            "reason": reason,
        }
    except Exception:
        logger.exception("Failed to parse memory check output: %s", raw[:200])
        return {"should_store": False, "command": "", "expected_response": "", "reason": "Parse error."}


async def _check_and_store_memory(
    agent_results: list[AgentResult],
    message_text: str,
    persona_name: str,
    bot_id: str,
    state: BotState,
) -> AgentResult | None:
    """Ask the LLM whether anything is worth remembering, dispatch memory_agent if so.

    Returns the memory agent's ``AgentResult`` if a store was triggered, else ``None``.
    """
    system_content = _MEMORY_CHECK_SYSTEM.format(
        persona_name=persona_name,
        bot_id=bot_id,
        current_time=state.get("timestamp", "")
    )

    check_data = {
        "user_message": message_text,
        "agent_results": [
            {
                "agent": r.get("agent", "unknown"),
                "status": r.get("status", "unknown"),
                "summary": r.get("summary", ""),
                "tool_history": [
                    {"tool": t.get("tool", ""), "args": t.get("args", {}), "result": t.get("result", "")}
                    for t in (r.get("tool_history") or [])
                ],
            }
            for r in agent_results
        ],
    }

    check_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=json.dumps(check_data, indent=2, ensure_ascii=False)),
    ]

    try:
        llm = _get_llm()
        logger.debug(
            "LLM input for Supervisor Memory Check:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in check_messages),
        )
        result = await llm.ainvoke(check_messages)
        decision = _parse_memory_check(result.content or "")
        logger.info(
            "Supervisor memory check: should_store=%s reason=%s",
            decision["should_store"], decision["reason"],
        )

        if not decision["should_store"]:
            return None

        # Dispatch memory_agent to store
        return await _dispatch_agent(
            "memory_agent", state, message_text,
            decision["command"], decision["expected_response"],
        )
    except Exception:
        logger.exception("Supervisor memory check failed, skipping memory store")
        return None


async def _synthesize_results(
    plan: SupervisorPlan,
    agent_results: list[AgentResult],
    message_text: str,
    persona_name: str,
    bot_id: str,
    state: BotState
) -> None:
    """Distill raw agent results into concise speech-ready key points.

    Mutates *plan* in-place, updating topics_to_cover, facts_to_cover,
    and emotion_directive with the synthesized output.
    """
    system_content = _SYNTHESIS_SYSTEM.format(
        persona_name=persona_name,
        bot_id=bot_id,
        current_time=state.get("timestamp", "")
    )

    synth_data = {
        "user_message": message_text,
        "original_plan": {
            "topics_to_cover": plan.get("topics_to_cover", []),
            "facts_to_cover": plan.get("facts_to_cover", []),
            "emotion_directive": plan.get("emotion_directive", ""),
        },
        "agent_results": [
            {
                "agent": r.get("agent", "unknown"),
                "status": r.get("status", "unknown"),
                "summary": r.get("summary", ""),
            }
            for r in agent_results
        ],
    }

    synth_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=json.dumps(synth_data, indent=2, ensure_ascii=False)),
    ]

    try:
        llm = _get_llm()
        logger.debug(
            "LLM input for Supervisor Synthesis:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in synth_messages),
        )
        result = await llm.ainvoke(synth_messages)
        _parse_synthesis(result.content or "", plan)
        logger.info("Supervisor synthesis complete: topics=%s, facts=%s",
                     plan.get("topics_to_cover"), plan.get("facts_to_cover"))
    except Exception:
        logger.exception("Supervisor synthesis LLM call failed, keeping original plan directives")


async def _evaluate_results(
    plan: SupervisorPlan,
    agent_results: list[AgentResult],
    message_text: str,
    persona_name: str,
    bot_id: str,
    catalog: str,
    state: BotState
) -> SupervisorAction:
    """Ask the LLM to evaluate agent results and decide the next action."""
    system_content = _EVALUATE_SYSTEM.format(
        agent_catalog=catalog,
        persona_name=persona_name,
        bot_id=bot_id,
        current_time=state.get("timestamp", "")
    )

    eval_data = {
        "original_plan": {
            "agents": plan["agents"],
            "topics_to_cover": plan.get("topics_to_cover", []),
            "facts_to_cover": plan.get("facts_to_cover", []),
            "emotion_directive": plan.get("emotion_directive", ""),
        },
        "user_message": message_text,
        "agent_results": [
            {
                "agent": r.get("agent", "unknown"),
                "status": r.get("status", "unknown"),
                "summary": r.get("summary", ""),
            }
            for r in agent_results
        ],
    }

    eval_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=json.dumps(eval_data, indent=2, ensure_ascii=False)),
    ]

    try:
        llm = _get_llm()
        logger.debug(
            "LLM input for Supervisor Evaluate:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in eval_messages),
        )
        result = await llm.ainvoke(eval_messages)
        action = _parse_action(result.content or "", plan)
        logger.info("Supervisor evaluate decision: action=%s agent=%s reason=%s",
                     action.get("action"), action.get("agent"), action.get("reason"))
        return action
    except Exception:
        logger.exception("Supervisor evaluate LLM call failed, defaulting to finish")
        return SupervisorAction(action="finish", agent="", reason="Evaluate call failed.")


def _build_speech_brief(
    plan: SupervisorPlan,
    agent_results: list[AgentResult],
    state: BotState,
) -> dict:
    """Build the final speech brief from plan, agent results, and state."""
    personality = state.get("personality", {})
    character_state = state.get("character_state", {})
    affinity = state.get("affinity", 500)
    history = state.get("conversation_history", [])
    assembler_output = state.get("assembler_output", {})
    persona_name = personality.get("name", "assistant")

    affinity_block = _build_affinity_block(affinity)
    topics_to_cover = _build_topics_to_cover(
        plan.get("topics_to_cover", []),
        assembler_output.get("user_topic", "Unknown"),
    )
    facts_to_cover = _build_facts_to_cover(
        plan.get("facts_to_cover", []),
    )
    return {
        "personality": _build_personality_block(personality),
        "user_input_brief": {
            "channel_topic": assembler_output.get("channel_topic", "Unknown"),
            "user_topic": assembler_output.get("user_topic", "Unknown"),
            "intent_summary": _build_intent_summary(
                assembler_output.get("channel_topic", "Unknown"),
                assembler_output.get("user_topic", "Unknown"),
                topics_to_cover,
            ),
        },
        "response_brief": {
            "should_respond": True,
            "response_goal": _build_response_goal(topics_to_cover, facts_to_cover),
            "response_language": plan.get("response_language", "Match the user's current language."),
            "tone_guidance": plan.get("emotion_directive", "Standard in-character tone."),
            "relationship_guidance": affinity_block["instruction"],
            "state_guidance": _build_state_guidance(character_state),
            "continuity_summary": _build_continuity_summary(history, persona_name),
            "topics_to_cover": topics_to_cover,
            "facts_to_cover": facts_to_cover,
            "unknowns_or_limits": _build_unknowns_or_limits(agent_results),
        },
    }


async def persona_supervisor(state: BotState) -> dict:
    """Plan, dispatch, and evaluate agents in a loop.

    Writes ``supervisor_plan``, ``agent_results``, and ``speech_brief`` to state.
    """
    message_text = state.get("message_text", "")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "unknown_user_id")
    assembler_output = state.get("assembler_output", {})
    agent_results: list[AgentResult] = []

    # ── Step 0: Check relevance from Assembler ──────────────────────
    logger.info("Supervisor Step 0: Checking relevance from assembler")
    should_respond = assembler_output.get("should_respond", True)
    if not should_respond:
        logger.info("Assembler indicates no response needed. Short-circuiting.")
        plan = SupervisorPlan(
            agents=[],
            instructions={},
            response_language="N/A",
            topics_to_cover=[],
            facts_to_cover=[],
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
    persona_name = personality.get("name", "assistant")
    bot_id = state.get("bot_id", "unknown_bot_id")

    catalog = _build_agent_catalog()

    # ── Step 1: Initial planning LLM call ───────────────────────────
    logger.info("Supervisor Step 1: Initial planning LLM call")
    system_content = _PLANNING_SYSTEM.format(
        agent_catalog=catalog,
        persona_name=persona_name,
        bot_id=bot_id,
        current_time=state.get("timestamp", "")
    )

    human_data = {
        "current_message": {
            "speaker": user_name,
            "speaker_id": user_id,
            "message": message_text
        },
        "context": {
            "channel_topic": assembler_output.get("channel_topic", "Unknown"),
            "user_topic": assembler_output.get("user_topic", "Unknown"),
            "user_memory": user_memory,
        }
    }

    human_content = json.dumps(human_data, indent=2, ensure_ascii=False)

    planning_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    try:
        llm = _get_llm()
        logger.debug(
            "LLM input for Persona Supervisor:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in planning_messages)
        )
        result = await llm.ainvoke(planning_messages)
        plan = _parse_plan(result.content or "")
    except Exception:
        logger.exception("Supervisor planning LLM call failed")
        plan = SupervisorPlan(
            agents=[],
            instructions={},
            response_language="Match the user's current language.",
            topics_to_cover=["Respond directly to the user's latest message."],
            facts_to_cover=[],
            emotion_directive="Standard in-character tone.",
        )

    logger.info("Supervisor plan: agents=%s", plan["agents"])

    # ── Step 2: Dispatch initial agents ─────────────────────────────
    logger.info("Supervisor Step 2: Dispatching initial agents")
    for agent_name in plan["agents"]:
        instruction = plan.get("instructions", {}).get(agent_name, {})
        command = instruction.get("command", "") if isinstance(instruction, dict) else ""
        expected_response = instruction.get("expected_response", "") if isinstance(instruction, dict) else ""
        agent_result = await _dispatch_agent(agent_name, state, message_text, command, expected_response)
        agent_results.append(agent_result)

    # ── Step 3: Evaluate-dispatch loop ──────────────────────────────
    logger.info("Supervisor Step 3: Evaluate-dispatch loop")
    if agent_results:
        for iteration in range(MAX_SUPERVISOR_ITERATIONS):
            action = await _evaluate_results(
                plan, agent_results, message_text, persona_name, bot_id, catalog, state
            )

            next_action = action.get("action", "finish")
            if next_action == "finish":
                logger.info("Supervisor evaluate: finishing (iteration %d). Reason: %s",
                             iteration, action.get("reason", ""))
                break

            target_agent = action.get("agent", "")
            new_instruction = action.get("instruction", {})
            command = new_instruction.get("command", "") if isinstance(new_instruction, dict) else ""
            expected_response = new_instruction.get("expected_response", "") if isinstance(new_instruction, dict) else ""

            if next_action == "retry":
                logger.info("Supervisor evaluate: retrying %s (iteration %d). Reason: %s",
                             target_agent, iteration, action.get("reason", ""))
            elif next_action == "escalate":
                logger.info("Supervisor evaluate: escalating to %s (iteration %d). Reason: %s",
                             target_agent, iteration, action.get("reason", ""))

            # Update plan tracking
            if target_agent not in plan["agents"]:
                plan["agents"].append(target_agent)
            plan["instructions"][target_agent] = AgentInstruction(
                command=command,
                expected_response=expected_response,
            )

            agent_result = await _dispatch_agent(target_agent, state, message_text, command, expected_response)
            agent_results.append(agent_result)
        else:
            logger.warning("Supervisor evaluate loop hit max iterations (%d)", MAX_SUPERVISOR_ITERATIONS)

    # ── Step 4: Memory check — store noteworthy information ────────
    logger.info("Supervisor Step 4: Memory check - storing noteworthy information")
    memory_result = await _check_and_store_memory(
        agent_results, message_text, persona_name, bot_id, state,
    )
    if memory_result is not None:
        agent_results.append(memory_result)

    # ── Step 5: Synthesize agent results into speech-ready key points ─
    logger.info("Supervisor Step 5: Synthesizing agent results into speech-ready key points")
    if agent_results:
        await _synthesize_results(
            plan, agent_results, message_text, persona_name, bot_id, state
        )

    # ── Step 6: Build speech brief ──────────────────────────────────
    logger.info("Supervisor Step 6: Building speech brief")
    speech_brief = _build_speech_brief(plan, agent_results, state)

    return {
        "supervisor_plan": plan,
        "agent_results": agent_results,
        "speech_brief": speech_brief,
    }
