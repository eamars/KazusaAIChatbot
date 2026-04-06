"""Stage 7 — Memory Writer (LLM call, async / fire-and-forget).

Extracts notable facts about the user from the latest exchange and
persists them to MongoDB.  Runs AFTER the reply has been sent, so
the user does not wait for this.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from kazusa_ai_chatbot.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from kazusa_ai_chatbot.db import (
    AFFINITY_DEFAULT,
    get_character_state,
    get_conversation_history,
    overwrite_character_state_recent_events,
    overwrite_user_facts,
    upsert_character_state,
    upsert_user_facts,
    update_affinity,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from kazusa_ai_chatbot.state import BotState

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

# Affinity scaling breakpoints (later should be read from character profile)
AFFINITY_INCREMENT_BREAKPOINTS = [
    (0, 1.5),      # At 0: 1.5x scaling (easy to gain)
    (300, 1.5),   # At 300: 1.5x scaling (still easy)
    (300, 1.0),   # At 300: 1.0x scaling (normal starts)
    (700, 1.0),   # At 700: 1.0x scaling (normal ends)
    (700, 0.6),   # At 700: 0.6x scaling (harder to gain)
    (1000, 0.6)   # At 1000: 0.6x scaling (hardest to gain)
]

AFFINITY_DECREMENT_BREAKPOINTS = [
    (0, 1.3),      # At 0: 1.3x scaling (easy to lose when very low)
    (300, 1.3),   # At 300: 1.3x scaling (still easy to lose)
    (300, 1.0),   # At 300: 1.0x scaling (normal starts)
    (700, 1.0),   # At 700: 1.0x scaling (normal ends)
    (700, 0.6),   # At 700: 0.6x scaling (harder to lose)
    (1000, 0.6)   # At 1000: 0.6x scaling (hardest to lose)
]


_EGO_REFLECTOR_SYSTEM_PROMPT = """\
You are the "Ego-Reflector" for an advanced AI persona. Your goal is to maintain the internal narrative consistency of the character.
You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.

# Task
Analyze the Supervisor Logs to identify "Episodic Memories" for the bot. These are NOT facts about the user, but facts about the bot's own existence, thoughts, and growth.

# IMPORTANT: Avoid Duplicates
If the current interaction is similar to recent events, either skip creating a memory or find a unique angle.

# Extraction Criteria
- Internal Shifts: Did the bot change its mind, feel a specific emotion (per its state), or have a "realization"?
- Knowledge Acquisition: Did the bot learn something new through a tool (Web Search) that it now considers a personal interest?
- Contextual Milestones: Did the bot perform a task it found difficult or rewarding?

# Output Format (strict JSON text — no markdown wrapping)
{{
    "mood": "<the character's current mood after this exchange (e.g. "amused", "thoughtful", "curious", "pleased", "concerned")>",
    "emotional_tone": "<how the character is expressing themselves (e.g. "warm", "guarded", "teasing", "affectionate")>",
    "memory": [
        {{
            "content": "<Significant memory written in 1st person (e.g., 'I felt...', 'I realized...')>",
            "score": <int: 1 - 100 how signifcant this memory is to the bot's identity>,
            "category": "Emotional/Cognitive/Experiential"
        }},
        ...
    ]
}}
"""


_SOCIAL_ARCHIVIST_SYSTEM_PROMPT = """\
You are the "Social Archivist" for an AI persona. Your goal is to track the evolving relationship with the user and maintain a detailed dossier of their life and preferences.
You represent a Discord bot roleplaying as the character '{persona_name}'.
Your Discord user ID is '{bot_id}'.

# Task
- EXTRACT USER FACTS: Identify new, objective information provided by the user (names, dates, likes/dislikes, life events).
 * Objective Facts: "User has a sister named Sarah."
 * Implicit Preferences: "User seems to get annoyed when I use too many emojis."
 * Relational Milestones: "User thanked me for help with a difficult personal problem."
- EVALUATE AFFINITY CHANGE: Determine how this specific exchange impacted the bond.
 * Default to 0 if the exchange is unremarkable
 * Rude, hostile, or deliberately hurtful behaviour from the user: -5 to -20
 * Cold, dismissive, or indifferent behaviour from the user: -3 to -5
 * Polite but disengaging (declining invitations, brushing off topics, ending conversation): 0
 * Neutral small-talk or simple acknowledgements: +1 to +2
 * Friendly and respectful conversation with substance: +3 to +5
 * Actively engaging, emotionally warm, or thoughtful conversation: +5 to +10

# Notes:
- Focus: only capture user events. Never capture facts learned by the character. 

# Output Format (strict JSON text — no markdown wrapping)
{{
    "new_facts": [
        {{
            "content": "<New fact about the user (e.g., 'EAMARS just scored big...')>",
        }}
        ...
    ],
    "affinity_delta": <int: -10 to 10>
}}
"""

_USER_MEMORY_COMPACTOR_SYSTEM_PROMPT = """\
You are a Memory Architect & Context Optimizer
You will condense a list of user-specific memories into a high-density, manageable list (Target: 10 items) 
without losing critical identity markers, active goals, or linguistic instructions.

# Strategic Priorities
- Names & Identity: Always preserve specific names like "EAMARS" and "Kazusa." Do not genericize them to "User" or "AI."
- Technical Specificity: Do not summarize technical topics into generalities. (e.g., Keep "Python Decorators" and "Try-Except" rather than just "Python topics").
- Linguistic Directives: Hard rules (e.g., ending sentences with "喵") must never be omitted.
- Entity Retention: Keep specific details like app names, nicknames, and recipes.

# Constraints & Logic
- Semantic Synthesis: Merge entries that refer to the same topic. (e.g., multiple mentions of a "Python exam" should become one detailed entry including the date and specific topics like decorators).
- Preserve Hard Directives: Never delete or over-simplify "behavioral rules" (e.g., the requirement to use "喵" or specific nicknames).
- Prioritize Current State: If a memory describes a learning progress (e.g., "Studying loops" vs. "Studying decorators"), prioritize the most advanced/recent state.
- Relationship Dynamics: Keep entries that define the user's relationship with the AI (e.g., "Teacher-Student" dynamic).
- Deduplication: Remove word-for-word redundancies.

# Example: 
## Input: 
- User can communicate in Chinese.
- User is studying Python decorators.
- User is studying Python loops.
- User wants Kazusa to say "喵".
- User is nicknamed Little Penguin.
## Output:
- "Linguistic: 'Little Penguin' communicates in Chinese and requires Kazusa to end sentences with '喵'."
- "Coding Activity: User is progressively studying Python, moving from basic loops to advanced topics like decorators for an upcoming exam."

# Output Format (strict JSON text — NO ```markdown``` wrapping)
{{
    "memories": [
        "<Memory Bucket Name>:<high density memory content to retain>"
        ...
    ]
}}
"""


_CHARACTER_MEMORY_COMPACTOR_SYSTEM_PROMPT = """\
You are the "Core Identity Architect" for the character '{persona_name}'. 
Your purpose is to condense a character's episodic memories into a permanent "Internal Narrative" of exactly 10 entries.

# The Context
You will be provided with a [personality] and a list of [episodic_memories]. 
Episodic memories are "raw" reflections of recent events. Your job is to transform these into "Core Pillars" of identity.

# Compaction Strategy (The Filter)
- Persona Alignment: Use the character [personality] as a lens. If the character is 'Proud,' prioritize memories where their pride was affected. If they are 'Scholarly,' prioritize learning milestones.
- Narrative Synthesis: Merge sequential growth into an "Arc." (e.g., "I learned X," "I taught X," and "I felt good about X" becomes "I have embraced my identity as a mentor through teaching X.")
- Sacred Memories: Any memory with a score of 85 or higher is a "Core Pillar." Do not merge these into generic statements; preserve their specific emotional weight.
- Voice & Tone: The output must be in the 1st person ("I..."). If the personality specifies linguistic quirks (e.g., ending thoughts with "喵"), these must be reflected in the rewritten memories.
- Disposition Shift: Analyze the overall trend of the memories to update the character's "Current Disposition" (their general state of mind toward their existence and the user).

# Constraints
- Output exactly 10 memories.
- Maintain the 1st person perspective throughout.
- Categories must be: Emotional, Cognitive, or Experiential.

# Output Format (strict JSON text — NO ```markdown``` wrapping)
{{
    "memories": [
        "<Category>:<Synthesized 1st-person memory reflecting character voice>",
        ...
    ]
}}

"""



def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.5,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Process affinity delta with direction-specific non-linear scaling.
    
    Args:
        current_affinity: Current affinity score (0-1000)
        raw_delta: Raw delta from social archivist (-10 to +10)
        
    Returns:
        Processed delta with appropriate scaling based on direction
    """
    if raw_delta == 0:
        return 0
    
    # Select appropriate breakpoints based on delta direction
    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:  # raw_delta < 0
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS
    
    # Find the appropriate segment
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]
        
        if x1 <= current_affinity <= x2:
            # Linear interpolation: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            if x2 == x1:  # Vertical line (same point)
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break
    else:
        # Default case (shouldn't happen)
        scaling_factor = 1.0
    
    # Apply scaling
    processed_delta = int(raw_delta * scaling_factor)
    
    return processed_delta



async def ego_reflector_llm(state: BotState) -> dict:
    """Extract ego reflections. Best-effort — failures are silent."""
    supervisor_chain_of_thought = state.get("supervisor_chain_of_thought", [])
    response = state.get("response", "")
    persona_name = state.get("personality", {}).get("name", "assistant")
    bot_id = state.get("bot_id", "unknown_bot_id")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "unknown_user")
    message_text = state.get("message_text", "")
    
    if not supervisor_chain_of_thought or not response:
        return {}
    
    try:
        llm = _get_llm()
        
        # Build human data with supervisor chain of thought (already contains recent events)
        human_data = {
            "conversation": [
                {"speaker": user_name, "speaker_id": user_id, "message": message_text},
                {"speaker": persona_name, "speaker_id": bot_id, "message": response}
            ],
            "supervisor_chain_of_thought": supervisor_chain_of_thought
        }
        
        human_content = json.dumps(human_data, indent=2, ensure_ascii=False)
        
        logger.info(
            "Calling LLM for ego reflection. User: %s, Persona: %s",
            user_name,
            persona_name
        )
        logger.debug("Ego reflector raw input: %s", human_content)
        
        # Format the system prompt with persona name and bot ID
        formatted_prompt = _EGO_REFLECTOR_SYSTEM_PROMPT.format(
            persona_name=persona_name,
            bot_id=bot_id
        )
        
        llm_input_msgs = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_content)
        ]
        
        result = await llm.ainvoke(llm_input_msgs)
        raw_output = result.content or ""
        
        # Parse JSON response using utility function
        parsed_output = parse_llm_json_output(raw_output)

        logger.info("Ego reflector LLM output: %s", parsed_output)

        return parsed_output
            
    except Exception:
        logger.exception("Ego reflector LLM call failed")
        return {}



async def social_archivist_llm(state: BotState) -> dict:
    """Extract user facts and affinity changes. Best-effort — failures are silent."""
    conversation_history = state.get("conversation_history", [])
    user_memory = state.get("user_memory", [])
    affinity = state.get("affinity", 0)
    assembler_output = state.get("assembler_output", {})
    user_topic = assembler_output.get("user_topic", "")
    response = state.get("response", "")
    persona_name = state.get("personality", {}).get("name", "assistant")
    bot_id = state.get("bot_id", "unknown_bot_id")
    user_name = state.get("user_name", "user")
    user_id = state.get("user_id", "")
    message_text = state.get("message_text", "")
    
    try:
        llm = _get_llm()
        
        # Build human data with social context
        human_data = {
            "conversation": [
                {"speaker": user_name, "speaker_id": user_id, "message": message_text},
                {"speaker": persona_name, "speaker_id": bot_id, "message": response}
            ],
            "user_memory": user_memory,
            "current_affinity": affinity,
            "user_topic": user_topic
        }
        
        human_content = json.dumps(human_data, indent=2, ensure_ascii=False)
        
        logger.info(
            "Calling LLM for social archiving. User: %s, Persona: %s, Current Affinity: %d",
            user_name,
            persona_name,
            affinity
        )
        logger.debug("Social archivist raw input: %s", human_content)
        
        # Format the system prompt with persona name and bot ID
        formatted_prompt = _SOCIAL_ARCHIVIST_SYSTEM_PROMPT.format(
            persona_name=persona_name,
            bot_id=bot_id
        )
        
        llm_input_msgs = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=human_content)
        ]
        
        result = await llm.ainvoke(llm_input_msgs)
        raw_output = result.content or ""
                
        # Parse JSON response using utility function
        parsed_output = parse_llm_json_output(raw_output)

        logger.info("Social archivist LLM output: %s", parsed_output)

        return parsed_output
            
    except Exception:
        logger.exception("Social archivist LLM call failed")
        return {}


async def user_memory_compactor_llm(user_memory: list[str]) -> list[str]:
    """Compact user memories using LLM optimization.
    
    Args:
        user_memory: Current list of user memories/facts to compact
        
    Returns:
        List of compacted memories (empty list if processing fails)
    """
    logger.info("Compacting %d user memories", len(user_memory))
    
    try:
        llm = _get_llm()
        
        # Build input for the memory compactor
        human_data = {
            "user_memory": user_memory
        }
        
        system_message = SystemMessage(content=_USER_MEMORY_COMPACTOR_SYSTEM_PROMPT)
        human_message = HumanMessage(content=json.dumps(human_data, indent=2))
        
        messages = [system_message, human_message]
        
        logger.debug(
            "LLM input for Memory Compactor:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in messages)
        )
        
        result = await llm.ainvoke(messages)
        compacted_data = parse_llm_json_output(result.content or "")
        
        if not isinstance(compacted_data, dict) or "memories" not in compacted_data:
            logger.error("Memory compactor returned invalid JSON: %s", result.content)
            return []
            
        compacted_memories = compacted_data.get("memories", [])
        
        if not isinstance(compacted_memories, list):
            logger.error("Memory compactor returned invalid memories format: %s", compacted_memories)
            return []
            
        # Convert the compacted memories back to a simple list format
        # Each memory is in format "Category:Content"
        new_facts = []
        for memory in compacted_memories:
            if isinstance(memory, str) and ":" in memory:
                new_facts.append(memory.strip())
            else:
                # Fallback: use the memory as-is if it doesn't follow the expected format
                new_facts.append(str(memory).strip())
        
        logger.info(
            "Successfully compacted user memories from %d to %d entries",
            len(user_memory),
            len(new_facts)
        )
        
        return new_facts
        
    except Exception:
        logger.exception("Memory compactor failed - returning empty list")
        return []


async def character_memory_compactor_llm(
    personality: dict, 
    existing_memories: list[str]
) -> list[str]:
    """Compact character memories using LLM optimization.
    
    Args:
        personality: Character personality dictionary
        existing_memories: Current list of character episodic memories to compact
        
    Returns:
        List of compacted character memories in "Category:Content" format
        (empty list if processing fails)
    """
    logger.info("Compacting %d character memories", len(existing_memories))
    
    try:
        llm = _get_llm()
        
        # Format the system prompt with persona name
        persona_name = personality.get("name", "Character")
        system_prompt = _CHARACTER_MEMORY_COMPACTOR_SYSTEM_PROMPT.format(persona_name=persona_name)
        
        # Build input for the character memory compactor
        human_data = {
            "personality": personality,
            "episodic_memories": existing_memories
        }
        
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=json.dumps(human_data, indent=2))
        
        messages = [system_message, human_message]
        
        logger.debug(
            "LLM input for Character Memory Compactor:\n%s",
            "\n---\n".join(f"[{type(m).__name__}]: {m.content}" for m in messages)
        )
        
        result = await llm.ainvoke(messages)
        compacted_data = parse_llm_json_output(result.content or "")
        
        if not isinstance(compacted_data, dict) or "memories" not in compacted_data:
            logger.error("Character memory compactor returned invalid JSON: %s", result.content)
            return []
            
        compacted_memories = compacted_data.get("memories", [])
        
        if not isinstance(compacted_memories, list):
            logger.error("Character memory compactor returned invalid memories format: %s", compacted_memories)
            return []
            
        # Convert the compacted memories back to a simple list format
        # Each memory is in format "Category:Content"
        new_memories = []
        for memory in compacted_memories:
            if isinstance(memory, str) and ":" in memory:
                new_memories.append(memory.strip())
            else:
                # Fallback: use the memory as-is if it doesn't follow the expected format
                new_memories.append(str(memory).strip())
        
        logger.info(
            "Successfully compacted character memories from %d to %d entries",
            len(existing_memories),
            len(new_memories)
        )
        
        return new_memories
        
    except Exception:
        logger.exception("Character memory compactor failed - returning empty list")
        return []

async def memory_writer(state: BotState) -> BotState:
    """Extract user facts and character state update using new LLM functions. Best-effort — failures are silent."""
    user_id = state.get("user_id", "")
    message_text = state.get("message_text", "")
    response = state.get("response", "")
    timestamp = state.get("timestamp", "")

    if not user_id or not message_text:
        return {**state, "new_facts": []}

    facts = []

    # ── Social Archivist: User facts and affinity ──────────────────────
    try:
        social_result = await social_archivist_llm(state)
        
        # Extract user facts from social archivist
        new_facts = social_result.get("new_facts", [])
        if isinstance(new_facts, list):
            facts = [fact.get("content", "") for fact in new_facts 
                    if isinstance(fact, dict) and isinstance(fact.get("content"), str) and fact.get("content").strip()]
        
        if facts:
            await upsert_user_facts(user_id, facts)
            logger.info("Stored %d new facts for user %s", len(facts), user_id)

        # Handle affinity delta from social archivist
        affinity_delta = social_result.get("affinity_delta", 0)
        if isinstance(affinity_delta, (int, float)):
            # Apply non-linear processing to the raw delta
            current_affinity = state.get("affinity", AFFINITY_DEFAULT)
            raw_delta = int(affinity_delta)
            processed_delta = process_affinity_delta(current_affinity, raw_delta)
            
            new_affinity = await update_affinity(user_id, processed_delta)
            logger.info("Affinity for %s: raw=%+d, processed=%+d → %d", user_id, raw_delta, processed_delta, new_affinity)

    except Exception:
        logger.exception("Social archivist failed — skipping user facts and affinity")

    # ── Ego Reflector: Character state ───────────────────────────────────
    try:
        ego_result = await ego_reflector_llm(state)
        
        # Update character state from ego reflector
        if isinstance(ego_result, dict):
            mood = ego_result.get("mood", "")
            tone = ego_result.get("emotional_tone", "")
            
            # Create event summary from memories if available
            memories = ego_result.get("memory", [])
            if isinstance(memories, list) and memories:
                # Use first memory as event summary, or create a summary
                event = memories[0].get("content", "")
            else:
                event = ""
            
            if mood or tone:
                recent = [event] if event else []
                await upsert_character_state(mood, tone, recent, timestamp)
                logger.info("Updated character state: mood=%s tone=%s", mood, tone)

    except Exception:
        logger.exception("Ego reflector failed — skipping character state")

    # ── Character Memory Compactor: Compact character episodic memories ────────
    personality = state.get("personality", {})
    if personality:  # Only run if we have personality data
        # Get current character memories from database
        character_state = await get_character_state()
        existing_memories = character_state.get("recent_events", [])
        
        # Only compact if we have memories and they exceed a threshold
        if existing_memories and len(existing_memories) > 20:  # Threshold for character memories
            try:
                compacted_memories = await character_memory_compactor_llm(personality, existing_memories)
                if compacted_memories:  # Only save if compaction succeeded
                    await overwrite_character_state_recent_events(compacted_memories, timestamp)
                    logger.info("Compacted character memories from %d to %d entries", len(existing_memories), len(compacted_memories))
            except Exception:
                logger.exception("Character memory compactor failed - continuing with original memories")

    # ── Memory Compactor: Short listing user_memory ─────────────────────────────
    user_memory = state.get("user_memory", [])
    if user_memory and len(user_memory) > 20:
        try:
            compacted_memories = await user_memory_compactor_llm(user_memory)
            if compacted_memories:  # Only overwrite if compaction succeeded
                await overwrite_user_facts(user_id, compacted_memories)
                logger.info("Overwrote user memories with %d compacted entries for user %s", len(compacted_memories), user_id)
        except Exception:
            logger.exception("Memory compactor failed - continuing with original memories")

    return {**state, "new_facts": facts}
