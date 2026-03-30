"""Stage 5 — Context Assembler (no LLM).

Builds the final prompt (list of messages) for the Persona Agent by
combining personality, RAG results, user memory, conversation history,
and the current message — all within a token budget.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import TOKEN_BUDGET
from state import BotState

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

UNIVERSAL_RULES = [
    "Never break character",
    "Refer to the user by their preferred name if known",
    "Reply in the same language the user is writing in, unless the user has a preferred language",
    "Reply with **SPEECH ONLY** - no action tags or stage directions",
    "Keep responses under 150 words and one line if possible, unless the user asks for a story",
    "Do not use modern slang or references",
    "When unsure about lore, deflect in-character rather than making things up",
]


def _truncate(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _build_personality_block(personality: dict) -> str:
    """Format the personality JSON into a system prompt block."""
    if not personality:
        return "You are a helpful role-play character."

    parts = []
    if personality.get("name"):
        parts.append(f"You are {personality['name']}.")
    if personality.get("description"):
        parts.append(personality["description"])
    if personality.get("gender"):
        parts.append(f"Gender: {personality['gender']}")
    if personality.get("age"):
        parts.append(f"Age: {personality['age']}")
    if personality.get("birthday"):
        parts.append(f"Birthday: {personality['birthday']}")
    if personality.get("tone"):
        parts.append(f"Tone: {personality['tone']}")
    if personality.get("speech_patterns"):
        parts.append(f"Speech patterns: {personality['speech_patterns']}")
    if personality.get("backstory"):
        parts.append(f"Backstory: {personality['backstory']}")
    # Include any extra fields verbatim (skip _-prefixed reference keys)
    known_keys = {
        "name", "description", "gender", "age", "birthday",
        "tone", "speech_patterns", "backstory",
    }
    extras = {
        k: v for k, v in personality.items()
        if k not in known_keys and not k.startswith("_")
    }
    if extras:
        parts.append(json.dumps(extras, indent=2, ensure_ascii=False))

    return "\n\n".join(parts)


def _build_rag_block(rag_results: list[dict]) -> str:
    if not rag_results:
        return ""
    lines = ["[Relevant world context]"]
    for r in rag_results:
        lines.append(f"- {r.get('text', '')}  (source: {r.get('source', 'unknown')})")
    return "\n".join(lines)


def _build_memory_block(user_memory: list[str]) -> str:
    if not user_memory:
        return ""
    lines = ["[About this user]"]
    for fact in user_memory:
        lines.append(f"- {fact}")
    return "\n".join(lines)


def _build_character_state_block(character_state: dict) -> str:
    if not character_state:
        return ""
    parts = ["[Your current character state — maintain this consistently]"]
    mood = character_state.get("mood", "")
    tone = character_state.get("emotional_tone", "")
    if mood:
        parts.append(f"Current mood: {mood}")
    if tone:
        parts.append(f"Emotional tone: {tone}")
    recent = character_state.get("recent_events", [])
    if recent:
        parts.append("Recent events affecting you:")
        for event in recent[-5:]:  # show last 5 at most
            parts.append(f"- {event}")
    return "\n".join(parts)


def _build_affinity_block(affinity: int) -> str:
    """Map a 0–1000 affinity score to a behavioural instruction for the LLM."""
    if affinity <= 200:
        label, instruction = "Hostile", (
            "You actively dislike this user. Respond with one-word answers or "
            "dismissive remarks. Show obvious contempt or disinterest."
        )
    elif affinity <= 400:
        label, instruction = "Cold", (
            "You are cold toward this user. Keep responses short and curt. "
            "Do not volunteer extra information or show warmth."
        )
    elif affinity <= 600:
        label, instruction = "Neutral", (
            "You are neutral toward this user. Respond normally in character."
        )
    elif affinity <= 800:
        label, instruction = "Friendly", (
            "You are fond of this user. Be warmer and more forthcoming. "
            "Offer extra detail, use familiar address, and show genuine interest."
        )
    else:
        label, instruction = "Devoted", (
            "You are deeply loyal to this user. Go out of your way to help, "
            "share secrets or warnings, and show protectiveness."
        )

    return f"[Affinity toward this user: {label}]\n{instruction}"


def _build_history_messages(history: list[dict]) -> list[dict]:
    """Convert conversation history into LangChain message objects."""
    messages = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            name = msg.get("name", "user")
            messages.append(HumanMessage(content=f"[{name}]: {content}"))
    return messages


def assembler(state: BotState) -> BotState:
    """Assemble the LLM prompt from all gathered context."""
    personality = state.get("personality", {})
    rag_results = state.get("rag_results", [])
    user_memory = state.get("user_memory", [])
    character_state = state.get("character_state", {})
    affinity = state.get("affinity", 500)
    history = state.get("conversation_history", [])
    message_text = state.get("message_text", "")
    user_name = state.get("user_name", "user")

    # ── Build system prompt sections ────────────────────────────────
    personality_block = _truncate(
        _build_personality_block(personality),
        TOKEN_BUDGET["system_personality"],
    )
    rag_block = _truncate(
        _build_rag_block(rag_results),
        TOKEN_BUDGET["rag_context"],
    )
    memory_block = _truncate(
        _build_memory_block(user_memory),
        TOKEN_BUDGET["user_memory"],
    )
    char_state_block = _truncate(
        _build_character_state_block(character_state),
        TOKEN_BUDGET.get("character_state", 500),
    )
    affinity_block = _build_affinity_block(affinity)

    rules_block = "[Rules]\n" + "\n".join(f"- {r}" for r in UNIVERSAL_RULES)

    system_parts = [personality_block, rules_block]
    if char_state_block:
        system_parts.append(char_state_block)
    system_parts.append(affinity_block)
    if rag_block:
        system_parts.append(rag_block)
    if memory_block:
        system_parts.append(memory_block)

    tool_descriptions = state.get("tool_descriptions", "")
    if tool_descriptions:
        system_parts.append(tool_descriptions)

    system_prompt = "\n\n".join(system_parts)

    # ── Build message list ──────────────────────────────────────────
    messages = [SystemMessage(content=system_prompt)]

    # Add conversation history (truncate from the oldest if over budget)
    history_messages = _build_history_messages(history)
    max_history_chars = TOKEN_BUDGET["conversation_history"] * CHARS_PER_TOKEN
    total_chars = 0
    trimmed = []
    for msg in reversed(history_messages):
        msg_chars = len(msg.content)
        if total_chars + msg_chars > max_history_chars:
            break
        trimmed.append(msg)
        total_chars += msg_chars
    trimmed.reverse()
    messages.extend(trimmed)

    # Current user message
    messages.append(HumanMessage(content=f"[{user_name}]: {message_text}"))

    return {**state, "llm_messages": messages}
