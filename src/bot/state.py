from __future__ import annotations

from typing import Any, TypedDict


class RagResult(TypedDict):
    text: str
    source: str
    score: float


class ChatMessage(TypedDict):
    role: str        # "user" | "assistant"
    user_id: str     # unique identifier (Discord user/bot ID)
    name: str        # display name
    content: str


class CharacterState(TypedDict, total=False):
    mood: str              # e.g. "melancholic", "playful", "irritated"
    emotional_tone: str    # e.g. "warm", "guarded", "teasing"
    recent_events: list[str]  # short summaries of recent notable interactions
    updated_at: str        # ISO timestamp of last update


class ToolCall(TypedDict):
    tool: str              # fully-qualified tool name (e.g. "mcp-searxng__search")
    args: dict[str, Any]   # arguments passed to the tool
    result: str            # text result returned by the tool


class BotState(TypedDict, total=False):
    # --- Stage 1: intake ---
    user_id: str
    user_name: str
    channel_id: str
    guild_id: str
    message_text: str
    timestamp: str
    should_respond: bool

    # --- Stage 2: router ---
    retrieve_rag: bool
    retrieve_memory: bool
    rag_query: str

    # --- Stage 3: RAG ---
    rag_results: list[RagResult]

    # --- Stage 4: memory ---
    conversation_history: list[ChatMessage]
    user_memory: list[str]
    character_state: CharacterState
    affinity: int  # 0–1000 affinity score toward current user

    # --- Stage 5: assembler ---
    llm_messages: list[dict]
    tool_descriptions: str  # prompt block listing available tools

    # --- Stage 6: persona (supervisor) ---
    response: str
    tool_history: list[ToolCall]  # tools called during this turn

    # --- Stage 7: memory writer ---
    new_facts: list[str]

    # --- metadata ---
    personality: dict
