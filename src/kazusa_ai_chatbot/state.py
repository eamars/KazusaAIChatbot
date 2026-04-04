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


class AgentResult(TypedDict):
    agent: str             # agent name, e.g. "web_search_agent"
    status: str            # "success" | "error"
    summary: str           # condensed output for speech_agent
    tool_history: list[ToolCall]  # tool calls made by this agent


class SupervisorPlan(TypedDict):
    agents: list[str]      # which agents to invoke, e.g. ["web_search_agent"]
    content_directive: str # what information to include in the output
    emotion_directive: str # how to generate the output (emotion, tone, style)


class AssemblerOutput(TypedDict):
    channel_topic: str
    user_topic: str
    should_respond: bool


class BotState(TypedDict, total=False):
    # --- Stage 1: intake ---
    user_id: str
    user_name: str
    channel_id: str
    guild_id: str
    bot_id: str  # the bot's own Discord user ID (for mention filtering)
    message_text: str
    timestamp: str
    should_respond: bool

    # --- Stage 3: RAG ---
    rag_results: list[RagResult]

    # --- Stage 4: memory ---
    conversation_history: list[ChatMessage]
    user_memory: list[str]
    character_state: CharacterState
    affinity: int  # 0–1000 affinity score toward current user

    # --- Stage 5: relevance_agent ---
    assembler_output: AssemblerOutput

    # --- Stage 6a: persona_supervisor ---
    supervisor_plan: SupervisorPlan
    agent_results: list[AgentResult]
    speech_human_data: dict

    # --- Stage 6b: speech_agent ---
    response: str

    # --- Stage 7: memory writer ---
    new_facts: list[str]

    # --- metadata ---
    personality: dict
