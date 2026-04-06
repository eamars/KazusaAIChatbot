from __future__ import annotations

from typing import Any, TypedDict


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


class AgentResult(TypedDict, total=False):
    agent: str             # agent name, e.g. "web_search_agent"
    status: str            # agent-authored response code, e.g. "success", "needs_context", "needs_clarification", "error"
    summary: str           # condensed output for speech_agent
    tool_history: list[ToolCall]  # tool calls made by this agent


class AgentInstruction(TypedDict, total=False):
    command: str
    expected_response: str


class SupervisorAction(TypedDict, total=False):
    """Decision from the supervisor's evaluate step."""
    action: str            # "finish" | "retry" | "escalate"
    agent: str             # target agent for retry/escalate
    instruction: AgentInstruction  # new instruction for retry/escalate
    reason: str            # why the supervisor chose this action


class SupervisorPlan(TypedDict):
    agents: list[str]      # which agents to invoke, e.g. ["web_search_agent"]
    instructions: dict[str, AgentInstruction]
    response_language: str
    topics_to_cover: list[str]
    facts_to_cover: list[str]
    emotion_directive: str # how to generate the output (emotion, tone, style)


class AssemblerOutput(TypedDict):
    channel_topic: str
    user_topic: str
    should_respond: bool
    reason_to_respond: str
    use_reply_feature: bool


class UserInputBrief(TypedDict, total=False):
    channel_topic: str
    user_topic: str
    intent_summary: str


class ResponseBrief(TypedDict, total=False):
    should_respond: bool
    response_goal: str
    response_language: str
    tone_guidance: str
    relationship_guidance: str
    state_guidance: str
    continuity_summary: str
    topics_to_cover: list[str]
    facts_to_cover: list[str]
    unknowns_or_limits: list[str]


class SpeechBrief(TypedDict, total=False):
    personality: dict[str, Any]
    user_input_brief: UserInputBrief
    response_brief: ResponseBrief


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

    # --- Stage 2: relevance_agent context loading ---
    conversation_history: list[dict[str, str]]
    user_memory: list[str]
    character_state: CharacterState
    affinity: int  # 0–1000 affinity score toward current user

    # --- Stage 2: relevance_agent analysis ---
    assembler_output: AssemblerOutput
    use_reply_feature: bool

    # --- Stage 3: persona_supervisor ---
    supervisor_plan: SupervisorPlan
    agent_results: list[AgentResult]
    speech_brief: SpeechBrief
    supervisor_chain_of_thought: list[dict]

    # --- Stage 4: speech_agent ---
    response: str

    # --- Stage 5: memory writer ---
    new_facts: list[str]

    # --- metadata ---
    personality: dict
