# Role-Play Discord Chatbot

A Discord chatbot that responds in-character using a configurable personality (JSON), long-term user memory, channel conversation history, and **MCP tool calling** — built with **LangGraph**, **LM Studio** (local Qwen 3.5 27B), and **MongoDB**.

## Architecture

The bot uses a **lean hybrid pipeline**: most stages are pure code (no LLM), with **3–5 synchronous LLM calls** per message (1 relevance check + 1 supervisor planning + 0–N tool agents + 1 speech agent) and **1 deferred LLM call** after the reply is sent. All generative LLM calls use a native JSON structure for passing context rather than string-concatenated prompt blocks.

```
                    ┌─────────────┐
                    │   Discord    │
                    └──────┬──────┘
                           │ raw event
                    ┌──────▼──────┐
                    │  1. Intake   │  no LLM — mention filter
                    └──────┬──────┘
                           │ clean state (may exit early)
                    ┌──────▼──────────┐
                    │  2. Relevance   │  ★ LLM CALL + MongoDB context load
                    └──────┬──────────┘
                           │ conversation_history + user_memory + character_state + affinity + assembler_output
                    ┌──────▼──────────────┐
                    │3a. Supervisor        │
                    │  ┌────────────────┐  │
                    │  │ LLM Planning   │  │  ★ LLM CALL (which agents? content/emotion)
                    │  └───────┬────────┘  │
                    │  ┌────────────────┐  │
                    │  │ Agent Dispatch │  │  ★ LLM CALL × N (tool agents)
                    │  └────────────────┘  │
                    └──────────┬───────────┘
                           │ supervisor_plan + agent_results + speech_brief
                    ┌──────▼──────┐
                    │3b. Speech   │  ★ LLM CALL (in-character reply)
                    └──────┬──────┘
                           │ response (or silence)
                    ┌──────▼──────┐
                    │  Discord     │  send reply
                    └──────┬──────┘
                           │ fire-and-forget (user doesn't wait)
                    ┌──────▼──────┐
                    │ 4. Mem Write │  ★ LLM CALL (extract user facts)
                    └──────────────┘
```

### LangGraph Graph (compiled)

Stages 1–3 are wired as a LangGraph `StateGraph`. Stage 4 (Memory Writer) runs **outside** the graph as an async task after the reply is sent.

```
START → intake → relevance_agent → persona_supervisor → speech_agent → END
```

## Pipeline Stages

### Stage 1 — Intake (no LLM)
Normalises the raw Discord message: strips mention markup, extracts user/channel metadata, sets `should_respond` flag. Early exit if nothing to respond to.

**Mention filtering** — if the message mentions other Discord users but *not* the bot, `should_respond` is set to `False` immediately, short-circuiting the entire graph with zero LLM calls. Messages with no mentions pass through normally.

### Stage 2 — Relevance Agent (LLM call + context loading)
Before calling the LLM, the relevance agent loads conversational context directly from MongoDB:
- **Conversation history** — last N messages for the channel
- **User facts** — long-term memory extracted by previous Memory Writer runs (e.g., "User prefers to be called Commander")
- **Character state** — current mood and emotional tone persisted across exchanges
- **Affinity score** — per-user affinity (0–1000) that modulates the bot's warmth toward each user

It then determines if the bot should engage in the conversation by analyzing the current message plus that loaded context. It outputs a JSON structure with `channel_topic`, `user_topic`, and `should_respond`.

If the relevance agent returns `should_respond: false`, the downstream supervisor **short-circuits** with an empty plan and a "stay silent" directive — no planning LLM call, no agents dispatched, and the speech agent returns an empty response. This is a **fail-open** design: parse failures or LLM crashes default to responding.

### Stage 3a — Persona Supervisor (LLM calls)
The supervisor orchestrates the agent execution plan.

#### Phase 1 — Planning
If the message is relevant:

1. **Build a prompt** with the relevance agent's topic analysis and the available agent catalog.
2. **Call the planning LLM** to return a JSON `SupervisorPlan` containing:
   - `agents`: list of agent names to invoke (e.g. `["web_search_agent"]`, or `[]` if none needed)
   - `content_directive`: instruction for what factual info or topics to include in the reply
   - `emotion_directive`: instruction for what tone, mood, and style the speech agent should use
3. Execute each requested agent sequentially in isolated contexts.
4. Synthesize a sanitized `speech_brief` for the Speech Agent.
5. Write `supervisor_plan`, `agent_results`, and `speech_brief` to state.

Each agent runs in its own LLM context with only the information it needs. If an agent crashes or hits a context limit, the supervisor catches the error and records it as `AgentResult(status="error")`. The supervisor then converts internal context, memory, and tool output into a speech-safe brief rather than passing raw data to the speech agent.

#### Available Agents

| Agent | Description |
|-------|-------------|
| `web_search_agent` | Searches the internet via MCP search tools. Gets only the user query + search tool descriptions. Runs its own `<tool_call>` loop, then summarises raw results. |
| *(extensible)* | New agents are added by subclassing `BaseAgent` and calling `register_agent()` in `graph.py`. |

#### MCP Tool Calling

Tools are provided by external **MCP servers** (HTTP/SSE). At startup, the bot connects to all configured servers and discovers available tools. Tool names are namespaced as `{server}__{tool}` internally, but the original name is sent to the MCP server.

Configured via the `MCP_SERVERS` environment variable (JSON string):

```json
{"mcp-searxng": {"url": "http://host:4001/mcp"}, "playwright": {"url": "http://host:8931/mcp"}}
```

### Stage 3b — Speech Agent (LLM call)
Always runs last. Generates the final **in-character reply** from a native JSON HumanMessage `speech_brief` containing only:
- Approved `personality` context
- Supervisor-authored interpretation of user intent
- Response guidance, continuity summary, and key points to cover
- Allowed personalization hints and limits on what not to claim

The speech agent never receives raw conversation history, raw user memory, raw character state, raw affinity scores, or raw tool output. If the supervisor marks `should_respond: false` in the brief, the speech agent **returns an empty response** without making an LLM call.

The speech agent's LLM context is free of tool descriptions and internal state, keeping the token budget focused on personality and final expression quality. The supervisor is responsible for translating internal state into guidance such as relationship tone, continuity summary, and key points to cover.

### Stage 4 — Memory Writer (deferred LLM call)
Runs **after** the reply is sent to Discord (fire-and-forget `asyncio.create_task`). Extracts from each exchange using a native JSON HumanMessage payload:
- **User facts** — new facts about the user (e.g., preferred name)
- **Character state** — updated mood and emotional tone
- **Affinity delta** — integer (-20 to +10) indicating how the exchange changes the bot's feeling toward the user
- **Agent results** — if agents were invoked during the turn, their names, status, and summaries are included in the extraction prompt so the LLM can reason about them

All three are persisted to MongoDB. Affinity deltas are clamped to `[-20, +10]` by non-LLM code before applying. Best-effort — failures are logged and silently skipped.

## Project Structure

```
src/
  main.py                  # CLI entry point
  config.py                # env vars, token budgets, MCP_SERVERS
  state.py                 # BotState + AgentResult + SupervisorPlan TypedDicts
  db.py                    # MongoDB async helpers + schema (TypedDict)
  graph.py                 # LangGraph StateGraph wiring + agent registration
  discord_bot.py           # Discord client, message handling, MCP lifecycle
  mcp_client.py            # McpManager — MCP server connections + tool execution
  tools.py                 # build_tool_prompt_block() for agent prompt injection
  utils.py                 # Shared helpers (history formatting)
  agents/
    __init__.py
    base.py                # BaseAgent ABC + AGENT_REGISTRY
    speech_agent.py        # Stage 3b — in-character reply generation
    web_search_agent.py    # Web search via MCP (isolated context)
  nodes/
    intake.py              # Stage 1 (mention filtering)
    relevance_agent.py     # Stage 2 (context loading + relevance gate)
    persona_supervisor.py  # Stage 3a (LLM planning + agent dispatch)
    memory_writer.py       # Stage 4 (facts + character state + affinity + agent results)
personalities/
  example.json             # sample personality ("Zara")
  kazusa.json              # full personality with _reference section
.env.example               # environment variable template
requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```env
DISCORD_TOKEN=your_discord_bot_token
MONGODB_URI=mongodb://your_connection_string
LLM_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=your-embedding-model-name

# Optional: MCP tool servers (JSON string)
MCP_SERVERS={"mcp-searxng": {"url": "http://host:4001/mcp"}}
MAX_TOOL_ITERATIONS=3
```

### 3. LM Studio

Load two models in LM Studio:
- **Chat model** — e.g., Qwen 3.5 27B (used by Supervisor, Speech Agent, Tool Agents, and Memory Writer)
- **Embedding model** — e.g., nomic-embed-text (used for semantic search over stored conversation history and user facts)

### 4. MongoDB

Set up the following collections in your database:
- **`conversation_history`** — auto-populated by the bot, with embeddings for semantic history search utilities
- **`user_facts`** — auto-populated by the Memory Writer (facts + affinity score per user), with embeddings for semantic user-fact search utilities
- **`character_state`** — auto-populated by the Memory Writer (mood, tone, recent events)

### 5. Create a personality

Create a JSON file following the schema in `personalities/example.json`:

```json
{
    "name": "Character Name",
    "description": "Who this character is",
    "gender": "Female",
    "age": 25,
    "birthday": "March 15",
    "tone": "sardonic, loyal, terse",
    "speech_patterns": "How the character talks",
    "backstory": "Character history"
}
```

The assembler reads these known keys: `name`, `description`, `gender`, `age`, `birthday`, `tone`, `speech_patterns`, `backstory`. Any other top-level keys are passed through as JSON.

Keys prefixed with `_` (e.g., `_reference`) are **ignored** by the assembler — use this to store visual descriptions, appearance details, or other reference data that shouldn't be injected into the prompt. See `kazusa.json` for an example.

## Usage

```bash
# Default: listen in ALL channels
python src/kazusa_ai_chatbot/main.py --personality personalities/example.json

# Listen in specific channels only
python src/kazusa_ai_chatbot/main.py --personality personalities/example.json --channels 123456789 987654321

# Respond to @mentions only
python src/kazusa_ai_chatbot/main.py --personality personalities/example.json --no-listen-all

# With debug logging
python src/kazusa_ai_chatbot/main.py --personality personalities/example.json --log-level DEBUG
```

## Design Decisions

- **Context loading inside relevance agent** — conversation history, user facts, character state, and affinity are loaded in one place before relevance analysis. This keeps the graph linear and removes state merge complexity.
- **Memory Writer outside the graph** — the user sees the reply after 1 LLM call; fact extraction happens in the background.
- **Token budget system** — explicit allocation prevents context overflow on models with limited context windows (~32K for Qwen 3.5 27B).
- **Embeddings retained for internal search utilities** — semantic embeddings are still stored for `conversation_history` and `user_facts` so local search scripts and retrieval utilities can query those collections, but lore-based RAG retrieval is no longer part of the chatbot response pipeline.
- **Affinity system** — per-user 0–1000 score (default 500) that slowly shifts based on conversation quality. The LLM proposes a delta; non-LLM code clamps it. This avoids wild swings while letting the bot gradually warm up or cool down toward individual users.
- **Universal rules in relevance agent** — behavioural rules ("never break character", etc.) are hardcoded rather than per-personality, ensuring consistency across all characters.
- **`_`-prefixed keys ignored** — personality JSON can store reference data (appearance, art notes) under `_reference` without wasting prompt tokens.
- **Two-layer "should I reply?" system** — Layer 1 (intake) uses rule-based mention filtering (zero cost). Layer 2 (Relevance Agent) uses an LLM call to analyze the conversational context (topics, recent history) and decides if the bot should engage. Both are fail-open: if uncertain or if the LLM fails, the bot defaults to responding.
- **Supervisor + sub-agent architecture** — the persona supervisor consumes the relevance agent's context analysis and plans which specialist agents to invoke. This provides failure isolation (a crashed agent doesn't kill the reply) and context separation (tool agents don't bloat the speech prompt).
- **Sanitized speech boundary** — the supervisor sees internal state, memory, and agent output, but the speech agent only receives a distilled `speech_brief` with `personality`, interpreted user intent, response guidance, and approved key points. This reduces accidental leakage of hidden state and tool output.
- **Prompt-based `<tool_call>` tags** in tool agents — ensures compatibility with Qwen and other local models served via LM Studio that may not support the `tools` parameter.
- **MCP for tooling** — tools are served by external MCP servers over HTTP, making them language-agnostic and independently deployable. New agents can be added by subclassing `BaseAgent` without changing existing code.
