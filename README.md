# Role-Play Discord Chatbot

A Discord chatbot that responds in-character using a configurable personality (JSON), long-term user memory, RAG-powered world knowledge, and **MCP tool calling** — built with **LangGraph**, **LM Studio** (local Qwen 3.5 27B), and **MongoDB**.

## Architecture

The bot uses a **lean hybrid pipeline**: most stages are pure code (no LLM), with only **1–3 synchronous LLM calls** per message (1 base + up to 2 tool-calling iterations) and **1 deferred LLM call** after the reply is sent.

```
                    ┌─────────────┐
                    │   Discord    │
                    └──────┬──────┘
                           │ raw event
                    ┌──────▼──────┐
                    │  1. Intake   │  no LLM
                    └──────┬──────┘
                           │ clean state
                    ┌──────▼──────┐
                    │  2. Router   │  no LLM (keyword/regex)
                    └──────┬──────┘
                           │ retrieval flags
                    ┌──────┴──────┐
              ┌─────▼─────┐ ┌─────▼─────┐
              │ 3. RAG    │ │ 4. Memory │  no LLM, parallel
              └─────┬─────┘ └─────┬─────┘
                    └──────┬──────┘
                           │ rag_results + history + user_memory
                    ┌──────▼──────┐
                    │ 5. Assemble │  no LLM (prompt builder)
                    └──────┬──────┘
                           │ llm_messages
                    ┌──────▼──────┐
                    │ 6. Persona  │  ★ LLM CALL (supervisor + tool loop)
                    └──────┬──────┘
                           │ response
                    ┌──────▼──────┐
                    │  Discord     │  send reply
                    └──────┬──────┘
                           │ fire-and-forget (user doesn't wait)
                    ┌──────▼──────┐
                    │ 7. Mem Write │  ★ LLM CALL (extract user facts)
                    └──────────────┘
```

### LangGraph Graph (compiled)

Stages 1–6 are wired as a LangGraph `StateGraph`. Stage 7 (Memory Writer) runs **outside** the graph as an async task after the reply is sent.

```
START → intake → router →┬→ rag_retriever    ─┬→ assembler → persona_agent → END
                         └→ memory_retriever ─┘
```

## Pipeline Stages

### Stage 1 — Intake (no LLM)
Normalises the raw Discord message: strips mention markup, extracts user/channel metadata, sets `should_respond` flag. Early exit if nothing to respond to.

### Stage 2 — Router (no LLM)
Rule-based decision tree using keyword matching and message length heuristics. Sets `retrieve_rag` and `retrieve_memory` flags. No LLM call — avoids unreliable routing on smaller models.

### Stage 3 — RAG Retriever (no generative LLM)
Embeds the query via LM Studio's `/v1/embeddings` endpoint, then runs a MongoDB Atlas `$vectorSearch` against the `lore` collection. Returns top-K relevant chunks with scores. Runs **in parallel** with Stage 4.

### Stage 4 — Memory Retriever (no LLM)
Four MongoDB lookups (parallel with Stage 3):
- **Conversation history** — last N messages for the channel
- **User facts** — long-term memory extracted by previous Memory Writer runs (e.g., "User prefers to be called Commander")
- **Character state** — current mood and emotional tone persisted across exchanges
- **Affinity score** — per-user affinity (0–1000) that modulates the bot's warmth toward each user

### Stage 5 — Context Assembler (no LLM)
Builds the final prompt within a **token budget**:

| Section              | Budget    |
|----------------------|-----------|
| Personality (system) | ~15K tok  |
| Character state      | ~500 tok  |
| Universal rules      | (fixed)   |
| Affinity instruction | (fixed)   |
| RAG context          | ~2K tok   |
| User memory          | ~500 tok  |
| Conversation history | ~4K tok   |
| Current message      | ~500 tok  |

**Universal rules** (e.g., "Never break character", "Keep responses under 200 words") are hardcoded in the assembler and injected into every prompt regardless of personality.

**Affinity** maps the user's 0–1000 score to a behavioural label (Hostile / Cold / Neutral / Friendly / Devoted) and injects a tone instruction.

Truncates oldest history first, then lowest-scored RAG chunks if over budget.

### Stage 6 — Persona Agent (LLM supervisor)
The persona agent is an **inline LLM supervisor** with an optional tool-calling loop:

1. Call the LLM with the assembled messages (including tool descriptions in the system prompt).
2. If the response contains a `<tool_call>{"name": "...", "args": {...}}</tool_call>` tag, parse and execute the tool via MCP.
3. Append the tool result and re-invoke the LLM (up to `MAX_TOOL_ITERATIONS`, default 3).
4. Return the final text response (with any `<tool_call>` tags stripped).

This avoids the overhead of a separate routing LLM — the persona decides in-context whether it needs a tool, with **zero extra latency** on normal (no-tool) messages.

#### MCP Tool Calling

Tools are provided by external **MCP servers** (HTTP/SSE). At startup, the bot connects to all configured servers, discovers available tools, and generates a prompt block describing them. Tool names are namespaced as `{server}__{tool}` internally, but the original name is sent to the MCP server.

Configured via the `MCP_SERVERS` environment variable (JSON string):

```json
{"mcp-searxng": {"url": "http://host:4001/mcp"}, "playwright": {"url": "http://host:8931/mcp"}}
```

### Stage 7 — Memory Writer (deferred LLM call)
Runs **after** the reply is sent to Discord (fire-and-forget `asyncio.create_task`). Extracts from each exchange:
- **User facts** — new facts about the user (e.g., preferred name)
- **Character state** — updated mood and emotional tone
- **Affinity delta** — integer (-20 to +10) indicating how the exchange changes the bot's feeling toward the user
- **Tool history** — if tools were called during the turn, the tool names, arguments, and results are included in the extraction prompt so the LLM can reason about them

All three are persisted to MongoDB. Affinity deltas are clamped to `[-20, +10]` by non-LLM code before applying. Best-effort — failures are logged and silently skipped.

## Project Structure

```
src/
  main.py                  # CLI entry point
  config.py                # env vars, token budgets, MCP_SERVERS
  state.py                 # BotState TypedDict + ToolCall (shared graph state)
  db.py                    # MongoDB async helpers + schema (TypedDict)
  graph.py                 # LangGraph StateGraph wiring
  discord_bot.py           # Discord client, message handling, MCP lifecycle
  mcp_client.py            # McpManager — MCP server connections + tool execution
  tools.py                 # build_tool_prompt_block() for LLM prompt injection
  nodes/
    intake.py              # Stage 1
    router.py              # Stage 2
    rag.py                 # Stage 3
    memory.py              # Stage 4
    assembler.py           # Stage 5 (universal rules, affinity, tool descriptions)
    persona.py             # Stage 6 (supervisor loop with <tool_call> parsing)
    memory_writer.py       # Stage 7 (facts + character state + affinity + tool history)
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
- **Chat model** — e.g., Qwen 3.5 27B (used by Persona Agent and Memory Writer)
- **Embedding model** — e.g., nomic-embed-text (used by RAG Retriever)

### 4. MongoDB

Set up the following collections in your database:
- **`lore`** — world/knowledge documents with an `embedding` field and a [vector search index](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/)
- **`conversation_history`** — auto-populated by the bot
- **`user_facts`** — auto-populated by the Memory Writer (facts + affinity score per user)
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
python src/main.py --personality personalities/example.json

# Listen in specific channels only
python src/main.py --personality personalities/example.json --channels 123456789 987654321

# Respond to @mentions only
python src/main.py --personality personalities/example.json --no-listen-all

# With debug logging
python src/main.py --personality personalities/example.json --log-level DEBUG
```

## Design Decisions

- **Rule-based router** instead of LLM-based — smaller local models are unreliable at routing decisions; keyword/regex is fast and deterministic.
- **Parallel retrieval** — RAG and Memory run concurrently since they're independent DB lookups with no LLM.
- **Memory Writer outside the graph** — the user sees the reply after 1 LLM call; fact extraction happens in the background.
- **Each parallel node returns only its owned fields** — prevents state clobbering during LangGraph fan-out merge.
- **Token budget system** — explicit allocation prevents context overflow on models with limited context windows (~32K for Qwen 3.5 27B).
- **Affinity system** — per-user 0–1000 score (default 500) that slowly shifts based on conversation quality. The LLM proposes a delta; non-LLM code clamps it. This avoids wild swings while letting the bot gradually warm up or cool down toward individual users.
- **Universal rules in assembler** — behavioural rules ("never break character", etc.) are hardcoded rather than per-personality, ensuring consistency across all characters.
- **`_`-prefixed keys ignored** — personality JSON can store reference data (appearance, art notes) under `_reference` without wasting prompt tokens.
- **Inline supervisor over separate routing LLM** — the persona agent decides in-context whether to call a tool. This avoids an extra LLM call on every message (most messages don't need tools) and sidesteps the routing reliability problem with local models.
- **Prompt-based `<tool_call>` tags** instead of native OpenAI function calling — ensures compatibility with Qwen and other local models served via LM Studio that may not support the `tools` parameter.
- **MCP for tooling** — tools are served by external MCP servers over HTTP, making them language-agnostic and independently deployable. New tools can be added by spinning up a new MCP server without changing bot code.
