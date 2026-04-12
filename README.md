# Role-Play Discord Chatbot

A Discord chatbot that responds in-character using a configurable personality (JSON), long-term user memory, channel conversation history, and **MCP tool calling** — built with **LangGraph**, **LM Studio** (OpenAI-compatible API), and **MongoDB**.

## Architecture

The bot uses a **multi-stage persona simulation pipeline**. When a message arrives, a **Relevance Agent** decides whether to respond, then a **Persona Supervisor** orchestrates a 5-stage cognitive pipeline — decontextualization, research, cognition, dialog generation, and consolidation — each stage backed by its own LLM subgraph. All generative LLM calls use native JSON structures for context passing.

```
                    ┌─────────────┐
                    │   Discord    │
                    └──────┬──────┘
                           │ raw event + context loading
                    ┌──────▼──────────┐
                    │ Relevance Agent │  ★ LLM CALL (should I respond?)
                    └──────┬──────────┘
                           │ should_respond / channel_topic / user_topic
                    ┌──────▼──────────────────────────────────────┐
                    │         Persona Supervisor v2                │
                    │                                              │
                    │  Stage 0: Message Decontextualizer  ★ LLM   │
                    │       ↓                                      │
                    │  Stage 1: Research Subgraph     ★ LLM × N   │
                    │       ↓                                      │
                    │  Stage 2: Cognition Subgraph    ★ LLM × 3   │
                    │   (subconscious → conscious → social filter) │
                    │       ↓                                      │
                    │  Stage 3: Dialog Agent          ★ LLM × 1-3 │
                    │   (generator ↔ evaluator loop)               │
                    │       ↓                                      │
                    │  Stage 4: Consolidation Subgraph ★ LLM × 3+ │
                    │   (state update, relationship, facts → DB)   │
                    └──────────┬───────────────────────────────────┘
                           │ final_dialog + future_promises
                    ┌──────▼──────┐
                    │   Discord    │  send reply + save conversation
                    └─────────────┘
```

### LangGraph Graph (compiled)

The top-level graph is two nodes with a conditional edge:

```
START → relevance_agent ─┬─ should_respond=true  → persona_supervisor2 → END
                         └─ should_respond=false → END
```

The Persona Supervisor internally builds a second `StateGraph` with 5 linear stages, each of which may contain nested subgraphs.

## Pipeline Stages

### Relevance Agent (LLM call)
The Discord bot (`discord_bot.py`) loads all context from MongoDB before invoking the graph:
- **Conversation history** — last N messages for the channel
- **User profile** — facts, affinity score, last relationship insight
- **Character state** — current mood, global vibe, reflection summary

The relevance agent then analyzes the message + context to decide if the bot should engage. It outputs `should_respond`, `use_reply_feature`, `channel_topic`, and `user_topic`.

If `should_respond: false`, the graph short-circuits to END — no further LLM calls.

### Persona Supervisor v2 — Stage 0: Message Decontextualizer (LLM call)
Clarifies ambiguous user input by resolving pronouns, references, and implicit context from chat history. For example, "I saw him yesterday" → "I saw John yesterday". Outputs `decontexualized_input` for downstream stages.

### Persona Supervisor v2 — Stage 1: Research Subgraph (LLM calls)
Dispatches research tasks to specialist agents based on the query nature:

| Agent | Description |
|-------|-------------|
| `memory_retriever_agent` | Searches conversation history, user facts, and persistent memory via MongoDB. Uses tool-calling with `search_user_facts`, `search_conversation`, `get_conversation`, `search_persistent_memory`. |
| `web_search_agent` | Searches the internet via MCP tools (e.g., SearXNG). Multi-turn LLM loop with planning, execution, and evaluation. |

An **evaluator** determines if the retrieved information is sufficient or if another research iteration is needed (up to `MAX_PERSONA_SUPERVISOR_STAGE1_RETRY`).

### Persona Supervisor v2 — Stage 2: Cognition Subgraph (3 LLM calls)
A 3-layer cognitive simulation:

1. **Subconscious** — emotional appraisal and interaction subtext analysis
2. **Consciousness** — internal monologue, logical stance, character intent
3. **Social Filter** — action directives (speech guide, content anchors, style filter) that control how the character expresses itself

Outputs: `internal_monologue`, `action_directives`, `emotional_appraisal`, `character_intent`, `logical_stance`, `interaction_subtext`.

### Persona Supervisor v2 — Stage 3: Dialog Agent (1–3 LLM calls)
A generator-evaluator loop that produces the final in-character reply:

- **Generator** — converts cognition outputs into natural dialog, split into 1–2 message segments
- **Evaluator** — checks for fatal errors (logic contradictions, missing facts, structure violations) and soft guideline adherence. Dynamically relaxes thresholds on retries.

Loop runs up to `MAX_DIALOG_AGENT_RETRY` times. Outputs: `final_dialog` (list of message strings).

### Persona Supervisor v2 — Stage 4: Consolidation Subgraph (3+ LLM calls)
Runs inline (not deferred) after dialog generation to persist the interaction's effects:

1. **Global State Updater** — updates mood, global vibe, and reflection summary
2. **Relationship Recorder** — generates diary entry, affinity delta (with non-linear scaling breakpoints), and relationship insight
3. **Facts Harvester** — extracts new user facts and future promises (with evaluator loop up to `MAX_FACT_HARVESTER_RETRY`)
4. **DB Writer** — persists all outputs to MongoDB (character state, user facts, affinity, relationship insight, memory)

### MCP Tool Calling

Tools are provided by external **MCP servers** (Streamable HTTP). At startup, the bot connects to all configured servers and discovers available tools. Tool names are namespaced as `{server}__{tool}` internally, but the original name is sent to the MCP server.

Configured via the `MCP_SERVERS` environment variable (JSON string):

```json
{"mcp-searxng": {"url": "http://host:4001/mcp"}, "playwright": {"url": "http://host:8931/mcp"}}
```

## Project Structure

```
src/kazusa_ai_chatbot/
  main.py                  # CLI entry point
  config.py                # env vars, affinity breakpoints, retry limits
  state.py                 # DiscordProcessState TypedDict
  db.py                    # MongoDB async helpers + document schemas (TypedDict)
  discord_bot.py           # Discord client, graph wiring, message handling
  mcp_client.py            # McpManager — MCP server connections + tool execution
  utils.py                 # Shared helpers (JSON parsing, affinity mapping, history trimming)
  agents/
    dialog_agent.py        # Stage 3 — generator/evaluator dialog loop
    memory_retriever_agent.py  # Research agent — MongoDB memory/history search
    web_search_agent2.py   # Research agent — web search via MCP tools
  nodes/
    relevance_agent.py             # Relevance gate (context analysis + should_respond)
    persona_supervisor2.py         # Top-level 5-stage persona orchestrator
    persona_supervisor2_schema.py  # GlobalPersonaState TypedDict
    persona_supervisor2_msg_decontexualizer.py  # Stage 0
    persona_supervisor2_research_subgraph.py    # Stage 1
    persona_supervisor2_cognition.py            # Stage 2
    persona_supervisor2_consolidator.py         # Stage 4
src/scripts/               # Standalone utility scripts (embedding creation, search, debug)
personalities/
  example.json             # sample personality
  kazusa.json              # full personality with _reference section
tests/                     # pytest suite (mocked unit tests + live integration tests)
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
LLM_MODEL=your-model-name
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=your-embedding-model-name

# Optional: MCP tool servers (JSON string)
MCP_SERVERS={"mcp-searxng": {"url": "http://host:4001/mcp"}}
```

### 3. LLM Server

Run an OpenAI-compatible API server (e.g., LM Studio, vLLM, Ollama) with:
- **Chat model** — used by all LLM stages (relevance, cognition, dialog, consolidation, research agents)
- **Embedding model** — used for semantic search over conversation history, user facts, and memory

### 4. MongoDB

Collections are auto-created on first use:
- **`conversation_history`** — chat messages with embeddings for semantic search
- **`user_facts`** — per-user facts, affinity score, relationship insight, with embeddings
- **`character_state`** — single global document for mood, vibe, reflection summary
- **`memory`** — persistent named memories with embeddings

Vector search indexes can be created via the utility scripts in `src/scripts/`.

### 5. Create a personality

Create a JSON file following the schema in `personalities/example.json`. The personality JSON should include at minimum:
- `name`, `description`, `gender`, `age`, `birthday`
- `personality_brief` with `logic`, `tempo`, `defense`, `quirks`, `taboos`, `mbti`

Keys prefixed with `_` (e.g., `_reference`) are ignored — use these for visual descriptions or other non-prompt data. See `kazusa.json` for a full example.

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

- **5-stage cognitive pipeline** — separating decontextualization, research, cognition, dialog, and consolidation gives each stage a focused LLM context and allows independent iteration. The cognition layer (subconscious → conscious → social filter) models a human-like decision process rather than a single prompt.
- **Generator-evaluator dialog loop** — the dialog agent generates candidate replies and an evaluator checks for fatal errors (logic contradictions, missing facts). This catches issues before sending while avoiding excessive retries via dynamic threshold relaxation.
- **Inline consolidation** — unlike the previous deferred memory writer, consolidation now runs as part of the graph. This ensures character state, facts, and affinity are updated before the next message arrives.
- **Non-linear affinity scaling** — affinity deltas are scaled by breakpoints: easy to gain/lose at extremes, normal in the middle, harder at high levels. This prevents runaway affinity while allowing meaningful relationship progression.
- **Affinity system** — per-user 0–1000 score (default 500) with 21 behavioral tiers from "Contemptuous" to "Unwavering". The LLM proposes a delta; non-LLM code applies non-linear scaling and clamping.
- **Research subgraph with evaluator** — the research stage can dispatch multiple specialist agents and re-evaluate whether enough information has been gathered, preventing premature or insufficient research.
- **MCP for tooling** — tools are served by external MCP servers over Streamable HTTP, making them language-agnostic and independently deployable.
- **Context loading in Discord bot** — conversation history, user profile, and character state are loaded once in `discord_bot.py` before graph invocation, keeping nodes stateless and testable.
- **`_`-prefixed personality keys ignored** — personality JSON can store reference data (appearance, art notes) under `_reference` without wasting prompt tokens.
