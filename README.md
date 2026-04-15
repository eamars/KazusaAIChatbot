# Kazusa AI Chatbot

A platform-agnostic AI chatbot that responds in-character using a configurable personality (JSON), long-term user memory, channel conversation history, and **MCP tool calling** — built with **FastAPI**, **LangGraph**, **LM Studio** (OpenAI-compatible API), and **MongoDB**.

The brain runs as a standalone service; IM adapters (Discord, debug web UI, etc.) communicate via a simple HTTP API.

## Architecture

The bot uses a **multi-stage persona simulation pipeline**. When a message arrives, a **Relevance Agent** decides whether to respond, then a **Persona Supervisor** orchestrates a 5-stage cognitive pipeline — decontextualization, research, cognition, dialog generation, and consolidation — each stage backed by its own LLM subgraph. All generative LLM calls use native JSON structures for context passing.

```
┌──────────────────┐          ┌──────────────────┐
│ Discord Adapter  │──HTTP──▶│                   │
└──────────────────┘         │  Kazusa Brain     │       ┌──────────┐
┌──────────────────┐         │  (FastAPI)        │◀─────▶│ MongoDB  │
│ Debug Web UI     │──HTTP──▶│                   │       └──────────┘
└──────────────────┘         │  POST /chat       │       ┌──────────┐
┌──────────────────┐         │  GET  /health     │◀─────▶│ LLM API  │
│ QQ / WeChat / …  │──HTTP──▶│  POST /event      │       └──────────┘
└──────────────────┘         └────────┬──────────┘
                                      │
                              ┌───────▼────────┐
                              │ LangGraph      │
                              │ Pipeline       │
                              └───────┬────────┘
                                      │
                    ┌─────────────────────────────────────────┐
                    │         Persona Supervisor v2            │
                    │                                          │
                    │  Stage 0: Message Decontextualizer  ★    │
                    │  Stage 1: Research Subgraph     ★ × N    │
                    │  Stage 2: Cognition Subgraph    ★ × 3    │
                    │  Stage 3: Dialog Agent          ★ × 1-3  │
                    │  Stage 4: Consolidation         ★ × 3+   │
                    └─────────────────────────────────────────┘
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
src/
  kazusa_ai_chatbot/                 # Brain service package
    service.py                       # FastAPI app — /chat, /health, /event routes
    scheduler.py                     # Async event scheduler (MongoDB-backed)
    config.py                        # env vars, affinity breakpoints, retry limits
    state.py                         # DiscordProcessState TypedDict
    db.py                            # MongoDB helpers + document schemas + db_bootstrap()
    discord_bot.py                   # Legacy Discord client (kept for reference)
    mcp_client.py                    # McpManager — MCP server connections + tool execution
    utils.py                         # Shared helpers (JSON parsing, affinity, history trim)
    agents/
      dialog_agent.py                # Stage 3 — generator/evaluator dialog loop
      memory_retriever_agent.py      # Research agent — MongoDB memory/history search
      web_search_agent2.py           # Research agent — web search via MCP tools
    nodes/
      relevance_agent.py             # Relevance gate (context analysis + should_respond)
      persona_supervisor2.py         # Top-level 5-stage persona orchestrator
      persona_supervisor2_schema.py  # GlobalPersonaState TypedDict
      persona_supervisor2_msg_decontexualizer.py  # Stage 0
      persona_supervisor2_research_subgraph.py    # Stage 1
      persona_supervisor2_cognition.py            # Stage 2
      persona_supervisor2_consolidator.py         # Stage 4
  adapters/                          # IM adapters (outside brain package)
    discord_adapter.py               # Thin Discord→HTTP adapter
    debug_adapter.py                 # Browser-based debug chat UI
  scripts/                           # Standalone utility scripts
personalities/
  example.json
  kazusa.json
tests/
Dockerfile
docker-compose.yml
requirements.txt
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=roleplay_bot

# LLM (OpenAI-compatible)
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=your-model-name
EMBEDDING_BASE_URL=http://localhost:1234/v1
EMBEDDING_MODEL=your-embedding-model-name

# Brain service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
PERSONALITY_PATH=personalities/kazusa.json

# Discord adapter (only needed if running the Discord adapter)
DISCORD_TOKEN=your_discord_bot_token

# Optional: MCP tool servers (JSON string)
MCP_SERVERS={"mcp-searxng": {"url": "http://host:4001/mcp"}}
```

### 3. LLM Server

Run an OpenAI-compatible API server (e.g., LM Studio, vLLM, Ollama) with:
- **Chat model** — used by all LLM stages (relevance, cognition, dialog, consolidation, research agents)
- **Embedding model** — used for semantic search over conversation history, user facts, and memory

### 4. MongoDB

The brain service runs `db_bootstrap()` on startup which automatically creates all required collections and indexes:
- **`conversation_history`** — chat messages with embeddings for semantic search
- **`user_profiles`** — per-user facts, affinity score, platform accounts, relationship insight
- **`character_state`** — single global document for mood, vibe, reflection summary
- **`memory`** — persistent named memories with embeddings
- **`scheduled_events`** — future events (follow-up messages, etc.)

Vector search indexes are created best-effort (requires MongoDB Atlas for full vector search).

### 5. Create a personality

Create a JSON file following the schema in `personalities/example.json`. The personality JSON should include at minimum:
- `name`, `description`, `gender`, `age`, `birthday`
- `personality_brief` with `logic`, `tempo`, `defense`, `quirks`, `taboos`, `mbti`

Keys prefixed with `_` (e.g., `_reference`) are ignored — use these for visual descriptions or other non-prompt data. See `kazusa.json` for a full example.

## Deployment

### Local Development

```bash
# 1. Start the brain service
uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000

# 2a. Debug adapter — browser chat at http://localhost:8080
python -m adapters.debug_adapter --brain-url http://localhost:8000 --port 8080

# 2b. Discord adapter
python -m adapters.discord_adapter --brain-url http://localhost:8000

# 2c. Discord adapter with channel filter
python -m adapters.discord_adapter --brain-url http://localhost:8000 --channels 123456789
```

### Docker Deployment

```bash
# Start brain + MongoDB
docker-compose up -d kazusa-brain mongo

# Optionally, uncomment and start adapters in docker-compose.yml:
# docker-compose up -d discord-adapter
# docker-compose up -d debug-adapter
```

The `docker-compose.yml` defines:
- **`kazusa-brain`** — the FastAPI service on port 8000
- **`mongo`** — MongoDB 7 with persistent volume
- **`discord-adapter`** (commented) — Discord client forwarding to brain
- **`debug-adapter`** (commented) — Web chat UI on port 8080

Environment variables are read from `.env` or can be set in the compose file.

### Legacy Discord Bot (deprecated)

```bash
# Direct Discord bot (bypasses brain service — kept for reference)
python src/kazusa_ai_chatbot/main.py --personality personalities/example.json
```

## Design Decisions

- **5-stage cognitive pipeline** — separating decontextualization, research, cognition, dialog, and consolidation gives each stage a focused LLM context and allows independent iteration. The cognition layer (subconscious → conscious → social filter) models a human-like decision process rather than a single prompt.
- **Generator-evaluator dialog loop** — the dialog agent generates candidate replies and an evaluator checks for fatal errors (logic contradictions, missing facts). This catches issues before sending while avoiding excessive retries via dynamic threshold relaxation.
- **Inline consolidation** — unlike the previous deferred memory writer, consolidation now runs as part of the graph. This ensures character state, facts, and affinity are updated before the next message arrives.
- **Non-linear affinity scaling** — affinity deltas are scaled by breakpoints: easy to gain/lose at extremes, normal in the middle, harder at high levels. This prevents runaway affinity while allowing meaningful relationship progression.
- **Affinity system** — per-user 0–1000 score (default 500) with 21 behavioral tiers from "Contemptuous" to "Unwavering". The LLM proposes a delta; non-LLM code applies non-linear scaling and clamping.
- **Research subgraph with evaluator** — the research stage can dispatch multiple specialist agents and re-evaluate whether enough information has been gathered, preventing premature or insufficient research.
- **MCP for tooling** — tools are served by external MCP servers over Streamable HTTP, making them language-agnostic and independently deployable.
- **Brain + adapter separation** — the brain service is platform-agnostic; IM adapters are thin HTTP clients. This enables multi-platform support (Discord, QQ, WeChat) without modifying the cognitive pipeline.
- **Context loading in service** — conversation history, user profile, and character state are loaded once in the `/chat` handler before graph invocation, keeping nodes stateless and testable.
- **`_`-prefixed personality keys ignored** — personality JSON can store reference data (appearance, art notes) under `_reference` without wasting prompt tokens.
- **DB bootstrap on startup** — all collections and indexes are verified/created at service start, so deployment requires zero manual setup beyond the environment variables.
