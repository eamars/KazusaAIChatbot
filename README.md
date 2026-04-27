# Kazusa AI Chatbot

Kazusa AI Chatbot is a platform-agnostic character chatbot brain built with
FastAPI, LangGraph, MongoDB, and OpenAI-compatible chat and embedding APIs. It
is designed for long-running character interaction: it remembers people, tracks
conversation context, retrieves relevant evidence, reasons through a staged
persona pipeline, and writes durable relationship state back after each turn.

The brain is separate from its adapters. Discord, NapCat QQ, and the browser
debug UI all talk to the same HTTP service, so the character logic stays in one
place while platform code remains thin.

For setup, operations, API details, and test commands, see
[docs/HOWTO.md](docs/HOWTO.md).

## What It Does

- Runs a staged LangGraph persona pipeline: relevance, decontextualization,
  RAG2 research, cognition, dialog, and background consolidation.
- Maintains durable character and user memory in MongoDB, including identity
  links, diary-style relationship notes, objective facts, active commitments,
  image summaries, channel history, and scheduled events.
- Uses a progressive RAG2 supervisor with helper agents for profile recall,
  conversation search, persistent memories, entity lookup, user lists, and
  optional web search through MCP.
- Caches helper-agent retrieval through Cache2, an in-memory session LRU with
  dependency-aware invalidation from conversation saves and consolidator writes.
- Keeps user-derived context out of system prompts: retrieved evidence, chat
  history, dialog output, and consolidation payloads are carried in human
  messages, preserving a clean trusted/untrusted prompt boundary.
- Supports multimodal input by describing image attachments before relevance
  and downstream persona stages.

## Architecture

```text
Adapters
  Discord / NapCat QQ / Debug UI
        |
        v
FastAPI brain service
  POST /chat
  GET  /health
  POST /event
        |
        v
Top-level LangGraph
  listen_only gate
  multimedia descriptor
  relevance agent
  persona_supervisor2
        |
        v
Persona supervisor v2
  Stage 0: message decontextualizer
  Stage 1: RAG2 research supervisor
  Stage 2: cognition subgraph
  Stage 3: dialog generator/evaluator
  Stage 4: background consolidation
        |
        v
MongoDB + Cache2 + OpenAI-compatible model APIs
```

## Technical Highlights

**RAG2 as the only research path**

RAG2 decomposes a turn into unknown slots, dispatches specialized helper
agents, evaluates whether each result resolved the slot, and projects known
facts into a compact `rag_result` payload. Structured profile and image bundles
are preserved where cognition needs them; bulky evidence is summarized before
it reaches later stages.

**Cache2 with event-driven correctness**

Cache2 caches helper-agent outputs at the agent boundary. Entries declare their
data dependencies, and write paths emit `CacheInvalidationEvent` objects after
durable changes. `save_conversation` invalidates conversation-history entries;
the consolidator invalidates user-profile and character-state entries after
successful writes. Cache2 is session-scoped and does not depend on MongoDB
write-through cache collections.

**LLM-first memory consolidation**

The consolidator turns a completed interaction into persistent state: character
mood, relationship observations, objective facts, commitments, user image
updates, character self-image updates, and scheduled future events. Duplicate
prevention is handled through LLM instructions plus database idempotency, not
content heuristics over user text.

**Prompt boundary discipline**

Character configuration and agent instructions stay in system prompts.
User-derived material, including retrieval evidence and prior dialog, is sent
through human-message payloads. This keeps the prompt architecture easier to
audit and reduces accidental elevation of retrieved or user-provided content.

**Adapter-neutral service core**

The brain exposes a compact HTTP surface while platform adapters handle the
transport details. This keeps the character pipeline reusable across chat
systems and makes local debugging possible through the same `/chat` contract
used by production adapters.

## Main Runtime Collections

| Collection | Purpose |
| --- | --- |
| `conversation_history` | Stored user and assistant messages plus embeddings |
| `user_profiles` | Identity mapping, profile memory, affinity, commitments, user image |
| `character_state` | Singleton character profile, runtime mood/vibe, self image |
| `memory` | Append-only long-term fact and promise records |
| `scheduled_events` | Pending future events |

Legacy write-through cache collections are intentionally absent from the
runtime model.

## Repository Map

```text
src/kazusa_ai_chatbot/
  db/         MongoDB access, schemas, bootstrap, profile and memory operations
  nodes/      LangGraph persona, cognition, dialog, consolidation, and RAG2 nodes
  rag/        RAG2 helper agents, retrieval tools, image/profile retrieval, Cache2
  service.py  FastAPI brain service

adapters/     Debug UI, Discord, and NapCat QQ adapters
personalities/ Character profile examples and local personality data
scripts/      Operational scripts such as legacy collection cleanup
docs/         Setup and operational documentation
tests/        Deterministic, live DB, and live LLM tests
```

## Documentation

- [docs/HOWTO.md](docs/HOWTO.md): local setup, environment variables, service
  startup, adapters, HTTP API, testing, and migration notes.
- [development_plans/rag_cache2_design.md](development_plans/rag_cache2_design.md):
  Cache2 architecture and invalidation model.
