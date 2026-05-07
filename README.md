# Kazusa AI Chatbot

Kazusa AI Chatbot is a platform-agnostic character brain for long-running chat interaction.

The project is built around a simple idea: a character should not be a stateless prompt wrapped in a chat adapter. Kazusa has a service core that listens, decides whether to respond, recalls relevant context, reasons through her own stance, speaks in character, and then quietly updates durable memory after the turn is over.

The same brain can be reached from Discord, QQ, a browser debug UI, or another adapter that speaks the service API. Platform code stays thin; the character logic, memory, retrieval, and scheduling live in one place.

For setup, operations, environment variables, API details, and test commands, see [docs/HOWTO.md](docs/HOWTO.md).

## What This Project Is

Kazusa is an experimental digital-character runtime with:

- A staged conversation pipeline instead of a single giant prompt.
- Long-term user and relationship memory.
- Short-term conversation-flow continuity.
- Evidence retrieval over profiles, memories, conversation history, and optional web sources.
- A process-local input queue that drops burst noise before RAG while preserving addressed messages.
- Background consolidation that turns completed interactions into durable state.
- A reflection cycle that reviews past interaction outside the live response path.
- Scheduled follow-through for accepted future promises.
- Adapter-neutral deployment across chat platforms.

The project is designed for local or OpenAI-compatible model runtimes, including weaker models with limited practical context-following ability. The architecture favors explicit boundaries, compact intermediate state, and specialist sub-systems over one prompt trying to infer everything.

## What It Is Not

Kazusa is not a generic assistant shell, a command router, or a tool-use demo.

The core runtime is character-centered: retrieval provides evidence, cognition decides what that evidence means for Kazusa in the current moment, and dialog generation owns the final wording. Durable memory is written after the response path, not by interrupting the user-facing turn with a long agent loop.

## High-Level Flow

```text
Chat platform / debug client
        |
        v
Brain service
  - enqueue inbound messages
  - prune burst noise before RAG
  - save dropped user messages without replying
        |
        v
Listen gate + perception
  - normalize incoming message
  - describe attachments when needed
  - decide whether Kazusa should respond
        |
        v
Persona turn
  - clarify the user's current message
  - retrieve relevant evidence
  - load short-term conversation flow
  - reason through stance, intent, and response goals
  - generate Kazusa's reply
        |
        +-------------------------> response to platform
        |
        v
Background consolidation
  - record conversation progress
  - update durable user/character memory
  - update relationship state and image summaries
  - run slower reflection and promotion work outside live chat
  - schedule accepted future follow-through
```

The response path is kept bounded. Heavier memory writes, image updates, cache invalidation, and scheduling happen after the user-facing reply is already available. The service still waits for those post-response writes before consuming the next queued chat item, so the next RAG pass does not read stale durable facts or stale Cache2 entries.

## Memory Horizons

Kazusa uses several memory horizons rather than treating all context as one pile of chat history.

```text
Immediate surface
  recent message text and local tone

Short-term flow
  current episode, open loops, topic momentum, repeated moves to avoid

Retrieved evidence
  profiles, memories, conversation history, user lookup, web facts when needed

Durable memory
  identity links, relationship notes, objective facts, milestones, commitments

Scheduled future actions
  accepted promises that should fire later through platform adapters
```

This separation matters. Recent chat helps Kazusa sound locally present, short-term flow helps her avoid looping or reopening stale threads, RAG provides factual grounding, and durable memory preserves relationship continuity across sessions.

## Core Subsystems

**Brain Service**

The service is the stable HTTP-facing core. It receives platform-neutral chat requests, queues them, prunes noisy bursts, runs the surviving turn pipeline, persists conversation rows, exposes health data, and coordinates startup/shutdown work. Dropped queued messages are still saved as user conversation rows, but they do not run relevance, RAG, cognition, dialog, or consolidation.

**Persona Pipeline**

The persona pipeline is the main response path. It separates relevance, message clarification, retrieval, cognition, and dialog generation so each stage has a narrower responsibility.

**Conversation Progress**

Conversation Progress is short-term operational memory. It tracks the current local episode so cognition can continue, deepen, pivot, or close a conversation naturally without rereading full raw history every turn.

**RAG 2**

RAG 2 is the evidence-retrieval system. It decomposes a query into missing facts, dispatches specialist retrieval agents, and returns compact evidence for cognition. It retrieves facts; it does not decide Kazusa's feelings or final wording.

**Database Layer**

The database layer stores conversation history, user identities, durable memories, character state, scheduled events, and short-lived episode state. It owns storage mechanics and embeddings, while higher-level modules own semantic interpretation.

**Dispatcher And Scheduler**

The dispatcher converts accepted future promises into validated scheduled tasks. The scheduler persists and fires them later through registered platform adapters. This is how Kazusa can follow through on a promised later message without blocking the current turn.

**Reflection Cycle**

Reflection is the slow background sense-making loop. It reads completed conversation windows, stores inspectable hourly and daily reflection runs, and may promote a small amount of durable lore or self-guidance through the memory-evolution boundary. Raw reflection output does not enter normal cognition directly; only promoted reflection context is eligible, and that context is gated by configuration.

**Adapters**

Adapters connect chat platforms to the brain service. They translate platform events into the service API and deliver responses back to the platform.

## Architectural Principles

**LLM-first semantics, deterministic mechanics**

LLMs decide semantic questions: whether Kazusa should answer, what evidence is needed, what a memory means, whether a user request became an accepted promise, and how Kazusa should frame her reply. Deterministic code handles structure: validation, persistence, limits, cache invalidation, scheduling, and adapter delivery.

**Bounded response latency**

The normal response path avoids unbounded multi-agent exploration. Retrieval and cognition are structured; consolidation and scheduling run in the background.

**Evidence is not persona**

RAG evidence answers "what is known?" Cognition answers "what does this mean for Kazusa right now?" Dialog answers "how does she say it?"

**Memory has ownership**

Short-term flow, retrieved evidence, durable memories, and scheduled commitments are separate systems. They overlap in purpose, but they do not replace each other.

**Platform-neutral core**

Kazusa's identity, cognition, memory, and retrieval do not belong to Discord, QQ, or any one adapter. Adapters are transport edges around the same brain.

## Project Shape

```text
Adapters and clients
        |
        v
Brain service
        |
        +-- global input queue
        |     prune burst noise -> persist dropped messages
        |
        +-- turn pipeline
        |     relevance -> retrieval -> cognition -> dialog
        |
        +-- short-term conversation progress
        |
        +-- RAG 2 evidence retrieval
        |
        +-- background consolidation
        |
        +-- reflection cycle
        |
        +-- dispatcher and scheduler
        |
        v
MongoDB + model APIs + platform adapters
```

For deeper technical introductions:

- [Conversation Progress](src/kazusa_ai_chatbot/conversation_progress/README.md)
- [RAG 2](src/kazusa_ai_chatbot/rag/README.md)
- [Reflection Cycle](src/kazusa_ai_chatbot/reflection_cycle/README.md)
- [Dispatcher](src/kazusa_ai_chatbot/dispatcher/README.md)
- [Database](src/kazusa_ai_chatbot/db/README.md)

## Current Direction

The project is moving toward an autonomous digital-life engine: a character that can maintain continuity, remember responsibly, retrieve evidence when needed, and follow through on accepted commitments while remaining platform independent.

The goal is not maximal tool use. The goal is a believable, inspectable character runtime where each subsystem has a clear reason to exist.

## Getting Started

Use [docs/HOWTO.md](docs/HOWTO.md) for local setup and operation.

That guide covers environment variables, model endpoints, MongoDB, service startup, adapters, API usage, and testing.
