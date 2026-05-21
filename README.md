<div align="center">
  <img src="resources/avatar.png" alt="Kazusa avatar" width="420" height="420" />

<h1>Kazusa Cognitive Core</h1>

<p><strong>A self-evolving character cognition runtime for persistent digital presence.</strong></p>

<p>
    <a href="README_CN.md">简体中文</a>
    ·
    <a href="docs/HOWTO.md">HOWTO</a>
  </p>

<p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-brain_service-009688?logo=fastapi&logoColor=white" />
    <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-cognition_pipeline-1C3C3C" />
    <img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-memory_store-47A248?logo=mongodb&logoColor=white" />
    <img alt="License" src="https://img.shields.io/badge/License-AGPL--3.0-blue" />
  </p>
</div>

## What Kazusa Achieves

Kazusa is not a generic assistant shell. It is a psychological model of a
self-evolving character brain: a runtime that keeps identity, relationship
continuity, retrieval, cognition, dialog, memory, reflection, and future
follow-through inside one inspectable service core.

The same brain can be reached from Discord, NapCat QQ, the browser debug UI, or
another adapter that speaks the service API. Adapters stay thin. The brain
service consumes typed message-envelope fields instead of parsing raw Discord,
QQ, or debug-wire syntax.

At a high level, Kazusa provides:

| Capability                       | What it means                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Platform-neutral character brain | Discord, QQ, debug UI, and future adapters feed the same FastAPI brain service.                                                    |
| Typed message boundary           | Platform syntax is normalized into `MessageEnvelope` fields before cognition or RAG sees it.                                       |
| Bounded live response path       | Queueing, relevance, RAG, cognition, action routing, and L3 surfaces are explicit stages with caps and inspectable payloads.       |
| Multi-horizon memory             | Recent chat, short-term conversation flow, retrieved evidence, durable memory, and scheduled commitments remain separate.          |
| RAG 2 evidence retrieval         | Helper agents retrieve user profiles, memories, conversation history, live facts, web evidence, and recall state.                  |
| Layered cognition                | Cognition decides stance, boundaries, judgment, style, action needs, and response goals before selected L3 surfaces render output. |
| Background consolidation         | Completed episodes update durable memory, relationship state, Cache2 invalidation, images, and progress from text plus action/surface traces. |
| Reflection outside chat          | Hourly, daily, and promoted reflection runs are stored as audit records and only promoted context can enter normal cognition.      |
| Scheduled follow-through         | Accepted future promises can become validated scheduled tasks delivered later through registered adapters.                         |
| Event logging observability      | Runtime, LLM, RAG, action routing, surfaces, reflection, self-cognition, dispatcher, consolidation, and DB operations emit sanitized operational events. |

## What You Can Build

| Use case                             | Why Kazusa fits                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Persistent character companion       | The runtime keeps relationship memory, short-term flow, character state, and reflection separate but connected.                  |
| Group-chat character bot             | Queue pruning, typed addressees, native reply hydration, and adapter-specific delivery let the brain survive noisy channels.     |
| Local model character lab            | Route-specific OpenAI-compatible model settings let weaker local models handle narrower, staged prompts.                         |
| Memory and RAG experiments           | RAG 2, Cache2, scoped user memory, shared memory evolution, and conversation search are modular enough to inspect independently. |
| Cross-platform adapter experiments   | New adapters only need to normalize platform events into the service contract and render returned messages.                      |
| Promise and follow-through workflows | Accepted future commitments can be validated, persisted, deduplicated, and delivered later through registered adapters.          |

## Supported LLMs

Kazusa is designed around OpenAI-compatible endpoints rather than one hosted
vendor. All OpenAI-compatible chat completion endpoints are technically
supported, and route-specific configuration lets different stages use different
models when needed.

In practice, Kazusa can be configured like a model routing table: lightweight
or local models can handle most structured reasoning, while a different hosted
model can be assigned to a stage where you want stronger voice or generation
quality. The route names below are the configuration handles documented in the
HOWTO. One working-style configuration looks like this:

| Route                      | Example model                            | Example source             |
| -------------------------- | ---------------------------------------- | -------------------------- |
| `RELEVANCE_AGENT_LLM`      | `local-model`                            | `http://localhost:1234/v1` |
| `VISION_DESCRIPTOR_LLM`    | `local-model`                            | `http://localhost:1234/v1` |
| `MSG_DECONTEXTUALIZER_LLM` | `local-model`                            | `http://localhost:1234/v1` |
| `RAG_PLANNER_LLM`          | `local-model`                            | `http://localhost:1234/v1` |
| `RAG_SUBAGENT_LLM`         | `local-model`                            | `http://localhost:1234/v1` |
| `WEB_SEARCH_LLM`           | `local-model`                            | `http://localhost:1234/v1` |
| `COGNITION_LLM`            | `local-model`                            | `http://localhost:1234/v1` |
| `DIALOG_GENERATOR_LLM`     | `deepseek-v4-flash`                      | `https://api.deepseek.com` |
| `DIALOG_EVALUATOR_LLM`     | `local-model`                            | `http://localhost:1234/v1` |
| `CONSOLIDATION_LLM`        | `local-model`                            | `http://localhost:1234/v1` |
| `JSON_REPAIR_LLM`          | `local-model`                            | `http://localhost:1234/v1` |
| `EMBEDDING`                | `text-embedding-nomic-embed-text-v2-moe` | `http://localhost:1234/v1` |

The table is an example, not a fixed requirement. Any route can point to any
OpenAI-compatible endpoint that can satisfy that stage's latency and quality
needs.

Tested chat model families:

- Gemma 4 26B MoE
- Qwen3.6 27B
- DeepSeek v4

Kazusa also requires an OpenAI-compatible embeddings endpoint for conversation
history, memory retrieval, and vector search features. Local deployments
commonly use LM Studio or another OpenAI-compatible end points.

## Architecture At A Glance

```text
Discord / NapCat QQ / Debug UI / future adapters
        |
        | typed ChatRequest + MessageEnvelope
        v
FastAPI brain service
        |
        v
Process-local input queue
  - collapse nearby follow-ups
  - drop burst noise before RAG
  - persist dropped user rows without replying
        |
        v
Listen gate and perception
  - hydrate reply context
  - describe image inputs when needed
  - decide whether Kazusa should answer
        |
        v
Persona turn
  - decontextualize the current message
  - retrieve evidence through RAG 2
  - load short-term conversation progress
  - reason through stance, boundary, style, and intent
  - initialize zero-or-more semantic actions through L2d
  - run selected L3 text/action handlers
  - emit surface outputs and action results
        |
        +-----------------------------> adapter bridge delivers visible surfaces
        |
        v
Post-turn work
  - persist assistant surface rows and delivery tracking
  - record conversation progress
  - consolidate durable memory and state from the episode trace
  - invalidate stale Cache2 entries
  - schedule accepted future promises
  - run reflection and growth workers outside live chat
        |
        v
MongoDB + model routes + optional MCP web tools + platform callbacks
```

Visible adapter delivery follows selected text surface outputs. Private action
results, scheduled-action results, no-visible-output decisions, and private
finalization still feed episode-trace consolidation without creating adapter
sends.

The core boundary is deliberately narrow:

```text
adapter/debug client -> brain service -> queue/intake -> typed episode/RAG
-> cognition/L2d -> selected L3 surfaces/action handlers
-> episode-trace consolidation -> scheduler/reflection
```

## Design Principles

**LLM-first semantics, deterministic mechanics**

LLM stages judge meaning: response relevance, missing evidence, memory meaning,
accepted promises, character stance, action choice, and surface intent.
Deterministic code owns validation, persistence, limits, cache invalidation,
scheduling, adapter delivery, and auditability.

**Evidence is not persona**

RAG answers "what is known?" Cognition answers "what does this mean for Kazusa
right now?" L2d answers "which actions or surfaces are needed?" L3/dialog
answers "how should the selected surface render it?"

**Memory has ownership**

Kazusa does not flatten all context into one prompt. Immediate surface text,
conversation progress, retrieved evidence, durable memory, promoted reflection,
and scheduled commitments each have a separate lifecycle.

**Reflection does not shortcut into live chat**

Reflection is slower sense-making work. Raw reflection output is stored for
inspection, but normal cognition only receives bounded, promoted, gated context.

**Adapters are transport edges**

Platform adapters parse platform events, normalize typed envelopes, call the
brain service, and deliver returned messages. Character identity, memory, RAG,
cognition, and scheduling remain in the platform-neutral core.

## Runtime Layers

| Layer                    | Owns                                                                                    | Key docs                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Adapters                 | Discord, NapCat QQ, debug UI transport and platform rendering                           | [HOWTO](docs/HOWTO.md)                                                                 |
| Brain service            | HTTP API, queue, graph startup, health, delivery receipts, runtime adapter registration | [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)                     |
| Message envelope         | Typed inbound content, mentions, replies, attachments, addressees, broadcast state      | [Message Envelope ICD](src/kazusa_ai_chatbot/message_envelope/README.md)               |
| Conversation progress    | Short-term episode state used by cognition to avoid loops and stale reopenings          | [Conversation Progress](src/kazusa_ai_chatbot/conversation_progress/README.md)         |
| RAG 2                    | Slot-driven helper-agent retrieval and Cache2 evidence projection                       | [RAG 2](src/kazusa_ai_chatbot/rag/README.md)                                           |
| Cognition and dialog     | Character stance, boundaries, judgment, style, visual directives, and final wording     | [Cognition Nodes](src/kazusa_ai_chatbot/nodes/README.md)                              |
| Action spec              | L2d action residues, capability registry, evaluator, results, surfaces, and traces      | [Action Spec](src/kazusa_ai_chatbot/action_spec/README.md)                            |
| Consolidation            | Durable target routing, write-intent validation, and target-specific persistence        | [Consolidation ICD](src/kazusa_ai_chatbot/consolidation/README.md)                    |
| Database                 | MongoDB collection ownership, embeddings, indexes, public persistence helpers           | [Database ICD](src/kazusa_ai_chatbot/db/README.md)                                     |
| Event logging            | Sanitized operational telemetry, status snapshots, statistics, and export contracts     | [Event Logging ICD](src/kazusa_ai_chatbot/event_logging/README.md)                     |
| Dispatcher and scheduler | Validated delayed tool execution for accepted future promises                           | [Dispatcher](src/kazusa_ai_chatbot/dispatcher/README.md)                               |
| Reflection cycle         | Background reflection runs, promotion gates, prompt-safe reflection context             | [Reflection Cycle ICD](src/kazusa_ai_chatbot/reflection_cycle/README.md)               |
| Memory evolution         | Curated shared memory lifecycle, lineage, seed reset, promoted memory writes            | [Memory Evolution ICD](src/kazusa_ai_chatbot/memory_evolution/README.md)               |
| Global character growth  | Slow promoted-trait drift from approved reflection memory                               | [Global Character Growth ICD](src/kazusa_ai_chatbot/global_character_growth/README.md) |
| Proactive output         | Permissioned preview/outbox contracts for future autonomous contact paths               | [Proactive Output ICD](src/kazusa_ai_chatbot/proactive_output/README.md)               |

## Quick Start

Kazusa expects MongoDB plus OpenAI-compatible chat and embedding endpoints. LM
Studio works for local development, but any compatible endpoint can be used.
All route-specific model environment variables are documented in
[docs/HOWTO.md](docs/HOWTO.md).

```powershell
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

Load a character profile before starting the brain:

```powershell
python -m scripts.load_character_profile personalities/kazusa.json
```

Run the brain service:

```powershell
kazusa-brain --host 0.0.0.0 --port 8000
```

Or use Uvicorn directly:

```powershell
uvicorn kazusa_ai_chatbot.service:app --host 0.0.0.0 --port 8000
```

Run the browser debug adapter:

```powershell
python -m adapters.debug_adapter --brain-url http://localhost:8000 --port 8080
```

Then open `http://localhost:8080`.

## Repository Map

```text
src/
  adapters/                    Platform adapters and debug UI
  kazusa_ai_chatbot/
    brain_service/             Service API, graph, intake, health, post-turn glue
    message_envelope/          Typed adapter-to-brain message contract
    nodes/                     Persona, cognition, dialog, consolidation stages
    action_spec/               Modality-neutral action contracts, registry, results
    consolidation/             Durable target routing and consolidation ICD
    rag/                       RAG 2 helper agents, hybrid retrieval, Cache2
    conversation_progress/     Short-term episode memory
    db/                        MongoDB facade, schemas, collection owners
    event_logging/             Sanitized operational telemetry interface and ICD
    dispatcher/                Delayed task validation and adapter handoff
    reflection_cycle/          Background reflection and promotion
    memory_evolution/          Shared memory lifecycle and seed reset
    global_character_growth/   Slow promoted character-growth traits
    proactive_output/          Permissioned proactive preview contracts
  scripts/                     Operator and maintenance CLIs
docs/
  HOWTO.md                     Setup, runtime commands, environment, tests
development_plans/             Approved, archived, and reference plan registry
tests/                         Deterministic, live DB, and live LLM test suites
resources/
  avatar.png                   README avatar asset
```

## Testing

Default test runs exclude live DB and live LLM tests through `pytest.ini`.

```powershell
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

Live LLM tests must be run one case at a time with output inspected. Live DB
tests require MongoDB. See [docs/HOWTO.md](docs/HOWTO.md#testing) for the
project testing contract.

## Project Status

Kazusa Cognitive Core is alpha-stage experimental infrastructure for a
persistent digital character. The main runtime is usable as a local brain
service with adapters, memory, retrieval, reflection, and scheduling, but some
autonomous-contact surfaces intentionally remain permissioned preview contracts
rather than production sends.

## Documentation Index

| Document                                                                 | Purpose                                                           |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [README.md](README.md)                                                   | Project overview and architecture map                             |
| [README_CN.md](README_CN.md)                                             | Simplified Chinese project overview                               |
| [docs/HOWTO.md](docs/HOWTO.md)                                           | Local setup, environment variables, run commands, adapters, tests |
| [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)       | HTTP endpoint contracts and adapter obligations                   |
| [Message Envelope ICD](src/kazusa_ai_chatbot/message_envelope/README.md) | Typed inbound message contract                                    |
| [Database ICD](src/kazusa_ai_chatbot/db/README.md)                       | Persistence ownership and collection contracts                    |
| [Action Spec](src/kazusa_ai_chatbot/action_spec/README.md)               | Modality-neutral action contracts and trace handoff               |
| [Consolidation ICD](src/kazusa_ai_chatbot/consolidation/README.md)       | Durable target routing and write-intent validation                |
| [Event Logging ICD](src/kazusa_ai_chatbot/event_logging/README.md)        | Sanitized telemetry interface, event taxonomy, and ops statistics |
| [RAG 2](src/kazusa_ai_chatbot/rag/README.md)                             | Retrieval architecture and evidence projection                    |
| [Cognition Nodes](src/kazusa_ai_chatbot/nodes/README.md)                 | Layered cognition, dialog, and node-package design contracts      |
| [Development Plans Registry](development_plans/README.md)                | Active, archived, reference, and roadmap documents                |

## License

Kazusa Cognitive Core is released under the
[GNU Affero General Public License v3.0](LICENSE).
