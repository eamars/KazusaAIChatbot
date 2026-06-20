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
| Bounded live response path       | Queueing, relevance, the cognition resolver, selected evidence capabilities, action routing, and L3 surfaces are explicit stages with caps and inspectable payloads. |
| Multi-horizon memory             | Recent chat, short-term conversation flow, retrieved evidence, durable memory, and scheduled commitments remain separate.          |
| Internal monologue residue       | A short private residue lane carries bounded first-person reasons from completed episodes into the next L2a cognition pass.       |
| RAG 2 evidence retrieval         | Demand-driven helper agents retrieve user profiles, memories, conversation history, live facts, web evidence, and recall state.   |
| Layered cognition                | Cognition decides stance, boundaries, judgment, style, action needs, and response goals before selected L3 surfaces render output. |
| Background consolidation         | Completed episodes update durable memory, relationship state, Cache2 invalidation, images, and progress from text plus action/surface traces. |
| Background artifact handoff      | Bounded text-only artifact work can be queued during a live turn and later re-enter cognition as a source-bound result episode.                |
| Reflection outside chat          | Hourly, daily, and promoted reflection runs are stored as audit records and only promoted context can enter normal cognition.      |
| Idle self-cognition              | Background source cases can enter the same resolver-backed persona path, with source-bound delivery and normal consolidation rules. |
| Calendar follow-through          | Accepted future promises and due commitments can become durable calendar triggers that run fresh cognition later.                  |
| Event logging observability      | Runtime, LLM, RAG, action routing, surfaces, reflection, self-cognition, dispatcher, consolidation, and DB operations emit sanitized operational events. |

## What You Can Build

| Use case                             | Why Kazusa fits                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Persistent character companion       | The runtime keeps relationship memory, short-term flow, character state, and reflection separate but connected.                  |
| Group-chat character bot             | Queue pruning, typed addressees, native reply hydration, and adapter-specific delivery let the brain survive noisy channels.     |
| Local model character lab            | Route-specific OpenAI-compatible model settings let weaker local models handle narrower, staged prompts.                         |
| Memory and RAG experiments           | RAG 2, Cache2, scoped user memory, shared memory evolution, and conversation search are modular enough to inspect independently. |
| Cross-platform adapter experiments   | New adapters only need to normalize platform events into the service contract and render returned messages.                      |
| Idle cognition and reflection labs   | Self-cognition and reflection use bounded source packets and shared cognition boundaries without turning adapters into agents.   |
| Promise and follow-through workflows | Accepted future commitments can be validated, persisted, deduplicated, and revisited later through durable calendar triggers.    |

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
| `BOUNDARY_CORE_LLM`        | `local-model`                            | `http://localhost:1234/v1` |
| `BACKGROUND_ARTIFACT_LLM`  | `local-model`                            | `http://localhost:1234/v1` |
| `BACKGROUND_WORK_LLM`      | `local-model`                            | `http://localhost:1234/v1` |
| `CODING_AGENT_LLM`         | `local-model`                            | `http://localhost:1234/v1` |
| `DIALOG_GENERATOR_LLM`     | `deepseek-v4-flash`                      | `https://api.deepseek.com` |
| `CONSOLIDATION_LLM`        | `local-model`                            | `http://localhost:1234/v1` |
| `JSON_REPAIR_LLM`          | `local-model`                            | `http://localhost:1234/v1` |
| `EMBEDDING`                | `text-embedding-nomic-embed-text-v2-moe` | `http://localhost:1234/v1` |

The table is an example, not a fixed requirement. Any route can point to any
OpenAI-compatible endpoint that can satisfy that stage's latency and quality
needs.

`CODING_AGENT_LLM` is optional. When its base URL, API key, and model are all
omitted, standalone coding-agent reading uses `BACKGROUND_WORK_LLM` provider
settings. If any of `CODING_AGENT_LLM_BASE_URL`,
`CODING_AGENT_LLM_API_KEY`, or `CODING_AGENT_LLM_MODEL` is set, all three must
be set together.

Chat LLM calls are routed through `LLInterface`. Each module owns its route,
model, generation budget, and thinking toggle via `LLMCallConfig`; the
interface owns backend detection, provider sessions, request mapping, response
normalization, and reload retry. Public token budget config uses
`max_completion_tokens`. Thinking is disabled by default. When enabled, the
interface currently maps provider-specific thinking controls for Gemma 4,
Qwen3-family model names, and Qwen-compatible Qwopus 3.x model names. The
runtime contract is documented in the
[LLM Interface ICD](src/kazusa_ai_chatbot/llm_interface/README.md).

Tested chat model families:

- Gemma 4 26B MoE
- Qwen3.6 27B
- DeepSeek v4

Kazusa also requires an OpenAI-compatible embeddings endpoint for conversation
history, memory retrieval, and vector search features. Local deployments
commonly use LM Studio or another OpenAI-compatible end points.

## Architecture At A Glance

```mermaid
flowchart TD
    A[Discord, NapCat QQ, debug UI, future adapters]
    B[FastAPI brain service]
    C[Process-local chat queue]
    D[Intake and typed episode assembly]
    E{Service graph}
    F[Media descriptor]
    G[Relevance gate]
    H[Conversation progress, promoted reflection context, L2a residue]
    J[Adapter delivery and delivery receipts]
    K[Post-turn memory, progress, residue, consolidation]
    L[(MongoDB, model routes, Cache2, web/MCP runtimes)]
    M[Calendar scheduler, reflection, self-cognition workers]
    N[No persona turn / empty ChatResponse]

    A -->|ChatRequest + MessageEnvelope| B
    B --> C
    C -->|drop/collapse policy, persist inbound rows| D
    D --> E
    E -->|attachments| F
    F --> G
    E -->|text only| G
    G -->|should respond| H
    G -->|should not respond| N
    H --> P0
    P3 -->|visible text surfaces| J
    P4 -->|private actions or no visible surface| K
    J --> K
    K --> L
    M -->|calendar-triggered source cases| P1
    M --> L

    subgraph Persona["Persona turn"]
        P0[Message decontextualizer]
        P1[Bounded cognition resolver]
        P2[Memory lifecycle specialist]
        P3[L3 text surface and dialog]
        P4[Private no-response/action trace]
        P0 --> P1
        P1 --> P2
        P2 -->|speak action selected| P3
        P2 -->|no visible surface| P4
    end

    subgraph Resolver["Resolver recurrence"]
        R1[L1 affect and subtext]
        R2[L2a consciousness + L2b boundary]
        R3[L2c1 judgment + L2c2 social context]
        R4[L2d action and capability selection]
        R1 --> R2
        R2 --> R3
        R3 --> R4
    end

    subgraph Capabilities["Cognition-selected capabilities"]
        C0[Deterministic capability executor]
        C1[RAG 2 or web/current evidence]
        C2[Human clarification or approval blocker]
        C3[Private self-goal resolution]
        C0 --> C1
        C0 --> C2
        C0 --> C3
    end

    P1 --> R1
    R4 -->|capability requested| C0
    C1 -->|observation| R1
    C2 -->|observation or pending resume| R1
    C3 -->|observation| R1
    R4 -->|terminal action specs| P2
```

Kazusa's live response path is a cognition core, not a chatbot shell or a
generic tool harness. Adapters normalize platform events into the typed service
contract; the brain service owns queueing, identity, reply hydration, history,
episode construction, and graph execution.

The resolver preserves the same L1 -> L2 -> L2d cognition stack on every
cycle. L2d may finish with selected action specs, or it may request one bounded
capability observation such as RAG 2 evidence, web/current evidence, human
clarification, approval preparation, or private self-goal resolution. The
observation is projected into the next cognition cycle; evidence never speaks
as persona by itself.

Selected visible text surfaces go back to adapters through `ChatResponse` and
delivery receipts. Private action results, no-visible-output decisions, and
surface traces can still feed post-turn progress, consolidation, Cache2
invalidation, residue recording, calendar state, reflection, and
self-cognition without creating a platform send.

Background work requests are selected by cognition as `background_work_request`,
validated and queued by deterministic action-spec execution, and acknowledged
only after a durable pending job exists. A route-only background-work router
chooses the worker and semantic task after the live turn; the text-artifact
worker has its own task router and generator stages. Completed results return
as `background_work_result_ready` cognition rather than being sent directly by
workers. Legacy background-artifact rows remain compatibility data, not the new
top-level runtime contract.

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
and calendar-scheduled commitments each have a separate lifecycle.

The internal monologue residue lane is a separate short-lived lane. It stores
one compact first-person reason from a completed episode and projects it only
into L2a as `internal_monologue_residue_context`. It is not
`reflection_summary`, durable memory, visible dialog planning, or calendar
input.

**Reflection does not shortcut into live chat**

Reflection is slower sense-making work. Raw reflection output is stored for
inspection, but normal cognition only receives bounded, promoted, gated context.

**Adapters are transport edges**

Platform adapters parse platform events, normalize typed envelopes, call the
brain service, and deliver returned messages. Character identity, memory, RAG,
cognition, and calendar scheduling remain in the platform-neutral core.

## Runtime Layers

| Layer                    | Owns                                                                                    | Key docs                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Adapters                 | Discord, NapCat QQ, debug UI transport and platform rendering                           | [HOWTO](docs/HOWTO.md)                                                                 |
| Control console          | Local operator auth, service lifecycle, process logs, audit, static UI, debug-chat handoff | [Control Console ICD](src/control_console/README.md)                                  |
| Brain service            | HTTP API, queue, graph startup, health, delivery receipts, runtime adapter registration | [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)                     |
| Message envelope         | Typed inbound content, mentions, replies, attachments, addressees, broadcast state      | [Message Envelope ICD](src/kazusa_ai_chatbot/message_envelope/README.md)               |
| LLM interface            | Backend-compatible chat LLM invocation, provider sessions, diagnostics, and reload retry | [LLM Interface ICD](src/kazusa_ai_chatbot/llm_interface/README.md)                    |
| Conversation progress    | Short-term episode state used by cognition to avoid loops and stale reopenings          | [Conversation Progress](src/kazusa_ai_chatbot/conversation_progress/README.md)         |
| Internal monologue residue | Short-lived private first-person residue loaded only into L2a cognition               | [Internal Monologue Residue ICD](src/kazusa_ai_chatbot/internal_monologue_residue/README.md) |
| Cognition resolver       | Bounded recurrence state, capability observations, HIL/pending resume, and cycle traces | [Cognition Resolver ICD](src/kazusa_ai_chatbot/cognition_resolver/README.md)            |
| RAG 2                    | Slot-driven helper-agent retrieval and Cache2 evidence projection                       | [RAG 2](src/kazusa_ai_chatbot/rag/README.md)                                           |
| Cognition and dialog     | Character stance, boundaries, judgment, style, visual directives, and final wording     | [Cognition Nodes](src/kazusa_ai_chatbot/nodes/README.md)                              |
| Action spec              | L2d action residues, capability registry, evaluator, results, surfaces, and traces      | [Action Spec](src/kazusa_ai_chatbot/action_spec/README.md)                            |
| Consolidation            | Durable target routing, write-intent validation, and target-specific persistence        | [Consolidation ICD](src/kazusa_ai_chatbot/consolidation/README.md)                    |
| Database                 | MongoDB collection ownership, embeddings, indexes, public persistence helpers           | [Database ICD](src/kazusa_ai_chatbot/db/README.md)                                     |
| Event logging            | Sanitized operational telemetry, status snapshots, statistics, and export contracts     | [Event Logging ICD](src/kazusa_ai_chatbot/event_logging/README.md)                     |
| Calendar scheduler       | Durable typed trigger timing for future cognition, commitment due checks, and reflection phase slots | [Calendar Scheduler ICD](src/kazusa_ai_chatbot/calendar_scheduler/README.md) |
| Dispatcher               | Adapter-facing delivery validation and callback transport helpers                       | [Dispatcher](src/kazusa_ai_chatbot/dispatcher/README.md)                               |
| Self-cognition           | Idle source collection, self-cognition episodes, route tracking, and source-bound delivery | [Self-Cognition](src/kazusa_ai_chatbot/self_cognition/README.md)                    |
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

Normal local operation starts the buildless Python/FastAPI control console,
then uses the console to start or stop the brain and adapters:

```powershell
kazusa-control-console --host 127.0.0.1 --port 8765
```

Run the brain service directly only when bypassing the console for
development:

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
  control_console/              Local operator console, lifecycle, logs, audit, static UI
  adapters/                    Platform adapters and debug UI
  kazusa_ai_chatbot/
    brain_service/             Service API, graph, intake, health, post-turn glue
    message_envelope/          Typed adapter-to-brain message contract
    llm_interface/             Chat LLM invocation compatibility layer and ICD
    cognition_resolver/        Bounded resolver loop, capability observations, HIL state
    nodes/                     Persona, cognition, and dialog stages
    action_spec/               Modality-neutral action contracts, registry, results
    consolidation/             Durable consolidation helpers, target routing, and ICD
    rag/                       RAG 2 helper agents, hybrid retrieval, Cache2
    conversation_progress/     Short-term episode memory
    internal_monologue_residue/ Short-lived private residue lane for L2a
    db/                        MongoDB facade, schemas, collection owners
    event_logging/             Sanitized operational telemetry interface and ICD
    calendar_scheduler/        Durable typed trigger scheduler and migration script support
    dispatcher/                Adapter-facing delivery validation and handoff
    self_cognition/            Idle self-cognition triggers, tracking, and delivery
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
service with adapters, memory, retrieval, self-cognition, reflection, and
scheduling, but some autonomous-contact surfaces intentionally remain
permissioned preview contracts rather than production sends.

## Documentation Index

| Document                                                                 | Purpose                                                           |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| [README.md](README.md)                                                   | Project overview and architecture map                             |
| [README_CN.md](README_CN.md)                                             | Simplified Chinese project overview                               |
| [docs/HOWTO.md](docs/HOWTO.md)                                           | Local setup, environment variables, run commands, adapters, tests |
| [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)       | HTTP endpoint contracts and adapter obligations                   |
| [Message Envelope ICD](src/kazusa_ai_chatbot/message_envelope/README.md) | Typed inbound message contract                                    |
| [LLM Interface ICD](src/kazusa_ai_chatbot/llm_interface/README.md)       | Chat model invocation, provider compatibility, and route diagnostics |
| [Database ICD](src/kazusa_ai_chatbot/db/README.md)                       | Persistence ownership and collection contracts                    |
| [Internal Monologue Residue ICD](src/kazusa_ai_chatbot/internal_monologue_residue/README.md) | Short-lived private residue lifecycle and L2a-only contract |
| [Action Spec](src/kazusa_ai_chatbot/action_spec/README.md)               | Modality-neutral action contracts and trace handoff               |
| [Consolidation ICD](src/kazusa_ai_chatbot/consolidation/README.md)       | Durable target routing and write-intent validation                |
| [Event Logging ICD](src/kazusa_ai_chatbot/event_logging/README.md)        | Sanitized telemetry interface, event taxonomy, and ops statistics |
| [Cognition Resolver ICD](src/kazusa_ai_chatbot/cognition_resolver/README.md) | Bounded resolver loop, capability observations, HIL, and traces |
| [RAG 2](src/kazusa_ai_chatbot/rag/README.md)                             | Retrieval architecture and evidence projection                    |
| [Cognition Nodes](src/kazusa_ai_chatbot/nodes/README.md)                 | Layered cognition, dialog, and node-package design contracts      |
| [Self-Cognition](src/kazusa_ai_chatbot/self_cognition/README.md)          | Idle cognition source collection, tracking, and delivery          |
| [Development Plans Registry](development_plans/README.md)                | Active, archived, reference, and roadmap documents                |

## License

Kazusa Cognitive Core is released under the
[GNU Affero General Public License v3.0](LICENSE).
