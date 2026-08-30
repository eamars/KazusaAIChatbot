<div align="center">
  <img src="resources/avatar.png" alt="Kazusa avatar" width="420" height="420" />

<h1>Kazusa Cognitive Core</h1>

<p><strong>A self-evolving character cognition runtime for persistent digital presence.</strong></p>

<h2>DSH Standard task runtime</h2>

<p>DSH is Kazusa's sole production route for bounded multi-step task
resolution. The Python Brain owns character judgment, durable task binding,
cognition recurrence, dialog, and delivery; the separately built Node sidecar
owns DSH Standard execution. Build the sidecar with
<code>corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile</code>
and <code>corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build</code>,
then start <code>node sidecars/dsh_resolution/dist/src/main.js</code> after
the Brain is ready.</p>

<p>The runtime uses <code>kazusa.dsh-resolution-rpc.v2</code>,
<code>dsh_resolution_intake.v2</code>, profile
<code>kazusa-resolver-standard-v2</code>, and pinned DSH release
<code>0.1.1-rc.2</code>. Configure the authenticated sidecar, Brain, gateway,
data/workspace/Python paths, and six <code>AGENTIC_RESOLVER_LLM_*</code>
route fields in the <a href="docs/HOWTO.md#dsh-standard-sidecar">HOWTO</a>;
secrets remain local.</p>

<p>
    <a href="README_CN.md">简体中文</a>
    ·
    <a href="docs/HOWTO.md">HOWTO</a>
    ·
    <a href="CHANGELOG.md">Changelog</a>
  </p>

<p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-brain_service-009688?logo=fastapi&logoColor=white" />
    <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-cognition_pipeline-1C3C3C" />
    <img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-memory_store-47A248?logo=mongodb&logoColor=white" />
    <img alt="Release" src="https://img.shields.io/badge/Release-v1.0.0-6f42c1" />
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

For local setup, jump to [Quick Start](#quick-start) and the
[HOWTO](docs/HOWTO.md). For subsystem ownership, use
[Runtime Layers](#runtime-layers).

## DSH Task Runtime

`task_resolution_request` enters one shared `AgenticResolverRuntime`; there is
no parallel production task executor. The Brain creates a durable
`dsh_task_binding.v1`, issues fresh fenced authority, and opens or continues the
DSH sidecar session. Checkpoints and terminal outcomes return as typed
`TaskResolutionResultV1` observations and re-enter normal cognition, dialog,
dispatcher, and adapter delivery.

The wire contracts are `kazusa.dsh-resolution-rpc.v2` and
`dsh_resolution_intake.v2`. Session state lives under
`<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/`; the profile/store identity is
`kazusa-resolver-standard-v2` /
`dsh-sqlite-0.1.1-rc.2-standard-v2`.

DSH Standard retains its native filesystem, shell, coding, jobs, tests, web,
approval, and sandbox tools. Kazusa adds fourteen storage-independent semantic
tools for conversation, memory, people, active context, calendar, attached
media, and public media. The controller-owned `submit_resolution` is the sole
model-owned terminal operation. Semantic results expose bounded entities,
opaque references, and evidence receipts; they never become persona or final
wording by themselves.

Questions, approval decisions, and plan review are internal character-cognition
episodes. The Brain produces a typed answer, rejection, or one-shot grant for
the same fenced thread and segment; these internal decisions are not relayed as
user prompts. Durable accepted-task controls support `continue`, `summarize`,
and `cancel` on the same opaque task/session binding.

See the [DSH integration architecture](docs/architecture/dsh_integration_architecture.md),
[Agentic Resolver architecture](docs/architecture/agentic_resolver_architecture.md),
[DSH interaction README](src/kazusa_ai_chatbot/dsh_interaction/README.md), and
[semantic gateway README](src/kazusa_ai_chatbot/dsh_tool_gateway/README.md).

Core terms used throughout this README:

- **Adapter**: platform transport that normalizes Discord, QQ, debug UI, or
  future events into the brain service API.
- **MessageEnvelope**: the typed inbound message contract consumed by the
  brain, RAG, and cognition stages.
- **RAG3 local context resolver**: cognition-selected local/private context
  evidence resolver; it returns evidence and does not decide persona stance or
  final wording.
- **Cognition resolver**: the bounded L1/L2/L2d loop that decides stance,
  action needs, and whether more evidence is needed.
- **L3/dialog**: the final visible wording stage after cognition has decided
  what kind of surface should exist.
- **Accepted task/background work**: durable delayed work accepted by the
  character, persisted by deterministic code, and re-entered through cognition.

At a high level, Kazusa provides:

| Capability                       | What it means                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Platform-neutral character brain | Discord, QQ, debug UI, and future adapters feed the same FastAPI brain service.                                                    |
| Typed message boundary           | Platform syntax is normalized into `MessageEnvelope` fields before cognition or RAG sees it.                                       |
| Bounded live response path       | Typed intake, frontline relevance, turn settlement, settled relevance, the cognition resolver, selected evidence capabilities, action routing, and L3 surfaces are explicit stages with caps and inspectable payloads. |
| Multi-horizon memory             | Recent chat, short-term conversation flow, retrieved evidence, durable memory, and scheduled commitments remain separate.          |
| Internal monologue residue       | A short private residue lane carries the exact bounded G-stage first-person monologue from completed episodes into the next goal-cognition pass. |
| Task resolution                  | One DSH runtime executes bounded foreground or accepted background work with fenced authority, typed checkpoints, and cognition re-entry. |
| Layered cognition                | Cognition decides stance, boundaries, judgment, style, action needs, and response goals before selected L3 surfaces render output. |
| Background consolidation         | Completed episodes update durable memory, relationship state, Cache2 invalidation, images, and progress from text plus action/surface traces. |
| Accepted delayed work            | Accepted reminders and DSH tasks are persisted, resumed by bounded workers, and returned through cognition rather than sent directly. |
| Reflection outside chat          | Hourly, daily, and promoted reflection runs are stored as audit records and only promoted context can enter normal cognition.      |
| Idle self-cognition              | Background source cases can enter the same resolver-backed persona path, with source-bound delivery and normal consolidation rules. |
| Calendar follow-through          | Accepted future promises and due commitments can become durable calendar triggers that run fresh cognition later.                  |
| Event logging observability      | Runtime, LLM, RAG, action routing, surfaces, reflection, self-cognition, dispatcher, consolidation, and DB operations emit sanitized operational events. |

## What You Can Build

| Use case                             | Why Kazusa fits                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Persistent character companion       | The runtime keeps relationship memory, short-term flow, character state, and reflection separate but connected.                  |
| Group-chat character bot             | Frontline relevance and turn settlement handle noisy channels.                                                                 |
| Local model character lab            | Route-specific OpenAI-compatible model settings let weaker local models handle narrower, staged prompts.                         |
| Memory and RAG experiments           | RAG3, Cache2, scoped user memory, shared memory evolution, and conversation search are modular enough to inspect independently. |
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
| `COGNITION_LLM_CHARACTER_CARRYOVER` | `local-model`                     | `http://localhost:1234/v1` |
| `COGNITION_V3_CHAIN_LLM`   | `local-model`                            | `http://localhost:1234/v1` |
| `COGNITION_V3_SIDECAR_LLM` | `sidecar-model`                          | `http://localhost:1234/v1` |
| `AGENTIC_RESOLVER_LLM`     | `local-model`                            | `http://localhost:1234/v1` |
| `DIALOG_GENERATOR_LLM`     | `deepseek-v4-flash`                      | `https://api.deepseek.com` |
| `CONSOLIDATION_LLM`        | `local-model`                            | `http://localhost:1234/v1` |
| `JSON_REPAIR_LLM`          | `local-model`                            | `http://localhost:1234/v1` |
| `EMBEDDING`                | `text-embedding-nomic-embed-text-v2-moe` | `http://localhost:1234/v1` |

The table is an example, not a fixed requirement. Any route can point to any
OpenAI-compatible endpoint that can satisfy that stage's latency and quality
needs.

`COGNITION_LLM` remains a required shared non-core cognition route. The
agentic cognition runtime requires `COGNITION_V3_CHAIN_LLM` and accepts one
complete optional `COGNITION_V3_SIDECAR_LLM` bundle. Each route owns a
complete endpoint, credential, model, completion-budget, and thinking bundle.
`COGNITION_LLM_CHARACTER_CARRYOVER` is the dedicated state-only background
operational carry-over route and has a maximum completion budget of 8,192
tokens.
The runtime constructs one primary chain and an optional sidecar; the generic
`COGNITION_LLM` route remains shared non-core plumbing. The agentic loop uses
a serialized primary chain with a single-stream sidecar. Its appraisal-stage layout is
fixed as `fixed_a1_a2`; the caller configures
`COGNITION_V3_TURN_DEADLINE_SECONDS` (`30..600`). The request-window ceiling is
50,000 tokens normally and conditionally 65,000 when the declared serving
window supports it. Timing evidence is non-streaming elapsed milliseconds;
the runtime makes no TTFT claim.

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
commonly use LM Studio or another OpenAI-compatible endpoint.

## Architecture At A Glance

This is the complete top-level map, not the shortest path through one chat
turn. Read the solid live path first:
`adapter -> brain service -> queue/intake -> evidence -> cognition -> dialog -> persistence/scheduler`.
Then use the subgraphs as ownership maps for helper agents, resolver
capabilities, web sources, complex-task research, accepted tasks, background
workers, and durable maintenance systems.

Ownership tags in node labels are intentional: `[LLM]` nodes make semantic
judgments, `[deterministic]` nodes validate or move state, and `[worker]` nodes
execute bounded delayed work. Exact subagent naming and documentation
vocabulary are covered by the
[Subagent Interface Guide](docs/SUBAGENT_INTERFACES.md).

The active chat intake path has two bounded relevance decisions. The frontline
route is a compact per-message `discard/start/append` judge. Accepted group
messages settle in a six-second quiet window with a ten-second hard deadline;
the settled route then chooses `ignore/proceed/wait`. Private-message timing
and adjacency-only private coalescing remain intact. The settlement coordinator
owns open-slot projection, bounded silent-prelude promotion, enqueue-time
deadlines, and the pre-deadline ingress barrier. One response owner receives
the assembled reply; appended request futures complete silently. A valid
`proceed` is atomically claimed before persona preparation and cognition run.
For group chat, admission requires either evidence-grounded interaction
relevance—such as typed addressing, an explicit group invitation, complete-name
address, or grounded continuity—or a concrete intersection with bounded active
character state. Relevance owns that semantic judgment. Recipient identity is
kept separate from the reason to speak, allowing a believable state-driven
interjection without treating another participant as the character.
Coalesced private fragments are shown to frontline as one logical input. The
four-image description budget is shared across reassessments, and omitted
media is explicit so settled relevance can fail closed before cognition.

```mermaid
flowchart TD
    A["Adapters<br/>Discord, NapCat QQ, debug UI"]
    B["Brain service<br/>typed intake, queue, relevance"]
    R["RAG3 / local context<br/>bounded evidence"]
    C["Cognition<br/>stance, boundaries, response goals"]
    D["DSH task edge<br/>fenced sidecar session and semantic tools"]
    L["L3 and dialog<br/>visible rendering"]
    P["Persistence and consolidation<br/>memory, progress, traces"]
    S["Scheduler, reflection, self-cognition<br/>outside live wording"]
    O["Dispatcher and adapter delivery"]

    A --> B --> R --> C
    C -->|ordinary response| L
    C -->|task_resolution_request| D
    D -->|typed checkpoint or result| C
    L --> O
    C --> P
    L --> P
    P --> S
    S -->|typed source episode| B
```

Kazusa's live response path is a cognition core, not a chatbot shell or a
generic tool harness. Adapters normalize platform events into the typed service
contract; the brain service owns queueing, identity, reply hydration, history,
episode construction, and graph execution.

### Short-horizon operational carry-over

The singleton character `CharacterCognitionStateV2` is the only persistent
short-horizon global posture. A settled turn first waits for the bounded
predecessor barrier, then reuses one immutable interaction-style snapshot for
relevance, V2 cognition, and L3 surface. Eligible background consolidation may
derive one source-free character operational update through the dedicated
carry-over route; current message, history, and conversation progress remain
the authority for facts and topic.

The latest cognition graph exposes only its source-owned,
`cognition_context_consumption.v1` projection under
`l2.reasoning.detail.context_consumption`. It records bounded consumed
character/relationship/style selections and typed health without source ids,
raw messages, evidence references, prompts, or private facts. The Control
Console renders that payload directly alongside persisted and elapsed-effective
character posture.

RAG3 resolves ordinary local/private context and projects bounded evidence.
`web_agent3` owns approved source retrieval. DSH owns multi-step task execution
through its native and semantic tool catalogs. Background work owns durable DSH
continuation and `future_speak`; none of these evidence or execution owners
decides persona stance or final wording.

The resolver preserves the same L1 -> L2 -> L2d cognition stack on every
cycle. L2d may finish with selected action specs or request a bounded capability
observation. A `task_resolution_request` opens or continues the fenced DSH
session and returns one prompt-safe observation to cognition. The separate
first-cycle shared-memory prewarm may project confirmed shared-memory rows into
L2a; it is not a task result and does not let retrieved evidence become
persona.

Selected visible text surfaces go back to adapters through `ChatResponse` and
delivery receipts. Private action results, no-visible-output decisions, and
surface traces can still feed post-turn progress, consolidation, Cache2
invalidation, residue recording, calendar state, reflection, and
self-cognition without creating a platform send.

Current-turn cognition partitions model-facing context into
`current_observation`, `direct_facts`, `participant_continuity`,
`conditional_character_context`, and `continuation_state`. G owns the exact
private first-person motivation and P owns the visible assertion boundary. L3
uses both as typed expression context in one semantic content-planning call;
deterministic code copies the caller-owned addressee plan and empty visible
boundary list. The public text-surface output copies P's exact epistemic
boundary for dialog, while the private monologue remains outside the dialog
payload. Dialog may describe a physical or external effect as completed only
when an `executed` permitted-action result supplies that fact.

Generic task work starts through `task_resolution_request`. Foreground work
uses a bounded DSH budget; a committed checkpoint can promote the same session
to an accepted task and task-orchestrator job. `future_speak` remains a separate
scheduled action lifecycle. Completed accepted tasks return as canonical
`tool_result` cognition sources, and status checks read current task state
without creating work.

## Real Debug Example Flows

The first three examples below were captured through the real debug `/chat`
interface, which sends the same typed chat request shape into the brain service
as runtime adapters. Example 4 illustrates a bounded DSH research result that
returns through cognition rather than bypassing the character brain. The examples were captured on July 2, 2026, then translated
to English and condensed for a README audience. They are not full trace dumps.
Internal ids, cache keys, raw database rows, and implementation field names are
intentionally omitted. The diagrams render typed payloads as readable prose.

Read each diagram from left to right. Every example uses the same five
checkpoints:

1. **Message / Request** is what the chat platform or debug client receives.
2. **Extract** is a human-readable summary of the typed, platform-neutral
   message envelope and hydrated context the brain receives.
3. **Context / Evidence** is retrieved conversation evidence, reply context, or
   structured task state used for the decision.
4. **Decision** is the character-level judgment for chat turns, or the
   task-level synthesis rule for bounded DSH results.
5. **Output** is what the user sees, the durable handoff created for later
   work, or the semantic packet returned to the next stage.

This mirrors the system boundary: adapters normalize platform events, RAG
returns evidence, cognition decides the character stance, dialog owns visible
wording, and deterministic subsystems own validation, persistence, scheduling,
adapter delivery, and durable task lifecycle.

### Example 1: Private Continuity Recall

This private-chat example shows how the system answers a follow-up question by
using recent conversation context instead of treating the message as isolated.

```mermaid
flowchart TD
    A["1. Message<br/>Kazusa, do you remember what I was worried about for tomorrow?"]
    B["2. Extract<br/>Private follow-up. The user is checking whether Kazusa remembers a specific prior worry."]
    C["3. Context / Evidence<br/>Recent conversation: the user was worried about going blank during technical interview questions."]
    D["4. Decision<br/>Treat this as a continuity and trust check. Answer from remembered context, then lightly check on the user."]
    E["5. Output<br/>I remember. You were worried you might go blank when technical questions come up in tomorrow's interview, and that was making you nervous. Are you okay right now?"]

    A --> B --> C --> D --> E
```

The important transfer is the remembered concern. The adapter only needs to
send a clean private message into the brain. RAG/retrieval supplies the earlier
interview worry as evidence, but it does not write the reply. Cognition decides
that the user is testing continuity, so the dialog response confirms the memory
and adds a small emotional check-in.

### Example 2: Group Reply And Mention Grounding

This group-chat example shows how a reply target and a direct mention become
semantic context. The character understands both the technical question and
the social pressure of being asked to take sides.

```mermaid
flowchart TD
    A["1. Message<br/>@Kazusa do you agree with Alex, or is the quality drop too risky?"]
    B["2. Extract<br/>Group message. The user directly mentions Kazusa and replies to Alex's proposal."]
    C["3. Context / Evidence<br/>Reply context: Alex suggested deploying the smaller model first."]
    D["4. Decision<br/>Answer the direct question, but avoid casually choosing sides in a group disagreement."]
    E["5. Output<br/>About what Alex said... starting with a smaller model, I am not sure I can say which is better. A quality drop is definitely something to worry about, but I should not casually pick a side."]

    A --> B --> C --> D --> E
```

The important transfer is the combination of direct address and reply context.
The adapter normalizes platform-specific mention and reply syntax into typed
envelope fields; the README diagram renders those fields as readable prose.
Cognition can then judge the social situation: Kazusa was invited into a
disagreement, so the visible answer acknowledges the quality risk without
pretending to have enough basis to overrule either person.

### Example 3: Accepted Future Reminder Handoff

This example shows a user-facing delayed task. The character accepts the
reminder in the live chat, while deterministic subsystems create durable work
for the future.

```mermaid
flowchart TD
    A["1. Message<br/>Kazusa, please remind me on 2026-07-04 at 09:00 to review my interview notes."]
    B["2. Extract<br/>Future reminder request. Time: 2026-07-04 09:00. Reminder: review interview notes."]
    C["3. Context / Evidence<br/>Structured task state: requester, future time, reminder objective, and chat scope."]
    D["4. Decision<br/>Accept the low-pressure request and acknowledge the exact time and objective."]
    E["5. Output<br/>Visible reply: Got it. July 4, 2026 at 9 AM, I will remind you to review your interview notes.<br/>Durable handoff: accepted task persisted; future_speak/background_work schedules future cognition.<br/>At due time, self-cognition, dialog, and dispatcher decide the actual send."]

    A --> B --> C --> D --> E
```

The important transfer is the future task, not the queue machinery. Cognition
decides whether Kazusa should accept the reminder. After that decision,
deterministic code stores the accepted task and queues the internal future
work. In implementation terms, cognition selects a `future_speak` action spec;
deterministic execution persists its accepted task and schedules a
`future_cognition` calendar run. At due time, self-cognition, dialog, and
dispatcher decide whether and how to send the reminder. The background worker
does not write final chat text directly.

### Example 4: Complex Public Research Packet

This DSH case shows how a broad benchmark request is decomposed into
source-bound evidence and a comparison packet. The typed result returns to
cognition before any visible answer is rendered. The benchmark numbers are captured trace content from July 2, 2026,
not current hardware guidance.

```mermaid
flowchart TD
    A["1. Message<br/>Compare RTX5090 with R9700 in terms of Qwen3.6 27B and 35B, and Gemma4 31/26B performance, with Q4 if possible."]
    B["2. Extract<br/>Public benchmark task. Treat R9700 as the AMD 32GB GPU target used by the collected evidence. Compare RTX 5090 and R9700 across Qwen3.6 27B/35B and Gemma4 31B/26B. Include Q4 quantization when evidence exists."]
    C["3. Context / Evidence<br/>RTX 5090 branch: Qwen3.6 27B about 130 tokens/s on dual RTX 5090 FP8; Gemma4 31B about 231 tokens/s in a coding task; Gemma4 26B Q4_K_M runnable with about 16GB VRAM.<br/>R9700 branch: source-reported Qwen3.6 35B and 27B throughput in the low-40 tokens/s range; Gemma4 31B about 39 tokens/s; Gemma4 26B availability noted, exact R9700 throughput unclear."]
    D["4. Decision<br/>Return a bounded knowledge packet. Compare only source-supported values, preserve caveats, and mark missing same-prompt head-to-head data."]
    E["5. Output<br/>Investigation packet: captured source snippets favored RTX 5090 for speed and setup maturity, while R9700 remained viable but backend-sensitive. Direct same-prompt Q4 comparisons and several model-specific throughputs were still missing."]

    A --> B --> C --> D --> E
```

The captured DSH work tree shows how the task is broken down. The planner first
separates evidence collection from comparison. Evidence branches collect facts
for each GPU, model-availability checks reuse already-collected evidence, and
the final packet keeps unsupported comparisons explicit.

```mermaid
flowchart TD
    R["Root<br/>Compare RTX5090 vs R9700 for Qwen3.6 27B/35B and Gemma4 31B/26B, Q4 if possible"]
    P["Planner split<br/>Collect benchmark evidence first, then compare metrics"]
    A["RTX 5090 evidence<br/>Qwen3.6 27B: about 130 tokens/s on dual RTX 5090 FP8<br/>Gemma4 31B: about 231 tokens/s in a coding task<br/>Gemma4 26B: Q4_K_M runnable, about 16GB VRAM<br/>Qwen3.6 35B: viable, no precise throughput found"]
    B["R9700 evidence<br/>Qwen3.6 35B and 27B: source-reported low-40 tokens/s range<br/>Gemma4 31B: about 39 tokens/s<br/>Gemma4 26B: related availability found, exact R9700 throughput unclear"]
    C["Model availability check<br/>Reuse the RTX 5090 and R9700 benchmark branches because they already carry Qwen3.6 and Gemma4 evidence"]
    D["Comparison packet<br/>Captured evidence favored RTX 5090 for speed and setup maturity<br/>Still needed: same prompt and hardware setup, RTX 5090 Qwen3.6 35B number, R9700 Gemma4 26B number"]

    R --> P
    R --> D
    P --> A
    P --> B
    P --> C
    C -. reuses .-> A
    C -. reuses .-> B
    A --> D
    B --> D
```

The important transfer is the boundary between evidence and conclusion. The
resolver turns one broad request into smaller evidence jobs. Each job returns a
short source-bound summary plus caveats. When a later branch asks something
already answered, the tree points back to that existing evidence instead of
treating it as a new fact. The final packet is useful to an AI developer
because it separates what the system can say now from what still needs
verification before making a confident public comparison.

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
the exact compact current-turn `private_monologue` produced by G and projects
post-turn residue only into goal cognition as
`internal_monologue_residue_context`. Prior residue is not
`reflection_summary`, durable memory, visible dialog planning, or calendar
input.

**Reflection does not shortcut into live chat**

Reflection is slower sense-making work. Raw reflection output is stored for
inspection, but normal cognition only receives bounded, promoted, gated context.
The reflection worker also owns the daily sleep/wake affect-settling pass that
smooths persistent character mood and global vibe outside the live response
path.

**Adapters are transport edges**

Platform adapters parse platform events, normalize typed envelopes, call the
brain service, and deliver returned messages. Character identity, memory, RAG,
cognition, and calendar scheduling remain in the platform-neutral core.

## Runtime Layers

| Layer                    | Owns                                                                                    | Key docs                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Adapters                 | Discord, NapCat QQ, debug UI transport and platform rendering                           | [Adapter ICD](src/adapters/README.md), [HOWTO](docs/HOWTO.md#adapters)                |
| Control console          | Local operator auth, service lifecycle, process logs, audit, static UI, debug-chat handoff | [Control Console ICD](src/control_console/README.md)                                  |
| Brain service            | HTTP API, queue, graph startup, health, delivery receipts, runtime adapter registration | [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)                     |
| DSH interaction          | Brain-owned internal approval/question/plan-review judgment and one-shot grants | [DSH interaction](src/kazusa_ai_chatbot/dsh_interaction/README.md) |
| DSH semantic gateway     | Typed storage-independent semantic tools, opaque references, evidence, and replay-safe worker | [DSH tool gateway](src/kazusa_ai_chatbot/dsh_tool_gateway/README.md) |
| DSH Standard sidecar     | Mounted official DSH runtime, native tools, sessions, checkpoints, and terminal receipts | [Sidecar README](sidecars/dsh_resolution/README.md) |
| Message envelope         | Typed inbound content, mentions, replies, attachments, addressees, broadcast state      | [Message Envelope ICD](src/kazusa_ai_chatbot/message_envelope/README.md)               |
| LLM interface            | Backend-compatible chat LLM invocation, provider sessions, diagnostics, and reload retry | [LLM Interface ICD](src/kazusa_ai_chatbot/llm_interface/README.md)                    |
| Conversation progress    | Short-term episode state used by cognition to avoid loops and stale reopenings          | [Conversation Progress](src/kazusa_ai_chatbot/conversation_progress/README.md)         |
| Internal monologue residue | Short-lived private first-person residue loaded only into L2a cognition               | [Internal Monologue Residue ICD](src/kazusa_ai_chatbot/internal_monologue_residue/README.md) |
| Cognition resolver       | Bounded recurrence state, capability observations, pending user prerequisites, and cycle traces | [Cognition Resolver ICD](src/kazusa_ai_chatbot/cognition_resolver/README.md)            |
| Task resolution          | DSH admission, durable task binding, checkpoint recurrence, and terminal result projection | [Task Resolution ICD](src/kazusa_ai_chatbot/task_resolution/README.md)                 |
| Local context resolver   | RAG3 local/private evidence graph and prompt-safe evidence projection                    | [Local Context Resolver ICD](src/kazusa_ai_chatbot/local_context_resolver/README.md)   |
| Cognition and dialog     | Character stance, boundaries, judgment, style, visual directives, and final wording     | [Cognition Nodes](src/kazusa_ai_chatbot/nodes/README.md)                              |
| Action spec              | L2d action residues, capability registry, evaluator, results, surfaces, and traces      | [Action Spec](src/kazusa_ai_chatbot/action_spec/README.md)                            |
| Accepted task            | User-facing lifecycle for delayed work accepted by the character                        | [Accepted Task ICD](src/kazusa_ai_chatbot/accepted_task/README.md)                    |
| Background work          | Internal task-orchestrator/future-speak execution and result handoff                    | [Background Work ICD](src/kazusa_ai_chatbot/background_work/README.md)                |
| Consolidation            | Durable target planning, lane routing/review, write-intent validation, and target-specific persistence | [Consolidation ICD](src/kazusa_ai_chatbot/consolidation/README.md)                    |
| Database                 | MongoDB collection ownership, embeddings, indexes, public persistence helpers           | [Database ICD](src/kazusa_ai_chatbot/db/README.md)                                     |
| Event logging            | Sanitized operational telemetry, status snapshots, statistics, and export contracts     | [Event Logging ICD](src/kazusa_ai_chatbot/event_logging/README.md)                     |
| Calendar scheduler       | Durable typed trigger timing for future cognition, commitment due checks, and reflection phase slots | [Calendar Scheduler ICD](src/kazusa_ai_chatbot/calendar_scheduler/README.md) |
| Dispatcher               | Adapter-facing delivery validation and callback transport helpers                       | [Dispatcher](src/kazusa_ai_chatbot/dispatcher/README.md)                               |
| Self-cognition           | Idle source collection, self-cognition episodes, route tracking, and source-bound delivery | [Self-Cognition](src/kazusa_ai_chatbot/self_cognition/README.md)                    |
| Reflection cycle         | Background reflection runs, promotion gates, prompt-safe reflection context             | [Reflection Cycle ICD](src/kazusa_ai_chatbot/reflection_cycle/README.md)               |
| Memory evolution         | Curated shared memory lifecycle, lineage, seed reset, promoted memory writes            | [Memory Evolution ICD](src/kazusa_ai_chatbot/memory_evolution/README.md)               |
| Character identity growth | Reviewed, root-counted global identity revisions and latest-only runtime projection     | [Character Identity Growth](src/kazusa_ai_chatbot/character_identity_growth/README.md) |
| Episode trace and lifecycle | Immutable `episode_trace.v2` settlement and idempotent post-turn audit records       | [Brain Service ICD](src/kazusa_ai_chatbot/brain_service/README.md)                    |

Other project documents:

| Document                                                | Purpose                                                           |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| [README_CN.md](README_CN.md)                            | Simplified Chinese project overview                               |
| [docs/HOWTO.md](docs/HOWTO.md)                          | Local setup, environment variables, run commands, adapters, tests |
| [Documentation Guide](docs/DOCUMENTATION_GUIDE.md)      | Document roles, source hierarchy, module README rules, parity     |
| [Subagent Interface Guide](docs/SUBAGENT_INTERFACES.md) | Cross-family subagent and worker documentation vocabulary         |
| [Future Architecture](docs/FUTURE_ARCHITECTURE.md)      | Independently maintained future architecture direction           |
| [Architecture References](docs/architecture/)            | Current architecture and contract references                     |
| [Development Plans Registry](development_plans/README.md) | Active plans, archive, and long-term roadmap                    |

## Quick Start

Kazusa expects MongoDB plus OpenAI-compatible chat and embedding endpoints. LM
Studio works for local development, but any compatible endpoint can be used.
Before starting the service, create a local `.env` with MongoDB, chat route,
and embedding settings. All route-specific model environment variables are
documented in [docs/HOWTO.md](docs/HOWTO.md#local-setup).

```powershell
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

The service requires a manually seeded character identity ledger. Before the
first startup against a clean database, load one complete canonical profile:

```powershell
venv\Scripts\python -m scripts.load_character_profile personalities\example.json
```

Startup reads the latest immutable identity revision and fails before intake
when no revision exists. Existing ledgers remain database-authoritative. A
replacement profile requires the explicit revisioned operator-reset command
documented in [docs/HOWTO.md](docs/HOWTO.md#character-profile).

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

### Agentic cognition startup

Operators configure the complete `COGNITION_V3_CHAIN_LLM_*` bundle,
optionally set the all-or-nothing
`COGNITION_V3_SIDECAR_LLM_*` bundle, and use the fixed `fixed_a1_a2`
appraisal-stage layout with the configured turn deadline and context window
described in `docs/HOWTO.md`. The former engine selector and per-stage route
bundles are removed from the runtime contract.

Useful read-only evidence commands are:

```powershell
venv\Scripts\python -m pytest -q tests/integration/cognition_core_v3/test_chain_observability.py
venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD
```

### DSH Standard startup

Start the Brain with MongoDB and wait for `/health` plus authenticated
`/runtime/dsh/health` to report its durable DSH store and cognition judge.
Build/start `sidecars/dsh_resolution` as a separate process, then require
authenticated `system.health` to report `route`, `standard`,
`semantic_worker`, `web`, and `brain` readiness. The sidecar and Brain may be
restarted independently; readiness still follows that dependency. Required
DSH and route settings are in the [HOWTO DSH runbook](docs/HOWTO.md#dsh-standard-sidecar).

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
    dsh_interaction/           Brain-owned internal DSH judgment and grant lineage
    dsh_tool_gateway/          Typed semantic catalog and framed worker boundary
    message_envelope/          Typed adapter-to-brain message contract
    llm_interface/             Chat LLM invocation compatibility layer and ICD
    cognition_resolver/        Bounded resolver loop and capability observations
    nodes/                     Persona, cognition, and dialog stages
    action_spec/               Modality-neutral action contracts, registry, results
    accepted_task/             User-facing accepted delayed-work lifecycle
    background_work/           Internal task-orchestrator and future-speak execution
    task_resolution/           DSH task admission, binding, continuation, and result projection
    consolidation/             Durable consolidation helpers, lane routing, and ICD
    local_context_resolver/    RAG3 local/private evidence graph and retained projection
    rag/                       RAG3 evidence leaves, retrieval utilities, and Cache2 policy
    conversation_progress/     Short-term episode memory
    internal_monologue_residue/ Short-lived private residue lane for L2a
    db/                        MongoDB facade, schemas, collection owners
    event_logging/             Sanitized operational telemetry interface and ICD
    calendar_scheduler/        Durable typed trigger scheduler and migration script support
    dispatcher/                Adapter-facing delivery validation and handoff
    self_cognition/            Idle self-cognition triggers, tracking, and delivery
    reflection_cycle/          Background reflection and promotion
    memory_evolution/          Shared memory lifecycle and seed reset
    character_identity_growth/ Reviewed identity semantics and runtime projection
    character_profile.py       Canonical manual-seed profile validation
    db/internal_action_latches.py  Durable internal-thought continuation latches
  scripts/                     Operator and maintenance CLIs
sidecars/
  dsh_resolution/              Mounted official DSH Standard runtime and RPC sidecar
docs/
  HOWTO.md                     Setup, runtime commands, environment, tests
  FUTURE_ARCHITECTURE.md       Independently maintained future architecture
  architecture/                Current architecture and contract references
development_plans/             Active plans, historical archive, and roadmap
tests/                         Deterministic, live DB, and live LLM test suites
resources/
  avatar.png                   README avatar asset
```

## Testing

Default test runs exclude live DB and live LLM tests through `pytest.ini`.

```powershell
venv\Scripts\python -m pytest -q
```

When a production module under cognition, cognition-resolver, or a named
direct-node ownership boundary changes, run the exact source-to-test impact
check from the recorded baseline:

```powershell
venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run
```

The command validates the ownership manifest, verifies exact pytest node
collection, and runs the mapped deterministic unit tests. The canonical unit
layout mirrors source modules under `tests/unit/`; integration and live-LLM
tests remain supplemental evidence.

Live LLM tests must be run one case at a time with output inspected. Live DB
tests require MongoDB. See [docs/HOWTO.md](docs/HOWTO.md#testing) for the
project testing contract.

## Project Status

Kazusa Cognitive Core v1.0.0 is the first stable release of the local runtime
for a persistent digital character. The main runtime provides adapters,
memory, retrieval, self-cognition, reflection, and scheduling through one
inspectable brain service. Some autonomous-contact surfaces intentionally
remain permissioned preview contracts rather than production sends; operators
must configure permissions and adapter delivery explicitly before enabling
them.

## License

Kazusa Cognitive Core is released under the
[GNU Affero General Public License v3.0](LICENSE).
