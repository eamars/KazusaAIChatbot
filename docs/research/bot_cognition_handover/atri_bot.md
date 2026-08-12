# ATRI-bot cognition research handover

Evidence labels: **Fact** is visible in the reviewed upstream snapshot or
source tree; **README claim** is stated by the project documentation and may
need source-level confirmation; **Inference** is the architectural assessment.
The review concerns observable application contracts, not provider-private
chain-of-thought.

## Snapshot

- **Fact:** The requested upstream is [114514ggb/ATRI-bot](https://github.com/114514ggb/ATRI-bot), default branch `main`, reviewed at commit
  `3dcae6d33279ced3ad238ba86460e64e48228582` (`更新下载的方式`, 2026-08-07).
- **Fact:** The project is a NapCat/OneBot QQ group bot using Python,
  PostgreSQL, pgvector, PGroonga, Docker, an event bus, a message queue, a
  pipeline, plugins, and an asynchronous LLM subsystem.
- **Inference:** ATRI-bot is more structured than a simple prompt wrapper—it
  has typed JSON decisions, memory extraction, hybrid retrieval, model
  fallback, and tool loops—but the evidence does not establish an Explicit
  Cognitive Trajectory equivalent to Kazusa’s target architecture.

## Verified source evidence

| Area | Evidence | Classification |
| --- | --- | --- |
| Product architecture | [`README.en.md`](https://github.com/114514ggb/ATRI-bot/blob/3dcae6d33279ced3ad238ba86460e64e48228582/README.en.md) and [`README.md`](https://github.com/114514ggb/ATRI-bot/blob/3dcae6d33279ced3ad238ba86460e64e48228582/README.md) | README claim |
| Chat entry and prompt | `atribot/LLMchat/chat.py`, `prepare_model_prompt.py`, `LLM_supervisor.py` in the [source tree](https://github.com/114514ggb/ATRI-bot/tree/3dcae6d33279ced3ad238ba86460e64e48228582/atribot/LLMchat) | Fact: current source ownership |
| Memory and RAG | `LLMchat/memory/{memory_system,memory_extractor,memory_retriever,memory_consolidator,user_info_system}.py`; `LLMchat/RAG/{rag,vector_store}.py` | Fact: current source ownership |
| Tools and agents | `LLMchat/tools/`, `LLMchat/MCP/`, `LLMchat/agent/`, `LLMchat/sandbox/` | Fact: current source ownership |
| Runtime boundary | `core/platform/`, `core/platform/message_queue.py`, `core/pipeline/`, `core/event_bus/`, `core/type/`, `plugins/` | Fact: current source ownership |
| Decisions | README LLM pipeline describes JSON `speak`, `update`, and `silence` decisions plus Function Calling | README claim |
| Retrieval | README describes pgvector + PGroonga dual recall, RRF fusion, importance/access/time-decay scoring | README claim; implementation modules exist |
| Memory maintenance | README describes LLM extraction, conflict clustering, merge/deduplication, and category half-lives | README claim; maintenance modules exist |

## End-to-end cognition flow

### Source/documented flow

The README describes this path:

1. NapCat delivers events through a OneBot adapter into a message queue.
2. A whitelist pipeline and EventBus route commands, plugin events, ordinary
   chat, and initiative-chat events.
3. `GroupChat.step()`/`PrivateChat` build a prompt from recent group or user
   context, a user profile, recent memory, emoji guidance, and Skills.
4. `LLMCoordinator` sends the primary model request. Function Calling can
   invoke local tools or MCP tools in a loop.
5. The model returns a JSON decision list. `speak` produces segmented text or
   emoji output, `update` changes the user profile, and `silence` produces no
   reply. Tool calls continue through the function-calling loop before the
   final decision.
6. Context is written back; when it exceeds the token limit, old turns are
   compressed into an LLM-generated summary.
7. After chat, memory extraction turns events/preferences into embeddings and
   PostgreSQL rows; later retrieval uses hybrid recall. Scheduled maintenance
   merges conflicts and deletes expired items.

The project also documents active topic participation and a
`schedule_self_trigger` tool that can initiate a later group-chat thought.
These are autonomous contact capabilities, not proof of a persistent semantic
trajectory.

### Assessment

The `speak/update/silence` JSON is a meaningful semantic control carrier and
gives ATRI-bot a stronger decision boundary than free-form text-only bots. It
still does not show separate event appraisal, state interpretation, competing
goal bids, workspace collapse, or a commit record that must precede output.
The `update` branch appears to be a model-selected persistence operation in
the documented flow, so its ownership and validation need careful review before
it can be treated as durable character cognition.

## Memory/retrieval/persona/state

- **README claim:** Short-term memory is a per-group/per-user sliding window;
  over-limit context is summarized by an LLM and inserted into the context.
- **README claim:** Long-term memory extracts per-user events and group topics,
  embeds them into PostgreSQL/pgvector, and recalls with pgvector plus PGroonga
  full-text search. RRF combines the two paths, then importance, access
  frequency, and category-specific time decay affect ranking.
- **README claim:** Categories include preference, fact, experience, emotion,
  group topic, knowledge, domain, and guideline. The documented half-lives
  range from days for group topics to roughly ten years for knowledge/domain/
  guidelines.
- **README claim:** `MemoryConsolidator` builds similarity graphs, clusters
  related memories, asks an LLM to merge or expand content, updates the winning
  memory/vector, deletes redundant rows, and periodically purges expired rows.
- **README claim:** `UserSystem` stores name, relationship, personality, recent
  topics, and style preferences in a per-user JSON profile injected into each
  prompt and updated after conversations.
- **Fact:** The source tree contains the named memory, RAG, user-info, and
  consolidation modules. This verifies ownership and a real implementation
  surface, not every README-described default or runtime path.

**Inference:** ATRI-bot’s memory design is a strong practical retrieval and
maintenance system. It treats memory as facts/preferences/events plus a user
profile, but the supplied evidence does not show universal provenance handles,
episode snapshots, current-evidence conflict precedence, or a formal rule
that retrieved memory cannot become persona stance. Category decay is an
operational lifecycle, not an affect/state-transition model.

## Reasoning/monologue visibility

- **README claim:** The model emits structured JSON decisions rather than only
  prose. The `speak`/`update`/`silence` branches are application-visible
  control results.
- **Fact:** The source tree includes prompt construction, model supervision,
  JSON utilities, agent runners, token management, and tests for tool search.
- **Inference:** These are semantic outputs and operational traces, not a
  protected semantic trajectory. No supplied contract distinguishes model-
  private reasoning, safe decision explanation, and optional character residue
  the way Kazusa’s ECT target does. JSON fields should not be treated as
  permission to expose hidden model reasoning.

## Tools/actions/output

- **README claim:** ATRI-bot supports 17 tools, Function Calling, MCP, Skills,
  web search/extraction, memory search/storage, Python and Shell sandboxing,
  file transfer, speech/image sending, sub-agents, and scheduled self-trigger.
- **Fact:** The current tree contains local tool packages, MCP managers and
  executors, sub-agent runners, Skills parsing/validation, and Docker sandbox
  classes.
- **README claim:** Permission is checked through blacklist/user/admin/root
  levels and command arguments are type-validated.
- **Inference:** Deterministic permission and tool mechanics are an ATRI-bot
  strength. The documented LLM loop still combines semantic decision and tool
  selection in one overall request path; the supplied evidence does not prove
  that a capability result must re-enter a cognition owner before `speak`, or
  that delivery success is separate from the model’s selected decision.
- **README claim:** Output can be segmented and decorated with emojis/stickers;
  a TTS integration is optional. Platform sending is owned by OneBot/network
  modules rather than the prompt itself.

## Bounds/failure/observability

- **README claim:** The system is fully asynchronous, uses queues and database
  pools, rotates model keys, and falls back through configured standby models
  when the primary model fails.
- **README claim:** Short-term context has a token limit and summary path;
  retrieval uses bounded candidate sets (the documented example uses top 40
  per recall path before fusion); memory decay and scheduled cleanup bound
  durable growth.
- **Fact:** The source tree contains timer/rate-limit utilities, cache
  lifecycle management, token management, validation helpers, logs, tests,
  and the pipeline/event abstractions.
- **Inference:** These controls improve availability and capacity, but fallback
  model selection is not semantic regeneration, and retrieval ranking is not
  stance validation. The supplied evidence does not show typed contract-error
  states, bounded semantic regeneration, resolver recurrence caps, or a
  protected replay record explaining why one intention won.

## Strengths

1. Structured JSON `speak/update/silence` decisions make the response branch
   explicit and easier to test than unconstrained prose.
2. Hybrid vector/full-text recall with RRF and time decay is a practical fit
   for noisy group-chat memory; the category model gives retrieval different
   lifetimes.
3. User profiles, memory extraction, consolidation, deduplication, and
   expiry address continuity beyond a fixed prompt window.
4. Asynchronous pipeline/event-bus/plugin design, model fallback, queues, and
   permission checks are sound operational foundations.
5. Sub-agents, MCP, Skills, and sandbox capabilities provide useful extension
   points for research and action.

## Limitations

1. No supplied evidence establishes a typed semantic trajectory with explicit
   appraisals, competing goals, workspace arbitration, selected intention,
   state replacement, and trace lineage.
2. The JSON decision list is structured, but `speak`, `update`, and `silence`
   are top-level outcomes rather than a complete causal model of why a stance
   won or how evidence changed it.
3. User profile and retrieved-memory text are injected into prompts; the
   evidence does not prove a universal boundary between evidence, persona,
   relationship meaning, and final wording.
4. LLM profile updates, memory extraction, and memory merging can mutate
   durable state in background paths. Ownership, conflict policy, privacy, and
   idempotency need source-level audit for high-assurance use.
5. Model fallback protects availability but can change semantic behavior; it is
   not equivalent to stage-owned bounded regeneration with contract evaluation.
6. Proactive scheduling and sandbox/sub-agent capabilities expand agency and
   require explicit target, permission, audit, and delivery boundaries.

## Implications for Kazusa ECT

- Reuse ATRI-bot’s structured decision idea for a compact final semantic
  carrier, but split it into Kazusa-owned stages: appraise the episode, form
  complete evidence-linked bids, collapse the workspace, commit replacement
  state, then plan action/surface/dialog.
- Keep `speak`/`silence` as a possible surface disposition, not the cognition
  engine. Silence must be grounded in the absence of a surviving reason to
  speak, not only a low score or a gate result.
- Treat ATRI memory results as RAG evidence. Preserve source/scope/occurrence
  handles and let cognition interpret whether a remembered preference, event,
  or relationship fact changes the current stance.
- The category half-life and consolidation patterns are useful post-turn
  maintenance ideas, but they should remain outside live wording and pass
  deterministic privacy, ownership, conflict, and promotion gates.
- Put MCP, sub-agent, sandbox, self-trigger, and message-send capabilities
  behind modality-neutral action requests and deterministic authorization. A
  tool’s success must return as an observation; it must not directly create a
  visible completion claim or durable stance.
- Keep model fallback at the provider/runtime layer. If a semantic stage
  fails, use the owning stage’s bounded contract repair or fail-closed policy;
  do not silently substitute an unvalidated semantic decision.

## Evidence matrix

| Dimension | ATRI-bot evidence | Assessment versus ECT |
| --- | --- | --- |
| Admission | OneBot adapter, queue, whitelist pipeline, EventBus, command/chat routes | Strong transport/admission mechanics; semantic admission ownership unresolved |
| Cognition output | JSON `speak`, `update`, `silence` decision list | Better structured control; no complete trajectory contract evidenced |
| Memory | Sliding context, LLM summary, pgvector + PGroonga hybrid RAG, categories/decay | Strong retrieval/maintenance; evidence-vs-stance boundary unresolved |
| Persona | Per-user JSON profile with relationship/personality/preferences | Useful continuity; not a typed multi-axis state reducer |
| Tools | Function Calling, MCP, 17 tools, sub-agents, Skills, sandbox | Broad capability surface; authorization and cognition re-entry need audit |
| Output | Segmented text, emoji/sticker, TTS option, OneBot delivery | Separate transport exists; commit-before-expression not evidenced |
| Bounds | Token compression, top-k recall, fallback models, queues, timer/rate limits | Good availability/capacity; semantic fail-closed behavior unclear |
| Growth | LLM profile updates, memory consolidation, decay | Durable evolution exists as maintenance; promotion/identity gates not evidenced |
| Trace | Logs, token management, tests, pipeline/event records | No supplied protected semantic replay record |
| Typed trajectory | No supplied appraisals, goal bids, workspace, selected intention, state commit schema | Not evidenced in ATRI-bot |

## Open questions

- What exact JSON schema and validator enforce the `return` decision list, and
  can one response contain conflicting `speak` and `silence` items?
- Does `update` mutate the profile immediately, through a deterministic write
  boundary, or only after a post-turn review?
- Which source modules implement the README’s hybrid SQL, category decay,
  consolidation, and profile update paths, and what provenance is retained?
- How are memory/profile conflicts with current messages resolved?
- Does the function-calling loop cap rounds and tool cost independently from
  provider fallback, and what happens when it exhausts its cap?
- Can `schedule_self_trigger` send autonomously only after explicit permission,
  target validation, and adapter delivery checks?
- Which fields are retained in logs or model-visible context, and is there a
  protected semantic replay surface separate from raw prompts and outputs?
