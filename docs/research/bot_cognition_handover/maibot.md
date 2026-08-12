# MaiBot cognition research handover

Research status: complete. Evidence was collected on 2026-08-12 from a
read-only shallow checkout and official MaiBot documentation. This handover is
the authoritative result; it deliberately discusses observable application
contracts and provider-facing fields, not private token-level chain-of-thought.

Evidence labels used below:

- **Verified from source**: implemented behavior observed in the pinned source.
- **Documentation claim**: stated by official documentation but not treated as
  implementation proof until matched to source.
- **Inference/assessment**: architectural implication derived from the verified
  evidence, including the Kazusa comparison.

## 1. Project identity and snapshot

**Verified from official sources and checkout**

- Canonical project: **MaiBot / MaiSaka**, official repository
  [Mai-with-u/MaiBot](https://github.com/Mai-with-u/MaiBot).
- Official quick-start documentation instructs users to clone that repository
  and identifies `dev` as the development branch:
  [Getting started](https://docs.mai-mai.org/en/manual/getting-started/).
- Research checkout: `main`, commit
  `1b8d13300a5add5f029106b0ee2d80b59996917c` (`1b8d133`), timestamp
  `2026-08-12T02:05:49Z`, subject `docs: 更新仓库状态图`.
- The current `dev` ref was separately observed at
  `240725cbb8e4573a4f08534b844cd2b85eff13ba`. Tag `1.0.0-rc.2` was observed
  at the older commit `864ea04e74b91f0d188f219d8e29af367e781aa6`; it is not the
  snapshot used for source claims.
- The repository describes itself as a lifelike LLM agent/digital lifeform
  with a plugin system, rather than a feature-complete generic assistant. That
  is product positioning, not proof of a particular cognition mechanism:
  [official repository description](https://github.com/Mai-with-u/MaiBot).

**Scope and confidence**

Confidence is high for the Planner/replyer loop, tool routing, memory
boundaries, and source-level bounds listed below. Confidence is medium for
configuration combinations because no live MaiBot service or database was
started. Confidence is low for documentation-only claims where the source did
not expose an equivalent contract.

## 2. Evidence inventory

| Area | Verified source evidence | Status / significance |
| --- | --- | --- |
| Product flow | `prompts/zh-CN/maisaka_chat.prompt:1-31`; `src/maisaka/reasoning_engine.py:972-1126` | Planner analyzes, acts, and can run multiple internal rounds. |
| Planner contract | `prompts/zh-CN/maisaka_chat.prompt:1-31`; `prompts/zh-CN/maisaka_chat_focus.prompt:1-38` | Free-form explicit analysis is an application-owned decision artifact; `reply` is the visible-output route. |
| Planner request assembly | `src/maisaka/chat_loop_service.py:906-1122`; `:1125-1249` | Context selection, injected references, tools, hooks, provider response, and native reasoning are separate fields. |
| Admission/relevance | `src/maisaka/reply_necessity.py:8-23,136-194,212-236`; `src/maisaka/turn_gates.py:38-99,118-196`; `src/maisaka/turn_scheduler.py:17-132` | Default source path performs deterministic pre-Planner scheduling using mention/content/pressure/frequency signals. |
| Actions and tools | `src/maisaka/reasoning_engine.py:428-510,2001-2101`; `src/maisaka/builtin_tool/tool_search.py:10-90`; `src/maisaka/builtin_tool/wait.py:10-70` | Visible and deferred tools, sequential execution, structured failures, and bounded waiting are implemented. |
| Visible reply | `src/maisaka/builtin_tool/reply.py:83-195,295-574`; `src/chat/replyer/maisaka_generator_base.py:539-724,875-1280` | Planner chooses when/where; replyer owns wording; deterministic post-process and delivery follow. |
| Mid-term memory | `src/maisaka/memory/mid_term.py:37-46,99-194,316-377,606-722,849-947`; `prompts/zh-CN/mid_term_memory_summary.prompt:1-10` | Short-term history is summarized and recalled by embedding; references are marked internal. |
| Long-term memory | `src/A_memorix/README.md:18-75`; `src/A_memorix/core/runtime/services/memory_search_service.py:19-153,263-366`; `src/A_memorix/core/retrieval/dual_path.py:101-132,1177-1429` | A_Memorix implements vector/graph/sparse retrieval, scopes, modes, filtering, and optional PPR. |
| Person state | `src/maisaka/memory/person_profile.py:197-276`; `src/A_memorix/core/utils/person_profile_service.py:1048-1239`; `src/A_memorix/core/runtime/services/profile_admin_service.py:19-60,249-296` | Snapshot-based person profiles, relationship text, evidence, TTL, and manual override exist. |
| Background persistence | `src/services/memory_flow_service.py:29-128,333-424,427-541,666-698`; `src/A_memorix/core/runtime/services/ingest_service.py:27-116,243-462` | Sent replies can enqueue person-fact and chat-summary writeback; writes are evidence-aware and often asynchronous. |
| Episode/consolidation memory | `src/A_memorix/core/utils/episode_service.py:1-9,404-498`; `src/A_memorix/core/runtime/services/ingest_service.py:493-644`; `src/A_memorix/core/utils/summary_importer.py:596-769` | LLM segmentation/import is bounded by validation, deterministic fallback, leases, CAS publication, and retry state. |
| Reasoning visibility | `src/maisaka/chat_loop_service.py:80-100,1031-1122`; `src/maisaka/display/runtime_mixin.py:34-56,592-690`; `src/maisaka/monitor/events.py:19-20,300-336,540-584` | Explicit Planner text can be shown to operator/debug surfaces; native provider reasoning is stored separately. |
| Observability | `src/maisaka/monitor/event_store.py:17-42,81-128`; `src/maisaka/monitor/events.py:403-449` | Monitor events are persisted with sanitization and bounded retention; retries/errors are observable. |
| Optionality | `src/config/official_configs.py:1050-1112,1318-1572,1848-2027,2585-2789,4960-5039` | A_Memorix, heuristic recall, behavior learning, rich reply, PPR, relation vectors, and display/debug features are configuration-controlled; several are disabled by default or marked experimental. |

Official feature descriptions used for claim/source comparison:

- [Message pipeline](https://docs.mai-mai.org/en/manual/features/message-pipeline)
  describes receive/check/context/decision/render/send.
- [MaiSaka reasoning](https://docs.mai-mai.org/en/manual/features/maisaka-reasoning)
  claims listening, deciding whether to speak, memory gathering, iterative
  thinking, and learning after conversation.
- [Memory system](https://docs.mai-mai.org/en/manual/features/memory-system)
  describes optional long-term memory and automatic extraction/update/merge/
  decay; the source confirms substantial memory machinery but the optional
  configuration and background boundaries matter.
- [A_Memorix configuration](https://docs.mai-mai.org/en/manual/configuration/amemorix-config)
  documents vector/graph retrieval settings, top-k, PPR, timeout, and
  concurrency controls.
- [Event pipeline hooks](https://docs.mai-mai.org/en/develop/event-pipeline-hooks)
  documents EventBus/HookDispatcher extension points; hooks remain optional
  extension surfaces, not core cognition ownership.

## 3. End-to-end cognition flow

### 3.1 Verified from source

1. **Inbound message admission and scheduling.** The runtime receives a
   normalized `SessionMessage`, updates its pending cache, focus state, and
   optional reply-effect observer, then schedules a turn. With reply-necessity
   mode enabled, `ReplyNecessityTurnGate` scores direct address, mentions,
   questions/requests/opinions, pressure, frequency, and recent self presence;
   the hard trigger is score `>= 80`. The legacy frequency path uses a pending
   message threshold and bounded idle compensation. A pure idle period cannot
   trigger with no real pending message. See
   [`reply_necessity.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reply_necessity.py#L136-L194),
   [`turn_gates.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/turn_gates.py#L38-L196),
   and [`turn_scheduler.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/turn_scheduler.py#L60-L134).

2. **Planner context construction.** `chat_loop_step` selects recent context,
   preserves tool-call/result pairs and context-restore references, injects
   deferred-tool reminders, optional heuristic memory, person profiles, jargon,
   and mid-term memory references, then calls the configured Planner with the
   current visible tool definitions. Planner history explicitly filters the
   mid-term summary message because the selected recall reference is injected
   separately. See [`chat_loop_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/chat_loop_service.py#L906-L1122)
   and [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L428-L557).

3. **Semantic decision and action loop.** The Planner prompt asks for current
   judgment, suggestion, next plan, and reasons, while making the Planner
   distinct from the character and replyer. It can call `query_memory`, profile
   lookup, `reply`, `wait`, focus tools, browser/plugin/MCP tools, and deferred
   tools discovered through `tool_search`. Tool calls append their results to
   the context and continue another internal round. A no-tool Planner response
   ends the current thinking cycle immediately; the function name mentions
   retry, but the implementation ends the cycle rather than regenerating. See
   [`maisaka_chat.prompt`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/prompts/zh-CN/maisaka_chat.prompt#L1-L31)
   and [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L574-L672).

4. **Interruption and bounded recurrence.** New messages can interrupt a
   streaming Planner request. The runtime waits for the configured quiet period,
   ingests the new messages, and retries the Planner while pending messages and
   the internal-round budget permit. The loop is bounded by
   `_max_internal_rounds = MAX_INTERNAL_ROUNDS`; the constant's declaration was
   not needed to establish the boundary and its numeric value remains an
   unresolved snapshot detail. `planner_interrupt_max_consecutive_count` is a
   separate config bound and defaults to `0` in the current source config. See
   [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L733-L797,L972-L1091)
   and [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L600-L644).

5. **Visible wording and delivery.** Only `reply` creates visible output.
   It validates the target message, passes the Planner's explicit current
   thought/guide/reference to the MaiSaka replyer, and lets the replyer read
   real chat history. The replyer system prompt requests only the visible
   colloquial response. Post-processing can split/correct the text, then the
   deterministic send service delivers segments and records the guided reply.
   Rich attachments and structured expression intent are optional/experimental.
   See [`reply.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/builtin_tool/reply.py#L83-L195,L295-L574)
   and [`maisaka_generator_base.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/chat/replyer/maisaka_generator_base.py#L539-L724,L875-L1280).

6. **Post-turn memory and learning.** When context is trimmed, MaiBot may
   create a mid-term summary, refresh behavior references, and launch background
   expression/behavior/jargon learning. Separately, after a sent reply, bounded
   workers can extract user-supported facts and create automatic chat summaries
   after a message threshold. These paths do not determine the already-selected
   visible reply. See [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L1318-L1414)
   and [`memory_flow_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/services/memory_flow_service.py#L427-L541,L666-L698).

### 3.2 Documentation claim versus implementation assessment

The official pipeline/reasoning pages accurately describe the broad shape:
receive, decide whether to speak, gather context, reason over multiple rounds,
render, and learn afterward. Source confirms the Planner/action loop and
post-turn learning paths. The documentation's human-like language should not
be read as evidence of a persistent cognitive-state transition model: the
current source has a Planner loop, profiles, memory, behavior/style learning,
and runtime state, but no single typed affect/relationship trajectory
equivalent to Kazusa ECT.

## 4. Memory, retrieval, and persistence boundaries

### 4.1 Verified from source

- **Mid-term chat recall** is a bounded context-compaction mechanism. It selects
  user messages from trimmed history, sends at most `16,000` source characters
  to a summary prompt, stores a summary plus up to five recall cues, and keeps
  summary/reference text bounded. A later query is embedded and the best cue is
  recalled only above cosine threshold `0.8`; the injected reference says it is
  internal and should not be quoted verbatim. Parse failure skips insertion.
  [`mid_term.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/memory/mid_term.py#L37-L46,L99-L194,L316-L377,L606-L722,L849-L947)
- **Person profile state** is evidence-derived and snapshot-based. The profile
  can contain identity settings, relationship settings, stable facts,
  interaction preferences, recent interactions, and uncertain notes. It has
  an evidence fingerprint, TTL/cache reuse, and a separate manual override.
  Planner profile injection is bounded to at most three people by current
  config and is marked internal. [`person_profile.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/memory/person_profile.py#L197-L276),
  [`person_profile_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/utils/person_profile_service.py#L1048-L1239).
- **Heuristic long-term recall** is explicitly optional and defaults to
  disabled in `AMemorixIntegrationConfig`. When enabled it requires a minimum
  message window, has a cache TTL/cooldown/new-message threshold, limits the
  number and size of hits, and applies same-chat/person and cross-chat policy
  checks. [`heuristic_injector.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/memory/heuristic_injector.py#L34-L133,L168-L242,L255-L409),
  [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L1395-L1528).
- **A_Memorix retrieval** supports `search`, `time`, `hybrid`, `episode`, and
  `aggregate`; it filters chat/person scope, removes non-user-visible hits,
  applies dynamic thresholds, can use vector and BM25 paths, graph relation
  recall, and optional Personalized PageRank. Defaults include top-k limits,
  PPR alpha `0.85`, PPR timeout `1.5s`, concurrency `4`, and parallel retrieval
  in the current config/source. Relation vectorization is separately
  configurable and documented as off by default. [`A_memorix/README.md`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/README.md#L18-L75),
  [`memory_search_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/runtime/services/memory_search_service.py#L19-L153),
  [`dual_path.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/retrieval/dual_path.py#L101-L132,L1177-L1429),
  [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L2585-L2789).
- **Person-fact writeback** only runs after sent replies, resolves a known target
  person, supplies direct user evidence, and caps extracted facts at five. The
  queue is capped at 256 and drops on overflow. The prompt explicitly excludes
  facts supplied only by the bot. [`memory_flow_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/services/memory_flow_service.py#L29-L128,L333-L424).
- **Chat-summary writeback** is an asynchronous queue with cap 256, defaults to
  a 36-message threshold, restores its last trigger cursor from durable metadata,
  and uses an `external_id` for idempotency. The A_Memorix importer reviews a
  bounded number of previous summaries and rejects tool/prompt/injection,
  speculation, jokes, and unconfirmed facts in its prompt contract.
  [`memory_flow_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/services/memory_flow_service.py#L427-L541),
  [`summary_importer.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/utils/summary_importer.py#L63-L112,L449-L505,L596-L729).
- **Episode materialization** groups source paragraphs, reuses unchanged input
  fingerprints, calls an LLM segmenter, validates complete paragraph coverage,
  and uses a deterministic rule fallback when the model/parser/coverage fails.
  Source rebuilds use leases/heartbeats, revision CAS publication, a maximum
  retry parameter defaulting to three in the current service path, and capped
  exponential backoff. [`episode_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/utils/episode_service.py#L1-L9,L404-L498),
  [`ingest_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/A_memorix/core/runtime/services/ingest_service.py#L493-L644).

### 4.2 Boundary assessment

MaiBot distinguishes evidence retrieval from wording reasonably well: memory
tools return results to the Planner, profile/heuristic/mid-term blocks are
marked as internal references, and replyer history excludes tool results and
reference messages. The boundary is primarily prompt and scope policy. The
source does not show a typed evidence/provenance contract carried through every
semantic decision; free-form `reference_info`, profile text, and Planner prose
remain influential prompt text.

## 5. Affect, relationship, persona state, and learning

### Verified from source

- Persona identity and behavior style are prompt/config inputs to Planner and
  replyer. `emotion_trait` is an experimental static personality suffix with
  `rational_calm`, `neutral`, or `sentimental`; it is not a per-episode affect
  reducer. [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L33-L55,L1050-L1112).
- Person profiles can store relationship settings and relation edges, but the
  implementation represents them as profile/evidence text and retrieval data.
  It does not expose a typed, elapsed, cause-first relationship/affect state
  machine comparable to Kazusa's Cognition Core V2. This is a source comparison
  assessment, not a claim that profile text has no behavioral effect.
- Behavior/style/jargon learning runs after history trimming or through
  configured learners. It learns reusable scene-behavior-result patterns and
  expression style, but it is configuration-controlled; behavior learning is
  marked experimental and defaults to false in the current config.
  [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L1318-L1414),
  [`learn_behavior.prompt`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/prompts/zh-CN/learn_behavior.prompt#L1-L42),
  [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L1050-L1080).

### Inference/assessment

MaiBot's persona adaptation is emergent and prompt-mediated: profile evidence,
behavior references, expression habits, and the current Planner analysis shape
the next action. Kazusa ECT instead makes affect, relationship, identity, and
goal state explicit persistent semantic domains. MaiBot should therefore be
compared as a flexible Planner-plus-memory system, not as an equivalent typed
affective trajectory implementation.

## 6. Reasoning/monologue visibility and safety

### Verified from source

- `ChatResponse.reasoning` is explicitly documented in code as provider-native
  reasoning for observation and distinct from the Planner's explicit body.
  `reasoning_engine.py` stores the Planner body as `last_reasoning_content`,
  passes it into `ToolInvocation.reasoning`, and includes it in reply-effect
  context. [`chat_loop_service.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/chat_loop_service.py#L80-L100,L1031-L1122),
  [`reasoning_engine.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/reasoning_engine.py#L622-L672,L1452-L1605).
- The explicit Planner text is therefore an application-owned analysis/plan
  surface, not a promise about hidden provider cognition. It is sent to tools
  as operational context and can be recorded in tool/reply-effect diagnostics.
- `show_maisaka_thinking` defaults true in the current config. When enabled,
  CLI/debug display and `planner.finalized` monitoring can expose Planner text,
  prompt previews, tool calls, and tool results. Monitor events other than
  stage-only events are persisted after payload sanitization. [`official_configs.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/config/official_configs.py#L4960-L5039),
  [`runtime_mixin.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/display/runtime_mixin.py#L34-L56,L592-L690),
  [`events.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/monitor/events.py#L300-L336,L540-L584).
- Memory/profile references tell the model to use them for internal reasoning
  and not quote them verbatim. Focus mode prompt text warns against cross-chat
  privacy leakage. These are prompt and routing safeguards, not a typed privacy
  proof. [`maisaka_chat_focus.prompt`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/prompts/zh-CN/maisaka_chat_focus.prompt#L1-L38),
  [`person_profile.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/memory/person_profile.py#L206-L215),
  [`heuristic_injector.py`](https://github.com/Mai-with-u/MaiBot/blob/1b8d13300a5add5f029106b0ee2d80b59996917c/src/maisaka/memory/heuristic_injector.py#L384-L403).

### Inference/assessment

MaiBot exposes a useful, inspectable reasoning narrative, but that narrative is
not automatically safe to treat as user-facing truth, durable persona state, or
authorization. The safest interpretation is the one implemented by its own
boundaries: Planner text is a short-lived action rationale; replyer text is the
only visible wording; memory/profile blocks are evidence-like references; native
provider reasoning remains observation-only.

## 7. Determinism, bounds, and failure handling

### Verified from source

- Context is capped by configured group/private context sizes; selection keeps
  tool-call/result pairs and uses a cache-stability expansion ratio. Planner
  recurrence is bounded by `MAX_INTERNAL_ROUNDS`.
- `wait` clamps seconds to non-negative values and stops after the configured
  consecutive-wait limit, default `3`. `tool_search` and memory/profile tools
  normalize result limits; `query_memory` caps the limit to `20`.
- Tool failures are returned as structured failures and appended to context so
  the Planner can choose a follow-up or alternate action. There is no generic
  automatic tool retry. Missing registry, missing target message, replyer
  failure, empty text, post-process failure, and send failure all remain visible
  failure outcomes.
- Replyer hooks may request regeneration up to `REPLYER_MAX_HOOK_RETRIES = 3`;
  after the cap the last response is used. This is deterministic bounded
  regeneration around a visible-text hook, not semantic stance repair.
- Mid-term memory uses deterministic JSON extraction plus `json_repair`, then
  skips invalid/empty summaries. Person-profile classification falls back to
  rule-based buckets when its model is unavailable or invalid.
- Episode segmentation uses strict coverage validation and deterministic
  `fallback_rule`; summary import returns a failed result on model/parse/import
  failure. The summary importer parser itself uses regex extraction plus
  `json.loads` and does not perform bounded semantic regeneration; this is a
  concrete weaker boundary than the episode and profile paths.
- Provider orchestration has model-attempt retry and task hard-timeout handling
  in `src/llm_models/model_client/base_client.py:878-1139`; the exact provider
  retry policy is configuration-dependent.
- Monitor storage is bounded to `10,000` records and `72` hours in
  `src/maisaka/monitor/event_store.py:17-42,81-128`, with media/data-url
  sanitization before persistence.

### Inference/assessment

MaiBot has many practical runtime bounds and graceful degradation paths. Its
largest semantic weak point is not missing limits; it is that the core Planner
decision remains free-form text plus tool-call validity. A malformed or
semantically unsupported Planner rationale can still influence the next action
unless a tool boundary, hook, or runtime state check catches it. This contrasts
with Kazusa's typed evaluator/regeneration/fail-closed contracts.

## 8. Verified strengths

1. **Clear Planner/replyer split.** The Planner decides whether and how to act;
   the replyer owns natural visible wording. This reduces accidental exposure
   of analysis and gives output rendering its own history and retry boundary.
2. **Action-oriented cognition.** Tools are first-class and deferred discovery
   keeps the common prompt smaller while preserving extensibility through
   plugins, browser, and MCP providers. Focus-mode tools make cross-chat
   attention an explicit action rather than hidden context mutation.
3. **Useful memory layering.** Mid-term compression, person profiles, explicit
   long-term queries, heuristic recall, graph relations, episodes, and
   background writeback serve different time horizons and scopes.
4. **Operational pragmatism.** Queues, idempotent external IDs, profile caches,
   vector-degraded writes, episode fallback, leases, CAS publication, tool
   failure results, wait limits, and bounded replyer retries make the system
   suitable for a long-running bot.
5. **Strong observability for development.** Prompt previews, stage cards, tool
   details, model/token/timing metrics, retry/error events, planner-finalized
   events, and a bounded replay ledger make behavior inspectable when the
   operator enables the relevant surfaces.
6. **Evidence-aware memory prompts.** The source repeatedly excludes bot-only
   suggestions, prompt injections, jokes, speculation, and unconfirmed facts
   from durable person/summary memory. This is a meaningful integrity defense,
   even though it is not a fully typed provenance contract.

## 9. Verified limitations and implementation gaps

1. The public documentation's “lifelike reasoning” framing is broader than the
   implementation evidence. The source shows recurrent Planner analysis and
   learning helpers, not a typed persistent affect/relationship trajectory.
2. Admission is partly deterministic and pre-Planner. Mention/content/pressure
   scores can decide whether the semantic Planner receives a turn; this is
   efficient but can miss grounded reasons that are not represented by the
   lexical scoring rules, or admit a turn whose deeper reason is weak.
3. Planner analysis has no closed schema, evidence-handle requirement, or
   semantic evaluator. Similarity replacement above `0.9` addresses repetition,
   not unsupported stance, privacy, or authorization.
4. Profile and memory references are mostly free-form text. Their prompt labels
   and scope checks help, but they can still compete with current conversation
   evidence without a universal deterministic provenance role.
5. Core memory is optional. `AMemorixPluginConfig.enabled` defaults false;
   integration controls, relation-vector settings, heuristic recall, behavior
   learning, rich reply, and debug surfaces vary independently. Documentation
   claims should therefore be read as capability descriptions, not guaranteed
   deployment behavior.
6. Background writeback is intentionally lossy under pressure: bounded queues
   drop work when full, and worker failures are logged while the live response
   continues. This protects latency but can lose a learning opportunity.
7. Summary import parsing is materially weaker than the newer episode/profile
   paths: malformed JSON produces a failed import rather than a bounded repair or
   stage-owned regeneration, and no semantic validator is visible after parsing.
8. Monitor/debug visibility is powerful but broad. Explicit Planner text,
   prompts, tool arguments/results, and reply-effect context can be persisted or
   displayed when enabled; retention and media sanitization reduce risk but do
   not make the narrative a privacy-safe public API.

## 10. Explicit comparison implications for Kazusa ECT

The following are **inference/assessment**, grounded in the observed MaiBot
source and the current Kazusa subsystem contracts. Kazusa references are local
workspace evidence, not claims about MaiBot.

| Observed MaiBot design | Evidence-backed implication for Kazusa ECT |
| --- | --- |
| Deterministic reply-necessity/frequency gate before the Planner, then free-form Planner judgment | Keep Kazusa's two semantic relevance stages as the grounded admission owner. MaiBot demonstrates the value of cheap scheduling, but a deterministic score should remain a scheduling aid rather than replace the LLM-owned reason-to-speak judgment described in `src/kazusa_ai_chatbot/relevance/README.md:19-51,80-150`. |
| One recurrent Planner emits analysis plus tool calls; no-tool ends the round | Preserve MaiBot's practical iterative action loop as a latency/UX pattern, but keep Kazusa's typed Cognition Core owners and exact route/capability contracts. The Planner equivalent should propose within the ECT boundary, not become a second untyped stance owner. `src/kazusa_ai_chatbot/cognition_core_v2/README.md:45-50,70-111`. |
| Memory/profile references are internal prompt blocks, while replyer receives filtered real chat history | This supports Kazusa's RAG-evidence-versus-cognition boundary. MaiBot shows internal references are useful, but Kazusa should retain explicit source scope/provenance and prevent free-form memory text from becoming affect, permission, or final stance. `src/kazusa_ai_chatbot/cognition_resolver/README.md:78-103`; `src/kazusa_ai_chatbot/rag/README.md` prompt-facing safety sections. |
| Planner rationale and provider native reasoning are separate fields | Keep Kazusa's `selected_bid_reason`/`private_monologue` distinction and protected trace boundary. Treat visible rationale as an inspectable semantic artifact, not hidden CoT and not a direct adapter surface. `src/kazusa_ai_chatbot/cognition_core_v2/README.md:144-179`; `src/kazusa_ai_chatbot/internal_monologue_residue/README.md:14-18,76-129`. |
| Profile relationship text and static emotion suffix shape behavior without a typed affect reducer | MaiBot identifies a real capability gap that Kazusa ECT is designed to cover: persistent affect/relationship state needs cause/evidence/decay/immutability rules, not only prompt wording. `src/kazusa_ai_chatbot/cognition_core_v2/README.md:3-38,252-302`. |
| Asynchronous fact/summary writeback, idempotent memory ingestion, and episode CAS workers | Reuse the operational pattern—background persistence, idempotency, bounded queues, leases, and retry state—while preserving Kazusa's deterministic target planning, lane validation, and state commit order. `src/kazusa_ai_chatbot/consolidation/README.md:37-53,124-148,300-369`. |
| Tool errors return to the Planner; visible reply delivery remains a separate deterministic action | Keep resolver observations as evidence only. Kazusa's resolver contract explicitly forbids a capability result from becoming persona/affect/relationship/final stance and commits final cognition before action/dialog. `src/kazusa_ai_chatbot/cognition_resolver/README.md:78-119`. |
| Rich monitoring exposes prompts, explicit analysis, tool results, and retries | Retain MaiBot's operator usefulness but apply Kazusa's protected trace policy: bounded semantic telemetry, redacted/private fields, no raw prompt or residue leakage, and no automatic trace-to-cognition feedback. `src/kazusa_ai_chatbot/cognition_resolver/README.md:119-129`; `src/kazusa_ai_chatbot/internal_monologue_residue/README.md:139-159`. |
| Free-form Planner plus prompt safeguards handles privacy and evidence by convention | Prefer Kazusa's fail-closed typed validation for roles, permissions, evidence handles, relational willingness, and surface authority. MaiBot is evidence that prompt instructions alone are an insufficient ownership boundary for a higher-assurance ECT path. `src/kazusa_ai_chatbot/cognition_core_v2/README.md:205-235,275-302`. |

## 11. Concise evidence matrix for parent synthesis

| Dimension | What MaiBot actually implements | What is documentation-only or optional | Confidence |
| --- | --- | --- | --- |
| Admission | Deterministic frequency/reply-necessity scheduler; Planner only after a trigger | Broader “decide whether to speak” narrative belongs to docs plus Planner behavior | High |
| Cognition | Free-form Planner analysis, recurrent tool/action rounds, explicit `wait`/`reply` | No typed universal stance/evidence schema | High |
| Internal thought | Planner body retained as action rationale; provider-native reasoning separate | Human-like “thinking” is product framing; display is config-controlled | High |
| Memory | Mid-term summary/embedding recall; A_Memorix vector/graph/BM25/PPR; profiles; background writeback | A_Memorix plugin and heuristic recall are optional; relation vectors separately configurable | High |
| Affect/relationship | Static emotion trait, profile relationship text/edges, learned behavior/style | No verified persistent affect/relationship transition engine | Medium-high |
| Tools | Builtins, deferred discovery, plugins/browser/MCP providers, focus tools | Plugin/browser/MCP/rich reply/structured expression features depend on config/runtime | High |
| Output | Replyer-only visible wording, post-processing, deterministic send and history sync | Hook rewrites are extension behavior, not core semantic ownership | High |
| Persistence | Async fact/summary queues, idempotent memory writes, snapshots, episode CAS workers | Learning/summary paths can be disabled or drop queued work under pressure | High |
| Bounds | Context caps, internal-round bound, wait cap, top-k/char caps, replyer retry cap, provider retry/timeout | Exact provider policy and `MAX_INTERNAL_ROUNDS` value were not pinned here | High for existence; medium for numeric defaults |
| Failure behavior | Structured tool failures, skip/fallback paths, episode fallback, retry/backoff/leases | Planner semantic failures lack a universal typed evaluator/fail-closed contract | High |
| Observability | Prompt previews, planner/tool monitor events, retry/error events, bounded replay ledger | Visibility can include explicit analysis/prompt content when enabled | High |

## 12. Unresolved questions

1. What is the exact numeric value of `MAX_INTERNAL_ROUNDS` in the current
   `main` snapshot, and how does it interact with provider hard timeouts under a
   slow tool chain?
2. In a real configured deployment, which A_Memorix integration flags are
   enabled together, and does the host service enforce the documented scope
   policy identically for every tool and background writer?
3. Which plugin/browser/MCP providers are present in the canonical distribution
   versus deployment-specific extensions, and what independent authorization
   checks exist at each provider boundary?
4. How often do the free-form Planner rationale, profile text, and memory
   references conflict in practice, and is there an operator-facing semantic
   evaluator beyond the inspected hooks and runtime validation?
5. Does the current release have additional affect/relationship state in
   deployment modules not reached by this source slice, or is profile text and
   static `emotion_trait` the complete public mechanism?

Overall assessment: MaiBot is a capable, production-oriented Planner/action
architecture with unusually rich memory and observability plumbing. Its most
important architectural lesson for Kazusa is the value of explicit action
recurrence, separate reply rendering, asynchronous memory maintenance, and
operational bounds. Its main contrast with Kazusa ECT is semantic ownership:
MaiBot leaves more judgment in a free-form Planner narrative and prompt-scoped
references, while Kazusa makes affect, relationship, evidence provenance,
authorization, and failure-closed state transitions explicit typed contracts.
