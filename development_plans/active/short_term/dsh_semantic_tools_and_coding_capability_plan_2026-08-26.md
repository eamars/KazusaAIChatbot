# DSH Plan 2: Standard Semantic Tools, Native Coding, And Brain Interaction

## Plan Status And Ownership

- **Status:** `approved` — architecture and execution contract approved by the
  user on 2026-08-28; production execution still follows the repository's
  explicit implementation authorization boundary.
- **Plan class:** multi-process capability expansion, model-route migration,
  and narrow Brain interaction integration.
- **Predecessor:**
  `development_plans/archive/completed/short_term/dsh_standalone_sidecar_and_resolution_interface_plan_2026-08-26.md`
  (Plan 1, completed with all gates green).
- **Successor boundary:**
  `development_plans/active/short_term/dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md`
  (Plan 3 retains the production `task_resolution_request` cutover, preserves
  the current post-selector Brain handover, and deletes only the legacy task
  executors replaced by DSH).
- **Architecture authority:** the user approves architecture and promotion;
  the active primary engineering agent owns implementation, verification,
  evidence, and closure. This plan binds no implementation worker or model.
- **Implementation order:** tests first, production second, documentation
  last. A phase cannot advance until its gate is recorded.

## Objective

Turn the completed Plan 1 standalone resolver into a full DeepSeek Harness
Standard agent that:

1. uses the official pinned DSH `standard` preset for its default coding and
   general-purpose tools;
2. adds only storage-independent Kazusa Brain and memory capabilities that
   the pinned DSH Standard catalog does not provide;
3. uses the native public-web capabilities published by DSH's default
   provider, isolated from Kazusa's current webagent implementation;
4. inherits DSH Standard's native workspace, permission, sandbox, tool-loop,
   and provider behavior without a Kazusa policy overlay;
5. sends every approval, question, and plan-review interaction to Kazusa
   Brain, which makes the semantic judgment itself or relays the question
   through Kazusa dialog and adapters;
6. routes DSH through a new project-owned `AGENTIC_RESOLVER_LLM_*` route,
   initially `qwen27b-5090`; and
7. preserves Plan 1's durable intake, fencing, checkpoint, evidence, and
   terminal-exhaust guarantees under a deliberate V2 contract epoch.

Plan 2 creates the Brain interaction bridge needed by DSH tools. It does not
route production `task_resolution_request` into DSH, map exhaust into
`TaskResolutionResultV1`, or delete any legacy production path. Those actions
remain atomic Plan 3 scope.

## Exit State

At Plan 2 exit, a canonical standalone V2 intake can use the complete pinned
DSH Standard toolset, storage-independent Kazusa Brain and memory tools, the
default DSH public-web tools, and native coding tools. DSH retains its Standard
autonomy. A DSH question, approval request, or plan review reaches the Brain
interaction broker. Brain can answer or decide immediately, or render and
deliver a user-facing question before safely resuming the same DSH thread.

The model-facing Kazusa catalog remains understandable when every tool
description is removed: each name states a verb and semantic noun, each
argument states a domain concept, cardinality is explicit, and every result is
a typed semantic entity or opaque continuation reference. MongoDB collections,
documents, field paths, query operators, aggregation stages, indexes, ObjectIds,
and storage schemas remain private Kazusa implementation details.

The initial DSH model is `qwen27b-5090` through the project's local
OpenAI-compatible endpoint. The model id, endpoint, credential reference,
capacity, completion budget, and thinking mode are configuration, not source
constants.

## Mandatory Skills And Rules

- Apply `development-plan` for execution state, change-radius amendments,
  source-to-test traceability, review, and closure.
- Apply `local-llm-architecture` before changing model-facing tool names,
  schemas, prompts, or routing. Keep model inputs semantic and storage-free.
- Apply `no-prepost-user-input` to every DSH question, approval, and relayed
  user reply. Cognition owns interpretation; deterministic code preserves the
  typed decision.
- Apply `py-style` before any Python edit, `cjk-safety` whenever a Python edit
  contains CJK text, and `test-style-and-execution` before creating, changing,
  or running tests. Use the project virtual environment for Python commands.
- Preserve pre-existing worktree changes, capture the owned-path baseline, and
  obtain a plan amendment before expanding the catalog, architecture, or
  production change surface.

## Fixed Architectural Decisions

### 1. Official Standard Is The Coding Capability

The sidecar boots the official host plane from the installed
`@deepseek-ai/dsh-base/cordis.patch.yml` through
`@deepseek-ai/dsh-app-boot`, applying it over the local empty
`sidecars/dsh_resolution/config/root.cordis.yml`. It then loads the official
`standard` agent preset shipped inside the same pinned `@deepseek-ai/dsh`
installation. It resolves that preset by package path and mounts it through
`@deepseek-ai/dsh-agent-presets`. Neither official file is copied, forked, or
restated.

The pin remains:

- DSH packages: `0.1.1-rc.2`;
- upstream source commit:
  `b150a551b8d465e31e418e1b2eaf5e79bbb7d28e`;
- `@deepseek-ai/cordis`: `4.0.1`;
- `@deepseek-ai/schemastery`: `3.18.1`.

`composition.ts` owns one narrow in-memory overlay over the installed base
bundle. Its row-id changes are exact:

- disable `hmr`, `settings`, `credentials`, `llm-deepseek`,
  `session-persistence-jsonl`, and `session-telemetry-otel`;
- configure `llm-pi-ai` with only the canonical project route and configure
  `agent-default-model` to that route/model;
- retain the pinned Standard workspace, permission, sandbox, and web-provider
  configuration unchanged; and
- insert only the installed Standard-preset loader, the existing SQLite
  persistence plugin, the Plan 1 terminal tool, the Kazusa semantic tools, the
  host-only credentials provider, and the Brain interaction provider.

The overlay changes configuration and adds Kazusa seams; it does not duplicate
an upstream plugin row. `standard_profile.spec.ts` resolves both installed
official files, records both SHA-256 digests, dumps the composed tree, and
asserts every overlay target exactly once before startup can become ready.

The direct dependency delta in `sidecars/dsh_resolution/package.json` is:

- add `@deepseek-ai/dsh`, `@deepseek-ai/dsh-agent-presets`,
  `@deepseek-ai/dsh-app-boot`, `@deepseek-ai/dsh-base`,
  `@deepseek-ai/dsh-credentials`, `@deepseek-ai/dsh-launch-environment`,
  `@deepseek-ai/dsh-llm-pi-ai`, `@deepseek-ai/dsh-tool-web`, and
  `@deepseek-ai/dsh-web`, all at `0.1.1-rc.2`;
- retain the Plan 1 direct core/session/SQLite dependencies still imported by
  sidecar source; and
- remove the direct `@deepseek-ai/dsh-llm-deepseek` dependency. The installed
  base bundle may retain it transitively, but the `llm-deepseek` row is disabled
  and no DeepSeek route may register.

All enabled tools in that exact Standard preset remain enabled. Platform
selection remains upstream-owned: native Windows exposes `pwsh`, while POSIX
hosts expose `bash`. Standard's deliberately disabled optional Codex and
Claude provider rows remain disabled. Kazusa adds no coding wrapper, patch
language, repository index, product-manager agent, programmer agent, command
parser, or coding workflow.

The official Standard preset and base bundle own their web tool rows and
provider selection. Kazusa neither inserts a web provider nor rewrites those
rows. The accepted catalog records whichever native `web_search` and
`web_fetch` capabilities the pinned Standard composition actually publishes.

### 2. DSH-Native Capability Precedence

DSH owns a capability whenever the pinned Standard profile already supplies
it. Kazusa does not inject a second model-facing tool for:

- shell, filesystem read/search/write/edit, image-file reading, jobs, or code;
- skills, goals, plan mode, todo, subagents, workflows, or Ralph;
- user questions or plan review;
- web search or URL fetch;
- text transformation or deterministic calculation.

Text transformation and calculation use Standard `pwsh`/`bash`. Workspace
images use Standard `read_image` when the selected DSH route declares image
input; the initial Qwen route declares text only, so upstream returns its
typed unsupported-image result. Kazusa's session-media tool is limited to
opaque media already attached to a Kazusa conversation and cannot accept a
filesystem path or URL.

Startup obtains the actual Standard registry, then compares that complete name
set with the fixed Kazusa semantic catalog. The DSH
registry is authoritative: a colliding Kazusa registration is omitted, the
native DSH tool remains available under the original name, and health reports
the omitted semantic registration as `native_precedence`. The runtime never
renames or aliases either tool. A DSH pin upgrade still requires a catalog
review and plan amendment before release.

#### Semantic Tool Minimality Gate

```text
Semantic question:
  Which Kazusa Brain or memory capability, absent from DSH Standard, does the
  current DSH task need to invoke?
Inputs required:
  Semantic task arguments and opaque references previously issued by Brain or
  a Kazusa semantic result.
Outputs required:
  Typed semantic entities, evidence, pagination references, and idempotent
  mutation outcomes consumed by DSH or submit_resolution.
Deterministic owners:
  Interface authentication, structural validation, reference lineage,
  storage mapping, persistence, idempotency, and result serialization.
Rejected complexity:
  Database schemas, backend query modes, Kazusa web tools, duplicate Standard
  tools, tool-choice routers, semantic post-filters, DSH prompt guardrails,
  runtime step/call budgets, and alternate fallback executors.
Evidence before expansion:
  A confirmed missing DSH-native capability plus a user-approved catalog
  amendment and description-stripped selection test.
```

### 3. Canonical DSH Model Route

Plan 2 introduces this project route:

| Environment field | Initial local value or rule |
|---|---|
| `AGENTIC_RESOLVER_LLM_BASE_URL` | `http://localhost:8080/v1` |
| `AGENTIC_RESOLVER_LLM_API_KEY` | project secret; value is never written to this plan, logs, prompts, or shell environment |
| `AGENTIC_RESOLVER_LLM_MODEL` | `qwen27b-5090` |
| `AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS` | `50176` |
| `AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS` | `8192` |
| `AGENTIC_RESOLVER_LLM_THINKING_ENABLED` | `true` |

`kazusa_ai_chatbot.config.AgenticResolverRouteSettingsV1` is the Python
configuration owner. The sidecar reads the same field names at host startup,
builds one route named `kazusa-agentic-resolver`, and publishes its route
digest in health and intake compatibility metadata.

`main.ts` resolves `sidecars/dsh_resolution/package.json` from its compiled
module location, takes that package directory's `../..` as the repository
root, and passes that absolute path to DSH's layered environment loader. It
takes one immutable project environment snapshot before extracting and
removing secrets; invocation cwd never selects the `.env`. TypeScript validates
the same required fields and bounds as
`AgenticResolverRouteSettingsV1`. The controller sends the non-secret route
descriptor and digest in V2 intake; the sidecar recomputes them and rejects a
mismatch before session creation. The digest includes field names, endpoint,
model, capacities, thinking/compatibility values, and credential
reference name, never the credential value.

The sidecar replaces `@deepseek-ai/dsh-llm-deepseek` with
`@deepseek-ai/dsh-llm-pi-ai` configured as:

- protocol `openai-completions`;
- model id, context window, and max tokens from the canonical fields;
- `compat.supportsDeveloperRole: false`;
- `compat.maxTokensField: max_completion_tokens`;
- `compat.thinkingFormat: qwen-chat-template`;
- `compat.chatTemplateKwargs.enable_thinking` bound to DSH thinking state;
- reasoning level `high` when thinking is enabled and `off` when disabled;
- text-only input for the initial route.

A deterministic fake OpenAI endpoint test pins the outgoing system role,
tool schema, max-token field, Qwen thinking kwargs, and multi-turn tool replay
before a real Qwen gate runs.

The remaining host fields are also exact and share the project's normal
environment/config loading path:

| Environment field | Plan 2 rule |
|---|---|
| `KAZUSA_DSH_SIDECAR_URL` | retain Plan 1's loopback HTTP `/rpc` URL |
| `KAZUSA_DSH_RPC_TOKEN` | retain Plan 1's opaque RPC secret |
| `KAZUSA_DSH_DATA_ROOT` | retain the absolute runtime-data root; V2 selects its new epoch below |
| `AGENTIC_RESOLVER_WORKSPACE_ROOT` | required absolute canonical path; initial value is the repository root |
| `KAZUSA_DSH_BRAIN_URL` | `http://127.0.0.1:8000`; loopback HTTP only |
| `KAZUSA_DSH_BRAIN_SHARED_SECRET` | new opaque secret shared only by Brain and sidecar host code |
| `KAZUSA_DSH_TOOL_GATEWAY_SECRET` | new opaque secret passed only to authority owners and the isolated worker |
| `KAZUSA_DSH_PYTHON_EXECUTABLE` | absolute path to `venv\Scripts\python.exe` |

The Brain URL and sidecar URL validators reject non-loopback hosts. Startup
validates required paths and non-secret bounds without printing secret values.
Web-provider configuration remains entirely owned by the pinned DSH Standard
composition; Kazusa webagent, SearXNG, and URL-reader configuration never enter
the sidecar route.

This route supersedes RAG/coding model routes for all new DSH resolver work in
Plan 2. Existing `RAG_*`, `BACKGROUND_WORK_*`, and `CODING_AGENT_*` fields stay
operational only for their unchanged current production callers. Plan 3
deletes those callers and their obsolete route fields in the same big-bang
cutover; Plan 2 neither redirects nor breaks them early.

### 4. DSH-Native Autonomy And Workspace

The task workspace is the absolute canonical value supplied through
`AGENTIC_RESOLVER_WORKSPACE_ROOT`; its initial project value is the repository
root. The controller places that workspace identity in model-hidden runtime
context and the immutable DSH session header. DSH Standard owns access behavior
for that workspace and any path it handles through native policy.

Kazusa supplies the admitted absolute workspace as session context and leaves
the pinned Standard permission preset, sandbox mode, approval behavior,
subagent behavior, tool scheduling, and execution policy unchanged. Plan 2
adds no Kazusa model-step cap, tool-call cap, retained-byte cap, soft deadline,
hard deadline, shell wrapper, command filter, plan instruction, or retry policy
to DSH.

When native DSH policy itself produces a question or approval request, the
existing Brain interaction provider handles that event. Kazusa does not create
an additional pre-execution permission layer. Subagents and workflows inherit
upstream DSH behavior and the root Brain interaction provider.

The sidecar records DSH-reported workspace and enforcement facts in health
diagnostics without converting them into Kazusa policy. Standard remains the
sole execution-policy implementation.

### 5. Secrets Stay On The Host Plane

`main.ts` reads the RPC token, model credential, Brain bridge secret, and
semantic-worker secret into host-only closures before creating an Agent, then
removes those names from `process.env`. The LLM adapter resolves its credential
through a host-only in-memory DSH credentials provider. The semantic worker is
started with an explicit minimal environment instead of inheriting the
sidecar environment.

The DSH shell receives only the upstream managed `DSH_*` environment and the
explicit non-secret allowlist. Tests execute native `pwsh`/`bash` and prove it
cannot observe:

- `AGENTIC_RESOLVER_LLM_API_KEY`;
- `KAZUSA_DSH_RPC_TOKEN`;
- `KAZUSA_DSH_BRAIN_SHARED_SECRET`;
- `KAZUSA_DSH_TOOL_GATEWAY_SECRET`; or
- any serialized capability token.

Repository instructions remain available through Standard's native
`agent-instructions` loader. The current repository rule prohibiting `.env`
inspection remains part of the agent instruction contract.

### 6. Brain Judges Every DSH Question

The sidecar mounts no direct-human answerer. One Brain-backed provider receives
DSH `approval/request` and `userQuestions.ask`. `ask_user_question` and
`exit_plan_mode` retain their native DSH tools and schemas while Kazusa Brain
judges every resulting question. Brain may answer or accept it itself, reject
it when that is the Brain's semantic decision, or relay a paraphrased or direct
question to the user through normal dialog and adapter delivery.

The request path is:

```text
DSH approval / question / plan review
    -> sidecar BrainInteractionClient
       -> signed loopback POST /runtime/dsh/interactions
          -> deterministic auth, scope, lease, replay, and size checks
          -> Kazusa cognition P-stage semantic decision
             answer | allow_once | reject | relay_to_user
          -> deterministic enactment
    <- signed immediate decision OR checkpoint_required
```

The Brain LLM owns semantic interpretation. Deterministic code validates the
closed contract, exact identity, request digest, scope, lease, expiry, and
one-shot consumption. It does not classify user text with keywords, regexes,
or post-LLM acceptance rules.

For `relay_to_user`:

1. Brain persists `DshInteractionPendingV1` before delivery.
2. Cognition supplies the response goal; Kazusa dialog owns visible wording;
   the dispatcher and adapter own delivery.
3. Brain records the delivered platform message id and returns
   `checkpoint_required` only after the pending row and delivery receipt are
   durable.
4. The controller uses the agreed `resolution.request_checkpoint` operation at
   the next safe boundary and returns a checkpointed exhaust. The interaction
   remains unresolved rather than being represented to DSH as a rejection.
5. A later user reply enters normal `/chat`. Exact platform/channel/user and
   reply-message lineage project the pending request into cognition.
6. Cognition emits `answer`, `allow_once`, `reject`, or `continue_waiting`.
   The pending owner persists that semantic result and schedules one
   controller continuation after the normal Brain state commit.
7. A question answer enters the continuation delta. An approval creates one
   short-lived grant bound to thread, segment, tool, arguments digest,
   workspace, scope, and policy. The next matching DSH retry consumes it once
   and maps the new DSH `callId` to `allowed-once`.

DSH approvals are active-turn and one-shot, so an old `callId` is never kept
open across a chat turn. A resumed call receives a new DSH `callId`; the
Brain-authored grant authorizes only the one semantically identical retry.

Subagents retain upstream behavior: a child that cannot ask a user returns
the blocker to its parent, and only the root interaction provider reaches
Brain.

### 7. Public Web Uses The Default DSH Provider

The model sees only the native web tools published by the pinned DSH Standard
composition. Their names, schemas, provider selection, scheduling, retrieval,
rendering, limits, errors, and evidence presentation remain DSH-owned. Kazusa
adds no web tool, provider adapter, observer, network policy, prompt guidance,
fallback, or translation layer. The current Kazusa webagent, SearXNG wrapper,
and URL-reader implementation remain outside the DSH process and import graph.

The composed-profile test records the native web names actually published by
the pinned upstream package. A future DSH pin change that changes those names
or removes a required native web capability returns to plan review.

### 8. Kazusa Semantic Gateway Is Storage-Independent

The sidecar owns one persistent Python worker over length-prefixed JSON on
stdio. The worker runs from the project virtual environment as
`python -m kazusa_ai_chatbot.dsh_tool_gateway.worker`. V2 intake carries one
signed activation-authority envelope. Host code validates that envelope and
derives a new, model-invisible HMAC-SHA256 claim for each semantic call. Each
claim authenticates:

- thread, segment, activation, and lease epoch;
- Brain conversation reference;
- scope and audience fingerprints;
- profile version, catalog digest, model-route digest, and policy epoch;
- the exact semantic operation, arguments digest, and issued-reference digest;
  and
- semantic call id, issued-at, expiry, token id, nonce, and idempotency key.

The activation grants the complete published Kazusa semantic catalog for the
Brain-authorized conversation scope. It does not create a second semantic
permission decision per tool call. Deterministic code authenticates the
interface, validates structure and reference lineage, maps semantic operations
to Kazusa-owned services, and returns the committed semantic outcome. It does
not interpret, filter, reclassify, or rewrite the DSH decision.

The model-facing catalog is exact:

| Tool | Semantic arguments | Result and Kazusa service owner |
|---|---|---|
| `kazusa_search_conversation_history` | `query`, optional `time_range` (`start_at`, `end_at`), optional `max_results`, optional opaque `next_page_ref` | conversation-entry summaries and opaque `conversation_entry_ref` values; conversation history service |
| `kazusa_read_conversation_entries` | one or more `conversation_entry_ref` values | complete prompt-safe conversation entries; conversation history service |
| `kazusa_summarize_conversation_participants` | optional semantic `time_range`, optional `max_people`, optional `next_page_ref` | participant summaries and opaque `person_ref` values; conversation history service |
| `kazusa_search_memories` | `query`, optional `subject_scope` (`current_user`, `active_character`, `shared_world`, or `all`), optional `memory_kinds` (`profile_fact`, `relationship`, `commitment`, `experience`, `world_knowledge`), optional `max_results`, optional `next_page_ref` | memory summaries and opaque `memory_ref` values; memory service chooses retrieval mechanics |
| `kazusa_read_memories` | one or more `memory_ref` values | complete prompt-safe semantic memory records; memory service |
| `kazusa_remember_information` | `subject` (`current_user`, `active_character`, or `shared_world`), `information`, `memory_kind`, `reason`, and source `conversation_entry_ref` or `current_task` provenance | committed memory outcome and `memory_ref`; existing Kazusa memory consolidation/persistence service |
| `kazusa_revise_memory` | `memory_ref`, revised semantic information, and reason | committed revision outcome for the same semantic memory identity; existing Kazusa memory evolution service |
| `kazusa_change_memory_lifecycle` | `memory_ref`, semantic transition (`activate`, `complete`, `cancel`, or `archive`), and reason | committed lifecycle outcome; existing reviewed memory-lifecycle service |
| `kazusa_find_people_by_name` | `display_name`, `match_relation` (`exact`, `contains`, `starts_with`, `ends_with`), optional `max_results`, optional `next_page_ref` | candidate summaries and opaque `person_ref` values; people service |
| `kazusa_read_person_profiles` | one or more `person_ref` values | prompt-safe profile and relationship context; people service |
| `kazusa_recall_active_context` | requested semantic kinds (`commitments`, `progress`, `history`, `calendar`) and optional `max_results` | current active-context entries with semantic provenance; deterministic recall service, outside the RAG review graph |
| `kazusa_read_calendar_context` | semantic view (`schedules`, `recent_runs`, `pending_runs`), optional `max_results`, optional `next_page_ref` | prompt-safe calendar entries; calendar service |
| `kazusa_inspect_attached_media` | opaque `attached_media_ref` and a semantic question | one prompt-safe inspection of media already attached to the Brain conversation; media-inspection service |

This catalog and the Brain interaction provider jointly give DSH access to the
Brain and its memory capabilities. Brain interaction uses native DSH question
and approval surfaces, so the catalog adds no duplicate `ask_brain`, user
question, approval, dialog, or delivery tool.

Tool schemas use domain vocabulary only. Search callers state what they need;
the Kazusa service selects keyword, vector, hybrid, collection, index, query,
deduplication, and projection mechanics. Opaque references exist only for
semantic continuation. They carry no collection name, document key, ObjectId,
field path, or backend cursor. `next_page_ref` supports complete authorized
traversal while keeping each response prompt-sized.

All results use `KazusaSemanticCapabilityResultV1`:

```text
schema_version
status: ok | empty | denied | invalid | timeout | unavailable
entities: typed semantic entities
page: {has_more, next_page_ref}
evidence: EvidenceReceiptV2[]
mutation: null | {outcome, semantic_ref, idempotency_key}
error: null | {code, safe_message}
```

Each page is prompt-sized and every mutation is idempotent. A worker restart
replays the committed result for the same idempotency key; it does not repeat
an uncertain mutation. Raw Mongo documents, embeddings, database names,
collection names, storage field names, media bytes, access-control data,
credentials, internal prompts, and hidden LLM traces never enter tool names,
arguments, results, descriptions, or DSH evidence.

Names follow `kazusa_<verb>_<semantic noun>` and express cardinality through
the noun or argument schema. Acceptance includes a description-stripped
catalog review: DSH receives names and JSON schemas with descriptions removed
and must still select the correct capability and construct valid arguments.
Static tests reject ambiguous generic verbs (`query`, `execute`, `operate`,
`access_data`) and storage vocabulary in every model-facing name, argument,
enum, result, example, and generated prompt.

Gateway handlers may call only the named Kazusa semantic services. They do not
call a RAG graph, task resolver, planner, specialist selector, coding harness,
webagent, or complete-answer synthesizer. DSH alone chooses and sequences the
tools; Kazusa services own storage mapping and committed persistence.

### 9. Multi-Tool Loop And Terminal Authority

Plan 1's `validSingleAction` rule is replaced atomically. A normal model step
may use any admitted Standard, native-web, or Kazusa semantic tool. DSH owns
tool execution and parallel scheduling across root and child sessions.

`submit_resolution` remains the only terminal authority. A terminal assistant
step must contain exactly one `submit_resolution` call and no sibling tool
call. Plain assistant text is non-terminal. A mixed or structurally invalid
terminal call receives the terminal tool's ordinary typed validation result;
DSH decides whether and how to continue its native loop. Kazusa adds no
regeneration instruction, attempt cap, model-step cap, tool-call cap, retained
tool-byte cap, soft deadline, or hard deadline to the DSH session.

Every evidence id cited by `submit_resolution` must exist in the current
authorized evidence ledger. Coding artifacts resolve through DSH's admitted
workspace contract and carry a content digest. Evidence from a previous
incompatible segment is rejected as an interface-integrity error.

Plan 3 retains the current Brain-owned inline execution budget (currently 30
seconds) solely to decide whether the caller waits for a result or promotes
the same task to the existing background worker. Expiry of that caller wait
invokes the agreed `resolution.request_checkpoint` operation. It does not
cancel, constrain, or retune the DSH reasoning loop.

## Contract And Persistence Epoch

Plan 2 is a big-bang standalone contract update with no V1 shim:

| Contract | Plan 2 value |
|---|---|
| RPC protocol | `kazusa.dsh-resolution-rpc.v2` |
| Intake | `dsh_resolution_intake.v2` |
| Resolution thread | `resolution_thread_store.v2` |
| Segment | `resolver_session_segment.v2` |
| Profile | `kazusa-resolver-standard-v2` |
| Semantic result | `kazusa_semantic_capability_result.v1` |
| Evidence receipt | `evidence_receipt.v2` |
| Brain interaction | `dsh_brain_interaction.v1` |
| Brain pending row | `dsh_interaction_pending.v1` |
| SQLite epoch | `dsh-sqlite-0.1.1-rc.2-standard-v2` |

V2 intake adds model-hidden `brain_conversation_ref`, canonical workspace,
interaction authority, route digest, and signed semantic-tool capability.
Controller thread creation stores the real Brain conversation reference
instead of substituting the resolution thread id.

The V2 SQLite epoch uses a new store path and never opens the Plan 1 V1 file.
Kazusa semantic-service calls require the V2 gateway contract. Existing V1
standalone rows remain historical and are not converted, resumed, or treated
as fallback state.

`DshBrainInteractionRequestV1` binds kind (`approval`, `question`, or
`plan_review`), thread/session/fence identity, DSH `callId`, tool name,
arguments digest, bounded transient details, conversation scope, nonce, and
expiry. `DshBrainInteractionDecisionV1` binds the same request digest and one
closed decision. Requests and responses use a dedicated
`KAZUSA_DSH_BRAIN_SHARED_SECRET`, timestamp window, nonce replay store, and
constant-time MAC verification over canonical JSON.

`src/kazusa_ai_chatbot/db/dsh_interactions.py` is the sole raw Mongo owner for
collection `dsh_interaction_store`. One `dsh_interaction_store.v1` document per
interaction contains the immutable signed request identity, decision state,
optional delivery lineage, optional reply result, and optional one-shot grant.
This collection name and every document/index field remain implementation-only
and never enter a DSH tool contract, prompt, description, argument, or result.
Indexes are exact:

- unique `interaction_id`;
- unique `(issuer, nonce)` for durable replay exclusion;
- `(status, platform, platform_channel_id, global_user_id,
  delivered_platform_message_id, expires_at)` for exact reply matching; and
- `(grant_status, resolution_thread_id, segment_id, tool_name,
  arguments_digest, workspace_fingerprint, scope_fingerprint, policy_epoch,
  expires_at)` for exact grant lookup.

Grant consumption is one atomic `find_one_and_update` from `available` to
`consumed`, matching every bound field and an unexpired timestamp. No generic
pending-interaction repository, adapter parser, or process-local grant cache is
authoritative. Interaction rows remain as audit records after expiry.

Interaction limits are fixed: canonical request/response bodies are at most
32 KiB, transient detail text is at most 8,000 characters, a Brain-authored
question answer is at most 2,000 characters, signed timestamp skew is 60
seconds, an active sidecar request expires after five minutes, a relayed
pending request expires after 24 hours, and a resumed one-shot grant expires
after ten minutes.

## Failure And Recovery Semantics

| Failure | Required disposition |
|---|---|
| Base bundle, Standard preset, or package drift | sidecar startup fails before RPC health becomes ready |
| Model route mismatch or missing credential | startup/first call fails with a typed route fault; no fallback model |
| Semantic worker malformed frame or crash | restart the worker; replay a committed idempotent result or return `unavailable` while preserving uncertain mutation state |
| Expired/replayed capability or opaque page/reference token | typed interface rejection before the semantic service call |
| Native web provider failure | preserve the native DSH web result without Kazusa fallback or reinterpretation |
| Brain auth/replay failure | interaction rejected and activation checkpointed/faulted; no local decision |
| Brain temporarily unavailable | safe checkpoint with `BRAIN_INTERACTION_UNAVAILABLE`; controller may retry under the same fence rules |
| User relay delivery failure | pending row records failure; sidecar receives no `checkpoint_required` acknowledgement until durable state is coherent |
| User reply does not match pending lineage | ordinary chat only; no grant or DSH continuation |
| Grant mismatch, expiry, or second use | DSH approval remains ungranted and a fresh Brain decision is required |
| Sidecar restart after committed tool result | rebuild evidence and interaction state from durable events and return the exact committed outcome |
| Invalid terminal submission | return the terminal tool's typed structural result and leave continuation to DSH |
| Existing Brain inline wait expires | request a cooperative checkpoint through `resolution.request_checkpoint`, promote the same DSH thread through Plan 3's current background-worker path, and preserve committed events |

## Exact Change Inventory

### Create — Production

- `src/kazusa_ai_chatbot/db/dsh_interactions.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/__init__.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/contracts.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/authority.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/catalog.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/conversation.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/memory.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/people.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/recall_calendar.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/media.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/dispatch.py`
- `src/kazusa_ai_chatbot/dsh_tool_gateway/worker.py`
- `src/kazusa_ai_chatbot/dsh_interaction/__init__.py`
- `src/kazusa_ai_chatbot/dsh_interaction/contracts.py`
- `src/kazusa_ai_chatbot/dsh_interaction/auth.py`
- `src/kazusa_ai_chatbot/dsh_interaction/decision.py`
- `src/kazusa_ai_chatbot/dsh_interaction/pending.py`
- `src/kazusa_ai_chatbot/dsh_interaction/resume.py`
- `src/kazusa_ai_chatbot/dsh_interaction/service.py`
- `sidecars/dsh_resolution/config/root.cordis.yml`
- `sidecars/dsh_resolution/src/composition.ts`
- `sidecars/dsh_resolution/src/model_route.ts`
- `sidecars/dsh_resolution/src/secret_broker.ts`
- `sidecars/dsh_resolution/src/semantic_gateway.ts`
- `sidecars/dsh_resolution/src/brain_interaction.ts`

### Modify — Production And Configuration

- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/db/resolution_threads.py`
- `src/agentic_resolver/__init__.py`
- `src/agentic_resolver/contracts.py`
- `src/agentic_resolver/controller.py`
- `src/agentic_resolver/errors.py`
- `src/agentic_resolver/fingerprints.py`
- `src/agentic_resolver/persistence.py`
- `src/agentic_resolver/rpc.py`
- `src/agentic_resolver/runtime.py`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
- `sidecars/dsh_resolution/package.json`
- `sidecars/dsh_resolution/pnpm-lock.yaml`
- `sidecars/dsh_resolution/src/contracts.ts`
- `sidecars/dsh_resolution/src/evidence.ts`
- `sidecars/dsh_resolution/src/main.ts`
- `sidecars/dsh_resolution/src/operations.ts`
- `sidecars/dsh_resolution/src/profile.ts`
- `sidecars/dsh_resolution/src/rpc.ts`
- `sidecars/dsh_resolution/src/runtime.ts`
- `sidecars/dsh_resolution/src/submit_resolution.ts`
- `tests/ownership/source_test_impact_manifest.json`
- operator-local `.env` using the exact `AGENTIC_RESOLVER_LLM_*`, workspace,
  sidecar, worker, and Brain bridge fields in this plan; secret values remain
  uncommitted and never printed.

### Delete Or Remove

- Remove the direct `@deepseek-ai/dsh-llm-deepseek` dependency and Plan 1
  profile composition; disable the transitive official-base row.
- Remove Plan 1's `KAZUSA_DSH_MODEL` configuration surface.
- Remove Plan 1's `validSingleAction` every-step restriction and minimal
  submit-only persona composition.
- Delete no Python production module, legacy route, legacy resolver, RAG
  owner, or coding-agent module in Plan 2.

### Reuse Without Modification

- `src/kazusa_ai_chatbot/db/conversation.py`
- `src/kazusa_ai_chatbot/db/memory.py`
- `src/kazusa_ai_chatbot/db/memory_evolution.py`
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
- `src/kazusa_ai_chatbot/db/users.py`
- `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`
- `src/kazusa_ai_chatbot/rag/recall/collectors/`
- `src/kazusa_ai_chatbot/media_inspection/`

If implementation discovers that one of these leaves cannot satisfy the
frozen schema without a production edit, execution returns to Phase 1, adds
that exact source path and its tests to this plan through an approved
amendment, then proceeds.

### Retain For Plan 3

- `src/kazusa_ai_chatbot/task_resolution/`
- `src/kazusa_ai_chatbot/rag/`
- `src/kazusa_ai_chatbot/local_context_resolver/`
- `src/kazusa_ai_chatbot/complex_task_resolver/`
- `src/kazusa_ai_chatbot/coding_agent/`
- current accepted-task/background routing and their existing model-route
  configuration.

Plan 2 adds no new import from these graph/harness owners to the DSH gateway.
Static tests allow imports only from the listed deterministic leaf owners.

### Create Or Modify — Tests And Fixtures (Phase 1)

- Modify `tests/test_config.py`.
- Modify `tests/test_agentic_resolver_contracts.py`.
- Modify `tests/test_agentic_resolver_controller.py`.
- Modify `tests/test_agentic_resolver_decommission.py`.
- Modify `tests/test_agentic_resolver_evidence.py`.
- Modify `tests/test_agentic_resolver_fingerprints.py`.
- Modify `tests/test_agentic_resolver_live_db.py`.
- Modify `tests/test_agentic_resolver_live_llm.py`.
- Modify `tests/test_agentic_resolver_persistence.py`.
- Modify `tests/test_agentic_resolver_rpc.py`.
- Modify `tests/test_agentic_resolver_runtime.py`.
- Modify `tests/test_agentic_resolver_sidecar_process.py`.
- Modify `sidecars/dsh_resolution/tests/contracts.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/evidence.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/lifecycle.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/process.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/profile.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/rpc.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/runtime.spec.ts`.
- Modify `sidecars/dsh_resolution/tests/submit_resolution.spec.ts`.
- Create `tests/test_dsh_tool_gateway_contracts.py`.
- Create `tests/test_dsh_tool_gateway_authority.py`.
- Create `tests/test_dsh_tool_gateway_conversation.py`.
- Create `tests/test_dsh_tool_gateway_memory.py`.
- Create `tests/test_dsh_tool_gateway_people.py`.
- Create `tests/test_dsh_tool_gateway_recall_calendar.py`.
- Create `tests/test_dsh_tool_gateway_media.py`.
- Create `tests/test_dsh_tool_gateway_worker.py`.
- Create `tests/test_dsh_brain_interaction_contracts.py`.
- Create `tests/test_dsh_brain_interaction_auth.py`.
- Create `tests/test_dsh_brain_interaction_persistence.py`.
- Create `tests/test_dsh_brain_interaction_decision.py`.
- Create `tests/test_dsh_brain_interaction_pending.py`.
- Create `tests/test_dsh_brain_interaction_resume.py`.
- Create `tests/test_dsh_brain_interaction_service.py`.
- Create `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py`.
- Create `tests/test_dsh_standard_profile_live_llm.py`.
- Create `tests/test_dsh_brain_interaction_live_llm.py`.
- Create `tests/test_dsh_plan2_e2e_live_llm.py`.
- Create `sidecars/dsh_resolution/tests/standard_profile.spec.ts`.
- Create `sidecars/dsh_resolution/tests/model_route.spec.ts`.
- Create `sidecars/dsh_resolution/tests/secret_broker.spec.ts`.
- Create `sidecars/dsh_resolution/tests/semantic_gateway.spec.ts`.
- Create `sidecars/dsh_resolution/tests/brain_interaction.spec.ts`.
- Create `sidecars/dsh_resolution/tests/terminal_policy.spec.ts`.
- Create a self-contained coding fixture under
  `tests/fixtures/dsh_standard_coding/` with one small defect and its existing
  deterministic test. It imports no Kazusa coding-agent code.

`tests/test_test_impact_manifest.py` is run unchanged; Plan 2 adds no edit to
that validator source.

### Documentation (Phase 3)

- Modify `README.md`.
- Modify `docs/HOWTO.md`.
- Modify `docs/architecture/dsh_integration_architecture.md`.
- Modify `docs/architecture/agentic_resolver_architecture.md`.
- Modify `src/agentic_resolver/README.md`.
- Modify `sidecars/dsh_resolution/README.md`.
- Modify `src/kazusa_ai_chatbot/brain_service/README.md`.
- Modify `src/kazusa_ai_chatbot/cognition_core_v3/README.md`.
- Create `src/kazusa_ai_chatbot/dsh_tool_gateway/README.md`.
- Create `src/kazusa_ai_chatbot/dsh_interaction/README.md`.

Documentation must state the exact native/custom catalog boundary, the
storage-independent semantic schemas, Qwen route, DSH-native workspace policy,
default DSH web-provider boundary, interaction state machine, V2 epochs,
startup commands, health fields, fault recovery, and Plan 3 deferrals.

## Source-To-Test Ownership Matrix

Every changed or created Python/TypeScript semantic owner has at least one
direct deterministic gate. The exact node names below are part of this work
contract and must collect before Phase 1 closes.

| Source path | Required exact deterministic node |
|---|---|
| `src/kazusa_ai_chatbot/config.py` | `tests/test_config.py::test_agentic_resolver_route_settings_are_strict_and_load_initial_qwen_route` |
| `src/kazusa_ai_chatbot/db/resolution_threads.py` | `tests/test_agentic_resolver_persistence.py::test_v2_thread_persists_brain_workspace_route_and_interaction_identity` |
| `src/kazusa_ai_chatbot/db/dsh_interactions.py` | `tests/test_dsh_brain_interaction_persistence.py::test_interaction_store_indexes_reply_lookup_and_atomic_one_shot_grant_consumption` |
| `src/agentic_resolver/__init__.py` | `tests/test_agentic_resolver_contracts.py::test_public_resolver_exports_only_v2_product_contracts` |
| `src/agentic_resolver/contracts.py` | `tests/test_agentic_resolver_contracts.py::test_v2_intake_separates_model_input_from_workspace_tool_and_brain_authority` |
| `src/agentic_resolver/controller.py` | `tests/test_agentic_resolver_controller.py::test_interaction_checkpoint_and_resume_preserve_exact_thread_segment_and_fence` |
| `src/agentic_resolver/errors.py` | `tests/test_agentic_resolver_contracts.py::test_v2_runtime_and_interaction_fault_codes_are_closed` |
| `src/agentic_resolver/fingerprints.py` | `tests/test_agentic_resolver_fingerprints.py::test_v2_digests_bind_standard_catalog_route_workspace_and_policy` |
| `src/agentic_resolver/persistence.py` | `tests/test_agentic_resolver_persistence.py::test_v1_rows_are_historical_and_never_resumed_as_v2` |
| `src/agentic_resolver/rpc.py` | `tests/test_agentic_resolver_rpc.py::test_v2_rpc_reconciles_committed_interaction_checkpoint_without_duplicate_execution` |
| `src/agentic_resolver/runtime.py` | `tests/test_agentic_resolver_runtime.py::test_runtime_builds_v2_authority_from_canonical_project_route_and_workspace` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/__init__.py` | `tests/test_dsh_tool_gateway_contracts.py::test_public_gateway_exports_are_bounded_contracts_only` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/contracts.py` | `tests/test_dsh_tool_gateway_contracts.py::test_semantic_result_uses_entities_opaque_page_refs_and_idempotent_mutation_outcomes` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/authority.py` | `tests/test_dsh_tool_gateway_authority.py::test_activation_authenticates_complete_catalog_scope_fence_reference_lineage_and_replay` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/catalog.py` | `tests/test_dsh_tool_gateway_contracts.py::test_description_stripped_catalog_is_self_explanatory_storage_independent_and_excludes_standard_capabilities` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/conversation.py` | `tests/test_dsh_tool_gateway_conversation.py::test_conversation_services_use_semantic_queries_opaque_refs_pagination_and_provenance` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/memory.py` | `tests/test_dsh_tool_gateway_memory.py::test_memory_services_search_read_remember_revise_and_change_lifecycle_without_storage_vocabulary` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/people.py` | `tests/test_dsh_tool_gateway_people.py::test_people_services_return_semantic_candidates_profiles_and_opaque_person_refs` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/recall_calendar.py` | `tests/test_dsh_tool_gateway_recall_calendar.py::test_recall_and_calendar_services_return_semantic_entries_and_opaque_pagination` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/media.py` | `tests/test_dsh_tool_gateway_media.py::test_attached_media_inspection_accepts_only_brain_issued_semantic_refs` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/dispatch.py` | `tests/test_dsh_tool_gateway_worker.py::test_dispatch_exposes_only_approved_semantic_services_and_routes_no_standard_or_graph_capability` |
| `src/kazusa_ai_chatbot/dsh_tool_gateway/worker.py` | `tests/test_dsh_tool_gateway_worker.py::test_worker_replays_committed_idempotent_results_and_preserves_uncertain_mutation_state` |
| `src/kazusa_ai_chatbot/dsh_interaction/__init__.py` | `tests/test_dsh_brain_interaction_contracts.py::test_public_interaction_exports_exclude_adapter_and_dsh_internal_types` |
| `src/kazusa_ai_chatbot/dsh_interaction/contracts.py` | `tests/test_dsh_brain_interaction_contracts.py::test_request_decision_pending_and_grant_contracts_are_exact_and_kind_specific` |
| `src/kazusa_ai_chatbot/dsh_interaction/auth.py` | `tests/test_dsh_brain_interaction_auth.py::test_mac_timestamp_nonce_digest_and_constant_time_validation_fail_closed` |
| `src/kazusa_ai_chatbot/dsh_interaction/decision.py` | `tests/test_dsh_brain_interaction_decision.py::test_brain_semantic_decision_is_enacted_without_keyword_or_post_llm_reclassification` |
| `src/kazusa_ai_chatbot/dsh_interaction/pending.py` | `tests/test_dsh_brain_interaction_pending.py::test_relay_pending_matches_exact_delivered_reply_lineage_and_expires_closed` |
| `src/kazusa_ai_chatbot/dsh_interaction/resume.py` | `tests/test_dsh_brain_interaction_resume.py::test_user_resolution_schedules_one_same_thread_continuation_and_one_shot_matching_grant` |
| `src/kazusa_ai_chatbot/dsh_interaction/service.py` | `tests/test_dsh_brain_interaction_service.py::test_signed_loopback_interaction_returns_immediate_decision_or_durable_checkpoint_required` |
| `src/kazusa_ai_chatbot/brain_service/contracts.py` | `tests/test_dsh_brain_interaction_service.py::test_brain_service_exposes_only_versioned_internal_dsh_request_and_response_models` |
| `src/kazusa_ai_chatbot/service.py` | `tests/test_dsh_brain_interaction_service.py::test_service_relay_uses_cognition_dialog_dispatcher_then_resumes_after_normal_chat_commit` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | `tests/test_dsh_brain_interaction_decision.py::test_global_state_carries_typed_pending_interaction_and_semantic_decision` |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | `tests/test_dsh_brain_interaction_decision.py::test_persona_projects_pending_dsh_context_and_returns_canonical_p_stage_decision` |
| `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py` | `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py::test_response_plan_requires_exact_kind_compatible_dsh_decision_only_when_context_exists` |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py::test_p_stage_decision_survives_canonical_output_without_deterministic_semantic_rewrite` |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py::test_p_prompt_assigns_decision_to_brain_and_visible_wording_to_dialog` |
| `src/kazusa_ai_chatbot/cognition_shared/contracts.py` | `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py::test_cognition_input_validates_pending_dsh_context_as_untrusted_bounded_evidence` |
| `sidecars/dsh_resolution/src/contracts.ts` | `sidecars/dsh_resolution/tests/contracts.spec.ts > V2 contracts > rejects V1 and separates model-visible input from authority` |
| `sidecars/dsh_resolution/src/evidence.ts` | `sidecars/dsh_resolution/tests/evidence.spec.ts > V2 evidence > rebuilds semantic native and artifact receipts after restart` |
| `sidecars/dsh_resolution/src/main.ts` | `sidecars/dsh_resolution/tests/process.spec.ts > V2 process > starts only after route standard worker web and brain health are ready` |
| `sidecars/dsh_resolution/src/operations.ts` | `sidecars/dsh_resolution/tests/lifecycle.spec.ts > V2 operation lifecycle > interaction deferral is idempotent under operation replay` |
| `sidecars/dsh_resolution/src/profile.ts` | `sidecars/dsh_resolution/tests/standard_profile.spec.ts > official Standard profile > mounts installed standard without a copied preset` |
| `sidecars/dsh_resolution/src/rpc.ts` | `sidecars/dsh_resolution/tests/rpc.spec.ts > V2 RPC > requires loopback bearer and V2 protocol` |
| `sidecars/dsh_resolution/src/runtime.ts` | `sidecars/dsh_resolution/tests/terminal_policy.spec.ts > autonomous multi-tool runtime > applies no Kazusa step call byte or deadline budget and accepts only sole terminal submit` |
| `sidecars/dsh_resolution/src/submit_resolution.ts` | `sidecars/dsh_resolution/tests/submit_resolution.spec.ts > V2 submit_resolution > rejects foreign segment evidence and out-of-workspace artifacts` |
| `sidecars/dsh_resolution/src/composition.ts` | `sidecars/dsh_resolution/tests/standard_profile.spec.ts > official Standard profile > catalog retains the complete pinned Standard set and adds only noncolliding Kazusa semantic tools` |
| `sidecars/dsh_resolution/src/model_route.ts` | `sidecars/dsh_resolution/tests/model_route.spec.ts > Qwen route > sends system tools max tokens and qwen thinking kwargs through pi-ai` |
| `sidecars/dsh_resolution/src/secret_broker.ts` | `sidecars/dsh_resolution/tests/secret_broker.spec.ts > secret isolation > native shell cannot read host credentials tokens or bridge secrets` |
| `sidecars/dsh_resolution/src/semantic_gateway.ts` | `sidecars/dsh_resolution/tests/semantic_gateway.spec.ts > semantic gateway > attaches invisible authority and persists bounded evidence receipts` |
| `sidecars/dsh_resolution/src/brain_interaction.ts` | `sidecars/dsh_resolution/tests/brain_interaction.spec.ts > Brain interaction > maps decisions exactly and checkpoints relay without direct user surface` |

`tests/ownership/source_test_impact_manifest.json` gains one row for every new
Python source owner and retains exact existing rows for modified sources. The
manifest validator itself is gated by
`tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`.

## Cross-Boundary And Live Acceptance Nodes

These nodes supplement direct owner tests and are also frozen:

| Boundary | Exact node |
|---|---|
| Sidecar to framed Python worker | `tests/test_dsh_tool_gateway_worker.py::test_real_sidecar_worker_round_trip_preserves_authority_result_and_evidence` |
| Sidecar to Brain immediate answer | `tests/test_dsh_brain_interaction_service.py::test_real_sidecar_question_is_answered_by_brain_without_user_delivery` |
| Sidecar to Brain approval | `tests/test_dsh_brain_interaction_service.py::test_real_sidecar_outside_workspace_retry_consumes_one_brain_grant` |
| Sidecar to Brain media cache | `tests/test_dsh_brain_interaction_service.py::test_signed_media_inspection_resolves_only_exact_scoped_cache_ref` |
| Brain relay and later chat reply | `tests/test_dsh_brain_interaction_service.py::test_relay_checkpoints_delivers_matches_reply_and_resumes_same_thread` |
| Native web isolation | `tests/test_agentic_resolver_sidecar_process.py::test_default_dsh_web_provider_uses_only_native_names_and_imports_no_kazusa_webagent` |
| Python V2 evidence projection | `tests/test_agentic_resolver_evidence.py::test_v2_evidence_receipts_bind_native_semantic_and_artifact_provenance` |
| Existing sidecar profile suite | `sidecars/dsh_resolution/tests/profile.spec.ts > V2 profile invariants > rejects V1 epoch and verifies installed base and standard digests` |
| Existing sidecar runtime suite | `sidecars/dsh_resolution/tests/runtime.spec.ts > V2 runtime protocol > executes normal multi-tool steps before sole terminal` |
| Standard coding without legacy harness | `tests/test_agentic_resolver_sidecar_process.py::test_standard_pwsh_reads_edits_runs_fixture_test_and_imports_no_kazusa_coding_agent` |
| Cold restart | `tests/test_agentic_resolver_sidecar_process.py::test_v2_cold_restart_rebuilds_standard_session_evidence_pending_interaction_and_terminal` |
| Real semantic-service persistence | `tests/test_agentic_resolver_live_db.py::test_v2_gateway_abstracts_storage_and_preserves_idempotent_memory_mutations` |
| Real Mongo pending interaction | `tests/test_agentic_resolver_live_db.py::test_v2_brain_pending_and_one_shot_grant_survive_service_restart` |
| Real Qwen conversation, people, and memory reads | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_conversation_people_and_memory_read_tools` |
| Real Qwen memory mutation lifecycle | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_memory_write_revision_lifecycle_and_readback` |
| Real Qwen recall, calendar, and media context | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_active_recall_calendar_and_attached_media_tools` |
| Real Qwen semantic/mixed resolution | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_selects_description_stripped_semantic_and_native_tools_then_submits_grounded_terminal` |
| Existing real resolver path | `tests/test_agentic_resolver_live_llm.py::test_qwen27b_v2_resolution_round_trip_preserves_thread_and_terminal_contract` |
| Real Qwen native coding | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_standard_mode_repairs_fixture_with_native_workspace_tools` |
| Real Qwen approval behavior | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_outside_workspace_request_round_trips_through_brain_once` |
| Real cognition immediate decision | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_answers_or_rejects_dsh_request_from_context` |
| Real cognition relay/resume | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_relays_ambiguous_permission_then_interprets_user_reply` |
| Plan 3 boundary | `tests/test_agentic_resolver_decommission.py::test_plan2_adds_only_interaction_bridge_and_does_not_cut_over_task_resolution` |

## Real LLM Coverage And Final Sign-Off

Real-LLM coverage is defined by advertised Plan 2 product behavior:

- every P2 release-gate capability;
- each of the thirteen Kazusa semantic tools; and
- DSH Standard at the advertised product-capability level: native coding,
  native web, questions/approvals, autonomous tool execution, and terminal
  submission. Individual upstream helper tools are not separate Kazusa
  coverage obligations.

Phase 1 implements this frozen feature-to-live-node matrix:

| Advertised feature | Exact real-LLM coverage owner |
|---|---|
| `search_conversation`, `read_conversation_entries`, `summarize_conversation_participants`, `find_people`, `read_people_profiles`, `search_memory`, and `read_memory` | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_conversation_people_and_memory_read_tools` |
| `remember_memory`, `revise_memory`, and `set_memory_lifecycle`, including semantic-service readback through the storage abstraction | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_memory_write_revision_lifecycle_and_readback` |
| `get_active_recall`, `get_calendar_entries`, and `get_attached_media_context` | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_active_recall_calendar_and_attached_media_tools` |
| Self-explanatory Kazusa semantic names, DSH-native web, autonomous mixed tool selection, evidence binding, and sole terminal submission | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_selects_description_stripped_semantic_and_native_tools_then_submits_grounded_terminal` |
| DSH Standard native coding against the dedicated Plan 2 fixture | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_standard_mode_repairs_fixture_with_native_workspace_tools` |
| Approval request, Brain decision, one-shot authority, and same-session continuation | `tests/test_dsh_standard_profile_live_llm.py::test_qwen27b_outside_workspace_request_round_trips_through_brain_once` |
| Brain answers or rejects a DSH question from existing context | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_answers_or_rejects_dsh_request_from_context` |
| Brain relays a DSH question, judges the user reply, and resumes the same DSH goal | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_relays_ambiguous_permission_then_interprets_user_reply` |
| V2 intake, thread/session continuity, autonomous multi-tool execution, evidence receipts, and terminal contract | `tests/test_agentic_resolver_live_llm.py::test_qwen27b_v2_resolution_round_trip_preserves_thread_and_terminal_contract` |
| Unchanged pre-Plan 3 production task-resolution handover | `tests/test_task_resolution_persona_e2e_live_llm.py::test_live_inline_result_returns_grounded_dialog` |

Every advertised feature is exercised through at least one real-LLM node;
deterministic registry, schema, or handler tests cannot substitute for that
model-facing execution. The coding live cases use the new DSH fixture and no
legacy Kazusa coding harness entry point.

The execution and remediation cycle is fixed:

1. Run every real-LLM coverage node individually with `-q -s`, inspect it, and
   record its observed behavior and failure mode before moving to the next
   node. A semantic failure remains recorded while execution continues through
   the complete coverage matrix. Only an unavailable/invalid model backend or
   broken test harness stops the coverage pass because later results would not
   be valid evidence.
2. After the complete pass, group failures by shared root cause and apply one
   consolidated remediation pass. Remediation targets the common semantic or
   interface contract rather than individual expected words or one fixture.
3. Verify remediation with the minimum real-LLM nodes that faithfully
   reproduce each failure, plus only the mapped deterministic owner nodes and
   directly affected neighboring live behavior. A minor issue does not trigger
   the full repository suite or the complete live matrix.
4. When a failure is not reproducible, the implementation owner inspects the
   original trace, exact backend/configuration, and nearest model-facing
   boundary, then builds the smallest faithful reproducer. Broad-suite runs or
   repeated unstructured model calls do not replace reproduction.

Final sign-off uses between one and five real-LLM E2E nodes, selecting the
smallest set that covers the completed Plan 2 surface. The maximum set is:

- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_context_people_memory_recall_and_calendar`
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_memory_create_revise_lifecycle_and_readback`
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_attached_media_native_web_and_semantic_evidence`
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_native_coding_repairs_and_verifies_workspace_fixture`
- `tests/test_dsh_plan2_e2e_live_llm.py::test_e2e_brain_judgment_checkpoint_relay_resume_and_terminal`

These E2E nodes are the final agent-behavior sign-off. Each uses the configured
real local model and full sidecar boundary, permits semantic variation, and
judges task completion, grounded tool use, Brain ownership, and terminal
behavior. Passing deterministic or patched tests cannot waive a failed E2E
behavioral judgment.

## Execution Roles

### `p2_implementation_owner`

- **Responsibility:** deliver the complete Plan 2 runtime, catalog, Brain
  interaction, tests, documentation, evidence, and lifecycle updates.
- **Owned surface:** every create/modify/delete path in the Exact Change
  Inventory, with unrelated worktree paths excluded.
- **Authority:** edit and verify the owned surface after an explicit production
  implementation command; make local mechanics decisions inside the fixed
  contracts; request an amendment for any catalog, architecture, route,
  provider, policy, or change-radius expansion.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety` when applicable,
  `test-style-and-execution`, and `python-venv` when environment work applies.
- **Capability floor:** production Python and TypeScript, DSH/Cordis profile
  composition, async process/RPC lifecycle, cognition contracts, semantic
  persistence abstraction, and deterministic/live verification.
- **Independence requirement:** none.
- **Acceptance output:** one release candidate with the exact diff, mapped
  deterministic and live evidence, catalog/route/profile digests, and all ten
  functional gates green.
- **Gate:** enters after explicit production authorization and Phase 0; exits
  only after P2-P3 and the mapped source/test inventory pass.

### `p2_independent_reviewer`

- **Responsibility:** independently verify scope, DSH autonomy, native
  precedence, storage abstraction, Brain judgment ownership, test evidence,
  and absence of Plan 3 production cutover.
- **Owned surface:** read-only review of the complete candidate, plan, evidence,
  and mapped tests.
- **Authority:** issue material findings and pass or block sign-off; remediation
  remains with `p2_implementation_owner`.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, and the test policy for evidence review.
- **Capability floor:** independent architecture/code review across Python,
  TypeScript, DSH tooling, cognition, persistence, and live verification.
- **Independence requirement:** a different executor from the implementation
  candidate and every remediation pass.
- **Acceptance output:** a written pass/fail decision tied to the ten release
  gates and exact residual risks.
- **Gate:** enters after the implementation owner presents a complete green
  candidate; exits only with zero unresolved material finding.

## Mandatory Execution Phases

### Phase 0 — Approval And Baseline

1. Record `git status --short` and the exact owned path baseline.
2. Confirm the active plan status is `approved`, then change it to
   `in_progress` immediately before the first test edit.
3. Verify `venv\Scripts\python`, Node, Corepack, and pnpm 11.7.0.
4. Record the installed DSH package/version/commit evidence and the exact
   official Standard preset digest.
5. Validate the non-secret canonical route fields and workspace path without
   printing credentials.

**Gate P2-P0:** clean/understood baseline, approved plan, available toolchain,
and exact upstream pin recorded.

### Phase 1 — Tests, Fixtures, And Ownership First

1. Apply only the frozen `package.json` dependency delta, regenerate
   `pnpm-lock.yaml`, and install it. This is the sole production/configuration
   prerequisite allowed before the test diff because Vitest must resolve the
   official pinned packages to collect.
2. Create/modify every test and fixture listed above.
3. Update the source-test impact manifest for every planned owner.
4. Add V2 contract, exact catalog, namespace collision, description-stripped
   selection, storage-vocabulary exclusion, model-route, DSH-native policy,
   secret isolation, semantic-service authority and mutation idempotency, Brain
   decision, relay/resume, restart, terminal, and static non-import assertions.
5. Implement the frozen advertised-feature-to-real-LLM-node coverage matrix
   above and create all five named E2E sign-off nodes. Each of the thirteen
   semantic tools and every advertised Plan 2 product capability has at least
   one real-LLM owner.
6. Run exact collection for every frozen pytest node and Vitest test name.
7. Run the deterministic matrix and record expected red failures. A failure is
   admissible only when it names a planned missing/changed production symbol
   or behavior. Test syntax, fixture, collection, and unrelated failures are
   fixed in Phase 1.

Dependency preparation commands are exact:

```powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --lockfile-only
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
```

**Gate P2-P1:** all exact nodes collect, the entire test/fixture/manifest diff
exists, and every red result maps to a frozen production requirement.

### Phase 2 — Production

Implement in this blocking order while continually running the Phase 1 tests:

1. V2 Python/TypeScript contracts, fingerprints, persistence epoch, and route
   settings.
2. Official base/Standard composition, pi-ai Qwen route, secret broker, and
   health diagnostics.
3. Fixed storage-independent catalog, capability signer/verifier, framed
   worker, semantic-service handlers, idempotent mutations, opaque pagination,
   and evidence receipts.
4. Default DSH web-provider isolation and native catalog verification.
5. Brain interaction contracts, auth, cognition P-stage decision, pending
   storage, Kazusa dialog/delivery relay, one-shot grant, and controller resume.
6. Autonomous multi-tool runtime, interaction checkpoint, terminal validation,
   restart/replay, removal of Plan 1's minimal restrictions, and removal of
   Kazusa-imposed DSH budgets.
7. Lockfile integrity and source-test ownership completion.

Run at minimum:

```powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution typecheck
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution test
venv\Scripts\python -m compileall -q src/agentic_resolver src/kazusa_ai_chatbot/db/dsh_interactions.py src/kazusa_ai_chatbot/dsh_tool_gateway src/kazusa_ai_chatbot/dsh_interaction
venv\Scripts\python -m pytest -q tests/test_config.py tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_controller.py tests/test_agentic_resolver_decommission.py tests/test_agentic_resolver_evidence.py tests/test_agentic_resolver_fingerprints.py tests/test_agentic_resolver_persistence.py tests/test_agentic_resolver_rpc.py tests/test_agentic_resolver_runtime.py tests/test_agentic_resolver_sidecar_process.py
venv\Scripts\python -m pytest -q tests/test_dsh_tool_gateway_contracts.py tests/test_dsh_tool_gateway_authority.py tests/test_dsh_tool_gateway_conversation.py tests/test_dsh_tool_gateway_memory.py tests/test_dsh_tool_gateway_people.py tests/test_dsh_tool_gateway_recall_calendar.py tests/test_dsh_tool_gateway_media.py tests/test_dsh_tool_gateway_worker.py
venv\Scripts\python -m pytest -q tests/test_dsh_brain_interaction_contracts.py tests/test_dsh_brain_interaction_auth.py tests/test_dsh_brain_interaction_persistence.py tests/test_dsh_brain_interaction_decision.py tests/test_dsh_brain_interaction_pending.py tests/test_dsh_brain_interaction_resume.py tests/test_dsh_brain_interaction_service.py tests/unit/cognition_core_v3/test_dsh_interaction_contract.py
venv\Scripts\python -m pytest -q tests/test_test_impact_manifest.py
```

After the deterministic and cross-process candidate is ready, run the complete
real-LLM feature coverage matrix once, collecting and inspecting all semantic
failures before remediation. Apply one consolidated remediation pass, then run
only the minimum faithful reproduction nodes and their mapped deterministic
owners. Phase 3 runs the final one-to-five E2E sign-off set.

**Gate P2-P2:** deterministic, cross-process, and real-DB owners are green;
every advertised feature has been exercised by the real local model; the full
coverage pass and consolidated failure inventory exist; every remediation has
a faithful minimal real-LLM verification; TypeScript typecheck/build/tests
pass; and no production or live case imports a legacy coding/resolver graph
through a DSH tool.

### Phase 3 — Documentation And Closure

1. Update every documentation path in the inventory.
2. Replace Plan 1 submit-only/KAZUSA_DSH_MODEL instructions with Standard V2
   startup, canonical route, native web, workspace policy, Brain interaction,
   worker, and recovery instructions.
3. Run documentation/static assertions, the impact validator, the mapped
   deterministic documentation owners, and the final one-to-five E2E real-LLM
   sign-off nodes against the accepted runtime.
4. Record exact commands/results, health output with secrets redacted, catalog
   digest, route digest, policy epoch, store epoch, sandbox enforcement fact,
   final diff, and gate decisions.
5. Mark this plan `completed`, move it to
   `development_plans/archive/completed/short_term/`, and update the registry.
   Plan 3 remains draft until separately refined and approved.

**Gate P2-P3:** documentation matches the accepted runtime; every release gate
below is green without waiver; the final diff contains only planned paths.

## Functional Release Gates

| Gate | Green condition |
|---|---|
| P2-G1 — Official Base and Standard | The installed pinned base bundle and Standard preset are mounted by reference, both digests match the recorded pin, the complete enabled catalog is visible, and no copied/forked or Kazusa policy composition exists. |
| P2-G2 — Canonical Qwen route | DSH uses only `AGENTIC_RESOLVER_LLM_*`, initially `qwen27b-5090`; fake-wire and real-Qwen tool-loop tests pass; secrets remain host-only. |
| P2-G3 — Semantic Brain/memory catalog | The thirteen fixed `kazusa_*` semantic tools provide conversation, people, profile, memory read/write/lifecycle, active recall, calendar, and attached-media capabilities through opaque semantic references. Description-stripped and storage-vocabulary tests pass; a simulated collision retains the DSH-native tool and reports `native_precedence`. |
| P2-G4 — Native public web | The model sees only the native web capabilities published by the default pinned DSH provider; static and process tests prove isolation from Kazusa webagent, SearXNG, and URL-reader code. |
| P2-G5 — Native coding autonomy | Standard reads, edits, writes, commands, jobs, and tests through its unchanged native policy; Kazusa injects no coding tool, harness, command filter, sandbox overlay, or execution instruction. |
| P2-G6 — Brain judgment | Every DSH approval, question, and plan-review interaction reaches Brain; immediate Brain decision, paraphrased/direct relay, delivery, reply matching, checkpoint, resume, expiry, restart, and native one-shot approval behavior pass. Brain owns whether the user is involved. |
| P2-G7 — Durable lifecycle | V2 intake/thread/segment/store epochs, operation idempotency, leases, evidence replay, interaction replay, checkpoint, terminal, and cold restart retain Plan 1 guarantees while Kazusa adds no DSH step/call/byte/deadline budget. |
| P2-G8 — Terminal integrity | Multi-tool steps are accepted; only a sole valid `submit_resolution` terminates; findings/artifacts are authorized and evidence-bound; structural rejection returns to the autonomous DSH loop without a Kazusa regeneration policy. |
| P2-G9 — Plan 3 boundary | Production `task_resolution_request`, accepted-task/background resolution, legacy RAG/resolver/coding callers, and their current routes remain unchanged; no fallback or partial cutover is introduced. |
| P2-G10 — Real-LLM coverage and E2E sign-off | Every advertised Plan 2 feature and each of the thirteen Kazusa semantic tools is exercised in the complete real-local-model coverage pass before remediation. Failures are consolidated, fixed in one pass, and verified with the minimum faithful real-LLM reproductions. The final sign-off uses no more than five named E2E nodes and every selected node passes behavioral inspection. |

All ten gates are blockers. Passing handler unit tests without real sidecar,
real model, or Brain relay/resume evidence is insufficient.

## Out Of Scope

- Production `task_resolution_request` routing to DSH.
- Mapping DSH exhaust to `TaskResolutionResultV1`.
- Accepted-task/background DSH scheduling, delivery cutover, or deployment
  drain.
- Deleting legacy task-resolution, RAG, local/complex resolver, coding-agent,
  prompt, test, model-route, or checkpoint owners.
- A compatibility shim, V1 converter, dual route, shadow route, fallback
  resolver, or percentage rollout.
- Modifying or forking DSH source.
- A Kazusa model-facing web, filesystem, shell, coding, text, computation,
  question, plan, goal, todo, subagent, workflow, or Ralph tool.
- A Kazusa-imposed DSH prompt policy, model-step limit, tool-call limit,
  retained-byte limit, execution deadline, retry loop, sandbox overlay, or
  permission overlay.
- Moving character judgment, visible wording, scheduling, adapter delivery, or
  storage ownership into DSH. DSH invokes Kazusa semantic persistence services;
  Kazusa remains the persistence owner.

## Approval Boundary

This document contains no unresolved implementation choice. Its approved state
authorizes only the exact Plan 2 inventory and gates once the user separately
commands production implementation.
Any production path expansion, tool-catalog change, DSH pin change, model-route
shape change, permission-policy change, or absorption of Plan 3 deletion scope
requires an explicit plan amendment and user approval before its test edit.
