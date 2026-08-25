# Agentic Resolver Target Architecture

## Document control

| Field | Value |
| --- | --- |
| Status | Target architecture for review |
| Date | 2026-08-25 |
| Scope | Replacement of Kazusa task-resolution DAGs |
| Current implementation ICD | src/agentic_resolver/README.md |
| Stable caller | Kazusa brain action selector through task_resolution_request |
| Supersedes | Standalone-first, four-facade, and DAG-backed resolver direction |

This document defines the renewed target architecture. It is not a statement
that the target is already implemented. Until a cutover is approved and
completed, src/agentic_resolver/README.md remains the implementation contract
for the current standalone prototype.

## Executive decision

Kazusa will use one agentic resolver as the resolution engine behind the
existing brain-owned task-resolution boundary.

The brain action selector remains the caller. task_resolution_request remains
the request surface. The current inline, direct-background, promotion, resume,
result, and observation mechanics remain the brain-facing contract.

Inside that boundary, the agentic resolver replaces the current task
resolution, RAG, internal resolver, complex resolver, and external/web
orchestration DAGs. The model receives a catalog of eligible base-level
semantic tools and decides, step by step, which tool to call, whether another
tool is needed, and when evidence is sufficient to submit a resolution.

The resolver never decides whether a job belongs in the background, whether a
response should be visible, what Kazusa's character stance is, or how the
final user-facing message is worded. Those decisions remain with the brain.

This is a big-bang replacement inside the resolution layer while preserving
the stable boundary above it. The target does not introduce a second live
resolver path or a compatibility vocabulary between old graphs and new tools.

## Confirmed use-case boundary

The immediate call sequence is:

    platform adapter or debug client
        -> Kazusa brain
        -> action selector
        -> task_resolution_request
        -> task-resolution service
        -> agentic resolver session
        -> TaskResolutionResult
        -> ResolverObservation
        -> cognition
        -> dialog or another brain-owned action

The brain owns:

- whether the selected action is task resolution;
- foreground versus background priority;
- the inline deadline and deterministic promotion decision;
- accepted-task notification, queueing, resume, and delivery;
- character judgment, relationship boundaries, and response goals;
- whether to speak and the visible wording.

The agentic resolver owns:

- interpreting the bounded resolution objective supplied by the brain;
- selecting eligible tools from the current catalog;
- forming typed tool arguments;
- evaluating returned evidence and deciding the next resolution step;
- requesting clarification or approval through typed terminal outcomes;
- submitting a structured resolution with evidence provenance.

Deterministic runtime code owns:

- schema validation and catalog construction;
- code-bound scope, permission, and target checks;
- tool dispatch, timeouts, quotas, and result-size limits;
- durable event recording and call/result correlation;
- idempotency, outcome verification, and crash recovery;
- foreground deadlines, background scheduling, and delivery;
- persistence constraints and audit records.

## Direction discarded by this architecture

The following previous design choices are superseded wherever they conflict
with this document:

1. A standalone resolver as the long-term product boundary.
2. A fixed four-tool facade made of local_context, public_research, coding,
   and text_computation.
3. Keeping task, RAG, complex, or web DAGs hidden behind those facade tools.
4. Selecting one specialist route before evidence gathering begins.
5. Treating Phase 2 as merely adding more adapters to the same facade model.
6. Allowing an adapter to return prose that bypasses a typed terminal result.
7. Letting the resolver choose foreground/background execution or user
   delivery.
8. Allowing self-authored skills to become active from one successful session.

Existing leaf executors may be retained where their ownership is appropriate.
Graph routers, graph state vocabularies, branch selectors, graph checkpoints,
and facade-only compatibility mappings do not remain on the live target path.

## Smallest model-owned contract

For each model step, the resolver model does exactly one of these:

- call one named tool using schema-valid arguments;
- call skill to load one curated skill version;
- call submit_resolution with the terminal structured outcome.

The first integrated release accepts one complete native tool call per model
step. A session can make many successive calls. This makes local-model output
easier to validate and replay without removing the model's freedom to explore.

Assistant prose is not runtime control. Private reasoning may be streamed for
diagnostics, but only a validated native call changes state.

submit_resolution is the normal terminal operation. Its outcome is one of:

- resolved: sufficient evidence supports an answer;
- partial: useful evidence exists with explicit remaining limitations;
- needs_user_input: a missing fact or ambiguity prevents continuation;
- approval_required: the next operation requires brain/user approval;
- unavailable: no eligible capability or accessible source can complete it;
- failed: the session reached a typed unrecoverable failure.

These are the existing TaskResolutionResult terminal statuses. deferred is
runtime-owned and is never selected through submit_resolution.

The controller projects the validated terminal call into the existing
TaskResolutionResultV1 fields: semantic objective, status, scene context, goal
continuation reference, evidence state, excerpts, handles, prompt-safe
summary, structured evidence, completed subgoals, remaining needs, checkpoint,
and coding-run context. The public schema version and caller projection remain
stable. Its opaque checkpoint payload carries the agentic session identifier
and revision after cutover rather than legacy graph state.

## Tool exposure policy

### Meaning of all interfaces

Kazusa exposes all eligible resolution-facing base-level interfaces to the
agentic resolver as tools by default.

Eligibility means the interface:

- performs one explicit semantic operation useful to resolution;
- has a typed input and typed result;
- has a code-enforced authorization scope;
- declares side effects, idempotency, timeouts, and provenance behavior;
- can be called without transferring cognition, dialog, delivery, or
  scheduling ownership to the resolver.

The catalog is capability-complete, not unrestricted. Every public Kazusa
brain interface must have one of two registry states:

- exposed, with a versioned tool manifest; or
- excluded, with a documented ownership or safety reason.

A catalog completeness test fails when an eligible interface has neither
state. This prevents the system from quietly returning to a small hand-picked
facade.

The following are explicit exclusions:

- raw MongoDB collections, database clients, and arbitrary query execution;
- action selection, cognition, dialog, character state judgment, and delivery;
- adapters and platform credentials;
- task queues, schedulers, checkpoint internals, and accepted-task mutation;
- graph nodes, stage prompts, raw LLM clients, caches, and trace stores;
- unrestricted shell or filesystem access;
- persistence writes owned by memory consolidation or another brain action;
- the resolver's own session and dispatch internals.

These exclusions are not hidden capabilities. They are ownership boundaries.
If a resolution use case needs one, Kazusa must define a bounded semantic
interface and expose that interface through the same manifest process.

### Tool manifest

Every exposed tool has one canonical manifest:

| Field | Purpose |
| --- | --- |
| name and version | Stable call identity and migration control |
| description | Model-facing semantic purpose and non-purpose |
| input_schema | Complete argument contract |
| output_schema | Complete result contract |
| semantic_domain | Memory, conversation, people, web, media, coding, and so on |
| cardinality | One item, page, aggregate, or bounded stream |
| trusted_scope | Code-derived user, character, channel, workspace, or target scope |
| side_effect_class | read_only, proposal, reversible_write, or irreversible_write |
| approval_policy | None, brain approval, or explicit user approval |
| idempotency | Retry rule and key behavior |
| outcome_verification | How an uncertain side effect is checked |
| timeout_and_limits | Duration, calls, bytes, rows, and page bounds |
| provenance | Source identifiers, timestamps, and freshness semantics |
| executor | The deterministic implementation boundary |
| refusal_contract | Typed reasons the executor can decline |

Trusted scope is injected by runtime code from the brain request and
authorization context. The model cannot broaden a user, character, channel,
conversation, workspace, or network scope by placing new values in arguments.

### Initial leaf catalog

The target first catalog contains at least these semantic leaf tools:

| Tool | Operation |
| --- | --- |
| conversation_search | Search scoped conversation turns by semantic or lexical query |
| conversation_list | Read bounded turns around known matches or a time window |
| conversation_aggregate | Compute a bounded participant, topic, or event summary over selected turns |
| memory_search | Search durable scoped memories and their provenance |
| memory_read | Read selected memory records and supporting source references |
| person_resolve | Resolve mentioned people to scoped person identities |
| profile_read | Read allowed profile facts for resolved people |
| active_recall | Read active agreements, commitments, reminders, and open loops |
| current_context_read | Read current scene and conversation context owned by the brain |
| calendar_read | Read authorized calendar facts needed by the task |
| web_search | Discover public sources for a stated query |
| web_read | Fetch and extract one authorized public source |
| media_inspect | Inspect one supplied or authorized media object |
| calculate | Perform deterministic arithmetic or structured computation |
| transform_text | Perform bounded text transformation without external retrieval |
| coding_read | Inspect authorized repository files and diagnostics |
| coding_propose | Produce a bounded code-change proposal or patch artifact |
| accepted_task_status | Read the status of a brain-owned accepted task |
| skill | Load one curated, versioned skill body into the session |
| submit_resolution | Submit the terminal typed resolution |

The list grows as eligible Kazusa interfaces are identified. Adding a tool
does not require adding a graph branch.

### Leaf-tool rule

No live tool hides a task, RAG, complex, or web orchestration DAG.

A leaf tool may contain deterministic implementation details such as:

- database access through the canonical data facade;
- embedding lookup or lexical search;
- pagination, caching, and deduplication;
- one bounded domain model where the operation itself is atomic, such as
  inspecting a supplied image;
- source parsing and provenance extraction;
- permission and size enforcement.

A leaf tool does not:

- choose a different specialist;
- decide an open-ended sequence of other resolver tools;
- synthesize the final cross-source answer;
- decide whether Kazusa should respond;
- hide graph state, graph checkpoints, or graph fallback behavior.

Search choices that materially affect meaning are explicit arguments. For
example, conversation_search exposes semantic, lexical, hybrid, time, sender,
and channel constraints rather than internally routing among opaque RAG
branches.

### Standard tool result

Each executor returns a typed envelope equivalent to:

    {
      "call_id": "call_...",
      "tool": "conversation_search",
      "tool_version": "1",
      "status": "succeeded",
      "data": {},
      "evidence": [
        {
          "source_type": "conversation_turn",
          "source_id": "...",
          "observed_at": "...",
          "scope": "..."
        }
      ],
      "warnings": [],
      "next_page": null,
      "error": null
    }

The runtime validates this envelope before it returns to the model. Tool
results are evidence, not persona, stance, or final wording.

## Native agent loop

The conceptual loop is:

    create or resume durable session
    build authorized tool catalog
    assemble bounded model context

    while step budget remains:
        request one native call from the resolver model
        validate name, schema, scope, permission, and budget

        if call is submit_resolution:
            validate terminal result and evidence references
            persist terminal event
            return TaskResolutionResult to the brain

        persist tool/call before dispatch
        execute the leaf tool
        persist paired tool/result
        append bounded result context

        if foreground deadline is reached:
            checkpoint the same session
            return the existing deferred result to brain-owned promotion

    submit a typed budget-exhausted failure

The model can revise its search based on evidence. For example, a vague memory
query may lead to conversation_search, then conversation_list around two
matches, then person_resolve, then memory_read, before submit_resolution.

The loop has explicit caps for model steps, calls per tool, total tool calls,
wall time, model tokens, evidence bytes, and repeated equivalent calls.
Repetition protection reports the observed duplicate to the model once and
terminates after the configured bound.

## DeepSeek Harness evidence and session contract

This design was checked against the real recorded sessions in the supplied
DeepSeek Harness copy at C:\workspace\deepseek-harness, commit
b150a551b8d465e31e418e1b2eaf5e79bbb7d28e.

The principal traces are:

- examples/jsonrpc-agent/tests/snapshots/bash-tool/session.jsonl
- examples/acp-agent/tests/snapshots/skill-load/session.jsonl
- examples/headless-agent/tests/snapshots/advanced-toolchain/session.jsonl

Observed trace facts:

| Trace | Recorded behavior |
| --- | --- |
| bash-tool | deepseek-official/deepseek-v4-flash emits call_00_Ry17evSfTr0uJnHhg3X93070 in the assistant message; the same ID appears in tool/call and tool/result before the second model step |
| skill-load | The model calls skill with call_skill_load; the runtime returns the selected full skill body under that ID; the next model step continues after loading |
| advanced-toolchain | Successive steps select different tools from the request-header catalog, demonstrating that the catalog does not predetermine one fixed branch |

Commit-pinned upstream copies:

- https://github.com/deepseek-ai/deepseek-harness/blob/b150a551b8d465e31e418e1b2eaf5e79bbb7d28e/examples/jsonrpc-agent/tests/snapshots/bash-tool/session.jsonl
- https://github.com/deepseek-ai/deepseek-harness/blob/b150a551b8d465e31e418e1b2eaf5e79bbb7d28e/examples/acp-agent/tests/snapshots/skill-load/session.jsonl

The bash-tool trace records this order:

1. request/header containing the available tool contract;
2. streamed assistant reasoning and tool-call chunks;
3. an assembled assistant/message;
4. a durable tool/call with exact call ID, tool name, and raw arguments;
5. a tool/result paired by the same call ID;
6. step/end;
7. a later model step that produces the final response;
8. turn/end.

The skill-load trace uses the same native call/result sequence. The model calls
skill with a name, the runtime returns the complete skill body, and the next
step continues with the loaded instructions.

The architecture adopts those mechanics, with Kazusa-specific strengthening:

- the authorized tool catalog and its digest are captured at session start;
- a call is persisted before dispatch;
- every result is correlated by exact call ID;
- terminal output is a submit_resolution call rather than free-form prose;
- foreground/background control stays outside the loop;
- permissions and target scopes are code-bound;
- crash state distinguishes a call that was not started from a call whose
  outcome is unknown.

### Durable session events

The session store is append-only at the event boundary. It records:

- session/created;
- request/header with objective, trusted scope, model, limits, catalog digest,
  and skill-catalog digest;
- assistant/reasoning_chunk where protected diagnostics permit it;
- assistant/tool_call_chunk;
- assistant/message;
- tool/call;
- tool/result;
- step/end;
- session/checkpoint;
- session/promoted;
- session/resumed;
- resolution/submitted;
- session/failed;
- session/ended.

Reasoning is diagnostic data with restricted retention. It is not evidence,
does not enter TaskResolutionResult, and is not replayed as a trusted
instruction.

The database facade owns an agentic session header plus append-only events.
The existing task checkpoint stores the agentic session identifier and latest
committed revision. Existing accepted-task and background records retain their
scheduling ownership.

### Crash and uncertainty states

Recovery classifies a dispatched call as one of:

- tool_not_started;
- tool_outcome_unknown;
- tool_failed;
- tool_succeeded.

Read-only or manifest-declared idempotent calls may be retried under their
retry policy. A side-effecting call with an unknown outcome is never retried
blindly. The runtime uses outcome_verification, returns approval_required or
needs_user_input when verification cannot settle it, and preserves the
uncertain event for audit.

## Foreground and background execution

The current calling mechanism is preserved.

### Priority now

1. The action selector emits task_resolution_request with priority now.
2. The task-resolution service starts or resumes an agentic session.
3. The loop runs under the brain-owned inline deadline.
4. A completed session returns the existing resolution result.
5. If the deadline expires, the same durable session is checkpointed.
6. Existing deterministic promotion logic promotes that session and returns
   the current deferred outcome.
7. The background worker resumes the same session rather than restarting the
   search.

### Priority background

1. The action selector emits task_resolution_request with priority background.
2. The brain-owned service creates the accepted task and initial session
   checkpoint before any resolution tool executes.
3. Existing queue and worker mechanics run the agentic session.
4. Completion returns through the existing resume and delivery path.

The resolver sees an execution budget and checkpoint signal. It does not
select, override, or reinterpret priority.

## Context policy for a local model

The model receives only the context needed for the current decision:

- the brain-supplied objective and trusted scope;
- current tool manifests or a stable compact catalog;
- loaded skill instructions;
- recent native calls and normalized results;
- a compact evidence ledger with stable evidence IDs;
- the remaining deterministic budgets;
- the latest checkpoint summary.

Large conversation turns, memory bodies, pages, media, and code remain in tool
storage until explicitly read. Results use bounded pages and references.

Compaction is an atomic runtime operation. It produces:

- an untrusted concise session summary;
- a structured ledger of established facts and source IDs;
- unresolved questions;
- attempted queries and their outcomes;
- active limits and approvals.

The original events remain durable. The summary cannot grant permissions,
change trusted scope, invent evidence, or replace required source references.

## Skills

### Immediate phase: curated skill loading

Skills enhance the resolver's procedure without replacing tools.

The skill catalog is curated and versioned. The model calls skill by name when
the objective matches. The runtime returns the selected immutable skill body,
records its version, and adds it to the bounded context. A session captures a
skill-catalog digest so a resumed run uses the same versions unless an
explicit migration is approved.

A skill may teach:

- a reliable sequence for ambiguous memory recall;
- source-quality and cross-checking practice;
- repository-specific coding investigation;
- Chinese localization conventions;
- recovery from common tool errors.

A skill cannot:

- grant a permission or broaden trusted scope;
- introduce an undeclared tool;
- bypass validation, approval, or budgets;
- claim facts about the user, character, or world;
- decide foreground/background priority or visible wording.

### Next phase: experience-derived skill development

Past sessions become candidates for skill improvement through an offline,
auditable pipeline:

    completed sessions
        -> candidate mining
        -> SkillCandidate
        -> knowledge classification
        -> static validation
        -> original-session replay
        -> held-out replay and regression evaluation
        -> independent or human approval
        -> immutable catalog version
        -> future sessions

SkillCandidate contains:

- the recurring task pattern;
- proposed procedural instructions;
- supporting session and event IDs;
- measured success and failure cases;
- required tools and permissions;
- expected benefit and regression risks;
- proposed evaluation cases.

Before promotion, the pipeline classifies extracted material:

- reusable procedure belongs in a skill candidate;
- factual knowledge belongs in curated knowledge with provenance;
- user memory or preference belongs in memory through its owning policy;
- a promise or active commitment belongs in active recall;
- secrets, transient tool output, and private reasoning are rejected.

The agent may propose a SkillCandidate. It cannot silently activate one.
Promotion requires static checks, replay on the originating cases, held-out
cases, safety and permission regression checks, and the configured independent
or human approval. Activated versions are immutable and reversible by catalog
selection.

## Kazusa use cases

### 1. Recall a specific event from conversation

User request:

    Do you remember the night I called the blue comet our marker?

Possible resolver loop:

1. conversation_search with the distinctive phrase and a broad authorized
   time range;
2. conversation_list around the strongest matches;
3. person_resolve if pronouns or group identities are ambiguous;
4. profile_read only if relationship identity is relevant;
5. submit_resolution with the event, uncertainty, and cited turn IDs.

The model decides whether more digging is useful. Conversation retrieval
returns evidence; cognition decides how Kazusa emotionally interprets and
speaks about it.

### 2. Recall a durable preference

User request:

    Which local model did I say I preferred for casual conversation, and why?

Possible loop:

1. memory_search for local-model preference;
2. memory_read for the selected memory and provenance;
3. conversation_search against the source period if the reason is incomplete
   or two memories conflict;
4. submit_resolution with both the preference and provenance.

### 3. Check an active agreement

User request:

    What did we agree I would finish today?

Possible loop:

1. active_recall for current commitments;
2. calendar_read for authorized due-time context;
3. conversation_search if the commitment has conflicting revisions;
4. submit_resolution with status and source references.

This does not create, edit, or complete the commitment. Those mutations remain
with the owning brain action and permission path.

### 4. Answer a current technical question

User request:

    What changed in the retail RTX 5090 specification since launch?

Possible loop:

1. web_search restricted toward first-party sources;
2. web_read the launch and current official specification pages;
3. calculate if units or deltas need normalization;
4. submit_resolution with dated sources and discrepancies.

No public_research DAG selects branches behind the call. The resolver makes
each search/read decision.

### 5. Find who discussed two topics in a group

User request:

    Who in our group talked about RTX 6000 and GLM 5.2, and what did each say?

Possible loop:

1. conversation_search for each topic;
2. conversation_aggregate over the matched turn IDs;
3. person_resolve for display identities;
4. conversation_list around ambiguous statements;
5. submit_resolution with participant-level evidence.

### 6. Recover from a blocked public source

User request:

    Summarize what that Reddit thread concluded.

Possible loop:

1. web_read the supplied URL;
2. if blocked, web_search for the same title, authorized mirrors, or primary
   sources cited by the thread;
3. read useful alternatives;
4. submit resolved with explicit source differences, or needs_user_input when
   the requested content remains unavailable.

The resolver reports the gap rather than manufacturing thread content.

### 7. Investigate a coding task in the background

The brain selects task_resolution_request with priority background for a
repository investigation.

Possible resolver loop after the existing queue resumes it:

1. load a repository investigation skill;
2. coding_read the relevant contracts and files;
3. coding_read diagnostics or allowed tests;
4. coding_propose a bounded change artifact;
5. submit_resolution with the proposal, verification, and any required
   approval.

The brain owns background scheduling and later delivery. A proposal does not
authorize production modification.

### 8. Apply a curated procedure

User request:

    Localize this architecture note into natural Simplified Chinese.

Possible resolver loop:

1. skill loads the approved Chinese-localization procedure;
2. transform_text applies the bounded transformation;
3. submit_resolution returns the artifact and declared terminology choices.

The skill improves technique. transform_text remains the executable
capability.

## Failure behavior

| Condition | Owner and result |
| --- | --- |
| Unknown tool | Runtime rejects call; model receives typed contract error within retry cap |
| Invalid arguments | Runtime rejects before dispatch; bounded model regeneration |
| Scope escalation | Runtime rejects; recorded permission failure |
| Tool timeout | Typed tool_failed result; model may choose another eligible tool |
| Source unavailable | Model explores alternatives or submits unavailable/needs_user_input |
| Conflicting evidence | Model reads more sources or submits explicit uncertainty |
| Repeated equivalent calls | Runtime warns once and terminates at repetition bound |
| Inline deadline | Runtime checkpoints; brain-owned logic promotes same session |
| Context limit | Atomic compaction with durable source ledger |
| Process crash | Replay events; classify pending call outcome before retry |
| Side-effect outcome unknown | Verify through manifest or request approval/input |
| Malformed terminal result | Producing model receives bounded regeneration request |
| Step or cost budget exhausted | Typed failed result with evidence already gathered |

All model-authored contracts pass through the canonical LLM JSON parser where
the owning stage permits syntax repair. Deterministic repair may normalize
structure and declared bounds; it does not invent semantic values or override
model decisions. Invalid terminal semantics follow bounded regeneration and
then fail closed.

## Cutover and decommission

The implementation must use one canonical contract update across the
resolution layer.

### Before cutover

1. Approve an implementation plan derived from this architecture.
2. Inventory every existing resolution-facing interface.
3. Classify each interface as exposed or excluded.
4. Define and test all initial tool manifests.
5. Add durable session/event storage through the database facade.
6. Implement the native loop behind the existing task-resolution service.
7. Verify inline, direct-background, promotion, resume, and result projection.
8. Build replay tests from real Harness-shaped call/result sessions.
9. Drain active legacy DAG checkpoints; block cutover while any remain.

### Atomic cutover

1. Route task-resolution service execution to the agentic resolver.
2. Move all required leaf executors to canonical tool manifests.
3. Remove live task/RAG/complex/web DAG routing.
4. Remove the four facade adapters and their specialist-selection vocabulary.
5. Remove legacy graph checkpoint and resume code after the drain gate.
6. Update callers, tests, ICDs, tracing, and operational runbooks together.

### After cutover

There is one live resolution path. Rollback uses deployment/version rollback
and compatible durable agentic events; it does not keep a dormant legacy graph
path in production.

## Acceptance conditions

Architecture implementation is complete only when:

- the action selector and task_resolution_request caller contract is
  unchanged;
- foreground/background authority remains with the brain;
- inline deadline promotion resumes the same agentic session;
- every eligible resolution-facing interface is exposed or explicitly
  excluded;
- no live tool hides a resolver orchestration DAG;
- the model dynamically chooses successive native tool calls;
- tool/call is durable before dispatch and tool/result uses the same call ID;
- terminal output is a validated submit_resolution call;
- trusted scope and permissions cannot be expanded by model arguments;
- memory and conversation evidence retains source provenance;
- RAG evidence is never treated as persona or final stance;
- cognition and dialog continue to own character judgment and wording;
- skill loading is native, versioned, and replayable;
- experience-derived skills require evaluation and approval before activation;
- the fixed four-facade path and old graph checkpoints are absent after
  cutover;
- deterministic replay, crash recovery, permission, deadline, and
  background-resume tests pass.

## Architectural invariants

1. The brain selects the action; the resolver resolves the objective.
2. The brain owns foreground/background mode and delivery.
3. The model chooses tools; deterministic code enforces execution.
4. All eligible base-level resolution interfaces are tools by default.
5. Every exclusion is explicit and reviewable.
6. A tool is a semantic leaf, never a hidden orchestration graph.
7. Retrieval supplies evidence, not character judgment.
8. submit_resolution is the only normal semantic terminal.
9. Every dispatched call has a durable, exactly correlated result or a typed
   uncertain-outcome state.
10. Skills teach procedure; they do not grant capability or store personal
    facts.
11. Learned skill activation is an offline governed promotion, not an
    in-session self-modification.
12. The renewed resolver replaces the conflicting DAG and four-facade design
    in one resolution-layer cutover.
