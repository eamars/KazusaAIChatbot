# DSH Plan 2 Standard Runtime Integration

## Document control

| Field | Value |
|---|---|
| Status | Current accepted Plan 2 runtime boundary; Plan 3 cutover remains future/draft |
| Date | 2026-08-26 |
| Scope | DSH Standard capability-ready sidecar and Brain interaction bridge |
| Stable production boundary | `task_resolution_request` and existing accepted/background routing remain unchanged |
| Related architecture | `docs/architecture/agentic_resolver_architecture.md` |
| Decision owner | Kazusa architecture |
| DSH baseline | DeepSeek Harness `0.1.1-rc.2`, composed from the installed official base and Standard preset |

## Current Plan 2 boundary

Plan 2 is the accepted, capability-ready DSH Standard runtime. It is a
separately built Node sidecar plus a Python control plane and Brain-owned
interaction bridge. The process lifecycles are independent, but the sidecar
reports ready only when the Brain's authenticated interaction health is ready.
Plan 3 remains the future/draft production cutover: it does not yet route
`task_resolution_request`, accepted/background work, legacy resolvers or
coding callers through DSH, and it does not change their exhaust mapping.

The pinned contracts are RPC `kazusa.dsh-resolution-rpc.v2`, intake
`dsh_resolution_intake.v2`, profile `kazusa-resolver-standard-v2`, DSH
`0.1.1-rc.2`, and store epoch `dsh-sqlite-0.1.1-rc.2-standard-v2`. Session
data lives at
`<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/sessions.sqlite`; the framed semantic
worker owns the adjacent `semantic-outcomes.sqlite`.

The official DSH base and Standard preset are mounted by reference. Standard
native filesystem, shell, coding, jobs, tests, public web, approval, and
sandbox tools take name precedence. Kazusa adds exactly thirteen
storage-independent semantic tools: `kazusa_search_conversation_history`,
`kazusa_read_conversation_entries`,
`kazusa_summarize_conversation_participants`, `kazusa_search_memories`,
`kazusa_read_memories`, `kazusa_remember_information`,
`kazusa_revise_memory`, `kazusa_change_memory_lifecycle`,
`kazusa_find_people_by_name`, `kazusa_read_person_profiles`,
`kazusa_recall_active_context`, `kazusa_read_calendar_context`, and
`kazusa_inspect_attached_media`. `submit_resolution` is controller-owned and
is the sole model-owned terminal operation. Kazusa adds no coding/web wrapper,
command filter, DSH budget, sandbox overlay, or generic workflow tool.

For approval, question, or plan-review requests, an authenticated Brain
request becomes a targeted runtime-authored cognition observation plus
pending semantic context, not a user permission. P-stage owns the semantic
answer/reject/allow-once/relay decision. `relay_to_user` uses normal dialog
and adapter delivery; an exact reply resumes the same thread and segment, and
deterministic code atomically matches tool, executable arguments, workspace,
scope, and policy before consuming a one-shot grant. Checkpoint, restart,
replay, and transport-loss recovery remain durable control-plane concerns.

### Plan 2 startup and readiness

Configure the required DSH sidecar, RPC, absolute data/workspace/Python,
loopback Brain, shared-secret, tool-gateway, and six
`AGENTIC_RESOLVER_LLM_*` environment fields; the initial documented route is
qwen27b-5090 with 50,176 context tokens, 8,192 completion tokens, and
thinking enabled. Start Brain with its durable Mongo DSH interaction store
and cognition judge, build/start the sidecar separately, then require
authenticated `system.health=ready` with `route`, `standard`,
`semantic_worker`, `web`, and `brain` readiness. Brain
`GET /runtime/dsh/health` reports configured, durable-store, and
cognition-judge readiness. See the [HOWTO](../HOWTO.md#run-the-plan-2-dsh-standard-sidecar).

## 1. Executive decision (future Plan 3 cutover)

The following ownership and topology describe the future Plan 3 cutover; they
are not the current production routing boundary.

Kazusa SHOULD integrate DeepSeek Harness as the **agent runtime inside the existing brain-owned task-resolution boundary**. DSH does not become the chatbot framework and does not own cognition, dialog, memory policy, user delivery, foreground/background priority, approval UX, or background scheduling.

The integration boundary is:

```text
platform adapter
    -> Kazusa brain
    -> action selector
    -> task_resolution_request
    -> Kazusa Resolution Controller
         -> ResolutionThread
         -> DSH Integration Sidecar
              -> DSH agent/session/tool loop
              -> Kazusa semantic leaf tools
              -> bounded public web capability
              -> submit_resolution
         <- DSH exhaust / checkpoint
    <- TaskResolutionResultV1
    -> cognition
    -> dialog or another brain-owned action
```

The central architectural unit is a **ResolutionThread**, owned by Kazusa. A DSH session normally represents one durable resolution lineage inside that thread. It is **not** one session per incoming message, one session per user, one session per platform conversation, or one session per loosely classified topic.

A DSH session may survive many model/tool steps and multiple Kazusa turns when the brain determines that later input genuinely continues the same resolution goal. The DSH live Agent activation, however, is short-lived: it is materialized only while work is running, checkpointed, then disposed. The durable DSH event log remains available for cold resume.

This architecture introduces two explicit integration planes:

1. **Control plane** — create, resume, amend, checkpoint, cancel, inspect, and dispose a DSH activation.
2. **Data plane** — the typed Kazusa-to-DSH intake and the typed DSH-to-Kazusa exhaust.

The brain never consumes DSH assistant prose as an authoritative answer. A resolution becomes authoritative only when the DSH resolver invokes the Kazusa-defined `submit_resolution` terminal tool, or when deterministic runtime code returns a brain-owned non-terminal runtime outcome such as `deferred`.

## 2. Goals and non-goals

The goals and exclusions in this section describe the future Plan 3 cutover;
the current Plan 2 operational boundary is frozen above.

### 2.1 Goals

This integration MUST:

- preserve `task_resolution_request` as the stable brain-facing action;
- preserve brain ownership of cognition, dialog, delivery, priority, and scheduling;
- replace custom generic agent-loop work with DSH session, tool, persistence, and model-loop primitives;
- support iterative private RAG and online research without hiding another agentic graph behind a leaf tool;
- support durable foreground-to-background promotion without restarting the investigation;
- provide explicit session lifecycle rules for direct messages, multi-turn follow-ups, and group chats;
- prevent cross-audience or cross-scope evidence leakage when deciding whether a DSH session can be resumed;
- keep model-visible context narrow and resolver-specific;
- return a typed, prompt-safe evidence product to the brain rather than a second chatbot response;
- keep crash uncertainty, side-effect verification, and authorization deterministic.

### 2.2 Non-goals

This integration MUST NOT:

- make DSH responsible for deciding whether Kazusa should speak;
- give DSH ownership of Kazusa's persona, relationship stance, emotional judgment, or response wording;
- use a single long-lived DSH session as user memory;
- use DSH session history as a replacement for Kazusa conversation storage or durable memory;
- expose raw database clients, unrestricted shell, unrestricted filesystem, or platform credentials to the resolver profile;
- let DSH directly ask the end user for clarification or approval;
- let the DSH goal-round driver become an independent background scheduler;
- let DSH session IDs become public brain or platform identities;
- treat semantic topic similarity alone as sufficient reason to resume a session.

## 3. Terminology and ownership

### 3.1 Platform conversation

A Discord DM, group channel, web-chat conversation, Telegram group, or equivalent platform context. It can live for months or years. It is owned by Kazusa/platform adapters.

### 3.2 Brain/cognitive context

Kazusa-owned current scene, character state, relationship judgment, action goals, and conversation interpretation. This is not copied wholesale into DSH.

### 3.3 ResolutionThread

A Kazusa-owned durable identity for one coherent resolution goal lineage. It can span multiple DSH turns and, when safety requires rotation, more than one DSH session segment.

Examples:

- “Find the laptop model the user mentioned last month.”
- “Research whether this GPU topology supports P2P.”
- “Find a dinner venue for this group under the given constraints.”

### 3.4 DSH session segment

One DSH durable session log used by one ResolutionThread under a compatible audience, authorization scope, resolver profile, and tool-contract epoch. Most ResolutionThreads use exactly one segment. A new segment is created when the old model-visible history is no longer safe or compatible to reuse.

### 3.5 DSH live activation

The process-local DSH `Agent` materialized from a session by `ctx.agents.create(...)` or `ctx.agents.resume(...)`. A durable session can exist while no live activation exists.

### 3.6 Resolution turn

One waking DSH turn driven by an initial objective, a Kazusa continuation, or a runtime resume marker. A turn may contain many DSH steps/tool calls.

## 4. Ownership matrix

| Concern | Owner | Notes |
|---|---|---|
| User/platform identity | Kazusa/platform | Never inferred from DSH history |
| Platform conversation/channel | Kazusa/platform | May contribute to trusted scope |
| Audience/visibility policy | Kazusa | Critical resume boundary |
| Character/persona state | Kazusa brain | Not a DSH responsibility |
| Action selection | Kazusa brain | Determines whether resolution is invoked |
| Foreground/background priority | Kazusa brain/runtime | Priority and checkpoint timing remain outside the autonomous Standard loop |
| ResolutionThread identity | Kazusa Resolution Controller | Stable semantic task lineage |
| DSH session ID | DSH adapter under Kazusa control | Internal implementation identity |
| DSH event log | DSH | Durable source of agent/tool history |
| Resolver objective | Kazusa brain | DSH interprets but does not broaden it |
| Semantic tool selection | DSH resolver model | Within the authorized catalog |
| Tool authorization/scope | Kazusa deterministic runtime | Model cannot broaden scope in arguments |
| Tool execution | Kazusa leaf executor or bounded DSH provider | Must obey manifest/policy |
| Evidence ledger | Kazusa integration layer | DSH may reference only registered evidence IDs |
| Terminal resolution decision | DSH model through `submit_resolution` | Validated by deterministic runtime |
| `deferred` / promotion | Kazusa runtime | Never submitted by the model |
| Approval/user clarification UX | Kazusa brain/dialog | DSH returns typed requests only |
| Background queue/scheduler | Kazusa | DSH jobs/goal driver are not the product scheduler |
| Final visible reply | Kazusa cognition/dialog | DSH output is evidence, not final speech |

**Normative rule:** DSH owns the contents and replay mechanics of a resolver session. Kazusa owns the meaning, authorization, lifecycle, and product consequences of that session.

## 5. Target deployment topology

```text
+---------------------------------------------------------------+
|                         Kazusa Python                         |
|                                                               |
|  Brain -> Action Selector -> task_resolution_request          |
|                          |                                    |
|                          v                                    |
|                Resolution Controller                          |
|        - ResolutionThread store                               |
|        - lifecycle / leases                                   |
|        - priority / deadlines                                 |
|        - audience + scope policy                              |
|        - TaskResolutionResult mapping                         |
+--------------------------+------------------------------------+
                           | local authenticated RPC
                           v
+---------------------------------------------------------------+
|                Kazusa DSH Integration Sidecar                 |
|                        Node / TypeScript                       |
|                                                               |
|  pinned DSH composition                                       |
|  - agent/session/tool spine                                   |
|  - persistence + checkpoint policy                            |
|  - Kazusa resolver profile                                    |
|  - autonomous Standard multi-tool loop                       |
|  - tool proxy + evidence ledger                               |
|  - submit_resolution -> concludeTurn()                        |
|  - mounted Standard native web provider                       |
|                                                               |
|  NO: duplicate Kazusa shell/fs/coding/web wrappers,           |
|      Kazusa loop budget/sandbox/policy overlay                |
+------------------------+-------------------+------------------+
                         |                   |
                         v                   v
              Kazusa Semantic Gateway     DSH Standard native web
              conversation/memory/etc.    search provider
```

The sidecar SHOULD be a long-lived process capable of hosting many DSH sessions. Creating a new DSH session MUST NOT imply spawning a new sidecar process.

The production bridge SHOULD be a Kazusa-specific protocol over DSH's public agent/session services rather than exposing the stock Python SDK contract directly. The current SDK is useful for prototyping but its JSON-RPC wire has no per-session close or prompt-cancel methods; SDK-created agents remain live until process shutdown. The sidecar therefore needs direct lifecycle control through `ctx.agents` or an equivalent Kazusa-owned bridge.

## 6. DSH resolver profile

### 6.1 Profile objective

Plan 2 uses a thin Kazusa overlay on the installed official DSH base and
Standard preset:

```text
kazusa-resolver-standard-v2
```

It retains the Standard autonomous multi-tool loop and native capabilities for
filesystem, shell, coding, jobs, tests, public web, questions, approval
policy, and sandbox execution. Kazusa contributes the semantic gateway, Brain
interaction bridge, evidence ledger, and terminal-integrity validation.

### 6.2 Required components

The profile SHOULD include:

- DSH session service and agent loop;
- DSH session persistence, preferably a production-appropriate backend;
- DSH semantic checkpoint policy;
- the chosen LLM adapter/model route;
- tool registry and tool timeout policy;
- Kazusa semantic gateway and Brain interaction bridge;
- Kazusa semantic tool proxy;
- Kazusa evidence-ledger integration;
- `submit_resolution` terminal tool;
- mounted Standard native web provider if public research is allowed;
- context compaction only for genuinely long resolution sessions;
- observability hooks for session events and usage.

### 6.3 Explicit exclusions

The thin Kazusa overlay excludes duplicate or bypass capability paths:

- custom shell, filesystem, coding, or public-web wrappers;
- custom question or approval UX that bypasses the Brain interaction bridge;
- custom jobs, goals, scheduler, or generic workflow tools;
- command filters, sandbox overlays, or DSH budget overlays;
- alternate agent-loop or continuation policy.

The installed Standard preset remains autonomous and exposes its native
filesystem, shell, coding, jobs, tests, public web, questions, approval, and
sandbox capabilities under native DSH policy. The Brain remains the owner of
user-facing interaction and grant decisions. Native provider limitations are
returned as typed DSH outcomes rather than hidden by the semantic gateway.

### 6.4 Profile pinning

Every DSH session segment MUST record:

- exact DSH revision/release;
- `resolver_profile_version`;
- model route/model ID;
- tool-catalog digest;
- policy epoch;
- scope/audience fingerprints.

Because DSH is currently in developer preview and explicitly warns of compatibility-breaking changes, production MUST pin a tested revision. Resume across an incompatible profile or tool-contract epoch MUST rotate to a new DSH session segment rather than silently replaying under changed semantics.

## 7. Architectural decision: session granularity

### 7.1 Rejected mappings

The following mappings are rejected:

```text
one DSH session per incoming message      -> too fragmented
one DSH session per user                  -> context + privacy contamination
one DSH session per platform conversation -> unrelated task contamination
one DSH session per broad topic           -> ambiguous semantic boundary
```

### 7.2 Chosen mapping

The default mapping is:

```text
one ResolutionThread
    -> one DSH session segment
    -> zero or one live DSH Agent activation at a time
```

A ResolutionThread represents one coherent goal lineage. Later input MAY resume its segment if Kazusa determines that the new request genuinely continues that goal and the previous model-visible evidence remains admissible.

### 7.3 Continuation rule

A prior DSH session is resumable only if all of the following are true:

1. the brain supplies an explicit `goal_continuation_ref` or equivalent thread identity;
2. the requested work is genuinely a continuation/refinement of the same resolution goal;
3. the previous segment still exists and is not administratively expired;
4. the new audience and trusted execution scope are compatible with every item already admitted into model-visible history;
5. the resolver profile/tool-contract epoch is compatible;
6. reuse is materially useful; otherwise a clean session is preferred.

Topic similarity alone MUST NOT satisfy item 1 or 2.

### 7.4 Group chat rule

A group chat MUST NOT have one shared DSH session merely because messages share a channel.

Example:

```text
Group G123
  -> ResolutionThread R1: dinner venue
  -> ResolutionThread R2: RTX research
  -> ResolutionThread R3: locate hotel link
```

Several participants MAY contribute to one thread if they are modifying the same goal and the audience/scope policy remains compatible.

If a prior session contains evidence retrieved under a private or narrower audience scope, it MUST NOT be resumed into a group-visible context. The safe first implementation SHOULD require exact `audience_fingerprint` and `scope_fingerprint` equality for session reuse. A future compatibility checker MAY allow a change only when it proves that all prior admitted evidence remains authorized.

## 8. ResolutionThread and session-segment records

Kazusa SHOULD persist thread metadata independently of DSH storage.

### 8.1 ResolutionThreadRecordV2

```json
{
  "schema_version": "resolution_thread_store.v2",
  "resolution_thread_id": "res_...",
  "brain_conversation_ref": "conv_...",
  "root_goal_ref": "goal_...",
  "current_segment_id": "seg_...",
  "state": "active|waiting_user|waiting_approval|checkpointed|completed|failed|abandoned|expired",
  "priority": "now|background",
  "audience_fingerprint": "sha256:...",
  "scope_fingerprint": "sha256:...",
  "created_at": "...",
  "updated_at": "...",
  "last_terminal_status": null,
  "continuation_eligible_until": "..."
}
```

### 8.2 ResolverSessionSegmentV2

```json
{
  "schema_version": "resolver_session_segment.v2",
  "segment_id": "seg_...",
  "resolution_thread_id": "res_...",
  "dsh_session_id": "dsh_...",
  "resolver_profile_version": "kazusa-resolver-standard-v2",
  "dsh_revision": "<pinned revision>",
  "tool_catalog_digest": "sha256:...",
  "policy_epoch": "2026-08-26.1",
  "scope_fingerprint": "sha256:...",
  "audience_fingerprint": "sha256:...",
  "model_route": "resolver-model-a",
  "state": "cold|live|checkpointed|terminal|rotated|archived",
  "last_committed_seq": 173,
  "parent_segment_id": null,
  "rotation_reason": null,
  "created_at": "...",
  "last_used_at": "..."
}
```

A thread may contain more than one segment only when DSH history has to be deliberately severed. Typical rotation reasons are `scope_changed`, `audience_changed`, `profile_incompatible`, `tool_contract_incompatible`, `retention_expired`, or `administrative_reset`.

## 9. Control-plane interface

The Kazusa brain SHOULD NOT call DSH APIs directly. The Resolution Controller calls a stable Kazusa DSH sidecar API.

Recommended operations:

```text
resolution.open
resolution.continue
resolution.amend
resolution.request_checkpoint
resolution.cancel
resolution.inspect
resolution.dispose_activation
```

### 9.1 `resolution.open`

Creates a ResolutionThread/segment if required, materializes a DSH Agent, seeds runtime policy, and admits the initial intake.

### 9.2 `resolution.continue`

Cold-resumes or reuses the current segment and admits a Kazusa-approved continuation turn. Used after `needs_user_input`, `approval_required`, a related follow-up, or background promotion.

### 9.3 `resolution.amend`

Allows the brain to modify an in-flight goal without sending a raw platform message to DSH. The sidecar SHOULD translate a high-priority amendment into DSH steering at the next safe step boundary. Ordinary follow-ups SHOULD remain queued as subsequent turns.

### 9.4 `resolution.request_checkpoint`

Requests a cooperative checkpoint. It SHOULD prevent the next model step at the next safe boundary, flush the session, return a checkpoint, and leave the durable session resumable.

### 9.5 `resolution.cancel`

Stops active execution for an explicit brain-owned reason. Cancellation is not the normal foreground-to-background path; promotion prefers cooperative checkpointing.

### 9.6 `resolution.dispose_activation`

Disposes the process-local Agent handle after it is idle/checkpointed. It does not delete durable session history.

## 10. Intake interface: Kazusa brain to DSH

### 10.1 Design principle

DSH receives a **bounded resolution objective**, not the raw conversation transcript and not Kazusa's complete cognitive state.

The brain is responsible for turning the user/platform scene into a resolution request. DSH is responsible for evidence gathering and structured resolution within that objective.

The intake is divided into two classes:

- **runtime-only authority** — consumed by deterministic sidecar/tool code and never treated as model instructions;
- **model-visible resolution content** — the minimum information the resolver model needs to pursue the task.

### 10.2 Current standalone `DSHResolutionIntakeV2`

Plan 2 already implements this exact standalone intake. A future Plan 3
production routing cutover must consume this canonical contract or replace it
atomically under a separately approved plan; this document does not define a
parallel V1 shape.

```json
{
  "schema_version": "dsh_resolution_intake.v2",
  "mode": "start|continue",
  "request_id": "rrq_...",
  "operation_id": "operation_...",
  "operation_payload_digest": "sha256:...",
  "resolution_thread_id": "res_...",
  "segment_id": "seg_...",
  "brain_conversation_ref": "chat:...",
  "workspace_root": "C:/absolute/workspace",
  "route_digest": "sha256:...",
  "model_input": {
    "objective": "Determine ...",
    "facts": []
  },
  "semantic_tool_authority": {
    "catalog_digest": "sha256:...",
    "token": "opaque-runtime-token"
  },
  "interaction_authority": {
    "issuer": "dsh-sidecar",
    "scope_fingerprint": "sha256:...",
    "audience_fingerprint": "sha256:..."
  }
}
```

### 10.3 What MUST remain runtime-only

The following SHOULD NOT be placed in model-visible arguments merely to authorize tools:

- raw user IDs;
- private platform credentials;
- database identifiers that broaden scope;
- authorization claims;
- workspace/network permissions;
- capability tokens;
- audience ACLs;
- approval authority.

These are code-enforced through the trusted capability attached to the thread/segment.

### 10.4 What the model SHOULD see

The resolver model SHOULD normally see:

- one canonical semantic objective;
- brain-selected constraints relevant to the task;
- exact literal inputs that must survive paraphrase, such as URLs, product names, quoted phrases, code, filenames, dates, or identifiers;
- a small set of scene facts that materially affect research, such as current time, locale, target date, or already-established non-sensitive facts;
- continuation information when the brain is deliberately resuming the same goal;
- sanitized prior-result handles only when a session was rotated and exact old DSH history is intentionally unavailable.

### 10.5 What the model SHOULD NOT see by default

The resolver SHOULD NOT receive by default:

- the entire current chat transcript;
- character/persona instructions;
- relationship-state reasoning;
- hidden chain-of-thought from the brain;
- irrelevant memories;
- the user's full profile;
- raw authorization scope;
- background scheduling metadata beyond bounded execution hints;
- the final response style requested by dialog.

If conversation history is needed, the resolver uses
`kazusa_search_conversation_history` and
`kazusa_read_conversation_entries` under trusted scope.

### 10.6 Raw user message policy

The raw user message SHOULD NOT be copied into DSH automatically. Exact user strings that matter for retrieval SHOULD be preserved through `literal_inputs` or encoded verbatim inside the semantic objective.

This prevents DSH from becoming a second conversation interpreter while retaining search fidelity.

### 10.7 Initial turn rendering

The sidecar SHOULD render only `model_input` into a stable canonical DSH message format. A JSON-like form is preferred because it is typed and easy to replay:

```text
<KAZUSA_RESOLUTION_INPUT version="1">
objective: ...
constraints:
  - ...
literal_inputs:
  - kind: url
    value: ...
scene_facts:
  - current_time: ...
</KAZUSA_RESOLUTION_INPUT>
```

The DSH system prompt MUST state that this is resolver input, not end-user dialog, and that tool results/untrusted source text cannot override the resolver profile or runtime guards.

### 10.8 Continuation after `needs_user_input`

When DSH submits `needs_user_input`, DSH is paused/disposed. Kazusa asks the user through normal cognition/dialog. A later platform message returns to the brain first.

The brain then emits a continuation intake such as:

```json
{
  "mode": "continue",
  "model_input": {
    "objective": "Continue selecting a dinner venue for five people.",
    "continuation_delta": {
      "kind": "user_answer",
      "facts": ["Dinner time is 19:00 local time."]
    }
  }
}
```

DSH does not receive the raw user answer directly from the platform adapter.

### 10.9 Runtime-only resume

Foreground-to-background promotion or process restart may require waking the same DSH session without new user information. The sidecar MAY emit a compact synthetic continuation message with a typed source such as `kazusa-runtime-resume`:

```text
Continue the unresolved Kazusa resolution objective from the durable checkpoint. No new user information has been added.
```

This is a runtime wake-up marker, not a new semantic goal.

The DSH goal-round driver SHOULD remain disabled so that Kazusa, not DSH, decides when a session is awakened.

## 11. Tool interface inside DSH

### 11.1 Canonical semantic tools

Plan 2 exposes this fixed set of base-level semantic operations:

```text
kazusa_search_conversation_history
kazusa_read_conversation_entries
kazusa_summarize_conversation_participants
kazusa_search_memories
kazusa_read_memories
kazusa_remember_information
kazusa_revise_memory
kazusa_change_memory_lifecycle
kazusa_find_people_by_name
kazusa_read_person_profiles
kazusa_recall_active_context
kazusa_read_calendar_context
kazusa_inspect_attached_media
submit_resolution
```

The exact catalog is capability-complete but authorized per thread. Native
DSH Standard tools are mounted separately and take name precedence; Kazusa
does not wrap coding or public web.

### 11.2 Tool proxy

The model calls canonical names. The DSH sidecar proxy attaches runtime authority invisibly:

```text
model
  -> kazusa_search_memories(query="summer holiday")

sidecar
  -> Kazusa Tool Gateway
     Authorization: ResolverCapability <opaque>
     body: {"query": "summer holiday"}
```

The model cannot supply a different user/channel/workspace to broaden access.

### 11.3 StandardToolResult

Each tool SHOULD return a validated envelope:

```json
{
  "call_id": "call_...",
  "tool": "kazusa_search_conversation_history",
  "tool_version": "1",
  "status": "succeeded",
  "data": {},
  "evidence": [
    {
      "evidence_id": "ev_...",
      "source_type": "conversation_turn",
      "source_id": "...",
      "observed_at": "...",
      "scope_fingerprint": "sha256:..."
    }
  ],
  "warnings": [],
  "next_page": null,
  "error": null
}
```

The evidence ledger registers `evidence_id` values before results become usable by `submit_resolution`.

### 11.4 Autonomous multi-tool turns and singleton terminal

DSH Standard may emit and execute multiple native and Kazusa semantic tool
calls in one autonomous turn, within the native DSH policy and the
deterministic authority, frame, and evidence bounds. Kazusa validates each
call/result pair and preserves its event and evidence correlation.

`submit_resolution` is the only terminal call. A turn that contains it must
contain that sole call and no native or semantic sibling calls. The terminal
result and every cited evidence receipt are validated before
`ToolExecution.concludeTurn()` commits the terminal exhaust.

If a call, result, or terminal structure is invalid, the boundary returns a
typed structural rejection to the autonomous DSH loop. DSH decides whether
and how to continue under its own loop policy; Kazusa does not silently
execute a subset or decide whether the loop continues. Runtime failure remains
fail-closed when the session cannot continue safely.

## 12. Exhaust interface: DSH to Kazusa brain

### 12.1 Exhaust principle

The brain consumes a **structured resolution product**, not DSH assistant prose and not DSH private reasoning.

There are three top-level run outcomes:

```text
terminal      -> validated submit_resolution
checkpointed  -> brain-owned deferred/promotion path
runtime_fault -> bridge/runtime failure without a valid terminal result
```

### 12.2 `submit_resolution` schema

Recommended model-facing terminal call:

```json
{
  "status": "resolved|partial|needs_user_input|approval_required|unavailable|failed",
  "summary": "Prompt-safe factual summary for Kazusa cognition.",
  "findings": [
    {
      "finding_id": "f_1",
      "claim": "...",
      "confidence": "high|medium|low",
      "evidence_ids": ["ev_1", "ev_2"]
    }
  ],
  "completed_subgoals": ["..."],
  "remaining_needs": ["..."],
  "clarification_request": null,
  "approval_request": null,
  "artifact_refs": [],
  "warnings": []
}
```

For `needs_user_input`, `clarification_request` is required. For `approval_required`, `approval_request` is required. The terminal tool MUST validate all cited evidence IDs against the current thread's evidence ledger.

On successful validation, the tool calls DSH `ToolExecution.concludeTurn()`. This marks the tool result as terminal for the current DSH turn. Free-form assistant text does not terminate resolution.

`needs_user_input` and `approval_required` remain valid terminal statuses in
the schema. Live Standard native question and approval hooks normally use the
authenticated Brain interaction checkpoint path, so they can pause and
resume the autonomous DSH loop without requiring a terminal status.

### 12.3 `clarification_request` field

```json
{
  "question_id": "q_...",
  "reason": "A specific missing fact prevents safe continuation.",
  "question": "What time should the dinner booking target?",
  "answer_type": "free_text|single_choice|multi_choice|boolean|date_time",
  "choices": [],
  "required": true
}
```

Kazusa cognition/dialog decides whether and how to ask it.

### 12.4 `approval_request` field

```json
{
  "approval_request_id": "appr_...",
  "action_summary": "Create calendar event ...",
  "tool": "calendar_write",
  "side_effect_class": "reversible_write|irreversible_write",
  "reason": "Explicit user approval required by manifest policy.",
  "proposal_ref": "proposal_..."
}
```

The model SHOULD NOT receive authority merely because it emitted this request. Kazusa obtains approval and later resumes the thread with an approval result and a fresh capability decision.

### 12.5 Current standalone `DSHResolutionExhaustV2`

The sidecar maps the terminal tool and runtime/session metadata into the
canonical V2 exhaust. Its kinds are `terminal`, `checkpointed`,
`runtime_fault`, and `canceled`:

```json
{
  "kind": "terminal",
  "terminal": {
    "status": "resolved",
    "summary": "...",
    "findings": [],
    "completed_subgoals": [],
    "remaining_needs": [],
    "clarification_request": null,
    "approval_request": null,
    "artifact_refs": [],
    "warnings": []
  },
  "evidence": [],
  "identity": {
    "resolution_thread_id": "res_...",
    "segment_id": "seg_...",
    "scope_fingerprint": "sha256:...",
    "audience_fingerprint": "sha256:...",
    "policy_epoch": "dsh-standard-policy-v2"
  },
  "usage": {},
  "last_committed_seq": 173
}
```

Plan 2 returns this standalone exhaust directly to its current callers. The
future Plan 3 draft would project it into the existing brain-facing
`TaskResolutionResultV1`; that production routing and projection are not part
of the current Plan 2 runtime.

### 12.6 Checkpointed/deferred exhaust

`deferred` is never a `submit_resolution` status. When a foreground deadline triggers promotion, the sidecar returns:

```json
{
  "kind": "checkpointed",
  "checkpoint": {
    "reason": "foreground_deadline",
    "dsh_session_id": "dsh_...",
    "last_committed_seq": 173,
    "segment_id": "seg_...",
    "scope_fingerprint": "sha256:...",
    "tool_catalog_digest": "sha256:..."
  },
  "identity": {
    "resolution_thread_id": "res_...",
    "segment_id": "seg_..."
  },
  "last_committed_seq": 173
}
```

Kazusa maps this to its existing runtime-owned deferred result and promotion mechanics.

### 12.7 Diagnostics are a separate channel

The following MUST NOT be folded into evidence or treated as brain facts:

- DSH chain-of-thought/reasoning streams;
- intermediate assistant prose;
- retry narration;
- uncommitted provider chunks;
- tool UI presentation metadata unrelated to provenance.

These may be retained in restricted diagnostics subject to policy, but cognition consumes only validated exhaust fields and cited evidence.

## 13. Session lifecycle

### 13.1 State model

Kazusa owns the product lifecycle; DSH owns activation/session mechanics.

```text
                   +------------------+
                   |       NEW        |
                   +--------+---------+
                            |
                            v
                 +----------+-----------+
                 |   ACTIVE_FOREGROUND  |
                 +----+----------+------+
                      |          |
            terminal  |          | checkpoint
                      v          v
        +-------------+--+    +--+----------------+
        | WAITING /      |    | CHECKPOINTED_BG  |
        | COMPLETED/etc. |    +---------+---------+
        +-------+--------+              |
                | continuation          | worker resume
                v                       v
             ACTIVE <------------- ACTIVE_BACKGROUND
                |
                v
        terminal / waiting / failed
```

A completed/waiting/checkpointed thread normally has **no live DSH Agent**. Only durable session data remains.

### 13.2 Create

For a new ResolutionThread:

1. Kazusa creates the thread record and trusted scope.
2. The sidecar allocates a DSH session segment and records profile/catalog fingerprints.
3. For direct background priority, Kazusa creates the accepted task and an initial durable checkpoint before any external resolution tool executes.
4. The sidecar creates the DSH Agent and installs per-agent policy/tool scope.
5. The initial intake is admitted as the first waking turn.
6. DSH executes until terminal, checkpoint, cancellation, or failure.

### 13.3 Normal terminal completion

When `submit_resolution` succeeds:

1. `concludeTurn()` marks the terminal tool result.
2. The sidecar waits for the turn to become idle and flushes persistence.
3. The sidecar returns the typed exhaust.
4. Kazusa updates the ResolutionThread state.
5. The live DSH Agent handle is disposed.
6. Durable session history is retained for continuation/audit according to retention policy.

`resolved` does not mean the durable DSH session is immediately deleted. It means there is no active execution.

### 13.4 `needs_user_input`

1. DSH submits the typed clarification request.
2. Sidecar flushes and disposes the activation.
3. Thread state becomes `waiting_user`.
4. Kazusa decides whether/how to ask the user.
5. User response re-enters normal brain processing.
6. If the brain confirms same-goal continuation and scope compatibility, the sidecar cold-resumes the same DSH session and admits a continuation intake.

### 13.5 `approval_required`

Same as `needs_user_input`, except the thread waits for brain/user approval. Approval MUST be revalidated at resume time and converted into runtime capability; it is not merely injected as trusted model text.

### 13.6 Related follow-up after `resolved`

A completed session MAY be reopened only when the brain deliberately links the new request through `goal_continuation_ref` and the reuse gate passes.

Example:

```text
R100: "What's the weather at Mt Hutt tomorrow?" -> resolved
next user turn: "And Thursday?"
brain: same goal lineage -> resume R100
```

A later unrelated request creates a new ResolutionThread even if it comes from the same user/conversation.

### 13.7 Foreground-to-background promotion

Promotion SHOULD be cooperative first and hard-abort only as a fallback.

**Soft deadline path:**

1. Kazusa sets `promotion_requested`.
2. Current model/tool step is allowed to finish.
3. At the next DSH `agent/pre-step` safe boundary, Kazusa policy rejects entry into another model step.
4. The sidecar flushes the DSH event log.
5. Sidecar returns `kind=checkpointed`.
6. Kazusa deterministic promotion logic enqueues the existing thread.
7. Live Agent handle is disposed.
8. Background worker cold-resumes the same DSH session and admits a compact runtime-resume wake-up message.

**Hard deadline fallback:**

If the sidecar cannot reach a safe boundary before the hard deadline, Kazusa may abort the active turn. After flush/reload, any incomplete tool dispatch is classified through DSH durable repair plus the Kazusa tool manifest. Side-effecting operations with uncertain outcome are never blindly retried.

### 13.8 Direct background priority

1. Brain selects `priority=background`.
2. Kazusa creates accepted-task record and ResolutionThread.
3. Initial DSH session metadata/checkpoint is made durable.
4. Background worker materializes the DSH Agent and admits the initial intake.
5. Completion returns through the existing brain resume/delivery path.

DSH does not choose to become background work.

### 13.9 Cold resume after process restart

1. Kazusa loads ResolutionThread/segment metadata.
2. Sidecar verifies pinned DSH/profile/catalog compatibility.
3. `ctx.agents.resume({ resumeSessionId })` reconstructs the DSH session.
4. DSH persistence restores event-derived model history.
5. DSH crash repair closes any interrupted call boundary into `TOOL_NOT_STARTED` or `TOOL_OUTCOME_UNKNOWN` semantics.
6. Kazusa manifest policy decides whether a tool may be retried, verified, or escalated.
7. A synthetic runtime-resume turn wakes the resolver if work is still pending.

### 13.10 Session rotation

A ResolutionThread gets a **new DSH session segment** rather than resuming old history when:

- audience changes in a way that could expose old evidence;
- trusted scope changes and prior evidence admissibility cannot be proven;
- the session's resolver profile/tool epoch is incompatible with the deployed runtime;
- retention has expired;
- administrative reset is requested;
- the event log is corrupt or cannot be safely migrated.

A rotated segment MAY receive a sanitized handoff containing only re-authorized `TaskResolutionResult` summaries and evidence handles. It MUST NOT inherit the unsafe old DSH transcript.

### 13.11 Retention and expiry

Three lifetimes are distinct:

1. **live activation lifetime** — normally seconds/minutes; dispose when idle and no immediate continuation is expected;
2. **continuation eligibility** — a configurable product window during which cold resume is allowed if the brain links the goal;
3. **durable audit/session retention** — longer, policy-driven storage lifetime.

The architecture SHOULD NOT depend on DSH's current built-in session listing/deletion behavior for large-scale retention. Current DSH persistence documentation notes no deletion/retention API and unpaginated listing. Kazusa SHOULD keep its own indexed thread metadata and either use a custom persistence backend or perform archival/deletion as controlled backend maintenance.

## 14. Concurrency and live amendments

### 14.1 Single execution owner

A DSH session segment MUST have at most one live Agent activation and one Kazusa execution lease at a time.

Kazusa SHOULD use a compare-and-set revision/lease on the ResolutionThread/segment before create/resume. Duplicate workers must fail closed rather than run the same side-effect-capable session concurrently.

### 14.2 New user message while resolution is running

The platform message always enters Kazusa brain first. The brain may decide:

- unrelated -> create another ResolutionThread;
- ordinary same-goal follow-up -> queue a continuation turn;
- urgent correction to current objective -> issue `resolution.amend`, translated to DSH steering at the next safe step boundary;
- cancellation -> cancel the thread;
- priority change -> update Kazusa scheduling and request checkpoint if necessary.

Raw platform messages are never inserted directly into the DSH inbox.

## 15. Persistence, checkpoints, and crash uncertainty

DSH's persisted unit is the append-only `SessionEvent`; message history is derived from that log. The sidecar SHOULD compose DSH's checkpoint policy so state is flushed before model requests, before top-level tool bodies that may produce side effects, and at pre-step boundaries.

Kazusa retains its stronger semantic classifications:

```text
tool_not_started
tool_outcome_unknown
tool_failed
tool_succeeded
```

The DSH persistence layer now repairs interrupted histories into `TOOL_NOT_STARTED`/`TOOL_OUTCOME_UNKNOWN` model-visible states, which aligns well with Kazusa's intended crash model. However, the **meaning** of safe retry remains a Kazusa manifest concern.

Rules:

- read-only or explicitly idempotent calls may be retried according to manifest policy;
- a write with unknown outcome is never blindly repeated;
- `outcome_verification` is attempted first;
- if verification cannot settle the result, DSH must return `approval_required`, `needs_user_input`, or an explicit partial/failure outcome;
- the uncertain event remains auditable.

## 16. Evidence and provenance

### 16.1 Evidence ledger

Every evidence-producing tool result SHOULD register stable evidence records outside free-form DSH prose.

Recommended fields:

```json
{
  "evidence_id": "ev_...",
  "resolution_thread_id": "res_...",
  "segment_id": "seg_...",
  "source_type": "conversation_turn|memory|web|calendar|file|other",
  "source_id": "...",
  "source_uri": null,
  "retrieved_at": "...",
  "observed_at": "...",
  "scope_fingerprint": "sha256:...",
  "content_digest": "sha256:...",
  "excerpt": "bounded prompt-safe excerpt",
  "metadata": {}
}
```

### 16.2 Evidence is not cognition

Evidence tools return facts, excerpts, identifiers, timestamps, and uncertainty. They do not return Kazusa's persona stance or final wording.

### 16.3 Compaction

DSH compaction MAY be enabled for unusually long resolver sessions. It is a context-management mechanism, not a memory subsystem.

Compaction MUST NOT invalidate evidence IDs or erase the external evidence ledger. `submit_resolution` validates against the ledger rather than relying on whether the full source text remains in the current compacted context.

## 17. Web and RAG boundary

### 17.1 Private RAG

Kazusa retains ownership of conversation/memory storage and deterministic retrieval executors. DSH owns the **choice and sequence** of semantic RAG operations.

Example using the current Plan 2 semantic names:

```text
kazusa_search_conversation_history
    -> kazusa_read_conversation_entries
    -> kazusa_find_people_by_name
    -> kazusa_search_memories
    -> submit_resolution
```

A leaf may perform embedding search, lexical search, pagination, filtering, or provenance extraction internally. A leaf MUST NOT hide another open-ended agentic router/DAG that selects other resolver tools or synthesizes the whole cross-source answer.

### 17.2 Public web

DSH Standard's native web provider is the public discovery seam. Kazusa does
not wrap it as a semantic gateway tool; native results still remain subject
to the Standard policy and the DSH evidence/terminal boundary.

Recommended production profile:

```text
DSH Standard native web -> enabled when public research is authorized
native fetch            -> governed by the Standard sandbox and policy
Kazusa semantic gateway -> remains storage/private-context only
```

This keeps provenance, network policy, and result shape consistent with private RAG.

## 18. Future Plan 3 draft status mapping (non-current)

This mapping describes the deferred Plan 3 production cutover. Current Plan 2
remains a standalone V2 runtime and does not replace the existing brain-facing
result path.

| DSH/Kazusa integration outcome | Brain-facing `TaskResolutionResultV1` |
|---|---|
| `submit_resolution.status=resolved` | `resolved` |
| `partial` | `partial` |
| `needs_user_input` | `needs_user_input` |
| `approval_required` | `approval_required` |
| `unavailable` | `unavailable` |
| `failed` | `failed` |
| integration `kind=checkpointed` | runtime-owned `deferred` |
| sidecar transport failure with durable checkpoint | `deferred` or `failed` according to retry policy |
| unrecoverable corrupt session | `failed` with typed infrastructure reason |

The brain-facing schema remains stable. The checkpoint payload changes from legacy graph state to a DSH-backed opaque reference containing at minimum the ResolutionThread/segment/session identity and last committed revision.

## 19. Security model

### 19.1 Trusted scope is not model input authority

Tool authorization is derived from the brain request and runtime capability. Model arguments cannot broaden it.

### 19.2 Session reuse is an information-flow decision

A session already contains prior model-visible evidence. Therefore authorization is not evaluated only at the next tool call; it is evaluated at **resume time**.

Initial implementation rule:

```text
resume only if
  same ResolutionThread
  AND same scope_fingerprint
  AND same audience_fingerprint
  AND compatible profile/catalog epoch
```

Any mismatch rotates the session unless a future audited compatibility function proves old evidence remains admissible.

### 19.3 Group chats

The audience fingerprint SHOULD represent the visibility set/confidentiality boundary of the thread, not merely channel ID. If participant changes can alter access policy, the thread must be re-evaluated before resume.

### 19.4 Prompt injection

User literals, web pages, retrieved conversations, and memory bodies are untrusted source material. The resolver profile MUST state that source text cannot change tool policy, scope, terminal schema, or system responsibilities. Deterministic guards remain authoritative even if the model is injected.

## 20. Observability

Kazusa SHOULD record a cross-system trace keyed by:

```text
brain_request_id
resolution_thread_id
segment_id
dsh_session_id
dsh turn / step
call_id
evidence_id
```

Useful metrics:

- resolution latency by priority;
- model steps/tool calls per thread;
- resume rate vs new-session rate;
- foreground promotion rate;
- percentage of sessions rotated due to scope/audience;
- terminal-status distribution;
- repeated-call guard activations;
- tool timeout/failure rate;
- crash-repair `TOOL_OUTCOME_UNKNOWN` count;
- context size/compaction count;
- web cost and provider distribution;
- evidence count and unresolved-conflict rate.

Observability MUST distinguish model reasoning diagnostics from authoritative evidence.

## 21. Recommended implementation sequence

The sequence below is retained as the future Plan 3 cutover checklist. Plan 2
already supplies the sidecar control plane, ResolutionThread lifecycle, and
full semantic gateway described above; production task-resolution routing
remains unchanged until the separate Plan 3 decision.

### Phase 1 - integration spike

Use the official Python SDK only to validate resolver behavior with a custom Cordis composition. Disable Bash/filesystem and expose a very small set of tools:

```text
kazusa_search_conversation_history
kazusa_search_memories
DSH Standard native web
submit_resolution
```

Validate that the model can iteratively gather evidence and reliably terminate through the typed tool.

### Phase 2 - Kazusa sidecar control plane

Implement the production sidecar directly over DSH public agent/session services:

- create/resume agent;
- event subscription;
- checkpoint/flush;
- per-session dispose;
- cancel;
- stable RPC contract to Python.

Do not let the stock SDK wire become the long-term product boundary while it lacks per-session close/cancel.

### Phase 3 - ResolutionThread lifecycle

Implement:

- thread and segment metadata;
- reuse gate;
- group-chat audience fingerprints;
- cold resume;
- scope/profile rotation;
- live activation leases;
- retention policy.

### Phase 4 - full semantic tool bridge

Plan 2 exposes the capability-complete Kazusa leaf catalog with manifests,
trusted scopes, evidence ledger, idempotency, and outcome verification.

### Phase 5 - foreground/background cutover

Move the existing brain-owned inline/deferred/background lifecycle to the same DSH session:

- soft checkpoint boundary;
- hard abort fallback;
- worker cold resume;
- accepted-task delivery.

### Phase 6 - hardening and fault injection

Test:

- multiple model tool calls in one autonomous turn;
- forged scope/tool arguments;
- forged evidence IDs;
- group-chat audience change;
- session resume under incompatible profile;
- process crash after `tool/call` but before `tool/result`;
- uncertain irreversible write;
- foreground deadline while a tool is running;
- duplicate background worker lease;
- DSH restart and cold resume;
- context compaction followed by valid evidence submission;
- attempts to web-fetch localhost/private addresses if fetch is enabled.

## 22. Architectural invariants

The following invariants are release blockers:

1. The brain is always the caller of task resolution.
2. DSH never receives raw authority to broaden trusted scope.
3. One DSH session is not a user, conversation, or topic memory.
4. A DSH session normally maps to one ResolutionThread lineage.
5. A scope/audience incompatibility rotates the DSH session instead of reusing unsafe history.
6. At most one live activation/lease exists per DSH session segment.
7. A new platform message always returns through Kazusa brain before it can amend/resume DSH.
8. Assistant prose is never a terminal runtime signal.
9. `submit_resolution` is the only model-owned terminal operation.
10. `deferred` is runtime-owned.
11. Foreground promotion resumes the same durable session when safe; it does not restart research.
12. Side-effecting calls with unknown outcomes are never blindly retried.
13. Evidence IDs must be registered and authorized before terminal submission.
14. Final user-visible wording remains brain/dialog owned.
15. The DSH deployment revision/profile is pinned and compatibility checked on resume.

## 23. Final proposed boundary

The resulting architecture can be summarized as:

```text
Kazusa Cognitive Architecture
    |
    +-- cognition / character / dialog
    +-- action selection
    +-- task scheduling and delivery
    +-- memory ownership and authorization
    |
    +-- Resolution Controller
          |
          +-- ResolutionThread (Kazusa identity)
          |     |
          |     +-- DSH Session Segment 0
          |     +-- DSH Session Segment 1  [only after safe rotation]
          |
          +-- Control plane
          |     create / resume / amend / checkpoint / cancel / dispose
          |
          +-- Data plane
                Intake: bounded semantic objective + literals + scene facts
                Exhaust: structured findings + evidence + typed terminal needs
                         |
                         v
                DeepSeek Harness
                  - durable event-sourced session
                  - native agent/tool loop
                  - semantic tool choice
                  - web/RAG orchestration
                  - submit_resolution
```

This gives Kazusa the benefit of a reusable agent harness while keeping the architectural property that matters most: **DSH resolves bounded evidence tasks; Kazusa remains the mind and product authority.**

## 24. References reviewed for this decision

- **[KAZ-AR]** Kazusa, *Agentic Resolver Target Architecture*: https://github.com/eamars/KazusaAIChatbot/blob/main/docs/architecture/agentic_resolver_architecture.md
- **[DSH-README]** DeepSeek Harness README / developer preview warning: https://github.com/deepseek-ai/deepseek-harness/blob/master/README.md
- **[DSH-ARCH]** DeepSeek Harness architecture and plugin extension points: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md
- **[DSH-CORE]** Agent/session APIs, `ctx.agents`, `followup`, `steer`, `inject`, lifecycle hooks: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/core.md
- **[DSH-SESSION]** Session event model and derived history: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md
- **[DSH-PERSIST]** Durable persistence, resume, crash repair: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.md
- **[DSH-PERSIST-PKG]** Persistence model experience and current retention/list limitations: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/session/session-persistence/README.md
- **[DSH-CHECKPOINT]** Semantic checkpoint policy: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/session/session-checkpoint-policy/README.md
- **[DSH-AGENT-LOOP]** Programmatic `ctx.agents.create` / `ctx.agents.resume`: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/README.md
- **[DSH-TOOLS]** Tool execution and `ToolExecution.concludeTurn()`: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.md
- **[DSH-INJECT]** Implemented decision separating `inject()` from turn execution: https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-24-separate-context-injection-from-turn-execution.md
- **[DSH-GOAL]** Same-session persisted goal domain: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/goal/README.md
- **[DSH-WEB]** Provider-neutral web seam: https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/web.md
- **[DSH-WEB-FETCH]** Deferred SSRF protection decision: https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-06-24-web-capability-seam.md
- **[DSH-SDK]** Python SDK: https://github.com/deepseek-ai/deepseek-harness/blob/master/python/sdk/README.md
- **[DSH-SDK-WIRE]** Current SDK JSON-RPC lifecycle limitations: https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/sdk/server/README.md
## Historical Plan 1 boundary (superseded by Plan 2)

The first prototype used an independent authenticated loopback sidecar and a
pre-V2 RPC, profile, intake, and store vocabulary. That prototype was a
staged control-plane seam without semantic gateway tools or a Brain
interaction edge. It is retained here only as historical context; the current
operational contract is the Plan 2 boundary in this document and the [V2
control-plane README](../../src/agentic_resolver/README.md).
