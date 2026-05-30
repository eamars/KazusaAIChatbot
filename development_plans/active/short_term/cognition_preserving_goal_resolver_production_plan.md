# Cognition-Preserving Goal Resolver Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a production goal resolver as bounded recurrence around Kazusa's existing L1 -> L2 -> L2d cognition core, with demand-driven evidence/tool observations and no bypass around memory-driven personality cognition.

**Architecture:** The baseline persona turn becomes a one-cycle resolver. Complex turns repeat the preserved cognition subgraph after bounded capability observations. RAG, web/current facts, human clarification, approval preparation, and self-resolution are selected by L2d as cognition-owned capabilities; deterministic code only validates, executes, limits, records, and routes observations back into the next cognition cycle.

**Tech Stack:** Python, LangGraph, existing Kazusa cognition nodes, existing RAG2 supervisor, action-spec contracts, FastAPI brain service, MongoDB-backed conversation progress and action-attempt audit, pytest, real LLM validation through the existing debug workflow.

---

## Plan Metadata

- Plan class: large production architecture implementation
- Status: in_progress
- Branch: `resolver-goal-poc`
- Owner: Codex
- Created: 2026-05-30
- Related superseded POC: `development_plans/archive/superseded/goal_resolver_poc_plan.md`
- Related references:
  - `development_plans/reference/designs/cognition_contracts_design.md`
  - `development_plans/reference/designs/cognition_core_evolution_progression.md`
  - `src/kazusa_ai_chatbot/nodes/README.md`
  - `src/kazusa_ai_chatbot/action_spec/README.md`
  - `src/kazusa_ai_chatbot/brain_service/README.md`

## Non-Negotiable Architecture Decisions

Kazusa is a cognition core system, not a generic assistant shell, coding
harness, or OpenClaw-style tool runner. This plan must preserve that boundary.

Every semantic thinking step must pass through the existing cognition stack:

```text
L1 subconscious appraisal
  -> L2 conscious interpretation, boundary, judgment, and social context
  -> L2d action/capability/surface selection
```

The resolver is not an external planner/verifier. It is a deterministic
recurrence controller around the cognition core:

```text
decontextualizer
  -> cognition cycle 1: L1 -> L2 -> L2d
  -> selected capability observation, if any
  -> cognition cycle 2: L1 -> L2 -> L2d
  -> repeat within caps
  -> selected L3 surface or private/no-response finalization
```

RAG becomes demand-driven evidence selected by cognition. The first cognition
cycle still receives the always-present personality substrate:

- character profile and boundary profile;
- current user profile and relationship summary;
- recent interaction context;
- conversation progress;
- internal monologue residue;
- promoted reflection context;
- local time and scene pressure.

Targeted RAG, conversation search, public web/current facts, human
clarification, guarded action preparation, and local artifact inspection are
capabilities selected by cognition when that substrate is insufficient.

## Production Scope

### In Scope

- Add a production resolver package that stores bounded cycle state and
  observations.
- Add a cognition capability-request contract separate from final action
  specs, so evidence requests do not turn `action_spec` into a generic tool
  dispatcher.
- Extend L2d to emit zero or more resolver capability requests in addition to
  final action specs.
- Run existing `call_cognition_subgraph` once per resolver cycle.
- Move RAG from mandatory pre-cognition stage to a demand-driven capability
  when the resolver is enabled.
- Keep the old RAG-first path behind a compatibility switch until live
  validation passes.
- Add HIL terminal handling that surfaces a minimal question through L3 text
  and stores resume-safe state.
- Add self-resolver support for self-cognition/internal-thought episodes with
  no unapproved outward side effects.
- Add trace artifacts and event logging for every resolver cycle.
- Add deterministic tests for graph shape, contract validation, caps, and
  safety; add real LLM validation cases for output quality.

### Out of Scope

- Arbitrary shell, database, filesystem, or adapter tools.
- Direct tool handlers calling cognition.
- Replacing L1, L2, L2d, L3, RAG workers, consolidation, scheduler, or adapter
  delivery with a generic agent harness.
- Deterministic keyword classifiers over user intent.
- Code-side semantic overrides of LLM-selected user preferences, commitments,
  permissions, or HIL decisions.
- Removing the legacy RAG-first path before shadow and live validation pass.

## Current System Contract Audit

Current live graph:

```text
brain_service.graph
  START
  -> multimedia_descriptor_agent, when needed
  -> relevance_agent
  -> load_conversation_episode_state
  -> persona_supervisor2
  -> END
```

Current persona graph:

```text
stage_0_msg_decontexualizer
  -> stage_1_research
       RAG2 supervisor
       project_known_facts(...)
       state["rag_result"]
  -> stage_2_cognition
       call_cognition_subgraph
       L1 -> L2a/L2b/L2c1/L2c2 -> L2d
       state["action_specs"]
  -> stage_2_memory_lifecycle
  -> stage_3_action or stage_3_no_response
  -> episode_trace
```

Allowed production change surface:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- new `src/kazusa_ai_chatbot/cognition_resolver/`
- existing RAG request construction helpers around `stage_1_research`
- event logging helpers only for sanitized cycle telemetry
- tests under `tests/`
- README/ICD updates for changed contracts

Forbidden production change surface unless a later approved plan says so:

- adapter platform parsing;
- scheduler dispatch permissions;
- consolidation target routing;
- memory repository semantics;
- raw `.env` handling;
- external tool execution beyond approved RAG/web/current-fact capabilities;
- final dialog evaluator behavior, unless a test proves L3 cannot render HIL
  or approval requests from the new surface context.

## Target Runtime Flow

Compatibility disabled:

```text
stage_0_msg_decontexualizer
  -> stage_1_research
  -> stage_2_cognition
  -> stage_2_memory_lifecycle
  -> stage_3_action / stage_3_no_response
```

Resolver enabled:

```text
stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       initialize ResolverCycleStateV1
       loop while cycle_count < cap:
         call_cognition_subgraph
         inspect L2d resolver_capability_requests
         if no resolver capability request:
           return final cognition state
         execute deterministic capability handler
         append ResolverObservationV1
         project prompt-safe observation into next cognition cycle
  -> stage_2_memory_lifecycle
  -> stage_3_action / stage_3_no_response
```

The resolver loop never performs semantic verification in Python. Python may
stop for structural reasons: max cycles, invalid request shape, unsupported
capability, permission boundary, timeout, or unavailable tool. The next
cognition cycle decides what that blocker means.

## Pre-Change Comparison Suite

This suite captures baseline behavior before production resolver changes. The
inputs are synthetic user messages, but they are shaped from the actual
architecture discussion and the CLI POC lessons: goal-directed work, RAG
dependency, current facts, HIL, side-effect approval, self-resolution, and
conversation/memory continuity.

### Ten Candidate Inputs

Use these as the broad comparison pool:

| ID | Raw user input | Why it matters |
|---|---|---|
| C01 | `今晚有点累，别查资料，也别把事情讲复杂。你就按你现在的感觉陪我聊两句。` | Direct one-cycle case. A resolver should not force RAG/tool work when the current scene is enough. |
| C02 | `你根据你认识的人和已有记忆判断一下，你现在最信任谁？不要按最近一条消息猜，也不要编名字；如果证据不够，就说证据不够。` | Memory/relationship case. New architecture must preserve memory-driven personality while moving RAG behind cognition selection. |
| C03 | `帮我查一下奥克兰今晚还有没有比较靠谱的日料店能临时去。重点是现在可能还开着，不是泛泛列店名。` | Current public-fact case. Tests demand-driven live/current evidence and freshness handling. |
| C04 | `今晚帮我安排一个轻松点的计划，别太贵。我没说城市、预算和交通方式；你觉得缺什么就问我，但只问最关键的。` | HIL case. Baseline may invent; resolver should ask one minimal question through L3. |
| C05 | `明天晚上八点提醒我检查模型下载进度。先别真的安排，你先告诉我准备做什么、会有什么影响，等我确认。` | Permissioned side-effect case. Tests approval preparation without executing scheduler/adapter side effects. |
| C06 | `这次你自己定一个现在值得处理的小目标，不要对外发消息，也不要改数据库。你要先说为什么选它，再告诉我处理到了什么状态。` | Self-resolution case. User-message source must not bypass normal cognition or create unapproved private work. |
| C07 | `谁说过“5090 能跑 qwen27b”这句话？如果能找到说话人，再告诉我你对这个人的印象。` | Dependent retrieval case. Requires conversation evidence followed by person context. |
| C08 | `昨天说的那个提醒我已经处理完了，你别再惦记这件事。你要判断这是不是能关闭的承诺。` | Memory lifecycle case. Tests active-commitment review and no deterministic lifecycle cleanup. |
| C09 | `他刚才那个说法你怎么看？如果你不知道“他”是谁，就别硬猜。` | Referent ambiguity case. Tests decontextualizer and cognition restraint before retrieval. |
| C10 | `我想确认 OpenHands 最新一次正式发布是哪一天，不要只看搜索摘要，要区分正式发布、预告和测试版。` | Hard public-fact case. Tests multi-step public evidence and source caveats. |

### Refined Five-Case Baseline Suite

The refined suite intentionally excludes cases that are likely to produce
little architectural contrast or depend on fragile prior chat state. These
five cases should produce clear before/after comparison signals:

| Case | Raw user input | Expected baseline signal | Expected resolver signal |
|---|---|---|---|
| B01 direct | `今晚有点累，别查资料，也别把事情讲复杂。你就按你现在的感觉陪我聊两句。` | Current RAG initializer should ideally return no slots; response quality shows whether the baseline can avoid over-retrieval. | One cognition cycle, no resolver capability request, L3 only if character chooses to speak. |
| B02 memory | `你根据你认识的人和已有记忆判断一下，你现在最信任谁？不要按最近一条消息猜，也不要编名字；如果证据不够，就说证据不够。` | Baseline should run RAG before cognition and may answer from relationship/profile evidence or insufficiency. | First cognition cycle should request `rag_evidence`; second cycle should ground stance in observation. |
| B03 current facts | `帮我查一下奥克兰今晚还有没有比较靠谱的日料店能临时去。重点是现在可能还开着，不是泛泛列店名。` | Baseline should route live/current evidence before cognition; output may expose freshness limitations. | Cognition should request live/current evidence only after judging that the task needs it. |
| B04 HIL | `今晚帮我安排一个轻松点的计划，别太贵。我没说城市、预算和交通方式；你觉得缺什么就问我，但只问最关键的。` | Baseline may either ask a question or produce a generic plan; this reveals whether current RAG-first flow invents missing user-owned constraints. | Cognition should select `human_clarification`, receive a blocked observation, then L3 asks one minimal question. |
| B05 approval | `明天晚上八点提醒我检查模型下载进度。先别真的安排，你先告诉我准备做什么、会有什么影响，等我确认。` | Baseline may schedule, promise, or only speak. This reveals current side-effect/approval boundary behavior. | Cognition should select approval preparation, not execute side effects, then L3 explains the pending action and waits. |

Baseline artifacts are stored under:

```text
test_artifacts/cognition_resolver/baseline_20260530/
```

Required baseline review artifact:

```text
test_artifacts/cognition_resolver/baseline_20260530/baseline_comparison_inputs_and_outputs.md
```

The review must include, per refined case:

- raw user input;
- `/chat` request metadata and debug mode;
- raw response messages;
- observed RAG/cognition/dialog path from logs when available;
- whether the case is expected to show direct, RAG, HIL, or approval contrast
  after the production resolver is implemented;
- quality notes grounded in the captured response and logs.

### Captured Baseline Result

Baseline capture completed on 2026-05-30 from an isolated service started from
this checkout on port `8010`. The successful request bundle is:

```text
test_artifacts/cognition_resolver/baseline_20260530/raw_baseline_responses.json
```

The human-readable comparison report is:

```text
test_artifacts/cognition_resolver/baseline_20260530/baseline_comparison_inputs_and_outputs.md
```

Important baseline observations:

- B01 direct chat already behaves as the intended one-cycle resolver: no RAG
  slots, no capability loop, visible dialog only.
- B02 memory judgment runs RAG before cognition, cannot confirm a trusted
  person, and correctly refuses to invent. The trace also exposes current
  routing weakness between person context and memory evidence.
- B03 current facts uses live context plus public web evidence and returns a
  useful answer, but the log shows URL-read 403 failures that are hidden by the
  final wording. Resolver telemetry must preserve those observation details.
- B04 HIL asks one useful location question, but the current system represents
  HIL only as ordinary dialog. The resolver must make the `waiting_for_user`
  terminal state explicit.
- B05 approval avoids promising the reminder, but it still invents plausible
  technical check steps from weak evidence. The resolver must expose both the
  pending approval state and the evidence gap.

### Lessons From The POC And Baseline

- The earlier CLI POC was useful as a harness experiment only. Its
  planner/verifier/finalizer shape must not become production architecture
  because it bypasses Kazusa's preserved cognition core.
- The production resolver must be recurrence around `L1 -> L2 -> L2d`, not an
  assistant-style external tool loop. Every semantic stage, including whether
  evidence is enough, must return to cognition.
- The baseline proves final dialog alone is insufficient evidence for review.
  The production implementation must emit a human-readable per-cycle trace:
  cycle index, cognition summary, capability request, capability observation,
  terminal reason, held action specs, and final surface decision.
- RAG should move from mandatory pre-cognition execution to a demand-driven
  resolver capability while preserving always-present personality substrate.
- HIL, approval, unsupported capability, timeout, and max-cycle stops must be
  explicit resolver statuses. Python may stop the loop structurally, but L2d
  decides what that stop means for the character and final surface.

### Independent Review Remediation Contract

An independent review on 2026-05-30 found the first production plan was aligned
architecturally but not executable. This plan therefore adds the following
non-optional implementation constraints before code work may proceed:

- `call_cognition_subgraph()` must propagate `resolver_capability_requests`
  from L2d back into `GlobalPersonaState`; otherwise the resolver loop cannot
  observe requests.
- The resolver loop must initialize a `project_known_facts(...)` compatible
  empty `rag_result` before the first cognition cycle when mandatory
  `stage_1_research` is skipped.
- RAG capability execution must receive the L2d `objective`. The capability
  objective becomes the fresh RAG query, while the original decontextualized
  user input remains in context for grounding.
- Every cycle must produce a prompt-safe trace row with L1, L2, L2d, capability
  request, observation, terminal status, held action specs, and final surface
  decision summaries.
- HIL and approval stops must create durable pending-resume records scoped to
  platform, channel, user, source message, expiry, and capability kind. A later
  user turn resumes through normal cognition; deterministic code does not
  decide whether the new user text satisfies the pending question or approval.
- `stage_1_goal_resolver` must load matching unexpired pending HIL/approval
  rows before the first cognition cycle and project them into
  `resolver_context`. L2d must emit a structural close/supersede/continue
  instruction after interpreting the user turn; deterministic code only applies
  that instruction to the pending ledger row.
- The rejected CLI POC plan must not remain the active executable plan. It is a
  superseded reference only.

## Contract Design

### ResolverCycleStateV1

Stored in `GlobalPersonaState["resolver_state"]` and projected into
`CognitionState["resolver_context"]` as one bounded semantic string plus a
small structural status object.

```python
class ResolverCycleStateV1(TypedDict):
    schema_version: Literal["resolver_cycle_state.v1"]
    cycle_index: int
    max_cycles: int
    status: Literal[
        "running",
        "terminal",
        "blocked",
        "max_cycles",
        "waiting_for_user",
        "waiting_for_approval",
    ]
    original_decontexualized_input: str
    observations: list[ResolverObservationV1]
    cycle_traces: list[ResolverCycleTraceV1]
    held_action_specs: list[ActionSpecV1]
    pending_resume: NotRequired[ResolverPendingResumeV1]
    terminal_reason: str
```

### ResolverCapabilityRequestV1

Emitted by L2d as `resolver_capability_requests`. It is not an action spec and
does not represent final user-visible behavior.

```python
class ResolverCapabilityRequestV1(TypedDict):
    schema_version: Literal["resolver_capability_request.v1"]
    capability_kind: Literal[
        "rag_evidence",
        "web_evidence",
        "human_clarification",
        "approval_preparation",
        "self_goal_resolution",
    ]
    objective: str
    reason: str
    priority: Literal["now", "background"]
```

Rules:

- `objective` is ordinary semantic text, not backend slot syntax.
- RAG-specific slot generation stays inside the existing RAG initializer.
- Web/current-fact low-level details stay inside the RAG/web capability owner.
- Human clarification must include one minimal question in `objective`.
- Approval preparation must describe the proposed side effect, expected effect,
  and approval need in `objective`.
- Self goal resolution is private-only and cannot request adapter delivery.

### ResolverPendingResolutionV1

Emitted by L2d when the current cognition cycle is interpreting a pending HIL
or approval row that was projected into `resolver_context`.

```python
class ResolverPendingResolutionV1(TypedDict):
    schema_version: Literal["resolver_pending_resolution.v1"]
    resume_id: str
    decision: Literal["continue_waiting", "answered", "approved", "rejected", "superseded"]
    reason: str
```

This is structural state chosen by cognition. Deterministic code may use it to
close, keep, or supersede the ledger row. Deterministic code must not infer
approval from keywords such as "yes" or "ok".

### ResolverObservationV1

Produced by deterministic capability handlers.

```python
class ResolverObservationV1(TypedDict):
    schema_version: Literal["resolver_observation.v1"]
    observation_id: str
    capability_kind: str
    request_objective: str
    request_reason: str
    status: Literal["succeeded", "blocked", "failed"]
    prompt_safe_summary: str
    rag_result: NotRequired[dict]
    pending_resume_id: NotRequired[str]
    evidence_refs: list[EvidenceRefV1]
    created_at_utc: str
```

Only `prompt_safe_summary` and existing prompt-safe `rag_result` projection are
visible to cognition. IDs and raw handler metadata remain deterministic trace
data.

### ResolverCycleTraceV1

Produced once per cognition cycle. It is the review surface for the user's
requirement to see every iteration.

```python
class ResolverCycleTraceV1(TypedDict):
    schema_version: Literal["resolver_cycle_trace.v1"]
    cycle_index: int
    status_before_cycle: str
    l1_emotional_appraisal: str
    l1_interaction_subtext: str
    l2_internal_monologue_summary: str
    l2_logical_stance: str
    l2_character_intent: str
    l2_judgment_note: str
    l2d_resolver_capability_requests: list[ResolverCapabilityRequestV1]
    l2d_action_specs_summary: list[str]
    selected_capability_kind: str
    observation_ids: list[str]
    final_surface_decision: str
    terminal_reason: str
    created_at_utc: str
```

The trace stores generated cognition artifacts, not hidden model reasoning. It
must clip long text before event logging or artifact writing.

### ResolverPendingResumeV1

Pending HIL and approval state is stored through the existing generic
action-attempt ledger (`self_cognition_action_attempts`) using resolver-specific
`action_kind` values. This avoids a new collection while preserving durable
scope, expiry, and idempotency.

```python
class ResolverPendingResumeV1(TypedDict):
    schema_version: Literal["resolver_pending_resume.v1"]
    resume_id: str
    capability_kind: Literal["human_clarification", "approval_preparation"]
    status: Literal[
        "waiting_for_user",
        "waiting_for_approval",
        "closed",
        "expired",
        "superseded",
    ]
    platform: str
    platform_channel_id: str
    global_user_id: str
    source_message_id: str
    prompt_safe_question: str
    prompt_safe_approval_summary: str
    created_at_utc: str
    expires_at_utc: str
```

The resolver must load and project matching pending rows into
`resolver_context` before the first resolver cognition cycle. It must not
deterministically interpret a later user input as "answered" or "approved";
that interpretation must pass through `L1 -> L2 -> L2d` and return
`ResolverPendingResolutionV1`.

## File Structure

Create:

- `src/kazusa_ai_chatbot/cognition_resolver/__init__.py`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
  - TypedDict contracts, structural validators, prompt-safe projection helpers.
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
  - State initialization, observation append, cap handling, empty RAG result.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
  - Deterministic capability dispatcher for allowed resolver capabilities.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
  - Recurrence controller that calls `call_cognition_subgraph`.
- `src/kazusa_ai_chatbot/cognition_resolver/telemetry.py`
  - Sanitized event rows and debug summaries.
- `src/kazusa_ai_chatbot/cognition_resolver/pending.py`
  - Pending HIL/approval record builders, ledger persistence, and prompt-safe
    resume projection.
- `tests/test_cognition_resolver_contracts.py`
- `tests/test_cognition_resolver_loop.py`
- `tests/test_cognition_resolver_persona_graph.py`
- `tests/test_cognition_resolver_l2d_contract.py`

Modify:

- `src/kazusa_ai_chatbot/config.py`
  - Add `COGNITION_RESOLVER_ENABLED`, `COGNITION_RESOLVER_MAX_CYCLES`, and
    `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add resolver state/context fields to `GlobalPersonaState` and
    `CognitionState`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Pass resolver context into cognition and propagate
    `resolver_capability_requests` out of L2d results.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  - Extend prompt and parser to emit resolver capability requests.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  - Validate `resolver_capability_requests`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Add resolver-enabled graph path while preserving legacy graph path.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Document resolver recurrence and demand-driven RAG.
- `src/kazusa_ai_chatbot/action_spec/README.md`
  - Clarify that resolver evidence capabilities are not final action specs.
- `docs/HOWTO.md`
  - Document feature flags and validation commands.
- `development_plans/README.md`
  - Register this plan.

## Implementation Tasks

### Task 1: Branch, Gates, And Plan Registry

**Files:**

- Modify: `development_plans/README.md`
- Modify: this plan file

- [ ] **Step 1: Verify branch and workspace**

Run:

```powershell
git status --short --branch
```

Expected:

```text
## resolver-goal-poc...origin/resolver-goal-poc
```

If the workspace has unrelated user changes, do not reset them. Record them in
the task notes and avoid staging them.

- [ ] **Step 2: Register this plan**

Add this row under `Active Short-Term Plans`:

```markdown
| [cognition_preserving_goal_resolver_production_plan.md](active/short_term/cognition_preserving_goal_resolver_production_plan.md) | Large production architecture implementation plan | in_progress |
```

Move the rejected POC plan out of the active execution set:

```powershell
git mv development_plans\active\short_term\goal_resolver_poc_plan.md development_plans\archive\superseded\goal_resolver_poc_plan.md
```

Remove its row from `Active Short-Term Plans` and add it to the superseded
records table. The archived POC remains historical evidence only.

Remove the rejected CLI POC implementation artifacts so the production source
baseline matches `main` before new resolver implementation starts:

```powershell
git rm -r -- resources\goal_resolver_poc src\kazusa_ai_chatbot\goal_resolver_poc src\scripts\run_goal_resolver_poc.py test_artifacts\goal_resolver_poc
git diff --name-status main -- src resources test_artifacts\goal_resolver_poc
```

Expected: no diff for those production/resource artifact paths after the
removal is staged.

- [ ] **Step 3: Commit the plan-only baseline and old POC removal**

Run:

```powershell
git add development_plans\README.md development_plans\active\short_term\cognition_preserving_goal_resolver_production_plan.md development_plans\archive\superseded\goal_resolver_poc_plan.md
git diff --cached --check
git commit -m "Plan cognition-preserving goal resolver"
```

Expected: commit succeeds with only plan/archive files and rejected POC
implementation removals staged. No new production resolver code is added in
this commit.

### Task 2: Add Resolver Contracts

**Files:**

- Create: `src/kazusa_ai_chatbot/cognition_resolver/__init__.py`
- Create: `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- Test: `tests/test_cognition_resolver_contracts.py`

- [x] **Step 1: Write failing contract tests**

Add tests for:

- valid `ResolverCapabilityRequestV1`;
- invalid unknown `capability_kind`;
- empty `objective` rejection;
- valid `ResolverObservationV1`;
- prompt-safe summary clipping;
- no raw IDs in cognition projection except observation aliases.

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py -q
```

Expected: fail because the package does not exist.

- [x] **Step 2: Implement contract module**

Define constants:

```python
RESOLVER_CYCLE_STATE_VERSION = "resolver_cycle_state.v1"
RESOLVER_CAPABILITY_REQUEST_VERSION = "resolver_capability_request.v1"
RESOLVER_OBSERVATION_VERSION = "resolver_observation.v1"
RESOLVER_CYCLE_TRACE_VERSION = "resolver_cycle_trace.v1"
RESOLVER_PENDING_RESUME_VERSION = "resolver_pending_resume.v1"
RESOLVER_PENDING_RESOLUTION_VERSION = "resolver_pending_resolution.v1"
ALLOWED_RESOLVER_CAPABILITIES = frozenset((
    "rag_evidence",
    "web_evidence",
    "human_clarification",
    "approval_preparation",
    "self_goal_resolution",
))
```

Expose functions:

```python
validate_resolver_capability_request(value: object) -> ResolverCapabilityRequestV1
validate_resolver_observation(value: object) -> ResolverObservationV1
validate_resolver_cycle_trace(value: object) -> ResolverCycleTraceV1
validate_resolver_pending_resume(value: object) -> ResolverPendingResumeV1
validate_resolver_pending_resolution(value: object) -> ResolverPendingResolutionV1
project_observations_for_cognition(observations: list[ResolverObservationV1]) -> str
project_pending_resume_for_cognition(pending: ResolverPendingResumeV1 | None) -> str
```

Validation may check structure, enum membership, required string fields,
bounded length, and list shape. It must not judge whether a user goal is
semantically satisfied.

- [x] **Step 3: Run tests**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py -q
```

Expected: pass.

- [x] **Step 4: Commit**

Run:

```powershell
git add src\kazusa_ai_chatbot\cognition_resolver tests\test_cognition_resolver_contracts.py
git commit -m "Add cognition resolver contracts"
```

Completed on 2026-05-30. Verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_resolver\contracts.py tests\test_cognition_resolver_contracts.py
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py -q
```

Result: 9 passed.

### Task 3: Add Resolver State Helpers

**Files:**

- Create: `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- Test: `tests/test_cognition_resolver_contracts.py`

- [x] **Step 1: Add failing tests**

Add tests that verify:

- `new_resolver_state()` initializes cycle 0 with no observations;
- `new_resolver_state()` initializes `cycle_traces=[]` and no
  `pending_resume`;
- `append_observation()` increments aliases as `resolver_obs_1`;
- `append_cycle_trace()` stores one clipped trace row per cognition cycle;
- observations are capped to the configured prompt projection count;
- `build_empty_rag_result()` returns a `project_known_facts` compatible shape
  with empty evidence and answer.
- `ensure_initial_resolver_inputs()` adds empty `rag_result`,
  `resolver_state`, and prompt-safe `resolver_context` before first cognition
  when mandatory RAG is skipped.

- [x] **Step 2: Implement state helpers**

Expose:

```python
new_resolver_state(
    *,
    decontexualized_input: str,
    max_cycles: int,
) -> ResolverCycleStateV1

append_observation(
    state: ResolverCycleStateV1,
    observation: ResolverObservationV1,
) -> ResolverCycleStateV1

append_cycle_trace(
    state: ResolverCycleStateV1,
    trace: ResolverCycleTraceV1,
) -> ResolverCycleStateV1

build_empty_rag_result(
    *,
    current_user_id: str,
    character_user_id: str,
) -> dict

ensure_initial_resolver_inputs(
    state: GlobalPersonaState,
    *,
    max_cycles: int,
) -> GlobalPersonaState
```

`build_empty_rag_result()` must use the existing `project_known_facts(...)`
helper so downstream cognition receives the current RAG projection shape.
`ensure_initial_resolver_inputs()` is mandatory in `stage_1_goal_resolver` and
`call_cognition_resolver_loop`; tests must fail if the first cycle can enter
`call_cognition_subgraph()` without `rag_result`.

- [x] **Step 3: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py -q
git add src\kazusa_ai_chatbot\cognition_resolver\state.py tests\test_cognition_resolver_contracts.py
git commit -m "Add cognition resolver state helpers"
```

Completed on 2026-05-30. Verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_resolver\state.py src\kazusa_ai_chatbot\cognition_resolver\contracts.py tests\test_cognition_resolver_contracts.py
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py -q
```

Result: 14 passed. `ruff` was not available in the project virtual
environment.

### Task 4: Add L2d Resolver Capability Output

**Files:**

- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- Test: `tests/test_cognition_resolver_l2d_contract.py`

- [x] **Step 1: Write failing L2d parser tests**

Patch the L2d LLM call to return:

```json
{
  "resolver_capability_requests": [
    {
      "schema_version": "resolver_capability_request.v1",
      "capability_kind": "rag_evidence",
      "objective": "检索当前用户与这个问题有关的关系和记忆证据。",
      "reason": "没有记忆证据时无法可靠判断。",
      "priority": "now"
    }
  ],
  "action_requests": []
}
```

Assert that `call_action_initializer()` returns the normalized resolver request
and no final action spec.

Add a second test around `call_cognition_subgraph()` with a patched cognition
subgraph result that contains `resolver_capability_requests`. Assert the
returned `GlobalPersonaState` update includes the same normalized requests.

- [x] **Step 2: Extend schema fields**

Add optional fields:

```python
resolver_state: NotRequired[ResolverCycleStateV1]
resolver_context: NotRequired[str]
resolver_capability_requests: NotRequired[list[ResolverCapabilityRequestV1]]
resolver_cycle_trace: NotRequired[ResolverCycleTraceV1]
pending_resolver_resume: NotRequired[ResolverPendingResumeV1]
resolver_pending_resolution: NotRequired[ResolverPendingResolutionV1]
```

Use imports from `kazusa_ai_chatbot.cognition_resolver.contracts`.

- [x] **Step 3: Extend L2d prompt contract**

Update the L2d prompt so the model can emit either:

- `resolver_capability_requests` for evidence/clarification/approval/self
  resolution needed before final stance; or
- `action_requests` for terminal surfaces/private actions.

The prompt must state:

```text
If more evidence or human-owned input is required before final action, emit
resolver_capability_requests and leave action_requests empty.
Only emit action_requests when the current cognition cycle is ready to
externalize a visible surface or private action.
```

- [x] **Step 4: Normalize and validate resolver requests**

Add `_normalize_resolver_capability_requests(parsed)` next to
`_normalize_action_requests(parsed)`. It must call
`validate_resolver_capability_request()` and drop malformed rows with a warning.

- [x] **Step 5: Propagate requests out of cognition**

In `call_cognition_subgraph()`:

- include `resolver_state`, `resolver_context`, and pending-resume prompt
  projection in the initial `CognitionState`;
- read `resolver_capability_requests = result.get(...)`;
- include `resolver_capability_requests` in the returned persona-state update;
- read and validate `resolver_pending_resolution` when L2d emits one, and
  include it in the returned persona-state update;
- include it in the sanitized log preview without raw prompt text.

This is a blocking contract. If this return field is missing, the resolver loop
will terminate incorrectly.

- [x] **Step 6: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_l2d_contract.py -q
git add src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_cognition_resolver_l2d_contract.py
git commit -m "Let L2d select resolver capabilities"
```

Completed on 2026-05-30. Verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_output_contracts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_cognition_resolver_l2d_contract.py tests\test_persona_supervisor2_action_initializer.py tests\test_cognition_prompt_contract_text.py
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py tests\test_cognition_resolver_l2d_contract.py tests\test_persona_supervisor2_action_initializer.py -q
venv\Scripts\python -m pytest tests\test_l2d_l3_surface_handoff.py tests\test_multi_source_cognition_stage_03_prompt_selection.py -q
```

Result: 25 passed for prompt/L2d resolver tests and 49 passed for existing
cognition handoff/output-contract tests.

### Task 5: Extract Reusable RAG Capability Execution

**Files:**

- Create: `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- Test: `tests/test_cognition_resolver_loop.py`

- [x] **Step 1: Write failing tests for RAG capability execution**

Build a minimal `GlobalPersonaState` fixture with:

- decontextualized input;
- character profile global user id;
- current user id;
- empty chat history;
- prompt-safe message context.

Patch `call_quote_aware_rag_supervisor()` to return one known fact. Assert that
executing `rag_evidence` returns a `ResolverObservationV1` with `status`
`succeeded` and a `rag_result` compatible with the current projection.

Add assertions that:

- the RAG supervisor `fresh_query` is the L2d resolver request `objective`;
- the original `decontexualized_input` is still present in RAG context;
- the observation stores `request_objective` and `request_reason`;
- malformed or empty objectives fail structural validation before RAG dispatch.

- [x] **Step 2: Extract a reusable RAG function**

Move the body of `stage_1_research()` that builds the RAG request and projects
known facts into a helper:

```python
async def run_rag_evidence_for_persona_state(
    state: GlobalPersonaState,
    *,
    agent_name: str,
    objective: str | None = None,
) -> dict:
```

Keep `stage_1_research()` behavior identical by calling the helper and returning
`{"rag_result": rag_result}`.

When `objective` is provided, it must become the `fresh_query` passed to
`call_quote_aware_rag_supervisor()`. The helper must still include the original
decontextualized user input in the RAG context under a prompt-safe
`original_user_request` or equivalent field, so RAG remains grounded in the
actual turn.

- [x] **Step 3: Implement capability dispatcher**

Expose:

```python
async def execute_resolver_capability_request(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1
```

For `rag_evidence`, call `run_rag_evidence_for_persona_state(...)` with the
request objective.

For `web_evidence`, use the existing RAG/web path by feeding the objective into
the RAG request context; do not call arbitrary HTTP directly in this task.

For `human_clarification`, return `status="blocked"` and a prompt-safe summary
containing the minimal question.

For `approval_preparation`, return `status="blocked"` and a prompt-safe summary
containing the proposed action and approval need.

For `self_goal_resolution`, return `status="blocked"` unless the current
episode is `internal_thought` or `self_cognition`; this prevents user-facing
turns from spawning private self-goals as an implementation side effect.

- [x] **Step 4: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
git add src\kazusa_ai_chatbot\cognition_resolver\capabilities.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_cognition_resolver_loop.py
git commit -m "Add cognition resolver capability execution"
```

Completed on 2026-05-30. Verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_resolver\contracts.py src\kazusa_ai_chatbot\cognition_resolver\capabilities.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_cognition_resolver_loop.py
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_contracts.py -q
venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_dialog_event_logging.py tests\test_persona_supervisor2_rag_skip_shape.py -q
```

Result: 18 passed for resolver capability/contracts and 27 passed for legacy
RAG integration/event/skip-shape compatibility.

### Task 6: Implement Cognition Recurrence Loop

**Files:**

- Create: `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- Test: `tests/test_cognition_resolver_loop.py`

- [x] **Step 1: Write failing loop tests**

Patch `call_cognition_subgraph` behavior through a fake function sequence:

1. cycle 1 returns one `rag_evidence` resolver request;
2. capability executor returns one observation;
3. cycle 2 returns one `speak` action spec and no resolver request.

Assert:

- cognition was called twice;
- capability execution happened once between the two cognition calls;
- final state contains the second cycle action spec;
- the observation summary was present in the second cognition input.
- the first cognition call receives a valid empty `rag_result` even though
  mandatory `stage_1_research` did not run;
- two `ResolverCycleTraceV1` rows are present, each with L1/L2/L2d summaries
  and terminal/final-surface fields.

- [x] **Step 2: Implement loop function**

Expose:

```python
async def call_cognition_resolver_loop(
    state: GlobalPersonaState,
    *,
    call_cognition_subgraph_func: Callable[[GlobalPersonaState], Awaitable[GlobalPersonaState]],
    execute_capability_func: Callable[[ResolverCapabilityRequestV1, GlobalPersonaState], Awaitable[ResolverObservationV1]],
    max_cycles: int,
    capability_timeout_seconds: float,
) -> GlobalPersonaState
```

Loop rules:

- initialize `rag_result`, `resolver_state`, and `resolver_context` before the
  first cognition cycle by calling `ensure_initial_resolver_inputs()`;
- call full cognition once per cycle;
- capture one `ResolverCycleTraceV1` after each cognition call using the
  returned L1/L2/L2d fields;
- if `resolver_capability_requests` is empty, return final cognition result;
- if a resolver request exists, execute at most the first `priority="now"`
  request in this implementation slice;
- wrap capability execution with `asyncio.wait_for(...,
  timeout=capability_timeout_seconds)`;
- on timeout, append a `ResolverObservationV1` with `status="failed"`,
  `capability_kind` from the request, `request_objective`, `request_reason`,
  and a prompt-safe timeout summary, then run the next cognition cycle;
- append observation to `resolver_state`;
- append the observation id and selected capability kind to the current cycle
  trace;
- project observations into `resolver_context`;
- update `rag_result` only from observation `rag_result`;
- continue until cap;
- when cap is reached, set `resolver_state.status="max_cycles"` and run one
  final cognition cycle with the cap blocker as an observation.

- [x] **Step 3: Pass resolver context into cognition**

In `call_cognition_subgraph`, include:

```python
"resolver_state": state.get("resolver_state"),
"resolver_context": state.get("resolver_context", ""),
"pending_resolver_resume": state.get("pending_resolver_resume"),
```

The cognition prompt payload builders must receive only prompt-safe
`resolver_context` and prompt-safe pending-resume projection, not raw
observation dicts or ledger records.

- [x] **Step 4: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
git add src\kazusa_ai_chatbot\cognition_resolver\loop.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py tests\test_cognition_resolver_loop.py
git commit -m "Add cognition resolver recurrence loop"
```

Implementation notes:

- `call_cognition_resolver_loop()` merges cognition updates into the prior
  `GlobalPersonaState` each cycle so required persona inputs are preserved.
- First-cycle input setup is mandatory through
  `ensure_initial_resolver_inputs()`, including an empty projected
  `rag_result` when legacy `stage_1_research` is skipped.
- Max-cycle blocker observations reuse the last real capability request kind,
  objective, and reason. This keeps `ResolverObservationV1` inside the approved
  capability taxonomy instead of inventing a fake system capability.
- Review coverage includes two-cycle RAG recurrence, timeout observation,
  max-cycle final cognition, prompt-safe observation projection, and trace
  shape validation.

Verification:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
# 7 passed
```

### Task 7: Integrate Resolver Into Persona Graph Behind Feature Flags

**Files:**

- Modify: `src/kazusa_ai_chatbot/config.py`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- Test: `tests/test_cognition_resolver_persona_graph.py`

- [x] **Step 1: Add failing graph tests**

Test two modes:

- resolver disabled: graph still calls `stage_1_research` before cognition;
- resolver enabled: graph calls resolver loop after decontextualizer and does
  not call mandatory `stage_1_research`.

- [x] **Step 2: Add config flags**

Add:

```python
COGNITION_RESOLVER_ENABLED = _bool_from_env("COGNITION_RESOLVER_ENABLED", "false")
COGNITION_RESOLVER_MAX_CYCLES = _bounded_int_from_env(
    "COGNITION_RESOLVER_MAX_CYCLES",
    "3",
    minimum=1,
    maximum=5,
)
COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS = _bounded_float_from_env(
    "COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS",
    "45.0",
    minimum=1.0,
    maximum=180.0,
)
```

Use the existing config helper style in `config.py`.

`COGNITION_RESOLVER_SHADOW_MODE` is intentionally out of scope for this
production slice. It requires a separate contract because shadow mode would run
two semantic pipelines in one live turn and must prove it cannot duplicate side
effects, pending approvals, scheduler events, or persistence.

- [x] **Step 3: Add resolver stage**

In `persona_supervisor2.py`, create:

```python
async def stage_1_goal_resolver(state: GlobalPersonaState) -> dict:
    initialized = ensure_initial_resolver_inputs(...)
    initialized = await load_matching_pending_resume_into_state(initialized)
    return await call_cognition_resolver_loop(initialized, ...)
```

Graph routing:

```text
if COGNITION_RESOLVER_ENABLED:
    stage_0_msg_decontexualizer -> stage_1_goal_resolver -> stage_2_memory_lifecycle
else:
    stage_0_msg_decontexualizer -> stage_1_research -> stage_2_cognition -> stage_2_memory_lifecycle
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_persona_graph.py -q
git add src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_cognition_resolver_persona_graph.py
git commit -m "Wire cognition resolver behind feature flag"
```

Implementation notes:

- Resolver-disabled mode preserves the existing graph sequence:
  `stage_0_msg_decontexualizer -> stage_1_research -> stage_2_cognition`.
- Resolver-enabled mode routes:
  `stage_0_msg_decontexualizer -> stage_1_goal_resolver`, then directly to
  memory lifecycle. Mandatory pre-cognition RAG is skipped and the resolver
  loop owns any demand-driven RAG request selected by cognition.
- `stage_1_goal_resolver` initializes resolver inputs before calling the loop,
  including the empty projected `rag_result`, and passes the existing
  `call_cognition_subgraph` and resolver capability executor into the loop.
- Pending HIL/approval resume loading remains in Task 8, where the durable
  resume ledger contract is implemented. Task 7 only establishes the feature
  flag and graph integration point.
- Review follow-up added config default/bound checks and asserts that the graph
  passes the intended cognition and capability callables into the resolver loop.

Verification:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_cognition_resolver_persona_graph.py
venv\Scripts\python -m pytest tests\test_config.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_loop.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py -q
# 71 passed
```

### Task 8: Add HIL And Approval Terminal Surfacing

**Files:**

- Modify: `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- Modify: `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- Create: `src/kazusa_ai_chatbot/cognition_resolver/pending.py`
- Modify: `src/kazusa_ai_chatbot/action_spec/README.md`
- Modify: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- Test: `tests/test_cognition_resolver_loop.py`

- [ ] **Step 1: Add failing HIL tests**

Test that a `human_clarification` resolver request:

- records a blocked observation;
- writes one `ResolverPendingResumeV1` row to the existing action-attempt
  ledger with `action_kind="resolver_pending_hil"`;
- scopes the pending row to platform, channel, user, source message, and expiry;
- re-enters one final cognition cycle;
- causes L2d to select `speak` with a minimal question;
- does not execute adapter delivery directly from the resolver.

Add a next-turn resume test:

- seed an unexpired `resolver_pending_hil` ledger row for the same platform,
  channel, and user;
- run `stage_1_goal_resolver` with a follow-up user message;
- assert the pending row is loaded and projected into cognition before cycle 1;
- fake L2d returns `resolver_pending_resolution.decision="answered"`;
- assert deterministic code marks the pending row `closed` only because L2d
  emitted that resolution.

- [ ] **Step 2: Add failing approval tests**

Test that an `approval_preparation` resolver request:

- records a blocked observation;
- writes one `ResolverPendingResumeV1` row to the existing action-attempt
  ledger with `action_kind="resolver_pending_approval"`;
- scopes the pending row to platform, channel, user, source message, and expiry;
- re-enters one final cognition cycle;
- causes L2d to select `speak` or private finalization explaining the prepared
  action;
- does not execute the side effect.

Add a next-turn approval resume test:

- seed an unexpired `resolver_pending_approval` ledger row for the same scope;
- run `stage_1_goal_resolver` with a follow-up user message;
- assert the pending approval is projected into cognition;
- fake L2d returns `resolver_pending_resolution.decision="approved"`;
- assert deterministic code closes the pending row as approved without running
  the side effect in the resolver stage.

- [ ] **Step 3: Implement pending-resume persistence**

In `pending.py`, expose:

```python
build_pending_resume_record(...)
upsert_pending_resume(...)
load_matching_pending_resume(...)
apply_pending_resolution(...)
project_pending_resume_for_cognition(...)
```

Use the existing `action_spec.attempt_ledger.upsert_action_attempt()` and
`list_action_attempts()` facade. Do not add a new collection in this slice.

Pending records are deterministic state only. On the next user turn,
`stage_1_goal_resolver` must load a matching unexpired pending row and project
it into `resolver_context`, but cognition must decide whether the user answered
the question, approved the action, rejected it, or changed topic. Deterministic
code may apply `ResolverPendingResolutionV1` only after L2d emits it.

Expired rows are not projected to cognition; deterministic code may mark them
`expired` by timestamp validation.

- [ ] **Step 4: Implement blocked-observation cycle behavior**

For blocked HIL/approval observations, the loop must run one more full
cognition cycle and then return. If the final cycle again requests the same
blocked capability, the loop returns a private no-response surface plus event
log warning rather than repeating forever.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
git add src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\action_spec\README.md src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py tests\test_cognition_resolver_loop.py
git commit -m "Add HIL and approval resolver terminals"
```

### Task 9: Add Self-Resolver Constraints

**Files:**

- Modify: `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- Modify: `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- Test: `tests/test_cognition_resolver_loop.py`

- [ ] **Step 1: Add failing self-resolver tests**

Create two fake episode states:

- `trigger_source="user_message"` requesting `self_goal_resolution`;
- `trigger_source="internal_thought"` requesting `self_goal_resolution`.

Assert user-message source is blocked. Assert internal-thought source returns a
private observation and never creates a user-visible send by itself.

- [ ] **Step 2: Implement source constraints**

Allow `self_goal_resolution` only when the cognitive episode trigger is
`internal_thought` or a future explicit `self_cognition` trigger. The handler
may produce prompt-safe private progress text, but execution must be private
and bounded.

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
git add src\kazusa_ai_chatbot\cognition_resolver tests\test_cognition_resolver_loop.py
git commit -m "Constrain self resolver capability"
```

### Task 10: Add Sanitized Telemetry And Debug Artifacts

**Files:**

- Create: `src/kazusa_ai_chatbot/cognition_resolver/telemetry.py`
- Modify: `src/kazusa_ai_chatbot/event_logging/README.md`
- Test: `tests/test_cognition_resolver_loop.py`

- [ ] **Step 1: Add failing telemetry tests**

Assert telemetry rows include:

- cycle count;
- per-cycle L1 emotional appraisal and interaction subtext summaries;
- per-cycle L2 stance, intent, judgment, and internal monologue summary;
- per-cycle L2d resolver request and action-spec summaries;
- capability kind;
- observation status;
- terminal reason;
- pending resume status when present;
- duration label.

Assert telemetry rows exclude:

- raw prompt text;
- raw message bodies;
- raw platform ids;
- raw DB ids;
- credentials or callback URLs.

Add timeout tests in `tests/test_cognition_resolver_loop.py`:

- capability executor sleeps longer than `capability_timeout_seconds`;
- loop records a failed timeout observation;
- timeout observation is present in the next cognition input;
- terminal trace contains the timeout status without raw prompt text.

- [ ] **Step 2: Implement telemetry helpers**

Expose:

```python
build_resolver_cycle_event(...)
build_resolver_terminal_event(...)
write_human_readable_resolver_trace(...)
```

Return plain dicts ready for existing event logging. If no dedicated event log
method exists, log through existing runtime/pipeline event helpers with
component `nodes.cognition_resolver`.

`write_human_readable_resolver_trace(...)` writes bounded local debug artifacts
under `test_artifacts/cognition_resolver/` for validation. It must not be part
of normal production persistence and must not include raw DB ids, credentials,
or raw adapter wire text.

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q
git add src\kazusa_ai_chatbot\cognition_resolver\telemetry.py src\kazusa_ai_chatbot\event_logging\README.md tests\test_cognition_resolver_loop.py
git commit -m "Add cognition resolver telemetry"
```

### Task 11: Documentation And Operator Guide

**Files:**

- Modify: `src/kazusa_ai_chatbot/nodes/README.md`
- Modify: `src/kazusa_ai_chatbot/action_spec/README.md`
- Modify: `docs/HOWTO.md`

- [ ] **Step 1: Update cognition node docs**

Document the resolver-enabled persona graph:

```text
stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       repeated call_cognition_subgraph cycles
       demand-driven resolver capabilities
  -> stage_2_memory_lifecycle
  -> stage_3_action / stage_3_no_response
```

State that every cycle runs L1 -> L2 -> L2d.

- [ ] **Step 2: Update action-spec docs**

Clarify that final actions and evidence capabilities are separate:

```text
ActionSpecV1 represents selected surfaces/private actions.
ResolverCapabilityRequestV1 represents evidence or blocker requests that must
feed another cognition cycle before final action selection.
```

- [ ] **Step 3: Update HOWTO**

Document:

```env
COGNITION_RESOLVER_ENABLED=false
COGNITION_RESOLVER_MAX_CYCLES=3
COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS=45.0
```

Add validation commands:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py -q
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add src\kazusa_ai_chatbot\nodes\README.md src\kazusa_ai_chatbot\action_spec\README.md docs\HOWTO.md
git commit -m "Document cognition resolver architecture"
```

### Task 12: Real LLM Validation

**Files:**

- Create: `test_artifacts/cognition_resolver/`
- Modify: this plan file with execution evidence after validation

- [ ] **Step 1: Run deterministic tests**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_cognition_resolver_l2d_contract.py -q
```

Expected: all pass.

- [ ] **Step 2: Rerun exact B01 direct-response live case**

Use the debug adapter or existing character-test workflow with resolver enabled
and max cycles 3. Input must exactly match the baseline case:

```text
今晚有点累，别查资料，也别把事情讲复杂。你就按你现在的感觉陪我聊两句。
```

Expected:

- one cognition cycle;
- no RAG capability request;
- L2d selects `speak` or silence based on character judgment;
- visible output, if any, is rendered by L3.

- [ ] **Step 3: Rerun exact B02 memory/RAG-demand live case**

Input:

```text
你根据你认识的人和已有记忆判断一下，你现在最信任谁？不要按最近一条消息猜，也不要编名字；如果证据不够，就说证据不够。
```

Expected:

- first cognition cycle requests `rag_evidence`;
- RAG observation re-enters cognition;
- later cognition cycle selects final surface or evidence-backed insufficiency.

- [ ] **Step 4: Rerun exact B03 current-fact live case**

Input:

```text
帮我查一下奥克兰今晚还有没有比较靠谱的日料店能临时去。重点是现在可能还开着，不是泛泛列店名。
```

Expected:

- cognition requests current/public evidence;
- observation contains freshness caveat or current evidence;
- final stance distinguishes current from stale information.

- [ ] **Step 5: Rerun exact B04 HIL live case**

Input:

```text
今晚帮我安排一个轻松点的计划，别太贵。我没说城市、预算和交通方式；你觉得缺什么就问我，但只问最关键的。
```

Expected:

- cognition requests `human_clarification`;
- blocked observation re-enters cognition;
- L3 asks one minimal question;
- no fabricated location, budget, transport, or preference.

- [ ] **Step 6: Rerun exact B05 approval live case**

Input:

```text
明天晚上八点提醒我检查模型下载进度。先别真的安排，你先告诉我准备做什么、会有什么影响，等我确认。
```

Expected:

- cognition requests `approval_preparation` or produces an explicit approval
  wait-state through L2d;
- no scheduler, adapter, or DB side effect executes as a reminder;
- a pending approval resume record is written with scope and expiry;
- L3 explains the pending action and waits for confirmation.

- [ ] **Step 7: Run one self-resolver dry run**

Use an internal-thought/self-cognition trigger packet with a private bounded
goal. Expected:

- shared cognition path runs;
- no adapter send occurs unless a later selected `speak` goes through normal
  delivery permission flow;
- trace shows private observation and final private action/no-response.

- [ ] **Step 8: Write validation report**

Create:

```text
test_artifacts/cognition_resolver/cognition_resolver_validation_report.md
```

The report must include for each case:

- input;
- baseline artifact path and baseline visible response;
- cycle count;
- L1/L2/L2d output summary per cycle;
- selected resolver capability, if any;
- observation summary;
- final action/surface;
- pass/fail assessment and residual risk.

- [ ] **Step 9: Commit validation artifacts only if explicitly requested**

By default, do not force-add ignored `test_artifacts`. If the user asks to
commit reports, force-add only the final human-readable report and bounded raw
trace summaries, not cache directories or large sandbox outputs.

## Acceptance Criteria

- Resolver disabled path preserves existing production graph behavior.
- Resolver enabled path runs after decontextualizer and before mandatory RAG.
- Every semantic decision runs through `call_cognition_subgraph`.
- Every resolver cycle includes full L1 -> L2 -> L2d output.
- `call_cognition_subgraph` propagates `resolver_capability_requests` from L2d.
- Resolver-enabled first cycle always receives a valid empty `rag_result` if no
  evidence has been requested yet.
- RAG is called only when L2d emits `rag_evidence`.
- RAG capability execution uses the L2d request objective as the evidence
  query and preserves the original user request as context.
- HIL and approval blockers re-enter cognition and surface through L3, not
  through direct resolver text.
- HIL and approval blockers create durable pending-resume rows scoped to
  platform/channel/user/source message with expiry.
- Next-turn HIL and approval resumes load unexpired pending rows into
  cognition and close/approve/reject/supersede them only from an L2d
  `ResolverPendingResolutionV1`.
- Capability timeouts become failed observations that re-enter cognition and
  are visible in per-cycle trace output.
- Self-resolver capability is private-only unless normal L2d/L3/delivery
  pipeline later selects visible speech.
- No deterministic keyword classifier decides user intent, preferences,
  commitments, permissions, or goal satisfaction.
- Deterministic code owns only validation, caps, permission boundaries,
  execution, telemetry, and state persistence.
- Real LLM validation report reruns exact B01-B05 baseline inputs and includes
  at least three multi-cycle cases plus one internal self-resolver dry run.

## Rollout Plan

1. Merge with `COGNITION_RESOLVER_ENABLED=false`.
2. Enable in local debug only and run deterministic plus live validation.
3. Compare resolver trace against legacy RAG-first outputs for at least 20
   previous real conversations.
4. Enable resolver for private debug adapter only.
5. Enable resolver for one low-traffic live adapter channel.
6. Keep legacy path available for one full validation window after live
   enablement.

## Risks And Controls

| Risk | Control |
|---|---|
| Resolver becomes external agent planner | L2d emits capability requests; no standalone planner/verifier is added. |
| RAG evidence disappears for memory-heavy character behavior | Always-present personality substrate remains in cycle 1; L2d can request `rag_evidence`; validation includes memory-heavy cases. |
| Local LLM emits malformed capability requests | Structural validator drops malformed rows; cognition receives a bounded blocker only when needed. |
| Tool loops increase latency | `COGNITION_RESOLVER_MAX_CYCLES` defaults to 3; one capability request executes per cycle in the first production slice. |
| Capability handler stalls | `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS` converts timeout into a failed observation that re-enters cognition. |
| HIL becomes direct assistant wording | HIL returns blocked observation, then L3 renders the visible question. |
| Side effects run without approval | `approval_preparation` records blocked observation and never executes action. |
| Pending approval is closed by keyword | L2d must emit `ResolverPendingResolutionV1`; deterministic code only applies that structural cognition output. |
| Self resolver leaks into user-visible output | Self-resolver source gate blocks user-message source; visible speech still requires normal L2d `speak` plus delivery pipeline. |
| Python semantically rewrites user intent | User interpretation remains in LLM prompts and output contracts; code performs structural validation only. |

## Completion Rule

This plan is complete only when:

- all implementation tasks are checked off;
- deterministic tests pass;
- live validation report exists and has been inspected;
- independent code review has no blocking findings;
- the feature remains behind a documented rollout flag;
- this plan is moved to `development_plans/archive/completed/short_term/` with
  execution evidence.
