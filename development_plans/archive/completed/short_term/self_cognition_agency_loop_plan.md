# self cognition agency loop plan

## Summary

- Goal: Implement the next self-cognition slice as a production-owned module
  under `src/kazusa_ai_chatbot/self_cognition/`. The module must reuse the
  current RAG2 and L1/L2/L3 cognition interfaces, write local tracking
  artifacts, and integrate an opt-in idle worker that can hand
  self-cognition-selected `send_message` candidates to the existing dispatcher
  and scheduler without depending on `experiments/*`.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `database-data-pull` when using real exported data, and `cjk-safety` before
  editing Python files that contain CJK text.
- Overall cutover strategy: compatible. Live runtime behavior remains off by
  default through `SELF_COGNITION_ENABLED=false`; when enabled, self-cognition
  may schedule only cognition-selected `send_message` candidates through the
  existing `TaskDispatcher`.
- Highest-risk areas: accidental production writes or sends, bypassing the
  shared cognition stack, weak duplicate suppression for expired commitments,
  reflection artifacts becoming triggers, and broad LLM/RAG context growth.
- Acceptance criteria: artifacts prove the tracking and dispatcher handoff
  contracts; focused tests, planned real LLM smoke, static gates, and
  independent review pass.

## Context

Self-cognition is intended to become an idle agency loop. Stages 1-6 proved the
production-owned dry-run module and user-reviewed real-data quality. This
follow-up stage integrates the module into the service runtime behind
`SELF_COGNITION_ENABLED=false` by default. Runtime integration must reuse the
existing dispatcher and scheduler for outward delivery; deleting
`experiments/*` must not break the module, worker, tests, smoke, or docs.

Reference documents: `development_plans/archive/superseded/self_cognition_reasoning_basis.md`,
`development_plans/archive/superseded/self_cognition_tracking_icd.md`, and
`development_plans/archive/superseded/self_cognition_loop_architecture.md`.
The reasoning basis holds research and examples; this plan is the execution
contract.

## Mandatory Skills

- `development-plan-writing`: plan changes and execution evidence.
- `local-llm-architecture`: source-packet, RAG, cognition, or prompt-like work.
- `no-prepost-user-input`: action selection, commitment interpretation, or send decision handling.
- `py-style`: Python edits.
- `test-style-and-execution`: test edits or runs.
- `database-data-pull`: real chat/progress/memory/user exports for case files.
- `cjk-safety`: Python files containing CJK prompt text or expected strings.

## Mandatory Rules

- After context compaction or major checklist sign-off, reread this entire plan
  before continuing.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use `venv\Scripts\python` for Python tests and scripts.
- Do not read `.env`.
- Modify production code only in the files named in `Change Surface`.
- Do not write production conversation history, conversation progress, user
  memory, character state, reflection state, Cache2 state, or adapter output.
- The only allowed production write is the existing dispatcher/scheduler write
  when `SELF_COGNITION_ENABLED=true` and shared cognition selected outward
  contact.
- Do not call adapters directly, production consolidation persistence,
  reflection persistence, memory persistence, or normal `/chat` service
  entrypoints from self-cognition.
- Do not create a standalone self-cognition LLM path. The module must call the
  existing cognition graph through the current shared cognition interface.
- Do not build a separate retrieval stack. Use existing RAG2 interfaces when
  a case needs retrieval.
- Do not use hourly reflection, daily reflection, self-reflection promotion, or
  promoted reflection artifacts as triggers. Reflection-derived state may only
  be a modifier after a valid visible/actionable trigger exists.
- Do not add a deterministic semantic proactive permission policy. Cognition
  owns the social decision to send; deterministic code owns target shape,
  idempotency, duplicate suppression, retry/backoff, budget, and audit.
- Do not route raw self-cognition internal monologue into reflection, stable
  memory, or production conversation progress.
- Do not use `think_only` as the production answer for outward-capable
  self-cognition. Dry-run/no-write behavior is represented by local artifacts
  and explicit route records.
- Keep CJK text in Python source safe per `cjk-safety`.

## Must Do

- Implement reusable self-cognition logic under
  `src/kazusa_ai_chatbot/self_cognition/`.
- Add only the minimal central config needed by this slice in
  `src/kazusa_ai_chatbot/config.py`.
- Add deterministic tests proving source framing, artifact shape, route
  classification, no forced action on model silence, action-attempt
  idempotency, and duplicate suppression.
- Add or update the module README so it owns the canonical ICD and dry-run
  command contract.
- Add opt-in service integration that starts a self-cognition idle worker only
  when `SELF_COGNITION_ENABLED=true`.
- Add local self-cognition action-attempt tracking so repeated idle ticks do
  not resend the same due occurrence after a candidate or handoff attempt.
- Convert action candidates to the existing `send_message` dispatcher shape;
  do not introduce a proactive permission policy or a parallel sender.
- Preserve existing production RAG2, cognition, and dialog entrypoints as
  imported engines only.
- Produce all required local artifact types:
  `self_cognition_trigger_record.json`,
  `self_cognition_run_record.json`,
  `self_cognition_rag_request.json` when RAG is used,
  `self_cognition_rag_output.json` when RAG is used,
  `self_cognition_cognition_input_after_rag.json`,
  `self_cognition_cognition_output.json`,
  `self_cognition_route_effect.json`,
  `self_cognition_action_candidate.json` when an outward action is considered,
  `self_cognition_action_attempt.json` when an outward action is considered,
  `self_cognition_dispatch_result.json` when a live handoff is attempted,
  and `self_cognition_loop_trace.md`.
- Model these dry-run cases with exported or synthetic local inputs:
  before-due commitment, past-due commitment, duplicate idle tick for the same
  due occurrence, no-action/progress-maintenance case, noisy group rejection,
  and bounded RAG follow-up case.
- Move the canonical ICD content into
  `src/kazusa_ai_chatbot/self_cognition/README.md`; leave the reference ICD as
  a pointer or transition note.

## Deferred

- Do not create durable MongoDB collections for self-cognition tracking.
- Do not create a separate `self_cognition_tracking` module; tracking is a
  sub-area of `src/kazusa_ai_chatbot/self_cognition/`.
- Do not wire self-cognition into `/chat`, conversation-history writes,
  conversation-progress writes, consolidation, reflection, memory writes,
  Cache2 mutation, or direct adapter sends.
- Do not add cross-platform proactive fanout, media sends, new tools beyond
  `send_message`, or external orchestrator frameworks.
- Do not mutate personality or stable character state from hourly
  self-cognition.
- Do not implement review UI, operator approval UI, media sends, or cross-target
  fanout.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Production runtime | compatible | Preserve existing behavior unless `SELF_COGNITION_ENABLED=true`. When enabled, start one service-owned idle worker and use only `TaskDispatcher` for outward `send_message` handoff. |
| Production module | bigbang | Put reusable self-cognition logic in `src/kazusa_ai_chatbot/self_cognition/`; do not depend on `experiments/*`. |
| Central config | compatible | Add only the minimal `SELF_COGNITION_*` constants listed in `Configuration Contract`; no broad env surface or legacy aliases. |
| Dry-run artifacts | bigbang | Replace the old artifact shape with the tracking-artifact shape listed in this plan. No compatibility shim for obsolete artifact names is required. |
| Tests | bigbang | Update or add focused deterministic tests for config, tracking, route classification, and harness behavior. Keep existing relevant framing tests. |
| ICD location | migration | Move the canonical ICD into `src/kazusa_ai_chatbot/self_cognition/README.md` during this plan and leave a pointer from the reference ICD. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- If an area is `bigbang`, rewrite obsolete experiment references instead of
  preserving old artifact names.
- If an area is `migration`, follow the exact migration phase listed in this
  plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce alternate production call paths, compatibility
  layers, fallback paths, extra features, or unrelated cleanup.
- The agent must search existing production code before adding helpers. Shared
  self-cognition behavior belongs in the production module.
- The agent must not create, import, or test against code under `experiments/`.
- Any change outside `src/kazusa_ai_chatbot/self_cognition/`,
  `src/kazusa_ai_chatbot/config.py`, `src/kazusa_ai_chatbot/service.py`,
  read-only DB helper files named in `Change Surface`,
  `src/scripts/run_self_cognition_dry_run.py`,
  `tests/test_self_cognition_*.py`, `tests/test_config.py`,
  service integration tests named in `Change Surface`, this plan, and the
  reference docs listed in `Change Surface` is forbidden unless the plan is
  updated first.
- If the plan and code disagree, preserve the plan intent and record the
  discrepancy before changing behavior.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

After this plan is executed, `src/kazusa_ai_chatbot/self_cognition/` exposes a
reusable self-cognition core plus an opt-in idle worker. The module CLI/script
can still run selected idle self-cognition case files and produce local
tracking artifacts that show:

- why the idle run started;
- what visible evidence and semantic due/actionability state were used;
- whether RAG2 was called and what bounded evidence it returned;
- what shared cognition input/output was used;
- which route was selected;
- which action attempt or outbox candidate was created, held, or suppressed;
- whether a production dispatcher handoff was skipped, rejected, or accepted.

Production behavior remains unchanged by default. When
`SELF_COGNITION_ENABLED=true`, the service starts one idle worker. That worker
reads visible/actionable production data through approved read interfaces,
invokes the self-cognition module, persists only local self-cognition tracking
artifacts, and hands non-duplicate `send_message` candidates to the existing
dispatcher/scheduler path. It never calls adapters directly and never writes
conversation progress, consolidation, reflection, memory, or character state.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Executable scope | Implement the reusable dry-run core in `src/kazusa_ai_chatbot/self_cognition/`; use `src/scripts/run_self_cognition_dry_run.py` for manual smoke. | Production logic and smoke commands must survive deletion of `experiments/*`. |
| Reasoning engine | Reuse existing RAG2 and L1/L2/L3 cognition interfaces. | Self-cognition must exercise the current character cognition stack. |
| Storage | Write local self-cognition tracking artifacts and a local attempt ledger only. | The module needs auditability and repeat suppression without adding production DB collections. |
| Trigger source | Use visible/actionable dialog, commitments, progress, pending action state, or bounded follow-up topics. | Reflection artifacts are modifiers only, not triggers. |
| Duplicate suppression | Use action-attempt idempotency based on source identity, due time, target scope, and action kind. | Generated text is not a stable identity. |
| Outbound execution | Use the existing `TaskDispatcher` for `send_message` candidates when live integration is enabled. | This reuses scheduler validation/deduplication and avoids a parallel proactive sender. |
| Config surface | Keep the `SELF_COGNITION_*` surface minimal: enable flag, worker interval, per-tick cap, tracking directory, source/RAG budgets, and trigger-source flags. | These are deploy-time runtime controls; route names, statuses, and artifact names stay as module constants. |
| ICD placement | Move the canonical ICD into `src/kazusa_ai_chatbot/self_cognition/README.md`. | Runtime module contracts should live with the module once it exists. |

## Configuration Contract

Follow the current `src/kazusa_ai_chatbot/config.py` pattern: module-level
constants, environment-variable parsing, fail-fast validation, and focused
`tests/test_config.py` coverage. Do not introduce a new settings framework.

Add only these central config items:

| Constant | Env var | Default | Validation | Purpose |
|---|---|---|---|---|
| `SELF_COGNITION_ENABLED` | `SELF_COGNITION_ENABLED` | `false` | bool string parse | Prevent accidental live activation. When true, service startup starts the idle worker. |
| `SELF_COGNITION_WORKER_INTERVAL_SECONDS` | same | `3600` | positive int | Idle worker sleep between ticks. |
| `SELF_COGNITION_MAX_CASES_PER_TICK` | same | `3` | positive int | Hard cap on self-cognition cases processed in one worker tick. |
| `SELF_COGNITION_TRACKING_DIR` | same | `self_cognition_runs` | non-empty string | Local artifact and attempt-ledger root for the self-cognition tracking system. |
| `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT` | same | `4000` | positive int | Max source packet text passed into cognition. |
| `SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT` | same | `4000` | positive int | Max RAG evidence text included in cognition. |

Add these bool trigger flags with default `true`: `SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED`, `SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED`, `SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED`, `SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED`, `SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED`, and `SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED`.

Do not add config for route names, artifact filenames, case names, action
statuses, idempotency identity, RAG supervisor call count, cognition call
count, dialog render call count, topic limit, route priority, or dispatcher
tool names. Those are fixed contracts or named module constants, not operator
settings.

Trigger flags control source collector eligibility only. They must not override
cognition's social decision to send, hold, search, or stay silent. Do not add a
reflection trigger flag; promoted reflection remains a modifier only after
another valid trigger exists.

## Contracts And Data Shapes

Artifacts are local files produced by the production module, live worker, or
dry-run harness. They are not production database schemas. In live mode, the
dispatcher may also create normal `scheduled_events` rows after the
self-cognition action candidate exists.

### Public Module Interface

Implement these functions under `kazusa_ai_chatbot.self_cognition`. Existing
code may use private helpers, but these public functions must exist and tests
must target them.

- `tracking.build_idempotency_key(source_kind, source_id, due_at, target_scope, action_kind) -> str`
  - Build a stable key from source identity, due time, normalized target
    scope, and action kind.
  - Do not include generated message text.
- `tracking.build_trigger_record(case) -> dict`
  - Return the `self_cognition_trigger_record.json` shape.
- `tracking.build_run_record(case, trigger_record, selected_route, budget) -> dict`
  - Return the `self_cognition_run_record.json` shape.
- `tracking.build_route_effect(run_record, route, consumer, effect_summary, next_topic=None) -> dict`
  - Return the `self_cognition_route_effect.json` shape with
    `production_write=False`.
- `tracking.classify_route(case, cognition_output, action_attempt=None) -> str`
  - Derive the selected route from case metadata, cognition output, and optional
    action-attempt state.
  - Do not force `action_candidate` merely because a commitment is past due.
    If cognition output does not select outward contact, return the silent,
    audit, or progress route indicated by cognition/case state.
  - If cognition selects outward contact and `action_attempt.status` is
    `duplicate_suppressed`, return `action_candidate` while leaving candidate
    creation suppressed.
- `tracking.build_action_attempt(case, trigger_record, existing_attempts) -> dict`
  - Return `candidate`, `held`, `duplicate_suppressed`, or
    `closed_no_action`.
  - Use existing local attempts only; do not query production storage.
- `tracking.build_action_candidate(case, action_attempt, text) -> dict | None`
  - Return a `send_message`-shaped candidate only for a non-duplicate outward
    action attempt.
  - Always set `production_handoff=False`.
- `handoff.build_raw_tool_call(action_candidate) -> RawToolCall`
  - Convert a non-empty self-cognition action candidate into the existing
    dispatcher `send_message` tool-call shape.
- `handoff.dispatch_action_candidate(case, action_attempt, action_candidate, dispatcher, now) -> dict`
  - Call `TaskDispatcher.dispatch(...)` exactly once for one candidate and
    return a local dispatch result artifact.
  - Do not call scheduler or adapters directly.
- `runner.run_self_cognition_case(case, output_dir, rag_client=None, cognition_client=None) -> dict`
  - Orchestrate one dry-run case and return artifact names to paths.
  - Optional clients are test seams; production defaults must call existing
    RAG2 and shared cognition interfaces.
- `worker.run_self_cognition_worker_tick(...) -> SelfCognitionWorkerResult`
  - Collect eligible cases, load local prior attempts, run cases, optionally
    dispatch non-duplicate candidates, and append local attempt-ledger rows.
- `worker.start_self_cognition_worker(...)` and `worker.stop_self_cognition_worker(...)`
  - Own the process-local service worker handle.
- `artifacts.write_tracking_artifacts(output_dir, artifacts) -> dict[str, str]`
  - Write only files under `output_dir`.
  - Return artifact names to paths.
- `artifacts.read_action_attempt_ledger(root) -> list[dict]`
  - Read the local attempt ledger for duplicate suppression.
- `artifacts.append_action_attempt_ledger(root, attempt) -> None`
  - Append one local attempt row after a case creates or suppresses an outward
    action.

`self_cognition_trigger_record.json`:

```python
{
    "trigger_id": str,
    "trigger_kind": str,
    "target_scope": {
        "platform": str,
        "platform_channel_id": str,
        "channel_type": str,
        "user_id": str | None,
    },
    "source_refs": list[dict],
    "semantic_due_state": str | None,
    "actionability": str,
    "status": str,
}
```

`self_cognition_run_record.json`:

```python
{
    "run_id": str,
    "trigger_id": str,
    "idle_timestamp": str,
    "output_mode": "silent" | "preview" | "scheduled_action_request",
    "selected_route": str,
    "status": str,
    "evidence_refs": list[dict],
    "budget": {
        "rag_calls": int,
        "cognition_calls": int,
        "dialog_calls": int,
        "topic_limit": int,
    },
}
```

`self_cognition_route_effect.json`:

```python
{
    "run_id": str,
    "route": str,
    "consumer": str,
    "production_write": False,
    "effect_summary": str,
    "next_topic": dict | None,
}
```

`self_cognition_action_attempt.json`:

```python
{
    "attempt_id": str,
    "run_id": str,
    "trigger_id": str,
    "source_kind": str,
    "source_id": str,
    "target_scope": dict,
    "action_kind": str,
    "due_at": str | None,
    "idempotency_key": str,
    "status": (
        "candidate"
        | "held"
        | "duplicate_suppressed"
        | "closed_no_action"
    ),
}
```

`self_cognition_action_candidate.json`:

```python
{
    "attempt_id": str,
    "target_platform": str,
    "target_channel": str,
    "target_channel_type": str,
    "text": str,
    "execute_at": str | None,
    "dispatch_shape": "send_message",
    "production_handoff": False,
}
```

`self_cognition_dispatch_result.json`:

```python
{
    "attempt_id": str,
    "idempotency_key": str,
    "production_handoff": bool,
    "dispatcher_called": bool,
    "scheduled_event_ids": list[str],
    "rejections": list[str],
    "status": "not_requested" | "accepted" | "rejected",
}
```

Local action-attempt ledger rows use the action-attempt shape plus optional
`dispatch_status`, `scheduled_event_ids`, and `recorded_at` fields. The ledger
exists only under `SELF_COGNITION_TRACKING_DIR`.

Forbidden compatibility shapes:

- Do not preserve `self_cognition_consolidation_candidate.json` as the primary
  route artifact. Use `self_cognition_route_effect.json`.
- Do not represent an outward candidate only as free text. It must be tied to
  an action attempt.
- Do not infer duplicate identity from generated message text.

### Dry-Run Case Contract

Implement these case names exactly in dry-run case files and tests. The module
must not ship built-in synthetic fixtures for these cases; callers provide a
case file.

| Case | Trigger kind | Due state | Expected primary route | Required artifacts |
|---|---|---|---|---|
| `commitment_before_due` | `active_commitment_due_check` | `future_due` | `progress_maintenance` | trigger, run, cognition input/output, route effect, trace |
| `commitment_past_due` | `active_commitment_due_check` | `past_due` | `action_candidate` only if cognition selects outward contact; otherwise record the silent/audit/progress route as an observation | trigger, run, cognition input/output, route effect, action attempt and action candidate only when contact is selected, trace |
| `commitment_duplicate_tick` | `active_commitment_due_check` | `past_due` | `action_candidate` with duplicate suppression only if cognition selects outward contact; otherwise record the silent/audit/progress route as an observation | trigger, run, route effect, synthetic prior attempt fixture, action attempt when contact is selected, trace; no new action candidate |
| `private_no_action` | `recent_direct_dialog_review` | none | `audit_only` or `progress_maintenance` | trigger, run, cognition input/output, route effect, trace |
| `group_noise_rejected` | `group_chat_trigger_review` | none | `silent_no_write` or `audit_only` | trigger, run, route effect, trace; no RAG, no action attempt |
| `topic_rag_followup` | `bounded_followup_topic` | none | `action_candidate`, `progress_maintenance`, or `audit_only` after RAG | trigger, run, RAG request/output, cognition input/output, route effect, trace |

Dry-run command contract:

```text
venv\Scripts\python -m scripts.run_self_cognition_dry_run --case-file <path> --output-dir <path>
```

The command must reject missing files, malformed case files, and unknown case
names with a nonzero exit and a clear message. It must never default to a
production write mode.

The `commitment_duplicate_tick` fixture must supply a synthetic prior
`candidate` or `sent` attempt for the same idempotency key. The command must
not depend on a previous dry-run or production storage to create duplicate
state.

## LLM Call And Context Budget

- Production live `/chat` call count: unchanged.
- Deterministic tests: zero LLM calls.
- Dry-run single case without RAG: at most one background cognition call and,
  only when cognition selects outward contact without explicit
  `[ACTION_CANDIDATE]` text, at most one dialog rendering call.
- Dry-run single case with RAG: at most one background RAG2 supervisor invocation
  and one background cognition call. It may also use the same conditional
  single dialog rendering call. Internal RAG2 helper-agent calls or continuation
  iterations are governed by existing RAG2 caps and do not count as additional
  self-cognition supervisor invocations.
- Dry-run multi-case commands must execute cases sequentially and write a separate
  output directory per case.
- Real LLM verification must run one case at a time and inspect output before
  running the next case.
- Default context cap: 50k tokens.
- Source packet budget: use `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`.
- RAG result budget: use `SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT`.
- Drop policy: if exported evidence exceeds budget, keep source refs and the
  highest-signal bounded excerpts; do not pass raw unbounded histories to the
  LLM.
- No retry loops are authorized for ordinary cognition, dialog, or RAG failures
  in this dry-run slice. A failed LLM/RAG call records a failed run artifact and
  stops that case.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Add only the `SELF_COGNITION_*` constants in `Configuration Contract`.
- `src/kazusa_ai_chatbot/service.py`
  - Start and stop the self-cognition worker only when
    `SELF_COGNITION_ENABLED=true`, after dispatcher runtime is configured.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Add read-only query support only if the worker needs active-commitment
    discovery that is not available through an existing public helper.
- `src/kazusa_ai_chatbot/self_cognition/__init__.py`
  - Export the public dry-run, tracking, handoff, and worker interfaces.
- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add typed models and literal constants for trigger, run, route effect,
    action attempt, action candidate, case metadata, routes, statuses, artifact
    filenames, and fixed internal limits.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Build source packets, semantic due-state labels, route-effect candidates,
    and bounded evidence projections.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Own artifact construction, idempotency key construction, action-attempt
    status transitions, route classification, and duplicate suppression.
- `src/kazusa_ai_chatbot/self_cognition/artifacts.py`
  - Write local artifacts under the requested output directory and maintain the
    local action-attempt ledger.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Orchestrate one dry-run case, write the required artifacts, call RAG2 only
    when needed, call the shared cognition graph, conditionally call dialog
    rendering for candidate wording, and suppress duplicates using local
    action-attempt state.
- `src/kazusa_ai_chatbot/self_cognition/handoff.py`
  - Convert action candidates to `RawToolCall` and dispatch through the
    existing `TaskDispatcher`.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Collect bounded visible/actionable source cases for the worker. Start with
    active commitments; do not use reflection as a trigger.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Own worker tick orchestration, local ledger loading, per-tick limits, and
    service start/stop handle.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Own the canonical ICD, controlled-handoff contract, public interfaces,
    commands, artifacts, and supported case schema.
- `src/scripts/run_self_cognition_dry_run.py`
  - Add a manual dry-run command with `--case-file` and `--output-dir`.
- `tests/test_config.py`
  - Add focused config coverage for the minimal self-cognition settings.
- `tests/test_reflection_cycle_stage1c_service.py` or a new focused service
  test file
  - Cover service startup/shutdown wiring for the self-cognition worker.
- `tests/test_self_cognition_framing.py`
  - Keep or update framing tests so the source packet does not bias toward
    passive waiting.
- `development_plans/archive/superseded/self_cognition_tracking_icd.md`
  - Convert it to a pointer or transition note after the module README owns
    the canonical ICD.

### Create

- `tests/test_self_cognition_tracking.py`
  - Cover tracking artifacts, idempotency, duplicate suppression, route
    classification, and forbidden production handoff flags.
- `tests/test_self_cognition_dry_run_cli.py`
  - Cover the dry-run command's case-file loading, output directory behavior,
    and rejection of unsupported case names.
- `tests/test_self_cognition_integration.py`
  - Cover worker tick behavior, ledger repeat suppression, dispatcher handoff,
    and disabled/busy skip behavior without LLM, DB, scheduler, or adapters.

## Test Contract

Add or preserve these exact deterministic tests. They must not call the LLM,
RAG, database, scheduler, dispatcher, or adapters.

Exact tests:

- `tests/test_config.py`: `test_self_cognition_config_defaults_are_minimal`,
  `test_self_cognition_char_limits_fail_fast_when_invalid`,
  `test_self_cognition_trigger_flags_default_true_and_parse_false`.
- `tests/test_self_cognition_framing.py`:
  `test_self_cognition_framing_presents_agency_without_silence_bias`.
- `tests/test_self_cognition_tracking.py`:
  `test_build_idempotency_key_ignores_generated_text`,
  `test_build_idempotency_key_changes_when_due_occurrence_changes`,
  `test_classify_route_returns_action_candidate_when_cognition_selects_contact`,
  `test_classify_route_does_not_force_action_for_past_due_silence`,
  `test_before_due_commitment_writes_progress_route_without_action_candidate`,
  `test_past_due_contact_decision_writes_action_attempt_and_candidate_without_handoff`,
  `test_duplicate_contact_decision_suppresses_same_due_occurrence`,
  `test_duplicate_tick_fixture_supplies_prior_attempt_state`,
  `test_group_noise_rejected_without_rag_or_action`,
  `test_artifact_writer_uses_expected_file_names`,
  `test_dry_run_command_rejects_unknown_case_name`.
- `tests/test_self_cognition_integration.py`:
  `test_worker_tick_loads_ledger_before_running_case`,
  `test_worker_tick_dispatches_candidate_through_task_dispatcher`,
  `test_worker_tick_records_dispatch_rejection_without_adapter_send`,
  `test_worker_tick_suppresses_duplicate_due_occurrence_from_ledger`,
  `test_worker_tick_defers_when_primary_interaction_is_busy`,
  `test_active_commitment_source_builds_due_case_from_memory_unit`.
- Service wiring test:
  `test_lifespan_starts_self_cognition_worker_only_when_enabled`.

### Keep

- `experiments/**`
  - Do not read, import, modify, or depend on this tree. It is planned for
    removal after this plan.
- `development_plans/archive/superseded/self_cognition_reasoning_basis.md`
  - Reference only unless execution discovers a factual correction.
- `development_plans/archive/superseded/self_cognition_loop_architecture.md`
  - Historical reference only.

## Implementation Order

1. Load mandatory skills and reread this plan, the ICD, and the reasoning
   basis.
2. Inspect current files:
   - `src/kazusa_ai_chatbot/config.py`
   - `tests/test_config.py`
   - `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`
   - `src/scripts/run_reflection_cycle_readonly.py`
   - existing `tests/test_self_cognition*.py` files, if any.
3. Add or update the deterministic tests listed in `Test Contract`.
4. Run:
   `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q`
   - Expected before implementation: existing config tests may pass; new
     self-cognition tests fail because the module, config constants, script, or
     artifact writer contracts are missing.
   - Record the result in `Execution Evidence`.
5. Update `src/kazusa_ai_chatbot/config.py`.
   - Add only the constants in `Configuration Contract`.
   - Use existing parsing helpers; add helper code only if required for
     fail-fast validation.
6. Create `src/kazusa_ai_chatbot/self_cognition/`.
   - Add `__init__.py`, `README.md`, `models.py`, `projection.py`,
     `tracking.py`, `artifacts.py`, and `runner.py`.
7. Update `src/kazusa_ai_chatbot/self_cognition/models.py`.
   - Add literal constants for case names, route names, artifact filenames,
     action-attempt statuses, and fixed internal limits.
   - Add typed dictionaries only when tests or public functions use them.
8. Update `src/kazusa_ai_chatbot/self_cognition/projection.py`.
   - Add deterministic semantic labels for due state and group-noise
     rejection.
   - Pass semantic descriptors to cognition; do not pass raw noise metrics or
     unbounded histories.
9. Update `src/kazusa_ai_chatbot/self_cognition/tracking.py`,
   `artifacts.py`, and `runner.py`.
   - Build trigger, run, route effect, route classification, action attempt,
     and action candidate artifacts through `tracking.py`.
   - Write artifacts through `artifacts.write_tracking_artifacts(...)`.
   - Ensure duplicate suppression runs before action candidate creation.
   - Ensure `commitment_past_due` does not force action when cognition selects
     silence.
   - Ensure all action candidates have `production_handoff=False`.
10. Create `src/scripts/run_self_cognition_dry_run.py`.
   - Add `--case-file`.
   - Add `--output-dir`.
   - Reject missing files, malformed case files, and unknown case names before
     creating output directories.
11. Mandatory pause gate: run real-data dry-run integration trials.
   - Use exported real data to create case files under
     `test_artifacts/self_cognition_cases/`.
   - Include at least one real active or expired promise case and one
     no-action case; inspect cognition output, route effect, action attempt,
     and message candidate.
   - Stop and report artifacts before any future workflow-integration plan.
12. Move the canonical ICD content into
    `src/kazusa_ai_chatbot/self_cognition/README.md`.
13. Update `development_plans/archive/superseded/self_cognition_tracking_icd.md`
    so it points to the module README as canonical.
14. Remove old artifact-name references from production module code, tests,
    script, and docs.
15. Re-run focused deterministic tests.
- They must pass before any dry-run smoke.
16. Run static forbidden-write, no-experiments-dependency, and
    obsolete-artifact greps listed in
    `Verification`.
17. Run dry-run smoke cases one at a time only if local LLM/RAG prerequisites are
    available. If unavailable, record the blocker and do not substitute
    production calls or live sends.
18. Run the independent code review gate.
19. Record execution evidence and leave unfinished stages unchecked if handing
    off.
20. User-approved integration stage: add deterministic tests from
    `tests/test_self_cognition_integration.py` and the service wiring test.
    Run them before implementation and record the expected failures.
21. Add the new minimal config settings:
    `SELF_COGNITION_WORKER_INTERVAL_SECONDS`,
    `SELF_COGNITION_MAX_CASES_PER_TICK`, and
    `SELF_COGNITION_TRACKING_DIR`.
22. Implement local attempt-ledger helpers in `artifacts.py`.
23. Implement `handoff.py` so action candidates become existing dispatcher
    `RawToolCall` objects and `TaskDispatcher.dispatch(...)` is the only
    production handoff.
24. Implement `sources.py` with bounded active-commitment case collection from
    visible/actionable data. Use existing read interfaces where possible; add
    only a read-only DB helper if discovery cannot be done through an existing
    public helper.
25. Implement `worker.py` with a single tick API, start/stop handle, busy
    deferral, per-tick case cap, ledger-backed duplicate suppression, and
    local artifact output directories.
26. Wire `service.py` to start/stop the self-cognition worker only when
    `SELF_COGNITION_ENABLED=true` and after the dispatcher runtime exists.
27. Update the module README and reference ICD pointer for live integration
    side effects and the dispatcher handoff contract.
28. Run focused deterministic tests:
    `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py tests/test_self_cognition_integration.py tests/test_reflection_cycle_stage1c_service.py -q`
29. Run planned real LLM smoke one case at a time using existing data. Inspect
    each output and record whether the live cognition selected silence,
    progress, or action. Do not batch real LLM smoke.
30. Run updated static greps and `git diff --check`.
31. Run a fresh independent code review gate after deterministic and real LLM
    evidence has been inspected.

## Progress Checklist

- [x] Stage 1 - Contract tests added
  - Covers steps 1-4. Files: `tests/test_config.py`,
    `tests/test_self_cognition_framing.py`,
    `tests/test_self_cognition_tracking.py`,
    `tests/test_self_cognition_dry_run_cli.py`.
  - Verify: run the focused pytest command in `Implementation Order` step 4.
  - Evidence/sign-off: record baseline result in `Execution Evidence`.

- [x] Stage 2 - Production module implemented
  - Covers steps 5-10. Files: `src/kazusa_ai_chatbot/config.py`,
    `src/kazusa_ai_chatbot/self_cognition/**`,
    `src/scripts/run_self_cognition_dry_run.py`.
  - Verify: focused deterministic tests pass, including `classify_route`
    contact and silence cases.
  - Evidence/sign-off: record changed files and pytest output.

- [x] Stage 3 - Real-data dry-run integration pause
  - Covers step 11.
  - Verify: run at least one real promise case and one no-action case through
    `src/scripts/run_self_cognition_dry_run.py`.
  - Evidence/sign-off: record input files, outputs, candidate text if any, and
    user review result before any future workflow-integration plan.

- [x] Stage 4 - Documentation and ICD relocation rule updated
  - Covers steps 12-14. Files: `src/kazusa_ai_chatbot/self_cognition/README.md`,
    `development_plans/archive/superseded/self_cognition_tracking_icd.md`.
  - Verify: `rg -n "canonical ICD|SC-TRACKING-ICD-001|kazusa_ai_chatbot.self_cognition" src/kazusa_ai_chatbot/self_cognition/README.md development_plans/archive/superseded/self_cognition_tracking_icd.md`.
  - Verify: `rg -n "self_cognition_consolidation_candidate" src/kazusa_ai_chatbot/self_cognition src/scripts tests` returns no matches; exit code 1 is acceptable.
  - Verify: `rg -n "experiments" src/kazusa_ai_chatbot/self_cognition src/scripts tests/test_self_cognition*.py` returns no matches; exit code 1 is acceptable.
  - Evidence/sign-off: record grep output and doc paths.

- [x] Stage 5 - Verification complete
  - Covers steps 15-17.
  - Verify: run every command in `Verification`.
  - Evidence/sign-off: record command, exit code, and relevant output summary.

- [x] Stage 6 - Independent code review complete
  - Covers step 18.
  - Verify: run the `Independent Code Review` gate.
  - Evidence/sign-off: record findings, fixes, commands rerun, residual risks,
    and approval status.

- [x] Stage 7 - Runtime integration tests and implementation
  - Covers steps 20-27. Files: `src/kazusa_ai_chatbot/self_cognition/**`,
    `src/kazusa_ai_chatbot/config.py`, `src/kazusa_ai_chatbot/service.py`,
    optional read-only DB helper, `tests/test_self_cognition_integration.py`,
    and the service wiring test.
  - Verify: new deterministic tests fail before implementation, then pass.
  - Evidence/sign-off: record red/green outputs and changed files.

- [x] Stage 8 - Deterministic and real LLM integration verification
  - Covers steps 28-30.
  - Verify: run the focused deterministic pytest command, static greps,
    `git diff --check`, and planned real LLM smoke cases one at a time.
  - Evidence/sign-off: record each real LLM input/output judgment, including
    silence as an acceptable observation.

- [x] Stage 9 - Final independent code review and sign-off
  - Covers step 31.
  - Verify: rerun the independent code review gate after integration evidence.
  - Evidence/sign-off: record findings, fixes, reruns, residual risks, and final
    approval status.

## Verification

### Static Greps

- Command:
  `rg -n "save_conversation|call_consolidation_subgraph|mark_scheduled_event|insert_user_memory_units|delete_many|update_many|adapter\\.send" src/kazusa_ai_chatbot/self_cognition src/scripts/run_self_cognition_dry_run.py -g "*.py"`
  - Expected: no matches. Exit code 1 from `rg` is acceptable and means no
    matches.
- Command:
  `rg -n "TaskDispatcher|dispatch\\(" src/kazusa_ai_chatbot/self_cognition -g "*.py"`
  - Expected: matches only the approved dispatcher handoff module or worker
    orchestration, never dry-run CLI code, adapters, scheduler direct calls, or
    consolidation code.
- Command:
  `rg -n "trigger_source=user_message|build_user_message_consolidation_origin" src/kazusa_ai_chatbot/self_cognition src/scripts/run_self_cognition_dry_run.py -g "*.py"`
  - Expected: no matches. Exit code 1 from `rg` is acceptable.
- Command:
  `rg -n "dialog_agent|final_dialog" src/kazusa_ai_chatbot/self_cognition src/scripts/run_self_cognition_dry_run.py -g "*.py"`
  - Expected: matches only the runner's conditional dialog-render fallback;
    dry-run CLI, handoff, worker, adapters, scheduler, and consolidation paths
    must not consume `final_dialog`.
- Command:
  `rg -n "canonical ICD|SC-TRACKING-ICD-001|kazusa_ai_chatbot.self_cognition" src/kazusa_ai_chatbot/self_cognition/README.md development_plans/archive/superseded/self_cognition_tracking_icd.md`
  - Expected: matches proving the module README owns the canonical ICD and the
    reference document points to it.
- Command:
  `rg -n "experiments" src/kazusa_ai_chatbot/self_cognition src/scripts tests/test_self_cognition*.py`
  - Expected: no matches. Exit code 1 from `rg` is acceptable.
- Command:
  `rg -n "self_cognition_consolidation_candidate" src/kazusa_ai_chatbot/self_cognition src/scripts tests`
  - Expected: no matches. Exit code 1 from `rg` is acceptable. The active plan
    may mention the obsolete filename only to forbid it; code, tests, and
    module docs must not keep the old artifact name.
- Command:
  `rg -n "SELF_COGNITION_ENABLED|start_self_cognition_worker|stop_self_cognition_worker" src/kazusa_ai_chatbot/service.py tests`
  - Expected: matches proving service wiring is explicit and test-covered.

### Deterministic Tests

- Command:
  `venv\Scripts\python -m pytest tests/test_config.py -q`
  - Expected: pass.
- Command:
  `venv\Scripts\python -m pytest tests/test_self_cognition_framing.py -q`
  - Expected: pass.
- Command:
  `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q`
  - Expected: pass.
- Command:
  `venv\Scripts\python -m pytest tests/test_self_cognition_integration.py tests/test_reflection_cycle_stage1c_service.py -q`
  - Expected: pass.
- Command:
  `git diff --check`
  - Expected: pass.

### Planned Real LLM Smoke

Run only after deterministic tests pass.

After deterministic integration tests pass, run exported or collected real-data
case files one at a time. Minimum: one active or expired promise case, one
no-action private case, and one dispatcher-handoff simulation where the
candidate handoff is disabled or patched so no real adapter send occurs during
the test run. Record input, cognition output, route effect, action attempt,
dispatch result if any, and message candidate if any. Silence is acceptable if
the artifacts explain the decision.

- Command:
  `venv\Scripts\python -m scripts.run_self_cognition_dry_run --case-file test_artifacts/self_cognition_cases/commitment_before_due.json --output-dir test_artifacts/self_cognition_dry_run/commitment_before_due`
  - Expected: writes trigger, run, cognition input/output, route effect, loop
    trace, and no action candidate.
- Command:
  `venv\Scripts\python -m scripts.run_self_cognition_dry_run --case-file test_artifacts/self_cognition_cases/commitment_past_due.json --output-dir test_artifacts/self_cognition_dry_run/commitment_past_due`
  - Expected: if cognition selects outward contact, writes action attempt and
    action candidate with `production_handoff=false`. If cognition selects
    silence, writes the selected silent/audit/progress route as an observation;
    this is not a smoke-test failure.
- Command:
  `venv\Scripts\python -m scripts.run_self_cognition_dry_run --case-file test_artifacts/self_cognition_cases/commitment_duplicate_tick.json --output-dir test_artifacts/self_cognition_dry_run/commitment_duplicate_tick`
  - Expected: fixture includes a synthetic prior attempt for the same
    idempotency key. If cognition selects outward contact, writes
    duplicate-suppressed action attempt and no new send candidate. If cognition
    selects silence, writes the selected silent/audit/progress route as an
    observation; this is not a smoke-test failure.

If local LLM, RAG, or exported data prerequisites are unavailable, record the
blocker and do not substitute production calls or live sends.

Deterministic tests prove the tracking, idempotency, duplicate-suppression, and
artifact contracts. Dry-run smoke runs are behavioral observations of the current
LLM/RAG framing; they do not prove that self-cognition will always produce
useful agency or that a past-due commitment will always become a send
candidate.

## Independent Plan Review

Audit status: completed during plan tightening. Inputs included the plan
registry, plan-writing references, current plan, reasoning basis, ICD,
`config.py`, and existing module README patterns.

Findings fixed in this plan:

- Split research basis from executable instructions.
- Added mandatory executable-plan sections, public interfaces, exact tests,
  case contract, verification, and evidence gates.
- Moved the target from experiment-owned logic to
  `src/kazusa_ai_chatbot/self_cognition/`.
- Made `src/kazusa_ai_chatbot/self_cognition/README.md` the canonical ICD.
- Added `tracking.classify_route(...)` as the tested route owner.
- Made action-candidate smoke expectations conditional on cognition output.
- Required synthetic prior attempt state for duplicate-tick case files.
- Defined the RAG cap as one RAG2 supervisor invocation.
- Added obsolete-artifact and no-`experiments/*` dependency greps.
- Added per-trigger-source enablement flags, all enabled by default.
- Added a mandatory real-data dry-run pause before any future workflow wiring.

Approval status: approved for dry-run module execution, then amended after user
acceptance for opt-in runtime integration. Placeholder scan, contract
consistency, granularity, and scope-control self-review passed.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused tests, static checks,
  execution evidence, next-stage handoff notes, and path-safe PowerShell
  commands.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The production self-cognition module writes the required dry-run tracking
  artifacts locally.
- Deterministic tests prove source framing, route artifacts, action-attempt
  idempotency, duplicate suppression, and route classification, including the
  rule that a past-due commitment does not force action when cognition selects
  silence.
- Deterministic integration tests prove ledger-backed repeat suppression,
  dispatcher handoff shape, rejected handoff recording, busy deferral, and
  service start/stop wiring.
- Static greps show no forbidden production writes, adapter sends,
  consolidation, reflection, memory, or normal `/chat` path is called from
  self-cognition code.
- Static greps show dispatcher calls are limited to the approved handoff/worker
  boundary.
- Static greps show no dependency on `experiments/*` from the module, script,
  or self-cognition tests.
- Static greps show obsolete artifact names are removed from module code,
  tests, script, and module docs.
- The module README states the canonical ICD, controlled-handoff contract,
  public interfaces, config surface, and supported case schema.
- The reference ICD points to the module README as canonical after
  implementation.
- Stage 3 artifacts show one real promise case and one no-action case.
- Execution evidence records focused tests, static greps, planned real LLM
  smoke status or blocker, and independent code review result after integration.
- Dry-run smoke results are recorded as behavioral observations. A silence outcome
  for `commitment_past_due` or `commitment_duplicate_tick` is not a failure if
  deterministic tracking tests pass.
- Production behavior is unchanged when `SELF_COGNITION_ENABLED=false`.
- When `SELF_COGNITION_ENABLED=true`, the only allowed outward side effect is
  dispatcher/scheduler handling of a non-duplicate self-cognition
  `send_message` candidate selected by shared cognition.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Self-cognition accidentally writes forbidden production state | Keep only dispatcher/scheduler handoff allowed; grep forbidden write paths | Static greps and code review |
| Dry-run module bypasses real cognition | Require existing cognition graph call and framing test | Focused tests and code review |
| Duplicate expired commitment creates repeated sends | Add action-attempt idempotency, local attempt ledger, and duplicate-suppression tests across ticks | Tracking and integration tests |
| LLM context grows unbounded | Enforce source packet and RAG result budgets | Tests or artifact inspection plus code review |
| Accidental dependency on removable `experiments/*` | Forbid imports and add no-experiments grep | Static grep and code review |
| Config surface grows too broad | Limit central config to approved constants and trigger flags | Config tests and code review |
| Module quality is unproven before workflow wiring | Use Stage 3 user-approved artifacts, then run planned real LLM smoke after integration | Stage 3 evidence, Stage 8 smoke, user review |
| ICD remains stranded in development plans after module implementation | Move canonical ICD into module README and leave reference pointer | ICD grep gate |
| Dispatcher handoff bypasses existing delivery controls | Convert to `RawToolCall` and call `TaskDispatcher.dispatch(...)`; never call scheduler or adapters directly | Integration tests and dispatcher grep |

## Execution Evidence

Record evidence here during implementation.

- Stage 1 contract test baseline: 2026-05-13 ran `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q`; exit code 1 during collection because `kazusa_ai_chatbot.self_cognition` does not exist yet. This is the expected pre-implementation failure surface for the new module.
- Stage 2 tracking implementation tests: implemented `src/kazusa_ai_chatbot/self_cognition/`, `src/scripts/run_self_cognition_dry_run.py`, minimal `SELF_COGNITION_*` config, and focused tests. Initial post-implementation run passed `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q` with 51 passed. After review fixes, focused runs passed: config 35 passed, framing 1 passed, tracking/CLI 17 passed. After the follow-up repeat-suppression fix, the combined focused command passed 55 tests.
- Stage 3 real-data dry-run pause: exported real data into `test_artifacts/self_cognition_cases/exports/` and created case files `commitment_past_due_real_673225019.json`, `commitment_past_due_real_673225019_due_missed.json`, and `private_no_action_real_673225019.json`. Ran real dry-run outputs under `test_artifacts/self_cognition_dry_run/commitment_past_due_real_673225019_rerun/`, `test_artifacts/self_cognition_dry_run/private_no_action_real_673225019/`, `test_artifacts/self_cognition_dry_run/commitment_past_due_real_673225019_due_missed/`, and `test_artifacts/self_cognition_dry_run/commitment_past_due_real_673225019_due_missed_rerun/`. Verified the real promise and no-action reruns wrote local artifacts only; both final verified route effects were `audit_only` with `production_write=false`. One intermediate due-missed run produced outward-contact cognition anchors, which exposed the need for route classification to honor existing cognition intent/stance; that fix is implemented and covered by deterministic tests. User accepted the quality on 2026-05-13 and approved continuing into runtime integration.
- Stage 4 documentation and ICD grep: moved canonical ICD content into `src/kazusa_ai_chatbot/self_cognition/README.md` and changed `development_plans/archive/superseded/self_cognition_tracking_icd.md` into a pointer. `rg -n "canonical ICD|SC-TRACKING-ICD-001|kazusa_ai_chatbot.self_cognition" src/kazusa_ai_chatbot/self_cognition/README.md development_plans/archive/superseded/self_cognition_tracking_icd.md` returned the expected module README and pointer matches.
- Stage 5 verification commands: `venv\Scripts\python -m pytest tests/test_config.py -q` passed 35; `venv\Scripts\python -m pytest tests/test_self_cognition_framing.py -q` passed 1; `venv\Scripts\python -m pytest tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q` passed 17 before the follow-up repeat-suppression fix. After that fix, `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py -q` passed 55. Static greps for production writes/sends, normal user-message consolidation path, `experiments`, and obsolete artifact names returned no matches; the ICD grep returned expected matches.
- Dry-run smoke status: real-data smoke was run one case at a time. The initial source packet was too technical and biased the model toward treating the input as system noise; framing was changed to a character-facing idle self-check and rerun. Final verified real-data runs selected `audit_only`/silence; this is recorded as behavioral observation, not a deterministic failure.
- Stage 7 runtime integration: added opt-in service worker wiring behind
  `SELF_COGNITION_ENABLED=false`, local action-attempt ledger read/append,
  active-commitment source collection through read-only production interfaces,
  dispatcher handoff through `TaskDispatcher.dispatch(...)`, and module README
  controlled-handoff ICD updates. Added deterministic integration coverage for
  worker ledger loading, accepted handoff, rejected handoff, same-tick duplicate
  suppression, busy deferral, handoff context mapping, active-commitment source
  projection, and service start/stop wiring. The initial Stage 7 red run failed
  during collection because `sources` did not exist yet; the post-implementation
  focused run passed 11 tests, then 8 integration tests after the review fix.
- Stage 8 deterministic and real LLM verification: final deterministic command
  `venv\Scripts\python -m pytest tests/test_config.py tests/test_self_cognition_framing.py tests/test_self_cognition_tracking.py tests/test_self_cognition_dry_run_cli.py tests/test_self_cognition_integration.py tests/test_reflection_cycle_stage1c_service.py -q`
  passed 73 tests after the post-review dialog-budget and active-commitment
  query fixes. Real LLM smoke was rerun one case at a time under
  `test_artifacts/self_cognition_integration_smoke/2026-05-13_run_03/`.
  Cases: `commitment_past_due_0930_resolved_by_later_chat_673225019`,
  `private_recent_idle_673225019`, `topic_rag_followup_graphrag_54369546`, and
  `group_noise_recent_54369546`. The due-missed commitment selected
  `action_candidate`, used one dialog render call, and wrote only a local
  candidate with `production_handoff=false`. The private and topic cases chose
  `audit_only`; the topic case invoked one RAG2 supervisor call before
  cognition. The group-noise case short-circuited with no RAG, cognition, or
  dialog calls.
- Stage 8 static verification: `py_compile` passed for the changed production
  and test files. Forbidden-write/send grep over `src/kazusa_ai_chatbot/self_cognition`
  and the dry-run CLI returned no matches. No-`experiments/*`, obsolete artifact,
  and normal user-message consolidation path greps returned no matches in
  self-cognition code/tests. Dialog grep matched the runner's conditional
  rendering fallback only. Dispatcher grep matched only `handoff.py` and
  `worker.py`. `git diff --check` exited 0.
- Stage 9 independent code review: initial active-agent review reread the
  plan, style rules, local-LLM architecture rule, no-pre/post-processing rule,
  and inspected the integration diff after tests and real LLM smoke. Findings
  fixed: worker ledger was originally loaded only once per tick, so same-tick
  duplicate cases could miss newly recorded attempts; the worker now rereads
  the ledger before each case and has a deterministic same-tick duplicate test.
  The worker loop now catches character-profile-provider failures inside the
  process-boundary handler. The cognition packet no longer labels live-capable
  runs with the old dry-run latch; it uses `local_tracking`. The
  action-candidate route-effect text and README dispatch result schema were
  updated to match the controlled-handoff contract. External review on
  2026-05-13 then found hidden `final_dialog` usage, undocumented dialog call
  budget, active-commitment query prioritization risk, a dead timestamp
  parameter, and two test-name mismatches. Post-review fixes removed the
  `final_dialog` string split, documented and counted the conditional dialog
  render call, used `current_timestamp` with parsed due timestamps to
  prioritize due active commitments, and renamed the integration tests to match
  the plan contract.
- Residual risks: real LLM behavior remains stochastic and the final verified
  real-data promise rerun chose silence; useful agency quality is not proven by
  deterministic gates. Live behavior remains disabled by default. When enabled,
  the worker can create normal dispatcher/scheduler effects only for
  non-duplicate `send_message` candidates selected by shared cognition. The
  current plan still does not write conversation progress, consolidation,
  reflection, memory, Cache2, adapter output, or character state.
