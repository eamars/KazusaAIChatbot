# self cognition agency loop plan

## Summary

- Goal: Implement the next self-cognition slice as a production-owned module
  under `src/kazusa_ai_chatbot/self_cognition/`. The module must reuse the
  current RAG2 and L1/L2/L3 cognition interfaces, write local dry-run artifacts,
  and prove trigger, route, action-attempt, duplicate-suppression, and
  no-production-handoff behavior without depending on `experiments/*`.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `database-data-pull` when using real exported data, and `cjk-safety` before
  editing Python files that contain CJK text.
- Overall cutover strategy: compatible. This adds a production module and
  minimal config, but must not wire self-cognition into live runtime behavior.
- Highest-risk areas: accidental production writes or sends, bypassing the
  shared cognition stack, weak duplicate suppression for expired commitments,
  reflection artifacts becoming triggers, and broad LLM/RAG context growth.
- Acceptance criteria: artifacts prove the tracking contract; focused tests and
  static gates pass; independent review passes.

## Context

Self-cognition is intended to become an idle agency loop, but this slice is not
production autonomy. It adds a production module with no live scheduler,
dispatcher, adapter, durable storage, or consolidation integration. Dry-run and
smoke commands must use the module or `src/scripts/`; deleting `experiments/*`
must not break the module, tests, smoke, or docs.

Reference documents: `development_plans/reference/designs/self_cognition_reasoning_basis.md`,
`development_plans/reference/designs/self_cognition_tracking_icd.md`, and
`development_plans/reference/designs/self_cognition_loop_architecture.md`.
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
- Modify production code only for the new `self_cognition` module, central
  config constants, and tests named in `Change Surface`.
- Do not write production database rows, scheduler rows, conversation history,
  conversation progress, user memory, character state, reflection state, or
  adapter output.
- Do not call dispatcher, scheduler mutation, adapter send, production
  consolidation persistence, or normal `/chat` service entrypoints.
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
- Preserve existing production RAG2 and cognition entrypoints as imported
  engines only.
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
- Do not wire the module into production dispatcher, scheduler, adapter,
  conversation-history writes, conversation-progress writes, consolidation,
  reflection, memory writes, or Cache2 mutation.
- Do not add cross-platform proactive fanout, media sends, new tools beyond
  `send_message`, or external orchestrator frameworks.
- Do not mutate personality or stable character state from hourly
  self-cognition.
- Do not implement review UI, operator approval UI, or runtime auto-send mode.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Production runtime | compatible | Preserve existing behavior. Do not wire self-cognition into live `/chat`, scheduler, dispatcher, adapters, consolidation, progress, or reflection. |
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
  `src/kazusa_ai_chatbot/config.py`, `src/scripts/run_self_cognition_dry_run.py`,
  `tests/test_self_cognition_*.py`, `tests/test_config.py`, this plan, and the
  reference docs listed in `Change Surface` is forbidden unless the plan is
  updated first.
- If the plan and code disagree, preserve the plan intent and record the
  discrepancy before changing behavior.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

After this plan is executed, `src/kazusa_ai_chatbot/self_cognition/` exposes a
reusable dry-run self-cognition core. The module CLI/script can run selected
idle self-cognition case files and produce local tracking artifacts that show:

- why the idle run started;
- what visible evidence and semantic due/actionability state were used;
- whether RAG2 was called and what bounded evidence it returned;
- what shared cognition input/output was used;
- which route was selected;
- which action attempt or outbox candidate was created, held, or suppressed;
- that no production handoff, production write, scheduler insert, dispatcher
  dispatch, or adapter send happened.

The target state is a production-owned dry-run module. Production behavior
remains unchanged because no live worker, dispatcher, scheduler, adapter, or
persistence integration is wired.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Executable scope | Implement the reusable dry-run core in `src/kazusa_ai_chatbot/self_cognition/`; use `src/scripts/run_self_cognition_dry_run.py` for manual smoke. | Production logic and smoke commands must survive deletion of `experiments/*`. |
| Reasoning engine | Reuse existing RAG2 and L1/L2/L3 cognition interfaces. | Self-cognition must exercise the current character cognition stack. |
| Storage | Write local dry-run artifacts only. | Durable storage and production projections require separate approved plans. |
| Trigger source | Use visible/actionable dialog, commitments, progress, pending action state, or bounded follow-up topics. | Reflection artifacts are modifiers only, not triggers. |
| Duplicate suppression | Use action-attempt idempotency based on source identity, due time, target scope, and action kind. | Generated text is not a stable identity. |
| Outbound execution | Emit handoff-shaped candidates only. | Real dispatcher/scheduler handoff is deferred until tracking quality is proven. |
| Config surface | Add only a small set of `SELF_COGNITION_*` constants to `config.py`; use named module constants for fixed internal limits. | Avoid magic numbers without turning every internal value into an env setting. |
| ICD placement | Move the canonical ICD into `src/kazusa_ai_chatbot/self_cognition/README.md`. | Runtime module contracts should live with the module once it exists. |

## Configuration Contract

Follow the current `src/kazusa_ai_chatbot/config.py` pattern: module-level
constants, environment-variable parsing, fail-fast validation, and focused
`tests/test_config.py` coverage. Do not introduce a new settings framework.

Add only these central config items:

| Constant | Env var | Default | Validation | Purpose |
|---|---|---|---|---|
| `SELF_COGNITION_ENABLED` | `SELF_COGNITION_ENABLED` | `false` | bool string parse | Prevent accidental live activation when a future worker is wired. This plan does not wire the worker. |
| `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT` | same | `16000` | positive int | Max source packet text passed into cognition. |
| `SELF_COGNITION_RAG_EVIDENCE_CHAR_LIMIT` | same | `16000` | positive int | Max RAG evidence text included in cognition. |

Add these bool trigger flags with default `true`: `SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED`, `SELF_COGNITION_TRIGGER_CONVERSATION_PROGRESS_ENABLED`, `SELF_COGNITION_TRIGGER_RECENT_DIRECT_DIALOG_ENABLED`, `SELF_COGNITION_TRIGGER_PENDING_OUTBOX_ENABLED`, `SELF_COGNITION_TRIGGER_BOUNDED_TOPIC_FOLLOWUP_ENABLED`, and `SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED`.

Do not add config for route names, artifact filenames, case names, action
statuses, idempotency identity, RAG supervisor call count, cognition call
count, topic limit, or production handoff. Those are fixed contracts or named
module constants, not operator settings.

Trigger flags control source collector eligibility only. They must not override
cognition's social decision to send, hold, search, or stay silent. Do not add a
reflection trigger flag; promoted reflection remains a modifier only after
another valid trigger exists.

## Contracts And Data Shapes

Artifacts are local dry-run files produced by the production module or dry-run
harness. They are not production database schemas.

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
- `runner.run_self_cognition_case(case, output_dir, rag_client=None, cognition_client=None) -> dict`
  - Orchestrate one dry-run case and return artifact names to paths.
  - Optional clients are test seams; production defaults must call existing
    RAG2 and shared cognition interfaces.
- `artifacts.write_tracking_artifacts(output_dir, artifacts) -> dict[str, str]`
  - Write only files under `output_dir`.
  - Return artifact names to paths.

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
- Dry-run single case without RAG: at most one background cognition call.
- Dry-run single case with RAG: at most one background RAG2 supervisor invocation
  and one background cognition call. Internal RAG2 helper-agent calls or
  continuation iterations are governed by existing RAG2 caps and do not count
  as additional self-cognition supervisor invocations.
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
- No retry loops are authorized for ordinary cognition or RAG failures in this
  dry-run slice. A failed LLM/RAG call records a failed run artifact and stops
  that case.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Add only the `SELF_COGNITION_*` constants in `Configuration Contract`.
- `src/kazusa_ai_chatbot/self_cognition/__init__.py`
  - Export the public dry-run and tracking interfaces.
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
  - Write local dry-run artifacts under the requested output directory only.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Orchestrate one dry-run case, write the required artifacts, call RAG2 only
    when needed, call the shared cognition graph, and suppress duplicates using
    local action-attempt state.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Own the canonical ICD, no-production-write contract, public interfaces,
    commands, artifacts, and supported case schema.
- `src/scripts/run_self_cognition_dry_run.py`
  - Add a manual dry-run command with `--case-file` and `--output-dir`.
- `tests/test_config.py`
  - Add focused config coverage for the minimal self-cognition settings.
- `tests/test_self_cognition_framing.py`
  - Keep or update framing tests so the source packet does not bias toward
    passive waiting.
- `development_plans/reference/designs/self_cognition_tracking_icd.md`
  - Convert it to a pointer or transition note after the module README owns
    the canonical ICD.

### Create

- `tests/test_self_cognition_tracking.py`
  - Cover tracking artifacts, idempotency, duplicate suppression, route
    classification, and forbidden production handoff flags.
- `tests/test_self_cognition_dry_run_cli.py`
  - Cover the dry-run command's case-file loading, output directory behavior,
    and rejection of unsupported case names.

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

### Keep

- `experiments/**`
  - Do not read, import, modify, or depend on this tree. It is planned for
    removal after this plan.
- `development_plans/reference/designs/self_cognition_reasoning_basis.md`
  - Reference only unless execution discovers a factual correction.
- `development_plans/reference/designs/self_cognition_loop_architecture.md`
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
13. Update `development_plans/reference/designs/self_cognition_tracking_icd.md`
    so it points to the module README as canonical.
14. Remove old artifact-name references from production module code, tests,
    script, and docs.
15. Re-run focused deterministic tests.
- They must pass before any dry-run smoke.
16. Run static no-production-write, no-experiments-dependency, and
    obsolete-artifact greps listed in
    `Verification`.
17. Run dry-run smoke cases one at a time only if local LLM/RAG prerequisites are
    available. If unavailable, record the blocker and do not substitute
    production calls or live sends.
18. Run the independent code review gate.
19. Record execution evidence and leave unfinished stages unchecked if handing
    off.

## Progress Checklist

- [ ] Stage 1 - Contract tests added
  - Covers steps 1-4. Files: `tests/test_config.py`,
    `tests/test_self_cognition_framing.py`,
    `tests/test_self_cognition_tracking.py`,
    `tests/test_self_cognition_dry_run_cli.py`.
  - Verify: run the focused pytest command in `Implementation Order` step 4.
  - Evidence/sign-off: record baseline result in `Execution Evidence`.

- [ ] Stage 2 - Production module implemented
  - Covers steps 5-10. Files: `src/kazusa_ai_chatbot/config.py`,
    `src/kazusa_ai_chatbot/self_cognition/**`,
    `src/scripts/run_self_cognition_dry_run.py`.
  - Verify: focused deterministic tests pass, including `classify_route`
    contact and silence cases.
  - Evidence/sign-off: record changed files and pytest output.

- [ ] Stage 3 - Real-data dry-run integration pause
  - Covers step 11.
  - Verify: run at least one real promise case and one no-action case through
    `src/scripts/run_self_cognition_dry_run.py`.
  - Evidence/sign-off: record input files, outputs, candidate text if any, and
    user review result before any future workflow-integration plan.

- [ ] Stage 4 - Documentation and ICD relocation rule updated
  - Covers steps 12-14. Files: `src/kazusa_ai_chatbot/self_cognition/README.md`,
    `development_plans/reference/designs/self_cognition_tracking_icd.md`.
  - Verify: `rg -n "canonical ICD|SC-TRACKING-ICD-001|kazusa_ai_chatbot.self_cognition" src/kazusa_ai_chatbot/self_cognition/README.md development_plans/reference/designs/self_cognition_tracking_icd.md`.
  - Verify: `rg -n "self_cognition_consolidation_candidate" src/kazusa_ai_chatbot/self_cognition src/scripts tests` returns no matches; exit code 1 is acceptable.
  - Verify: `rg -n "experiments" src/kazusa_ai_chatbot/self_cognition src/scripts tests/test_self_cognition*.py` returns no matches; exit code 1 is acceptable.
  - Evidence/sign-off: record grep output and doc paths.

- [ ] Stage 5 - Verification complete
  - Covers steps 15-17.
  - Verify: run every command in `Verification`.
  - Evidence/sign-off: record command, exit code, and relevant output summary.

- [ ] Stage 6 - Independent code review complete
  - Covers step 18.
  - Verify: run the `Independent Code Review` gate.
  - Evidence/sign-off: record findings, fixes, commands rerun, residual risks,
    and approval status.

## Verification

### Static Greps

- Command:
  `rg -n "TaskDispatcher|save_conversation|call_consolidation_subgraph|mark_scheduled_event|insert_user_memory_units|delete_many|update_many|adapter\\.send|dispatch\\(" src/kazusa_ai_chatbot/self_cognition src/scripts/run_self_cognition_dry_run.py -g "*.py"`
  - Expected: no matches. Exit code 1 from `rg` is acceptable and means no
    matches.
- Command:
  `rg -n "trigger_source=user_message|build_user_message_consolidation_origin|final_dialog" src/kazusa_ai_chatbot/self_cognition src/scripts/run_self_cognition_dry_run.py -g "*.py"`
  - Expected: no matches. Exit code 1 from `rg` is acceptable.
- Command:
  `rg -n "canonical ICD|SC-TRACKING-ICD-001|kazusa_ai_chatbot.self_cognition" src/kazusa_ai_chatbot/self_cognition/README.md development_plans/reference/designs/self_cognition_tracking_icd.md`
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

### Dry-Run CLI Smoke

Run only after deterministic tests pass.

Before any future workflow integration, run exported real-data case files:
minimum one active or expired promise case and one no-action case. Record input,
cognition output, route effect, action attempt, and message candidate if any.
Silence is acceptable if the artifacts explain the decision.

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

Approval status: approved for dry-run module execution. Placeholder scan,
contract consistency, granularity, and scope-control self-review passed.

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
- Static greps show no production write, scheduler, dispatcher, adapter,
  consolidation, or normal `/chat` path is called from self-cognition code.
- Static greps show no dependency on `experiments/*` from the module, script,
  or self-cognition tests.
- Static greps show obsolete artifact names are removed from module code,
  tests, script, and module docs.
- The module README states the canonical ICD, no-production-write contract,
  public interfaces, config surface, and supported case schema.
- The reference ICD points to the module README as canonical after
  implementation.
- Stage 3 artifacts show one real promise case and one no-action case.
- Execution evidence records focused tests, static greps, dry-run smoke status or
  blocker, and independent code review result.
- Dry-run smoke results are recorded as behavioral observations. A silence outcome
  for `commitment_past_due` or `commitment_duplicate_tick` is not a failure if
  deterministic tracking tests pass.
- Production behavior is unchanged.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dry-run module accidentally writes production state | Keep all writes local and grep for forbidden production write paths | Static greps and code review |
| Dry-run module bypasses real cognition | Require existing cognition graph call and framing test | Focused tests and code review |
| Duplicate expired commitment creates repeated sends | Add action-attempt idempotency and duplicate-suppression tests | Tracking tests and duplicate dry-run case |
| LLM context grows unbounded | Enforce source packet and RAG result budgets | Tests or artifact inspection plus code review |
| Accidental dependency on removable `experiments/*` | Forbid imports and add no-experiments grep | Static grep and code review |
| Config surface grows too broad | Limit central config to approved constants and trigger flags | Config tests and code review |
| Module quality is unproven before workflow wiring | Pause for real-data dry-run trial | Stage 3 evidence and user review |
| ICD remains stranded in development plans after module implementation | Move canonical ICD into module README and leave reference pointer | ICD grep gate |

## Execution Evidence

Record evidence here during implementation.

- Stage 1 contract test baseline:
- Stage 2 tracking implementation tests:
- Stage 3 real-data dry-run pause:
- Stage 4 documentation and ICD grep:
- Stage 5 verification commands:
- Dry-run smoke status:
- Independent code review:
- Residual risks:
