# multi source cognition architecture stage 08 internal thought cognition dry run plan

## Summary

- Goal: Add an internal-thought dry-run path that builds
  `CognitiveEpisode(trigger_source=internal_thought)` from private cognition
  residue and optional action-latch input without public output,
  conversation-history pollution, durable writes, scheduler dispatch, adapter
  sends, or RAG broadening.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for current `/chat`; bigbang
  dry-run-only entrypoint for internal-thought cognition. No recursive loop,
  dialog, delivery, consolidation, persistence writes, scheduler dispatch, or
  adapter sends are enabled.
- Highest-risk areas: private thought leaking into public prompts or history,
  adding an unbounded loop, reusing reflection semantics for a different
  trigger, generated thought writing memory, prompt byte drift, and action
  latch semantics being misread as permission to execute.
- Acceptance criteria: private residue, public scene residue, action-latch,
  episode, and audit contracts exist; internal-thought prompt selection is
  source-aware and audit-only; the dry-run runner performs at most one injected
  cognition subgraph call; `/chat`, Stage 03, Stage 06, and Stage 07 gates
  still pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
Lifecycle: Stage 08 is approved for execution from post-Stage-07 `main`.
Stage 07 is completed in the parent ledger and registry.

## Context

Stage 07 introduced the first non-chat dry-run entrypoint for promoted
reflection artifacts. Stage 08 applies the same source-aware pattern to private
internal thought and action residue. This stage does not approve autonomous
loops, proactive output, action execution, or durable writes.

The experimental cognition reference contains monologue and action-latch
concepts, but production code must not import experiment modules. This stage
translates only the contract idea: a private internal stimulus can enter shared
cognition in a single dry-run pass and produce an audit record.

## Stage Handoff

### From Stage 07

Stage 08 starts from completed Stage 07 artifacts:

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`
- reflection dry-run prompt selector variant
- audit-only dry-run return shape
- evidence that Stage 07 introduced no service, dialog, RAG, consolidation,
  persistence, scheduler, adapter, or transport wiring
- parent ledger row for `stage_07` set to `completed`
- registry row for Stage 07 set to `completed | completed`

Stage 07 evidence carried into Stage 08:

- Branch: `stage-07-reflection-cognition-dry-run`
- Commits: implementation `9288f42`, evidence `3b65b8e`, merge `b6370d9`
- Tests: Stage 07 focused `14 passed`; Stage 03 regression `36 passed`;
  Stage 06 origin-policy gates `9 passed`; Stage 00 baseline `11 passed`
- Stage 07 independent code review: completed; one unused test import and one
  overlong helper signature were fixed in-scope; rerun evidence passed

### To Stage 09

After Stage 08, Stage 09 can rely on:
- `src/kazusa_ai_chatbot/internal_thought_cognition.py` defines residue,
  latch, public scene, and audit contracts;
- selector variant `internal_thought_internal_monologue` and payload key
  `internal_thought_residue` exist without `/chat` regression;
- Stage 08 focused tests prove no public-history, RAG, write, scheduler,
  dispatcher, adapter, cache, or persistence leakage;
- public scene residue remains contract-only: not produced, merged, persisted,
  or exposed by Stage 08.

Stage 09 must keep `internal_thought_cognition.py` as Keep and must not use
internal-thought residue, action latch, or public scene residue as image/audio
percept input.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle, exact contracts,
  progress evidence, independent review gates, and handoff.
- `local-llm-architecture`: enforce bounded background cognition, explicit
  source ownership, context caps, and no response-path call increase.
- `no-prepost-user-input`: do not reinterpret generated thought or action latch
  text as a user instruction, preference, fact, permission, or commitment.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-07 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not route internal thought through `/chat`, `persona_supervisor2(...)`,
  service adapters, message-envelope intake, dialog, or post-turn persistence.
- Do not call `call_consolidation_subgraph(...)`.
- Do not call scheduler, dispatcher, cache invalidation, adapter send, or
  conversation persistence functions.
- Do not add a recursive loop. Stage 08 dry run is exactly one injected
  cognition subgraph invocation per audit request.
- Do not add retry, repair, fallback, compatibility, queue, worker, or
  background-runner paths.
- Do not store internal thought in public conversation history, user memory,
  character state, group scene residue, normal RAG evidence, or normal adapter
  output.
- Do not broaden Stage 04 RAG adapter runtime acceptance for
  `internal_thought`.
- Do not change existing `text_chat_user_message` or Stage 07 reflection prompt
  bytes.
- Do not deterministically classify, approve, execute, persist, or convert
  action-latch text into commitments, permissions, facts, preferences, or
  scheduled actions. Preserve action-latch data only as audit-only structured
  input for the single dry run.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in `Execution
  Evidence`.

## Must Do

- Create private internal-thought residue, public scene residue, and
  action-latch TypedDict contracts in
  `src/kazusa_ai_chatbot/internal_thought_cognition.py`.
- Add an internal-thought episode builder for exactly:
  `trigger_source="internal_thought"`,
  `input_sources=["internal_monologue"]`, and
  `output_mode in {"think_only", "preview", "silent"}`.
- Add prompt selection for exactly one internal-thought dry-run variant:
  `internal_thought_internal_monologue`.
- Add internal-thought prompt-map entries for every L1/L2/L3 handler without
  changing text-chat or reflection prompt bytes.
- Add source-payload projection for the internal-thought variant inside
  `build_cognition_prompt_source_payload(...)`.
- Add a dry-run audit runner that calls the injected cognition callable at most
  once and returns an audit dict with the exact contract in this plan.
- Add tests proving no public or durable write path is called.
- Add tests proving private residue never appears in `chat_history_recent`,
  `chat_history_wide`, `final_dialog`, `consolidation_state`, RAG payloads,
  adapter output, scheduler calls, dispatcher calls, cache invalidation calls,
  or conversation persistence calls.
- Add tests proving public scene residue is contract-only in this stage:
  no public scene residue instance is produced, merged, persisted, or exposed
  by the runner.
- Run every Verification command and record evidence.

## Deferred

- Enabling internal-thought durable writes.
- Enabling loops, retries, action execution, proactive output, or scheduled
  action requests.
- Merging private cognition residue into any runtime public scene residue.
- Producing public scene residue instances from the dry-run runner.
- RAG retrieval for internal thought.
- Dialog generation from internal thought.
- Multimodal input sources and proactive transport.

## Cutover Policy

Overall strategy: compatible for `/chat`, bigbang dry-run-only for internal
thought.

| Area | Policy | Instruction |
|---|---|---|
| Current `/chat` | compatible | Preserve graph, prompts, RAG, dialog, consolidation, and persistence. |
| Internal-thought entrypoint | bigbang | Add one dry-run entrypoint. No fallback through `/chat`. |
| Loop policy | bigbang | Exactly one injected cognition subgraph invocation. No recursive loop, retry, or repair call. |
| Writes and output | bigbang | No dialog, no persistence, no scheduler, no cache, and no adapter sends. |
| Public scene residue | bigbang | Define the contract only. Do not construct, merge, persist, or expose runtime public scene residue. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by adding shims,
  fallbacks, feature flags, dual paths, or compatibility layers.
- If an area is `bigbang`, implement only the named new behavior and prove no
  old or alternate path was retained.
- If an area is `compatible`, preserve only the explicitly listed current
  behavior.
- Any change to a cutover policy requires user approval before implementation.

Rollback path: remove the internal-thought dry-run module, remove selector and
prompt-map additions for the internal-thought variant, and remove focused
tests. No database rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local fixture helper names;
- local variable names inside tests;
- assertion ordering.

Not allowed:

- adding feature flags, fallback paths, loop runners, queues, background
  workers, alternate graph entrypoints, or repair calls;
- importing from `experiments/cognition_core_next/`;
- creating extra source labels, output modes, prompt variants, public-scene
  states, persistence schemas, audit fields, or runner statuses;
- touching service, adapters, dispatcher, scheduler, RAG internals, dialog, or
  consolidation;
- adding raising-only helpers or pass-through wrappers;
- using broad helper freedom; private helpers are allowed only for repeated
  structural validation or local table lookup, and only inside files already in
  Change Surface.

If the agent needs a file outside Change Surface, stop and update this plan
before continuing.

## Target State

The internal-thought dry-run path is:

```text
InternalThoughtResidue + optional InternalActionLatch
-> build_internal_thought_cognitive_episode(...)
-> run_internal_thought_cognition_dry_run(...)
-> call_cognition_subgraph_func(dry_run_state)
-> InternalThoughtCognitionDryRunAudit
```

No service, adapter, RAG adapter, dialog, consolidator, scheduler, dispatcher,
cache, or database module calls this path in Stage 08.

## Contracts And Data Shapes

### Module

Create exactly:

`src/kazusa_ai_chatbot/internal_thought_cognition.py`

### Public Type Aliases

```python
InternalThoughtDryRunOutputMode = Literal["think_only", "preview", "silent"]
InternalThoughtDryRunStatus = Literal[
    "completed",
    "skipped_busy",
    "skipped_empty_residue",
]
```

### TypedDict Contracts

```python
class InternalThoughtResidue(TypedDict):
    residue_id: str
    internal_monologue: str
    source: Literal["runtime_internal_thought"]


class InternalActionLatch(TypedDict):
    latch_id: str
    action_text: str
    latch_reason: str
    status: Literal["audit_only"]


class PublicSceneResidue(TypedDict):
    scene_residue_id: str
    source_episode_id: str
    summary: str
    visibility: Literal["public_scene_candidate"]
    merge_status: Literal["not_merged_stage_08"]


class InternalThoughtCognitionDryRunAudit(TypedDict):
    status: InternalThoughtDryRunStatus
    skip_reason: str
    cognition_called: bool
    episode_id: str
    residue_id: str
    action_latch_id: str
    trigger_source: Literal["internal_thought"]
    input_sources: list[Literal["internal_monologue"]]
    output_mode: InternalThoughtDryRunOutputMode
    prompt_variant: Literal["internal_thought_internal_monologue"]
    prompt_keys: list[str]
    cognition_output_keys: list[str]
```

The runner must not return raw cognition values, generated dialog, generated
facts, generated promises, private monologue text, public scene residue, RAG
evidence, or adapter payloads. It returns only the fields above.

### Error Class

```python
class InternalThoughtCognitionDryRunError(ValueError):
    """Raised when internal-thought dry-run inputs violate the local contract."""
```

### Builder Signature

```python
def build_internal_thought_cognitive_episode(
    *,
    residue: InternalThoughtResidue,
    timestamp: str,
    time_context: TimeContextDoc,
    action_latch: InternalActionLatch | None = None,
    output_mode: InternalThoughtDryRunOutputMode = "think_only",
) -> CognitiveEpisode:
```

Builder rules:

- Reject unsupported `output_mode`, empty ids/text, residue source other than
  `runtime_internal_thought`, and action-latch status other than `audit_only`.
- Enforce hard caps before building the episode:
  `internal_monologue <= 4000`, `action_text <= 1000`, and
  `latch_reason <= 1000` characters.
- Build one `CognitiveEpisode` with one `internal_monologue` percept whose
  `visibility` is `model_visible`.
- Percept content must be canonical JSON with keys `residue` and
  `action_latch`; `action_latch` must be `{}` when no latch is supplied.
- Use `episode_id = "internal_thought:dry_run:<sha256-prefix>"` where the
  prefix is the first 16 hex characters of the canonical percept content
  digest.
- Use `target_scope` fixed to platform/channel/user/display value
  `internal_thought` or `internal_thought_dry_run`, with no addressed users and
  `target_broadcast=False`.
- Use `origin_metadata` fixed to platform `internal_thought`, message id
  `internal_thought:dry_run`, empty active-turn ids, and debug modes
  `{"think_only": True, "no_remember": True}`.
- Call `validate_cognitive_episode(episode)` before returning.

### Runner Signature

```python
async def run_internal_thought_cognition_dry_run(
    *,
    residue: InternalThoughtResidue,
    character_profile: CharacterProfileDoc,
    user_profile: UserProfileDoc,
    timestamp: str,
    time_context: TimeContextDoc,
    is_primary_interaction_busy: Callable[[], bool],
    call_cognition_subgraph_func: Callable[
        [GlobalPersonaState],
        Awaitable[GlobalPersonaState],
    ],
    action_latch: InternalActionLatch | None = None,
    output_mode: InternalThoughtDryRunOutputMode = "think_only",
) -> InternalThoughtCognitionDryRunAudit:
```

Runner rules:

- Validate `output_mode` before the busy check.
- Busy skip audit: `status="skipped_busy"`,
  `skip_reason="primary_interaction_busy"`, `cognition_called=False`, empty
  `episode_id`, supplied ids, and empty prompt/output key lists.
- Empty-residue skip audit: `status="skipped_empty_residue"`,
  `skip_reason="internal_thought_residue_empty"`, `cognition_called=False`,
  empty `episode_id`, supplied ids, and empty prompt/output key lists.
- For a runnable residue, call `build_internal_thought_cognitive_episode(...)`
  exactly once.
- Build `dry_run_state` locally with empty `chat_history_wide`,
  `chat_history_recent`, `reply_context`, `promoted_reflection_context`,
  `final_dialog`, `new_facts`, and `future_promises`.
- Build `rag_result` with the same empty shape used by Stage 07. Do not call
  Stage 04 RAG adapter, RAG runtime, cache runtime, or retrieval functions.
- Set `user_input` and `decontexualized_input` to the fixed string:
  `Internal thought dry run over private cognition residue.`
- Set `should_respond=False` and debug modes
  `{"think_only": True, "no_remember": True}`.
- Preserve action latch only inside the episode source payload. Do not convert
  it into `action_directives`, commitments, scheduled actions, or output.
- Await `call_cognition_subgraph_func(dry_run_state)` exactly once.
- Return `status="completed"`, `skip_reason=""`, `cognition_called=True`, the
  episode id, the supplied ids, the exact variant name, the exact prompt-key
  list, and sorted cognition output keys.

### Prompt Selection Contract

Update `CognitionPromptVariant` with exactly:

```python
"internal_thought_internal_monologue"
```

Add exactly one selector branch:

- `trigger_source == "internal_thought"`
- `input_sources == ["internal_monologue"]`
- `output_mode in {"think_only", "preview", "silent"}`
- `prompt_key == f"{stage}.internal_thought_internal_monologue"`

Do not enumerate internal-thought rejection labels in static deny lists. Use a
closed allow-list branch.

### Source Payload Contract

`build_cognition_prompt_source_payload(...)` must return exactly this mapping
for `internal_thought_internal_monologue`:

```python
{
    "internal_thought_residue": {
        "residue_id": str,
        "internal_monologue": str,
        "action_latch": dict[str, str],
    },
}
```

`action_latch` is `{}` when no latch was supplied. The payload must be derived
by parsing the episode percept's canonical JSON content. If the
`internal_monologue` percept is missing, duplicated, malformed, or structurally
wrong, raise `CognitionPromptSelectionError` directly.

### Prompt-Key List

Use this exact list in the module and tests:

```python
[
    "l1_subconscious.internal_thought_internal_monologue",
    "l2a_consciousness.internal_thought_internal_monologue",
    "l2b_boundary_core.internal_thought_internal_monologue",
    "l2c_judgment_core.internal_thought_internal_monologue",
    "l3_contextual_agent.internal_thought_internal_monologue",
    "l3_style_agent.internal_thought_internal_monologue",
    "l3_content_anchor_agent.internal_thought_internal_monologue",
    "l3_preference_adapter.internal_thought_internal_monologue",
    "l3_visual_agent.internal_thought_internal_monologue",
]
```

## LLM Call And Context Budget

- Current `/chat` before Stage 08: unchanged existing live response path.
- Current `/chat` after Stage 08: unchanged. No new response-path LLM calls,
  no new RAG calls, no new dialog calls, and no new consolidation calls.
- Internal-thought dry run before Stage 08: unsupported.
- Internal-thought dry run after Stage 08: one background-only injected
  cognition subgraph invocation when not skipped. The implementation must not
  add retries, repair calls, evaluator calls, dialog calls, or RAG calls.
- Inside the injected cognition subgraph, existing L1/L2/L3 cognition calls are
  reused at most once per stage for the dry-run state:
  L1 subconscious, L2a consciousness, L2b boundary core, L2c judgment core,
  L3 contextual, L3 style, L3 content anchor, L3 preference, and L3 visual.
- Context cap: assume a 50k token model context cap.
- Hard source caps before prompt entry:
  `internal_monologue <= 4000` characters,
  `action_text <= 1000` characters, and
  `latch_reason <= 1000` characters.
- Cap policy: reject over-cap residue or latch inputs with
  `InternalThoughtCognitionDryRunError`; do not silently truncate or summarize
  private thought.
- Prompt input limit verification: focused tests must build an over-cap
  residue and prove the builder raises before any cognition call.
- Latency impact: no live response-path impact because Stage 08 does not wire
  the dry run into service, adapters, scheduler, or dispatcher.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Trigger ownership | Use `internal_thought`, not `reflection_signal`. | Internal residue is generated by runtime cognition, not reflection promotion. |
| Input source | Use only `internal_monologue`. | Keeps private monologue separate from media, memory, and dialog text. |
| Prompt variant | Use `internal_thought_internal_monologue`. | Names the trigger and source without implying action execution. |
| Loop policy | One injected cognition subgraph invocation only. | Bounded dry-run evidence before any loop design. |
| Public scene residue | Define contract only; do not produce runtime instances. | Satisfies parent Stage 08 contract handoff without public leakage. |
| Persistence | Return audit dict only. | Stage 06 denies non-chat writes and Stage 10 owns output. |
| Action latch | Audit-only structured input. | Prevents deterministic permission, commitment, or scheduled-action interpretation. |
| Experiment code | Translate concepts, do not import. | Experiment modules are reference only. |

## Change Surface

Target ownership boundary: internal-thought dry-run contracts, prompt selection,
prompt-map admission, focused tests, and lifecycle records.

### Create

- `src/kazusa_ai_chatbot/internal_thought_cognition.py` — private residue,
  public scene residue, action-latch, episode-builder, and dry-run audit
  contracts.
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py` —
  builder, selector, source-payload, prompt-render, bounded-run,
  no-public-leak, no-write, prompt-fingerprint, and context-cap tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  — add the single internal-thought dry-run prompt variant and source payload
  projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` — add the
  L1 internal-thought prompt-map entry without changing existing prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` — add L2
  internal-thought prompt-map entries without changing existing prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` — add L3
  internal-thought prompt-map entries without changing existing prompt bytes.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md`
  — record progress, verification, independent code review, execution
  evidence, and final lifecycle status.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  — move Stage 08 from `approved` to `completed` after implementation.
- `development_plans/README.md` — move Stage 08 from `approved | ready` to
  `completed | completed` after implementation.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/*.py`
- adapter, dispatcher, scheduler, cache, and database modules
- `experiments/cognition_core_next/**`

## Implementation Order

1. Reread this plan, Stage 07 `Execution Evidence`, the parent ledger row, and
   registry row.
2. Add focused import/contract tests in
   `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
   for:
   `InternalThoughtResidue`, `InternalActionLatch`, `PublicSceneResidue`,
   `InternalThoughtCognitionDryRunAudit`,
   `InternalThoughtCognitionDryRunError`,
   `build_internal_thought_cognitive_episode`, and
   `run_internal_thought_cognition_dry_run`.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q`
   - Expected before implementation: import error for
     `kazusa_ai_chatbot.internal_thought_cognition`.
3. Implement only the type aliases, TypedDicts, error class, constants, and
   builder in `src/kazusa_ai_chatbot/internal_thought_cognition.py`.
   - Verify the same focused test command.
   - Expected after this step: builder tests pass; runner and selector tests
     still fail if already present.
4. Add selector and source-payload tests in the Stage 08 test file for:
   supported tuple, unsupported trigger/source/mode, exact prompt keys,
   malformed percept JSON, duplicate internal-monologue percepts, and exact
   payload shape.
5. Update
   `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
   with the internal-thought selector branch and source-payload projection.
   - Verify Stage 08 selector/source-payload tests and
     `tests/test_multi_source_cognition_stage_03_prompt_selection.py`.
6. Add prompt-map entries in L1/L2/L3 files by mapping the new variant to the
   existing prompt constants only. Do not edit prompt constant text.
   - Verify Stage 08 prompt-render tests and Stage 03 selector tests.
7. Add runner tests for busy skip, empty-residue skip, one cognition call,
   empty history/RAG/dialog/write surfaces, audit-only return fields, no public
   scene residue production, no action execution, and over-cap rejection.
8. Implement the runner in `internal_thought_cognition.py`.
   - Verify the Stage 08 focused test file.
9. Add prompt fingerprint tests in the Stage 08 test file covering all existing
   text-chat and reflection prompt constants that Stage 08 touches indirectly.
   - Verify the Stage 08 focused test file and Stage 07 focused test file.
10. Run every command in `Verification`.
11. Run the `Independent Code Review` gate and remediate in-scope findings.
12. After review approval, update Stage 08 lifecycle rows to completed, record
    execution evidence, and sign off.

## Progress Checklist

- [ ] Stage 1 - prerequisite evidence carried forward.
  - Covers Step 1. Verify parent ledger and registry show Stage 07 completed
    and Stage 08 approved; record the Stage 07 branch, commits, merge commit,
    and test results.
  - Handoff: reread this plan, then start Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - internal thought episode contract complete.
  - Covers Steps 2-3. Verify focused contract/builder tests pass after the
    expected red import failure; record red/green output.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 - prompt selection and source payload complete.
  - Covers Steps 4-5. Verify Stage 08 selector/source-payload tests and Stage
    03 selector tests pass; record command output.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - prompt maps and fingerprints complete.
  - Covers Steps 6 and 9. Verify Stage 08 prompt-render/fingerprint tests plus
    Stage 07 and Stage 03 gates pass; record fingerprint and test output.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - bounded dry-run runner complete.
  - Covers Steps 7-8. Verify bounded-run, no-public-leak, no-write,
    no-scene-residue, and context-cap tests pass; record command output.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 6 - full verification complete.
  - Covers Step 10. Verify every Verification command passes or has an allowed
    no-match exit; record command output.
  - Handoff: run independent code review before lifecycle completion.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 7 - independent code review and lifecycle complete.
  - Covers Steps 11-12. Verify `Independent Code Review` passes, findings are
    fixed or recorded, affected checks are rerun, and lifecycle rows are
    updated; record review, fixes, reruns, lifecycle rows, and residual risks.
  - Handoff: Stage 09 plan can be reviewed after this checkpoint is signed.
  - Sign-off: `<agent/date>` after review approval and lifecycle update.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`

### Change Surface Gate

Run this from the feature branch:

```powershell
$allowed = @('development_plans/README.md','development_plans/active/short_term/multi_source_cognition_architecture_plan.md','development_plans/active/short_term/multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md','src/kazusa_ai_chatbot/internal_thought_cognition.py','src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py','src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py','src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py','src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py','tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py')
$changed = git diff --name-only main...HEAD
$outside = $changed | Where-Object { $allowed -notcontains $_ }
if ($outside) { $outside; exit 1 }
'CHANGE_SURFACE_OK'
```

Expected result: `CHANGE_SURFACE_OK`.

### Static Greps

- `rg -n "save_conversation|save_assistant_message|call_consolidation_subgraph|dispatcher\\.dispatch|runtime\\.invalidate|get_rag_cache2_runtime|schedule|adapter" src\kazusa_ai_chatbot\internal_thought_cognition.py`

  Expected result: no matches. `rg` exit code `1` is acceptable.
- `rg -n "\"internal_thought\"|\"internal_monologue\"" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`

  Expected result: no matches. Stage 08 must not wire internal thought into
  service, persona supervisor, or RAG.
- `rg -n "PublicSceneResidue|public_scene_candidate|not_merged_stage_08" src\kazusa_ai_chatbot -g "*.py"`

  Expected result: matches only in
  `src\kazusa_ai_chatbot\internal_thought_cognition.py` and
  `tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`.
- `git diff --check`

  Expected result: exit code `0`. Git line-ending normalization warnings are
  acceptable only when no whitespace errors are reported.

### Prompt Fingerprint Guard

Add and run Stage 08 tests that verify existing text-chat and reflection prompt
constants touched by the prompt-map additions retain their exact UTF-8 byte
lengths and SHA-256 digests from the pre-Stage-08 baseline. The implementation
agent must capture the baseline from post-Stage-07 `main` before editing L1/L2/L3.

Expected result: every text-chat and reflection prompt fingerprint matches the
post-Stage-07 baseline.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not author the plan. If no separate reviewer is available, the active agent
must reread the parent architecture plan, completed Stage 07 artifacts, this
plan, and relevant source/test context from a fresh-review posture.

Review scope: prior artifacts, architecture alignment, readiness, exact
contracts, change surface, verification gates, progress evidence, creativity
suppression, and clean Stage 07/08/09/10 boundaries.

Review result on 2026-05-10:

- Fixed blockers: stale Stage 07 handoff, parent public-scene-residue carryover,
  LLM/context budget, independent code review gate, lifecycle change surface,
  inherited Stage 06 gates, exact audit contract, and action-latch semantic
  guard.
- Approval status: approved for Stage 08 execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off,
lifecycle completion, merge, or handoff. Prefer a reviewer that did not
implement the change. If no separate reviewer is available, the active agent
must reread this plan, inspect the full diff from a fresh-review posture, and
record that no separate reviewer was available.

Review scope: project style, code quality, design weaknesses, plan alignment,
prompt/RAG payload leaks, persistence risk, action-latch semantic drift,
regression coverage, execution evidence, Stage 09 handoff, and path-safe
commands.

Fix concrete findings directly only when the fix is inside the approved Change
Surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

Stage 08 is complete when:

- internal-thought residue, public-scene-residue, action-latch, episode, and
  audit contracts exist;
- selector supports only the approved internal-thought dry-run tuple;
- source-payload projection returns exactly `internal_thought_residue`;
- dry-run runner invokes cognition at most once and writes nothing;
- dry-run runner does not produce public scene residue instances;
- private residue does not appear in public dialog, conversation history,
  RAG runtime, consolidation, scheduler, dispatcher, adapter, cache, or
  persistence paths;
- action latch remains audit-only and is not converted into permissions,
  commitments, scheduled actions, dialog, or durable state;
- `/chat`, Stage 03, Stage 06, and Stage 07 regression gates still pass;
- independent code review is complete and recorded before lifecycle completion.

## Plan Self-Review

Approved self-review on 2026-05-10:

- **Coverage:** every Stage 08 parent scope item maps to a contract,
  implementation step, verification gate, or explicit deferral.
- **Open-marker scan:** no unresolved Stage 07 evidence, broad implementation
  wording, or open implementation decisions remain.
- **Contract consistency:** trigger source, input source, output modes,
  variant name, prompt keys, function names, TypedDict names, audit fields,
  and command paths match across sections.
- **Granularity:** checkpoints split prerequisite evidence, builder contract,
  selector/source payload, prompt maps, runner, verification, and review.
- **Verification:** no-loop, no-write, no-public-leak, no-public-scene-output,
  context-cap, prompt-fingerprint, change-surface, and prior-stage gates are
  explicit.

## Execution Handoff

Intended execution mode: sequential implementation on a feature branch forked
from post-Stage-07 `main`.

Next action: fork the Stage 08 branch from `main`, reread this approved plan,
load the mandatory skills, and start at Progress Checklist Stage 1.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Private thought leaks publicly | No service/dialog/consolidation/RAG wiring and audit-only return | Static greps, focused no-public-leak tests |
| Unbounded self-loop appears | Exactly one injected cognition subgraph invocation; no retry or repair | Bounded-run test and static grep |
| Reflection and internal sources blur | Separate variant, labels, contracts, and tests | Selector and prompt-key tests |
| Generated thought writes memory | No consolidation calls plus inherited Stage 06 write-denial gates | Static greps and origin-policy tests |
| Action latch becomes execution permission | Audit-only latch contract and no-pre/post-user-input guard | No action-directive, scheduler, dispatcher, and persistence tests |
| `/chat` prompt drift | Preserve existing prompt bytes | Stage 00, Stage 03, Stage 07, and Stage 08 fingerprint gates |
| Public scene residue leaks early | Contract-only residue and no runner production | Public-scene grep and no-scene-residue tests |

## Completion Artifact Contract

When Stage 08 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- internal-thought dry-run selector variant
- internal-thought source-payload projection
- internal-thought L1/L2/L3 prompt-map entries
- audit-only dry-run return shape
- public scene residue contract with no runtime production
- parent ledger row for `stage_08` flipped to `completed`
- registry row flipped to `completed | completed`
- execution evidence in this plan naming branch, commit, checks, independent
  code review, lifecycle rows, residual risks, and sign-off

The artifact must not include service wiring, persistence, database writes,
scheduler dispatch, adapter output, RAG broadening, recursive loops, action
execution, or proactive behavior.

## Execution Evidence
Record during implementation:

- Stage 07 evidence reread:
- Independent plan review:
  completed on 2026-05-10; approved for execution after review-derived fixes.
- Branch and commit:
- Static checks: compile, change surface, greps, diff check, and prompt
  fingerprint guard:
- Tests: focused and prior-stage regression gates:
- Independent code review:
- Completion diff review, lifecycle records, residual risks, and sign-off:
