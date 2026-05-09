# multi source cognition architecture stage 07 reflection trigger cognition dry run plan

## Summary

- Goal: Add a reflection-triggered dry-run path that builds a
  `CognitiveEpisode(trigger_source=reflection_signal)` from promoted reflection
  context and runs shared cognition in audit-only mode.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for current `/chat`; dry-run only for
  reflection-triggered cognition. No dialog, delivery, consolidation,
  persistence writes, scheduler dispatch, or adapter sends are enabled.
- Highest-risk areas: treating reflection as fake user input, raw reflection
  leakage into prompts, prompt drift for current `/chat`, background LLM
  latency, and accidentally enabling writes despite Stage 06 origin policy.
- Acceptance criteria: a reflection episode builder and dry-run audit runner
  exist; reflection prompt selection is source-aware and audit-only; `/chat`
  prompt fingerprints and regression gates remain unchanged; no durable write
  or delivery path is introduced.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: approved on 2026-05-10 after Stage 06 completion evidence was
carried forward and this plan passed independent plan review. Execute only from
a feature branch forked from `main` after the parent ledger and registry show
`stage_06` completed and `stage_07` approved.

## Context

Stages 01 through 05 put current `/chat` on the episode, RAG, cognition, and
consolidation-origin contracts. Stage 06 makes consolidator writes explicitly
origin-gated. Stage 07 is the first non-chat trigger admission stage, but it
must remain a dry run.

Reflection input must come from promoted/gated reflection context only. Raw
hourly or daily reflection run output must not enter cognition. The existing
`build_promoted_reflection_context(...)` boundary is the only allowed context
source for this stage.

This stage deliberately does not broaden the Stage 04 RAG adapter for
reflection. Reflection dry-run state must use the existing empty projected RAG
shape plus promoted reflection context. Reflection retrieval is deferred to a
later plan.

## Stage Handoff

### From Stage 06

Stage 07 expects these completed artifacts:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
- `tests/test_consolidation_origin_policy.py`
- `tests/test_consolidator_origin_policy_db_writer.py`
- evidence that denied origins produce no durable writes, no scheduler
  dispatch, and no Cache2 invalidation
- parent ledger row for `stage_06` set to `completed`

Carried-forward Stage 06 evidence:

- Branch: `stage-06-consolidator-per-write-origin-policy`
- Commit: `2ea7526` (`Implement stage 06 consolidation write policy`)
- Full deterministic suite:
  `venv\Scripts\python -m pytest` passed with
  `907 passed, 217 deselected`.
- Focused write-policy gates passed:
  `tests\test_consolidation_origin_policy.py`,
  `tests\test_consolidator_origin_policy_db_writer.py`,
  `tests\test_consolidator_efficiency.py`,
  `tests\test_db_writer_cache2_invalidation.py`, and adjacent
  consolidation, scheduler, service-background, RAG, and Stage 00-04
  regression gates recorded in Stage 06 `Execution Evidence`.
- Parent ledger row is `completed`.
- Registry row is `completed | completed`.

### To Stage 08

After Stage 07, Stage 08 can rely on:

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py` defining
  `build_reflection_signal_cognitive_episode(...)`,
  `run_reflection_cognition_dry_run(...)`, and
  `ReflectionCognitionDryRunAudit`;
- the selector variant value
  `"reflection_signal_reflection_artifact"` and prompt keys shaped as
  `"<stage>.reflection_signal_reflection_artifact"`;
- a non-chat dry-run entrypoint pattern that does not call `/chat`;
- reflection-source prompt selection and prompt-render tests;
- the audit-only return shape listed in `Contracts And Data Shapes`;
- current `/chat` regression evidence after adding a future source variant.

Stage 08 must not reuse reflection-specific names for internal thought. It must
add its own trigger, input-source, and audit labels.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle and handoff.
- `local-llm-architecture`: keep the dry-run path bounded, background-only, and
  source-aware.
- `no-prepost-user-input`: do not reinterpret reflection output as user facts,
  user instructions, or accepted commitments.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-06 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not route reflection through `/chat`, `persona_supervisor2(...)`,
  service adapters, message-envelope intake, dialog, or post-turn persistence.
- Do not call `call_consolidation_subgraph(...)`.
- Do not call `save_conversation(...)`, `save_assistant_message(...)`,
  dispatcher transport, scheduler dispatch, or adapter send functions.
- Do not broaden Stage 04 RAG adapter runtime acceptance for
  `reflection_signal`.
- Do not add a new live `/chat` LLM call or increase current `/chat` prompt
  payloads.
- Do not modify existing `text_chat_user_message` prompt constants except for
  mechanical lookup-table wiring. Their byte fingerprints must remain stable.
- Reflection dry runs may use background cognition LLM calls only when an
  explicit busy probe says primary interaction is not busy.
- The dry-run runner must accept an injected cognition callable for tests.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes to `completed`, merge, or
  sign-off, run the `Independent Code Review` gate and record the result in
  `Execution Evidence`.
- The only new prompt variant value is exactly
  `"reflection_signal_reflection_artifact"`. The only new reflection prompt key
  shape is exactly `"<stage>.reflection_signal_reflection_artifact"`.
- If promoted reflection context contains no `promoted_lore` and no
  `promoted_self_guidance`, the dry-run runner must return a skipped audit
  with `status="skipped_empty_context"` and must not build a cognitive episode
  or call cognition.
- Prompt payloads must not include keys named `cognitive_episode`,
  `trigger_source`, or `input_sources`. They must include promoted reflection
  content only under `reflection_artifact` for the reflection variant.

## Must Do

- Add `build_reflection_signal_cognitive_episode(...)` exactly as defined in
  `Contracts And Data Shapes`. It consumes promoted reflection context as the
  only reflection material and accepts only timestamp, time context, and output
  mode as non-reflection metadata.
- Add reflection prompt selection for exactly:
  `trigger_source="reflection_signal"`,
  `input_sources=["reflection_artifact"]`, and
  `output_mode in {"think_only", "preview", "silent"}`.
- Add reflection dry-run prompt-map entries for every L1/L2/L3 handler without
  changing current text-chat prompt bytes. The reflection variant maps to the
  existing prompt constants; source awareness is carried by the model-facing
  `reflection_artifact` payload field, not by rewriting current prompt text.
- Add `run_reflection_cognition_dry_run(...)` exactly as defined in
  `Contracts And Data Shapes`. It builds source-aware cognition state, calls
  the injected cognition callable at most once, and returns
  `ReflectionCognitionDryRunAudit`.
- Add tests proving busy dry runs call no cognition LLMs and non-busy dry runs
  call the cognition callable once.
- Add prompt-render tests proving reflection prompts receive reflection
  percept context and do not receive `cognitive_episode`, `trigger_source`, or
  raw reflection run documents in model-facing payloads.
- Run every Verification command and record evidence.
- Keep lifecycle rows as `approved | approved` before implementation. Flip
  lifecycle rows to `completed | completed` only after verification and
  independent code review pass.

## Deferred

- Reflection durable writes.
- Reflection RAG retrieval.
- Reflection output delivery, dialog generation, scheduled action requests, or
  proactive sends.
- Raw reflection run ingestion into cognition.
- Changes to memory promotion, daily reflection prompts, interaction-style
  updates, scheduler policy, adapters, or transport.
- Internal thought, action latch, multimodal inputs, and proactive output.

## Cutover Policy

Overall strategy: compatible for `/chat`, dry-run-only for reflection.

| Area | Policy | Instruction |
|---|---|---|
| Current `/chat` | compatible | Preserve current graph, prompt bytes, RAG request shape, dialog, and consolidation behavior. |
| Reflection entrypoint | bigbang | Add one dry-run entrypoint. No fallback through `/chat`. |
| Reflection writes | bigbang | No writes, no consolidation, no scheduler, no cache invalidation, and no adapter sends. |
| Reflection RAG | compatible | Keep Stage 04 future-source rejection; use empty projected RAG shape in dry-run state. |

Rollback path: remove the reflection dry-run module, remove reflection prompt
variant wiring, remove focused tests, and restore selector to text-chat-only.
No database rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names inside tests;
- assertion ordering;
- fixture helper names.

Not allowed:

- adding feature flags, fallback paths, compatibility shims, wrapper layers, or
  alternate graph entrypoints;
- adding persistent audit storage;
- editing service startup, reflection worker scheduling, adapter delivery, RAG
  internals, dialog, or consolidation;
- creating extra source labels, output modes, or prompt variants;
- using raw hourly/daily reflection run documents as model input;
- adding raising-only helpers or pass-through wrappers.

If a required file outside Change Surface must change, stop and update this
plan before continuing.

## Target State

The reflection dry-run path is:

```text
promoted reflection context
-> build_reflection_signal_cognitive_episode(...)
-> run_reflection_cognition_dry_run(...)
-> call_cognition_subgraph_func(dry_run_state)
-> ReflectionCognitionDryRunAudit
```

The dry-run state must not be created by calling `/chat` or
`persona_supervisor2(...)`. Compatibility fields required by the current
cognition subgraph must be populated with the explicit reflection dry-run
placeholder values listed in `Contracts And Data Shapes`. Model-facing
reflection prompt payloads must frame the input as a reflection artifact, not
as a user utterance.

## Contracts And Data Shapes

Add these public contracts to
`src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`.

```python
ReflectionDryRunOutputMode = Literal["think_only", "preview", "silent"]
ReflectionDryRunStatus = Literal[
    "completed",
    "skipped_busy",
    "skipped_empty_context",
]


class ReflectionCognitionDryRunError(ValueError):
    """Raised when reflection dry-run inputs violate the local contract."""


class ReflectionCognitionDryRunAudit(TypedDict):
    status: ReflectionDryRunStatus
    skip_reason: str
    cognition_called: bool
    episode_id: str
    trigger_source: Literal["reflection_signal"]
    input_sources: list[Literal["reflection_artifact"]]
    output_mode: ReflectionDryRunOutputMode
    prompt_variant: Literal["reflection_signal_reflection_artifact"]
    prompt_keys: list[str]
    cognition_output_keys: list[str]
```

Skipped audit values are exact:

- Busy skip:
  `status="skipped_busy"`, `skip_reason="primary_interaction_busy"`,
  `cognition_called=False`, `episode_id=""`,
  `trigger_source="reflection_signal"`,
  `input_sources=["reflection_artifact"]`,
  `output_mode=<supplied output_mode>`,
  `prompt_variant="reflection_signal_reflection_artifact"`,
  `prompt_keys=[]`, and `cognition_output_keys=[]`.
- Empty-context skip:
  `status="skipped_empty_context"`,
  `skip_reason="promoted_reflection_context_empty"`,
  `cognition_called=False`, `episode_id=""`,
  `trigger_source="reflection_signal"`,
  `input_sources=["reflection_artifact"]`,
  `output_mode=<supplied output_mode>`,
  `prompt_variant="reflection_signal_reflection_artifact"`,
  `prompt_keys=[]`, and `cognition_output_keys=[]`.
- Completed audit:
  `status="completed"`, `skip_reason=""`, `cognition_called=True`,
  non-empty `episode_id`, the nine prompt keys listed below, and
  `cognition_output_keys` sorted from the returned cognition dict.

Prompt keys for a completed reflection dry run are exact and ordered:

```python
[
    "l1_subconscious.reflection_signal_reflection_artifact",
    "l2a_consciousness.reflection_signal_reflection_artifact",
    "l2b_boundary_core.reflection_signal_reflection_artifact",
    "l2c_judgment_core.reflection_signal_reflection_artifact",
    "l3_contextual_agent.reflection_signal_reflection_artifact",
    "l3_style_agent.reflection_signal_reflection_artifact",
    "l3_content_anchor_agent.reflection_signal_reflection_artifact",
    "l3_preference_adapter.reflection_signal_reflection_artifact",
    "l3_visual_agent.reflection_signal_reflection_artifact",
]
```

Use this exact builder signature:

```python
def build_reflection_signal_cognitive_episode(
    *,
    promoted_reflection_context: PromotedReflectionContext,
    timestamp: str,
    time_context: TimeContextDoc,
    output_mode: ReflectionDryRunOutputMode = "think_only",
) -> CognitiveEpisode:
```

Builder behavior is exact:

- Treat the context as empty when both `promoted_lore` and
  `promoted_self_guidance` are absent or empty. In that case raise
  `ReflectionCognitionDryRunError("promoted reflection context is empty")`.
- Derive `episode_id` as
  `reflection:dry_run:<digest>`, where `<digest>` is the first 16 hex
  characters of the SHA-256 digest of canonical JSON rendered from
  `promoted_reflection_context` with `sort_keys=True` and
  `ensure_ascii=False`.
- Use `trigger_source="reflection_signal"`.
- Use `input_sources=["reflection_artifact"]`.
- Use the supplied `output_mode`; reject any other output mode by raising
  `ReflectionCognitionDryRunError("reflection output_mode is not supported")`.
- Create exactly one percept:
  `percept_id="reflection:artifact:promoted_context"`,
  `input_source="reflection_artifact"`, `visibility="model_visible"`,
  `content=<canonical JSON string>`, and
  `metadata={"source": "promoted_reflection_context"}`.
- Use target scope placeholders:
  `platform="reflection_cycle"`,
  `platform_channel_id="reflection_dry_run"`,
  `channel_type="reflection_dry_run"`,
  `current_platform_user_id="reflection_cycle"`,
  `current_global_user_id="reflection_cycle"`,
  `current_display_name="reflection_cycle"`,
  `target_addressed_user_ids=[]`, and `target_broadcast=False`.
- Use origin metadata placeholders:
  `platform="reflection_cycle"`,
  `platform_message_id="reflection:dry_run"`,
  `active_turn_platform_message_ids=[]`,
  `active_turn_conversation_row_ids=[]`, and
  `debug_modes={"think_only": True, "no_remember": True}`.
- Call `validate_cognitive_episode(episode)` before returning.

Use this exact runner signature:

```python
async def run_reflection_cognition_dry_run(
    *,
    promoted_reflection_context: PromotedReflectionContext,
    character_profile: CharacterProfileDoc,
    user_profile: UserProfileDoc,
    timestamp: str,
    time_context: TimeContextDoc,
    is_primary_interaction_busy: Callable[[], bool],
    call_cognition_subgraph_func: Callable[
        [GlobalPersonaState],
        Awaitable[GlobalPersonaState],
    ],
    output_mode: ReflectionDryRunOutputMode = "think_only",
) -> ReflectionCognitionDryRunAudit:
```

Runner behavior is exact:

- Reject unsupported `output_mode` with
  `ReflectionCognitionDryRunError("reflection output_mode is not supported")`
  before checking the busy probe.
- Call `is_primary_interaction_busy()` after output-mode validation. If it
  returns `True`, return the busy skipped audit and call no other injected or
  LLM-backed callable.
- If promoted context is empty, return the empty-context skipped audit and do
  not call cognition.
- Build the episode with `build_reflection_signal_cognitive_episode(...)`.
- Build `dry_run_state` directly as a `GlobalPersonaState` dict. Do not call
  `/chat`, `persona_supervisor2(...)`, the RAG adapter, dialog,
  consolidation, persistence, scheduler, dispatcher, or adapters.
- Required dry-run state placeholders:
  `user_input="Reflection dry run over promoted reflection artifact."`,
  `prompt_message_context={"body_text": "", "addressed_to_global_user_ids": [], "broadcast": False, "mentions": [], "attachments": []}`,
  `user_multimedia_input=[]`, `platform="reflection_cycle"`,
  `platform_channel_id="reflection_dry_run"`,
  `channel_type="reflection_dry_run"`,
  `platform_message_id="reflection:dry_run"`,
  `platform_user_id="reflection_cycle"`,
  `global_user_id="reflection_cycle"`,
  `user_name="reflection_cycle"`,
  `platform_bot_id="reflection_cycle"`,
  `chat_history_wide=[]`, `chat_history_recent=[]`, `reply_context={}`,
  `indirect_speech_context=""`, `channel_topic=""`,
  `debug_modes={"think_only": True, "no_remember": True}`,
  `should_respond=False`,
  `decontexualized_input="Reflection dry run over promoted reflection artifact."`,
  `referents=[]`, `promoted_reflection_context=<input context>`, and
  `cognitive_episode=<built episode>`.
- Required empty RAG shape:
  `{"answer": "", "user_image": {"user_memory_context": empty_user_memory_context()}, "user_memory_unit_candidates": [], "character_image": {}, "third_party_profiles": [], "memory_evidence": [], "recall_evidence": [], "conversation_evidence": [], "external_evidence": [], "supervisor_trace": {"loop_count": 0, "unknown_slots": [], "dispatched": []}}`.
- Required initial output placeholders:
  `internal_monologue=""`, `action_directives={}`,
  `interaction_subtext=""`, `emotional_appraisal=""`,
  `character_intent=""`, `logical_stance=""`, `final_dialog=[]`,
  `mood=""`, `global_vibe=""`, `reflection_summary=""`,
  `subjective_appraisals=[]`, `affinity_delta=0`,
  `last_relationship_insight=""`, `new_facts=[]`, and
  `future_promises=[]`.
- Await `call_cognition_subgraph_func(dry_run_state)` exactly once.
- Return the completed audit. Do not return raw cognition output values except
  sorted key names in `cognition_output_keys`.

Selector contract in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
is exact:

- Extend `CognitionPromptVariant` to
  `Literal["text_chat_user_message", "reflection_signal_reflection_artifact"]`.
- Keep the current text-chat branch byte-for-byte compatible in behavior.
- For `trigger_source="reflection_signal"`,
  `input_sources=["reflection_artifact"]`, and
  `output_mode in {"think_only", "preview", "silent"}`, return:
  `{"variant": "reflection_signal_reflection_artifact", "prompt_key": f"{stage}.reflection_signal_reflection_artifact"}`.
- Reject every other reflection tuple with `CognitionPromptSelectionError`.

Add one non-wrapper projection helper to the selector module:

```python
def build_cognition_prompt_source_payload(
    *,
    episode: CognitiveEpisode,
    selection: CognitionPromptSelection,
) -> dict[str, object]:
```

This helper returns `{}` for `text_chat_user_message`. For
`reflection_signal_reflection_artifact`, it returns exactly
`{"reflection_artifact": <the single reflection_artifact percept content>}`.
It must not return `cognitive_episode`, `trigger_source`, `input_sources`, raw
reflection run documents, origin metadata, or target scope fields. It raises
`CognitionPromptSelectionError` if the reflection percept is missing or not
unique. L1/L2/L3 handlers must call this helper and update their existing
`HumanMessage` dicts with the returned payload.

## LLM Call And Context Budget

- Current `/chat` before Stage 07: unchanged existing live response path.
- Current `/chat` after Stage 07: unchanged. No new response-path LLM calls,
  no prompt payload growth, and no RAG/dialog/consolidation call-count change.
- Reflection dry run when busy: zero cognition subgraph calls and zero LLM
  calls.
- Reflection dry run when promoted context is empty: zero cognition subgraph
  calls and zero LLM calls.
- Reflection dry run when not busy and context is non-empty: exactly one
  injected cognition subgraph invocation. That subgraph uses the existing
  L1/L2/L3 cognition LLM calls; Stage 07 must not add repair loops, evaluator
  calls, dialog calls, RAG calls, or summarizer calls.
- Context cap: promoted reflection context is the already gated
  `PromotedReflectionContext` shape. Do not add raw hourly/daily run documents,
  transcripts, full reflection run docs, source message refs, user ids, or
  private details.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Reflection source | Use only `build_promoted_reflection_context(...)` output. | Raw reflection runs are not safe model-facing cognition input. |
| Entry graph | Add a dry-run runner instead of reusing `/chat`. | Reflection is a different trigger source and must not pretend to be a user message. |
| RAG | Use empty projected RAG shape in Stage 07. | Keeps Stage 04 future-source RAG rejection intact while testing shared cognition admission. |
| Output | Return audit dict only. | Later stages own persistence and proactive behavior. |
| Prompt bytes | Preserve text-chat prompt fingerprints. | `/chat` regression risk must remain measurable. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py` — reflection
  episode builder and dry-run audit runner.
- `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py` —
  reflection builder, selector, prompt-render, dry-run, and no-write tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  — add the single reflection dry-run prompt variant and
  `build_cognition_prompt_source_payload(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` — add the
  L1 reflection prompt-map entry without changing text-chat prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` — add L2
  reflection prompt-map entries without changing text-chat prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` — add L3
  reflection prompt-map entries without changing text-chat prompt bytes.
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py` — update
  the existing unsupported reflection rejection to remain a negative case by
  using an unsupported reflection output mode or unsupported input-source tuple;
  do not weaken the text-chat assertions.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md`
  — record progress and evidence during execution.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  and `development_plans/README.md` — set Stage 07 to approved before
  execution; flip lifecycle rows to completed only after verification and
  independent code review pass.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- adapter, dispatcher, scheduler, and database modules

## Implementation Order

1. Reread Stage 06 `Execution Evidence`.
2. Add reflection episode-builder tests.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py -q`
   - Expected before implementation: import error for the dry-run module.
3. Implement `reflection_cycle/cognition_dry_run.py`.
   - Verify: compile the module and rerun Step 2 tests.
4. Add selector tests for reflection dry-run variant and existing rejection
   cases. Update
   `tests\test_multi_source_cognition_stage_03_prompt_selection.py` so its
   current reflection rejection remains negative through an unsupported
   reflection tuple.
5. Wire the selector variant, `build_cognition_prompt_source_payload(...)`,
   and prompt maps.
   - Verify: Stage 03 selector tests plus Stage 07 tests pass.
6. Add dry-run runner tests for busy probe, no writes, and one injected
   cognition call.
7. Add prompt-render tests with mocked LLMs for the reflection variant.
8. Add a text-chat prompt fingerprint guard in the Stage 07 test file using
   the exact byte lengths and SHA-256 digests listed in `Verification`.
9. Run the full Verification section.
10. Run the `Independent Code Review` gate and remediate in-scope findings.
11. Record evidence and sign off only after each checklist stage passes.

## Progress Checklist

- [ ] Stage 1 - prerequisite evidence carried forward.
  - Covers: Step 1.
  - Verify: Stage 06 row is `completed`.
  - Evidence: Stage 06 branch, commit, and test results recorded.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - reflection episode builder complete.
  - Covers: Steps 2-3.
  - Verify: focused builder tests pass after expected red import failure.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 - reflection prompt selection and prompt maps complete.
  - Covers: Steps 4-5.
  - Verify: Stage 03 and Stage 07 selector/render tests pass.
  - Evidence: prompt fingerprint and test output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - audit-only dry-run runner complete.
  - Covers: Steps 6-8.
  - Verify: no-write and prompt-render tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - full verification complete.
  - Covers: Step 9.
  - Verify: every Verification command passes or has an allowed no-match exit.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 6 - independent code review and lifecycle handoff complete.
  - Covers: Steps 10-11.
  - Verify: `Independent Code Review` gate passes, review findings are fixed
    or recorded as residual non-blockers, affected verification commands are
    rerun, and parent ledger plus registry rows are flipped to completed only
    after review approval.
  - Evidence: review findings, fixes, rerun commands, residual risks, and
    lifecycle row diffs recorded.
  - Handoff: Stage 08 draft is eligible for review after lifecycle update.
  - Sign-off: `<agent/date>` after review approval and lifecycle update.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\cognition_dry_run.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`

### Change Surface Gate

- Run this PowerShell gate from the feature branch:

  ```powershell
  $allowed = @(
      "src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py",
      "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py",
      "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py",
      "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py",
      "src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py",
      "tests/test_multi_source_cognition_stage_03_prompt_selection.py",
      "tests/test_multi_source_cognition_stage_07_reflection_dry_run.py",
      "development_plans/active/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md",
      "development_plans/active/short_term/multi_source_cognition_architecture_plan.md",
      "development_plans/README.md"
  )
  $changed = git diff --name-only main...HEAD
  $diff = Compare-Object -ReferenceObject $allowed -DifferenceObject $changed
  if ($diff) { $diff; exit 1 }
  ```

  Expected result: no output and exit code `0`. If any path appears, stop and
  update this plan before continuing.

### Static Greps

- `rg -n "call_consolidation_subgraph|save_conversation|save_assistant_message|dispatcher\\.dispatch|runtime\\.invalidate|get_rag_cache2_runtime" src\kazusa_ai_chatbot\reflection_cycle\cognition_dry_run.py`

  Expected result: no matches. `rg` exit code `1` is acceptable.

- `rg -n "\"reflection_signal\"|\"reflection_artifact\"" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`

  Expected result: no matches. Stage 07 must not wire reflection into service,
  persona supervisor, or RAG.

- `git diff --check`

### Prompt Fingerprint Guard

Add and run a Stage 07 test that checks these exact UTF-8 byte lengths and
SHA-256 digests for existing text-chat prompt constants:

| Prompt constant | Bytes | SHA-256 |
|---|---:|---|
| `_COGNITION_SUBCONSCIOUS_PROMPT` | 3884 | `93b4a80fa69aa7479d77699622aa632dd47a8515c475c91a0921bcdb302dc938` |
| `_COGNITION_CONSCIOUSNESS_PROMPT` | 11993 | `241fb639de242e2d7fc964da922a8b0ea2ac0d9c4f5b2b762df210c34805a5e5` |
| `_BOUNDARY_CORE_PROMPT` | 9694 | `dee7b322eb0d8637a3ee95b386560786042911cd0acca93b7c30896638ef26d1` |
| `_JUDGEMENT_CORE_PROMPT` | 6532 | `ca4e88cc3854cbdb63372ad3b20644575ef9eb74abdc8637212fedc0ca5b3b89` |
| `_CONTEXTUAL_AGENT_PROMPT` | 5112 | `4a2f7735c9f6b45637f329ad10581124360a24049444be43efb43cd2d802baae` |
| `_STYLE_AGENT_PROMPT` | 6430 | `c0f66e0d744688afa4b105f20573708d295057856fa924c0102c0d5605cb6340` |
| `_CONTENT_ANCHOR_AGENT_PROMPT` | 11088 | `9bf38821e24a561cec5c887f54432a4bff7b84131efb6c997d26edab8e0bbea0` |
| `_PREFERENCE_ADAPTER_PROMPT` | 7017 | `f5b0363c0d1ea1f28770237d27908cbfd56a86410c7c64d9522c44e1c284f88d` |
| `_VISUAL_AGENT_PROMPT` | 7597 | `68b1a35d43bfa28c46c91274d946faa9c7edf206f25fa414dabc822592626294` |

Expected result: every fingerprint matches. A mismatch is a blocker unless the
plan is updated and approved again.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Independent Plan Review

Run this gate before execution or handoff. Prefer a reviewer that did not draft
the plan. If no separate reviewer is available, the active agent must reread
the parent architecture plan, Stage 06 execution evidence, this plan, Stage 08
handoff expectations, and the relevant source/test contracts from a fresh-review
posture.

Review scope:

- Previous-stage artifacts are named, completed, and carried forward.
- The proposed scope aligns with the project modular design and the top-level
  architecture plan.
- The stage is unblocked: dependencies, decisions, status, registry rows, and
  required artifacts are present.
- The plan gives full, concrete instructions for an implementation agent:
  contracts, change surface, exact file paths, verification gates, progress
  checklist, and evidence requirements.
- Agent creativity is tightly bounded: no unresolved choices, broad verbs,
  optional fallbacks, compatibility shims, private helper freedom, or unowned
  side paths remain.
- Boundaries between Stage 06, Stage 07, and Stage 08 are explicit, with clean
  handoff and no overlapping or missing ownership.

Review result recorded on 2026-05-10:

- Blockers found: stale Stage 06 handoff evidence, missing independent code
  review gate, underspecified dry-run contracts, undefined empty-context
  behavior, open-ended prompt variant naming, and missing exact change-surface
  gate.
- Fixes applied: Stage 06 evidence carried forward; status and lifecycle set to
  approved; exact builder, selector, prompt-payload, runner, audit, skip, prompt
  key, LLM-budget, fingerprint, change-surface, and review contracts added.
- Approval status: approved for Stage 07 execution after these edits.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off,
lifecycle completion, merge, or handoff to Stage 08. Prefer a reviewer that did
not implement the change. If no separate reviewer is available, the active
agent must reread this plan, inspect the full diff from a fresh-review posture,
and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and
  path-safe commands for directories containing spaces.

Fix concrete findings directly only when the fix is inside the approved change
surface. If a fix would cross the approved boundary or alter the contract, stop
and update this plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

Stage 07 is complete when:

- reflection dry-run episode construction exists and validates;
- prompt selection supports only the approved reflection dry-run source tuple;
- prompt payload projection adds only `reflection_artifact` for reflection
  variants and never exposes `cognitive_episode`, `trigger_source`, or
  `input_sources`;
- text-chat prompt fingerprints and regression gates remain unchanged;
- dry-run audit calls injected cognition at most once and only when not busy;
- empty promoted reflection context returns a skipped audit without calling
  cognition;
- no dialog, consolidation, persistence, scheduler, cache, adapter, or RAG
  runtime path is enabled for reflection;
- independent code review passes; and
- parent ledger and registry rows are completed only after verification and
  independent code review pass.

## Plan Self-Review

Approval self-review on 2026-05-10:

- **Coverage:** parent Stage 07 scope maps to builder, selector, prompt-render,
  audit-runner, and regression checks.
- **Placeholder scan:** Stage 06 evidence is carried forward; no implementation
  decision is delegated to the agent.
- **Contract consistency:** trigger source, input source, and output modes match
  the parent architecture; exact variant, prompt key, builder, runner, audit,
  skip, payload, and empty RAG shapes are defined.
- **Granularity:** checkpoints split prerequisite evidence, builder, prompt
  selection, dry-run runner, full verification, and independent code review.
- **Verification:** no-write, no-service-wiring, prompt non-consumption, and
  prior-stage gates are explicit; change-surface and prompt-fingerprint gates
  are exact.

## Execution Handoff

Intended execution mode: sequential implementation on a feature branch forked
from post-Stage-06 `main`.

Next action: fork the Stage 07 branch from `main`, reread this approved plan,
load mandatory skills, and start at Progress Checklist Stage 1. If any
verification failure requires a file outside Change Surface or a contract
change, stop and update this plan before continuing.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Reflection is treated as user text | Separate builder and dry-run runner; no `/chat` entry | Static greps and dry-run tests |
| Raw reflection leaks into prompts | Use only promoted reflection context | Builder and prompt-render tests |
| Current `/chat` prompt drift | Preserve prompt bytes and rerun baseline | Prompt fingerprint and Stage 00 tests |
| Background work affects live latency | Busy probe and injected callable | Busy dry-run test |
| Reflection writes memory | No consolidation or persistence calls | Static greps and no-write tests |

## Completion Artifact Contract

When Stage 07 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`
- `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py`
- reflection dry-run selector variant
- `build_cognition_prompt_source_payload(...)`
- audit-only dry-run return shape
- parent ledger row for `stage_07` flipped to `completed`
- registry row flipped to `completed | completed`
- execution evidence in this plan naming branch, commit, checks, and sign-off

The artifact must not include service wiring, delivery, consolidation writes,
database persistence, scheduler dispatch, RAG broadening, or proactive output.

## Execution Evidence

Record during implementation:

- Stage 06 evidence reread: Stage 06 completed on branch
  `stage-06-consolidator-per-write-origin-policy`, commit `2ea7526`, with
  full deterministic suite `907 passed, 217 deselected`; parent ledger and
  registry rows show Stage 06 completed.
- Independent plan review: completed on 2026-05-10; blockers fixed in this
  plan; approved for execution.
- Branch:
- Commit:
- Static compile:
- Change surface gate:
- Static greps:
- Prompt fingerprint guard:
- Focused tests:
- Prior stage regression gates:
- Independent code review:
- Completion diff review:
- Lifecycle records:
- Sign-off:
