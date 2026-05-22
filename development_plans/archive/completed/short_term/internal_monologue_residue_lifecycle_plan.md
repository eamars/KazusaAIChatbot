# internal monologue residue lifecycle plan

## Summary

- Goal: Move the accepted internal monologue residue POC into the production
  codebase as a bounded, configurable rolling-window carry-over that explains
  why the character feels a certain way without using `reflection_summary` as
  hidden live cognition state.
- Plan class: high_risk_migration
- Status: completed
- Owner decision: POC evidence is good enough for self-cognition behavior;
  production implementation and plan cleanup are complete.
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `debug-llm`, `database-data-pull`, `py-style`,
  `test-style-and-execution`, and `cjk-safety` for Python files containing
  CJK prompt text.
- Overall cutover strategy: additive storage, bigbang prompt consumption,
  bigbang experiment cleanup.
- Highest-risk areas: third-person recorder voice, stale residue becoming
  action pressure, accidental direct consumers outside L2a, and leaving
  `reflection_summary` as a second live carry-over path.
- Acceptance criteria: L2a receives one compiled prompt-facing residue string
  built from a configurable rolling window; recorder output is first-person and
  bounded; empty or skipped residues are valid; raw prior residue is not
  consumed by L1, L2b, L2d, L3, dialog, adapter delivery, or the scheduler;
  experiments are removed.

## Context

The accepted POC showed the product value:

- Self-cognition can produce a mood shift such as guardedness, fatigue, or low
  mood for a concrete reason.
- A later `/chat` turn benefits when L2a sees that reason, not only the mood
  label.
- The rolling-window prompt improved quality because a strong recent signal can
  remain available and be amplified or softened by later evidence.
- Current-event evidence must remain able to override or soften old residue.

The production issue is not that Kazusa feels low. The issue is that the next
live cognition often receives `mood` / `global_vibe` without a clear private
cause. `reflection_summary` has been filling that gap unintentionally. This
plan replaces that hidden behavior with an explicit short-lived private
residue lane.

Important POC conclusions to preserve:

- The recorder prompt must be first-person because the stored text is fed back
  into future character cognition.
- Character name and ambient condition belong in the recorder system message.
- The recorder must know whether the environment is group or private.
- The recorder should not receive broad case-kind fields unless the field is
  directly needed for the semantic question.
- Mood and recent `reflection_summary` are not recorder inputs by default;
  they duplicate downstream state and increase the chance of circular residue.
- The prompt-facing carry-over is one string, not a packet of many semantic
  fields.
- Human-readable evaluation reports must be authored by the agent from raw
  trace data. Scripts may emit JSON or trace artifacts, but must not generate
  Markdown review reports.

## Mandatory Skills

- `development-plan-writing`: preserve lifecycle, progress checklist,
  execution evidence, and independent review gates.
- `local-llm-architecture`: keep prompt inputs minimal, semantic, and bounded;
  keep deterministic decisions outside prompts.
- `debug-llm`: audit recorder and L2a prompt quality, and author evaluation
  reports from raw trace data.
- `database-data-pull`: use read-only production diagnostics when needed; do
  not read `.env`.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files containing CJK prompt text.

## Mandatory Rules

- Use `venv\Scripts\python` for Python commands.
- Use `apply_patch` for manual edits.
- Use PowerShell `-LiteralPath` for filesystem paths that may contain spaces.
- Do not read `.env`.
- Do not mutate production data from diagnostics or tests.
- Do not store provider-hidden chain-of-thought. In this codebase,
  `internal_monologue` means the explicit generated character-facing thought
  artifact produced by L2a.
- Do not persist full prior `internal_monologue` as the carry-over unit.
  Persist only the compact recorder-produced residue string plus operational
  metadata needed for scope, ordering, validation, and audit.
- Do not make residue an action trigger, response-ratio controller,
  permission rule, delivery rule, scheduler rule, adapter rule, or memory
  promotion rule.
- Deterministic code owns scope selection, window size, ordering, age labels,
  prompt budget, validation, retry eligibility, persistence, and deletion of
  old experiment files.
- The recorder LLM owns only the semantic compression question: what private
  first-person reason should remain available to future cognition?
- One deterministic repair retry is allowed when a non-empty recorder candidate
  fails structural validation. The retry must receive the original semantic
  inputs plus the validation failure reason only. If the retry still fails,
  skip persistence and log a sanitized skip reason.
- Empty recorder output is valid when nothing should carry over. Empty output
  must not require `current_speaker_name`, exact names, or any other speaker
  validation.
- L2a Consciousness is the only direct live cognition consumer of prompt-facing
  internal monologue residue.
- L1, L2b, L2c1, L2c2, L2d, L3, dialog, adapter delivery, scheduler dispatch,
  and conversation-progress recording must not receive raw prior residue
  directly.
- Downstream stages may see residue influence only after L2a rewrites it into
  current-turn `internal_monologue`, `logical_stance`, and
  `character_intent`.
- `reflection_summary` must not remain a live carry-over input once residue is
  wired. It may remain as legacy/audit character-state text.
- Raw reflection output must not enter normal cognition directly.
- Runtime prompts added or modified by this plan must be written in Chinese.
  Follow the existing LLM prompt language policy: schema keys, JSON field
  names, IDs, URLs, code, commands, and model labels may remain in their
  required original language, but prompt instructions and generated free-text
  guidance must be Chinese.
- Any prompt modification must rewrite the affected full prompt block as one
  coherent logic flow. Do not append new constraint paragraphs, patch notes, or
  one-off correction blocks to the end of an existing prompt.
- Prompt audits must check for first-person perspective, group/private ambient
  condition, character name in the system message, minimal human payload, and
  absence of development/process terminology. The audit must also confirm the
  prompt was rewritten coherently instead of extended by appended corrections.
- Vague relation wording such as `对方` is allowed to pass through in this
  stage. Do not reject, retry, skip, or fail prompt audit solely because a
  residue uses vague relation wording.
- Scripts must not generate Markdown reports. Raw JSON trace is allowed; the
  agent authors Markdown review with `debug-llm`.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before completion, lifecycle status change, merge, or sign-off, run the
  Independent Code Review gate and record the result in Execution Evidence.

## Must Do

- Delete the internal-monologue-residue POC code and sequence files from
  `experiments/`, including ignored `__pycache__` files for those scripts.
- Move the accepted behavior into production code under a new production-owned
  module, `src/kazusa_ai_chatbot/internal_monologue_residue/`.
- Write `src/kazusa_ai_chatbot/internal_monologue_residue/README.md` as an ICD
  for the module. It must define document control, ownership boundary, runtime
  callers, public facade, storage contract, prompt contract, validation
  contract, lifecycle, configuration, telemetry, test seams, and forbidden
  consumers.
- Update both root project READMEs, `README.md` and `README_CN.md`, so the
  public architecture overview documents the internal monologue residue lane,
  L2a-only consumption, and the separation from `reflection_summary`.
- Introduce a new MongoDB collection named
  `internal_monologue_residue_state` for residue rows. Do not store residue in
  `character_state`, conversation progress, reflection collections, action
  attempts, or conversation history.
- Add production DB facade helpers and bootstrap/index support for residue
  persistence. Runtime callers must not use raw MongoDB handles outside
  `kazusa_ai_chatbot.db`.
- Add configurable rolling-window controls:
  - `INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE`, default `5`, min `1`, max `10`.
  - `INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT`, default `3000`, min
    `200`, max `3000`.
  - `INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT`, default `220`, min `80`, max
    `500`.
- Store each accepted recorder result as one first-person residue string plus
  operational metadata. Do not store a wide semantic packet.
- Project selected rows into one prompt-facing string. Include computed
  time-since-generation for each row. Do not expose raw DB ids, action ids,
  delivery ids, source packets, message bodies, or prior full monologues.
- Feed the compiled residue string only to L2a.
- Keep current input and current RAG/conversation evidence primary in L2a.
  Residue is soft private background.
- Remove `reflection_summary` from L1 live carry-over when L2a residue is
  wired. Do not leave dual carry-over.
- Add a recorder LLM prompt in Chinese using first-person framing. Character
  name and ambient condition must be in the system message.
- Implement recorder and L2a prompt edits as full prompt rewrites, not appended
  correction blocks.
- Keep recorder human payload minimal:
  `internal_monologue`, current speaker display name when applicable, exact
  name candidates, ambient evidence summary, and incoming rolling-window
  residue string.
- Do not pass mood or recent `reflection_summary` into the recorder unless a
  future approved plan proves the recorder cannot preserve causal residue
  without them.
- Validate recorder output before persistence:
  first-person voice, row character cap, no third-person self-reference such
  as `角色`, and no obvious prompt/process leakage.
- Do not add deterministic validation, repair retry, skip behavior, or tests
  that reject `对方`, `那个人`, `某人`, `他`, `她`, or similar vague relation
  wording in this stage.
- Implement one deterministic repair retry for invalid non-empty candidates.
- Treat empty, skipped, or no-carry results as successful no-write outcomes.
  They must not fail because a speaker name is absent.
- Record sanitized telemetry for load/write/skip/retry outcomes. Do not log
  residue text, raw prompts, source packets, or raw message bodies.
- Add deterministic tests for window selection, projection, validation, retry,
  vague-reference pass-through, empty-output handling, DB helper boundaries,
  prompt boundaries, and experiment cleanup.
- Add focused integration tests proving:
  - self-cognition can write a validated residue;
  - later `/chat` can load that residue into L2a;
  - normal `/chat` response-path LLM call count does not increase;
  - L1 no longer consumes `reflection_summary` as live carry-over.
- Audit the recorder and L2a prompt with `debug-llm` using raw trace data and
  an agent-authored Markdown review.

## Deferred

- Defer upstream attribution drift and exact-name enforcement. This plan allows
  vague relation wording to pass through; it does not redesign source
  attribution inside earlier cognition stages.
- Do not redesign RAG, conversation progress, adapters, dispatcher, scheduler,
  memory evolution, reflection promotion, or consolidation target routing.
- Do not add a response-ratio controller or deterministic speak/silence rule.
- Do not add a generic memory-provider abstraction.
- Do not add a broad emotion ontology.
- Do not backfill historical `reflection_summary` values into residue rows.
- Do not remove the `reflection_summary` field from persisted
  `character_state`.
- Do not make scheduled events carry raw residue.
- Do not feed raw residue directly to dialog or L3 content anchors.
- Do not keep production behavior dependent on files under `experiments/`.

## Cutover Policy

Overall strategy: additive storage, bigbang prompt consumption, bigbang
experiment cleanup.

- New production module: compatible. Add
  `src/kazusa_ai_chatbot/internal_monologue_residue/` alongside existing
  modules.
- New DB collection and helpers: compatible. Add
  `internal_monologue_residue_state`; an empty collection means empty
  prompt-facing residue.
- Experiment files: bigbang. Delete the POC scripts and sequence JSON files
  from `experiments/`; production code and tests become the only executable
  implementation.
- Recorder: compatible. Start writing new residue rows after completed
  episodes without deleting `character_state.reflection_summary`.
- L2a consumption: bigbang. Once wired, L2a consumes the compiled residue
  string as the only direct prompt-facing residue input.
- L1 `reflection_summary` consumption: bigbang. Remove live carry-over from L1
  during the same integration stage that adds L2a residue.
- Existing `reflection_summary` storage: compatible. Keep for legacy/audit
  display only.
- Historical data: compatible. Do not backfill.
- Tests: bigbang. Prompt contract tests must enforce the new consumer boundary.

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a dual-prompt transition by default.
- For bigbang areas, delete or rewrite legacy references instead of preserving
  both old and new paths.
- For compatible areas, preserve only the surfaces explicitly listed here.
- Any change to cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose private helper names only when public contracts and
  ownership boundaries remain intact.
- The agent must not add feature flags, alternate fallback paths,
  compatibility prompt shims, broad registries, extra response-path LLM calls,
  or unrelated cleanup.
- Changes outside the listed Change Surface require plan revision or explicit
  user approval.
- If existing helpers can be reused inside the approved surface, reuse them.
- If plan and code disagree, preserve this plan's stated intent and record the
  discrepancy before changing code.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

```text
completed chat or self-cognition episode
  -> post-episode recorder reads current explicit L2a internal_monologue
  -> recorder returns at most one compact first-person residue string
  -> deterministic validation accepts, repairs once, or skips
  -> DB stores the residue string with scope and created_at metadata
  -> next relevant trigger loads the latest N rows by deterministic scope
  -> projection compiles them into one age-labeled residue string
  -> L2a uses that string as soft private background
  -> L2a emits current internal_monologue, stance, and intent
  -> downstream stages consume only current derived cognition
```

The residue answers:

```text
What private reason still colors my current interpretation?
```

It does not answer:

```text
Should I speak?
What action should I take?
What should I tell the user verbatim?
What durable memory should be written?
```

## Design Decisions

- Direct live consumer: L2a only.
- Prompt-facing shape: one string named
  `internal_monologue_residue_context`.
- Storage shape: one residue string per accepted row plus operational metadata.
- Window strategy: deterministic rolling window, not per-row semantic decay
  fields.
- Decay signal: computed age labels in the compiled string plus recency order.
- Recorder perspective: first person, because the output re-enters the
  character's own cognition.
- Recorder model route: use the existing `COGNITION_LLM` route unless a later
  approved routing plan adds a dedicated route.
- Recorder retry: one deterministic repair retry for invalid non-empty output,
  excluding vague relation wording.
- Empty output: valid no-write result.
- Vague references: allowed pass-through in this stage; do not reject or retry
  solely because output contains wording such as `对方`.
- Mood and `reflection_summary`: excluded from recorder input by default.
- Existing `reflection_summary`: storage-compatible, no longer live carry-over.
- Experiments: removed after production migration.

## Contracts And Data Shapes

### Production Module

Create:

```text
src/kazusa_ai_chatbot/internal_monologue_residue/
```

The module README must be ICD-styled, following the repository's subsystem
README pattern. It must include:

- document control and owning package;
- module purpose and non-goals;
- runtime caller and DB boundary;
- public facade contract;
- storage row contract;
- prompt contract and prompt-authoring rules;
- validation and retry contract;
- lifecycle and configuration;
- telemetry contract;
- tests and audit gates;
- forbidden consumers and forbidden data leakage.

Public facade:

```python
async def load_residue_context(
    *,
    trigger_scope: ResidueTriggerScope,
    current_timestamp_utc: str,
) -> ResidueLoadResult: ...

async def record_completed_episode_residue(
    *,
    completed_state: Mapping[str, Any],
    current_timestamp_utc: str,
) -> ResidueRecordResult: ...

def project_residue_window(
    *,
    rows: Sequence[InternalMonologueResidueRow],
    current_timestamp_utc: str,
    context_char_limit: int,
) -> str: ...
```

Runtime code imports from the module facade, not storage internals.

### Storage Row

Collection name:

```text
internal_monologue_residue_state
```

Conceptual row:

```python
{
    "residue_id": str,
    "character_id": str,
    "scope_key": str,
    "scope_kind": "character_global" | "group_scene" | "user_thread",
    "platform": str,
    "platform_channel_id": str,
    "channel_type": str,
    "global_user_id": str,
    "residue_text": str,
    "source_kind": "chat" | "self_cognition",
    "source_refs": list[dict[str, str]],
    "created_at": str,
}
```

Rules:

- `residue_text` is the only semantic carry-over text.
- `scope_key`, `scope_kind`, platform fields, `global_user_id`, `source_kind`,
  `source_refs`, and `created_at` are operational metadata.
- `source_refs` must contain sanitized identifiers only, not raw message text.
- Do not store full prior `internal_monologue`.
- Do not store recorder prompt text.
- Do not store final dialog as residue.
- Do not store residue rows in `character_state`, conversation progress,
  reflection collections, action attempts, or conversation history.

### Scope Selection

Use deterministic scope keys:

- `character_global`: character-level private residue from self-cognition that
  can explain later mood in `/chat`.
- `group_scene`: same platform and group channel.
- `user_thread`: same platform, channel or private surface, and global user id.

Load candidates for the current trigger, rank exact user thread first, group
scene second, character global third, then keep the newest
`INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE` rows across the eligible set.

### Prompt-Facing Projection

L2a receives one string, for example:

```text
最近仍可能影响我的私念残留，越新的越重要：
- 8分钟前，关于当前群聊：我还记得 Tobacco 用提拉米苏和赌约逗我，让我既期待又防备。
- 2小时前，关于整体心境：我被冷场和反复旁观耗得有些低落，所以会先确认 Tobacco 是不是真的在接住我。
```

Projection rules:

- Return an empty string when no row is eligible.
- Include time-since-generation computed at projection time.
- Keep the whole string within `INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT`.
- Prefer newer rows when the character limit is tight.
- Do not expose raw IDs, source packets, action metadata, delivery metadata,
  message bodies, or full previous monologues.

### Recorder Prompt Contract

System message owns stable identity and environment:

```text
我叫 {character_name}。
我现在整理的是{ambient_condition}里的私念残留。
```

Human message owns only current-run material:

```json
{
  "internal_monologue": "current explicit L2a monologue",
  "incoming_window_context": "compiled residue string before this turn",
  "current_speaker_name": "display name or empty",
  "exact_name_candidates": ["display names"],
  "ambient_evidence": "short group/private scene descriptor"
}
```

Recorder output:

```json
{
  "residue_text": "first-person residue string, or empty string"
}
```

Prompt rules:

- Prompt instructions must be Chinese. Keep JSON keys and required schema
  labels unchanged.
- Rewrite the full prompt when changing it. Do not append late correction
  blocks after the main task, name policy, or output contract.
- Write as my private first-person residue.
- Preserve the reason behind my feeling, expectation, guard, fatigue, or
  softening.
- Prefer exact display names when a person is part of the cause, but do not
  force this when the recorder naturally writes vague relation wording.
- If no meaningful private cause remains, output an empty string.
- Do not write visible dialog, action instructions, durable user facts, or
  process labels.

### Validation And Retry

Validate non-empty recorder output:

- `residue_text` is a string and within `INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT`.
- Text is first-person and does not describe the character as `角色`,
  `{character_name}`, or another third-person subject.
- Text does not contain obvious prompt/process terms such as `system message`,
  `语义表达层`, prompt field names, or recorder instructions.
- Do not validate against vague relation wording. `对方`, `那个人`, `某人`,
  `他`, `她`, and similar wording pass through when the other validation checks
  pass.

If validation fails for a non-empty candidate, run one repair retry with the
failure reason. If the retry fails, skip persistence and emit sanitized
telemetry.

Empty output bypasses speaker/name validation and records a no-write result.

## LLM Call And Context Budget

Before:

- L1 receives `character_profile.reflection_summary` as live affective
  background.
- L2a receives mood and current evidence but no first-class causal residue.
- Self-cognition can update `mood`, `global_vibe`, and `reflection_summary`.

After:

- L1 receives current mood, global vibe, relationship intuition, and current
  stimulus evidence, but not `reflection_summary` as live carry-over.
- L2a receives one compiled residue string.
- Recorder runs only after the episode has completed.

Budget rules:

- Normal `/chat` visible response-path LLM call count must not increase.
- Recorder calls must run after response completion or in background.
- Self-cognition may run the recorder during its background path.
- Recorder uses one LLM call normally and at most one repair retry.
- L2a residue context is capped by
  `INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT`.
- Each stored row is capped by `INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT`.

## Change Surface

Create:

- `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`
- `src/kazusa_ai_chatbot/internal_monologue_residue/__init__.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/models.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/projection.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/loader.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py`
- `src/kazusa_ai_chatbot/internal_monologue_residue/runtime.py`
- `tests/test_internal_monologue_residue_projection.py`
- `tests/test_internal_monologue_residue_loader.py`
- `tests/test_internal_monologue_residue_recorder.py`
- `tests/test_internal_monologue_residue_prompt_boundaries.py`
- `tests/test_internal_monologue_residue_integration.py`

Modify:

- `README.md`
- `README_CN.md`
- `src/kazusa_ai_chatbot/config.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py` or the current bootstrap owner
- `src/kazusa_ai_chatbot/db/README.md`
- add a DB-owned helper module under `src/kazusa_ai_chatbot/db/`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/brain_post_turn.py` or the current service-owned
  background post-turn owner
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- prompt contract and self-cognition tests named in Verification

Delete:

- `experiments/internal_monologue_residue_poc.py`
- `experiments/internal_monologue_residue_multiturn_poc.py`
- `experiments/internal_monologue_residue_decay_poc.py`
- `experiments/internal_monologue_residue_window_poc.py`
- `experiments/internal_monologue_residue_window_compare_poc.py`
- `experiments/internal_monologue_residue_window_compare_ten_turn_poc.py`
- `experiments/internal_monologue_residue_six_turn_sequence.json`
- `experiments/internal_monologue_residue_ten_turn_sequence.json`
- generated `experiments/__pycache__/internal_monologue_residue*.pyc`

Keep out of scope:

- `src/kazusa_ai_chatbot/dialog_agent.py`: must not consume raw residue.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`: must
  not consume raw residue.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2c2.py`: must
  not consume raw residue.
- adapters: no adapter changes.
- scheduler and dispatcher: no delivery, permission, or scheduling behavior
  changes.
- `conversation_progress`: keep external conversation-flow memory separate
  from private affective residue.

## Data Migration

No backfill is approved.

Storage work is additive:

- create the new collection/indexes through DB bootstrap;
- tolerate missing or empty collection;
- do not mutate existing `character_state` rows;
- do not delete `reflection_summary` from storage;
- do not migrate historical `reflection_summary` into residue rows.

## Overdesign Guardrail

- Actual problem: mood can carry from self-cognition into `/chat`, but the
  next cognition lacks the reason unless it relies on overloaded
  `reflection_summary`.
- Minimal change: store compact first-person residue strings, select a bounded
  rolling window, compile one prompt-facing string, and feed it only to L2a.
- Ownership boundaries: deterministic code owns scope, window, age labels,
  validation, retry eligibility, caps, persistence, and telemetry; recorder LLM
  owns only semantic compression; L2a owns current interpretation; downstream
  stages consume only current derived cognition.
- Rejected complexity: no broad packet, no mood field, no decay fields, no
  response gating, no action trigger, no direct dialog/L3/L2d consumption, no
  backfill, no experiment dependency, no new model route, no adapter changes,
  and no generic memory provider.
- Evidence threshold for future expansion: add fields, consumers, longer
  windows, new routes, or semantic decay only after production diagnostics show
  the one-string rolling window fails a concrete use case.

## Implementation Order

1. Load mandatory skills and reread this plan.
2. Update module contract tests first.
   - Add projection, loader, recorder, prompt-boundary, and integration tests.
   - Run focused tests and record expected missing-symbol or missing-behavior
     failures.
3. Add configuration parsing.
   - Add the three `INTERNAL_MONOLOGUE_RESIDUE_*` settings with fail-fast
     validation.
   - Add tests for defaults and invalid values if the project has config tests.
4. Implement the production module.
   - Add models, projection, validation, recorder, loader, and runtime facade.
   - Keep public interface narrow.
   - Write the module README as an ICD before treating the module contract as
     complete.
   - Write the recorder prompt in Chinese as a coherent full prompt block.
5. Add DB facade helpers and bootstrap/index support.
   - Add semantic helpers for insert, latest-window query, and sanitized
     cleanup if needed.
   - Create/bootstrap indexes for `internal_monologue_residue_state`.
   - Keep raw MongoDB access inside the DB package.
6. Wire loader into normal `/chat`.
   - Load eligible residue before cognition without adding response-path LLM
     calls.
7. Wire loader into self-cognition.
   - Self-cognition uses the same production module, not experiment code.
8. Wire L2a prompt consumption.
   - Add `internal_monologue_residue_context` to L2a human payload only.
   - Rewrite the full L2a prompt in Chinese with soft-background guidance.
     Do not append a residue paragraph onto the old prompt.
9. Remove L1 `reflection_summary` live carry-over.
   - Delete live prompt binding from L1.
   - Keep storage and consolidation compatibility.
10. Wire post-episode recorder.
    - Normal `/chat`: run after response completion through the service-owned
      post-turn path.
    - Self-cognition: run after completed cognition/action/dialog state exists.
11. Add sanitized telemetry.
    - Record loaded/written/skipped/retried status and scope labels only.
12. Delete experiment POC files.
13. Update root READMEs and the module ICD.
    - Update `README.md` and `README_CN.md`.
    - Ensure `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`
      documents the production interface and ownership boundary.
14. Run prompt audit with `debug-llm`.
    - Use raw trace data.
    - The agent writes the Markdown review; scripts do not.
15. Run full verification.
16. Run independent code review and remediate findings inside the approved
    change surface.

## Progress Checklist

- [x] Stage 1 - contract tests added and red
  
  - Covers: Implementation Order step 2.
  - Verify: focused residue tests fail only for planned missing symbols or
    behavior.
  - Evidence: focused test batch fails during collection with
    `ModuleNotFoundError: No module named
    'kazusa_ai_chatbot.internal_monologue_residue'`.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 2 - config and production module implemented
  
  - Covers: steps 3-4.
  - Verify: py_compile and focused projection/recorder tests pass; module
    README is ICD-styled.
  - Evidence: `src/kazusa_ai_chatbot/config.py` defines the three bounded
    `INTERNAL_MONOLOGUE_RESIDUE_*` settings; production module lives under
    `src/kazusa_ai_chatbot/internal_monologue_residue/`; focused residue
    suite passed with `18 passed in 1.91s`; static compile passed.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 3 - DB facade and bootstrap complete
  
  - Covers: step 5.
  - Verify: DB helper tests pass; static grep shows no raw MongoDB access
    outside DB internals for residue.
  - Evidence: DB facade added in
    `src/kazusa_ai_chatbot/db/internal_monologue_residue.py`; bootstrap and
    schema support added for `internal_monologue_residue_state`; focused
    residue integration tests passed with the full focused suite.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 4 - loaders wired into chat and self-cognition
  
  - Covers: steps 6-7.
  - Verify: integration tests prove eligible residue loads and empty collection
    degrades to empty context.
  - Evidence: chat loader wiring is in
    `src/kazusa_ai_chatbot/service.py`; self-cognition loader wiring is in
    `src/kazusa_ai_chatbot/self_cognition/runner.py`; focused residue
    integration tests passed, including configured character-id fallback.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 5 - L2a-only prompt consumption complete
  
  - Covers: steps 8-9.
  - Verify: prompt-boundary tests prove only L2a receives residue and L1 no
    longer receives `reflection_summary` carry-over.
  - Evidence: prompt-boundary tests passed; static grep for
    `internal_monologue_residue_context` in L1/L2d/L3/dialog produced no
    matches; static grep for `reflection_summary` in L1 produced no matches.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 6 - post-episode recorder wired
  
  - Covers: steps 10-11.
  - Verify: completed chat and self-cognition episodes write validated compact
    residue or sanitized no-write outcomes.
  - Evidence: chat post-turn recorder is in
    `src/kazusa_ai_chatbot/brain_service/post_turn.py` and service
    background wiring; self-cognition recorder is called after consolidation;
    focused integration test
    `test_post_turn_records_internal_monologue_residue_in_background` passed.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 7 - experiment cleanup and docs complete
  
  - Covers: steps 12-13.
  - Verify: `rg --files -u experiments | rg "internal_monologue_residue"`
    has no residue POC matches; `README.md`, `README_CN.md`, and module ICD
    document production behavior.
  - Evidence: experiment grep produced no matches; README grep found both
    root READMEs documenting the residue lane and L2a-only boundary.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 8 - prompt audit complete
  
  - Covers: step 14.
  - Verify: `debug-llm` review records no prompt blockers, confirms Chinese
    prompt instructions, and confirms prompt edits are coherent rewrites rather
    than appended correction blocks.
  - Evidence: agent-authored review saved at
    `test_artifacts/llm_reviews/internal_monologue_residue_prompt_audit_20260522.md`;
    raw traces saved under `test_artifacts/llm_traces/`, including the live
    L2a trace and recorder no-write trace.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 9 - full verification complete
  
  - Covers: step 15.
  - Verify: every Verification command passes or has an approved blocker.
  - Evidence: focused residue suite `18 passed in 1.91s`; adjacent
    deterministic suite `62 passed in 2.74s`; static compile passed; static
    greps matched expected no-output boundaries; `git diff --check` passed
    with CRLF warnings only; live LLM cognition smoke passed and was
    inspected.
  - Sign-off: `Codex/2026-05-22`.

- [x] Stage 10 - independent code review complete
  
  - Covers: step 16.
  - Verify: review findings are closed or explicitly accepted as residual
    risk.
  - Evidence: independent reviewer found three issues: wrong-schema recorder
    output could become false empty no-write, nodes README had stale L1
    reflection carry-over wording, and residue ICD lacked document-control
    metadata. All three were fixed and reviewer re-check reported no remaining
    blockers.
  - Sign-off: `Codex/2026-05-22`.

## Verification

Run from repository root with the project virtual environment.

Static compile:

```powershell
venv\Scripts\python -m py_compile `
  src\kazusa_ai_chatbot\internal_monologue_residue\models.py `
  src\kazusa_ai_chatbot\internal_monologue_residue\projection.py `
  src\kazusa_ai_chatbot\internal_monologue_residue\loader.py `
  src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py `
  src\kazusa_ai_chatbot\internal_monologue_residue\runtime.py
```

Focused tests:

```powershell
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_projection.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_loader.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_recorder.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_integration.py -q
```

Adjacent integration tests:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests\test_service_background_consolidation.py -q
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py -q
venv\Scripts\python -m pytest tests\test_cognition_live_llm_prompt_contracts.py -q
```

Live LLM tests from the listed files must run one case at a time with output
inspected under `test-style-and-execution` and `debug-llm`.

Static greps:

```powershell
rg -n "internal_monologue_residue_context" `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py `
  src\kazusa_ai_chatbot\nodes\dialog_agent.py
```

Expected: no matches. Exit code `1` is acceptable.

```powershell
rg -n "reflection_summary|character_reflection_summary" `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py
```

Expected: no live prompt consumption of `reflection_summary`.

```powershell
rg -n "internal_monologue_residue.*scheduled|scheduled.*internal_monologue_residue" `
  src\kazusa_ai_chatbot
```

Expected: no scheduled-event raw residue payload path.

```powershell
rg -n "_id|action_attempt_id|delivery_tracking_id|adapter_message_id" `
  src\kazusa_ai_chatbot\internal_monologue_residue\projection.py
```

Expected: no prompt-facing projection of raw storage/action/delivery ids.

```powershell
rg --files -u experiments | rg "internal_monologue_residue"
```

Expected: no matches after cleanup. Exit code `1` is acceptable.

```powershell
rg -n "internal monologue residue|internal_monologue_residue|私念残留|内心独白残留" `
  README.md README_CN.md
```

Expected: both root README files document the feature at architecture level.

```powershell
git diff --check
```

Expected: no whitespace errors.

Live DB diagnostics:

- Use only when MongoDB is available and the user authorizes production
  inspection.
- Export bounded recent residue rows after implementation.
- Confirm rows are stored in `internal_monologue_residue_state`.
- Confirm rows contain compact first-person `residue_text`, operational scope,
  and `created_at`.
- Confirm prompt-facing projection excludes raw ids and message bodies.
- Confirm self-cognition-created `character_global` residue can be selected
  for later `/chat`.

## Independent Plan Review

Run before marking this plan `approved`.

Review scope:

- Architecture alignment with adapter -> brain service -> queue/intake -> RAG
  -> cognition -> dialog -> persistence/consolidation -> scheduler/reflection.
- L2a-only direct consumption.
- Removal of `reflection_summary` as live carry-over.
- One-string rolling-window projection.
- Configurable window and prompt budgets.
- First-person recorder prompt quality.
- Chinese prompt instructions and full-prompt rewrite discipline.
- Vague-reference pass-through and deterministic repair retry for structural
  failures.
- ICD-styled module interface documentation.
- Empty/no-write recorder handling.
- Experiment cleanup and production ownership.
- Verification coverage for boundaries, prompt shape, storage caps, retry, and
  self-cognition-to-chat integration.

Record blockers, non-blocking findings, required edits, and approval status in
Execution Evidence. Do not approve while blockers remain.

## Independent Code Review

Run after all Verification commands pass and before final sign-off.

Review scope:

- Project style and skill compliance.
- Plan alignment with Must Do, Deferred, L2a-only consumption, one-string
  projection, cutover policy, and verification gates.
- Code quality and design risk, especially hidden fallback paths, stale
  `reflection_summary` prompt use, accidental vague-reference rejection, raw
  id leakage, raw monologue persistence, direct L2d/dialog consumption,
  response-path LLM call increases, appended prompt correction blocks, missing
  ICD module contract, and accidental adapter/scheduler changes.
- Regression quality across focused residue tests, prompt contract tests,
  self-cognition integration, service post-turn behavior, and sanitized DB
  diagnostics.

Fix findings only when the fix is inside this plan's approved change surface.
If a finding requires a new public contract or broader architecture, stop and
revise the plan before changing code.

## Acceptance Criteria

This plan is complete when:

- Internal monologue residue POC files are removed from `experiments/`.
- Production residue code lives under
  `src/kazusa_ai_chatbot/internal_monologue_residue/`.
- Root `README.md` and `README_CN.md` document the new production behavior.
- Residue rows use the new MongoDB collection
  `internal_monologue_residue_state`.
- DB access goes through DB facade helpers.
- Residue window size and prompt budgets are configurable through the approved
  settings.
- Each stored residue carries one compact first-person `residue_text`.
- Projection compiles selected rows into one age-labeled string for L2a.
- Module README is ICD-styled and defines the production interface contract.
- Recorder prompt is audited and uses first-person Chinese framing with
  character name and ambient condition in the system message.
- Recorder and L2a prompts are rewritten as coherent full Chinese prompt
  blocks when modified; no appended correction blocks remain.
- Recorder validation rejects third-person self-reference and prompt/process
  leakage, and one deterministic repair retry is implemented for structural
  failures.
- Vague references such as `对方` are allowed to pass through at this stage.
- Empty recorder output is a valid no-write result and does not fail speaker
  validation.
- Self-cognition can write a validated residue explaining the reason behind a
  low or guarded mood.
- A later `/chat` can load that residue into L2a.
- L2a is the only direct live cognition consumer of prompt-facing residue.
- L1 no longer uses `reflection_summary` as live emotional carry-over.
- L2b, L2d, L3, dialog, adapters, and scheduler do not receive raw prior
  residue directly.
- Normal `/chat` response-path LLM call count does not increase.
- All Verification gates pass.
- `debug-llm` prompt audit passes with an agent-authored Markdown review.
- Independent code review approves the implementation.
- Registry and plan lifecycle records follow `development_plans/README.md`.

## Plan Self-Review

Performed by Codex on 2026-05-22 while rewriting the draft after owner cleanup
decisions.

- Placeholder scan: no unresolved placeholder or option remains.
- Minimality: the previous broad residue packet was removed; prompt-facing
  carry-over is one string.
- Execution relevance: experiment-first work was replaced by production
  module work and experiment deletion.
- Risk handling: first-person perspective, empty no-write handling,
  vague-reference pass-through, and prompt audit are explicit gates.
- Lifecycle: plan status is `completed`; owner approved production
  implementation after POC cleanup decisions, and execution evidence has been
  recorded before archive.

## Independent Plan Audit Result

Performed by Codex on 2026-05-22 from a fresh-review posture after rewriting
the plan.

- Inputs reviewed: updated plan, `development_plans/README.md`, plan contract
  references, `README.md`, `docs/HOWTO.md`, nodes README, self-cognition
  README, conversation-progress README, DB README, and the current L1/L2a
  source boundaries.
- Findings fixed during audit: narrowed the experiment-cleanup grep to
  residue-specific files, removed old packet-era `settled` wording, and
  removed placeholder wording from the self-review line.
- Owner revision after audit: exact-name enforcement is relaxed for this
  stage; vague relation wording such as `对方` must pass through validation and
  prompt audit.
- Owner revision after audit: prompt instructions must be Chinese, prompt edits
  require coherent full-prompt rewrites instead of appended blocks, and the new
  module interface must be documented in ICD style.
- Owner revision after audit: root `README.md` and `README_CN.md` must be
  updated, and residue persistence uses the new
  `internal_monologue_residue_state` collection.
- Approval blockers after fixes: none found in the plan text.
- Status decision: production implementation passed the independent code
  review gate and is complete.

## Risks

- Residue becomes hidden action pressure.
  Mitigation: L2d/dialog never receive raw residue; prompt-boundary tests and
  static greps enforce this.
- Stale private reason colors unrelated chats.
  Mitigation: deterministic scope ranking, configurable window size, age
  labels, and context budget.
- Recorder stores vague references.
  Mitigation: accepted as non-blocking for this stage; prompt may prefer exact
  names, but validation must not reject or retry vague relation wording.
- Recorder voice shifts to third person.
  Mitigation: first-person prompt, validation, retry, and prompt audit.
- Empty/no-carry cases fail validation.
  Mitigation: empty output bypasses speaker/name validation and records a
  no-write outcome.
- `reflection_summary` remains a parallel live carry-over path.
  Mitigation: bigbang L1 prompt removal with static grep and prompt contract
  tests.
- Normal chat latency increases.
  Mitigation: recorder runs post-episode/background; response-path LLM count
  must not increase.
- Event logs leak sensitive text.
  Mitigation: log labels/status/scope only, not residue text or prompts.

## Execution Evidence

- 2026-05-22 Stage 1 red tests:
  `venv\Scripts\python -m pytest
  tests\test_internal_monologue_residue_projection.py
  tests\test_internal_monologue_residue_loader.py
  tests\test_internal_monologue_residue_recorder.py
  tests\test_internal_monologue_residue_prompt_boundaries.py
  tests\test_internal_monologue_residue_integration.py -q`
  exited `1` during collection because
  `kazusa_ai_chatbot.internal_monologue_residue` does not exist yet. This is
  the expected missing-module failure for the new production module.

- 2026-05-22 production implementation checkpoint:
  - New production module:
    `src/kazusa_ai_chatbot/internal_monologue_residue/`.
  - New DB facade and collection:
    `src/kazusa_ai_chatbot/db/internal_monologue_residue.py` and
    `internal_monologue_residue_state`.
  - Chat and self-cognition both load residue into L2a only; post-episode
    recorder writes compact validated rows or records sanitized no-write
    outcomes.
  - Configured character-id fallback was normalized to
    `CHARACTER_GLOBAL_USER_ID` for both self-cognition loading and recorder
    row writing.

- 2026-05-22 verification checkpoint:
  - Focused residue tests:
    `venv\Scripts\python -m pytest
    tests\test_internal_monologue_residue_projection.py
    tests\test_internal_monologue_residue_loader.py
    tests\test_internal_monologue_residue_recorder.py
    tests\test_internal_monologue_residue_prompt_boundaries.py
    tests\test_internal_monologue_residue_integration.py -q`
    passed with `18 passed in 1.91s`.
  - Static compile of residue, DB, service, self-cognition, L1, and L2 files
    passed.
  - Adjacent deterministic tests:
    `venv\Scripts\python -m pytest
    tests\test_self_cognition_integration.py
    tests\test_service_background_consolidation.py
    tests\test_cognition_prompt_contract_text.py -q`
    passed with `62 passed in 2.74s`.
  - Static greps found no direct raw residue consumer in L1, L2d, L3, or
    dialog; no `reflection_summary` use in L1 prompt; no scheduled raw residue
    path; no raw id projection; no residue POC files under `experiments/`.
  - `git diff --check` passed with CRLF warnings only.
  - Live LLM cognition smoke passed:
    `tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_photo_request_chinese`
    with `-m live_llm`.
  - Live recorder smoke produced accepted empty no-write output for a light
    private photo/weather case.
  - Agent-authored debug review:
    `test_artifacts/llm_reviews/internal_monologue_residue_prompt_audit_20260522.md`.

- 2026-05-22 independent code review checkpoint:
  - Reviewer reported three findings and all were remediated.
  - Added regression test:
    `tests\test_internal_monologue_residue_recorder.py::test_record_completed_episode_retries_wrong_schema_output`.
    It failed before the fix and passed after the recorder parsed-payload
    validation change.
  - Post-fix focused residue suite passed with `19 passed in 1.93s`.
  - Post-fix adjacent deterministic suite passed with `62 passed in 2.80s`.
  - Reviewer re-check reported no remaining blockers; residual risk noted by
    reviewer was limited to the re-check scope, while the parent agent reran
    the broader deterministic verification listed above.

- 2026-05-22 recorder prompt style alignment:
  - `_RECORDER_PROMPT` was rewritten as a coherent full prompt block to align
    with cognition/consolidation prompt structure:
    `# 语言政策`, `# 核心任务`, `# 证据身份`, `# 生成步骤`,
    `# 私念视角契约`, `# 输入格式`, and `# 输出格式`.
  - Functional contract stayed unchanged: one `residue_text` string, first
    person, empty no-write allowed, vague relation wording allowed, and no
    new model-facing fields.
  - Focused residue suite passed with `19 passed in 1.88s`.
  - Live recorder smoke for a light private photo/weather case produced an
    accepted empty no-write result.
  - Agent-authored debug review:
    `test_artifacts/llm_reviews/internal_monologue_residue_prompt_style_review_20260522.md`.

- 2026-05-22 lifecycle cleanup:
  - Plan status changed to `completed`.
  - Plan moved from `development_plans/active/short_term/` to
    `development_plans/archive/completed/short_term/`.
  - Registry updated so completed residue work is no longer executable from
    `active/`.

## Glossary

- `internal_monologue`: explicit generated character-facing thought produced
  by L2a for the current cognition.
- `internal_monologue_residue`: compact private first-person carry-over string
  distilled from completed cognition.
- `internal_monologue_residue_context`: the one compiled prompt-facing string
  consumed by L2a.
- `rolling window`: latest N eligible residue rows selected by deterministic
  scope and recency.
- `time-since-generation`: human-readable age label computed during projection.
- `character_global`: character-level private residue from self-cognition that
  can explain later mood in `/chat`.
- `group_scene`: group-channel residue scoped to the same platform and group.
- `user_thread`: user-scoped residue for the same platform and interaction
  surface.
- `reflection_summary`: legacy/current singleton character-state summary that
  must stop acting as live cognition carry-over in this plan.
