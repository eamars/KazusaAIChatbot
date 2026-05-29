# self cognition group review participant context plan

## Summary

- Goal: implement the successful flow-focused participant-context POC for
  `group_chat_review` production self-cognition, while reducing the POC
  contract to the smallest prompt-facing shape needed for target discipline.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang for group-review participant-context
  source hydration; compatible for existing self-cognition runner, delivery
  target, RAG2 supervisor, conversation progress, reflection activity-window,
  and action-attempt shapes
- Highest-risk areas: turning participant context into a roster prompt,
  adding full RAG latency to every group-review slot, fabricating a semantic
  user target for group review, leaking storage identifiers into prompts, and
  causing the character to address every visible participant instead of the
  current social beat, and leaving experiment-only POC code in place after the
  production path supersedes it.
- Acceptance criteria: production group-review cases receive bounded
  flow-focused participant context before cognition; group review still does
  not invoke the full RAG supervisor; prompt-facing context contains one
  primary social beat plus aggregate background-flow context; deterministic
  tests and one-at-a-time real LLM comparison pass; experiment-only POC code is
  removed before code review; independent code review approves the
  implementation.

## Context

Current production path:

```text
reflection cycle
  -> collect_group_chat_review_cases(...)
  -> build_group_activity_windows(...)
  -> _build_group_review_case(...)
  -> runner.build_self_cognition_case_artifacts_async(...)
  -> projection.build_source_packet(...)
  -> shared cognition graph
  -> selected L3/dialog only when L2d selects speak
```

`group_chat_review` is not in `RAG_BACKED_CASE_NAMES`, so the runner does not
call `projection.build_rag_request(...)` or `call_rag_supervisor(...)` for
group review. That boundary must remain.

The experiment showed that a participant mini-retrieval path improves active
group-review speech by giving cognition relationship and continuity evidence.
The roster-shaped POC also showed a failure: when the prompt listed every
participant with equal-looking evidence, the character sometimes tried to
reply to everyone. The second POC fixed that failure by collapsing the
participant evidence into one flow focus.

This plan implements the production version as a source hydration step for
group-review cases. It does not implement the experiment's roster contract.
It does not make group review a normal RAG-backed self-cognition case.

## Mandatory Skills

- `development-plan`: load before editing this plan, registry rows, lifecycle
  status, execution evidence, or sign-off state.
- `local-llm-architecture`: load before changing prompt-facing context,
  self-cognition source packets, RAG helper use, cognition state, dialog
  inputs, or LLM call budgets.
- `py-style`: load before editing Python source files.
- `cjk-safety`: load before editing Python files that contain CJK string
  literals or prompt-facing CJK text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running real LLM comparison cases or writing
  human-readable output-quality reports.

## Mandatory Rules

- Execute this plan only after status is changed to `approved` or
  `in_progress` by explicit user instruction.
- Do not modify production code while this plan is `draft`.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or
  final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in execution
  evidence.
- RAG retrieves evidence. Cognition decides whether the observed group scene
  gives the character a reason to speak. Dialog renders the selected surface.
- Do not add `CASE_GROUP_CHAT_REVIEW` to `RAG_BACKED_CASE_NAMES`.
- Do not call the full RAG supervisor for group-review participant context.
- Do not add a feature flag, environment variable, compatibility mode, or
  alternate production path for this change.
- Do not add a display-name identity lookup fallback in the first production
  implementation. Use `global_user_id` from source rows when present; otherwise
  omit profile and conversation hydration for that participant.
- Do not fabricate `target_scope.user_id` for group review. The group scene
  remains the semantic target; same-channel delivery target remains separate.
- Do not expose raw storage ids, embeddings, raw UTC timestamps, raw adapter
  syntax, `delivery_target`, action-attempt ids, or helper worker payloads in
  prompt-facing participant context.
- Do not provide a prompt-facing roster of participant profiles. Only the
  primary social beat receives relationship, engagement, and conversation
  evidence.
- Deterministic code owns candidate scoring, caps, prompt payload bounds,
  timestamp windows, source-row projection, and helper-result validation.
- LLM helper agents own only their existing specialist evidence tasks. They
  must not receive delivery, scheduler, adapter capability, or persistence
  decisions.
- Real LLM tests must run one case at a time with output inspected before the
  next case.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual edits.
- Check `git status --short --untracked-files=all` before editing.
- Do not read `.env`.

## Must Do

- Add production group-review participant-context hydration before
  self-cognition cognition starts.
- Keep group review outside the full RAG supervisor path.
- Add a bounded internal participant-row projection to reflection activity
  windows so source collection can see participant ids and direct-address
  metadata without changing rendered visible dialogue.
- Add a self-cognition-owned participant-context module that selects one
  primary social beat from bounded visible group rows.
- Hydrate only the primary social beat with user profile, relationship label,
  user engagement guidance, and at most one bounded prior-conversation
  evidence lookup.
- Attach the prompt-facing context under
  `case["conversation_progress"]["participant_context"]`.
- Use aggregate background-flow context instead of background participant
  profiles.
- Preserve `target_scope.user_id=None` for ambient and directly addressed
  group-review cases.
- Preserve source-aligned group delivery target behavior.
- Add deterministic tests for candidate selection, prompt-facing context
  projection, caps, missing identity degradation, no full RAG supervisor
  invocation, and group target preservation.
- Add one-at-a-time real LLM comparison over the prior ten speaking-slot cases
  and write a human-readable report with actual outputs.
- Remove the experiment-only mini-RAG POC script and temporary production
  comparison runner after production real LLM comparison evidence is recorded
  and before independent code review starts.
- Update self-cognition documentation to describe the participant-context
  source hydration and explicitly say it is not the full RAG supervisor path.
- Update this plan and the development-plans registry lifecycle only after
  verification and review evidence exists.

## Deferred

- Do not redesign group-review cadence, reflection scope selection, activity
  window labels, sleep-period handling, delivery binding, idempotency, or
  action-attempt storage.
- Do not change the normal `/chat` RAG path.
- Do not change RAG2 initializer, dispatcher, evaluator, finalizer, Cache2, or
  `project_known_facts(...)`.
- Do not change L2d, L3, dialog, or evaluator prompts unless deterministic and
  real LLM evidence proves the participant context cannot be consumed through
  the existing source packet.
- Do not add a roster-shaped participant context to production.
- Do not hydrate background participant profiles, background relationship
  labels, or background conversation evidence.
- Do not add display-name identity lookup fallback.
- Do not add web lookup.
- Do not add retries beyond existing helper-agent `max_attempts=1` use for the
  single conversation-evidence lookup.
- Do not migrate or rewrite historical conversation, reflection,
  self-cognition action-attempt, or user-profile data.
- Do not add database collections, indexes, or durable participant-context
  storage.
- Do not add response-ratio tuning, cooldown changes, deterministic speech
  suppression, or keyword filters.
- Do not delete ignored raw comparison evidence under `test_artifacts` during
  the POC cleanup step unless the owner explicitly requests local evidence
  cleanup.

## Cutover Policy

Overall strategy: bigbang for the production group-review participant-context
hydration; compatible for existing storage and runner contracts.

| Area | Policy | Instruction |
|---|---|---|
| Group-review participant context | bigbang | Attach the new flow-focused context directly for eligible group-review cases. Do not preserve a roster mode. |
| Full RAG supervisor | compatible | Preserve the existing behavior where group review does not call the full RAG supervisor. |
| Conversation progress shape | compatible | Add `participant_context` as a group-review-only nested prompt field. Do not change stored conversation-progress schemas. |
| Reflection activity windows | compatible | Add an internal participant-row projection without changing rendered visible-context text. |
| Semantic target user | compatible | Preserve `target_scope.user_id=None` for group review. |
| Delivery target | compatible | Preserve same-channel group delivery target binding. |
| Historical data | compatible | No migration, backfill, or durable schema change. |
| Tests | bigbang | Replace experiment-only assumptions with production deterministic and real LLM verification. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- Bigbang areas must rewrite the behavior directly, not preserve legacy or
  experiment variants behind flags.
- Compatible areas preserve only the compatibility surfaces explicitly listed
  in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

For every collected `group_chat_review` case with non-empty visible context,
source collection attaches a bounded prompt-facing participant context:

```python
case["conversation_progress"]["participant_context"] = {
    "source": "group_review_participant_context",
    "focus_mode": (
        "direct_reply"
        | "group_pile_on"
        | "continue_visible_thread"
        | "ambient_observation"
    ),
    "guidance": str,
    "primary_reply_target": {
        "display_name": str,
        "reply_target_fit": "high" | "medium" | "low",
        "role_in_window": list[str],
        "relationship_label": str,
        "relationship_band": "positive" | "neutral" | "negative" | "unknown",
        "last_relationship_insight": str,
        "engagement_guidelines": list[str],
        "nearby_conversation_evidence": list[str],
        "visible_samples": list[str],
    },
    "background_flow": {
        "mode": "none" | "side_thread" | "multi_person_pile_on" | "ambient_group",
        "summary": str,
        "participant_count_label": "single" | "few" | "many",
    },
}
```

When no usable primary participant exists, `primary_reply_target` is `{}`,
`focus_mode` is `ambient_observation`, and `background_flow` still describes
the bounded visible group scene.

The prompt-facing shape deliberately contains no `participants` list. The
visible dialogue rows still provide exact local wording. Participant context
adds only the relationship/continuity evidence needed to judge one social
beat.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Integration point | Hydrate during `collect_group_review_cases(...)` before runner invocation. | Group review source collection owns the activity window and can add source evidence before cognition. |
| Retrieval scope | Primary participant only. | The failure mode was roster fan-out; primary-only hydration gives relationship evidence without encouraging replies to every participant. |
| Background participants | Aggregate background-flow summary only. | The visible transcript already shows names and text; profiles for background people are not needed for the current semantic question. |
| Identity source | Use `global_user_id` already present in source messages. | Reflection input rows carry identity metadata; display-name lookup adds latency and ambiguity. |
| Missing identity | Degrade to visible-only participant context. | A missing id should not block group review or trigger an unreliable lookup. |
| Conversation lookup | At most one `ConversationEvidenceAgent.run(..., max_attempts=1)` call for the primary participant. | This matches the POC quality improvement while keeping latency bounded. |
| User profile | Directly read `get_user_profile(...)` for the primary participant when `global_user_id` exists. | Profile read is deterministic storage access, not full RAG supervisor planning. |
| Engagement guidance | Use `build_user_engagement_relevance_context(...)` for the primary participant only. | Existing style-image projection already provides bounded relationship-facing guidance. |
| Prompt surface | Attach under `conversation_progress.participant_context`. | The self-cognition source packet already renders conversation progress into cognition. |
| Runtime label | Use `group_review_participant_context`, not `mini_rag`. | Runtime prompts should not contain experiment or implementation-history language. |

## Contracts And Data Shapes

### Internal Window Rows

Extend `GroupActivityWindow` with an internal, non-rendered participant-row
field:

```python
participant_rows: list[dict[str, Any]]
```

Rows are bounded to the same source-window message set and include only fields
needed for candidate selection and helper lookup:

```python
{
    "timestamp": str,
    "role": str,
    "display_name": str,
    "body_text": str,
    "platform_message_id": str,
    "global_user_id": str,
    "platform_user_id": str,
    "addressed_to_global_user_ids": list[str],
    "mentions": list[dict[str, str]],
    "is_directed_at_character": bool,
    "reply_context": dict[str, Any],
}
```

`participant_rows` must not be rendered by
`projection.render_source_packet_text(...)`.

### New Self-Cognition Module

Create:

```text
src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py
```

Public entrypoint:

```python
async def build_group_review_participant_context(
    *,
    participant_rows: list[dict[str, Any]],
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    window_start_utc: str,
    current_timestamp_utc: str,
    conversation_agent: ConversationEvidenceAgent | None = None,
) -> dict[str, Any] | None:
    ...
```

The function returns `None` only when there are no user-authored participant
rows after filtering. Otherwise it returns the target-state payload above.

### Candidate Rules

Candidate selection is deterministic:

- Ignore non-user rows.
- Group rows by first non-empty `global_user_id`, then `platform_user_id`, then
  `display_name`.
- Add role `direct_cue` when addressed ids, mention metadata, explicit
  directed flag, or visible text points to the active character.
- Add role `reply_to_character` when native reply metadata targets the active
  bot account.
- Add role `latest_speaker` for the latest user row in the window.
- Add role `topic_author` when a participant has more than one visible row.
- Add role `dominant_speaker` when a participant has at least three visible
  rows.
- Compute `reply_target_fit` as `high` for direct cue or reply-to-character,
  `medium` for latest topic/dominant speaker, and `low` otherwise.
- Select primary from high-fit participants by latest timestamp first, then
  candidate score. If no high-fit participant exists, apply the same rule to
  medium-fit participants, then low-fit participants.
- Set `focus_mode="group_pile_on"` when two or more high-fit participants are
  present; otherwise derive `direct_reply`, `continue_visible_thread`, or
  `ambient_observation` from the selected primary fit.

### Prompt-Facing Guidance

Use concise semantic guidance:

- `group_pile_on`: address the shared pile-on or accusation as one beat; do
  not answer each participant one by one.
- `direct_reply`: reply to the primary target or current thread only; do not
  fan out.
- `continue_visible_thread`: follow the current discussion flow without
  widening the reply.
- `ambient_observation`: use participant context only as background; speak
  only if the visible group flow itself gives enough reason.

### Trace And Artifacts

Production event logs must not store raw conversation bodies or helper worker
payloads. Dry-run artifacts may include the case payload and cognition input
as they do today. Helper worker payloads should not be embedded in
`conversation_progress.participant_context`.

## LLM Call And Context Budget

Before this plan for `group_chat_review`:

| Stage | Calls | Blocking | Context |
|---|---:|---|---|
| Full RAG supervisor | 0 | none | not invoked |
| Cognition graph | 1 | response path | source packet, visible context, empty RAG result, conversation progress |
| L3/dialog | 0 or 1 | response path only when L2d selects speak | selected text directives and dialog state |

After this plan for `group_chat_review`:

| Stage | Calls | Blocking | Context |
|---|---:|---|---|
| Full RAG supervisor | 0 | none | not invoked |
| Primary conversation evidence helper | 0 or 1 | source hydration before cognition | primary participant id, same group channel, 72-hour closed lookback, capped evidence |
| User profile and engagement projection | deterministic DB calls | source hydration before cognition | primary `global_user_id` only |
| Cognition graph | 1 | response path | source packet plus compact `participant_context` |
| L3/dialog | 0 or 1 | response path only when L2d selects speak | unchanged |

Hard caps:

- Participant row input: current 15-minute group activity window only.
- Primary hydrated participant count: `1`.
- Conversation-evidence helper calls per group-review case: `1`.
- Conversation helper attempts: `max_attempts=1`.
- Conversation lookback: `72` hours ending immediately before the activity
  window start.
- Conversation evidence lines: `3`.
- Visible samples per primary: `3`, each capped to `160` characters.
- Prompt-facing participant context: must be capped before source-packet
  rendering so the total source packet remains under
  `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`.

No new response-path retry loop is allowed.

## Change Surface

### Delete

- `experiments/group_chat_review_mini_rag_ablation.py`
  - Remove after the production implementation and real LLM evidence supersede
    the experiment-only POC path.
  - Do not remove ignored raw evidence under `test_artifacts` as part of this
    cleanup step.
- `experiments/group_chat_review_participant_context_production_comparison.py`
  - Remove after Stage 5 writes production comparison evidence.
  - This file is temporary evidence tooling and must not be part of the final
    implementation diff.

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`
  - Add internal `participant_rows` projection to `GroupActivityWindow`.
  - Keep existing `visible_context` rendering unchanged.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Call `build_group_review_participant_context(...)` for group-review cases
    and attach the result under `conversation_progress.participant_context`.
  - Allowed test seam: add a keyword-only
    `participant_context_builder` parameter to
    `collect_group_review_cases(...)` only, defaulting to the production
    builder. Do not add this seam to worker-facing
    `collect_group_chat_review_cases(...)`.
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document that group review uses bounded primary-participant context without
    invoking the full RAG supervisor.
- `tests/test_self_cognition_group_review_source.py`
  - Add source-collection tests for participant context attachment, target
    preservation, visible-context rendering, and helper seam behavior.
- `tests/test_self_cognition_tracking.py`
  - Add runner/projection tests proving group review still omits
    `self_cognition_rag_request.json` and `self_cognition_rag_output.json`.

### Create

- `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py`
  - Own deterministic candidate selection, primary participant hydration,
    prompt-facing projection, and helper-result caps for group review.
- `tests/test_self_cognition_group_review_participant_context.py`
  - Focused deterministic tests for candidate roles, primary selection,
    background-flow projection, caps, missing identity degradation, and
    helper-call count.
- `experiments/group_chat_review_participant_context_production_comparison.py`
  - Temporary Stage 5 evidence runner only.
  - It must load the exact source-id list in `Real LLM Evaluation`, run the
    implemented production participant-context path for those cases, compare
    against the stored current baseline from the prior POC evidence, and write
    raw JSON plus a human-readable report.
  - It must be deleted in Stage 6 before independent code review.

### Keep

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - No production edits are allowed in this plan. If implementation evidence
    shows `runner.py` must change, stop and amend this plan before editing.
  - `RAG_BACKED_CASE_NAMES` must remain unchanged. Group review must remain
    outside the full RAG supervisor path.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - No production edits are allowed in this plan. If a focused test proves the
    existing compact JSON rendering drops `participant_context`, stop and amend
    this plan before editing.
- `src/kazusa_ai_chatbot/nodes/*`
  - Do not change cognition, L2d, L3, dialog, or evaluator prompts in this
    plan.
- `src/kazusa_ai_chatbot/rag/*`
  - Do not change RAG2 supervisor, initializer, dispatcher, helper agents,
    Cache2, or projection contracts.
- `src/kazusa_ai_chatbot/db/*`
  - Do not add collections, indexes, or schema migrations.

## Overdesign Guardrail

- Actual problem: active group-review self-cognition lacks participant
  relationship and continuity evidence, while roster-shaped evidence encourages
  replies that try to address everyone.
- Minimal change: add one bounded source-hydration module that selects one
  primary social beat and attaches primary-only participant evidence to the
  existing `conversation_progress` prompt surface.
- Ownership boundaries: deterministic code selects candidates, caps payloads,
  validates helper outputs, and preserves target/delivery boundaries; RAG
  helpers retrieve bounded evidence; cognition judges whether to speak and
  what stance to take; dialog renders wording.
- Rejected complexity: full RAG supervisor invocation, roster prompt,
  background participant profiles, display-name lookup fallback, feature flags,
  new schemas, new storage, web lookup, extra retries, prompt rewrites,
  response-ratio tuning, deterministic suppression, adapter changes, and
  compatibility modes.
- Evidence threshold: add a rejected complexity only after a follow-up
  diagnostic shows a repeated production failure that cannot be fixed by
  primary-only participant context, existing visible context, or current L2/L3
  contracts, and after the user approves a new plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, feature flags,
  or extra features.
- The responsible agent must search the codebase before implementing helpers.
  If equivalent behavior exists, move or reuse it in the narrowest appropriate
  module instead of duplicating it.
- Changes outside the listed `Change Surface` require a written justification
  before editing and must remain directly tied to the approved contract.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, broad refactors, or test fixture
  rewrites outside this plan.
- If implementation evidence shows the existing prompt cannot use the new
  context correctly, stop and request plan approval for a prompt-change
  amendment before editing prompts.
- If native subagents are unavailable during execution, stop before production
  implementation unless the user explicitly requests fallback execution.

## Implementation Order

1. Parent establishes focused tests for the new participant-context module.
   - File: `tests/test_self_cognition_group_review_participant_context.py`.
   - Expected before implementation: import failure for the new module.
2. Parent establishes source-integration tests.
   - File: `tests/test_self_cognition_group_review_source.py`.
   - Expected before implementation: missing participant context in collected
     group-review cases.
3. Parent verifies group review remains outside full RAG supervisor.
   - File: `tests/test_self_cognition_tracking.py`.
   - Expected baseline: existing group-review runner tests pass and no RAG
     artifacts are emitted.
4. Parent starts exactly one production-code subagent with this approved plan,
   mandatory skills, focused failing tests, and the production change surface.
5. Production-code subagent implements internal participant rows in
   `reflection_cycle/activity_windows.py`.
6. Production-code subagent implements
   `self_cognition/group_review_participant_context.py`.
7. Production-code subagent wires `sources.collect_group_review_cases(...)` to
   attach prompt-facing context.
8. Parent runs focused module tests and integration tests.
9. Parent runs regression tests listed in `Verification`.
10. Parent creates the temporary Stage 5 production comparison runner under
    `experiments/`.
11. Parent runs one-at-a-time real LLM comparison over the previous ten
    speaking-slot cases and writes a report with actual outputs.
12. Parent removes the experiment-only mini-RAG POC script and the temporary
    production comparison runner, then verifies no production or test code
    references the experiment-only paths.
13. Parent starts exactly one independent code-review subagent after planned
    verification passes.
14. Parent fixes in-scope review findings, reruns affected verification, and
    records evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static
  checks, validation work, and evidence while the production-code subagent
  edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification and POC cleanup pass; reviews the plan, diff, and
  evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused participant-context tests established
  - Covers: implementation order step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py -q`
  - Evidence: record expected import failure or focused failure before
    production implementation.
  - Handoff: next stage establishes source integration and RAG-boundary
    baseline tests.
  - Sign-off: `Codex/2026-05-29` after expected missing-module failure was
    recorded in `Execution Evidence`.
- [x] Stage 2 - source integration and RAG-boundary baseline established
  - Covers: implementation order steps 2 and 3.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
  - Evidence: record expected missing-context failure or current source
    baseline, plus the current group-review no-full-RAG baseline.
  - Handoff: next stage starts production-code subagent.
  - Sign-off: `Codex/2026-05-29` after expected source failures and
    no-full-RAG tracking baseline were recorded in `Execution Evidence`.
- [x] Stage 3 - production module and source wiring complete
  - Covers: implementation order steps 4 through 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py tests\test_self_cognition_group_review_source.py -q`
  - Evidence: record changed files, helper-call caps, and test output.
  - Handoff: next stage starts regression verification.
  - Sign-off: `Codex/2026-05-29` after focused tests, compile, changed files,
    and helper-call cap evidence were recorded.
- [x] Stage 4 - group-review RAG boundary regression complete
  - Covers: implementation order steps 8 and 9.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
  - Evidence: record that group review emits no full RAG request/output
    artifacts and target scope remains group-scoped.
  - Handoff: next stage starts real LLM comparison.
  - Sign-off: `Codex/2026-05-29` after tracking regression evidence was
    recorded.
- [x] Stage 5 - real LLM comparison complete
  - Covers: implementation order steps 10 and 11.
  - Verify: run the exact command in `Real LLM Evaluation`.
  - Evidence: record the report path, raw summary path, source ids, aggregate
    speak counts, case-level quality notes, and any regressions.
  - Handoff: next stage removes the experiment-only POC path.
  - Sign-off: `Codex/2026-05-29` after raw summary, human-authored report,
    validation output, aggregate speak counts, and case-level quality notes
    were recorded.
- [x] Stage 6 - experiment POC cleanup complete
  - Covers: implementation order step 12.
  - Verify:
    `foreach ($path in @('experiments\group_chat_review_mini_rag_ablation.py', 'experiments\group_chat_review_participant_context_production_comparison.py')) { if (Test-Path -LiteralPath $path) { throw "$path still present" } }`
  - Evidence: record removed path, `git status --short --untracked-files=all`,
    and the POC cleanup verification output.
  - Handoff: next stage starts independent code review.
  - Sign-off: `Codex/2026-05-29` after cleanup evidence is recorded.
- [x] Stage 7 - independent code review complete
  - Covers: implementation order steps 13 and 14.
  - Verify: independent review subagent inspects the approved plan, full diff,
    focused tests, regression output, real LLM report, and execution evidence.
  - Evidence: record findings, fixes, rerun commands, residual risks, and
    review approval status.
  - Handoff: plan can move to completion only after review approval.
  - Sign-off: `Codex/2026-05-29` after review evidence is recorded.

## Execution Evidence

- Stage 1 focused participant-context tests:
  - Command:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py -q`
  - Result: exited with code `1` during collection, expected before production
    implementation.
  - Evidence: `ImportError: cannot import name 'group_review_participant_context' from 'kazusa_ai_chatbot.self_cognition'`.
- Stage 2 source-integration baseline:
  - Command:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
  - Result: exited with code `1`, expected before production
    implementation; `8 passed, 2 failed`.
  - Evidence: missing `GroupActivityWindow.participant_rows` and
    `collect_group_review_cases(...)` does not yet accept
    `participant_context_builder`.
- Stage 2 group-review no-full-RAG baseline:
  - Command:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
  - Result: exited with code `0`; `43 passed`.
  - Evidence: `test_group_chat_review_does_not_invoke_full_rag_supervisor`
    passed with zero full-RAG calls, no RAG request artifact, and no RAG
    output artifact.
- Stage 3 production module and source wiring:
  - Production-code subagent: `Ampere`
    (`019e7300-48a6-72a1-9802-4d3867daa5e2`).
  - Changed production files:
    `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`,
    `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py`,
    `src/kazusa_ai_chatbot/self_cognition/sources.py`, and
    `src/kazusa_ai_chatbot/self_cognition/README.md`.
  - Focused command:
    `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py tests\test_self_cognition_group_review_source.py -q`
  - Result: exited with code `0`; `15 passed`.
  - Compile command:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\activity_windows.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py`
  - Compile result: exited with code `0`.
  - Evidence: participant rows are internal to group activity windows,
    `conversation_progress.participant_context` is attached during
    group-review source collection, prompt-facing context uses
    `context_shape="single_flow_focus"`, no prompt-facing `participants` or
    `selected_reply_target` field is emitted, and the builder hydrates only
    one primary participant with one conversation-evidence helper call using
    `max_attempts=1`.
- Stage 4 group-review RAG boundary regression:
  - Command:
    `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
  - Result: exited with code `0`; `43 passed`.
  - Evidence: `test_group_chat_review_does_not_invoke_full_rag_supervisor`
    passed after production source hydration; group review still records zero
    full-RAG calls and emits no `self_cognition_rag_request.json` or
    `self_cognition_rag_output.json` artifact.
- Stage 5 real LLM comparison:
  - Command: exact Stage 5 command from `Real LLM Evaluation`, with output
    directory
    `test_artifacts\experiments\group_chat_review_participant_context\20260529T092925Z`.
  - Result: exited with code `0`.
  - Raw summary:
    `test_artifacts/experiments/group_chat_review_participant_context/20260529T092925Z/summary.json`.
  - Human-authored review:
    `test_artifacts/experiments/group_chat_review_participant_context/20260529T092925Z/production_participant_context_review_10_cases.md`.
  - Validation command:
    `venv\Scripts\python -c "... stage5_validation_ok ..."`
  - Validation result: exited with code `0`;
    `case_count=10`, source ids matched the plan order,
    `production_writes="none"`, all cases had
    `conversation_progress.participant_context.source="group_review_participant_context"`,
    all contexts had `context_shape="single_flow_focus"`, no context had
    prompt-facing `participants` or `selected_reply_target`, no production case
    emitted a full-RAG request/output artifact, and production spoke in
    `8/10` cases.
  - Aggregate speak counts: stored current baseline `6/10`, previous mini-RAG
    POC `10/10`, production participant context `8/10`.
  - Case quality notes: cases 3 through 10 passed; cases 1 and 2 were
    warnings because production stayed silent. Case 1 silence was defensible
    after the visible flow moved to technical file-sending, but it exposed a
    residual candidate-selection risk where an earlier row mentioning the
    character name can overweight the current direct accuser. Case 2 silence
    was defensible because the scene drifted to side-thread content, but it
    may be a missed opportunity to challenge topic-change evasiveness.
  - Regression check: no case was judged `fail`; case 2 did not show roster
    fan-out and did not address multiple visible participants as separate
    reply targets.
- Stage 6 experiment POC cleanup:
  - Removed path:
    `experiments/group_chat_review_mini_rag_ablation.py`.
  - Removed path:
    `experiments/group_chat_review_participant_context_production_comparison.py`.
  - Absence check:
    `foreach ($path in @('experiments\group_chat_review_mini_rag_ablation.py', 'experiments\group_chat_review_participant_context_production_comparison.py')) { if (Test-Path -LiteralPath $path) { throw "$path still present" } }`
  - Absence result: exited with code `0`; output
    `cleanup_files_absent_ok`.
  - Path-name grep:
    `rg -n "group_chat_review_mini_rag_ablation|group_chat_review_participant_context_production_comparison" src tests development_plans\README.md`
  - Path-name grep result: no matches; output `experiment_path_grep_ok`.
  - Old-contract grep:
    `rg -n "selected_reply_target|mini_rag" src\kazusa_ai_chatbot\self_cognition`
  - Old-contract grep result: no matches; output
    `old_poc_contract_grep_ok`.
  - `git status --short --untracked-files=all` after cleanup showed only the
    active plan/registry, production source/doc changes, and deterministic
    test changes; no `experiments/*` runner files remained visible.
- Stage 7 independent code review:
  - Review subagent: `Einstein`
    (`019e731f-5f05-7923-aa6d-8f1991da3417`).
  - Review scope: active plan, full implementation diff, deterministic tests,
    static checks, Stage 5 real LLM evidence, Stage 6 POC cleanup evidence,
    and development-plan registry state.
  - Review result: production code path approved. The reviewer found no
    hidden full-RAG invocation, prompt-facing roster leak, `selected_reply_target`
    or `mini_rag` production leakage, target-user fabrication,
    delivery/action metadata leakage, helper-call cap violation, or blocking
    code defect.
  - Finding addressed: stale lifecycle text in `Independent Plan Review`
    still described this document as a draft. The text was updated to reflect
    that plan-review issues were addressed before execution and this plan is
    now `in_progress`.
  - Finding classified: `development_plans/README.md` and the worktree also
    contain `resolver_stage_trace_poc_plan.md`. That resolver plan and its
    registry row are unrelated worktree scope for this implementation and are
    excluded from this plan's completion evidence; they were not removed or
    reverted.
  - Minor evidence note: Stage 5 `summary.json` has confusing top-level
    `production_active_text` values for cases 1 and 2, while nested
    `production.selected_text`, route artifacts, and the human-authored
    report agree that production spoke in `8/10` cases. This does not
    invalidate the Stage 5 gate.
  - Residual risk: case 1's documented candidate-selection risk remains as
    follow-up tuning because earlier character-name text cues can overweight
    the wrong primary participant. It did not produce a bad visible reply in
    Stage 5 and is not a blocker for this plan.
  - Lifecycle closeout: plan status set to `completed` and moved to
    `development_plans/archive/completed/short_term/` after Stage 7 evidence
    was recorded.

## Verification

### Static Checks

- `rg -n "CASE_GROUP_CHAT_REVIEW" src\kazusa_ai_chatbot\self_cognition\runner.py`
  - Expected: matches constants or case handling only; `CASE_GROUP_CHAT_REVIEW`
    must not appear inside `RAG_BACKED_CASE_NAMES`.
- `rg -n "participants|selected_reply_target|mini_rag" src\kazusa_ai_chatbot\self_cognition`
  - Expected: no prompt-facing roster field named `participants`, no
    experiment label `mini_rag`, and no old POC-only `selected_reply_target`
    contract in production self-cognition code.
- `rg -n "UserLookupAgent|PersonContextAgent|Web" src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py`
  - Expected: no matches for `UserLookupAgent`, `PersonContextAgent`, or web
    lookup usage.
- `rg -n "delivery_target|action_attempt|platform_channel_id" src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py`
  - Expected: no prompt-facing projection of delivery or action-attempt
    metadata. `platform_channel_id` may appear only as helper context for
    conversation evidence.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py -q`

### Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\activity_windows.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py`

### Real LLM Evaluation

- Source artifact set:
  `test_artifacts/experiments/group_chat_review_mini_rag_ablation/20260529T075429Z/`.
- Baseline summary:
  `test_artifacts/experiments/group_chat_review_mini_rag_ablation/20260529T075429Z/summary.json`.
- Diagnostic audit source:
  `test_artifacts/diagnostics/group_chat_review_mini_rag_speaking_10_20260529/`.
- Case source ids, in required execution order:

| Case | Source id |
|---|---|
| 1 | `scope_e3945048f57d:2026-05-20T00:45:00+00:00:2026-05-20T01:00:00+00:00` |
| 2 | `scope_e3945048f57d:2026-05-26T05:15:00+00:00:2026-05-26T05:30:00+00:00` |
| 3 | `scope_e3945048f57d:2026-05-21T03:30:00+00:00:2026-05-21T03:45:00+00:00` |
| 4 | `scope_e3945048f57d:2026-05-26T04:45:00+00:00:2026-05-26T05:00:00+00:00` |
| 5 | `scope_e3945048f57d:2026-05-20T03:00:00+00:00:2026-05-20T03:15:00+00:00` |
| 6 | `scope_e3945048f57d:2026-05-20T03:15:00+00:00:2026-05-20T03:30:00+00:00` |
| 7 | `scope_e3945048f57d:2026-05-25T06:45:00+00:00:2026-05-25T07:00:00+00:00` |
| 8 | `scope_e3945048f57d:2026-05-20T01:45:00+00:00:2026-05-20T02:00:00+00:00` |
| 9 | `scope_e3945048f57d:2026-05-20T01:30:00+00:00:2026-05-20T01:45:00+00:00` |
| 10 | `scope_60e91207fbef:2026-05-26T02:30:00+00:00:2026-05-26T02:45:00+00:00` |

- Temporary runner contract:
  `experiments/group_chat_review_participant_context_production_comparison.py`
  must serialize live LLM calls one case at a time, use the implemented
  production participant-context source path, compare against the stored
  current baseline outputs in the baseline summary, and write no production DB
  mutations.
- Exact command:

```powershell
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$out = "test_artifacts\experiments\group_chat_review_participant_context\$stamp"
venv\Scripts\python experiments\group_chat_review_participant_context_production_comparison.py `
  --audit-dir test_artifacts\diagnostics\group_chat_review_mini_rag_speaking_10_20260529 `
  --baseline-summary test_artifacts\experiments\group_chat_review_mini_rag_ablation\20260529T075429Z\summary.json `
  --output-dir $out `
  --source-id "scope_e3945048f57d:2026-05-20T00:45:00+00:00:2026-05-20T01:00:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-26T05:15:00+00:00:2026-05-26T05:30:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-21T03:30:00+00:00:2026-05-21T03:45:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-26T04:45:00+00:00:2026-05-26T05:00:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-20T03:00:00+00:00:2026-05-20T03:15:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-20T03:15:00+00:00:2026-05-20T03:30:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-25T06:45:00+00:00:2026-05-25T07:00:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-20T01:45:00+00:00:2026-05-20T02:00:00+00:00" `
  --source-id "scope_e3945048f57d:2026-05-20T01:30:00+00:00:2026-05-20T01:45:00+00:00" `
  --source-id "scope_60e91207fbef:2026-05-26T02:30:00+00:00:2026-05-26T02:45:00+00:00"
```

- Required evidence files under `$out`:
  `summary.json`, `production_participant_context_review_10_cases.md`, and
  one raw artifact directory per case.
- Stage 5 passes only when:
  - `summary.json` records exactly the ten source ids above in the same order.
  - `summary.json.case_count` is `10`.
  - `summary.json.run_context.production_writes` is `none`.
  - Every production case includes
    `conversation_progress.participant_context.source` equal to
    `group_review_participant_context`.
  - Every production participant context uses
    `context_shape="single_flow_focus"` and has no prompt-facing
    `participants` or `selected_reply_target` field.
  - No production case directory contains `self_cognition_rag_request.json` or
    `self_cognition_rag_output.json`.
  - At least eight of ten production cases produce selected visible text; any
    non-speaking case includes L2d action-spec evidence and a human judgment
    that silence is appropriate for the scene.
  - The report includes stored current baseline text, production participant
    context output, visible context, `focus_mode`, primary focus, output text
    or L2d result, and `pass`, `warn`, or `fail` quality judgment per case.
  - No case is judged `fail`; warnings are recorded as residual quality risks.
  - Case 2 no longer shows roster fan-out: production output must not address
    multiple visible participants as separate reply targets.

### POC Cleanup

- `foreach ($path in @('experiments\group_chat_review_mini_rag_ablation.py', 'experiments\group_chat_review_participant_context_production_comparison.py')) { if (Test-Path -LiteralPath $path) { throw "$path still present" } }`
  - Expected: command exits without throwing.
- `rg -n "group_chat_review_mini_rag_ablation|group_chat_review_participant_context_production_comparison" src tests development_plans\README.md`
  - Expected: no matches for experiment-only path names in source, tests, or
    the development-plans registry.
- `rg -n "selected_reply_target|mini_rag" src\kazusa_ai_chatbot\self_cognition`
  - Expected: no matches for the old POC-only selected target contract or
    `mini_rag` labels in production self-cognition code.

## Independent Plan Review

Review performed on 2026-05-29 in two passes:

- Drafting-agent self-review after rereading the plan contract, execution
  gates, local LLM architecture constraints, and this full plan.
- Independent explorer subagent review of this plan, the registry row, project
  rules, development-plan rules, execution gates, and local LLM architecture
  constraints.

Review scope:

- Lifecycle order from draft plan through real LLM comparison, POC cleanup,
  independent code review, and final sign-off.
- Alignment with source ownership boundaries: source hydration, bounded helper
  evidence retrieval, cognition judgment, dialog rendering, and persistence.
- Overdesign pressure: no roster prompt, no full RAG supervisor, no feature
  flag, no prompt rewrite, no durable schema, and no background profile
  hydration.
- Verification quality for deterministic tests, static checks, real LLM
  evidence, cleanup evidence, and review evidence.

Findings addressed:

- Blocking finding: the previous checklist handed directly from real LLM
  comparison to independent code review and did not remove the experiment-only
  mini-RAG POC path first. Addressed by adding a `Delete` change surface item,
  implementation order step 12, Stage 6 cleanup gate, cleanup verification,
  and an acceptance criterion.
- Medium finding: the cleanup instruction could have removed raw comparison
  evidence needed for inspection. Addressed by preserving ignored
  `test_artifacts` evidence unless the owner explicitly requests cleanup.
- Medium finding: the plan lacked a recorded independent plan-review section
  even though review was requested before execution. Addressed by adding this
  section with scope, findings, and resolution status.
- Medium finding: the first cleanup grep draft was too broad because it would
  have rejected future regression tests that mention an old experiment-only
  field only to assert its absence. Addressed by limiting the old-contract grep
  to production self-cognition code and using a separate path-name grep across
  source, tests, and registry files.
- Blocking independent-review finding: the real LLM gate was not executable
  enough because it lacked source artifact paths, source ids, command, runner
  contract, and pass criteria. Addressed by binding Stage 5 to the
  `20260529T075429Z` artifact set, listing all ten source ids, adding the
  exact PowerShell command, specifying the temporary runner contract, and
  defining hard pass criteria.
- Medium independent-review finding: the progress checklist mixed step 3 into
  Stage 1 even though Stage 1 only ran the new module test, and its handoff
  skipped Stage 2. Addressed by limiting Stage 1 to step 1 and moving the
  RAG-boundary baseline into Stage 2 with the source integration baseline.
- Medium independent-review finding: the change surface left conditional
  production-code freedom in `sources.py`, `runner.py`, and `projection.py`.
  Addressed by defining the only allowed source test seam and replacing
  conditional `Keep` wording with stop-and-amend rules.

Follow-up independent review result:

- No blockers or medium issues remain from the prior review.
- The reviewer confirmed the real LLM gate, progress checklist, change
  surface, POC cleanup ordering, and lifecycle gating are correctly specified.
- Residual risks: Stage 5 depends on the named `test_artifacts` inputs
  remaining available; live LLM output quality still requires human judgment
  under the listed fail conditions; the temporary comparison runner is deleted
  before code review, so review relies on recorded evidence rather than final
  tree source for that runner.

Plan-review status: all identified plan-contract issues were addressed before
execution. This plan moved to `in_progress` by explicit user instruction
before production implementation began.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final
sign-off. The parent agent must create one independent code-review subagent
through the current harness's native subagent capability. If native subagents
are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused tests, real LLM evidence,
  static checks, execution evidence, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
execution evidence.

## Acceptance Criteria

This plan is complete when:

- `group_chat_review` production cases attach bounded
  `conversation_progress.participant_context` before cognition.
- The prompt-facing participant context has no roster-shaped `participants`
  field and hydrates only one primary social beat.
- Background participants are represented only as aggregate background-flow
  context.
- Group review still emits no full RAG supervisor request/output artifacts.
- `target_scope.user_id` remains `None` for group-review cases.
- Same-channel group delivery target behavior remains unchanged.
- All deterministic tests, static checks, compile checks, and one-at-a-time
  real LLM comparison gates pass.
- The real LLM report shows the previous case 2 roster fan-out failure is
  corrected and records any remaining quality risks.
- The experiment-only mini-RAG POC script and temporary production comparison
  runner are removed before independent code review starts, while ignored raw
  comparison evidence remains available.
- `src/kazusa_ai_chatbot/self_cognition/README.md` documents the new
  participant-context path and its RAG boundary.
- Independent code review approves the implementation or all blocking findings
  are fixed and re-reviewed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Added helper latency slows reflection cadence | Hydrate primary only; one conversation-evidence call; no full RAG supervisor; no identity lookup fallback | LLM call budget review and real LLM comparison timing notes |
| Prompt still treats context as a roster | Production shape has no `participants` list and no background profiles | Static grep and case 2 real LLM report |
| Missing `global_user_id` loses profile evidence | Degrade to visible-only primary context instead of unreliable lookup | Missing-identity deterministic test |
| Source rows leak internal ids into prompt | Keep ids in internal `participant_rows`; project only semantic fields | Source-packet rendering test and static review |
| Candidate selection picks stale participant | Primary selection prefers latest high/medium-fit cue over raw score | Case 5 deterministic and live comparison |
| Group pile-on over-targets one participant | `group_pile_on` guidance tells cognition to answer shared beat once | Case 7 real LLM comparison |
| POC cleanup removes inspection evidence | Delete only experiment runner source files; preserve ignored `test_artifacts` evidence | Stage 6 cleanup evidence and git status |
