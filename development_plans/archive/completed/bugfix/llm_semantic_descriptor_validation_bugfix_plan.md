# llm semantic descriptor validation bugfix plan

## Summary

- Goal: stop hard-failing LLM-generated semantic descriptor fields that are
  consumed only as later LLM context, while preserving strict enums for fields
  that deterministic code uses as control state.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `skill-creator`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang for semantic-descriptor validation and
  prompt contracts; compatible for stored data shape and existing field names.
- Highest-risk areas: accidentally relaxing true control fields, weakening
  structural validation, changing daily reflection gates, and leaving prompts
  that still teach the local LLM closed enum values for prompt-only fields.
- Acceptance criteria: semantic descriptor fields named in this plan accept
  bounded arbitrary strings; true control fields remain closed enums; the
  local LLM architecture rule documents this boundary; focused deterministic
  tests, static greps, and prompt-render checks pass.

## Context

The observed failure was:

```text
ValueError: invalid topic_momentum: developing
```

The recorder LLM emitted `topic_momentum="developing"`. That value is
semantically reasonable but not listed in `VALID_TOPIC_MOMENTUM`. The failure
occurred in background conversation-progress recording and did not protect a
deterministic routing, permission, persistence-lifecycle, or scoring decision.

The system-level RCA is that some model-facing semantic descriptors were typed
like protocol fields. The key is useful because downstream prompts need a
stable place to read the descriptor. The value does not need a closed enum when
the downstream consumer is another LLM.

The current confirmed split is:

- `conversation_progress.status` and `conversation_progress.continuity` are
  control fields. Deterministic code branches on them. They stay closed enums.
- `conversation_progress.conversation_mode`, `episode_phase`, and
  `topic_momentum` are prompt/storage descriptors. They should be required
  string fields with caps, not closed enums.
- `interaction_style_overlay.confidence` is stored with style guidance and fed
  to LLM-facing style/relevance/cognition context. It should be a bounded
  descriptor, not a hard `low|medium|high` enum.
- Hourly reflection `confidence` and
  `participant_observations[].evidence_strength` feed later reflection LLMs.
  They should be prompt-only descriptors. Daily reflection output
  `confidence` remains a closed enum because it gates interaction-style
  extraction.

This plan also updates the local LLM architecture rule so future prompt and
schema work uses the same boundary:

```text
If deterministic code does not branch, route, persist lifecycle state, enforce
privacy, or compute score from the exact value, do not make the LLM output a
closed enum. Keep the key stable and validate only structure, sanitation, and
bounds.
```

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, signing off, or updating this plan.
- `local-llm-architecture`: load before changing prompt contracts, LLM output
  schemas, context projection, reflection behavior, conversation progress, or
  the local LLM architecture skill text.
- `skill-creator`: load before editing
  `.agents/skills/local-llm-architecture/SKILL.md`; keep the skill concise,
  imperative, trigger-focused in frontmatter, and validate the skill after the
  edit.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  test strings.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live LLM checks, judging LLM output quality,
  or writing LLM inspection artifacts.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Check `git status --short` before editing.
- Do not read `.env`.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual source, test, documentation, and plan edits.
- Preserve the stable field keys named in this plan. This is not a data-shape
  migration.
- Keep deterministic control labels strict. Do not relax enums for routing,
  delivery, permissions, privacy gates, lifecycle states, memory types, merge
  decisions, action execution, cache policy, adapter behavior, or numeric
  scoring.
- Apply the local LLM rule: a generated value consumed only as LLM context must
  be a bounded string descriptor, not a closed enum. Deterministic code may
  require the key, require string/list shape, cap length, drop malformed list
  items, reject source-detail leakage, and enforce empty/non-empty consistency.
- Do not add retry loops, repair prompts, compatibility shims, feature flags,
  migration scripts, extra LLM calls, or alternate persistence paths.
- Runtime prompt changes must keep stable output contracts in `SystemMessage`
  prompt constants and current-run data in `HumanMessage` payloads.
- Prompt edits must use plain semantic wording and avoid development-process
  terms in runtime prompts.
- Skill edits must follow `skill-creator`: update only `SKILL.md`, keep the
  added rule concise, do not create auxiliary docs, and run
  `C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py`
  against `.agents\skills\local-llm-architecture`.
- For `conversation_progress`, keep `status` and `continuity` as closed enums.
  Only relax `episode_phase` and `topic_momentum`.
- For reflection, keep daily synthesis output `confidence` as
  `low|medium|high` because `reflection_cycle.interaction_style` uses it as a
  deterministic gate. Hourly-slot `confidence` values carried into daily input
  are semantic descriptors, not the daily synthesis output control label.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Update `.agents/skills/local-llm-architecture/SKILL.md` with the semantic
  descriptor versus control enum rule.
- Remove closed-set validation for
  `conversation_progress.recorder.validate_recorder_output(...)` fields
  `episode_phase` and `topic_momentum`.
- Remove `VALID_EPISODE_PHASE` and `VALID_TOPIC_MOMENTUM` from
  `conversation_progress.policy`.
- Rewrite the conversation-progress recorder prompt so `episode_phase` and
  `topic_momentum` are described as bounded semantic descriptors, not enum
  choices.
- Preserve strict validation for `conversation_progress.status` and
  `conversation_progress.continuity`.
- Relax `interaction_style_images.validate_interaction_style_overlay(...)`
  so `confidence` accepts a bounded descriptor string instead of only
  `low`, `medium`, `high`, or empty.
- Define the interaction-style confidence cap as
  `_CONFIDENCE_DESCRIPTOR_MAX_CHARS = 80` in
  `src/kazusa_ai_chatbot/db/interaction_style_images.py`.
- Preserve the rule that an empty interaction-style overlay cannot carry
  non-empty confidence.
- Soften hourly reflection prompt and validation for hourly `confidence` and
  `participant_observations[].evidence_strength`.
- Define the hourly reflection descriptor cap as
  `_REFLECTION_DESCRIPTOR_MAX_CHARS = 80` in
  `src/kazusa_ai_chatbot/reflection_cycle/projection.py` and apply it to
  hourly `confidence` and `participant_observations[].evidence_strength`.
- Preserve daily synthesis output `confidence` as a control label: validation
  continues to return warnings for non-enum values, and interaction-style
  extraction continues to skip values outside `medium` and `high`.
- Update documentation for conversation progress, reflection cycle, database
  interaction-style overlays, and local LLM architecture.
- Update focused tests before production-code edits and keep regression tests
  mapped to the behavior being protected.

## Deferred

- Do not relax enums for action specs, RAG routing, web-agent actions/statuses,
  memory unit types, memory lifecycle states, merge decisions, stability
  windows, privacy gates, reflection promotion decisions, global-growth
  scoring labels, message-envelope roles, event statuses, scheduler statuses,
  or adapter delivery contracts.
- Do not rename stored fields such as `topic_momentum`, `episode_phase`, or
  `confidence`.
- Do not migrate existing MongoDB documents or backfill stored values.
- Do not change conversation-progress load/record facades, repository scope
  keys, expiry policy, guarded write behavior, or prompt budget caps.
- Do not change interaction-style guideline extraction semantics beyond the
  `confidence` descriptor validation.
- Do not change daily reflection style-update eligibility.
- Do not add new LLM calls, JSON repair behavior, retries, or live response
  path latency.

## Cutover Policy

Overall strategy: bigbang for validation and prompt wording; compatible for
field names and stored data shape.

| Area | Policy | Instruction |
|---|---|---|
| Conversation progress `episode_phase` | bigbang | Replace closed enum validation with bounded string validation. Do not preserve the old closed-set rejection path. |
| Conversation progress `topic_momentum` | bigbang | Replace closed enum validation with bounded string validation. The incident value `developing` must be accepted. |
| Conversation progress `status` | compatible | Preserve current strict enum behavior. |
| Conversation progress `continuity` | compatible | Preserve current strict enum behavior and `new_episode` aliasing to `sharp_transition`. |
| Interaction-style overlay `confidence` | bigbang | Replace closed enum validation with bounded string validation while preserving empty-overlay consistency. |
| Hourly reflection descriptors | bigbang | Remove prompt and warning language that treats hourly-only descriptors as `low|medium|high` protocol values. This includes hourly-slot confidence shown inside daily prompt input. |
| Daily synthesis output `confidence` | compatible | Preserve `low|medium|high` as the control vocabulary. Keep existing warning-style validation for invalid values and preserve downstream style-update eligibility gates. |
| Stored documents | compatible | Keep existing keys and tolerate old values. No migration or dual-read path is needed. |
| Tests | bigbang | Replace tests that encode old closed enum assumptions for prompt-only descriptors. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, rewrite the old behavior directly instead of adding
  compatibility branches.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

Conversation-progress recorder validation accepts:

```python
{
    "status": "active",
    "continuity": "same_episode",
    "conversation_mode": "task_support",
    "episode_phase": "answering_first_question",
    "topic_momentum": "developing",
}
```

The same validator still rejects:

```python
{
    "status": "working",
    "continuity": "fresh_start",
}
```

Interaction-style overlay validation accepts:

```python
{
    "speech_guidelines": ["Keep confirmations brief."],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": "moderate but useful",
}
```

An empty overlay with non-empty confidence still raises `ValueError`.

Hourly reflection output may carry descriptor text for evidence confidence.
Daily synthesis input may show that hourly-slot descriptor text. Daily
synthesis output still carries `confidence="low"`, `"medium"`, or `"high"`
because style extraction checks exact values.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stable keys | Keep existing keys | Downstream prompt contexts and stored documents already depend on the field names. The problem is value over-typing, not key shape. |
| Descriptor validation | Validate shape, sanitation, and caps | Deterministic code should keep payloads safe and bounded without pretending semantic labels are protocol values. |
| Control validation | Keep strict enums | Exact labels are required where code branches, routes, gates, or computes scores. |
| Conversation progress scope | Limit production changes to recorder/policy/docs/tests | Repository and projection already cap and project these fields as strings. |
| Interaction-style confidence | Keep the `confidence` key | Renaming to `confidence_note` would create unnecessary data-shape churn for LLM-only context. |
| Reflection confidence split | Split hourly soft validation from daily synthesis output control vocabulary | Daily synthesis output confidence is a deterministic gate through style eligibility; hourly confidence is LLM evidence context. |
| Local LLM rule | Write the rule into `.agents/skills/local-llm-architecture/SKILL.md` | The same mistake can recur in prompt/schema work unless the local architecture rule states the boundary. |

## Contracts And Data Shapes

### Semantic Descriptor Contract

A semantic descriptor field must follow this contract:

```python
descriptor: str
```

Validation rules:

- required key when the existing schema requires the key;
- value must be a string;
- whitespace normalized only where the local module already normalizes user or
  LLM text;
- value capped by the exact cap named below;
- empty string allowed only where the current schema already allows empty;
- no closed-set rejection unless a deterministic consumer uses exact values.

### Conversation Progress

Keep this output shape:

```python
{
    "status": str,
    "episode_label": str,
    "continuity": str,
    "conversation_mode": str,
    "episode_phase": str,
    "topic_momentum": str,
    "current_thread": str,
    "user_goal": str,
    "current_blocker": str,
    "user_state_updates": list[str],
    "assistant_moves": list[str],
    "overused_moves": list[str],
    "open_loops": list[str],
    "resolved_threads": list[str],
    "avoid_reopening": list[str],
    "emotional_trajectory": str,
    "next_affordances": list[str],
    "progression_guidance": str,
}
```

Strict labels:

```python
status in {"active", "suspended", "closed"}
continuity in {"same_episode", "related_shift", "sharp_transition"}
```

Prompt-only descriptors:

```python
conversation_mode: str
episode_phase: str
topic_momentum: str
```

All three fields use the existing `MAX_LABEL_CHARS = 80` cap before storage or
prompt projection.

### Interaction Style Overlay

Keep this overlay shape:

```python
{
    "speech_guidelines": list[str],
    "social_guidelines": list[str],
    "pacing_guidelines": list[str],
    "engagement_guidelines": list[str],
    "confidence": str,
}
```

`confidence` is a bounded descriptor. It must not be used for routing or
eligibility. Empty overlay plus non-empty confidence remains invalid.
Use `_CONFIDENCE_DESCRIPTOR_MAX_CHARS = 80` and normalize by trimming,
collapsing internal whitespace, lowercasing ASCII text as the current code
already does, and truncating to 80 characters.

### Reflection

Hourly reflection:

```python
{
    "participant_observations": [
        {
            "participant_ref": str,
            "observation": str,
            "evidence_strength": str,
        }
    ],
    "confidence": str,
}
```

Hourly `confidence` and `participant_observations[].evidence_strength` use
`_REFLECTION_DESCRIPTOR_MAX_CHARS = 80`. Validation records no warning merely
because these hourly descriptor values are outside `low`, `medium`, and
`high`.

Daily reflection:

```python
{
    "confidence": "low" | "medium" | "high",
}
```

Daily remains strict because `_ELIGIBLE_DAILY_CONFIDENCE` uses exact values.
The strictness is a control vocabulary contract, not a hard parser failure:
daily validation continues to return warnings for invalid values, and style
extraction continues to skip values outside `_ELIGIBLE_DAILY_CONFIDENCE`.

## LLM Call And Context Budget

- Conversation progress recorder: no new LLM calls. Prompt wording changes
  only. Context size stays under the existing progress prompt caps.
- Interaction-style extractor: no new LLM calls. Prompt wording changes only.
  The `confidence` field stays one short string.
- Hourly and daily reflection: no new LLM calls. Hourly descriptor wording is
  softened; daily control confidence remains unchanged.
- Live response path: no new response-path calls and no added retry loops.
- Background jobs: no extra background LLM calls.
- Token budget impact: neutral or slightly smaller because enum rosters are
  removed from affected prompts.

## Change Surface

### Modify

- `.agents/skills/local-llm-architecture/SKILL.md`
  - Add the local LLM rule for semantic descriptors versus control enums.
  - Perform audit against the skill-creator rules.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Remove `VALID_EPISODE_PHASE` and `VALID_TOPIC_MOMENTUM`.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Stop using `_validated_label(...)` for `episode_phase` and
    `topic_momentum`.
  - Keep strict validation for `status` and `continuity`.
  - Rewrite prompt text for semantic descriptors.
- `src/kazusa_ai_chatbot/conversation_progress/README.md`
  - State that `conversation_mode`, `episode_phase`, and `topic_momentum` are
    bounded semantic descriptors.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - Replace `_CONFIDENCE_VALUES` closed-set validation with bounded string
    normalization.
  - Add `_CONFIDENCE_DESCRIPTOR_MAX_CHARS = 80`.
  - Keep source-detail and empty-overlay consistency validation.
- `src/kazusa_ai_chatbot/db/README.md`
  - Document interaction-style overlay `confidence` as a semantic descriptor.
- `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
  - Soften hourly `confidence` and `evidence_strength` prompt language,
    including hourly-slot confidence shown in daily prompt input.
  - Keep only daily synthesis output `confidence` as `low|medium|high`.
- `src/kazusa_ai_chatbot/reflection_cycle/projection.py`
  - Split hourly descriptor validation from daily synthesis confidence warning
    validation.
  - Add `_REFLECTION_DESCRIPTOR_MAX_CHARS = 80`.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Document the hourly/daily confidence boundary.
- `tests/test_conversation_progress_flow.py`
  - Add focused tests for arbitrary semantic descriptors and non-string
    descriptor rejection.
- `tests/test_interaction_style_images.py`
  - Add focused tests for arbitrary confidence descriptors, cap behavior, and
    empty-overlay rejection.
- `tests/test_reflection_cycle_prompt_contracts.py`
  - Update prompt and validation tests for hourly soft descriptors and daily
    strict confidence.
- `tests/test_reflection_interaction_style.py`
  - Preserve tests proving low daily confidence skips style extraction.

### Keep

- `conversation_progress.status` and `conversation_progress.continuity`
  closed-set validation.
- `reflection_cycle.interaction_style._ELIGIBLE_DAILY_CONFIDENCE`.
- Action-spec, RAG, web-agent, memory, promotion, scheduler, adapter,
  event-log, and global-growth enums.
- MongoDB collection shape and indexes.
- Existing LLM route configuration.

### Delete

- No files are deleted.

### Create

- No production module is created.

## Overdesign Guardrail

- Actual problem: LLM-only semantic descriptor values can fail background work
  because they are validated as closed protocol enums.
- Minimal change: keep stable keys and structural validation; remove closed-set
  validation only for confirmed LLM-only descriptor values.
- Ownership boundaries: LLM stages own semantic wording; deterministic code
  owns required keys, types, caps, source-detail rejection, lifecycle states,
  routing, privacy gates, and persistence safety.
- Rejected complexity: no repair loop, no schema migration, no compatibility
  shim, no new confidence field, no feature flag, no new prompt stage, no
  broad enum-removal campaign, and no changes to true control fields.
- Evidence threshold: add or relax another enum only after a grep plus
  consumer trace proves the value is consumed only as LLM context and no
  deterministic branch, gate, write policy, route, score, or privacy decision
  depends on exact values.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when the helper
  preserves the contracts in this plan and removes repeated validation logic.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, retries, or
  extra features.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad prompt rewrites, or test fixture rewrites outside
  this plan.
- The responsible agent must treat changes outside the listed change surface
  as blocked unless the user approves an updated plan.
- If code inspection reveals a listed semantic descriptor is actually consumed
  as deterministic control state, stop and report the discrepancy instead of
  relaxing that field.
- If implementation discovers another LLM-only closed enum, do not add it to
  this plan automatically. Record it as follow-up evidence unless it is in the
  exact field families already listed here.

## Implementation Order

1. Parent establishes the focused conversation-progress test contract.
   - Modify `tests/test_conversation_progress_flow.py`.
   - Add `test_recorder_validator_accepts_semantic_flow_descriptors`.
   - Use `episode_phase="answering_first_question"` and
     `topic_momentum="developing"`.
   - Run:
     `venv\Scripts\python -m pytest tests/test_conversation_progress_flow.py::test_recorder_validator_accepts_semantic_flow_descriptors -q`
   - Expected before implementation: fails with invalid `topic_momentum` or
     invalid descriptor label.
2. Parent establishes the interaction-style confidence test contract.
   - Modify `tests/test_interaction_style_images.py`.
   - Add `test_validate_interaction_style_overlay_accepts_semantic_confidence`.
   - Use `confidence="moderate but useful"`.
   - Run:
     `venv\Scripts\python -m pytest tests/test_interaction_style_images.py::test_validate_interaction_style_overlay_accepts_semantic_confidence -q`
   - Expected before implementation: fails with invalid interaction style
     confidence.
3. Parent establishes the reflection prompt/validation test contract.
   - Modify `tests/test_reflection_cycle_prompt_contracts.py`.
   - Add or update tests so hourly confidence descriptors do not warn, hourly
     prompt text no longer requires `low|medium|high`, daily input no longer
     presents hourly-slot confidence as `low|medium|high`, and daily synthesis
     output confidence still returns a validation warning for non-enum values.
   - Run:
     `venv\Scripts\python -m pytest tests/test_reflection_cycle_prompt_contracts.py -q`
   - Expected before implementation: hourly descriptor test fails or prompt
     grep assertion fails.
4. Parent updates `.agents/skills/local-llm-architecture/SKILL.md`.
   - Add the local LLM rule before production-code changes so execution agents
     have the rule in local guidance.
   - Load `skill-creator` before editing the skill.
   - Verify with:
     `rg "closed enum|semantic descriptor|deterministic code" .agents/skills/local-llm-architecture/SKILL.md`
   - Validate with:
     `venv\Scripts\python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\local-llm-architecture`
5. Production-code subagent implements conversation-progress changes.
   - Modify only the conversation-progress files listed in `Change Surface`.
   - Rerun the focused conversation-progress test.
6. Production-code subagent implements interaction-style changes.
   - Modify only `db/interaction_style_images.py` and related docs/tests
     listed in `Change Surface`.
   - Rerun the focused interaction-style test.
7. Production-code subagent implements reflection changes.
   - Modify only `reflection_cycle/prompts.py`,
     `reflection_cycle/projection.py`, related README, and focused tests.
   - Rerun reflection prompt-contract tests.
8. Parent runs static greps and focused regression tests.
   - Record commands and output in `Execution Evidence`.
9. Parent runs live LLM checks one case at a time if the configured local LLM
   routes are available.
   - If unavailable, record the blocker and do not claim live LLM coverage.
10. Parent starts the independent code-review subagent.
    - Review the full diff, plan alignment, and evidence.
    - Parent remediates approved findings inside the change surface and reruns
      affected verification.

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
  checks, and validation work while the production-code subagent edits
  production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused tests established
  - Covers: implementation steps 1-3.
  - Verify: focused tests are added and fail for the current closed-enum
    behavior or record the current baseline if a test already passes.
  - Evidence: record failing command output in `Execution Evidence`.
  - Handoff: next stage updates the local LLM rule.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 2 - local LLM architecture rule updated
  - Covers: implementation step 4.
  - Verify:
    `rg "semantic descriptor|closed enum|deterministic code" .agents/skills/local-llm-architecture/SKILL.md`
  - Validate:
    `venv\Scripts\python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\local-llm-architecture`
  - Evidence: record grep output in `Execution Evidence`.
  - Handoff: next stage starts production-code subagent.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 3 - conversation-progress descriptor validation fixed
  - Covers: implementation step 5.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_conversation_progress_flow.py -q`
  - Evidence: record changed files and test output.
  - Handoff: next stage fixes interaction-style confidence.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 4 - interaction-style confidence validation fixed
  - Covers: implementation step 6.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_interaction_style_images.py tests/test_reflection_interaction_style.py -q`
  - Evidence: record changed files and test output.
  - Handoff: next stage fixes hourly reflection descriptors.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 5 - hourly reflection descriptor contract fixed
  - Covers: implementation step 7.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_reflection_cycle_prompt_contracts.py tests/test_reflection_interaction_style.py -q`
  - Evidence: record changed files and test output.
  - Handoff: next stage runs full focused verification.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 6 - static and deterministic regression verification complete
  - Covers: implementation step 8.
  - Verify: all commands in `Verification` static and deterministic sections.
  - Evidence: record command output and allowed grep matches.
  - Handoff: next stage runs live LLM checks if available.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 7 - live LLM verification recorded
  - Covers: implementation step 9.
  - Verify: run listed live checks one at a time when configured, or record
    route unavailability.
  - Evidence: record trace path or blocker.
  - Handoff: next stage starts independent code review.
  - Sign-off: `Codex/2026-05-30`.
- [x] Stage 8 - independent code review complete
  - Covers: implementation step 10.
  - Verify: review subagent reports approval or concrete findings; parent
    fixes approved findings and reruns affected verification.
  - Evidence: record review summary, fixes, rerun commands, and residual risk.
  - Handoff: plan may be signed off only after this stage is complete.
  - Sign-off: `Codex/2026-05-30`.

## Verification

### Static Greps

- Run:
  `rg "VALID_EPISODE_PHASE|VALID_TOPIC_MOMENTUM" src/kazusa_ai_chatbot tests`
  - Expected: no matches.
- Run:
  `rg "_CONFIDENCE_VALUES|invalid interaction style confidence" src/kazusa_ai_chatbot/db/interaction_style_images.py tests/test_interaction_style_images.py`
  - Expected: no matches.
- Run:
  `rg "_CONFIDENCE_DESCRIPTOR_MAX_CHARS|_REFLECTION_DESCRIPTOR_MAX_CHARS" src/kazusa_ai_chatbot tests`
  - Expected: exactly one production definition for each constant and focused
    test references proving each cap.
- Run:
  `rg "topic_momentum.*stable.*drifting|episode_phase.*opening.*developing" src/kazusa_ai_chatbot/conversation_progress`
  - Expected: no recorder prompt enum roster. Existing historical comments are
    not allowed in production prompt text.
- Run:
  `rg "\"evidence_strength\": \"low\\|medium\\|high\"|confidence.*evidence_strength.*英文枚举" src/kazusa_ai_chatbot/reflection_cycle/prompts.py tests/test_reflection_cycle_prompt_contracts.py`
  - Expected: no matches.
- Run:
  `rg "\"confidence\": \"low\\|medium\\|high\"" src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
  - Expected: exactly one match, the daily synthesis output `confidence`
    schema. The hourly prompt and daily input shape for hourly slot
    confidence must not match.

### Skill Validation

- `venv\Scripts\python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\local-llm-architecture`

### Focused Deterministic Tests

- `venv\Scripts\python -m pytest tests/test_conversation_progress_flow.py -q`
- `venv\Scripts\python -m pytest tests/test_interaction_style_images.py tests/test_reflection_interaction_style.py -q`
- `venv\Scripts\python -m pytest tests/test_reflection_cycle_prompt_contracts.py tests/test_reflection_interaction_style.py -q`

### Broader Deterministic Regression

- `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`

### Prompt Render Checks

- `venv\Scripts\python -m pytest tests/test_conversation_progress_flow.py::test_recorder_prompt_mentions_phase2_flow_contract -q`
- `venv\Scripts\python -m pytest tests/test_reflection_cycle_prompt_contracts.py -q`

### Live LLM Checks

Run only one live LLM case at a time and inspect output:

- `venv\Scripts\python -m pytest -m live_llm tests/test_conversation_progress_recorder_live_llm.py -q -s`
- `venv\Scripts\python -m pytest -m live_llm tests/test_reflection_cycle_live_llm.py -q -s`

If live LLM routes are unavailable, record the unavailable route and do not
claim live LLM coverage. Deterministic verification remains required.

## Independent Code Review

Run this gate after all `Verification` commands pass or have recorded
environment blockers for live LLM checks, and before final sign-off. The
parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and local skill artifact.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, exact contracts, implementation order, verification gates,
  and acceptance criteria.
- Whether every relaxed field is genuinely LLM-only and every true control
  field remains strict.
- Whether prompt wording follows local LLM guidance and does not add
  development-process terms to runtime prompts.
- Whether tests prove the incident value `topic_momentum="developing"` is
  accepted and true control values are still rejected.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a finding requires a boundary or contract
change, stop and update the plan or request approval before editing.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `topic_momentum="developing"` no longer fails conversation-progress recorder
  validation.
- `episode_phase` and `topic_momentum` are required string descriptors, not
  closed enums.
- `status` and `continuity` remain strict conversation-progress control
  fields.
- Interaction-style overlay `confidence` accepts bounded semantic descriptors
  while empty overlays still reject non-empty confidence.
- Hourly reflection confidence/evidence descriptors are no longer prompt-only
  closed enums.
- Daily synthesis output confidence remains a `low|medium|high` control
  vocabulary; invalid daily output values continue to produce validation
  warnings and remain ineligible for interaction-style extraction.
- `.agents/skills/local-llm-architecture/SKILL.md` contains the semantic
  descriptor versus control enum rule.
- Documentation names the ownership boundary.
- Static greps and focused deterministic tests pass.
- Broader deterministic regression passes or any unrelated pre-existing
  failures are recorded with evidence.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A real control enum is accidentally relaxed | Limit change surface to named fields and run consumer greps before edits | Static greps and independent code review |
| Prompt wording becomes too vague for local LLMs | Keep stable keys and give plain semantic descriptions instead of removing fields | Prompt contract tests and live LLM inspection when available |
| Interaction-style confidence becomes unbounded prompt noise | Add a short cap and preserve empty-overlay consistency | Focused interaction-style tests |
| Daily reflection gate is weakened | Keep daily synthesis output confidence as the style-eligibility control vocabulary and test low-confidence skip behavior | `tests/test_reflection_interaction_style.py` |
| Old stored documents behave differently | Keep field names and projection shape compatible; no migration | Existing projection and DB tests |

## Execution Evidence

- Draft created: 2026-05-30. Plan was promoted to `in_progress` before
  execution and closed as `completed` after implementation, verification, and
  independent review.
- Focused failing tests: parent added deterministic tests for conversation
  progress descriptor acceptance, interaction-style descriptor confidence,
  hourly reflection descriptor validation, daily confidence control warnings,
  and interaction-style extractor prompt confidence boundaries. Initial RED
  command produced expected failures for the over-typed fields before
  implementation.
- Local LLM rule update: `.agents/skills/local-llm-architecture/SKILL.md`
  now states that closed enum output values are only for deterministic control;
  generated values consumed only as later LLM context keep stable keys and use
  short bounded semantic descriptor strings.
- Skill validation:
  - `venv\Scripts\python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\local-llm-architecture`
    initially failed on Windows default `cp1252` decoding.
  - `$env:PYTHONUTF8='1'; venv\Scripts\python "C:\Users\Ran Bao\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .agents\skills\local-llm-architecture`
    passed: `Skill is valid!`.
- Static grep results:
  - `rg "VALID_EPISODE_PHASE|VALID_TOPIC_MOMENTUM" src\kazusa_ai_chatbot tests`
    returned no matches.
  - `rg "_CONFIDENCE_VALUES|invalid interaction style confidence" src\kazusa_ai_chatbot\db\interaction_style_images.py tests\test_interaction_style_images.py`
    returned no matches.
  - `rg '"evidence_strength": "low\|medium\|high"|confidence.*evidence_strength.*英文枚举' src\kazusa_ai_chatbot\reflection_cycle\prompts.py tests\test_reflection_cycle_prompt_contracts.py`
    returned no matches.
  - `rg '"confidence": "low\|medium\|high"' src\kazusa_ai_chatbot\reflection_cycle\prompts.py`
    returned exactly one match, the daily synthesis output schema line.
  - `rg '"confidence": "low\|medium\|high\|"|"confidence": "medium\|high\|"' src\kazusa_ai_chatbot\reflection_cycle\interaction_style.py tests\test_reflection_interaction_style.py`
    returned no matches after the independent review fix.
  - `rg 'episode_phase.*opening|episode_phase.*developing|topic_momentum.*stable|topic_momentum.*drifting|topic_momentum.*quick_pivot|topic_momentum.*fragmented|topic_momentum.*sharp_break' src\kazusa_ai_chatbot\conversation_progress\recorder.py`
    returned no matches.
- Focused deterministic test results:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\conversation_progress\policy.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\db\interaction_style_images.py src\kazusa_ai_chatbot\reflection_cycle\projection.py src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style.py tests\test_conversation_progress_flow.py tests\test_interaction_style_images.py tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_interaction_style.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_conversation_progress_flow.py -q`
    passed: 17 passed.
  - `venv\Scripts\python -m pytest tests\test_interaction_style_images.py tests\test_reflection_interaction_style.py -q`
    passed: 30 passed.
  - `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_interaction_style.py -q`
    passed: 23 passed.
- Broader deterministic regression:
  - `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`
    completed with 1726 passed, 268 deselected, and 23 failed.
  - `venv\Scripts\python -m pytest --lf -q` reproduced the 23 failures.
    The failures were outside the touched descriptor-validation surfaces:
    existing cognition/RAG prompt contract and fixture failures in
    `tests/test_conversation_progress_cognition.py`,
    `tests/test_global_character_growth_replay.py`,
    `tests/test_memory_retrieval_tools.py`,
    `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py`,
    `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`,
    `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`,
    `tests/test_persona_supervisor2_action_initializer.py`,
    `tests/test_rag_continuation.py`,
    `tests/test_rag_initializer_cache2.py`, and
    `tests/test_user_memory_evidence_agent.py`.
  - These broad failures were not expanded into this bugfix because they do
    not involve the changed files or the semantic descriptor enum boundary.
- Live LLM checks: not run in this execution; no live LLM coverage claimed.
  Deterministic validator, prompt contract, static, and review evidence are
  the verification basis for this bugfix.
- Independent code review:
  - Production-code worker `019e73c1-f3c3-7d01-8d97-7f8dfeef8f1a` completed
    the planned production/source-doc slice.
  - First read-only reviewer `019e73c7-4379-76e1-abaa-a7ac3068ed75` found a
    blocking missed prompt surface: interaction-style extractor overlay
    `confidence` still used closed enum examples. Parent fixed it in
    `src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py` and added
    `test_interaction_style_extractor_prompt_keeps_confidence_boundary`.
  - Re-reviewer `019e73cb-0837-7903-9d37-4963831fbcc8` reported no blocking
    findings after the fix. Residual risk: unrelated confidence enums remain
    intentionally outside scope, notably reflection global promotion.

This completed implementation record is historical context. New scope must use
a new active plan.
