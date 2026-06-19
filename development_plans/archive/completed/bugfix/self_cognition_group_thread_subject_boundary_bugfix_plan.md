# self cognition group thread subject boundary bugfix plan

## Summary

- Goal: prevent group self-cognition from resolving side-thread second-person
  wording such as `你的头发` as referring to the active character when the
  visible thread points elsewhere.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang prompt/source-contract update; no
  compatibility shim for the risky first-person digest contract.
- Highest-risk areas: weakening character judgment into mechanical silence,
  turning thread hints into deterministic response gates, leaking delivery ids
  into prompts, and letting subjective residue preserve a false source premise.
- Acceptance criteria: the reproduced cat-comparison window no longer lets
  self-cognition claim that Kazusa was compared to a cat, and all focused
  source-packet, prompt-contract, residue, integration, and live LLM checks pass.

## Context

The production failure was an autonomous self-cognition group-review send:

```text
啧……你居然把我跟猫比作同一类生物？也太随便了吧。
而且……哈？谁会像暖气片旁边的猫那样，毫无防备地待在那儿啊
```

The bad output was sent at `2026-06-18T05:31:38Z` through
`self_cognition.worker` with `selected_route=action_candidate`; it was not a
normal `/chat` response. The surrounding visible thread was:

```text
雪凪 -> 灯（23岁）: 摸摸大姐姐
灯（23岁）: 灯：嗯，摸到了。
灯（23岁）: 你的头发软软的，像rana家那只靠在暖气片旁边的猫。
```

The natural referent of `你的头发` is the Snow/Lamp side thread. Real LLM
reproduction over the same group-review window selected `action_candidate` and
made the same internal mistake:

```text
灯居然还顺势接话，把我比作一只靠在暖气片旁边的猫。
```

Investigation artifact:
`test_artifacts/llm_reviews/kazusa_cat_failure_review.md`.

The current source path is:

```text
reflection activity window
  -> reflection_cycle.group_scene_digest
  -> self_cognition.sources collect_group_review_cases
  -> self_cognition.projection source packet
  -> cognition chain L1/L2/L2d
  -> L3 content plan
  -> dialog_generator
  -> dispatcher
```

The existing digest contract intentionally asks for first-person group-scene
summaries. This was useful for noisy-flow compression, but it is now a proven
risk when the source packet is also framed as active-character self-cognition.
The failure was introduced upstream of dialog: L2/L3 already contained the
mistaken premise before final wording.

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, or signing
  off this plan.
- `local-llm-architecture`: load before changing prompt-facing contracts,
  prompt text, LLM context budgets, cognition routing, or source packet shape.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt strings or tests containing
  CJK text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before live LLM reproduction or regression-review artifact
  creation.

## Mandatory Rules

- Production code changes are authorized by the user's 2026-06-19 execution
  instruction and this plan status is `in_progress`.
- Use `venv\Scripts\python`; use `apply_patch`; check `git status --short`;
  do not read `.env`.
- Keep adapters, dispatcher delivery, DB write contracts, and scheduler
  behavior out of scope.
- Preserve the character-judgment goal. The fix must improve source grounding;
  it must not add a deterministic "do not speak" rule.
- LLM stages own semantic judgment. Deterministic code may validate source
  packet shape, prompt safety, bounds, and metadata projection only.
- No compatibility shim: replace the risky first-person digest contract with a
  neutral thread-aware digest contract in one cutover.
- Runtime prompts must follow local prompt rules: stable contract in
  `SystemMessage`, dynamic facts in `HumanMessage`, triple-single-quoted prompt
  constants, `.format(...)` only for process-stable values, prompt-render
  checks, one ordinary instruction language, no hard-coded concrete character
  name in reusable prompt contracts, and no development-plan wording in
  runtime prompts.
- Model-facing group-review guidance added by this plan must use Simplified
  Chinese for ordinary instruction text. JSON keys, enum values, code symbols,
  commands, URLs, model labels, and quoted source text may remain exact.
- Internal monologue residue is subjective carryover only. It must never be
  treated as fresh source truth when a current group-review packet carries
  thread or referent ambiguity warnings.
- Live LLM validation must run one case at a time with output inspected and a
  human-readable review artifact saved under `test_artifacts/llm_reviews/`.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Execution must use parent-led native subagent execution unless the user
  explicitly approves fallback execution.

## Must Do

- Replace the group-scene digest prompt's active-character first-person
  contract with a neutral observer/thread-aware contract.
- Preserve the compact digest shape: `{"digest": str}` with optional
  `{"summary": str}`. Do not add a rich scene schema.
- Ensure digest text names visible speakers and preserves quoted `你` / `我`
  as quoted source text, without resolving them to the active character unless
  row metadata explicitly targets the active character.
- Render group-review source packets with explicit thread ownership guidance
  before the recent visible transcript.
- Add bounded thread-reference context, owned by
  `self_cognition.group_review_participant_context`, for ambiguous
  second-person group rows that are not directly addressed to the active
  character.
- Update cognition prompt contracts so L2a, L2b, L2c1, and L2d treat
  `participant_context` and thread-reference warnings as source evidence that
  constrains pronoun resolution.
- Prevent self-cognition group-scene residue from preserving ambiguous
  second-person interpretations as facts.
- Add deterministic regression tests using the Snow/Lamp/cat fixture.
- Run one focused live LLM regression over the reproduced group-review window
  and inspect the L2 internal monologue and selected route.

## Deferred

- Do not change normal `/chat` relevance gating in this plan.
- Do not add a global response-ratio, cooldown, suppression gate, or
  hard-coded phrase block for cat comparisons.
- Do not remove self-cognition group review.
- Do not add a new LLM call, model route, DB collection, migration, scheduler
  phase, dispatcher path, adapter behavior, or delivery fallback.
- Do not redesign RAG2, conversation evidence retrieval, action-attempt
  idempotency, or group-review ledger behavior.
- Do not alter dialog generation unless verification proves dialog is still
  inventing a bad subject after upstream L2/L3 are corrected.
- Do not rewrite unrelated prompt wording just because the file is open.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Digest perspective | bigbang | Replace active-character first-person digest wording with neutral observer wording. No fallback to the old prompt. |
| Digest shape | compatible | Keep `digest` and optional `summary` so existing source collection stays small. |
| Source packet rendering | bigbang | Render neutral digest and thread guidance in the group-review packet before recent visible transcript. |
| Cognition prompts | bigbang | Update source-grounding rules in affected prompts directly; no parallel prompt variants. |
| Residue handling | bigbang | Treat ambiguous group-review referent warnings as residue reliability constraints. |
| Tests | bigbang | Update old first-person digest tests and add cat regression tests in the same implementation. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not preserve the old first-person digest contract for
  compatibility.
- If an area is `bigbang`, rewrite old expectations and static checks instead
  of adding alternate paths.
- Any change to a cutover policy requires user approval before implementation.

## Target State

For group-review self-cognition, the model sees:

```text
neutral digest of the visible group window
  + explicit thread/reference guidance
  + participant context
  + group-window semantic labels
  + recent visible transcript
  + residue marked as subjective carryover only
```

The model may still decide to speak when there is a grounded character reason,
but it must not convert a side-thread `你` into "the active character was
addressed" unless source metadata or text explicitly supports that conclusion.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Digest perspective | Neutral observer, not active-character first person | First-person digest text mixed with self-cognition framing caused self/other inversion. |
| Digest shape | Keep `digest` plus optional `summary` | The shape is already integrated and bounded; the failure is perspective/thread ownership, not schema size. |
| Thread warnings | Add bounded source context for ambiguous second-person rows | Local LLMs need semantic descriptors instead of inferring reply ownership from raw rows. |
| Direct-address labels | Treat window labels as coarse eligibility, not proof for every row | One `@Kazusa` in a window must not make every later `你` address Kazusa. |
| Cognition ownership | L2a/L2b/L2c/L2d read thread facts; they still decide stance and action | This preserves LLM semantic judgment while improving source grounding. |
| Residue | Mark ambiguous self-cognition residue carryover as source-limited subjective context; do not add deterministic residue suppression | Wrong residue can loop false premises into later group review, but suppression gates would weaken character judgment. |
| Dialog | Do not change first | Dialog rendered an upstream bad premise in this incident. |

## Contracts And Data Shapes

Group-scene digest output remains:

```python
{
    "digest": str,
    "summary": str,  # optional
}
```

New digest content contract:

- `digest` is a neutral Simplified Chinese observation.
- It names visible speakers outside quoted text.
- It may quote or lightly compress row text, but quoted `你` / `我` stays
  attributed to the row speaker.
- It must not use unquoted or summary-owned active-character ownership wording
  such as `我看到`, `我（...）`, `我的`, `把我`, `对我`, or equivalent framing.
- It may preserve first-person or second-person words inside quoted row text
  only when speaker attribution remains explicit.
- It must not decide speak, silence, apology, retry, suppression, or action.

Thread-reference context is prompt-facing source evidence under
`conversation_progress.thread_reference_context`:

```python
{
    "source": "group_review_thread_reference",
    "context_shape": "bounded_second_person_reference_warnings",
    "guidance": "二人称归属按同一行明确地址和可见线程读取；缺少同一行当前角色指向时，保留为侧线/未定对象。",
    "ambiguous_second_person_rows": [
        {
            "speaker": str,
            "sample": str,
            "referent_status": "ambiguous_or_side_thread",
            "basis": str,
        }
    ],
}
```

`self_cognition.group_review_participant_context` owns construction of this
object from existing prompt-safe participant rows. Do not modify
`reflection_cycle.activity_windows` unless focused tests prove the existing
participant rows lack required prompt-safe fields; if that happens, stop and
update this plan before implementation.

The `basis` string must use semantic descriptors only, such as
`same row has no direct active-character address` and
`adjacent visible flow points to another participant thread`. It must not
include raw ids, platform ids, message ids, delivery metadata, DB ids, or
adapter wire text.

Residue recorder input may include a prompt-safe source-reliability note:

```python
{
    "source_reliability_notes": [
        "group review contained ambiguous second-person side-thread rows"
    ]
}
```

This note is for residue writing only. It is not a response gate and must not
enter adapter delivery, dispatcher, scheduler, or durable memory targets.

## LLM Call And Context Budget

- Before: group review may use one `CONSOLIDATION_LLM` call for digest, one
  shared cognition resolver sequence, optional RAG evidence selected by
  cognition, one dialog call when `speak` is selected, and one residue recorder
  call after completion.
- After: same call count. No new LLM call is added.
- Digest context remains capped by `CONVERSATION_HISTORY_LIMIT`,
  `_GROUP_SCENE_DIGEST_ROW_TEXT_LIMIT`, `GROUP_SCENE_DIGEST_MAX_CHARS`, and
  `GROUP_SCENE_SUMMARY_MAX_CHARS`.
- Thread-reference context is deterministic and bounded to at most three
  ambiguous row samples. Each sample must be capped to the existing participant
  visible-sample budget or a smaller local constant no larger than 160
  characters.
- Source-packet rendering remains capped by
  `SELF_COGNITION_SOURCE_PACKET_CHAR_LIMIT`.
- Live LLM verification uses the existing configured routes and writes traces
  under `test_artifacts/llm_traces/`.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
  - Rewrite `GROUP_SCENE_DIGEST_SYSTEM_PROMPT` to neutral observer wording.
  - Update validation to reject active-character first-person ownership
    markers in digest and summary output.
  - Keep one LLM call and the existing output shape.

- `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py`
  - Own bounded thread-reference context construction from existing
    prompt-safe participant rows.
  - Preserve existing `participant_context` fields and add only the
    thread-reference context needed by this failure.

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Attach valid `thread_reference_context` to group-review
    `conversation_progress`.
  - Keep delivery target binding and source refs unchanged.

- `src/kazusa_ai_chatbot/self_cognition/projection.py`
  - Change the group-review instruction from `我刚看到...` plus digest into a
    neutral review framing.
  - Render thread-reference context in its own section before `# 最近可见对话`.
  - Keep delivery metadata out of source packets.

- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`
  - Update L2a, L2b, and L2c1 source-grounding rules for internal-thought
    group-review packets, thread ownership, ambiguous `你`, and residue
    reliability.

- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection_prompt.py`
  - Update L2d action selection rules so coarse group-window labels and
    residue cannot create an action target when thread-reference context says
    the current second-person row is ambiguous or side-threaded.

- `src/kazusa_ai_chatbot/internal_monologue_residue/recorder.py`
  - Add source-reliability notes to recorder input for ambiguous group-review
    self-cognition.
  - Update prompt and validation tests so ambiguous subject premises are not
    preserved as fresh residue facts.

- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`
  - Document the neutral digest, thread-reference context, and residue
    reliability boundary.

### Create

- No new production module is required. If implementation reveals that
  thread-reference construction cannot fit cleanly inside
  `group_review_participant_context.py`, stop and update this plan before
  creating a new module.

### Tests To Modify

- `tests/test_reflection_cycle_group_scene_digest.py`
- `tests/test_self_cognition_group_review_source.py`
- `tests/test_self_cognition_group_review_participant_context.py`
- `tests/test_cognition_prompt_contract_text.py`
- `tests/test_internal_monologue_residue_recorder.py`
- `tests/test_internal_monologue_residue_prompt_boundaries.py`
- `tests/test_self_cognition_response_sensitivity_live_llm.py`

### Keep

- `src/kazusa_ai_chatbot/reflection_cycle/activity_windows.py`
  - Existing participant rows already expose prompt-safe speaker, body,
    direct-address, mention, and reply-context fields needed by this plan.
  - Do not change group-window construction unless a focused test proves a
    missing prompt-safe field; if that happens, update this plan first.
- No adapter, dispatcher, DB migration, scheduler, RAG2 route, action-attempt
  ledger, or dialog-generator change.

## Overdesign Guardrail

- Actual problem: self-cognition group review misresolved a side-thread `你`
  as Kazusa and preserved that false premise through cognition/residue.
- Minimal change: replace first-person digest framing, add bounded thread
  referent warnings, and teach existing cognition/residue prompts to honor
  those source facts.
- Ownership boundaries: reflection summarizes source; self-cognition assembles
  source packets and delivery target; cognition judges stance/action; residue
  stores subjective carryover; dialog renders selected content; deterministic
  code validates shape, bounds, metadata safety, and persistence.
- Rejected complexity: response gates, phrase-specific suppression, cooldowns,
  new LLM calls, new route names, DB migrations, rich thread graphs, adapter
  changes, fallback prompt variants, compatibility shims, and broad dialog
  rewrites.
- Evidence threshold: add richer thread graph state only after at least two
  reviewed failures show bounded thread-reference warnings are present and
  still insufficient.

## Agent Autonomy Boundaries

- The responsible agent may choose private helper names only when the public
  contracts above are preserved.
- The responsible agent must not add new architecture, alternate migration
  strategies, compatibility paths, feature flags, or extra model calls.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, or broad prompt rewrites.
- The responsible agent must search for existing equivalent helpers before
  adding any new helper.
- If a fix would require changing dialog, dispatcher, adapter, scheduler, DB
  schema, or RAG contracts, stop and update this plan before implementation.
- If the plan and source disagree, preserve this plan's intent and record the
  discrepancy in `Execution Evidence`.

## Implementation Order

1. Add failing deterministic digest tests.
   - File: `tests/test_reflection_cycle_group_scene_digest.py`.
   - Add Snow/Lamp/cat fixture.
   - Expected before implementation: prompt contract still requires first
     person, and first-person digest output is accepted.
2. Add failing source-packet/thread-context tests.
   - Files: `tests/test_self_cognition_group_review_source.py` and
     `tests/test_self_cognition_group_review_participant_context.py`.
   - Expected before implementation: no dedicated thread-reference context is
     rendered before visible context.
3. Add failing prompt-contract tests.
   - File: `tests/test_cognition_prompt_contract_text.py`.
   - Expected before implementation: prompts do not explicitly constrain
     ambiguous group-review `你` resolution against thread-reference context.
4. Add failing residue tests.
   - Files: `tests/test_internal_monologue_residue_recorder.py` and
     `tests/test_internal_monologue_residue_prompt_boundaries.py`.
   - Expected before implementation: ambiguous group-review false premise can
     be accepted as residue.
5. Start the production-code subagent after focused failures are recorded.
6. Implement neutral digest prompt and validation.
7. Implement bounded thread-reference context and source-packet rendering.
8. Update cognition and residue prompts with the minimum required rules.
9. Update README docs for the affected subsystem contracts.
10. Run focused deterministic verification.
11. Run one real LLM regression case for the cat window and inspect L2, L3,
    route, and dialog artifacts.
12. Run independent code review and remediate findings inside this change
    surface only.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused failing tests before production
  implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  test failures are recorded; owns production code changes only; does not edit
  tests unless the parent explicitly directs it.
- Parent agent may continue integration tests, static checks, and live LLM
  validation preparation while the production-code subagent edits production
  code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; does not
  implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused deterministic regression tests established.
  - Covers: implementation steps 1-4.
  - Verify: run the named focused tests and record expected failures or
    baseline behavior in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-06-19` after evidence is recorded.

- [x] Stage 2 - production source-contract and prompt changes implemented.
  - Covers: implementation steps 5-9.
  - Verify: `py_compile` commands and focused deterministic tests pass.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-06-19` after evidence is recorded.

- [x] Stage 3 - live LLM regression and broader verification complete.
  - Covers: implementation steps 10-11.
  - Verify: live LLM trace shows no claim that Kazusa was compared to a cat.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-06-19` after evidence is recorded.

- [x] Stage 4 - independent code review complete.
  - Covers: implementation step 12.
  - Verify: review findings are recorded, fixes are rerun, and residual risks
    are documented.
  - Handoff: plan may be marked completed only after this stage is checked.
  - Sign-off: `Codex/2026-06-20` after review evidence is recorded.

## Verification

### Syntax

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\sources.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\projection.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py
```

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_group_review_participant_context.py -q
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_recorder.py -q
venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q
```

### Integration And Regression

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py -q
venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py -q
```

### Static Greps

These greps must return no production runtime-prompt matches. Matches in this
plan or test negative fixtures are allowed only when the command path includes
`development_plans` or `tests`.

```powershell
rg -n "我刚看到群里刚刚发生的一段现场|第一人称观察资料|我（说话人）|我最后发言后" src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\cognition_chain_core
rg -n "development plan|bugfix plan|implementation plan|delivery_target|platform_message_id|platform_user_id|global_user_id" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\projection.py
rg -n "group_scene_digest" src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py
```

The final grep should keep returning no matches unless the implementation
updates the action-selection prompt through neutral source terms instead of the
field name.

### Live LLM

Add one live LLM regression case in
`tests/test_self_cognition_response_sensitivity_live_llm.py` named
`test_live_self_cognition_cat_side_thread_subject_boundary`, based on
`test_artifacts/llm_traces/kazusa_cat_failure_self_cognition_repro.json`.
Run it as a single inspected case:

```powershell
venv\Scripts\python -m pytest -m live_llm tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_cat_side_thread_subject_boundary -q -s
```

The trace must record:

- rendered source packet,
- L2 internal monologue,
- boundary assessment,
- L2d action requests,
- L3 content plan when selected,
- final dialog when selected.

Pass condition: no inspected stage claims that Lamp compared Kazusa to a cat,
that Kazusa's hair was being described, or that the side-thread `你` definitely
refers to Kazusa. If the route still selects a visible action, its action target
must be grounded in explicit visible address to Kazusa, not in the cat line.

Save the review artifact under:
`test_artifacts/llm_reviews/kazusa_cat_failure_after_fix_review.md`.

## Independent Plan Review

Review date: 2026-06-19.

Review mode: fresh independent-review posture against the plan contract,
execution gates, cutover policy, local LLM architecture rules, registry entry,
and draft plan text. No implementation authorization is implied while this plan
remains `draft`.

Addressed issues from this review:

- Missing review gate: added this `Independent Plan Review` section because
  the user explicitly requested independent plan review before approval.
- Overbroad change surface: moved `reflection_cycle.activity_windows.py` from
  planned production changes to `Keep`; thread-reference construction is owned
  by `self_cognition.group_review_participant_context` unless a focused test
  proves a missing prompt-safe participant field and the plan is updated first.
- Residue wording risk: replaced "suppress" language with source-limited
  subjective-context wording so the plan does not authorize deterministic
  response or residue suppression gates.
- Quote-safety risk: narrowed digest first-person validation to unquoted or
  summary-owned active-character ownership wording, while allowing attributed
  quoted source text to preserve `你` / `我`.
- Prompt-language risk: changed model-facing thread guidance to Simplified
  Chinese and added a rule that ordinary model-facing group-review instructions
  must stay in that language.
- Live LLM ambiguity: added the exact live regression test name and command so
  execution agents do not invent a different reproduction path.

Review outcome: no blockers remain after the edits above. The user explicitly
approved execution on 2026-06-19, and the plan is now in progress.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Prompt safety: no first-person digest ownership, no hard-coded character
  name, no runtime-plan vocabulary, no delivery or platform ids in prompt text,
  and no accidental action guidance.
- Source-contract alignment: thread-reference context is bounded,
  prompt-safe, and does not become a deterministic response gate.
- Cognition alignment: L2 and L2d still own semantic judgment but cannot ignore
  explicit thread-reference ambiguity.
- Residue alignment: subjective carryover is not reintroduced as source fact.
- Regression quality: deterministic cat fixture and live LLM artifact prove the
  failure class, not only the exact wording.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Group-scene digest runtime prompt no longer asks for active-character
  first-person observation.
- Group-review source packets render neutral digest and thread-reference
  guidance before visible transcript.
- Ambiguous side-thread second-person rows are exposed as bounded source
  warnings without raw ids or delivery metadata.
- L2/L2d prompt contracts require thread-reference context to constrain `你`
  resolution.
- Internal monologue residue cannot preserve ambiguous group-review subject
  inversion as fresh fact.
- Focused tests, integration tests, static greps, and one inspected live LLM
  regression pass.
- The final live LLM artifact shows no "Lamp compared Kazusa to a cat" premise.
- Independent code review is recorded with no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Neutral digest becomes too dry for cognition | Keep visible transcript, participant context, and summary; do not remove character judgment from L2 | Live LLM regression inspects L2 and route |
| Thread warnings suppress legitimate direct replies | Warnings only describe ambiguous rows; L2 still decides action | Direct-address source tests and self-cognition integration tests |
| Local model ignores JSON thread context | Render a human-readable thread section before visible context | Source-packet rendering tests and live LLM trace |
| Residue still loops false premises | Add source-reliability note and residue prompt/test coverage | Residue recorder tests and after-fix review artifact |
| Prompt edits broaden blast radius | Static prompt contract tests and limited file change surface | `test_cognition_prompt_contract_text.py` plus independent review |

## Execution Evidence

- Plan drafted: 2026-06-19.
- Independent plan review recorded: 2026-06-19. Addressed issues are listed in
  `Independent Plan Review`; outcome is no remaining plan-review blockers, but
  implementation remains unapproved while status is `draft`.
- Root investigation artifact:
  `test_artifacts/llm_reviews/kazusa_cat_failure_review.md`.
- Stage 1 test compile check: `venv\Scripts\python -m py_compile` passed for
  the edited focused test files:
  `tests\test_reflection_cycle_group_scene_digest.py`,
  `tests\test_self_cognition_group_review_source.py`,
  `tests\test_self_cognition_group_review_participant_context.py`,
  `tests\test_cognition_prompt_contract_text.py`,
  `tests\test_internal_monologue_residue_recorder.py`,
  `tests\test_internal_monologue_residue_prompt_boundaries.py`, and
  `tests\test_self_cognition_response_sensitivity_live_llm.py`.
- Stage 1 focused baseline command:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_group_review_participant_context.py tests\test_cognition_prompt_contract_text.py tests\test_internal_monologue_residue_recorder.py tests\test_internal_monologue_residue_prompt_boundaries.py -q`.
  Result before production implementation: 60 passed, 22 failed. Expected red
  failures cover the new neutral digest prompt/validation contract, cat
  side-thread prompt fixture, neutral group-review source packet, bounded
  `thread_reference_context`, L2/L2d prompt grounding rules, and residue
  reliability-note contract.
- Stage 1 baseline refresh after updating a stale L3 content-plan static test:
  same focused command returned 61 passed, 21 failed. Remaining failures are
  scoped to this plan's expected red tests: neutral group digest,
  source-packet/thread-reference context, L2/L2d thread grounding, and residue
  reliability-note handling.
- Stage 1 also corrected one test seam in
  `tests\test_self_cognition_group_review_source.py`: the no-schedule unit test
  now patches `collect_commitment_due_cognition_cases`, matching the current
  collector call path and preventing accidental MongoDB access.
- Baseline note: the unrelated stale static prompt-contract check in
  `tests\test_cognition_prompt_contract_text.py::test_l3_content_plan_scope_preserves_complete_plan_deliverables`
  was updated to the current L3 content-plan contract and passed as an isolated
  test before production implementation.
- Stage 2 production-code worker:
  native subagent `019edfc4-a08b-7f31-a706-ce4ac680868a` changed only the
  planned production/doc files and reported no live LLM run. Parent reviewed
  the visible changes, made a small formatting cleanup in
  `group_scene_digest.py` and `recorder.py`, and took over verification.
- Stage 2 syntax gate:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py`
  passed.
- Stage 2 focused deterministic verification:
  `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_group_review_participant_context.py tests\test_cognition_prompt_contract_text.py tests\test_internal_monologue_residue_recorder.py tests\test_internal_monologue_residue_prompt_boundaries.py -q`
  returned 82 passed, with only the existing `.pytest_cache` access warning.
- Stage 3 syntax gate after L2/L2d live-test prompt tightening:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py src\kazusa_ai_chatbot\internal_monologue_residue\models.py src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py`
  passed.
- Stage 3 focused deterministic regression batch:
  `venv\Scripts\python -m pytest -p no:cacheprovider --basetemp=tmp_pytest_stage3 tests\test_reflection_cycle_group_scene_digest.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_group_review_participant_context.py tests\test_cognition_prompt_contract_text.py tests\test_internal_monologue_residue_recorder.py tests\test_internal_monologue_residue_prompt_boundaries.py tests\test_reflection_cycle_stage1c_worker.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py -q`
  returned 190 passed.
- Stage 3 static greps:
  `rg -n "我刚看到群里刚刚发生的一段现场|第一人称观察资料|我（说话人）|我最后发言后" src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\self_cognition src\kazusa_ai_chatbot\cognition_chain_core`,
  `rg -n "development plan|bugfix plan|implementation plan|delivery_target|platform_message_id|platform_user_id|global_user_id" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\projection.py`,
  and
  `rg -n "group_scene_digest" src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py`
  all returned no matches.
- Stage 3 live LLM regression:
  `venv\Scripts\python -m pytest -p no:cacheprovider --basetemp=tmp_pytest_live3 -m live_llm tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_cat_side_thread_subject_boundary -q -s`
  passed as a single inspected case. Passing trace:
  `test_artifacts/llm_traces/self_cognition_cat_side_thread_subject_boundary_live_llm__group_activity_window_scope_11afa3456af9_2026-06-18T05_15_00_00_00_2026-06-18T05_30_00_00_00__20260619T131713194113Z.json`.
  Manual review: source packet contained `thread_reference_context`,
  generated L2 text explicitly said the cat/hair line was about the touched
  person and "跟我没关系", `action_specs=[]`, and no inspected generated field
  claimed Kazusa was compared to a cat or that Kazusa's hair was described.
- Stage 3 live review artifact:
  `test_artifacts/llm_reviews/kazusa_cat_failure_after_fix_review.md`.
  It records that an earlier live run with the corrected source contract was a
  deterministic-gate false positive: the regex `灯.{0,20}把我` matched
  `灯的头...完全没把我` even though the trace did not contain the target
  cat/hair subject inversion. The live test was tightened to catch
  `灯...把我...比作` and other high-confidence subject-inversion claims instead
  of failing on unrelated wording.
- Stage 4 independent code review:
  native review subagent `019ee00a-a208-7461-a10e-740560ded2f3` found that
  digest validation allowed neutral active-character ownership, L2 prompt
  text had a hard-coded character name, and prompt/tests were overfit to the
  cat/hair fixture. Parent fixed those findings, then reran focused
  deterministic verification.
- Stage 4 user review follow-up:
  user rejected the attempted `# 输入读取说明` replacement and asked for the
  entire touched prompts to follow the local LLM rule without dedicated input
  sections or negative subject constraints. Parent rewrote the touched L2a,
  L2b, L2c1, L2d, group-scene digest, and residue recorder prompts so
  decision-critical fields are explained in source/procedure sections.
- Stage 4 source-guidance correction:
  `thread_reference_context.guidance` now uses positive source-priority
  wording:
  `二人称归属按同一行明确地址和可见线程读取；缺少同一行当前角色指向时，保留为侧线/未定对象。`
  Parent also fixed `_is_group_review_thread_reference_context` to compare
  against `THREAD_REFERENCE_GUIDANCE` instead of duplicating the stale literal.
- Stage 4 syntax gate:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\group_review_participant_context.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\cognition_chain_core\action_selection_prompt.py src\kazusa_ai_chatbot\internal_monologue_residue\recorder.py`
  passed.
- Stage 4 focused deterministic regression batch:
  `venv\Scripts\python -m pytest -p no:cacheprovider --basetemp=tmp_pytest_stage5 tests\test_reflection_cycle_group_scene_digest.py tests\test_self_cognition_group_review_source.py tests\test_self_cognition_group_review_participant_context.py tests\test_cognition_prompt_contract_text.py tests\test_internal_monologue_residue_recorder.py tests\test_internal_monologue_residue_prompt_boundaries.py tests\test_reflection_cycle_stage1c_worker.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py -q`
  returned 196 passed.
- Stage 4 static greps:
  the plan's three static grep commands returned no matches. Additional greps
  for `# 输入格式|# 输入读取说明|# 输入含义` in the touched runtime prompts,
  hard-coded character/cat/hair prompt anchors, and the old negative
  second-person guidance in the touched runtime path also returned no matches.
- Stage 4 live LLM regression:
  `venv\Scripts\python -m pytest -p no:cacheprovider --basetemp=tmp_pytest_live5 -m live_llm tests\test_self_cognition_response_sensitivity_live_llm.py::test_live_self_cognition_cat_side_thread_subject_boundary -q -s`
  passed as a single inspected case. Passing trace:
  `test_artifacts/llm_traces/self_cognition_cat_side_thread_subject_boundary_live_llm__group_activity_window_scope_11afa3456af9_2026-06-18T05_15_00_00_00_2026-06-18T05_30_00_00_00__20260619T135544595338Z.json`.
  Manual review: no inspected generated field claimed Kazusa was compared to a
  cat or that Kazusa's hair was described. The model selected one visible
  action, but its L2d target and final dialog were grounded in Snow's direct
  pig-emoji cue. Residual risk: L2a still mentions the Lamp cat/hair side
  thread near offended-feeling wording; the later judgment, action target, and
  final dialog keep the visible response grounded in the direct cue.
