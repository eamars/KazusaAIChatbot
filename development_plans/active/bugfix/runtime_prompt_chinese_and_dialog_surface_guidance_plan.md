# Runtime prompt Chinese and dialog surface guidance plan

## Summary

- Goal: convert every runtime prompt added or modified relative to
  `origin/main` to Chinese ordinary instructions and restore an organic
  text-dialog ownership boundary that discourages action narration without
  rejecting, repairing, filtering, or failing it.
- Plan class: large.
- Status: in_progress.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`, and
  `no-prepost-user-input`.
- Overall cutover strategy: bigbang prompt-language and ownership correction on
  the current branch; no dual-language prompt or compatibility renderer.
- Highest-risk areas: semantic translation drift, local-model instruction
  salience, action-description overcorrection, prompt-contract tests tied to
  English wording, and accidental changes to routing or output schemas.
- Acceptance criteria: all affected prompts use Chinese ordinary instructions;
  schema keys and enums remain exact; upstream text planning contains no
  intentional gesture/staging guidance; dialog organically favors spoken text;
  action narration remains pass-through variation; focused verification passes;
  the full frozen 20+20 suite is not run.

## Context

The fresh 40-turn replay at `6c81d43a` produced structurally marked action or
stage narration in all 38 turns that reached the dialog generator. Protected
traces show that content planning and dialog generation explicitly advertise
action description in plain, bracketed, first-person, and third-person forms.
The surface verifier explicitly accepts it, so no repair occurs.

The user's clarified direction distinguishes runtime policy from execution
review policy. Runtime prompts should organically discourage action narration;
if the local model still emits it, the response continues without rejection,
repair, deterministic filtering, or failure classification. The earlier phrase
“let action description pass” was an instruction to Codex's review workflow,
not content to place in Kazusa's prompts.

The user also selected branch-wide option A: every runtime prompt added or
modified since the production baseline must use Chinese ordinary instructions.
Machine-facing schema keys, enum values, IDs, code, URLs, model labels, and
quoted source text retain their exact language.

This plan supersedes the action-description permission clauses in
`cognition_core_v2_live_character_judgment_rebalance_plan.md` and the hard
action-description rejection clauses in
`dialog_visible_speech_and_semantic_fidelity_bugfix_plan.md`. It preserves the
remaining character-judgment, contradiction, role-direction, execution-truth,
terminal-visual, and one-repair contracts.

## Mandatory Skills

- `development-plan`: lifecycle, evidence, and scope control.
- `local-llm-architecture`: prompt minimality, semantic ownership, and local
  model attention constraints.
- `py-style`: Python source review and edits.
- `cjk-safety`: safe CJK prompt literals and immediate syntax checks.
- `test-style-and-execution`: focused deterministic and individually inspected
  real-LLM checks.
- `debug-llm`: agent-authored human-readable evidence for any real-LLM run.
- `no-prepost-user-input`: no deterministic semantic action-description
  classifier or output rewrite.

## Mandatory Rules

- Use Chinese for ordinary prompt instructions, explanations, procedures,
  headings, and generated free-text directions.
- Preserve JSON keys, enum values, role handles, IDs, code, URLs, commands,
  model labels, capability names, and quoted source text exactly when their
  contracts require it.
- Use triple-single-quoted prompt constants and `.format(...)` for stable
  prompt rendering. Keep per-run facts in human messages.
- Use `当前角色` and `当前用户` in Chinese free text. Keep `self`,
  `current_user`, `other`, and `none` only where the structured role contract
  requires those exact enum tokens.
- Text content planning owns visible meaning, interaction development, active
  boundaries, role direction, and execution truth. It does not intentionally
  request gestures, staging, body movement, or action-description forms.
- Style planning owns wording, sentence shape, rhythm, hesitation, and
  punctuation. Emotional and character traits are expressed through language.
- Dialog renders what the character naturally says or types. It softly turns
  action tendencies into wording and cadence without mechanical repeated
  prohibitions.
- Action narration is not a fatal, unacceptable, verifier, repair, parser, or
  deterministic filtering condition. If the model emits it, preserve the
  generated output and continue.
- The surface verifier retains only false character-brain capability-execution
  truth. Semantic and role verifiers retain their existing hard classes.
- The visual stage remains a terminal image-directive owner and may describe
  physical composition because its output has no dialog consumer.
- Preserve all structured output schemas, exact fields, cardinality bounds,
  model routes, sampling parameters, call counts, and retry limits.
- The full frozen 20 group plus 20 private test suite is forbidden for this
  change.
- Real-LLM checks, if run, execute individually and are inspected before the
  next case.
- No captured corpus phrase, concrete character name, user id, channel id, or
  expected dialog is inserted into a runtime prompt.
- No subagent performs implementation. The user previously directed the root
  agent to carry the implementation forward; the root agent owns edits,
  verification, review, and reporting.
- After context compaction or a major checklist sign-off, reread this entire
  plan before continuing.
- Before lifecycle completion or merge, perform the final review gate and
  record its availability and outcome in Execution Evidence.

## Must Do

1. Inventory runtime prompt assignments added or modified relative to
   `origin/main`, including composed prompt families whose changed fragments
   make the final prompt branch-modified.
2. Translate all affected ordinary instructions to Chinese without changing
   schemas, enums, role direction, capability semantics, output contracts, or
   model-call behavior.
3. Correct the content, style, dialog, repair, and surface-verifier action
   narration policy according to the Mandatory Rules.
4. Update prompt-contract tests that assert English prose or the retired action
   permission/rejection policies.
5. Reconcile active plan and subsystem documentation wording with this plan.
6. Run CJK syntax checks immediately after each Python edit batch.
7. Run prompt rendering, static prompt-language audit, and focused deterministic
   tests only.
8. Run at most a small set of individually inspected real-LLM cases if needed
   to expose the post-change prompt behavior; author a readable review before
   relying on those results.
9. Perform a final root-agent code review against every user direction before
   reporting completion.

## Deferred

- Full frozen 20+20 replay.
- Model, route, sampling, context-budget, graph, schema, persistence, database,
  action registry, resolver, relevance algorithm, or adapter redesign.
- Character profile, affinity, relationship, memory, or state retuning.
- Deterministic action-description detection, parenthesis stripping, content
  rewriting, or keyword suppression.
- New verifier, repair stage, retry, fallback renderer, feature flag, prompt
  compatibility path, or dual-language prompt.
- General translation of untouched production-baseline prompts.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Branch-modified prompts | replace | Chinese ordinary instructions; exact machine tokens remain. |
| Text surface | replace | Plan speech meaning and interaction beats without intentional staging guidance. |
| Dialog generator | replace | Organic Chinese rendering contract that favors spoken/typed chat. |
| Repair | replace | Chinese hard-error repair contract; action narration stays outside repair. |
| Verifiers | replace | Chinese semantic/role/execution contracts; no action-narration judgment. |
| Tests | replace | Assert stable contracts and language boundary, not retired English wording. |
| Documentation | reconcile | One canonical action-narration policy from this plan. |

No old/new prompt selection, compatibility alias, fallback prompt, or rollout
flag is permitted.

## Target State

```text
Chinese cognition and surface information carriers
  -> content planning chooses meaningful, vivid speech beats
  -> style planning describes language and cadence
  -> dialog produces natural character speech
  -> semantic and role checks protect hard meaning direction
  -> surface check protects capability-execution truth
  -> action narration, if still generated, passes through as model variation
```

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Action narration | Softly discourage; never gate or repair | Matches the clarified instruction and avoids tuning around weak-model variation. |
| Upstream ownership | No intentional action-description guidance | Prevents dialog from receiving staging as required semantic content. |
| Prompt tone | Organic positive procedure | Avoids mechanical strengthened wording and instruction accretion. |
| Prompt language | Chinese ordinary instructions branch-wide | Matches the cognition-chain carrier and local-model language context. |
| Machine tokens | Preserve exact | Protects deterministic parsers and structured contracts. |
| Verification size | Focused only | The user forbids the expensive full 40-turn suite. |
| Visual prompt | Retain physical image directives | It is terminal and has no dialog consumer. |

## Change Surface

Production prompt owners:

- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
- `src/kazusa_ai_chatbot/consolidation/lane_router.py`
- `src/kazusa_ai_chatbot/consolidation/reflection.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`

Audit-only prompt owners already using Chinese ordinary instructions remain
unchanged unless the final audit finds English ordinary prose added by this
branch.

Tests and documentation may change only where directly coupled to these prompt
contracts or the superseded action-narration wording.

## Overdesign Guardrail

- Actual problem: branch-modified prompts mix English instructions and encode
  action narration as either encouraged output or a hard failure.
- Minimal change: translate the affected prompt contracts and correct ownership
  language while preserving code paths and schemas.
- Ownership boundaries: LLM prompts own semantic generation and review;
  deterministic code owns shape, limits, routing, and execution.
- Rejected complexity: filters, new fields, flags, agents, verifiers, retries,
  fallback prompts, dual-language paths, and model changes.
- Evidence threshold: a later observed hard failure outside current verifier
  ownership plus user approval is required before adding another control.

## Agent Autonomy Boundaries

- The root agent may choose Chinese phrasing that preserves the exact semantic
  contract and organic tone.
- The root agent may not change schemas, call graphs, routes, sampling,
  persistence, or action capability semantics.
- Changes outside the named prompt, test, plan, and documentation surfaces
  require user clarification before implementation.
- The root agent may remove retired English/action-policy prompt text directly.
- The root agent may not add compatibility paths or unrelated cleanup.

## Implementation Order

1. Add/update focused language and action-ownership prompt tests; record the
   current expected failures.
2. Translate V2 cognition, action, workspace, and surface prompt families.
3. Translate dialog generation, repair, and verifier prompt families while
   applying the clarified action-narration policy.
4. Translate branch-modified consolidation and relevance prompt families.
5. Audit already-Chinese modified prompts and all composed final prompts.
6. Reconcile tests, active plan references, and subsystem documentation.
7. Run syntax, rendering, static audit, and focused deterministic tests.
8. Run limited individual real-LLM checks only if needed and author the review.
9. Perform the final root-agent review and record evidence.

## Execution Model

The user-directed fallback execution model applies: the root agent performs all
implementation, test, verification, and review work without subagents. The
root agent establishes the focused test contract before production edits and
records the independent-review gate as unavailable under the active no-subagent
instruction rather than claiming an independent approval.

## Progress Checklist

- [x] Checkpoint A - prompt inventory and focused test contract complete.
- [x] Checkpoint B - V2 cognition/action/workspace/surface prompts translated.
- [x] Checkpoint C - dialog generation/repair/verifier prompts corrected.
- [x] Checkpoint D - consolidation and relevance prompts translated.
- [x] Checkpoint E - tests and documentation reconciled.
- [x] Checkpoint F - focused verification and final root review complete.
- [x] Checkpoint G - independent review availability and lifecycle status recorded.

## Verification

### Static and syntax

- `venv\Scripts\python.exe -m py_compile <each changed Python prompt file>`
- Prompt rendering for every `.format(...)` template succeeds.
- Branch-modified runtime prompts contain Chinese ordinary instructions; exact
  schema/enum tokens remain.
- No runtime prompt advertises action description as an output form or treats
  it as a verifier/repair failure.
- No full 20+20 replay command is run.

### Focused tests

- Prompt-contract tests for cognition, action planning, surface, dialog,
  consolidation routing, and relevance pass in targeted files only.
- Existing structural tests confirm exact output fields and role/action enums.
- Any real-LLM case runs alone and receives an agent-authored readable review.

## Independent Code Review

An independent subagent review is unavailable under the active user and system
no-subagent instructions. The root agent performs a separate final review pass
over the complete diff, verifies every user direction, records this limitation,
and leaves lifecycle completion pending if project policy requires independent
approval.

## Acceptance Criteria

- Every branch-added or branch-modified runtime prompt uses Chinese ordinary
  instructions.
- Action narration is organically discouraged in text planning and rendering.
- Upstream text planning supplies no intentional action-description guidance.
- Action narration remains pass-through behavior and never triggers verifier,
  repair, deterministic filter, or test failure.
- Visual directives remain terminal and image-oriented.
- Structured fields, enums, routes, sampling, call counts, and retry bounds are
  unchanged.
- Focused checks pass without running the full 40-turn suite.
- The final report identifies verification performed, real-LLM evidence if any,
  independent-review availability, and residual model variability.

## Execution Evidence

- Prompt inventory covered all 13 modified production Python modules. The
  branch-added V2 prompts and every runtime prompt changed relative to the
  production baseline now use Chinese ordinary instructions while preserving
  exact schema, enum, role, route, and capability tokens.
- Text surface planning now owns speech meaning and interaction development;
  style planning owns language and cadence; dialog renders what the character
  naturally says or sends. The terminal visual prompt remains image-oriented.
- The dialog hard-error taxonomy remains contradiction within one response,
  direct conflict with current user input, unique role-direction reversal, and
  false capability-execution claims. Action narration has no verifier, repair,
  parser, filter, or fatal-error path.
- A string-stripped AST comparison against the pre-change `HEAD` checked all
  13 modified production Python files and reported `non_string_ast_diffs=none`.
  Therefore executable control flow, schemas, routes, call counts, retries,
  sampling configuration, and persistence behavior are unchanged.
- The anti-cheat scan reported `captured_case_material=none` for the frozen QQ
  identifiers, character-specific names, and captured examples. The scoped
  prompt-policy scan reported
  `action_narration_specific_prompt_terms=none` in modified runtime sources.
- `venv\Scripts\python.exe -m py_compile` succeeded for every modified Python
  production and test file. `git diff --check` succeeded.
- The consolidated focused deterministic suite passed with `151 passed,
  4 deselected`. It covered prompt language and rendering, semantic source
  ownership, action planning and authorization, V2 dependencies and facade
  integration, text/visual separation, dialog hard-error repair, relevance,
  consolidation routing, interaction style, preferences, and conversation
  progress.
- The four deselections are live-database integration cases. No real-LLM case
  and no frozen 20+20 replay was run, matching the user's verification scope.
- The root agent performed the complete implementation and final review. An
  independent subagent review was unavailable under the active no-subagent
  instruction. Plan status remains `in_progress` pending user review and
  sign-off.
