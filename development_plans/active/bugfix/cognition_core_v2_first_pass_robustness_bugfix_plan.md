# cognition core v2 first-pass robustness bugfix plan

Status: draft

Plan class: high_risk_migration

Cutover: staged bigbang per stage, no runtime flags

## Summary

- Goal: remove the six deterministic hard-failure paths in `cognition_core_v2`,
  make every bounded retry carry the information the local model needs to fix
  its output, and rebalance the per-stage completion-token and prompt-character
  budgets so the cognition stages stop failing closed on irreducible context.
- Plan class: `high_risk_migration` (production LLM generation limits, prompt
  contracts, and failure dispositions change across twelve routes).
- Status: `draft`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `python-venv`,
  `no-prepost-user-input`.
- Overall cutover strategy: seventeen staged bigbang changes. Each stage is one
  contract cutover with no compatibility path, no feature flag, and no dual
  code path. Rollback is a source revert of the stage.
- Highest-risk areas: goal-cognition prompt budget, appraisal provenance
  validation, relational-willingness semantic authority on retry, and the
  per-route completion-token reduction.
- Acceptance criteria: see `Acceptance Criteria`.

The plan's length budget is intentionally exceeded. The plan owner explicitly
authorized full procedural detail over compression.

## Context

Static analysis of `src/kazusa_ai_chatbot/cognition_core_v2` found fourteen
defects across four areas: first-pass generation grounding, evaluator feedback
quality, token budgeting, and failure recovery. The defects that abort a turn
are deterministic, not stochastic: they fire whenever the character identity
grows past roughly half its allowed size, or whenever an appraisal family
touches seven or more distinct entities.

Two defects were confirmed by executing module functions directly rather than
by reading:

1. `_validate_handles` in `semantic_appraisal.py` hardcodes an upper bound of
   `8` for every handle list, including the two provenance lists that
   deterministic code derives and accumulates across up to eight micro items.
   A synthetic eight-goal `goal_threat_outcome` family with one proposition and
   one delta per item fails merged validation at item 7 with
   `selected roles handles must contain between 0 and 8 items`.
2. `character_identity` and `character_constraints` are injected into the
   appraisal state partition and the goal-cognition semantic context, but are
   absent from both reduction orders. `TEXT_LIMIT_BY_PATH` in
   `character_identity_growth/models.py` allows `backstory` 6000,
   `description` 2400, `self_image.self_concept` 2400, and five personality
   fields at 1200-1600 each, so the `moral_identity`, `existential_drive`, and
   `goal_cognition` identity partitions can reach roughly 20,000 characters
   against an 8,000-character appraisal cap and a 21,700-character goal payload
   budget.

The existing maximum-shape budget tests in
`tests/test_cognition_core_v2_prompt_budget_continuity.py` max out evidence,
constraints, scene context, and state cardinality, but pass the ~500-character
stub identity from `tests/cognition_core_v2_test_helpers.py`. The largest
irreducible prompt block in production is the one block the budget tests do not
exercise.

One pre-existing deterministic test assertion is already arithmetically false
and must be recorded as a baseline before any edit:
`tests/test_cognition_core_v2_live_character_judgment.py::test_live_character_prompts_fit_local_model_attention_caps`
asserts `len(GOAL_COGNITION_PROMPT) <= 2200` and
`len(REQUIRED_SELECTION_GOAL_PROMPT) <= 2400`, while the actual lengths are
2300 and 2501.

Adjacent improvement areas intentionally left for later plans:

- `nodes/dialog_agent.py` `DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS = 50000` combined
  with a 25,000-token completion reservation on the dialog route can exceed a
  50k context window on backends that reject rather than clamp.
- The dialog verifier chain, `state_reducers.py` terminal-transition guards,
  and `emotion_derivation.py` were not audited.
- `project_model_visible_percepts` percept-driven surface overflow.

## Mandatory Skills

Execution must read and apply these skills in full before the corresponding
work:

- `.agents/skills/development-plan/SKILL.md` plus
  `references/plan_contract.md`, `references/execution_gates.md`, and
  `references/cutover_policy.md`: load before touching this plan or executing
  any stage.
- `.agents/skills/local-llm-architecture/SKILL.md`: load before editing any
  prompt constant, prompt payload, model call, or context budget.
- `.agents/skills/py-style/SKILL.md` plus both
  `references/positive_constraints.md` and
  `references/negative_constraints.md`: load before editing any `.py` file.
- `.agents/skills/cjk-safety/SKILL.md`: load before editing any Python file
  containing Chinese prompt text.
- `.agents/skills/test-style-and-execution/SKILL.md`: load before adding,
  changing, or running any test.
- `.agents/skills/python-venv/SKILL.md`: load before running Python.
- `.agents/skills/no-prepost-user-input/SKILL.md`: load before editing any
  prompt that shapes user-visible wording.

After any automatic context compaction, reread this entire plan and every
mandatory skill before continuing implementation, verification, handoff, or
final reporting. After signing off any major progress-checklist stage, reread
this entire plan before starting the next stage.

## Mandatory Rules

Project and skill rules, copied here because execution context may compact:

- Use `venv\Scripts\python.exe` for every Python and pytest invocation. Do not
  use a global interpreter. If `venv\` is absent, create it at the project root
  per the `python-venv` skill before running anything.
- Quote every path that contains spaces. Prefer `-LiteralPath` in PowerShell.
- Run deterministic tests in batches. Run live LLM tests one at a time and
  inspect the emitted artifact before starting the next one.
- Every prompt constant is a triple-single-quoted string `'''...'''`. Do not
  introduce triple-double-quoted prompt literals, f-string prompt literals,
  f-string `SystemMessage` content, or `.replace(...)` prompt rendering.
- Any Python string whose content contains Chinese typographic quotes must use
  single-quote delimiters. Do not retype existing Chinese prompt text through a
  file-write tool when the exact bytes already exist in the repository; extract
  and rewrite those bytes with a Python script instead.
- Run `py_compile` on every edited Python file that contains Chinese text
  immediately after editing it, before running any test.
- Keep the `SystemMessage` static for the Python session on every route. All
  per-run facts belong in the `HumanMessage`.
- Do not introduce a new term, acronym, module name, stage name, or
  development-process concept into a runtime prompt. Prompt vocabulary must be
  grounded in the model-facing input or defined in the same prompt.
- Do not mention this plan, its stages, or any migration language in source
  code, comments, or prompts.
- Every raw model response passes through
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before contract
  evaluation. Do not add a second parser.
- Deterministic code owns validation, budgeting, handle authority, retry
  counting, and dispositions. The LLM owns semantic judgment. Do not move a
  routing, permission, feasibility, delivery, or retry decision into a prompt.
- Deterministic code may drop or truncate model-authored structure only where
  this plan names the exact rule. It must never invent, paraphrase, default, or
  copy a semantic value the model did not author.
- Named constants only. Do not introduce a bare numeric literal for a budget,
  cap, floor, or limit.
- Search for an existing helper before adding one. `middle_truncate_text`
  already exists twice in this package and must be deduplicated rather than
  copied a third time.
- Do not add a JSON-repair LLM call, verifier LLM, classifier LLM, feature
  flag, compatibility shim, alias field, or extra healthy-path model call.
- Do not raise any per-route `max_completion_tokens` above its current value.
- Preserve unrelated user changes in the worktree. `docs/CODING_AGENT_CAPABILITY_ASSESSMENT.md`
  and `src/scripts/count_code.py` are untracked and must remain untouched.
- Before final completion, lifecycle status change, merge, or sign-off, run the
  `Independent Code Review` gate and record the result in `Execution Evidence`.
- `Execution Model` uses parent-led native subagent execution. Do not fall back
  to single-agent execution without explicit user approval.

## Must Do

- Record the pre-change deterministic baseline for every named test file,
  including the two already-false prompt-length assertions.
- Lower the cognition-owned `max_completion_tokens` defaults and add an
  explicit per-call timeout to every cognition-owned `LLMCallConfig`.
- Change every cognition prompt cap to count the system prompt plus the dynamic
  payload, and raise the appraisal and goal caps to their new values.
- Make `character_identity`, `character_constraints`, and `scene_context`
  reducible under fixed floors in both the appraisal and goal budget fitters.
- Bound the two derived appraisal provenance handle lists by the size of their
  permitted sets instead of the hardcoded `8`.
- Return the accepted appraisal prefix when one micro item exhausts its
  attempts, instead of discarding the family.
- Give the generic goal producer the `accepted_degraded` disposition its own
  policy table already declares.
- Drop stale bids whose persistent goal no longer exists before workspace
  collapse instead of raising `internal_invariant`.
- Retain the stage system prompt and pass the exact validation error on every
  surface-stage repair.
- Retain the initial goal system prompt on every goal regeneration so the
  relational-willingness ordering rules survive the retry.
- Put the offending value and the permitted allowlist into every appraisal
  handle, path, and text validation error.
- Reduce appraisal handle authority in lockstep with appraisal state-row
  removal.
- Add an exact output skeleton to the appraisal prompt and the generic goal
  prompt.
- Drop invalid planner rows individually and retry only when the model proposed
  rows and none survived.
- Add a bounded reduction pass to the surface stage before its typed cap
  failure.
- Correct the appraisal attempt-limit policy row and its fixture.
- Delete the dead Chinese-keyed `roles` projection stub and its dead filter.
- Bind the `outcome_pending` proposition subject to its four permitted entity
  kinds.
- Deduplicate `middle_truncate_text` into `prompt_budget.py`.
- Update `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, `docs/HOWTO.md`,
  and every affected test assertion in the same change as the code.
- Run every command in `Verification` and record the result.
- Complete the independent code review gate.

## Deferred

- Do not change `DEFAULT_LLM_MAX_COMPLETION_TOKENS`. Non-cognition consumers
  (`coding_agent`, `consolidation`, `rag`, `complex_task_resolver`,
  `web_search`, `dialog_generator`, `json_repair`) keep 25000.
- Do not change any `*_BASE_URL`, `*_API_KEY`, `*_MODEL`, or `.env` value.
- Do not change `COGNITION_LLM_CHARACTER_CARRYOVER_MAX_COMPLETION_TOKENS`; it
  is already bounded at 8192.
- Do not change `DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS`, the dialog verifier chain,
  or any file under `nodes/dialog_agent.py`.
- Do not change any runtime attempt count: `V2_MODEL_TOTAL_ATTEMPTS`,
  `V2_VERIFIER_TOTAL_ATTEMPTS`, `GOAL_COGNITION_ATTEMPT_LIMIT`,
  `SEMANTIC_APPRAISAL_ATTEMPT_LIMIT` (2), `SEMANTIC_APPRAISAL_ITEM_LIMIT`,
  `ACTION_PLANNING_ATTEMPT_LIMIT`, `ACTION_AUTHORIZATION_ATTEMPT_LIMIT`,
  `SURFACE_STAGE_ATTEMPT_LIMIT`, `WORKSPACE_COLLAPSE_ATTEMPT_LIMIT`, and
  `COGNITION_SAFE_RETRY_LIMIT` all keep their current runtime values.
  Correcting the `semantic_appraisal` policy-table row from 3 to 2 aligns the
  declared record with the unchanged runtime limit and is in scope.
- Do not add lockstep handle reduction to goal cognition; its handle authority
  comes from `_role_bindings`, whose summaries are never reduced.
- Do not truncate `episode.visible_percepts` in the surface reduction pass.
  Percept-driven surface overflow keeps the existing degraded-surface outcome.
- Do not redesign `_compact_permitted_delta_path_domains` into an explicit path
  allowlist.
- Do not change `state_reducers.py`, `state_models.py`, `emotion_derivation.py`,
  `transition_guards.py`, `character_carryover.py`, or any persisted state
  schema.
- Do not add a maximum-size-identity parameterization to any test outside the
  files named in `Change Surface`.
- Do not fix unrelated failing tests discovered during baseline capture. Report
  them and ask.

## Cutover Policy

Each stage is one bigbang contract cutover:

- The old cap accounting, the old provenance bound, the old repair prompts, and
  the old dispositions have no runtime fallback and no compatibility path.
- `_SURFACE_REPAIR_PROMPT`, `GOAL_COGNITION_REPAIR_PROMPT`, and
  `REQUIRED_SELECTION_GOAL_REPAIR_PROMPT` are deleted, not aliased.
- Production code, prompt text, README, `docs/HOWTO.md`, and test assertions
  move together inside the same stage.
- Rollback is a source revert of the stage commit. There is no runtime toggle,
  environment switch, or dual code path.
- No stage requires a data migration. No persisted document, queue envelope,
  database index, or adapter contract changes.

## Target State

```text
one cognition turn
  -> every cognition-owned model call carries an explicit timeout and a
     right-sized completion cap
  -> every cognition prompt cap counts system + dynamic characters
  -> appraisal packet
       identity/constraints reduce under fixed floors before state rows drop
       state-row removal drops the matching handle from question authority
       derived provenance lists are bounded by their permitted-set size
       one exhausted micro item returns the accepted prefix
       still-irreducible packet omits that family with typed diagnostics
  -> goal cognition packet
       supplemental -> scene -> constraints -> identity -> evidence floor
       regeneration reuses the initial system prompt plus repair_feedback
       exhausted generic attempts may deliver one handle-degraded bid
       still-unrecoverable required branch keeps the typed safe-retry failure
  -> workspace collapse
       bids whose persistent goal no longer exists are dropped with a warning
       cap or contract exhaustion keeps the first complete registry-order bid
  -> action planning
       invalid rows drop individually; all-dropped triggers one repair
       exhaustion returns the empty blocked proposal
  -> surface stages
       bounded reduction runs before the typed cap failure
       repair reuses the stage system prompt and carries the exact error
       exhaustion returns the validated degraded surface
```

No cognition stage aborts a turn because required context is irreducible under
a cap that deterministic code can still reduce.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Completion-token scope | Lower only cognition-owned route defaults; leave `DEFAULT_LLM_MAX_COMPLETION_TOKENS` at 25000 | Plan owner authorized a cognition-scoped reduction. Non-cognition consumers are unaudited. |
| Completion-token values | Three code-owned defaults: appraisal 2048, structured 1024, semantic 8192 | Each value exceeds the stage's own contract maximum output. 8192 restores the historical project default for prose-heavy owners. |
| Per-call timeout | One env-backed `COGNITION_STAGE_TIMEOUT_SECONDS`, default 120, bounds 10.0-600.0 | A single bound is sufficient; per-stage timeouts would be speculative flexibility. |
| Cap accounting | Every cognition cap counts system prompt plus dynamic payload | Three of six caps silently excluded a 0.7k-6.8k-character system prompt. One rule removes the inconsistency. |
| Appraisal cap value | 20000 total | Payload budget ~17.1k, 2.1x today. Worst-case 20000 chars + 2048 completion is 44% of a 50k window. |
| Goal cap value | 36000 total | Payload budget ~33.7k. Fits a maximum identity plus constraints plus scene plus floored evidence, with 12% window margin at 8192 completion. |
| Identity reduction | Fixed nine-step field truncation order; `boundaries`, `mbti`, and identity scalars never reduce | Boundary profile and name/age/gender are permission- and role-relevant; backstory prose is not. |
| Identity reduction position | Appraisal reduces identity before dropping state rows; goal cognition reduces identity last | Appraisal reasons about entities and evidence; goal cognition reasons about character judgment. |
| Provenance bound | Bound the two derived lists by `len(allowed)` | A derived duplicate-free subset of the permitted set can never exceed the permitted set. No magic number. |
| Appraisal item exhaustion | Return the accepted prefix; re-raise only when nothing was accepted | The loop already treats empty and exact-repeat items as bounded termination. Contract exhaustion joins that class. |
| Generic goal degradation | Drop invalid handles from `evidence_handles` and `target_role_handles` only | The README already documents this disposition for selection turns. `relational_willingness.evidence_handles` is a permission gate and stays fatal. |
| Stale goal bids | Drop the bid in the facade before collapse | A goal terminalized by this turn's own appraisal makes its branch bid stale by construction. |
| Goal repair prompt | Delete both repair system prompts; reuse the initial system prompt and move structural instructions into `repair_feedback` | The repair prompts discarded the relational-willingness ordering rules while retaining the schema, which can relax a boundary gate on attempts 2-3. |
| Surface repair prompt | Delete `_SURFACE_REPAIR_PROMPT`; reuse the stage system prompt and add `contract_error` | The generic repair prompt removed the stage output contract from the conversation, making attempts 2-3 near-random. |
| Planner row containment | Drop invalid rows; raise only when rows were proposed and none survived | Satisfies both README statements: individual row dropping, and a contract error that still earns one repair. |
| Surface reduction | Tail-drop `supporting_bids` then `semantic_affect`, then raise | Bounded, reviewable, and does not touch percept content. |
| Appraisal attempt policy | Add `V2_APPRAISAL_TOTAL_ATTEMPTS = 2` and use it in both the policy table and the module | The policy table declared 3 while the module used 2, so `validate_v2_attempt_record` would reject a truthful appraisal record. |
| Dead `roles` projection | Delete the Chinese-keyed stub and the handle-keyed filter that can never match it | The only reader is a filter whose key domains cannot intersect. |

## Contracts And Data Shapes

### New configuration constants

`src/kazusa_ai_chatbot/config.py`:

```python
SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS = 2_048
COGNITION_STRUCTURED_DEFAULT_MAX_COMPLETION_TOKENS = 1_024
COGNITION_SEMANTIC_DEFAULT_MAX_COMPLETION_TOKENS = 8_192
SURFACE_CONTENT_DEFAULT_MAX_COMPLETION_TOKENS = 8_192
SURFACE_PREFERENCE_DEFAULT_MAX_COMPLETION_TOKENS = 4_096
SURFACE_VISUAL_DEFAULT_MAX_COMPLETION_TOKENS = 2_048
COGNITION_STAGE_TIMEOUT_SECONDS = _bounded_float_from_env(
    "COGNITION_STAGE_TIMEOUT_SECONDS",
    "120",
    minimum=10.0,
    maximum=600.0,
)
```

`SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` keeps its existing name and
becomes `2_048`. No new `*_MAX_COMPLETION_TOKENS` environment variable is
added; the twelve existing route variables keep their names and only their
code-owned defaults change.

Route-to-default assignment:

| Route env prefix | New default constant |
|---|---|
| `COGNITION_LLM_APPRAISAL_EVENT_AGENCY` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_APPRAISAL_MORAL_IDENTITY` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE` | `SEMANTIC_APPRAISAL_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` | `COGNITION_SEMANTIC_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_GOAL_ACTIVE_BRANCH` | `COGNITION_SEMANTIC_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_ACTION_PLANNING` | `COGNITION_SEMANTIC_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_WORKSPACE_COLLAPSE` | `COGNITION_STRUCTURED_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_ACTION_AUTHORIZATION` | `COGNITION_STRUCTURED_DEFAULT_MAX_COMPLETION_TOKENS` |
| `COGNITION_LLM_RESOLVER_AUTHORIZATION` | `COGNITION_STRUCTURED_DEFAULT_MAX_COMPLETION_TOKENS` |

### Prompt cap contract

Every cap below counts the system prompt plus the dynamic payload:

| Constant | Before | Before accounting | After | After accounting |
|---|---:|---|---:|---|
| `SEMANTIC_APPRAISAL_PROMPT_CAP` | 8000 | human only | 20000 | system + human |
| `SEMANTIC_APPRAISAL_REPAIR_PROMPT_CAP` | 10000 | dynamic only | 24000 | system + all dynamic |
| `GOAL_COGNITION_PROMPT_CAP` | 24000 | system + human | 36000 | system + human |
| `ACTION_PLANNING_PROMPT_CAP` | 24000 | human only | 32000 | system + human |
| `ACTION_AUTHORIZATION_PROMPT_CAP` | 16000 | human only | 20000 | system + human |
| `RESOLVER_AUTHORIZATION_PROMPT_CAP` | 24000 | human only | 24000 | system + human |
| `SURFACE_STAGE_PROMPT_CAP` | 24000 | human only | 32000 | system + human |
| `SURFACE_STAGE_REPAIR_PROMPT_CAP` | 24000 | human only | 32000 | system + human |
| `WORKSPACE_COLLAPSE_PROMPT_CAP` | 24000 | system + human | 24000 | system + human |

### Shared reduction helpers

`src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py` gains one public
truncation function and three single-step reducers. Each reducer mutates its
argument in place and returns whether one bounded reduction step was applied.
Each returns `False` only when the projection is already at its floor.

```python
IDENTITY_TEXT_FLOORS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("core", "backstory"), 600),
    (("core", "description"), 400),
    (("self_image", "self_concept"), 400),
    (("personality", "quirks"), 300),
    (("personality", "taboos"), 300),
    (("personality", "logic"), 300),
    (("personality", "tempo"), 300),
    (("personality", "defense"), 300),
)
MAX_REDUCED_GROWTH_EDGES = 2
MAX_REDUCED_STANDARD_DESCRIPTION_CHARS = 120
SCENE_TEXT_FLOORS: tuple[tuple[str, int], ...] = (
    ("public_group_scene", 400),
    ("conversation_continuity", 400),
    ("semantic_scene", 300),
    ("semantic_temporal_context", 200),
)


def middle_truncate_text(value: str, maximum_chars: int) -> str:
    """Retain both semantic ends while removing the middle of long text."""


def reduce_identity_projection(identity: dict[str, Any]) -> bool:
    """Apply the next bounded identity reduction step for one prompt packet."""


def reduce_constraints_projection(constraints: dict[str, Any]) -> bool:
    """Apply the next bounded character-constraint reduction step."""


def reduce_scene_context_projection(scene_context: dict[str, Any]) -> bool:
    """Apply the next bounded scene-context reduction step."""
```

`reduce_identity_projection` applies exactly one step per call, in this order:

1. the first entry in `IDENTITY_TEXT_FLOORS` whose current text is longer than
   its floor, middle-truncated to that floor;
2. once every text floor is reached, truncate
   `self_image.current_growth_edges` to `MAX_REDUCED_GROWTH_EDGES` entries.

`reduce_identity_projection` never reads or writes `core.name`, `core.gender`,
`core.age`, `core.birthday`, `personality.mbti`, or any `boundaries` field.
Missing keys are skipped, not created.

`reduce_constraints_projection` applies exactly one step: middle-truncate every
`standards[*].description` longer than
`MAX_REDUCED_STANDARD_DESCRIPTION_CHARS`. It never removes `drives`,
`standards`, `meaning_state`, or any standard row.

`reduce_scene_context_projection` applies exactly one step: the first entry in
`SCENE_TEXT_FLOORS` whose current text is longer than its floor.

`prompt_budget._middle_truncate_text` is renamed to the public
`middle_truncate_text`. `workspace._middle_truncate_text` is deleted and
`workspace._fit_workspace_prompt_payload` imports the shared function.

### Appraisal payload fitting contract

```python
def _fit_appraisal_payload(
    payload: dict[str, Any],
    *,
    system_prompt_chars: int,
) -> tuple[str, frozenset[str]]:
    """Fit one appraisal packet and return its text and surviving handles."""
```

Fixed reduction order, one step per loop iteration, re-serializing after each
step:

1. `reduce_identity_projection(state["character_identity"])`
2. `reduce_constraints_projection(state["character_constraints"])`
3. state-row removal in the existing order
   `("knowledge_gaps", "events", "threats", "goals", "affect",
   "relationship", "character_operational_context")`, dropping the tail row of
   a non-empty list or popping a non-list value, exactly as today; `"roles"` is
   removed from this order because that projection is deleted
4. `fit_evidence_texts_to_budget(...)` at the existing
   `MIN_PROMPT_EVIDENCE_TEXT_CHARS` floor
5. `CognitionContextLimitError`

Whenever step 3 removes an element that owns a prompt handle, the same
iteration removes that handle from the question authority:

| Removed element | Handle dropped |
|---|---|
| a `goals` row | that row's `handle` |
| a `threats` row | that row's `handle` |
| a `events` row | that row's `handle` |
| a `knowledge_gaps` row | that row's `handle` |
| `relationship` | `r1` |
| `affect`, `character_operational_context` | none |

Dropping handle `H` removes `H` from `question["permitted_role_handles"]`, from
each of `question["handle_field_domains"]["subject_handle"]`,
`["object_handle"]`, and `["entity_handle"]`, from
`question["candidate_origin_evidence"]`, and from every
`question["permitted_delta_path_domains"][*]["handles"]`. A domain entry whose
`handles` list becomes empty is removed.

The returned `frozenset[str]` is the surviving `permitted_role_handles` set.
`appraise_semantic_question` narrows `item_question["permitted_role_handles"]`
and `item_question["permitted_delta_paths"]` to that set before validation, so
the item validator never accepts a handle the prompt did not show.

### Appraisal provenance bound

```python
def _validate_handles(
    value: Any,
    allowed: set[str],
    label: str,
    *,
    minimum: int = 1,
    maximum: int = MAX_APPRAISAL_OBJECT_HANDLES,
) -> list[str]:
```

`MAX_APPRAISAL_OBJECT_HANDLES = 8` is the new named constant for the two
model-authored per-object lists. Call-site maxima:

| Call site | `maximum` |
|---|---|
| proposition `evidence_handles` | `MAX_APPRAISAL_OBJECT_HANDLES` |
| delta `evidence_handles` | `MAX_APPRAISAL_OBJECT_HANDLES` |
| `selected_evidence_handles` | `len(evidence_handles)` |
| `selected_role_handles` | `len(question["permitted_role_handles"])` |

### Goal cognition payload fitting contract

`_fit_goal_prompt_payload` keeps its signature. Its fixed reduction order, one
step per loop iteration:

1. existing `_GOAL_SUPPLEMENTAL_CONTEXT_ORDER` removal, unchanged
2. `reduce_scene_context_projection(projected_context["scene_context"])`
3. `reduce_constraints_projection(projected_context["character_constraints"])`
4. `reduce_identity_projection(projected_context["character_identity"])`
5. `fit_evidence_texts_to_budget(...)` at `MIN_PROMPT_EVIDENCE_TEXT_CHARS`
6. `PromptBudgetError`

`"roles"` is removed from `_GOAL_SUPPLEMENTAL_CONTEXT_ORDER` because that
projection is deleted.

### Generic goal degradation contract

```python
def _degraded_goal_bid_draft(
    parsed: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    require_relational_willingness: bool,
    episode_handles: set[str] | None,
) -> GoalBidDraftV2 | None:
    """Project a complete generic bid after dropping invalid handle entries."""
```

Rules:

- return `None` when `parsed` is not a `Mapping`;
- return `None` when neither `evidence_handles` nor `target_role_handles` is a
  list;
- build a candidate copy retaining only entries of `evidence_handles` that are
  strings present in `evidence_handles`, and only entries of
  `target_role_handles` that are strings present in `role_handles`;
- return `None` when neither list changed, so an unchanged candidate never
  takes the degraded path;
- run the unmodified `validate_goal_bid_draft(...)` on the candidate and return
  `None` on any failure;
- never touch `relational_willingness`, any prose field,
  `expected_consequences`, or `confidence`.

The degraded draft is reachable only on the final attempt of a non-selection
branch. The trace step records `parse_status="degraded"` and
`status="degraded"`, matching the existing selection path.

### Stale-bid filter contract

`facade._run_cognition` gains one private projection:

```python
def _bids_with_live_goals(
    bids: Sequence[ActionBidV2],
    state: Mapping[str, Any],
) -> tuple[list[ActionBidV2], list[str]]:
    """Keep bids whose non-ordinary persistent goal still exists in state."""
```

It returns the retained bids in input order and the dropped branch ids. The
facade appends `stale_goal_bid_dropped:<branch_id>` to `warnings` for each
dropped id, passes the retained list to `_ordinary_relational_decision`,
`_workspace_goal_contexts`, `collapse_authoritative_relational_bid`, and
`collapse_bids`, and keeps the unfiltered list in the
`branch_execution` validation event under the existing `generated_bids` key.

### Surface repair contract

```python
def _surface_repair_messages(
    *,
    payload: Mapping[str, Any],
    system_prompt: str,
    invalid_candidate: str,
    reason: str,
    contract_error: str,
    stage_name: str,
    safe_checkpoint: str,
    attempt_count: int,
) -> list[SystemMessage | HumanMessage]:
```

The returned list is `[SystemMessage(content=system_prompt),
HumanMessage(content=<repair payload>)]`. The repair payload is:

```python
{
    "surface": payload,
    "contract_repair": {
        "repair_instruction": SURFACE_REPAIR_INSTRUCTION,
        "reason": reason,
        "contract_error": contract_error[:SURFACE_STAGE_ERROR_CAP],
        "invalid_candidate": _bounded_repair_text(invalid_candidate),
    },
}
```

`SURFACE_REPAIR_INSTRUCTION` is a new module-level Chinese string carrying the
retained content of the deleted `_SURFACE_REPAIR_PROMPT`: preserve the original
character judgment, affect direction, relationship direction, selected
intention, capability results, and facts; repair only the field set, field
types, lengths, list cardinality, and JSON syntax; return only the JSON object
this stage's output contract defines. `SURFACE_STAGE_ERROR_CAP = 500`.

`_surface_prompt_text` gains a `system_prompt_chars: int` keyword and applies
the reduction order before raising:

1. tail-drop `payload["supporting_bids"]` entries while more than
   `MIN_SURFACE_SUPPORTING_BIDS` remain, then remove the key;
2. tail-drop `payload["semantic_affect"]` entries, then remove the key;
3. raise the existing typed `surface_<stage>_context_limit` failure.

`MIN_SURFACE_SUPPORTING_BIDS = 2`.

### Goal repair contract

`GOAL_COGNITION_REPAIR_PROMPT` and `REQUIRED_SELECTION_GOAL_REPAIR_PROMPT` are
deleted. Every regeneration reuses `initial_system_prompt`. The structural
instructions those prompts carried move into `repair_feedback` as a new
bounded field:

```python
repair_feedback["repair_instruction"] = list[str]
```

Two module-level tuples supply the exact strings:
`GENERIC_GOAL_REPAIR_INSTRUCTIONS` and
`SELECTION_GOAL_REPAIR_INSTRUCTIONS`. Each entry is one Chinese sentence
already present in the deleted prompt, extracted byte-for-byte from the current
source rather than retyped. `_build_goal_repair_feedback(...)` gains a
`selection_required`-driven selection between the two tuples; every other field
it returns is unchanged.

### Authorization cap accounting

`invoke_semantic_authorizer` changes `base_prompt_chars` from a `HumanMessage`
sum to a sum over every message. `authorize_action_requests` and
`authorize_resolver_requests` add `len(<SYSTEM_PROMPT>)` to their pre-invocation
`prompt_text` length comparison.

### Planner row containment

`_normalize_action_request_rows` and `_normalize_resolver_request_rows` log and
skip an invalid row instead of raising, then raise
`ValueError("every proposed <kind> request row was unusable")` only when
`values` was non-empty and `normalized` is empty.

### Proposition subject binding

```python
_PROPOSITION_SUBJECT_KIND_SETS = {
    "outcome_pending": frozenset({
        "goal",
        "event",
        "threat",
        "knowledge_gap",
    }),
}
```

`_validate_proposition` checks `_PROPOSITION_SUBJECT_KINDS` first, then
`_PROPOSITION_SUBJECT_KIND_SETS`, raising when the subject handle's kind is
outside the permitted set.

### Validation error contract

One private helper renders a capped allowlist:

```python
MAX_ERROR_ALLOWLIST_ITEMS = 40


def _allowlist_hint(values: Sequence[str]) -> str:
    """Render a bounded sorted allowlist for one contract error message."""
```

It returns `json.dumps(sorted_values[:MAX_ERROR_ALLOWLIST_ITEMS])` plus a
`" (+N more)"` suffix when truncated. Errors that must carry a hint:

| Function | Error must name |
|---|---|
| `_validate_delta` | the rejected `target_path` and the permitted paths |
| `_validate_handles` | the rejected handles and the allowed set |
| `_validate_proposition` subject | the rejected handle and the permitted role handles |
| `_validate_proposition` object | the rejected handle and the permitted role handles |
| `_validate_proposition` kind | the rejected kind and the permitted kinds in full |
| `_require_text` | the field label and its maximum character count |

`_require_text(value, maximum=200)` becomes
`_require_text(value, label, *, maximum=MAX_APPRAISAL_SEMANTIC_TEXT_CHARS)`
with `MAX_APPRAISAL_SEMANTIC_TEXT_CHARS = 200` and
`MAX_APPRAISAL_DELTA_REASON_CHARS = 300`.

### Attempt policy correction

`model_attempt_policy.py` adds `V2_APPRAISAL_TOTAL_ATTEMPTS = 2`, sets
`V2_MODEL_OWNER_POLICIES["semantic_appraisal"]["total_attempt_limit"]` to it,
and exports it through `cognition_core_v2/__init__.py`.
`semantic_appraisal.SEMANTIC_APPRAISAL_ATTEMPT_LIMIT` becomes
`V2_APPRAISAL_TOTAL_ATTEMPTS`.
`tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`
`owners.semantic_appraisal.total_attempt_limit` becomes `2`.

No public state schema, database document, queue envelope, or adapter contract
changes anywhere in this plan.

## LLM Call And Context Budget

Context-window cap: 50k tokens per model call.

Estimation method: character-based upper bound. Every prompt character is
counted as at most one token, which is the pessimistic ceiling for dense CJK
tokenization. Realistic mixed CJK/ASCII JSON tokenizes at roughly 0.6-0.8
tokens per character, so the figures below overstate real usage by 25-40%.

### Call counts

| Surface | Before | After |
|---|---:|---:|
| Appraisal calls per family | up to 8 items x 2 attempts | unchanged |
| Appraisal families per turn | up to 6 | unchanged |
| Goal branch attempts | up to 3 per branch | unchanged |
| Goal branches per turn | up to 14 | unchanged |
| Workspace collapse attempts | up to 3 | unchanged |
| Action planning attempts | up to 3 | unchanged |
| Action / resolver authorization attempts | up to 3 each | unchanged |
| Surface content / preference attempts | up to 3 each | unchanged |
| Surface visual attempts | up to 3 | unchanged |
| New model calls added | n/a | 0 |

Call counts do not change. Two changes reduce calls in practice: a bounded
appraisal item termination ends a family without its second attempt, and a
degraded generic goal bid can avoid the service safe-retry graph invocation.

### Per-call worst-case window use

| Stage | Input chars before | Completion before | Ceiling before | Input chars after | Completion after | Ceiling after | Margin after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic appraisal | 10 888 | 25 000 | 35 888 | 20 000 | 2 048 | 22 048 | 56% |
| Appraisal repair | 12 888 | 25 000 | 37 888 | 24 000 | 2 048 | 26 048 | 48% |
| Goal cognition | 24 000 | 25 000 | 49 000 | 36 000 | 8 192 | 44 192 | 12% |
| Workspace collapse | 24 000 | 25 000 | 49 000 | 24 000 | 1 024 | 25 024 | 50% |
| Action planning | 30 814 | 25 000 | 55 814 | 32 000 | 8 192 | 40 192 | 20% |
| Action authorization | 18 157 | 25 000 | 43 157 | 20 000 | 1 024 | 21 024 | 58% |
| Resolver authorization | 24 710 | 25 000 | 49 710 | 24 000 | 1 024 | 25 024 | 50% |
| Surface content plan | 26 397 | 25 000 | 51 397 | 32 000 | 8 192 | 40 192 | 20% |
| Surface preference | 25 062 | 25 000 | 50 062 | 32 000 | 4 096 | 36 096 | 28% |
| Surface visual | 24 626 | 25 000 | 49 626 | 32 000 | 2 048 | 34 048 | 32% |
| Surface dialog repair | 26 339 | 25 000 | 51 339 | 32 000 | 8 192 | 40 192 | 20% |

Three stages exceeded the 50k ceiling before this change (action planning,
surface content plan, surface preference, surface dialog repair). Every stage
is inside the ceiling after it, with the smallest margin at goal cognition.

### Blocking, latency, and truncation

- Every cognition-owned call gains `timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS`
  (120 s default). No cognition call is currently bounded in wall-clock time.
- Worst-case single-call generation drops from 25 000 to 8 192 tokens, and to
  2 048 on the six appraisal routes that carry the highest per-turn call count.
- Truncation and drop policy is fully specified in
  `Contracts And Data Shapes`. Every reduction step has a named floor. No stage
  silently discards required evidence, a required handle, or a permission
  decision.
- Verification tests: the maximum-shape and exact-cap tests named in
  `Verification`, extended to a maximum-size identity.

No new response-path model call is introduced. No cap is raised above the 50k
window. No `max_completion_tokens` value increases.

## Change Surface

Target ownership boundary: `src/kazusa_ai_chatbot/cognition_core_v2`.

Two changes fall outside that boundary and carry justification:

- `src/kazusa_ai_chatbot/config.py` owns every route default and every
  environment-backed constant in this project. The package cannot own its own
  route defaults without duplicating the config contract, which
  `py-style N-011` forbids. The change is additive plus twelve default-value
  edits, all bounded by grep verification.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` and
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py` are the only
  constructors of the cognition and surface `LLMCallConfig` objects.
  `cognition_core_v2` receives them by injection and cannot set
  `timeout_seconds` itself. Both edits are confined to the config literals and
  `_surface_config`.

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - add the three cognition completion defaults, three surface completion
    defaults, and `COGNITION_STAGE_TIMEOUT_SECONDS`;
  - repoint twelve `COGNITION_LLM_*_MAX_COMPLETION_TOKENS` defaults.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - add `timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS` to all twelve V2
    `LLMCallConfig` literals.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - add a `max_completion_tokens` parameter to `_surface_config` and pass the
    three surface defaults plus `timeout_seconds`.
- `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`
  - rename `_middle_truncate_text` to `middle_truncate_text`;
  - add the three floors tuples, three reducers, and their constants.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py`
  - new caps and cap accounting;
  - identity and constraints reduction in `_fit_appraisal_payload`;
  - lockstep handle drop and the `frozenset` return;
  - `_validate_handles` maximum parameter and call-site maxima;
  - bounded item termination in `appraise_semantic_question`;
  - `_require_text` label and maximum;
  - `_allowlist_hint` and the six actionable errors;
  - `_PROPOSITION_SUBJECT_KIND_SETS`;
  - `SEMANTIC_APPRAISAL_ATTEMPT_LIMIT` sourced from the policy module;
  - output skeleton appended to `SEMANTIC_APPRAISAL_PROMPT`.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - new cap value;
  - scene, constraints, and identity reduction in `_fit_goal_prompt_payload`;
  - `"roles"` removed from `_GOAL_SUPPLEMENTAL_CONTEXT_ORDER`;
  - `_degraded_goal_bid_draft` and its final-attempt wiring;
  - deletion of both repair prompts and the `repair_instruction` feedback
    field;
  - output skeleton appended to `GOAL_COGNITION_PROMPT`.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - add `_bids_with_live_goals` and wire it between bid collection and both
    collapse paths; no other change.
- `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`
  - delete `_middle_truncate_text` and import the shared function.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
  - new cap value and system-prompt-inclusive accounting;
  - individual row dropping in both normalizers.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py`
  - new cap value, system-prompt-inclusive accounting in
    `invoke_semantic_authorizer` and in the `authorize_action_requests`
    pre-check.
- `src/kazusa_ai_chatbot/cognition_core_v2/resolver_authorization.py`
  - system-prompt-inclusive pre-check.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
  - new cap values and accounting;
  - delete `_SURFACE_REPAIR_PROMPT`, add `SURFACE_REPAIR_INSTRUCTION`;
  - `_surface_repair_messages` signature and payload;
  - `_surface_prompt_text` reduction pass.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`
  - delete the Chinese-keyed `payload["roles"]` stub.
- `src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py`
  - correct the `_select_evidence_rows` docstring to describe actual behavior.
- `src/kazusa_ai_chatbot/cognition_core_v2/model_attempt_policy.py`
  - add `V2_APPRAISAL_TOTAL_ATTEMPTS` and correct the appraisal policy row.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`
  - export `V2_APPRAISAL_TOTAL_ATTEMPTS`.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - new cap values and accounting rule; identity/constraints/scene reduction;
    provenance bound; bounded appraisal item termination; generic goal
    degradation; stale-bid drop; surface repair contract; goal repair contract;
    appraisal attempt count; planner row containment; surface reduction pass.
- `docs/HOWTO.md`
  - twelve `COGNITION_LLM_*_MAX_COMPLETION_TOKENS` example values and one new
    `COGNITION_STAGE_TIMEOUT_SECONDS` example line.

### Create

- No new module, package, prompt file, or fixture file.

### Delete

- `surface_stages._SURFACE_REPAIR_PROMPT`
- `goal_cognition.GOAL_COGNITION_REPAIR_PROMPT`
- `goal_cognition.REQUIRED_SELECTION_GOAL_REPAIR_PROMPT`
- `workspace._middle_truncate_text`
- the `payload["roles"]` stub in `state_projection.project_state_for_prompt`
- the dead `roles` filter in `semantic_appraisal._project_question_state`
- the `"roles"` entry in both reduction orders

### Tests

- `tests/cognition_core_v2_test_helpers.py`
  - add `maximum_character_identity()` and
    `maximum_identity_context()` returning an identity at every
    `TEXT_LIMIT_BY_PATH` ceiling with five growth edges.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
- `tests/test_cognition_core_v2_dependencies.py`
- `tests/test_cognition_core_v2_failures.py`
- `tests/test_cognition_core_v2_integration.py`
- `tests/test_cognition_core_v2_model_retry_continuity.py`
- `tests/test_cognition_core_v2_stage_model_routing.py`
- `tests/test_cognition_core_v2_projection.py`
- `tests/test_cognition_core_v2_alignment_gates.py`
- `tests/test_cognition_core_v2_live_character_judgment.py`
- `tests/test_cognition_core_v2_contracts.py`
- `tests/test_action_selection_prompt_contract.py`
- `tests/test_config.py`
- `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`

### Keep

- Every file under `src/kazusa_ai_chatbot/cognition_core_v2` not listed above,
  including `state_reducers.py`, `state_models.py`, `emotion_derivation.py`,
  `emotion_definitions.py`, `transition_guards.py`, `character_carryover.py`,
  `branch_activation.py`, `dependency_graph.py`, `parallel_executor.py`,
  `output_projection.py`, `diagnostics.py`, `validation_cli.py`,
  `contracts.py`, and `surface.py`.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`.
- Every non-cognition consumer of `DEFAULT_LLM_MAX_COMPLETION_TOKENS`.

### Lifecycle

- this plan;
- `development_plans/README.md` Active Bugfix Plans table.

## Overdesign Guardrail

- Actual problem: `cognition_core_v2` aborts a turn, or silently omits every
  appraisal family, through six deterministic code paths that a grown character
  identity or a multi-entity turn triggers by construction; and four bounded
  retry paths give the local model too little information to converge.
- Minimal change: make three already-projected context blocks reducible, bound
  two derived lists by their own permitted sets, convert three all-or-nothing
  raises into the dispositions the module's own policy table and README already
  declare, retain the system prompt and validation error on repair, and
  right-size the completion reservation that starves the input budget.
- Ownership boundaries: `config.py` owns route defaults and environment
  bounds. The two `nodes/` builders own `LLMCallConfig` construction.
  `prompt_budget.py` owns deterministic truncation and reduction.
  `semantic_appraisal.py` and `goal_cognition.py` own their packet fitting and
  structural validation. `facade.py` owns bid eligibility before collapse.
  The LLM owns every semantic judgment, including the relational-willingness
  stance, which deterministic code never derives, upgrades, or rewrites.
- Rejected complexity: no new module; only the seven new or renamed functions
  enumerated in `Agent Autonomy Boundaries`; no new LLM stage, verifier, judge,
  or JSON-repair call; no feature flag or environment switch; no compatibility
  alias for any deleted prompt constant; no generic reduction framework or
  strategy registry; no per-stage timeout constant; no new
  `*_MAX_COMPLETION_TOKENS` environment variable; no token-counting library or
  tokenizer dependency; no pre-flight token guard in the provider adapter; no
  explicit-path allowlist replacing `permitted_delta_path_domains`; no lockstep
  handle reduction in goal cognition; no percept truncation in the surface
  reduction pass; no retry-count increase anywhere.
- Evidence threshold: adding a tokenizer-backed budget, a provider-side context
  guard, an explicit delta-path allowlist, or a percept reduction pass requires
  a live-LLM trace showing that the character-based ceiling in
  `LLM Call And Context Budget` was breached in production, or a reproduced
  failure that the fixed reduction orders in this plan cannot contain.

## Agent Autonomy Boundaries

The responsible agent may:

- choose local implementation mechanics that preserve every contract in
  `Contracts And Data Shapes`;
- add the named test helpers and test cases;
- make mechanical documentation and test-assertion updates required by a stage
  cutover;
- stop early and report a blocker.

The responsible agent must not:

- introduce new architecture, an alternate migration strategy, a compatibility
  layer, a fallback path, or an extra feature;
- add a helper, wrapper, alias, flag, or abstraction beyond these seven, which
  are the only new or renamed functions this plan authorizes:
  `prompt_budget.middle_truncate_text` (rename of the existing private
  function), `prompt_budget.reduce_identity_projection`,
  `prompt_budget.reduce_constraints_projection`,
  `prompt_budget.reduce_scene_context_projection`,
  `facade._bids_with_live_goals`,
  `goal_cognition._degraded_goal_bid_draft`, and
  `semantic_appraisal._allowlist_hint`;
- change a file outside `Change Surface`;
- change a numeric budget, cap, floor, or attempt count to a value other than
  the one this plan states;
- perform unrelated cleanup, formatting churn, dependency upgrade, prompt
  rewrite, or broad refactor;
- fix an unrelated failing test discovered during baseline capture;
- derive, upgrade, or rewrite a `relational_willingness` stance in
  deterministic code;
- widen the generic goal degradation beyond the two handle lists.

The responsible agent must stop and request a plan amendment when:

- a stated cap, floor, or default cannot satisfy its own maximum-shape test;
- a deleted prompt constant turns out to have a consumer outside
  `Change Surface`;
- a reduction order cannot reach the cap without dropping required evidence, a
  required handle, or a permission decision;
- the plan and the code disagree about an existing contract;
- an unrelated worktree change overlaps a target file and cannot be preserved;
- native subagent capability is unavailable at an implementation or review
  gate.

## Implementation Order

The order is test-contract-first per stage. Each stage establishes its focused
test expectation, records the pre-implementation result, implements, then
re-runs the same test.

1. Reread this plan, every mandatory skill, `development_plans/README.md`,
   `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, `docs/HOWTO.md`,
   `git status`, and every file in `Change Surface`.
2. Capture the deterministic baseline. Record which assertions already fail,
   specifically the two prompt-length assertions in
   `test_live_character_prompts_fit_local_model_attention_caps`.
3. Add `maximum_character_identity()` and `maximum_identity_context()` to
   `tests/cognition_core_v2_test_helpers.py`.
4. Repoint `_maximum_appraisal_context` and `_maximum_goal_context` in
   `tests/test_cognition_core_v2_prompt_budget_continuity.py` to
   `maximum_identity_context()`, run the maximum-shape tests, and record the
   expected `CognitionContextLimitError` and `PromptBudgetError` failures.
5. Implement the completion-token and timeout rebalance in `config.py` and both
   `nodes/` builders. Build the config constants before the route edits so a
   missing name fails at import.
6. Implement `middle_truncate_text`, the three floors tuples, and the three
   reducers in `prompt_budget.py`. Delete
   `workspace._middle_truncate_text` and repoint its call site.
7. Implement the new cap values and the system-prompt-inclusive accounting in
   `semantic_appraisal.py`, `goal_cognition.py`, `action_selection.py`,
   `action_authorization.py`, `resolver_authorization.py`, and
   `surface_stages.py`.
8. Implement the identity, constraints, and scene reduction steps in
   `_fit_appraisal_payload` and `_fit_goal_prompt_payload` in their stated
   orders. Run `py_compile` on both files immediately.
9. Re-run the maximum-shape tests from step 4 and record the pass.
10. Add the failing provenance-overflow test and the failing bounded-termination
    test to `tests/test_cognition_core_v2_failures.py`, and record their
    pre-implementation failures.
11. Implement the `_validate_handles` maximum parameter, its four call-site
    maxima, and the bounded item termination in `semantic_appraisal.py`.
    Re-run step 10's tests.
12. Add the failing generic-goal-degradation test to
    `tests/test_cognition_core_v2_failures.py` and record its failure.
13. Implement `_degraded_goal_bid_draft` and its final-attempt wiring. Re-run
    step 12's test.
14. Add the failing stale-goal-bid test to
    `tests/test_cognition_core_v2_integration.py` and record its failure.
15. Implement `_bids_with_live_goals` and its facade wiring. Re-run step 14's
    test.
16. Add the failing surface-repair-contract test to
    `tests/test_cognition_core_v2_dependencies.py` and record its failure.
17. Implement the surface repair contract and delete `_SURFACE_REPAIR_PROMPT`.
    Run `py_compile` on `surface_stages.py`. Re-run step 16's test.
18. Update the five `REQUIRED_SELECTION_GOAL_REPAIR_PROMPT` assertions and the
    `test_required_selection_regeneration_feedback_counts_toward_cap` assertions
    in `tests/test_cognition_core_v2_dependencies.py` to expect the initial
    system prompt, and record their failures.
19. Implement the goal repair contract: extract the retained Chinese
    instruction sentences from the current source with a Python script, delete
    both repair prompts, add the two instruction tuples and the
    `repair_instruction` feedback field. Run `py_compile` on
    `goal_cognition.py`. Re-run step 18's tests.
20. Add the failing actionable-error tests to
    `tests/test_cognition_core_v2_failures.py` and record their failures.
21. Implement `_allowlist_hint`, the `_require_text` signature change, the six
    actionable errors, and `_PROPOSITION_SUBJECT_KIND_SETS`. Re-run step 20's
    tests.
22. Add the failing lockstep-reduction test to
    `tests/test_cognition_core_v2_prompt_budget_continuity.py` and record its
    failure.
23. Implement the lockstep handle drop, the `frozenset` return from
    `_fit_appraisal_payload`, and the `item_question` narrowing. Re-run step
    22's test.
24. Raise the two prompt-length caps in
    `tests/test_cognition_core_v2_live_character_judgment.py` to their new
    values and add a `SEMANTIC_APPRAISAL_PROMPT` cap assertion.
25. Append the output skeletons to `SEMANTIC_APPRAISAL_PROMPT` and
    `GOAL_COGNITION_PROMPT` by editing the existing literals in place. Run
    `py_compile` on both files. Re-run step 24's assertions and every prompt
    content assertion in
    `tests/test_cognition_core_v2_action_planning_bugfix.py`,
    `tests/test_cognition_core_v2_dependencies.py`, and
    `tests/test_cognition_core_v2_live_character_judgment.py`.
26. Add the failing planner-row-containment test to
    `tests/test_action_selection_prompt_contract.py` and record its failure.
27. Implement individual row dropping in both planner normalizers. Re-run step
    26's test.
28. Add the failing surface-reduction test to
    `tests/test_cognition_core_v2_prompt_budget_continuity.py` and record its
    failure.
29. Implement the `_surface_prompt_text` reduction pass. Re-run step 28's test.
30. Correct the appraisal attempt policy in `model_attempt_policy.py`, its
    export in `__init__.py`, and
    `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`. Re-run
    `tests/test_cognition_core_v2_model_retry_continuity.py`.
31. Delete the `payload["roles"]` stub, the dead filter, and both `"roles"`
    reduction-order entries. Update
    `tests/test_cognition_core_v2_projection.py:104`. Correct the
    `_select_evidence_rows` docstring.
32. Update `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and
    `docs/HOWTO.md`.
33. Run the full deterministic verification suite and every static grep.
34. Run each live LLM case in `Verification` separately and inspect its
    artifact before starting the next.
35. Start one independent code-review subagent, remediate valid findings inside
    the approved change surface, and re-run affected verification.
36. Record every command, result, artifact path, residual risk, and sign-off in
    `Execution Evidence`, then update `development_plans/README.md`.

## Execution Model

- Parent agent owns orchestration, all test code, verification, static checks,
  execution evidence, review-feedback remediation, lifecycle updates, and final
  sign-off.
- Parent agent establishes each stage's focused test contract and records the
  expected failure or baseline before production implementation of that stage
  starts.
- Production-code subagent: exactly one native subagent, started after the step
  4 baseline is recorded. It owns production code changes only. It does not
  edit tests unless the parent explicitly directs it. It closes after the
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  documentation, and plan-progress updates while the production-code subagent
  edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes. It reviews the plan, the full diff, and the
  evidence, reports findings to the parent, and does not implement fixes.
- Parent agent runs every live LLM test itself, one at a time.
- If native subagent capability is unavailable at an implementation or review
  gate, stop before execution and report the blocker. Do not switch to
  single-agent execution without explicit user approval.

## Progress Checklist

- [ ] Stage 0 - baseline captured
  - Covers: steps 1-2.
  - Files: none.
  - Verify: the deterministic batch in `Verification / Deterministic baseline`.
  - Evidence: full pass/fail list, and explicit confirmation that
    `test_live_character_prompts_fit_local_model_attention_caps` already fails
    on `GOAL_COGNITION_PROMPT` and `REQUIRED_SELECTION_GOAL_PROMPT`.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 1 - maximum-identity test contract established
  - Covers: steps 3-4.
  - Files: `tests/cognition_core_v2_test_helpers.py`,
    `tests/test_cognition_core_v2_prompt_budget_continuity.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q`
  - Evidence: the recorded `CognitionContextLimitError` for each of the five
    identity-bearing appraisal families and the `PromptBudgetError` for goal
    cognition.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - completion-token and timeout rebalance implemented
  - Covers: step 5.
  - Files: `src/kazusa_ai_chatbot/config.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`,
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`,
    `tests/test_cognition_core_v2_stage_model_routing.py`,
    `tests/test_config.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_cognition_core_v2_stage_model_routing.py -q`
    and the route grep in `Verification / Static Greps`.
  - Evidence: test output plus the grep result proving no cognition route still
    defaults to `DEFAULT_LLM_MAX_COMPLETION_TOKENS`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - shared reduction helpers implemented
  - Covers: step 6.
  - Files: `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`,
    `src/kazusa_ai_chatbot/cognition_core_v2/workspace.py`.
  - Verify: `venv\Scripts\python.exe -m py_compile` on both files, then
    `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q`
    and the `_middle_truncate_text` grep.
  - Evidence: compile result, test output, grep returning matches only in
    `prompt_budget.py`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - cap accounting and context reduction implemented
  - Covers: steps 7-9.
  - Files: `semantic_appraisal.py`, `goal_cognition.py`,
    `action_selection.py`, `action_authorization.py`,
    `resolver_authorization.py`, `surface_stages.py`.
  - Verify: `py_compile` on all six files, then the full budget-continuity and
    dependencies batches.
  - Evidence: the step 4 failures now pass; the exact-cap and cap-plus-one
    tests pass at the new values.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - appraisal provenance bound and bounded item termination
  - Covers: steps 10-11.
  - Files: `semantic_appraisal.py`, `tests/test_cognition_core_v2_failures.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_failures.py -q`
  - Evidence: the recorded pre-implementation failure text
    `selected roles handles must contain between 0 and 8 items`, then the pass;
    plus the accepted-prefix return proof.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - generic goal degradation implemented
  - Covers: steps 12-13.
  - Files: `goal_cognition.py`, `tests/test_cognition_core_v2_failures.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_dependencies.py -q`
  - Evidence: a handle-only defect degrades; a prose, consequence,
    `confidence`, or `relational_willingness` defect still raises
    `goal_bid_structure_exhausted`.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 - stale goal bids dropped before collapse
  - Covers: steps 14-15.
  - Files: `facade.py`, `tests/test_cognition_core_v2_integration.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_integration.py -q`
  - Evidence: a terminalized-goal bid is dropped with
    `stale_goal_bid_dropped:<branch_id>` and the turn completes; the unfiltered
    bid list remains in the `branch_execution` validation event.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 8 - surface repair contract implemented
  - Covers: steps 16-17.
  - Files: `surface_stages.py`,
    `tests/test_cognition_core_v2_dependencies.py`.
  - Verify: `py_compile` on `surface_stages.py`, then
    `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_prompt_budget_continuity.py -q`
    and the `_SURFACE_REPAIR_PROMPT` grep.
  - Evidence: the repair request retains the stage system prompt and carries
    the exact `contract_error`; grep returns no matches.
  - Handoff: next agent starts at Stage 9.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 9 - goal repair semantic authority restored
  - Covers: steps 18-19.
  - Files: `goal_cognition.py`,
    `tests/test_cognition_core_v2_dependencies.py`.
  - Verify: `py_compile` on `goal_cognition.py`, then
    `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_dependencies.py tests\test_cognition_core_v2_relational_willingness.py -q`
    and the two repair-prompt greps.
  - Evidence: every regeneration reuses the initial system prompt; the
    relational-willingness ordering sentences are present in the regeneration
    request; both greps return no matches.
  - Handoff: next agent starts at Stage 10.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 10 - actionable validator errors implemented
  - Covers: steps 20-21.
  - Files: `semantic_appraisal.py`,
    `tests/test_cognition_core_v2_failures.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_failures.py tests\test_cognition_core_v2_contracts.py -q`
  - Evidence: each of the six errors names the rejected value and its
    allowlist; the allowlist is capped at
    `MAX_ERROR_ALLOWLIST_ITEMS` with the `(+N more)` suffix.
  - Handoff: next agent starts at Stage 11.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 11 - lockstep handle reduction implemented
  - Covers: steps 22-23.
  - Files: `semantic_appraisal.py`,
    `tests/test_cognition_core_v2_prompt_budget_continuity.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py tests\test_cognition_core_v2_alignment_gates.py -q`
  - Evidence: after a state-row drop, the removed handle is absent from
    `permitted_role_handles`, all three `handle_field_domains` lists,
    `candidate_origin_evidence`, and every
    `permitted_delta_path_domains[*].handles`; the item validator rejects that
    handle.
  - Handoff: next agent starts at Stage 12.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 12 - output skeletons added
  - Covers: steps 24-25.
  - Files: `semantic_appraisal.py`, `goal_cognition.py`,
    `tests/test_cognition_core_v2_live_character_judgment.py`.
  - Verify: `py_compile` on both source files, then
    `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_live_character_judgment.py tests\test_cognition_core_v2_action_planning_bugfix.py tests\test_cognition_core_v2_dependencies.py -q`
  - Evidence: the skeleton block is present in both prompts; every existing
    prompt content assertion still passes; the new length caps hold.
  - Handoff: next agent starts at Stage 13.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 13 - planner row containment implemented
  - Covers: steps 26-27.
  - Files: `action_selection.py`,
    `tests/test_action_selection_prompt_contract.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_cognition_core_v2_action_planning_bugfix.py -q`
  - Evidence: one invalid row among valid rows is dropped and logged; all rows
    invalid raises once and reaches the empty blocked proposal after three
    attempts.
  - Handoff: next agent starts at Stage 14.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 14 - surface reduction pass implemented
  - Covers: steps 28-29.
  - Files: `surface_stages.py`,
    `tests/test_cognition_core_v2_prompt_budget_continuity.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_prompt_budget_continuity.py -q`
  - Evidence: an over-cap surface payload reduces and reaches the model; a
    payload still over cap after reduction returns the validated degraded
    surface with zero model calls.
  - Handoff: next agent starts at Stage 15.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 15 - policy, projection, and docstring hygiene complete
  - Covers: steps 30-31.
  - Files: `model_attempt_policy.py`, `__init__.py`, `state_projection.py`,
    `semantic_source_planner.py`,
    `tests/fixtures/cognition_core_v2_retry_exhaustion_cases.json`,
    `tests/test_cognition_core_v2_projection.py`,
    `tests/test_cognition_core_v2_model_retry_continuity.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_model_retry_continuity.py tests\test_cognition_core_v2_projection.py -q`
    and the `roles` grep.
  - Evidence: a truthful appraisal attempt record with
    `total_attempt_limit=2` passes `validate_v2_attempt_record`; grep shows no
    remaining Chinese-keyed `roles` projection.
  - Handoff: next agent starts at Stage 16.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 16 - documentation updated
  - Covers: step 32.
  - Files: `src/kazusa_ai_chatbot/cognition_core_v2/README.md`,
    `docs/HOWTO.md`.
  - Verify: the documentation greps in `Verification / Static Greps`.
  - Evidence: no stale `8,000-character`, `24,000-character`, or `25000`
    cognition value remains; every new contract is described.
  - Handoff: next agent starts at Stage 17.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 17 - full deterministic verification passes
  - Covers: step 33.
  - Files: none.
  - Verify: every command in `Verification / Deterministic regression` and
    `Verification / Static Greps`.
  - Evidence: complete command output plus explicit comparison against the
    Stage 0 baseline.
  - Handoff: next agent starts at Stage 18.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 18 - live LLM verification passes
  - Covers: step 34.
  - Files: none.
  - Verify: every command in `Verification / Live LLM`, run one at a time with
    its artifact inspected before the next.
  - Evidence: per-case artifact path, model route, observed attempts, observed
    latency, and the agent's judgment on whether the behavior satisfies the
    contract.
  - Handoff: next agent starts at Stage 19.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 19 - independent code review complete
  - Covers: step 35.
  - Files: whatever the review requires inside the approved change surface.
  - Verify: re-run every affected focused test, static check, and regression
    gate after remediation.
  - Evidence: reviewer identity, files reviewed, findings, fixes applied,
    commands rerun, residual risks, approval status.
  - Handoff: next agent starts at Stage 20.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 20 - acceptance criteria and lifecycle signed off
  - Covers: step 36.
  - Files: this plan, `development_plans/README.md`.
  - Verify: every line in `Acceptance Criteria` is satisfied and evidenced.
  - Evidence: final `git diff --stat`, the completed `Execution Evidence`
    section, and the registry status change.
  - Handoff: none. Plan reaches `completed`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Deterministic baseline

Run before any edit and record the exact result:

```powershell
$cognitionV2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest $cognitionV2Tests -m "not live_llm and not live_db and not live_internet" -q
venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py -q
```

Expected baseline exception:
`test_cognition_core_v2_live_character_judgment.py::test_live_character_prompts_fit_local_model_attention_caps`
fails because `len(GOAL_COGNITION_PROMPT)` is 2300 against an asserted 2200 and
`len(REQUIRED_SELECTION_GOAL_PROMPT)` is 2501 against an asserted 2400. Record
this as pre-existing. Any other failure must be reported to the user before
implementation continues.

### Deterministic regression

```powershell
$cognitionV2Tests = Get-ChildItem -LiteralPath 'tests' -Filter 'test_cognition_core_v2*.py' | ForEach-Object { $_.FullName }
venv\Scripts\python.exe -m pytest $cognitionV2Tests -m "not live_llm and not live_db and not live_internet" -q
venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_action_selection_prompt_contract.py tests\test_action_selection_payload.py tests\test_service_input_queue.py -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\cognition_core_v2\prompt_budget.py src\kazusa_ai_chatbot\cognition_core_v2\semantic_appraisal.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py src\kazusa_ai_chatbot\cognition_core_v2\workspace.py src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py src\kazusa_ai_chatbot\cognition_core_v2\action_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\resolver_authorization.py src\kazusa_ai_chatbot\cognition_core_v2\surface_stages.py src\kazusa_ai_chatbot\cognition_core_v2\state_projection.py src\kazusa_ai_chatbot\cognition_core_v2\semantic_source_planner.py src\kazusa_ai_chatbot\cognition_core_v2\model_attempt_policy.py
git diff --check
```

Required deterministic assertions, each with at least one named test:

- every appraisal family fits at maximum state cardinality, maximum
  constraints, maximum evidence, **and** maximum identity, with zero
  `CognitionContextLimitError`;
- goal cognition fits the same maximum shape with zero `PromptBudgetError`;
- the exact-cap and cap-plus-one boundary tests hold at every new cap value;
- every cap comparison includes its system prompt length;
- `reduce_identity_projection` never alters `core.name`, `core.gender`,
  `core.age`, `core.birthday`, `personality.mbti`, or any `boundaries` field,
  and returns `False` at floor;
- `reduce_constraints_projection` never removes a standard row;
- `reduce_scene_context_projection` returns `False` at floor;
- an eight-item `goal_threat_outcome` family across eight distinct goals passes
  merged validation;
- one exhausted micro item at index 3 returns the two accepted items and
  records the bounded-termination failure detail;
- one exhausted micro item at index 1 still raises
  `semantic_appraisal_contract_exhausted`;
- a generic ordinary bid whose only defect is an unknown `evidence_handles`
  entry degrades and reaches collapse;
- a generic ordinary bid with an invalid `relational_willingness` still raises
  `goal_bid_structure_exhausted`;
- a bid whose persistent goal is absent from final state is dropped with
  `stale_goal_bid_dropped:<branch_id>` and the turn completes;
- dropping every bid still reaches the deterministic silence result;
- the surface repair request contains the stage system prompt and the exact
  validation error text;
- every goal regeneration request contains the initial system prompt;
- `repair_feedback.repair_instruction` is non-empty for both producer kinds;
- each of the six appraisal errors names the rejected value and its allowlist;
- a dropped appraisal state row removes its handle from all five question
  authority locations;
- both prompt skeletons parse as JSON after brace extraction;
- one invalid planner row among valid rows is dropped and logged;
- an all-invalid planner row set raises once and lands on the empty blocked
  proposal;
- an over-cap surface payload reduces and reaches the model;
- `validate_v2_attempt_record` accepts an appraisal record with
  `total_attempt_limit=2`;
- no cognition route config has `timeout_seconds=None`;
- no cognition route `max_completion_tokens` exceeds 8192.

### Static Greps

Each grep states its exact expected result. `rg` exit code 1 with no output is
the accepted result for a zero-match expectation.

```powershell
rg -n "_SURFACE_REPAIR_PROMPT" src tests
rg -n "GOAL_COGNITION_REPAIR_PROMPT|REQUIRED_SELECTION_GOAL_REPAIR_PROMPT" src tests
rg -n "_middle_truncate_text" src\kazusa_ai_chatbot\cognition_core_v2
rg -n "DEFAULT_LLM_MAX_COMPLETION_TOKENS" src\kazusa_ai_chatbot\config.py
rg -n "timeout_seconds" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py
rg -n "25000|25_000" src\kazusa_ai_chatbot\cognition_core_v2 docs\HOWTO.md
rg -n "8,000-character|24,000-character" src\kazusa_ai_chatbot\cognition_core_v2\README.md
rg -n '"roles"' src\kazusa_ai_chatbot\cognition_core_v2
rg -n "COGNITION_STAGE_TIMEOUT_SECONDS" src\kazusa_ai_chatbot docs\HOWTO.md
```

Expected results:

- `_SURFACE_REPAIR_PROMPT`: zero matches.
- `GOAL_COGNITION_REPAIR_PROMPT|REQUIRED_SELECTION_GOAL_REPAIR_PROMPT`: zero
  matches.
- `_middle_truncate_text`: zero matches. The shared function is
  `middle_truncate_text`.
- `DEFAULT_LLM_MAX_COMPLETION_TOKENS`: matches only on its own definition and
  on non-cognition consumers. Zero matches inside any `COGNITION_LLM_*`
  default. Forbidden match: any `COGNITION_LLM_*_MAX_COMPLETION_TOKENS` still
  defaulting to it.
- `timeout_seconds` in the two `nodes/` files: at least thirteen matches,
  covering all twelve cognition routes plus `_surface_config`.
- `25000|25_000` in `cognition_core_v2` and `docs/HOWTO.md`: zero matches.
- `8,000-character|24,000-character` in the package README: zero matches.
- `"roles"` in `cognition_core_v2`: zero matches.
- `COGNITION_STAGE_TIMEOUT_SECONDS`: matches in `config.py`, both `nodes/`
  files, and `docs/HOWTO.md`.

If any forbidden match appears, stop and fix it before continuing. Do not
suppress a grep with an inline exception.

### Live LLM

Run each command separately and inspect the newly written artifact under
`test_artifacts/llm_traces/` before running the next:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_live_llm.py -q -s --collect-only
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_live_character_judgment.py -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_relational_willingness_live_llm.py -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_workspace_live_llm.py -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_action_planning_live_llm.py -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_cognition_core_v2_surface_owner_live_llm.py -q -s
```

The first command is a collection-only listing used to enumerate the individual
case node ids. Every subsequent live case must then be run one node id at a
time, not as a file batch.

Per-case inspection must confirm:

- no call was truncated by the new `max_completion_tokens` value;
- no call hit `COGNITION_STAGE_TIMEOUT_SECONDS`;
- no appraisal family was omitted with `semantic_appraisal_context_limit`;
- the relational-willingness stance is model-authored and unchanged by
  deterministic code;
- observed per-call latency is lower than or equal to the baseline;
- the artifact records the route, model, attempts, raw output, parsed output,
  and the final disposition.

Record the agent's judgment on each case, not only the pytest status.

### Smoke

- Service imports without error:
  `venv\Scripts\python.exe -c "import kazusa_ai_chatbot.service"`.
- `venv\Scripts\python.exe -c "import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"`
  and the same for `persona_supervisor2_l3_surface`.

### Database

No database change. No database verification is required.

## Independent Code Review

Run this gate after every `Verification` command passes and before final
sign-off. The parent agent must create one independent code-review subagent
through the harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

The reviewer receives this plan, the full `git diff`, every verification
result, and every live LLM artifact path.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  fixture, and documentation artifact, including `py-style` positive and
  negative constraints, `cjk-safety` delimiter rules, docstrings, named
  constants, and path-safe commands.
- Whether any deterministic code path derives, upgrades, defaults, or rewrites
  a semantic value the model did not author, with specific attention to
  `relational_willingness`, `selection`, prose fields, and
  `expected_consequences`.
- Whether the goal and surface repair paths genuinely retain their initial
  system prompt and semantic authority, and whether the
  relational-willingness ordering rules are present in every regeneration
  request.
- Whether each reduction order is exactly as specified, one step per iteration,
  with the stated floors, and whether any reduction can remove required
  evidence, a required handle, or a permission decision.
- Whether the lockstep handle drop covers all five question authority
  locations, and whether the item validator can still accept a handle the
  prompt did not show.
- Whether the provenance bound is derived from the permitted set rather than a
  magic number.
- Whether any cap comparison still omits its system prompt.
- Whether any new helper, wrapper, flag, alias, mode, fallback path, or
  abstraction was added beyond the four functions this plan names.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, `Contracts And Data Shapes`, `Implementation Order`, and
  `Acceptance Criteria`.
- Regression and handoff quality: baseline preserved, new tests mapped to named
  risks, stale static gates corrected, live artifacts inspected one at a time,
  and no unplanned file changed.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a fix would cross that boundary or alter a
contract, stop and update this plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Every appraisal family and goal cognition fit their maximum valid prompt
  shape including a maximum-size character identity, with zero
  `CognitionContextLimitError` and zero `PromptBudgetError`.
- Every cognition prompt cap counts its system prompt plus its dynamic payload,
  and no stage's worst-case input plus completion reservation exceeds 50k
  tokens under the character-based ceiling.
- No cognition-owned `LLMCallConfig` has `timeout_seconds=None`, and no
  cognition-owned `max_completion_tokens` exceeds 8192.
- `DEFAULT_LLM_MAX_COMPLETION_TOKENS` is unchanged at 25000 and no
  non-cognition consumer's completion budget changed.
- The two derived appraisal provenance lists are bounded by their permitted-set
  size, and an eight-item family across eight distinct entities completes.
- An exhausted appraisal micro item returns the accepted prefix and records a
  bounded-termination failure detail; an exhausted first item still omits the
  family.
- A generic goal bid whose only defect is an unknown handle entry reaches
  collapse as a degraded bid; every other defect still raises
  `goal_bid_structure_exhausted`.
- A bid whose persistent goal no longer exists is dropped with a warning
  instead of raising `internal_invariant`.
- Every surface-stage repair request contains the stage system prompt and the
  exact validation error.
- Every goal regeneration request contains the initial system prompt, including
  the relational-willingness ordering rules.
- Every appraisal handle, path, kind, and text validation error names the
  rejected value and its bounded allowlist.
- A dropped appraisal state row removes its handle from `permitted_role_handles`,
  all three `handle_field_domains` lists, `candidate_origin_evidence`, and every
  `permitted_delta_path_domains[*].handles`, and the item validator rejects it.
- `SEMANTIC_APPRAISAL_PROMPT` and `GOAL_COGNITION_PROMPT` each contain one exact
  output skeleton, and every existing prompt content assertion still passes.
- An invalid planner row is dropped individually; an all-invalid row set raises
  once and lands on the empty blocked proposal.
- An over-cap surface payload reduces and reaches the model; a still-over-cap
  payload returns the validated degraded surface with zero model calls.
- `V2_MODEL_OWNER_POLICIES["semantic_appraisal"]["total_attempt_limit"]` is 2,
  the fixture matches, and `validate_v2_attempt_record` accepts a truthful
  appraisal record.
- `_SURFACE_REPAIR_PROMPT`, `GOAL_COGNITION_REPAIR_PROMPT`,
  `REQUIRED_SELECTION_GOAL_REPAIR_PROMPT`,
  `workspace._middle_truncate_text`, the Chinese-keyed `roles` projection, and
  its dead filter are absent from the source tree.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md` and `docs/HOWTO.md`
  describe the final contracts, with no stale cap or token value.
- Every deterministic command passes, every static grep returns its expected
  result, and the only remaining deviation from the Stage 0 baseline is the
  intended one.
- Every live LLM case was run individually, its artifact inspected, and its
  behavior judged acceptable.
- The independent code review has no unresolved findings.
- `Execution Evidence` is complete and `development_plans/README.md` reflects
  the final status.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A lower `max_completion_tokens` truncates a legitimate long output | Every value exceeds its stage's own contract maximum; the smallest value serves the stage whose output is capped at ~700 characters | Live LLM inspection confirms no truncation; deterministic tests assert no value below the contract maximum |
| The 120 s timeout cuts a slow but healthy local call | Bounds are 10.0-600.0 and the value is environment-overridable without a code change | Live LLM inspection records per-call latency against the timeout |
| Identity reduction removes character-defining context and flattens voice | `boundaries`, `mbti`, and identity scalars never reduce; identity reduces last in goal cognition; floors retain 600 characters of backstory and 400 of self-concept | Deterministic tests assert the never-reduced fields; live character-judgment cases are inspected for voice drift |
| Goal degradation softens a boundary gate | Degradation touches only two handle lists; `relational_willingness` and every prose field stay fatal | A dedicated test asserts an invalid `relational_willingness` still raises |
| Reusing the initial system prompt on repair confuses the model about which fields to return | `repair_feedback` still carries the exact required field list, field types, allowed handles, and validation error | Deterministic tests assert both the system prompt identity and the feedback content |
| Dropping a stale bid silently loses a legitimate motive | Only non-ordinary bids whose goal id is absent from final state are dropped, each with a named warning, and the unfiltered list stays in the trace | A test asserts the warning, the trace content, and that a live-goal bid is never dropped |
| Individual planner row dropping hides a systematic contract failure | Every drop is logged, and an all-invalid set still raises once so one repair attempt runs | A test asserts both the partial-drop log and the all-invalid raise |
| Lockstep reduction removes a handle the question still needs | Reduction only shrinks; `_validate_question_handle_authority` still passes; `self` and `current_user` are never in a reducible list | A test asserts `self` and `current_user` survive full reduction |
| Editing Chinese prompt literals corrupts encoding | Single-quote delimiters, byte-exact extraction for retained sentences, and immediate `py_compile` on every edited file | `py_compile` after every prompt edit; prompt content assertions |
| Raising caps pushes a stage over the 50k window | The budget table states every stage's worst-case ceiling and margin under a pessimistic character-as-token bound | Deterministic exact-cap tests plus live LLM inspection |
| Twenty stages drift from the plan across sessions | One stage sign-off at a time, mandatory full-plan reread after each major stage and after compaction, and a final independent review | `Progress Checklist` sign-off lines and `Execution Evidence` |

## Execution Evidence

Populate during execution. Do not pre-fill.

- Pre-edit git state:
- Stage 0 deterministic baseline result:
- Pre-existing failures confirmed:
- Stage 1 expected-failure evidence:
- Config and route grep results:
- `py_compile` results per edited file:
- Stage 4 maximum-identity pass evidence:
- Stage 5 provenance-overflow pre-fix error text and post-fix result:
- Stage 6 degradation matrix results:
- Stage 7 stale-bid warning and trace evidence:
- Stage 8 surface repair request evidence:
- Stage 9 goal regeneration request evidence:
- Stage 10 actionable error samples:
- Stage 11 lockstep reduction evidence:
- Stage 12 prompt skeleton and length evidence:
- Stage 13 planner containment evidence:
- Stage 14 surface reduction evidence:
- Stage 15 policy and projection evidence:
- Documentation grep results:
- Full deterministic regression result:
- Live LLM commands, artifact paths, routes, models, attempts, latency, and
  per-case judgment:
- Observed latency change versus baseline:
- Independent reviewer identity and files reviewed:
- Review findings, fixes applied, and commands rerun:
- Residual risks:
- Final `git diff --stat`:
- Final sign-off:
