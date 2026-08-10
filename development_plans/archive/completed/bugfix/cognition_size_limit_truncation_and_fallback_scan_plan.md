# cognition size-limit truncation and fallback scan plan

## Summary

- Goal: scan every runtime context-size boundary and replace recoverable
  size-only rejection with deterministic truncation, row trimming, or optional
  context removal before the owning contract is validated.
- Status: completed
- Scope boundary: cognition input projections, adjacent prompt/context
  builders, persisted semantic-text producers, and their deterministic tests.
- Change direction: fit bounded context at the producer boundary, preserve
  required semantic fields and provenance, and keep size-only overflow out of
  fail-closed error paths. Every producer must publish a packet within its
  declared budget, and every consumer must apply the same owner-specific
  deterministic fit as a last-resort guard before strict validation.
- Acceptance state: the Cognition Core V2 producer/consumer contract is
  implemented, verified, and covered by deterministic and real-LLM tests.

## Context

The reviewed runtime correlation id is
`chat:qq:ch_1f677493d7a52025:438485259`, with trace id
`llmtrace_175868f4ff924dfd8832229a600eea9f`. Its brain-service event failed
with `CognitionContractError: relationship operational context is oversized`
before a Cognition Core V2 model call. The original 914-character incident
shape is also recorded under
`chat:qq:ch_732699d6699040ae:1938352034`: two model-authored causal rows were
present, and trimming the offending resolved event description from 109 to 80
characters reduced the verified projection to 885 characters.

The live regression fixture uses the exact reviewed correlation and trace as
source evidence, but the relationship packet is an explicitly labelled
synthetic reconstruction of the 914-character shape because the protected
trace export contains metadata rather than the full Cognition V2 input packet.

The hard relationship-context guard and its producer were introduced by
`32d59aeb` on 2026-08-03. The Aug 5 context-fade/sleep and prompt changes did
not alter this boundary. The failure occurred before a Cognition Core V2
model call, so model-route and timeout settings were not the immediate cause.

## Confirmed Decisions

| Topic | Decision |
|---|---|
| Size-only overflow | The producer fits the packet before publication; the consumer repeats the owner-specific deterministic fit as a last-resort guard before validation. |
| Producer output invariant | No producer returns or persists a recoverable context packet whose canonical serialized size exceeds its declared consumer budget. |
| Consumer recovery | A consumer receiving an oversized recoverable packet truncates bounded free text and drops optional rows in the declared order before invoking strict validation. |
| Cognition consumer boundary | `validate_cognition_core_input` applies the relationship and character operational-context fits to a deep copy before their strict validators, replaces the fitted nested mappings in its returned payload, and protects direct validator callers as well as the facade and graph builder. |
| Canonical fit implementation | Each owner has one canonical fit helper and one budget constant. Projection code and the validator call that helper; they do not maintain parallel trim logic or hard-coded limits. |
| Serialized-size definition | Size is `len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))`, matching the validator’s current representation. Tests cover CJK, combining text, and astral/emoji characters using Python’s decoded-string character count, not UTF-8 byte length. |
| Truncation primitive | Owner-specific reduction order remains mandatory, while bounded free-text reduction reuses `cognition_core_v2.prompt_budget.middle_truncate_text` and its existing marker. No owner-blind project-wide truncator is introduced. |
| Character digest fit | The final character operational packet, including the 64-character `context_digest` field, must fit. Use a fixed digest placeholder while fitting or an equivalent deterministic final-packet calculation, then recompute the digest and recheck the final size. |
| Post-fit invariant | If fields pass their individual structural bounds but the required envelope still cannot fit after permitted reductions, raise the existing typed `CognitionContextLimitError` as an implementation/resource invariant. Individual required-field bound violations remain structural `CognitionContractError` failures. |
| Semantic authority | The producing LLM remains the semantic owner; deterministic code may preserve, shorten, or omit bounded context but may not invent a replacement decision. |
| Required fields | Preserve exact identifiers, handles, enums, provenance, and required current-episode facts. |
| Optional rows | Drop rows in a declared deterministic priority order after text trimming. |
| Structural failures | Missing fields, wrong types, unsupported values, conflicting fields, and unrecoverable shape errors remain fail-closed. |
| Model/settings changes | No model, endpoint, timeout, or `.env` change is part of this plan. |
| Existing database rows | The incident row is already repaired. Bulk database migration is outside this plan and requires a separate approved scope. |
| Compatibility | Use one canonical fitted shape; add no aliases, dual schemas, or compatibility mapper. |

## Scope And Change Direction

First inventory all size checks and all callers that convert an over-budget
payload into rejection, blocking, an empty result, or a typed operational
failure. Classify each check as transport/prompt context, persisted semantic
text, required semantic output, or operational resource limit.

For recoverable prompt/context data, the owning producer performs this fixed
fit sequence before returning or persisting the packet:

1. Serialize with the same deterministic encoding used by the validator and
   record the original character count.
2. Trim bounded free-text fields using the existing
   `middle_truncate_text` primitive and a field-specific Unicode character
   budget.
3. Remove lowest-priority optional rows or lanes while preserving required
   current-turn facts, handles, provenance, and target identity. For
   relationship context, rows are already ordered by salience and recency;
   remove from the end of that stable order, causal rows before affect rows,
   with the existing deterministic tie-breaks.
4. Re-serialize and repeat the declared deterministic reductions until the
   budget is met.
5. Record the final size, trimmed fields, dropped rows, and fallback level in
   bounded diagnostics; pass only the fitted shape to the model and validator.

At each consumer boundary, apply the same owner-specific fit to a deep copy of
the received packet before strict validation. For Cognition Core V2 this
consumer pass is inside `validate_cognition_core_input`; the fitted nested
mapping must be placed into the returned payload so the facade and downstream
prompt projection cannot receive the original oversized mapping. This
defensive pass covers legacy persisted rows, direct callers, and future
producers that fail to uphold the producer invariant. The consumer must
preserve identifiers, handles, enum values, timestamps, provenance, and
required current-turn facts while trimming only declared free text or
removing optional context. It must not mutate the caller’s mapping, invent
semantic content, or invoke an LLM to repair size. The final size check remains
an invariant of both fit passes, not a data-driven rejection path. If fields
pass their individual structural bounds but required structure remains over
the aggregate cap, raise `CognitionContextLimitError`; individual field-bound
violations remain structural contract errors. The fit result retains bounded
original/final sizes, trim/drop fields, and fallback level for logging and
deterministic inspection; these diagnostics are not added to the public
cognition input schema. No runtime turn may fail solely because recoverable
context text is over its aggregate limit.

The character operational fit must include the digest field in its final-size
calculation. A fixed 64-character digest placeholder may reserve that space
before row reductions; the actual digest is then recomputed and the complete
packet is checked again.

## Mandatory Skills

Execution must read and apply these skills before changing the corresponding
surface:

- `.agents/skills/development-plan/SKILL.md`
- `.agents/skills/local-llm-architecture/SKILL.md`
- `.agents/skills/llm-trace-debug/SKILL.md`
- `.agents/skills/debug-llm/SKILL.md`
- `.agents/skills/py-style/SKILL.md`
- `.agents/skills/cjk-safety/SKILL.md`
- `.agents/skills/test-style-and-execution/SKILL.md`
- `.agents/skills/database-data-pull/SKILL.md` for read-only persisted-data
  inventory or incident evidence.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python and pytest.
- Keep the live response path bounded and inspectable.
- Keep deterministic validation, persistence, limits, and diagnostics in code;
  do not add an LLM repair call for context packing.
- Do not truncate identifiers, evidence handles, enum values, timestamps, or
  required semantic fields. Increase the fixed envelope budget or remove
  optional context when those fields alone approach the cap.
- Do not add a generic project-wide, owner-blind string truncator that hides
  semantic ownership. Each owner defines its field order, row priority, and
  fallback diagnostics, while bounded text reduction reuses
  `prompt_budget.middle_truncate_text`.
- Apply canonical JSON parsing before evaluating raw model output. Normalize
  only recoverable size bounds, then run the existing strict contract.
- Preserve raw model output and normalized size-recovery evidence in protected
  diagnostics when capture is enabled; never put secrets in the artifact.
- Keep unrelated worktree changes and the repaired database row intact.
- Run deterministic tests in batches. Run any live LLM case one at a time and
  inspect its human-readable artifact.

## Must Do

### 1. Build the project-wide inventory

Search source, tests, and subsystem documentation for:

- `oversized`, `exceeds`, `hard cap`, `context limit`, `ContextLimitError`,
  `over_hard_cap`, `MAX_*_CHARS`, `prompt_budget`, `json.dumps`, and serialized
  length checks;
- every `raise` or blocked/empty-result branch reached only because a payload
  is too large;
- every producer that writes model-authored descriptions, summaries,
  conversation-progress text, evidence text, identity-growth candidates, or
  cognition operational context.

Create a matrix with owner, consumer, serialized budget, mandatory fields,
optional fields, current failure behavior, safe trim order, and test coverage.
The matrix must include at least:

- `cognition_core_v2/contracts.py` and `state_projection.py`;
- `nodes/persona_supervisor2_cognition.py` and Cognition Core V2 producers;
- `conversation_progress/recorder.py`, `repository.py`, and `runtime.py`;
- `character_identity_growth/projection.py`, `llm.py`, and `runner.py`;
- coding-agent context-budget owners and their stage callers;
- the corresponding tests, READMEs, HOWTO budget documentation, and trace
  diagnostics.

The inventory must explicitly classify existing fitting infrastructure as
already safe, recoverable-overflow, or a true resource guard. At minimum,
review `cognition_core_v2/prompt_budget.py`, conversation-progress caps,
identity-growth reductions, and coding-agent context budgets before adding or
moving helpers. Do not rewrite a boundary whose cap is a true resource limit.

### 2. Repair Cognition Core V2 fitting

- Add deterministic relationship-context fitting at the relationship
  projection boundary before `validate_cognition_core_input` sees the value.
- Put the canonical owner-specific fit helper and budget constant in one
  shared Cognition Core V2 budget owner, reusing
  `prompt_budget.middle_truncate_text`; import that helper from both the
  projection and validator without duplicating limits or trim order.
- Trim `causal_context.semantic_summary` first with the existing middle
  truncation marker, then drop the lowest-priority causal rows, then drop
  optional affect rows in the documented stable order if required.
- Preserve relationship axes, relationship identity, freshness labels, and
  the exact handle/provenance mapping.
- Route character operational context through the same kind of explicit
  pre-validation fit, including the final digest field, and verify that its
  existing row reductions cannot leave an over-budget value.
- Add the consumer-side last-resort fit for both operational context packets
  inside `validate_cognition_core_input`, operating on a deep copy and
  returning the fitted mappings. The consumer must recover an oversized
  packet supplied by a direct caller or legacy row using the same fixed
  trim/drop order and must complete validation without a size-only crash.
- Keep the final validator strict about shape and types. Its size assertion
  remains an invariant check after producer and consumer fitting, rather than
  the normal recovery mechanism.
- Preserve the existing structural failure behavior: an oversized packet with
  missing fields, wrong types, unsupported values, or conflicting fields must
  still raise `CognitionContractError` rather than being made valid by fitting.
- Emit bounded size-recovery diagnostics that identify the owner, original and
  final serialized sizes, limit, fields trimmed, rows dropped, and fallback
  level.

### Executed inventory classification

The project-wide scan grouped current size checks by owning contract and
classified each boundary before implementation. No remaining adjacent owner
was found to have the same recoverable Cognition V2 size-only rejection.

| Owner and boundary | Declared budget | Required/optional handling | Classification and disposition | Verification |
| --- | --- | --- | --- | --- |
| Cognition V2 relationship operational context | 900 serialized chars | Identity, axes, freshness, provenance required; causal summaries and affect rows reducible | Recoverable context overflow; canonical producer and consumer fit implemented | 914/900/901, Unicode, row-drop, builder, consumer, live tests |
| Cognition V2 character operational context | 1200 serialized chars including 64-char digest | Identity/axes and digest integrity required; pressures/affect reducible | Recoverable context overflow; canonical producer and consumer fit implemented | Digest boundary, re-fit stability, consumer, builder tests |
| Cognition V2 stage prompt budgets | Owner-specific stage caps | Existing stage reductions preserve required evidence and use typed fallbacks when required context cannot fit | Mixed: existing recoverable stage fitting is already safe; true required-stage/resource failures remain typed | `test_cognition_core_v2_prompt_budget_continuity.py`, failures batch |
| Persona supervisor Cognition V2 input builder | Delegated to the two operational-context budgets | Must publish the fitted nested packets | Producer boundary delegated to canonical Cognition V2 fits; no second vocabulary added | `test_cognition_core_v2_context_size_input_builder.py` |
| Conversation-progress recorder scene/event prompts | 8,000 scene / 24,000 event serialized human-payload chars | Optional historical turns are removed; accepted turn and complete prior event ledger are required | Existing deterministic reduction is safe; required-ledger overflow is a true semantic/resource guard and remains typed | Recorder/repository/runtime tests and README contract |
| Conversation-progress projections and persistence | Field caps plus 16,000-char active packet cap | Field-specific normalization and compaction preserve stored identity/lineage | Persisted semantic/state contract and resource guard, not a safe text-only prompt packet; unchanged | Conversation-progress projection, compaction, repository, and cache tests |
| Character identity-growth projection/LLM | Configured identity prompt character budget; at most eight candidates | Older optional candidates are removed; current identity and evidence are preserved | Existing recoverable candidate removal is safe; required identity/evidence overflow is a true semantic prompt guard and remains `IdentityPromptBudgetError` | Identity-growth contract/prompt tests and README |
| Coding-agent prompt/context owners | 50,000-char / 42,000-token hard input caps plus stage targets | Files, observations, evidence rows, assignments, and excerpts are pruned by their stage owners | Operational resource limits and safety caps; stage-specific pruning already exists, and hard-cap failures remain typed/observable | Coding-agent context, reading, writing, action-loop, and source-intake tests |
| Persisted model-authored semantic text and trace diagnostics | Field-specific state contracts | Writers validate/normalize their owned fields; no historical rewrite | Existing persistence boundary; no avoidable Cognition V2 overflow remains after projection fit; no migration required | State/projection and trace artifact checks |

The scan covered `oversized`, `exceeds`, hard-cap/context-limit errors,
`over_hard_cap`, `MAX_*_CHARS`, prompt-budget helpers, canonical JSON length
checks, and blocked/empty-result branches in the listed owners. The remaining
typed failures are required semantic contract failures, malformed input,
provider failures, or true operational resource limits rather than
recoverable free-text overflow.

### 3. Repair adjacent recoverable context boundaries

For every inventory row classified as recoverable context-size overflow:

- implement the owner-specific trim/drop order;
- apply that fit before the packet is published by the producer;
- apply the same fit again at the consuming boundary before strict validation
  or model invocation;
- preserve the current accepted turn and mandatory ledger/provenance facts;
- keep optional context removal deterministic and observable;
- ensure the caller receives a valid bounded packet rather than a context-limit
  rejection, empty semantic result, or stale prior packet solely because text
  was too long;
- leave true resource exhaustion, provider failure, malformed structure, and
  missing required semantic decisions on their existing typed failure paths.

### 4. Protect persistence boundaries

Audit model-authored writes for descriptions and summaries that later feed a
bounded projection. Retain a field-specific persisted maximum where the state
contract requires one, but add deterministic normalization at the owning write
boundary so future model output cannot create an avoidable over-budget shape.
Do not rewrite unrelated historical rows or run a bulk data migration under
this plan.

### 5. Add regression and boundary coverage

Add tests for every repaired owner covering:

- exact 900/901-character boundary payloads and the 914-character incident
  shape;
- producer output is at or below the declared budget before publication;
- consumer input that is deliberately one or more characters over budget is
  fitted and accepted without a size-only exception;
- CJK, combining text, and astral/emoji Unicode with the canonical decoded
  string character count;
- multiple optional rows with stable salience/recency tie-breaking;
- long text plus long row lists in the same payload;
- fixed required fields with all optional context removed;
- recovery diagnostics, caller-input immutability, and final strict-contract
  acceptance;
- a structurally invalid oversized packet that still raises the existing
  structural contract error;
- the character operational context digest-after-fit boundary;
- the incident fixture: 914 characters before fitting and at most 900 after
  fitting at both producer and consumer boundaries, with no
  `relationship operational context is oversized` failure;
- deterministic coverage in `test_cognition_core_v2_operational_projection.py`,
  `test_cognition_core_v2_contracts.py`, and
  `test_cognition_core_v2_prompt_budget_continuity.py` (or focused new modules
  when their fixtures do not fit);
- one real-LLM Cognition Core V2 case using the synthetic incident-shaped
  fixture, with source provenance labelled as reconstruction, that proves the
  fitted context reaches the goal/workspace bidding path and records raw model
  output, parsed output, route, and human review data. The live gate is
  structural (`no failure`, expected stage reached, and artifact present); a
  particular `admitted_bid` is review evidence, not a pass/fail requirement.

Add one deterministic integration test through
`build_cognition_input_from_global_state` so the production caller is covered,
not only a hand-built payload helper. Assert the returned relationship packet
is fitted and bounded after the builder returns.

### 6. Update operational documentation

Document that size-only overflow is handled by deterministic fitting, list the
owner-specific reduction order, and distinguish recoverable context overflow
from structural contract errors and external provider/resource failures.
Update the short-horizon operational-context section in
`src/kazusa_ai_chatbot/cognition_core_v2/README.md` and the Cognition Core V2
budget section in `docs/HOWTO.md`, including the 900/1200 packet caps,
producer/consumer fit contract, digest accounting, and typed post-fit
invariant behavior.

## Deferred

- No model or endpoint replacement.
- No `.env` or route-setting change.
- No blanket removal of contract validation.
- No LLM summarizer, verifier, classifier, or repair stage for truncation.
- No bulk database rewrite, historical semantic cleanup, or automatic
  migration of all existing user profiles.
- No changes to character judgment, willingness, response gating, or dialog
  wording beyond receiving a valid fitted context.
- No unrelated prompt-budget redesign in a subsystem whose inventory proves
  its size cap is an operational resource guard rather than recoverable
  context overflow.

## Target State

```text
model/persistence producer
  -> canonical parse or validated state
  -> owner-specific deterministic fit
       -> trim free text
       -> drop optional rows/lanes by fixed priority
       -> record bounded recovery diagnostics
  -> consumer-side owner-specific defensive fit inside the canonical validator
       -> replace fitted nested packet in returned validated payload
  -> strict shape/type/semantic validation
  -> model call or downstream stage
```

No recoverable context-size violation reaches the live graph as a rejection.
Required semantic decisions remain model-authored, and deterministic code only
normalizes bounded transport/context representation.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`: own bounded
  relationship and character operational context fitting.
- `src/kazusa_ai_chatbot/cognition_core_v2/prompt_budget.py`: provide the
  canonical middle-truncation primitive and shared owner-specific budget
  helper surface without duplicating limits.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`: fit a deep copy at
  the validator boundary, return the fitted result, and retain strict shape
  and type checks plus the typed post-fit invariant.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: verify the
  production input builder receives the fitted projection and add no second
  fit vocabulary.
- affected subsystem READMEs, `docs/HOWTO.md`, and focused tests.

The adjacent owners were reviewed but left unchanged because their remaining
hard failures are required semantic-ledger limits, required identity/evidence
prompt limits, or true coding-agent resource/safety caps. Their existing
owner-specific optional-row pruning remains in force; none matched the
recoverable free-text overflow fixed here.

### Create

- Focused deterministic tests or fixtures for each repaired owner when an
  existing test module cannot express the boundary matrix cleanly.
- A deterministic Cognition Core V2 input-builder integration regression for
  the relationship-context overflow.
- A Cognition Core V2 live-LLM incident-shaped regression test for the
  relationship-context overflow and bidding path. Its metadata must state
  that the packet is a synthetic reconstruction from the reviewed trace.
- A human-readable debug review artifact for the incident fixture and the
  post-fix integration result.

### Delete

- No files or persisted collections.

### Keep

- `llm_trace_runs` and `llm_trace_steps` capture behavior and recovery
  metadata according to their existing retention/capture mode.
- Strict structural/semantic contract failures and external provider/resource
  failures.
- Current model routes, service settings, adapter contracts, and database
  ownership boundaries.

## Agent Autonomy Boundaries

The implementation agent may choose local helper names, fixture decomposition,
and command order within the listed owners. It must keep the fixed
trim/drop order, required-field preservation, canonical validator placement,
copy-and-return behavior, digest accounting, diagnostics, and acceptance
criteria unchanged. It may not introduce parallel fit helpers, an
owner-blind truncator, or an `admitted_bid` live-test gate.

The implementation agent must request a plan amendment before changing public
schemas, changing semantic authority, adding an LLM call, changing model or
environment settings, introducing a compatibility layer, or expanding into a
database migration.

## Verification

- Run the inventory search and retain the complete owner matrix.
- Run focused deterministic tests for every changed owner, including the
  exact-boundary/Unicode/row-drop/structural-negative matrix and the
  `build_cognition_input_from_global_state` integration test.
- Run the incident fixture through the production input builder and verify a
  fitted context at or below its declared limit before publication.
- Pass a deliberately oversized incident-shaped packet through the consumer
  boundary and verify deterministic fitting, strict acceptance, and no
  size-only failure.
- Verify the consumer returns a fitted copy and leaves the caller’s original
  mapping unchanged.
- Verify the final character operational packet, including its digest, is at
  or below 1200 characters.
- Inspect trace/diagnostic output for original size, final size, trim/drop
  reason, and absence of size-only failure disposition.
- Run the normal non-live test batch for the affected modules.
- Run a live LLM regression case only when the changed boundary reaches a live
  prompt; execute and inspect one case at a time.
- Verify `/health` and `/ops/runtime-status` after deployment/restart when the
  implementation is eventually executed.

## Execution record

- 2026-08-06: DeepSeek implementation handoff supplied the canonical fit
  implementation; parent review corrected digest stability, required-field
  error classification, and minimum-loss summary fitting.
- 2026-08-06: Independent DeepSeek alignment review inspected the final source
  and plan, found no remaining Cognition V2 root-cause defect, and verified the
  affected deterministic batch. Its surfaced issues were consumed in the
  implementation and tests.
- 2026-08-06: `py_compile` passed for changed source/tests.
- 2026-08-06: focused boundary batch passed, 56 tests; the affected
  deterministic Cognition V2 batch passed, 144 tests.
- 2026-08-06: producer and consumer real-LLM cases passed separately; both
  reached goal bidding and action planning with parsed outputs (12 producer
  calls, 11 consumer calls), with no failure disposition. Human-readable
  review evidence is recorded at
  `test_artifacts/llm_traces/cognition_core_v2_context_size_live_llm__review.md`.

## Acceptance Criteria

- Every current project size-limit rejection in the scanned runtime owners is
  listed and classified in the executed inventory matrix.
- Every recoverable context-size path fits deterministically before producer
  publication and again before model or downstream contract invocation.
- No producer emits an oversized recoverable packet.
- The consumer recovers a deliberately oversized incident-shaped packet
  without a size-only crash and keeps required relationship identity, axes,
  freshness, and provenance intact.
- `validate_cognition_core_input` is the canonical Cognition V2 consumer fit
  boundary and returns the fitted nested packets to all callers.
- The incident fixture no longer raises a size-only contract error and keeps
  required relationship identity, axes, and provenance intact.
- A structurally invalid oversized packet still raises the typed structural
  contract error, while an irreducible required-only overflow raises the typed
  context-limit invariant.
- Character operational context remains within 1200 characters after digest
  insertion, not only before digest insertion.
- No truncator invents semantic content or changes a model-authored decision.
- Structural, semantic, provider, and true resource failures remain typed and
  observable.
- Focused tests cover exact-boundary, overflow, Unicode, row-trim, and
  required-field cases for every modified owner.
- Documentation and human-readable debug evidence describe the final behavior.

The original incident failure is no longer a live Cognition V2 error path:
recoverable relationship overflow is fitted at both producer and consumer
boundaries, while malformed packets and true required/resource failures retain
their typed dispositions.

## Cutover Policy

Overall strategy: bigbang.

- Runtime context fitting, validators, tests, and documentation move to the
  canonical truncation/trim behavior together.
- The previous size-only rejection paths are removed rather than aliased.
- Rollback is a source revert of the complete implementation; no runtime flag
  or dual behavior is added.
