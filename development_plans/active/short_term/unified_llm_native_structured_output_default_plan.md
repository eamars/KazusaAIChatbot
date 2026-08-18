# Unified LLM Native Structured Output Default Plan

## Plan Status

- **Status:** `draft`
- **Decision date:** 2026-08-19
- **Plan type:** Decision record and pre-impact implementation contract
- **Execution authorization:** None. This draft records the policy decision; it does not authorize production-code, prompt, schema, or test changes.
- **Parallel boundary:** `development_plans/active/short_term/cognition_v3_cache_affine_semantic_chain_bigbang_plan.md`
- **Cutover strategy:** Native structured output becomes the default while the existing fallback, validation, and JSON-repair behavior remains available and bounded.

## Summary

This plan records the decision to move structured-output stages from prompt-level
soft fences to provider-native structured output through the unified LLM
interface. The decision applies across models and providers rather than only to
the model that motivated this investigation.

The plan intentionally stops before source mapping and impact-radius analysis.
Those artifacts are mandatory before this plan can become an executable
implementation plan, but they are not authored in this decision-record phase.

## Binding Decisions

The following decisions are made and recorded as the policy for the later
implementation phase.

### 1. Native structured output is the default for all models

- Every call that has a structured-output contract uses the unified LLM
  interface's native structured-output request by default.
- The default is model- and provider-independent. A model does not remain on a
  prompt-only soft fence merely because it is weaker, newer, or configured by a
  different route.
- A call that intentionally has a free-form contract remains free-form. “All
  models” means that every model capable of serving a structured-output stage
  receives the same native-default policy; it does not force an artificial JSON
  schema onto final prose or other intentionally free-form surfaces.
- Native transport shape is owned by the structured-output contract supplied to
  the interface. Semantic judgment, grounding, and character/business meaning
  remain owned by the producing LLM stage and its semantic instructions.

### 2. Prompts stop duplicating the wire-format contract

- Structured prompts remove hard-coded JSON object examples and other
  schema-shaped examples whose purpose is to teach serialization.
- Prompts retain the semantic explanations needed for the stage to make the
  correct decision: field meaning where useful, decision criteria, grounding
  requirements, boundaries, refusal or silence meaning, and other domain
  semantics.
- Prompts no longer require global boilerplate such as “no markdown fence”,
  “return only JSON”, or a field-by-field reproduction of the exact wire
  format merely to enforce serialization. The native contract owns that
  responsibility.
- The implementation agent has explicit case-by-case judgment over the
  remaining prompt wording. It may express semantic concepts in the form that
  best serves each stage, while preserving the native contract and the
  semantic behavior under test.

### 3. Every structured prompt receives a real-LLM regression test

- Each structured-output prompt must be exercised through a real LLM test using
  the actual unified interface and a real configured model route.
- Mocked or patched model tests may verify deterministic plumbing, but they do
  not satisfy the prompt-behavior requirement.
- The live test must compare the post-change behavior with a captured baseline
  for the same prompt case and model route. The comparison is semantic and
  contract-oriented rather than byte-for-byte wording equality.
- The later test inventory must cover representative normal, boundary, and
  failure-sensitive cases for each prompt. A prompt is not considered migrated
  because only one unrelated prompt has passed.
- Live cases are run one at a time, inspected, and recorded as durable review
  artifacts with enough context to determine whether the behavior drifted.

### 4. Structured output is a first-class unified-interface capability

- The unified LLM interface exposes a provider-neutral way for a caller to
  supply a structured-output contract.
- The interface owns translation of that contract to the selected provider's
  native request mechanism. Callers do not create provider-specific structured
  output clients or prompt-format shims.
- The interface preserves the response information required by the existing
  parser, checks, repair path, diagnostics, and stage evaluator.
- The exact signature, schema representation, provider feature matrix, and
  source ownership are implementation details for the deferred impact-mapping
  phase. The provider-neutral capability and native-default policy are binding
  now.

### 5. Existing fallback, checks, and JSON repair are preserved

- The current fallback behavior remains available after native structured output
  becomes the default.
- Existing structural parsing, deterministic cleanup, contract checks,
  semantic validation, normalization, error classification, bounded retry or
  regeneration behavior, and JSON repair remain part of the response path.
- Native output is still subject to the existing checks. Native transport
  success is not treated as semantic validation success.
- Existing fallback triggers, attempt caps, repair ownership, failure
  dispositions, and fail-closed behavior are preserved unless a separately
  approved plan explicitly changes them.
- JSON repair remains structural recovery. It may repair permitted syntax or
  object-shape defects according to the existing routine, but it does not invent
  semantic values, change a domain decision, or bypass later validation.
- The implementation must retain evidence of whether a result used the native
  path, the preserved fallback, or repair, so regressions in recovery behavior
  remain diagnosable.

## Decision Rationale

Prompt-level JSON examples and soft fences communicate an intention but do not
reliably constrain the model's transport output. They also duplicate the shape
contract, consume prompt space, and create drift between prompt examples and
the actual evaluator.

Native structured output gives the interface/provider boundary ownership of
serialization and substantially reduces markdown-fence and malformed-shape
failures. It does not replace semantic instructions or domain validation, so
the plan keeps semantic explanations and the existing deterministic recovery
path. Real-LLM baseline tests are required because native shape compliance does
not prove that prompt meaning, grounding, silence, refusal, or other semantic
decisions remain stable.

## Scope For This Draft

### In scope now

- Recording the five binding decisions above.
- Defining the intended ownership boundary between semantic prompts, native
  structured transport, and existing validation/recovery.
- Defining the evidence and approval gates required before implementation.
- Registering this plan as a separate draft alongside the parallel Cognition V3
  plan.

### Explicitly deferred now

- Source-file, symbol, call-site, route, schema, and prompt inventory.
- Impact-radius analysis and ownership assignment to exact files.
- Exact pytest node IDs and the source-to-test traceability matrix.
- Provider capability mapping, unsupported-feature behavior details, and the
  concrete unified-interface signature.
- Prompt edits, production-code edits, schema implementation, test additions,
  and rollout execution.
- Any modification, supersession, or status change to the parallel Cognition V3
  plan.

The deferred items are execution gates, not policy questions. The five policy
decisions are closed by this document; implementation mechanics remain open
until the required mapping is performed.

## Parallel-Work Coexistence Rule

This plan is independent of the Cognition V3 cache-affine semantic-chain plan
and does not authorize edits to its files or contracts. While that work is in
progress, an implementation agent must first refresh the source and impact
map against the actual shared workspace state. Any overlap with Cognition V3
must be resolved through an explicit sequencing decision or plan amendment
before production changes are made.

The native structured-output policy is the cross-cutting target state for the
later implementation, but this draft does not assume which Cognition V3
contracts, prompts, or interface surfaces will exist after its work lands.

## Target State Contract

The later implementation must produce this boundary:

```text
semantic stage instructions + native structured contract
        -> unified LLM interface
        -> provider-native structured request (default)
        -> existing parser/check/repair and stage evaluation
        -> typed stage result or bounded failure
```

The contract has four distinct responsibilities:

| Responsibility | Owner | Required behavior |
|---|---|---|
| Semantic meaning and decision criteria | Producing LLM stage | Explain what the result means and how to decide it; keep grounding and domain semantics. |
| Wire shape and serialization | Native structured contract plus unified interface | Request the declared shape natively for every structured-output model call. |
| Structural and domain acceptance | Existing deterministic checks and owning evaluator | Continue to validate, normalize, reject, regenerate, or fail closed under current rules. |
| Recovery and diagnostics | Existing fallback/repair boundary | Preserve current fallback and JSON-repair behavior and make the disposition observable. |

## Mandatory Skills And Rules For Later Execution

- Apply `development-plan` for promotion, ownership, gates, and execution
  evidence.
- Apply `local-llm-architecture` to preserve the boundary between native shape,
  semantic prompting, deterministic validation, and bounded recovery.
- Apply `debug-llm` to create human-readable live-LLM comparison artifacts and
  inspect semantic drift.
- Apply `test-style-and-execution` for all test changes and execution. Run live
  LLM cases one at a time and inspect each result.
- Apply `py-style` before writing or reviewing Python implementation or test
  code.
- Keep the native request model-agnostic and provider-neutral at the unified
  interface boundary.
- Keep semantic ownership with the producing LLM stage and deterministic
  ownership with validation, limits, persistence, and recovery code.
- Preserve the existing fallback, checks, JSON repair, attempt caps, and
  fail-closed outcomes.
- Do not authorize production changes from this draft alone. Promotion to
  `approved` or `in_progress` requires the deferred impact map and an explicit
  implementation authorization.

## Later Execution Roles And Handoffs

Exact file ownership is intentionally deferred. The later executable plan must
assign these boundary owners to exact repository files and symbols before any
production edit:

| Role | Boundary responsibility | Required handoff evidence |
|---|---|---|
| Unified-interface owner | Add the provider-neutral native structured-output capability and preserve response/recovery compatibility. | Interface contract, provider-routing evidence, native/fallback disposition evidence, and focused tests. |
| Prompt-contract owner | Remove serialization examples and fence boilerplate while retaining semantic explanations, using case-by-case wording. | Prompt diff, semantic intent review, baseline comparison, and live test artifacts for every structured prompt. |
| Validation/recovery owner | Confirm existing checks, fallback, normalization, JSON repair, caps, and fail-closed outcomes remain unchanged in behavior. | Recovery-path regression evidence, including native success, native failure/unsupported, malformed output, and repair cases. |
| Live-LLM verification owner | Run and inspect real model cases and compare them with the pre-change baseline. | One-at-a-time run log, durable raw/normalized artifacts, semantic drift disposition, and review summary. |
| Independent reviewer | Review cross-cutting ownership, parallel-plan overlap, contract compatibility, and acceptance evidence. | Written approval or a bounded list of required corrections. |

No role may expand the scope into Cognition V3 or alter fallback semantics
without an explicit plan amendment and owner decision.

## Test Impact And Traceability

The exact source-to-test matrix is intentionally omitted from this draft at the
owner's direction. Adding guessed paths, symbols, or pytest node IDs would
violate the deferred-impact boundary.

Before this plan can be promoted to an executable implementation contract, the
implementation owner must add one row for every affected source or governed
artifact. Each row must name:

1. the exact repository-relative source or prompt artifact;
2. the exact symbol or contract;
3. the semantic owner;
4. the exact pytest node ID or live-test case identifier;
5. the test mode (`regular`, `live-LLM`, or `live-DB` where applicable); and
6. the regression prevented or behavior proven.

The matrix must include every structured prompt, the unified-interface native
request path, the preserved fallback path, existing checks, and JSON repair.
The real-LLM requirement in Decision 3 applies to each prompt row; a mocked
test alone cannot close that row.

## Promotion And Execution Gates

### Gate 0 — Decision record

- This document is registered as `draft`.
- All five binding decisions are explicit.
- The parallel Cognition V3 boundary is named.
- No source map, impact-radius claim, or production change is made.

### Gate 1 — Pre-implementation impact mapping

- The parallel-plan workspace state is inspected before assigning ownership.
- The exact source/prompt/schema/interface inventory is complete.
- The exact test-impact and traceability matrix is complete.
- Shared-surface sequencing with Cognition V3 is resolved.
- A later status change to `approved` or `in_progress` is explicitly authorized.

### Gate 2 — Implementation contract

- The unified interface exposes the provider-neutral native structured-output
  request.
- Native structured output is the default for every structured-output model
  call across all models/providers.
- Prompt serialization examples and fence boilerplate are removed while
  semantic explanations remain.
- Existing fallback/check/repair behavior has a locked baseline and an
  explicit compatibility test set.

### Gate 3 — Behavioral verification

- Every structured prompt has a real-LLM test through the unified interface.
- Each live case is run individually, inspected, and recorded.
- Post-change behavior remains within the approved semantic baseline, including
  grounding, refusal/silence, literal fidelity, and contract interpretation.
- Native, fallback, check-failure, and JSON-repair dispositions are observable
  and match the preserved behavior contract.

### Gate 4 — Independent review and handoff

- The exact source/test matrix is reviewed.
- No prompt has regained a JSON example or fence instruction solely as a hidden
  replacement for native shape enforcement.
- No fallback, check, repair, cap, or fail-closed rule was weakened.
- The plan records implementation commit(s), test commands, live artifacts,
  reviewer disposition, and final status before archival or continued work.

## Acceptance Criteria For The Later Implementation

The implementation is acceptable only when all of the following are true:

1. Structured-output stages use native structured output by default through the
   unified LLM interface for every model/provider route in scope.
2. The prompt set no longer relies on hard-coded JSON examples or mandatory
   markdown-fence instructions for serialization, while its semantic
   explanations remain effective and reviewed case by case.
3. Every structured prompt has inspected real-LLM evidence showing no
   unacceptable semantic drift from its captured baseline.
4. The unified interface carries the structured contract without
   provider-specific call sites or stage-local transport shims.
5. Existing fallback, checks, normalization, bounded retry/regeneration,
   JSON-repair, diagnostics, and fail-closed behavior remain preserved.
6. The exact source/test traceability matrix and implementation evidence are
   complete before the plan is considered executable or complete.

## Current Handoff State

- **Decision:** recorded and closed.
- **Plan registration:** complete in `development_plans/README.md`.
- **Source and impact mapping:** deliberately deferred.
- **Production implementation:** not started and not authorized by this draft.
- **Next authorized planning action:** after the parallel Cognition V3 work is
  sufficiently stable, perform the exact impact mapping and propose the
  executable-plan promotion for approval.
