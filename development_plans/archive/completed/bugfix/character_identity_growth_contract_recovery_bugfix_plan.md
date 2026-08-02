# character identity growth contract recovery bugfix plan

## Summary

- Goal: raise routed identity-growth success to at least 95% on the frozen 185-episode Asuna replay cohort, including at least 40/42 recovered historical failures, while preserving valid semantic rejection as a successful no-change or rejected outcome.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `test-style-and-execution`, `debug-llm`; load `cjk-safety` before any Python edit that adds CJK string content.
- Overall cutover strategy: bigbang replacement of the model-facing proposal/review wire contract; preserve the existing internal policy, candidate, revision, and run contracts.
- Highest-risk areas: proposal-stage output burden, semantic no-change fidelity, replay-cohort comparability, bounded retry context, and real local-LLM behavior.
- Acceptance criteria: at least 176/185 replayed episodes and at least 40/42 historical failure roots reach a valid semantic disposition within the bounded attempt cap; no expected rejection ends as a contract failure; no candidate or revision is created from an invalid stage result; affected deterministic and one-at-a-time live-LLM gates pass.

## Context

The Asuna review found 189 identity-growth runs, including 42 failed runs (22.2%). Thirty-nine failed during proposal regeneration and three failed during review regeneration. The failed trace steps were contract errors; no provider-error family was observed. Failed runs created no candidate or revision, but they consumed the three-attempt cap and remained pipeline failures instead of valid semantic outcomes.

The routed episode cohort is 185 runs: 143/185 reached a valid end-to-end disposition (77.30%), 146/185 completed proposal within the attempt cap (78.92%), and the review succeeded in 143/146 valid-proposal cases (97.95%). Proposal first-attempt validity was 80/185 (43.24%). The target is 176/185 or better end-to-end, with 40/42 or better in the historical failure subset and 42/42 as the zero-terminal-failure goal. The four daily-reflection records are outside this routed episode denominator.

The July 30–August 2 post-alias cohort still passed only 114/149 runs (76.51%), with 34 proposal failures and one review failure. This establishes that short string aliases and provenance handling are secondary improvements; the dominant bottleneck remains proposal-stage output burden and cross-field contract coupling.

The observed failure families were:

- a single semantic episode produced patches that were no-ops against the current identity;
- long evidence or candidate identifiers were omitted, copied incorrectly, or invented by near-string reproduction;
- the model omitted or mistyped `value_kind` or the matching replacement field;
- `corroborate_candidate` was emitted with a null or invalid candidate reference;
- the review stage reconstructed `accepted_changes` and copied explanatory prompt wording into the JSON object;
- retries retained the rejected model object and appended repair text, allowing the same malformed structure to anchor later attempts.

The evidence artifact is `test_artifacts/diagnostics/asuna_character_identity_growth_failed_review_20260802.md`. The prior completed plan is `development_plans/archive/completed/bugfix/cognition_core_v2_character_identity_growth_bigbang_plan.md`; it remains closed. This plan is a separate reliability follow-up for the semantic-stage contract and retry boundary. Its scope includes the existing Step K retry behavior because the new failure evidence shows that retained invalid output still permits repeated contract failure.

The current ownership boundary is:

- `projection.py` supplies current identity, evidence, and candidates;
- `llm.py` owns proposal/review prompts, model calls, canonical parsing, retries, and stage results;
- `validation.py` validates closed model output and maps it to the internal decision shape;
- `policy.py` owns semantic disposition and candidate/revision eligibility;
- `runner.py` owns run persistence and fail-closed mutation boundaries.

The 42-root failure subset contains 35 QQ/private roots and seven debug roots. Seven debug traces retain full protected output; QQ traces are metadata-only. The cohort gate therefore measures contract recovery on prompt-safe reconstructed inputs, with semantic-quality and privacy behavior measured separately by the ten existing live cases.

Adjacent improvements intentionally remain outside this plan: changing the consolidation model, tuning character-growth semantics, changing evidence eligibility, changing the attempt cap, redesigning the database schema, changing cognition/dialog consumers, and adding new background agents.

## Mandatory Skills

- `development-plan`: load before changing plan status, executing any stage, recording evidence, or closing the plan.
- `local-llm-architecture`: load before changing prompts, model-facing contracts, retries, or semantic-stage ownership. Preserve LLM semantic judgment and deterministic structural validation.
- `py-style`: load before editing `src/**/*.py` or Python tests. Apply the project fail-fast, explicit-data, and exception-handling rules.
- `test-style-and-execution`: load before changing or running tests. Establish deterministic contract tests first; run every live-LLM case individually and inspect its artifact.
- `debug-llm`: load before creating or reviewing LLM quality artifacts, contract-recovery replays, or live-LLM evidence.
- `llm-trace-debug`: load when retrieving protected trace evidence for any new failure investigation; keep raw trace handling inside the protected trace boundary.
- `cjk-safety`: load before adding CJK content to Python string literals; keep new contract text ASCII unless CJK content is required by an existing fixture.

## Mandatory Rules

- The LLM owns authorship, durability, global applicability, contradiction, privacy, and semantic acceptance/rejection.
- Deterministic code owns JSON parsing entry, structural validation, path/type validation, index-to-repository mapping, persistence, limits, and fail-closed behavior.
- Every raw model response passes through `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` before semantic evaluation.
- The model-facing prompt contains semantic evidence cards but no repository evidence IDs, candidate IDs, database IDs, channel IDs, or opaque persistence handles.
- Deterministic mapping may translate a validated path, typed replacement, and bounded prompt index into the existing internal representation. It must not invent, rewrite, or suppress semantic decisions.
- Deterministic reason-code derivation is a sanitized disposition label only. It may use the LLM-owned action/verdict and semantic dimensions to select an existing reason code; it must not create a new semantic judgment.
- An invalid proposal or review result never enters policy, candidate persistence, revision persistence, scheduling, delivery, or cognition.
- Valid `no_change` and `reject` outcomes are successful semantic dispositions. Contract exhaustion remains a typed pipeline error and remains fail-closed; it must never be relabeled as character rejection.
- Preserve evaluated root lineage on semantic no-change and stage-failure runs. Do not create a candidate or revision from either outcome.
- Use the project virtual environment: `venv\Scripts\python.exe`.
- Never read `.env` during this plan. Use existing configuration guards and explicit test fixtures.
- Parent-owned tests establish the focused failing contract before production implementation begins.
- Run regular deterministic tests in batches. Run each live-LLM case one at a time with `-s`, inspect the generated artifact, and record the disposition.
- Run live-DB tests only through the repository’s explicit live-DB guard and only with a separately authorized test database.
- Do not introduce compatibility readers, alias layers, dual model shapes, fallback model calls, or alternate entrypoints. The wire-contract replacement is bigbang.
- After automatic context compaction, reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off every major progress stage, reread this entire plan before starting the next stage.
- Before final completion, lifecycle status change, merge, or sign-off, run the Independent Code Review gate and record its result in `Execution Evidence`.
- Execute only after this plan reaches `approved` or `in_progress` under `development_plans/active/`, and only after the user explicitly authorizes implementation.

## Must Do

- Replace the model-facing proposal patch shape with one uniform `path` plus `replacement` object; infer the internal value kind from the allowed path.
- Remove model-emitted `schema_version`, `reason_code`, and review `accepted_change_kind`; attach schema identity at the stage boundary and derive the existing internal reason/kind fields deterministically from validated LLM-owned fields.
- Replace model-facing evidence and candidate handles with bounded one-based prompt-local indices and deterministically restore repository identifiers after validation.
- Remove `accepted_changes` from the model-facing review output; on review acceptance, copy the already validated proposal patches deterministically into the existing internal review result.
- Rewrite proposal/review output instructions and expected-format examples to the v2 wire contract and remove the explanatory replacement-field phrase that the model copied as a JSON key.
- Reset each contract retry to the original system and semantic human context, plus one bounded structured repair instruction. Do not include the previous malformed model object in the next attempt.
- Preserve the canonical parser, three-attempt cap, provider/contract distinction, fail-closed stage error, root retention, and existing internal V1 policy/persistence shapes.
- Dynamically omit `corroborate_candidate` from proposal action guidance when the bounded candidate set is empty.
- Render the path/type registry once in the static system prompt and remove duplicate schema/version and allowed-path metadata from the dynamic human payload.
- Rewrite the no-change guidance so the model returns `no_change` when evidence adds no materially new durable identity meaning; retain only genuinely changed, evidence-supported paths in mixed patches; patch a sibling field only when the evidence directly contradicts that field.
- Define one typed contract-violation object with a bounded `violations` list; persist only stable `stage.code` values in existing run `validation_error_codes` and send field/expected details only through protected trace metadata and retry repair text.
- Record stable, bounded contract error codes and fields without exposing raw identifiers or private transcript content in repair prompts or public projections.
- Freeze prompt-safe replay inputs for all 185 routed episodes before production implementation, run the current V1 baseline, then run the V2 cohort against the identical inputs and model configuration.
- Require at least 176/185 end-to-end replay successes and at least 40/42 successes in the historical failure subset; retain 42/42 as the zero-terminal-failure target.
- Add deterministic tests for the wire contract, index mapping, no-op recovery, review copying, retry reset, handle absence, fail-closed persistence, and root retention.
- Run all existing identity-growth deterministic gates and all existing live-LLM cases individually after implementation.
- Update the character-identity-growth README to describe numeric prompt indices and the separation between model-facing V2 wire data and internal V1 decisions.

## Deferred

- Do not change the semantic policy rules for authorship, durability, contradiction, privacy, cadence, or promotion.
- Do not change the consolidation model, route, temperature, top-p, completion-token cap, or three-attempt limit.
- Do not change `policy.py`, database collections, revision schema, candidate lifecycle, cognition projections, dialog wording, or adapter delivery unless a focused test proves the unchanged internal contract cannot accept the mapped result.
- Do not add a heuristic classifier for user input, a keyword gate, pre-processing, or post-processing that changes an LLM semantic decision.
- Do not convert exhausted contract errors into `no_change`, `reject`, or any other semantic disposition.
- Do not preserve the old model-facing V1 output as a compatibility path.
- Do not add raw QQ transcript fixtures or expand protected trace capture for this bugfix. The replay fixture contains only protected, prompt-safe reconstructed cards and deterministic lineage metadata.
- Do not upgrade dependencies, change unrelated prompts, or perform formatting-only cleanup.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Proposal model output | bigbang | Accept only the V2 proposal fields defined below; schema identity is selected by the stage boundary rather than emitted by the model. |
| Review model output | bigbang | Accept only the V2 review fields defined below; the model does not emit `accepted_changes`, `accepted_change_kind`, or `reason_code`. |
| Prompt provenance | bigbang | Use one-based evidence/candidate indices; no raw or opaque repository handles enter the prompt. |
| Retry context | bigbang | Start each retry from the original system/human context plus one structured repair instruction; do not append malformed output. |
| Internal policy and persistence | retained | Preserve the existing validated V1 decision, candidate, revision, run, and root-lineage shapes. |
| Historical traces and runs | retained history | Read-only historical records remain unchanged; runtime code does not parse historical model output as a new input contract. |
| Tests | bigbang | Rewrite model-output fixtures and prompt assertions to V2; retain policy and persistence assertions against the internal V1 result. |

### Cutover Policy Enforcement

- The execution agent must follow the selected policy for each area.
- The agent must not preserve the old model-facing shape through a fallback, alias, adapter, dual reader, or conditional branch.
- Existing internal V1 shapes may remain only after deterministic V2 validation and mapping has completed.
- Any change to this cutover policy requires user approval before implementation.

## Target State

The proposal stage returns the existing internal `IdentityProposalDecisionV1` only after validating the V2 model wire object and mapping bounded indices and uniform replacements. The review stage returns the existing internal `IdentityReviewDecisionV1` only after validating the V2 review judgment; accepted internal patches are copied from the validated proposal rather than reconstructed by the model. The model emits neither schema/version metadata nor sanitized reason codes; the stage boundary supplies the internal schema version and derives the existing reason code from validated LLM-owned fields.

For every model attempt:

- the model sees the same bounded semantic context;
- evidence rows are addressed as `evidence_index` values from `1` through `N`;
- candidate rows are addressed as `candidate_index` values from `1` through `M`;
- raw repository identifiers are absent from system and human prompts;
- the retry repair instruction contains only stable error code, field, and expected shape;
- the next attempt does not see the prior malformed JSON object.

The model’s action guidance contains `corroborate_candidate` only when at least one candidate row is present. The static path/type registry appears once in the system prompt; the dynamic payload contains no duplicated allowed-path list or schema-version metadata.

Expected outcomes:

- expected semantic rejection completes as `no_change` or `rejected` with no candidate/revision mutation;
- valid growth reaches the unchanged policy and persistence path;
- malformed output is regenerated within three attempts when the model can repair it;
- exhausted contract recovery remains an auditable `failed` run with retained roots and no mutation;
- the frozen 185-episode replay reaches at least 176 valid dispositions, the 42-root failure subset reaches at least 40 valid dispositions, and the final target remains zero terminal contract failures.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Model patch shape | Use `path` plus one `replacement` field | Removes the tagged-union key-choice burden from the weaker local model while keeping path-specific deterministic type validation. |
| Model output minimum | Omit model-emitted schema/version, reason, and review accepted-kind fields | Removes metadata that the stage already owns or can derive from the LLM-owned judgment, reducing exact-key and cross-field failures. |
| Model provenance | Use one-based prompt-local indices assigned after final candidate trimming | Eliminates long-ID copying and makes provenance a bounded selection task; the paired replay gate proves whether it improves reliability beyond existing aliases. |
| Review patch ownership | Deterministically copy accepted proposal patches | The review judges acceptance; it does not need to serialize an already validated patch a second time. |
| Reason-code ownership | Derive the existing internal reason code from validated action/verdict and semantic dimensions | Preserves the LLM’s semantic fields while removing a redundant model choice that frequently conflicts with them. |
| Candidate action set | Include corroboration guidance only when candidates exist | Prevents an impossible action/candidate combination from entering the model’s decision space. |
| Dynamic prompt payload | Keep the path/type registry in the static system prompt and remove duplicate allowed-path/schema metadata from the human payload | Reduces context and exact-key burden without removing semantic evidence or current identity. |
| No-change judgment | Return `no_change` when evidence adds no materially new durable identity meaning; patch sibling fields only for direct evidence contradiction | Prevents contract optimization from turning already-satisfied evidence into unrelated identity changes. |
| Retry context | Rebuild from immutable base messages | Prevents malformed-output anchoring and retry-context growth. |
| Retry diagnostics | Use a bounded `violations` list and protected trace metadata | Provides all actionable repair facts without leaking raw output or consuming one retry per missing field. |
| Internal representation | Retain existing V1 internal decisions | Keeps policy, candidate, revision, run, and downstream consumers stable. |
| Contract version | Select V2 at the stage boundary rather than requiring a model-emitted version field | Makes the bigbang boundary explicit while removing one exact-key failure source. |
| Error reporting | Use stable bounded codes and fields | Improves repair guidance and future RCA without leaking raw IDs or private content. |
| No-op handling | Require the semantic owner to return `no_change`; treat a no-op patch as a recoverable contract error | Preserves semantic ownership and prevents deterministic code from silently rewriting an arbitrary judgment. |

## Contracts And Data Shapes

### Proposal wire contract

The proposal stage selects V2 at the boundary. The model must return exactly these keys; it emits no schema/version field and no reason-code field:

```json
{
  "action": "no_change | explicit_self_redefinition | inferred_growth | corroborate_candidate",
  "candidate_index": null,
  "proposed_changes": [
    {"path": "self_image.self_concept", "replacement": "..."}
  ],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "global_applicability": "global | scoped | absent",
  "confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_abstraction": "...",
  "evidence_indices": [1],
  "contradiction_candidate_indices": []
}
```

`replacement` is a JSON string, integer, semantic-band string, closed-enum string, or string list according to `path`. The validator maps it to the existing internal tagged patch with the appropriate `value_kind` and replacement key. The model never emits `value_kind` or a replacement-key variant.

`candidate_index` is null for new proposals and `no_change`; corroboration requires exactly one valid candidate index. The prompt lists `corroborate_candidate` only when at least one candidate is present. Evidence and contradiction lists are bounded, contain unique one-based integers, and map only to rows displayed in the same prompt.

The validator derives the internal `reason_code` after validating the model-owned fields. The fixed precedence is:

| Proposal condition | Internal reason code |
|---|---|
| `action=no_change` and `private_detail_risk=high` | `privacy_blocked` |
| `action=no_change` and contradiction indices are present | `contradiction_blocked` |
| `action=no_change` otherwise | `proposal_no_change` |
| `action=explicit_self_redefinition` with readiness dimensions valid | `candidate_ready` |
| `action=inferred_growth` or `corroborate_candidate` with high confidence, durable/global relevance, and low privacy risk | `candidate_ready` |
| `action=inferred_growth` or `corroborate_candidate` without readiness dimensions | `candidate_emerging` |

An explicit action that fails its required readiness/authorship dimensions is a cross-field contract error. The derivation selects an existing label; it does not change the action or any semantic field.

### Review wire contract

The review stage selects V2 at the boundary. The model must return exactly these keys; it emits no schema/version, accepted-kind, or reason-code field:

```json
{
  "verdict": "accept | reject | no_change",
  "selected_candidate_index": null,
  "rejected_candidate_indices": [],
  "character_authorship": "self_declared | inferred | absent",
  "identity_relevance": "durable | ephemeral | absent",
  "coherence": "coherent | conflicting | absent",
  "global_applicability": "global | scoped | absent",
  "review_confidence": "low | medium | high",
  "private_detail_risk": "low | high",
  "character_owned_summary": "...",
  "privacy_safe_evidence_summaries": ["..."]
}
```

The review wire object contains no `accepted_changes`. For `verdict=accept`, deterministic validation copies the validated proposal’s internal `proposed_changes` into internal `accepted_changes` after checking candidate, contradiction, authorship, coherence, privacy, and readiness invariants. Internal `accepted_change_kind` is derived from the proposal action for new changes or from the selected candidate’s existing change kind for corroboration, then checked against the review’s model-owned authorship.

The validator derives the internal review `reason_code` with this fixed precedence:

| Review condition | Internal reason code |
|---|---|
| `verdict=no_change` | `proposal_no_change` |
| `verdict=reject` and `private_detail_risk=high` | `privacy_blocked` |
| `verdict=reject` and `coherence=conflicting` or rejected contradiction candidates are present | `contradiction_blocked` |
| `verdict=reject` otherwise | `review_rejected` |
| `verdict=accept` with high confidence, durable/coherent/global relevance, and low privacy risk | `candidate_ready` |
| `verdict=accept` otherwise | `candidate_emerging` |

### Retry error descriptor

The repair instruction contains a bounded list of all known violations:

```json
{
  "violations": [
    {
      "code": "invalid_index",
      "field": "evidence_indices[0]",
      "expected": "integer from 1 through N"
    }
  ]
}
```

The typed stage-facing `IdentityContractViolation` contains the same bounded `violations` list. Each item has only `code`, `field`, and `expected`. Allowed stable error codes are `malformed_json`, `missing_required_key`, `unknown_key`, `wrong_type`, `unsupported_value`, `invalid_index`, `invalid_provenance`, `semantic_noop`, `cross_field_inconsistency`, and `handle_leakage`. The descriptor must not contain raw model output, raw repository IDs, private message text, or exception text that could contain either. The stage appends only `stage.code` values such as `proposal.invalid_index` to the existing run `validation_error_codes`; the full bounded descriptor is passed through the existing protected `validation_error` trace field and the retry message.

### Dynamic prompt payload

The static system prompt contains the closed path/type registry exactly once. The dynamic human payload contains current identity, redacted evidence cards, bounded candidate rows, and semantic summaries, but omits prompt-visible `schema_version`, `allowed_paths`, evidence IDs, candidate IDs, and duplicated path/type registries. Internal input schema versions remain validated before rendering and are not emitted to the model.

## LLM Call And Context Budget

The affected stages are background-only `CONSOLIDATION_LLM` calls using the existing local consolidation model and call configuration.

| Measure | Before | After |
|---|---|---|
| Nominal stage calls | One proposal plus one review | One proposal plus one review |
| Maximum semantic-stage calls | Three attempts per stage; six total | Three attempts per stage; six total |
| Canonical JSON-repair calls | Up to one existing `JSON_REPAIR_LLM` call per malformed semantic attempt; up to six total | Same existing bound; up to six repair calls and twelve total model/helper calls across proposal and review |
| Base prompt budget | 18,000 characters | 18,000 characters |
| Retry input | Original context plus prior AI output and repair message; context grows per attempt | Original context plus one bounded repair descriptor; no prior AI output |
| Model-visible provenance | Short string aliases derived from repository handles | One-based numeric indices only |
| Output schema | Tagged patch union, duplicated dynamic schema metadata, and review patch copy | Uniform replacement patch, minimal semantic judgment, and one static path/type registry |
| New response-path calls | None | None |
| Provider configuration | Existing temperature, top-p, model, and completion cap | Unchanged |

The renderer must keep the required identity/evidence context within the existing 18,000-character budget. The repair descriptor is capped at 800 characters and is added only to the retry message, keeping the bounded semantic attempt input below 18,800 characters. The plan adds no model/helper call; the existing canonical parser’s `JSON_REPAIR_LLM` path remains counted in the worst-case twelve-call bound. Verification must record semantic attempts, JSON-repair calls when exposed by the trace, rendered prompt size, and validation-error sequence for each live artifact.

## Change Surface

### Delete

- No files are deleted.
- Remove the retired V1 model-output instructions, expected-format examples, and review replacement-copy wording from `llm.py` prompt constants. Internal V1 decision fields remain because policy and persistence consume them.

### Modify

- `src/kazusa_ai_chatbot/character_identity_growth/models.py`
  - Add non-model V2 wire-boundary constants and typed wire shapes without a model-emitted schema field.
  - Keep `IdentityPatchV1`, `IdentityProposalDecisionV1`, and `IdentityReviewDecisionV1` as internal mapped contracts.
- `src/kazusa_ai_chatbot/character_identity_growth/validation.py`
  - Validate V2 uniform replacements and one-based indices.
  - Map V2 wire data to the existing internal V1 decisions.
  - Copy validated proposal patches for accepted internal review results.
  - Derive internal reason codes and accepted change kind using the fixed tables in `Contracts And Data Shapes`.
  - Define one typed `IdentityContractViolation` with bounded `violations` items and wrap all stage-facing V2 validation failures in it.
- `src/kazusa_ai_chatbot/character_identity_growth/llm.py`
  - Rewrite proposal/review prompts and expected formats.
  - Replace string-handle aliasing/restoration with index mapping/restoration, assigning indices after final candidate trimming.
  - Normalize every prompt-visible candidate/proposal patch to the uniform shape.
  - Remove dynamic schema/version, allowed-path, and duplicated path/type metadata from the human payload.
  - Pass the bounded violation descriptor through the existing protected `validation_error` trace argument.
  - Reset retry messages to the immutable base context plus a bounded descriptor.
  - Preserve canonical parsing, trace recording, attempt caps, and typed stage errors.
- `src/kazusa_ai_chatbot/character_identity_growth/README.md`
  - Document the V2 model-facing boundary, numeric indices, internal V1 mapping, and review ownership.
- `tests/test_character_identity_growth_validation.py`
  - Add V2 wire validation, replacement mapping, index bounds, and review-copy tests.
- `tests/test_character_identity_growth_prompt_contracts.py`
  - Update prompt-schema assertions and add forbidden-handle/forbidden-phrase assertions.
- `tests/test_character_identity_growth_policy.py`
  - Replace model-output fixtures with V2 wire responses while retaining internal policy assertions.
  - Add no-op, malformed-shape, invalid-index, and retry-reset sequences.
- `tests/test_character_identity_growth_runner.py`
  - Verify valid no-change/rejection and exhausted-stage failure preserve roots and create no candidate/revision.
- `tests/test_character_identity_growth_observability.py`
  - Verify stable stage/code error projection and distinction between semantic rejection and pipeline failure.
- `tests/test_character_identity_growth_live_llm.py`
  - Update the capture assertions for internal V1 results and inspect prompt-safe numeric indices.
- `tests/test_character_identity_growth_failure_cohort_live_llm.py`
  - Create the one-time V1 baseline harness and the post-cutover V2 replay gate for the frozen 185-episode cohort and 42-root failure subset.

### Create

- No production module is created. Add the failure-cohort test harness in `tests/test_character_identity_growth_failure_cohort_live_llm.py`; it owns only replay orchestration and artifact writing, while the single semantic-stage boundary remains the production owner.
- Generate the protected, prompt-safe replay artifact at `test_artifacts/diagnostics/asuna_identity_growth_replay_v1.json` before production implementation. It contains 185 routed episode inputs, a marked 42-root failure subset, redacted cards, current identity/candidate snapshots, lineage digests, source-fidelity labels, and no raw QQ transcript.
- Generate `test_artifacts/diagnostics/asuna_identity_growth_replay_v1_baseline.json` before the V2 implementation and `test_artifacts/diagnostics/asuna_identity_growth_replay_v2_result.json` after it. These are execution evidence artifacts, not compatibility runtime data.

### Keep

- `src/kazusa_ai_chatbot/character_identity_growth/projection.py` input lineage and bounded evidence construction.
- `src/kazusa_ai_chatbot/character_identity_growth/policy.py` semantic policy ownership.
- `src/kazusa_ai_chatbot/character_identity_growth/runner.py` candidate/revision persistence and fail-closed mutation boundary, except for test-proven mapping adjustments inside the approved internal contract.
- Existing database schemas, trace privacy modes, model route, prompt budget, and attempt cap.

## Overdesign Guardrail

- Actual problem: valid semantic decisions are being lost as terminal contract failures because the local model must serialize too many internal details and redundant semantic labels at once, with proposal-stage failure dominating the cohort.
- Minimal change: remove model-emitted schema/reason/accepted-kind metadata, simplify the proposal/review wire shape, make provenance numeric and local, reset contract retries, and prove the result on the frozen replay cohort while preserving the internal V1 domain contract.
- Ownership boundaries: LLM owns semantic judgment; `validation.py` owns structural/type/index mapping; `llm.py` owns prompt and retry mechanics; `policy.py` owns disposition; `runner.py` owns persistence and fail-closed mutation.
- Rejected complexity: no model upgrade, extra agent, extra retry, compatibility reader, dual wire shape, heuristic semantic filter, fallback semantic decision, database migration, or new public service path.
- Evidence threshold: the current plan must reach 176/185 end-to-end and 40/42 failure-cohort successes. Add a separate plan only if a failure family survives those gates and cannot be addressed by the bounded prompt/validator boundary without changing semantic ownership.

## Agent Autonomy Boundaries

- The responsible agents may choose local implementation mechanics only when they preserve the exact V2 wire shapes, index rules, internal V1 mapping, retry reset, and ownership boundaries in this plan.
- The production-code subagent must not alter policy semantics, persistence schemas, model routing, attempt limits, or downstream cognition/dialog consumers.
- The responsible agents must not add compatibility layers, alternate prompt shapes, fallback paths, private pass-through wrappers, or new semantic correction logic.
- Any production edit outside `models.py`, `validation.py`, and `llm.py` requires the strong justification recorded in this plan and a focused test proving the target modules cannot own the behavior.
- Test fixtures may use synthetic redacted evidence; they must not include raw private QQ transcripts or repository identifiers in model-visible content.
- If code and plan disagree, preserve the plan’s stated ownership and report the discrepancy before changing scope.
- If native subagent execution is unavailable, stop before implementation and report the blocker until the user explicitly approves fallback execution.

## Implementation Order

1. Parent baseline and focused contract tests.
   - Record `git status --short`, current HEAD, the existing focused selector result, and `git diff --check`.
   - Build and freeze `test_artifacts/diagnostics/asuna_identity_growth_replay_v1.json` from the protected boundary for all 185 routed episode inputs, marking the 42 historical failures and preserving only prompt-safe reconstructed cards plus lineage digests.
   - Run the current V1 one-time baseline harness against those identical inputs and record proposal, review-after-valid-proposal, end-to-end, and failure-cohort rates in `asuna_identity_growth_replay_v1_baseline.json`.
   - Update the named validation, prompt-contract, policy, runner, and observability tests with V2 expectations and the new failure-mode sequences.
   - Run the focused selector before production edits. Record the expected failure because V2 constants and mapping are not yet implemented.
2. Production contract implementation.
   - Start exactly one native production-code subagent with this plan, the mandatory skills, the focused test contract, and ownership limited to `models.py`, `validation.py`, and `llm.py`.
   - Add V2 wire types/constants, uniform replacement validation, numeric index validation, deterministic mapping, reason/kind derivation, typed violations, and accepted-change copying.
   - Keep internal V1 return shapes stable for policy and runner callers.
3. Prompt/rendering implementation.
   - Replace prompt output instructions and examples with V2.
   - Render final bounded evidence/candidate rows with one-based indices assigned after trimming, dynamically omit corroboration guidance when there are no candidates, and remove repository identifiers from every system/human prompt string.
   - Remove duplicate schema/version, allowed-path, and path/type metadata from the dynamic human payload while keeping the static path/type registry once in the system prompt.
   - Normalize prompt-visible current-candidate and proposal patches to `path` plus `replacement`.
4. Retry implementation.
   - Rebuild every contract retry from the original two-message base.
   - Add one safe descriptor message containing all bounded violations without rejected output or raw exception text.
   - Run the protected V1-versus-reset-retry comparison on the same synthetic failure sequences before locking the retry contract.
   - Keep provider errors separate, preserve trace attempt recording, and retain the three-attempt cap.
5. Parent module verification.
   - Rerun the contract, validation, prompt, and policy selectors.
   - Inspect failed-attempt message arrays and assert the prior malformed object is absent from the retry input.
6. Parent integration verification.
   - Run runner, integration, worker, projection, observability, and module-boundary selectors.
   - Verify valid no-change/rejection and exhausted-stage failure preserve roots, create no candidate/revision, and project stable error codes.
7. Live-LLM verification.
   - Run the frozen 185-episode V2 replay one case at a time against the same model/configuration used for the V1 baseline; inspect each artifact and calculate the 185, 42, proposal, review, and end-to-end rates.
   - Require at least 176/185 end-to-end successes and at least 40/42 successes in the historical failure subset. A missing replay input, skipped case, or contract failure counts outside the numerator and blocks sign-off.
   - Run each existing identity-growth live case individually through the real consolidation model.
   - Inspect every artifact for prompt handle absence, V2 output recovery, attempt count, validation codes, semantic disposition, and no unintended persistence.
8. Final regression and review.
   - Run the repository non-live selector and static checks.
   - Start exactly one independent code-review subagent after verification passes.
   - Remediate only in-scope findings, rerun affected gates, record evidence, and request approval before lifecycle closeout.

## Execution Model

- The parent agent owns orchestration, test code, verification, execution evidence, review remediation, lifecycle updates, and final sign-off.
- The parent establishes the focused test contract and records the expected pre-implementation failure before production implementation starts.
- Exactly one native production-code subagent edits production code after the focused test gate. It edits only the approved production files and does not edit tests unless the parent explicitly directs an in-scope correction.
- The parent may run integration tests, static checks, artifact review, and non-overlapping verification while the production-code subagent works.
- Exactly one native independent code-review subagent reviews the final diff and evidence after planned verification passes. It reports findings and does not implement fixes.
- If native subagent capability is unavailable, execution stops before production edits unless the user explicitly approves fallback execution.

## Progress Checklist

- [x] Stage 1 — baseline and focused V2 contract tests established.
  - Covers: implementation step 1; protected replay artifact, V1 baseline artifact, `tests/test_character_identity_growth_failure_cohort_live_llm.py`, `tests/test_character_identity_growth_validation.py`, `tests/test_character_identity_growth_prompt_contracts.py`, `tests/test_character_identity_growth_policy.py`, and `tests/test_character_identity_growth_runner.py`.
  - Verify: 185 routed inputs and marked 42-root subset are frozen; V1 baseline rates are recorded; focused selector records the expected pre-implementation failures; `git status --short` and `git diff --check` are recorded.
  - Evidence: replay hash, baseline artifact path, baseline rates, command output, and changed test list in `Execution Evidence`.
  - Handoff: production-code subagent starts only after this stage is signed.
  - Sign-off: parent/date after evidence is recorded; reread this plan before Stage 2.
- [x] Stage 2 — V2 wire contracts and deterministic mapping complete.
  - Covers: `models.py` and `validation.py`; proposal replacement mapping, candidate/evidence index bounds, reason/kind derivation, review-copy behavior, and typed bounded violations.
  - Verify: validation and policy contract selectors pass; no internal V1 consumer changes are required.
  - Evidence: test output, changed symbols, and internal-shape assertion results.
  - Handoff: prompt/rendering implementation begins.
  - Sign-off: parent/date after evidence is recorded; reread this plan before Stage 3.
- [x] Stage 3 — prompt-safe indices and V2 prompt text complete.
  - Covers: `llm.py` prompt constants, bounded renderer, prompt-visible candidate/proposal normalization, and README contract text.
  - Verify: prompt-contract selector passes; raw repository identifiers and the retired replacement phrase are absent from rendered prompts.
  - Evidence: prompt snapshots or assertions, prompt character counts, and static grep results.
  - Handoff: retry implementation begins.
  - Sign-off: parent/date after evidence is recorded; reread this plan before Stage 4.
- [x] Stage 4 — bounded retry recovery complete.
  - Covers: `_run_identity_stage`, typed repair descriptor, provider/contract error distinction, protected `validation_error` propagation, and trace attempt recording.
  - Verify: retry tests prove immutable base context, absence of rejected output, all-violation repair descriptors, three-attempt cap, V1-versus-reset-retry comparison, and fail-closed exhaustion.
  - Evidence: focused test output, comparison artifact, and one inspected synthetic retry trace.
  - Handoff: integration verification begins.
  - Sign-off: parent/date after evidence is recorded; reread this plan before Stage 5.
- [x] Stage 5 — identity integration and persistence boundary verified.
  - Covers: runner, policy, integration, worker, projection, observability, and module-boundary tests.
  - Verify: valid rejection/no-change and exhausted failure retain roots; no candidate/revision is created from invalid output; existing accepted growth remains valid.
  - Evidence: selector output, run-result assertions, and static ownership checks.
  - Handoff: live-LLM verification begins.
  - Sign-off: parent/date after evidence is recorded; reread this plan before Stage 6.
- [x] Stage 6 — real local-LLM regression matrix verified.
  - Covers: the frozen 185-episode replay, its 42-root failure subset, and all ten existing cases in `tests/test_character_identity_growth_live_llm.py`, each run one at a time.
  - Verify: replay reaches at least 176/185 end-to-end and 40/42 failure-cohort successes; every case produces an inspectable artifact; expected rejection cases complete semantically; accepted cases preserve policy outcomes; no artifact contains a raw repository handle in the model-visible prompt.
  - Evidence: the completed replay has 185/185 end-to-end valid dispositions, 42/42 historical-failure valid dispositions, 185/185 valid reviews, zero terminal contract/provider failures, and all ten existing live cases pass individually.
  - Handoff: final full regression and independent review.
  - Sign-off: parent/2026-08-02 after the complete cohort artifact and per-case live evidence were recorded; reread this plan before Stage 7.
- [x] Stage 7 — final regression, independent code review, and handoff complete.
  - Covers: non-live suite, static checks, diff review, independent code-review subagent, remediation, and lifecycle record.
  - Verify: all acceptance criteria pass; the one-pass architecture review findings were remediated within scope; the identity-growth selector is 129/129, static checks pass, the repository-wide collection result is limited to the four recorded pre-existing fixture errors, and `git diff --check` is clean.
  - Evidence: final commands, completed cohort artifact, recovery audit, review artifact, changed-file list, residual test-fixture risks, and user handoff.
  - Handoff: the plan is archived under `development_plans/archive/completed/`.
  - Sign-off: parent/2026-08-02 under the user-directed one-pass review model; all surfaced architecture findings are remediated and no terminal replay failure remains.

## Verification

### Focused deterministic gates

Run with the project interpreter:

```powershell
venv\Scripts\python.exe -m pytest `
  tests/test_character_identity_growth_validation.py `
  tests/test_character_identity_growth_prompt_contracts.py `
  tests/test_character_identity_growth_policy.py `
  tests/test_character_identity_growth_runner.py `
  tests/test_character_identity_growth_observability.py -q
```

Expected result: all affected tests pass after implementation. Before implementation, the new V2 tests fail for the missing V2 contract/mapping behavior and no production code is changed until that baseline is recorded.

### Integration and boundary gates

```powershell
venv\Scripts\python.exe -m pytest `
  tests/test_character_identity_growth_projection.py `
  tests/test_character_identity_growth_integration.py `
  tests/test_character_identity_growth_worker_integration.py `
  tests/test_character_identity_growth_observability.py `
  tests/test_character_identity_growth_module_boundary.py -q
```

Expected result: zero failures. The selectors must prove that internal V1 policy/persistence consumers remain unchanged and that stage failures never create a candidate or revision.

### Static checks

- `rg -n "one matching replacement field" src tests` must return zero matches. Exit code `1` for no matches is expected.
- `rg -n "identity-evidence:|identity-candidate:" src/kazusa_ai_chatbot/character_identity_growth/llm.py` must return zero model-facing prompt literals. Synthetic test data may contain source-shaped identifiers; rendered-prompt assertions are the authority for test-time leakage.
- Prompt-template assertions must prove that model output examples contain no `schema_version`, `reason_code`, `accepted_change_kind`, or `accepted_changes` fields and that the static path/type registry appears once.
- Rendered-prompt tests must assert every evidence/candidate row uses numeric indices and that every source identifier is absent from both system and human prompt strings.
- `git diff --check` must pass with zero whitespace errors.

### Repository regression

```powershell
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm and not live_internet" -q
```

Expected result: zero new failures relative to the recorded baseline. Any pre-existing failure must be listed with its baseline evidence and remain outside the affected identity-growth selectors.

### Cohort reliability gate

The recovery-complete `test_artifacts/diagnostics/asuna_identity_growth_replay_v1.json` contains 185 routed episode inputs, the 42 historical failure-root markers, deterministic current identity/candidate snapshots, prompt-safe evidence cards, lineage digests, and source-fidelity labels. The seven full-capture debug roots retain exact redacted prompt-safe reconstruction. The remaining 178 cases retain their metadata-only source-fidelity labels while protected trace envelopes, bounded detail-free summaries, or explicit no-evidence sentinels provide replay inputs; no raw transcript content is stored in the manifest.

Run the current implementation once per replay case and retain `test_artifacts/diagnostics/asuna_identity_growth_replay_v1_baseline.json`. After V2 implementation, run the same node IDs, same frozen inputs, same model, same route, and same model configuration once per process and retain `test_artifacts/diagnostics/asuna_identity_growth_replay_v2_result.json`:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm "tests/test_character_identity_growth_failure_cohort_live_llm.py::test_live_replay_case[<case_id>]" -q -s
```

The harness must emit separate proposal, review-after-valid-proposal, end-to-end, and failure-subset counts. A routed case succeeds when proposal and review reach a valid semantic disposition within their three semantic attempts; `no_change` and `rejected` count as success, while contract exhaustion, missing input, skipped execution, or uninspectable artifact count outside the numerator. The canonical parser’s existing `JSON_REPAIR_LLM` calls are included in attempt/artifact accounting but do not alter the semantic-attempt cap.

Required thresholds:

- end-to-end: at least 176/185 (95.14%);
- historical failure subset: at least 40/42 (95.24%);
- review success after a valid proposal: at least 95% of valid proposals;
- target: 185/185 and 42/42 with zero terminal contract failures.

The ten existing semantic-quality cases below remain separate gates and do not count toward the cohort reliability numerator. A replay result below either mandatory threshold blocks sign-off and requires another scoped plan revision.

### Real local-LLM gates

Run each test separately; do not run the file as a batch:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_explicit_self_redefinition_is_character_authored -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_user_imposed_identity_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_inferred_growth_matches_existing_candidate -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_private_detail_is_abstracted_or_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_close_relationship_can_shape_global_identity -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_scoped_relationship_fact_is_not_identity -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_repeated_semantics_do_not_fake_independence -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_ephemeral_roleplay_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_contradictory_growth_is_rejected -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests/test_character_identity_growth_live_llm.py::test_live_reversal_requires_fresh_evidence -q -s
```

Expected result: each case writes an inspectable artifact, stays within three semantic attempts per stage, contains no raw repository identifiers in model-visible prompts, and reaches the expected semantic/policy disposition. A contract failure requires artifact inspection and remediation before the next gate.

### Protected trace and persistence checks

- Inspect the seven available full-capture debug failure modes through the existing protected trace workflow; use synthetic redacted responses for deterministic replay and keep QQ metadata-only traces metadata-only.
- Inspect the V1/V2 cohort artifacts for identical case IDs, model/configuration, prompt-safe input digests, semantic attempt counts, canonical JSON-repair counts, and failure disposition.
- For a semantic rejection, assert `disposition` is `no_change` or `rejected`, `candidate_id` is null, and no revision write occurred.
- For contract exhaustion, assert `disposition` is `failed`, roots and source count are retained, `validation_error_codes` contain stable stage/code values, and no candidate/revision write occurred.
- For accepted growth, assert the internal policy result and candidate/revision path match the pre-fix behavior.

## Independent Plan Review

Before changing this plan from `draft` to `approved`, run one independent plan review through the native review capability.

Review scope:

- The prior completed plan and Asuna failure artifact are correctly carried forward without modifying the completed record.
- The V2 wire shapes, internal V1 boundary, numeric index mapping, retry reset, and review ownership are complete and mutually consistent.
- The change surface is bounded to the identity semantic-stage contract and its tests/docs.
- No compatibility path, fallback semantic decision, unowned helper, open design choice, or broad cleanup remains.
- Every Must Do item maps to an implementation step, focused test, integration gate, and acceptance criterion.

Record blockers, required edits, non-blocking findings, and approval status in the plan before execution. Keep the plan `draft` while blockers remain.

## Independent Code Review

Run after all planned verification gates pass and before completion, lifecycle closeout, merge, or sign-off. The parent must start exactly one independent code-review subagent through the native capability. If native subagents are unavailable, stop and retain the plan’s active status until the user explicitly approves fallback execution.

Review inputs:

- this approved plan and the completed identity-growth parent plan;
- the full implementation diff and changed-file list;
- focused, integration, repository, live-LLM, trace, and persistence evidence;
- updated README and lifecycle records.

Review scope:

- Python style, test style, exception handling, parser usage, prompt safety, and command/path safety;
- exact V2 wire contract and deterministic mapping without semantic rewriting;
- absence of raw IDs and private content from model-visible prompts;
- retry context reset, attempt caps, provider/contract distinction, and failure persistence;
- no compatibility shims, alternate paths, unplanned files, or downstream ownership drift;
- acceptance/rejection behavior, regression coverage, evidence accuracy, and handoff quality.

The parent may fix findings only within this plan’s Change Surface. A finding requiring a new contract, boundary, fallback, migration, or unrelated module requires plan update and user approval before implementation. Record findings, fixes, rerun commands, residual risks, and review approval in `Execution Evidence`.

## Execution Evidence

Execution started under explicit user authorization. Stages 1 through 6 are
verified; Stage 7 remains open for final regression and lifecycle handoff.

- Baseline and focused test contract: `git status --short` and
  `git diff --check` were recorded before edits. Parent added the focused V2
  contract tests in `tests/test_character_identity_growth_validation.py`,
  `tests/test_character_identity_growth_prompt_contracts.py`, and
  `tests/test_character_identity_growth_policy.py`. The pre-implementation
  selector failed at collection because the planned
  `IdentityContractViolation` symbol is not yet present in `validation.py`.
  Initial frozen manifest before protected-input recovery:
  `test_artifacts/diagnostics/asuna_identity_growth_replay_v1.json`, SHA-256
   `5B36D04362BF8B623F265D78A95BEE2558D351A70EE136D16EF2B55CF6D02A41`, with
   185 routed episodes and 42 historical failures. V1 summary:
  `test_artifacts/diagnostics/asuna_identity_growth_replay_v1_baseline.json`,
  SHA-256 `94EBA9280E96D9D3633996F82E8C946C5D51FD6BC59DB075C9745D9C77D4F928`,
  records 80/185 proposal first-attempt, 146/185 valid proposals, 143/146
  valid reviews, and 143/185 end-to-end valid dispositions. Protected recovery
  then completed the same manifest with 185/185 prompt-safe replay inputs while
  preserving source-fidelity labels. The completed manifest SHA-256 is
  `C80EB8DEFA84864CE61AB9BCCDF83452571B400338A29BD78BA847850D0071A0`.
- V2 contract implementation: exactly one native DeepSeek-V4-Flash 0731
  implementation pass edited only `models.py`, `validation.py`, and `llm.py`.
  The parent corrected typed empty-object classification and internal V1
  review-kind preservation. The validation, prompt, and policy selectors now
   pass with 60 focused tests.
- Prompt and retry implementation: V2 uniform replacements, numeric indices,
  no-candidate action restriction, no-change semantic guidance, immutable
  retry context, bounded violation descriptors, and prompt metadata stripping
  are covered by the focused selectors. The full identity non-live selector
   passes 129 tests with 215 live tests deselected.
- Integration and persistence-boundary verification: contract, projection,
  causal-lineage, longitudinal-policy, integration, worker, observability,
   and module-boundary selectors are included in the 129-test pass; production
   modules compile and `git diff --check` is clean.
- One-at-a-time live-LLM artifacts: all 185 cohort replay cases pass
  individually, including 42/42 historical failures, and all ten existing
  live-LLM cases pass individually. The V2
  result artifact is
  `test_artifacts/diagnostics/asuna_identity_growth_replay_v2_result.json`,
   SHA-256 `94E053A498D8C4C012C1037E61E679044B3D53D8C4AEEA35F645AB382682D7A8`.
  Its cohort counts are proposal 185/185, review 185/185, end-to-end
  185/185, historical 42/42, and terminal failures 0. The recovery audit is
  `test_artifacts/diagnostics/asuna_identity_growth_replay_recovery_audit.json`.
  The parent-authored review is
  `test_artifacts/diagnostics/asuna_identity_growth_replay_v2_review.md`.
   The model-visible prompt audit found zero repository-handle, schema, or
   dynamic-allowed-path matches.
- Independent plan review: one read-only `gpt-5.6-sol` max architecture pass completed. Verdict was `DO NOT APPROVE` because the proposal contract remained too large, the 95% replay gate was absent, retry diagnostics were singular, no-change guidance conflicted with the current prompt, typed error propagation was underspecified, and the JSON-repair call budget was undercounted. The plan now removes model-emitted schema/reason/accepted-kind metadata, defines fixed derivation tables and typed violations, adds the 185/42 paired replay gate, corrects no-change guidance, and counts the twelve-call worst case.
- Independent code review: one native read-only architecture review completed
  with a `BLOCKED — do not approve completion or sign-off` decision before the
  recovery pass. The review identified six actionable items. The parent
  remediated ordered prompt-index mapping, strict V2-only semantic-stage
  validators with separately named internal V1 wrappers, inferred-growth
  candidate-index rejection, bounded multi-key structural violations with
  `stage.code` error names and an 800-character retry descriptor, and
  replay-artifact aggregation with completeness and 176/185 plus 40/42
  threshold enforcement. The remaining coverage finding is closed: protected
  recovery supplied all 178 missing prompt-safe inputs and the complete live
  gate now passes 185/185 with zero terminal failures.
- Reviewer remediation verification: the new ordered-index, strict-boundary,
  inferred-growth, multi-violation, and retry-bound tests pass; the full
  non-live identity-growth selector passes 129/129; all 185 cohort replays and
  all ten existing live cases pass individually. The cohort gate records the
  complete denominator and fails closed when the artifact is incomplete or
  below threshold.
- Repository-wide non-live regression: the planned command was attempted and
  stopped at collection on four pre-existing environment/fixture errors
  (`experiments.cognition_core_v2_real_conversation_replay` missing, the
  Asuna personality profile missing, and two missing conversation-history
  fixture imports). The affected identity-growth selector remains green at
  129/129; no new failure was observed in the scoped surface.
- Final lifecycle status and handoff:
  Stages 6 and 7 are complete. The plan is archived and the registry row points
  to the completed historical record.

## Acceptance Criteria

This plan is complete when:

- the model-facing proposal contract is V2 with uniform `path`/`replacement` patches and no model-emitted schema/version or reason field;
- the model-facing review contract is V2 and contains no `accepted_changes`, `accepted_change_kind`, or `reason_code` field;
- evidence and candidate provenance uses bounded one-based prompt indices, with zero raw repository IDs in rendered system or human prompts;
- the existing internal V1 decision, policy, candidate, revision, run, and root-lineage contracts remain valid;
- every invalid model output is parsed canonically, classified by one typed bounded violation object, propagated as `stage.code` plus protected field/expected metadata, and regenerated from the immutable base context with at most three semantic attempts;
- valid semantic rejection completes as `no_change` or `rejected`, with no candidate/revision mutation;
- exhausted contract recovery remains visibly `failed`, retains evaluated roots, and performs no candidate/revision mutation;
- deterministic identity-growth contract, policy, runner, integration, worker, projection, observability, and module-boundary selectors pass;
- the frozen 185-episode replay reaches at least 176/185 end-to-end successes and the 42-root historical failure subset reaches at least 40/42 successes, with proposal and review-after-valid-proposal rates recorded separately;
- all ten existing identity-growth live-LLM cases pass individually with inspected artifacts and zero terminal contract failures;
- the captured Asuna failure modes have deterministic regression coverage and no longer reproduce as terminal failures;
- the repository non-live regression has zero new failures relative to baseline;
- the independent plan review is approved before execution;
- the independent code review is approved after verification, with findings remediated or recorded as approved residual risk;
- execution evidence, lifecycle status, and user handoff are complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Simplifying metadata still leaves proposal-stage output burden too high | Remove model-emitted schema/reason/accepted-kind fields, remove duplicated dynamic metadata, constrain unavailable actions, and require the 185/42 replay thresholds | Paired V1/V2 cohort artifacts and proposal/end-to-end rate gates |
| Uniform replacement mapping changes a typed value incorrectly | Infer expected type only from the closed allowed path table and test every path family | Validation selector and internal V1 result assertions |
| Numeric indices become misaligned after candidate trimming | Trim optional candidates before assigning final indices and test protected/removed candidates | Prompt renderer and index-bound tests |
| Review copying bypasses semantic review | Copy only after review verdict and all review invariants pass; keep review semantic fields validated | Review validator and policy tests |
| Retry reset removes useful repair information or repairs only the first defect | Keep a bounded `violations` list, run a paired V1/reset-retry comparison, and pass details through protected trace metadata | Retry sequence tests, comparison artifact, and live artifacts |
| A contract failure is hidden as a semantic rejection | Preserve typed exhaustion and failed run disposition; never synthesize semantic decisions | Runner persistence and health tests |
| Existing internal consumers depend on model-facing fields | Keep deterministic V1 mapping at the LLM boundary and run integration/module-boundary selectors | Full identity integration gates |
| Local-model behavior regresses after prompt simplification | Run the frozen 185/42 cohort and ten semantic cases individually with the same model/configuration and inspect every artifact before sign-off | Paired cohort artifacts, live-LLM matrix, and independent review |
