# Public Documentation DSH Boundary Cleanup Plan

## Summary

- Goal: restore product-facing and generic architecture documentation to their
  stable abstraction boundaries while retaining DSH details in their proper
  operator, integration, and module-owned documents.
- Status: completed
- Scope boundary: top-level presentation plus non-DSH documents that currently
  duplicate DSH identity, protocol, persistence, or execution mechanics.
- Change direction: replace backend-specific presentation with stable task-
  resolution concepts; preserve current runtime behavior and all DSH-owned
  documentation.
- Acceptance state: completed without runtime or test-suite changes; all
  documentation and verification criteria passed.

## Scope And Change Direction

The pre-DSH top-level documentation at `4fab6a9a` establishes the presentation
baseline: Kazusa is introduced as a character cognition runtime, while task
execution is described as a capability rather than as the project's identity.
Current source and DSH-owned ICDs remain the authority for implementation
details.

This plan removes DSH-specific names and mechanics from:

- `README.md`
- `README_CN.md`
- `docs/SUBAGENT_INTERFACES.md`
- `docs/architecture/cognition_contracts_design.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`

It retains DSH detail in `docs/HOWTO.md`, DSH and resolver architecture
references, `src/agentic_resolver/README.md`, `sidecars/dsh_resolution/README.md`,
the task-resolution/interaction/gateway ICDs, and direct operational contract
owners such as Brain service, accepted task, background work, action spec, and
database documentation.

## Mandatory Skills

- `development-plan`: plan lifecycle, scope control, evidence, and closeout.
- `chinese-translation`: native Simplified Chinese reconstruction and bilingual
  fidelity for the paired landing pages.
- `test-style-and-execution`: execution of existing deterministic documentation
  checks without adding prompt, wording, deletion, or process-only tests.

## Mandatory Rules

- Keep runtime source, schemas, configuration, and behavior unchanged.
- Keep DSH-owned documentation complete and current.
- Describe task resolution by stable product capability in generic owners.
- Preserve unrelated documentation and existing user changes.
- Add no static phrase, source-absence, decommission, or documentation-policing
  tests.

## Must Do

- Restore both landing pages to Kazusa-first presentation and semantic parity.
- Remove DSH protocol, release, route, storage, authority, sidecar, tool-count,
  and startup detail from both landing pages.
- Remove the DSH runtime from the subagent-family taxonomy because it is not a
  subagent family.
- Keep cognition, Nodes, RAG, and self-cognition documents at their owned task-
  resolution handoff boundaries.
- Preserve links from operator and DSH-owned documents to the detailed runtime
  contracts.

## Deferred

- Runtime or configuration changes.
- Rewriting direct DSH owners or their detailed ICDs.
- General documentation redesign unrelated to the DSH leakage.
- New documentation-string tests.

## Target State

- The paired top-level READMEs contain no DSH-specific term or identifier.
- Generic architecture documents name `task resolution`, typed observations,
  accepted delayed work, and ownership boundaries without naming the backend.
- The subagent guide covers actual subagent and worker families only.
- Direct DSH owners continue to document exact runtime contracts and operations.

## Execution Roles

### Documentation implementation owner

- Responsibility: classify ownership, edit the declared documentation surface,
  maintain bilingual fidelity, and record verification evidence.
- Owned surface: the seven governed documents and this plan/registry entry.
- Authority: documentation-only edits within the declared boundary.
- Applicable skills: `development-plan`, `chinese-translation`, and
  `test-style-and-execution`.
- Capability floor: repository history comparison, architecture ownership
  analysis, native English and Simplified Chinese editing, and deterministic
  verification.
- Independence requirement: none; the change is documentation-only and the
  user's boundary decision is explicit.
- Acceptance output: scoped diff, zero-leakage scan, bilingual review, passing
  affected documentation tests, and completed archived plan.
- Gate: clean execution baseline before edits; all acceptance criteria and
  evidence recorded before archival.

## Test Impact And Traceability

| Governed artifact | Contract changed | Semantic owner | Exact deterministic nodes | Mode | Regression prevented |
| --- | --- | --- | --- | --- | --- |
| `README.md` | Product presentation | Project documentation | none; deliberate exclusion of static wording tests | Manual/static review | Backend implementation presented as project identity |
| `README_CN.md` | Chinese product presentation and parity | Project documentation | none; deliberate exclusion of static wording tests | Manual/static review | Implementation-heavy or semantically divergent localization |
| `docs/SUBAGENT_INTERFACES.md` | Cross-family taxonomy | Project documentation | none; deliberate exclusion of static wording tests | Manual/static review | A non-subagent runtime represented as a subagent family |
| `docs/architecture/cognition_contracts_design.md` | Generic task-capability ownership | Cognition architecture | `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner`; `tests/test_self_cognition_architecture_docs.py::test_cognition_contracts_doc_names_selected_self_cognition_speak_delivery` | Deterministic | Backend mechanics displacing stable cognition contracts |
| `src/kazusa_ai_chatbot/nodes/README.md` | Node-to-task-resolution handoff | Cognition nodes | `tests/test_cognition_observability_docs.py::test_runtime_readmes_document_prewarm_and_observation_carriers` | Deterministic | Node documentation claiming persistence/executor ownership |
| `src/kazusa_ai_chatbot/rag/README.md` | RAG/task boundary | RAG | none; deliberate exclusion of static wording tests | Manual/static review | Task backend leaking into evidence-owner documentation |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | Self-cognition task handoff | Self-cognition | `tests/test_self_cognition_architecture_docs.py::test_canonical_self_cognition_readme_defines_delivery_target_before_cognition`; `tests/test_self_cognition_architecture_docs.py::test_canonical_self_cognition_docs_are_single_delivery_authority`; `tests/test_self_cognition_architecture_docs.py::test_canonical_self_cognition_readme_documents_outcome_dimensions` | Deterministic | Self-cognition documentation claiming task binding and recovery mechanics |

## Change Surface

### Modify

- The seven governed documentation files listed above.
- `development_plans/README.md` for active and completed lifecycle registration.

### Create

- This bugfix plan, later moved unchanged except for execution record and status
  to `development_plans/archive/completed/bugfix/`.

### Keep

- Production code, tests, configuration, DSH-owned documentation, HOWTO, and
  direct operational contract documentation.

## Agent Autonomy Boundaries

The implementation owner may choose local prose and heading mechanics that
preserve the fixed ownership boundary and bilingual meaning. Any runtime,
schema, configuration, direct DSH-contract, or unrelated documentation change
requires a plan amendment and user decision.

## Verification

- Scan the governed generic documents for DSH-specific terms and identifiers.
- Confirm DSH-owned and operator documentation retains its current details.
- Review English and Chinese headings, capabilities, examples, startup path,
  project status, and links for semantic parity.
- Run the exact deterministic nodes in the traceability matrix.
- Run `git diff --check` and inspect the complete scoped diff.

## Acceptance Criteria

- Both landing pages present Kazusa rather than its task-execution backend.
- All seven generic documents respect their declared ownership.
- Detailed DSH documentation remains available in the correct owners.
- Existing affected deterministic documentation tests pass.
- The final diff contains documentation and plan lifecycle changes only.

## Progress Checklist

- [x] Classify all active DSH documentation references by owner.
- [x] Capture clean worktree baseline and owned file set.
- [x] Correct the paired top-level READMEs.
- [x] Correct adjacent generic documentation leakage.
- [x] Complete verification and diff review.
- [x] Record evidence, mark completed, and archive the plan.

## Execution Evidence

- Baseline: clean `HEAD` at `7bb8e42a`.
- Pre-DSH presentation reference: `4fab6a9a`.
- Executor resolution: `/root` using the current primary Codex model and local
  filesystem/test tools; dynamic assignment chosen because the task requires
  full conversation context, bilingual judgment, repository history, and
  immediate plan lifecycle ownership without a separate handoff.
- Governed leakage scan: zero DSH-specific terms or identifiers across the
  seven generic documents.
- Owner classification scan: remaining active references are confined to the
  HOWTO, DSH/resolver architecture, sidecar/control-plane packages, and direct
  Brain, cognition-interaction, task, action, accepted-work, background-work,
  gateway, and database contract owners.
- Bilingual review: English and Chinese headings, capability descriptions,
  architecture map, research flow, runtime layers, startup path, repository
  map, status, and links are semantically aligned. The stale Chinese claim that
  research bypasses visible dialog was corrected to cognition-first rendering.
- Existing deterministic documentation checks: 6 passed in 0.98 seconds.
- All governed local Markdown links resolve; all Markdown fence counts are
  balanced.
- `git diff --check`: passed with line-ending notices only.
- Scope review: documentation and plan lifecycle files only; production source,
  configuration, tests, and direct DSH-owned documents are unchanged.
