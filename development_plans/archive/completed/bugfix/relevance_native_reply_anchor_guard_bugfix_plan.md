# relevance native reply anchor guard bugfix plan

## Summary

- Goal: make settled relevance express a clear semantic native-reply preference
  while preventing delivery from quoting an obsolete opening fragment of an
  assembled turn.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, and
  `test-style-and-execution`.
- Overall cutover strategy: bigbang prompt clarification and deterministic
  delivery guard on the existing `use_reply_feature` contract.
- Highest-risk areas: local-model instability in the Boolean decision and a
  response owner that differs from the effective latest fragment.
- Acceptance criteria: real-LLM gates establish the semantic decision; an
  assembled follow-up cannot quote its obsolete opener; a single-fragment
  response can preserve an allowed quote; no LLM call or adapter contract is
  added.

## Context

Settled relevance already emits `use_reply_feature`, but its prompt defines no
decision rule. The service passes the Boolean to adapters, which quote the
inbound platform message owned by the response future. When a follow-up is
appended to an open turn, that response owner can remain the older opener even
though the assembled turn's effective meaning comes from its newest fragment.
The visible answer can therefore quote the wrong message.

The user approved a narrow ownership split: settled relevance decides whether
native anchoring is socially useful, and deterministic service delivery permits
the request only when the response owner is the effective latest fragment.

## Mandatory Skills

- `development-plan`: govern this execution record and lifecycle.
- `local-llm-architecture`: keep the local-model question bounded and delivery
  feasibility deterministic.
- `no-prepost-user-input`: preserve the LLM semantic choice; apply only a
  delivery compatibility condition after cognition.
- `debug-llm`: retain raw traces and author a human-readable review from real
  input and output.
- `py-style`: govern every Python edit.
- `test-style-and-execution`: separate real-LLM semantic gates from
  deterministic delivery tests and run live cases one at a time.

## Mandatory Rules

- Add and run the real-LLM semantic gates before production implementation.
- Run each real-LLM test individually and inspect its newest raw trace before
  running the next case.
- Author the human-readable LLM review from inspected evidence; test code may
  emit structured evidence only.
- Keep stable prompt rules in the system message and current-run evidence in
  the human payload.
- Keep the existing relevance route, one settled LLM call, schema, adapter API,
  response-owner lifecycle, and cognition graph contract.
- Preserve the LLM's semantic Boolean in graph state. Apply delivery feasibility
  only while building `ChatResponse.use_reply_feature`.
- Use the project virtual environment for Python and pytest commands.
- After context compaction or a major checklist sign-off, reread this plan
  before continuing.
- Before lifecycle completion, perform the parent-only independent review the
  user approved by requiring this work to remain single-agent.

## Must Do

- Add real-LLM gates for a specific direct group question, an ordinary private
  message, and a whole-group invitation.
- Define the positive `use_reply_feature` decision procedure in settled
  relevance without adding prompt payload fields or examples shaped around the
  test wording.
- Suppress native reply delivery when an assembled turn's response owner is not
  its effective latest fragment.
- Add deterministic coverage for obsolete-owner suppression and allowed
  single-fragment passthrough.
- Align relevance and brain-service ICD wording with the ownership split.
- Run focused and affected regression suites and inspect the final diff.

## Deferred

- Do not change frontline relevance, settlement timing, append selection,
  response-future ownership, adapter rendering, cognition, dialog, persistence,
  or control-console telemetry.
- Do not add an LLM call, agent, retry, repair path, compatibility shim, feature
  flag, configuration value, or quote-target remapping.
- Do not redesign private-message settlement or infer platform capabilities in
  relevance.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| Settled prompt | bigbang | Define the existing Boolean directly. |
| Service delivery | bigbang | Gate the existing Boolean at response construction. |
| Adapter contract | retained | Keep first-message native reply behavior unchanged. |
| Tests | bigbang | Add semantic and deterministic gates for the clarified contract. |

## Target State

Settled relevance returns `use_reply_feature=true` only for a proceeding group
turn where native reply anchoring materially clarifies the specific addressed
message or speaker. It returns false for private conversation, whole-group
invitations, and non-proceed actions. The service then permits the native reply
only when the response owner is the effective latest fragment. A mismatch
degrades to an unquoted visible response.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Semantic owner | Settled relevance | Social usefulness is a language judgment. |
| Delivery owner | Brain service | Response-owner identity and quote feasibility are deterministic. |
| Mismatch behavior | Suppress the native quote | This prevents a misleading anchor while preserving dialog. |
| Target remapping | Keep the current owner | Remapping requires a broader response/future and adapter contract. |
| LLM workload | Keep one existing settled call | The Boolean fits the current bounded role. |

## Change Surface

### Modify

- `tests/test_relevance_turn_settlement_live_llm.py`: add three semantic gates.
- `tests/test_persona_relevance_agent.py`: assert the prompt contract.
- `tests/test_service_input_queue.py`: prove delivery suppression and
  single-fragment passthrough.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`: define the
  existing Boolean.
- `src/kazusa_ai_chatbot/service.py`: apply the response-owner/latest-fragment
  delivery condition.
- `src/kazusa_ai_chatbot/relevance/README.md`: document semantic ownership.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document delivery ownership.
- `development_plans/README.md`: track lifecycle.

### Create

- `test_artifacts/debug/relevance_native_reply_anchor_review.md`: ignored local
  human-readable before/after review artifact.

### Keep

- Adapters, API schemas, settlement coordinator, cognition, dialog, and control
  console remain unchanged.

## Overdesign Guardrail

- Actual problem: an undefined LLM Boolean can cause an assembled response to
  quote an obsolete opener.
- Minimal change: clarify the Boolean and add one response-construction
  conjunction using existing fragment and owner data.
- Ownership boundaries: the LLM owns social usefulness; service code owns
  executable target compatibility; adapters own native rendering.
- Rejected complexity: target selection fields, platform capability inputs,
  alternate response owners, extra model calls, retries, helpers, and UI work.
- Evidence threshold: a future requirement to quote a non-owner fragment would
  require a separately approved response/delivery contract change.

## Agent Autonomy Boundaries

- The parent may choose local expression layout while preserving this exact
  ownership split.
- Changes stay inside the listed files and behaviors.
- Equivalent existing behavior must be reused when found.
- Unrelated cleanup, formatting churn, prompt rewrites, and broad refactors are
  outside scope.
- A contract change discovered during execution requires user direction.

## Implementation Order

1. Add the three real-LLM tests and record their pre-change outputs.
2. Add deterministic prompt and service tests and record the expected baseline
   failure where applicable.
3. Clarify the settled prompt without changing its payload or output schema.
4. Add the response-construction delivery guard.
5. Update the two ICDs.
6. Run live cases individually, author the comparison review, then run focused
   and affected deterministic suites.
7. Perform the parent-only independent review, remediate in-scope findings, and
   close the plan lifecycle.

## Execution Model

- The user previously required single-agent execution and independent review
  without subagents; that instruction is the approved fallback execution path.
- The parent owns tests, production code, verification, evidence, review,
  lifecycle updates, and final sign-off.
- The parent establishes test contracts before production edits.

## Progress Checklist

- [x] Stage 1 - real-LLM semantic gates captured before implementation.
  - Sign-off: parent, 2026-07-16. L21 reproduced `proceed/false`; L22 and
    L23 passed `proceed/false`; each newest trace was inspected before the next
    case and recorded in the readable review.
- [x] Stage 2 - deterministic prompt and delivery contracts established.
  - Sign-off: parent, 2026-07-16. The prompt-contract and obsolete-owner
    tests failed on the missing behavior; the single-fragment passthrough test
    passed on the retained behavior.
- [x] Stage 3 - prompt and service guard implemented with ICD alignment.
  - Sign-off: parent, 2026-07-16. The prompt defines the existing Boolean, the
    service guards only final native-reply delivery, both ICDs reflect the
    ownership split, all three focused tests pass, and production files compile.
- [x] Stage 4 - live and deterministic verification completed with readable
  review evidence.
  - Sign-off: parent, 2026-07-16. Final L21-L23 live gates passed individually
    with inspected traces; 75 affected relevance/settlement/service tests, two
    retained quote-delivery tests, and five control-console projection/graph
    tests passed. Compilation and `git diff --check` passed.
- [x] Stage 5 - parent-only independent review passed and lifecycle closed.
  - Sign-off: parent, 2026-07-16. Review tightened the deterministic tests to
    prove the LLM Boolean reaches graph state unchanged before delivery
    compatibility is applied. Focused and 75-test regressions passed again;
    no unresolved finding remains.

## Verification

### Real LLM

Run each new case separately with `-q -s`, inspect its newest trace, and record
the semantic judgment in the review artifact.

### Focused deterministic

- `venv\Scripts\python -m pytest tests/test_persona_relevance_agent.py -q`
- Run the named native-reply tests in `tests/test_service_input_queue.py`.

### Affected regressions

- `venv\Scripts\python -m pytest tests/test_relevance_turn_settlement.py tests/test_relevance_turn_settlement_graph.py tests/test_persona_relevance_agent.py tests/test_service_input_queue.py -q`
- Compile the two changed production Python files.
- Inspect `git diff --check` and `git status --short`.

## Independent Code Review

The parent rereads this plan, the diff, test output, live traces, and the human
review from a fresh-review posture. Review prompt ownership, LLM workload,
semantic passthrough, deterministic target compatibility, private-message
impact, test realism, code-test coupling, ICD alignment, and unintended
control-console impact. Fix every in-scope finding and rerun affected gates.

## Acceptance Criteria

This plan is complete when all real-LLM gates meet their semantic contract,
the obsolete-owner delivery test passes, existing single-message quote behavior
passes, no added model call or adapter/UI change exists, regressions pass, and
the independent review has no unresolved finding.

## Execution Evidence

- Pre-change live evidence: L21 failed only the required native-anchor Boolean;
  L22 and L23 passed. Raw traces and the agent-authored review are under
  `test_artifacts/`.
- Focused tests: pre-implementation run produced two expected failures
  (`test_settled_prompt_defines_native_reply_anchor_semantics` and
  `test_native_reply_is_suppressed_for_obsolete_response_owner`) and one
  retained-behavior pass (`test_native_reply_reaches_single_fragment_response`).
  Post-implementation rerun passed all three focused tests; both changed
  production Python files compiled.
- Final live evidence and review: L21 passed `proceed/true`; L22 and L23 passed
  `proceed/false`. The final prompt was 4,970, 4,870, and 4,928 characters for
  the three cases. Each trace was inspected before the next case and the
  before/after review is in
  `test_artifacts/debug/relevance_native_reply_anchor_review.md`.
- Regression tests: 75 affected relevance/settlement/service tests passed; two
  existing direct quote and mention-coexistence tests passed; five
  control-console client/cognition-graph tests passed with one existing
  dependency deprecation warning. Production/test compilation and
  `git diff --check` passed.
- Independent review: parent-only fresh review covered plan alignment, prompt
  ownership and 564-character load increase, one-call route preservation,
  semantic passthrough, response-owner compatibility, private-message impact,
  test realism and anti-cheat quality, adapter boundaries, ICD alignment, and
  control-console projection/graph behavior. One test-quality finding was
  fixed and reverified. No unresolved risk or UI adjustment remains.
