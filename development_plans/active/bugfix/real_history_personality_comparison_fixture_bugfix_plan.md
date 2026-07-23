# real history personality comparison fixture bugfix

## Summary

- Goal: produce a semantically valid 20-case Asuna/Kazusa comparison from
  direct Kazusa conversation-history rows.
- Plan class: medium
- Status: completed
- Mandatory skills: development-plan, debug-llm, character-test,
  test-style-and-execution, local-llm-architecture, py-style, cjk-safety
- Cutover: invalidate the contaminated paired report, preserve Kazusa source
  rows as the comparison base, deterministically role-bind the same rows to
  Asuna, and retain the old artifacts as RCA evidence.
- Highest-risk areas: identity leakage during role mapping, hidden prompt-time
  transformation, and treating technical trace completion as behavior
  acceptance.

## Context

The previous paired run selected rows addressed to 杏山千纱 and then injected
the active profile's typed mention and addressed id for both profiles. The
Asuna request therefore carried contradictory identity evidence. The report
printed only the source body and hid the effective envelope and transformed
input.

The corrected comparison is intentionally role-bound. The source population is
real conversation history in which Kazusa is the addressed character. The
Kazusa run preserves the source role and source event representation. The Asuna
run receives a deterministic projection of the same event: every source
Kazusa identity-bearing element is replaced with the active Asuna identity
before seeding or sending the request.

The user evaluates the character through source input, each profile's actual
input, private monologue, and final visible dialog. Technical pass status is
supporting evidence only.

The old report and artifacts remain available for RCA. They do not count as
comparison sign-off evidence.

## Mandatory Skills

- development-plan: maintain this plan through implementation and verification.
- debug-llm: expose actual input and output and keep technical validation
  separate from human quality judgment.
- character-test: preserve isolated per-case raw artifacts and the live graph
  boundary for later execution.
- test-style-and-execution: keep live cases one at a time and inspect each
  artifact before proceeding.
- local-llm-architecture: preserve typed adapter-to-brain ownership and keep
  identity decisions at the harness boundary for this test-only projection.
- py-style: apply the repository Python coding constraints before editing.
- cjk-safety: preserve UTF-8 and safe Python string handling for Chinese test
  data.

## Mandatory Rules

- Production relevance, decontextualization, cognition, dialog, and adapter
  code remain unchanged.
- Main comparison cases are exactly twenty direct Kazusa source user rows.
  Each selected row must have Kazusa in its typed addressed ids or typed bot
  mention and must have bounded assistant context.
- The Kazusa artifact preserves the source current body, raw wire text,
  source mentions, source addressed ids, source reply metadata, source user
  identity, and source bounded context. Transport-only debug channel and
  replay message identifiers may be isolated for the test database.
- The Asuna artifact uses the same source row and bounded context, then
  deterministically maps all Kazusa identity-bearing data before the service
  receives it:
  - explicit names and aliases in body and raw wire text;
  - typed mention ids, display names, and raw mention text;
  - addressed global ids;
  - assistant display names, platform ids, and global ids;
  - reply target identity fields and source identity strings used in prompt
    context;
  - any other nested model-visible source identity strings in the bounded
    source rows.
- Asuna mapping preserves every other user's identity and the semantic event
  content. It records the mapping entries and any excluded rows.
- No raw `@杏山千纱`, Kazusa alias, source Kazusa platform id, or source Kazusa
  global id may occur in the Asuna effective envelope, transformed bounded
  context, dynamic human prompt payload, or decontextualized input.
- Each raw artifact contains `source_input`, `effective_input`,
  `identity_mapping`, `excluded_rows`, transformed/effective context,
  `decontextualized_input`, `private_monologue`, and `visible_dialog`.
- The direct exact-replay routing guard is separate from the main comparison.
  It replays a Kazusa-directed source row without injecting the active target;
  it is not included in the twenty paired cases or the primary report.
- `technical_status=passed` proves execution and trace completeness only.
  `semantic_validity` records fixture identity checks separately.
- Live LLM execution uses the guarded test database, one case at a time, with
  each raw artifact inspected before the next case.
- A completed run with an empty visible message list is a captured response
  surface, not a fixture failure; the report renders it as `（无可见输出）`.
- Operational errors remain visible as the actual returned dialog text and are
  recorded as response-surface evidence rather than rewritten or suppressed.
- Do not alter personality JSON, production source, or unrelated dirty files.

## Target State

The main comparison contains twenty direct Kazusa source turns. Kazusa sees
the original role-bound source event. Asuna sees the same event after a fully
auditable deterministic identity projection to Asuna. Both profiles therefore
receive semantically matched interaction events without contradictory identity
evidence.

Every case artifact contains:

- `source_input`: the original source message record;
- `effective_input`: the exact profile-bound request input;
- `identity_mapping`: each source-to-active identity replacement;
- `excluded_rows`: rows intentionally excluded from the bounded context;
- `effective_context`: the rows seeded for that profile;
- `decontextualized_input`: the producing-stage output captured from trace;
- `private_monologue` and `visible_dialog` for human inspection.

The primary human report is authored from inspected artifacts and prints only
source input, each profile's actual input, private monologue, and visible
dialog. The existing Python script is a raw-evidence validator/support
generator; it does not author human judgment or quality conclusions.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Main population | Twenty direct Kazusa source rows | Exercises the role-bound failure mode that invalidated the previous run. |
| Kazusa side | Preserve source role and event data | Establishes the real-history baseline. |
| Asuna side | Deterministic source-to-Asuna identity projection | Keeps the event constant while changing only the addressed character identity. |
| Identity scope | Text, typed metadata, reply metadata, assistant identity, and nested prompt-visible strings | Prevents hidden Kazusa leakage through a secondary field. |
| Other participants | Preserve their names and ids | Keeps social event semantics intact. |
| Routing guard | Separate exact source replay without active-target injection | Tests target ownership independently from personality comparison. |
| Evidence | Raw source/effective inputs plus trace-derived decontextualized input | Makes prompt-time changes inspectable. |
| Report ownership | Human-readable review from inspected raw evidence | Prevents deterministic scripts from making behavioral judgments. |
| Production scope | Test harness and evidence support only | The observed failure is in fixture construction. |

## Contracts And Data Shapes

The source selector must prove before execution:

- role is `user`;
- the row is not attachment-only;
- Kazusa is present in typed addressed ids or a typed bot mention;
- the selected row has recent assistant context;
- the bounded context is retained as source rows without silent omission.

The Kazusa projection must prove:

- current source body and raw wire text are unchanged;
- source mentions and addressed ids are unchanged;
- source assistant identity fields remain source identity;
- any changes are transport-only replay isolation fields.

The Asuna projection must prove:

- source body/content is transformed only through the explicit identity map;
- all source Kazusa aliases and ids are absent from effective model input;
- active Asuna mention and addressed id are present when the source targeted
  Kazusa;
- other participant identity and event content are preserved;
- every replacement is recorded in `identity_mapping`;
- `excluded_rows` is present even when empty.

The semantic validity gate fails a case when:

- the effective target does not match the intended profile mode;
- the Kazusa side differs from its source event outside transport isolation;
- the Asuna side contains source Kazusa identity in any model-visible dynamic
  payload;
- the trace lacks required decontextualized input or monologue fields for the
  main comparison case;
- the response message field or response-surface classification is missing;
- a routing-guard trace lacks a relevance disposition or proceeds for a
  non-active target.

## LLM Call And Context Budget

- Use the existing service graph and configured route budgets.
- Use full protected local trace capture for each later live run.
- Use one isolated channel per profile and case.
- Run no parallel live LLM cases.
- Seed only the eight-row bounded context selected from the real export.
- Keep worker toggles disabled as in the existing isolated live harness.

## Change Surface

- `tests/test_real_history_personality_e2e_live_llm.py`: direct source
  selection, source-preserving Kazusa projection, deterministic Asuna role
  mapping, identity guards, routing guard, and raw artifact capture.
- `tests/build_real_history_personality_report.py`: deterministic raw-evidence
  validator/support generator; no human quality judgment.
- `tests/test_real_history_personality_fixture_contract.py`: deterministic
  mapping, source-preservation, routing-guard, and evidence-shape tests.
- No production source, personality JSON, database bootstrap, or unrelated
  dirty-file changes.

## Overdesign Guardrail

- Actual problem: the old harness sent Kazusa identity-bearing history to
  Asuna while claiming it was an Asuna comparison input.
- Minimal change: keep direct Kazusa history as the source, apply one explicit
  recursive identity projection for Asuna, guard the effective payload, and
  expose the result.
- Ownership: the test harness owns replay projection and evidence; relevance
  owns runtime semantic interpretation; cognition owns monologue intent;
  dialog owns visible wording.
- Rejected complexity: production target rewriting, prompt changes,
  compatibility aliases, output rewriting, and judge models.

## Implementation Order

1. Update this plan with the role-bound source contract.
2. Change selection to twenty direct Kazusa source rows.
3. Add source-preserving and Asuna canonicalization projections.
4. Add hard identity checks and dynamic trace-input extraction.
5. Update raw artifacts and the evidence-support validator.
6. Add deterministic fixture and evidence-shape tests.
7. Run static checks, deterministic tests, and live collection checks.
8. Reset the guarded database and run each comparison case one at a time with
   inspection between cases.
9. Run the separate exact-replay routing guard and author the human report
   after inspecting the raw artifacts.

## Progress Checklist

- [x] RCA confirmed the contradictory Asuna envelope.
- [x] Existing report classified as invalid paired sign-off evidence.
- [x] Role-bound source/mapping contract updated from neutral comparison.
- [x] Direct Kazusa source selector implemented and verified.
- [x] Kazusa source-preserving projection implemented and verified.
- [x] Asuna identity projection and leakage guards implemented and verified.
- [x] Raw evidence support validator and deterministic tests pass.
- [x] Py-compile, diff check, and collection checks pass.
- [x] Live comparison and routing guard executed in the guarded test database.
- [x] Independent review and final sign-off completed after the artifact,
  report, and regression audit.

## Acceptance Criteria

- The harness selects exactly twenty direct Kazusa source rows.
- Kazusa effective input matches the source role/event representation.
- Asuna effective input contains no source Kazusa identity and contains the
  mapped Asuna identity where the source targeted Kazusa.
- Other users and event content remain present after mapping.
- Each artifact exposes source input, effective input, mapping, exclusions,
  transformed context, decontextualized input, monologue, and dialog.
- The support validator rejects missing, mismatched, or contaminated pairs.
- The primary human report remains limited to the requested evidence fields.
- No production or personality files change.

## Execution Evidence

- Main artifacts: `test_artifacts/personality_comparison/role_bound_20_case/`.
- The main comparison contains 20 direct Kazusa history rows and 20 Asuna
  projections. Both sides have `technical_status=passed`, passed fixture
  validity, passed semantic validity, a decontextualized input, a private
  monologue, and captured visible response data. The Asuna artifacts were
  collected immediately before the response-surface classification field was
  added; their raw `visible_dialog` and `response` fields remain complete, and
  the post-change Kazusa reruns exercise the new classification gate.
- The separate routing guard contains four real Asuna runs under
  `test_artifacts/personality_comparison/routing_guard/asuna/`. All four
  settled on `discard` in the frontline relevance stage and produced no
  visible dialog. One transient Mongo connection failure was retried after a
  guarded database cleanup.
- The current report is
  `test_artifacts/personality_comparison/role_bound_20_case_report.md` and
  contains only source input, each profile's actual input, private monologue,
  and visible dialog.
- Real response-surface findings are preserved in the report: Asuna case 20
  returned the actual internal response-path error text after a
  `model_contract_invalid` fail-closed result; Kazusa case 14 completed with a
  private monologue and no visible dialog; Kazusa case 07 preserves its raw
  `[角色名称]` placeholder.
- During execution, the guardrail caught and corrected two harness false
  positives: the Kazusa baseline was initially subjected to Asuna source-leak
  checks, and the routing guard initially ignored `frontline_relevance_agent`
  `intake_action=discard`. Deterministic regression coverage now checks the
  latter path, and the four-case live routing guard revalidated it.
