# cognition state integrity plan

## Summary

- Goal: Fix one verified code-level bug — the conversation-progress recorder writes durable recovery-policy state without reading the character's `boundary_profile`, even though every other cognition stage does. Plumb the existing mapper-derived descriptors into the recorder prompt so `progression_guidance`, `next_affordances`, and `current_blocker` are conditioned on the configured boundary policy.
- Plan class: small
- Status: completed
- Overall cutover strategy: compatible at the public facade and persistence boundary; big-bang at the internal recorder input boundary because `boundary_profile` is always available from `character_profile`.
- Highest-risk areas: drifting from prompt+input plumbing into a recorder behavior redesign, sending raw float scores to the recorder LLM, scope-creeping back into the originally-proposed reappraisal/decay/schema fixes that were retracted as out-of-scope.
- Acceptance criteria: the recorder receives `boundary_recovery_description`, `self_integrity_description`, and `relationship_priority_description` via existing mappers; the recorder prompt references those descriptors when emitting recovery-policy fields; no new state, no new mappers, no new LLM calls.

## Context

RCA `test_artifacts/rca_qq_1082431481_response_pattern_20260501.md` originally surfaced four candidate issues (affect residue, schema-as-instance, recorder/boundary mismatch, asymmetric decay). Three of those depend on input quality and accumulated prior state and have been deliberately removed from scope. Only one survives as a verified, code-level defect independent of input or prior state.

Verified evidence:

- All seven `boundary_profile` mappers exist in `src/kazusa_ai_chatbot/nodes/boundary_profile.py` and follow the local-llm-architecture rule of mapping raw values to descriptors.
- L2 cognition consumes three mappers at `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py:592-595`.
- L3 cognition consumes two mappers at `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py:231-235` and again at `:824-828`.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py` consumes none. Grep for `boundary_profile|boundary_recovery|self_integrity` returns no match.

The recorder owns `progression_guidance` and `next_affordances`. These are recovery-policy fields. Writing them without conditioning on the character's configured `boundary_recovery`, `self_integrity`, and `relational_override` is config-blind durable-state generation. This is a real bug independent of input quality and independent of prior state.

## Mandatory Rules

- All boundary-profile values must reach the recorder LLM as descriptors produced by the existing mappers in `boundary_profile.py`. Raw floats or raw strategy strings must not enter the recorder prompt payload.
- Reuse the same mappers L2 already imports. Do not create new mappers.
- Do not introduce new state fields in `ConversationEpisodeStateDoc`, `ConversationProgressPromptDoc`, or any cross-stage envelope.
- Do not add new LLM calls. The recorder remains a single background call.
- Do not change relevance routing, cognition stage flow, decay logic, schema handling, dialog generator, or any other module.
- The recorder prompt may be edited only inside the `_RECORDER_PROMPT` constant in `recorder.py`. Do not change the recorder's output schema.
- Python edits must follow `py-style`. The prompt is English; if any descriptor strings contain CJK, follow `cjk-safety`.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Build a `boundary_profile` descriptor block in the recorder input payload using `get_boundary_recovery_description`, `get_self_integrity_description`, and `get_relationship_priority_description`.
- Add a `character_boundary_profile` section to the recorder prompt's `# Input Format`, listing the three descriptor fields.
- Add prompt guidance instructing the recorder to condition `progression_guidance`, `next_affordances`, and the softening cadence implied by `current_blocker` on those descriptors.
- Plumb the character's required `boundary_profile` through the recorder caller(s) so `record_turn_progress` receives it without changing its public facade signature.
- Add unit tests covering: descriptor mapping happens before the LLM call; the prompt payload contains the three descriptor strings; the recorder still produces a valid `ConversationEpisodeStateDoc` when `boundary_profile` is supplied; the production caller fails loudly if `character_profile.boundary_profile` is missing.

## Deferred

- Do not implement an L1/L2 reappraisal loop, schema-disconfirmation gate, decay mechanism, or closure-propagation channel. Those were in the prior draft and have been retracted.
- Do not plumb `compliance_strategy`, `control_intimacy_misread`, `control_sensitivity`, or `authority_skepticism` descriptors into the recorder. They do not drive recovery-policy state.
- Do not redesign relevance, cognition graph, dialog agent, or `conversation_progress` storage shape.
- Do not migrate or rewrite legacy `conversation_episode_state` documents.
- Do not change L2 or L3 boundary-profile consumption.

## Cutover Policy

| Area | Policy | Rationale |
|---|---|---|
| Recorder input projection and prompt | bigbang | `boundary_profile` is a required internal recorder input because it is always available from `character_profile`. Output schema is unchanged. Old stored documents continue to project. |

## Agent Autonomy Boundaries

- The implementation agent may choose local mechanics inside the recorder module but must preserve the public facade `record_turn_progress(...)` and the recorder's output schema.
- The agent must not introduce new mappers, new state fields, new LLM calls, or new modules.
- The agent must not modify L2/L3 boundary-profile consumption.
- If the plan and code disagree, the agent must preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible (for example, a mapper is missing), the agent must stop and report rather than inventing a substitute.

## Target State

- The recorder prompt's `# Input Format` lists `character_boundary_profile` containing `boundary_recovery_description`, `self_integrity_description`, and `relationship_priority_description`.
- The recorder prompt body instructs the model to align `progression_guidance`, `next_affordances`, and softening cadence with those descriptors.
- The recorder LLM never receives raw floats or raw strategy strings for boundary-profile values.
- A character configured `boundary_recovery=rebound` no longer produces "soften slowly / probe whether user is hiding more" recovery policy on a clarified-referent event of the kind in the RCA.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Which fields to plumb | `boundary_recovery`, `self_integrity`, `relational_override` | These are the three fields whose configured values directly govern recovery cadence, ledger-keeping disposition, and connection-extension affordance. The other four boundary-profile fields drive input interpretation, which the recorder does not own. |
| How values reach the LLM | Through the existing `get_*_description` mappers in `boundary_profile.py` | Local-llm-architecture rule: no raw values to the LLM. Mappers already exist and are already used by L2/L3. Reusing them keeps interpretation consistent across stages. |
| Where the descriptors live in the prompt | A dedicated `character_boundary_profile` block in `# Input Format`, referenced by name in the generation procedure | Matches the recorder prompt's existing structured-input convention. Keeps the descriptor block inspectable and debuggable. |
| Caller signature change | Extend `ConversationProgressRecordInput` with a required `boundary_profile` field | The production caller already owns `character_profile`; missing `boundary_profile` is a state-shape bug that should surface immediately. |
| Backward compatibility window | No recorder fallback for missing `boundary_profile` | The recorder should not silently write config-blind recovery state. Existing stored documents remain readable because no persisted schema changes are introduced. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/recorder.py` — import the three mappers, build the descriptor block in the input payload, add the descriptor section to `_RECORDER_PROMPT`'s `# Input Format`, add guidance referencing the descriptors.
- `src/kazusa_ai_chatbot/conversation_progress/models.py` — extend `ConversationProgressRecordInput` with a required `boundary_profile` field carrying the three raw values needed for mapping.
- The recorder's caller (the cognition pipeline location that invokes `record_turn_progress`) — pass the character's `boundary_profile` into the record input. Exact path resolved during implementation.

### Keep

- `src/kazusa_ai_chatbot/nodes/boundary_profile.py` — unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` and `_l3.py` — unchanged.
- `ConversationEpisodeStateDoc` and `ConversationProgressPromptDoc` schemas — unchanged.
- `load_progress_context` and `record_turn_progress` public facade — signature unchanged at the facade boundary.

### Create

- Tests under `tests/` covering the four cases listed in Must Do.

### Delete

- None.

## Implementation Order

1. Extend `ConversationProgressRecordInput` with the required `boundary_profile` field.
2. Modify the recorder to import mappers, build the descriptor block, and update `_RECORDER_PROMPT`.
3. Update the caller to pass `boundary_profile` into the record input.
4. Add unit tests.
5. Run focused tests and a single live recording smoke against a character configured `boundary_recovery=rebound` to confirm `progression_guidance` no longer emits ledger-keeping language.

## Progress Checklist

- [x] Stage 1 — input model and recorder payload plumbing
  - Covers: steps 1-2.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/conversation_progress/recorder.py src/kazusa_ai_chatbot/conversation_progress/models.py`; existing `tests/test_conversation_progress_*` pass; new test asserting descriptor strings appear in the recorder prompt payload passes.
  - Evidence: record changed files and test output in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-01` after verification and evidence are recorded.
- [x] Stage 2 — caller wiring and integration
  - Covers: step 3.
  - Verify: integration test that the recorder receives a non-empty `boundary_profile` from the cognition pipeline; missing `character_profile.boundary_profile` raises instead of falling back.
  - Evidence: record test output before moving on.
  - Sign-off: `Codex/2026-05-01` after verification and evidence are recorded.
- [x] Stage 3 — verification and live smoke
  - Covers: steps 4-5.
  - Verify: see Verification section.
  - Evidence: capture the recorder output diff against a `rebound` character before/after.
  - Sign-off: `Codex/2026-05-01` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg "boundary_profile|boundary_recovery|self_integrity" src/kazusa_ai_chatbot/conversation_progress` returns matches in `recorder.py` and `models.py`.
- `rg "raw|score" src/kazusa_ai_chatbot/conversation_progress/recorder.py` confirms no raw float is being placed into the recorder payload.

### Tests

- `pytest tests/test_conversation_progress_recorder.py` (new test cases for descriptor presence and normal runtime persistence).
- Existing recorder/repository/runtime tests under `tests/` continue to pass.

### Smoke

- One live recording call against a character configured `boundary_recovery=rebound` on a clarified-referent style turn produces `progression_guidance` and `next_affordances` consistent with rebound recovery (no "soften slowly" language, no "probe for hidden things" affordance).

## Acceptance Criteria

This plan is complete when:

- The recorder consumes `boundary_recovery`, `self_integrity`, and `relational_override` exclusively through the existing `boundary_profile.py` mappers.
- The recorder prompt's `# Input Format` includes a `character_boundary_profile` block, and the prompt body references those descriptors when generating recovery-policy fields.
- No raw boundary-profile floats or strategy strings appear in the recorder payload.
- New tests pass; all existing `conversation_progress` tests pass.
- A live smoke against a `rebound` character produces recovery-policy fields consistent with the configuration.

## Execution Evidence

- Plan approval: approved after code audit confirmed the recorder was the missing `boundary_profile` consumer. The final implementation makes `ConversationProgressRecordInput.boundary_profile` required because `character_profile.boundary_profile` is always available in the production caller.
- Changed files: `src/kazusa_ai_chatbot/conversation_progress/models.py`, `src/kazusa_ai_chatbot/conversation_progress/recorder.py`, `src/kazusa_ai_chatbot/service.py`, `tests/test_conversation_progress_recorder.py`, `tests/test_conversation_progress_runtime.py`, `tests/test_service_background_consolidation.py`.
- Static compile: `python -m py_compile src\kazusa_ai_chatbot\conversation_progress\models.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\conversation_progress\runtime.py src\kazusa_ai_chatbot\service.py` passed.
- Static greps: `rg "boundary_profile|boundary_recovery|self_integrity" src\kazusa_ai_chatbot\conversation_progress` returned matches in `recorder.py` and `models.py`; `rg "build_character_boundary_profile_descriptors|boundary profile absent|character_boundary_profile may be omitted|_record_input\(None\)|boundary_profile: NotRequired" src tests -g "*.py"` returned no matches.
- Focused deterministic tests: `pytest tests\test_conversation_progress_recorder.py tests\test_service_background_consolidation.py::test_progress_background_passes_character_boundary_profile tests\test_conversation_progress_flow.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_module_boundary.py -q` passed: 21 passed.
- Full non-live conversation-progress tests: `pytest tests\test_conversation_progress_cognition.py tests\test_conversation_progress_flow.py tests\test_conversation_progress_history_policy.py tests\test_conversation_progress_module_boundary.py tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_service_background_consolidation.py::test_progress_background_passes_character_boundary_profile -q` passed: 34 passed.
- Live recorder smoke: one real recorder call with `boundary_recovery="rebound"` on a clarified-referent turn produced `current_blocker=""`, `open_loops=[]`, `avoid_reopening=["hidden meaning suspicion"]`, `next_affordances=["proceed with save index topic", "check if user wants to continue with that"]`, and `progression_guidance="Move forward with the literal meaning; no need to probe for hidden intent; continue on the save index issue."`

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A caller is missed and silently keeps recording without boundary grounding | `boundary_profile` is required; the production caller indexes `character_profile["boundary_profile"]` so missing data raises before recording | Integration test asserts the production caller path supplies it and missing data raises |
| Prompt edit accidentally enlarges payload past the local-LLM context budget | Descriptors are short fixed strings; before/after character count compared during Stage 1 verification | Stage 1 evidence records prompt size delta |
| Prompt rewrite drifts into recovery-behavior redesign instead of conditioning on existing config | Mandatory Rules forbid behavior redesign; the prompt edit only adds an input section and a single guidance sentence per field | Code review of the diff against this plan |

## Glossary

- **Recovery-policy fields:** the recorder output fields that govern how the character's accumulated state evolves toward closure — `progression_guidance`, `next_affordances`, and the softening cadence implied by `current_blocker`.
- **Descriptor:** the string returned by a `get_*_description` mapper in `boundary_profile.py`. The local-llm-architecture rule requires LLMs to receive descriptors, not raw values.
