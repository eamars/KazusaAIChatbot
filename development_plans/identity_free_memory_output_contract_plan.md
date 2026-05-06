# identity_free_memory_output_contract_plan

## Summary

- Goal: Enforce an identity-free first-person output contract for LLM-generated, character-facing memory text, then sanitize existing polluted database rows after the code-side guard is in place.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `local-llm-architecture`, `no-prepost-user-input`, `py-style`, `cjk-safety`, `test-style-and-execution`, `database-data-pull`, `python-venv`
- Overall cutover strategy: bigbang for future LLM output enforcement; migration for existing database content.
- Highest-risk areas: durable memory loss from over-strict validation, accidental input rewriting, stale embeddings/cache after DB repair, broad prompt drift outside memory surfaces.
- Acceptance criteria: no future prompt-facing character memory prose persists generated character names or assistant-role labels; existing active polluted memory rows are sanitized through a reviewed dry-run/apply flow; verification gates pass.

## Context

`user_memory_context` currently projects `user_memory_units` semantic text verbatim into cognition. Existing data shows generated memory text has mixed active-character references such as `助理`, `助手`, `角色`, `千纱`, `杏山千纱`, and first-person `我`. The immediate visible failure is an objective fact containing `助理多次回以'亲爱的'`, but the broader problem is that the memory writer has no output-side identity contract.

The user explicitly confirmed:

- Input must remain as-is.
- The scope must be strictly forced at the LLM output side.
- Database sanitization must be included as the final stage after the code-side fix.

This plan treats natural-language input payloads, chat history rows, and user utterances as source evidence. It does not rewrite or normalize source input before LLM calls. The fix is output-side: prompts tell memory LLM stages how generated memory prose must refer to the active character, and validators reject nonconforming generated output before persistence.

## Mandatory Skills

- `local-llm-architecture`: load before editing prompts, LLM handlers, cognition/RAG/memory context contracts, or background LLM behavior.
- `no-prepost-user-input`: load before changing consolidator memory, facts, promise, or relationship output handling. This plan must not introduce deterministic user-input interpretation.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt strings or tests containing Chinese/Japanese text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before read-only database exports and dry-run inspection.
- `python-venv`: load before running Python scripts, tests, or dependency commands.

## Mandatory Rules

- Input remains as-is. Do not rewrite, normalize, filter, translate, or project `decontexualized_input`, `chat_history_recent`, `final_dialog`, raw conversation history, user text, or source evidence to hide character names or role labels.
- Enforce identity rules only on LLM-generated output fields before persistence or before downstream generated-memory reuse.
- Do not add deterministic logic that decides whether a user request, preference, commitment, or fact should exist. The LLM still owns semantic extraction and channel selection.
- Output validation may reject or drop invalid generated text, but it must not rewrite generated text into a corrected semantic meaning.
- Durable character-facing memory prose must use `我` to refer to the active character.
- Durable character-facing memory prose must not generate active-character identity labels: `助理`, `助手`, `assistant`, `Assistant`, `角色`, `千纱`, `杏山千纱`, `Kazusa`, `Kyōyama Kazusa`.
- Generated memory prose should paraphrase direct user quotes that contain active-character names when the quote is not essential. Do not copy user-supplied active-character names into durable generated memory prose.
- Preserve storage role vocabulary where it is a machine field. Do not rename `conversation_history.role="assistant"` or other storage/transport role fields.
- Do not sanitize prompt-facing memory at read/projection time as the primary fix. Projection should reveal stored data quality, not hide it.
- Do not broaden this plan into a general memory quality rewrite, dedup overhaul, RAG redesign, affinity change, or role vocabulary refactor.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Add an output-side identity contract for all LLM-generated, durable, character-facing memory text listed in this plan.
- Add structural output validation that rejects generated memory text containing forbidden active-character references.
- Keep all source input payloads unchanged.
- Update memory-unit extractor and rewrite prompts to require first-person `我` and forbid generated identity labels.
- Update relationship/global-state/character-self-image memory-like prompts listed in `Change Surface` to follow the same output contract where they produce durable character-facing memory prose.
- Add tests proving invalid generated identity labels are rejected before persistence.
- Add a database sanitization CLI with dry-run and apply modes.
- Run database sanitization only after code-side tests pass.
- Recompute embeddings for sanitized `user_memory_units`.
- Invalidate affected RAG/user-profile/character-state caches after applying database changes.

## Deferred

- Do not rewrite raw `conversation_history`.
- Do not rename storage roles, adapter roles, message-envelope roles, or `assistant_moves`.
- Do not change RAG routing, retrieval slot planning, user lookup, or live context behavior.
- Do not redesign memory schemas beyond adding validation/report metadata needed by this plan.
- Do not change personality profile canonical `name`.
- Do not sanitize seed/world knowledge by forcing `我` where no active-character speaker is guaranteed.
- Do not add a new response-path LLM call.

## Cutover Policy

Overall strategy: bigbang for future generated-output enforcement; migration for existing database content.

| Area | Policy | Instruction |
|---|---|---|
| Future generated memory output | bigbang | Once code lands, every covered output boundary rejects nonconforming generated memory prose. No compatibility fallback that persists old-style identity text. |
| Source input payloads | compatible | Preserve existing input shapes and content. Existing prompts may receive the same source fields as before. |
| Stored conversation roles | compatible | Keep `role="assistant"` and related storage/transport vocabulary unchanged. |
| Existing active `user_memory_units` | migration | Sanitize through dry-run report, reviewed apply, embedding recompute, and cache invalidation. |
| Existing non-active memory rows | migration | Scan and report in dry-run. Apply only if the row is prompt-facing or explicitly included by the migration command. |
| Tests | bigbang | Update tests to assert the new output contract; do not keep tests that expect generated character names in durable memory prose. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by adding fallback persistence of old-style generated identity text.
- For bigbang areas, reject or drop nonconforming generated output instead of preserving it.
- For migration areas, follow the exact dry-run/apply phases in this plan.
- For compatible areas, preserve only the compatibility surfaces explicitly listed here.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside the listed change surface as high-scrutiny changes and report why they are necessary before editing.
- The agent may add helper functions or a new module only under the public interface defined in this plan.
- The agent must search for existing equivalent helpers before creating a new helper.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve this plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

Prompt-facing, character-owned generated memory text uses `我` for the active character and never stores generated active-character names or assistant-role labels. The LLM may see unchanged source inputs that contain `assistant`, `千纱`, `杏山千纱`, or user-authored names, but its generated durable memory output must not copy those labels as the active-character subject.

Covered future output fields:

- `user_memory_units.fact`
- `user_memory_units.subjective_appraisal`
- `user_memory_units.relationship_signal`
- `relationship_recorder.subjective_appraisals`
- `relationship_recorder.last_relationship_insight`
- `global_state_updater.reflection_summary`
- `character self_image.recent_window[].summary`
- `character self_image.historical_summary` when produced by the compression LLM

The projected `user_memory_context` remains a mechanical projection of stored `user_memory_units`; it should become clean because persisted memory units are clean.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Character reference in memory prose | Use `我` | It avoids user-supplied short-name contamination and assistant-role translation leakage. |
| Input handling | Preserve input as-is | The user explicitly required this, and the no-prepost-user-input rule forbids semantic input rewriting. |
| Enforcement point | Prompt plus output validation | Prompts guide the LLM; validators prevent polluted persistence when prompts fail. |
| Invalid generated output | Reject/drop the generated output field or candidate | Rewriting output locally would create semantic post-processing. Rejection is structural contract enforcement. |
| Read-time projection | Do not sanitize on read | Read-time cleanup hides bad data and lets polluted durable rows remain. |
| Existing DB repair | Run after code-side fix | Prevents the same contamination from being reintroduced during or after migration. |
| Response-path LLM budget | No new response-path calls | This is background consolidation and migration work; chat latency must not increase. |

## Contracts And Data Shapes

### New Module

Create `src/kazusa_ai_chatbot/memory_identity_contract.py`.

Public constants:

```python
CHARACTER_MEMORY_SELF_REFERENCE = "我"
FORBIDDEN_GENERATED_CHARACTER_REFERENCES = (
    "助理",
    "助手",
    "assistant",
    "Assistant",
    "角色",
    "千纱",
    "杏山千纱",
    "Kazusa",
    "Kyōyama Kazusa",
)
```

Public functions:

```python
def identity_contract_instruction() -> str:
    """Return reusable prompt text for character-facing memory output."""

def identity_contract_violations(
    generated_fields: dict[str, object],
    *,
    field_names: tuple[str, ...],
) -> list[dict[str, str]]:
    """Return structural violations in LLM-generated output fields.

    Only inspect fields named by field_names.
    Do not inspect source input fields.
    Do not rewrite values.
    """

def has_identity_contract_violation(
    generated_fields: dict[str, object],
    *,
    field_names: tuple[str, ...],
) -> bool:
    """Return whether generated fields violate the identity contract."""
```

Violation item shape:

```python
{
    "field": "fact",
    "term": "助理",
    "message": "generated memory output must use 我 for the active character",
}
```

The module owns only structural output validation and prompt-instruction text. It must not parse user intent, classify commitments, infer acceptance, or rewrite text.

### Prompt Contract Text

Every covered LLM prompt must include an output identity section equivalent to:

```text
# Output Identity Contract
- This contract applies only to fields you generate in the JSON output.
- Source inputs may contain names, role labels, and quotes; use them as evidence only.
- In generated character-facing memory prose, refer to the active character as “我”.
- Do not generate “助理”, “助手”, “assistant”, “Assistant”, “角色”, “千纱”, “杏山千纱”, “Kazusa”, or “Kyōyama Kazusa” as the active-character subject.
- If a user quote contains an active-character name, paraphrase it unless the exact quote is essential evidence.
- Return valid JSON only.
```

### Output Rejection Rules

- Memory-unit extractor candidate: if `fact`, `subjective_appraisal`, or `relationship_signal` violates the identity contract, reject the whole candidate and log a warning.
- Memory-unit rewrite output: if any rewritten semantic field violates the contract, raise `ValueError`; the existing update path must skip that candidate without writing.
- Relationship recorder: drop invalid `subjective_appraisals` items; if `last_relationship_insight` violates the contract, set it to empty so it is not persisted.
- Global state updater: if `reflection_summary` violates the contract, set it to empty so `upsert_character_state` preserves the existing value.
- Character image session/compress outputs: if generated summary violates the contract, skip appending/replacing that generated summary.

## LLM Call And Context Budget

### Runtime Code Path

Before:

- No additional LLM calls for identity validation.
- Background consolidation calls already include global state updater, relationship recorder, facts harvester/evaluator, memory-unit extractor, merge/rewrite/stability when candidates exist, and character image summary/compress when needed.

After:

- No new response-path LLM calls.
- No new background LLM calls for validation.
- Prompt text increases by a small fixed block, estimated under 800 Chinese/ASCII characters per covered prompt.
- Validation is deterministic structural checking over generated output only.
- Invalid output is dropped or skipped; no retry loop is added.

### Database Sanitization Path

- The migration CLI may make one background LLM rewrite call per polluted row in dry-run mode.
- The migration CLI is offline/operator-run only and must not run in the chat response path.
- Each rewrite payload contains one row's semantic fields plus the output identity contract.
- Hard cap: default `--limit 100` unless explicitly overridden.
- Dry-run output must record raw input, proposed output, validation status, and errors.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/memory_identity_contract.py`
  - New public contract module for prompt instruction text and generated-output validation.
- `src/scripts/sanitize_memory_identity_contract.py`
  - Dry-run/apply database sanitization CLI.
- `tests/test_memory_identity_contract.py`
  - Focused tests for validation helper behavior.
- `tests/test_memory_identity_output_contracts.py`
  - Integration-style tests for consolidator output rejection.
- `tests/test_memory_identity_sanitization_script.py`
  - CLI/unit tests for dry-run and apply mechanics with mocked DB/LLM.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - Add prompt identity contract to extractor and rewrite prompts.
  - Validate extractor candidates and rewrite outputs before persistence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - Add prompt identity contract to global state updater and relationship recorder prompts.
  - Validate generated `reflection_summary`, `subjective_appraisals`, and `last_relationship_insight`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Add prompt identity contract to self-image session and compression prompts.
  - Validate generated summaries before appending/replacing.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Add or expose a safe update path for sanitizer apply if existing `update_user_memory_unit_semantics(..., increment_count=False)` is insufficient for audit metadata.
- `tests/test_user_memory_units_rag_flow.py`
  - Add regression tests for candidate rejection and rewrite rejection.
- `tests/test_consolidator_reflection_prompts.py`
  - Add regression tests for invalid reflection/relationship output handling.

### Keep

- `src/kazusa_ai_chatbot/utils.py::trim_history_dict`
  - Keep source history projection unchanged for this plan.
- `src/kazusa_ai_chatbot/time_context.py::format_history_for_llm`
  - Keep input row content unchanged except existing timestamp formatting.
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`
  - Keep projection mechanical unless tests need imports updated. Do not add read-time sanitization.
- `conversation_history.role`
  - Keep `user | assistant` storage vocabulary unchanged.

## Data Migration

Migration happens only after code-side enforcement and tests pass.

### Dry-Run Command

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_identity_contract `
  --dry-run `
  --scan-active-user-memory-units `
  --scan-user-profiles `
  --scan-character-state `
  --limit 500 `
  --output test_artifacts\identity_memory_sanitization_dry_run.json
```

Dry-run report shape:

```json
{
  "query": {
    "dry_run": true,
    "scopes": ["user_memory_units", "user_profiles", "character_state"],
    "limit": 500
  },
  "records": [
    {
      "collection": "user_memory_units",
      "document_key": "unit_id",
      "document_id": "87e746a7b9f846c6add65ebbf6d59959",
      "status": "ready|blocked|unchanged",
      "violations": [{"field": "fact", "term": "助理"}],
      "before": {
        "fact": "...",
        "subjective_appraisal": "...",
        "relationship_signal": "..."
      },
      "after": {
        "fact": "...",
        "subjective_appraisal": "...",
        "relationship_signal": "..."
      },
      "validation_errors": []
    }
  ]
}
```

The dry-run must not mutate MongoDB.

### Apply Command

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_identity_contract `
  --apply `
  --input test_artifacts\identity_memory_sanitization_dry_run.json `
  --output test_artifacts\identity_memory_sanitization_apply_report.json
```

Apply behavior:

- Apply only records with `status="ready"` and no validation errors.
- Preserve document IDs, unit types, lifecycle timestamps unless the existing update helper requires `updated_at`.
- For `user_memory_units`, recompute embeddings through the normal helper path.
- Do not increment `count` for sanitation-only rewrites.
- Add an audit/merge-history entry indicating identity-contract sanitation where the collection supports audit metadata.
- Invalidate affected cache sources after writes.
- Leave blocked records unchanged and list them in the apply report.

### Post-Apply Verification

- Export affected rows again.
- Verify no active prompt-facing covered field contains forbidden generated character references.
- Verify representative `identify_user_image 673225019 --platform qq` output no longer shows `助理`, `助手`, `角色`, `千纱`, `杏山千纱`, `Kazusa` in generated memory prose.

## Implementation Order

1. Add module tests for `memory_identity_contract.py`.
   - Expected baseline before implementation: import or symbol failure.
2. Add integration tests for memory-unit extractor/rewrite output rejection.
   - Expected baseline: invalid generated terms are accepted or not checked.
3. Add tests for relationship/global-state/character-image output rejection.
   - Expected baseline: invalid generated terms are accepted or not checked.
4. Add tests for sanitizer dry-run/apply mechanics with mocked LLM and DB helpers.
   - Expected baseline: script missing.
5. Implement `memory_identity_contract.py`.
6. Wire memory-unit prompt contract and output validation.
7. Wire reflection prompt contract and output validation.
8. Wire character self-image prompt contract and output validation.
9. Implement sanitizer CLI.
10. Run focused tests and fix only within approved scope.
11. Run broader consolidation/memory tests.
12. Run dry-run sanitation report against the database.
13. Review dry-run report for blocked rows and obvious semantic damage.
14. Apply sanitation after review.
15. Re-export and verify sanitized database state.

## Progress Checklist

- [ ] Stage 1 - Contract module tests written
  - Covers: `tests/test_memory_identity_contract.py`.
  - Verify: focused test fails before module implementation or documents existing missing symbols.
  - Evidence: record baseline result in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - Contract module implemented
  - Covers: `src/kazusa_ai_chatbot/memory_identity_contract.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_memory_identity_contract.py -q`.
  - Evidence: record changed files and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - Memory-unit output enforcement complete
  - Covers: `persona_supervisor2_consolidator_memory_units.py`, `tests/test_user_memory_units_rag_flow.py`, `tests/test_memory_identity_output_contracts.py`.
  - Verify: focused memory-unit tests pass.
  - Evidence: record prompt render check and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - Reflection and self-image output enforcement complete
  - Covers: `persona_supervisor2_consolidator_reflection.py`, `persona_supervisor2_consolidator_images.py`, reflection/image tests.
  - Verify: focused consolidator reflection/image tests pass.
  - Evidence: record prompt render check and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - Sanitizer CLI implemented and tested
  - Covers: `src/scripts/sanitize_memory_identity_contract.py`, `tests/test_memory_identity_sanitization_script.py`.
  - Verify: sanitizer tests pass with mocked DB/LLM.
  - Evidence: record test output and sample dry-run fixture.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - Full code-side verification passed
  - Covers: all code-side enforcement before touching production-like DB rows.
  - Verify: all commands in `Verification` code-side gates pass.
  - Evidence: record complete command output summaries.
  - Handoff: next agent starts database dry-run.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 7 - Database dry-run report produced and reviewed
  - Covers: read-only export/sanitization dry-run.
  - Verify: dry-run command writes report; report has no invalid JSON and lists ready/blocked rows.
  - Evidence: record output path, counts, and blocked-row count.
  - Handoff: next agent applies only if review is complete.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 8 - Database sanitation applied and verified
  - Covers: apply command, embedding recompute, cache invalidation, post-apply export.
  - Verify: post-apply greps and `identify_user_image` smoke pass.
  - Evidence: record apply report path, affected counts, and smoke output summary.
  - Handoff: plan can move to completed after acceptance criteria are checked.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg -n "助理|助手|assistant|Assistant|角色|千纱|杏山千纱|Kazusa|Kyōyama" src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Allowed matches: prompt contract forbidden-term list and tests/fixtures explicitly asserting rejection.
- `rg -n "role.*assistant|assistant.*role" src/kazusa_ai_chatbot/utils.py src/kazusa_ai_chatbot/time_context.py`
  - Expected: existing storage/input role references remain; no source-input sanitization helper is added.
- `rg -n "sanitize.*history|normalize.*history|rewrite.*input|replace.*千纱" src/kazusa_ai_chatbot`
  - Expected: no new input-side rewrite helpers for this plan.

### Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_memory_identity_contract.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_identity_output_contracts.py -q
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidator_reflection_prompts.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_identity_sanitization_script.py -q
```

### Prompt Rendering

Run a lightweight import/render check for modified prompt templates:

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\memory_identity_contract.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py `
  src\scripts\sanitize_memory_identity_contract.py
```

### Database Dry Run

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_identity_contract `
  --dry-run `
  --scan-active-user-memory-units `
  --scan-user-profiles `
  --scan-character-state `
  --limit 500 `
  --output test_artifacts\identity_memory_sanitization_dry_run.json
```

### Database Apply

Run only after code-side verification and dry-run review:

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_identity_contract `
  --apply `
  --input test_artifacts\identity_memory_sanitization_dry_run.json `
  --output test_artifacts\identity_memory_sanitization_apply_report.json
```

### Post-Apply Smoke

```powershell
venv\Scripts\python.exe -m scripts.identify_user_image 673225019 --platform qq
venv\Scripts\python.exe -m scripts.export_user_memories 673225019 --platform qq --raw --limit 100 --output test_artifacts\identity_memory_post_apply_user_673225019.json
```

Post-apply report and export must not show forbidden generated character references in covered prompt-facing fields.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| LLM output with useful memory is dropped due to forbidden term in quoted user text | Prompt tells LLM to paraphrase user quotes; tests include quote-like cases | Memory-unit output tests |
| Code accidentally rewrites input history | Static grep and mandatory no-input-rewrite rule | Static greps and code review |
| Sanitizer changes semantics too aggressively | Dry-run report includes before/after and blocked rows; apply uses dry-run output | Dry-run review and post-apply smoke |
| Stale embeddings after repair | Use normal semantic update helper with embedding recompute | Apply report and vector field update test/mocking |
| Cache still serves polluted context | Emit/trigger cache invalidation for affected user/character sources | Apply report includes invalidation counts |
| Prompt examples reintroduce concrete character names | Static grep allows only forbidden-term lists/tests | Static grep |

## Operational Steps

1. Complete code-side stages and tests.
2. Run database dry-run in the project venv.
3. Review report counts and blocked rows. Do not manually edit MongoDB.
4. Apply only the validated dry-run report.
5. Re-export affected data.
6. Run `identify_user_image` smoke for the known polluted QQ user.
7. Record all evidence in this plan or an execution record before marking complete.

## Execution Evidence

- Static grep results:
- Focused test results:
- Prompt render / py_compile results:
- Database dry-run report:
- Database apply report:
- Post-apply export:
- `identify_user_image` smoke:
- Blocked rows / residual risk:

## Acceptance Criteria

This plan is complete when:

- Covered LLM prompts include the output identity contract.
- Covered LLM handlers reject/drop nonconforming generated memory output before persistence or generated-memory reuse.
- No source input rewriting or history normalization was introduced.
- `conversation_history.role="assistant"` and storage role vocabulary remain unchanged.
- Focused module, integration, and sanitizer tests pass.
- Database dry-run report is produced and reviewed.
- Database apply report shows sanitized ready rows were updated and blocked rows were left unchanged.
- Embeddings for sanitized `user_memory_units` are recomputed through the normal path.
- Cache invalidation is performed for affected sources.
- Post-apply export and `identify_user_image 673225019 --platform qq` no longer show forbidden generated character references in covered prompt-facing memory fields.

## Glossary

- Character-facing memory: Generated text consumed later as the active character's internal context, relationship memory, self-image, or prompt-facing user memory.
- Forbidden generated character reference: A generated active-character label that should not appear in covered durable memory prose, including `助理`, `助手`, `assistant`, `角色`, `千纱`, `杏山千纱`, and romanized forms.
- Input remains as-is: Source evidence text is passed through unchanged. The system may add prompt instructions and validate generated output, but it must not rewrite source messages or user text to make the LLM's job easier.
