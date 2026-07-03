# adapter semantic identity boundary and memory pollution plan

## Summary

- Goal: stop adapter-originated inconsistent participant labels, platform syntax, and occurrence placeholders from entering durable semantic storage; repair existing polluted conversation rows and quarantine or repair derived memory pollution.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `py-style`, `cjk-safety`, `test-style-and-execution`, `local-llm-architecture`, `memory-knowledge-maintenance`.
- Overall cutover strategy: bigbang for new adapter/envelope semantics; migration for existing stored data and derived memory pollution.
- Highest-risk areas: QQ reply hydration, Discord reply excerpts, storage-boundary validation, existing conversation embeddings, user profile display names, derived `memory` and `user_memory_units` rows.
- Acceptance criteria: new `/chat` user rows cannot persist `@mentioned-user-N`, platform-qualified fallback labels, or raw platform syntax outside `raw_wire_text`; same platform user id uses one canonical display-name hydration policy in sender, mention, and reply fields; migration dry-run reports dirty rows; apply repairs deterministic rows and leaves ambiguous memory rows inactive or review-listed.

## Context

### Development Report

The observed failure is a boundary failure before storage. A QQ message can be stored and later retrieved as:

```text
[timestamp] speaker reply_to group-alias: @real-nickname text;
reply object: group-alias: @mentioned-user-1 text
```

The root cause is not RAG projection. RAG and conversation-history renderers are thin readers of durable fields. The inconsistent data is created in adapter normalization, passed through the typed envelope, then stored by brain intake and `save_conversation(...)`.

Confirmed source findings:

- QQ mention display names use `select_qq_display_name(...)` in `src/adapters/napcat_qq_adapter/mention_hydration.py`.
- QQ reply target display names currently use a different card-first policy in `src/adapters/napcat_qq_adapter/reply_hydration.py`.
- QQ sender display name currently uses raw sender nickname in `src/adapters/napcat_qq_adapter/ws_adapter.py`.
- QQ segment reply metadata can provide partial reply fields in `src/adapters/napcat_qq_adapter/inbound_segments.py` and then block richer platform lookup in `reply_hydration.py`.
- QQ replied-message excerpt projection can be called without a mention label map, causing occurrence placeholders such as `@mentioned-user-1`.
- Discord top-level body text normalizes mention tags, but `referenced_message.content` is copied into `reply_excerpt` without the same projection.
- Brain intake stores `display_name`, `body_text`, `raw_wire_text`, `mentions`, and `reply_context` as supplied after envelope resolution.
- Existing migration support already scans `body_text` and `reply_context.reply_excerpt` for dirty transport syntax through `src/scripts/migrate_conversation_history_envelope.py` and `src/kazusa_ai_chatbot/db/script_operations.py`.

The current contract documented `@mentioned-user-N` as a fallback token. That token is now rejected for persisted semantic fields because it is occurrence-local, not identity-stable. The corrected boundary also rejects platform-qualified fallback labels such as `@qq-user:<id>` because the brain receives platform-agnostic semantic text; platform ids belong in typed metadata and raw replay fields.

### Post-Review Root Cause Correction

The first implementation attempt mitigated the visible symptom by replacing legacy occurrence placeholders and adding storage-boundary rejection. It did not address the direct QQ failure mode that produced the bad reply excerpt.

The direct root cause was in NapCat adapter reply hydration: `apply_replied_message_metadata(...)` projected the replied message excerpt before final mention display-name hydration had all labels available. For a reply excerpt containing `[CQ:at,qq=673225019]`, the early projection ran with an incomplete display map. Production legacy code turned that into `@mentioned-user-1`; the first mitigation would have turned it into `@user`. Both outcomes are lossy because the platform id is no longer available to the later normalizer for consistent display-name selection.

The final fix keeps raw reply CQ evidence adapter-internal until hydration is complete. Reply metadata now carries internal `reply_mention_display_names`; `QQAdapter._hydrate_reply_display_names(...)` merges reply-segment labels with current-message labels, hydrates reply-only mentions through the existing NapCat member lookup path, updates `reply_to_display_name` when the target id is resolved, and removes the internal field before building the brain-facing envelope. The final projection still happens inside the adapter normalizer, after the display map is complete.

### Memory Pollution Impact

Memory pollution exists in three layers:

- Direct conversation pollution: dirty `body_text`, `display_name`, `mentions.display_name`, and `reply_context` can be retrieved by recent history, RAG conversation evidence, reflection, and group scene review.
- Embedding pollution: dirty `body_text` changes conversation-history document embeddings. Dirty `reply_excerpt` does not directly drive the row embedding today, but it enters quote-aware RAG, current-event grounding, decontextualization, and evidence packets.
- Derived memory pollution: consolidation, reflection, shared memory promotion, and user memory units can preserve wrong identity facts or misleading statements such as a group alias being treated as a person, or `@mentioned-user-1` being treated as a stable participant.

The repair strategy is ordered: stop new writes first, repair source conversation rows second, then audit and repair or deactivate derived memory that was produced from polluted source rows.

## Mandatory Skills

- `development-plan`: load before changing this plan, executing it, reviewing it, or updating lifecycle status.
- `py-style`: load before editing Python production or script files.
- `cjk-safety`: load before editing any Python file that contains CJK string literals.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing RAG, cognition, prompt-facing context, memory, consolidation, or any LLM-adjacent data contract.
- `memory-knowledge-maintenance`: load before modifying curated shared `memory` rows or shared-memory maintenance workflows.

## Mandatory Rules

- Do not read `.env` during this plan.
- Production-code changes are authorized by the 2026-07-03 user instruction to execute this plan without subagents.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Do not introduce compatibility shims, dual read/write paths, adapter fallback bridges, or legacy placeholder support.
- Do not add response-path LLM calls. All new-write prevention, auditing, and deterministic repair logic belongs to adapters, message-envelope validation, database maintenance helpers, or scripts.
- Do not move platform parsing into the brain, RAG, cognition, dialog, consolidation, or database layers.
- Do not make LLM prompts interpret raw platform ids, CQ syntax, Discord syntax, or occurrence placeholders.
- Raw platform syntax is allowed only in `raw_wire_text` and explicit audit exports.
- `@mentioned-user-N`, `@mentioned-role-N`, `#mentioned-channel-N`, and `@mentioned-entity-N` are forbidden in persisted semantic fields after this plan.
- Platform-qualified fallback tokens are forbidden in semantic fields. If a display label cannot be resolved, semantic text uses a platform-neutral fallback such as `@user`, while platform ids remain only in typed metadata and `raw_wire_text`.
- If an adapter cannot produce typed identity metadata for a semantic mention or reply target that came from platform syntax, the request must fail closed before storage rather than writing an occurrence placeholder.
- Database migration must run dry-run before apply.
- Database apply steps must export backups before mutation and record counters after mutation.
- Re-embedding must run for any conversation row whose `body_text` changes.
- User-profile repair must be keyed by platform id and must not infer aliases from free text alone.
- Derived memory repair must prefer deterministic repair from repaired source rows; ambiguous memory must be archived, superseded, rejected, or exported for human review instead of remaining active.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- The `Execution Model` uses explicit user-approved fallback execution without subagents for this run.

## Must Do

- Replace all adapter user/bot mention fallback occurrence tokens with platform-neutral semantic fallback tokens.
- Update the message-envelope and adapter ICDs so they forbid occurrence placeholders in persisted semantic fields.
- Unify QQ participant label selection across sender, mentions, reply targets, segment reply metadata, and reply excerpt projection.
- Normalize Discord reply excerpts with the same semantic projection as Discord top-level body text.
- Add a storage-boundary semantic validator that rejects platform syntax and occurrence placeholders in persisted semantic fields before `save_conversation(...)`.
- Add focused adapter tests that reproduce the current QQ and Discord leaks and prove the new outputs.
- Add service/intake tests proving invalid semantic fields are rejected before persistence.
- Extend migration/audit support for dirty conversation rows, including occurrence placeholders and platform syntax in `body_text`, `display_name`, `mentions.display_name`, `reply_context.reply_to_display_name`, and `reply_context.reply_excerpt`.
- Add dry-run and apply paths for deterministic conversation repair.
- Recompute conversation embeddings for rows whose `body_text` changes.
- Add derived-memory audit reporting for active `memory` and `user_memory_units` rows that contain forbidden markers or references to repaired dirty conversation rows.
- Add deterministic repair or deactivation paths for derived memory where source evidence is unambiguous.
- Update tests that currently assert `@mentioned-user-N` output.
- Record execution evidence and migration counters in this plan before completion.

## Deferred

- Do not redesign RAG helper-agent routing.
- Do not rewrite cognition, dialog, or consolidation prompts to explain platform syntax.
- Do not add new LLM repair stages.
- Do not create a compatibility mode that preserves `@mentioned-user-N`.
- Do not add a new durable identity system outside `user_profiles`.
- Do not infer human aliases from message text.
- Do not repair ambiguous derived memory with generated guesses.
- Do not modify curated global memory JSONL unless a separate curated-memory change is required after the audit.
- Do not change adapter outbound delivery-mention rendering except where tests require replacing old fallback-token expectations.
- Do not run live database apply operations without explicit operator approval after dry-run evidence.

## Cutover Policy

Overall strategy: bigbang for runtime semantics, migration for stored data.

| Area | Policy | Instruction |
| --- | --- | --- |
| New adapter semantic text | bigbang | Replace occurrence fallback tokens directly. Do not keep `@mentioned-user-N` support for new writes. |
| QQ participant labels | bigbang | Use one canonical selector for sender, mentions, reply targets, and reply excerpt mention maps. |
| Discord reply excerpts | bigbang | Normalize reply excerpts before envelope construction. Do not store raw referenced-message content. |
| Message-envelope ICD | bigbang | Rewrite the fallback-token contract to platform-neutral semantic labels and forbidden occurrence/platform-qualified placeholders. |
| Brain storage boundary | bigbang | Reject invalid semantic envelope fields before persistence. Do not auto-repair live invalid payloads in the brain. |
| Conversation history rows | migration | Run dry-run audit, backup export, deterministic repair, re-embedding, and post-apply verification. |
| User profiles | migration | Repair only platform-account display labels that have deterministic platform id evidence. |
| Derived memory rows | migration | Repair deterministic rows, supersede or archive ambiguous active rows, and export manual review rows. |
| Tests | bigbang | Update old tests that assert occurrence placeholders and add new boundary tests. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `migration`, follow the exact migration phases and cleanup gates listed in this plan.
- If an area is `compatible`, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Runtime target:

```text
platform event
  -> adapter parses platform syntax
  -> adapter selects stable platform participant labels
  -> adapter projects semantic text and typed metadata
  -> brain validates semantic storage boundary
  -> conversation row is stored with clean semantic fields
  -> RAG, cognition, dialog, reflection, and consolidation read semantic fields
```

Observable behavior:

- QQ same platform user id has one visible label policy in the stored row's speaker, mention display, reply target display, and reply excerpt mentions.
- Discord reply excerpts do not store `<@...>`, `<@&...>`, `<#...>`, or custom emoji transport markers.
- Unknown display labels become platform-neutral fallback tokens such as `@user`, not occurrence placeholders or platform-qualified stand-ins.
- `raw_wire_text` remains available for audit and replay.
- Invalid adapter payloads fail before durable storage.
- Dirty historical rows can be counted, exported, repaired, re-embedded, and verified.
- Polluted active memory rows are not left active after audit if deterministic repair is unavailable.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| QQ canonical label source | Use normalized `nickname -> name -> card -> platform-neutral fallback`. | This matches the current mention selector and prevents group-card reply targets from diverging from sender and mentions. |
| Unknown user mention fallback | Use platform-neutral tokens, for example `@user`, `@role`, `#channel`, or `@entity`. | The brain-facing text must be platform-agnostic; platform ids stay in typed metadata and raw replay fields. |
| Unknown reply display fallback | Use platform-neutral labels such as `user`, `role`, `channel`, or `entity`. | Display fields should be semantic labels, not platform lookup stand-ins. |
| Missing stable id | Reject before storage when semantic text would need an identity fallback but no platform entity id exists. | Writing an occurrence placeholder creates unrecoverable ambiguity. |
| Brain validator role | Validate and reject only. | Adapters own platform parsing; the brain must not guess platform syntax semantics. |
| Migration repair source | Prefer typed metadata and parent conversation rows over raw text parsing. | Repaired rows should be tied to stable ids and stored facts, not free-text guesses. |
| Derived memory action | Repair when deterministic, deactivate when ambiguous. | Active polluted memory is worse than missing memory because it repeatedly contaminates future context. |
| LLM use | No new response-path LLM call. Offline manual review may use exported artifacts outside this plan's runtime path only after explicit approval. | The live chatbot path must stay bounded for local models. |

## Contracts And Data Shapes

### Semantic Token Contract

Forbidden persisted semantic tokens:

```text
@mentioned-user-N
@mentioned-role-N
#mentioned-channel-N
@mentioned-entity-N
```

Allowed platform-neutral fallback tokens:

```text
@user
@role
#channel
@entity
@character
```

Forbidden platform-qualified fallback tokens:

```text
@qq-user:<platform_user_id>
@discord-user:<platform_user_id>
@discord-role:<platform_role_id>
#discord-channel:<platform_channel_id>
qq-user:<platform_user_id>
discord-user:<platform_user_id>
```

The responsible agent may implement equivalent helper names, but the external behavior above is fixed.

### Adapter Helper Contract

Update or create adapter-owned helpers under `src/adapters`:

```python
def semantic_entity_fallback_label(
    *,
    entity_kind: str,
    mention_context: bool,
) -> str:
    ...
```

Required behavior:

- Return a non-empty platform-neutral semantic label.
- Never return occurrence-count labels.
- Never return platform-qualified id labels.
- Keep platform-specific parsing in adapter modules.

### Storage Boundary Validator Contract

Create a message-envelope validation entry point owned by `kazusa_ai_chatbot.message_envelope`:

```python
def validate_semantic_storage_fields(
    *,
    platform: str,
    display_name: str,
    envelope: MessageEnvelope,
) -> None:
    ...
```

Required behavior:

- Inspect `display_name`, `body_text`, `mentions[].display_name`, `reply.display_name`, and `reply.excerpt`.
- Reject `[CQ:`, Discord user/role/channel mention tags, Discord custom emoji tags, and occurrence fallback placeholders.
- Do not inspect or reject `raw_wire_text` for platform syntax.
- Raise `ValueError` with adapter scope context safe for logs.
- Perform structural validation only; do not rewrite values.

Brain intake must call this validator before user conversation rows are persisted.

### Migration Report Shape

The migration dry-run must write a JSON report under `test_artifacts/diagnostics/` or a caller-specified output path:

```json
{
  "generated_at": "ISO-8601 UTC",
  "dry_run": true,
  "conversation_history": {
    "dirty_rows": 0,
    "repairable_rows": 0,
    "ambiguous_rows": 0,
    "requires_reembedding": 0
  },
  "user_profiles": {
    "dirty_accounts": 0,
    "repairable_accounts": 0,
    "ambiguous_accounts": 0
  },
  "derived_memory": {
    "dirty_user_memory_units": 0,
    "dirty_shared_memory_units": 0,
    "repairable_units": 0,
    "deactivate_units": 0,
    "manual_review_units": 0
  }
}
```

The apply report must include the same counters plus mutation counts and backup file paths.

### Derived Memory Lifecycle Contracts

Add a shared-memory rejection API owned by `kazusa_ai_chatbot.memory_evolution`:

```python
async def reject_memory_unit(
    *,
    active_unit_id: str,
    reason: str,
    storage_timestamp_utc: str,
) -> EvolvingMemoryDoc:
    ...
```

Required behavior:

- Accept only an active shared-memory unit.
- Set `status` to `rejected`.
- Set `updated_at` to `storage_timestamp_utc`.
- Preserve lineage, evidence refs, privacy review, source metadata, and existing content for audit.
- Use the memory-evolution write guard.
- Invalidate Cache2 with `source="memory"` before returning.

Add a maintenance-only user-memory archive helper in `kazusa_ai_chatbot.db.script_operations`:

```python
async def archive_user_memory_unit_for_semantic_identity_repair(
    *,
    unit_id: str,
    reason: str,
    storage_timestamp_utc: str,
) -> dict[str, object]:
    ...
```

Required behavior:

- Match only `status="active"`.
- Set `status="archived"`, `archived_at`, and `updated_at`.
- Append a `merge_history` entry with operation `semantic_identity_repair_archive`.
- Return matched and modified counts.
- Do not call `update_user_memory_unit_lifecycle(...)`, because that runtime helper is active-commitment-only.

## LLM Call And Context Budget

- Before: no LLM call is required for adapter normalization, message-envelope validation, storage writes, or conversation-history migration.
- After: no LLM call is added to the live response path.
- Response-path budget impact: zero calls, zero added prompt context.
- Background/offline impact: deterministic audit and migration only.
- Manual review artifacts may later be inspected by a human or an explicitly approved offline LLM workflow, but that is outside this plan.
- Local LLM safety: runtime prompts continue to receive semantic identity labels and cleaned text; they are not asked to infer raw platform syntax or storage schema.

## Change Surface

### Modify

- `src/adapters/envelope_common.py`: replace occurrence fallback formatting with platform-neutral semantic fallback token support.
- `src/adapters/discord_adapter.py`: normalize reply excerpts with the same projection used for top-level Discord body text; collect referenced-message mention maps when available.
- `src/adapters/napcat_qq_adapter/mention_hydration.py`: keep the canonical QQ display selector for hydrated display labels.
- `src/adapters/napcat_qq_adapter/reply_hydration.py`: use the canonical QQ selector; preserve raw reply excerpt evidence until final display-name hydration; carry reply mention labels internally.
- `src/adapters/napcat_qq_adapter/inbound_segments.py`: use the canonical selector for segment reply sender labels and avoid blocking richer hydration when fields are partial.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`: use the canonical selector for top-level sender display names and profile display-name input.
- `src/adapters/README.md`: update adapter ICD fallback-token and boundary rules.
- `src/adapters/napcat_qq_adapter/README.md`: update QQ label and reply hydration contracts.
- `src/kazusa_ai_chatbot/message_envelope/README.md`: update semantic fallback contract and forbidden occurrence placeholders.
- `src/kazusa_ai_chatbot/message_envelope/types.py` if type comments need updated documentation only.
- `src/kazusa_ai_chatbot/brain_service/intake.py`: call the storage-boundary semantic validator before persistence.
- `src/kazusa_ai_chatbot/brain_service/README.md`: update adapter responsibilities that currently name occurrence placeholders.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add maintenance helpers for dirty semantic-field scanning and deterministic repair.
- `src/kazusa_ai_chatbot/memory_evolution/__init__.py`: export the shared-memory rejection API.
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`: implement `reject_memory_unit(...)` with write guard and Cache2 invalidation.
- `src/kazusa_ai_chatbot/memory_evolution/README.md`: document the maintenance rejection API and its limited use.
- `src/scripts/migrate_conversation_history_envelope.py`: extend the existing migration or route to a new focused repair script while preserving dry-run-first behavior.
- `src/scripts/create_conversation_history_embedding.py`: reuse existing re-embedding entry point as the approved re-embedding step; modify only if the migration needs a narrower batch selector.
- `development_plans/README.md`: register this active bugfix plan.

### Create

- `src/kazusa_ai_chatbot/message_envelope/semantic_validation.py`: storage-boundary semantic validator.
- `src/scripts/audit_semantic_identity_pollution.py`: read-only audit/report CLI for conversation rows, user profiles, shared memory, and user memory units.
- `src/scripts/repair_semantic_identity_pollution.py`: dry-run/apply repair CLI for deterministic repairs and deactivation actions.
- Focused tests in existing files rather than new broad suites:
  - `tests/test_adapter_envelope_normalizers.py`
  - `tests/test_runtime_adapter_registration.py`
  - `tests/test_conversation_history_migration_script.py`
  - `tests/test_message_envelope.py`
  - `tests/test_service_background_consolidation.py`
  - `tests/test_memory_evolution_repository.py`
  - `tests/test_user_memory_unit_lifecycle.py`

### Keep

- Keep `raw_wire_text` as audit/replay text that may contain platform syntax.
- Keep adapter ownership of platform parsing.
- Keep RAG, cognition, dialog, reflection, and consolidation as consumers of semantic fields rather than platform parsers.
- Keep existing memory-evolution public APIs for shared memory lifecycle.

### Delete

- Delete no production modules in this plan.
- Remove old occurrence-placeholder expectations from tests and docs.

## Data Migration

Migration runs after new-write prevention has passed focused tests.

### Phase 1 - Backup And Dry Run

- Export affected `conversation_history` rows matching forbidden markers.
- Export affected `user_profiles` platform accounts by platform id and display label.
- Export affected active `user_memory_units`.
- Export affected active shared `memory` rows through approved memory export tooling or `db.script_operations`.
- Write a dry-run report with repairability classification.

### Phase 2 - Conversation Repair

- Repair `body_text` only when typed mention metadata, platform id fallback, or parent row evidence gives an unambiguous replacement.
- Repair `reply_context.reply_excerpt` from parent conversation row `body_text` when `reply_to_message_id` resolves exactly.
- Repair `reply_context.reply_to_display_name` from parent row `display_name` when `reply_to_platform_user_id` and platform message id match.
- Repair `mentions.display_name` using canonical display labels or platform-neutral fallback labels.
- Leave ambiguous rows unchanged and export them in the manual-review file.
- Recompute embeddings for rows whose `body_text` changes.

### Phase 3 - User Profile Repair

- Update platform-account `display_name` only when the same `(platform, platform_user_id)` has a deterministic canonical label from repaired conversation rows or adapter lookup evidence.
- Do not add suspected aliases from this migration.
- Do not merge user profiles in this plan.

### Phase 4 - Derived Memory Repair

- For `user_memory_units`, repair semantic text only when the polluted token maps to a repaired conversation source or a single platform id. Archive active rows that still contain forbidden markers after deterministic repair.
- For `user_memory_units`, use `archive_user_memory_unit_for_semantic_identity_repair(...)` for active non-repairable rows. Do not use `update_user_memory_unit_lifecycle(...)` for this maintenance action.
- For shared `memory`, use `memory_evolution.supersede_memory_unit(...)` when a deterministic clean replacement can be built. If no deterministic replacement exists, call `memory_evolution.reject_memory_unit(...)` and include the row in the manual-review export.
- Preserve evidence refs, lineage, timestamps, and privacy review metadata on repaired replacements.
- Invalidate memory Cache2 after any shared-memory mutation.

### Phase 5 - Verification And Cleanup

- Re-run dirty-row scan and require zero forbidden markers in active semantic fields except rows explicitly exported as manual review and excluded from active retrieval.
- Re-run focused RAG/conversation projection tests to ensure retrieval surfaces cleaned data.
- Record migration counters and backup paths in `Execution Evidence`.

## Overdesign Guardrail

- Actual problem: adapter-normalized semantic fields currently store inconsistent participant labels, raw platform syntax, and occurrence-local placeholder labels that later pollute retrieval and memory.
- Minimal change: fix adapter normalization, add a deterministic storage-boundary validator, update the message-envelope contract, and provide dry-run-first maintenance repair for already polluted rows.
- Ownership boundaries: adapters own platform parsing and label selection; message envelope owns typed semantic invariants; brain service owns validation and persistence coordination; database maintenance owns repair mechanics; RAG/cognition/dialog/consolidation consume cleaned semantic fields; LLM stages do not parse platform syntax.
- Rejected complexity: no response-path LLM repair, no compatibility mode, no prompt-side platform syntax explanations, no new identity collection, no alias inference, no background auto-repair loop, no dual storage fields, no adapter-to-brain fallback mapper.
- Evidence threshold: add new architecture only after a concrete platform cannot provide stable ids or display labels under this contract and a separate approved plan defines that platform's identity boundary.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target module as high-scrutiny changes.
- The target ownership boundary is adapter normalization, message-envelope validation, brain intake validation call site, and maintenance migration scripts.
- Changes to RAG, cognition, dialog, reflection, consolidation, scheduler, dispatcher, or LLM prompts are forbidden unless a test proves the semantic validator or migration cannot be integrated without that exact change.
- If equivalent helper behavior already exists, abstract or move it into the appropriate common location instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent adds focused tests for semantic fallback tokens in `tests/test_adapter_envelope_normalizers.py`.
   - Expected pre-implementation result: tests fail because occurrence placeholders are still produced.
2. Parent adds QQ runtime tests in `tests/test_runtime_adapter_registration.py` for sender, mention, reply target, segment reply, and replied-message excerpt consistency.
   - Expected pre-implementation result: tests fail on card-first reply display, partial hydration return, or placeholder excerpt.
3. Parent adds Discord reply excerpt normalization test in `tests/test_adapter_envelope_normalizers.py`.
   - Expected pre-implementation result: test fails because raw `<@...>` remains in reply excerpt.
4. Parent adds storage-boundary validation tests in `tests/test_message_envelope.py` and `tests/test_service_background_consolidation.py`.
   - Expected pre-implementation result: tests fail because no semantic validator exists or is not called.
5. Parent starts the production-code subagent after the focused test contract is established.
   - Ownership: production adapter, message-envelope, brain-intake, docs, and migration helper changes only.
6. Production-code subagent implements adapter fallback token and QQ/Discord normalization changes.
7. Production-code subagent implements `semantic_validation.py` and wires it into brain intake.
8. Parent adds migration dry-run and deterministic repair tests in `tests/test_conversation_history_migration_script.py`.
   - Expected pre-implementation result: tests fail until migration helpers detect placeholders and display-field pollution.
9. Production-code subagent implements maintenance helpers and scripts.
10. Parent adds derived-memory audit and repair tests in memory-evolution and user-memory lifecycle test files.
11. Production-code subagent implements deterministic derived-memory repair or deactivation paths.
12. Parent runs focused tests and loops back to the failing stage until green.
13. Parent runs broader adapter, service, migration, and memory verification commands.
14. Parent runs dry-run audit only when an operator has approved live DB inspection.
15. Parent starts the independent code-review subagent after all planned verification passes.
16. Parent remediates review findings inside this change surface and reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established.
  - Covers: implementation steps 1-4.
  - Verify: run the exact focused tests and record expected failures.
  - Evidence: list failing tests and failure reasons in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-07-03`.
- [x] Stage 2 - runtime new-write prevention implemented.
  - Covers: implementation steps 5-7.
  - Verify: adapter and message-envelope focused tests pass.
  - Evidence: changed files, test command output, and static grep results.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-07-03`.
- [x] Stage 3 - migration and audit tooling implemented.
  - Covers: implementation steps 8-11.
  - Verify: migration and memory audit tests pass.
  - Evidence: dry-run report fixture output and test command output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-07-03`.
- [x] Stage 4 - full verification completed.
  - Covers: implementation steps 12-14.
  - Verify: all commands under `Verification` pass or are recorded as operator-blocked.
  - Evidence: command outputs, live DB approval status, dry-run counters when run.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-07-03`.
- [x] Stage 5 - independent code review completed.
  - Covers: implementation steps 15-16.
  - Verify: review subagent reports no unresolved blockers; affected tests are rerun after fixes.
  - Evidence: review findings, fixes, rerun commands, residual risks, approval status.
  - Handoff: plan can move to completion only after this stage is signed.
  - Sign-off: `Codex/2026-07-03`.

## Verification

### Static Greps

- `rg "@mentioned-(user|role|entity)-\\d+|#mentioned-channel-\\d+" src tests`
  - Expected after implementation: matches only in migration tests, historical documentation explaining forbidden legacy markers, and dirty-fixture strings.
  - Forbidden after implementation: production runtime code creating those tokens for new messages.
- `rg "\\[CQ:|<@!?\\d+>|<@&\\d+>|<#\\d+>" src\\kazusa_ai_chatbot src\\adapters tests`
  - Expected after implementation: parser regexes, raw-wire tests, migration dirty fixtures, and outbound rendering tests may match.
  - Forbidden after implementation: tests or production code expecting those markers in `body_text` or `reply_excerpt`.
- `rg "project_qq_semantic_text\\([^\\n]*,\\s*\\{\\s*\\}" src\\adapters`
  - Expected after implementation: zero matches. Exit code 1 from `rg` is acceptable.

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_runtime_adapter_registration.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_message_envelope.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_conversation_history_migration_script.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_memory_evolution_repository.py tests\test_user_memory_unit_lifecycle.py -q`

### Broader Regression

- `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py tests\test_cognition_current_event_grounding.py tests\test_memory_retrieval_tools.py tests\test_rag_projection.py -q`
- `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`

### Migration Dry Run

- `venv\Scripts\python.exe -m scripts.audit_semantic_identity_pollution --output test_artifacts/diagnostics/semantic_identity_pollution_dry_run.json`
  - Expected: exits 0 and writes counters without mutating DB.
  - Requires explicit live DB inspection approval.
- `venv\Scripts\python.exe -m scripts.repair_semantic_identity_pollution --dry-run --output test_artifacts/diagnostics/semantic_identity_repair_dry_run.json`
  - Expected: exits 0 and reports planned repairs without mutating DB.
  - Requires explicit live DB inspection approval.

### Migration Apply

- `venv\Scripts\python.exe -m scripts.repair_semantic_identity_pollution --apply --backup-dir test_artifacts/backups/semantic_identity_boundary`
  - Expected: exits 0, writes backup files first, reports mutation counters, and refuses to run without backup output.
  - Requires explicit operator approval after dry-run review.
- `venv\Scripts\python.exe -m scripts.create_conversation_history_embedding --only-missing`
  - Expected: exits 0 after any migration that marks rows for re-embedding or leaves no rows needing embedding repair.
  - Use the existing script flags if implementation adds a narrower row selector.

## Independent Plan Review

Plan review was performed by the drafting agent from a fresh review posture because the current user request asked for plan review but did not explicitly authorize subagent delegation.

Review scope:

- Architecture alignment with adapter, message-envelope, brain-service, database, RAG, and memory-evolution ICDs.
- Instruction completeness for production-code agents.
- Creativity suppression and no compatibility shim policy.
- Data migration safety and memory pollution handling.
- Verification specificity.

Surfaced issues and resolutions:

| Finding | Severity | Resolution |
| --- | --- | --- |
| The initial high-level plan did not name exact change files or test files. | Blocker | Added `Change Surface`, `Implementation Order`, and `Verification` with exact paths and commands. |
| The initial high-level plan did not define the replacement for `@mentioned-user-N`. | Blocker | First addressed with platform-qualified fallback contracts, then corrected after boundary review to platform-neutral semantic fallbacks with platform ids kept in typed metadata. |
| The initial high-level plan did not state whether the brain should repair or reject invalid live payloads. | Blocker | Set the brain validator to reject only; adapters own repair before `/chat`. |
| The initial high-level plan did not separate source conversation repair from derived memory repair. | Blocker | Added ordered `Data Migration` phases. |
| The initial high-level plan risked broad prompt/RAG changes. | Blocker | Added mandatory no-prompt, no-response-path-LLM, and forbidden change-surface rules. |
| The initial high-level plan did not define migration evidence. | Blocker | Added dry-run/apply report shape, backup requirements, and execution evidence requirements. |
| Existing docs still permit occurrence placeholders. | Blocker | Added ICD updates to `Must Do` and `Change Surface`. |
| Derived-memory deactivation was underspecified for shared `memory`. | Blocker | Added `memory_evolution.reject_memory_unit(...)` contract with write guard and Cache2 invalidation. |
| Generic `user_memory_units` archiving could be confused with the active-commitment lifecycle helper. | Blocker | Added maintenance-only `archive_user_memory_unit_for_semantic_identity_repair(...)` and explicitly forbade using `update_user_memory_unit_lifecycle(...)` for this repair. |

Approval status: all review blockers identified in this planning pass were addressed. Execution was authorized by the user's 2026-07-03 instruction to execute the plan without subagents and prove mitigation/sanitation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

For this execution, the user explicitly required no subagent. The independent review gate was therefore performed as a no-subagent fallback review by the parent agent against the diff, plan, tests, static scans, migration reports, and final database audit.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt-adjacent, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, prompt/RAG payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused and regression tests, execution evidence, migration dry-run reports, backup behavior, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Adapter and message-envelope docs forbid occurrence fallback placeholders and platform-qualified fallback labels in persisted semantic fields.
- QQ sender, mention, reply target, and reply excerpt paths use one canonical participant label policy.
- Discord reply excerpts are normalized with the same semantic projection as top-level Discord content.
- Brain intake rejects invalid semantic fields before user-row persistence.
- Focused tests prove no new `@mentioned-user-N`, `@qq-user:<id>`, or raw platform syntax enters `body_text` or `reply_excerpt`.
- Existing tests that expected occurrence placeholders are updated to platform-neutral fallback tokens.
- Conversation-history migration dry-run reports dirty, repairable, ambiguous, and re-embedding counters.
- Migration apply backs up rows before mutation and recomputes embeddings for changed `body_text` rows.
- Active derived memory rows containing forbidden markers are repaired, superseded, archived, rejected, or exported for manual review; they are not silently left active.
- Full non-live test suite passes or any remaining failures are documented as unrelated with evidence.
- Independent code review has no unresolved blockers.

## Final Consistency Check

Two implementation attempts were compared before closure:

- Attempt 1 mitigated the symptom by replacing legacy occurrence placeholders, adding semantic-field validation, and repairing historical data. That was necessary but incomplete because it still allowed the QQ reply excerpt to be projected before hydration had enough identity evidence.
- Attempt 2 fixed the direct root cause at the NapCat adapter boundary. Replied-message raw CQ text remains adapter-internal until display-name hydration merges current-message labels, reply-segment labels, and reply-only NapCat lookups. The brain-facing envelope receives only platform-agnostic semantic text and typed metadata.
- Adjacent paths were checked: Discord reply excerpt projection, QQ current-message mentions, QQ reply target display, QQ reply-only mentions, brain storage validation, conversation repair tooling, and derived-memory quarantine helpers.
- Boundary rule after review: the adapter may parse CQ/Discord syntax and use platform ids for lookup and typed metadata; the brain validates and stores semantic fields but does not parse platform wire syntax or interpret platform ids as text.

## Execution Evidence

- Draft creation: completed on 2026-07-03, then archived at `development_plans/archive/completed/bugfix/adapter_semantic_identity_boundary_and_memory_pollution_plan.md`.
- Registry update: completed on 2026-07-03 in `development_plans/README.md`.
- Plan review: drafting-agent review completed; blockers listed and addressed in `Independent Plan Review`.
- Focused test failures before implementation:
  - New adapter tests initially failed because QQ and Discord fallback paths still produced legacy occurrence placeholders or raw reply excerpts.
  - Final root-cause regression tests initially failed because QQ reply excerpt projection happened before display-name hydration, producing a group-card reply target or `@user` instead of the resolved mention label.
  - Message-envelope tests initially failed because `validate_semantic_storage_fields` did not exist.
  - Service intake test initially failed because invalid semantic fields were not rejected before persistence.
  - Memory lifecycle tests initially failed because shared-memory rejection and user-memory semantic-identity archive helpers did not exist.
  - Migration tests initially failed because the semantic-identity repair script did not exist.
- Production implementation evidence:
  - Adapter fallback labels now use platform-neutral semantic tokens through `semantic_entity_fallback_label(...)`.
  - QQ sender, mention, reply target, and reply excerpt paths use the canonical QQ label policy after adapter-side hydration is complete.
  - QQ reply hydration preserves raw reply excerpt evidence adapter-internal and removes internal `reply_mention_display_names` before envelope construction.
  - Discord reply excerpts are normalized with the same projection as top-level Discord content.
  - Brain intake calls `validate_semantic_storage_fields(...)` before `save_conversation(...)`.
  - Maintenance helpers and scripts were added for semantic-identity audit, backup, deterministic repair, re-embedding, profile repair, user-memory archive, and shared-memory rejection.
- Migration dry-run evidence:
  - `scripts.audit_semantic_identity_pollution --output test_artifacts\diagnostics\semantic_identity_pollution_dry_run.json`: initial report found `16633` dirty conversation rows, `0` dirty user-profile accounts, `0` dirty user-memory units, and `0` dirty shared-memory units.
  - `scripts.repair_semantic_identity_pollution --dry-run --batch-size 50000 --output test_artifacts\diagnostics\semantic_identity_repair_dry_run.json`: final pre-apply report found `16634` dirty conversation rows, `16634` repairable, `0` ambiguous, and `1035` requiring re-embedding.
- Migration apply evidence:
  - Primary apply repaired `16635` conversation rows and re-embedded `1035`; backups were written under `test_artifacts\backups\semantic_identity_boundary\`.
  - Follow-up repair passes handled rows written by a concurrently running old deployment during verification: `9`, then `2`, then `4`, then `5` conversation rows.
  - Final read-only signoff audit `test_artifacts\diagnostics\semantic_identity_pollution_absolute_final.json` at `2026-07-03T07:49:51.234650+00:00` reported `0` dirty conversation rows, `0` dirty user-profile accounts, `0` dirty user-memory units, and `0` dirty shared-memory units.
- Verification evidence:
  - `venv\Scripts\python.exe -m py_compile ...` for all touched Python files: passed.
  - `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py -q`: `93 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_message_envelope.py tests\test_conversation_history_migration_script.py -q`: `16 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`: `30 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_memory_evolution_repository.py tests\test_user_memory_unit_lifecycle.py -q`: `11 passed`.
  - `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py tests\test_cognition_current_event_grounding.py tests\test_memory_retrieval_tools.py tests\test_rag_projection.py -q`: `68 passed`.
  - `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`: `2705 passed, 2 skipped, 408 deselected`.
  - `rg "project_qq_semantic_text\([^\n]*,\s*\{\s*\}" src\adapters`: no matches.
  - Legacy placeholder grep now matches only negative assertions, migration fixtures, forbidden-marker docs, and historical/archive text; no production runtime creator remains.
  - Raw platform syntax grep matches parser regexes, raw-wire fixtures, migration dirty fixtures, outbound native-render tests, and safety tests; no new semantic storage expectation remains.
  - Final root-cause verification on 2026-07-03: `test_napcat_reply_excerpt_reuses_hydrated_current_mention_label` and `test_napcat_reply_excerpt_hydrates_reply_only_mentions` both passed after failing on the pre-fix path.
  - Final closeout compile on 2026-07-03: `venv\Scripts\python.exe -m py_compile ...` for touched adapter, brain, migration, memory, and test Python files passed.
  - Final closeout focused regression on 2026-07-03: `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py tests\test_message_envelope.py tests\test_conversation_history_migration_script.py tests\test_service_background_consolidation.py tests\test_memory_evolution_repository.py tests\test_user_memory_unit_lifecycle.py -q`: `155 passed`.
  - Final static boundary scan on 2026-07-03 found `@mentioned-*`, platform-qualified fallback labels, and raw platform syntax only in validators, forbidden-marker docs, and negative or dirty fixtures; no adapter runtime generation path remained.
  - Final stale-plan scan found no target-state references to the rejected id-bearing fallback design.
  - Final `git diff --check` passed with line-ending warnings only.
- Independent code review evidence:
  - Per user instruction, no subagent was used. Parent-agent fallback review inspected the plan, tracked and untracked diffs, runtime boundary changes, scripts, tests, static scans, DB reports, and `git diff --check`.
  - `git diff --check` passed with line-ending warnings only.
  - No unresolved code blockers found.
  - Residual operational risk: a running old deployment wrote dirty QQ reply-excerpt rows while verification and signoff were running. The database was repaired back to zero, but that running process must be restarted or redeployed with this code to prevent further old-format writes outside this workspace.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Rejecting invalid live payloads can surface adapter bugs as request failures. | Fix adapters first and run focused adapter tests before wiring validator into persistence. | Adapter tests and service intake tests pass. |
| Platform-qualified fallback tokens expose platform ids in semantic text. | Forbid them in semantic fields; keep platform ids only in typed metadata, lookup code, and raw replay fields. | Storage-boundary tests and static scans. |
| Historical rows cannot all be deterministically repaired. | Export ambiguous rows and remove polluted derived memory from active retrieval instead of guessing. | Dry-run report and post-apply dirty scan. |
| Re-embedding can be skipped after `body_text` changes. | Migration apply records changed row ids and runs conversation embedding repair. | Embedding repair command evidence. |
| User profile display labels can be overcorrected. | Update only by exact platform id evidence. | User-profile repair report includes repairable and ambiguous counters. |
| Derived memory may lose useful information when archived or superseded. | Preserve backups, evidence refs, and manual review exports. | Backup files and derived-memory report. |
| Scope can drift into prompt/RAG redesign. | Keep prompt and RAG code out of change surface unless a failing integration test proves a necessary contract update. | Static diff review and independent code review. |
