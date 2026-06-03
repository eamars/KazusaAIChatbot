# group chat user style image plan

## Summary

- Goal: Let strongly attributed group-chat evidence update `user_style_image`
  through the same canonical user-style extraction and persistence path used by
  private-chat reflection.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, and `no-prepost-user-input` before touching any
  user-input interpretation path.
- Overall cutover strategy: compatible and additive. Existing private
  user-style generation and group-channel style generation must keep working.
- Coordination gate: do not execute this plan while
  `development_plans/active/short_term/reflection_phase_scheduled_group_review_plan.md`
  is `in_progress`.
- Highest-risk areas: adjacent-user attribution, broadcast self-cognition
  leakage, creating a second user-style semantic path, raw group transcript
  leakage into style storage, and accidental RAG or `user_memory_units`
  exposure.
- Acceptance criteria: group-chat user-style capture uses structural
  attribution only, excludes unaddressed broadcast rows, shares one
  user-style extractor/upsert path with private chat, preserves existing
  consumers and storage shape, and is covered by deterministic tests.

## Context

Current `user_style_image` generation only comes from private daily
reflections. Group daily reflections write `group_channel_style`, not
user-scoped style. This causes production QQ user-style coverage to be sparse
because Kazusa talks mostly in group chat.

RCA conclusions to carry into implementation:

- `user_style_image` is abstract handling guidance for one `global_user_id`.
  It is not `user_image`, not `user_memory_units`, and not RAG evidence.
- Existing database state confirmed one real QQ user-style image for
  `qq:673225019`; other user-style rows were debug users. Group-channel style
  rows exist but are keyed by channel, not user.
- Recent self-cognition audit showed many group self-cognition messages are
  stored as `broadcast=true`, `addressed_to_global_user_ids=[]`, with no
  persisted `mentions` or `reply_context`, even when visible text names one or
  more users. These rows must not become user-style evidence.
- Last-7-day strict eligibility audit, excluding `qq:673225019`, showed the
  hard rule is still usable: 27 QQ users had at least one eligible event, 11
  had at least 8 eligible events, and 7 had at least 16 eligible events.
- The strongest candidate was `qq:925059922`
  (`global_user_id=eaf9e90d-9caa-443a-8af5-715daa9d9917`) with 43 eligible
  group events and no existing `user_style_image`.
- Implementation inspection found `db.conversation_reflection
  .list_reflection_scope_messages(...)` is the correct read boundary for
  reflection windows, but its current projection does not expose the structural
  attribution fields required by this plan. This plan authorizes only the
  narrow projection extension listed in `Change Surface`.

## Mandatory Skills

- `development-plan`: load before approving, executing, or updating this plan.
- `local-llm-architecture`: load before changing reflection prompts,
  background LLM calls, source packets, or style extraction contracts.
- `py-style`: load before editing Python production or test files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `no-prepost-user-input`: load before any change that might interpret user
  commands, commitments, preferences, or reply-style requests.

## Mandatory Rules

- This draft is not executable until its status is changed to `approved` or
  `in_progress`.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Deterministic code owns source eligibility, target attribution, row
  filtering, caps, persistence validation, and duplicate suppression.
- The LLM owns only the semantic style judgment after deterministic code has
  produced a strongly attributed single-user source packet.
- The LLM-facing extractor payload must be an allowlist, not a dump of the
  source packet. Include only fields required for style abstraction; keep
  source ids, user ids, attribution metadata, thresholds, DB fields, and audit
  lineage out of the prompt.
- Group source message reads must go through
  `kazusa_ai_chatbot.db.conversation_reflection.list_reflection_scope_messages`;
  reflection-cycle code must not access MongoDB or `conversation_history`
  directly.
- Do not extend `ReflectionWorkerResult` or interaction-style storage schemas.
  Count a daily doc as succeeded when any approved style write occurs for that
  doc; log detailed participant-source counts instead of changing result shape.
- Do not infer target users from visible text, display names, nicknames,
  pronouns, or keyword matching.
- Do not include unaddressed assistant broadcast rows in user-style evidence.
  Broadcast rows qualify only when a persisted structural target maps to
  exactly one target user.
- Group prompt-facing evidence rows must include compact message content as
  style signal and must not include ids, display names, reply metadata, raw
  wire text, source refs, or adjacent-user rows.
- Do not write group-derived observations to `user_memory_units`, RAG evidence,
  `promoted_reflection_context`, dialog state, or self-cognition state.
- Do not create a second group-specific user-style output schema, sanitizer,
  storage slot, or runtime consumer path.
- Do not read `.env`. Do not run live DB inspection in this plan.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.

## Must Do

- Refactor user-style generation around one canonical user-style source
  contract and one canonical source-to-overlay processor.
- Keep private daily reflection user-style generation on that same canonical
  processor.
- Add a deterministic group participant source builder for group daily
  reflection windows.
- Extend the reflection message projection to expose only the structural
  fields needed for attribution: `addressed_to_global_user_ids`, `broadcast`,
  `reply_context.reply_to_platform_user_id`, and
  `reply_context.reply_to_current_bot`. Do not project raw wire text, reply
  excerpts, reply display names, or full reply metadata for this feature.
- Include group user-authored rows only when the target user's row is
  structurally addressed to the active character.
- Include assistant rows only when persisted structural metadata maps the row
  to the same target user.
- Exclude adjacent users and unaddressed assistant broadcast rows from
  target-user evidence.
- Preserve the existing `interaction_style_images` user document shape and
  `upsert_user_style_image(...)` persistence path.
- Keep group-channel style generation separate and unchanged except for
  sharing daily-doc iteration where necessary.
- Add deterministic tests for eligibility, exclusion, source processing,
  duplicate suppression, and existing private/group-channel behavior.

## Deferred

- Do not generate full cognition-facing `user_image` or `user_memory_context`
  from group chat.
- Do not write group-derived facts, commitments, relationship insights, or
  profile memories.
- Do not backfill historical `interaction_style_images`.
- Do not persist new `delivery_mentions` or adapter-side hidden reply metadata
  in this plan. Do not consume new delivery-mention fields in this plan; use
  only the structural fields listed in `Must Do`.
- Do not parse display names from assistant dialog text to recover targets.
- Do not change RAG, dialog, L3 consumers, relevance consumers, adapters, or
  self-cognition delivery behavior.
- Do not add a live response-path LLM call.

## Cutover Policy

Overall strategy: compatible and additive.

| Area | Policy | Instruction |
|---|---|---|
| Existing private user style | compatible | Preserve current private daily behavior while routing it through the canonical source processor. |
| Group-channel style | compatible | Keep `group_channel_style` generation and runtime use unchanged. |
| Group-derived user style | additive | Add only strongly attributed group participant sources as an additional user-style source. |
| Storage | compatible | Reuse `interaction_style_images`; do not migrate or rewrite existing documents. |
| Runtime consumers | compatible | Keep existing `build_interaction_style_context` and engagement-context consumers unchanged. |

## Target State

```text
private daily reflection
  -> private user source builder
  -> canonical user-style source processor
  -> existing sanitizer
  -> upsert_user_style_image(...)

group daily reflection
  -> group-channel style generation, unchanged
  -> reflection DB helper reloads bounded group window
  -> deterministic group participant source builder
  -> canonical user-style source processor
  -> existing sanitizer
  -> upsert_user_style_image(...)
```

Normal chat continues to read user style through existing runtime projections.
No RAG path retrieves or projects style images.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Semantic path | Use one canonical user-style source processor for private and group-derived user style. | Prevents divergent rules for the same stored data slot. |
| Attribution owner | Deterministic code decides whether a group row belongs to one target user. | The LLM must not infer users from noisy group text. |
| Broadcast rows | Exclude assistant broadcast rows without persisted structural target metadata that maps to exactly one user. | Recent self-cognition rows are often semantically directed but structurally unaddressed. |
| Source threshold | Require strong group evidence before capture. | Strict recent-data audit still found enough candidates for high-confidence generation. |
| Storage | Reuse `interaction_style_images` user documents. | The target artifact is still `user_style_image`. |
| Backfill | Do not backfill historical rows. | Avoids surprise writes and keeps rollout bounded to future daily runs. |
| Group/doc independence | Process group-channel style and group participant user style as independent branches for the same daily doc. | An empty or rejected group-channel overlay must not suppress valid participant sources, and no participants must not suppress group-channel style. |

## Contracts And Data Shapes

Create one internal source module with these names:

```python
class UserStyleEvidenceRow(TypedDict):
    role: Literal["target_user", "character"]
    text: str


class UserStyleSourcePacket(TypedDict):
    source_id: str
    source_kind: Literal["private_daily", "group_participant_daily"]
    global_user_id: str
    channel_type: Literal["private", "group"]
    daily_confidence: Literal["medium", "high"]
    attribution_basis: str
    conversation_quality_patterns: list[str]
    synthesis_limitations: list[str]
    evidence_rows: list[UserStyleEvidenceRow]
    source_reflection_run_ids: list[str]
```

`src/kazusa_ai_chatbot/reflection_cycle/interaction_style_sources.py` must
export these deterministic helpers:

```python
def build_private_daily_user_style_source(
    *,
    daily_doc: CharacterReflectionRunDoc,
    global_user_id: str,
) -> UserStyleSourcePacket | None: ...


def build_group_participant_user_style_sources(
    *,
    daily_doc: CharacterReflectionRunDoc,
    messages: list[dict[str, Any]],
    character_global_user_id: str,
) -> list[UserStyleSourcePacket]: ...


def user_style_source_to_extractor_payload(
    *,
    source: UserStyleSourcePacket,
    current_overlay: dict,
) -> dict: ...
```

`evidence_rows` are prompt-facing, bounded, and sanitized. They must not
include platform ids, message ids, global user ids, display names, reply
metadata, raw wire text, source refs, or adjacent-user rows. The deterministic
builder's internal attribution field allowlist is `platform_user_id`,
`global_user_id`, `addressed_to_global_user_ids`, `broadcast`,
`reply_context.reply_to_platform_user_id`, and legacy
`reply_context.reply_to_current_bot`. These fields must not enter the
extractor payload.

`user_style_source_to_extractor_payload(...)` must return exactly this
LLM-facing shape:

```python
{
    "channel_type": "private|group",
    "daily_confidence": "medium|high",
    "conversation_quality_patterns": list[str],
    "synthesis_limitations": list[str],
    "evidence_rows": [
        {"role": "target_user|character", "text": "prompt-safe content"}
    ],
    "current_overlay": InteractionStyleOverlayDoc,
}
```

Do not include `source_id`, `source_kind`, `global_user_id`,
`attribution_basis`, `source_reflection_run_ids`, platform ids, addressee
fields, reply metadata, row counts, threshold values, or eligibility reasons
in this payload.

Group source eligibility:

- Candidate user must have non-empty `global_user_id` and must not be the
  active character.
- For each group daily doc, build `character_platform_user_ids` from assistant
  rows whose `global_user_id` is the active character, and build each target's
  platform-id set from target-user rows in the same window.
- User-authored eligible rows are rows from the target user where
  `addressed_to_global_user_ids` contains the active character, or
  `reply_context.reply_to_current_bot == true` for legacy rows, or
  `reply_context.reply_to_platform_user_id` matches one of the active
  character platform ids from the same window.
- Assistant eligible rows are assistant rows where exactly one structural
  target maps to the target user. Supported proof is either the normalized
  `addressed_to_global_user_ids` set equals `{target_global_user_id}` or
  `reply_context.reply_to_platform_user_id` matches one of the target user's
  platform ids from the same window.
- Assistant rows with `broadcast == true` follow the same exact-target rule.
  Broadcast assistant rows without exact structural target proof are excluded.
- Assistant rows structurally addressed to multiple users are excluded from
  every single-user style packet.
- Do not use body text, display names, nicknames, quote text, or pronouns as
  target proof.
- Duplicate suppression means: emit each target user at most once per group
  daily doc, dedupe repeated source rows inside one source packet, and skip a
  user-style write when the current stored style doc already lists the daily
  run id. Do not add a historical source-run ledger or new DB shape.

Minimum group source strength:

- At least 3 eligible target-user rows.
- At least 3 eligible assistant-to-target rows.
- At least 8 total eligible rows.
- Cap per group daily reflection document to the top 5 users by eligible row
  count, then assistant-to-target count, then target-user row count.

LLM source payload caps:

- At most 24 evidence rows per source packet.
- At most 160 characters per evidence row after whitespace compaction.
- Prefer balanced recent target-user and assistant-to-target rows.
- Empty or rejected extractor output must not overwrite an existing overlay.

Extractor payload contract:

- Add `evidence_rows` to the existing interaction-style extractor payload.
- Private sources must provide an empty `evidence_rows` list and keep using the
  existing daily abstract quality fields.
- Group participant sources must provide only `target_user` and `character`
  evidence rows for the target user.
- The extractor prompt must state that evidence rows are style signals only
  and that output must remain abstract handling guidance, never event memory.
- The extractor prompt must not ask the LLM to validate attribution,
  thresholds, duplicate status, source lineage, or persistence eligibility.

## LLM Call And Context Budget

Current path: one background interaction-style LLM call per eligible private
daily doc and one background group-channel style LLM call per eligible group
daily doc.

Target path: keep the existing calls and add at most 5 background user-style
LLM calls per eligible group daily doc.

This plan does not add live response-path LLM calls.

Context budget for each user-style source call:

- Use the existing consolidation route.
- Keep total dynamic payload below 8,000 characters.
- Drop oldest or lowest-priority evidence rows before exceeding the cap.
- When no source remains after caps and thresholds, skip without write.
- Do not add retry loops, repair prompts, or second-pass LLM calls for this
  feature.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/db/conversation_reflection.py`
  - Extend `_REFLECTION_MESSAGE_PROJECTION` only with
    `addressed_to_global_user_ids`, `broadcast`, and
    `reply_context.reply_to_platform_user_id`, and
    `reply_context.reply_to_current_bot`.
  - Do not project `raw_wire_text`, `reply_context.reply_excerpt`,
    `reply_context.reply_to_display_name`, or full reply metadata.
- `src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py`
  - Add or call the canonical user-style source processor.
  - Route private daily user-style generation through the canonical processor.
  - For group daily docs, load the bounded conversation window through
    `list_reflection_scope_messages(...)`, build participant sources, and
    process them independently from group-channel style.
  - Keep the existing interaction-style LLM as the only extractor and update
    its payload contract to accept `evidence_rows`.
- `tests/test_reflection_interaction_style.py`
  - Add focused tests for source building, eligibility, exclusion, and shared
    processing.
- `tests/test_reflection_cycle_readonly.py`
  - Update the projection allowlist test to require structural attribution
    fields and forbid raw reply/display/wire fields.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Document that group-derived user-style sources are background-only,
    structurally attributed, and separate from group-channel style.

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/interaction_style_sources.py`
  - Own deterministic source packet construction and eligibility logic.

### Keep

- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - Keep storage shape and runtime projections unchanged. Stop execution and
    update this plan before making any change to this file.
- `src/kazusa_ai_chatbot/db/schemas.py`
  - Do not change schema definitions in this plan.
- Runtime cognition, relevance, RAG, dialog, adapter, and self-cognition
  modules.
  - Do not change consumers or delivery behavior in this plan.

## Overdesign Guardrail

- Actual problem: real QQ users lack `user_style_image` because current user
  style generation only uses private daily reflections.
- Minimal change: add strongly attributed group participant sources to the
  existing daily interaction-style update path.
- Ownership boundaries: deterministic code proves attribution and caps source
  evidence; the LLM abstracts style guidance; existing validators sanitize
  output; existing DB helpers persist one current user-style document.
- Rejected complexity: no full user memory generation, no RAG exposure, no
  display-name target inference, no self-cognition direct writes, no backfill,
  no runtime consumer changes, and no new storage collection.
- Persisted delivery-mention or native-reply target support is out of scope.
  A later plan is required before adding those inputs.

## Agent Autonomy Boundaries

- The responsible agent must keep deterministic group source construction in
  `interaction_style_sources.py` and semantic overlay extraction in
  `interaction_style.py`.
- The responsible agent must not introduce alternate generation rules for
  private and group user-style overlays.
- The responsible agent must not implement semantic keyword filters, display
  name parsing, or post-LLM correction of style decisions.
- A changed reflection phase scheduler worker contract blocks execution of
  this plan until this plan is updated.
- The responsible agent must not perform unrelated cleanup, broad prompt
  rewrites, dependency changes, or formatting churn.

## Implementation Order

1. Parent updates `tests/test_reflection_cycle_readonly.py::
   test_db_interface_uses_message_field_allowlist` to require
   `addressed_to_global_user_ids`, `broadcast`, and
   `reply_context.reply_to_platform_user_id`, and
   `reply_context.reply_to_current_bot`, while still forbidding raw wire, reply
   excerpt, and reply display-name fields.
2. Parent adds focused source-builder tests in
   `tests/test_reflection_interaction_style.py`:
   `test_group_user_style_sources_use_structural_targets_only`,
   `test_group_user_style_sources_exclude_adjacent_user_and_broadcast_noise`,
   `test_group_user_style_sources_accept_hidden_reply_target`,
   `test_group_user_style_source_thresholds_and_top_five_cap`, and
   `test_group_user_style_source_payload_hides_operational_metadata`.
3. Parent adds integration tests in `tests/test_reflection_interaction_style.py`:
   `test_run_daily_interaction_style_update_writes_group_participant_user_style`
   and
   `test_group_participant_style_skips_already_applied_daily_run`.
4. Parent runs the focused tests and records the expected failures before
   production implementation.
5. Production-code subagent extends the DB reflection projection exactly as
   specified in `Change Surface`.
6. Production-code subagent creates `interaction_style_sources.py` with the
   exact source packet types and helper names from `Contracts And Data Shapes`.
7. Production-code subagent refactors private user-style extraction to build a
   private source packet and call the canonical source-to-overlay processor.
8. Production-code subagent integrates group participant processing in
   `_process_group_daily_doc` without changing group-channel style behavior.
9. Parent reruns focused tests, updates only approved-scope issues, and records
   evidence.
10. Parent updates `reflection_cycle/README.md`.
11. Parent runs all verification commands and static boundary checks.
12. Parent runs Independent Code Review, remediates approved-scope findings,
    reruns affected verification, and records final evidence.

## Execution Model

- Parent owns test contract, orchestration, verification, evidence, review
  remediation, lifecycle updates, and final sign-off.
- Parent establishes focused tests before production implementation starts.
- Use exactly one production-code subagent for production edits and one
  independent code-review subagent after verification.
- Native subagents are required. Execution stops when the harness lacks native
  subagent support. This plan has no single-agent execution path.

## Progress Checklist

- [x] Stage 1 - coordination and test contract established.
  - Covers: steps 1-4.
  - Verify: focused tests are added and baseline failures or current behavior
    are recorded in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-06-04` after evidence is recorded.
- [x] Stage 2 - deterministic source and DB projection implemented.
  - Covers: steps 5-6.
  - Verify: source-builder tests and projection allowlist test pass.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex / 2026-06-04` after evidence is recorded.
- [x] Stage 3 - canonical processor integration complete.
  - Covers: steps 7-9.
  - Verify: interaction-style focused tests pass, including private-path
    regression and group participant integration.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex / 2026-06-04` after evidence is recorded.
- [x] Stage 4 - docs, full verification, review, and lifecycle complete.
  - Covers: steps 10-12.
  - Verify: all `Verification` commands pass, Independent Code Review
    approves, and any review fixes are verified.
  - Handoff: move the plan to completed after evidence is recorded.
  - Sign-off: `Codex / 2026-06-04` after final evidence is recorded.

## Verification

Run from the repository root with the project venv:

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_interaction_style.py -q
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_readonly.py -q
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py -q
venv\Scripts\python.exe -m pytest tests\test_rag_projection.py -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\conversation_reflection.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style_sources.py
```

Static boundary checks:

```powershell
rg -n "user_style_image|interaction_style_context|interaction_style_images" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py
rg -n "get_db|conversation_history" src\kazusa_ai_chatbot\reflection_cycle\interaction_style*.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style_sources.py
rg -n "raw_wire_text|reply_excerpt|reply_to_display_name|reply_to_message_id" src\kazusa_ai_chatbot\reflection_cycle\interaction_style*.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style_sources.py
rg -n "display_name|body_text|platform_user_id|addressed_to_global_user_ids|reply_context" src\kazusa_ai_chatbot\reflection_cycle\interaction_style*.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style_sources.py
```

The first command must show no production RAG exposure. The second and third
commands must show no direct DB access and no raw reply/wire leakage in the
reflection-cycle style modules. The fourth command must be reviewed to confirm
that occurrences are limited to deterministic attribution fields and
prompt-safe content projection, with no target attribution from visible text or
display names and no ids in extractor payloads.

Do not run live LLM calls, live DB inspection, or live DB writes in this plan.

## Independent Plan Review

Review completed on 2026-06-03 during drafting. Findings addressed:

- The original change surface failed to build group participant sources because
  `list_reflection_scope_messages(...)` did not project structural target
  fields. Resolution: add the narrow DB projection change and projection test.
- The original source contract left the new module interface open. Resolution:
  fix exact `TypedDict` and helper names in `Contracts And Data Shapes`.
- The original implementation order and checklist were too coarse for
  execution handoff. Resolution: name focused tests, expected baseline
  failures, staged verification, and sign-off gates.
- The registry listed the active scheduler plan as `draft` while the plan file
  is `in_progress`. Resolution: update the registry row without editing the
  active scheduler plan.

Before changing this plan from `draft` to `approved`, rerun this review against
the current code and active scheduler plan state. Record new findings in
`Execution Evidence` before execution starts.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The reviewer must check structural attribution, shared private/group
processing, forbidden runtime/RAG/memory changes, eligibility tests, cap tests,
duplicate suppression, empty output, sanitizer rejection, and verification
evidence. Record findings, fixes, commands rerun, residual risks, and approval
status in Execution Evidence.

## Acceptance Criteria

This plan is complete when:

- Private daily user-style generation still writes through
  `upsert_user_style_image(...)`.
- Group daily reflection can produce user-style sources only for users with
  strong structural attribution.
- Group participant source construction uses the reflection DB helper with the
  approved projection fields, not raw MongoDB access.
- Assistant broadcast rows without structural targets are excluded from
  user-style evidence.
- Hidden structural reply targets count only when they map deterministically
  to exactly one target user in the same group window.
- The extractor payload contains only the allowlisted semantic fields and
  prompt-safe role/text evidence rows; operational source metadata is absent.
- Group-derived user style uses the same extractor, sanitizer, empty-output
  skip, source-lineage skip, and persistence path as private user style.
- `group_channel_style` behavior remains separate and unchanged.
- Existing runtime style consumers keep their current payload shapes.
- RAG and `user_memory_units` do not receive style-image data.
- All Verification commands pass and Independent Code Review approves.

## Risks

- Adjacent-user contamination: structural attribution only and negative broadcast-row tests.
- Extra background LLM cost: evidence thresholds and top-5 users per group daily doc cap.
- Missing projection fields: explicit DB-helper projection change and allowlist test.
- Divergent semantics or event leakage: one canonical processor, prompt-safe row projection, sanitizer tests, and boundary grep.
- Interference with active scheduler work: draft status and coordination gate.

## Execution Evidence

- 2026-06-04: User approved the plan and requested implementation with
  subagents. Registry no longer lists
  `reflection_phase_scheduled_group_review_plan.md`; the referenced active
  plan file is absent, so the coordination gate is clear.
- 2026-06-04: Stage 1 test contract established. Added focused deterministic
  tests for the reflection DB projection allowlist, private extractor
  `evidence_rows=[]` payload, group source eligibility/exclusion/hidden reply
  attribution/top-five cap/minimal payload, group participant writes, and
  already-applied user-source skip. Baseline command:
  `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_readonly.py::test_db_interface_uses_message_field_allowlist tests\test_reflection_interaction_style.py::test_extract_user_style_overlay_uses_only_safe_daily_signal_payload tests\test_reflection_interaction_style.py::test_group_user_style_sources_use_structural_targets_only tests\test_reflection_interaction_style.py::test_group_user_style_sources_exclude_adjacent_user_and_broadcast_noise tests\test_reflection_interaction_style.py::test_group_user_style_sources_accept_hidden_reply_target tests\test_reflection_interaction_style.py::test_group_user_style_source_thresholds_and_top_five_cap tests\test_reflection_interaction_style.py::test_group_user_style_source_payload_hides_operational_metadata tests\test_reflection_interaction_style.py::test_run_daily_interaction_style_update_writes_group_participant_user_style tests\test_reflection_interaction_style.py::test_group_participant_style_skips_already_applied_daily_run -q`.
  Result: expected failure, 9 failed. Failures prove the missing projection
  fields, missing `evidence_rows` private payload, missing
  `interaction_style_sources` module, and missing group participant update path.
- 2026-06-04: Production-code subagent `Kierkegaard`
  (`019e8dc3-669e-7e01-89da-37c78a6cd723`) started for the approved Stage 2
  and Stage 3 production write scope only.
- 2026-06-04: Production-code subagent completed the approved production
  scope in `conversation_reflection.py`, `interaction_style_sources.py`, and
  `interaction_style.py`. Parent review found one approved-scope attribution
  gap: non-character `role="assistant"` rows could be counted as character
  evidence. Added
  `test_group_user_style_sources_exclude_other_assistant_rows` and fixed the
  builder to require assistant evidence rows to match the active character
  `global_user_id`.
- 2026-06-04: Parent Stage 2/3 verification passed after the fix. Focused
  command including the added non-character assistant negative case: 10 passed.
  `venv\Scripts\python.exe -m pytest tests\test_reflection_interaction_style.py -q`:
  18 passed. `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_readonly.py -q`:
  9 passed. `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\db\conversation_reflection.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style_sources.py`:
  passed.
- 2026-06-04: Stage 4 README update completed. Documented that
  group-derived user-style sources are background-only, structurally
  attributed, separate from `group_channel_style`, and forbidden from feeding
  ids, reply metadata, raw wire text, adjacent-user rows, RAG, memory, dialog,
  or self-cognition state.
- 2026-06-04: Full Verification commands passed:
  `tests\test_reflection_interaction_style.py` 18 passed,
  `tests\test_reflection_cycle_readonly.py` 9 passed,
  `tests\test_interaction_style_images.py` 20 passed,
  `tests\test_rag_projection.py` 32 passed, and `py_compile` passed for the
  three changed Python production files. Static boundary checks: no matches
  for RAG style exposure, no matches for direct DB access in style modules, no
  matches for raw reply/wire fields, and structural-field matches are limited
  to deterministic attribution plus compact prompt-safe evidence text. The
  PowerShell wildcard form in the plan was rerun with explicit file paths
  after `rg` treated `interaction_style*.py` as a literal Windows path.
- 2026-06-04: Independent Code Review subagent `Pasteur`
  (`019e8dd1-38ad-7632-b3e8-79cfe5f7831c`) returned
  `CHANGES_REQUESTED`. Finding: unexpected exceptions in the group-channel
  and group-participant branches still shared one failure domain, so a
  group-channel branch failure could suppress participant writes, and a
  participant failure after a group-channel write could convert an approved
  write into a failed daily doc.
- 2026-06-04: Review remediation completed. Added deterministic coverage for
  group-channel unexpected failure not suppressing participant user-style
  writes and participant unexpected failure not erasing group-channel success.
  Updated `_process_group_daily_doc` so both branches run independently, branch
  exceptions are logged, any approved branch write counts the daily doc as
  succeeded, and branch errors are re-raised only when no branch writes.
  Review-regression command: 2 passed. Full rerun:
  `tests\test_reflection_interaction_style.py` 20 passed,
  `tests\test_reflection_cycle_readonly.py` 9 passed,
  `tests\test_interaction_style_images.py` 20 passed,
  `tests\test_rag_projection.py` 32 passed, and `py_compile` passed for the
  three changed Python production files. Static boundary checks remained clean
  with the same interpretation as above.
- 2026-06-04: Independent Code Review re-review returned
  `CHANGES_REQUESTED`. Finding: multiple participant user sources still shared
  one failure domain, so a later participant-source exception could erase an
  earlier approved participant write when the group-channel branch did not
  write.
- 2026-06-04: Second review remediation completed. Added
  `test_later_group_participant_failure_preserves_prior_user_success`.
  Updated `_process_group_participant_user_styles` so each user source is
  isolated, source exceptions are logged without user/source identifiers,
  source processing continues, any prior participant write returns success,
  and source errors are re-raised only when no participant source writes.
  Review-regression command: 3 passed. Full rerun:
  `tests\test_reflection_interaction_style.py` 21 passed,
  `tests\test_reflection_cycle_readonly.py` 9 passed,
  `tests\test_interaction_style_images.py` 20 passed,
  `tests\test_rag_projection.py` 32 passed, and `py_compile` passed for the
  three changed Python production files. Static boundary checks remained clean
  with the same interpretation as above.
- 2026-06-04: Independent Code Review subagent `Pasteur` re-reviewed the
  second remediation and returned `APPROVED`. Reviewer ran the focused
  three-test partial-success regression command: 3 passed. No blocking
  findings remained. Plan status changed to `completed` for archive move.
