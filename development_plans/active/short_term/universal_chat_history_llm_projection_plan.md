# universal chat history llm projection plan

## Summary

- Goal: Route every model-facing conversation-history payload through one
  central projection helper so full, narrow, filtered, retrieved, and
  background conversation windows use the same logging-style transcript format.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang for model-facing conversation-history
  projection; internal storage/state contracts remain structured.
- Highest-risk areas: prompt regressions in relevance, decontextualization,
  RAG conversation evidence, memory extraction, reflection, and cognition
  style/social stages; accidental loss of deterministic identity fields before
  code needs them; privacy drift in reflection; stale tests expecting dict rows.
- Acceptance criteria: every LLM payload that presents conversation-history
  rows uses the same central helper, the helper emits logging-style strings,
  no model-facing conversation-history prompt keeps local bespoke row dicts,
  focused and integration tests pass, and the observed `蚝爹油` decontextualizer
  live case is rerun with an inspectable trace.

## Context

The immediate failure was a decontextualizer run for QQ group `227608960`.
The raw input contained `蚝爹油` as both a recent chat speaker and a
`scope_users` entry, but the model treated the phrase as an ungrounded term
instead of a visible participant. The current decontextualizer sends
`chat_history` as JSON row dictionaries with `role`, platform ids, global ids,
timestamps, display names, message text, and delivery metadata. The display
name is present, but it is not salient to weaker local models.

The user decision is broader than the decontextualizer: whenever conversation
history is presented to an LLM, the presentation must be consistent regardless
of whether the window is full, narrow, filtered, current-user scoped, retrieved
by RAG, or selected for background/reflection work. The temporary presentation
format is logging style:

```text
[2026-06-13 15:04:19] 蚝爹油: 捡垃圾不是乐趣么
[2026-06-13 15:14:43] 1816 reply_to 杏山千纱: @杏山千纱 那蚝爹油跟你啥关系
```

There is no `broadcast` label in the line format. In a group channel, ordinary
visible chat lines are implicitly visible to the channel. Explicit reply or
addressing metadata may be rendered only when it exists and materially helps
the model understand adjacency.

Current known model-facing history projection families:

- Live decontextualizer: `persona_supervisor2_msg_decontexualizer.py`
  receives `chat_history` from `persona_supervisor2.py`.
- Relevance: `persona_relevance_agent.py` sends `conversation_history`.
- RAG runtime context: `rag/prompt_projection.py` sends `chat_history_recent`
  and `chat_history_wide` into initializer and dispatcher payloads.
- RAG conversation evidence worker judges: conversation search/filter/keyword
  workers pass retrieved conversation rows to LLM judge stages.
- RAG conversation evidence public projection:
  `rag/conversation_evidence/projection.py` builds row summaries and packets
  from retrieved conversation rows.
- Cognition chain social/style/visual stages:
  `cognition_chain_core/stages/l2c2.py` and `stages/l3.py` send small
  `chat_history` windows.
- Conversation progress recorder:
  `conversation_progress/recorder.py` sends `chat_history_recent`.
- Consolidation memory-unit extraction:
  `consolidation/memory_units.py` and
  `memory_writer_prompt_projection.py` send `chat_history_recent`.
- Reflection and group-scene digest:
  `reflection_cycle/projection.py` and `reflection_cycle/group_scene_digest.py`
  project conversation rows into LLM payloads.
- Self-cognition routes reuse the normal persona/cognition path and inherit
  the same projection once live persona consumers are migrated.

Internal storage rows, service graph state, RAG worker raw result contracts,
cache dependencies, adapter envelopes, and persistence remain structured. The
new rule governs only model-facing presentation of conversation-history rows.

## Mandatory Skills

- `development-plan`: load before executing, reviewing, approving, or signing
  off this plan.
- `local-llm-architecture`: load before changing prompt-facing payloads or LLM
  stage contracts.
- `py-style`: load before editing Python source.
- `cjk-safety`: load before editing Python files containing Chinese strings or
  prompt/test examples.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running and reviewing live LLM tests or prompt
  quality comparisons.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Do not change database storage shapes, adapter envelope shapes, queue state,
  RAG cache dependency keys, or deterministic identity matching data as part of
  this plan.
- Do not add compatibility shims, alternate projection modes, feature flags, or
  fallback-to-old-history payloads.
- Do not make LLM stages consume raw platform ids, global ids, conversation row
  ids, or message ids merely to understand transcript order or speaker labels.
  Deterministic code may keep those fields internally.
- The central helper is the only model-facing formatter for conversation
  history. Local call sites may slice/filter/select rows before the helper, but
  they must not format their own transcript lines.
- Live LLM tests must run one case at a time and must produce inspectable raw
  trace plus agent-authored review artifacts.

## Must Do

- Create one central conversation-history LLM projection helper.
- Migrate every model-facing conversation-history payload to the central
  helper.
- Keep structured row data available to deterministic code before projection.
- Use logging-style transcript strings as the prompt-facing representation.
- Render explicit `reply_to` information when a row has visible reply metadata.
- Omit broadcast labels from the universal history line format.
- Update prompt text that currently tells models to read `display_name`,
  `speaker_name`, `speaker_ref`, or structured chat-history row fields.
- Add focused tests for the helper and integration tests for all migrated LLM
  payload families.
- Update live LLM trace tests so the observed `蚝爹油` case proves the
  decontextualizer receives flattened history lines.
- Run the verification commands in this plan and record evidence.

## Deferred

- Do not redesign decontextualizer referent output schema.
- Do not redesign `scope_users`, `prompt_message_context`, `reply_context`, or
  any other non-history input field.
- Do not change RAG retrieval ranking, query generation, cache policies, or
  storage filters.
- Do not change reflection persistence policy or durable memory write policy.
- Do not add new LLM calls, retries, repair prompts, or model routes.
- Do not convert non-conversation evidence, user profiles, memory rows,
  relationship rows, or web evidence to logging-style strings.
- Do not remove deterministic ids from internal rows where validation,
  active-turn exclusion, cache invalidation, or source references need them.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Model-facing conversation-history projection | bigbang | Replace bespoke dict/list transcript projections with the central helper in one implementation. No local fallback to old row dictionaries. |
| Internal graph/service state | compatible | Keep existing structured `chat_history_wide` and `chat_history_recent` state shapes so deterministic code keeps identity and delivery metadata. |
| RAG worker raw results | compatible | Workers may keep structured rows internally. Only LLM judge/finalizer/context payloads must receive helper-rendered transcript strings. |
| Prompt text | bigbang | Rewrite prompt field descriptions to describe transcript lines instead of row dict fields. |
| Tests | bigbang | Update prompt payload assertions and fixture baselines to the new transcript shape. Do not preserve old test expectations. |
| Database | compatible | No data migration and no collection changes. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy prompt-facing references
  instead of preserving them.
- If an area is `compatible`, preserve only the surfaces explicitly listed in
  this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

All LLM calls that include conversation-history material receive a list of
plain transcript strings produced by one public helper. A full window, narrow
window, current-user interaction window, RAG-filtered result, reflection scope,
or group-scene digest input all use the same line grammar:

```text
[<local timestamp>] <speaker display name>: <visible message text>
[<local timestamp>] <speaker display name> reply_to <reply display name>: <visible message text>
<speaker display name>: <visible message text>
<speaker display name> reply_to <reply display name>: <visible message text>
```

Timestamp is included when available after local-time formatting. If timestamp
is absent or invalid, the line starts with the speaker label rather than an
invented time. The helper preserves message text language and literal
mentions. Attachment descriptions already projected into `body_text` remain in
the line; if a caller passes raw prompt-safe attachments, the helper appends
bounded image blocks through the existing image-text projection behavior.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Central ownership | Create `src/kazusa_ai_chatbot/conversation_history_prompt_projection.py` | A dedicated module avoids overloading time utilities and makes grep-based enforcement straightforward. |
| Public entrypoint | `project_conversation_history_for_llm(rows, *, character_name='', max_rows=None) -> list[str]` | One helper controls the line format for all model-facing history. |
| Line helper visibility | Keep per-row rendering private unless tests require direct access | Callers should depend on one list-level projection function. |
| Input row shape | Accept mapping rows with common fields: `timestamp`, `display_name`, `role`, `body_text`, `content`, `text`, `reply_context`, and `attachments` | Existing callers have slightly different internal row shapes; the helper normalizes only prompt-facing text. |
| Assistant fallback speaker | Use `character_name` for assistant rows missing `display_name`; otherwise use `display_name` | This preserves the visible active-character name without hard-coding Kazusa. |
| Missing speaker | Use `unknown` only when neither `display_name` nor assistant fallback exists | Avoids dropping lines while making incomplete data visible in tests. |
| Reply rendering | Render `reply_to` when `reply_context.reply_to_display_name` or top-level `reply_to_display_name` is present | Reply adjacency is useful context; raw reply ids are not. |
| Broadcast rendering | Do not render broadcast state | Visible group messages are implicitly visible; adding a label adds noise. |
| Internal ids | Strip from prompt-facing transcript lines | Raw ids help deterministic code, not LLM semantic interpretation. |
| Reflection | Use the same helper for conversation rows sent to reflection LLMs | The user requirement prioritizes universal presentation over local bespoke deidentified row shape. |

## Contracts And Data Shapes

Public helper contract:

```python
def project_conversation_history_for_llm(
    rows: Iterable[Mapping[str, Any]],
    *,
    character_name: str = "",
    max_rows: int | None = None,
) -> list[str]:
    """Return model-facing logging-style conversation transcript lines."""
```

Input rules:

- `rows` are already selected, filtered, and ordered by the caller.
- If `max_rows` is provided, the helper keeps the last `max_rows` rows while
  preserving chronological order.
- `timestamp` may be storage UTC, local text, or absent. The helper uses
  existing local-time formatting when storage UTC is detected and otherwise
  preserves non-empty timestamp text.
- Message text comes from `body_text`, then `content`, then `text`.
- Reply target comes from `row["reply_context"]["reply_to_display_name"]`
  first, then top-level `reply_to_display_name`.
- Attachments may contribute bounded image blocks only through existing
  prompt-safe text projection; raw image bytes never enter transcript lines.

Output examples:

```python
[
    '[2026-06-13 15:04:19] 蚝爹油: 捡垃圾不是乐趣么',
    '[2026-06-13 15:14:43] 1816 reply_to 杏山千纱: @杏山千纱 那蚝爹油跟你啥关系',
]
```

Forbidden model-facing shapes after cutover:

```python
[
    {'display_name': '蚝爹油', 'body_text': '捡垃圾不是乐趣么'},
]
```

```python
[
    {'speaker_name': '蚝爹油', 'body_text': '捡垃圾不是乐趣么'},
]
```

```python
[
    {'speaker_ref': 'participant_1', 'text': '捡垃圾不是乐趣么'},
]
```

## LLM Call And Context Budget

No new LLM calls are added. The affected calls keep their existing call counts
and blocking behavior.

| LLM call family | Before | After | Context impact |
|---|---|---|---|
| Decontextualizer | One response-path call with structured `chat_history` rows | One response-path call with transcript lines | Lower token and schema-noise load; same row limit. |
| Relevance | One response-path call with `conversation_history` rows | One response-path call with transcript lines | Lower metadata load; same wide-history limit. |
| RAG initializer/dispatcher | Existing response-path calls with runtime context history rows | Same calls with transcript lines | Lower metadata load; no extra retrieval. |
| RAG conversation worker judges | Existing helper calls with tool result rows | Same calls with transcript-line result rows where conversation rows are shown to LLM | Lower row metadata load; deterministic raw rows retained outside prompt. |
| Cognition social/style/visual | Existing response-path calls with tiny history row windows | Same calls with transcript lines | Similar or lower context use. |
| Conversation progress and consolidation | Existing background/post-turn calls | Same calls with transcript lines | Lower metadata load; memory prompts must rely on transcript speaker labels. |
| Reflection/group digest | Existing background calls | Same calls with transcript lines | Lower metadata load; display names become visible transcript labels. |

Use `50k tokens` as the default cap. Verification should compare character
length of representative before/after payloads for decontextualizer, relevance,
RAG context, memory extraction, and reflection. After projection, each payload
must be no larger than before unless attachment text already existed in the
old prompt-safe row.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/conversation_history_prompt_projection.py`
  - Owns the universal model-facing conversation-history formatter.
- `tests/test_conversation_history_prompt_projection.py`
  - Focused helper tests for timestamps, speakers, replies, attachments,
    slicing, missing fields, and no id leakage.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Project `input_msg["chat_history"]` with the central helper and update
    prompt text to describe transcript lines.
- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`
  - Project `human_data["conversation_history"]` with the central helper and
    update relevance prompt wording if it references row fields.
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
  - Project `chat_history_recent` and `chat_history_wide` runtime-context
    fields through the central helper.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py`
  - Keep tool outputs structured for deterministic consumers, but add or use
    a prompt-facing projection point before any LLM consumes conversation
    result rows.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/projection.py`
  - Use central transcript lines for conversation evidence summaries,
    projection payload rows, and packets where message rows are surfaced to
    later LLM stages.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/filter.py`
  - Project `get_conversation` judge input result rows through the central
    helper.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/keyword.py`
  - Project keyword-search judge input result rows through the central helper.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/search.py`
  - Project semantic/hybrid-search judge input result rows through the central
    helper.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2c2.py`
  - Replace `_surface_history_for_social_context` row projection with central
    transcript projection.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Replace `_surface_history_for_style` and `_surface_history_for_visual`
    row projections with central transcript projection.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Replace `_project_recorder_chat_history` with central transcript
    projection and update prompt wording from `speaker_name/body_text` to
    transcript lines.
- `src/kazusa_ai_chatbot/consolidation/memory_units.py`
  - Project `chat_history_recent` through the central helper before memory
    extractor prompts.
- `src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`
  - Remove local speaker-row projection and route `chat_history_recent`
    through the central helper.
- `src/kazusa_ai_chatbot/reflection_cycle/projection.py`
  - Replace `_project_messages_for_prompt` message dict rows with central
    transcript lines under `conversation.messages`.
- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
  - Replace `message_rows` dict projection with central transcript lines and
    update prompt wording to read transcript lines.
- Relevant tests and fixtures under `tests/`
  - Update prompt payload shape assertions and fixture baselines that currently
    expect chat-history dict rows.
- `src/kazusa_ai_chatbot/rag/README.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`, and narrow subsystem READMEs if
  they document conversation-history prompt shape.

### Keep

- `src/kazusa_ai_chatbot/utils.py::trim_history_dict`
  - Keeps structured service-state history rows for deterministic consumers.
- `src/kazusa_ai_chatbot/time_boundary.py::format_storage_utc_history_for_llm`
  - Remains a timestamp row copier, not the universal transcript formatter.
- `src/kazusa_ai_chatbot/service.py`
  - Keeps storing and passing structured `chat_history_wide` and
    `chat_history_recent` through graph state.
- Database collections and indexes
  - No migration.

## Overdesign Guardrail

- Actual problem: LLM-facing conversation history is presented in several
  bespoke shapes, making weak local models miss obvious speaker context and
  making prompt behavior hard to tune centrally.
- Minimal change: keep internal structured rows, but route every
  model-facing conversation-history projection through one helper that emits
  logging-style transcript strings.
- Ownership boundaries: deterministic code owns row selection, filtering,
  ordering, identity matching, cache invalidation, active-turn exclusion, and
  source ids; the LLM receives only transcript evidence for semantic reading.
- Rejected complexity: no projection modes, no feature flag, no compatibility
  fallback to old dict rows, no new LLM calls, no schema aliases, no adapter
  changes, no storage migration, no prompt-specific local formatters.
- Evidence threshold: add a new projection mode or exception only after an
  observed failing prompt trace proves that the universal transcript format
  cannot express a required conversation-history fact and deterministic code
  cannot supply that fact outside the transcript.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the listed change surface
  as high-scrutiny changes and justify them in `Execution Evidence`.
- The responsible agent must search the codebase before adding local helpers.
  If equivalent conversation-history formatting exists, replace it with the
  central helper instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, preserve the plan's stated intent and record
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent establishes the focused helper test contract in
   `tests/test_conversation_history_prompt_projection.py`.
2. Parent runs the focused helper tests and records the expected missing-module
   failure.
3. Parent starts one production-code subagent for production implementation.
4. Production-code subagent creates the central helper module and makes the
   focused helper tests pass.
5. Parent adds or updates integration tests for each prompt family while the
   production-code subagent migrates production call sites.
6. Production-code subagent migrates live response-path prompt call sites:
   decontextualizer, relevance, RAG runtime context, cognition social/style,
   and visual history.
7. Production-code subagent migrates RAG conversation-evidence worker judge
   inputs and public projection payloads.
8. Production-code subagent migrates post-turn/background prompt call sites:
   conversation progress, consolidation memory extraction, reflection, and
   group-scene digest.
9. Parent updates fixtures, prompt-contract tests, and live LLM trace tests.
10. Parent runs focused tests, integration tests, static greps, py compile, and
    live LLM tests listed in `Verification`.
11. Parent starts one independent code-review subagent after planned
    verification passes.
12. Parent remediates review findings inside the approved change surface,
    reruns affected verification, records evidence, and signs off.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused helper contract established
  - Covers: implementation order steps 1-2.
  - Files: `tests/test_conversation_history_prompt_projection.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_conversation_history_prompt_projection.py -q`.
  - Expected before implementation: fails because
    `kazusa_ai_chatbot.conversation_history_prompt_projection` does not exist.
  - Evidence: record command and failure in `Execution Evidence`.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 2 - central helper implemented
  - Covers: implementation order steps 3-4.
  - Files: `src/kazusa_ai_chatbot/conversation_history_prompt_projection.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_conversation_history_prompt_projection.py -q`.
  - Evidence: focused helper tests pass.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 3 - response-path prompt consumers migrated
  - Covers: implementation order step 6.
  - Files: decontextualizer, relevance, RAG runtime context, L2c2, and L3
    stage files listed in `Change Surface`.
  - Verify: run the response-path tests listed in `Verification`.
  - Evidence: record changed files and test output.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 4 - RAG conversation evidence migrated
  - Covers: implementation order step 7.
  - Files: RAG conversation evidence projection and workers listed in
    `Change Surface`.
  - Verify: run the RAG tests listed in `Verification`.
  - Evidence: record changed files and test output.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 5 - background/reflection prompt consumers migrated
  - Covers: implementation order step 8.
  - Files: conversation progress, consolidation, memory writer, reflection,
    and group-scene digest files listed in `Change Surface`.
  - Verify: run the background/reflection tests listed in `Verification`.
  - Evidence: record changed files and test output.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 6 - live LLM validation completed
  - Covers: implementation order step 10.
  - Files: live LLM test traces under `test_artifacts/llm_traces/`.
  - Verify: run each live LLM test one case at a time and inspect raw output.
  - Evidence: raw trace paths plus agent-authored review artifact paths.
  - Sign-off: parent/date after evidence is recorded.
- [ ] Stage 7 - independent code review completed
  - Covers: implementation order steps 11-12.
  - Verify: independent code-review subagent reports no unresolved blockers.
  - Evidence: review findings, fixes, rerun commands, and residual risks.
  - Sign-off: parent/date after review approval is recorded.

## Verification

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_conversation_history_prompt_projection.py -q`
  - Expected after implementation: pass.
- `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_ground_display_name_target_from_observed_group_reply -q -s -m live_llm`
  - Expected after implementation: run one case, produce raw trace and review
    artifact; quality judgment must inspect whether `蚝爹油` is grounded as a
    visible participant.

### Integration And Regression Tests

- `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py tests\test_adapter_readable_mentions_live_llm.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_relevance_reply_to_bot_live_llm.py tests\test_relevance_sensitivity_live_llm.py -q -m "not live_llm"`
- `venv\Scripts\python.exe -m pytest tests\test_rag_phase3_capability_agents.py tests\test_rag_phase3_initializer_live_llm.py tests\test_rag_agent_package_prompt_stability.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_history_policy.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_user_memory_units_rag_flow.py tests\test_consolidation_evidence_hardening_live_llm.py -q -m "not live_llm"`
- `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_group_scene_digest.py tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_interaction_style.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_live_llm_prompt_contracts.py tests\test_cognition_chain_connector_mapping.py -q -m "not live_llm"`

### Static Greps

- `rg -n "_project_recorder_chat_history|_surface_history_for_social_context|_surface_history_for_style|_surface_history_for_visual|_project_messages_for_prompt|_digest_message_rows" src\kazusa_ai_chatbot`
  - Expected after migration: no local bespoke conversation-history LLM
    projection helpers remain, or remaining matches are deleted function names
    in tests documenting removal.
- `rg -n "\"chat_history\"\\s*:\\s*format_storage_utc_history_for_llm|\"conversation_history\"\\s*:\\s*format_storage_utc_history_for_llm|\"chat_history_recent\"\\s*:\\s*format_storage_utc_history_for_llm|\"chat_history_wide\"\\s*:\\s*format_storage_utc_history_for_llm" src\kazusa_ai_chatbot`
  - Expected after migration: zero matches in LLM human-payload builders.
- `rg -n "speaker_name|speaker_ref|display_name.*body_text|body_text.*display_name" src\kazusa_ai_chatbot tests`
  - Expected after migration: matches are allowed only in deterministic
    storage, user/profile code, non-conversation evidence, or tests that
    explicitly validate internal structured rows. Prompt instructions for
    model-facing conversation history must not require these row fields.

### Static Python Checks

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\conversation_history_prompt_projection.py`
- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\persona_relevance_agent.py src\kazusa_ai_chatbot\rag\prompt_projection.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\consolidation\memory_units.py src\kazusa_ai_chatbot\memory_writer_prompt_projection.py src\kazusa_ai_chatbot\reflection_cycle\projection.py src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`

### Prompt Payload Evidence

Capture or inspect representative human payloads for:

- Decontextualizer observed QQ case.
- Relevance group-chat case.
- RAG initializer context with `chat_history_recent` and `chat_history_wide`.
- Conversation evidence worker judge input.
- Conversation progress recorder input.
- Memory-unit extractor input.
- Reflection hourly payload.
- Group-scene digest payload.

For each payload, record that conversation history appears as `list[str]`
transcript lines and not as dict rows.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, live LLM review artifacts, and path-safe commands for
  Windows paths.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `project_conversation_history_for_llm` is the sole production helper used for
  model-facing conversation-history transcript formatting.
- All LLM payloads that present conversation history use `list[str]` logging
  lines, regardless of whether the source window is full, narrow, filtered,
  scoped, retrieved, or background/reflection selected.
- No model-facing conversation-history prompt requires row fields such as
  `display_name`, `speaker_name`, `speaker_ref`, or `body_text`.
- Internal structured state and deterministic identity/source metadata remain
  available before projection.
- The observed `蚝爹油` live decontextualizer case is rerun with an
  inspectable trace and review artifact.
- All verification commands either pass or have recorded, user-approved
  exceptions.
- Independent code review reports no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Reflection loses previous deidentification behavior | Treat visible display names as transcript labels only and keep reflection output privacy validation intact | Reflection prompt-contract tests and code review inspect persistence boundaries. |
| Memory extraction loses speaker ownership | Logging lines preserve speaker before the colon and reply targets when present | Memory-unit extractor tests inspect prompt payload and output quality. |
| RAG evidence loses source refs | Keep structured rows inside deterministic RAG worker/projection code and project only LLM-facing payloads | RAG conversation evidence tests verify resolved refs, active-turn exclusion, and source hints. |
| Existing prompt tests fail from expected shape changes | Update tests to assert central transcript shape instead of old dict rows | Prompt-contract tests pass. |
| Live local LLM still fails decontextualizer grounding | Keep failure as quality evidence and compare raw traces; do not add one-off prompt hacks outside this plan | Live LLM trace and review artifact record result. |

## Execution Evidence

Pre-execution draft. Record command output, trace paths, review findings, and
stage sign-offs here during implementation.
