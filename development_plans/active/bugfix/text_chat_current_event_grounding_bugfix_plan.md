# text chat current-event grounding bugfix plan

## Summary

- Goal: stop text-chat cognition from inverting current-turn actor, addressee,
  reply, and target ownership when the typed current-message facts already
  exist but are not visible to the L2/L3 semantic prompts.
- Plan class: large.
- Status: draft.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`, `debug-llm`.
- Overall cutover strategy: bigbang for the prompt-facing text-chat cognition
  payload; compatible for existing graph state, storage, RAG, decontextualizer,
  dialog, adapters, and public contracts.
- Highest-risk areas: overfitting the tiramisu regression, leaking raw ids or
  wire syntax into prompts, duplicating `prompt_message_context` without a
  stable contract, adding deterministic semantic labels, widening dialog/RAG
  scope, and increasing response-path latency.
- Acceptance criteria: L2a Consciousness, L2c1 Judgment Core, and L3 Content
  Plan receive a bounded prompt-safe `current_event_grounding` payload for
  plain text-chat turns; no raw platform ids, global ids, message ids,
  `raw_wire_text`, attachment binary data, or platform wire markers enter that
  payload; the Kazusa victory-anchor live LLM repro passes after inspection;
  related mention, referent, and stale-dialog regressions remain green.

## Context

The observed production failure happened around the Kazusa message:

```text
诶……在这种事情上宣布胜利？你真是幼稚得要命。
唔，不过是提拉米苏的话……
好啦，这次算你赢了！快给我！
```

The surrounding conversation established the opposite ownership:

- 2026-06-23 16:46:57 +12: Kazusa said she had probably won.
- 2026-06-23 16:48:13 +12: the user replied to Kazusa and wrote that Kazusa
  won, then offered tiramisu.
- 2026-06-23 16:49:31 +12: Kazusa replied as if the user won.

The failure mode is current-turn actor ownership inversion. The relevant
current-message facts are present in graph state:

- `user_input` contains the current visible text.
- `prompt_message_context` contains prompt-safe typed current-message structure.
- `reply_context` contains the replied-to display name and excerpt.
- `referents` carries decontextualizer output.

The decontextualizer receives these fields and can preserve the visible
referents. `run_cognition_chain(...)` and `run_text_surface_planning(...)`
also preserve the fields in internal state. The gap is later: plain text-chat
L2/L3 prompt payloads can rely on `decontextualized_input`, upstream
monologue, `referents`, and source payloads, but the source payload for
`text_chat_user_message` is empty. L3 Content Plan can therefore be asked to
resolve visible content without the original current message and reply facts.

Prior completed plans show why this must be a boundary repair rather than a
one-off wording patch:

- `typed_message_envelope_stage2_plan.md` moved platform wire parsing to
  adapters and established clean `body_text`, typed mentions, typed reply
  targets, and typed addressing.
- `prompt_safe_message_context_plan.md` created prompt-safe current-message
  projection and required typed addressing, reply context, mentions, and
  attachment summaries to be preserved without storage-only fields.
- `universal_chat_history_llm_projection_plan.md` normalized conversation
  history projection, but explicitly deferred redesigning `prompt_message_context`
  and `reply_context`.
- `decontextualizer_scope_users_referent_bugfix_plan.md` kept referent
  judgment before RAG/cognition and rejected deterministic likely-referent
  labels, evidence-role ranking, retry loops, and RAG-side pronoun resolution.
- `dialog_anchor_authority_stale_history_bugfix_plan.md` made dialog render
  the L3 semantic plan instead of reinterpreting facts from stale history.
- `l3_dialog_content_plan_contract_bugfix_plan.md` made L3 Content Plan the
  owner of resolved semantic content before dialog runs.

Existing live LLM tests cover mention readability, decontextualizer scoped
users, referent clarification, reply-to-bot relevance, dialog mention target
selection, L3/dialog content-plan handoff, and stale-history dialog authority.
They do not cover plain text-chat current-message/reply grounding being
present in L2/L3 semantic prompts.

The current repro test artifact is
`tests/test_kazusa_victory_anchor_live_llm.py`. Its initial failing trace
recorded that `reply_context`, `prompt_message_context`, and `user_input` were
absent from the L3 prompt payload, and the model produced a plan equivalent to
"算你赢了".

## Mandatory Skills

- `development-plan`: load before editing this plan, moving lifecycle status,
  executing stages, recording evidence, or signing off.
- `local-llm-architecture`: load before changing cognition prompt payloads,
  prompt text, L2/L3 stage boundaries, or LLM context budgets.
- `py-style`: load before editing Python production code.
- `cjk-safety`: load before editing Python files containing CJK test fixtures
  or prompt strings.
- `test-style-and-execution`: load before adding, changing, or running
  deterministic, patched LLM, or live LLM tests.
- `debug-llm`: load before running live LLM tests or writing review artifacts
  for LLM traces.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Execute this plan only after the user approves it and explicitly authorizes
  production-code changes. This draft alone does not authorize production edits.
- Use parent-led native subagent execution for implementation. If native
  subagents are unavailable, stop before execution unless the user explicitly
  requests fallback execution.
- Do not read `.env`.
- Do not add response-path LLM calls, retry loops, evaluator loops, repair
  prompts, summarizers, or fallback paths.
- Do not add deterministic semantic labels such as winner, loser, owner,
  recipient, target, inviter, or requester. Deterministic code may only project
  prompt-safe facts already present in graph state.
- Do not expose raw platform ids, global ids, message ids, `raw_wire_text`,
  CQ/Discord wire markers, attachment URLs, `base64_data`, embeddings,
  database ids, source refs, adapter ids, scheduler ids, or delivery ids to
  L2/L3 prompt payloads.
- LLM stages own semantic judgment. Deterministic code owns whitelist
  projection, truncation, validation, and prompt payload construction.
- RAG remains evidence retrieval. Do not make RAG resolve current pronouns,
  current actor ownership, or reply-thread semantics.
- Decontextualizer remains the referent-rewrite owner. Do not move
  decontextualization into L2/L3 and do not weaken existing `referents`
  behavior.
- Dialog remains a renderer of `content_plan`. Do not pass raw current-message
  text or reply context to dialog under this plan.
- Adapters, message-envelope storage, database schemas, conversation history,
  queueing, delivery, consolidation, scheduler, reflection, and self-cognition
  source collection are out of production scope for this plan.
- Real LLM tests must run one case at a time with `-q -s -m live_llm`; inspect
  the emitted trace before running the next live case.
- Prompt changes must be coherent rewrites of the affected prompt sections,
  not appended negative rules or test-shaped examples.
- Do not hard-code Kazusa, tiramisu, victory, or any captured failure nouns in
  reusable runtime prompts.

## Must Do

- Create a bounded prompt-safe projection named `current_event_grounding` for
  model-facing current-message facts.
- Add deterministic tests proving the projection includes visible text,
  speaker display name, mention display names, direct active-character
  addressing descriptor, reply display name, and reply excerpt while excluding
  raw ids and wire syntax.
- Add deterministic L2a Consciousness, L2c1 Judgment Core, and L3 Content Plan
  payload tests proving plain text-chat turns include `current_event_grounding`.
- Update L2a Consciousness prompt text so it treats `current_event_grounding`
  as the highest-priority source for current speaker, visible text, direct
  address, and reply ownership when forming the internal monologue.
- Update L2c1 Judgment Core prompt text so it can correct a candidate stance
  or monologue that conflicts with current-event grounding.
- Update L3 Content Plan prompt text so it preserves subject, addressee,
  recipient, reply, and quoted-text ownership from current-event grounding
  before dialog renders final wording.
- Keep `decontextualized_input` and `referents` as existing inputs. The new
  payload augments them; it does not replace them.
- Keep `build_cognition_prompt_source_payload(...)` behavior for non-chat
  source variants unless the implementation proves a narrowly required local
  helper call is cleaner for the approved target stages.
- Run the existing Kazusa victory-anchor live LLM repro one case at a time
  before production edits to confirm the failure, then after implementation to
  confirm the fix.
- Produce a human-readable live LLM review artifact under
  `test_artifacts/llm_reviews/`.
- Run focused deterministic, live LLM, and static verification gates listed in
  this plan.

## Deferred

- Do not redesign `CognitionChainInputV1`, `CognitionTextSurfaceInputV1`,
  `CognitiveEpisode`, source labels, or `model_visible_percepts`.
- Do not move current-event grounding into adapter code, storage schemas, RAG
  prompt projection, conversation-history projection, or dialog payloads.
- Do not add a universal prompt payload that every LLM stage receives.
- Do not add a general thread graph, full-history retry, resolver loop retry,
  candidate ranking, semantic overlap scoring, or LLM-as-judge evaluator.
- Do not add compatibility aliases, parallel field names, feature flags, prompt
  variants, or fallback mappers for old payload shapes.
- Do not alter mention rendering, reply hydration, outbound addressing,
  `target_addressed_user_ids`, delivery mentions, or adapter reply behavior.
- Do not rework decontextualizer prompts, RAG slot planning, Cache2 keys,
  conversation progress, memory lifecycle, or self-cognition group-review
  source packets unless a compile failure is directly caused by the approved
  edits.
- Do not batch-run live LLM tests.

## Cutover Policy

Overall strategy: bigbang for the model-facing text-chat cognition payload.

| Area | Policy | Instruction |
|---|---|---|
| `current_event_grounding` payload | bigbang | Add one canonical prompt-facing shape and wire it directly into approved L2/L3 semantic stages. No alias, fallback name, or alternate shape. |
| Existing graph state | compatible | Keep `user_input`, `prompt_message_context`, `reply_context`, and `referents` unchanged. |
| Public cognition contracts | compatible | Do not change public `CognitionChainInputV1` or `CognitionTextSurfaceInputV1` schemas unless an implementation blocker proves a field is missing from the public contract. |
| Prompt source variants | bigbang local | Plain text-chat approved stages receive grounding through the new payload. Non-chat reflection, internal-thought, and background variants do not receive chat grounding. |
| Dialog/RAG/decontext/adapters | compatible | Preserve existing behavior and call counts. |
| Tests | bigbang | Implement the projection and payload tests named in `Implementation Order`, and remove old expectations that plain text-chat semantic stages lack current-event grounding. |

## Target State

For a normal visible text-chat turn, the semantic path is:

```text
typed message envelope and reply context
  -> prompt-safe graph state
  -> decontextualizer produces decontextualized_input and referents
  -> L2a Consciousness reads decontextualized_input + current_event_grounding
  -> L2c1 Judgment Core can correct candidate ownership conflicts
  -> L3 Content Plan reads current_event_grounding before final content_plan
  -> dialog renders content_plan without reinterpreting current message facts
```

The model-facing `current_event_grounding` is a compact evidence packet, not a
semantic answer. It tells L2/L3 what is visibly present in the current turn:
who spoke, what the clean current text says, which display names were mentioned
or addressed, and what message was replied to.

The LLM still decides the character's stance, interpretation, content plan,
and wording. Deterministic code only keeps the prompt-safe current-event facts
available at the semantic stages that need them.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix owner | Add current-event grounding to L2a, L2c1, and L3 Content Plan. | These stages form or correct semantic interpretation and response content. Dialog must not reinterpret. |
| Payload type | Use one bounded `current_event_grounding` dict. | A named dict is inspectable and avoids scattering raw `prompt_message_context` / `reply_context` through prompt payloads. |
| Existing fields | Preserve existing `user_input`, `prompt_message_context`, `reply_context`, `decontextualized_input`, and `referents`. | They remain graph-state source facts and existing prompt inputs. The new payload is a prompt-facing projection, not a state/storage replacement. |
| Deterministic logic | Whitelist and truncate prompt-safe fields only. | The code should not infer "who won" or any other semantic actor relation. |
| Display names | Include display names, not ids. | Local LLMs need readable identity labels; ids are operational metadata. |
| Prompt wording | Define the field semantically and positively. | Avoid negative-rule accretion and test-shaped examples. |
| Source selection helper | Prefer a small shared projection helper over duplicating extraction in each stage. | L2a, L2c1, and L3 need identical prompt-safe current-event facts and identical id-leak checks. |
| Dialog | Do not pass `current_event_grounding` to dialog. | Dialog renders `content_plan`; L3 must resolve semantic content before dialog. |
| RAG | Do not change RAG. | The issue is not evidence retrieval and current-turn RAG exclusion remains intentional. |

## Contracts And Data Shapes

Create a prompt-facing projection with this canonical top-level key:

```python
{
    "current_event_grounding": {
        "speaker_display_name": str,
        "current_message_text": str,
        "mentions": list[str],
        "addressing": {
            "addresses_active_character": bool,
            "addressed_display_names": list[str],
            "broadcast": bool,
        },
        "reply": {
            "reply_to_display_name": str,
            "reply_excerpt": str,
            "reply_to_active_character": bool,
        },
    }
}
```

Projection rules:

- `speaker_display_name` comes from the current user display name already in
  cognition state.
- `current_message_text` comes from `prompt_message_context.body_text` when
  non-empty; otherwise it falls back to `user_input`.
- `mentions` contains only non-empty mention display names from
  `prompt_message_context.mentions`.
- `addressing.addresses_active_character` is derived by comparing
  prompt-safe state identity internally; only the boolean leaves the helper.
- `addressing.addressed_display_names` contains readable names only. It may
  include the active character runtime name when the typed address points to
  the active character. It may include mentioned user display names when those
  names are already present in the current prompt-safe mention list.
- `addressing.broadcast` copies the prompt-safe boolean value.
- `reply.reply_to_display_name` and `reply.reply_excerpt` come from
  `reply_context`.
- `reply.reply_to_active_character` is derived internally and emitted only as a
  boolean.
- Empty strings and empty lists are allowed. The field shape stays stable.
- The projection must cap `current_message_text`, `reply_excerpt`, individual
  display names, and list lengths with deterministic constants.
- The projection must never emit ids, raw wire syntax, storage-only fields, or
  binary attachment data.

Forbidden compatibility shapes:

```python
{"current_turn_anchor": "..."}
{"winner": "..."}
{"owner": "..."}
{"likely_referent": "..."}
{"reply_context": {"reply_to_platform_user_id": "..."}}
{"prompt_message_context": {"addressed_to_global_user_ids": ["..."]}}
```

Create exactly one public local helper inside
`src/kazusa_ai_chatbot/cognition_chain_core/current_event_grounding.py`:

```python
def build_current_event_grounding_for_llm(
    *,
    user_input: object,
    prompt_message_context: Mapping[str, Any],
    reply_context: Mapping[str, Any],
    speaker_display_name: str,
    active_character_display_name: str,
    active_character_global_user_id: str,
) -> dict[str, Any]:
    """Return the value for human_payload["current_event_grounding"]."""
```

The approved L2/L3 stages must assign this return value to
`msg["current_event_grounding"]`. Existing code must not import private helper
functions from this module.

## LLM Call And Context Budget

No new LLM calls are added.

Affected response-path calls:

| Stage | Before | After | Context impact |
|---|---|---|---|
| L2a Consciousness | One existing `COGNITION_LLM` call with decontextualized input, L1 fields, RAG, profile, memory, progress, and source payload. | Same call, plus bounded `current_event_grounding`. | Adds at most about 2,500 serialized characters before tokenization. |
| L2c1 Judgment Core | One existing `COGNITION_LLM` call with candidate monologue/stance, boundary, affinity, referents, and source payload. | Same call, plus bounded `current_event_grounding`. | Adds at most about 2,500 serialized characters. |
| L3 Content Plan | One existing `COGNITION_LLM` call with decontextualized input, referents, RAG, L2 outputs, selected surface intent, resolver state, memory lifecycle, style context, progress, and source payload. | Same call, plus bounded `current_event_grounding`. | Adds at most about 2,500 serialized characters. |

Using the default 50k-token planning cap, the added payload is small relative
to the existing prompts. Latency should not materially change because call
count is unchanged. If prompt traces show the helper exceeds its cap, stop and
tighten deterministic truncation; do not add a summarizer.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/cognition_chain_core/current_event_grounding.py`
  - Owns the prompt-safe current-event grounding projection helper and local
    constants.
- `tests/test_cognition_current_event_grounding.py`
  - Focused deterministic tests for projection shape, id stripping, truncation,
    and L2/L3 prompt payload inclusion.
- `tests/test_kazusa_victory_anchor_live_llm.py`
  - Keep or update the existing live LLM repro so it is part of the tracked
    test suite for this bug.
- `test_artifacts/llm_reviews/current_event_grounding_text_cognition_<timestamp>.md`
  - Human-readable review of live LLM traces from this plan.

### Modify

- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`
  - Add `current_event_grounding` to L2a Consciousness and L2c1 Judgment Core
    human payloads.
  - Rewrite only the affected prompt sections to define the new field and its
    authority.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - Add `current_event_grounding` to L3 Content Plan human payload.
  - Rewrite only the affected content-plan prompt sections.
- `src/kazusa_ai_chatbot/cognition_chain_core/graph_state.py`
  - Update internal state typing only if the implementation stores
    `current_event_grounding` in state instead of building it at each approved
    stage.
- `src/kazusa_ai_chatbot/cognition_chain_core/README.md`
  - Document the new L2/L3 current-event grounding contract if the prompt
    payload shape becomes part of the local cognition-core ICD.
- Existing prompt-contract or baseline tests that assert old payload keys.
  - Update only the assertions that conflict with the new approved payload.

### Keep

- `src/kazusa_ai_chatbot/cognition_chain_core/chain.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/surface.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/prompt_selection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
- `src/kazusa_ai_chatbot/rag/**`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- adapters, database schemas, conversation storage, delivery, scheduler,
  reflection, self-cognition, and consolidation modules.

If implementation discovers that one `Keep` file must change to compile or to
preserve the approved contract, stop and record the exact blocker before
editing it.

## Overdesign Guardrail

- Actual problem: plain text-chat semantic stages can invert current-turn actor
  or addressee ownership because prompt-safe current-message/reply facts exist
  in state but are not visible in the L2/L3 semantic prompts that decide
  interpretation and content.
- Minimal change: build one bounded prompt-safe `current_event_grounding`
  projection from existing state and add it only to L2a Consciousness, L2c1
  Judgment Core, and L3 Content Plan payloads with coherent prompt wording.
- Ownership boundaries: adapters normalize wire syntax; brain/service state
  carries typed current-message fields; decontextualizer rewrites referents;
  RAG retrieves evidence; L2/L3 decide semantic interpretation and response
  content; dialog renders `content_plan`; deterministic code validates,
  truncates, and keeps unsafe fields out of prompts.
- Rejected complexity: deterministic semantic actor labels, special-case
  victory/tiramisu logic, `current_turn_anchor`, raw `prompt_message_context`
  passthrough, dialog reinterpretation, RAG changes, full thread graph,
  history-window changes, new LLM calls, retry/evaluator loops, compatibility
  aliases, prompt variants, feature flags, and adapter/storage migrations.
- Evidence threshold: add richer thread ownership state, extra prompt stages,
  or broader current-event projection only after at least two reviewed failures
  show that the approved `current_event_grounding` field was present in traces
  and still insufficient for a different recurring ownership class.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's payload shape, stage ownership, and forbidden
  fields.
- The responsible agent must search for existing prompt-safe projection helpers
  before adding the new helper. Reuse existing truncation or whitelist patterns
  where they fit without changing unrelated semantics.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, prompt variants,
  feature flags, or extra features.
- The responsible agent must treat changes outside `cognition_chain_core` and
  the listed tests as blockers unless a compile/import failure proves the
  outside change is required.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If tests reveal unrelated failures, report them and keep this plan focused
  unless the failure blocks the approved change.
- If the plan and code disagree, preserve the plan's stated ownership boundary
  and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Refresh the baseline.
   - Command: `git status --short`.
   - Confirm only this plan, its registry row, and the existing or planned
     repro test are in scope.
   - Read this plan, `README.md`, `docs/HOWTO.md`,
     `src/kazusa_ai_chatbot/cognition_chain_core/README.md`,
     `src/kazusa_ai_chatbot/nodes/README.md`, and the source/test files listed
     in `Change Surface`.

2. Run the existing live repro before production edits.
   - Command:
     `venv\Scripts\python.exe -m pytest tests\test_kazusa_victory_anchor_live_llm.py::test_live_content_plan_preserves_kazusa_as_victory_subject -q -s -m live_llm`
   - Expected before implementation: fails or records a prompt omission where
     `current_event_grounding`, `reply_context`, `prompt_message_context`, and
     current text are not available to L3 Content Plan.
   - Evidence: record trace path and behavior judgment.

3. Add focused deterministic projection tests.
   - File: `tests/test_cognition_current_event_grounding.py`.
   - Tests:
     - `test_current_event_grounding_projects_visible_names_without_ids`
     - `test_current_event_grounding_caps_text_and_reply_excerpt`
     - `test_current_event_grounding_omits_raw_wire_and_storage_fields`
   - Expected before implementation: helper import or contract assertions fail.

4. Add deterministic L2/L3 prompt-payload tests.
   - File: `tests/test_cognition_current_event_grounding.py`.
   - Tests:
     - `test_l2a_payload_includes_current_event_grounding_for_text_chat`
     - `test_l2c1_payload_includes_current_event_grounding_for_text_chat`
     - `test_l3_content_plan_payload_includes_current_event_grounding_for_text_chat`
   - Expected before implementation: captured payloads lack the new key.

5. Start one production-code subagent after the focused tests are established.
   - Ownership: production source under `src/kazusa_ai_chatbot/cognition_chain_core/`
     only.
   - The subagent must not edit tests except if the parent explicitly requests
     a narrow fixture correction.

6. Implement the projection helper.
   - Create `current_event_grounding.py`.
   - Add direct whitelist projection, truncation constants, and tests.
   - Ensure no helper output contains keys containing `id`, `raw`, `wire`,
     `base64`, `url`, `message_id`, `platform_user_id`, or
     `global_user_id`.

7. Wire approved L2/L3 stages.
   - Add the helper output to L2a Consciousness, L2c1 Judgment Core, and L3
     Content Plan message payloads.
   - Keep non-chat source variants free of chat grounding.

8. Rewrite affected prompt sections coherently.
   - L2a: current-event grounding is the direct current-message/reply evidence
     for speaker, visible text, addressee, and reply ownership.
   - L2c1: candidate L2a interpretation can be corrected when it conflicts
     with current-event grounding.
   - L3 Content Plan: content must preserve current-event ownership before
     dialog renders it.
   - Do not add captured-failure nouns or negative-only warning paragraphs.

9. Update documentation and stale tests.
   - Update `cognition_chain_core/README.md` only if the new payload becomes
     part of documented core behavior.
   - Update existing baseline tests only where they assert the old payload
     omission or old prompt wording.

10. Run focused deterministic tests and static checks.
    - Run the commands listed under `Verification`.
    - Fix only issues inside the approved change surface.

11. Run live LLM tests one at a time and inspect traces.
    - Run the victory-anchor repro.
    - Run the selected related live regressions.
    - Write the LLM review artifact.

12. Run independent code review.
    - Start one independent code-review subagent after verification passes.
    - Remediate only findings inside approved scope.
    - Rerun affected checks and record evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only
  under `src/kazusa_ai_chatbot/cognition_chain_core/`; does not edit tests
  unless the parent explicitly directs it; closes after planned production code
  changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - baseline and focused test contract established.
  - Covers: implementation steps 1-4.
  - Verify: baseline live repro command was run individually; deterministic
    projection/payload tests exist and fail for the expected missing helper or
    missing payload key.
  - Evidence: record `git status --short`, failing test summaries, and trace
    path in `Execution Evidence`.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - current-event grounding projection implemented.
  - Covers: implementation step 6.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_cognition_current_event_grounding.py::test_current_event_grounding_projects_visible_names_without_ids tests\test_cognition_current_event_grounding.py::test_current_event_grounding_caps_text_and_reply_excerpt tests\test_cognition_current_event_grounding.py::test_current_event_grounding_omits_raw_wire_and_storage_fields -q`
  - Evidence: record changed source files and focused test output.
  - Handoff: next stage wires approved L2/L3 stages.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - L2/L3 prompt payloads and prompts updated.
  - Covers: implementation steps 7-9.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_cognition_current_event_grounding.py -q`
  - Evidence: record payload-capture assertions and prompt-render checks.
  - Handoff: next stage runs regression verification.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - deterministic and static regression verification complete.
  - Covers: implementation step 10.
  - Verify: all deterministic, compile, and static grep commands in
    `Verification` pass or return the expected no-match status.
  - Evidence: record command outputs and any scoped fixes.
  - Handoff: next stage runs live LLM tests.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - live LLM verification and review artifact complete.
  - Covers: implementation step 11.
  - Verify: each live LLM command in `Verification` is run individually with
    trace inspection.
  - Evidence: record trace paths, model behavior judgment, and review artifact
    path.
  - Handoff: next stage runs independent code review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - independent code review complete.
  - Covers: implementation step 12 and `Independent Code Review`.
  - Verify: review subagent reports no unresolved blockers; any in-scope fixes
    are made; affected checks are rerun.
  - Evidence: record review findings, fixes, rerun commands, residual risks,
    and approval status.
  - Handoff: plan may be moved toward completion only after this stage is
    signed off.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Python Checks

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_chain_core\current_event_grounding.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l3.py tests\test_cognition_current_event_grounding.py tests\test_kazusa_victory_anchor_live_llm.py`

### Static Greps

- `rg -n "current_turn_anchor|winner|loser|tiramisu|提拉米苏|算你赢|Kazusa victory|victory-anchor" src\kazusa_ai_chatbot\cognition_chain_core`
  - Expected: no matches in runtime source.
  - Allowed: matches in tests and this plan only.
- `rg -n "platform_user_id|global_user_id|platform_message_id|reply_to_message_id|raw_wire_text|base64_data|addressed_to_global_user_ids" src\kazusa_ai_chatbot\cognition_chain_core\current_event_grounding.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l3.py`
  - Expected: matches are allowed only inside the projection helper's internal
    input reads, comparisons, forbidden-field tests, or comments explaining
    non-output fields. Stage prompt payloads must not serialize these keys.
- `rg -n "current_event_grounding" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\conversation_progress src\adapters`
  - Expected: no matches. The new payload belongs to cognition-core L2/L3
    prompts only.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_cognition_current_event_grounding.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_chain_core_contracts.py tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_l3_dialog_content_plan_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_referents_live_llm.py -q -m "not live_llm"`
- `venv\Scripts\python.exe -m pytest tests\test_adapter_readable_mentions_live_llm.py tests\test_dialog_anchor_boundary_live_llm.py -q -m "not live_llm"`

### Prompt Render Checks

- Render L2a Consciousness, L2c1 Judgment Core, and L3 Content Plan prompts
  through the same `.format(...)` path used at runtime.
- Expected: no missing placeholders, no literal unescaped braces errors, and
  each affected prompt mentions `current_event_grounding` in coherent input
  guidance and decision procedure sections.

### Real LLM Tests

Run each command one at a time with `-q -s -m live_llm`; inspect the trace
before running the next command.

- `venv\Scripts\python.exe -m pytest tests\test_kazusa_victory_anchor_live_llm.py::test_live_content_plan_preserves_kazusa_as_victory_subject -q -s -m live_llm`
  - Expected after implementation: L3 Content Plan preserves the current
    message's declared active-character victory and does not plan wording that
    treats the user as the winner.
- `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_decontext_scope_users_live_llm.py::test_live_scope_users_ground_display_name_target_from_observed_group_reply -q -s -m live_llm`
  - Expected: existing decontextualizer scoped-user grounding remains valid.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_referents_live_llm.py::test_live_CONTENT_PLAN_keeps_mixed_referent_question_narrow -q -s -m live_llm`
  - Expected: structured referents still control narrow clarification and are
    not overridden by current-event grounding.
- `venv\Scripts\python.exe -m pytest tests\test_dialog_anchor_boundary_live_llm.py::test_live_dialog_agent_keeps_content_plan_over_stale_history -q -s -m live_llm`
  - Expected: dialog still follows content plan and does not regain raw current
    or stale history authority.

### LLM Review Artifact

- Write `test_artifacts/llm_reviews/current_event_grounding_text_cognition_<timestamp>.md`.
- Include: command, model route if visible in trace, trace paths, prompt
  payload keys, observed current-event grounding payload, raw LLM output,
  parsed output, pass/fail judgment, and residual risk.

## Independent Plan Review

Run this gate before approval or execution. Because this draft was created by
the parent agent, prefer a reviewer that did not draft the plan when the user
authorizes subagent review. If no separate reviewer is authorized, the parent
agent must review from a fresh-review posture.

Review scope:

- The plan fixes the current-event grounding failure without special-casing the
  captured tiramisu/victory case.
- Previous-stage artifacts are carried forward correctly:
  `typed_message_envelope_stage2_plan.md`,
  `prompt_safe_message_context_plan.md`,
  `universal_chat_history_llm_projection_plan.md`,
  `decontextualizer_scope_users_referent_bugfix_plan.md`,
  `dialog_anchor_authority_stale_history_bugfix_plan.md`, and
  `l3_dialog_content_plan_contract_bugfix_plan.md`.
- Ownership boundaries are preserved: adapters normalize, decontext resolves
  references, RAG retrieves evidence, L2/L3 decide semantic interpretation,
  dialog renders, deterministic code validates and projects.
- `current_event_grounding` is bounded, prompt-safe, and free of raw ids,
  storage-only fields, and wire syntax.
- Change surface is narrow enough for the observed failure class and does not
  authorize RAG, dialog, adapter, storage, scheduler, or consolidation edits.
- Verification includes deterministic payload tests, live LLM quality evidence,
  prompt-render checks, static greps, and related regression tests.

Record blockers, non-blocking findings, required edits, residual risks, and
approval status in the final response or in `Execution Evidence` if execution
has started.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt payload leaks, raw id leakage,
  storage/persistence risk, brittle live fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression quality, including whether the live LLM test proves prompt
  behavior rather than only harness execution.

The parent agent fixes findings directly only when the fix is inside the
approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- L2a Consciousness, L2c1 Judgment Core, and L3 Content Plan payloads include
  `current_event_grounding` for normal text-chat turns.
- `current_event_grounding` contains only prompt-safe current-message text,
  display names, booleans, and bounded reply excerpt text.
- No raw ids, raw wire syntax, storage-only fields, adapter fields, database
  refs, binary attachment data, or delivery metadata are serialized into the
  new payload.
- Runtime prompts do not mention Kazusa, tiramisu, victory, or any captured
  failure noun as reusable examples.
- The Kazusa victory-anchor live LLM repro passes after trace inspection.
- Existing decontextualizer scoped-user, referent clarification, and dialog
  anchor-authority live regressions remain acceptable.
- No new response-path LLM call, retry loop, evaluator loop, summarizer,
  fallback path, compatibility alias, or feature flag is added.
- The independent code review gate reports no unresolved blockers.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Payload becomes a raw-envelope passthrough | Use a dedicated whitelist projection helper and id/wire static greps | Projection unit tests and static greps |
| Model overweights current message text and ignores decontext/referents | Prompt says grounding disambiguates ownership while `decontextualized_input` and `referents` remain semantic inputs | Mixed-referent live regression |
| Fix overfits the tiramisu case | Runtime prompts exclude captured nouns and deterministic code emits no semantic actor labels | Static grep and prompt review |
| Context grows too much for local LLM | Apply deterministic caps and no new LLM calls | Projection cap tests and trace payload size inspection |
| Dialog starts reinterpreting current facts | Do not change dialog payload; keep dialog anchor-authority regression | Dialog live regression |
| RAG or decontext ownership drifts | No RAG/decontext production changes in scope | Static greps and changed-file review |

## Execution Evidence

No execution has started under this draft plan.

## Plan Self-Review

- Review result: parent self-review performed on 2026-06-23. No separate
  review subagent was used because this request asked for plan review, not
  delegated subagent review. The review found one blocker: the helper function
  name was left to implementation. The blocker was fixed by naming
  `build_current_event_grounding_for_llm(...)` and its signature in
  `Contracts And Data Shapes`.
- Coverage: every `Must Do` item maps to an implementation step, progress
  checkpoint, and verification gate.
- Minimality: the plan adds one bounded prompt-facing projection and wires it
  only into L2a, L2c1, and L3 Content Plan. It rejects RAG, dialog, adapter,
  storage, retry, evaluator, and semantic-label expansions.
- Placeholder scan: this plan contains no unresolved design choices or
  placeholder implementation language.
- Contract consistency: the canonical field name is `current_event_grounding`
  across summary, contract, change surface, tests, and verification.
- Granularity: each checkpoint names exact files, tests, commands, expected
  evidence, and handoff point.
- Verification: deterministic tests cover projection and payload plumbing;
  live LLM tests cover the reproduced failure and adjacent regressions.
