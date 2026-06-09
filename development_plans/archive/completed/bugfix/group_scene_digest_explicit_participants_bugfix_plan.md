# group scene digest explicit participants bugfix plan

## Summary

- Goal: make reflection-owned group scene digests preserve visible participant names and explicit chronological closure so local-LLM self-cognition can read noisy group windows without resolving `participant_N` aliases or pronouns.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang replacement of the digest prompt payload
  identity projection; keep the one-string JSON output shape unchanged.
- Highest-risk areas: prompt bloat, implicit Chinese references, wrong speaker
  attribution, missing Kazusa's latest message in busy windows, and downstream
  cognition over-reading activity labels.
- Acceptance criteria: real LLM traces show digests with visible display names,
  no `participant_N`, no implicit speaker references, bounded length, and
  correct final-window chronology for at least five group-flow cases.
- Current execution state: completed after follow-up real LLM verification and
  final inline review. Earlier Stage 5 sign-off was invalidated by later real
  LLM proof; the corrected proof and final review are recorded below.

## Context

The previous digest fix added
`conversation_progress.group_scene_digest = {"digest": str}` as optional
source hydration for reflection-attached group self-cognition. Real QQ group
905393941 evidence showed the digest sees Kazusa's own prior message, but the
current prompt payload strips user display names into `participant_N` aliases.

That deidentification was over-conservative for this consumer. Self-cognition
already receives display names in `visible_context`, and the digest is not an
external analytics surface. The alias namespace forces a local LLM to reconcile
separate representations across prompt sections, which is exactly the failure
mode the digest was meant to reduce.

## Mandatory Skills

- `development-plan`: load before plan lifecycle edits, execution, or review.
- `local-llm-architecture`: load before changing prompt-facing context, prompt
  contracts, LLM input shaping, or LLM budget.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing Chinese strings.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running real LLM tests or writing human-readable
  LLM review artifacts.

## Mandatory Rules

- Do not touch production code unless this plan is `in_progress` or
  `approved` and the user has explicitly authorized implementation.
- Use `venv\Scripts\python`; use `apply_patch` for manual edits; check
  `git status --short`; do not read `.env`.
- Preserve the output contract exactly: `{"digest": str}`. Do not add fields,
  nested objects, confidence, target, action, route, or flow-state keys.
- The digest is source support only. It must not decide speak, silence,
  apology, retry, suppression, delivery, persistence, permission, or adapter
  feasibility.
- Preserve visible participant display names in the digest input and output.
  Do not use `participant_N` aliases for self-cognition-consumed digests.
- Do not expose global user ids, platform user ids, platform message ids,
  source refs, delivery targets, adapter wire syntax, raw attachment URLs, or
  database row ids.
- The prompt must not hard-code a concrete character name in static text.
  It must use the active character's visible display name from the current
  window payload when instructing or producing `我（<name>）` wording.
- The prompt must use a positive row-reading procedure that keeps
  `display_name` visible for summarized speakers. Live regression tests reject
  placeholder leakage, known self/other inversions, and logical contradictions.
- Control output length through deterministic bounding and compact prompt
  wording. Do not tell the digest LLM to produce a fixed number of sentences,
  bullets, speakers, clauses, or facts.
- Balance content and participants by summarizing only the meaningful visible
  flow, but every summarized speaker must be named.
- Truncate busy-window rows from the beginning/top/oldest side so the newest
  rows, including Kazusa's latest visible message, remain available.
- Real LLM tests must run one case at a time with trace inspection after each
  run. A passing schema assertion is not sufficient quality proof.
- After any automatic context compaction or major checklist sign-off, reread
  this plan before continuing.
- Before final completion or lifecycle update, run the independent code review
  gate and record findings in `Execution Evidence`.

## Must Do

- Add failing deterministic tests that prove digest prompt payloads preserve
  display names and keep newest rows when row limits are exceeded.
- Add at least five real LLM digest tests:
  1. the QQ 905393941 duplicate-thanks closed-flow failure;
  2. quiet group chat without Kazusa participation;
  3. one Kazusa participation that closes a direct request;
  4. multiple Kazusa participations in separate visible beats;
  5. very busy group chat where oldest rows are truncated and Kazusa's latest
     message must remain visible.
- Run the first real LLM test before production implementation and record the
  red failure mode and trace path.
- Update `group_scene_digest` prompt payload shaping so user rows carry
  visible display names instead of `participant_N` aliases.
- Keep the digest output one string inside JSON.
- Update deterministic tests that currently assert deidentification once the
  new identity-preserving contract is implemented.
- Update reflection-cycle and self-cognition READMEs to state that group scene
  digests preserve visible display names but not internal ids.

## Deferred

- No cognition prompt changes.
- No dialog, action-router, dispatcher, adapter, persistence, scheduler, RAG,
  or background-worker changes.
- No deterministic speak/silence gate.
- No duplicate-text detector, cooldown, response-ratio tuning, retry loop, new
  model route, feature flag, or database migration.
- No rich digest schema beyond `{"digest": str}`.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Digest prompt payload | bigbang | Replace `participant_N` aliases with visible display names for prompt-facing group digest rows. |
| Digest output | compatible | Preserve `{"digest": str}` exactly. |
| Raw ids | no-op | Continue excluding ids and adapter metadata. |
| Source packet rendering | compatible | Existing `conversation_progress` rendering carries the digest. |
| Tests | bigbang | Rewrite old alias/deidentification expectations to the explicit-name contract. |

## Target State

```text
selected group activity window
  -> digest prompt payload with visible display names and newest rows preserved
  -> one-string JSON digest with explicit names and no pronouns
  -> existing self_cognition source packet rendering
  -> existing cognition/action/dialog path unchanged
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Speaker identity | Preserve display names | The consumer already sees display names; aliases add reasoning burden. |
| Internal ids | Keep excluded | Display names are enough for scene comprehension and avoid persistence identity exposure. |
| Output shape | One string JSON | User explicitly rejected richer group-digest structure. |
| Length control | Deterministic output bounding plus compact prompt | Avoid over-regulating the LLM's information amount. |
| Truncation direction | Drop oldest rows first | Newest rows define current state and contain Kazusa's latest participation. |
| Prompt wording | Explicit anti-pronoun rule | Local LLMs need direct instruction, not implied style preference. |
| Downstream changes | None | The fix supports cognition input rather than changing cognition steps. |

## Contracts And Data Shapes

Output remains:

```python
{"digest": str}
```

Prompt-facing rows should expose only prompt-safe values:

```python
{
    "timestamp": str,
    "role": str,
    "display_name": str,
    "content_activity": "text" | "empty_or_media_only",
    "text": str,
}
```

For assistant rows, `display_name` should be the visible assistant display
name when present. The prompt must explain that assistant rows are
`我（<display_name>）` from the self-cognition perspective. For user rows,
`display_name` must be the visible participant name. If a row genuinely lacks
a display name, use a compact explicit fallback such as `未知发言者` instead
of inventing a participant number.

The normalizer continues to accept only a dict with exactly one non-empty
string field named `digest`, bounded by `GROUP_SCENE_DIGEST_MAX_CHARS`, and
without action guidance.

## LLM Call And Context Budget

- LLM calls remain at most one `CONSOLIDATION_LLM` call per selected group
  review window.
- Inputs remain the selected window rows and compact activity labels only.
- No RAG, DB lookups, relationship hydration, reflection output, adapter
  metadata, or delivery metadata enters the digest prompt.
- The system prompt must be triple-single quoted and colocated with the LLM
  call and parser.
- The human message must contain only the dynamic prompt payload.
- The prompt must not hard-code the concrete character name in static text.
  It must instruct the model to use the assistant row's visible display name
  from the dynamic payload for `我（<display_name>）` references.
- Do not add repair prompts or retry loops for parsed-but-invalid JSON.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
  - Replace alias-only speaker projection with visible display-name projection.
  - Tighten the system prompt for explicit names, no implicit references, and
    final visible event chronology.
  - Preserve newest-row truncation.
  - Keep one-string JSON validation and deterministic output bounding.

- `tests/test_reflection_cycle_group_scene_digest.py`
  - Replace deidentification expectations with display-name expectations.
  - Add oldest-row truncation coverage if not already present.
  - Keep invalid shape, action-guidance rejection, and one-string bounding.

- `tests/test_reflection_cycle_group_scene_digest_live_llm.py`
  - Add five or more real LLM cases with durable trace output.

- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document explicit display-name preservation and continued internal-id
    exclusion.

### Keep

- No changes to `self_cognition.projection`.
- No changes to cognition, dialog, action specs, dispatcher, adapter,
  persistence, RAG, scheduler, or reflection window selection.

## Overdesign Guardrail

- Do not add a roster map, participant dictionary, canonical user resolver, or
  RAG-backed identity hydration.
- Do not add digest subfields for final state, latest speaker, confidence, or
  actionability.
- Do not make the prompt list a fixed amount of information to output.
- Do not add deterministic semantic parsing over the digest string.
- Do not preserve old alias behavior behind compatibility options.

## Agent Autonomy Boundaries

- The execution agent may rename private helpers inside
  `group_scene_digest.py`.
- The execution agent may not change the one-string output contract, add new
  LLM routes, add new runtime config, or edit downstream cognition prompts.
- If the live LLM cannot satisfy the explicit-name contract after prompt and
  payload fixes, stop and report evidence instead of adding another LLM stage.

## Implementation Order

1. Preserve the already-recorded red baseline unless tests are changed:
   deterministic visible-name payload failure and duplicate-thanks live LLM
   failure are recorded in `Execution Evidence`.
2. If the red tests are modified during implementation, rerun the affected red
   test before production edits and record the new expected failure.
3. Update `group_scene_digest.py` prompt payload and prompt wording.
4. Run the deterministic digest tests and record the green result.
5. Run each real LLM case one at a time, inspecting traces after each run.
6. Update READMEs.
7. Run focused self-cognition source tests to confirm the digest still renders
   through `conversation_progress`.
8. Run the independent code review gate and address findings.

## Execution Model

Execution is approved inline without subagents because the user explicitly
requested no-subagent execution for this bugfix thread.

The checked Stage 1 item is pre-implementation evidence collection requested
by the user, not approval to edit production code. Remaining checklist stages
must stay unchecked until the plan is approved and their verification evidence
is recorded.

## Progress Checklist

- [x] Stage 1 - red tests drafted
  - Files: `tests/test_reflection_cycle_group_scene_digest.py`,
    `tests/test_reflection_cycle_group_scene_digest_live_llm.py`.
  - Verify: run the duplicate-thanks live LLM case one time.
  - Evidence: record command, failure reason, and trace path.
  - Sign-off: Codex/2026-06-10 after red deterministic and real LLM failures
    were recorded.
- [x] Stage 2 - production digest input and prompt fixed
  - Files: `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`.
  - Verify: deterministic digest tests pass.
  - Evidence: record command output.
  - Sign-off: Codex/2026-06-10 after focused digest tests passed.
- [x] Stage 3 - real LLM cases inspected
  - Files: live LLM trace artifacts under `test_artifacts/llm_traces/`.
  - Verify: run each live case one at a time with `-q -s`.
  - Evidence: write a human-readable review artifact using `debug-llm`.
  - Sign-off: Codex/2026-06-10 after all five live cases passed and final
    traces were reviewed.
- [x] Stage 4 - docs and integration verified
  - Files: reflection-cycle README, self-cognition README, group source tests.
  - Verify: focused source tests pass.
  - Evidence: record command output.
  - Sign-off: Codex/2026-06-10 after README updates and focused source tests.
- [x] Stage 5 - independent code review complete
  - Scope: full diff against this plan, prompt contract, tests, and docs.
  - Verify: rerun affected tests after review fixes.
  - Evidence: record review result and residual risks.
  - Sign-off: Codex/2026-06-10 after inline no-subagent review and final
    verification.
- [x] Stage 6 - post-review real LLM regression addressed
  - Files: `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`,
    `tests/test_reflection_cycle_group_scene_digest.py`,
    `tests/test_reflection_cycle_group_scene_digest_live_llm.py`.
  - Verify: rerun deterministic digest tests and the five real LLM cases one
    at a time with trace inspection.
  - Evidence: record failed proof, prompt simplification, latest trace set,
    and updated human-readable review artifact.
  - Sign-off: Codex/2026-06-10 after duplicate-thanks and multiple-participation
    regressions were reproduced, prompt was simplified, and latest live traces
    were inspected.
- [x] Stage 7 - final inline review after follow-up
  - Scope: full diff against this plan, latest LLM review artifact, prompt
    contract, deterministic tests, live LLM tests, and docs.
  - Verify: rerun affected deterministic, integration, and static checks after
    review fixes.
  - Evidence: record review result, commands, findings, fixes, and residual
    risks.
  - Sign-off: Codex/2026-06-10 after final inline review, static checks,
    deterministic tests, integration tests, and latest live trace review.

## Verification

Run real LLM tests one at a time:

```powershell
venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_duplicate_thanks_closed_flow -q -s
venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_quiet_group_without_kazusa -q -s
venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_one_kazusa_participation -q -s
venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_multiple_kazusa_participations -q -s
venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_latest_kazusa_row_in_busy_group -q -s
```

Real LLM assertions must stay focused on contract-critical facts: required
visible names, absence of `participant_N` aliases, absence of implicit speaker
references, bounded one-string JSON, and closed-flow chronology. Do not add
golden full-digest equality.

Run deterministic and integration checks:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py
venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest.py
venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest_live_llm.py
venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q
```

## Independent Code Review

Before completing the plan, review the full diff against:

- visible-name preservation;
- no internal-id exposure;
- no `participant_N` aliases in digest prompt payload or expected digest;
- no implicit reference instructions missing from the prompt;
- no output-shape expansion beyond `{"digest": str}`;
- no downstream cognition/action/dialog changes;
- real LLM trace evidence for all named cases.

## Acceptance Criteria

- Deterministic tests prove display names enter the digest prompt payload.
- Deterministic tests prove busy-window row limiting drops oldest rows and
  preserves newest rows.
- Real LLM traces for at least five cases show display names in the digest and
  no `participant_N` aliases.
- Closed-flow cases explicitly state Kazusa already responded and whether any
  newer user request appears after her latest message.
- Digest output stays within `GROUP_SCENE_DIGEST_MAX_CHARS`.
- The runtime output shape remains exactly `conversation_progress.group_scene_digest = {"digest": str}`.

## Risks

- A local LLM may still use implicit references despite prompt wording.
- Very busy windows may require content compression that omits low-signal
  speakers; this is acceptable only when omitted speakers are not summarized.
- Display names can be noisy, blank, or visually odd; the fix must preserve
  visible names when present without resolving profiles or ids.
- Existing tests currently assert deidentification and must be intentionally
  rewritten during implementation.

## Plan Self-Review

### 2026-06-10 Self-Review

Findings addressed: removed the ambiguous optional sixth live LLM case; clarified that checked Stage 1 is pre-implementation red-baseline evidence, not production-code approval; clarified static-vs-dynamic character naming; updated implementation order to start from the recorded red baseline unless tests change; tightened real LLM verification away from golden full-digest equality; and loosened one brittle closure phrase while preserving strict identity checks.

## Execution Evidence

### 2026-06-10 Red Test Baseline

- Syntax checks passed:
  - `venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest.py`
  - `venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest_live_llm.py`
- Deterministic red command:
  - `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py::test_group_scene_digest_payload_preserves_display_names_for_cognition -q`
  - Result: failed with `KeyError: 'display_name'`, proving current
    `message_rows` do not carry visible names.
- Real LLM red command:
  - `venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_duplicate_thanks_closed_flow -q -s`
  - Result: failed with `missing visible name: 总是跌倒的企鹅`.
  - Normalized digest still used `participant_1`, `participant_2`, and
    `participant_3`.
  - Trace:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__duplicate_thanks_closed_flow.json`.
  - Human review:
    `test_artifacts/llm_traces/group_scene_digest_explicit_participants_red_review.md`.
- After self-review loosened brittle closure wording, the duplicate-thanks
  live LLM red test was rerun:
  - Command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_duplicate_thanks_closed_flow -q -s`
  - Result: failed with `missing visible name: 总是跌倒的企鹅`.
  - Normalized digest still used `participant_1`, `participant_2`, and
    `participant_3`.
  - Trace:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__duplicate_thanks_closed_flow__20260609T122703614729Z.json`.

### 2026-06-10 Stage 2 Deterministic Verification

- Production file changed:
  `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`.
- The digest prompt payload now carries `display_name` and no
  `participant_N` alias field.
- Syntax checks passed:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
  - `venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest.py`
- Focused deterministic command passed:
  - `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py -q`
  - Result: 14 passed.

### 2026-06-10 Stage 3 Real LLM Verification

- Live LLM cases were run one at a time with trace inspection after each run.
- During live inspection, prompt wording was tightened to avoid:
  - speaker/addressee inversion such as `我（杏山千纱）回复杏山千纱`;
  - copying `对方` instead of preserving visible participant names;
  - treating `@杏山千纱` as proof of an earlier Kazusa message;
  - omitting Kazusa's own visible assistant-row text;
  - negative-prompt priming of the broken phrase
    `我没有在这个窗口中发言之外`.
- Superseded live regression command from the first implementation:
  - `venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py -q -s`
  - Result: 5 passed.
- Superseded trace set from the first implementation:
  - `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__duplicate_thanks_closed_flow__20260609T130522445297Z.json`
  - `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__quiet_group_without_kazusa__20260609T130525296633Z.json`
  - `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__one_kazusa_participation__20260609T130527905756Z.json`
  - `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__multiple_kazusa_participations__20260609T130532081979Z.json`
  - `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__busy_group_latest_kazusa__20260609T130538932059Z.json`
- Note: later proof invalidated this as final evidence; see
  `Post-Review Regression And Prompt Simplification`.
- Human-readable LLM review artifact:
  `test_artifacts/llm_reviews/group_scene_digest_explicit_participants_live_llm_review_20260610.md`.

### 2026-06-10 Stage 4 Docs And Integration Verification

- Documentation updated:
  - `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - `src/kazusa_ai_chatbot/self_cognition/README.md`
- The READMEs now document that group scene digest preserves visible
  `display_name` values, excludes internal ids and `participant_N` aliases,
  keeps the one-string JSON shape, and remains source hydration rather than
  action guidance.
- Focused source command passed:
  - `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
  - Result: 13 passed.

### 2026-06-10 Stage 5 Inline Review

- Review mode: inline no-subagent review, per user instruction.
- Reviewed:
  - production diff in
    `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`;
  - deterministic digest tests;
  - live LLM tests and final trace review artifact;
  - reflection-cycle and self-cognition README updates;
  - registry and plan evidence.
- Checks:
  - `git diff --check`
  - `rg -n "speaker_ref|participant_[0-9]|active_character" ...`
  - `rg -n "global_user_id|platform_user_id|platform_message_id|source_refs|delivery_target|adapter" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
  - `rg -n "我没有在这个窗口中发言之外|回复杏山千纱|能帮到杏山千纱|对方|有人" ...`
- Findings:
  - No downstream cognition, dialog, action, dispatcher, adapter,
    persistence, RAG, or scheduler code changed.
  - Digest prompt payload now exposes visible `display_name` and continues to
    exclude internal ids and adapter metadata.
  - Output contract remains exactly `{"digest": str}`.
  - Oldest-row truncation is preserved through
    `window.participant_rows[-_GROUP_SCENE_DIGEST_ROW_LIMIT:]`.
  - Real LLM tests cover the original duplicate-thanks failure, quiet no-Kazusa
    group activity, one Kazusa response, multiple Kazusa responses, and busy
    newest-row preservation.
  - Residual risk: the local LLM can produce redundant but consistent status
    wording; no deterministic cleanup was added because the plan keeps this
    fix in the prompt/input contract.
  - Working tree contains an unrelated untracked
    `dialog_one_bubble_layout_contract_bugfix_plan.md`; it was not touched as
    part of this bugfix.
- This review was later invalidated by fresh real LLM proof; final verification
  evidence is recorded in
  `Post-Review Regression And Prompt Simplification` and the final inline
  review section.
- Superseded verification evidence at the time was:
  - deterministic digest suite: 14 passed;
  - superseded five-case live batch: passed at that time;
  - focused self-cognition group review source suite: 13 passed.

### 2026-06-10 Post-Review Regression And Prompt Simplification

- Fresh proof after Stage 5 showed the previous completion claim was not
  valid:
  - Command:
    `venv\Scripts\python -m pytest -m live_llm tests\test_reflection_cycle_group_scene_digest_live_llm.py::test_live_digest_preserves_names_for_duplicate_thanks_closed_flow -q -s`
  - Result: failed.
  - Trace:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__duplicate_thanks_closed_flow__20260609T131241410389Z.json`.
  - Failure: digest rewrote Kazusa's own quote into
    `能帮到杏山千纱`, making Kazusa the object of her own reply.
- The first quote-preservation prompt update fixed the original duplicate case
  in repeated runs, but a multiple-participation live run exposed a different
  local-LLM failure:
  - Trace:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__multiple_kazusa_participations__20260609T132246941305Z.json`.
  - Failure: the model emitted repeated self-correction blocks and normalized
    to a truncated/corrupted digest.
- Research note:
  - The decontextualizer prompt keeps direct dialogue pronouns such as `你`
    and `我` unless there is a scoped reason to resolve them.
  - The digest prompt was therefore simplified to treat assistant-row text as
    quoted evidence and to preserve pronouns inside that quote.
- User-directed review note:
  - The quality bar is logical correctness for another LLM consumer, not exact
    wording. Parser-tolerated Markdown-wrapped JSON is acceptable; generated
    digest text still must preserve speaker identity, Kazusa's own rows, and
    chronological facts.
- Prompt follow-up:
  - Removed new negative-prompt accretion and retained a positive row-reading
    procedure:
    `message_rows` order, visible `display_name`, assistant rows as
    `我（display_name）说：“text”`, and a final chronology sentence derived
    from `activity_labels.assistant_presence` plus message order.
- Deterministic verification:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
    passed.
  - `venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest.py`
    passed.
  - `venv\Scripts\python -m py_compile tests\test_reflection_cycle_group_scene_digest_live_llm.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py -q`
    passed: 14 passed.
- Latest real LLM verification was run one case at a time with trace
  inspection:
  - duplicate-thanks closed flow:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__duplicate_thanks_closed_flow__20260609T133414684269Z.json`.
  - quiet group without Kazusa:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__quiet_group_without_kazusa__20260609T133433659184Z.json`.
  - one Kazusa participation:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__one_kazusa_participation__20260609T133453143967Z.json`.
  - multiple Kazusa participations:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__multiple_kazusa_participations__20260609T133353682166Z.json`.
  - busy group latest Kazusa:
    `test_artifacts/llm_traces/reflection_cycle_group_scene_digest_live_llm__busy_group_latest_kazusa__20260609T133513701506Z.json`.
- Human-readable LLM review artifact updated:
  `test_artifacts/llm_reviews/group_scene_digest_explicit_participants_live_llm_review_20260610.md`.

### 2026-06-10 Stage 7 Final Inline Review

- Review mode: inline no-subagent review, per user instruction.
- Reviewed:
  - `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`;
  - `tests/test_reflection_cycle_group_scene_digest.py`;
  - `tests/test_reflection_cycle_group_scene_digest_live_llm.py`;
  - `src/kazusa_ai_chatbot/reflection_cycle/README.md`;
  - `src/kazusa_ai_chatbot/self_cognition/README.md`;
  - plan registry and this active bugfix plan;
  - latest LLM review artifact.
- Commands and checks:
  - `git diff --check`
    - Result: no whitespace errors; only line-ending warnings were reported.
  - `venv\Scripts\python -m pytest tests\test_reflection_cycle_group_scene_digest.py -q`
    - Result: 14 passed.
  - `venv\Scripts\python -m pytest tests\test_self_cognition_group_review_source.py -q`
    - Result: 13 passed.
  - `rg -n "speaker_ref|participant_[0-9]|active_character" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
    - Result: zero matches.
  - `rg -n "global_user_id|platform_user_id|platform_message_id|source_refs|delivery_target|adapter" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
    - Result: zero matches.
  - `rg -n "不要|禁止|不使用|不能|只要|必须|Markdown|代码块|纠错|多个版本" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
    - Result: only deterministic action-guidance rejection markers matched;
      the digest prompt itself no longer carries the new negative-constraint
      wording.
- Latest live LLM evidence:
  - Each case was run individually and inspected; trace paths are recorded in
    `Post-Review Regression And Prompt Simplification`.
  - The acceptance bar used for the latest review is logical coherence for the
    downstream cognition LLM, not exact prose.
- Findings:
  - No downstream cognition, dialog, action, dispatcher, adapter, persistence,
    RAG, or scheduler code changed.
  - Production prompt payload now exposes visible `display_name` and continues
    to exclude internal ids and adapter metadata.
  - The runtime output contract remains exactly `{"digest": str}`.
  - Oldest-row truncation is preserved through
    `window.participant_rows[-_GROUP_SCENE_DIGEST_ROW_LIMIT:]`.
  - The original logical failure is covered by the latest duplicate-thanks
    trace: Kazusa's own quote remains `能帮到你`, not
    `能帮到杏山千纱`.
  - Residual risk: local LLM sentence placement may vary. Markdown-wrapped JSON
    is handled by the parser; the meaningful risk is a normalized digest that
    is logically contradictory or loses required speaker/content facts.
  - Working tree contains an unrelated untracked
    `dialog_one_bubble_layout_contract_bugfix_plan.md`; it was not touched as
    part of this bugfix.
