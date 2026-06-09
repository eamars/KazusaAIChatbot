# reflection group scene digest self-cognition bugfix plan

## Summary

- Goal: add one reflection-owned group digest string to group self-cognition
  source packets so noisy group flow is easier for cognition to read.
- Plan class: medium
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: compatible optional source-packet field; omit the
  field when digest generation fails or returns an empty string.
- Highest-risk areas: the digest becoming hidden action guidance, a rich schema
  coming back, raw ids leaking into prompt context, or reflection latency
  growing beyond one selected group-review window.
- Acceptance criteria: group-review cases may include
  `conversation_progress.group_scene_digest = {"digest": str}` and no
  cognition, RAG, dialog, action-router, dispatcher, or persistence contract
  changes.

## Context

The QQ `905393941` failure was not that Kazusa could not see her own prior
message. The source window contained it. The failure was that
reflection-attached group self-cognition asked the cognition stack to rebuild a
noisy group flow from raw bounded rows plus coarse activity labels, and the
scene compressed into "continuous questions / sharing knowledge."

Current ownership stays unchanged:

- `reflection_cycle` selects and prepares monitored group windows.
- `self_cognition.sources` assembles `group_chat_review` source cases.
- `self_cognition.projection` already renders `conversation_progress`.
- cognition judges stance and action; dialog words visible output.
- RAG retrieves missing evidence and is not used for summarizing the current
  already-selected window.

## Mandatory Skills

- `development-plan`: load before plan lifecycle edits, execution, or review.
- `local-llm-architecture`: load before changing prompt-facing context,
  LLM prompt contracts, LLM prompt strings, or LLM budget.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files with CJK prompt text.
- `test-style-and-execution`: load before changing or running tests.

## Mandatory Rules

- Do not edit production code from this draft until the user explicitly
  approves implementation and the plan status is `approved` or `in_progress`.
- Use `venv\Scripts\python`; use `apply_patch`; check `git status --short`;
  do not read `.env`.
- The digest contract is exactly one generated string inside JSON:
  `{"digest": str}`.
- The digest must be written from the active character's first-person
  observational perspective because the consumer receives the rendered source
  packet as `internal_thought` / `internal_monologue` material. It must read
  like "我看到..." or "这段群聊里...我..." source observation, not like a
  system narrator, user speech addressed to the character, or a third-person
  story about the character.
- Do not add `status`, `confidence`, `thread_summaries`,
  `latest_visible_turn`, `active_character_participation`,
  `noise_and_limitations`, deterministic flow-state fields, or any other
  semantic subfields.
- The digest is source support only. It must not decide speak, silence,
  apology, retry, suppression, or "already answered" outcome.
- Do not expose raw global user ids, platform ids, source refs, adapter wire
  syntax, attachment URLs, or delivery target metadata to the digest prompt.
- The digest LLM prompt implementation must explicitly follow the repo LLM
  prompt rules from `local-llm-architecture`: stable contract in
  `SystemMessage`, current-window rows only in `HumanMessage`, triple-single
  quoted prompt constant, `.format(...)` only for process-stable values,
  runtime prompt-render verification, one language for ordinary instructions,
  no hard-coded concrete character name, no development-plan wording in the
  runtime prompt, no prompt input that asks the LLM to decide routing,
  delivery, persistence, permission, action feasibility, or adapter
  feasibility, and no retry loop after parsed-but-invalid JSON.
- After context compaction or major checklist sign-off, reread this plan before
  continuing. Before final completion, run the `Independent Code Review` gate.
- Execution is parent-led with native subagents unless the user explicitly
  approves fallback execution.

## Must Do

- Add a reflection-owned builder for selected 15-minute group activity windows.
- The builder returns `{"digest": non_empty_string}` or no digest.
- Attach the digest under `conversation_progress.group_scene_digest` for
  `group_chat_review` cases only when valid.
- Let existing source-packet rendering carry the digest; do not edit cognition
  or dialog prompts.
- Add deterministic tests for parsing, bounding, deidentification, attachment,
  and rendering.
- Add a duplicate-answer-shaped fixture whose expected digest is one neutral
  first-person observational string and not an action recommendation.

## Deferred

- No cognition, graph, L2d payload, dialog, dispatcher, delivery, persistence,
  adapter, scheduler, or action-attempt changes.
- No rich digest schema and no deterministic flow-label projection.
- No RAG call, background artifact reuse, DB collection, historical migration,
  retry loop, feature flag, model route, cooldown, response-ratio control,
  duplicate-text detector, silence gate, or "already answered" suppression.
- No required live LLM quality artifact unless the user explicitly asks.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Source packet | compatible | Add optional `conversation_progress.group_scene_digest = {"digest": str}`. |
| Group-review collection | bigbang | Selected group-review cases call the digest builder before cognition; invalid output is omitted. |
| Cognition/action/dialog | compatible | Preserve existing graph, prompt, payload, and routing contracts. |
| RAG/background/persistence | no-op | Add no new retrieval, worker, collection, or migration. |
| Tests | bigbang | Add focused digest and source-packet tests in the same implementation. |

## Target State

```text
selected group messages
  -> reflection_cycle.activity_windows.GroupActivityWindow
  -> reflection_cycle.group_scene_digest.build_group_scene_digest(...)
  -> self_cognition.sources conversation_progress.group_scene_digest
  -> existing self_cognition.projection rendering
  -> existing cognition graph unchanged
```

The model sees raw bounded rows, existing activity labels, and one neutral
digest string. The digest helps with scene reading; it does not replace raw
evidence or command an action.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Digest shape | `{"digest": str}` only | This is the user-requested contract. |
| Owner | reflection-cycle source preparation | It summarizes the selected monitored group window. |
| LLM route | existing `CONSOLIDATION_LLM_*` | Background summary work needs no new route. |
| Digest perspective | first-person observational source summary | Consumers frame the rendered packet as `internal_thought_residue.internal_monologue`, then L2a forms the character's actual first-person `internal_monologue`. |
| Invalid output | omit the digest | Avoid an unavailable/status schema. |
| Downstream use | prompt-facing context only | No deterministic branch may read digest meaning. |

## Contracts And Data Shapes

Prompt-facing source shape:

```python
{
    "digest": str,
}
```

The builder uses `kazusa_ai_chatbot.utils.parse_llm_json_output` once. The
parsed object is accepted only when it is a JSON object with exactly one string
field named `digest`. Missing field, empty string, non-string value, extra
keys, parser failure, or LLM exception means no digest is attached. Do not add
a second LLM repair call or retry loop.

The string must be bounded by a named constant, neutral, deidentified, and
written as a compact first-person observational scene-flow summary. It may
mention visible active-character participation and uncertainty from
media-only/noisy rows. It must not contain action advice or policy terms.

Allowed perspective examples:

- `这段群聊里，participant_1 问了一个词义问题，我随后已经解释过；后面只出现了图片/空内容，没有新的文字继续追问。`
- `我看到这段群聊先有人问我问题，我已经在窗口内接过一次；最后的可见变化是媒体/空内容，文字线索不足。`

Forbidden perspective examples:

- `The active character answered the question, so she should not reply again.`
- `系统判断这个问题已经解决。`
- `你刚刚已经回答过了，不要再说。`

Prompt payload rows use abstract refs such as `participant_1` and
`active_character`, preserve chronological order, and include only bounded text
plus compact activity labels.

## LLM Call And Context Budget

- Before: group review has no digest-generation LLM call.
- After: at most one background `CONSOLIDATION_LLM` call per selected
  group-review window, outside live `/chat`.
- Input: already-selected bounded visible rows and compact activity labels.
- Excluded: raw ids, source refs, delivery metadata, adapter wire text,
  attachment URLs, DB rows, RAG results, and raw hourly/daily reflection output.
- Prompt rules: follow `local-llm-architecture` prompt rules explicitly. Keep
  stable instructions in the system prompt, dynamic rows in the human payload,
  use triple-single quoted prompt constants, render with `.format(...)` only
  for process-stable values, avoid hard-coded concrete character names, avoid
  development-process vocabulary, and run prompt-render verification in
  addition to `py_compile`.
- Output: strict JSON with one `digest` string; no retry or repair loop after
  parsed-but-invalid JSON.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/group_scene_digest.py`
  - `GROUP_SCENE_DIGEST_MAX_CHARS`
  - digest system prompt
  - `build_group_scene_digest(window) -> dict[str, str] | None`
  - private prompt-payload and strict parse helpers
  - prompt-render verification helper or test-visible prompt payload builder

- `tests/test_reflection_cycle_group_scene_digest.py`
  - payload deidentification and order
  - valid one-string JSON acceptance and bounding
  - digest prompt requires first-person observational perspective and no
    hard-coded concrete character name
  - invalid/multi-field/action-guidance output omission
  - duplicate-answer-shaped neutral first-person digest fixture

### Modify

- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - import the builder
  - add optional `scene_digest_builder` test seam
  - attach valid digest to `conversation_progress.group_scene_digest`

- `tests/test_self_cognition_group_review_source.py`
  - assert fake one-string digest reaches the rendered source packet
  - assert raw ids and delivery metadata stay out

- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - document the digest as source hydration, not RAG evidence or action
    guidance.

### Keep

- Do not modify `self_cognition.projection`; existing `conversation_progress`
  rendering must carry the digest.
- Do not modify cognition, dialog, action router, dispatcher, RAG,
  persistence, adapter, or scheduler contracts.

## Overdesign Guardrail

- Actual problem: noisy group self-cognition can misread flow even when visible
  rows are present.
- Minimal change: one optional digest string under
  `conversation_progress.group_scene_digest`.
- Ownership boundaries: reflection summarizes source; self-cognition assembles
  case and target; cognition judges action; dialog words speech; deterministic
  code validates and bounds.
- Rejected complexity: rich schema, deterministic flow labels, perspective
  variants, prompt edits outside digest builder, RAG/background reuse,
  persistence, gates, retries, feature flags, new routes, and duplicate
  suppression.
- Evidence threshold: add structure later only after reviewed regressions prove
  the one-string digest is present, safe, and insufficient.

## Agent Autonomy Boundaries

- Private helper names may change; public builder and one-string contract may
  not.
- Do not add digest keys beyond `digest`.
- Do not make digest content a deterministic control input.
- If an equivalent group-window digest builder already exists, reuse it instead
  of duplicating behavior.

## Implementation Order

1. Add failing digest contract tests in
   `tests/test_reflection_cycle_group_scene_digest.py`.
2. Add failing source attachment/rendering tests in
   `tests/test_self_cognition_group_review_source.py`.
3. Start the production-code subagent with ownership limited to the files in
   `Change Surface`.
4. Implement `group_scene_digest.py` with deidentified prompt payload,
   first-person observational prompt contract, one-string JSON parsing,
   bounding, prompt-render verification coverage, and omission on invalid
   output.
5. Wire source attachment in `self_cognition.sources`.
6. Update the two README files.
7. Run focused tests, full verification, and record evidence.
8. Run independent code review; fix only in approved scope and rerun affected
   verification.

## Execution Model

- Parent owns test contract, verification, evidence, review coordination,
  lifecycle updates, and final sign-off.
- Production-code subagent owns production files only and closes after planned
  implementation.
- Independent code-review subagent reviews the approved plan, diff, and
  evidence; it does not implement fixes.
- If native subagents are unavailable, stop before production-code edits unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - test contract added.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_reflection_cycle_group_scene_digest.py -q`
    and
    `venv\Scripts\python -m pytest tests/test_self_cognition_group_review_source.py -q`.
  - Evidence: record expected pre-implementation failures.
  - Sign-off: `<agent/date>`.
- [ ] Stage 2 - digest builder and source attachment implemented.
  - Verify focused tests plus:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py`
    and
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\sources.py`.
  - Evidence: record changed files and focused test output.
  - Sign-off: `<agent/date>`.
- [ ] Stage 3 - full verification and independent code review complete.
  - Verify all commands in `Verification`; rerun affected commands after any
    review fix.
  - Evidence: record command summaries, review findings, fixes, and residual
    risks.
  - Sign-off: `<agent/date>`.

## Verification

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\sources.py
venv\Scripts\python -m pytest tests/test_reflection_cycle_group_scene_digest.py -q
venv\Scripts\python -m pytest tests/test_reflection_cycle_activity_windows.py -q
venv\Scripts\python -m pytest tests/test_self_cognition_group_review_source.py -q
venv\Scripts\python -m pytest tests/test_self_cognition_integration.py -q
rg -n "thread_summaries|latest_visible_turn|active_character_participation|noise_and_limitations|confidence" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\sources.py
rg -n "should_speak|should_stay_silent|action_recommendation|resolved issue|resolved_issue|suppress" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py tests\test_reflection_cycle_group_scene_digest.py
rg -n "background_work|ConversationEvidenceAgent|rag_evidence" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py src\kazusa_ai_chatbot\self_cognition\sources.py
rg -n "Kazusa|杏山千纱|development plan|bugfix plan|implementation plan" src\kazusa_ai_chatbot\reflection_cycle\group_scene_digest.py
```

The first `rg` command must return no production digest-schema matches. The
second may return only negative-test strings in tests. The final `rg` command
must return no production runtime-prompt matches.

## Independent Plan Review

Before approval or execution, review this draft for architecture alignment,
one-string contract enforcement, first-person observational perspective,
absence of hidden action guidance, explicit LLM prompt-rule coverage, exact
change surface, source-packet safety, and realistic LLM budget. Record blockers
or approval in `Execution Evidence`.

## Independent Code Review

After verification passes and before completion, review the approved plan,
full diff, and evidence. Blockers include richer digest shape, prompt-safety
leaks, non-first-person or action-guidance digest perspective,
cognition/RAG/dialog/action-router scope creep, missing prompt-render checks,
weak regression tests, or invalid-output behavior that does not omit the
digest. Record findings, fixes, reruns, and approval in `Execution Evidence`.

## Acceptance Criteria

- Group-review packets attach `conversation_progress.group_scene_digest` only
  as `{"digest": str}`.
- The digest is reflection-owned, bounded, prompt-safe, neutral, optional, and
  written as first-person observational source context.
- Existing group-review packets without a digest remain valid.
- No RAG, background-work, cognition, action-router, dialog, dispatcher,
  persistence, adapter, or scheduler change is introduced.
- Focused tests, prompt-render verification, static checks, verification
  commands, and independent code review pass.

## Execution Evidence

- Plan drafted: 2026-06-09.
- Initial draft was too broad because it used a rich scene schema.
- Overdesign correction: 2026-06-09. User clarified the goal is one string
  output inside JSON. Plan reduced to `{"digest": str}` and all rich fields,
  deterministic flow labels, unavailable/status shape, and live-LLM artifact
  requirements were removed.
- Perspective decision: 2026-06-09. Consumer-code review showed the rendered
  source packet enters cognition as `internal_thought` / `internal_monologue`
  material and later reaches L2d as `conversation_progress` evidence. The
  digest is therefore specified as first-person observational source context,
  not system narration, user speech, third-person story, or action guidance.
- Plan review after perspective update: 2026-06-09. Findings addressed:
  removed a concrete character-name example from the forbidden perspective
  examples, and clarified that the builder may use the repo JSON parser once
  but must not add a second LLM repair call or retry loop.
- Follow-up review adjustment: 2026-06-09. Clarified the exact JSON parser
  helper and made the hard-coded-name/runtime-plan-wording static check's
  expected result explicit.
- Independent plan review: pending.
- Implementation, verification, and independent code review: pending.
