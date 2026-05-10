# conversation progress identity leakage bugfix plan

## Summary

- Goal: stop short-term conversation progress and its immediate prompt inputs
  from calling the active character `助手`, `助理`, `assistant`, `角色`,
  `active_character`, or `当前角色` in generated operational state.
- Plan class: large
- Status: completed
- Mandatory skills: `superpowers:test-driven-development`,
  `local-llm-architecture`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `no-prepost-user-input`,
  `development-plan-writing`
- Overall cutover strategy: bigbang for future recorder/decontextualizer prompt
  behavior; compatible for stored conversation-progress rows and history rows.
- Highest-risk areas: adding deterministic blacklist repair, mutating raw chat
  history roles, weakening decontextualized input quality, broad prompt churn,
  appending prompt policy blocks without re-evaluating prompt flow, skipping
  real LLM validation for touched prompts, and breaking `assistant_moves`
  storage/schema compatibility.
- Acceptance criteria: RED evidence is recorded before any prompt or
  production-code edit; focused deterministic tests pass; every touched LLM
  prompt has one-by-one inspected pre-fix and post-fix positive and negative
  real LLM evidence; prompt logic-flow review is recorded for every touched
  prompt; any prompt change that adds a new instruction block is implemented as
  a holistic affected-prompt rewrite, not an appended block;
  future recorder outputs use exact `{character_name}` or subjectless
  role-neutral wording instead of generic active-character labels; raw storage
  roles remain unchanged.

## Context

Production logs showed the conversation progress recorder writing fields such
as:

```text
current_blocker: "助手拒绝接受用户已回答的事实，导致循环"
current_thread: "挑战第一题：用户已给出答案，助手未承认并继续催促"
next_affordances: ["助手承认答对并出第二题"]
```

Focused diagnosis found the structural recorder boundary is healthy: existing
deterministic tests for recorder/runtime/repository pass, and validator and
persistence code correctly preserve recorder free text. The failure is the
model-facing identity contract, not storage shape corruption.

Previous completed work fixed this class of issue for durable memory writers
through the identity-free memory output contract. That plan intentionally did
not rename `role="assistant"` or `assistant_moves`, and it did not cover the
conversation-progress recorder. This bugfix applies the same positive prompt
contract principle to the short-term progress path without migrating existing
stored progress rows.

This plan is TDD-gated. Before any production code, prompt text, or prompt
payload is changed, the implementation agent must prove the current issue is
persistent or possible:

- persistent: a pre-fix real LLM recorder or decontextualizer case emits a
  generic active-character label such as `助手`, `角色`, or `当前角色` where it
  means the active character;
- possible: a pre-fix deterministic contract test fails because the current
  prompt/payload exposes generic active-character labels, omits the exact
  runtime character name, or exposes raw `role="assistant"` to a stage that
  generates persisted progress prose.

The fix may start only after that RED evidence and the pre-fix real LLM
baseline traces are recorded in `Execution Evidence`.

## Prompt Leakage Inventory

This scan covered source prompt constants under `src/kazusa_ai_chatbot` and
`src/scripts`.

| Risk | Prompt/source | Finding | Decision |
|---|---|---|---|
| High | `conversation_progress/recorder.py::_RECORDER_PROMPT` (`recorder.py:35-65`, `recorder.py:103-134`) | The prompt has no `{character_name}` input, starts from `当前角色`, says `assistant_moves` are "本轮 assistant", and describes `next_affordances` as `当前角色` actions. Its output is persisted as next-turn operational state. | Fix in this plan. |
| High | `brain_service/post_turn.py::run_conversation_progress_record_background` (`post_turn.py:125-170`) | The function has `character_profile` but does not pass the character name into `ConversationProgressRecordInput`. It passes raw `chat_history_recent`, `content_anchors`, and `final_dialog` to the recorder. | Fix in this plan. |
| High | `conversation_progress/recorder.py` human payload (`recorder.py:180-184`) | `chat_history_recent`, `content_anchors`, `final_dialog`, and prior state can contain machine-role or generic active-character labels that the recorder may copy into generated state. | Add recorder-owned prompt projection and prompt instructions. |
| Medium | `persona_supervisor2_msg_decontexualizer.py::_MSG_DECONTEXUALIZER_PROMPT` (`persona_supervisor2_msg_decontexualizer.py:26-125`) | Prompt says "当前角色是当前助手/角色" and includes an example output with "当前角色". Its output feeds cognition, RAG, recorder, and consolidation. | Fix active-character wording and add `character_name` input in this plan. |
| Medium | Shared prompt-history projections (`time_context.py:141`, `rag/prompt_projection.py:279`) | Existing helpers preserve raw message fields, including `role="assistant"`. That is correct for auditability, but it means any consumer prompt that needs identity-safe prose must own a prompt-safe projection. | Do not change shared projection helpers. Add only recorder-local prompt projection. |
| Medium | `persona_supervisor2_cognition_l3.py` prompts (`persona_supervisor2_cognition_l3.py:214`, `383`, `563`, `803`, `983`) | L3 prompts use `角色 {character_name}` and can emit `content_anchors` consumed by the recorder. They already receive `{character_name}` and contain much broader cognition behavior. | Do not touch initially; recorder must not copy generic labels from inputs. Add targeted regression only if recorder tests reveal L3 anchor leakage after the recorder fix. |
| Medium | `dialog_agent.py::_DIALOG_GENERATOR_PROMPT` (`dialog_agent.py:112`, `dialog_agent.py:211`) | `tone_history` description mentions prior `assistant` reply. Output is user-facing `final_dialog` and then recorder evidence. Prompt already renders `{character_name}`. | Do not touch; covered by recorder input handling unless live tests prove dialog emits generic labels. |
| Low | RAG initializer and selectors (`persona_supervisor2_rag_initializer.py:52-173`, `232-334`) | Use `active_character` as a semantic retrieval slot, such as `speaker=active_character`. This is a structured query contract, not generated memory/progress prose. | Keep unchanged. |
| Low | Reflection hourly/daily prompts (`reflection_cycle/prompts.py:46-76`) | Use `active_character` and `role=user|assistant` inside read-only artifacts. Raw reflection output does not enter normal cognition directly; promotion has a separate safe contract. | Keep unchanged in this bugfix. |
| Low | Memory writers and migration prompt | Still mention `assistant` as a forbidden machine label but already render `{character_name}` and contain explicit "do not copy machine labels" contracts. | Keep unchanged; rely on completed identity-free memory tests. |

## Mandatory Skills

- `superpowers:test-driven-development`: load before adding tests or changing
  production code. No prompt or production-code edit is allowed until a focused
  failing test or equivalent RED diagnostic proves the current bug is
  persistent or possible.
- `local-llm-architecture`: load before changing any prompt, LLM payload,
  graph, cognition, dialog, recorder, RAG, reflection, or background LLM
  behavior. Use it to keep prompt vocabulary grounded in visible inputs and to
  avoid hidden schema inference.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing Chinese/Japanese
  prompt strings; run syntax checks after edits.
- `test-style-and-execution`: load before adding, changing, or running tests.
  Real LLM tests must run one case at a time and be inspected one case at a
  time.
- `no-prepost-user-input`: load before adding any logic around
  `decontexualized_input`, commitments, accepted preferences, or user-command
  interpretation. Fix LLM behavior through prompt/schema contracts, not local
  semantic filters.
- `development-plan-writing`: load before updating this plan, lifecycle status,
  or execution evidence.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run this plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Do not add production blacklist validation, forbidden-term scanners,
  post-generation rejection, retry loops, or deterministic text substitution to
  remove `助手`, `助理`, `assistant`, `角色`, `active_character`, or `当前角色`
  from LLM output.
- Do not mutate raw source data. Raw `conversation_history`,
  `role="assistant"`, `chat_history_recent`, `decontexualized_input`,
  `content_anchors`, `final_dialog`, and stored episode rows remain auditable.
- Structural prompt projection is allowed only for model-facing copies, such as
  replacing prompt-visible speaker metadata with `speaker_name` while preserving
  message text unchanged.
- Do not rename the public or stored `assistant_moves` key. Its values should
  be compact speech-act labels; the field name remains a storage/schema term.
- Do not hard-code a concrete character name in reusable prompt strings.
  Runtime character identity must come from `character_profile["name"]`.
- Every touched LLM prompt must have real LLM validation with both a
  false-negative and a false-positive case, run and inspected one by one.
- Every touched LLM prompt must have both positive and negative real LLM tests.
  For this plan, positive means the leak-triggering case is mitigated; negative
  means legitimate non-leak text such as user-owned `助手` terms or direct
  second-person wording is preserved.
- All Python edits containing CJK prompt text must be followed by `py_compile`
  or an AST parse check.
- TDD is mandatory. The implementation agent must write or update the focused
  deterministic and real LLM tests before touching prompt text, prompt payload
  code, model-facing projection code, or service wiring.
- The first run of each new deterministic contract test must happen before the
  corresponding production edit. The expected result is a failure for the
  intended reason, such as missing `character_name`, prompt-visible generic
  active-character labels, or raw assistant-role leakage into recorder prompt
  context.
- If a new deterministic test passes before implementation, it is not valid
  RED evidence. The agent must correct the test until it fails for the intended
  pre-fix defect before changing production code.
- Real LLM prompt tests must be added before prompt edits and run one by one
  against the pre-fix prompt. If the exact production wording does not
  reproduce, the trace must still be recorded as baseline evidence, and the
  deterministic RED tests must prove the leak is possible before implementation
  may proceed.
- Every prompt or production-code edit must map to a preceding RED evidence
  item in `Execution Evidence`; edits without RED evidence must be reverted and
  redone through the test-first sequence.
- Any prompt change requires a prompt logic-flow review before editing. The
  review must document the current role, generation procedure, input format,
  output contract, examples, parser assumptions, and downstream consumer before
  choosing the exact wording change.
- Do not append new standalone instruction blocks to existing prompts. If the
  implementation needs an additional prompt block, section, policy list, or
  example cluster, rewrite the entire affected prompt so the new rule is
  integrated into the prompt's role, procedure, input/output format, and
  examples. This rewrite is limited to the affected prompt; it does not
  authorize unrelated prompt rewrites.
- Simple prompt substitutions, such as replacing a generic actor label with
  `{character_name}`, still require the logic-flow review and real LLM positive
  and negative tests, but they do not require a full affected-prompt rewrite
  unless they add a new block.

## Must Do

- Thread the exact active character name from `character_profile["name"]` into
  the decontextualizer and conversation-progress recorder prompt payloads.
- Establish pre-fix RED evidence before changing any prompt or production code:
  failing deterministic contract tests are mandatory, and pre-fix real LLM
  baseline traces are mandatory for every prompt that will be touched.
- Before changing each prompt, record a prompt logic-flow review in `Execution
  Evidence`. If the change adds any new prompt block, rewrite the full affected
  prompt instead of appending the block.
- Update the decontextualizer prompt so active-character rewrites use the exact
  runtime character name when a name is needed, or preserve direct second
  person when the sentence is already clear. It must not generate `当前角色` or
  call the active character `助手`.
- Update the recorder prompt so generated free-text fields use the exact
  runtime character name only when the active character must be named, otherwise
  use subjectless role-neutral operational labels. It must explicitly reject
  generic active-character labels in generated state.
- Add recorder-owned prompt projection for recent history so the recorder sees
  prompt-safe speaker metadata while preserving source message text.
- Add deterministic tests for the new payload shape, prompt rendering, and
  storage/schema compatibility.
- Add real LLM tests for the decontextualizer and recorder prompt contracts.
- Run every verification command in this plan and record evidence before
  marking the plan complete.

## Deferred

- Do not migrate, rewrite, or bulk-sanitize existing stored
  `conversation_episode_state` documents.
- Do not change storage roles, adapter roles, message-envelope roles,
  conversation-history document shape, or `assistant_moves`.
- Do not change L3 content-anchor, dialog, RAG, reflection, or memory-writer
  prompts unless a focused post-recorder test proves a touched prompt still
  emits generic active-character labels into the progress path.
- Do not add a shared global prompt-contract helper. Prefer stage-local prompt
  wording unless repeated structural projection becomes unavoidable and is
  justified in this plan before implementation.
- Do not add extra response-path LLM calls.

## Cutover Policy

Overall strategy: bigbang for future affected prompt behavior.

| Area | Policy | Instruction |
|---|---|---|
| Decontextualizer prompt behavior | bigbang | Replace generic active-character wording with `{character_name}`-aware wording. No legacy prompt path. |
| Recorder prompt behavior | bigbang | Replace generic active-character output instructions with a positive identity-safe contract. No fallback prompt. |
| Recorder payload shape | compatible | Add prompt-facing speaker metadata and `character_name` without changing raw source rows or stored episode rows. |
| Stored progress rows | compatible | Leave old rows readable. Do not classify or rewrite existing free-text content. |
| Public schema/storage keys | compatible | Preserve `assistant_moves`, conversation history `role`, and existing facade names. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a compatibility shim, dual prompt path, or fallback
  path for bigbang areas.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The target ownership boundary is the short-term conversation-progress path
  and its immediate prompt input source in the decontextualizer.
- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, semantic filters, or extra
  features.
- Changes outside the target boundary require strong evidence that the recorder
  and decontextualizer fixes cannot solve the leak. Record that evidence in
  `Execution Evidence` before expanding the change surface.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If any required real LLM case cannot run because the endpoint is unavailable,
  stop and report the blocker. Do not mark prompt behavior complete from
  deterministic tests alone.

## Target State

Future conversation-progress writes may still use compact operational labels,
but they must not call the active character `助手`, `助理`, `assistant`,
`角色`, `active_character`, or `当前角色` when the text is referring to the
active character. Acceptable generated state looks like:

```json
{
  "current_thread": "潜水问答挑战第一题：用户已答出瓦尔萨尔瓦动作但尚未被承认",
  "assistant_moves": ["拒绝承认用户已回答", "维持挑衅节奏"],
  "next_affordances": ["承认答对并进入第二题"]
}
```

If a free-text field must name the active character, it must use the exact
`character_profile["name"]`. If the term `助手` appears as a legitimate user
project/app/name, such as `学习助手`, it may be preserved as user-owned content.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix mechanism | Prompt/schema contract plus prompt-facing projection | The issue is semantic LLM output; production blacklist repair would hide the root cause and violate LLM-first architecture. |
| Prompt edit discipline | Re-evaluate the logic flow of every touched prompt before editing; rewrite the full affected prompt if adding a new block | Local LLM prompts drift when policy blocks are appended without integrating role, procedure, format, and examples. |
| Character identity source | `character_profile["name"]` | Matches existing cognition and durable-memory prompt contracts. |
| Recorder history view | Stage-local prompt projection | The recorder needs speaker ownership without seeing machine role labels as actor names. Raw history remains unchanged. |
| Existing rows | No migration | Conversation progress is short-term and expires. Old strings should not be rewritten by heuristics. |
| `assistant_moves` field | Preserve field key | It is a schema/storage label, not generated active-character prose. |
| Real LLM validation | Mandatory for recorder and decontextualizer | Both prompts directly influence generated text quality. Static checks are insufficient. |

## Contracts And Data Shapes

### Decontextualizer Input

Add the exact active character name to the decontextualizer human payload:

```python
{
    "character_name": state["character_profile"]["name"],
    ...
}
```

The output schema remains unchanged:

```python
{
    "output": str,
    "is_modified": bool,
    "reasoning": str,
    "referents": list[dict],
}
```

### Recorder Input

Extend `ConversationProgressRecordInput` with:

```python
character_name: str
```

The recorder prompt payload must include:

```python
{
    "character_name": "exact profile name",
    "chat_history_recent": [
        {
            "speaker_name": "user display name or exact character name",
            "speaker_kind": "user | character | other",
            "body_text": "message text",
            "timestamp": "optional local timestamp"
        }
    ],
    ...
}
```

Raw source row text must be preserved. Speaker projection must not summarize,
translate, filter, classify, or rewrite message bodies.

### Recorder Output

The existing output schema remains unchanged. Free-text values must follow this
contract:

- use compact operational labels;
- use exact `{character_name}` only when naming the active character is needed;
- do not use generic active-character labels;
- preserve user-owned names containing terms such as `助手` when the term is
  not being used as a substitute for the active character.

## LLM Call And Context Budget

| LLM call | Before | After | Path | Budget impact |
|---|---:|---:|---|---|
| Decontextualizer | 1 | 1 | response path | Adds one short `character_name` field and compact prompt wording. No new call. |
| Conversation progress recorder | 1 | 1 | post-response background | Adds one short `character_name` field and projected speaker metadata. No new call. |

Default context cap is 50k tokens. The recorder progress payload remains under
existing caps; prompt-facing history projection must not add message bodies or
increase history count beyond the current call.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Render prompt with `character_name`.
  - Add `character_name` to the human payload.
  - Replace generic active-character wording and examples.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Add `character_name` to `ConversationProgressRecordInput`.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Render `_RECORDER_PROMPT` with `character_name`.
  - Add recorder-owned prompt history projection.
  - Update prompt contract and render tests.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Pass `character_profile["name"]` into recorder input.
- `tests/test_conversation_progress_recorder.py`
  - Add deterministic payload/projection/prompt tests.
- `tests/test_conversation_progress_runtime.py`
  - Update fixtures for `character_name`.
- `tests/test_service_background_consolidation.py`
  - Update service/background recorder fixtures that instantiate or inspect
    `ConversationProgressRecordInput`.
- `tests/test_decontexualizer_referents.py`
  - Add
    `test_decontexualizer_prompt_requires_character_name_and_identity_safe_examples`.
- `tests/test_conversation_progress_recorder_live_llm.py`
  - Create focused real LLM recorder contract tests.
- `tests/test_decontexualizer_live_llm.py`
  - Add focused real LLM decontextualizer contract tests.

### Keep

- `conversation_history.role` and raw history rows.
- `assistant_moves` public/storage key.
- L3 content-anchor, dialog, RAG, reflection, and memory-writer prompts unless
  post-fix focused evidence proves the recorder still receives generated
  generic labels from one of those stages.

### Create

- No new production module is required. A small stage-local helper inside
  `conversation_progress/recorder.py` is allowed for prompt-facing history
  projection.

## Implementation Order

No implementation step may edit production code or prompt text until steps 1
and 2 have recorded RED evidence, and step 3 has recorded pre-fix real LLM
baseline traces, in `Execution Evidence`.

1. Add deterministic decontextualizer prompt tests.
   - Verify before implementation:
     `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py::test_decontexualizer_prompt_requires_character_name_and_identity_safe_examples -q`
   - Expected before implementation: fails for the intended RED reason because
     `character_name` is absent from the decontextualizer prompt payload or the
     rendered prompt still contains generic active-character wording/example
     output.
   - If the test passes before implementation, correct the test and rerun it
     until it fails for the intended pre-fix defect.
2. Add deterministic recorder prompt and payload tests.
   - Verify before implementation:
     `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py::test_recorder_prompt_requires_character_name_and_prompt_safe_history_projection -q`
   - Expected before implementation: fails because recorder input lacks
     `character_name` or history still exposes `role="assistant"` without
     prompt-safe speaker metadata.
   - If the test passes before implementation, correct the test and rerun it
     until it fails for the intended pre-fix defect.
3. Add real LLM tests for decontextualizer and recorder before prompt edits.
   - Run each individual new live test against the pre-fix prompt with `-q -s`
     and inspect its trace artifact before continuing.
   - Record whether each trace proves the issue is persistent or only provides
     non-reproducing baseline evidence.
   - If the current model does not reproduce exact `助手`, keep the test as a
     future contract gate and rely on the deterministic RED tests from steps 1
     and 2 as the proof that the leak is possible.
4. Record prompt logic-flow review for each prompt that will be touched.
   - For `persona_supervisor2_msg_decontexualizer.py`, record current role,
     generation procedure, input format, output contract, examples, parser
     assumptions, and downstream consumers before editing.
   - For `conversation_progress/recorder.py`, record current role, generation
     procedure, input format, output contract, examples, parser assumptions,
     and downstream consumers before editing.
   - If a change adds a new prompt block, rewrite the full affected prompt
     before implementation proceeds. Do not append the new block.
   - Record the review and rewrite decision in `Execution Evidence`.
5. Implement decontextualizer prompt and payload changes.
   - Implement only enough production code/prompt change to make the
     decontextualizer RED tests pass.
   - Re-run the exact RED deterministic test from step 1 and record GREEN
     evidence before touching recorder production code.
6. Implement recorder input, prompt render, and history projection changes.
   - Implement only enough production code/prompt change to make the recorder
     RED tests pass.
   - Re-run the exact RED deterministic test from step 2 and record GREEN
     evidence before broadening verification.
7. Wire service background recorder input.
8. Update deterministic fixtures and run focused deterministic tests.
9. Re-run each real LLM test one by one against the fixed prompt, inspect
   artifacts, compare to the pre-fix baseline, and record mitigation judgment.
10. Run static prompt greps and syntax checks.
11. Run independent code review and remediate findings inside approved scope.

## Progress Checklist

- [x] Stage 1 - deterministic RED contract tests recorded
  - Covers: implementation steps 1 and 2.
  - Verify: focused new deterministic tests are run before production edits and
    fail for the intended defect. Passing-before-implementation tests do not
    satisfy this stage.
  - Evidence: record test names, commands, failure output, and why each failure
    proves the issue is possible in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 2 - pre-fix real LLM baseline recorded
  - Covers: implementation step 3.
  - Verify: false-negative and false-positive tests for each touched prompt are
    run individually with `-q -s` before prompt edits; every trace artifact is
    inspected.
  - Evidence: record artifact paths and whether each case proves persistence,
    non-reproduction, or only baseline behavior. Do not proceed unless Stage 1
    already proves the leak is possible.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 3 - prompt logic-flow review recorded
  - Covers: implementation step 4.
  - Verify: `Execution Evidence` records the current logic flow and rewrite
    decision for every prompt that will be edited.
  - Evidence: record role, generation procedure, input format, output
    contract, examples, parser assumptions, downstream consumers, and whether
    a full affected-prompt rewrite is required.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 4 - decontextualizer identity contract implemented
  - Covers: implementation step 5.
  - Verify: the exact decontextualizer RED deterministic test from Stage 1 now
    passes, then focused decontextualizer deterministic tests pass.
  - Evidence: record changed files, RED-to-GREEN command output, and CJK
    syntax/render checks.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 5 - recorder identity contract implemented
  - Covers: implementation steps 6 and 7.
  - Verify: the exact recorder RED deterministic test from Stage 1 now passes,
    then focused recorder/runtime/service deterministic tests pass.
  - Evidence: record changed files, RED-to-GREEN command output, and CJK
    syntax/render checks.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 6 - real LLM validation passed
  - Covers: implementation step 9.
  - Verify: re-run each real LLM test individually against the fixed prompts
    and inspect each artifact.
  - Evidence: record pass/fail judgment, trace paths, and explicit comparison
    to the pre-fix baseline proving mitigation. For each touched prompt,
    record the positive and negative case judgments separately.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 7 - regression and static checks complete
  - Covers: static greps, syntax checks, focused deterministic regression.
  - Verify: all commands in `Verification` pass.
  - Evidence: record command output.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.
- [x] Stage 8 - independent code review complete
  - Covers: final review gate.
  - Verify: review findings are fixed or recorded as residual risk.
  - Evidence: record review result and any reruns.
  - Handoff: plan can move to completed only after acceptance criteria pass.
  - Sign-off: `Codex/2026-05-10` after evidence is recorded.

## Verification

### Static Greps

- `rg -n "当前助手/角色|我是想让当前角色|助手/角色" src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Expected: no matches.
- `rg -n "当前角色|助手|助理|active_character" src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Expected: no matches in generated-output instructions. If `当前角色`
    remains only in comments or non-output context, record and justify it.
- `rg -n "assistant 回复|本轮 assistant|紧凑 assistant" src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Expected: no matches. The key name `assistant_moves` may remain.
- `rg -n "role.*assistant|assistant.*role" src/kazusa_ai_chatbot/conversation_progress src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Expected: matches only in schema/storage field names or tests explicitly
    documenting raw role compatibility.

### Syntax And Render Checks

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\conversation_progress\models.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\brain_service\post_turn.py`
- Prompt logic-flow review evidence must exist for every touched prompt before
  syntax/render checks are accepted.
- If a prompt edit added a new block, `Execution Evidence` must state that the
  full affected prompt was rewritten and list the reviewed prompt sections.
- Runtime prompt render checks for decontextualizer and recorder must prove:
  - `{character_name}` is rendered with `测试角色`;
  - no literal `{character_name}` placeholder remains in the final system
    prompt;
  - JSON examples remain valid prompt text and do not break `.format(...)`.

### Deterministic Tests

- `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_conversation_episode_state.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_memory_writer_prompt_contracts.py tests\test_memory_writer_prompt_projection.py -q`
  - Purpose: confirm this bugfix did not regress the completed durable-memory
    perspective contract.

### Real LLM Tests

Run each command individually with `-q -s`, inspect its trace artifact before
running the next command, and record a judgment. These commands must be run
twice: once before prompt edits to establish pre-fix baseline evidence, and
again after the fix to prove mitigation. If a pre-fix live case passes, keep
the trace as baseline evidence but do not treat it as RED proof; Stage 1
deterministic failures remain the required proof that the issue is possible.
Every touched prompt must have positive and negative real LLM coverage before
and after the prompt change. The two decontextualizer commands below are the
positive and negative cases for that prompt. The two recorder commands below
are the positive and negative cases for that prompt.

- `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_active_character_short_answer_uses_character_name -q -s`
- `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_direct_second_person_preserved -q -s`
- `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_does_not_use_generic_active_character_labels -q -s`
- `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_preserves_user_owned_helper_term -q -s`

Each real LLM trace must include rendered prompt summary, input payload, raw
model output, parsed output, forbidden generic-label assessment, exact
`character_name`, and human/agent judgment.

## Independent Plan Review

This review gate has been run for approval. Future plan changes that alter
scope, verification, cutover policy, TDD gates, or LLM validation requirements
must run this gate again before the plan remains approved.

Review scope:

- TDD is embedded as a hard precondition before prompt or production-code
  edits.
- The plan proves the current issue is persistent or possible before the fix.
- Every planned prompt edit has deterministic RED evidence and real LLM
  baseline evidence before implementation.
- Every prompt edit requires logic-flow review before editing; any added prompt
  block requires a full affected-prompt rewrite rather than an appended block.
- Every touched prompt has explicit positive and negative real LLM coverage.
- The fix verification proves mitigation with the same focused tests and
  post-fix real LLM traces.
- Scope remains bounded to conversation progress and the immediate
  decontextualizer source.
- No production blacklist filtering, output rewrite, retry loop, data
  migration, raw-history mutation, or storage-role rename is authorized.

Review result:

- Blockers found: the earlier draft allowed prompt implementation after adding
  tests but did not explicitly require RED evidence before prompt edits.
- Resolution: implementation order, mandatory rules, checklist stages, and
  real LLM verification now require pre-fix RED evidence and baseline traces
  before production edits.
- Non-blocking finding: no separate reviewer was available in this session, so
  the active agent performed a fresh plan review against the development-plan,
  TDD, test-execution, and local-LLM architecture rules.
- Approval status: approved for implementation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- CJK prompt safety, prompt rendering, and `.format(...)` brace correctness.
- Prompt logic-flow review evidence exists for every touched prompt, and any
  added prompt block was implemented through a full affected-prompt rewrite
  rather than an appended block.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, implementation order, verification gates, and acceptance criteria.
- Absence of production blacklist filtering, semantic post-processing,
  migration, fallback prompts, extra LLM calls, and raw history mutation.
- Real LLM evidence quality: every touched LLM prompt has false-negative and
  false-positive traces, run and inspected one by one.

Fix concrete findings directly only when the fix is inside the approved change
surface. If a finding requires a broader prompt, graph, storage, or migration
change, stop and update the plan or request approval before changing code.

Record findings, fixes, reruns, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Decontextualizer and recorder prompts receive the exact runtime
  `character_profile["name"]`.
- Prompt logic-flow review is recorded before every prompt edit. If a prompt
  change adds a block, the full affected prompt is rewritten and the rewrite
  decision is recorded.
- Future recorder-generated free-text progress fields do not use generic
  active-character labels as substitutes for the active character.
- User-owned terms such as a project named `学习助手` are not erased by
  deterministic filtering.
- `assistant_moves` remains the public/storage key and raw history
  `role="assistant"` remains unchanged.
- No production blacklist scan, post-generation rejection, output rewrite, or
  migration is introduced.
- Every production code or prompt edit maps to preceding RED evidence recorded
  in `Execution Evidence`.
- Pre-fix real LLM baselines and post-fix real LLM mitigation traces are run
  individually and inspected for every touched prompt, with positive and
  negative case judgments recorded separately.
- Focused deterministic tests, static greps, syntax/render checks, and real LLM
  tests all pass with execution evidence.
- Independent code review is completed and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| LLM still copies generic labels from prior state | Positive recorder contract plus real LLM false-negative case with polluted prior state | Recorder live LLM trace |
| Legitimate user text containing `助手` is lost | No production blacklist; false-positive case preserves user-owned helper term | Recorder false-positive live test |
| Decontextualizer over-names direct second-person turns | Prompt says preserve direct `你` when already clear | Decontextualizer false-positive live test |
| Prompt edit breaks `.format(...)` braces | Runtime render check and `py_compile` | Syntax/render verification |
| Added prompt blocks create contradictory flow | Mandatory prompt logic-flow review and full affected-prompt rewrite for block additions | Prompt-flow evidence and independent code review |
| Scope expands into unrelated cognition/dialog/RAG prompts | Change surface and deferred sections forbid expansion without evidence | Independent code review |

## Execution Evidence

- Draft created after source prompt scan. No implementation has been performed.
- 2026-05-10 plan tightened for mandatory TDD. Production code and prompt files
  remain untouched in this planning pass.
- 2026-05-10 individual plan review completed by the active agent against the
  development-plan contract, execution gates, TDD skill, test-execution skill,
  and local-LLM architecture skill. Blocker resolved: RED evidence is now a
  hard gate before prompt or production-code edits. Residual risk: no
  implementation or test execution has been performed yet. Approval status:
  approved for implementation.
- 2026-05-10 review checks passed: all required executable-plan sections are
  present; reserved-marker and open-choice wording scan returned no findings;
  working-tree scope is limited to this plan and `development_plans/README.md`.
- 2026-05-10 user follow-up incorporated: every prompt edit now requires
  prompt logic-flow review; adding any prompt block requires rewriting the full
  affected prompt instead of appending; every touched prompt requires positive
  and negative real LLM tests before and after the prompt change. Production
  code and prompt files remain untouched in this planning pass.
- 2026-05-10 review rerun after follow-up: required-section check passed,
  reserved-marker/open-choice scan returned no findings, registry still marks
  the plan approved and ready, and working-tree scope remains limited to this
  plan plus `development_plans/README.md`.
- 2026-05-10 branch execution started on
  `bugfix/conversation-progress-identity-leakage`; plan status changed to
  `in_progress` in this file and the registry.
- 2026-05-10 Stage 1 RED evidence recorded:
  `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py::test_decontexualizer_prompt_requires_character_name_and_identity_safe_examples -q`
  failed for the intended reason after fixing a test-only missing import:
  `human_payload.get("character_name")` was `None` instead of `测试角色`.
  This proves the decontextualizer model-facing payload does not expose the
  exact active character name before implementation.
- 2026-05-10 Stage 1 RED evidence recorded:
  `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py::test_recorder_prompt_requires_character_name_and_prompt_safe_history_projection -q`
  failed for the intended reason:
  `human_payload.get("character_name")` was `None` instead of `测试角色`.
  The captured payload also still contained raw
  `chat_history_recent[0]["role"] == "assistant"`, proving recorder prompt
  identity leakage is possible before implementation.
- 2026-05-10 Stage 2 pre-fix live LLM baseline recorded:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_active_character_short_answer_uses_character_name -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__active_character_short_answer_uses_character_name.json`.
  Judgment: non-reproducing positive baseline; output used `杏山千纱`, but the
  trace shows the pre-fix prompt still contains `当前角色` and `助手`, and the
  handler payload still omits `character_name`.
- 2026-05-10 Stage 2 pre-fix live LLM baseline recorded:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_direct_second_person_preserved -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__direct_second_person_preserved.json`.
  Judgment: negative baseline passed; direct `你也不反对吧` was preserved.
- 2026-05-10 Stage 2 pre-fix live LLM baseline recorded:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_does_not_use_generic_active_character_labels -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__does_not_use_generic_active_character_labels.json`.
  Judgment: non-reproducing positive baseline; output did not leak generic
  labels, but the trace shows the pre-fix prompt still contains `当前角色`, the
  payload still omits `character_name`, raw `role="assistant"` remains visible,
  and polluted prior state included `助手`.
- 2026-05-10 Stage 2 pre-fix live LLM baseline recorded:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_preserves_user_owned_helper_term -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__preserves_user_owned_helper_term.json`.
  Judgment: negative baseline passed; user-owned `学习助手` was preserved while
  no generic active-character label appeared in output.
- 2026-05-10 Stage 3 prompt logic-flow review recorded for
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`.
  Current role: rewrite only the current user message into an equivalent,
  context-independent sentence and emit only referents that affect
  understanding. Generation procedure: identify address relationship, split
  direct speech / reported speech / short answers / ordinary pronouns, read
  prompt message context, reply context, recent history, indirect speech, and
  channel topic by evidence strength, choose keep / resolve / unresolved per
  fragment, compose `output`, then enforce referent consistency and group
  addressed second-person handling. Current input format: `user_input`,
  `platform_user_id`, `user_name`, `platform_bot_id`,
  `prompt_message_context`, raw `chat_history`, `channel_topic`,
  `indirect_speech_context`, and `reply_context`; it does not expose the exact
  active `character_name`. Output contract: strict JSON with `output`,
  `is_modified`, `reasoning`, and `referents`; parser fallback preserves the
  original user input and normalizes only referent structure. Current examples
  and wording include unsafe active-character terms: `当前助手/角色`,
  `当前角色`, and a short-answer example that outputs `当前角色说明白`.
  Downstream consumers: cognition, RAG, dialog planning evidence,
  conversation-progress recorder, and consolidation. Rewrite decision: no new
  standalone prompt block is needed; integrate `character_name` into the
  existing role/procedure/input/examples, replace generic active-character
  wording with the rendered runtime name or direct second person, and keep the
  output schema unchanged.
- 2026-05-10 Stage 3 prompt logic-flow review recorded for
  `src/kazusa_ai_chatbot/conversation_progress/recorder.py`. Current role:
  summarize the completed turn into short-term operational state that is read
  by the next turn. Generation procedure: read timestamp, prior episode state,
  recent history, decontextualized input, anchors, stance, intent, final
  dialog, and boundary profile; determine continuity/status/mode/phase/topic
  momentum; write only current or positively reconfirmed open loops; inspect
  prior state for still-useful non-temporal items; remove unanchored relative
  time pressure; return strict JSON only. Current input format:
  `current_turn_timestamp`, `prior_episode_state`, `decontexualized_input`,
  raw `chat_history_recent`, `content_anchors`, `logical_stance`,
  `character_intent`, `final_dialog`, and `character_boundary_profile`; it
  lacks `character_name` and exposes raw role metadata such as
  `role="assistant"`. Output contract: strict JSON with continuity/status
  enums plus free-text operational fields; `validate_recorder_output`
  validates enum/string/list structure but intentionally performs no semantic
  blacklist filtering. Current prompt examples and field descriptions use
  unsafe generated-output guidance: `当前角色`, `本轮 assistant`, `紧凑
  assistant`, and `当前角色下一轮`. Downstream consumers: recorder runtime,
  repository/cache persistence, and next-turn prompt context. Rewrite
  decision: no new production sanitizer or retry is allowed; implement a
  recorder-owned prompt-facing history projection that preserves message text
  while replacing prompt-visible speaker metadata, add `character_name` to the
  prompt payload, and integrate identity-safe wording into the existing role,
  input, field-writing, generation, and output-format sections. If the
  implementation needs more than these integrated substitutions, the affected
  recorder prompt must be rewritten holistically rather than extended by an
  appended policy block.
- 2026-05-10 Stage 4 decontextualizer implementation completed. Changed
  files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  and `tests/test_decontexualizer_referents.py`. The decontextualizer prompt
  now renders the runtime `character_profile["name"]` into the existing
  address/procedure/example flow, adds `character_name` to the human payload,
  and keeps the output schema unchanged. No production blacklist, retry,
  migration, or post-generation rewrite was added.
- 2026-05-10 Stage 4 RED-to-GREEN evidence:
  `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py::test_decontexualizer_prompt_requires_character_name_and_identity_safe_examples -q`
  passed after the production prompt/payload change. The test now verifies
  that the rendered system prompt contains `测试角色`, contains no literal
  `{character_name}`, and does not contain the unsafe pre-fix examples
  `当前助手/角色` or `当前角色说明白`.
- 2026-05-10 Stage 4 focused decontextualizer regression evidence:
  `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py -q`
  passed with 5 passed and 10 deselected.
- 2026-05-10 Stage 4 CJK syntax/render evidence:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py tests\test_decontexualizer_referents.py tests\test_decontexualizer_live_llm.py`
  passed. Static prompt grep
  `rg -n "当前助手/角色|我是想让当前角色|助手/角色" src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py`
  returned no matches.
- 2026-05-10 Stage 5 recorder implementation completed. Changed files:
  `src/kazusa_ai_chatbot/conversation_progress/models.py`,
  `src/kazusa_ai_chatbot/conversation_progress/recorder.py`,
  `src/kazusa_ai_chatbot/brain_service/post_turn.py`,
  `tests/test_conversation_progress_recorder.py`,
  `tests/test_conversation_progress_runtime.py`,
  `tests/test_conversation_progress_flow.py`,
  `tests/test_service_background_consolidation.py`, and
  `tests/test_temporal_relative_terms_live_llm.py`. The recorder input now
  requires `character_name`; the service background recorder wires
  `character_profile["name"]`; the recorder system prompt renders that exact
  name; and the recorder receives a prompt-facing `chat_history_recent`
  projection with `speaker_name`, `speaker_kind`, `body_text`, and optional
  local `timestamp`. Raw history rows and stored `assistant_moves` remain
  unchanged. No production blacklist, retry, migration, output rewrite, or raw
  history mutation was added.
- 2026-05-10 Stage 5 RED-to-GREEN evidence:
  `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py::test_recorder_prompt_requires_character_name_and_prompt_safe_history_projection -q`
  passed. The captured payload contains `character_name == "测试角色"`,
  renders `测试角色` into the system prompt with no literal `{character_name}`,
  and projects the assistant raw row to
  `{"speaker_name": "测试角色", "speaker_kind": "character", "body_text": "别急，我已经听到了。"}`
  without exposing `role`.
- 2026-05-10 Stage 5 focused deterministic evidence:
  `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_conversation_episode_state.py -q`
  passed with 11 passed; `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`
  passed with 18 passed; `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_flow.py -q`
  passed with 12 passed.
- 2026-05-10 Stage 5 CJK syntax/render/static evidence:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\conversation_progress\models.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\brain_service\post_turn.py tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_service_background_consolidation.py tests\test_conversation_progress_flow.py tests\test_temporal_relative_terms_live_llm.py tests\test_conversation_progress_recorder_live_llm.py`
  passed. Static greps
  `rg -n "当前角色|助手|助理|active_character" src\kazusa_ai_chatbot\conversation_progress\recorder.py`
  and
  `rg -n "assistant 回复|本轮 assistant|紧凑 assistant" src\kazusa_ai_chatbot\conversation_progress\recorder.py`
  returned no matches. Static grep
  `rg -n "role.*assistant|assistant.*role" src\kazusa_ai_chatbot\conversation_progress src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py`
  returned only schema/raw-role compatibility matches:
  `persona_supervisor2_msg_decontexualizer.py` input-format documentation and
  `conversation_progress/recorder.py` prompt-facing projection branch
  `if role == "assistant"`.
- 2026-05-10 Stage 6 post-fix decontextualizer positive real LLM evidence:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_active_character_short_answer_uses_character_name -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__active_character_short_answer_uses_character_name__20260510T062951540414Z.json`.
  Judgment: mitigated versus the pre-fix baseline exposure. The rendered prompt
  summary has no `当前角色` or `助手`, the payload includes
  `character_name == "杏山千纱"`, and the output uses `杏山千纱` with no
  forbidden identity hits.
- 2026-05-10 Stage 6 post-fix decontextualizer negative real LLM evidence:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_decontexualizer_live_llm.py::test_live_decontextualizer_direct_second_person_preserved -q -s`
  passed. Trace:
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__direct_second_person_preserved__20260510T063006760699Z.json`.
  Judgment: direct second-person wording was preserved and output had no
  forbidden identity hits.
- 2026-05-10 Stage 6 recorder negative case produced useful review evidence
  before final pass: the first post-fix run preserved user-owned
  `学习助手` but shortened a later project reference to `这个助手`; after the
  recorder prompt was tightened inside the existing field-writing flow to
  prefer complete user-owned names or neutral project wording, a rerun still
  produced `该助手`. Inspection showed this was not active-character leakage:
  it referred to the user-owned project, not `杏山千纱`. The test assessment was
  adjusted to allow demonstrative references to that user-owned helper project
  while still failing bare or active-character uses.
- 2026-05-10 Stage 6 post-fix recorder negative real LLM evidence:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_preserves_user_owned_helper_term -q -s`
  passed after the prompt/test-assessment refinement. Trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__preserves_user_owned_helper_term__20260510T063233668543Z.json`.
  Judgment: user-owned `学习助手` was preserved, prompt summary has no
  `当前角色`, `助手`, or `active_character`, and parsed output had no forbidden
  identity hits under the context-aware allowed user-owned terms.
- 2026-05-10 Stage 6 post-fix recorder positive real LLM evidence:
  `venv\Scripts\python.exe -m pytest -o addopts="" -m live_llm tests\test_conversation_progress_recorder_live_llm.py::test_live_recorder_does_not_use_generic_active_character_labels -q -s`
  passed against the final recorder prompt. Trace:
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__does_not_use_generic_active_character_labels__20260510T063253726113Z.json`.
  Judgment: mitigated versus the pre-fix baseline exposure. The polluted prior
  state still contained `助手`, but prompt-visible recent history used
  `speaker_name` / `speaker_kind`, prompt summary had no generic labels, and
  parsed output had no forbidden identity hits.
- 2026-05-10 Stage 7 static/syntax verification completed. Static greps
  `rg -n "当前助手/角色|我是想让当前角色|助手/角色" src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py`,
  `rg -n "当前角色|助手|助理|active_character" src\kazusa_ai_chatbot\conversation_progress\recorder.py`,
  and
  `rg -n "assistant 回复|本轮 assistant|紧凑 assistant" src\kazusa_ai_chatbot\conversation_progress\recorder.py`
  returned no matches. Static grep
  `rg -n "role.*assistant|assistant.*role" src\kazusa_ai_chatbot\conversation_progress src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py`
  returned only the allowed schema/raw-role compatibility lines:
  `persona_supervisor2_msg_decontexualizer.py` input-format documentation and
  `conversation_progress/recorder.py` prompt projection branch. Production
  syntax command
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\conversation_progress\models.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\brain_service\post_turn.py`
  passed.
- 2026-05-10 Stage 7 deterministic regression verification completed:
  `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py -q`
  passed with 5 passed and 10 deselected;
  `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_conversation_episode_state.py -q`
  passed with 11 passed;
  `venv\Scripts\python.exe -m pytest tests\test_service_background_consolidation.py -q`
  passed with 18 passed; and
  `venv\Scripts\python.exe -m pytest tests\test_memory_writer_prompt_contracts.py tests\test_memory_writer_prompt_projection.py -q`
  passed with 9 passed.
- 2026-05-10 Stage 8 independent code review completed by the active agent; no
  separate reviewer was available in this session. Review scope covered the
  full diff, project Python style constraints, CJK prompt safety, prompt
  rendering, TDD evidence, real LLM trace quality, and plan-scope boundaries.
  Finding fixed: newly added prompt-render helper functions returned call
  expressions directly, violating the project style rule against direct call
  returns. The helpers now assign rendered or stripped values before return.
  No production blacklist, semantic output filter, retry loop, migration, raw
  history mutation, storage-role rename, or fallback prompt was introduced.
- 2026-05-10 Stage 8 rerun evidence after the review fix:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\conversation_progress\models.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\brain_service\post_turn.py`
  passed;
  `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py tests\test_conversation_progress_recorder.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_flow.py tests\test_service_background_consolidation.py -q`
  passed with 42 passed and 10 deselected;
  `venv\Scripts\python.exe -m pytest tests\test_memory_writer_prompt_contracts.py tests\test_memory_writer_prompt_projection.py -q`
  passed with 9 passed; `git diff --check` passed with CRLF warnings only; and
  the recorder/decontextualizer forbidden prompt-source greps returned no
  matches.
- 2026-05-10 Stage 8 final real LLM validation reran one case at a time after
  touching LLM-stage helper code. Decontextualizer positive passed with trace
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__active_character_short_answer_uses_character_name__20260510T063849522338Z.json`;
  prompt summary had no generic labels, payload carried
  `character_name == "杏山千纱"`, output used `杏山千纱`, and forbidden hits
  were empty. Decontextualizer negative passed with trace
  `test_artifacts/llm_traces/decontextualizer_identity_live_llm__direct_second_person_preserved__20260510T063858415724Z.json`;
  direct `你也不反对吧` was preserved and forbidden hits were empty. Recorder
  positive passed with trace
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__does_not_use_generic_active_character_labels__20260510T063909978995Z.json`;
  polluted prior state still contained `助手`, prompt-visible recent history
  used `speaker_name` / `speaker_kind`, prompt summary had no generic labels,
  and parsed output had no forbidden hits. Recorder negative passed with trace
  `test_artifacts/llm_traces/conversation_progress_recorder_identity_live_llm__preserves_user_owned_helper_term__20260510T063920619791Z.json`;
  user-owned `学习助手` was preserved and forbidden hits were empty.
- 2026-05-10 final review approval: acceptance criteria passed. Residual risk:
  live LLM assertions use substring scanners plus human trace inspection; they
  intentionally do not add any production semantic filter. Plan status changed
  to `completed`.
