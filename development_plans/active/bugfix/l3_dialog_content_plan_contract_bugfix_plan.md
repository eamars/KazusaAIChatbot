# l3 dialog content plan contract bugfix plan

## Summary

- Goal: replace the `content_anchors: list[str]` L3-to-dialog contract with a
  native `content_plan: dict[str, str]` contract so dialog renders one resolved
  semantic plan instead of translating every anchor label into visible text.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`
- Overall cutover strategy: bigbang for runtime prompt/state/test contracts;
  no compatibility shim, no old-anchor projection, no dual read path.
- Highest-risk areas: dialog re-authoring missing content, L3 losing required
  technical facts or code blocks, stale `content_anchor` terminology leaking
  through memory lifecycle/consolidation/progress/telemetry, and live LLM tests
  passing structurally without human quality review.
- Acceptance criteria: source and tests use `content_plan`, deterministic
  contract tests pass, static greps show no runtime `content_anchor` contract
  remains, and seven one-at-a-time real LLM cases are inspected with a
  human-readable review artifact.

## Context

The observed group-chat output fragmentation came from a real contract error.
The current L3 content-anchor prompt emits a list such as `[DECISION]`,
`[ANSWER]`, `[SOCIAL]`, `[PROGRESSION]`, and `[SCOPE]`. The dialog generator
prompt then tells the model to first list mandatory deliverables from every
anchor category, while the evaluator fails the response if any anchor category
is bypassed. That makes the local LLM treat each label as a visible checklist
item, which explains outputs where every anchor becomes its own line or
paragraph.

The target is not a multi-message sender and not a deterministic formatter.
All visible text still returns through the current `final_dialog: list[str]`
shape and is joined into one bubble by the existing delivery path. The fix is
to change the upstream semantic handoff so dialog receives one resolved content
plan rather than a tagged list that looks like a set of independent obligations.

The upstream LLM must generate `content_plan` natively. It must not output old
anchor strings that deterministic code converts into a dict. If a response
needs a visible fact, conclusion, code block, refusal boundary, uncertainty, or
specific next step, L3 must put that semantic content into the plan before
dialog runs. Dialog may rephrase, compress, split lines, and add harmless
particles, but it must not fill missing facts, examples, topics, conclusions,
or questions.

The three basis scenarios for design and live validation are:

- casual group reply: prevent generic filler such as `那我们继续聊点轻松有趣的内容吧`
  when that content was not resolved upstream;
- technical comparison: preserve every provided GB300/Pro6000 number and the
  upstream conclusion while allowing multi-line rendering;
- code/fixed-format reply: preserve code blocks exactly inside one string
  value and one visible bubble.

Adjacent improvement areas intentionally left for later plans:

- no redesign of L2 cognition stance generation;
- no new LLM stage, repair loop, or deterministic semantic validator;
- no adapter delivery changes;
- no effort to improve factual correctness outside the provided content plan.

## Mandatory Skills

- `development-plan`: load before plan lifecycle edits, execution, review, or
  lifecycle status updates.
- `local-llm-architecture`: load before changing prompts, graph stage names,
  LLM input/output contracts, dialog/evaluator behavior, or consolidation
  prompt payloads.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt files containing CJK text;
  run syntax and prompt-render checks after edits.
- `test-style-and-execution`: load before adding, changing, or running
  deterministic, live LLM, or live DB tests.
- `debug-llm`: load before running live LLM cases, writing trace artifacts, or
  judging LLM output quality.

## Mandatory Rules

- Do not modify production code until this plan is approved or in progress and
  the user explicitly authorizes implementation.
- Do not preserve, accept, or project `content_anchors` in runtime code after
  cutover.
- Do not add a compatibility shim, fallback path, dual read, old-key alias,
  string-anchor parser, or old-anchor-to-plan converter.
- Do not hard-enumerate `content_plan` keys in deterministic validation.
  Deterministic code validates only that `content_plan` is a dict with string
  keys and string values.
- Prompts should teach the preferred keys `visible_goal`, `semantic_content`,
  `voice`, and `rendering`, but code must not reject missing, reordered, or
  spelling-drifted keys when all present values are strings.
- Each `content_plan` key has exactly one string value. No lists, nested
  objects, arrays of facts, separate `must_not_include` fields, optional
  content fields, or `facts_to_preserve` side channels.
- Ordinary string values should be compact single-paragraph text. The only
  newline exception is literal fixed-format content such as code, JSON, logs,
  configs, CLI output, patches, or tables that must retain line breaks.
- `semantic_content`, when present, is the only source of visible facts,
  claims, conclusions, code, examples, and concrete next topics. `visible_goal`
  explains purpose, `voice` shapes expression, and `rendering` shapes layout.
- Dialog may interpret wording but must not author missing semantic content.
- L3 content-plan generation must resolve vague progression before dialog. A
  phrase like "continue light topics" may appear only when that exact visible
  move is the intended content, not as a placeholder for dialog to invent.
- Code and fixed-format blocks are protected literal islands. Dialog must not
  change indentation, line order, fence language, blank lines, symbols, or
  code content.
- Keep `final_dialog: list[str]` and `mention_target_user: bool` unchanged.
- No hard line, segment, paragraph, or character budget is allowed. `rendering`
  may express qualitative length and formatting requirements.
- Any prompt changed by this plan must be rewritten as a coherent prompt flow
  that satisfies local-LLM prompt requirements. Do not append isolated warning
  bullets, negative constraints, migration notes, or one-off regression
  patches to the current prompt. The changed prompt must preserve a readable
  stage boundary, positive generation/review procedure, input-field semantics,
  and output contract.
- No new runtime LLM call, retry loop, repair prompt, graph node, adapter
  behavior, model route, temperature, timeout, or provider setting is allowed.
- Real LLM tests must run one case at a time with `-q -s`, and each trace must
  be inspected before the next case runs.
- Live LLM quality review must be agent-authored Markdown from raw traces. A
  passing pytest result is not quality approval.
- Use `venv\Scripts\python.exe` for Python commands. Do not read `.env`.
- Use `apply_patch` for manual edits.
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

## Must Do

- Replace L3 content-anchor output with native `content_plan: dict[str, str]`.
- Rename runtime prompt/stage/function/test terminology from content anchor to
  content plan where it refers to the L3-to-dialog text contract.
- Rewrite every changed runtime prompt as a coherent positive procedure, not
  an appended list of negative constraints. This includes the L3 content-plan
  prompt, dialog generator/evaluator prompts, conversation-progress recorder
  prompt, consolidation prompts, memory-lifecycle prompt text, and any
  self-cognition prompt text touched by the migration.
- Feed prompt-safe resolver goal progress and resolver observations into the
  content-plan LLM input instead of appending extra anchors after L3.
- Remove deterministic semantic append helpers that build extra `[ANSWER]`,
  `[FACT]`, or `[SCOPE]` strings after the content-plan LLM returns.
- Update dialog generator and evaluator prompts so they consume `content_plan`
  as one semantic plan, with `semantic_content` as the preferred visible
  semantic payload when present.
- Update dialog validation and telemetry to count `content_plan` entries
  rather than anchors.
- Update conversation progress recorder, consolidation facts/reflection, and
  self-cognition tracking to read `content_plan` values as medium-strength
  response-planning evidence.
- Update memory lifecycle context terminology that currently exposes
  `content_anchor_roles` to downstream L3. Use `content_plan_roles` entries
  with `role` and `instruction` strings so old anchor vocabulary does not
  re-enter the new L3 prompt.
- Update docs and tests to the new contract.
- Add deterministic contract tests before production prompt/code edits.
- Add real LLM tests for the three basis scenarios plus adjacent guard cases.
- Write a human-readable LLM debug review artifact after live validation.

## Deferred

- Do not change adapter delivery, newline joining, platform send count, or
  `final_dialog` schema.
- Do not alter L1, L2a, L2b, L2c1, L2c2, or L2d prompt ownership unless a
  static reference to the removed content-anchor contract must be renamed.
- Do not add deterministic semantic filters over user input or dialog output.
- Do not add `must_not_include`, optional visible content, continuation fields,
  fact lists, nested content units, or rejected-alternative fields.
- Do not add old-anchor compatibility, migration readers, aliases, or tests
  that prove old anchors still work.
- Do not update historical archived development plans except through registry
  lifecycle rules.
- Do not change database documents or backfill historical event/progress rows.
- Do not improve unrelated RAG, memory lifecycle, self-cognition routing, or
  consolidation behavior.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| L3 text semantic output | bigbang | Replace `content_anchors` with `content_plan`; no old key, parser, alias, or projection. |
| Prompt stage name | bigbang | Rename content-anchor stage terminology to content-plan terminology in runtime code, prompt selection, tests, and docs. |
| Dialog input | bigbang | Dialog consumes `content_plan`; it no longer lists `[DECISION]` / `[ANSWER]` deliverables. |
| Resolver progress handoff | bigbang | Move prompt-safe progress/observation summaries into L3 input; remove post-L3 anchor append. |
| Memory lifecycle handoff | bigbang | Rename `content_anchor_roles` to `content_plan_roles` in the prompt-safe context. |
| Consolidation/progress evidence | bigbang | Project `content_plan` values as medium-strength planning evidence; no legacy anchor extractor. |
| Event logging | bigbang | Replace `anchor_count` with `content_plan_entry_count`. |
| Output delivery | compatible | Keep `final_dialog` and `mention_target_user` unchanged. |
| Stored historical records | no-op | Do not migrate historical event or conversation-progress rows. |
| Tests | bigbang | Rewrite fixtures and assertions to the new plan contract; delete old anchor-label assumptions. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- For every bigbang area, remove or rewrite old references instead of
  preserving them.
- Historical records may still contain old field names in the database; runtime
  code must not read or write old fields after cutover.
- Any change to this cutover policy requires user approval before
  implementation.

## Data Migration

No database migration is part of this plan.

Historical event-log documents, trace artifacts, and archived plans may still
contain `content_anchors` or `anchor_count`. Runtime code and active tests must
not depend on those old fields. New runtime event payloads must use the new
content-plan telemetry field.

## Target State

```text
selected speak action
  -> L3 content-plan agent
       reads L2 stance, selected text intent, prompt-safe resolver progress,
       memory lifecycle context, interaction style, conversation progress,
       and source payloads
       returns content_plan: dict[str, str]
  -> L4 collector
       places content_plan under action_directives.linguistic_directives
  -> dialog generator/evaluator
       renders one visible bubble from content_plan, style, context, and voice
  -> post-turn progress/consolidation/tracking
       treat content_plan as medium-strength response-planning evidence
```

Preferred model-facing shape:

```json
{
  "content_plan": {
    "visible_goal": "接住轻松调侃，让对方感到角色被逗乐且相处舒服。",
    "semantic_content": "被对方逗乐了，有一点小窃喜；这种轻松相处方式让人觉得舒服。",
    "voice": "轻快、随和，不深究具体指代。",
    "rendering": "约35字；单个聊天气泡；2-3个自然短句；可自然改写 semantic_content，但不得补充 semantic_content 没有的事实、话题、问题或结论。"
  }
}
```

This shape is a prompt convention, not a deterministic enum. If a weaker model
returns `semantics` or another string key, code accepts the dict when every key
and value is a string. Dialog still must not invent missing content.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Semantic handoff | Use `content_plan: dict[str, str]` | A flat dict prevents every label from becoming a mandatory visible paragraph. |
| Preferred keys | Teach `visible_goal`, `semantic_content`, `voice`, `rendering` | The names are self-explanatory for weaker models and avoid overlapping labels. |
| Key validation | Validate shape only, not key enum | User explicitly rejected hard enumeration and accepts missing/spelling drift. |
| Value cardinality | One string per key | Lists/nested objects increase local-LLM stress and recreate multi-lane duplication. |
| Content authority | `semantic_content` preferred when present | Dialog needs one content payload but must still tolerate key drift. |
| Dialog freedom | Rephrase only | Wording interpretation is allowed; missing semantic content generation is not. |
| Resolver progress | LLM-resolved inside L3 | Deterministic post-L3 semantic append was part of the old anchor blast radius. |
| Code blocks | Literal content inside one string | The code exception preserves fixed-format blocks without adding list values. |
| Runtime calls | No new calls | The live chat path remains bounded for local LLM latency. |
| Compatibility | None | Old anchors are removed rather than adapted. |

## Contracts And Data Shapes

### L3 Output

```python
{
    "content_plan": dict[str, str],
}
```

Validation:

- top-level `content_plan` is required;
- `content_plan` must be a non-empty dict;
- every key must be a non-empty string;
- every value must be a string;
- normalize by stripping keys and values;
- drop entries whose stripped key or value is empty;
- after normalization, require at least one entry;
- no list, dict, number, bool, or null values are accepted;
- exact key names are not deterministically enumerated.

### L4 Action Directives

```python
{
    "action_directives": {
        "contextual_directives": {
            "social_distance": str,
            "emotional_intensity": str,
            "vibe_check": str,
            "relational_dynamic": str,
        },
        "linguistic_directives": {
            "rhetorical_strategy": str,
            "linguistic_style": str,
            "accepted_user_preferences": list[str],
            "content_plan": dict[str, str],
            "forbidden_phrases": list[str],
        },
        "visual_directives": {
            "facial_expression": list[str],
            "body_language": list[str],
            "gaze_direction": list[str],
            "visual_vibe": list[str],
        },
    }
}
```

### Dialog Input

Dialog receives the same `action_directives` envelope, but
`linguistic_directives.content_plan` replaces
`linguistic_directives.content_anchors`.

Dialog prompt rules:

- read `semantic_content` first when present;
- use other content-plan fields to understand purpose, voice, and layout;
- if key names drift, infer the closest role from key text and value text;
- never convert `visible_goal`, `voice`, or `rendering` into new facts;
- never generate facts, examples, technical judgments, code, next topics, or
  questions that are absent from the plan;
- preserve literal code/fixed-format blocks exactly.

### Consolidation And Conversation Progress

The projection helper becomes:

```python
content_plan_from_action_directives(action_directives: object) -> dict[str, str]
```

Consolidation and progress prompts should receive `content_plan` as a dict and
describe it as pre-dialog response-planning evidence. It remains weaker than
`final_dialog` for persistence decisions.

### Self-Cognition Tracking

Self-cognition marker fallback must scan all string values in `content_plan`
for existing route markers such as `PROGRESS_MAINTENANCE_MARKER`. It must not
look for old anchor labels or require a specific content-plan key.

### Forbidden Compatibility Shapes

These runtime shapes are removed:

```python
{"content_anchors": ["[DECISION] ...", "[ANSWER] ...", "[SCOPE] ..."]}
{"anchor_count": 3}
```

## LLM Call And Context Budget

- No new response-path or background LLM calls are added.
- The L3 content-plan agent reuses the existing COGNITION route and call site.
- The dialog generator and evaluator keep their current call count and retry
  behavior.
- Resolver goal progress and prompt-safe observations move from deterministic
  post-L3 append into the existing L3 human payload. This increases L3 input
  content only by data already available in state and previously appended to
  dialog.
- Prompt size should stay below the existing 50k default cap. Implementation
  must remove old anchor dependency-tree prose while adding the new compact
  content-plan procedure.
- Fixed-format blocks may increase one `semantic_content` string length, but
  they do not add new payload lanes.
- Verification uses live LLM tests one case at a time and records raw traces
  under `test_artifacts/llm_traces/` plus an agent-authored review under
  `test_artifacts/llm_reviews/`.

## Change Surface

### Modify

Primary runtime source:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Rename content-anchor prompt/call/logging/output to content-plan contract.
  - Rewrite the L3 prompt.
  - Add resolver goal progress and observation summaries to the L3 input.
  - Remove post-L3 anchor append helpers.
  - Collect `content_plan` in L4.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Rename graph node and imports from content-anchor to content-plan.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  - Replace the stage name with the content-plan stage name.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
  - Validate the new `content_plan` field shape.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Replace `content_anchors: list[str]` with `content_plan: dict[str, str]`.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Replace default empty content-anchor state with content-plan state.

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Rewrite generator/evaluator prompt flow for `content_plan`.
  - Update state validation expectations and event logging call.

Secondary runtime consumers:

- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Pass `content_plan` to conversation progress recording.

- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Replace `content_anchors` in `RecordTurnProgressInput`.

- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Update prompt input, reading order, and temporal rules to use
    `content_plan`.

- `src/kazusa_ai_chatbot/consolidation/schema.py`
  - Replace `content_anchors_from_action_directives` with
    `content_plan_from_action_directives`.

- `src/kazusa_ai_chatbot/consolidation/facts.py`
  - Update prompts and payloads to use `content_plan` as medium-strength
    planning evidence.

- `src/kazusa_ai_chatbot/consolidation/reflection.py`
  - Update prompts and payloads to use `content_plan`.

- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - Scan content-plan string values for route markers.

- `src/kazusa_ai_chatbot/event_logging/recording.py`
  - Rename dialog quality telemetry from `anchor_count` to a content-plan
    `content_plan_entry_count` field.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`
  - Rename prompt-safe downstream role field away from
    `content_anchor_roles` to `content_plan_roles`.
  - Rename each role item's visible text field from `anchor` to `instruction`.

- `src/kazusa_ai_chatbot/utils.py`
  - Remove or update the stale parser debug sample containing content anchors.

Documentation:

- `src/kazusa_ai_chatbot/nodes/README.md`
  - Update L3, L4, dialog, and examples to `content_plan`.

- `src/kazusa_ai_chatbot/event_logging/README.md`
  - Update dialog quality event field documentation.

- `development_plans/README.md`
  - Track this plan in the active bugfix registry and later lifecycle moves.

Tests:

- Update all tests returned by
  `rg -l "content_anchors|content_anchor|anchor_count" tests` at execution
  time.
- Current known affected test files include dialog tests, L3 prompt-contract
  tests, multi-source cognition tests, conversation-progress tests,
  consolidation tests, memory lifecycle tests, self-cognition tests,
  event-logging tests, image-cognition live tests, and temporal/live LLM tests.

### Create

- `tests/test_l3_dialog_content_plan_contract.py`
  - Focused deterministic contract tests for L3 output shape, L4 collection,
    dialog validation, projection helper, and no old-anchor fallback.

- `tests/test_l3_dialog_content_plan_live_llm.py`
  - Real LLM cases for L3 content-plan generation and dialog consumption.

- `test_artifacts/llm_reviews/l3_dialog_content_plan_contract_<timestamp>.md`
  - Agent-authored live LLM review artifact created during verification.

### Keep

- `final_dialog: list[str]` output schema.
- `mention_target_user: bool` output schema.
- Adapter delivery and one-bubble newline join behavior.
- Existing relevance, decontextualizer, L1/L2/L2d, RAG, scheduler, dispatcher,
  and database behavior unless a stale content-anchor reference is directly
  in the approved change surface.

## Overdesign Guardrail

- Actual problem: tagged `content_anchors` make dialog and evaluator treat
  every upstream role label as a visible deliverable, causing fragmented and
  over-covered output.
- Minimal change: replace the tagged list with one flat string-valued
  `content_plan` dict and update direct consumers to read that contract.
- Ownership boundaries: L3 owns response semantic planning; dialog owns
  wording, layout, one-bubble rendering, and literal block preservation;
  deterministic code owns shape validation, persistence payloads, telemetry,
  and graph wiring.
- Rejected complexity: compatibility shims, old-anchor parsers, key enums,
  nested content units, fact lists, must-not fields, optional content lanes,
  extra LLM calls, repair prompts, deterministic semantic filters, adapter
  changes, and delivery rewrites.
- Evidence threshold: new runtime stages, compatibility modes, or helper
  agents require repeated post-cutover real LLM failures that cannot be fixed
  by prompt-contract clarification inside this plan's change surface and a new
  user-approved plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the target modules named in
  `Change Surface` as out of scope unless a static grep proves they are stale
  references to the removed content-anchor contract.
- The responsible agent must remove stale content-anchor references when they
  are explicitly in scope and verified by greps and tests.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad refactors, or prompt rewrites outside this
  contract migration.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Confirm authorization and starting state.
   - Verify this plan is `approved` or `in_progress`.
   - Verify the user explicitly authorized implementation.
   - Run `git status --short`.
   - Record branch and dirty-worktree notes in `Execution Evidence`.

2. Load required skills and reread local contracts.
   - Load all mandatory skills.
   - Reread `development_plans/README.md`, `src/kazusa_ai_chatbot/nodes/README.md`,
     and this plan.
   - Record evidence before editing.

3. Establish focused deterministic test contract.
   - Add `tests/test_l3_dialog_content_plan_contract.py`.
   - Add tests proving:
     - L3 output contract requires `content_plan` dict values as strings;
     - wrong old `content_anchors` output fails validation;
     - L4 collector emits `linguistic_directives.content_plan`;
     - dialog action-directive validation rejects missing `content_plan`;
     - consolidation projection returns a dict;
     - self-cognition marker fallback scans plan values;
     - no deterministic key enum rejects noncanonical string keys.
   - Run the focused tests before production implementation.
   - Expected baseline: failures for missing new contract and old runtime shape.

4. Establish prompt-text and static-grep tests.
   - Update or add tests that assert dialog/L3 prompts mention `content_plan`
     and no longer mention old anchor labels as the semantic authority.
   - Add a static grep test or documented manual grep gate for runtime source.
   - Expected baseline: failures proving old prompt text remains.

5. Add live LLM test scaffolding.
   - Create `tests/test_l3_dialog_content_plan_live_llm.py`.
   - Use existing live endpoint skip helpers and `tests.llm_trace.write_llm_trace`.
   - Write traces before case-specific quality assertions.
   - Include raw L3 input, raw L3 output, dialog golden input, raw dialog
     output, joined one-bubble text, parser status, and route/model metadata.

6. Add L3 real LLM case A: casual overloaded source.
   - Simulated upstream state mirrors the reported anchor pressure: playful
     tone, comfort, and an Agent-development progression affordance.
   - Expected: `content_plan` is a flat dict of string values; the plan
     resolves what is actually visible; `semantic_content` does not contain
     three unrelated visible ideas when scope and current turn do not support
     them.

7. Add L3 real LLM case B: technical comparison.
   - Simulated state asks for GB300 vs Pro6000.
   - Expected: all supplied numbers and the final suitability conclusion are
     present in plan values, preferably under `semantic_content`.

8. Add L3 real LLM case C: code/fixed-format source.
   - Simulated state includes a prompt-safe Python code block in evidence or
     selected text intent.
   - Expected: the code block remains inside one string value and no separate
     list of code lines is produced.

9. Add dialog golden case D: casual content plan.
   - Dialog input uses this golden plan:

```json
{
  "visible_goal": "接住轻松调侃，让对方感到角色被逗乐且相处舒服。",
  "semantic_content": "被对方逗乐了，有一点小窃喜；这种轻松相处方式让人觉得舒服。",
  "voice": "轻快、随和，不深究具体指代。",
  "rendering": "约35字；单个聊天气泡；2-3个自然短句；可自然改写 semantic_content，但不得补充 semantic_content 没有的事实、话题、问题或结论。"
}
```

   - Expected: no generic continuation, no Agent-development topic, no extra
     question.

10. Add dialog golden case E: technical comparison.
    - Dialog input uses this golden plan:

```json
{
  "visible_goal": "回答用户对 GB300 和 Pro6000 的性能对比请求，并给出适用场景结论。",
  "semantic_content": "GB300: FP16 2250 TFLOPS, FP8 4500 TFLOPS, 288GB HBM3e, 带宽 12000 GB/s, TDP 1400W, FP32 90 TFLOPS。Pro6000: FP16 125 TFLOPS, FP8 2000 TFLOPS, 96GB GDDR7, 带宽约1792 GB/s, TDP 400W, FP32 125 TFLOPS。结论：GB300 更适合超大规模训练和推理；Pro6000 更适合较小规模推理。",
  "voice": "可以轻微调侃，但信息密度优先。",
  "rendering": "单个聊天气泡；允许多行短句；保留数值和单位；不得补充 semantic_content 没有的技术判断。"
}
```

    - Expected: all numeric facts and conclusion appear; no extra technical
      criteria such as "high concurrency" unless present in the plan.

11. Add dialog golden case F: code block.
    - Dialog input contains a `semantic_content` string with a fenced Python
      code block.
    - Expected: fenced block exists, indentation is preserved, code content is
      unchanged, and character voice stays outside the block.

12. Add dialog golden case G: private soft reply.
    - Dialog input has a warmer private-chat voice and a small practical
      conclusion in `semantic_content`.
    - Expected: private tone is warmer but no new promise or follow-up task is
      invented.

13. Run one baseline live dialog golden case before production edits.
    - Run the casual dialog golden case with `-m live_llm -q -s`.
    - Record trace path and manual judgment. If trace-writing is broken, fix
      only the test harness before production edits.

14. Start the production-code subagent.
    - Provide this approved plan, failing focused tests, baseline trace,
      mandatory skills, and exact production change surface.
    - Production-code subagent may edit production code only.
    - It must close after planned production changes are complete.

15. Parent continues test and documentation migration while production code is
    being edited.
    - Update tests and docs inside change surface.
    - Keep production edits owned by the production-code subagent unless the
      parent is remediating integration or review findings after subagent close.

16. Implement production cutover.
    - Rename content-plan stage/function/prompt/output.
    - Update L3 prompt and input payload.
    - Remove post-L3 anchor append helpers.
    - Update dialog prompts and validator.
    - Update progress, consolidation, self-cognition, memory lifecycle, event
      logging, docs, and utility sample.
    - For every changed prompt, rewrite the full affected prompt flow around
      the new `content_plan` contract. Do not append a warning paragraph to the
      old prompt. The diff must show that the stage boundary, generation or
      review procedure, input-field semantics, and output contract were
      reviewed together.

17. Run syntax and prompt-render checks.
    - `venv\Scripts\python.exe -m py_compile` on all changed Python files.
    - Runtime `.format(...)` render checks for changed prompt constants that
      use formatting.
    - Review prompt diffs manually and record that no changed prompt was
      implemented as an appended warning, appended negative constraint, or
      migration note.

18. Run focused deterministic tests.
    - Run the new focused contract test file.
    - Run updated dialog deterministic tests.
    - Run updated L3 surface handoff tests.

19. Run static greps.
    - `rg "content_anchors|content_anchor" src/kazusa_ai_chatbot`
    - Expected: zero matches after runtime cutover.
    - `rg "anchor_count" src/kazusa_ai_chatbot`
    - Expected: zero matches after telemetry rename.
    - `rg "content_anchors|content_anchor|anchor_count" tests`
    - Expected: matches are allowed only in negative contract tests inside
      `tests/test_l3_dialog_content_plan_contract.py` that intentionally build
      rejected old-shape payloads. Any positive fixture, live LLM case, helper,
      or regression test using old fields must be rewritten.

20. Run regression tests by affected cluster.
    - Dialog cluster.
    - L3/content-plan prompt-contract cluster.
    - Conversation progress cluster.
    - Consolidation cluster.
    - Self-cognition tracking/integration cluster.
    - Memory lifecycle cluster.
    - Event logging cluster.
    - Multi-source cognition prompt-selection/dry-run cluster.

21. Run live LLM cases one at a time.
    - Run cases A through G individually with `-m live_llm -q -s`.
    - Inspect each trace before running the next case.
    - Write the Markdown review artifact after all raw traces are inspected.

22. Run focused diff review.
    - Confirm no old-anchor compatibility path exists.
    - Confirm production changes align with `Change Surface`.
    - Confirm live review maps each basis scenario to observed outputs.

23. Start the independent code-review subagent.
    - Provide approved plan, full diff, deterministic results, static greps,
      live traces, and review artifact.
    - Review subagent reports findings only.

24. Parent remediates in-scope review findings.
    - Fix only within approved change surface.
    - Rerun affected deterministic, static, and live checks.

25. Request lifecycle sign-off.
    - Do not mark complete until user accepts evidence and review status.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
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

- [ ] Stage 1 - focused contract tests established
  - Covers: implementation steps 1-4.
  - Files: `tests/test_l3_dialog_content_plan_contract.py` plus updated
    prompt-text tests.
  - Verify: focused tests fail before implementation for missing
    `content_plan` contract.
  - Evidence: record failing assertion summary in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 2 - live LLM scaffolding and baseline recorded
  - Covers: implementation steps 5-13.
  - Files: `tests/test_l3_dialog_content_plan_live_llm.py`.
  - Verify: one baseline live case writes a trace.
  - Evidence: record trace path and manual judgment.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 3 - production cutover complete
  - Covers: implementation steps 14-17.
  - Files: primary runtime source files in `Change Surface`.
  - Verify: syntax and prompt-render checks pass.
  - Evidence: record production subagent summary, changed files, and command
    output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 4 - deterministic and static verification complete
  - Covers: implementation steps 18-20.
  - Files: source, docs, and affected tests.
  - Verify: focused tests, affected regression clusters, and static greps pass.
  - Evidence: record commands and results.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 5 - real LLM validation complete
  - Covers: implementation step 21.
  - Files: live LLM trace outputs and review artifact.
  - Verify: cases A-G run one at a time and raw traces are inspected.
  - Evidence: record trace paths, quality notes, and review artifact path.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after evidence is recorded.

- [ ] Stage 6 - independent code review and lifecycle sign-off complete
  - Covers: implementation steps 22-25.
  - Verify: independent review reports no unresolved blocking findings, or
    findings are remediated and affected checks rerun.
  - Evidence: record review summary, fixes, reruns, residual risks, and user
    sign-off.
  - Handoff: none.
  - Sign-off: `<agent/date>` after evidence is recorded.

## Verification

Syntax and prompt render:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_memory_lifecycle.py src\kazusa_ai_chatbot\conversation_progress\recorder.py src\kazusa_ai_chatbot\consolidation\facts.py src\kazusa_ai_chatbot\consolidation\reflection.py src\kazusa_ai_chatbot\event_logging\recording.py src\kazusa_ai_chatbot\self_cognition\tracking.py
venv\Scripts\python.exe -c "from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as m; m._CONTENT_PLAN_AGENT_PROMPT.format(character_name='测试角色'); print('OK')"
```

Focused deterministic tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_contract.py -q
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py -q
venv\Scripts\python.exe -m pytest tests\test_l2d_l3_surface_handoff.py -q
```

Affected regression clusters:

```powershell
venv\Scripts\python.exe -m pytest tests\test_conversation_progress_cognition.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_recorder.py -q
venv\Scripts\python.exe -m pytest tests\test_consolidator_facts_rag2.py tests\test_consolidator_reflection_prompts.py tests\test_consolidator_source_aware_payloads.py -q
venv\Scripts\python.exe -m pytest tests\test_memory_lifecycle_specialist.py tests\test_self_cognition_tracking.py tests\test_rag_dialog_event_logging.py tests\test_event_logging_interface.py -q
venv\Scripts\python.exe -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q
```

Static greps:

```powershell
rg "content_anchors|content_anchor" src\kazusa_ai_chatbot
rg "anchor_count" src\kazusa_ai_chatbot
rg "content_anchors|content_anchor|anchor_count" tests
```

Expected result: zero matches under `src\kazusa_ai_chatbot`. Under `tests`,
matches are allowed only in `tests\test_l3_dialog_content_plan_contract.py`
when the test intentionally constructs rejected old-shape payloads. Any other
test match is a blocking stale-contract reference.

Real LLM tests, one case at a time:

```powershell
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_casual_overloaded_source -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_technical_comparison -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_code_block_source -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_casual_golden -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_technical_golden -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_code_block_golden -m live_llm -q -s
venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_private_soft_reply -m live_llm -q -s
```

Every real LLM trace must be inspected before the next command runs. The agent
must write a Markdown review artifact under `test_artifacts/llm_reviews/`
covering run context, inputs, raw outputs, parsed outputs, quality notes,
validation results, and raw evidence paths.

## Plan Self-Review

Self-review performed on the draft before approval.

- Coverage: every `Must Do` item maps to implementation steps, change-surface
  files, verification gates, or acceptance criteria.
- Minimality: the plan changes the L3-to-dialog semantic contract and direct
  consumers only; it rejects compatibility, extra LLM calls, deterministic
  semantic filters, and adapter changes.
- Placeholder scan: completed with no unresolved placeholders, deferred work
  markers, broad copy-forward wording, or open decision language remaining.
- Contract consistency: `content_plan`, `content_plan_roles`,
  `content_plan_entry_count`, and test names are consistent across sections.
- Granularity: checklist stages have file targets, verification commands,
  evidence requirements, and handoff points.
- Verification: static greps now allow only negative old-shape contract tests;
  runtime source must have zero old-anchor matches.
- Surfaced issue fixed: live LLM case count now states seven A-G cases rather
  than six.
- Surfaced issue fixed: static grep gates no longer conflict with negative
  tests that intentionally construct rejected old `content_anchors` payloads.
- Surfaced issue fixed: this plan now includes an `Execution Evidence` section
  for approved execution records.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final
sign-off. The parent agent must create one independent code-review subagent
through the current harness's native subagent capability. If native subagents
are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with this plan's `Must Do`, `Deferred`, cutover policy, data
  shapes, implementation order, verification gates, and acceptance criteria.
- Absence of compatibility shims, old-anchor parsers, hard key enums,
  deterministic semantic filters, extra LLM calls, fallback prompts, or adapter
  delivery changes.
- Prompt quality for local LLMs: short positive procedures, no negative
  constraint pile, no appended warning blocks, no migration notes, and no
  hidden reliance on dialog to create missing content.
- Real LLM evidence quality: cases A-G ran one at a time, traces were
  inspected, and the Markdown review is agent-authored from raw evidence.

The parent fixes concrete findings directly only when the fix is inside the
approved change surface. If a fix changes contract or scope, stop and request
approval before changing code.

## Acceptance Criteria

This plan is complete when:

- Runtime code writes and reads `content_plan`, not `content_anchors`.
- No runtime compatibility shim, alias, converter, parser, or dual path for old
  anchors exists.
- Deterministic validation accepts `dict[str, str]` without hard-enumerating
  keys and rejects non-string values.
- L3 prompt produces a native content plan and resolves progression/content
  before dialog.
- Dialog prompt renders from `content_plan` and does not author missing
  semantic content.
- Every changed runtime prompt is rewritten as a coherent flow; no prompt
  change is implemented by appending warnings, appended negative constraints,
  or migration notes to the old prompt.
- Code/fixed-format blocks are preserved exactly inside one string value and
  one visible bubble.
- Conversation progress, consolidation, and self-cognition read
  `content_plan` as medium-strength planning evidence.
- Memory lifecycle no longer exposes old `content_anchor_roles` terminology to
  L3 and uses `content_plan_roles[].instruction` instead.
- Dialog quality telemetry no longer uses `anchor_count`.
- Static greps for `content_anchors`, `content_anchor`, and `anchor_count`
  pass under the searched runtime/test paths.
- Focused deterministic tests and affected regression clusters pass.
- Real LLM cases A-G produce durable traces and an agent-authored quality
  review artifact.
- The casual live cases do not invent generic continuation or Agent-topic
  content absent from `semantic_content`.
- The technical live cases preserve all provided numbers and do not add new
  technical criteria absent from `semantic_content`.
- The code live cases preserve fenced code formatting and keep character voice
  outside the block.
- Independent code review has no unresolved blocking findings.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Local LLM copies `semantic_content` too literally | Prompt dialog to render naturally while preserving semantic payload | Casual and private live dialog cases |
| Local LLM treats all dict keys as equal content | Prompt `semantic_content` as preferred payload and other keys as purpose/style/layout | Dialog golden tests and review artifact |
| L3 omits a required technical fact | Technical L3 live case requires every supplied number in plan values | L3 technical live case |
| Dialog invents missing technical conclusions | Dialog prompt forbids content absent from plan | Technical golden live case |
| Code block is reflowed or voice-contaminated | Literal block preservation rule and code live case | Code L3 and dialog live cases |
| Stale anchor terminology remains in a secondary prompt | Static grep and secondary consumer update | Static greps and regression clusters |
| Historical event rows contain old fields | No DB migration; scope limits runtime reads/writes only | Data migration section and event logging tests |
| Real LLM nondeterminism masks quality regressions | One-at-a-time traces plus human-authored review | Debug review artifact |

## Execution Evidence

Pre-execution status:

- Draft created and registered under `development_plans/active/bugfix/`.
- Self-review completed before approval.
- User approved execution on 2026-06-10.
- Plan status moved to `in_progress` before production-code execution.
- No implementation, production-code, test-code, or live LLM verification
  evidence has been recorded yet.

During approved execution, record each checklist sign-off here with command
outputs, trace paths, review findings, fixes, reruns, residual risks, and user
sign-off.
