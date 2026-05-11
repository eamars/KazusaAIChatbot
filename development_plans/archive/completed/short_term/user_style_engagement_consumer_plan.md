# user style engagement consumer plan

## Summary

- Goal: Route `user_style_image.engagement_guidelines` to the smallest correct consumers, rename the relevance module to `persona_relevance_agent`, and move the current vision descriptor ownership into the decontextualizer module without changing graph order.
- Plan class: medium
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for module rename, prompt, and in-process payload changes; compatible for stored `interaction_style_images`.
- Highest-risk areas: group relevance becoming too eager or too silent, style guidance overriding structural address rules, prompt-flow drift, and accidentally moving image description after relevance.
- Acceptance criteria: group relevance can see only bounded user engagement guidance on LLM paths; private relevance payloads remain unchanged; L3 content-anchor receives `interaction_style_context`; the relevance module no longer owns vision description; RAG and dialog still do not consume raw style images.

## Context

`interaction_style_images` stores sanitized user and group-channel style overlays. `build_interaction_style_context(...)` currently loads user style for L3 and group-channel style only for group chat.

The current path is optional multimedia descriptor -> relevance -> persona supervisor -> stage 0 message decontextualizer -> RAG -> cognition -> dialog. If relevance returns `should_respond=false`, the service graph exits before L3, so L3 cannot use `user_style_image.engagement_guidelines` to admit a group message already filtered out by relevance.

Read-only planning data pulled from MongoDB found 2 active user-style docs and 5 active group-channel docs with non-empty engagement guidelines. User-style examples are mostly per-user response moves: setting small interaction conditions, using displayed new information for challenge or feedback, and responding actively to clear user requests. These belong primarily after response admission. In group chat, a narrow subset can help relevance decide low-noise ambiguous participation, but it must not override group structural-address safety.

Current source ownership is also misleading: `nodes/relevance_agent.py` owns both relevance and image description. This plan renames the module to `persona_relevance_agent.py` and moves the vision descriptor code into `persona_supervisor2_msg_decontexualizer.py`. The service graph order must stay unchanged: image description still runs before persona relevance so relevance can see prompt-safe image summaries.

## Mandatory Skills

- `local-llm-architecture`: load before prompt, graph, or payload changes.
- `py-style`: load before editing Python.
- `cjk-safety`: load before moving or editing CJK prompt strings in Python.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Keep the response path bounded. Do not add a new LLM call.
- Preserve graph order: multimedia description remains before persona relevance. Do not move full message decontextualization before relevance.
- Do not feed raw database documents to prompts. Only prompt-facing projections are allowed.
- Move existing CJK prompt blocks by byte-preserving refactor mechanics, then run Python syntax checks before continuing.
- Any edited LLM prompt must be rewritten in-place so the procedure, evidence order, input format, examples, and output rules remain coherent. Do not append the new instruction as a trailing block.
- Do not change RAG, dialog, consolidation, reflection extraction, adapters, scheduler, persistence schemas, service API contracts, or deterministic text parsing behavior.
- Group relevance may use user engagement guidance only as a soft tie-breaker after structural evidence makes the message eligible for LLM judgment. It must not override reply metadata, direct-address metadata, `group_attention` hard skips, boundary decisions, `logical_stance`, active commitments, or open-loop obligations.
- Private relevance payloads must stay unchanged.
- L3 content-anchor may use engagement guidance only for `[SOCIAL]`, `[PROGRESSION]`, and follow-up shape; it must not rewrite `[DECISION]`, invent `[FACT]`, or answer unsupported questions.
- Prompt behavior must be verified with real LLM tests run one at a time and inspected one at a time. Pytest pass alone is not enough evidence.
- After context compaction or major checklist sign-off, reread this whole plan before continuing. Before final sign-off, run and record the independent code review gate.

## Must Do

- Rename `nodes/relevance_agent.py` to `nodes/persona_relevance_agent.py`.
- Move `_VISION_DESCRIPTOR_PROMPT`, `_vision_descriptor_llm`,
  `multimedia_descriptor_agent`, and private descriptor helpers into
  `persona_supervisor2_msg_decontexualizer.py`.
- Update service wiring to import `relevance_agent` from
  `persona_relevance_agent` and `multimedia_descriptor_agent` from the
  decontextualizer module.
- Add one DB-layer projection helper that returns a compact relevance-facing
  user engagement context for one `global_user_id`.
- Wire that projection into group persona relevance LLM payloads only, after
  existing deterministic group skip checks.
- Update the noisy group relevance prompt with bounded instructions for the new
  user engagement context.
- Add `interaction_style_context` to the L3 content-anchor prompt payload and
  prompt contract.
- Update focused tests for the module rename, vision ownership move, DB
  projection, group relevance payload shape, private relevance omission, and
  content-anchor payload shape.

## Deferred

- Do not change `group_channel_style_image` consumers.
- Do not add a new relevance sub-agent, routing stage, prompt evaluator, retry
  loop, feature flag, compatibility shim, alternate prompt, or fallback path.
- Do not change group attention thresholds or existing hard skip behavior.
- Do not migrate or rewrite existing `interaction_style_images` documents.
- Do not let dialog consume raw style images or `interaction_style_context`.
- Do not redesign `engagement_guidelines` extraction in reflection.

## Cutover Policy

Overall strategy: bigbang for in-process behavior; compatible for stored data.

| Area | Policy | Instruction |
|---|---|---|
| Relevance module rename | bigbang | Rename the module and update imports/tests. Do not keep a `relevance_agent.py` compatibility wrapper. |
| Vision ownership move | bigbang | Move descriptor code into the decontextualizer module while preserving graph order. |
| DB projection helper | compatible | Existing documents remain valid. Missing, inactive, empty, or invalid overlays return empty context. |
| Group relevance payload | bigbang | Add `user_engagement_context` directly to noisy-group LLM payloads only when the LLM is invoked. |
| Private relevance payload | compatible | Preserve current shape. Do not attach user engagement context. |
| L3 content-anchor payload | bigbang | Add `interaction_style_context` directly to the existing content-anchor payload. |
| RAG and dialog | compatible | Preserve current non-consumer boundary. |

Cutover enforcement: follow the selected policy for each area. Changing cutover
policy requires user approval.

## Agent Autonomy Boundaries

- Local variable names are flexible only when the contracts here remain intact.
- Do not edit outside `Change Surface` unless a direct compile/import blocker
  requires it; record any such change in `Execution Evidence`.
- Do not perform unrelated cleanup, broad refactors, dependency changes, or
  prompt rewrites beyond the prompt clauses named here.
- If code and plan disagree, preserve the plan intent and report the mismatch.
  If an instruction is impossible, stop and report the blocker.

## Target State

Source ownership:

```text
nodes/persona_relevance_agent.py
  owns: should_respond, reason_to_respond, use_reply_feature,
        channel_topic, indirect_speech_context, group_attention

nodes/persona_supervisor2_msg_decontexualizer.py
  owns: text decontextualization and current-turn image description helpers
```

Runtime order stays:

```text
multimedia_descriptor_agent -> relevance_agent -> persona_supervisor2
```

For group relevance LLM calls, the human payload includes:

```python
"user_engagement_context": {
    "engagement_guidelines": list[str],
    "confidence": str,
}
```

The field is omitted from private relevance payloads and may be empty in group
payloads when no active user style exists.

For L3 content-anchor calls, the human payload includes the existing
`interaction_style_context` shape used by L3 style and preference adapter.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Module name | Use `persona_relevance_agent.py` | Makes the module's persona-gating ownership explicit. |
| Relevance callable | Keep function name `relevance_agent` | Avoids unnecessary graph/service contract churn. |
| Vision move | Move code into decontextualizer module, preserve graph order | Vision description creates prompt-safe decontextualized media evidence, but relevance still needs it before gating. |
| User-style relevance consumer | Group relevance receives compact user engagement context only on LLM paths | Relevance is the only stage that can prevent L3 from being skipped. |
| Private relevance | No change | Private messages are already addressed; user engagement style is response planning there. |
| L3 consumer | Add context to `l3_content_anchor_agent` | Engagement guidelines affect follow-up questions, social moves, and progression. |
| Dialog/RAG | No raw style context | Dialog consumes action directives; RAG retrieves evidence. |

## Contracts And Data Shapes

### DB Projection Helper

Add to `src/kazusa_ai_chatbot/db/interaction_style_images.py`:

```python
async def build_user_engagement_relevance_context(
    global_user_id: str,
) -> dict:
    """Build relevance-facing engagement guidance for one user."""
```

Return `{"engagement_guidelines": list[str], "confidence": str}`.

Rules:

- Read only `user_style_image` for the supplied `global_user_id`.
- If missing, inactive, empty, or invalid, return
  `{"engagement_guidelines": [], "confidence": ""}`.
- Use the same overlay validation rules as runtime L3 style context.
- Return at most `RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT` engagement
  guideline strings.
- Do not include other guideline categories, source run ids, revision,
  timestamps, or style image ids.
- Export the helper from `src/kazusa_ai_chatbot/db/__init__.py`.

### Persona Relevance Payload

In `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`:

- Import `build_user_engagement_relevance_context`.
- Keep existing deterministic third-party reply, chaotic-noise, and
  medium/high-noise no-continuity short-circuits before loading the context.
- Only when `channel_type == "group"` and the LLM will be invoked, call the DB
  helper with `state["global_user_id"]`.
- If the optional DB helper raises `DatabaseOperationError`, log the exception
  text and use the empty context shape; do not catch internal state/key errors.
- Add the result as top-level `human_data["user_engagement_context"]`.
- Do not add it for private relevance.

Rewrite `_RELEVANCE_SYSTEM_NOISY_PROMPT` as an integrated prompt update: define `user_engagement_context` in the evidence hierarchy, generation path, input format, and examples where needed. It is sanitized per-user engagement guidance and only a soft tie-breaker after structural address, `group_attention`, and history continuity make the message eligible for LLM judgment. It cannot prove address, override reply-to-other metadata, override group-noise rules, or turn third-party conversation into a bot summons.

### Vision Descriptor Move

Move the existing descriptor prompt, LLM, parser helpers, persistence update,
prompt-message-context refresh, and cognitive-episode media replacement into
`persona_supervisor2_msg_decontexualizer.py`. Keep the callable name
`multimedia_descriptor_agent(state: IMProcessState) -> IMProcessState`. Do not
merge it into `call_msg_decontexualizer(...)`; service graph wiring still calls
it before persona relevance.

### L3 Content-Anchor Payload

In `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`:

- Add `interaction_style_context` to `call_content_anchor_agent(...)`'s human
  payload using the same fallback as `call_style_agent(...)`.
- Update `_CONTENT_ANCHOR_AGENT_PROMPT` input format and rules.

Rewrite `_CONTENT_ANCHOR_AGENT_PROMPT` as an integrated prompt update: add `interaction_style_context` to the input format and weave the rule into the parsing steps and anchor rules. Use `user_style.engagement_guidelines` only as soft guidance for `[SOCIAL]`, `[PROGRESSION]`, and follow-up shape. In group chat, apply `user_style` first and existing `group_channel_style` second according to `application_order`. Do not use interaction style to change `[DECISION]`, override `logical_stance`, invent facts, answer unsupported questions, or reopen avoided loops.

## LLM Call And Context Budget

No new LLM calls are added. The existing vision descriptor call count and graph
position remain unchanged, only its source module changes. Group relevance keeps
the existing relevance LLM call count and adds at most
`RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT` short user engagement strings plus
confidence on group LLM paths only. Private relevance has no payload change.
L3 content-anchor keeps the existing cognition LLM call and adds the compact
interaction-style context already used by other L3 agents, projected with
`L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT` per source.

If implementation would require a new response-path LLM call or moving message
decontextualization before relevance, stop and request user approval.

## Change Surface

### Rename

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  -> `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`
- `tests/test_relevance_agent.py`
  -> `tests/test_persona_relevance_agent.py`

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`: remove vision
  descriptor code; add user engagement relevance context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  own and export `multimedia_descriptor_agent` and descriptor helpers.
- `src/kazusa_ai_chatbot/service.py`: import relevance and multimedia
  descriptor from their new owning modules.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`: add the
  relevance-facing user engagement projection helper.
- `src/kazusa_ai_chatbot/db/__init__.py`: export the helper.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`: pass
  `interaction_style_context` to content-anchor; update its prompt.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: route
  content-anchor after the existing style-context loader so the loaded context
  actually reaches the content-anchor node.
- `src/scripts/run_touched_llm_regression.py`: update touched prompt module
  paths for renamed relevance and moved vision descriptor ownership.
- `tests/test_persona_relevance_agent.py`: update relevance tests for renamed
  module and new payload.
- `tests/test_msg_decontexualizer.py`: move vision descriptor tests here.
- `tests/test_cognition_interaction_style_context.py`: add or extend
  content-anchor payload test.
- `tests/test_cognition_live_llm_prompt_contracts.py`: add real LLM
  content-anchor style-context positive and guard cases.
- `tests/test_multi_source_cognition_image_input.py`: update relevance import
  path.
- `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`:
  update multimedia descriptor import/patch path.
- `tests/test_relevance_sensitivity_live_llm.py`: update import path and add
  real LLM false-negative and false-positive group relevance cases.
- `development_plans/README.md`: keep this plan row active.

### Keep

- `src/kazusa_ai_chatbot/brain_service/graph.py`: keep node order and graph
  semantics unchanged.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: no direct style-image
  consumption.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`: no
  interaction-style exposure.
- Adapter modules, service request contracts, RAG, reflection, scheduler, and
  persistence schemas.

## Implementation Order

1. Rename the relevance module and test file, update imports, and confirm existing relevance tests still target the same callable names.
2. Move vision descriptor code and tests into the decontextualizer module/test file; update service import while preserving graph order.
3. Update script and multimodal regression import paths for renamed/moved ownership.
4. Add focused tests for `build_user_engagement_relevance_context(...)`; expected before implementation: import or attribute failure.
5. Implement the DB helper and export it.
6. Add persona relevance tests covering group payload inclusion, private omission, and deterministic group skip preserving no-LLM behavior.
7. Implement relevance import, load order, payload field, and integrated noisy prompt rewrite.
8. Add real LLM relevance false-negative and false-positive cases; run and inspect each case individually, then revise only the approved prompt surface if the live output violates the contract.
9. Add content-anchor payload test; expected before implementation: fake content-anchor payload lacks `interaction_style_context`.
10. Implement content-anchor payload and integrated prompt rewrite.
11. Add real LLM content-anchor positive and guard cases; run and inspect each case individually, then revise only the approved prompt surface if needed.
12. Run focused tests, static greps, regression tests, real LLM prompt tests, and independent code review.

## Progress Checklist

- [x] Stage 1 - module rename and vision ownership move complete.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py tests\test_msg_decontexualizer.py -q`.
  - Evidence/sign-off: record import-path updates, graph-order preservation, and test output before ticking.
- [x] Stage 2 - DB projection contract complete.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py -q`.
  - Evidence/sign-off: record failing-before/pass-after result before ticking.
- [x] Stage 3 - group persona relevance integration complete.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py -q`.
  - Evidence/sign-off: record payload shape and test output before ticking.
- [x] Stage 4 - L3 content-anchor integration complete.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_cognition_interaction_style_context.py -q`.
  - Evidence/sign-off: record payload shape and test output before ticking.
- [x] Stage 5 - real LLM prompt validation complete.
  - Verify: run every command in `Real LLM Prompt Tests` individually with `-s`.
  - Evidence/sign-off: record false-negative/false-positive judgment and log artifact paths before ticking.
- [x] Stage 6 - regression verification and independent code review complete.
  - Verify: all commands in `Verification`, then review gate.
  - Evidence/sign-off: record command outputs, review findings, fixes, reruns, residual risks, and approval status before ticking.

## Verification

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_interaction_style_context.py -q`

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_cognition_preference_adapter.py tests\test_persona_supervisor2.py tests\test_multi_source_cognition_image_input.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`

### Real LLM Prompt Tests

Run each command alone with `-s`, inspect the emitted log/artifact, and record the model behavior judgment before running the next case:

- `venv\Scripts\python.exe -m pytest tests\test_relevance_sensitivity_live_llm.py::test_noisy_relevance_live_uses_user_engagement_without_false_negative -q -s`; expected: low-noise eligible group message with relevant user engagement guidance returns `should_respond=true`.
- `venv\Scripts\python.exe -m pytest tests\test_relevance_sensitivity_live_llm.py::test_noisy_relevance_live_does_not_turn_engagement_into_false_positive -q -s`; expected: low-noise third-person or non-addressed group talk remains `should_respond=false` even with active engagement guidance.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_content_anchor_live_uses_engagement_for_progression_without_false_negative -q -s`; expected: content-anchor uses engagement guidance for `[SOCIAL]` or `[PROGRESSION]` when the upstream decision already allows engagement.
- `venv\Scripts\python.exe -m pytest tests\test_cognition_live_llm_prompt_contracts.py::test_content_anchor_live_does_not_let_engagement_override_decision_false_positive -q -s`; expected: content-anchor preserves `[DECISION]` and does not invent `[FACT]` or unsupported `[ANSWER]` when engagement guidance encourages participation.

### Static Greps

- `rg -n "kazusa_ai_chatbot\.nodes\.relevance_agent|from kazusa_ai_chatbot.nodes import relevance_agent|nodes.relevance_agent" src tests`
  - Expected: zero matches.
- `rg -n "_VISION_DESCRIPTOR_PROMPT|_vision_descriptor_llm|multimedia_descriptor_agent" src\kazusa_ai_chatbot\nodes\persona_relevance_agent.py`
  - Expected: zero matches.
- `rg -n "interaction_style_context|user_style_image|group_channel_style_image" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py`
  - Expected: no production RAG exposure matches.
- `rg -n "user_engagement_context" src\kazusa_ai_chatbot tests`
  - Expected: matches only in `persona_relevance_agent.py`, relevant tests, and
    this plan unless directly justified in `Execution Evidence`.

### Compilation

- `venv\Scripts\python.exe -m compileall src\kazusa_ai_chatbot tests`

## Independent Plan Review

Completed on 2026-05-11 from a fresh-review posture against the registry,
current graph/source/test contracts, and plan-writing gates. Blockers found
and fixed before approval: missing `cjk-safety`, missing script and multimodal
test change surfaces, and missing optional DB-read failure behavior. Residual
unrelated registry/worktree drift is outside this plan. Approval: approved for
execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Review style, prompt safety, test style, ownership boundaries, hidden fallback
paths, prompt/RAG leaks, compatibility wrappers, blast radius, and alignment
with this plan. Persona relevance must not own vision description, and
decontextualizer must not move after relevance. Fix findings only inside the
approved change surface; otherwise stop and update the plan or request approval.

Completed on 2026-05-11. Findings fixed:

- The optional relevance DB projection handled invalid overlays but not
  malformed stored documents; added a red-first test and narrowed the helper
  fallback to projection/schema errors (`KeyError`, `TypeError`, `ValueError`).
- Prompt edits after config changes needed mandatory real LLM validation; reran
  false-negative and false-positive relevance tests one at a time and inspected
  durable traces.
- The interaction-style guideline limits were implicit; split them into
  storage, L3 projection, and relevance projection config constants with
  validation and focused deterministic tests.

No remaining plan-scope findings. Residual unrelated worktree drift in other
development-plan files remains outside this implementation.

## Acceptance Criteria

This plan is complete when:

- `nodes/relevance_agent.py` no longer exists and imports use
  `nodes/persona_relevance_agent.py`.
- The persona relevance module owns only response eligibility and no vision
  descriptor prompt, LLM, helper, or multimedia descriptor callable.
- `multimedia_descriptor_agent` is owned by
  `persona_supervisor2_msg_decontexualizer.py` and still runs before persona
  relevance.
- `user_style_image.engagement_guidelines` has a bounded group relevance
  consumer for LLM-judged group messages.
- Existing group relevance hard skips still run before the new context is
  loaded or applied.
- Private relevance payloads remain unchanged.
- L3 content-anchor receives `interaction_style_context`.
- RAG and dialog still do not consume raw style images or interaction-style
  context.
- All verification commands pass, or any blocked command is recorded with a
  concrete reason and approved residual risk.
- Real LLM false-negative and false-positive prompt tests are run one at a
  time, inspected, and judged acceptable with recorded artifacts.
- Independent code review is completed and recorded.
- The plan is archived under `development_plans/archive/completed/short_term/`.

## Execution Evidence

- Implementation not started. Independent plan review approved this execution
  contract on 2026-05-11.
- 2026-05-11 amendment: user-required prompt rewrite rule and real LLM
  false-negative/false-positive validation gates added before implementation.
- 2026-05-11 Stage 1: renamed relevance module/test, moved vision descriptor
  ownership to decontextualizer, fixed stale decontextualizer test fixture, and
  updated import paths. Verification:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_relevance_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py tests\test_msg_decontexualizer.py tests\test_persona_relevance_agent.py`
  passed; `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py tests\test_msg_decontexualizer.py -q`
  passed with 34 tests.
- 2026-05-11 Stage 2: added DB projection tests, observed expected missing
  `build_user_engagement_relevance_context` failures, then implemented and
  exported the helper. Verification:
  `venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py -q`
  passed with 11 tests.
- 2026-05-11 Stage 3: added red-first group relevance payload/fallback/skip
  tests; initial targeted run failed on missing `user_engagement_context` as
  expected. Integrated user engagement context only into the group LLM payload,
  after deterministic skips, and rewrote the noisy relevance prompt evidence
  hierarchy to treat engagement as a soft post-eligibility signal. Verification:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_relevance_agent.py tests\test_persona_relevance_agent.py`
  passed; `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py -q`
  passed with 23 tests.
- 2026-05-11 Stage 4: added red-first content-anchor payload and subgraph
  plumbing tests; initial targeted run failed because content-anchor lacked
  `interaction_style_context` and ran in parallel with the loader. Added the
  style context to the content-anchor payload with the existing fallback,
  rewrote the content-anchor prompt rules around `[SOCIAL]`/`[PROGRESSION]`,
  and routed content-anchor after the existing loader. Verification:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py tests\test_cognition_interaction_style_context.py`
  passed; `venv\Scripts\python.exe -m pytest tests\test_cognition_interaction_style_context.py -q`
  passed with 4 tests.
- 2026-05-11 Stage 5: added real LLM false-negative/false-positive tests for
  group relevance and content-anchor. Initial plan command without marker
  override was deselected by project `pytest.ini`; reran each case individually
  with `-m live_llm -q -s`. Relevance false-negative passed with
  `should_respond=true`; trace:
  `test_artifacts\llm_traces\relevance_sensitivity_live_llm__user_engagement_followup_false_negative.json`.
  Relevance false-positive passed with `should_respond=false`; trace:
  `test_artifacts\llm_traces\relevance_sensitivity_live_llm__user_engagement_third_person_false_positive.json`.
  Content-anchor positive passed with `[SOCIAL]` and `[PROGRESSION]` follow-up
  anchors; trace:
  `test_artifacts\llm_traces\cognition_prompt_contracts_live__prompt_contracts.content_anchor.engagement_progression_positive.json`.
  Content-anchor guard passed, preserving refusal and avoiding invented dosage
  facts; trace:
  `test_artifacts\llm_traces\cognition_prompt_contracts_live__prompt_contracts.content_anchor.engagement_decision_guard.json`.
- 2026-05-11 Stage 6: focused verification passed:
  `venv\Scripts\python.exe -m pytest tests\test_persona_relevance_agent.py -q`
  passed with 23 tests;
  `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py -q`
  passed with 13 tests;
  `venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py -q`
  passed with 12 tests after the review fix;
  `venv\Scripts\python.exe -m pytest tests\test_cognition_interaction_style_context.py -q`
  passed with 4 tests. Regression verification initially failed only on the
  content-anchor prompt fingerprint because the prompt was intentionally
  rewritten; updated that approved fingerprint and reran
  `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_cognition_preference_adapter.py tests\test_persona_supervisor2.py tests\test_multi_source_cognition_image_input.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`,
  which passed with 42 tests. Static greps passed: old relevance import path
  returned no matches, persona relevance has no vision descriptor symbols, RAG
  has no interaction-style exposure, and `user_engagement_context` is scoped to
  persona relevance/tests. `venv\Scripts\python.exe -m compileall src\kazusa_ai_chatbot tests`
  passed. `git diff --check` initially found one trailing-whitespace line in
  the moved decontextualizer code; after removal, `git diff --check` passed
  with only existing CRLF conversion warnings. Independent code review approved
  after fixing the malformed stored style document fallback.
- 2026-05-11 review follow-up: added positive integer validation for
  interaction-style config limits and reran mandatory live relevance prompt
  checks after prompt text changed. The false-positive relevance guard initially
  failed because the model treated third-person subject questions as direct
  address; rewrote the noisy relevance prompt in-place to clarify that boundary.
  Reruns passed and traces were inspected:
  `test_artifacts\llm_traces\relevance_sensitivity_live_llm__user_engagement_third_person_false_positive__20260511T082101559884Z.json`
  and
  `test_artifacts\llm_traces\relevance_sensitivity_live_llm__user_engagement_followup_false_negative__20260511T082119848396Z.json`.
- 2026-05-11 final config cleanup: made all interaction-style prompt/storage
  limits explicit in config:
  `INTERACTION_STYLE_STORAGE_GUIDELINES_PER_FIELD_LIMIT=5`,
  `L3_INTERACTION_STYLE_GUIDELINES_PER_FIELD_LIMIT=5`, and
  `RELEVANCE_USER_ENGAGEMENT_GUIDELINES_LIMIT=3`. Added red-first config and L3
  projection tests. Final verification passed:
  `venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_interaction_style_images.py tests\test_persona_relevance_agent.py -q`
  passed with 58 tests; `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\db\interaction_style_images.py src\kazusa_ai_chatbot\nodes\persona_relevance_agent.py tests\test_config.py tests\test_interaction_style_images.py`
  passed; `git diff --check` passed with only existing CRLF conversion
  warnings.
- 2026-05-11 closure: plan marked completed and moved to completed archive.
