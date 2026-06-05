# qq face projection empty input guard bugfix plan

## Summary

- Goal: preserve QQ face-only and inline-face messages as semantic prompt text, and stop still-empty no-content turns from crashing the resolver graph.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for QQ face projection and empty-content graph guard.
- Highest-risk areas: changing `body_text` semantics too broadly, accidentally routing QQ faces through multimedia/image-observation, and suppressing valid image-only attachment turns.
- Acceptance criteria: `[CQ:face,id=344]` persists as `<image>大怨种表情</image>` in adapter `body_text`; inline QQ faces keep their position; remaining empty no-content turns are persisted but do not invoke the graph; valid attachment/image-observation turns keep the existing media path.

## Context

The confirmed 2026-06-05 incident was a QQ face-only message:

```text
raw_wire_text = "[CQ:face,id=344]"
body_text = ""
attachments = []
```

The QQ adapter stripped all non-mention CQ markers through the generic
`_CQ_ANY_PATTERN` replacement. Because `face` is user-authored visible content,
not mere transport syntax, the adapter erased the only semantic input. The
turn then reached `stage_1_goal_resolver`, where resolver initialization
rejected empty `decontexualized_input`.

This is related to, but distinct from, the completed image-only resolver
bugfix. The earlier fix covers empty text with a usable `image_observation`.
This incident had no image attachment, no generated image observation, and no
semantic text because the adapter stripped the QQ face before brain intake.

The architectural boundary remains unchanged:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
```

Adapters normalize platform syntax into `MessageEnvelope.body_text` and typed
fields. The brain service must not parse QQ CQ syntax or know QQ face ids.

## Failure Modes And Required Behavior

| Failure mode | Required behavior | Reason |
|---|---|---|
| Known QQ face id is present in the table. | Render the exact mapped description as `<image>{description}</image>`. | Preserve visible authored content with the strongest available local semantic label. |
| QQ face id is not present in the table. | Render `<image>表情</image>`; do not drop the segment, guess a meaning, expose the id in `body_text`, or raise. | Unknown faces are still visible expressions, but the adapter must not invent semantics. |
| QQ face segment is a syntactically closed `[CQ:face...]` segment but lacks a usable `id`. | Render `<image>表情</image>` for the `face` segment. | A closed face segment still signals an expression more safely than an empty message. |
| QQ face segment has extra CQ parameters or the `id` parameter is not first. | Parse the `id` when available and ignore unrelated parameters. | NapCat/OneBot segment parameters may vary; adapter output should not depend on parameter ordering. |
| Multiple QQ faces appear in one message. | Render each face in order with spacing that keeps it separate from adjacent text and other faces. | QQ faces can be inserted anywhere; order and adjacency are authored content. |
| QQ face appears in reply excerpt projection. | Apply the same face projection to the excerpt because `project_qq_semantic_text()` owns both body and excerpt sanitization. | Reply excerpts can carry visible face-only context and should not be silently emptied. |
| Static table description contains `<`, `>`, or `&`. | Escape the description before placing it inside `<image>...</image>`. | Local constants are maintenance data; prompt boundaries must remain structurally safe. |
| Static table description is wrong or stale. | Keep behavior deterministic and update the table through a separate scoped mapping change; do not add runtime guessing or LLM interpretation. | Wrong local metadata is a maintenance issue, not a reason to make the live path non-deterministic. |
| QQ face projection still results in empty body for a future unhandled case. | Persist the user row, skip graph invocation, return no messages, and record a non-error pipeline event with `status="completed"` and `final_outcome="no_content"`. | The brain service must fail closed on no-content input rather than surfacing a resolver exception. |
| Empty text has a real image/audio description or image payload. | Do not suppress it; keep the existing multimedia/image-observation path. | The brain guard must not regress the completed image-only fix. |
| Empty survivor has collapsed non-empty content. | Do not suppress it; use the existing collapsed-content path. | Queue coalescing can create a meaningful combined turn even when one row is empty. |
| Reply-only message has no text, no face, and no media. | Treat it as no-content and skip the graph. | A reply marker alone is transport/addressing metadata, not enough semantic input for cognition. |

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, archiving, or signing off this plan.
- `local-llm-architecture`: load before changing adapter-to-brain contracts, graph routing, prompt-facing projection, cognition, or resolver behavior.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`; implementation also requires explicit user approval.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, docs, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Keep QQ protocol handling inside the QQ adapter boundary.
- Do not add QQ-specific logic to brain service, resolver, cognition, RAG, dialog, persistence, or prompts.
- Do not add a new message envelope field for QQ faces.
- Do not represent QQ faces as real attachments and do not trigger the multimedia descriptor for QQ system faces.
- Do not add a new LLM call, retry loop, runtime lookup, network fetch, or database lookup for QQ face interpretation.
- The QQ face lookup table must store final LLM-facing image descriptions, not apply a universal `{official_name}表情` formula.
- New Python CJK string literals must use single-quoted delimiters unless an escape sequence is required.
- After editing Python files with CJK literals, run `py_compile` against the production and test files touched by this plan.
- The brain-side guard must only suppress turns that have no prompt-usable authored text and no prompt-usable media description.
- Preserve valid image-only attachment behavior from `resolver_image_only_empty_input_bugfix_plan.md`.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do

- Add adapter-owned QQ face projection in `src/adapters/napcat_qq_adapter.py` before the generic CQ-stripper.
- Preserve QQ face inline position in `MessageEnvelope.body_text`.
- Render known QQ faces as inline prompt image blocks with exact static descriptions.
- Add at least the confirmed mapping:

```text
344 -> <image>大怨种表情</image>
```

- Store mapping values as final descriptions, for example `344: "大怨种表情"`, then render with `<image>{description}</image>`.
- Use a generic fallback for unknown QQ face ids:

```text
<image>表情</image>
```

- Keep `raw_wire_text` unchanged for audit and replay.
- Project malformed face segments and face segments without a known id to the same generic fallback.
- Parse QQ face `id` from any CQ parameter position within a syntactically closed face segment.
- Apply QQ face projection to reply excerpts because they use the same projection helper.
- Keep image/file/video CQ codes stripped from `body_text` unless they are represented through existing attachment handling.
- Add a brain-service guard that detects no-content turns after persistence and before graph invocation.
- For no-content turns, complete with an empty `ChatResponse` and do not call `_graph.ainvoke`.
- Record the no-content turn as a non-error pipeline event with `status="completed"`, `final_outcome="no_content"`, and `severity="info"`.
- Add focused tests for QQ face projection, inline placement, unknown fallback, malformed face fallback, segment-list face conversion, reply excerpt face projection, and empty no-content graph suppression.
- Add focused tests for extra face parameters, non-first `id` parameters, multiple adjacent faces, and escaping mapped descriptions that contain `<`, `>`, or `&`.

## Deferred

- Do not build a complete QQ face ontology in this plan.
- Do not fetch QQ face images or assets at runtime.
- Do not add custom HTML tags besides the existing `<image>...</image>` convention.
- Do not explain QQ protocol, face ids, or slang meanings to cognition.
- Do not add a fake `image/*` attachment for QQ system faces.
- Do not change attachment description projection, multimedia descriptor prompts, image observation contracts, RAG, dialog, consolidation, database schema, queue pruning policy, or persona graph topology.
- Do not backfill historical conversation rows in this plan.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| QQ `face` CQ projection | bigbang | Replace stripping with inline `<image>{description}</image>` rendering in `body_text`. |
| QQ face mapping | bigbang | Use adapter-local static final-description lookup values. Do not compute all descriptions from official labels. |
| Unknown QQ face ids | bigbang | Render `<image>表情</image>` instead of dropping the segment. |
| Brain empty no-content turns | bigbang | Persist the user row, skip graph invocation, record `final_outcome="no_content"`, and return no messages. |
| Existing image attachment path | compatible | Preserve the existing attachment, multimedia descriptor, and image-observation behavior. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- For bigbang areas, rewrite old behavior directly instead of adding compatibility branches.
- For compatible areas, preserve only the compatibility surfaces explicitly listed in this plan.
- Any change to this cutover policy requires user approval before implementation.

## Target State

Known QQ face-only input:

```text
raw_wire_text = "[CQ:face,id=344]"
body_text = "<image>大怨种表情</image>"
attachments = []
```

Known QQ face inside text:

```text
raw_wire_text = "我[CQ:face,id=344]服了"
body_text = "我 <image>大怨种表情</image> 服了"
```

Unknown QQ face:

```text
raw_wire_text = "[CQ:face,id=999999]"
body_text = "<image>表情</image>"
```

QQ face with extra parameters:

```text
raw_wire_text = "[CQ:face,foo=bar,id=344]"
body_text = "<image>大怨种表情</image>"
```

Multiple adjacent QQ faces:

```text
raw_wire_text = "[CQ:face,id=344][CQ:face,id=999999]"
body_text = "<image>大怨种表情</image> <image>表情</image>"
```

Still-empty no-content turn:

```text
body_text = ""
attachments = []
multimedia_input = []
graph invoked = false
response.messages = []
```

Valid image-only attachment turn:

```text
body_text = ""
attachments = [{"media_type": "image/jpeg", ...}]
multimedia_input contains image row
existing multimedia/image-observation path remains active
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix location for QQ face erasure | QQ adapter semantic projection. | QQ faces are platform-native visible content; adapters own platform syntax translation. |
| Prompt representation | Inline `<image>...</image>` in `body_text`. | This reuses the existing prompt-facing visual boundary and preserves inline placement. |
| Mapping value | Final static image description string. | Some official labels are not directly meaningful to an LLM; the table must carry the exact prompt label. |
| ID 344 mapping | `大怨种表情`. | This is the confirmed incident face and is concise, visual, and LLM-readable. |
| Unknown face behavior | `<image>表情</image>`. | Unknown faces should remain non-empty visible content without inventing meaning. |
| Attachments | Do not create fake attachment rows for QQ faces. | Fake `image/*` attachments would route through multimedia and lose inline position. |
| Brain fallback | Generic no-content no-op before graph invocation. | Even with adapter fixes, brain service should not graph-crash on valid empty envelopes from any adapter. |

## Contracts And Data Shapes

- `MessageEnvelope.body_text` remains a string containing user-authored semantic content and prompt-safe visual descriptions.
- `MessageEnvelope.raw_wire_text` remains the original CQ text or closest replay form.
- QQ face ids do not appear in `body_text`.
- QQ face descriptions are escaped before placement inside `<image>...</image>`.
- `attachments` remain reserved for real normalized media attachments.
- Unknown, missing, empty, non-numeric, or unusable QQ face ids all use the same generic prompt description: `表情`.
- A malformed face segment in this plan means a syntactically closed `[CQ:face...]` segment with no usable `id`; unclosed raw text is outside this plan's parser expansion.
- A mapped description value is the complete content inside the image boundary after escaping; code must not append a suffix such as `表情` at runtime.
- No-content guard predicate:

```text
empty stripped body_text
AND no collapsed combined_content
AND no multimedia_input row with non-empty description or image payload
AND no reply media description
```

The no-content guard must implement this predicate directly or delegate to a
helper with exactly this behavior.

## LLM Call And Context Budget

- Before: zero LLM calls for QQ face projection.
- After: zero LLM calls for QQ face projection.
- Before: empty no-content turns can reach relevance and resolver LLM stages.
- After: empty no-content turns stop before graph invocation, with zero graph/LLM calls.
- Prompt context increase is bounded by the number of QQ face segments in the current message and the static description length per segment.

## Change Surface

### Modify

- `src/adapters/napcat_qq_adapter.py`: add QQ face pattern, static final-description lookup, inline `<image>` rendering, and projection order before `_CQ_ANY_PATTERN`.
- `src/kazusa_ai_chatbot/service.py`: add generic no-content graph guard after user persistence and before `_graph.ainvoke`.
- `tests/test_adapter_envelope_normalizers.py`: add focused projection tests and update existing QQ face expectations.
- `tests/test_runtime_adapter_registration.py`: cover segment-list `face` events flowing to brain as inline `<image>` body text.
- `tests/test_service_input_queue.py`: cover no-content persisted turn that does not invoke the graph.
- `development_plans/README.md`: update lifecycle registry when plan status changes.

### Keep

- `src/kazusa_ai_chatbot/cognition_resolver/state.py`
- `src/kazusa_ai_chatbot/cognition_episode.py`
- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
- `src/kazusa_ai_chatbot/utils.py`
- RAG, cognition prompts, dialog prompts, consolidation, database schemas, and graph topology.

## Overdesign Guardrail

- Actual problem 1: QQ adapter erases a visible QQ face, leaving empty `body_text`.
- Minimal fix 1: adapter maps QQ face ids to inline `<image>` descriptions inside `body_text`.
- Actual problem 2: brain service can still graph-crash if an adapter supplies no text and no usable media.
- Minimal fix 2: brain service skips graph invocation for no-content turns after persistence.
- Rejected complexity: QQ ontology, runtime face asset lookup, fake attachments, new envelope fields, brain QQ handling, prompt changes, LLM interpretation, historical backfill, and generalized rich-message redesign.

## Agent Autonomy Boundaries

- Implementation freedom is limited to the files listed in `Change Surface`.
- Changes outside the listed files require user approval.
- Do not perform unrelated cleanup, formatting churn, dependency upgrades, broad refactors, prompt rewrites, or schema changes.
- If the plan and source code disagree, preserve this plan's ownership boundaries and report the discrepancy.
- If a required behavior cannot be implemented without changing a deferred area, stop and report the blocker.

## Implementation Order

1. Add failing adapter normalizer tests for:
   - `[CQ:face,id=344]` -> `<image>大怨种表情</image>`.
   - `我[CQ:face,id=344]服了` -> `我 <image>大怨种表情</image> 服了`.
   - unknown face id -> `<image>表情</image>`.
   - malformed face segment -> `<image>表情</image>`.
   - extra parameters and non-first `id` -> mapped description.
   - multiple adjacent faces -> ordered, separated image blocks.
   - injected mapping text containing `<`, `>`, or `&` -> escaped image-block content.
   - existing mention-plus-face case preserving mention and face.
2. Add failing adapter normalizer test for reply excerpt face projection.
3. Add failing runtime adapter test for segment-list `{"type": "face", "data": {"id": 344}}` reaching brain as `<image>大怨种表情</image>`.
4. Add failing service queue test where an empty `body_text`, no attachments, and no collapsed content persists the user row but does not invoke `_graph.ainvoke`.
5. Add service queue regression tests proving image-only attachments and collapsed non-empty content still invoke the graph.
6. Run the focused tests and record expected failures.
7. Implement QQ face projection in `src/adapters/napcat_qq_adapter.py`.
8. Run adapter-focused tests and fix only adapter-scope failures.
9. Implement the no-content graph guard in `src/kazusa_ai_chatbot/service.py`.
10. Run service queue tests and fix only guard-scope failures.
11. Run the full verification command set.
12. Run independent code review and address only plan-scope findings.
13. Update plan status, progress checklist, and execution evidence after successful implementation and review.

## Execution Model

- Execute only after user approval and status change to `approved` or `in_progress`.
- Parent owns orchestration, tests, verification, evidence, review remediation, lifecycle updates, and final sign-off.
- Native subagents are the default execution model required by the development-plan contract.
- If the user explicitly requests no subagents for this plan, use single-agent fallback execution and record that override in `Execution Evidence`.

## Progress Checklist

- [x] Stage 1 - tests established: add the adapter, runtime adapter, and service no-content tests; run focused tests and record expected failures. Sign-off: Codex, 2026-06-05.
- [x] Stage 2 - QQ face projection implemented: update adapter projection and pass adapter/runtime focused tests. Sign-off: Codex, 2026-06-05.
- [x] Stage 3 - brain no-content guard implemented: update service guard and pass service queue tests. Sign-off: Codex, 2026-06-05.
- [x] Stage 4 - verification complete: run all commands in `Verification` and record outputs. Sign-off: Codex, 2026-06-05.
- [x] Stage 5 - independent code review complete: record findings, remediations, reruns, and residual risk. Sign-off: Codex, 2026-06-05.
- [x] Stage 6 - lifecycle closed: update plan status and registry after approved completion. Sign-off: Codex, 2026-06-05.

## Verification

```powershell
venv\Scripts\python.exe -m pytest tests/test_adapter_envelope_normalizers.py -q
venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q
venv\Scripts\python.exe -m pytest tests/test_service_input_queue.py -q
venv\Scripts\python.exe -m py_compile src/adapters/napcat_qq_adapter.py src/kazusa_ai_chatbot/service.py tests/test_adapter_envelope_normalizers.py tests/test_runtime_adapter_registration.py tests/test_service_input_queue.py
git diff --check
```

## Independent Plan Review

The draft must pass review before approval. Review scope: RCA match, two failure modes represented separately, new-design failure modes covered, adapter/brain ownership boundary, no fake attachment path, no QQ-specific brain logic, exact ID 344 decision, unknown-id and malformed-face fallback behavior, static final-description lookup contract, and focused verification commands.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Review source diff, tests, plan alignment, CJK string safety, adapter projection order, unknown-face fallback, inline placement, graph no-op behavior, event logging outcome, and absence of QQ-specific brain/cognition/resolver logic.

## Acceptance Criteria

- `[CQ:face,id=344]` produces `body_text == "<image>大怨种表情</image>"`.
- Inline QQ faces preserve their authored position with readable spacing.
- Unknown QQ face ids produce `<image>表情</image>` instead of empty text.
- Malformed QQ face segments produce `<image>表情</image>` instead of empty text.
- Face segments with extra parameters or non-first `id` parameters still use the mapped description.
- Multiple adjacent QQ faces preserve order and remain separated.
- Face mapping descriptions containing `<`, `>`, or `&` are escaped inside the `<image>` block.
- QQ faces in reply excerpts are projected with the same rules as current-message body text.
- QQ face ids and CQ syntax remain absent from `body_text`.
- `raw_wire_text` still preserves original CQ syntax.
- Real image/file/video CQ segments are not misrepresented as QQ system faces.
- Empty no-content turns persist the user row, do not invoke `_graph.ainvoke`, and return no messages.
- Empty no-content turns record `status="completed"`, `final_outcome="no_content"`, and `severity="info"`.
- Valid image-only attachment turns still use the existing multimedia/image-observation path.
- No new LLM calls, prompts, envelope fields, database schema, fake attachments, or QQ-specific brain logic are added.
- All verification commands pass.

## Execution Evidence

- Plan creation: draft file created on 2026-06-05.
- Independent plan review: completed on 2026-06-05 without subagents because the user did not explicitly authorize delegation in this turn. Findings addressed in this revision: explicit extra-parameter/non-first-id behavior, bounded malformed-segment definition, multiple-face behavior, prompt-boundary escaping test coverage, CJK literal/py_compile requirements for test files, and acceptance criteria for the new-design failure modes.
- Implementation authorization: user explicitly requested workspace cleanup and plan execution with subagents on 2026-06-05; plan status moved to `in_progress`.
- Stage 1 evidence: removed unrelated pre-existing untracked QQ failure-mode test files per user cleanup request; `venv\Scripts\python.exe -m py_compile tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py tests\test_service_input_queue.py` exited 0; `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py tests\test_service_input_queue.py -q` failed as expected with 12 failures and 88 passes. Expected failures covered QQ face projection, missing face mapping constant, runtime NapCat face forwarding, and no-content graph suppression.
- Production-code subagent: completed on 2026-06-05. Changed `src/adapters/napcat_qq_adapter.py` and `src/kazusa_ai_chatbot/service.py` only; reported `py_compile`, focused pytest, and `git diff --check` passing.
- Stage 2 evidence: `venv\Scripts\python.exe -m pytest tests/test_adapter_envelope_normalizers.py -q` passed with 15 tests; `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q` passed with 52 tests.
- Stage 3 evidence: `venv\Scripts\python.exe -m pytest tests/test_service_input_queue.py -q` passed with 33 tests, including no-content graph suppression and image-only/collapsed-content regression coverage.
- Stage 4 evidence: `venv\Scripts\python.exe -m py_compile src/adapters/napcat_qq_adapter.py src/kazusa_ai_chatbot/service.py tests/test_adapter_envelope_normalizers.py tests/test_runtime_adapter_registration.py tests/test_service_input_queue.py` exited 0; `git diff --check` exited 0 with only LF-to-CRLF working-copy warnings.
- Independent code review: subagent `019e96c7-2ae0-70b2-8d92-e957166212d2` completed read-only review on 2026-06-05. Review found no production-code issues. One lifecycle finding noted incomplete checklist/evidence; this revision records stages 2-6 and completion evidence. Residual risk: static QQ face table intentionally maps only confirmed id `344`; unknown ids degrade to generic `<image>表情</image>` until future mapping changes.
- Lifecycle closure: plan status changed to `completed` and registry moved to completed bugfix archive on 2026-06-05.
