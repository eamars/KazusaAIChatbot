# rag2 mainline fusion recall quality plan

## Summary

- Goal: Fuse the useful current-branch private-memory recall behavior into the
  cleaner mainline RAG2 behavior, then fix shared RAG2 quality gaps without
  choosing one branch wholesale.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: bigbang for RAG2 prompt-facing language and
  evidence-format wording; compatible for existing `rag_result` top-level keys
  and supervisor trace; no database, index, cache, routing, graph, or model
  migration.
- Highest-risk areas: losing the current branch's private-memory recall gain,
  regressing mainline's cleaner group-chat synthesis, overfitting to the 20
  experiment cases, adding deterministic user-keyword matching, degrading
  retrieval through English translation, leaking raw ids or raw UTC timestamps,
  overvaluing source provenance instead of cognition usability, and changing
  cognition/dialog ownership. The fresh holdout adds one structural risk:
  treating flat retrieved conversation rows as authoritative evidence even
  when the user asks for a relation such as previous image, reply parent,
  follow-up line, or exact speaker attribution.
- Acceptance criteria: the fused implementation preserves mainline output
  cleanliness, preserves the current branch's private user-memory recall win,
  promotes direct candidate evidence when it answers the requested slot,
  produces Chinese prompt-facing RAG evidence consistently, evaluates quality
  from the cognition consumer's point of view, passes focused deterministic
  checks, and passes `debug-llm` review on the fixed 20-case branch-vs-mainline
  suite plus a fresh holdout set.

## Context

The branch-vs-mainline RAG2 review used 20 fixed real database-backed inputs:

```text
experiments/rag2_branch_quality/reports/branch_vs_mainline_quality_review_20260524.md
```

Raw artifacts:

```text
test_artifacts/rag2_branch_quality/current_branch_20260524/
test_artifacts/rag2_branch_quality/mainline_20260524/
```

Observed result:

| Area | Current branch | Mainline | Fusion decision |
|---|---|---|---|
| Private durable user memory | Better in specific cases, especially the user's cat memory and some continuity projection | Can miss user-specific facts and retrieve generic guidance memory | Preserve the current-branch private-memory recall behavior |
| Group-chat synthesis | Usable but sometimes noisier | Cleaner chronology, speaker attribution, and Chinese output | Keep mainline synthesis as the baseline |
| Negative-result answers | Often trace-like | More readable and directly consumable | Keep mainline-style negative-result formatting |
| Candidate promotion | Fails when exact answer remains an unconfirmed candidate | Same failure | Fix generically in evaluator/finalizer contract |
| Technical memory recall | Misses 4090/NVFP4 and Gemma 4 quantization | Same failure | Treat as shared coverage gap |
| Media recall | Misses metal embedded nut product image | Same failure | Treat as shared media-search gap |
| Output language | Mixed English and Chinese | Mostly Chinese | Make prompt-facing RAG output Chinese-first |

The earlier production RAG2 cognition-ready evidence work improved the
evidence contract, but the new experiment shows that branch-level changes must
be fused carefully. The current branch is not a general replacement for
mainline. The value is narrower: better scoped private-memory recall and
continuity projection. Mainline remains better for final evidence readability
and group-chat attribution.

The user has explicitly changed the language decision: previous discussion
favored English for internal RAG wording, but this plan supersedes that
decision. Prompt-facing RAG language must be Chinese to avoid translation drift
between Chinese chat data, retrieval queries, evidence summaries, and cognition
consumption.

The user has also clarified the evaluation basis: source/provenance returned by
RAG is useful for multi-turn RAG search, gap analysis, trace inspection, and
debugging, but it must not be treated as a primary quality metric for the
cognition-facing handoff. The quality question starts from the consumer:
whether cognition receives the facts, uncertainty, and concise context needed
to reason and generate. Source detail is supporting evidence only.

Fresh 20-case holdout review:

```text
experiments/rag2_branch_quality/reports/holdout20_current_vs_mainline_review_20260524.md
```

Raw artifacts:

```text
test_artifacts/rag2_branch_quality/holdout20_cases_20260524.jsonl
test_artifacts/rag2_branch_quality/holdout20_current_20260524/
test_artifacts/rag2_branch_quality/holdout20_mainline_20260524/
```

Holdout learning from the cognition-consumer view:

| Area | Learning | Design implication |
|---|---|---|
| Scoped user memory | Current branch materially improves several private user-memory cases where mainline returns no confirmed fact. | Preserve the scoped user-memory path and do not roll it back while fixing conversation evidence. |
| Conversation/media context | Current branch can find a seed row but miss the required adjacent image, reply parent, or follow-up line. | Conversation evidence needs a relational packet contract before final synthesis, not just more retrieved rows. |
| Group who-said-what | Current branch can identify a speaker but lose reply direction or exact line coverage. | Resolution for relational questions must require the requested relation, not only lexical or semantic coverage. |
| Too much information | Some current outputs contain the right fact plus unrelated side facts. | RAG must reduce evidence before cognition; downstream cognition must not be asked to filter broad evidence dumps. |
| Baseline behavior | Mainline often preserves more relational neighborhood but can expose large noisy evidence lists. | Borrow the relational strength, not the noisy flat-row output style. |

Structural conclusion: the next phase is not "retrieve more conversation
history." The core issue is the shape of accepted conversation evidence.
Flat retrieved rows, generic neighbor rows, and free-form final answers are too
weak as the primary cognition-facing contract for relational chat history.
RAG should first form a bounded evidence packet around a seed row, attach only
the required relations, then reduce that packet to one to three concise facts
before projection.

Bad or weak designs to remove or demote:

- Flat conversation rows must stop being treated as authoritative evidence for
  relational questions. Keep rows for trace/debug, but accepted public
  evidence should state the answered relation.
- Generic neighbor expansion must stop acting like a relevance boost. Neighbor
  rows should be promoted only when attached to a seed as a supported relation
  such as `previous_message`, `next_message`, or `reply_parent`.
- The LLM judge should not be the sole authority for `resolved=true` on
  relation-dependent questions. It may judge semantic relevance, but the
  packet must contain the required relation before the slot is accepted.
- The deterministic coverage checker is a guardrail, not a proof of relational
  correctness. It may confirm literal or value coverage, but it cannot prove
  reply direction, previous-image context, or topic-flow order.
- The free-form RAG `answer` should be treated as debug/fallback synthesis, not
  the strongest cognition input. Compact evidence facts and uncertainty should
  be primary.
- Broad retry/search-more behavior should not be the default recovery path.
  Prefer seed-specific relation fetch, packet reduction, or a single
  source-owned fallback when the first source returns nearby-but-not-answer
  evidence.

Relevant architecture documents:

- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `development_plans/archive/completed/short_term/rag2_cognition_ready_evidence_plan.md`
- `development_plans/reference/designs/rag2_recall_quality_experiment_plan.md`

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, updating, or
  signing off this plan.
- `local-llm-architecture`: load before changing RAG prompts, evaluator
  behavior, finalizer behavior, evidence projection, or cognition handoff.
- `debug-llm`: load before running live RAG2 comparisons, inspecting LLM output,
  or writing human-readable quality reports.
- `no-prepost-user-input`: load before adding any matching, filtering,
  promotion, or gating logic that could interpret user content.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files that contain Chinese prompt
  strings or Chinese test fixtures.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing
  implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute this plan while `Status` is `draft`. Implementation requires
  user approval and status `approved` or `in_progress`.
- Do not choose either branch wholesale. Use mainline behavior as the synthesis
  and formatting baseline, and port only the current-branch behavior that is
  proven to improve private durable user-memory recall.
- Do not import from `experiments.*`, `experiments/`, or `test_artifacts/` in
  production code or production tests. Experiment files are evidence and design
  references only.
- Do not add a new response-path LLM call. Use existing RAG2 stages and keep
  the current loop budget unless this plan is updated and approved.
- Do not change MongoDB databases, collections, indexes, vector indexes, Cache2
  schema, model routes, or adapter behavior.
- Do not change cognition, dialog, conversation progress, conversation graph,
  DAG, RAG3 routing, or consolidation behavior in this plan.
- RAG returns evidence. Cognition decides stance, boundaries, character
  judgment, and response goals. Dialog owns final visible wording.
- RAG must not shift evidence filtering responsibility to cognition. If RAG
  retrieves a large candidate set, RAG is responsible for reducing it into
  compact facts and explicit uncertainty before projection.
- Evaluate RAG output from the cognition consumer's point of view. Do not score
  an output higher merely because it includes source names, source ids, source
  collections, or provenance text. Score it higher only when the output gives
  cognition a more accurate, complete, compact, uncertainty-aware factual basis.
- The initializer remains semantic-only. It must not generate backend query
  parameters, MongoDB filters, index names, cache policy, or route mechanics.
- Stable code-level route identifiers must remain ASCII/English when they are
  deterministic contracts, such as `Memory-evidence:` or
  `Conversation-evidence:`. The semantic task text after the route token, all
  LLM-facing prompt instructions, generated search/refined queries, worker
  summaries, evaluator summaries, finalizer answers, and cognition-facing
  evidence text must be Chinese-first.
- Do not translate Chinese user input or Chinese evidence into English as an
  intermediate retrieval language. Preserve Chinese anchors and Chinese
  phrasing for search quality unless the source evidence itself is English.
- Do not add deterministic keyword anchors derived from user input. Deterministic
  code may validate stable non-user contracts such as route prefixes, enum
  values, schema keys, and source-type labels.
- Candidate promotion must be semantic and evidence-based. If a retrieved
  candidate directly answers the slot, the evaluator may promote it to
  confirmed evidence. Deterministic code may validate and sanitize the promoted
  shape, but must not decide semantic relevance from user-keyword matching.
- Conversation evidence promotion must respect relation requirements. For
  questions about previous images, replies, follow-up lines, topic flow, or
  who-said-what, a seed row is not sufficient unless the required relation is
  present and represented in the accepted evidence.
- Prompt-facing evidence must not expose raw adapter wire syntax such as
  `[CQ:...]`, raw source rows, raw storage ids, raw global user ids, embeddings,
  binary/base64 data, raw attachment URLs, raw UTC timestamps, or microsecond
  timestamps.
- Source/provenance detail belongs in trace/debug fields or short supporting
  evidence, not in the primary answer criteria. Public RAG text may include a
  short speaker/time cue only when it helps cognition understand who said what
  or when something happened.
- Prompt-facing timestamps must use configured local wall-clock time with
  second precision at most.
- Image evidence must use prompt-safe image descriptions or escaped
  `<image>...</image>` blocks where the current production projection requires
  that shape.

## Must Do

- Read the branch-vs-mainline review and raw artifacts before implementation.
- Diff the current branch against mainline and isolate the exact production
  changes that plausibly caused the private-memory recall win. Record the
  selected changes and rejected changes in execution evidence.
- Preserve mainline-style answer synthesis and group-chat formatting unless a
  targeted change is required by this plan.
- Port or recreate only the current-branch private-memory recall improvement
  that helps scoped user continuity, user-memory-unit retrieval, or memory
  projection.
- Convert RAG2 prompt-facing language to Chinese:
  - initializer prompt instructions and examples;
  - worker prompts where they produce search/refined queries or summaries;
  - evaluator and continuation prompt instructions;
  - finalizer prompt instructions;
  - public evidence labels and public answer wording.
- Keep deterministic route tokens stable. Localize the semantic task text after
  the route token and all prompt-facing language.
- Replace English evidence labels such as `Conclusion`, `Evidence summary`, and
  `Uncertainty` in prompt-facing public evidence with Chinese cognition-facing
  labels: `结论`, `上下文`, and `不确定性`.
- Add a generic candidate-promotion path for cases where candidate evidence
  directly answers the current slot but the helper result is marked unresolved.
  The `耗材厂` case must be covered by this generic behavior without
  hardcoding its terms.
- Preserve useful negative-result behavior: when no direct evidence is found,
  report the no-evidence conclusion and nearby commitments/facts separately.
- Add a bounded conversation evidence packet contract for relation-dependent
  conversation recall. A packet should start from a seed message and may attach
  only supported relation rows for this stage: previous message/image, next
  message, or reply parent. Reply-child and same-topic relation producers stay
  deferred until a source-owned implementation exists. The packet reducer must
  output compact cognition-facing facts rather than a broad row dump.
- Keep conversation packetization inside the existing RAG2 helper-agent
  boundary. Do not add a new response-path LLM call; use existing worker
  generation/judgment stages and deterministic relation fetching/validation.
- Preserve raw rows and source refs for trace/debug only. Public
  `conversation_evidence` should emphasize the answered fact, speaker, local
  time when useful, relation type, and uncertainty.
- Preserve or improve source sanitation: no raw message ids, raw global user
  ids, raw CQ syntax, or raw UTC timestamps in `rag_result.answer`,
  `memory_evidence`, `recall_evidence`, or `conversation_evidence`.
- Update the experiment quality matrix so source/provenance presence is not a
  scored dimension. The scored dimensions are cognition usability, coverage,
  factual accuracy, attribution where needed for meaning, uncertainty handling,
  compactness, and prompt safety.
- Update `src/kazusa_ai_chatbot/rag/README.md` and, if the cognition handoff
  wording changes, `src/kazusa_ai_chatbot/nodes/README.md`.
- Re-run the fixed 20-case branch-vs-mainline suite and a fresh 20-case holdout
  through real RAG2 after implementation. Write a `debug-llm` human-readable
  review that shows inputs and outputs.
- During implementation, run focused real LLM branch-vs-mainline probes when a
  targeted change is ready for feedback. Run each live case individually,
  inspect the output before continuing, and use the result to guide the next
  narrow edit. Do not batch-run live probes for a simple pass/fail signal.

## Deferred

- RAG3 router/interpreter work.
- Graph RAG, DAG, or conversation graph work.
- Full conversation graph replacement of recent history.
- New databases, collections, indexes, vector-search changes, or memory schema
  migrations.
- Conversation-progress redesign.
- Dialog style or persona voice changes outside RAG evidence wording.
- New LLM stages, extra response-path calls, fallback retries, or compatibility
  shims not named in this plan.
- Deterministic semantic matching over user input.
- Increasing conversation `top_k` or retry count as the main quality fix.
- Asking cognition or dialog to filter broad, noisy RAG evidence.

## Cutover Policy

Overall strategy: bigbang for prompt-facing RAG2 wording and prompt language;
compatible for `rag_result` keys and trace fields.

| Area | Policy | Instruction |
|---|---|---|
| RAG2 prompt-facing language | bigbang | Convert RAG2 LLM-facing instructions, examples, summaries, refined queries, and public evidence labels to Chinese-first wording. Do not keep parallel English prompt variants. |
| Code-level route tokens | compatible | Keep existing route-prefix contracts. The semantic task text after route tokens must be Chinese-first. |
| `rag_result` top-level keys | compatible | Preserve existing public keys and field types for cognition and consolidation consumers. |
| Supervisor trace | compatible | Preserve trace/debug fields for inspection. Raw ids may remain trace-only. |
| Retrieval architecture | compatible | Use existing RAG2 initializer, dispatcher, specialist agents, evaluator, finalizer, and projection boundaries. |
| Experiment code | bigbang | Do not import experiment code. After production verification, remove obsolete experiment code only if the user approves cleanup scope. |

## Cutover Policy Enforcement

- The execution agent must follow the selected policy for each area.
- If an area is `bigbang`, rewrite the old prompt-facing behavior instead of
  preserving parallel variants.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to cutover policy requires user approval before implementation.

## Target State

RAG2 keeps the existing architectural flow:

```text
initializer -> dispatcher -> specialist agent -> evaluator -> finalizer
  -> project_known_facts -> state["rag_result"] -> cognition
```

The fused target behavior is:

- Mainline remains the baseline for readable synthesis, group attribution,
  local chronology, and cognition-facing output shape.
- Current-branch private-memory recall improvements are preserved for scoped
  user continuity and user profile facts.
- RAG2 prompt-facing evidence is Chinese-first from retrieval planning through
  cognition-facing projection.
- Candidate evidence that directly answers a slot can become confirmed
  evidence through the evaluator contract.
- Nearby evidence remains available as nearby/unconfirmed evidence instead of
  being mixed into direct answers.
- Public RAG evidence is prompt-safe and cognition-ready:

```text
结论：直接事实答案，或明确说明没有确认事实。
上下文：
- 只保留 cognition 需要理解事实的说话人、时间或场景；不要堆叠来源名。
不确定性：剩余不确定性、冲突、或“无”。
```

## Design Decisions

- Use mainline as the behavioral baseline for final answer readability and
  group-chat synthesis.
- Port only proven current-branch private-memory recall improvements.
- Supersede the earlier English-internal-RAG decision. Chinese is the default
  LLM-facing RAG language because the source chat data, user input, retrieved
  evidence, and cognition consumer are primarily Chinese.
- Keep code identifiers stable where they are deterministic contracts. This
  avoids a broad dispatcher migration while still removing English translation
  from semantic retrieval content.
- Fix candidate promotion at the evaluator/finalizer contract level, not with
  deterministic user-keyword matching.
- Keep no-evidence cases honest. Do not convert nearby evidence into direct
  facts unless the evaluator can semantically justify the promotion.

## Contracts And Data Shapes

The public `rag_result` keys remain compatible:

```python
{
    "answer": str,
    "user_image": dict,
    "user_memory_unit_candidates": list[dict],
    "character_image": dict,
    "third_party_profiles": list[str],
    "memory_evidence": list[dict],
    "recall_evidence": list[dict],
    "conversation_evidence": list[str],
    "external_evidence": list[dict],
    "supervisor_trace": {
        "loop_count": int,
        "unknown_slots": list[str],
        "dispatched": list[dict],
    },
}
```

Prompt-facing evidence text inside those fields must use Chinese labels and
Chinese summaries. Trace payloads may keep internal identifiers and raw refs
needed for debugging, but they must not be copied into public evidence text.
The cognition-facing fields must not reward or emphasize source provenance for
its own sake. Include source-like details only when they are part of the fact's
meaning, such as speaker attribution in group chat, event time, or whether
evidence is confirmed versus nearby/unconfirmed.

Candidate-promotion records, if represented explicitly, must be trace-visible
and public-evidence-safe:

```python
{
    "resolved": True,
    "promotion_source": "candidate_evidence",
    "promotion_reason": "候选证据直接回答当前槽位。",
    "public_summary": "结论：...\n上下文：...\n不确定性：...",
}
```

This shape is illustrative. Implementation must reuse the existing production
fact-row and projection contracts where possible instead of adding unnecessary
new public fields.

## LLM Call And Context Budget

- Do not add a new RAG response-path LLM call.
- Lower the universal RAG2 supervisor loop cap from eight to four per user
  direction. Future optimization should make common recall paths fit within
  fewer than four loops instead of adding variable per-case budgets.
- Chinese prompt conversion must not increase prompt length materially. Replace
  English instructions in place rather than appending Chinese duplicates.
- Candidate promotion must run inside existing evaluator/finalizer calls or
  existing deterministic validation.

## Change Surface

Expected production inspection and change surface:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
- `src/kazusa_ai_chatbot/rag/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md` only if handoff wording changes.

Expected test and experiment surface:

- Focused production tests under `tests/` for prompt rendering, projection,
  candidate promotion, source sanitation, and private-memory recall behavior.
- Experiment-review matrix updates so the human-readable comparison evaluates
  cognition usability rather than provenance richness.
- Existing experiment runner under `experiments/rag2_branch_quality/` as
  comparison harness only.
- New `debug-llm` review artifact under an experiment report or
  `test_artifacts/` path after real LLM validation.

## Overdesign Guardrail

- Do not add graph search, RAG3 routing, a new memory index, a new media index,
  a new finalizer stage, or a second answer synthesizer.
- Do not make the initializer understand backend storage or direct database
  filters.
- Do not add compatibility prompt variants in both English and Chinese.
- Do not solve missed technical-memory cases by hardcoding technology terms.
- Do not solve the `耗材厂` case by hardcoding brand names or that phrase.
- Do not add new public `rag_result` fields unless existing fields cannot
  safely express the required evidence.

## Agent Autonomy Boundaries

- The execution agent may inspect branch diffs and choose the smallest code
  changes that preserve the current-branch private-memory win.
- The execution agent may rewrite RAG2 prompts into Chinese within the approved
  prompt surfaces.
- The execution agent may add focused tests and experiment report artifacts.
- The execution agent must not change the cutover policy, public state shape,
  database schema, model routing, graph architecture, or cognition/dialog
  prompts without user approval.
- If implementation reveals that preserving the private-memory win requires a
  broader memory schema or retrieval-index change, stop and report the blocker
  instead of expanding scope.

## Implementation Order

1. Parent rereads this plan, `rag/README.md`, `nodes/README.md`, and the
   branch-vs-mainline report.
2. Parent inspects the current branch versus mainline diff and records which
   current-branch changes are selected for fusion and which are rejected.
3. Parent adds focused tests for:
   - Chinese prompt-facing evidence labels;
   - no raw ids, raw CQ syntax, or raw UTC timestamps in public evidence;
   - candidate evidence promotion from an exact-answer candidate;
   - preservation of private user-memory recall behavior.
4. Parent runs the focused tests before implementation and records baseline
   failures or current behavior.
5. Parent starts one production-code subagent with this approved plan, the
   focused test contract, and the exact production change surface.
6. Production-code subagent implements the selected private-memory fusion,
   Chinese prompt conversion, candidate-promotion contract, and public evidence
   sanitation changes.
7. Parent updates integration tests or harness checks for `rag_result`
   compatibility and cognition-facing projection.
8. Parent runs focused tests, integration tests, static checks, and py_compile
   for touched Python files.
9. Parent runs focused real LLM branch-vs-mainline probes for the changed
   behavior as implementation feedback. Preferred probes include the cat/profile
   memory case, the `耗材厂` candidate-promotion case, and one mainline-clean
   group attribution case.
10. Parent runs the fixed 20-case real LLM suite and writes a `debug-llm`
   branch-quality report showing inputs and outputs.
11. Parent selects a fresh 20-case holdout from real DB-backed data, runs it
    one live LLM case at a time, and writes a second `debug-llm` report.
12. Parent starts an independent code-review subagent after verification
    passes.
13. Parent remediates review findings within this plan's change surface,
    reruns affected verification, records execution evidence, and updates plan
    lifecycle only after user-approved completion.

## Execution Model

- Execution is parent-led and uses native subagents.
- Parent owns test contract, integration verification, live LLM test execution,
  report writing, plan lifecycle updates, and final sign-off.
- Production-code subagent owns only production source changes listed in
  `Change Surface`.
- Independent code-review subagent owns review only and must not implement
  fixes.
- If native subagents are unavailable during execution, stop and report the
  blocker unless the user explicitly approves fallback execution.

## Progress Checklist

- [x] Stage 1 - branch/mainline diff audit recorded.
- [x] Stage 2 - focused tests added and baseline behavior recorded.
- [x] Stage 3 - production-code subagent completes approved fusion changes.
- [x] Stage 4 - integration compatibility checks pass.
- [x] Stage 5 - focused branch-vs-mainline live LLM probes completed during
      implementation feedback.
- [x] Stage 6 - fixed 20-case `debug-llm` review completed.
- [x] Stage 7 - fresh 20-case holdout `debug-llm` review completed.
- [x] Stage 8 - independent code review completed and findings resolved.
- [x] Stage 9 - execution evidence recorded and lifecycle status updated.

## Verification

Required deterministic verification:

```powershell
venv\Scripts\python.exe -m py_compile <touched Python files>
venv\Scripts\python.exe -m pytest <focused RAG2 tests> -q
```

Required live LLM verification:

- During implementation, run focused real LLM probes between the current branch
  and mainline when a targeted behavior changes. Use the same real input for
  both sides, inspect output one case at a time, and record a short
  `debug-llm` note or execution-evidence summary.
- Run the 20 fixed cases from
  `experiments/rag2_branch_quality/cases/rag2_branch_quality_cases_20260524.jsonl`
  one case at a time.
- Produce a human-readable `debug-llm` report showing each input and output.
- Run a fresh 20-case holdout set selected from real database-backed inputs.
- Produce a second human-readable `debug-llm` report showing each input and
  output.
- In both reports, do not count source/provenance richness as a positive quality
  metric. Treat source details as raw/supporting evidence and judge the primary
  output by cognition usability.

Quality gates:

- The cat/profile case must preserve the current branch's successful recall of
  the user's cat details.
- The group exact-phrase and Ab screenshot cases must preserve mainline's
  cleaner attribution and no message-id leakage.
- The `耗材厂` case must promote the candidate evidence into the answer through
  generic candidate-promotion behavior.
- Public RAG evidence must be Chinese-first and must not include raw ids, raw
  CQ syntax, raw UTC timestamps, or debug-only source rows.
- No new response-path LLM calls are introduced.
- The review conclusion must state whether cognition can consume the RAG output
  directly to form stance, judgment, and response goals without reading trace
  fields.

## Independent Plan Review

Before approval or execution, run an independent plan review focused on:

- whether the plan truly fuses branch strengths instead of picking a branch;
- whether Chinese prompt-facing language is specified without breaking route
  contracts;
- whether candidate promotion avoids deterministic user-keyword matching;
- whether quality scoring starts from the cognition consumer rather than source
  provenance richness;
- whether the change surface is small enough for production safety;
- whether verification proves both fixed 20-case behavior and fresh holdout
  behavior.

Record the review outcome in this plan before changing `Status` to `approved`
or `in_progress`.

## Independent Code Review

After planned implementation and verification pass, run an independent code
review over the full diff. The review must check:

- compliance with this plan and mandatory skills;
- prompt safety and CJK string safety;
- no imports from `experiments/` or `test_artifacts/`;
- no deterministic semantic matching over user input;
- no raw ids, raw CQ syntax, or raw UTC timestamps in public evidence;
- no cognition/dialog ownership drift;
- no evaluation claim that treats source/provenance richness as a primary RAG
  quality win;
- real LLM review artifacts match the claimed quality result.

Record findings, fixes, rerun commands, and residual risk in execution
evidence before sign-off.

## Acceptance Criteria

- Mainline-style group synthesis and answer cleanliness are preserved.
- Current-branch private durable user-memory recall improvement is preserved.
- Prompt-facing RAG language is Chinese-first across planning, retrieval
  summaries, evaluator/finalizer output, and cognition-facing evidence.
- Quality evaluation is cognition-consumer-first. Source/provenance detail is
  available for trace/debug and gap analysis, but is not a scored quality
  dimension for the RAG-to-cognition handoff.
- Candidate evidence that directly answers a slot is promoted generically.
- Existing `rag_result` top-level keys and field types remain compatible.
- No raw ids, raw CQ syntax, raw UTC timestamps, or debug source rows appear in
  public RAG evidence.
- Fixed 20-case real LLM review and fresh 20-case holdout review are completed
  with readable inputs and outputs.
- Independent code review is completed with findings resolved or explicitly
  accepted by the user.

## Execution Evidence

- 2026-05-24: User approved the plan and requested execution. Status changed
  to `in_progress`.
- 2026-05-24: User added an execution requirement to run focused real LLM
  branch-vs-mainline probes during implementation for realtime feedback where
  practical.
- 2026-05-24: Branch/mainline diff audit started. Selected for fusion:
  scoped `user_memory_units` recall that combines slot/current/original query,
  prompt-safe evidence formatting, trace-only source refs, and generic
  unresolved-candidate handling. Rejected as-is: English-first RAG2 prompt
  wording, English public evidence labels, source-heavy public evidence,
  strict deterministic coverage as the only promotion gate, and any new
  user-input keyword matching.
- 2026-05-24: Production-code subagent rewrote RAG2 prompt-bearing files into
  Chinese-first prompt-facing language while preserving stable JSON keys,
  route tokens, source text, URLs, filenames, and code/model labels. Follow-up
  local patches fixed the remaining continuation prompt headings, one recall
  fallback status string, and the RAG evidence documentation contract.
- 2026-05-24: Focused deterministic verification passed:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `45 passed`.
  `py_compile` passed for touched production and focused test Python files.
  `rg` found no stale English RAG2 marker phrases such as `Use English`,
  `# Input Format`, `# Output Format`, `Return valid JSON only`,
  `This RAG run found`, `Checked sources`, or `Nearby but unconfirmed`.
- 2026-05-24: Focused real LLM branch-vs-mainline probes completed one case at
  a time for `rag2-branch-quality-005`, `rag2-branch-quality-017`, and
  `rag2-branch-quality-019`. Review artifact:
  `experiments/rag2_branch_quality/reports/fusion_current_vs_mainline_probe_20260524.md`.
  Raw current-branch artifacts:
  `test_artifacts/rag2_branch_quality/fusion_current_probe_20260524/`.
  Raw mainline artifacts:
  `test_artifacts/rag2_branch_quality/fusion_mainline_probe_20260524/`.
  Result: current branch clearly improves scoped user-memory recall in case
  005, answers case 017 with fewer loops than mainline, and preserves answer
  correctness for case 019 while showing a remaining over-broad attachment
  planning weakness.
- 2026-05-24: Independent prompt-surface review found three issues: public
  evidence redaction still used English `[source id omitted]`, several
  evaluator prompt example values were English placeholder prose, and the
  dispatcher/initializer input examples used invalid JSON prose values. Fixed
  all three by using Chinese public redaction labels, Chinese placeholder
  values, and valid JSON examples. Fresh verification after the fixes:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `45 passed`;
  `py_compile` passed for the edited prompt/formatting files; `git diff
  --check` returned exit code 0.
- 2026-05-24: Parent reviewed touched LLM prompt surfaces again before fixed
  20-case validation. Additional prompt/contract fixes removed dispatcher
  internal-worker detail, made evaluator/finalizer source cues conditional,
  prevented readable message-id leakage, replaced placeholder initializer
  schema examples, and tightened raw-id/source-id handling in initializer,
  evaluator, and finalizer prompts.
- 2026-05-24: During live fixed-case validation, case 011 exposed initializer
  placeholder-slot copying and case 016/018 exposed raw UUID/source-id and
  invalid person-slot dependencies in prompt-facing initializer output. Fixed
  by prompt tightening plus structural normalization over stable slot grammar:
  raw source ids are sanitized from initializer slots and
  `speaker=person resolved in slot N` is dropped when slot N cannot resolve a
  person. This validation does not use deterministic keyword matching over
  user input.
- 2026-05-24: Deterministic verification after the additional fixes passed:
  `venv\Scripts\python.exe -m pytest tests/test_rag_evidence_formatting.py
  tests/test_rag_projection.py tests/test_rag_prompt_evidence_safety.py
  tests/test_rag_finalizer_time_context.py -q` returned `47 passed`;
  `venv\Scripts\python.exe -m pytest
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_drops_invalid_person_slot_reference
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_keeps_valid_person_slot_reference
  tests/test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_removes_source_ids
  tests/test_rag_prompt_evidence_safety.py tests/test_rag_evidence_formatting.py
  -q` returned `19 passed`; `py_compile` passed for edited prompt/source/test
  files.
- 2026-05-24: Fixed 20-case live RAG2 validation completed one case at a time
  against real DB-backed data. Human-readable `debug-llm` review artifact:
  `experiments/rag2_branch_quality/reports/fixed20_current_rag2_review_20260524.md`.
  Raw artifacts:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/`. Structured
  raw summary:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/_case_summary.json`.
  Result from cognition-consumer view: most private-memory and group-chat
  cases are usable, candidate promotion works for the `耗材厂`/X2D-style cases,
  and public message/source-id safety is improved. Remaining high-risk gaps:
  technical conversation recall (`Gemma 4` quantization), coined phrase recall
  (`星尘薄荷魔法`), media product-image recall (`金属预埋螺母`), and one official
  identity-memory lookup (`千纱生日和星座`).
- 2026-05-24: Final local verification for Stage 6 passed:
  `venv\Scripts\python.exe -m py_compile
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py
  src\kazusa_ai_chatbot\rag\evidence_formatting.py
  tests\test_persona_supervisor2_rag2_integration.py
  tests\test_rag_evidence_formatting.py
  tests\test_rag_prompt_evidence_safety.py` exited 0;
  `venv\Scripts\python.exe -m pytest tests\test_rag_evidence_formatting.py
  tests\test_rag_projection.py tests\test_rag_prompt_evidence_safety.py
  tests\test_rag_finalizer_time_context.py
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_drops_invalid_person_slot_reference
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_keeps_valid_person_slot_reference
  tests\test_persona_supervisor2_rag2_integration.py::test_normalize_initializer_slots_removes_source_ids
  -q` returned `50 passed`; `git diff --check` exited 0 with only CRLF
  normalization warnings.
- 2026-05-24: Failed fixed-run cases were compared against mainline using the
  same real inputs. Review artifact:
  `experiments/rag2_branch_quality/reports/failed_cases_current_vs_mainline_20260524.md`.
  Extracted evidence:
  `test_artifacts/rag2_branch_quality/fixed20_current_20260524/_failed_vs_mainline_summary.json`.
  Result: mainline clearly wins on `rag2-branch-quality-020` because it
  retrieves the durable birthday/zodiac memory in one loop; mainline is also
  cleaner and cheaper on `rag2-branch-quality-007`; both branches fail
  `rag2-branch-quality-004` and `rag2-branch-quality-018`, while current is
  safer than mainline on 018 because it no longer exposes a raw
  `global_user_id`. Next development priority is to restore mainline's durable
  character/world memory attribute recall while preserving current-branch
  private user-memory recall, then address exact/coined phrase load, media
  product-image recall, and technical conversation recall.
- 2026-05-24: Priority 1 was addressed only; other failed-case priorities
  remain deferred. Root cause: the durable-memory search specialist translated
  Chinese subject/attribute queries into English and generated only the subject
  as a literal anchor, so exact character/world attribute rows such as
  birthday/zodiac memories were ranked below broader subject memories. Fix:
  `persistent_memory_search_agent` prompt now requires preserving the source
  language and treating subject-plus-requested-attribute queries as literal
  anchor candidates generated by the LLM specialist, without adding
  deterministic keyword extraction from user input. Focused live RAG2 result:
  `rag2-branch-quality-020` now resolves in one loop with the answer
  `已确认杏山千纱（Kyōyama Kazusa）的生日为8月5日，星座为狮子座。该信息来源于持久化记忆中的明确记录。`
  Private-memory regression check `rag2-branch-quality-005` still resolves via
  `user_memory_evidence_agent`. Review artifact:
  `experiments/rag2_branch_quality/reports/priority1_durable_memory_recall_review_20260524.md`.
  Raw artifacts:
  `test_artifacts/rag2_branch_quality/priority1_after_current_20260524/`,
  `test_artifacts/rag2_branch_quality/priority1_after_attribute_prompt_probe_20260524.json`.
  Verification:
  `venv\Scripts\python.exe -m py_compile
  src\kazusa_ai_chatbot\rag\persistent_memory_search_agent.py
  tests\test_rag_hybrid_agents.py` passed, and
  `venv\Scripts\python.exe -m pytest
  tests\test_rag_hybrid_agents.py::test_persistent_memory_search_prompt_preserves_chinese_attribute_anchors
  tests\test_rag_hybrid_agents.py::test_persistent_memory_search_tool_fuses_semantic_and_keyword_rows
  tests\test_rag_hybrid_agents.py::test_persistent_memory_search_rejects_untrusted_source_filter
  tests\test_rag_hybrid_agents.py::test_persistent_memory_search_reapplies_trusted_source_filter
  -q` returned `4 passed`.
- 2026-05-24: Next-stage conversation relation packet improvement completed.
  Production changes stayed inside the existing RAG2 helper/projection
  boundary:
  `conversation_search_agent` now preserves prompt-safe local timestamps,
  normalizes seed timestamps back to storage UTC for neighbor queries, and
  annotates previous/next neighbor rows with stable seed relation metadata;
  `conversation_evidence_agent` now builds bounded seed-plus-relation packets,
  requires relation packets for relation-dependent slots, keeps direct
  semantic/keyword hits usable as packet seeds even when they are also
  neighbors, filters non-relation packet evidence to keyword-anchored seeds,
  and routes media-adjacent conversation slots to conversation search instead
  of memory; `persona_supervisor2_rag_projection` prefers selected packet
  summaries for public `conversation_evidence`; initializer prompt text now
  documents only supported relation tokens (`previous_message`,
  `next_message`, `reply_parent`) and the initializer cache version was bumped
  to `initializer_prompt:v20`. Unsupported advertised relation tokens
  `reply_child` and `same_topic_nearby` were removed from this stage because
  no source-owned producer exists yet.
- 2026-05-24: Independent code-review subagent reviewed the relation-packet
  diff. Findings: unsupported relation tokens were advertised, direct hits
  could become relation-tagged and lose packet seed status, and packet
  summaries could expand non-relation evidence. Fixes applied in this stage:
  contract narrowed to produced relation tokens, direct retrieval methods keep
  rows eligible as packet seeds, and non-relation public packets are limited
  to keyword-supported seeds.
- 2026-05-24: Focused real RAG2 live probe against real MongoDB data used
  holdout case `rag2-holdout20-20260524-018` from
  `test_artifacts/rag2_branch_quality/holdout20_cases_20260524.jsonl`.
  Input: `谁说 Google Drive 又不是第一次这样了？前面那张图大概是什么事情？`
  Final answer after fixes:
  `总是跌倒的企鹅：Google drive又不是第一次这样了；上一条消息包含一张社交媒体平台（类似 X/Twitter）上的帖子截图，发帖人 ID 为 @masahiroitosugi，内容讲述其 Google 账号因上传旧漫画数据到 Drive 而被封禁且申诉失败，发布时间为 2026 年 5 月 15 日晚上 8:46。`
  Raw artifact:
  `test_artifacts/rag2_branch_quality/conversation_packet_probe_final_20260524/rag2-holdout20-20260524-018.json`.
  Human-readable debug review:
  `experiments/rag2_branch_quality/reports/conversation_packet_probe_review_20260524.md`.
- 2026-05-24: Deterministic verification for the conversation-packet stage
  passed. `py_compile` passed for touched RAG/projection/cache/test Python
  files. Focused tests covering local timestamp preservation, UTC neighbor
  query bounds, relation-required unresolved behavior, packet reduction,
  direct-neighbor seed retention, non-relation packet filtering, media-slot
  routing, public projection, and initializer prompt version passed. Broader
  adjacent suite returned `126 passed`:
  `venv\Scripts\python.exe -m pytest tests\test_rag_hybrid_agents.py
  tests\test_rag_phase3_capability_agents.py tests\test_rag_projection.py
  tests\test_rag_prompt_evidence_safety.py
  tests\test_rag_initializer_cache2.py::test_initializer_prompt_version_bumped_for_capability_cutover
  tests\test_rag_initializer_cache2.py::test_initializer_prompt_version_bumps_to_v20_for_current_contract
  -q`. `git diff --check` exited 0 with only CRLF normalization warnings.
- 2026-05-24: Closeout cleanup completed. User rejected variable loop-budget
  complexity and directed a universal four-loop cap. Production RAG2 now uses
  `_MAX_LOOP_COUNT = 4`; the RAG README and this plan record that future
  optimization should make common recall paths fit below four loops instead of
  adding broader loop budgets. The evaluator now finalizes or drains queued
  evidence before asking the continuation LLM when there are pending evidence
  slots, Recall plus memory both miss, or already resolved evidence exists.
  Verification: `venv\Scripts\python.exe -m py_compile
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py
  tests\test_persona_supervisor2_rag2_integration.py` passed, and
  `venv\Scripts\python.exe -m pytest
  tests\test_persona_supervisor2_rag2_integration.py -q` returned
  `16 passed`. Final current-plus-holdout observation report:
  `experiments/rag2_branch_quality/reports/current20_and_holdout20_after_packets_review_20260524.md`.
- 2026-05-24: Plan status set to `completed` and archived under
  `development_plans/archive/completed/short_term/`. Follow-up direction was
  recorded in the RAG3 draft plan: improve first-pass routing precision,
  reduce loop demand below four, and do not use RAG3 as a way to add variable
  or broader loop budgets.
