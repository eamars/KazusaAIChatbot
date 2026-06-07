# l2d action router prompt separation plan

## Summary

- Goal: Finish the L2d separation by moving action initialization into a
  route-only JSON contract, reviewing and rewriting the affected prompts as
  coherent LLM flows, and correcting the remaining router/worker boundary where
  a routing LLM still emits worker-facing task parameters.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: bigbang for the action-router prompt used by L2d
  and background-worker route contract, compatible for downstream action-spec,
  queue, delivery, and adapter contracts.
- Highest-risk areas: L2d route regression, prompt-flow drift, route/parameter
  ownership bleed, live-chat latency, accidental raw-id exposure, background
  work promise without durable queue evidence, and overfitting to the POC cases.
- Acceptance criteria: L2d calls the top-level `action_router` module with a
  prompt-safe JSON payload, action capabilities are projected from runtime-safe
  affordances instead of hardcoded prompt prose, routers only route,
  worker-local generators own semantic task interpretation, and focused
  deterministic plus one-at-a-time live LLM evidence validates the flow.

## Context

The completed background-work migration reduced the original L2d overload, but
the remaining L2d action initializer still carries too much responsibility.
Current source inspection shows:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
  contains `_ACTION_INITIALIZER_PROMPT` and the live `call_action_initializer`
  entry point.
- `build_action_initializer_payload(...)` currently returns a prose action
  context string. The prompt explicitly describes the user message as a Chinese
  context string rather than JSON.
- The same static prompt manually lists resolver and action capabilities. The
  project already has prompt-safe action affordance projection in
  `src/kazusa_ai_chatbot/action_spec/registry.py`, but L2d does not use that
  projection as its prompt-facing action roster.
- `src/kazusa_ai_chatbot/background_work/router.py` follows the better pattern
  of stable system prompt plus runtime JSON, but its route output still
  includes a worker-facing `task` string.
- `src/kazusa_ai_chatbot/background_work/subagent/text_artifact.py` already
  splits worker-local classification from generation, but the classifier still
  emits both `task_type` and a cleaned `task`, which makes one LLM both route
  and generate semantic parameters.
- `src/kazusa_ai_chatbot/rag/web_agent3/README.md` defines the pattern this
  plan follows: the router emits only route fields, deterministic code
  dispatches, and selected subagents own source-local interpretation.

The target is not another background-work runtime migration. The queue,
delivery loop, result-ready cognition route, adapter delivery, and durable job
collection remain intact. This plan is the follow-up boundary and prompt-flow
cleanup after L2d separation.

The user has explicitly rejected placing the new action-router module under
`nodes`. The L2d node remains under `src/kazusa_ai_chatbot/nodes/` and points
to a top-level `src/kazusa_ai_chatbot/action_router/` module. The new module
must include its own ICD-styled README.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing action-router prompt flow,
  input shape, background-work routing, worker-subagent prompts, graph routing,
  parser contracts, or LLM context budgets.
- `no-prepost-user-input`: load before changing logic that accepts, rejects,
  persists, routes, or acts on user requests.
- `debug-llm`: load before live/local LLM runs, prompt comparison, prompt
  quality review, or trace artifact review.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK prompt
  strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.
- Plan status is not production-code authorization. Production-code edits
  require this plan to be approved or in progress and a direct user instruction
  to execute it.
- Any modification to an existing LLM prompt, prompt payload, parser contract,
  or output schema must pass the LLM modification rule before implementation:
  audit the information fed into the LLM, then warrant whether the affected
  prompt must be rewritten to preserve organic logic flow across role,
  procedure, input contract, output schema, examples, parser assumptions, and
  downstream consumers. If an existing prompt is touched, rewrite the affected
  prompt coherently instead of appending a patch rule.
- Any newly drafted LLM prompt or LLM stage must follow the LLM drafting rule:
  define the smallest semantic question, required inputs, output fields,
  deterministic owners, rejected inputs, rejected complexity, parser
  assumptions, downstream consumers, and required evidence before adding prompt
  scope.
- No LLM may both route/classify and generate semantic parameters, tool
  parameters, or artifacts. This applies to L2d, the background-work router,
  and worker-local task classifiers.
- L2d remains the graph stage that asks the top-level action-routing question,
  but `action_router` owns the reusable prompt, payload, contract, and
  normalization code. The L2d node delegates to it.
- The action-router LLM may choose route families and immediate
  visible-surface need; it must not choose workers, worker-local task types,
  handler ids, DB fields, adapter targets, job ids, delivery mechanics, tool
  arguments, or final visible text.
- The background-work router may choose only whether work can execute, which
  worker owns it, and why. It must not emit worker-facing task briefs, task
  types, tool parameters, files, final text, delivery decisions, or job ids.
- Worker subagents may interpret worker-local semantics only after deterministic
  dispatch selects the worker. Worker-local classifiers may classify but must
  not manufacture cleaned task prompts for generators.
- Deterministic code owns validation, materialization, queue persistence,
  idempotency, adapter delivery feasibility, retries, leases, limits,
  capability allowlists, and invariant enforcement.
- Do not add deterministic keyword routing or post-processing over raw user
  text to rescue LLM route decisions.
- Do not tune prompt wording to memorize the Stage 0 POC cases or the current
  fixture names.
- `action_spec.registry` may change only prompt-facing projection text needed
  by `action_router`. Capability schemas, handler ids, owner modules, action
  models, and action handlers are out of scope unless the plan is updated and
  reapproved.
- Live `/chat` must not wait for background worker routing, worker-local
  classification, artifact generation, web research, code execution, or tool
  loops.
- Live/local LLM checks must run one case at a time and be inspected one case
  at a time.

## Must Do

- Create a top-level `action_router` module with a mandatory ICD-styled README.
- Preserve `call_action_initializer(...)` as the integration entry point so the
  surrounding L2d graph does not churn. The L2d node imports and delegates to
  `action_router`; it does not own the action-router prompt or payload schema.
- Replace the prose action-initializer human payload with a prompt-safe JSON
  payload rendered through `json.dumps(..., ensure_ascii=False)`.
- Generate the action-capability roster from prompt-safe affordance projection,
  not from a manually duplicated capability list inside the action-router prompt.
- Generate resolver-capability affordances through a prompt-safe projection so
  the static prompt does not duplicate resolver capability definitions.
- Rewrite the action-router prompt as one coherent flow covering source
  recognition, pending resolver continuation, resolver-first evidence needs,
  goal progress, top-level action routing, visible-surface need, and strict
  output shape.
- Keep L2d output route-only. For `background_work_request`, L2d chooses that
  route and a route reason; deterministic materialization builds the queue
  summary from already prompt-safe state rather than accepting an LLM-produced
  worker task.
- Preserve existing downstream action-spec and queue compatibility by keeping
  `background_work_request` action specs and durable background-work job fields
  compatible with current handlers.
- Update the prompt-facing `background_work_request` action affordance so it
  no longer instructs the model to emit `task_brief`. The trusted
  `task_brief` action-spec param remains a deterministic materialization field.
- Correct the background-work router output so it no longer emits `task`.
- Correct `text_artifact` worker-local prompts so classification outputs only
  task type and reason, while generation receives the original queue/source
  summary and the chosen task type.
- Update deterministic tests and live LLM traces to show the exact source data
  used to judge pass, partial pass, and failure.
- Record prompt-flow review findings and LLM input audits in `Execution
  Evidence` before sign-off.

## Deferred

- Do not redesign the background-work queue, worker runtime, result delivery,
  adapter delivery, scheduler, reflection, proactive output, or consolidator.
- Do not add a repository-editing coding agent, shell execution worker,
  package-install worker, file mutation worker, browser automation worker,
  attachment worker, image worker, or web-research background worker.
- Do not add DB migration scripts or rewrite existing queued job documents.
- Do not add fallback LLM repair calls, compatibility shims, keyword fallback
  routers, or alternate dispatch paths.
- Do not remove `background_work_request`, the existing durable job collection,
  or the result-ready cognition delivery route.
- Do not change L3/dialog wording ownership.
- Do not change adapter protocols or message-envelope platform fields.

## Cutover Policy

Overall strategy: bigbang for L2d/action-router prompt and route contracts.
Local compatible surfaces listed below are the only retained compatibility
surfaces.

| Area | Strategy | Policy |
|---|---|---|
| L2d action router internals | bigbang | Replace the internal prompt and payload in one change while preserving the public `call_action_initializer(...)` result shape. |
| Action-spec output | compatible | Keep current action-spec kinds, validation, visibility, urgency, and target ownership. |
| Background-work queue | compatible | Keep durable job schema compatible; reinterpret `task_brief` as a deterministic queue/source summary, not an L2d-generated worker task. |
| Background-work router | bigbang | Remove LLM-produced `task` from router decisions and update tests in the same change. |
| Text-artifact worker | bigbang | Remove classifier-produced cleaned task strings and pass source summaries to generation. |
| Adapters and delivery | unchanged | No adapter or delivery contract changes. |

Cutover policy enforcement:

- If an area is `bigbang`, rewrite legacy prompt references instead of keeping
  fallback parsing, dual prompt paths, or old output aliases.
- If an area is `compatible`, preserve only the compatibility surface listed in
  the table.
- Any change to cutover policy requires user approval before implementation.

Rollback is code-level only: revert the action-router module integration and
background-worker route-contract edits together. Do not partially roll back the
action-router prompt without the parser/payload tests.

## Target State

The live response path keeps one L2d action-routing LLM call:

```text
cognition state
  -> action_router JSON payload builder
  -> route-only action-router prompt
  -> normalized route decisions
  -> deterministic action-spec materialization
  -> action evaluator / handler
```

Queued background work keeps the existing asynchronous shape:

```text
background_work_request action spec
  -> durable queue row
  -> worker tick
  -> background-work route-only worker selector
  -> deterministic worker dispatch
  -> worker-local classifier
  -> worker-local generator
  -> result-ready cognition delivery
```

No LLM stage performs both route selection and semantic parameter generation.
Routers return route fields. Generators receive selected route context and the
original prompt-safe source summaries.

## Design Decisions

- L2d stays the graph stage that asks the top-level action-routing question
  because it is the stage with access to cognition stance, boundary judgment,
  source type, resolver observations, and visible-response pressure.
- L2d does not become a worker parameter generator. Through `action_router`, it
  routes to
  `background_work_request`; deterministic code builds queue summaries from
  `decontextualized_input`, source metadata already safe for prompts, and the
  L2d route reason.
- The new `action_router` module is a code-organization and contract boundary,
  not a new response-path LLM stage.
- `action_router` is outside `nodes` because it is a reusable semantic routing
  capability. `nodes` owns graph orchestration and state threading; it should
  not own reusable prompt contracts.
- Runtime-varying fields move to `HumanMessage` JSON, matching Web Agent 3 and
  background-work router prompt structure.
- Stable role/procedure/output instructions remain in `SystemMessage` for
  prefix stability on local LLMs.
- Prompt-facing action capability descriptions come from the action registry
  projection. The static prompt may describe how to use a capability roster,
  but must not duplicate the roster as a hand-maintained hardcoded list.
- Resolver capability descriptions use a prompt-safe projection owned by the
  new `action_router` module. Do not hand-code resolver capability definitions
  in the static prompt.
- The background-work router is corrected to match its own route-only intent:
  `action`, `worker`, and `reason` only.
- The text-artifact task classifier becomes classification-only. The generator
  receives `task_type`, `queue_summary`, `source_summary`, and output limits.
- Existing durable field names may remain for compatibility when changing them
  would create a data migration. Ownership semantics are documented in the ICD.

## Action Router Input Audit Procedure

The action-router input contract must be designed before prompt rewriting. The
implementation agent must write the audit result into `Execution Evidence`
before editing production prompt code.

Use this minimal contract:

```text
Semantic question:
Given one completed cognition state, which resolver capabilities and top-level
action routes are semantically needed next, and is a visible surface needed now?

Inputs required:
Prompt-safe source summary, current semantic input, cognition judgment,
evidence status, resolver state, continuity clues, and capability affordances.

Output fields required downstream:
resolver_capability_requests, resolver_goal_progress, action_requests.

Deterministic owners:
state projection, raw-id stripping, capability projection, action-spec
materialization, queue summaries, validation, persistence, execution, delivery.

Rejected complexity:
worker selection, worker task generation, DB schema, adapter targeting, job
state, tool arguments, final visible wording, retries, repair prompts.

Evidence needed before adding complexity:
a failing live trace showing the router cannot decide a required top-level
route from the approved semantic sections.
```

For every candidate input field, record:

- upstream owner;
- model-facing semantic name;
- why the router needs it for the semantic question;
- downstream consumer affected by the decision;
- projection rule from raw state to prompt-safe value;
- maximum size or count;
- forbidden raw fields removed by the projection.

Approved input sections:

- `source`: trigger source, input source labels, output mode, channel type, and
  semantic source notes. No platform ids or adapter ids.
- `current_input`: decontextualized semantic input, bounded media summary, and
  a prompt-safe current-goal summary. No raw message ids or transport syntax.
- `cognition`: stance, intent, judgment note, boundary assessment, relationship
  signals, and bounded private reasoning summaries already produced upstream.
- `evidence`: RAG answer status, short evidence summaries, conversation
  progress, and active-commitment clues. No raw memory ids, source ids,
  collection names, or retrieval backend metadata.
- `resolver`: prompt-safe pending resume, resolver observations, missing
  evidence needs, and approved goal-progress context. No pending-row ids or
  operational resolver ids.
- `continuity`: continuity-relevant active commitments, accepted promises, and
  background-result cues as semantic summaries only.
- `capabilities`: prompt-safe resolver affordances and action affordances
  generated from projections.
- `work_seed`: a bounded prompt-safe source summary copied from upstream
  semantic input for later background queue materialization. The router may
  decide to use the background-work route, but it must not rewrite this seed.

Rejected input sections:

- adapter targets, platform user ids, global user ids, channel ids, message ids,
  action attempt ids, job ids, handler ids, schema versions, collection names,
  leases, retry counters, delivery state, worker registry internals, task
  types, tool parameters, filesystem paths, credentials, final L3 text, and raw
  numeric engagement telemetry.

Numeric or operational measurements must be converted before reaching the LLM.
For example, recent group activity should enter as semantic labels such as
`noise_level="high"` with a short explanation, not as raw message counts.

The JSON payload must be shallow and model-facing. It must not mirror the
internal graph state or database documents. If a field cannot be explained in
one sentence to the model, it stays out of the action-router payload.

## Contracts And Data Shapes

The `action_router` human payload is a JSON object with prompt-safe semantic
data only. It is not a serialized L2d state object:

```json
{
  "source": {
    "trigger_source": "user_message",
    "channel_type": "private",
    "output_mode": "visible_reply"
  },
  "current_input": {
    "decontextualized_input": "semantic summary",
    "media_summary": ""
  },
  "cognition": {
    "logical_stance": "CONFIRM",
    "character_intent": "PROVIDE",
    "judgment_note": "short judgment",
    "internal_monologue": "bounded private reasoning summary",
    "boundary_core_assessment": {},
    "relationship_signals": {}
  },
  "evidence": {
    "rag_answer": "",
    "memory_evidence": [],
    "conversation_progress": {},
    "active_commitment_clues": []
  },
  "resolver": {
    "pending_resume": null,
    "resolver_context": {},
    "previous_observations": []
  },
  "capabilities": {
    "resolver_affordances": [],
    "action_affordances": []
  },
  "work_seed": {
    "background_work_allowed": true,
    "source_summary": "prompt-safe upstream work seed copied later by code",
    "max_output_chars": 4000
  }
}
```

Forbidden `action_router` payload fields:

- raw platform ids, global user ids, DB ids, job ids, action attempt ids, lease
  fields, adapter names, handler ids, collection names, schema-version fields,
  credentials, filesystem paths, tool parameters, worker names, task types, and
  final visible text.

The `action_router` output remains normalized into the existing public result
shape consumed by L2d, but the raw model output is route-only:

```json
{
  "resolver_capability_requests": [
    {
      "capability_kind": "rag_evidence",
      "objective": "semantic evidence need",
      "reason": "why evidence is needed",
      "priority": "now"
    }
  ],
  "resolver_pending_resolution": {
    "decision": "answered",
    "reason": "why the active pending item is resolved"
  },
  "resolver_goal_progress": null,
  "action_requests": [
    {
      "capability": "speak",
      "decision": "visible_reply",
      "detail": "surface need, not final wording",
      "reason": "why visible response is needed"
    },
    {
      "capability": "background_work_request",
      "decision": "queue_async_work",
      "detail": "route-only acceptance boundary",
      "reason": "why background work is appropriate"
    }
  ]
}
```

Raw model output must not include schema-version fields or pending resolver
ids. `action_router` normalization attaches trusted schema versions and binds
`resolver_pending_resolution` to the single active pending row from state.

For `background_work_request`, raw action-router output must not contain `task_brief`,
`worker`, `work_kind`, `task_type`, `tool_args`, `artifact_text`, `file_path`,
or delivery targets.

The background-work router output becomes:

```json
{
  "action": "execute",
  "worker": "text_artifact",
  "reason": "why this worker owns the queued work"
}
```

The `text_artifact` task classifier output becomes:

```json
{
  "task_type": "coding_snippet",
  "reason": "why this task type fits"
}
```

The `text_artifact` generator receives the classifier result plus the original
queue/source summaries and returns only artifact result fields.

The durable `routed_task` job field remains for compatibility. Worker runtime
fills it from the deterministic queued summary or source summary, not from a
router LLM field.

## LLM Call And Context Budget

- L2d action routing remains one live-response LLM call. This plan must not add
  another response-path LLM stage.
- The action-router system prompt should shrink or remain comparable by removing the
  hardcoded capability roster and prose input-format mismatch.
- The action-router human message may grow modestly because JSON includes keys, but the
  content must stay bounded through prompt-safe projection and existing summary
  limits.
- `action_router` receives only prompt-safe semantic sections, not raw graph
  state. The input audit must reject any field that exists only to let the LLM
  decide operational behavior.
- Background worker LLM calls remain outside the live `/chat` path.
- Live LLM verification must run one case at a time with trace inspection.
- Prompt review evidence must include the rendered system prompt, rendered
  human JSON payload, raw model output, normalized output, and quality judgment
  for each inspected live case.

## Change Surface

Create these files:

- `src/kazusa_ai_chatbot/action_router/README.md`
- `src/kazusa_ai_chatbot/action_router/__init__.py`
- `src/kazusa_ai_chatbot/action_router/contracts.py`
- `src/kazusa_ai_chatbot/action_router/payload.py`
- `src/kazusa_ai_chatbot/action_router/prompt.py`
- `src/kazusa_ai_chatbot/action_router/router.py`
- `tests/test_action_router_payload.py`
- `tests/test_action_router_prompt_contract.py`

Modify these files:

- `development_plans/README.md`
- `development_plans/active/short_term/l2d_action_router_prompt_separation_plan.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/action_spec/README.md`
- `src/kazusa_ai_chatbot/action_spec/registry.py`
- `src/kazusa_ai_chatbot/background_work/README.md`
- `src/kazusa_ai_chatbot/background_work/models.py`
- `src/kazusa_ai_chatbot/background_work/router.py`
- `src/kazusa_ai_chatbot/background_work/providers.py`
- `src/kazusa_ai_chatbot/background_work/worker.py`
- `src/kazusa_ai_chatbot/background_work/subagent/text_artifact.py`
- `tests/test_persona_supervisor2_action_initializer.py`
- `tests/test_l2d_action_selection_cases.py`
- `tests/test_l2d_action_selection_live_llm.py`
- `tests/test_l2d_l3_surface_handoff.py`
- `tests/test_background_work_router.py`
- `tests/test_background_work_router_live_llm.py`
- `tests/test_background_work_providers.py`
- `tests/test_background_work_text_artifact.py`
- `tests/test_background_work_text_artifact_live_llm.py`
- `tests/test_action_spec_evaluator.py`

Remove no files.

Keep these areas unchanged except for test imports or assertions required by
the files above:

- `src/kazusa_ai_chatbot/action_spec/handlers/background_work.py`
- `src/kazusa_ai_chatbot/action_spec/models.py`
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`
- `src/kazusa_ai_chatbot/background_work/delivery.py`
- adapter modules
- scheduler, reflection, consolidator, and RAG retrieval modules

## Overdesign Guardrail

- Do not generalize this into a universal graph-agent framework.
- Do not introduce a plugin registry for L2d action routing.
- Do not add a second worker or a coding-agent implementation.
- Do not add new persistence collections.
- Do not rename durable DB fields only for terminology cleanliness.
- Do not split the L2d action router into multiple live-response LLM calls.
- Do not make deterministic code infer user intent from keywords.
- Do not put the new router package under `nodes`.

## Agent Autonomy Boundaries

- The implementation agent may reorganize helper functions inside the listed
  files when required to satisfy the contracts.
- The implementation agent may keep compatibility wrapper functions in
  `persona_supervisor2_cognition_l2d.py` if tests or callers import them.
- The implementation agent may add narrow test helper fixtures inside listed
  test files.
- The implementation agent must stop and report if satisfying this plan
  requires modifying files outside `Change Surface`, adding a DB migration,
  changing action-spec schemas or handlers, changing adapter delivery, or
  changing L3 visible wording ownership.
- The implementation agent must not change this plan's scope, status, or
  acceptance criteria without parent approval.

## Implementation Order

1. Load mandatory skills and reread required docs: `development_plans/README.md`,
   this plan, `README.md`, `docs/HOWTO.md`, `src/kazusa_ai_chatbot/nodes/README.md`,
   `src/kazusa_ai_chatbot/action_spec/README.md`,
   `src/kazusa_ai_chatbot/background_work/README.md`, and
   `src/kazusa_ai_chatbot/rag/web_agent3/README.md`.
2. Parent captures current-state evidence: rendered current
   `_ACTION_INITIALIZER_PROMPT`, one representative payload, current action
   registry projection, and the current background-work router/text-artifact
   output shapes.
3. Parent performs and records the LLM modification/input audit for
   `_ACTION_INITIALIZER_PROMPT`, `BACKGROUND_WORK_ROUTER_PROMPT`,
   `TEXT_ARTIFACT_TASK_ROUTER_PROMPT`, and `TEXT_ARTIFACT_GENERATOR_PROMPT`.
4. Parent writes focused failing tests:
   `tests/test_action_router_payload.py::test_action_router_payload_is_prompt_safe_json`,
   `tests/test_action_router_prompt_contract.py::test_action_router_normalizes_schema_free_resolver_requests`,
   `tests/test_action_router_prompt_contract.py::test_action_router_background_work_route_rejects_task_brief`,
   `tests/test_action_spec_evaluator.py::test_prompt_affordances_do_not_ask_l2d_for_background_task_brief`,
   `tests/test_background_work_router.py::test_background_work_router_decision_is_route_only`,
   and `tests/test_background_work_text_artifact.py::test_text_artifact_task_router_does_not_emit_clean_task`.
   Run them and record expected missing-module or old-contract failures.
5. Parent starts the single production-code subagent with this approved plan,
   mandatory skills, failing tests, and the exact `Change Surface`.
6. Production-code subagent implements all production changes in the listed
   files: top-level `action_router` module and ICD, L2d delegation,
   prompt-safe projection updates, background-work route-only decision, and
   text-artifact classifier/generator ownership correction.
7. Parent updates integration/live test tracing as needed inside listed test
   files while the production-code subagent owns production files.
8. Parent runs focused module and projection tests, then integration tests,
   then live LLM cases one at a time.
9. Parent starts the review subagent after deterministic verification passes,
   records findings, addresses accepted findings inside the approved surface,
   and reruns affected verification.
10. Parent records final execution evidence and updates lifecycle status only
    after acceptance criteria are met.

## Execution Model

- Parent-led native subagent execution is required for production-code changes.
- Use exactly one production-code subagent for implementation. The requested
  model is `gpt5.5 high` if available in the active tool surface.
- Use one independent review subagent for code review. The requested model is
  `gpt5.5 xhigh` if available in the active tool surface.
- The parent agent owns plan scope, prompt-flow audit acceptance, test
  selection, live LLM trace inspection, review-feedback decisions, final
  evidence recording, and user reporting.
- If the required production-code subagent or review subagent cannot be
  created, stop before production-code edits and report the missing capability.

## Progress Checklist

- [ ] Stage 0 - plan approval and baseline evidence
  - Covers: implementation steps 1-3.
  - Verify: current prompt, payload, registry projection, and worker prompt
    shapes are recorded in `Execution Evidence`.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 1 - focused tests establish the contract
  - Covers: implementation step 4.
  - Verify: focused tests are added and run with expected failures or baseline
    failures recorded.
  - Sign-off: `<agent/date>` after test output is recorded.
- [ ] Stage 2 - production implementation by one subagent
  - Covers: implementation steps 5-7.
  - Verify: subagent reports changed production files, commands run, blockers,
    and residual risks.
  - Sign-off: `<agent/date>` after parent reviews the subagent report.
- [ ] Stage 3 - deterministic verification
  - Covers: implementation step 8.
  - Verify: all deterministic commands in `Verification` pass, or failures are
    recorded and addressed before continuing.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 4 - live LLM verification
  - Covers: implementation step 8.
  - Verify: each live LLM case runs one at a time and trace quality is judged
    against the project goal.
  - Sign-off: `<agent/date>` after trace paths and judgments are recorded.
- [ ] Stage 5 - independent code review and remediation
  - Covers: implementation step 9.
  - Verify: review findings, accepted fixes, and rerun commands are recorded.
  - Sign-off: `<agent/date>` after review is clean or residual risks are accepted.
- [ ] Stage 6 - final lifecycle update
  - Covers: implementation step 10.
  - Verify: acceptance criteria are checked against evidence.
  - Sign-off: `<agent/date>` after status update.

## Verification

Static prompt and boundary checks:

```powershell
rg "用户消息是一段中文行动上下文字符串，不是 JSON|work_kind|task_type|coding_snippet|text_rewrite" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\action_router
```

Expected result: no matches in the action-router prompt or raw L2d output
schema, except test fixture text outside the searched source files.

```powershell
rg '"task":' src\kazusa_ai_chatbot\background_work\router.py src\kazusa_ai_chatbot\background_work\providers.py src\kazusa_ai_chatbot\background_work\subagent\text_artifact.py
```

Expected result: no router-decision or classifier-output field named `task`.
Generator payload tests may use source-summary names instead.

Deterministic tests:

```powershell
venv\Scripts\python -m pytest tests/test_action_router_payload.py tests/test_action_router_prompt_contract.py tests/test_persona_supervisor2_action_initializer.py tests/test_l2d_action_selection_cases.py tests/test_l2d_l3_surface_handoff.py -q
```

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_evaluator.py tests/test_background_work_router.py tests/test_background_work_providers.py tests/test_background_work_text_artifact.py tests/test_background_work_jobs.py tests/test_background_work_delivery.py -q
```

Live LLM tests must run one case at a time:

```powershell
$env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm -s
```

```powershell
venv\Scripts\python -m pytest tests/test_background_work_router_live_llm.py::test_background_work_router_live_case -q -m live_llm -s
```

```powershell
venv\Scripts\python -m pytest tests/test_background_work_text_artifact_live_llm.py::test_text_artifact_live_case -q -m live_llm -s
```

Each live trace must be inspected for:

- rendered system prompt;
- rendered human JSON payload;
- raw model output;
- normalized output;
- action specs or worker results;
- leakage errors;
- quality judgment aligned with the project goal.

## Independent Plan Review

Before production-code execution, an independent reviewer must check:

- the plan does not reopen completed background-work runtime scope;
- file lists are complete for the intended boundary correction;
- prompt rewrite and LLM drafting rules are explicit;
- `action_router` is outside `nodes`, with L2d delegating to it;
- the action-router input-audit procedure rejects raw graph state and
  operational fields;
- resolver output shape matches the downstream resolver contracts while keeping
  schema versions and pending ids deterministically bound;
- action registry projection changes are limited to prompt-facing affordance
  text and do not alter capability schemas or handlers;
- L2d remains route-only and does not generate worker-local parameters;
- background-work router and text-artifact classifier no longer combine routing
  with semantic parameter generation;
- verification includes deterministic and one-at-a-time live LLM evidence;
- no DB migration is required.

Review findings must be addressed in this plan before status changes from
`draft` to `approved`.

## Independent Code Review

After implementation and verification, the review subagent must inspect:

- route/parameter ownership boundaries across L2d, background-work router, and
  text-artifact worker;
- prompt-flow coherence after rewrites;
- prompt-safe payload projections and forbidden-field filtering;
- `nodes` owns L2d graph orchestration while `action_router` owns reusable
  prompt, payload, contract, and normalization code;
- compatibility of public L2d result shape and action-spec materialization;
- absence of raw ids, handler ids, DB fields, adapter fields, worker-local
  fields, or final visible text in action-router prompt inputs and outputs;
- deterministic test quality and live LLM trace quality.

The parent agent must address accepted review findings before final sign-off.

## Acceptance Criteria

- L2d action initialization delegates to top-level `action_router`, receives
  JSON human input, and no longer depends on a prose action-context string.
- `src/kazusa_ai_chatbot/action_router/README.md` exists and documents the
  module in ICD format.
- The static action-router prompt no longer manually duplicates the action capability
  roster; prompt-facing action affordances come from runtime-safe projections.
- `background_work_request` prompt affordance no longer instructs the model to
  emit `task_brief`; deterministic materialization builds the trusted
  action-spec param from `work_seed`.
- The action-router prompt is rewritten as a coherent flow and reviewed for source
  recognition, resolver continuation, resolver-first evidence needs, goal
  progress, top-level action routing, and visible-surface need.
- Raw L2d output for background work contains no worker, task type, task brief,
  tool parameter, job id, adapter target, or final visible text.
- Background-work router output contains no worker-facing task string.
- Text-artifact classifier output contains only task type and reason.
- Existing public action-spec, queue, and delivery contracts remain compatible.
- No new live-response LLM call is added.
- Focused deterministic tests pass.
- Required live LLM traces are produced and inspected one at a time.
- Independent code review is completed and accepted findings are addressed.

## Risks

- JSON payloads may increase token usage if projection is not bounded.
- Removing L2d-generated task briefs may reduce worker specificity unless the
  deterministic queue/source summary is clear.
- Local LLMs may need the prompt rewrite to be shorter and more procedural than
  the current accumulated prompt.
- Existing tests may assert old prose payloads or durable field terminology;
  update them without changing behavioral intent.

## Execution Evidence

- Status: not started.
- Independent plan review: performed on 2026-06-07; surfaced issues were
  cutover enforcement, resolver output shape, registry projection conflict,
  progress-gate granularity, and action-spec README scope; addressed in draft.
- Current-state prompt capture: pending.
- LLM modification audit: pending.
- Deterministic verification: pending.
- Live LLM verification: pending.
- Independent code review: pending.
