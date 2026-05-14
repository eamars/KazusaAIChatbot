# rag 2.1 initializer subagent contract plan

## Summary

- Goal: Shift RAG2 responsibility so the initializer chooses evidence paths and
  dependencies, while top-level capability agents own admission, parameter
  generation, worker selection, and source-authority envelopes.
- Plan class: large
- Status: draft
- Mandatory skills: `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible
- Highest-risk areas: breaking cached initializer strategies, increasing
  response-path latency, creating hidden dispatcher repair logic, leaking
  backend parameters into prompt-facing contracts, and weakening existing
  capability boundaries.
- Acceptance criteria: structured `slot_records` exist beside legacy
  `unknown_slots`, dispatcher routing is schema-driven and mechanical,
  top-level capability agents generate their own parameters, RAG evidence
  carries typed authority/failure status into cognition-facing projection, and
  no existing database data is changed.

## Context

The current RAG2 architecture intends this separation:

```text
initializer/planner -> semantic slot
dispatcher/router   -> choose specialist by prefix/capability
specialist agent    -> extract low-level parameters for its own domain
deterministic code  -> validate schema, limits, permissions, and execute
```

The implementation only partially enforces that boundary. The initializer emits
string slots with capability prefixes, but those strings still carry
capability-specific details such as speaker scope and dependency wording. The
dispatcher maps prefixes mechanically for recognized slots, then passes the
whole string through. Top-level capability agents already own much of the
worker choice, but admission and failure semantics are not consistently exposed
as a first-class contract.

The incident showed why this matters: a slot that should have been transcript
evidence contained current-episode wording, and the conversation-evidence
capability rejected it as Recall. A narrow bugfix addresses that immediate
failure. This RAG 2.1 plan addresses the broader system issue: route
compatibility, evidence authority, and parameter ownership must be explicit,
typed, and testable.

This plan does not mutate existing data. It is a compatible architecture
upgrade for future runs.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG prompt, routing,
  capability contracts, evidence projection, cognition-facing payloads, or LLM
  call budgets.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python tests or source containing Chinese
  or Japanese strings.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record
  the result in execution evidence.
- Do not mutate existing MongoDB data, conversation history rows,
  conversation-progress state, user memories, or existing cache rows.
- Do not add negative prompt constraints to fix the current-episode
  Conversation-evidence failure mode.
- Do not make the initializer generate MongoDB fields, aggregation syntax,
  worker names, search parameters, global user IDs, timestamp filters, limits,
  cache keys, or tool arguments.
- Do not add deterministic semantic repair or rerouting in the shared
  dispatcher after a capability refuses a slot.
- Do not add a response-path LLM call beyond the existing initializer,
  dispatcher fallback where still present, capability workers, and finalizer
  calls.
- Keep adapter, queue, persistence, consolidation, reflection, and scheduler
  behavior outside this plan unless a listed integration test proves a direct
  contract dependency.
- Keep all old `unknown_slots` trace consumers working during the migration by
  deriving string labels from structured slot records.
- Because this plan bumps initializer strategy/cache versions while forbidding
  existing cache-row mutation, runtime startup must stop purging stale
  initializer cache rows. Stale initializer rows must remain stored and be
  ignored by version-key filtering.

## Must Do

- Define a structured RAG strategy schema with `slot_records`.
- Keep legacy `unknown_slots` as derived labels during the compatible cutover.
- Bump initializer strategy/cache schema version so old cached strategies do
  not mix with RAG 2.1 records.
- Remove the startup stale-initializer purge path so Cache2 version bumps do
  not delete existing persistent initializer rows.
- Make the initializer output path-level facts only: capability, evidence need,
  dependencies, expected reference types, and freshness class.
- Make the dispatcher prefer `slot_records`, validate schema, map capability to
  agent through a registry, and pass typed runtime context through unchanged.
- Keep the legacy string-slot path only for old cached/test inputs during this
  compatible phase.
- Move capability-specific admission and parameter planning into top-level
  capability agents.
- Introduce a source-authority envelope in RAG known facts and projection.
- Preserve existing projection keys so cognition-facing callers do not require
  a bigbang state migration.
- Add deterministic tests for initializer schema parsing, dispatcher routing,
  capability admission/refusal, authority projection, and legacy compatibility.
- Add live LLM initializer checks one case at a time after deterministic tests
  pass.

## Deferred

- Do not remove legacy `unknown_slots` in this plan.
- Do not remove the dispatcher LLM fallback unless every legacy fallback caller
  is proven obsolete by greps and tests inside this plan.
- Do not redesign conversation-progress persistence.
- Do not repair historical incident data or cache rows.
- Do not add a new RAG capability.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Initializer output | compatible | Add `slot_records` and derive legacy `unknown_slots` labels. Do not delete `unknown_slots` consumers in this plan. |
| Cache2 strategy schema | compatible | Bump schema/version and keep old cache rows ignored by version mismatch. Do not migrate existing cache data. |
| Cache2 startup hydration | compatible | Remove the bootstrap-time stale initializer purge call. Hydration must load only current-version rows and leave stale rows stored. |
| Dispatcher | compatible | Prefer structured slots. Keep legacy string-prefix dispatch for old inputs during this plan. |
| Capability agents | compatible | Accept structured slot context while keeping existing `run(task, context, max_attempts)` call shape. |
| Web evidence | compatible | `web_evidence` routes to existing `web_search_agent2` as the one approved direct-worker exception in this plan. The dispatcher still attaches `rag_slot_record`; evaluator/projection wrap the direct web result in an authority envelope. Do not create a new `web_evidence_agent` in this plan. |
| Evidence projection | compatible | Add authority fields without removing existing summary/raw result fields. |
| Data | compatible | No existing data changes, migrations, or backfills. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative or broader strategy by default.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local helper names only when they preserve the public
  contracts in this plan.
- The agent must not introduce alternate retrieval architecture, fallback
  rerouting, multi-pass repair, new persistence, or unrelated prompt rewrites.
- The agent must treat edits outside RAG initializer, dispatcher, capability
  agents, RAG projection/evaluator, and their tests as out of scope unless a
  listed integration test proves the dependency.
- If equivalent validation or projection helpers already exist, reuse or extend
  them instead of duplicating logic.
- If a required instruction conflicts with current code, preserve the plan's
  stated ownership model and report the discrepancy before inventing a
  substitute.

## Target State

The initializer produces structured path-level strategy:

```json
{
  "strategy_schema_version": "rag_strategy:v2",
  "slot_records": [
    {
      "slot_id": "s1",
      "capability": "conversation_evidence",
      "need": "verify whether the active character previously asked for the yandere definition",
      "depends_on": [],
      "expected_refs": ["message"],
      "freshness": "historical"
    }
  ],
  "unknown_slots": [
    "Conversation-evidence: verify whether the active character previously asked for the yandere definition"
  ]
}
```

The dispatcher validates the slot schema, maps `capability` to the registered
top-level agent, and passes:

```python
{
    "rag_slot_record": {
        "slot_id": "s1",
        "capability": "conversation_evidence",
        "need": "...",
        "depends_on": [],
        "expected_refs": ["message"],
        "freshness": "historical",
    },
    "original_query": "...",
    "known_facts": [...],
    "runtime_context": {...},
}
```

Top-level capability agents perform admission and parameter planning inside
their domain. They return the existing result shape plus an authority envelope:

```python
{
    "resolved": True,
    "result": {
        "capability": "conversation_evidence",
        "selected_summary": "...",
        "primary_worker": "conversation_search_agent",
        "resolved_refs": [...],
        "authority_envelope": {
            "slot_id": "s1",
            "route_capability": "conversation_evidence",
            "source_system": "conversation_history",
            "authority": "prior_message_evidence",
            "failure_status": "resolved_authoritative",
            "allowed_uses": ["prior_message_wording", "speaker_attribution"],
            "freshness_basis": "historical_chat",
            "scope_global_user_id": None,
            "resolved": True,
            "supporting_refs": [...]
        }
    }
}
```

Unresolved cases also carry authority:

```python
{
    "authority_envelope": {
        "slot_id": "s1",
        "route_capability": "conversation_evidence",
        "source_system": "conversation_history",
        "authority": "prior_message_evidence",
        "failure_status": "unresolved_missing_context",
        "allowed_uses": [],
        "freshness_basis": "historical_chat",
        "scope_global_user_id": None,
        "resolved": False,
        "supporting_refs": []
    }
}
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Initializer responsibility | Initializer owns evidence path and dependency order only | This keeps weak local LLM planning bounded and avoids backend-parameter leakage. |
| Dispatcher responsibility | Dispatcher validates schema and maps capability mechanically | Shared orchestration should stay boring and inspectable. |
| Capability responsibility | Top-level capability agents own admission, worker selection, and parameter extraction | Domain agents have the context to decide search vs filter vs aggregate, memory exact vs semantic, profile lookup vs list, and Recall mode. |
| Failure handling | Capability refusals remain structured and are not rerouted by dispatcher | Rerouting after refusal recreates hidden semantic repair in the shared layer. |
| Authority handling | RAG projection carries authority envelopes into cognition-facing payloads | Downstream stages need source-authority semantics, not anonymous text blobs. |
| Compatibility | Keep legacy `unknown_slots` through this plan | Existing traces, tests, and cache-facing code still expect string slot labels. |
| Web evidence | Keep web as a direct-worker exception | The current code routes `Web-evidence:` to `web_search_agent2`; adding a new web capability is outside this plan. |
| Data | No data mutation or migration | The user explicitly disallowed existing data changes. |

## Contracts And Data Shapes

### `RAGStrategySlot`

```python
{
    "slot_id": str,
    "capability": str,
    "need": str,
    "depends_on": list[dict[str, str]],
    "expected_refs": list[str],
    "freshness": str,
}
```

Allowed `capability` values:

```text
live_context
conversation_evidence
memory_evidence
person_context
recall
web_evidence
```

Allowed `freshness` values:

```text
runtime
live
historical
durable
episode
```

Allowed `expected_refs` values:

```text
message
person
memory
profile
relationship
url
runtime_fact
live_fact
episode_state
```

### Initializer Output

```python
{
    "strategy_schema_version": "rag_strategy:v2",
    "slot_records": list[RAGStrategySlot],
    "unknown_slots": list[str],
}
```

`unknown_slots` is derived from `slot_records` for trace compatibility. The
initializer does not emit worker names, database fields, search filters,
global user IDs, timestamps, limits, cache keys, or tool arguments.

### Dispatcher Input And Output

Input state includes:

```python
{
    "slot_records": list[RAGStrategySlot],
    "current_slot_record": RAGStrategySlot,
    "unknown_slots": list[str],
    "current_slot": str,
    "known_facts": list[dict],
    "context": dict,
}
```

Dispatcher output includes:

```python
{
    "current_dispatch": {
        "agent_name": str,
        "task": str,
        "context": dict,
        "max_attempts": int,
        "slot_record": RAGStrategySlot,
        "route_source": "structured_slot" | "legacy_prefix" | "legacy_llm",
    }
}
```

### Authority Envelope

```python
{
    "slot_id": str,
    "route_capability": str,
    "source_system": str,
    "authority": str,
    "failure_status": str,
    "allowed_uses": list[str],
    "freshness_basis": str,
    "scope_global_user_id": str | None,
    "resolved": bool,
    "supporting_refs": list[dict],
}
```

Use `None` for unscoped evidence. Only scoped user-memory continuity,
explicitly user-scoped conversation evidence, profile evidence, and
relationship evidence may set `scope_global_user_id`.

Allowed `failure_status` values:

```text
no_evidence_needed
unresolved_missing_context
unresolved_incompatible_route
resolved_supporting_only
resolved_authoritative
stale_or_wrong_authority
```

### RAG Projection Authority Key

`rag_result` must preserve all existing public keys and value types. Add a new
sibling key:

```python
{
    "evidence_authority": list[AuthorityEnvelope],
}
```

Each resolved or unresolved top-level RAG fact contributes at most one
authority envelope. Existing `conversation_evidence`, `memory_evidence`,
`recall_evidence`, and `external_evidence` item shapes must not be changed in
this plan.

## LLM Call And Context Budget

- Before: initializer makes one planner call; dispatcher uses deterministic
  prefix mapping for recognized slots and may fall back to a dispatcher LLM;
  capability agents and workers use existing domain calls; finalizer uses one
  summarizer/finalizer call.
- After: no additional response-path LLM calls.
- Initializer prompt context changes from free-form slot strings to structured
  slot records. The payload remains under the existing 50k-token cap.
- Dispatcher structured-slot routing is deterministic and should reduce LLM
  fallback use.
- Capability agents receive a compact slot record plus the existing task label
  and runtime context.
- Authority envelopes are deterministic metadata and should not be expanded into
  long prompt prose.
- Verification must include prompt-render tests for initializer JSON examples
  and deterministic tests proving no new response-path call is introduced for
  structured dispatch.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py`
  - Add typed state fields or `TypedDict` definitions for strategy slots,
    dispatcher slot records, and authority envelopes.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Bump initializer strategy schema/version for RAG 2.1.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Remove the startup call that purges stale persistent initializer cache
    rows.
- `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py`
  - Keep `load_initializer_entries(...)` filtering by current version key.
  - Update stale-cache helper documentation to state that startup hydration
    ignores stale initializer rows instead of deleting them.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
  - Emit and validate `slot_records`.
  - Derive `unknown_slots` from `slot_records`.
  - Keep prompt wording path-level and schema-focused.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
  - Prefer structured `slot_records`.
  - Map capability to agent through the registry.
  - Preserve legacy prefix dispatch for old string slots.
  - Attach `rag_slot_record` for the direct `web_search_agent2` exception.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
  - Accept `rag_slot_record` in context and perform domain admission and
    parameter planning.
  - Emit authority envelopes.
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
  - Accept `rag_slot_record`, perform domain admission and parameter planning,
    and emit authority envelopes.
- `src/kazusa_ai_chatbot/rag/person_context_agent.py`
  - Accept `rag_slot_record`, perform domain admission and parameter planning,
    and emit authority envelopes.
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`
  - Accept `rag_slot_record`, perform domain admission and parameter planning,
    and emit authority envelopes.
- `src/kazusa_ai_chatbot/rag/recall_agent.py`
  - Accept `rag_slot_record`, perform Recall-mode admission, and emit authority
    envelopes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
  - Preserve authority envelope into `known_facts`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - Project authority envelope metadata into cognition-facing RAG payloads
    without removing existing summary fields.
- `tests/test_db.py`
  - Update bootstrap tests so startup no longer calls stale initializer purge.
- `tests/test_rag_cache2_persistent.py`
  - Add or update tests proving stale initializer rows remain stored and
    `load_initializer_entries(...)` hydrates only current-version rows.
- RAG tests under `tests/`
  - Add deterministic and live LLM tests listed in `Verification`.

### Keep

- Adapters, queue/intake, persistence, consolidation, reflection, and scheduler
  modules stay unchanged unless a test identifies a direct compile or contract
  break from the RAG state shape.
- Existing worker-level search/filter/aggregate implementations keep owning
  physical database parameters and execution limits.
- Existing `unknown_slots` traces remain available during this plan.

## Implementation Order

1. Add typed strategy-slot and authority-envelope contract tests.
   - Target tests: new focused tests in `tests/test_rag_strategy_slot_contract.py`.
   - Expected before implementation: missing schema helpers or validation
     failure.
2. Implement slot and envelope types.
   - Target: `persona_supervisor2_rag_types.py`.
   - Keep helpers deterministic and small.
3. Add initializer schema tests.
   - Cover valid `slot_records`, derived `unknown_slots`, invalid extra
     backend fields being rejected with a model-contract event, dependency
     validation, and cache version separation.
4. Update initializer output and Cache2 version.
   - Target: `persona_supervisor2_rag_initializer.py` and `rag/cache2_policy.py`.
   - Run prompt-render tests after every prompt edit.
5. Remove bootstrap-time stale initializer purge.
   - Target: `src/kazusa_ai_chatbot/db/bootstrap.py`,
     `tests/test_db.py`, and `tests/test_rag_cache2_persistent.py`.
   - Expected result: stale initializer rows are ignored by version filtering
     and are not deleted during startup hydration.
6. Add dispatcher structured-routing tests.
   - Cover each allowed capability and legacy string-slot fallback.
   - Cover invalid capability and invalid dependency behavior.
   - Cover `web_evidence` as the direct `web_search_agent2` exception.
7. Update dispatcher to prefer structured slots.
   - Target: `persona_supervisor2_rag_dispatch.py`.
   - Keep fallback path visible through `route_source`.
8. Add top-level capability admission tests.
   - Cover Conversation-evidence, Memory-evidence, Person-context,
     Live-context, and Recall accepted/refused cases.
9. Update top-level capability agents to consume `rag_slot_record`.
   - Keep the existing `run(task, context, max_attempts)` method signature.
   - Put structured slot data in context under `rag_slot_record`.
10. Add authority-envelope evaluator/projection tests.
   - Cover resolved authoritative, unresolved missing context,
     incompatible route, scoped memory, Recall episode state, and conversation
     transcript evidence.
   - Add
     `tests/test_rag_projection.py::test_project_known_facts_adds_evidence_authority_without_changing_existing_lists`.
11. Implement authority-envelope propagation through evaluator and projection.
12. Add live LLM initializer tests.
   - Run one case at a time and inspect output logs before moving to the next.
   - Cover active agreement Recall, exact phrase Conversation-evidence,
     active-character self-word Conversation-evidence, live weather/location,
     and user-list/person-context.
13. Run full verification.
14. Run independent code review and record findings before completion.

## Progress Checklist

- [ ] Stage 1 - slot and envelope contracts established
  - Covers: implementation order steps 1-2.
  - Verify: focused strategy-slot contract tests pass.
  - Evidence: record expected failure before implementation and pass after.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - initializer emits structured strategy
  - Covers: implementation order steps 3-4.
  - Verify: initializer schema, prompt-render, and cache-version tests pass.
  - Evidence: record changed files and command output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - Cache2 startup preserves stale initializer rows
  - Covers: implementation order step 5.
  - Verify: bootstrap and persistent Cache2 tests prove startup does not purge
    stale initializer rows and hydration ignores stale versions.
  - Evidence: record row-retention assertions and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - dispatcher structured routing complete
  - Covers: implementation order steps 6-7.
  - Verify: dispatcher structured-routing and legacy fallback tests pass.
  - Evidence: record route-source behavior and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - capability agents own admission and parameters
  - Covers: implementation order steps 8-9.
  - Verify: capability-agent admission/refusal tests pass.
  - Evidence: record worker-call assertions and test output.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - authority envelope reaches projection
  - Covers: implementation order steps 10-11.
  - Verify: evaluator/projection authority tests pass.
  - Evidence: record output shapes and test output.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 - live LLM and regression verification complete
  - Covers: implementation order steps 12-13.
  - Verify: live LLM tests run one case at a time and regression tests pass.
  - Evidence: record trace files, inspected outputs, and command results.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 8 - independent code review complete
  - Covers: implementation order step 14.
  - Verify: review full diff against this plan, project style, and regression
    risk; rerun affected tests after any review fix.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked complete only after this stage is signed off.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg -n "slot_records|rag_slot_record|authority_envelope" src tests`
  - Expected: matches only in RAG initializer, dispatcher, types, capability
    agents, evaluator/projection, and tests.
- `rg -n "global_user_id|platform_message_id|from_timestamp|to_timestamp|limit" src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py`
  - Expected: no initializer-generated backend parameter examples for RAG 2.1
    slot records. Existing runtime context references are allowed only when
    they are not model-output fields.
- `rg -n "negative constraint|forbidden_phrases" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag`
  - Expected: if this command returns matches, inspect `git diff` and verify
    this plan added no prompt-negative fix for the current-episode
    Conversation-evidence failure. Existing unrelated prompt text may remain.

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests/test_rag_strategy_slot_contract.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_initializer_cache2.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_db.py tests/test_rag_cache2_persistent.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py::test_project_known_facts_adds_evidence_authority_without_changing_existing_lists -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py -q`

### Live LLM Tests

Run each live LLM initializer case one at a time and inspect output before the
next case:

- `venv\Scripts\python.exe -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_active_agreement_to_recall -q -s`
- `venv\Scripts\python.exe -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_exact_phrase_to_conversation_evidence -q -s`
- `venv\Scripts\python.exe -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_active_character_self_words_stays_conversation -q -s`
- `venv\Scripts\python.exe -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_routes_character_local_temperature_to_live_context -q -s`
- `venv\Scripts\python.exe -m pytest -m live_llm tests/test_rag_phase3_initializer_live_llm.py::test_live_initializer_named_person_impression_stays_person_context -q -s`

Use the project's real-LLM test execution policy from
`test-style-and-execution`. Record trace artifact paths and inspected verdicts.

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_projection.py tests/test_rag_initializer_cache2.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_db.py tests/test_rag_cache2_persistent.py -q`

### Data Safety

- `git diff --stat`
  - Expected: source and test files plus this plan/registry updates only.
- No command in this plan writes to MongoDB, migrates collections, backfills
  rows, or edits exported incident artifacts.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and
  path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change
surface. If a fix would cross the approved boundary or alter the contract, stop
and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
execution evidence.

## Acceptance Criteria

This plan is complete when:

- `slot_records` are produced by the initializer and legacy `unknown_slots` are
  derived for compatibility.
- Dispatcher structured routing works for every allowed capability and records
  route source.
- Top-level capability agents own admission and parameter planning for their
  domains.
- Authority envelopes are present in known facts and cognition-facing RAG
  projection.
- `rag_result.evidence_authority` is added without changing the existing
  `conversation_evidence`, `memory_evidence`, `recall_evidence`, or
  `external_evidence` item shapes.
- Runtime startup no longer purges stale persistent initializer cache rows.
- `web_evidence` is handled through the existing direct `web_search_agent2`
  exception and receives an authority envelope through evaluator/projection.
- Existing string-slot tests and traces remain compatible during this plan.
- No LLM prompt-negative constraint was added for the incident failure mode.
- No existing database data was changed.
- Focused, regression, and required live LLM tests pass or blocked commands are
  recorded with concrete reasons.
- Independent code review is complete and recorded.
