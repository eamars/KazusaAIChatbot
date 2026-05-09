# multi source cognition architecture stage 04 rag cognitive episode adapter plan

## Summary

- Goal: Refactor RAG input construction so the persona graph builds current
  `/chat` RAG requests from `CognitiveEpisode` through a narrow adapter while
  preserving existing text-chat retrieval behavior.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` if Python prompt or test strings with CJK content are edited.
- Overall cutover strategy: compatible refactor of the RAG request-building
  boundary; `/chat` continues to send the same `original_query`, `character_name`,
  and RAG `context` to `call_rag_supervisor`.
- Highest-risk areas: changed RAG initializer inputs, Cache 2 key drift,
  prompt payload drift, retrieval routing drift, hidden fallback paths, and
  accidentally enabling non-chat sources before origin policy exists.
- Acceptance criteria: current text `/chat` retrieval inputs are equivalent
  before and after the adapter, frozen Stage 00 evidence still projects to the
  same `rag_result`, and non-chat trigger sources remain disabled.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_04`

Execution status: blocked. This plan must not be executed until `stage_02` and
`stage_03` are both marked `completed` in the parent ledger and this plan is
explicitly approved.

## Context

The top-level architecture requires the current `/chat` path to migrate first,
then prompt selection, then RAG input construction. Stage 04 is therefore a RAG
boundary refactor, not a new retrieval feature.

The current runtime shape is:

```text
persona_supervisor2.stage_1_research
-> call_rag_supervisor(
       original_query=state["decontexualized_input"],
       character_name=state["character_profile"]["name"],
       context={platform, channel, user, time, history, progress, ...}
   )
-> project_known_facts(...)
-> state["rag_result"]
```

The target shape is:

```text
persona_supervisor2.stage_1_research
-> build_text_chat_rag_request(cognitive_episode, legacy supplemental fields)
-> call_rag_supervisor(same original_query, same character_name, same context)
-> project_known_facts(...)
-> state["rag_result"]
```

Prior-stage artifacts that must exist before execution:

- Stage 00 baseline test:
  `tests/test_multi_source_cognition_stage_00_regression_baseline.py`.
- Stage 00 fixture and frozen evidence corpus:
  `tests/fixtures/multi_source_cognition_stage_00_cases.json`.
- Stage 01 contract module:
  `src/kazusa_ai_chatbot/cognition_episode.py`.
- Stage 01 contract tests:
  `tests/test_cognitive_episode_contract.py`.
- Stage 02 completion artifacts:
  `/chat` episode wiring in `src/kazusa_ai_chatbot/service.py`,
  optional `cognitive_episode` graph state fields, pass-through tests, and
  Stage 00 baseline rerun evidence.
- Stage 03 completion artifacts:
  source-aware prompt selector artifact, prompt-render tests, and Stage 00
  baseline rerun evidence.

Current blocking status on 2026-05-09:

- Stage 02 is still `draft`.
- Stage 03 is still `draft`.
- Stage 04 is a draft authoring artifact only.

## Mandatory Skills

- `development-plan-writing`: preserve parent-stage boundaries and lifecycle.
- `local-llm-architecture`: protect RAG initializer inputs, local-LLM prompt
  boundaries, context budgets, and live response latency.
- `no-prepost-user-input`: do not add deterministic semantic interpretation of
  user text while adapting RAG inputs.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00`, `stage_01`,
  `stage_02`, and `stage_03` are `completed`.
- Before editing, read the Stage 00, Stage 01, Stage 02, and Stage 03 execution
  evidence and confirm every artifact path listed in `Context` exists.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or
  final reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Keep the current `/chat` RAG `original_query` exactly equal to
  `state["decontexualized_input"]`.
- Keep the current `/chat` RAG `character_name` exactly equal to
  `state["character_profile"]["name"]`.
- Keep the current `/chat` RAG context keys and values equivalent to the
  pre-adapter `stage_1_research` context.
- Do not change `call_rag_supervisor` signature or return shape.
- Do not change `project_known_facts` behavior or `rag_result` public keys.
- Do not change RAG initializer, dispatcher, evaluator, finalizer, helper
  agents, Cache 2 policy, cache keys, cache invalidation, or retrieval tools.
- Do not add prompt text, prompt variants, prompt examples, prompt payload keys,
  or prompt selection rules in this stage.
- Do not add live `/chat` LLM calls or increase RAG loop caps.
- Do not add deterministic keyword routing, query classification, user-intent
  filtering, commitment detection, permission detection, or semantic repair
  over user text.
- Do not enable reflection, internal thought, scheduled recall, image, audio,
  proactive, or system-probe trigger sources in runtime.
- Do not add database schema changes, migrations, new persistence writes, new
  queues, new schedulers, new adapter behavior, or service startup behavior.

## Must Do

- Add a RAG episode adapter module with the exact public API in
  `Adapter Interface Contract`.
- Prove the adapter independently with module tests.
- Refactor `persona_supervisor2.stage_1_research` to use the adapter for the
  RAG supervisor request when RAG is not skipped.
- Preserve the current unresolved-referent RAG skip behavior and projected
  empty `rag_result` shape.
- Add equivalence tests proving the adapter emits the same `/chat` RAG request
  shape that `stage_1_research` built before this stage.
- Use the Stage 00 frozen evidence corpus to prove projected retrieval output
  remains equivalent.
- Document future `episode_focus` rules as non-executable notes only.
- Rerun Stage 00, Stage 01, Stage 02, and Stage 03 deterministic gates required
  by this plan.

## Deferred

- Stage 02 `/chat` episode wiring. This plan consumes it; it does not create it.
- Stage 03 prompt selector creation or prompt tuning.
- Consolidation origin metadata threading and origin policy.
- Reflection-triggered RAG, internal-thought RAG, scheduled-recall RAG, image
  RAG, audio RAG, proactive output, or transport policy.
- Changing the RAG supervisor graph, helper-agent roster, dispatch prefixes,
  initializer examples, evaluator summaries, finalizer prompt, or Cache 2
  persistence.
- Passing raw `CognitiveEpisode` objects into model-facing prompt payloads.
- Adding `episode_focus` to the live `/chat` RAG context.

## Cutover Policy

Policy: `compatible`.

The adapter becomes the only request-building path used by
`persona_supervisor2.stage_1_research` for text `/chat` RAG calls. The adapter
must produce the same `call_rag_supervisor` input contract as the current inline
code. There is no dual runtime path and no feature flag.

Rollback path: revert the adapter module, restore the inline
`call_rag_supervisor` argument construction in `stage_1_research`, remove the
Stage 04 tests and README notes, and rerun Stage 00 through Stage 03 gates. No
database rollback is required.

Stop condition: if the adapter changes `original_query`, `character_name`,
RAG context keys, projected `rag_result`, RAG prompt payloads, Cache 2 keys,
retrieval routing, debug-mode behavior, prompt rendering, or Stage 00 baseline
behavior, stop Stage 04 and create a bugfix plan.

## Agent Autonomy Boundaries

- The implementation agent must use the exact module path, public type names,
  public function name, function signature, and output shape in
  `Adapter Interface Contract`.
- The implementation agent may choose private helper names inside the adapter
  only when the public API and behavior remain unchanged.
- The implementation agent must not choose alternate context keys, alternate
  query source rules, alternate unsupported-trigger behavior, alternate
  fallback paths, or alternate future-source mappings.
- The implementation agent must not create a legacy fallback inside
  `stage_1_research`. If `cognitive_episode` is missing after Stage 02, fail
  loudly in tests and report the prerequisite regression.
- The implementation agent must not edit files outside `Change Surface`.
- If the plan and code disagree, preserve this plan's stated behavior and
  report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Current `/chat` RAG calls are built through a source-aware adapter boundary, but
retrieval remains behaviorally unchanged:

```text
cognitive_episode + decontexualized_input + legacy supplemental fields
-> RAGEpisodeRequest
-> call_rag_supervisor(original_query, character_name, context)
-> project_known_facts(...)
-> rag_result
```

Only `trigger_source="user_message"` with `input_sources=["dialog_text"]` is
accepted by runtime code in this stage. Every other trigger source raises the
adapter's structural unsupported-source error and remains for later child
plans.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Adapter module | Create `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`. | RAG owns request construction; persona graph should call a narrow RAG boundary. |
| Public API | Add one text-chat request builder and one adapter error class. | Stage 04 should not become a generic multi-source RAG framework. |
| Query source | Use `decontexualized_input` for `/chat` `original_query`. | Parent plan requires current text retrieval to remain unchanged. |
| Context source | Use `CognitiveEpisode` for source-neutral identity/origin fields and explicit supplemental arguments for existing RAG-only context. | Keeps the episode compact while preserving current RAG context. |
| Unsupported triggers | Raise `RAGEpisodeAdapterError` for non-`user_message` triggers. | Later stages must not silently reuse text-chat retrieval for non-chat sources. |
| RAG supervisor signature | Keep unchanged. | Avoids graph, prompt, Cache 2, and helper-agent churn. |
| Future source focus | Document future `episode_focus` rules only. | Stage 04 can define direction without enabling non-chat sources. |

## Adapter Interface Contract

Create `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`.

The module must import `CognitiveEpisode` and `validate_cognitive_episode` from
`kazusa_ai_chatbot.cognition_episode`. It must not import `service.py`,
`persona_supervisor2.py`, RAG supervisor modules, database functions, LLMs,
MCP tools, schedulers, or adapter code.

The module must expose exactly these public names:

```python
class RAGEpisodeAdapterError(ValueError):
    """Raised when an episode cannot be adapted into a RAG request."""


class RAGEpisodeRequest(TypedDict):
    original_query: str
    character_name: str
    context: dict[str, Any]
    current_user_id: str
    character_user_id: str


def build_text_chat_rag_request(
    *,
    episode: CognitiveEpisode,
    decontexualized_input: str,
    character_profile: dict[str, Any],
    user_profile: dict[str, Any],
    prompt_message_context: dict[str, Any],
    channel_topic: str,
    chat_history_recent: list[dict],
    chat_history_wide: list[dict],
    reply_context: dict[str, Any],
    indirect_speech_context: str,
    conversation_progress: dict[str, Any] | None = None,
    conversation_episode_state: dict[str, Any] | None = None,
    promoted_reflection_context: dict[str, Any] | None = None,
) -> RAGEpisodeRequest:
    """Build the current text `/chat` RAG supervisor request from an episode."""
```

`build_text_chat_rag_request` must:

- Call `validate_cognitive_episode(episode)` before reading nested fields.
- Raise `RAGEpisodeAdapterError` when `episode["trigger_source"]` is not
  `"user_message"`.
- Raise `RAGEpisodeAdapterError` when `episode["input_sources"]` is not exactly
  `["dialog_text"]`.
- Raise `RAGEpisodeAdapterError` when `character_profile["global_user_id"]` or
  `character_profile["name"]` is missing or not a non-empty string.
- Return `original_query` equal to the `decontexualized_input` argument.
- Return `character_name` equal to `character_profile["name"]`.
- Return `current_user_id` equal to
  `episode["target_scope"]["current_global_user_id"]`.
- Return `character_user_id` equal to `character_profile["global_user_id"]`.
- Return a one-level copied `context` dict with the exact keys and values in
  `RAG Context Contract`.
- Not mutate input dicts or lists.
- Not call RAG, LLMs, databases, cache runtime, schedulers, or adapters.

## RAG Context Contract

For current text `/chat`, the adapter must return a context dict equivalent to
the current inline `stage_1_research` context:

```python
{
    "platform": episode["target_scope"]["platform"],
    "platform_channel_id": episode["target_scope"]["platform_channel_id"],
    "channel_type": episode["target_scope"]["channel_type"],
    "character_profile": {
        "global_user_id": character_profile["global_user_id"],
        "name": character_profile["name"],
    },
    "active_turn_platform_message_ids": list(
        episode["origin_metadata"]["active_turn_platform_message_ids"]
    ),
    "active_turn_conversation_row_ids": list(
        episode["origin_metadata"]["active_turn_conversation_row_ids"]
    ),
    "global_user_id": episode["target_scope"]["current_global_user_id"],
    "user_name": episode["target_scope"]["current_display_name"],
    "user_profile": dict(user_profile),
    "current_timestamp": episode["timestamp"],
    "time_context": episode["time_context"],
    "prompt_message_context": dict(prompt_message_context),
    "channel_topic": channel_topic,
    "chat_history_recent": list(chat_history_recent),
    "chat_history_wide": list(chat_history_wide),
    "reply_context": dict(reply_context),
    "indirect_speech_context": indirect_speech_context,
    "conversation_progress": conversation_progress,
    "conversation_episode_state": conversation_episode_state,
    "promoted_reflection_context": promoted_reflection_context,
}
```

The context must not include these keys:

- `cognitive_episode`
- `message_envelope`
- `episode_focus`
- `trigger_source`
- `input_sources`
- raw percept payloads
- raw attachment payloads
- consolidation policy fields

## Future Episode Focus Contract

This section is documentation only for Stage 04. The implementation must not
enable these trigger sources.

Future child plans must produce `episode_focus` as follows:

| Trigger source | Future focus source | Runtime status in Stage 04 |
|---|---|---|
| `user_message` | Current `decontexualized_input` from the message decontextualizer. | Active for current `/chat`; not added as a context key. |
| `reflection_signal` | The promoted `reflection_artifact` percept content, not raw reflection output. | Disabled. |
| `internal_thought` | The approved private `internal_monologue` percept content or action-latch residue. | Disabled. |
| `scheduled_recall` | The validated recall or scheduled-event percept content. | Disabled. |
| `system_probe` | The diagnostic probe percept content. | Disabled. |
| `image_observation` input | A compact image summary percept, not raw binary data. | Disabled. |
| `audio_observation` input | A transcript or tone-summary percept, not raw audio data. | Disabled. |

Future source support belongs in later child plans after origin policy and dry
run audit paths exist.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
  - Public RAG episode adapter boundary defined above.
- `tests/test_rag_cognitive_episode_adapter.py`
  - Module tests for adapter request shape, input copying, unsupported trigger
    handling, unsupported input-source handling, required character fields, and
    no mutation.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Replace inline non-skip RAG supervisor request construction in
    `stage_1_research` with `build_text_chat_rag_request`.
  - Keep unresolved-referent skip behavior unchanged.
  - Keep `call_rag_supervisor` and `project_known_facts` calls semantically
    equivalent.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Add a short note that text `/chat` RAG request construction is owned by
    `rag.cognitive_episode_adapter`, while `call_rag_supervisor` remains the
    runtime entry point.
- `tests/test_persona_supervisor2_rag2_integration.py`
  - Add `cognitive_episode` to direct `stage_1_research` fixtures.
  - Assert captured RAG request fields and context match the pre-adapter shape.
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - Keep Stage 00 behavioral assertions intact and add any necessary
    `cognitive_episode` fixture state required after Stage 02.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md`
  - Update checklist and `Execution Evidence` only during execution.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  - Update the `stage_04` ledger row to `completed` only after verification
    passes.
- `development_plans/README.md`
  - Update the Stage 04 registry row only after completion. Both `Status` and
    `Execution` columns must move to `completed` in the same edit.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py`
- `src/kazusa_ai_chatbot/rag/cache2_*.py`
- RAG helper agents under `src/kazusa_ai_chatbot/rag/**`

### Forbidden

- L1/L2/L3 cognition prompts or prompt payload builders.
- Dialog prompts, dialog evaluator, response targeting, or delivery tracking.
- Consolidator prompts, write policy, persistence, cache invalidation, or
  reflection integration.
- Service endpoint, queue, adapter, scheduler, dispatcher, database schema,
  or startup files.
- Any file in `experiments/cognition_core_next/`.

## Implementation Order

1. Add adapter module tests.
   - Create `tests/test_rag_cognitive_episode_adapter.py`.
   - Use `build_text_chat_cognitive_episode` from Stage 01 to create valid
     episodes.
   - Assert exact `RAGEpisodeRequest` output shape and context key set.
   - Run the test and record the expected missing-module or missing-symbol
     failure in `Execution Evidence`.
2. Implement the adapter module.
   - Add `RAGEpisodeAdapterError`, `RAGEpisodeRequest`, and
     `build_text_chat_rag_request`.
   - Keep all behavior deterministic and structural.
   - Run module tests and iterate only inside the adapter module and adapter
     tests until they pass.
3. Add integration tests around `stage_1_research`.
   - Update direct research-state fixtures to include `cognitive_episode`.
   - Patch `call_rag_supervisor` and capture its arguments.
   - Assert the captured `original_query`, `character_name`, and context match
     the pre-adapter contract.
   - Assert unresolved-referent skip behavior still avoids RAG.
4. Refactor `persona_supervisor2.stage_1_research`.
   - Build `rag_request = build_text_chat_rag_request(...)`.
   - Call `call_rag_supervisor` with `rag_request["original_query"]`,
     `rag_request["character_name"]`, and `rag_request["context"]`.
   - Call `project_known_facts` with `rag_request["current_user_id"]` and
     `rag_request["character_user_id"]`.
5. Add frozen evidence corpus equivalence coverage.
   - Use `tests/fixtures/multi_source_cognition_stage_00_cases.json` top-level
     `frozen_evidence_corpus`.
   - Return the corpus `known_facts` from patched RAG supervisor.
   - Assert the projected `rag_result` matches the existing expected projection
     for answer, evidence categories, dispatched agents, source metadata, and
     scoped user-memory unit candidates.
6. Update `src/kazusa_ai_chatbot/rag/README.md`.
   - Add only the adapter-boundary note approved in `Change Surface`.
   - Do not change RAG supervisor, helper-agent, or Cache 2 semantics.
7. Run all verification gates.
8. Update this plan's checklist and `Execution Evidence`.
9. Update the parent ledger and registry only after every verification command
   passes.

## Progress Checklist

- [ ] Stage 1 - adapter module tests added.
  - Covers: `tests/test_rag_cognitive_episode_adapter.py`.
  - Verify: focused adapter test fails only because approved Stage 04 module or
    symbols do not exist yet.
  - Evidence: record command and expected failure in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - adapter module implemented.
  - Covers: `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`.
  - Verify: adapter module tests pass and `py_compile` passes.
  - Evidence: record public API names and test output.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - `stage_1_research` integration tests added.
  - Covers: `tests/test_persona_supervisor2_rag2_integration.py` and Stage 00
    baseline fixture updates required by Stage 02.
  - Verify: integration tests fail only until the adapter is wired.
  - Evidence: record command and failure summary.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 4 - persona RAG request builder refactored.
  - Covers: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`.
  - Verify: adapter and integration tests pass.
  - Evidence: record captured request equivalence output.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - frozen evidence and regression gates pass.
  - Covers: Stage 00 frozen corpus, Stage 00 baseline, Stage 01 contract tests,
    Stage 02 pass-through tests, Stage 03 prompt-render tests, and RAG adjacent
    tests.
  - Verify: every command in `Verification` passes.
  - Evidence: record exact command results.
  - Handoff: next agent updates lifecycle records.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - lifecycle records updated.
  - Covers: this plan, parent ledger, and registry.
  - Verify: rows show Stage 04 completed and artifact paths are named.
  - Evidence: record parent ledger and registry confirmation.
  - Handoff: Stage 04 is complete; Stage 05 remains blocked until separately
    created and approved.
  - Sign-off: `<agent/date>` after lifecycle updates are recorded.

## Verification

### Focused Adapter Tests

```powershell
venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py
```

### Integration And Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py
venv\Scripts\python -m pytest tests\test_rag_projection.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py
```

Stage 03 verification command must be copied from the completed Stage 03
execution evidence before Stage 04 execution starts. Do not invent that command
in this plan while Stage 03 does not yet exist.

### Adjacent Deterministic RAG Tests

```powershell
venv\Scripts\python -m pytest tests\test_rag_initializer_cache2.py tests\test_rag_phase3_supervisor_integration.py
```

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py
git diff --check
rg -n "cognitive_episode|RAGEpisode|build_text_chat_rag_request" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_initializer.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_dispatch.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_evaluator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py
```

The final `rg` command must return no matches. Exit code 1 from that specific
no-match grep is acceptable and must be recorded as expected.

No real LLM tests are required by Stage 04 unless Stage 03 execution evidence
requires a specific live prompt smoke gate. If any prompt payload changes, stop
and create a bugfix plan.

## Acceptance Criteria

This plan is complete when:

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py` owns the text
  `/chat` RAG request-building boundary.
- `persona_supervisor2.stage_1_research` uses
  `build_text_chat_rag_request` for non-skip RAG calls.
- The adapter's `/chat` `original_query`, `character_name`, context keys, and
  context values are equivalent to the pre-adapter inline construction.
- `call_rag_supervisor` signature and return shape remain unchanged.
- `project_known_facts` behavior and `rag_result` public keys remain unchanged.
- The Stage 00 frozen evidence corpus projects to the same answer, evidence
  categories, dispatched agents, source metadata, scoped user-memory unit
  candidates, answer availability, and unknown-slot categories.
- Non-chat trigger sources raise `RAGEpisodeAdapterError` and remain disabled.
- No prompts consume `cognitive_episode`, `RAGEpisodeRequest`, or raw percepts.
- No new live `/chat` LLM calls are added.
- Stage 00 through Stage 03 deterministic gates pass.
- Parent ledger and registry can point to this stage's adapter module, tests,
  RAG README note, and verification evidence.

## Data Migration

No database schema, collection, index, stored-document, cache persistence, or
migration work is allowed or required.

## Operational Steps

No service restart, scheduler operation, adapter operation, deployment step, or
manual runtime intervention is required for local verification. The character
must continue using the current `/chat` retrieval path, only with request
construction moved behind the adapter.

## Completion Artifact Contract

`stage_04` is not complete until `Execution Evidence` records:

- The adapter module path and public API names.
- The `persona_supervisor2.stage_1_research` integration path.
- The focused adapter test path.
- The integration test paths proving request equivalence.
- The Stage 00 frozen corpus assertion results.
- The Stage 00, Stage 01, Stage 02, and Stage 03 verification commands used as
  gates and their results.
- The no-match prompt/consolidator/RAG-internals grep result.
- Confirmation that non-chat triggers remain disabled.
- Confirmation that the parent ledger was updated so `stage_04` is complete.
- Confirmation that `development_plans/README.md` was updated so the Stage 04
  registry row is complete.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| RAG initializer or Cache 2 keys drift | Keep `original_query`, `character_name`, and context equivalent. | Captured request equality tests and adjacent RAG initializer/cache tests. |
| Prompt payloads change indirectly | Do not edit prompt files or RAG prompt modules. | Static no-match grep and Stage 03 prompt-render gate. |
| Adapter hides a legacy fallback | Require `cognitive_episode` after Stage 02 and fail if missing. | Integration tests with required episode state. |
| Non-chat sources accidentally run through text-chat RAG | Raise `RAGEpisodeAdapterError` for unsupported triggers/input sources. | Adapter negative tests. |
| Frozen evidence projection drifts | Keep `project_known_facts` unchanged. | Stage 00 frozen corpus equivalence test and `tests/test_rag_projection.py`. |

## LLM Call And Context Budget

No new LLM calls are allowed.

Before and after for live `/chat`:

| RAG area | Before | After |
|---|---:|---:|
| RAG supervisor call count | unchanged | unchanged |
| Initializer calls | unchanged | unchanged |
| Dispatcher/evaluator/finalizer calls | unchanged | unchanged |
| Helper-agent calls | unchanged | unchanged |
| RAG loop cap | unchanged | unchanged |
| Prompt text | unchanged | unchanged |
| Prompt payload keys | unchanged | unchanged |

For `/chat`, the adapter must keep prompt-facing RAG inputs within the current
budget by preserving the same `original_query` and context. No raw episode,
percept list, attachment payload, or future `episode_focus` value may be added
to prompt-facing RAG payloads in this stage.

## Execution Evidence

Draft only. No implementation has been executed from this plan.
