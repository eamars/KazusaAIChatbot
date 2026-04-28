# llm routing migration plan

## Summary

- Goal: Introduce named LLM routes so different chatbot responsibilities can move to different OpenAI-compatible providers or models without migrating every call site at once.
- Plan class: large
- Status: draft
- Overall cutover strategy: compatible
- Highest-risk areas: shared LLM factory behavior, prompt JSON contracts, tool-calling web search, cognition/dialog latency, live LLM test assumptions.
- Acceptance criteria: every production chat LLM call uses a named route; default configuration preserves current LM Studio behavior; route-specific env vars can move one route without affecting others.

## Context

The project currently centralizes nearly all chat LLM calls through `get_llm(...)` in `src/kazusa_ai_chatbot/utils.py`. `config.py` already defines primary, secondary, and preference model settings, but production call sites mostly call the primary route directly. Only the preference adapter currently uses `get_preference_llm(...)`.

The target architecture is not one model per call site. The target is a small set of named routes whose names are bonded to where each LLM is used in the code — not abstract semantic roles. This makes it grep-obvious which call sites a route controls, and it lets operators tier each route independently from the config without any code change.

Routes:

- `relevance_agent_llm`
- `vision_descriptor_llm`
- `msg_decontextualizer_llm`
- `rag_planner_llm`
- `rag_subagent_llm`
- `web_search_llm`
- `cognition_llm`
- `dialog_generator_llm`
- `dialog_evaluator_llm`
- `consolidation_llm`
- `json_repair_llm`
- embeddings remain separate through existing `EMBEDDING_*` settings.

The first implementation must preserve current behavior by default. Provider migration happens later by setting route-specific environment variables.

## Mandatory Rules

- Preserve the existing pipeline contracts. Do not redesign prompts, state schemas, graph topology, RAG slot semantics, memory schemas, or database writes as part of this routing migration.
- Do not add deterministic pre-processing or post-processing over user instructions. LLM interpretation remains prompt/schema driven.
- Do not make the initializer, dispatcher, or helper agents take on new responsibilities. Routing changes must only affect which model client each existing agent uses.
- Keep all route defaults compatible with the current primary `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- Keep embeddings separate. Do not route embeddings through chat LLM routes and do not change embedding dimensions or vector indexes.
- Preserve JSON output parsing behavior. This plan may route JSON-producing agents to another model, but must not change their output schemas.
- Do not add provider-specific SDKs. Continue using the existing OpenAI-compatible `ChatOpenAI` path unless a later plan explicitly approves a provider-specific client.
- Do not encode tier or provider choices in source code. Tiering belongs in config; the code only references named route helpers.
- Follow project Python style: small helpers, typed signatures where useful, no broad refactors, no unrelated formatting churn.
- For tests, regular pytest runs must remain offline by default. Do not require live LLM access for normal test verification.

## Must Do

- Add route-specific chat LLM configuration with environment-variable fallback to the primary model.
- Add named helper functions for all approved routes.
- Replace production chat LLM call sites so they use the route helper bonded to that call site.
- Keep `get_llm(...)` available as the low-level factory for backwards compatibility and tests.
- Update docs so operators know which environment variables control each route.
- Add tests proving route defaults and route overrides select the expected model/base URL/api_key.
- Add static verification commands that catch remaining unclassified production `get_llm(...)` calls.

## Deferred

- Do not migrate any route to a real external provider in code.
- Do not remove `SECONDARY_LLM_*` or `PREFERENCE_LLM_*` in this plan.
- Do not split `cognition_llm` into L1/L2/L3 sub-routes. Cognition latency is accepted in this iteration; the natural follow-up split lives along the existing `cognition_l1`/`l2`/`l3` file boundaries.
- Do not introduce model fallback chains, retry escalation, model health checks, cost accounting, or per-route rate limiters.
- Do not change prompt wording, retry limits, graph edges, RAG agent behavior, database schemas, or memory consolidation logic.
- Do not re-embed existing data or change `EMBEDDING_MODEL`.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Runtime route helpers | compatible | Add named helpers while keeping `get_llm(...)` as the low-level factory. |
| Existing default behavior | compatible | Every route falls back to the primary `LLM_*` values unless route-specific env vars are set. |
| Production call sites | migration | Replace direct production `get_llm(...)` usage with the route helper bonded to that call site. |
| `get_preference_llm(...)` | compatibility | Keep it as an alias or wrapper for preference-specific behavior until callers are migrated. |
| Embeddings | compatible | Leave existing `EMBEDDING_*` behavior unchanged. |
| Tests | migration | Add route tests and update mocks only where route helper names change imports. |

## Cutover Policy Enforcement

- The implementation agent must not choose a more aggressive strategy than this plan.
- If a call site is listed under a route, migrate it to that route; do not leave it on direct `get_llm(...)`.
- If a route-specific env var is absent, behavior must be identical to the current primary model configuration.
- Any provider-specific behavior, fallback chain, or retry policy requires a separate approved plan.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the route names and contracts in this plan.
- The agent must not introduce alternate route names, additional route groups, compatibility layers, or prompt rewrites.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or graph refactors.
- If a required call site cannot be migrated cleanly, the agent must stop and report the blocker instead of inventing a substitute.

## Target State

All production chat LLM clients are created through bonded route helpers:

```python
get_relevance_agent_llm(...)
get_vision_descriptor_llm(...)
get_msg_decontextualizer_llm(...)
get_rag_planner_llm(...)
get_rag_subagent_llm(...)
get_web_search_llm(...)
get_cognition_llm(...)
get_dialog_generator_llm(...)
get_dialog_evaluator_llm(...)
get_consolidation_llm(...)
get_json_repair_llm(...)
```

Each helper accepts the same practical tuning arguments currently used by `get_llm(...)`, including `temperature`, `top_p`, and passthrough keyword arguments such as `presence_penalty`.

Each helper uses this fallback pattern:

```text
<ROUTE>_LLM_BASE_URL -> LLM_BASE_URL
<ROUTE>_LLM_API_KEY  -> LLM_API_KEY
<ROUTE>_LLM_MODEL    -> LLM_MODEL
```

For example:

```text
COGNITION_LLM_BASE_URL
COGNITION_LLM_API_KEY
COGNITION_LLM_MODEL

DIALOG_GENERATOR_LLM_BASE_URL
DIALOG_GENERATOR_LLM_API_KEY
DIALOG_GENERATOR_LLM_MODEL

RAG_PLANNER_LLM_BASE_URL
RAG_PLANNER_LLM_API_KEY
RAG_PLANNER_LLM_MODEL
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Route count | Use 11 chat routes plus existing embeddings | Each route is bonded to a specific call site or tight cluster of sites; tiering is a config decision, not a code one. |
| Naming style | Bonded to where the LLM is used (e.g. `dialog_generator_llm`, `rag_planner_llm`) | Grep maps each route to a small, deterministic set of files; reviewers don't have to translate abstract roles. |
| Dialog split | `dialog_generator_llm` and `dialog_evaluator_llm` are separate routes | Generator drives user-visible quality; evaluator is a critic that can sit on a cheaper tier without hurting output. |
| Retrieval split | `rag_planner_llm`, `rag_subagent_llm`, `web_search_llm` | Planner needs better reasoning; subagents are cheap JSON workers; web search needs provider-side tool-calling, which is a deployment constraint not a quality one. |
| Cognition route | Single `cognition_llm` for now | All cognition layers go to one route in v1. Splitting along `l1`/`l2`/`l3` is deferred to a later plan once latency is measured. |
| Vision separation | `vision_descriptor_llm` distinct from text routes | Image input typically requires a multimodal endpoint independent of the chat endpoint. |
| Compatibility | Route env vars fall back to primary | Operators can migrate one route at a time and default behavior stays stable. |
| Existing `PREFERENCE_LLM_*` | Preserve as compatibility wrapper | It exists today and may be relied on by local env files. |
| Provider abstraction | Continue OpenAI-compatible `ChatOpenAI` | Current app and docs are built around OpenAI-compatible endpoints. |
| No fallback chain | Routes fail loudly when their endpoint is unreachable | Avoids hidden cross-tier spend; failover is a separate concern that requires its own plan. |

## Route Assignments

### `relevance_agent_llm`

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - `_relevance_agent_llm`

### `vision_descriptor_llm`

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - `_vision_descriptor_llm`

### `msg_decontextualizer_llm`

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - `_msg_decontexualizer_llm`

### `rag_planner_llm`

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - `_initializer_llm`
  - `_dispatcher_llm`

### `rag_subagent_llm`

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - `_evaluator_summarizer_llm`
  - `_finalizer_llm`
- `src/kazusa_ai_chatbot/rag/conversation_aggregate_agent.py`
  - `_extractor_llm`
- `src/kazusa_ai_chatbot/rag/conversation_filter_agent.py`
  - `_generator_llm`
  - `_judge_llm`
- `src/kazusa_ai_chatbot/rag/conversation_keyword_agent.py`
  - `_generator_llm`
  - `_judge_llm`
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`
  - `_generator_llm`
  - `_judge_llm`
- `src/kazusa_ai_chatbot/rag/persistent_memory_keyword_agent.py`
  - `_generator_llm`
  - `_judge_llm`
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`
  - `_generator_llm`
  - `_judge_llm`
- `src/kazusa_ai_chatbot/rag/relationship_agent.py`
  - `_extractor_llm`
- `src/kazusa_ai_chatbot/rag/user_list_agent.py`
  - `_extractor_llm`
- `src/kazusa_ai_chatbot/rag/user_lookup_agent.py`
  - `_extractor_llm`
  - `_picker_llm`

### `web_search_llm`

- `src/kazusa_ai_chatbot/rag/web_search_agent.py`
  - `_generator_llm` (must preserve `bind_tools(_ALL_TOOLS)`)
  - `_evaluator_llm`
  - `_finalizer_llm`

### `cognition_llm`

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
  - `_subconscious_llm`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - `_conscious_llm`
  - `_boundary_core_llm`
  - `_judgement_core_llm`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - `_contextual_agent_llm`
  - `_style_agent_llm`
  - `_content_anchor_agent_llm`
  - `_preference_adapter_llm`
  - `_visual_agent_llm`

### `dialog_generator_llm`

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - `_dialog_generator_llm`

### `dialog_evaluator_llm`

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - `_dialog_evaluator_llm`

### `consolidation_llm`

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  - `_facts_harvester_llm`
  - `_fact_harvester_evaluator_llm`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - `_global_state_updater_llm`
  - `_relationship_recorder_llm`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`
  - `_task_dispatcher_llm`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - `_user_image_session_summary_llm`
  - `_user_image_compress_llm`
  - `_character_image_session_summary_llm`
  - `_character_image_compress_llm`

### `json_repair_llm`

- `src/kazusa_ai_chatbot/utils.py`
  - `_parse_json_with_llm`

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Add route config constants or a route config helper for the approved route env vars.
- `src/kazusa_ai_chatbot/utils.py`
  - Keep `get_llm(...)`.
  - Add named route helper functions.
  - Route `_parse_json_with_llm` through `get_json_repair_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Use `get_relevance_agent_llm(...)` and `get_vision_descriptor_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Use `get_msg_decontextualizer_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Use `get_rag_planner_llm(...)` for `_initializer_llm` and `_dispatcher_llm`.
  - Use `get_rag_subagent_llm(...)` for `_evaluator_summarizer_llm` and `_finalizer_llm`.
- `src/kazusa_ai_chatbot/rag/*_agent.py` (excluding `web_search_agent.py`)
  - Use `get_rag_subagent_llm(...)` for all chat LLM clients.
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`
  - Use `get_web_search_llm(...)` for all chat LLM clients. Preserve `bind_tools(_ALL_TOOLS)` at the call site.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Use `get_cognition_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Use `get_dialog_generator_llm(...)` for the generator and `get_dialog_evaluator_llm(...)` for the evaluator.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_*.py`
  - Use `get_consolidation_llm(...)`.
- `docs/HOWTO.md`
  - Document route env vars and default fallback behavior.
- `tests/test_config.py` and/or `tests/test_utils.py`
  - Add route helper configuration tests.

### Keep

- `src/kazusa_ai_chatbot/db/_client.py`
  - Existing embedding client remains controlled by `EMBEDDING_*`.
- `pyproject.toml`
  - No dependency changes.
- Prompt text files and embedded prompts
  - No prompt migration in this plan.

## Implementation Order

1. Add route configuration support.
   - Prefer a small helper that resolves route env vars with fallback to primary values.
   - Ensure env var names exactly match the route names in this plan.

2. Add route helper functions in `utils.py`.
   - Each helper must call `get_llm(...)` with route-selected `model`, `base_url`, and `api_key`.
   - Preserve passthrough `**kwargs`.
   - Keep `get_preference_llm(...)` as a compatibility wrapper. It may call `get_cognition_llm(...)` or continue using `PREFERENCE_LLM_*`, but `_preference_adapter_llm` must be migrated to `get_cognition_llm(...)`.

3. Migrate low-risk single-call files.
   - `json_repair_llm`
   - `relevance_agent_llm`
   - `vision_descriptor_llm`
   - `msg_decontextualizer_llm`

4. Migrate cognition.
   - All L1/L2/L3 call sites use `get_cognition_llm(...)`.
   - Replace call-site imports and helper calls only. Do not edit prompt bodies.

5. Migrate dialog.
   - Generator uses `get_dialog_generator_llm(...)`; evaluator uses `get_dialog_evaluator_llm(...)`.

6. Migrate retrieval.
   - RAG supervisor planner roles (`_initializer_llm`, `_dispatcher_llm`) use `get_rag_planner_llm(...)`.
   - RAG supervisor evaluator/finalizer and all `rag/*_agent.py` (excluding web search) use `get_rag_subagent_llm(...)`.
   - Web search uses `get_web_search_llm(...)`. Preserve `.bind_tools(_ALL_TOOLS)`.

7. Migrate consolidation.
   - Replace helper calls only.
   - Do not change background execution or database write behavior.

8. Update docs.
   - Add a route configuration section to `docs/HOWTO.md`.
   - State that missing route vars fall back to primary `LLM_*`.
   - State that embeddings are still controlled by `EMBEDDING_*`.

9. Add or update tests.
   - Test default route fallback for every route.
   - Test route-specific override for at least `cognition_llm`, `dialog_generator_llm`, `dialog_evaluator_llm`, `rag_planner_llm`, `rag_subagent_llm`, and `web_search_llm` independently.
   - Test helper preserves per-call tuning parameters such as `temperature`, `top_p`, and arbitrary kwargs.

10. Run static greps and tests.

## Verification

### Static Greps

- `rg "get_llm\\(" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - `utils.py` low-level factory and route helper definitions.
    - Comments or docs if any.
  - Production agent modules should not directly instantiate chat clients through `get_llm(...)`.
- `rg "get_preference_llm\\(" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - compatibility wrapper definition only.
- `rg "SECONDARY_LLM|PREFERENCE_LLM" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - config compatibility definitions and wrapper implementation.
- `rg "RELEVANCE_AGENT_LLM|VISION_DESCRIPTOR_LLM|MSG_DECONTEXTUALIZER_LLM|RAG_PLANNER_LLM|RAG_SUBAGENT_LLM|WEB_SEARCH_LLM|COGNITION_LLM|DIALOG_GENERATOR_LLM|DIALOG_EVALUATOR_LLM|CONSOLIDATION_LLM|JSON_REPAIR_LLM" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - `config.py` route resolution.
    - `utils.py` helper definitions.

### Tests

- `pytest tests/test_config.py tests/test_utils.py -q`
- `pytest -m "not live_db and not live_llm" -q`

### Smoke

- Service imports without missing route helper errors:
  - `python -c "import kazusa_ai_chatbot.service"`
- Route helpers instantiate without live network calls:
  - Use mocked `ChatOpenAI` in tests; do not require an LLM endpoint.

## Acceptance Criteria

This plan is complete when:

- All listed production chat LLM call sites use the approved bonded route helpers.
- Default env configuration still routes all chat calls to the current primary `LLM_*` endpoint.
- Setting `COGNITION_LLM_MODEL` changes only cognition helper construction in tests.
- Setting `DIALOG_GENERATOR_LLM_MODEL` changes only the dialog generator helper, leaving `DIALOG_EVALUATOR_LLM_MODEL` independent.
- Setting `RAG_PLANNER_LLM_MODEL` changes only the planner helper, leaving `RAG_SUBAGENT_LLM_MODEL` independent.
- Web search is controlled by `WEB_SEARCH_LLM_*` and continues to bind tools.
- Embedding behavior remains controlled only by `EMBEDDING_*`.
- Docs describe every route and fallback behavior.
- Offline tests and static grep gates pass.

## Rollback / Recovery

- Code rollback path: revert the routing-helper commit and call-site migration commit.
- Data rollback path: none required; this plan does not change databases or embeddings.
- Irreversible operations: none.
- Required backup: none beyond normal source control.
- Recovery verification: run `pytest -m "not live_db and not live_llm" -q` after rollback.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Route helper accidentally changes default model | Every route falls back to primary `LLM_*`; add tests for defaults. | Route helper tests. |
| Web search tool binding breaks | Keep `bind_tools(_ALL_TOOLS)` at the call site; chosen `WEB_SEARCH_LLM_*` provider must support tool calling. | Existing web search tests plus import smoke. |
| Live LLM tests assume primary env vars only | Keep primary defaults and document route envs. | Offline tests pass; live tests can be run route-by-route later. |
| Prompt JSON contracts degrade on a new provider | No provider migration in this plan; only routing knobs. | Default-compatible tests and later live route characterization. |
| Cognition latency multiplies on a slow premium endpoint | Accepted in this plan; deferred to a future cognition-split plan. | Measure end-to-end dialog latency after operators point `COGNITION_LLM_*` at the premium endpoint. |
| Too many knobs confuse operations | Route names are bonded to call sites; HOWTO route table maps each name to its files. | HOWTO route table. |

## Operational Steps

After implementation, operators can migrate routes gradually:

1. Set only `COGNITION_LLM_BASE_URL`, `COGNITION_LLM_API_KEY`, and `COGNITION_LLM_MODEL` to test a larger cognition model.
2. Set `DIALOG_GENERATOR_LLM_*` to point user-visible reply generation at the premium endpoint.
3. Leave `DIALOG_EVALUATOR_LLM_*` on a cheaper tier; the evaluator is a critic and does not need parity with the generator.
4. Point `RAG_PLANNER_LLM_*` at a mid-tier model when planning quality matters; keep `RAG_SUBAGENT_LLM_*` on the cheap tier for high-volume JSON workers.
5. Confirm tool-calling support before setting `WEB_SEARCH_LLM_*` to a non-primary provider.
6. Move `CONSOLIDATION_LLM_*` only after immediate response quality is stable.
7. Do not change `EMBEDDING_*` without a separate embedding migration and reindex plan.

## Execution Evidence

- Static grep results:
- Test results:
- Service import smoke:
- Documentation updated:
