# llm routing migration plan

## Summary

- Goal: Introduce named LLM routes so different chatbot responsibilities can move to different OpenAI-compatible providers or models without migrating every call site at once.
- Plan class: large
- Status: draft
- Overall cutover strategy: migration
- Highest-risk areas: shared LLM factory behavior, prompt JSON contracts, tool-calling web search, cognition/dialog latency, live LLM test assumptions.
- Acceptance criteria: every production chat LLM call uses route-specific config constants with the existing `get_llm(...)` factory; `.env` explicitly defines every route; generic `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` are removed; the app crashes during config load if any required route variable is missing.

## Context

The project currently centralizes nearly all chat LLM calls through `get_llm(...)` in `src/kazusa_ai_chatbot/utils.py`. `config.py` defines a generic primary model setting today, but that generic route is intentionally removed by this refactor.

The target architecture is not one model per call site. The target is a small set of named routes whose names are bonded to where each LLM is used in the code — not abstract semantic roles. This makes it grep-obvious which call sites a route controls, and it lets operators tier each route independently from config without any code change.

This plan must stay simple: do not add new helper functions and do not add config shims. Add route config constants only, require every route env var explicitly, then pass those constants into the existing `get_llm(...)` factory at each call site.

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

The first implementation must preserve current behavior by writing explicit route values into `.env`. Provider migration happens later by editing those route-specific environment variables. Source code must not silently fall back from a route to generic `LLM_*` values, and generic chat `LLM_*` settings are removed/invalidated after the refactor.

## Mandatory Rules

- Preserve the existing pipeline contracts. Do not redesign prompts, state schemas, graph topology, RAG slot semantics, memory schemas, or database writes as part of this routing migration.
- Do not add deterministic pre-processing or post-processing over user instructions. LLM interpretation remains prompt/schema driven.
- Do not make the initializer, dispatcher, or helper agents take on new responsibilities. Routing changes must only affect which model client each existing agent uses.
- Do not implement route defaults or fallback shims in `config.py`. Route env vars are required.
- Update `.env` with every route variable populated using the current LM Studio values.
- Keep embeddings separate. Do not route embeddings through chat LLM routes and do not change embedding dimensions or vector indexes.
- Add explicit `EMBEDDING_API_KEY`; do not reuse a removed chat `LLM_API_KEY` for embeddings.
- Preserve JSON output parsing behavior. This plan may route JSON-producing agents to another model, but must not change their output schemas.
- Do not add provider-specific SDKs. Continue using the existing OpenAI-compatible `ChatOpenAI` path unless a later plan explicitly approves a provider-specific client.
- Do not encode tier or provider choices in source code. Tiering belongs in config; the code only references route config constants.
- Do not introduce any new LLM helper functions. Use the existing `get_llm(...)` factory directly with route-specific `model`, `base_url`, and `api_key` constants.
- Do not introduce config helper functions, fallback helper functions, shim dictionaries, compatibility adapters, or route resolution wrappers.
- Follow project Python style: small changes, typed signatures where useful, no broad refactors, no unrelated formatting churn.
- For tests, regular pytest runs must remain offline by default. Do not require live LLM access for normal test verification.

## Must Do

- Add route-specific chat LLM configuration as required environment variables with no fallback to any generic chat model.
- Update `.env` so every route variable is present and populated with the current LM Studio values.
- Remove generic `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` config constants and `.env` entries.
- Make `get_llm(...)` require explicit `model`, `base_url`, and `api_key`; no defaults.
- Add only route-specific config constants. Do not add route helper functions.
- Replace production chat LLM call sites so they call existing `get_llm(...)` with the route constants bonded to that call site.
- Keep `get_llm(...)` available as the explicit low-level factory, but remove its default model/base URL/API key.
- Update docs so operators know which environment variables control each route.
- Add tests proving route defaults and route overrides select the expected model/base URL/api_key.
- Add static verification commands that catch remaining unclassified production `get_llm(...)` calls.

## Deferred

- Do not migrate any route to a real external provider in code.
- Remove `SECONDARY_LLM_*`, `PREFERENCE_LLM_*`, `get_secondary_llm(...)`, and `get_preference_llm(...)` after the preference adapter is routed through `COGNITION_LLM_*`.
- Do not split `cognition_llm` into L1/L2/L3 sub-routes. Cognition latency is accepted in this iteration; the natural follow-up split lives along the existing `cognition_l1`/`l2`/`l3` file boundaries.
- Do not introduce model fallback chains, retry escalation, model health checks, config fallback shims, cost accounting, or per-route rate limiters.
- Do not change prompt wording, retry limits, graph edges, RAG agent behavior, database schemas, or memory consolidation logic.
- Do not re-embed existing data or change `EMBEDDING_MODEL`.

## Cutover Policy

Overall strategy: migration

| Area | Policy | Instruction |
|---|---|---|
| Runtime route constants | migration | Add named config constants while keeping `get_llm(...)` as the only LLM factory. |
| Generic chat `LLM_*` | bigbang | Remove generic chat `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` from config, `.env`, docs, and compose. |
| Required route env vars | bigbang | Each route constant must read its env var directly and crash if missing. No fallback to generic `LLM_*`. |
| `.env` values | migration | Populate every route env var in `.env` with the current LM Studio values before or with the code migration. |
| Production call sites | migration | Keep direct `get_llm(...)` usage, but pass the route constants bonded to that call site. |
| Existing helper functions | bigbang | Remove secondary/preference LLM helpers once their callers are migrated. Do not introduce new helpers. |
| Embeddings | compatible | Leave existing `EMBEDDING_*` behavior unchanged. |
| Tests | migration | Add route config tests and update mocks only where config imports change. |

## Cutover Policy Enforcement

- The implementation agent must not choose a more aggressive strategy than this plan.
- If a call site is listed under a route, migrate it to that route by passing the route constants into `get_llm(...)`.
- If a route-specific env var is absent, the application must fail during config load. Do not catch, repair, or silently substitute missing route config.
- Generic chat `LLM_*` constants must not exist after implementation.
- New helper functions are forbidden in this plan.
- Config shims and fallback wrappers are forbidden in this plan.
- Any provider-specific behavior, fallback chain, or retry policy requires a separate approved plan.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve the route names and contracts in this plan.
- The agent must not introduce alternate route names, additional route groups, helper functions, config shims, compatibility layers, fallback wrappers, or prompt rewrites.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or graph refactors.
- If a required call site cannot be migrated cleanly, the agent must stop and report the blocker instead of inventing a substitute.

## Target State

All production chat LLM clients are created through the existing `get_llm(...)` factory using bonded route config constants:

```python
_relevance_agent_llm = get_llm(
    temperature=0,
    top_p=1.0,
    model=RELEVANCE_AGENT_LLM_MODEL,
    base_url=RELEVANCE_AGENT_LLM_BASE_URL,
    api_key=RELEVANCE_AGENT_LLM_API_KEY,
)
```

Each route config constant is required. Use direct environment reads for all route variables:

```text
<ROUTE>_LLM_BASE_URL
<ROUTE>_LLM_API_KEY
<ROUTE>_LLM_MODEL
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
| Implementation style | Add config constants only; no new helper functions | Keeps the migration flat and avoids a second abstraction layer over `get_llm(...)`. |
| Dialog split | `dialog_generator_llm` and `dialog_evaluator_llm` are separate routes | Generator drives user-visible quality; evaluator is a critic that can sit on a cheaper tier without hurting output. |
| Retrieval split | `rag_planner_llm`, `rag_subagent_llm`, `web_search_llm` | Planner needs better reasoning; subagents are cheap JSON workers; web search needs provider-side tool-calling, which is a deployment constraint not a quality one. |
| Cognition route | Single `cognition_llm` for now | All cognition layers go to one route in v1. Splitting along `l1`/`l2`/`l3` is deferred to a later plan once latency is measured. |
| Vision separation | `vision_descriptor_llm` distinct from text routes | Image input typically requires a multimodal endpoint independent of the chat endpoint. |
| Required config | Route env vars are mandatory | Missing model links must crash early instead of silently using the wrong provider. |
| Existing secondary/preference routes | Remove legacy compatibility routes | Route-specific constants replace them; keeping hidden legacy routes would preserve the generic-primary model shape. |
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
  - Add route config constants for the approved route env vars.
  - Do not add a route config helper function.
  - Do not use `os.getenv(..., fallback)` for route constants.
  - Use direct required env reads for route constants so missing values crash during config load.
- `src/kazusa_ai_chatbot/utils.py`
  - Keep `get_llm(...)`.
  - Do not add named route helper functions.
  - Route `_parse_json_with_llm` by passing `JSON_REPAIR_LLM_*` constants into existing `get_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Import `RELEVANCE_AGENT_LLM_*` and `VISION_DESCRIPTOR_LLM_*` constants.
  - Pass those constants into existing `get_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Import `MSG_DECONTEXTUALIZER_LLM_*` constants and pass them into existing `get_llm(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - Use `RAG_PLANNER_LLM_*` constants for `_initializer_llm` and `_dispatcher_llm`.
  - Use `RAG_SUBAGENT_LLM_*` constants for `_evaluator_summarizer_llm` and `_finalizer_llm`.
- `src/kazusa_ai_chatbot/rag/*_agent.py` (excluding `web_search_agent.py`)
  - Use `RAG_SUBAGENT_LLM_*` constants for all chat LLM clients.
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`
  - Use `WEB_SEARCH_LLM_*` constants for all chat LLM clients. Preserve `bind_tools(_ALL_TOOLS)` at the call site.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Use `COGNITION_LLM_*` constants for all chat LLM clients.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Use `DIALOG_GENERATOR_LLM_*` constants for the generator and `DIALOG_EVALUATOR_LLM_*` constants for the evaluator.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_*.py`
  - Use `CONSOLIDATION_LLM_*` constants for all chat LLM clients.
- `docs/HOWTO.md`
  - Document route env vars and the required-explicit-config behavior.
- `.env`
  - Add all route-specific `*_LLM_BASE_URL`, `*_LLM_API_KEY`, and `*_LLM_MODEL` entries with values copied from the current LM Studio settings.
- `tests/test_config.py` and/or `tests/test_utils.py`
  - Add route config tests.

### Keep

- `src/kazusa_ai_chatbot/db/_client.py`
  - Existing embedding client remains controlled by `EMBEDDING_*`.
- `pyproject.toml`
  - No dependency changes.
- Prompt text files and embedded prompts
  - No prompt migration in this plan.

## Implementation Order

1. Add route configuration support.
   - Add constants directly in `config.py`.
   - Each route constant must use direct required environment access.
   - Do not use generic chat `LLM_*` as fallback for any route constant.
   - Missing route env vars must raise during import/config load.
   - Ensure env var names exactly match the route names in this plan.

2. Update `utils.py`.
   - Remove default model, base URL, and API key values from `get_llm(...)`.
   - Do not add new helper functions.
   - Route `_parse_json_with_llm` by passing `JSON_REPAIR_LLM_MODEL`, `JSON_REPAIR_LLM_BASE_URL`, and `JSON_REPAIR_LLM_API_KEY` into `get_llm(...)`.

3. Migrate low-risk single-call files.
   - `json_repair_llm`
   - `relevance_agent_llm`
   - `vision_descriptor_llm`
   - `msg_decontextualizer_llm`

4. Migrate cognition.
   - All L1/L2/L3 call sites use `get_llm(...)` with `COGNITION_LLM_*` constants.
   - Replace call-site imports and `get_llm(...)` arguments only. Do not edit prompt bodies.

5. Migrate dialog.
   - Generator uses `get_llm(...)` with `DIALOG_GENERATOR_LLM_*`; evaluator uses `get_llm(...)` with `DIALOG_EVALUATOR_LLM_*`.

6. Migrate retrieval.
   - RAG supervisor planner roles (`_initializer_llm`, `_dispatcher_llm`) use `get_llm(...)` with `RAG_PLANNER_LLM_*`.
   - RAG supervisor evaluator/finalizer and all `rag/*_agent.py` (excluding web search) use `get_llm(...)` with `RAG_SUBAGENT_LLM_*`.
   - Web search uses `get_llm(...)` with `WEB_SEARCH_LLM_*`. Preserve `.bind_tools(_ALL_TOOLS)`.

7. Migrate consolidation.
   - Replace `get_llm(...)` arguments only.
   - Do not change background execution or database write behavior.

8. Update docs.
   - Add a route configuration section to `docs/HOWTO.md`.
   - State that missing route vars crash config load.
   - State that embeddings are still controlled by `EMBEDDING_*`.

9. Add or update tests.
   - Test every route constant reads its explicit env var.
   - Test missing required route env vars fail during config loading.
   - Test route-specific override for at least `cognition_llm`, `dialog_generator_llm`, `dialog_evaluator_llm`, `rag_planner_llm`, `rag_subagent_llm`, and `web_search_llm` independently.
   - Test call-site construction or config values without introducing route helper functions.

10. Run static greps and tests.

## Verification

### Static Greps

- `rg "get_llm\\(" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - `utils.py` low-level explicit factory.
    - Production agent modules using `get_llm(...)` with route config constants.
    - Comments or docs if any.
- `rg "SECONDARY_LLM|PREFERENCE_LLM" src/kazusa_ai_chatbot -S`
  - Must return no matches.
- `rg "get_secondary_llm|get_preference_llm" src/kazusa_ai_chatbot -S`
  - Must return no matches.
- `rg "RELEVANCE_AGENT_LLM|VISION_DESCRIPTOR_LLM|MSG_DECONTEXTUALIZER_LLM|RAG_PLANNER_LLM|RAG_SUBAGENT_LLM|WEB_SEARCH_LLM|COGNITION_LLM|DIALOG_GENERATOR_LLM|DIALOG_EVALUATOR_LLM|CONSOLIDATION_LLM|JSON_REPAIR_LLM" src/kazusa_ai_chatbot -S`
  - Allowed matches:
    - `config.py` route constants.
    - Production call-site imports and `get_llm(...)` arguments.
- `rg "os\\.getenv\\(\"(RELEVANCE_AGENT_LLM|VISION_DESCRIPTOR_LLM|MSG_DECONTEXTUALIZER_LLM|RAG_PLANNER_LLM|RAG_SUBAGENT_LLM|WEB_SEARCH_LLM|COGNITION_LLM|DIALOG_GENERATOR_LLM|DIALOG_EVALUATOR_LLM|CONSOLIDATION_LLM|JSON_REPAIR_LLM)" src/kazusa_ai_chatbot/config.py -S`
  - Must return no matches. Route config must not use fallback-style `os.getenv`.

### Tests

- `pytest tests/test_config.py tests/test_utils.py -q`
- `pytest -m "not live_db and not live_llm" -q`

### Smoke

- Service imports without missing route config errors:
  - `python -c "import kazusa_ai_chatbot.service"`
- Route config tests run without live network calls:
  - Use mocked `ChatOpenAI` in tests; do not require an LLM endpoint.

## Acceptance Criteria

This plan is complete when:

- All listed production chat LLM call sites use existing `get_llm(...)` with the approved bonded route config constants.
- `.env` contains explicit values for every route-specific chat LLM variable.
- Missing route-specific env vars crash during config load.
- With the updated `.env`, all routes point to the current LM Studio endpoint until an operator edits them.
- Setting `COGNITION_LLM_MODEL` changes only cognition LLM construction in tests.
- Setting `DIALOG_GENERATOR_LLM_MODEL` changes only the dialog generator LLM construction, leaving `DIALOG_EVALUATOR_LLM_MODEL` independent.
- Setting `RAG_PLANNER_LLM_MODEL` changes only planner LLM construction, leaving `RAG_SUBAGENT_LLM_MODEL` independent.
- Web search is controlled by `WEB_SEARCH_LLM_*` and continues to bind tools.
- Embedding behavior remains controlled only by `EMBEDDING_*`.
- Docs describe every route and the required explicit config behavior.
- Offline tests and static grep gates pass.

## Rollback / Recovery

- Code rollback path: revert the route-config commit and call-site migration commit.
- Data rollback path: none required; this plan does not change databases or embeddings.
- Irreversible operations: none.
- Required backup: none beyond normal source control.
- Recovery verification: run `pytest -m "not live_db and not live_llm" -q` after rollback.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Missing route config crashes deploy | This is intended behavior; update `.env` with all required route values before code rollout. | Config-load failure test and `.env` review. |
| Web search tool binding breaks | Keep `bind_tools(_ALL_TOOLS)` at the call site; chosen `WEB_SEARCH_LLM_*` provider must support tool calling. | Existing web search tests plus import smoke. |
| Live LLM tests assume generic chat env vars | Update live tests to use the relevant route base URL. | Offline tests pass; live tests can be run route-by-route later. |
| Prompt JSON contracts degrade on a new provider | No provider migration in this plan; only routing knobs. | Explicit route config tests and later live route characterization. |
| Cognition latency multiplies on a slow premium endpoint | Accepted in this plan; deferred to a future cognition-split plan. | Measure end-to-end dialog latency after operators point `COGNITION_LLM_*` at the premium endpoint. |
| Too many knobs confuse operations | Route names are bonded to call sites; HOWTO route table maps each name to its files. | HOWTO route table. |

## Operational Steps

After implementation, operators can migrate routes gradually:

1. Set only `COGNITION_LLM_BASE_URL`, `COGNITION_LLM_API_KEY`, and `COGNITION_LLM_MODEL` to test a larger cognition model.
2. Set `DIALOG_GENERATOR_LLM_*` to point user-visible reply generation at the premium endpoint.
3. Leave `DIALOG_EVALUATOR_LLM_*` on a cheaper tier; the evaluator is a critic and does not need parity with the generator.
4. Point `RAG_PLANNER_LLM_*` at a mid-tier model when planning quality matters; keep `RAG_SUBAGENT_LLM_*` on the cheap tier for high-volume JSON workers.
5. Confirm tool-calling support before setting `WEB_SEARCH_LLM_*` to a different provider.
6. Move `CONSOLIDATION_LLM_*` only after immediate response quality is stable.
7. Do not change `EMBEDDING_*` without a separate embedding migration and reindex plan.

## Execution Evidence

- Static grep results:
- Test results:
- Service import smoke:
- Documentation updated:
