# service module separation stage1 plan

## Summary

- Goal: Split the oversized brain service implementation into focused internal modules without changing current functional behavior.
- Plan class: medium
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`.
- Overall cutover strategy: compatible.
- Highest-risk areas: preserving `kazusa_ai_chatbot.service:app`, private test monkeypatch surfaces, queue ordering, graph wiring, lifespan startup/shutdown, and post-turn write timing.
- Acceptance criteria: existing service entrypoints, request/response contracts, queue semantics, graph behavior, scheduler/dispatcher setup, cache hydration, and background work timing remain observably unchanged while implementation code moves behind a thinner compatibility module.

## Context

`src/kazusa_ai_chatbot/service.py` currently acts as the FastAPI app, API model contract, graph builder, lifespan manager, queue worker, identity/envelope hydrator, persistence coordinator, post-turn writer, runtime adapter registry, scheduler/dispatcher bootstrap, RAG cache hydration hook, and health endpoint.

This makes the service hard to inspect and expensive to import. A plain import of `kazusa_ai_chatbot.service` currently pulls graph, RAG, consolidator, and reflection modules before lifespan starts. The file is also the current ASGI target used by Docker, docs, and tests.

The first implementation step must be code separation only. It must not fix packaging, entrypoint, concurrency, latency, or runtime behavior. Broken or awkward entrypoints discovered during analysis are intentionally deferred.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before touching graph construction, LLM pipeline wiring, RAG startup, cognition/dialog/consolidation imports, or background LLM behavior.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not change current functional behavior in Stage 1.
- Keep `kazusa_ai_chatbot.service:app` working.
- Keep `from kazusa_ai_chatbot import service as service_module` working.
- Preserve all currently tested service-level symbols through `service.py`, including private helpers that tests monkeypatch or call directly.
- Do not convert `kazusa_ai_chatbot.service` into a package in Stage 1. Keep the file `src/kazusa_ai_chatbot/service.py` as the compatibility facade.
- Do not change `/chat`, `/health`, `/event`, `/runtime/adapters/register`, or `/runtime/adapters/heartbeat` schemas or response behavior.
- Do not change queue ordering, single-worker behavior, drop/collapse policy, response future behavior, or post-turn gating.
- Do not change graph nodes, prompts, LLM call count, RAG capability routing, cognition, dialog, consolidation semantics, scheduler behavior, dispatcher behavior, DB schemas, cache policy, or adapter API shape.
- Do not fix `pyproject.toml` console scripts, Dockerfile dependency setup, docs, or deployment commands in Stage 1.
- Do not introduce new environment variables, feature flags, runtime config, background workers, concurrency lanes, compatibility fallbacks, or alternate ASGI targets.
- Use dependency injection or thin wrappers when extracting helpers so monkeypatches applied to `service_module` still affect behavior.
- If a helper cannot be moved without changing behavior or private test surfaces, leave that helper in `service.py` and document the deferral.

## Must Do

- Create a focused internal service implementation package under `src/kazusa_ai_chatbot/brain_service/`.
- Move exactly the service implementation groups named in `Interface Contract` and `Change Surface` into focused modules while keeping `service.py` as the public compatibility facade.
- Preserve service-level imports and wrapper names for all currently referenced service symbols.
- Add or update tests only to prove compatibility and module boundaries, not to bless behavior changes.
- Run the focused service tests and static greps listed in `Verification`.
- Record execution evidence before marking any checklist stage complete.

## Deferred

- Do not fix `kazusa-ai-chatbot = "kazusa_ai_chatbot.main:main"`.
- Do not change `kazusa-brain = "kazusa_ai_chatbot.service:app"`.
- Do not move the ASGI target to `kazusa_ai_chatbot.brain_service.asgi:app`.
- Do not convert `service.py` into `service/__init__.py`.
- Do not optimize import time beyond what naturally falls out of extraction.
- Do not add multi-worker brain execution or use `BRAIN_EXECUTOR_COUNT`.
- Do not decouple post-turn consolidation from the current worker gate.
- Do not redesign runtime adapter registration, scheduler delivery, or dispatcher tool contracts.
- Do not change adapter modules or message-envelope contracts.
- Do not update deployment docs except in a later plan dedicated to entrypoints.

## Cutover Policy

| Area | Policy | Stage 1 instruction |
|---|---|---|
| Python import surface | compatible | `kazusa_ai_chatbot.service` remains the import and ASGI surface. |
| HTTP API | compatible | Endpoint paths, models, validation, and response shapes stay unchanged. |
| Queue/runtime behavior | compatible | Same single in-process queue worker and post-turn gate. |
| Graph/LLM pipeline | compatible | Same nodes, edges, prompts, model calls, and state keys. |
| Scheduler/dispatcher | compatible | Same startup wiring, pending index rebuild, and adapter registration behavior. |
| Database | compatible | No schema, collection, index, write-shape, or migration changes. |
| Packaging/deployment | compatible | No console-script, Docker, or command changes in this stage. |
| Tests | compatible | Existing service tests must continue to pass; import-path-only updates are allowed when necessary. |

## Agent Autonomy Boundaries

- The agent may choose local helper names inside `brain_service` modules only when the public `service.py` compatibility surface remains unchanged.
- The agent must not introduce new architecture, alternate migration strategies, behavior flags, fallback paths, or extra features.
- The agent must treat edits outside `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/brain_service/`, and service-focused tests as out of scope unless this plan explicitly names them.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- The agent must search for existing equivalent behavior before extracting helpers. Move existing logic; do not reimplement it from memory.
- If a moved function needs mutable service globals, either keep a wrapper in `service.py` or pass dependencies explicitly from `service.py`.
- If the plan and code disagree, preserve the plan's no-behavior-change intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

Stage 1 ends with `service.py` still owning the public import surface and ASGI target, while the named implementation groups live in focused modules:

```text
src/kazusa_ai_chatbot/
  service.py                     # compatibility facade and ASGI target
  brain_service/
    __init__.py
    contracts.py                 # Pydantic request/response models
    graph.py                     # graph construction helpers
    intake.py                    # envelope, reply, identity, and user-save helpers
    post_turn.py                 # assistant save, progress recording, consolidation helpers
    cache_startup.py             # RAG initializer cache hydration
    runtime_adapters.py          # runtime adapter registration payload helpers
    health.py                    # health payload construction helpers
```

`service.py` remains responsible for FastAPI route decorators, `app`, mutable process globals, worker lifecycle, `lifespan()`, and compatibility wrappers needed to preserve current test monkeypatch behavior. Later plans can reduce the facade further after entrypoint and private-test surfaces are intentionally migrated.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stage 1 public module | Keep `src/kazusa_ai_chatbot/service.py` | Avoids changing ASGI target, imports, and private test monkeypatch behavior. |
| Extracted package name | `kazusa_ai_chatbot.brain_service` | Avoids ambiguity with the existing `service.py` module while giving service code its own module boundary. |
| Cutover style | Compatible facade | Allows file separation without packaging or runtime command changes. |
| Dependency handling | Pass dependencies from `service.py` into extracted helpers | Preserves monkeypatches against `service_module` and avoids hidden duplicate globals. |
| Test strategy | Existing behavior tests first | This is a refactor; current tests are the behavioral contract. |
| Entry-point fixes | Deferred | Fixing console scripts or Docker changes observable packaging behavior and belongs in Stage 2. |
| Import-time optimization | Incidental only | Stage 1 is separation, not performance tuning. |

## Interface Contract

### `brain_service.contracts`

Owns Pydantic models currently defined in `service.py`.

`service.py` must import and re-export these names so existing code keeps working:

```python
AttachmentIn
DebugModesIn
MentionIn
ReplyTargetIn
AttachmentRefIn
MessageEnvelopeIn
ChatRequest
AttachmentOut
ChatResponse
EventRequest
Cache2AgentStatsResponse
Cache2HealthResponse
HealthResponse
RuntimeAdapterRegistrationRequest
RuntimeAdapterRegistrationResponse
```

### `brain_service.graph`

Owns graph construction mechanics, but not service-global dependencies.

Required callable shape:

```python
def build_graph(
    *,
    relevance_agent_node,
    multimedia_descriptor_agent_node,
    load_conversation_episode_state_node,
    persona_supervisor_node,
):
    ...
```

`service.py` must keep `_build_graph()` as a wrapper that passes the current service-level node symbols. Tests that monkeypatch `service_module.relevance_agent` or `service_module.persona_supervisor2` must still affect `_build_graph()`.

### `brain_service.intake`

Owns these extracted helper bodies:

```python
def compact_reply_context(reply_context: ReplyContext) -> ReplyContext: ...

async def hydrate_reply_context(req: ChatRequest) -> ReplyContext: ...

async def resolve_message_envelope_identities(
    req: ChatRequest,
    *,
    character_global_user_id: str,
    resolve_global_user_id_func,
) -> MessageEnvelope: ...

def active_turn_platform_message_ids(item: QueuedChatItem) -> list[str]: ...

async def save_user_message_from_item(
    item: QueuedChatItem,
    *,
    global_user_id: str,
    reply_context: ReplyContext,
    save_conversation_func,
    resolve_message_envelope_identities_func,
    message_envelope: MessageEnvelope | None = None,
    logger,
) -> None: ...

async def resolve_queued_user(
    item: QueuedChatItem,
    *,
    resolve_global_user_id_func,
    get_user_profile_func,
) -> tuple[str, dict]: ...
```

`service.py` must keep wrappers with the current names: `_compact_reply_context`, `_hydrate_reply_context`, `_resolve_message_envelope_identities`, `_active_turn_platform_message_ids`, `_save_user_message_from_item`, and `_resolve_queued_user`. Dropped/collapsed item orchestration stays in `service.py` because it owns queue completion and service logging.

### `brain_service.post_turn`

Owns these extracted helper bodies:

```python
async def save_assistant_message(
    result: dict,
    *,
    ensure_character_global_identity_func,
    save_conversation_func,
    now_func,
    logger,
) -> None: ...

async def run_consolidation_background(
    state: dict,
    *,
    call_consolidation_subgraph_func,
    personality: dict,
    logger,
) -> None: ...

async def run_conversation_progress_record_background(
    state: dict,
    *,
    record_turn_progress_func,
    logger,
) -> None: ...
```

`service.py` must keep `_save_assistant_message`, `_run_conversation_progress_record_background`, and `_run_consolidation_background` wrappers. The wrappers pass service-level dependencies so existing monkeypatches against `service_module` still affect behavior.

### `brain_service.cache_startup`

Owns this extracted helper body:

```python
async def hydrate_rag_initializer_cache(
    *,
    load_initializer_entries_func,
    get_rag_cache2_runtime_func,
    cache_name: str,
    max_entries: int,
    logger,
) -> int: ...
```

`service.py` must keep `_hydrate_rag_initializer_cache()` as the tested wrapper.

### `brain_service.runtime_adapters`

Owns this extracted helper body:

```python
def register_runtime_adapter_payload(
    req: RuntimeAdapterRegistrationRequest,
    *,
    status: str,
    register_remote_runtime_adapter_func,
) -> RuntimeAdapterRegistrationResponse: ...
```

`service.py` must keep `register_runtime_adapter`, `register_remote_runtime_adapter`, and `_register_runtime_adapter_payload` as compatibility wrappers.

### `brain_service.health`

Owns this extracted helper body:

```python
async def build_health_response(
    *,
    get_db_func,
    get_rag_cache2_runtime_func,
    logger,
) -> HealthResponse: ...
```

`service.py.health()` must remain the FastAPI route function and direct-call test surface.

## LLM Call And Context Budget

Before Stage 1:

- Response path: unchanged existing graph and nested LLM calls.
- Background path: unchanged consolidation, progress, scheduler dispatch, and reflection behavior.

After Stage 1:

- Response path: exactly the same LLM call count, prompts, graph nodes, state keys, and context payloads.
- Background path: exactly the same LLM call count, prompts, context payloads, and blocking behavior.

This plan must not add, remove, reorder, retry, or otherwise modify any LLM call. Any implementation that requires a prompt, graph, RAG, cognition, dialog, or consolidation behavior change is out of scope and must stop.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/brain_service/__init__.py`
  - Package marker and intentionally small public surface.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - Pydantic API models moved from `service.py`.
- `src/kazusa_ai_chatbot/brain_service/graph.py`
  - Dependency-injected graph construction helper.
- `src/kazusa_ai_chatbot/brain_service/intake.py`
  - Extracted intake helper bodies named in `Interface Contract`.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Extracted post-turn helper bodies named in `Interface Contract`.
- `src/kazusa_ai_chatbot/brain_service/cache_startup.py`
  - Extracted RAG initializer cache hydration body.
- `src/kazusa_ai_chatbot/brain_service/runtime_adapters.py`
  - Extracted runtime adapter payload helper body.
- `src/kazusa_ai_chatbot/brain_service/health.py`
  - Extracted health payload helper body.

### Modify

- `src/kazusa_ai_chatbot/service.py`
  - Keep as compatibility facade and ASGI target.
  - Import/re-export models from `brain_service.contracts`.
  - Replace moved implementation bodies with wrappers that pass current dependencies.
  - Keep route decorators, `app`, process globals, worker lifecycle, and `lifespan()`.

- Service-focused tests under `tests/`
  - Add import-surface tests and boundary tests where needed.
  - Do not rewrite tests to hide behavior changes.

### Keep

- `pyproject.toml`
- `Dockerfile`
- `docker-compose.yml`
- `docs/HOWTO.md`
- `src/adapters/*`
- `src/kazusa_ai_chatbot/chat_input_queue.py`
- Graph node, RAG, cognition, dialog, consolidator, scheduler, dispatcher, DB, and message-envelope behavior modules.

## Implementation Order

1. Baseline current behavior.
   - Run focused service tests before editing.
   - Record current failures, if any, without fixing unrelated failures.
2. Create `brain_service` package and move only Pydantic contracts first.
   - Keep all model names re-exported from `service.py`.
   - Run import and model-focused tests.
3. Extract graph construction with dependency injection.
   - Keep `service.py._build_graph()` as the wrapper.
   - Verify monkeypatched service-level graph nodes still affect graph construction.
4. Extract cache startup and runtime adapter payload helpers.
   - Keep `_hydrate_rag_initializer_cache()` and `_register_runtime_adapter_payload()` wrappers.
   - Run cache and runtime adapter registration tests.
5. Extract intake and post-turn helper bodies behind service wrappers.
   - Preserve service-level function names and monkeypatch behavior.
   - Run input queue, background consolidation, bot-side addressing, and health tests.
6. Keep `lifespan()` in `service.py` and run lifespan compatibility tests.
   - Do not create a lifespan runtime module in Stage 1.
   - Preserve startup/shutdown ordering exactly.
   - Run reflection-cycle service lifespan tests.
7. Run static greps and broader service verification.
   - Confirm deferred files and entrypoints were not changed.
   - Confirm no concrete adapter normalizers are imported by brain modules.

## Progress Checklist

- [x] Stage 1A - Baseline behavior recorded.
  - Covers: current focused service tests before edits.
  - Verify: run the first test group in `Verification`.
  - Evidence: record command output and any pre-existing failures.
  - Handoff: next agent starts at Stage 1B.
  - Sign-off: `Codex / 2026-05-06` after focused baseline passed in the project venv and the global-Python `tzdata` environment failure was recorded.
- [x] Stage 1B - Contracts extracted.
  - Covers: `brain_service/contracts.py` and service re-exports.
  - Verify: import checks and service model tests pass.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 1C.
  - Sign-off: `Codex / 2026-05-06` after import checks passed and focused service tests passed with contract re-exports.
- [x] Stage 1C - Graph, cache, and runtime adapter helpers extracted.
  - Covers: `graph.py`, `cache_startup.py`, `runtime_adapters.py`.
  - Verify: graph, cache, and runtime adapter tests pass.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 1D.
  - Sign-off: `Codex / 2026-05-06` after graph, cache hydration, and runtime adapter registration tests passed through the focused service suite.
- [x] Stage 1D - Intake and post-turn helpers extracted.
  - Covers: `intake.py`, `post_turn.py`, service wrappers.
  - Verify: queue, background consolidation, bot-side addressing, and health tests pass.
  - Evidence: changed files and test output.
  - Handoff: next agent starts at Stage 1E.
  - Sign-off: `Codex / 2026-05-06` after queue, background consolidation, bot-side addressing, and health tests passed.
- [x] Stage 1E - Lifespan compatibility and final checks complete.
  - Covers: `service.py.lifespan()`, final static greps, broader focused test suite.
  - Verify: all commands in `Verification` pass or pre-existing failures are documented.
  - Evidence: final command output and static grep results.
  - Handoff: Stage 1 complete; Stage 2 entrypoint work requires a new or superseding plan.
  - Sign-off: `Codex / 2026-05-06` after lifespan tests, static greps, import checks, and broader service-adjacent smoke passed.

## Verification

### Baseline And Focused Tests

Run before and after the refactor:

```powershell
pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q
```

Run targeted direct-call/import checks:

```powershell
python -c "from kazusa_ai_chatbot.service import app, ChatRequest, ChatResponse; print(type(app).__name__, ChatRequest.__name__, ChatResponse.__name__)"
python -c "import importlib.util; print(importlib.util.find_spec('kazusa_ai_chatbot.service') is not None)"
```

### Static Greps

These checks must pass after implementation:

```powershell
rg "kazusa_ai_chatbot\.main:main|kazusa-brain|kazusa_ai_chatbot\.service:app" pyproject.toml Dockerfile docker-compose.yml docs README.md
rg "from adapters|import adapters|adapters\." src\kazusa_ai_chatbot\brain_service src\kazusa_ai_chatbot\service.py
rg "BRAIN_EXECUTOR_COUNT" src\kazusa_ai_chatbot
```

Expected grep interpretation:

- Existing entrypoint strings must remain only where they already existed before Stage 1; Stage 1 must not edit those files.
- Adapter imports in brain service modules must return no matches.
- `BRAIN_EXECUTOR_COUNT` must remain unused outside its pre-existing config definition; Stage 1 must not introduce executor behavior.

### Broader Smoke

Run at least one broader service-adjacent suite after focused tests pass:

```powershell
pytest tests\test_persona_supervisor2.py tests\test_consolidator_efficiency.py tests\test_dispatcher.py -q
```

If live LLM or database-dependent tests are not run, record that explicitly in `Execution Evidence` and explain why.

## Acceptance Criteria

This plan is complete when:

- `service.py` remains the working ASGI target and compatibility import surface.
- API models are available from `kazusa_ai_chatbot.service` exactly as before.
- Existing service tests that call or monkeypatch private service symbols still pass.
- The internal implementation code is split into `kazusa_ai_chatbot.brain_service` modules.
- Queue ordering, graph execution, response creation, assistant persistence, progress recording, consolidation, cache hydration, scheduler startup, dispatcher setup, runtime adapter registration, and health behavior are unchanged.
- No entrypoint, Docker, docs, adapter, DB schema, prompt, graph behavior, RAG behavior, scheduler behavior, or LLM call-count changes are included.
- Verification commands pass, or any failure is clearly identified as pre-existing and unrelated.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Private monkeypatch tests stop affecting behavior | Keep wrappers in `service.py` and pass current service dependencies into extracted helpers | Existing service tests and targeted monkeypatch tests |
| Import target changes accidentally | Keep `service.py`; do not create `kazusa_ai_chatbot/service/` in Stage 1 | Import checks and unchanged deployment greps |
| Graph node binding changes | Build graph through dependency-injected helper called by service wrapper | Graph tests that monkeypatch service-level nodes |
| Startup ordering changes | Keep `lifespan()` wrapper and extract only leaf setup helpers | Reflection-cycle service lifespan tests |
| Post-turn gate changes | Preserve worker loop and await order | Background consolidation and input queue tests |
| Refactor hides a behavior change in tests | Run baseline before editing and compare focused test results after each stage | Execution evidence per checklist stage |

## Execution Evidence

- Baseline focused tests:
  - `pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` with global Python failed before edits because `tzdata` was unavailable and `Pacific/Auckland` could not load.
  - `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` passed: `56 passed in 3.35s`.
- Stage 1B contracts extraction tests:
  - `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.service import app, ChatRequest, ChatResponse; print(type(app).__name__, ChatRequest.__name__, ChatResponse.__name__)"` printed `FastAPI ChatRequest ChatResponse`.
  - `venv\Scripts\python.exe -c "import importlib.util; print(importlib.util.find_spec('kazusa_ai_chatbot.service') is not None)"` printed `True`.
  - `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` passed after contracts and helper extraction: `56 passed in 3.60s`.
- Stage 1C graph/cache/runtime adapter tests:
  - Changed files: `src/kazusa_ai_chatbot/brain_service/graph.py`, `src/kazusa_ai_chatbot/brain_service/cache_startup.py`, `src/kazusa_ai_chatbot/brain_service/runtime_adapters.py`, and wrapper calls in `src/kazusa_ai_chatbot/service.py`.
  - `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service\__init__.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\brain_service\graph.py src\kazusa_ai_chatbot\brain_service\intake.py src\kazusa_ai_chatbot\brain_service\post_turn.py src\kazusa_ai_chatbot\brain_service\cache_startup.py src\kazusa_ai_chatbot\brain_service\runtime_adapters.py src\kazusa_ai_chatbot\brain_service\health.py` passed.
  - `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` passed after graph/cache/runtime extraction: `56 passed in 3.35s`.
- Stage 1D intake/post-turn tests:
  - Changed files: `src/kazusa_ai_chatbot/brain_service/intake.py`, `src/kazusa_ai_chatbot/brain_service/post_turn.py`, and compatibility wrappers in `src/kazusa_ai_chatbot/service.py`.
  - `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` passed after intake/post-turn extraction: `56 passed in 3.35s`.
  - `venv\Scripts\python.exe -m pytest tests\test_bot_side_addressing.py -q` passed: `2 passed in 1.87s`.
- Stage 1E final focused tests:
  - `venv\Scripts\python.exe -c "from kazusa_ai_chatbot.service import app, ChatRequest, ChatResponse; print(type(app).__name__, ChatRequest.__name__, ChatResponse.__name__)"` printed `FastAPI ChatRequest ChatResponse`.
  - `venv\Scripts\python.exe -c "import importlib.util; print(importlib.util.find_spec('kazusa_ai_chatbot.service') is not None)"` printed `True`.
  - `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_service_background_consolidation.py tests\test_service_health.py tests\test_reflection_cycle_stage1c_service.py tests\test_runtime_adapter_registration.py tests\test_rag_cache2_persistent.py -q` passed in the final run: `56 passed in 3.35s`.
  - `git status --short` showed only the planned `src/kazusa_ai_chatbot/service.py`, `src/kazusa_ai_chatbot/brain_service/`, and this plan file changed.
- Static grep results:
  - `rg "kazusa_ai_chatbot\.main:main|kazusa-brain|kazusa_ai_chatbot\.service:app" pyproject.toml Dockerfile docker-compose.yml docs README.md` returned only existing entrypoint strings in `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, and `docs\HOWTO.md`; none of those files were edited.
  - `rg "from adapters|import adapters|adapters\." src\kazusa_ai_chatbot\brain_service src\kazusa_ai_chatbot\service.py` returned no matches, which is the expected result.
  - `rg "BRAIN_EXECUTOR_COUNT" src\kazusa_ai_chatbot` returned only `src\kazusa_ai_chatbot\config.py:BRAIN_EXECUTOR_COUNT = int(os.getenv("BRAIN_EXECUTOR_COUNT", "1"))`.
- Broader smoke results:
  - `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2.py tests\test_consolidator_efficiency.py tests\test_dispatcher.py -q` passed: `15 passed, 4 deselected in 2.08s`.
- Skipped verification, if any:
  - Live LLM and real database tests were not run. This stage is a behavior-preserving module split, and the approved verification gates are deterministic focused service tests, import checks, static greps, and service-adjacent smoke tests.
