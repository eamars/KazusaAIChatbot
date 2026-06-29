# past dialog cognition residual plan

## Summary

- Goal: deterministically attach compact trace-backed cognition residual to L2a when a Kazusa-authored past dialog is already attached as current-turn context, while treating missing trace data as ordinary forgetting.
- Plan class: large.
- Status: draft.
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `local-llm-architecture`, `no-prepost-user-input`.
- Overall cutover strategy: compatible for runtime behavior; bigbang for the new residual field contract and consumer boundary.
- Highest-risk areas: private cognition leakage into broad prompt payloads, accidental consumers outside L2a, keyword or LLM-gated retrieval, and over-broad use of raw trace data.
- Acceptance criteria: L2a is the only LLM consumer, unavailable trace data is omitted without projection, no new LLM call is introduced, direct reply/quote and conversation-evidence attachments use the same bounded projector, conversation-evidence candidate loading uses row ids rather than platform-message-id-only lookups, and verification greps/tests prove no public or downstream leakage.

## Context

Kazusa already records per-stage LLM trace steps in `llm_trace_steps`, and assistant conversation rows can carry `llm_trace_id`. The proposed behavior is not a new durable memory system. It is a best-effort private context lookup for cases where current processing has already attached a specific past Kazusa dialog as relevant context.

The driving use case is: when Kazusa is asked about a past message or a prior idea she expressed, the visible dialog alone can be insufficient. A compact residual from the cognition stages that produced that dialog can help the next L2a consciousness stage rebuild continuity. If trace data is absent, expired, captured in metadata mode, malformed, or not connected to a Kazusa-authored dialog, the system must attach nothing and let Kazusa forget.

Current implementation gaps:

- `conversation_history.llm_trace_id` is available for assistant outbound rows, but the live response path does not read past trace steps.
- `llm_trace_steps.parsed_output` contains useful stage output only when `LLM_TRACE_CAPTURE_MODE=full`; in metadata mode it is `{}` and must be treated as unavailable data.
- `_hydrate_reply_context` fetches a replied conversation row and projects visible message context, but it does not preserve a private L2a-only cognition side channel.
- `_hydrate_reply_context` only loads the replied DB row when visible reply metadata or attachments need hydration; residual lookup needs a separate private row lookup when `reply_to_message_id` is present, even if visible reply metadata is already complete.
- RAG conversation evidence projects public evidence strings and source references; it must not become a carrier for private cognition residual.
- `internal_monologue_residue_context` establishes the correct pattern: private cognition-adjacent context is narrow and L2a-only.
- The current RAG source refs used for conversation evidence are attached in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py` under `rag_result.supervisor_trace.dispatched[*].source_refs`, while L2 prompt projection already strips trace-only private keys from supervisor trace before model consumption.
- The current resolver loop applies RAG capability observations in `src/kazusa_ai_chatbot/cognition_resolver/loop.py` by copying `observation["rag_result"]` into cognition state. The private residual side channel must be attached there after `rag_result` is applied, not inside `ResolverObservationV1`, because resolver observations are stored and can become broader prompt context.

This plan maps the proposal to a bounded implementation that favors deterministic data attachment over semantic gating. Fetching residual is cheap; deciding whether to fetch based on user wording is the expensive and inconsistent part. Therefore the trigger is structural: a past Kazusa-authored dialog is already attached by an existing context path.

## Mandatory Skills

- `development-plan`: load before approving, executing, reviewing, or updating this plan.
- `py-style`: load before creating or editing Python source files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing cognition contracts, prompts, RAG integration, or trace-backed LLM context.
- `no-prepost-user-input`: load before touching any code path that could be interpreted as keyword routing or semantic gating over user text.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- Execution must use parent-led native subagents unless the user explicitly approves fallback execution.
- Do not modify, add, or remove production code unless the user explicitly approves execution of this plan or separately commands implementation.
- Do not read `.env`. Do not change `LLM_TRACE_CAPTURE_MODE`; this plan consumes whatever trace data already exists.
- Treat existing coding-agent worktree changes as unrelated unless the user explicitly brings them into scope.
- Do not add keyword matching, semantic user-text matching, regex phrase detection, or LLM-driven "should fetch residual" decisions.
- Deterministically attempt residual lookup only after existing code has already attached a past Kazusa-authored dialog through direct reply/quote context or conversation-evidence context.
- If there is no Kazusa-authored dialog, no `llm_trace_id`, no trace row, no parsed stage output, an expired trace, or a read/project failure, attach no residual context and continue the existing response path.
- The only LLM consumer of `past_dialog_cognition_context` is L2a consciousness. L1, L2b, L2c1, L2c2, L2d, L3, dialog generation, RAG evaluator/finalizer, consolidation, scheduler, reflection, adapters, and delivery code must not consume it.
- Do not add residual to `prompt_message_context`, `rag_result`, public `conversation_evidence`, message envelope projection, dialog output, durable memories, reflection promotion, scheduler inputs, adapter payloads, or logs intended for ordinary UI display.
- Do not pass `raw_messages`, `raw_response_text`, raw prompts, raw model responses, trace ids, database ids, or implementation-stage labels into the L2a prompt.
- Do not query conversation rows by `platform_message_id` alone for conversation-evidence residual candidates. Use `conversation_row_id` or `_id` refs from RAG source refs; skip refs that cannot be resolved without platform/channel scoping.
- Deterministic code owns trace lookup, stage filtering, size limits, ordering, and omission on failure.
- L2a owns current subjective use of the residual. The residual is weak context, not evidence of fact, not a command, not final stance, and not dialog wording.
- RAG continues to return evidence. RAG does not own persona stance or private cognition residual.
- New Python code must follow project fail-fast rules for required internal data, avoid silent broad exception swallowing, and keep structural validation separate from semantic judgment.
- New tests must use deterministic patched data. Live LLM tests are not required for this plan unless the executor adds a prompt-render smoke that must be inspected one case at a time.

## Must Do

- Add a runtime DB facade helper that reads selected `llm_trace_steps` by trace id and stage names, with a projection that excludes raw prompts and raw responses.
- Add a runtime DB facade helper that loads bounded conversation rows by `conversation_history` row ids for RAG source-ref candidate resolution.
- Create a dedicated `past_dialog_cognition` module that owns candidate filtering, trace-step projection, prompt-safe formatting, caps, omission behavior, and source-status diagnostics.
- Add a single optional state and prompt field named `past_dialog_cognition_context`.
- Wire direct reply/quote context so a replied Kazusa-authored assistant row with `llm_trace_id` triggers deterministic best-effort residual lookup.
- Wire conversation-evidence context so Kazusa-authored past dialog rows already retrieved as conversation evidence can trigger the same deterministic residual lookup through a private side channel, without inserting private residual into `rag_result` or public evidence strings.
- Keep `ResolverObservationV1` free of `past_dialog_cognition_context`; the loop may copy the private field into cognition state, but resolver observations must remain prompt-safe summaries and public evidence.
- Pass `past_dialog_cognition_context` through `ConversationContextPromptV1` into L2a only.
- Update the L2a prompt contract to describe this context as past private subjective context that is weaker than current visible input and retrieved evidence.
- Limit the initial projected stages to `l2a_conscious_framing` and `l2c1_judgment_synthesis`.
- Add focused module tests, integration tests, consumer-boundary tests, and static guardrail greps listed in `Verification`.
- Add or update local README documentation for the new module and affected cognition boundary.

## Deferred

- Do not create a durable cognition-residual collection.
- Do not backfill old conversations or traces.
- Do not require full trace capture for correctness.
- Do not add a new LLM summarizer, router, classifier, evaluator, repair call, or retry loop.
- Do not add semantic or keyword gating for "asked about the past", "what did you mean", quoted text, or similar phrases.
- Do not expose residual in debug-channel ordinary response payloads, adapter payloads, dialog text, public RAG evidence, or memory consolidation inputs.
- Do not use L2b, L2c2, L2d, L3, final dialog, raw prompt text, or raw response text as projected residual in this plan.
- Do not refactor unrelated RAG, cognition, trace-writing, adapter, scheduler, reflection, or consolidation architecture.
- Do not change coding-agent files as part of this plan.

## Cutover Policy

Overall strategy: compatible for runtime behavior; bigbang for the new residual contract and consumer boundary.

| Area | Policy | Instruction |
|---|---|---|
| Runtime behavior without trace data | compatible | Preserve the current response path when residual data is unavailable. Missing data means no residual field is projected. |
| New context field | bigbang | Use exactly `past_dialog_cognition_context`. Do not create aliases, compatibility names, fallback field names, or duplicate carrier fields. |
| Consumer boundary | bigbang | L2a is the only consumer. Delete or block any implementation that forwards the field to other stages. |
| Trace capture mode | compatible | Consume existing full-capture traces when present. Metadata-mode traces produce no projected residual. |
| RAG evidence payload | bigbang | Keep private residual out of `rag_result`, public `conversation_evidence`, and RAG finalizer text. |
| Tests and docs | bigbang | Add tests and docs for the new contract. Do not preserve undocumented private behavior. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative dual-path strategy by default.
- Any area marked `bigbang` must use the single contract listed here.
- Any change to this cutover policy requires user approval before implementation.

## Target State

When a current turn carries a past Kazusa-authored dialog through an existing attachment path, deterministic runtime code attempts to retrieve compact cognition residual from that dialog's `llm_trace_id`.

The completed response path is:

```text
current message
  -> existing reply/quote hydration and/or RAG conversation evidence
  -> deterministic Kazusa-dialog candidate extraction
  -> DB facade reads selected llm_trace_steps
  -> prompt-safe past_dialog_cognition projection
  -> persona state past_dialog_cognition_context
  -> CognitionChainInputV1.conversation_context.past_dialog_cognition_context
  -> L2a consciousness only
  -> downstream stages receive only normal L2a outputs, not the residual field
```

If any lookup step fails or yields no safe parsed output, the field is omitted or empty and the rest of the pipeline behaves as it does today.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Trigger | Deterministically attempt lookup when a past Kazusa-authored dialog is already attached. | This avoids inconsistent LLM/tool decisions and avoids brittle keyword matching over user text. |
| Availability | Missing trace data is ordinary forgetting. | The feature improves continuity when evidence exists but does not create a correctness dependency. |
| Consumer | L2a consciousness is the only LLM consumer. | L2a owns current private framing; other stages should see only normal cognition outputs. |
| Carrier | Use `past_dialog_cognition_context` in conversation context, not `prompt_message_context` or `rag_result`. | Existing broad payloads flow to too many consumers and would make leakage hard to audit. |
| Source data | Use `llm_trace_steps.parsed_output` only. | Parsed stage outputs are bounded and typed; raw prompts/responses are too broad and unsafe. |
| Initial stages | Project only `l2a_conscious_framing` and `l2c1_judgment_synthesis`. | This captures private framing plus settled judgment while excluding dialog wording and action selection. |
| Latency | Add bounded DB reads and no LLM calls. | The user identified fetch cost as low and decision cost as the risk. |
| RAG role | RAG may identify attached past dialog candidates, but private residual is built outside public RAG evidence. | RAG returns evidence; cognition owns private stance and continuity. |
| RAG source refs | Use `supervisor_trace.dispatched[*].source_refs` as trace-only row-id hints, then resolve rows through DB facade before filtering. | Source refs are already trace-only; row reload keeps author and `llm_trace_id` checks out of public RAG text. |
| Conversation row lookup | Resolve RAG evidence candidates by `conversation_row_id` or `_id`; do not use `platform_message_id` without platform/channel scope. | Platform message ids are adapter-local and unsafe as global lookup keys. |
| Resolver observation | Do not place residual in `ResolverObservationV1`. Attach it directly to cognition state inside the resolver loop after `rag_result` is applied. | Observations are stored in resolver state and can reach broader prompt context. |

## Contracts And Data Shapes

### Module Boundary

Create `src/kazusa_ai_chatbot/past_dialog_cognition/` as the single owner of residual projection.

Public entrypoint:

```python
async def build_past_dialog_cognition_context(
    candidates: Sequence[PastDialogCognitionCandidate],
    *,
    character_global_user_id: str,
    max_dialogs: int = 3,
    context_char_limit: int = 1800,
) -> PastDialogCognitionLookupResult:
    ...
```

The production implementation may use dataclasses or TypedDicts, but the public module contract must preserve this shape:

```python
{
    "past_dialog_cognition_context": str,
    "candidate_count": int,
    "selected_count": int,
    "status": str,
    "diagnostics": list[dict[str, str]],
}
```

Only `past_dialog_cognition_context` may enter the L2a prompt. Counts, statuses, source ids, trace ids, and diagnostics are for tests and trace/debug metadata only.

### Candidate Contract

Each candidate must represent a visible Kazusa-authored past dialog already attached by an existing context mechanism.

Required candidate fields:

```python
{
    "visible_text": str,
    "llm_trace_id": str,
    "created_at": object,
    "source": "reply_context" | "conversation_evidence",
    "role": str,
    "global_user_id": str,
    "conversation_row_id": str,
    "platform_message_id": str,
    "platform": str,
    "platform_channel_id": str,
}
```

Candidate construction must filter out non-assistant rows, non-Kazusa rows, empty visible text, and rows without `llm_trace_id` before trace lookup.

### Trace Read Contract

Add a DB facade helper in `src/kazusa_ai_chatbot/db/llm_tracing.py`:

```python
async def list_llm_trace_steps_for_trace_ids(
    trace_ids: Sequence[str],
    *,
    stage_names: Sequence[str],
) -> list[dict[str, Any]]:
    ...
```

The helper must project only:

```python
{
    "trace_id": str,
    "stage_name": str,
    "sequence": int,
    "parsed_output": dict,
    "created_at": object,
}
```

It must not project `raw_messages` or `raw_response_text`.

### Conversation Row Read Contract

Add a DB facade helper in `src/kazusa_ai_chatbot/db/conversation.py`:

```python
async def list_conversation_rows_by_row_ids(
    row_ids: Sequence[str],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ...
```

The helper must query `_id`/`conversation_row_id` values only, apply the provided limit, and return normal conversation row fields needed for candidate construction:

```python
{
    "_id": object,
    "conversation_row_id": str,
    "platform": str,
    "platform_channel_id": str,
    "role": str,
    "platform_message_id": str,
    "global_user_id": str,
    "display_name": str,
    "body_text": str,
    "llm_trace_id": str,
    "timestamp": str,
}
```

This helper must not accept `platform_message_id` by itself. Direct reply/quote hydration may continue using the existing platform/channel-scoped helper.

### Residual Projection Contract

The projector must read only these parsed-output fields:

```python
{
    "l2a_conscious_framing": [
        "internal_monologue",
        "logical_stance",
        "character_intent",
    ],
    "l2c1_judgment_synthesis": [
        "logical_stance",
        "character_intent",
        "judgment_note",
    ],
}
```

The L2a prompt-facing text must be compact natural-language context. It must not include trace ids, database ids, raw prompts, raw model responses, raw JSON blobs, or stage implementation labels.

### State And Prompt Contract

Add `past_dialog_cognition_context: str` to `ConversationContextPromptV1`.

Add the same optional string to the persona graph state only as a private cognition-input field. Do not add it to message-envelope prompt projection or RAG public projection.

L2a prompt instruction:

- Treat this as private context from the character's earlier cognition around an attached past dialog.
- Use it only to understand continuity around the anchored past dialog.
- Do not treat it as a fact source, user command, final answer, or dialog wording.
- Prefer current visible input and retrieved public evidence when they conflict.

## LLM Call And Context Budget

Before this plan:

- Response path L2a receives current message context, conversation context, retrieved evidence, and optional `internal_monologue_residue_context`.
- No trace-backed past-dialog cognition residual is supplied.

After this plan:

- No new LLM calls are added.
- L2a receives at most one additional bounded string, `past_dialog_cognition_context`.
- Maximum prompt-facing residual budget is `1800` characters across at most `3` past dialogs.
- Empty, unavailable, expired, malformed, metadata-mode, or non-Kazusa data contributes `0` characters.
- Direct reply/quote path performs at most one bounded trace lookup for the replied assistant row.
- Direct reply/quote path performs at most one platform/channel-scoped conversation row lookup for `reply_to_message_id`, even when visible reply metadata is already complete.
- Conversation-evidence path performs bounded trace lookup for at most three selected Kazusa-authored rows already present in evidence references.

The L2a context budget remains below the default `50k tokens` cap by using conservative character limits. The feature must not add response-path model latency beyond DB reads and projection.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/past_dialog_cognition/README.md`: module ownership, source data, consumer boundary, and forbidden consumers.
- `src/kazusa_ai_chatbot/past_dialog_cognition/__init__.py`: public exports.
- `src/kazusa_ai_chatbot/past_dialog_cognition/models.py`: candidate/result data shapes.
- `src/kazusa_ai_chatbot/past_dialog_cognition/projection.py`: parsed trace-step to prompt-safe text projection.
- `src/kazusa_ai_chatbot/past_dialog_cognition/runtime.py`: DB-backed context builder that applies filtering, caps, and omission behavior.
- `tests/test_past_dialog_cognition_context.py`: focused module and projector tests.
- `tests/test_past_dialog_cognition_reply_integration.py`: direct reply/quote side-channel tests.
- `tests/test_past_dialog_cognition_rag_integration.py`: conversation-evidence side-channel and public-payload leak tests.

### Modify

- `src/kazusa_ai_chatbot/db/llm_tracing.py`: add bounded read helper for selected trace steps.
- `src/kazusa_ai_chatbot/db/__init__.py`: export the new DB facade helper.
- `src/kazusa_ai_chatbot/db/conversation.py`: add `list_conversation_rows_by_row_ids` for bounded row-id lookup from RAG source refs.
- `src/kazusa_ai_chatbot/service.py`: add a private direct-reply candidate lookup that uses `get_conversation_by_platform_message_id(platform, platform_channel_id, reply_to_message_id)` when `reply_to_message_id` is present, regardless of whether visible reply metadata was already hydrated; wire resulting Kazusa-dialog candidates into the private residual builder without changing visible `prompt_message_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add optional private state field for `past_dialog_cognition_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: include the private field in `ConversationContextPromptV1` and pass it into cognition chain input.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`: after applying a RAG observation's `rag_result` to `cognition_state`, derive and attach private `past_dialog_cognition_context` directly to cognition state.
- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`: add `past_dialog_cognition_context` to conversation context contract.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`: update L2a system/human prompt handling to consume the field; no other stage in this file may consume it.
- `src/kazusa_ai_chatbot/nodes/README.md`: document L2a-only consumer boundary.
- `src/kazusa_ai_chatbot/llm_tracing/README.md` or nearest trace README: document that metadata-mode traces do not provide residual content.
- `tests/test_internal_monologue_residue_prompt_boundaries.py`: extend boundary assertions or add equivalent assertions proving the new field is L2a-only.

### Keep

- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`: do not add residual to visible message context.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/projection.py`: do not add private residual to public evidence projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`: do not place private residual into `rag_result`; source refs remain trace-only row-id hints.
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`: do not add `past_dialog_cognition_context` to `ResolverObservationV1`.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`: no residual consumer.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`: no residual consumer.
- `src/kazusa_ai_chatbot/consolidation/`, scheduler, reflection, and adapter modules: no residual consumer.

## Overdesign Guardrail

- Actual problem: Kazusa can be asked about a past Kazusa-authored dialog, and visible text alone can omit the cognition context that produced the earlier response.
- Minimal change: add a deterministic best-effort trace-backed residual side channel for already-attached Kazusa dialogs and expose it only to L2a.
- Ownership boundaries: DB facade reads trace steps; `past_dialog_cognition` validates and projects residual; RAG and reply hydration identify already-attached dialog candidates; L2a uses the private context for current subjective framing; downstream stages consume normal cognition outputs only.
- Rejected complexity: no new durable store, no backfill, no new LLM call, no semantic fetch decision, no keyword matching, no compatibility aliases, no broad prompt payload, no public RAG residual, no extra agents, no retry loop, no adapter exposure.
- Evidence threshold: add any broader consumer, projected stage, durable storage, semantic gate, or LLM summarizer only after a separate approved plan with focused failure evidence and new verification gates.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate cutover strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target module as high-scrutiny changes. Updating an existing module outside the target module or introducing a new code path, prompt, or variable requires the justification already listed in `Change Surface`.
- The responsible agent must search for existing equivalent behavior before adding helpers. If equivalent behavior exists, extract or reuse it through the appropriate project boundary instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent loads required skills, rereads this plan, records current `git status --short`, and confirms no production-code execution starts without approval.
2. Parent adds `tests/test_past_dialog_cognition_context.py` with failing coverage for projector behavior:
   - no candidate returns empty context;
   - metadata-mode empty `parsed_output` returns empty context;
   - L2a and L2c1 parsed fields are projected;
   - raw prompt/response fields are never projected;
   - max dialog and character caps are enforced.
3. Parent runs `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py -q` and records the expected missing-module or failing-test baseline.
4. Parent starts the production-code subagent with this approved plan, mandatory skills, focused test contract, and production ownership boundary.
5. Production-code subagent creates `past_dialog_cognition` models/projection/runtime and DB trace read helper, then reports changed files and focused test status.
6. Parent reruns `tests/test_past_dialog_cognition_context.py` and records the pass before integration wiring proceeds.
7. Parent adds or updates direct reply integration tests in `tests/test_past_dialog_cognition_reply_integration.py`:
   - replied Kazusa assistant row with full parsed trace reaches L2a input;
   - replied Kazusa assistant row reaches L2a input even when adapter-supplied reply metadata is already complete and `_hydrate_reply_context` does not need visible DB hydration;
   - replied non-Kazusa row produces no residual;
   - missing trace id, missing trace row, and empty parsed output produce no residual;
   - visible `prompt_message_context` remains unchanged.
8. Production-code subagent wires direct reply/quote candidate extraction and private residual attachment in the approved service path only.
9. Parent reruns reply integration tests and focused module tests.
10. Parent adds or updates conversation-evidence integration tests in `tests/test_past_dialog_cognition_rag_integration.py`:
    - Kazusa-authored evidence refs can produce private residual context;
    - public `rag_result` and `conversation_evidence` do not contain residual text;
    - no residual appears when evidence refs are non-Kazusa, missing trace, or unavailable.
11. Production-code subagent wires the RAG-to-cognition private side channel in `src/kazusa_ai_chatbot/cognition_resolver/loop.py` immediately after `cognition_state["rag_result"] = observation["rag_result"]`, outside public RAG projection and outside `ResolverObservationV1`.
12. Parent reruns RAG integration tests, reply tests, and module tests.
13. Parent adds or updates cognition contract and L2a prompt-boundary tests:
    - `ConversationContextPromptV1` accepts `past_dialog_cognition_context`;
    - L2a prompt receives the field;
    - L1, L2b, L2c1, L2c2, L2d, L3, dialog, and consolidation do not receive it.
14. Production-code subagent updates cognition contracts and L2a prompt handling only after tests define the boundary.
15. Parent runs the full verification commands in `Verification`.
16. Parent updates module README and affected architecture README files.
17. Parent runs static greps and records expected zero or allowed matches.
18. Parent starts the independent code-review subagent after all planned verification passes.
19. Parent remediates review findings only inside approved scope, reruns affected verification, records evidence, and leaves this plan in `draft` unless the user approved execution and lifecycle completion.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused module contract established
  - Covers: implementation steps 1-6.
  - Verify: `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py -q`.
  - Evidence: record baseline failure, changed files, and final focused-test pass in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - direct reply/quote private residual wiring complete
  - Covers: implementation steps 7-9.
  - Verify: `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_reply_integration.py tests\test_past_dialog_cognition_context.py -q`.
  - Evidence: record test output and confirm `prompt_message_context` stayed unchanged.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - conversation-evidence private side channel complete
  - Covers: implementation steps 10-12.
  - Verify: `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_rag_integration.py tests\test_past_dialog_cognition_context.py -q`.
  - Evidence: record test output and confirm public `rag_result` has no residual text.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - cognition contract and L2a-only prompt boundary complete
  - Covers: implementation steps 13-15.
  - Verify: boundary tests, py_compile, and static greps listed in `Verification`.
  - Evidence: record allowed matches and zero-match checks before moving on.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - docs and lifecycle evidence complete
  - Covers: implementation steps 16-17.
  - Verify: README diffs are scoped to residual ownership, trace capture caveat, and L2a-only boundary.
  - Evidence: record docs changed and final static grep output.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - independent code review complete
  - Covers: implementation steps 18-19.
  - Verify: independent review completed, findings resolved or recorded, affected checks rerun.
  - Evidence: record reviewer role, findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: plan can be completed only after user-approved execution and lifecycle update.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py -q`
  - Expected: passes after Stage 1.
- `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_reply_integration.py -q`
  - Expected: passes after Stage 2.
- `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_rag_integration.py -q`
  - Expected: passes after Stage 3.
- `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q`
  - Expected: passes with updated L2a-only boundary assertions or unchanged existing assertions plus new equivalent test coverage.

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_cognition_chain_core_contracts.py tests\test_cognition_chain_connector_mapping.py -q`
  - Expected: passes after contract wiring.
- `venv\Scripts\python -m pytest tests\test_rag_projection.py -q`
  - Expected: passes after RAG side-channel wiring.

### Python Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\db\llm_tracing.py src\kazusa_ai_chatbot\past_dialog_cognition\__init__.py src\kazusa_ai_chatbot\past_dialog_cognition\models.py src\kazusa_ai_chatbot\past_dialog_cognition\projection.py src\kazusa_ai_chatbot\past_dialog_cognition\runtime.py src\kazusa_ai_chatbot\cognition_chain_core\contracts.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py`
  - Expected: exits `0`.

### Static Greps

- `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l3.py src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\rag`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable for zero matches.
- `rg "raw_messages|raw_response_text" src\kazusa_ai_chatbot\past_dialog_cognition`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable for zero matches.
- `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_chain_core\stages src\kazusa_ai_chatbot\nodes`
  - Expected allowed matches only in L2a handling, cognition input construction, persona state schema, and tests. Any L3, dialog, consolidation, scheduler, adapter, or public RAG consumer match blocks sign-off.
- `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_resolver`
  - Expected allowed matches only in `src\kazusa_ai_chatbot\cognition_resolver\loop.py`. Any match in `contracts.py` or resolver-state projection blocks sign-off.
- `rg "past_dialog_cognition" src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py src\kazusa_ai_chatbot\rag`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable for zero matches unless a reviewed source-ref-only edit to `persona_supervisor2_rag_projection.py` was required; any residual text or `past_dialog_cognition_context` match blocks sign-off.
- `rg "past_dialog_cognition" src\kazusa_ai_chatbot\message_envelope src\kazusa_ai_chatbot\adapters src\kazusa_ai_chatbot\scheduler src\kazusa_ai_chatbot\reflection`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable for zero matches.
- `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_resolver\contracts.py`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable for zero matches.

### Prompt Render Check

- Add or reuse a deterministic prompt-render test that builds L2a input with non-empty `past_dialog_cognition_context`.
  - Expected: rendered L2a human payload contains the compact context and no trace ids, raw prompts, raw responses, or stage implementation labels.

## Independent Plan Review

Manual review performed without subagent, per user instruction.

Review inputs:

- `development_plans/README.md`
- `development_plans/active/short_term/past_dialog_cognition_residual_plan.md`
- `README.md`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/db/llm_tracing.py`
- `src/kazusa_ai_chatbot/llm_tracing/__init__.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/conversation_evidence/projection.py`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`

Findings addressed:

- Blocker: the plan named the RAG projection file under the wrong package. The plan now uses the real path, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`.
- Blocker: the plan used the vague path `src/kazusa_ai_chatbot/cognition_resolver/` or another current integration point. The plan now names `src/kazusa_ai_chatbot/cognition_resolver/loop.py` as the RAG-to-cognition private side-channel integration point.
- Blocker: returning residual through `ResolverObservationV1` would store private context in resolver state. The plan now places RAG side-channel attachment in `cognition_resolver/loop.py` after `rag_result` is applied and explicitly forbids adding the field to `ResolverObservationV1`.
- Blocker: the candidate contract required module-side non-Kazusa filtering but did not include `role` or `global_user_id`. The candidate shape now includes author and source identifiers.
- Blocker: the conversation-evidence lookup path did not prevent unsafe `platform_message_id`-only DB lookup. The plan now requires `list_conversation_rows_by_row_ids`, uses `conversation_row_id` or `_id`, and skips unscoped platform-message-id-only refs.
- Blocker: direct reply residual lookup could have been skipped when adapter-supplied reply metadata was already complete. The plan now requires a separate private platform/channel-scoped row lookup when `reply_to_message_id` is present.
- Non-blocking: static greps did not explicitly cover the real RAG projection file under `nodes/`. The verification section now includes a targeted grep for that file and the `rag` package.

Approval status: review blockers are addressed in this draft. The plan remains `draft`; execution is still not authorized.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, prompt/RAG payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused and regression tests, execution evidence, static-grep expectations, and path-safe commands.

The parent agent may fix review findings directly only when the fix is inside the approved change surface or the finding corrects review-only test/documentation evidence. If a finding requires a new consumer, new projected stage, new public payload, durable storage, or semantic gating, stop and request plan approval for the changed scope before modifying code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `past_dialog_cognition_context` exists as the single private residual field.
- Direct reply/quote context deterministically attempts residual lookup for Kazusa-authored assistant rows and omits the field when data is unavailable.
- Direct reply/quote residual lookup does not depend on visible reply metadata being incomplete.
- Conversation-evidence context deterministically attempts residual lookup for already-retrieved Kazusa-authored rows without placing residual in public RAG evidence.
- Resolver observations do not contain `past_dialog_cognition_context`.
- Conversation-evidence row resolution uses `conversation_row_id` or `_id`; it does not query `platform_message_id` without platform/channel scope.
- L2a receives bounded compact residual context when available.
- L1, L2b, L2c1, L2c2, L2d, L3, dialog, RAG finalizer, consolidation, scheduler, reflection, adapters, and message-envelope projection do not receive residual context.
- The implementation reads only selected `parsed_output` fields from selected trace stages.
- No new LLM call, keyword gate, semantic fetch classifier, retry loop, durable store, backfill, or trace-capture configuration change exists.
- All verification commands pass.
- Independent code review is complete and recorded in `Execution Evidence`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Residual leaks into public prompt payloads or dialog output. | Dedicated field, L2a-only wiring, forbidden public carriers. | Static greps and consumer-boundary tests. |
| Implementation adds brittle keyword matching over user text. | Structural trigger only after existing context attachment. | Code review scans for regex, keywords, and semantic gates. |
| Metadata-mode traces are mistaken for usable residual. | Empty `parsed_output` means no residual. | Focused projector test for metadata-mode step. |
| Raw trace prompts or responses enter L2a. | DB projection excludes raw fields and projector reads parsed fields only. | Focused test and static grep for `raw_messages|raw_response_text`. |
| RAG becomes responsible for private cognition. | Residual side channel is built after RAG evidence and outside public `rag_result`. | RAG integration leak tests and static grep under `rag`. |
| Residual is stored in resolver observations. | Attach residual in `cognition_resolver/loop.py` after `rag_result` state update and keep `ResolverObservationV1` unchanged. | Static grep blocks `past_dialog_cognition_context` in `cognition_resolver/contracts.py` and resolver-state projection tests. |
| Conversation-evidence candidate lookup selects the wrong row. | Reload by row id only; skip platform-message-id-only refs. | RAG integration tests cover ambiguous/missing row id refs. |
| Context grows too large for local LLM. | Max three dialogs and `1800` character cap. | Projector cap test and prompt-render check. |

## Execution Evidence

- Plan drafting:
  - Created while plan status is `draft`; no production code changes are authorized by this draft.
  - Registry row added under `development_plans/active/short_term/`.
  - Execution later explicitly authorized by user request on 2026-06-29; lifecycle status remains `draft`.
- Independent plan review:
  - Manual no-subagent review completed.
  - Blockers addressed: wrong RAG projection path, vague resolver integration point, unsafe resolver-observation carrier, under-specified candidate author fields, unsafe platform-message-id-only evidence lookup, and direct-reply row lookup dependency on incomplete visible metadata.
  - Non-blocking finding addressed: targeted static grep added for the real RAG projection file.
- Implementation evidence:
  - Parent established failing focused tests first: `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py -q` failed with missing `kazusa_ai_chatbot.past_dialog_cognition`.
  - Production-code subagent `Hypatia` completed one production pass with concern that static greps/final status were not yet run.
  - Parent added integration and prompt-boundary tests, added runtime RAG helper consolidation, fixed empty-context L2a omission, and updated docs.
  - Independent code review finding fixed: selected text-surface input no longer carries `past_dialog_cognition_context`; `persona_supervisor2_l3_surface.py`, `cognition_chain_core/surface.py`, and `cognition_chain_core/stages/l3.py` now have zero matches.
  - Independent code review finding fixed: residual DB failures now catch `PyMongoError`; broad `except Exception` was removed from `past_dialog_cognition` and the resolver residual attachment path.
- Static grep results:
  - `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l3.py src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\rag`: zero matches.
  - `rg "raw_messages|raw_response_text" src\kazusa_ai_chatbot\past_dialog_cognition`: zero matches.
  - `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_resolver`: matches only `loop.py`.
  - `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_resolver\contracts.py`: zero matches.
  - `rg "past_dialog_cognition" src\kazusa_ai_chatbot\message_envelope src\kazusa_ai_chatbot\calendar_scheduler src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\dispatcher src\kazusa_ai_chatbot\proactive_output src\kazusa_ai_chatbot\brain_service`: zero matches.
  - `rg "past_dialog_cognition" src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py src\kazusa_ai_chatbot\rag`: zero matches.
  - `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\cognition_chain_core\stages src\kazusa_ai_chatbot\nodes`: allowed matches only in L2a prompt/payload handling, persona cognition input/state plumbing, persona state schema, and `nodes/README.md`.
  - `rg "past_dialog_cognition_context" src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\cognition_chain_core\surface.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l3.py`: zero matches.
  - `rg -n "except Exception as exc" src\kazusa_ai_chatbot\past_dialog_cognition src\kazusa_ai_chatbot\cognition_resolver\loop.py`: zero matches.
- Test results:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\db\llm_tracing.py src\kazusa_ai_chatbot\db\conversation.py src\kazusa_ai_chatbot\past_dialog_cognition\__init__.py src\kazusa_ai_chatbot\past_dialog_cognition\models.py src\kazusa_ai_chatbot\past_dialog_cognition\projection.py src\kazusa_ai_chatbot\past_dialog_cognition\runtime.py src\kazusa_ai_chatbot\cognition_chain_core\contracts.py src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py`: passed.
  - `venv\Scripts\python -m pytest tests\test_past_dialog_cognition_context.py tests\test_past_dialog_cognition_reply_integration.py tests\test_past_dialog_cognition_rag_integration.py tests\test_past_dialog_cognition_prompt_boundaries.py -q`: 18 passed.
  - `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q`: 4 passed.
  - `venv\Scripts\python -m pytest tests\test_cognition_chain_core_contracts.py tests\test_cognition_chain_connector_mapping.py -q`: 15 passed.
  - `venv\Scripts\python -m pytest tests\test_rag_projection.py -q`: 35 passed.
  - `venv\Scripts\python -m pytest tests\test_cognition_resolver_loop.py -q`: 33 passed.
  - Combined rerun after review fixes: `venv\Scripts\python -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py tests\test_rag_projection.py tests\test_cognition_resolver_loop.py -q`: 72 passed.
- Real LLM comparison evidence:
  - Branch commit before comparison: `d301586 Add past dialog cognition residual context` on `feature/past-dialog-cognition-residual`.
  - Runtime data check found retained `llm_trace_steps` had zero non-empty `parsed_output` rows across stored stages; the comparison therefore reused real visible Kazusa assistant dialog rows and seeded synthetic parsed cognition residuals in a test-only `debug` channel.
  - Seed and comparison run id: `pdc_live_20260629T115757Z_9fc284db`.
  - Seed metadata: `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_seed_metadata.json`.
  - Human-readable review: `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_review.md`.
  - Structured comparison data: `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_comparison_data.json`.
  - Feature branch live runs:
    - `case_1_reddit_strategy`: completed; response artifact `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_feature_case_1_reddit_strategy_response.json`; trace export `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_feature_case_1_trace_export.json`.
    - `case_2_emotional_texture`: completed; response artifact `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_feature_case_2_emotional_texture_response.json`; trace export `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_feature_case_2_trace_export.json`.
  - Main baseline live runs:
    - `case_1_reddit_strategy`: completed; response artifact `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_main_case_1_reddit_strategy_response.json`; trace export `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_main_case_1_trace_export.json`.
    - `case_2_emotional_texture`: completed; response artifact `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_main_case_2_emotional_texture_response.json`; trace export `test_artifacts/past_dialog_cognition_live/pdc_live_20260629T115757Z_9fc284db_main_case_2_trace_export.json`.
  - One first-attempt feature request failed before model output because `local_timestamp` used ISO UTC instead of configured-local format; retry with empty `local_timestamp` completed successfully and is the recorded comparison output.
  - The live comparison used `debug_modes.no_remember=true`; normal conversation rows were still created, but background consolidation and durable memory writes were suppressed.
  - Qualitative read recorded in the review: case 2 improved because the branch output answered the concrete "哪里不对" question from residual context; case 1 was mixed because the branch output followed residual context more closely but ended with a sharper final line.
- Independent code review:
  - Review subagent `Lagrange` completed review.
  - Findings: selected L3 text-surface contract carried nested `past_dialog_cognition_context`; broad exception handlers could mask non-DB bugs.
  - Fixes: text-surface chain input is sanitized by `build_text_surface_chain_input_from_global_state`; residual DB-read degradation catches `PyMongoError` only.
  - Affected verification rerun and passed as listed above. Residual risk: this remains best-effort and depends on full-capture trace availability.
