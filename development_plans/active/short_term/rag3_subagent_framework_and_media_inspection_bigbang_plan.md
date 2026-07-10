# rag3 subagent framework and media inspection bigbang plan

## Summary

- Goal: Convert RAG3 local-context source handling into a first-class
  resolver-local subagent framework, convert existing source hydration into
  subagents, and add session-media inspection through the same RAG3 subagent
  path while exposing the shared media inspector to complex-task resolver
  media intake.
- Plan class: high_risk_migration
- Status: in_progress
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang. Replace the current fixed
  `source_hydration.py` bridge with resolver-local subagent discovery and
  dispatch in one final production shape. Do not keep a production compatibility
  shim, dual source-hydration path, or fallback to the old bridge.
- Highest-risk areas: live response-path local evidence, RAG3 prompt and graph
  contracts, L2d reachability, local-model latency, media privacy, raw binary
  handling, cache freshness, complex-resolver external media safety, and
  preserving cognition/dialog ownership.
- Acceptance criteria: RAG3 production local-context recall routes every
  source-backed node kind through the new subagent registry; existing memory,
  scoped-memory, conversation, person, recall, live-context, and supplied
  external evidence behavior is preserved through converted subagents; current
  and recent session media can be inspected on demand through a RAG3 media
  subagent; complex-task resolver has a bounded media-intake subagent that
  calls the same shared media inspector for externally fetched media; the
  generic media descriptor behavior is unchanged; no raw media is stored in
  MongoDB by this plan; deterministic, live-LLM, static, documentation, and
  independent-review gates pass before completion.

## Context

Current RAG3 is the production local-context recall path. Its implemented
shape is:

```text
local_context_recall
  -> resolve_local_context(request, context, options=None)
  -> graph planner
  -> active-node traversal
  -> fixed source_hydration.py bridge for selected source-backed node kinds
  -> active-node LLM
  -> packet synthesis
  -> retained rag_result projection
```

The current bridge is static. It maps supported node kinds to existing
source-owned helpers for memory, scoped memory, conversation, person, and
recall evidence. The local-context resolver contracts already define future
`LocalContextSubagentV1` request/result protocols, but there is no registry,
module discovery, or runtime dispatch equivalent to
`complex_task_resolver.subagent`.

The user-approved product direction is broader than a media-only patch:

- conversation media is local conversation context and belongs under RAG3;
- RAG3 should gain the same family-local subagent extension capability as the
  complex-task resolver;
- existing RAG3 source handlers must be converted to subagents, not bypassed
  by a media special case;
- the current generic media descriptor remains unchanged;
- raw media stays process/session cached and is not added to durable database
  storage;
- cache miss is a normal evidence gap;
- cognition may use RAG3 media evidence first and then select
  `public_answer_research` in a later resolver cycle when public/external
  search is needed;
- complex-task resolver also needs a path to inspect externally fetched media,
  using the same lower-level media inspector without routing public research
  through RAG3.

This plan treats the RAG3 subagent conversion as final production architecture.
It preserves the RAG3 public IO and prompt-facing `rag_result` projection while
rewriting the internal source-dispatch boundary.

Adjacent improvement areas intentionally left for later plans:

- video frame extraction, audio transcription, and non-image media inspection;
- browser-driven image search, reverse image search, or JavaScript-rendered
  media acquisition;
- persistent raw-media storage, blob collections, or object-storage integration;
- new L2d resolver capability kinds for direct media inspection.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  lifecycle-updating, or signing off this plan.
- `local-llm-architecture`: load before changing RAG3 graph, subagent,
  prompt, media-inspection, complex-resolver, cognition, or LLM-call behavior.
- `debug-llm`: load before creating or judging live LLM review artifacts for
  RAG3, media inspection, or complex resolver behavior.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files or tests containing CJK string
  literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution. If
  native subagents are unavailable, stop unless the user explicitly approves
  fallback execution.
- Preserve the character-brain boundary: RAG3 and complex-task resolver return
  evidence; cognition decides stance, whether more evidence is needed, whether
  to speak, and how evidence affects character judgment; dialog owns final
  visible wording.
- Preserve the public RAG3 entrypoint:
  `resolve_local_context(request, context, options=None)`. This plan may extend
  validated context, node-kind, artifact, subagent, and trace contracts, but it
  must not create a second public local-context resolver entrypoint.
- Preserve the prompt-facing `rag_result` surface consumed by cognition unless
  this plan explicitly extends it with sanitized media evidence rows.
- Convert every source-backed RAG3 node kind to resolver-local subagent
  dispatch. Do not leave memory, scoped-memory, conversation, person, recall,
  live-context, supplied-external, or media source handling as one-off branches
  in `service.py`.
- The generic media descriptor prompt, descriptor result schema, descriptor
  persistent cache semantics, and conversation attachment description update
  behavior are out of scope for behavioral changes. Cache population around
  the descriptor path is allowed only when the descriptor's visible output and
  prompt stay unchanged.
- Do not store raw media bytes or base64 payloads in MongoDB as part of this
  plan. Existing config-gated inline attachment persistence behavior remains
  unchanged.
- Raw media payloads, cache keys, content hashes, platform ids, database ids,
  graph ids, prompt text, stage traces, and external fetch internals must not
  enter RAG3 planner prompts, cognition-visible observations, `rag_result`,
  dialog prompts, or final user-facing text.
- Deterministic code owns media cache insertion, alias binding, byte caps,
  MIME checks, URL fetch validation, cache eviction, cache miss translation,
  timeout handling, schema validation, and prompt-safe projection.
- LLM stages own semantic graph planning, active-node semantic evidence
  judgment, media visual observation from supplied image media, and packet
  synthesis. LLM stages must not decide persistence, routing, adapter delivery,
  cache eviction, retry policy, or external fetch permission.
- RAG3 must not call complex-task resolver directly. Cross-hop behavior runs
  through cognition: RAG3 returns media/local evidence, cognition decides
  whether `public_answer_research` is needed in a later resolver cycle.
- Complex-task resolver may call the shared media inspector for externally
  fetched media through its own resolver-local media subagent. It must not
  route public media inspection through RAG3.
- Live LLM tests must run one case at a time with output inspected. Do not run
  live LLM cases in batches.
- Real LLM test coverage must include 2-5 realistic inputs for each new
  LLM-backed node, including at least one normal exact-answer case and one
  ambiguous or unsupported boundary case when relevant. Passing pytest status
  is harness evidence only; the parent agent must inspect output quality.
- Every live LLM run must produce an agent-authored Markdown review artifact
  from raw evidence. Tests and scripts may emit JSON, logs, traces, and model
  outputs, but they must not emit the human quality report.
- New or edited Python code must follow `py-style`: imports at module top,
  narrow `try` blocks, useful docstrings for non-trivial functions, required
  internal data read by indexing, named computed return values, no hidden
  broad exception handling, no pass-through helpers, no duplicate helper
  behavior, and no speculative flags, modes, or optional contract fields.
- Runtime prompt constants must stay adjacent to their LLM instance and
  handler, use triple-single-quoted strings, use `.format(...)` with named
  placeholders for stable template rendering, keep dynamic facts in the human
  message, avoid hard-coded character names, and expose explicit output
  contracts.
- Python files or tests containing CJK strings must use single-quoted string
  delimiters for risky CJK quotation content, read and write UTF-8 explicitly,
  reconfigure Windows stdout/stderr before printing CJK model output, and run
  syntax verification immediately after edits.

## Must Do

- Add a RAG3 resolver-local subagent package and registry under
  `kazusa_ai_chatbot.local_context_resolver.subagent`.
- Align RAG3 subagent registration metadata with the complex-task resolver
  pattern: `SUBAGENT`, `DESCRIPTION`, `SUPPORTED_ACTIONS`,
  `OWNED_NODE_KINDS`, `DEFAULT_ACTION`, `create`, and optional `is_enabled`.
- Convert existing static source hydration behavior into RAG3 subagents:
  memory, conversation, person, recall, live-context, and supplied-external
  source handling.
- Add a RAG3 media subagent that owns current and recent session media node
  kinds and calls the shared media inspector.
- Update the `local_context_recall` semantic affordance text in L2d/action
  selection so local conversation media is reachable without adding a new
  capability kind.
- Remove production use of `source_hydration.py` after converted subagents are
  wired and verified. Delete the module and replace its focused tests when no
  production or test import remains.
- Add a shared media-inspection package with a bounded public service contract
  usable by RAG3 and complex-task resolver.
- Add a process-local session media cache with a fixed rolling item cap and a
  byte cap. The first approved defaults are
  `MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE = 8`,
  `MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE = 16 * 1024 * 1024`,
  `MEDIA_SESSION_CACHE_MAX_ITEM_BYTES = 6 * 1024 * 1024`, and
  `MEDIA_SESSION_CACHE_TTL_SECONDS = 900`.
- Populate the session media cache from current-turn inbound media with bytes
  available in service intake. Preserve existing descriptor behavior and
  existing durable storage policy.
- Extend RAG3 context projection with prompt-safe session-media aliases and
  trusted lookup refs that are stripped before LLM prompts.
- Add exact RAG3 node kinds, media selector shape, `media_ref` artifact type,
  `media_evidence` rag-result rows, and validation while preserving existing
  node, graph, packet, and projection validation.
- Add a complex-task resolver media subagent for externally fetched image
  inspection through the shared media inspector.
- Add deterministic contract tests before production implementation, then
  implement production code to satisfy those tests.
- Add selected one-at-a-time live LLM review cases for RAG3 media inspection,
  RAG3 preserved source behavior, and complex-task media inspection.
- Update README, HOWTO, local-context resolver ICD, complex-task resolver ICD,
  message-envelope or media docs, and the development plan registry.

## Deferred

- Do not add a new L2d-visible direct media capability.
- Do not replace or retune the current generic media descriptor prompt.
- Do not add persistent raw media storage, object storage, blob collections, or
  a database migration for media bytes.
- Do not add dual RAG3 source-dispatch paths or compatibility shims around
  `source_hydration.py`.
- Do not merge RAG3 and complex-task resolver into one universal runtime
  package or shared registry.
- Do not add reverse image search, browser automation, OCR beyond the selected
  media model's answer, JavaScript execution, CAPTCHA handling, or image search
  providers.
- Do not implement video frame sampling, audio transcription, non-image media
  inspection, generic OCR extraction, or object-storage media retrieval in this
  plan.
- Do not increase cognition resolver max cycles, RAG3 traversal caps, or
  capability timeouts to compensate for weak media inspection.
- Do not let RAG3 perform public web search as part of local media inspection.
  Public or current external research remains a complex-task resolver concern.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| RAG3 subagent framework | bigbang | Introduce the registry and make it the only production source-dispatch path for source-backed RAG3 nodes. |
| Static `source_hydration.py` bridge | bigbang | Replace it with subagents and remove production imports. Delete the module after references are gone. |
| Existing RAG3 public IO | compatible retained surface | Preserve `resolve_local_context(request, context, options=None)` and `LocalContextResolutionPacketV1`. |
| Existing `rag_result` projection | compatible retained surface | Preserve existing fields and add only sanitized media evidence projection required by this plan. |
| Existing source helper semantics | migration | Move current memory, conversation, person, and recall helper usage behind subagents without changing source meaning. |
| Generic media descriptor | compatible retained behavior | Preserve descriptor prompt, result shape, persistent descriptor cache, and attachment-description update behavior. |
| Session media cache | bigbang new process-local source | Add process-local rolling cache for raw current/recent media. No database persistence. |
| Shared media inspector | bigbang new lower-level service | Add one media-inspection service callable by RAG3 and complex-task resolver. |
| Complex-task resolver media intake | compatible extension | Add a bounded media subagent without changing existing evidence or algorithmic subagent behavior. |
| Tests | bigbang replacement | Replace source-hydration tests with subagent registry, dispatch, projection, cache miss, media, and preserved-behavior tests. |
| Documentation | bigbang update | Present RAG3 as subagent-dispatched production local context after this plan completes. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative dual-path strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of
  preserving them.
- If an area is `migration`, follow the exact implementation and cleanup gates
  in this plan.
- If an area is `compatible retained surface`, preserve only the named public
  surface and rewrite private internals as needed.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Completed RAG3 local-context architecture:

```text
local_context_recall
  -> resolve_local_context(request, context, options=None)
  -> graph planner
  -> active-node traversal
  -> local_context_resolver.subagent registry
       -> memory subagent
       -> conversation subagent
       -> person subagent
       -> recall subagent
       -> live-context subagent
       -> supplied-external subagent
       -> media subagent
  -> active-node LLM with prompt-safe subagent source context
  -> packet synthesis
  -> retained rag_result projection
  -> next cognition resolver cycle
```

Session media inspection path:

```text
service intake receives inline media bytes
  -> process-local SessionMediaCache stores current/recent media under bounded caps
  -> generic descriptor still runs normally
  -> cognition selects local_context_recall when details are missing
  -> RAG3 planner creates current/recent media node
  -> RAG3 media subagent resolves prompt-safe media alias to cache entry
  -> shared media inspector answers exact question from the node objective
  -> media evidence is projected as local conversation evidence
  -> cognition decides whether answer, silence, clarification, or public research is needed
```

Complex external media path:

```text
public_answer_research
  -> complex-task resolver graph
  -> media_inspection_task node
  -> complex media subagent validates external media source and byte limits
  -> shared media inspector answers exact question
  -> complex knowledge packet returns public/source-bound media evidence
  -> cognition judges what the packet means
```

Cache miss behavior:

```text
RAG3 media subagent resolves alias
  -> cache hit: call shared media inspector
  -> cache miss: return unavailable result with bounded missing-evidence row
  -> RAG3 packet preserves evidence boundary
  -> cognition decides the visible or private consequence
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| RAG3 ownership | Conversation media belongs under RAG3 local context. | User-sent media is part of the conversation, not public research. |
| Subagent conversion scope | Convert all RAG3 source-backed source handling to subagents. | A media-only subagent would recreate a special-case bridge and block future extension. |
| Cutover strategy | Use a final bigbang internal rewrite while preserving RAG3 public IO. | The user explicitly wants the change final and expansion-ready without compatibility shims. |
| Registry shape | Mirror complex-task resolver subagent metadata while keeping a family-local package. | This gives parity without merging local/private and public/external resolver families. |
| L2d reachability | Keep the same capability names but update `local_context_recall` semantic affordance text to mention local conversation media. | Without this wording the feature is technically present but the resolver loop may not select it. Specialists still own media mechanics. |
| Cross-hop behavior | RAG3 and complex resolver do not call each other directly. | Cognition owns the decision to seek public research after local media evidence. |
| Shared media inspector | Add one lower-level inspector service used by both resolver families. | Perceptual question answering is shared capability; source ownership stays family-local. |
| Descriptor behavior | Preserve the existing generic descriptor unchanged. | The current descriptor works for most cases and remains the baseline perceptual summary. |
| Raw media cache | Use process-local session cache only. | The product requirement rejects database storage for raw media. |
| Cache size | Start with 8 media items and 16 MiB total per process/session scope. | This covers several recent turns while bounding memory and privacy exposure. |
| Media support phase | Implement image-only inspection in this plan. | Current ingestion reliably carries image bytes; audio and video need separate extraction contracts and are deferred. |
| Identity questions | Require supplied visual anchors or answer with uncertainty. | A question like "is this the active character?" cannot be confirmed from image bytes alone. |
| Complex media intake | Add a bounded complex media subagent for externally fetched image bytes. | This covers public/media research without routing external work through RAG3. |

## Contracts And Data Shapes

### RAG3 Subagent Registry

Create:

```text
src/kazusa_ai_chatbot/local_context_resolver/subagent/
  __init__.py
  memory.py
  conversation.py
  person.py
  recall.py
  live_context.py
  supplied_external.py
  media.py
```

Each module must declare:

```python
SUBAGENT: str
DESCRIPTION: str
SUPPORTED_ACTIONS: tuple[str, ...]
OWNED_NODE_KINDS: tuple[str, ...]
DEFAULT_ACTION: str
def create() -> LocalContextSubagentV1: ...
```

Optional:

```python
def is_enabled() -> bool: ...
```

Registry exports:

```python
create_subagents() -> dict[str, LocalContextSubagentV1]
subagent_for_node_kind(node_kind: str) -> str | None
subagent_capability_descriptions() -> list[dict[str, object]]
```

The registry validates duplicate subagent names, duplicate node-kind
ownership, missing default actions, unsupported default actions, and disabled
subagents.

### Converted RAG3 Subagents

| Subagent | Owned node kinds | Action | Source behavior |
|---|---|---|---|
| `memory` | `memory_evidence`, `scoped_memory` | `collect_memory_evidence` | Existing memory helper behavior, including current-user scoped memory projection. |
| `conversation` | `conversation_evidence` | `collect_conversation_evidence` | Existing conversation helper behavior, including active-turn exclusion and prompt-safe excerpts. |
| `person` | `person_context` | `collect_person_context` | Existing person/profile helper behavior. |
| `recall` | `recall_evidence` | `collect_recall_evidence` | Existing recall helper behavior for commitments, agreements, plans, and episode state. |
| `live_context` | `live_context` | `collect_live_context` | Deterministic local-time/current-context projection from resolver context. |
| `supplied_external` | `external_evidence` | `collect_supplied_external_evidence` | Project caller-supplied URL/public-content evidence already present in context; no web search. |
| `media` | `current_turn_media`, `recent_media` | `inspect_media` | Resolve prompt-safe media alias against session cache and call shared media inspector. |

`subtask` and `synthesis` remain graph/stage-owned structural node kinds. They
are not source-backed subagent node kinds.

### LocalContextSubagentRequestV1

Preserve the existing request envelope and add only dependency context:

```python
{
    "schema_version": "local_context_subagent_request.v1",
    "node_id": str,
    "subagent": str,
    "action": str,
    "objective": str,
    "payload": dict,
    "constraints": dict,
    "dependency_context": list[dict],
}
```

The model-facing active-node prompts receive semantic source rows, not this raw
subagent request envelope.

### LocalContextSubagentResultV1

Preserve the existing result envelope:

```python
{
    "schema_version": "local_context_subagent_result.v1",
    "resolved": bool,
    "status": "resolved|partial|invalid|unavailable|failed",
    "result": {
        "source_records": list[dict],
        "artifacts": list[LocalContextArtifactV1],
        "node_update": dict,
    },
    "attempts": int,
    "cache": dict,
    "trace": dict | list[str],
    "unresolved_items": list[str],
}
```

Status invariants:

- `resolved`: `resolved=True`, at least one useful source row or artifact, and
  no blocking unresolved item.
- `partial`: `resolved=True`, useful evidence exists, and uncertainty or
  missing subparts are recorded.
- `invalid`: request payload, media selector, URL, MIME, or schema validation
  failed before semantic inspection.
- `unavailable`: source exists but cannot be read now, including session-media
  cache miss.
- `failed`: a permitted external dependency such as the image model or bounded
  URL fetch failed after validation.

Every result is validated before merge. Allowed `node_update` fields are only
`investigation_summary`, `knowledge_we_know_so_far`,
`knowledge_still_lacking`, `recommended_next_iteration`,
`evidence_boundary_notes`, `status`, and `source_observation_ids`. Prompt-facing
rows are sanitized after subagent return and again during final `rag_result`
projection.

### RAG3 Planner Media Selector

Add two node kinds:

```text
current_turn_media
recent_media
```

Media nodes may carry this deterministic selector in their node payload:

```python
{
    "schema_version": "local_context_media_selector.v1",
    "selector_kind": "current|recent|alias",
    "alias": str | None,
    "ordinal": int | None,
    "question": str,
}
```

The planner owns only the semantic choice that local media evidence is needed
and the exact visual question to ask. Deterministic code resolves
`selector_kind`, `alias`, and `ordinal` against prompt-safe session media refs.
If the selector is ambiguous or no matching cache ref exists, the media
subagent returns `status="unavailable"` with `reason="cache_miss"` in
`result["node_update"]["evidence_boundary_notes"]` and does not call the media
inspector.

### RAG3 Media Context

Extend `LocalContextResolverContextV1` with optional trusted media refs:

```python
{
    "session_media_refs": [
        {
            "alias": "current_media_1",
            "media_kind": "image",
            "content_type": "image/png",
            "description": "generic descriptor summary if available",
            "recency": "current_turn|recent_turn",
            "turn_relation": "active_turn|recent_cached_turn",
            "source_summary": "current message attachment",
            "cache_ref": "process-local lookup ref",
            "descriptor_cache_key": "trusted descriptor cache key if present"
        }
    ]
}
```

Prompt compaction exposes `alias`, `media_kind`, `content_type`,
`description`, `recency`, `turn_relation`, and `source_summary`. Prompt
compaction strips `cache_ref`, `descriptor_cache_key`, source message ids,
content hashes, raw bytes, base64, and database ids.

Alias generation:

- Current active-turn image items become `current_media_1`,
  `current_media_2`, in stable intake order.
- Recent cached image items from earlier turns become `recent_media_1`,
  `recent_media_2`, newest first.
- Collapsed history, persisted descriptors, and rows whose raw bytes are absent
  may be listed only as descriptor context; they must not produce trusted
  `cache_ref` values for inspection.

### Session Media Cache

Create a process-local cache under the shared media-inspection package:

```python
put_session_media(scope, media_items) -> list[SessionMediaRefV1]
list_session_media_refs(scope) -> list[SessionMediaRefV1]
get_session_media(scope, cache_ref) -> SessionMediaPayloadV1 | None
clear_session_media(scope=None) -> None
```

`scope` is the exact tuple `(platform, platform_channel_id, global_user_id)`.
`get_session_media` must reject refs whose stored scope does not exactly match
the caller scope.

`SessionMediaPayloadV1` is trusted runtime data and must not be prompt-facing:

```python
{
    "cache_ref": str,
    "scope": tuple[str, str, str],
    "media_kind": "image",
    "content_type": str,
    "base64_data": str,
    "size_bytes": int,
    "content_hash": str,
    "created_at_monotonic": float,
    "last_seen_monotonic": float,
    "description": str,
}
```

Eviction policy:

- keep at most `MEDIA_SESSION_CACHE_MAX_ITEMS_PER_SCOPE = 8` image items per
  active session scope;
- keep cached bytes at or below
  `MEDIA_SESSION_CACHE_MAX_BYTES_PER_SCOPE = 16 * 1024 * 1024` per scope;
- reject any single image payload over `MEDIA_SESSION_CACHE_MAX_ITEM_BYTES =
  6 * 1024 * 1024`;
- expire entries older than `MEDIA_SESSION_CACHE_TTL_SECONDS = 900`;
- evict oldest entries first inside the matching scope;
- never persist payloads to MongoDB;
- never expose `cache_ref`, `content_hash`, or `base64_data` to prompts.

### Shared Media Inspector

Create package:

```text
src/kazusa_ai_chatbot/media_inspection/
  __init__.py
  contracts.py
  service.py
  session_cache.py
  image.py
  README.md
```

Public service contract:

```python
async def inspect_media(
    request: MediaInspectionRequestV1,
) -> MediaInspectionResultV1:
    ...
```

Request:

```python
{
    "schema_version": "media_inspection_request.v1",
    "source": "rag3_session_media|complex_external_media|test|live_llm_review",
    "media_kind": "image",
    "content_type": str,
    "base64_data": str,
    "question": str,
    "existing_descriptor": str,
}
```

Result:

```python
{
    "schema_version": "media_inspection_result.v1",
    "status": "answered|uncertain|unsupported|invalid_input|failed",
    "answer": str,
    "supporting_facts": list[str],
    "contradicting_facts": list[str],
    "uncertainty_notes": list[str],
    "evidence_boundary_notes": list[str],
}
```

Image inspection uses exactly one model call through `VISION_DESCRIPTOR_LLM`
with multimodal `image_url` payload data and the media-inspection prompt, not
the generic descriptor prompt. The stable system prompt defines the output
contract, uncertainty rules, instruction-boundary rule, and image-only scope.
The dynamic human payload contains the question and optional existing generic
descriptor. OCR-like text visible inside the image is evidence, not an
instruction channel. The inspector must not log raw base64, prompt text with
embedded base64, content hashes, cache refs, or platform ids.

Caller-owned translations:

- RAG3 media subagent translates session cache miss to
  `LocalContextSubagentResultV1(status="unavailable")`.
- Complex media subagent translates fetch failure to
  `ComplexTaskSubagentResultV1(status="failed")`.
- The shared inspector returns only image-inspection statuses listed in its
  result contract.

### Complex Resolver Media Intake

Add:

```text
src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py
```

Owned node kind:

```text
media_inspection_task
```

Supported action:

```text
inspect_media
```

The complex media subagent accepts only media bytes or directly fetchable media
URLs supplied by the complex resolver context or prior evidence artifacts. URL
fetching must be deterministic and isolated from prompt payloads:

- accept only `http` and `https` URLs;
- reject URLs with credentials, fragments used as data, unsupported schemes, or
  missing host;
- resolve DNS before connection and reject loopback, private, link-local,
  multicast, reserved, and local interface addresses;
- revalidate scheme, host, DNS/IP class, content type, and byte cap on every
  redirect;
- cap redirects, connection timeout, read timeout, decoded byte length, and
  final image dimensions;
- require image MIME and image magic/decoder validation before inspection;
- record only source URL, MIME, byte count, and validation status in
  source-bound evidence.

The complex media subagent calls the shared media inspector and returns the
existing `ComplexTaskSubagentResultV1` envelope. It must not add raw bytes to
`available_evidence`, planner prompts, or compact context.

## LLM Call And Context Budget

Default context-window cap: 50k tokens.

Before this plan:

- RAG3 normal single-source path: planner LLM plus active-node LLM, with
  optional static source hydration on active-node cache miss.
- RAG3 collapse review and bottom-up synthesis LLM calls run only when the
  graph state requires them.
- Generic media descriptor runs before relevance for current-turn media.
- Existing source specialists may call nested LLM helpers for memory, conversation, person, or recall evidence; those nested calls remain part of the response-path budget when selected.
- Complex-task resolver uses its existing planner, node resolver, subagent,
  collapse, and synthesis calls.

After this plan:

- RAG3 normal non-media single-source path keeps the same LLM count: planner
  LLM plus active-node LLM on active-node cache miss, plus any existing nested
  helper LLM calls already used by the selected source specialist. Static
  source hydration is replaced by subagent dispatch with the same source helper
  behavior.
- RAG3 media path adds one media-inspection LLM call only when a media subagent
  runs and the session cache hits. Cache miss adds no media-inspection LLM
  call.
- RAG3 planner prompt receives a compact semantic capability roster for
  source-backed subagents. The roster must stay short and static.
- RAG3 active-node prompt receives sanitized `source_records` from the selected
  subagent. It must not receive raw bytes, cache refs, hashes, ids, fetch
  internals, or stage traces.
- Complex media path adds one media-inspection LLM call when the complex media
  subagent has validated external media bytes.
- Generic descriptor call count remains unchanged.

Hard caps:

- Session media inspection: at most one media-inspection LLM call per media
  subagent run.
- RAG3 active-node Cache2 must not reuse stale media-inspection output across
  different image content. Implement either no active-node cache store for
  `current_turn_media` and `recent_media`, or include trusted content identity
  in the cache key while keeping that identity out of prompts and traces.
- Bump `RAG3_CACHE_POLICY_VERSION` when removing `source_hydration_enabled`
  from context and cache keys.
- RAG3 traversal caps remain at current resolver option hard caps unless
  existing code already enforces lower values.
- No JSON-repair LLM call is added for RAG3 subagent, media inspector, or
  complex media output.
- Live response-path timeout remains governed by the existing cognition
  resolver capability timeout.

Verification must record stage-call counts for selected live LLM cases and
confirm cache-miss media cases skip the media-inspection LLM, selected source
specialist cases preserve nested-call behavior, and media cache keys cannot
reuse evidence for different image content.

## Change Surface

### Delete

- `src/kazusa_ai_chatbot/local_context_resolver/source_hydration.py` after the
  converted subagent dispatch path has replacement tests and all references are
  removed.
- `tests/test_local_context_resolver_source_hydration.py` after equivalent
  subagent coverage exists.

### Modify

- `src/kazusa_ai_chatbot/local_context_resolver/contracts.py`: extend
  subagent, node-kind, artifact, context, validator, and projection contracts.
- `src/kazusa_ai_chatbot/local_context_resolver/service.py`: replace fixed
  source-hydration call sites with registry subagent dispatch, result merge,
  trace counters, and cache interactions.
- `src/kazusa_ai_chatbot/local_context_resolver/stages.py`: update planner and
  active-node prompt contracts for subagent capability roster and media node
  semantics while keeping prompt vocabulary short and semantic.
- `src/kazusa_ai_chatbot/local_context_resolver/cache.py`: remove
  `source_hydration_enabled` from cache key policy, bump the policy version,
  and either bypass active-node cache storage for media nodes or include trusted
  media content identity outside prompt-facing context.
- `src/kazusa_ai_chatbot/local_context_resolver/__init__.py`: export only
  stable public IO and approved validators.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py`: update the
  `local_context_recall` affordance text so L2d can select local recall for
  current/recent conversation image details.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`: pass prompt-safe
  session-media refs and trusted lookup refs into RAG3 context when present in
  persona state.
- `src/kazusa_ai_chatbot/service.py`: populate the process-local session media
  cache from inbound current-turn media bytes during intake.
- `src/kazusa_ai_chatbot/config.py`: add bounded session-media cache constants.
- `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py`: add
  `media_inspection_task` node-kind support and related validation.
- `src/kazusa_ai_chatbot/complex_task_resolver/service.py`: dispatch complex
  media-inspection nodes through the complex media subagent and merge bounded
  results.
- `src/kazusa_ai_chatbot/complex_task_resolver/stages.py`: update prompt
  contract for media-inspection task creation from external media evidence.
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/__init__.py`: discover
  and validate the media subagent.
- `tests/test_local_context_resolver_live_llm.py`: keep named source-behavior
  live gates available for one-at-a-time preserved behavior checks.
- `tests/test_complex_task_resolver_live_llm.py`: add named complex media live
  gates instead of relying on a broad `-k` selector.
- `README.md`, `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/local_context_resolver/README.md`,
  `src/kazusa_ai_chatbot/complex_task_resolver/README.md`,
  `src/kazusa_ai_chatbot/message_envelope/README.md`, and
  `development_plans/README.md`: document the new boundaries and plan
  lifecycle state.

### Create

- `src/kazusa_ai_chatbot/local_context_resolver/subagent/__init__.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/memory.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/conversation.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/person.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/recall.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/live_context.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/supplied_external.py`
- `src/kazusa_ai_chatbot/local_context_resolver/subagent/media.py`
- `src/kazusa_ai_chatbot/media_inspection/__init__.py`
- `src/kazusa_ai_chatbot/media_inspection/contracts.py`
- `src/kazusa_ai_chatbot/media_inspection/service.py`
- `src/kazusa_ai_chatbot/media_inspection/session_cache.py`
- `src/kazusa_ai_chatbot/media_inspection/image.py`
- `src/kazusa_ai_chatbot/media_inspection/README.md`
- `src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py`
- `tests/test_local_context_resolver_subagent_registry.py`
- `tests/test_local_context_resolver_subagent_dispatch.py`
- `tests/test_local_context_resolver_media_subagent.py`
- `tests/test_media_inspection_contracts.py`
- `tests/test_media_session_cache.py`
- `tests/test_complex_task_resolver_media_subagent.py`
- `tests/test_action_selection_media_affordance.py`
- `tests/test_rag3_media_inspection_live_llm.py`
- `tests/test_media_inspection_live_llm.py`

### Keep

- Existing generic media descriptor prompt, persistent descriptor cache, and
  conversation attachment-description update behavior.
- Existing RAG3 public entrypoint and retained `rag_result` projection surface.
- Existing complex-task resolver public entrypoint.
- Existing message-envelope attachment contract and durable storage policy.
- Existing cognition resolver capability names:
  `local_context_recall` and `public_answer_research`.

## Overdesign Guardrail

- Actual problem: RAG3 needs extensible source-owned subagents and on-demand
  exact-question media inspection for recent conversation media without
  rewriting the generic descriptor or storing raw media durably.
- Minimal change: replace the fixed RAG3 source-hydration bridge with a
  resolver-local subagent registry, convert existing source handlers into that
  registry, add process-local media cache refs to RAG3 context, and implement
  one shared image-first media inspector used by RAG3 and complex resolver.
- Ownership boundaries: cognition chooses whether local or public evidence is
  needed; RAG3 owns local/private conversation evidence; complex-task resolver
  owns public/external research; shared media inspector owns perceptual
  exact-question answers; deterministic code owns cache, validation, caps,
  fetch checks, and projection; dialog owns final wording.
- Rejected complexity: no new direct L2d media capability, no durable raw-media
  storage, no universal RAG3/complex registry, no descriptor replacement, no
  reverse image search, no browser automation, no video/audio expansion beyond
  image-first contracts, no retry loops, no compatibility bridge to
  `source_hydration.py`.
- Evidence threshold: add deferred capabilities only after tests or live
  artifacts show the image-first shared inspector and subagent registry cannot
  satisfy approved near-term cases within current latency and safety caps.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside RAG3, shared media
  inspection, complex media intake, service media-cache population, tests, and
  listed docs as high-scrutiny changes.
- The responsible agent may remove old source-hydration code when greps and
  tests prove the converted subagent path covers the old behavior.
- If an existing helper already performs the needed source retrieval, the
  responsible agent must call or move it behind the relevant subagent instead
  of duplicating retrieval logic.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors unless explicitly
  listed in `Must Do`.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent adds baseline failing tests: `tests/test_local_context_resolver_subagent_registry.py`, `tests/test_local_context_resolver_subagent_dispatch.py`, `tests/test_media_inspection_contracts.py`, `tests/test_media_session_cache.py`, `tests/test_conversation_history_envelope.py`, `tests/test_local_context_resolver_media_subagent.py`, `tests/test_complex_task_resolver_media_subagent.py`, and `tests/test_action_selection_media_affordance.py`.
2. Parent starts one production-code subagent with the approved plan, focused test contract, and production ownership boundary.
3. Production-code subagent implements RAG3 subagent contracts, registry, and converted memory/conversation/person/recall/live/supplied-external subagents.
4. Production-code subagent rewires RAG3 active-node source dispatch through the registry and removes production imports of `source_hydration`.
5. Production-code subagent implements shared media-inspection contracts, session cache, image inspector, service intake population, and media-cache configuration constants without changing generic descriptor behavior.
6. Production-code subagent implements RAG3 media node kinds, selector validation, media subagent, media artifacts, projection, cache miss behavior, and context compaction.
7. Production-code subagent implements complex-task resolver media subagent, node kind, dispatch, and bounded external media validation.
8. Production-code subagent updates L2d/action-selection affordance text while preserving `local_context_recall` as the only local recall capability kind.
9. Parent runs focused deterministic tests, replaces old source-hydration tests, deletes `source_hydration.py` after `rg` proves no imports remain, then runs integration and regression tests listed in `Verification`.
10. Parent runs selected live LLM cases one at a time, authors review artifacts, updates documentation and registry rows, runs independent code review, remediates approved-scope findings, reruns affected verification, records evidence, and updates lifecycle status only after review approval.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused test contract established. Covers step 1. Verify each new focused test file and record expected failure or baseline. Evidence: commands and failures in `Execution Evidence`. Handoff: Stage 2. Sign-off: `parent/2026-07-10`.
- [x] Stage 2 - RAG3 subagent framework conversion complete. Covers steps 2-4. Verify `venv\Scripts\python -m pytest tests\test_local_context_resolver_subagent_registry.py tests\test_local_context_resolver_subagent_dispatch.py -q`. Evidence: changed files, removed source-hydration references, test output. Handoff: Stage 3. Sign-off: `parent/2026-07-10`.
- [x] Stage 3 - shared media inspection and session cache complete. Covers step 5. Verify `venv\Scripts\python -m pytest tests\test_media_inspection_contracts.py tests\test_media_session_cache.py tests\test_media_descriptor_cache.py tests\test_conversation_history_envelope.py -q`. Evidence: descriptor preservation, inline attachment policy, cache tests. Handoff: Stage 4. Sign-off: `parent/2026-07-10`.
- [x] Stage 4 - RAG3 media subagent integration complete. Covers step 6. Verify `venv\Scripts\python -m pytest tests\test_local_context_resolver_media_subagent.py tests\test_local_context_resolver_projection.py tests\test_local_context_resolver_integration.py -q`. Evidence: cache hit/miss, projection, sanitization. Handoff: Stage 5. Sign-off: `parent/2026-07-10`.
- [x] Stage 5 - complex resolver media intake complete. Covers step 7. Verify `venv\Scripts\python -m pytest tests\test_complex_task_resolver_media_subagent.py tests\test_complex_task_resolver_contracts.py tests\test_complex_task_resolver_service.py -q`. Evidence: dispatch, URL safety, inspector invocation. Handoff: Stage 6. Sign-off: `parent/2026-07-10`.
- [ ] Stage 6 - cleanup, static checks, L2d reachability, and deterministic regression complete. Covers steps 8-9. Verify all static greps and deterministic commands in `Verification`. Evidence: grep outputs, `py_compile`, diff check, pytest results. Handoff: Stage 7. Sign-off: `<agent/date>`.
- [ ] Stage 7 - live LLM review complete. Covers step 10 live gates. Verify each live LLM case one at a time with `-s -m live_llm` and inspect the generated artifact. Evidence: command, status, stage-call counts, artifact path, qualitative review. Handoff: Stage 8. Sign-off: `<agent/date>`.
- [ ] Stage 8 - documentation and lifecycle update complete. Covers step 10 docs gates. Verify `rg` checks in `Verification` show docs and the active registry row describe the final architecture. Evidence: updated docs and registry row. Handoff: Stage 9. Sign-off: `<agent/date>`.
- [ ] Stage 9 - independent code review and remediation complete. Covers step 10 review gate. Verify independent review approval and rerun affected tests after remediation. Evidence: review subagent id, findings, fixes, rerun commands, residual risks, approval status. Handoff: plan completion only after sign-off. Sign-off: `<agent/date>`.

## Verification

### Static Greps

- `rg "source_hydration|hydrate_source_for_node" src\kazusa_ai_chatbot tests`
  returns no production imports after cleanup. Allowed matches are archived
  completed plans and explanatory documentation that states the old bridge is
  retired. In tests, allowed matches are only assertions that the retired
  bridge is absent.
- `rg "source_hydration_enabled" src\kazusa_ai_chatbot tests`
  returns no matches after cleanup except archived or explanatory docs that
  state the flag was removed.
- `rg "base64_data|content_hash|cache_ref" src\kazusa_ai_chatbot\cognition_chain_core src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\cognition_resolver src\kazusa_ai_chatbot\local_context_resolver src\kazusa_ai_chatbot\complex_task_resolver src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\media_inspection`
  must show no prompt-facing projection of raw media payloads, hashes, or
  lookup refs. Allowed matches are validators, trusted context handling,
  session cache internals, and explicit sanitizer tests.
- `rg "current_media_|recent_media_" src\kazusa_ai_chatbot tests` must show
  aliases only in RAG3 context, tests, docs, and prompt-safe projections.
- `rg "MEDIA_SESSION_CACHE" src\kazusa_ai_chatbot tests docs README.md` must
  show configuration, tests, and docs for the fixed rolling media cache.
- `rg "media_kind.*audio|media_kind.*video|audio\\||video\\|" src\kazusa_ai_chatbot\media_inspection src\kazusa_ai_chatbot\local_context_resolver src\kazusa_ai_chatbot\complex_task_resolver`
  must return no production contract matches for this plan.
- `rg "local conversation media|conversation image|recent image" src\kazusa_ai_chatbot\cognition_chain_core tests`
  must show the L2d affordance update and its focused test.
- `rg "rag3_subagent_framework_and_media_inspection_bigbang_plan" development_plans\README.md`
  must show one active short-term registry row while this plan remains active.

### Focused Deterministic Tests

```powershell
venv\Scripts\python -m pytest `
  tests\test_local_context_resolver_subagent_registry.py `
  tests\test_local_context_resolver_subagent_dispatch.py `
  tests\test_media_inspection_contracts.py `
  tests\test_media_session_cache.py `
  tests\test_local_context_resolver_media_subagent.py `
  tests\test_complex_task_resolver_media_subagent.py `
  tests\test_action_selection_media_affordance.py `
  -q
```

### RAG3 Regression Tests

```powershell
venv\Scripts\python -m pytest `
  tests\test_local_context_resolver_contracts.py `
  tests\test_local_context_resolver_graph.py `
  tests\test_local_context_resolver_projection.py `
  tests\test_local_context_resolver_integration.py `
  tests\test_local_context_resolver_standalone.py `
  tests\test_local_context_resolver_cache.py `
  -q
```

### Cognition And Capability Integration

```powershell
venv\Scripts\python -m pytest `
  tests\test_cognition_resolver_contracts.py `
  tests\test_cognition_resolver_loop.py `
  tests\test_cognition_resolver_persona_graph.py `
  tests\test_cognition_resolver_l2d_contract.py `
  tests\test_action_selection_prompt_contract.py `
  tests\test_cognition_chain_core_action_selection.py `
  tests\test_persona_supervisor2_cognition_prewarm.py `
  tests\test_persona_supervisor2_rag2_integration.py `
  -q
```

### Complex Resolver Regression Tests

```powershell
venv\Scripts\python -m pytest `
  tests\test_complex_task_resolver_contracts.py `
  tests\test_complex_task_resolver_graph.py `
  tests\test_complex_task_resolver_service.py `
  tests\test_complex_task_resolver_evidence.py `
  tests\test_complex_task_resolver_algorithmic.py `
  tests\test_complex_task_resolver_prompt_contract.py `
  -q
```

### Media Descriptor Preservation

```powershell
venv\Scripts\python -m pytest `
  tests\test_media_descriptor_cache.py `
  tests\test_conversation_history_envelope.py `
  tests\test_msg_decontexualizer.py `
  tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py `
  tests\test_service_input_queue.py `
  -q
```

### Static Python And Diff Checks

```powershell
venv\Scripts\python -m py_compile <changed-python-files>
git diff --check
```

`git diff --check` may report existing line-ending warnings only when they are
proven unrelated to this plan's edited files.

### Default Deterministic Regression

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm and not live_internet" -q
```

### Live LLM Review

Run one case at a time with output inspected:

```powershell
venv\Scripts\python -m pytest tests\test_rag3_media_inspection_live_llm.py::test_live_rag3_current_image_exact_question -s -m live_llm
venv\Scripts\python -m pytest tests\test_rag3_media_inspection_live_llm.py::test_live_rag3_current_image_identity_uncertainty -s -m live_llm
venv\Scripts\python -m pytest tests\test_rag3_media_inspection_live_llm.py::test_live_rag3_current_image_cache_miss_boundary -s -m live_llm
venv\Scripts\python -m pytest tests\test_media_inspection_live_llm.py::test_live_media_inspector_exact_visual_question -s -m live_llm
venv\Scripts\python -m pytest tests\test_media_inspection_live_llm.py::test_live_media_inspector_unsupported_non_visual_question -s -m live_llm
venv\Scripts\python -m pytest tests\test_local_context_resolver_live_llm.py::test_production_exact_phrase -s -m live_llm
venv\Scripts\python -m pytest tests\test_local_context_resolver_live_llm.py::test_production_scoped_memory -s -m live_llm
venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_media_inspection_external_image_exact_question -s -m live_llm
venv\Scripts\python -m pytest tests\test_complex_task_resolver_live_llm.py::test_live_media_inspection_external_image_fetch_refusal -s -m live_llm
```

Each live case must record:

- command and pass/fail status;
- stage-call counts;
- media-inspection call count;
- cache hit or cache miss status where relevant;
- artifact path;
- human-readable review of whether evidence, uncertainty, and boundaries are
  acceptable.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the completed RAG3 bigbang plan, this plan, current RAG3
ICD, complex-task resolver ICD, message-envelope ICD, and relevant tests from
a fresh-review posture.

Review scope:

- The plan preserves RAG3 public IO and cognition/dialog ownership.
- The plan converts all source-backed RAG3 source handling into subagents.
- The plan avoids compatibility shims and dual source-dispatch paths.
- The plan updates L2d reachability without adding a new capability kind.
- The media cache, prompt-safe aliasing, and DB non-persistence requirements
  are explicit and testable.
- The active-node cache cannot reuse media evidence across different image
  content.
- The media-inspection contract is image-only and does not include speculative
  audio, video, OCR, retry, browser, or reverse-search behavior.
- Complex external media intake has explicit URL fetch and SSRF safety rules
  and keeps raw bytes out of `available_evidence`.
- The shared media inspector can be used by RAG3 and complex-task resolver
  without merging their resolver families.
- The plan gives execution agents concrete contracts, paths, commands,
  verification gates, and acceptance criteria.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, raw media
  persistence risk, cache miss behavior, brittle fixtures, and avoidable blast
  radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`,
  `Change Surface`, contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including old source-hydration replacement,
  focused tests, regression tests, live LLM artifacts, documentation, and
  lifecycle evidence.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Execution Evidence

Record evidence here during execution. Leave entries empty until the matching
gate has actually run.

- Focused test contract baseline: initial collection failed only for absent planned modules; 9 focused deterministic tests pass after implementation.
- Production subagent handoff: user explicitly approved parent-led fallback without subagents; parent owns production and review work.
- RAG3 subagent conversion: registry, source adapters, dispatch, Cache2 v2, and retired bridge deletion completed; retired-source grep is clean.
- Shared media inspector and session cache: image-only contracts, scoped capped cache, inspector, config, and focused tests pass.
- RAG3 media subagent integration: selector aliases, cache miss boundary, sanitization, media artifacts, and non-cacheable media nodes pass focused tests.
- Complex media intake and URL safety: resolver-local media subagent checks URL scheme, DNS/IP, redirects, MIME, bytes, magic bytes, and decoded dimensions before inspection.
- L2d affordance reachability: `local_context_recall` now names conversation images; focused affordance test passes.
- Static grep results: no production/test matches for retired source bridge or enablement flag; registry row present; diff check passes with existing line-ending warnings only.
- Deterministic regression results: 100 focused RAG3/complex tests pass. Default collection hits unrelated fixture-package imports; fixture-excluded run exceeded 120 seconds.
- Live LLM review artifacts and quality judgments: exact RAG3 phrase trace plus two shared-inspector traces reviewed in `rag3_media_inspection_live_llm_review.md`; exact visual grounding and unsupported intent boundary accepted.
- Documentation and registry updates: README, HOWTO, RAG3 ICD, complex-resolver ICD, media ICD, dependency declaration, and active registry updated.
- Independent code review findings and remediation: parent-led fallback review fixed scoped-memory node identity, alias ordering, result validation, and source-row allowlists; focused reruns pass.
- Residual risks: remaining planned RAG3-media and complex-media live cases plus fixture-excluded default regression must run before final lifecycle completion.

## Acceptance Criteria

This plan is complete when:

- RAG3 has a resolver-local subagent registry with validated metadata and
  deterministic node-kind ownership.
- Every source-backed RAG3 node kind listed in this plan routes through a
  subagent, not through `source_hydration.py` or a direct branch in
  `service.py`.
- `source_hydration.py` is removed from production imports and deleted when no
  test import remains.
- Existing memory, scoped-memory, conversation, person, recall, live-context,
  and supplied-external behavior is preserved through converted subagents.
- The generic media descriptor prompt, result schema, persistent cache, and
  attachment-description update behavior remain unchanged.
- Current/recent session media is cached process-locally with fixed count and
  byte, item, and TTL caps; scope mismatch cannot read cached media; cache miss
  returns bounded missing evidence without a media-inspection LLM call.
- RAG3 media inspection uses prompt-safe aliases and never exposes raw media,
  hashes, cache refs, ids, or descriptor-cache internals to cognition or
  dialog.
- The shared media inspector supports image-first exact-question answering and
  returns typed uncertainty/evidence boundaries through the image-only
  contract.
- Complex-task resolver can inspect externally fetched image media through its
  own media subagent and the shared inspector after deterministic URL, MIME,
  byte, redirect, DNS/IP, and decoder validation.
- L2d can select existing `local_context_recall` for local conversation image
  details without a new direct media capability.
- RAG3 active-node cache policy is bumped and cannot reuse media evidence for a
  different image.
- Static greps prove no forbidden raw media or retired source-hydration
  production path leaks remain.
- Focused deterministic, RAG3 regression, cognition integration, complex
  resolver regression, media descriptor preservation, static, and default
  deterministic regression commands pass or record unrelated residual failures
  with evidence.
- Selected live LLM cases run one at a time and produce accepted review
  artifacts.
- README, HOWTO, local-context resolver ICD, complex-task resolver ICD,
  message-envelope or media docs, and development plan registry reflect the
  final architecture.
- Independent code review approves the diff after remediation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| RAG3 source behavior regresses during subagent conversion | Convert existing helper calls behind subagents and preserve projection contracts | RAG3 regression tests and source-behavior live LLM case |
| Media cache leaks raw payloads into prompts | Split trusted lookup refs from prompt-safe aliases and add sanitizer greps | Static greps and media subagent prompt projection tests |
| Cache miss causes confusing cognition evidence | Return explicit `cache_miss` missing-evidence rows and boundary notes | Cache miss deterministic and live LLM cases |
| L2d becomes too tool-heavy | Keep L2d capability set unchanged and route through `local_context_recall` or `public_answer_research` | Cognition resolver contract tests |
| Complex resolver and RAG3 boundaries blur | Use one shared media inspector below both families while keeping family-local subagents | Complex and RAG3 subagent tests plus docs review |
| Response latency increases | Preserve existing RAG3 LLM counts for non-media paths and add one media-inspection call only on cache hit | Live LLM stage-call evidence |
| Descriptor behavior changes accidentally | Keep descriptor prompt and schema unchanged and run descriptor cache/decontextualizer tests | Media descriptor preservation command |
| Raw media is persisted accidentally | Keep session cache process-local and preserve existing DB storage policy | Conversation storage tests and static greps |
