# DSH Semantic Tool Gateway

The gateway is the Kazusa-owned semantic boundary mounted into the official
DSH Standard profile. It exposes storage-independent operations as typed
entities, opaque references, and bounded evidence receipts. Storage schemas,
database identifiers, credentials, and authority tokens stay private to the
gateway and its callers.

## Fixed semantic catalog

The model-facing catalog is the `SEMANTIC_TOOL_NAMES` tuple in `catalog.py`:

| Name | Semantic operation |
| --- | --- |
| `kazusa_search_conversation_history` | Find relevant scoped conversation entries by meaning and optional time range. |
| `kazusa_read_conversation_entries` | Read complete conversation entries by opaque references. |
| `kazusa_summarize_conversation_participants` | Summarize participants observed in a bounded conversation range. |
| `kazusa_search_memories` | Search semantic memories by query, subject scope, and kind. |
| `kazusa_read_memories` | Read complete semantic memories by opaque references. |
| `kazusa_remember_information` | Retain information with an explicit subject, kind, reason, and provenance. |
| `kazusa_revise_memory` | Revise one memory identified by an opaque reference. |
| `kazusa_change_memory_lifecycle` | Apply one explicit lifecycle transition to a memory reference. |
| `kazusa_find_people_by_name` | Find people by display name and semantic relation matching. |
| `kazusa_read_person_profiles` | Read semantic profiles by opaque person references. |
| `kazusa_recall_active_context` | Recall active commitments, progress, history, or calendar context. |
| `kazusa_read_calendar_context` | Read authorized schedule or calendar-run context by view. |
| `kazusa_inspect_attached_media` | Inspect attached media by opaque reference and semantic question. |

`submit_resolution` is separate and controller-owned: it is the sole
model-facing terminal operation. Standard native tools take precedence by
name, so the gateway does not wrap coding, filesystem, shell, jobs, tests,
public web, approval, or sandbox capabilities.

## Worker and authority boundary

The sidecar uses a persistent length-prefixed JSON worker. The worker validates
the signed activation authority and exact call scope before dispatch, records
idempotent semantic mutations, and replays committed outcomes from the
adjacent `<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/semantic-outcomes.sqlite`.
DSH session events remain in `sessions.sqlite`; the two stores have separate
ownership. Result frames are bounded, and opaque continuation references are
authenticated and bound to scope, audience, profile, catalog, route,
workspace, policy, activation, lease, segment, and expiry.

The gateway owns service calls and semantic result shaping. It does not decide
character stance, visible wording, or whether a DSH approval/question should
be relayed; those decisions belong to the Brain and cognition boundary. See
the [DSH interaction README](../dsh_interaction/README.md), [sidecar
README](../../../sidecars/dsh_resolution/README.md), and [integration
architecture](../../../docs/architecture/dsh_integration_architecture.md).
