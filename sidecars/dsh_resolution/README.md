# DSH Resolution Sidecar

The sidecar is the independently operating DSH Standard execution boundary.
It authenticates Brain/resolver calls, persists session events and receipts,
owns native Standard tools, and forwards the exact semantic catalog.

The model-visible catalog has exactly fourteen rows in canonical order:

1. kazusa_search_conversation_history
2. kazusa_read_conversation_entries
3. kazusa_summarize_conversation_participants
4. kazusa_search_memories
5. kazusa_read_memories
6. kazusa_remember_information
7. kazusa_revise_memory
8. kazusa_change_memory_lifecycle
9. kazusa_find_people_by_name
10. kazusa_read_person_profiles
11. kazusa_recall_active_context
12. kazusa_read_calendar_context
13. kazusa_inspect_attached_media
14. kazusa_inspect_public_media

kazusa_inspect_public_media is limited to HTTP(S) image URLs and a semantic
question. The shared safety contract rejects credentials/fragments, private
or special-use DNS results, unsafe redirects after at most 3 hops, responses
over 6 MiB, and responses exceeding 15 seconds. PNG, JPEG, GIF, or WebP
MIME/magic agreement, Pillow decoding, and 1..8192 dimensions are required.
The sidecar forwards bounded vision evidence with source dsh_public_media,
never raw bytes or base64.

Any catalog schema change produces a new catalog digest. Eligible
terminal/checkpointed V2 threads rotate to a fresh segment when no interaction
is open; authority and grants bound to an earlier digest fail closed.

system.health is ready only when authenticated route, Standard,
semantic-worker, web, Brain, catalog, policy, workspace, profile, release,
and store checks agree. Build and type checking use the package-manager version
pinned in `package.json`.
Activation verification allows at most 50 ms of future issuance to account for
Python/Node wall-clock precision on Windows loopback calls. Expiration has no
grace period; MAC, scope, audience, generation, and policy checks still apply.

The installed Standard profile exposes native web search. A requested source
verification requires retrieved text; source links and prior knowledge alone
produce a partial result with the missing verification stated explicitly.
On Windows, confined-shell HTTPS may fail with `SEC_E_NO_CREDENTIALS`. DSH
uses the native shell's exact-command escalation contract and Brain judgment
for that observed failure. The approval decision controls the retry.

The reusable real-process verification owner is
`experiments/dsh_runtime_probe.py`. Its `sidecar-lifecycle` mode drives the
built Standard sidecar through authenticated semantic work, SQLite checkpoint
restart, and exact terminal replay with a deterministic model endpoint. Use it
to diagnose recovery after establishing real-model viability through the Brain.
The dedicated TypeScript test suite is retired; strict production type checking
and build remain. Fresh runtime repairs and verification are recorded in the
[completion record](../../development_plans/archive/completed/bugfix/dsh_runtime_completion_plan_2026-09-05.md).

`ResolutionSidecarRuntime.forProduction` is the sole executor construction
boundary. Its injected executor and controls perform real operation joining,
durable inspection/replay, and fenced forwarding. Fixtures and scripted
providers belong to verification support. The process probe injects terminal
response loss outside the sidecar and also covers RPC authentication,
activation authority rejection, independent sessions, and missing-worker
readiness. Shared launch and cleanup resources live in
`experiments/dsh_process_support.py`.
