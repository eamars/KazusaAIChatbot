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
and store checks agree. Build and sidecar tests use the package-manager version
pinned in `package.json`.

The reusable real-process verification owner is
`experiments/dsh_runtime_probe.py`. Its `sidecar-lifecycle` mode drives the
built Standard sidecar through authenticated semantic work, SQLite checkpoint
restart, and exact terminal replay with a deterministic model endpoint. Tests
assert public behavior and compatibility; installed package file hashes and
internal composition row counts are not release contracts.
