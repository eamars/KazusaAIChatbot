# DSH Resolution Sidecar

The sidecar is the independently operating DSH Standard execution boundary.
It authenticates Brain/resolver calls, persists session events and receipts,
owns native Standard tools, and forwards the exact semantic catalog.

The model-visible catalog has exactly fourteen rows. Plan 2 rows 1 through
13 remain byte-identical; Plan 3 adds only row 14:

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

The new row produces a new catalog digest. Eligible terminal/checkpointed V2
threads rotate to a fresh segment when no interaction is open; old authority
and grants fail closed. Open pre-cutover interactions and grants drain first.

system.health is ready only when authenticated route, Standard,
semantic-worker, web, Brain, catalog, policy, workspace, profile, release,
and store checks agree. Build and sidecar tests are run through the pinned
package-manager command recorded by Plan 3.
