# DSH Semantic Tool Gateway

The gateway publishes the description-free, storage-independent semantic
catalog mounted by the DSH Standard route. It contains exactly fourteen rows
in canonical order.

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

## Public media boundary

The additive tool accepts an HTTP(S) URL and a visual question. Fetching
rejects credentials and fragments, resolves DNS before connecting, rejects
private, loopback, link-local, multicast, reserved, and unspecified
addresses, and rechecks each redirect up to 3 times. The timeout is 15
seconds and the body limit is 6 MiB. MIME and magic bytes must agree on PNG,
JPEG, GIF, or WebP; Pillow must decode the image and dimensions must be
between 1 and 8192. The vision result identifies source dsh_public_media;
raw bytes and base64 are never model-facing.

## Catalog authority

Any catalog schema change produces a new semantic catalog digest. A terminal
or checkpointed V2 thread without an open interaction rotates to a fresh
segment carrying the current digest. Authority and grants bound to an earlier
digest fail closed. Native Standard names retain precedence and
submit_resolution remains the sole terminal surface.

Run gateway contract, media safety, authority, and worker tests with
venv\Scripts\python.
