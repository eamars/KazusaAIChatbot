# Repository Index ICD

`repository_index` owns persistent, immutable, workspace-local discovery of
safe repository text. It uses versioned SQLite/FTS5 snapshots keyed by the
resolved source identity, complete source manifest, exclusion policy, and
schema version.

## Ownership

- `identity.py` defines canonical source identity and the exclusion/content
  privacy policy.
- `builder.py` scans every in-scope regular file, streams safe text into
  bounded line-aware chunks, and atomically publishes complete snapshots.
- `search.py` owns literal, regex, symbol, path, bounded read, and cursor
  contracts. It returns repository-relative prompt-safe evidence only.
- `storage.py` pins complete snapshots for live run owners and reclaims only
  complete, unpinned snapshots outside active cursors.
- `overlay.py` owns created, modified, deleted, and renamed candidate paths.
  Overlay-owned paths suppress stale base rows.
- `regex_worker.py` owns the bounded process used for regular-expression
  evaluation.

The index never owns task interpretation, edit permission, approval, command
selection, or final implementation judgment. Those remain with the controller
and deterministic coding-run boundaries.

## Safety And Persistence

The index excludes environment, credential-like, binary, symlink, generated
run/index, cache, and out-of-scope paths before publication. Eligible files
that exceed a configured byte or storage resource return a typed
`index_resource_exhausted` result; eligible files are never silently dropped.

A snapshot database remains unavailable until its `complete` transaction is
committed and the building database is atomically published. Coding runs pin
the selected snapshot identity and never switch snapshots after their first
action.
An existing published database is reused only after its snapshot, source,
manifest, schema, policy, and complete-state fields match the requested
identity. Index, pin, overlay, and snapshot database symlinks are rejected.

Incomplete builds commit one file at a time through `build_file`. Replaying a
file first clears its file, chunk, FTS, symbol, and import rows, so publication
cannot contain resume duplicates. Compatible complete snapshots supply rows
for unchanged content hashes while the new snapshot assigns current file and
row identities.

Candidate searches page one merged keyset order. A cursor binds the base
snapshot, overlay revision, query, mode, glob, and final sort tuple; overlay
paths suppress base rows on every page, including tombstones and renames.
Regex evaluation for both immutable chunks and candidate overlay content runs
in bounded worker processes and returns no partial rows after a timeout.
