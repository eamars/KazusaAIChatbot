# NapCat QQ Adapter ICD

## Public Imports And CLI

The package root exports `NapCatWSAdapter`, `QQEnvelopeNormalizer`,
`project_qq_semantic_text`, `runtime_app`, and `main`. The command
`python -m adapters.napcat_qq_adapter` starts the same adapter CLI.

The CLI reads `BRAIN_URL`, `NAPCAT_WS_URL`, `NAPCAT_WS_TOKEN`, runtime callback
settings, and `NAPCAT_ACTIVE_GROUPS` from the environment after loading
`.env`. `NAPCAT_ACTIVE_GROUPS` is a comma- or whitespace-separated list of QQ
group ids used when `--channels` is omitted. Explicit `--channels` values take
precedence. When neither source provides groups, public groups are listen-only
and private chats remain active.

## Submodule Responsibility Table

`cli.py` owns argument and environment parsing. `runtime_api.py` owns the
FastAPI callback app and active adapter binding. `ws_adapter.py` owns websocket
lifecycle, OneBot API dispatch, brain forwarding, and delivery receipts.
`envelope_normalizer.py` owns typed envelope construction. `cq_projection.py`
owns CQ marker parsing and prompt-facing text projection. `face_catalog.py`
owns the static QQ face labels. `inbound_segments.py` owns segment-list to wire
text conversion. `mention_hydration.py`, `reply_hydration.py`, `attachments.py`,
and `outbound.py` own their named adapter helper domains.

## Inbound QQ Segment Flow

NapCat string messages are treated as canonical wire text. Structured segment
lists are converted to CQ-compatible wire text before envelope normalization so
one parser owns mention, reply, face, and image marker projection.

Reply hydration may preserve a replied message excerpt as adapter-internal CQ
wire evidence until mention display-name hydration has completed. The final
`MessageEnvelope.reply.excerpt` sent to the brain must be prompt-facing
semantic text and must not contain CQ markers, occurrence placeholders, or raw
platform ids.

## CQ Projection And Face Catalog Contract

CQ mentions become readable mention tokens. CQ replies are removed from body
text and represented through typed reply metadata. Known QQ face ids become
`<image>{description}</image>` text through the static catalog.

QQ display labels use the adapter's canonical selection order:
`nickname -> name -> card -> user`. The `card` group alias is only used when
profile nickname/name are unavailable. If a mentioned user lookup fails, the
semantic fallback is platform-neutral, for example `@user`; occurrence
placeholders such as `@mentioned-user-1` and platform-qualified labels such as
`@qq-user:<id>` are not valid envelope text.

## Runtime API Binding Contract

`runtime_api.py` is importable without `ws_adapter.py`. The websocket adapter
binds itself when its callback server starts and clears the binding on close
only when it still owns the active binding.

## Unknown Face Omission Contract

Unknown, missing, empty, non-numeric, or unusable QQ face ids are omitted from
semantic projection. A face-only unknown message therefore has empty body text;
inline unknown faces collapse adjacent authored text.

## Source Snapshot And Label Maintenance

`face_catalog.py` contains the accepted numeric QQ face id set from QFace commit
`e476a706a7e508849c6031c3654051a02639964f`. Runtime lookup is static and never
performs network, database, or model calls.

## Verification Commands

Run `venv\Scripts\python.exe -m py_compile` for all package Python files, then
the focused adapter normalizer and runtime adapter tests listed in the active
development plan.
