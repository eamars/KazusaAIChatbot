# QQ replied image description unavailable queue prune bugfix plan

## Summary

- Goal: document the observed failure where Kazusa could not retrieve the
  image description for a native QQ reply to an earlier image-only group
  message.
- Plan class: small
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: not selected; this draft is observation-only and
  does not authorize production-code, prompt, schema, test, queue, or database
  changes.
- Highest-risk areas: mistaking this for a QQ image-fetch failure, treating
  reply hydration as able to regenerate missing media descriptions, weakening
  queue pruning without preserving group-chat noise behavior, and exposing raw
  image bytes to prompt-facing reply context.
- Acceptance criteria: this draft is complete when the failure mode,
  production evidence, observed boundaries, and excluded implementation scope
  are recorded for later investigation.

## Context

This draft records the failure behind Kazusa's QQ group response:

```text
诶？
图片好像加载不出来呢……
我什么都看不见啊，是不是没发成功呀？
```

The assistant row is in QQ group `905393941` with platform message id
`798654675`. It was sent at `2026-06-07T13:30:17.0568+00:00`
(`2026-06-08T01:30:17.0568+12:00` local time).

Kazusa was answering Jigsaw, QQ user `411706805`, global user id
`fa874545-02e6-4127-a24e-30819f941d83`. Jigsaw's immediate addressed request
was platform message id `775464770`:

```text
@杏山千纱 这是啥动物，给我个鉴定
```

That request was a native QQ reply to an earlier image-only message with
platform message id `2118424974`.

## Evidence

Primary evidence artifacts:

- `test_artifacts/diagnostics/qq_905393941_recent_history.json`
- `test_artifacts/diagnostics/qq_905393941_image_failure_rows.json`
- `test_artifacts/diagnostics/event_log_ch905393941_1328_1332.json`
- `test_artifacts/diagnostics/rag_cache2_media_descriptor_recent.json`

Observed stored conversation sequence:

| UTC time | Speaker | Message id | Stored summary |
|---|---|---|---|
| `13:28:37.970887` | Jigsaw `411706805` | `2118424974` | Image-only group row with empty `body_text`, raw QQ image CQ text, and one stored image attachment. |
| `13:29:23.879092` | Runtime event | `chat:qq:ch_e30715c7af5bb11a:2118424974` | `queue_intake` event with `status="dropped"` for the image row. |
| `13:30:00.747269` | Jigsaw `411706805` | `775464770` | Bot-addressed native reply asking Kazusa to identify the animal in the replied image. |
| `13:30:17.0568` | Kazusa | `798654675` | Visible response saying the image did not load and she could not see anything. |

Observed stored row for the original image:

- `platform_message_id="2118424974"`
- `platform_channel_id="905393941"`
- `body_text=""`
- `raw_wire_text` contains a QQ image CQ segment with a `multimedia.nt.qq.com.cn`
  download URL.
- `attachments=[{"media_type": "image/jpeg", "storage_shape": "inline"}]`
- no stored `description`.

Observed reply hydration result on Jigsaw's later request:

```json
[
  {
    "media_kind": "image",
    "description": "",
    "summary_status": "unavailable"
  }
]
```

Observed comparison row:

- A later image in the same group, platform message id `1496773441`, did get a
  generated image description at `2026-06-08T01:31:50.630984+12:00`.
- This shows the media descriptor path was available when an image message
  survived into graph execution.

## Observed Failure Mode

The system persisted the original image-only QQ message, but the group-chat
input queue pruned it before the persona graph ran. Because the graph did not
run for that image row, the media descriptor never generated or persisted an
attachment description.

Later, when Jigsaw sent a bot-addressed native reply to that image, reply
hydration successfully found the replied conversation row, but the row only
contained an image attachment shell with no stored description. The prompt-safe
reply projection therefore exposed the image as unavailable rather than as a
described image.

The visible result is:

```text
Original image row exists.
Original image row has no persisted image description.
Later native reply points to that row.
Reply context can only project summary_status="unavailable".
Kazusa says she cannot see the image.
```

This is a live intake, queue, and media-description boundary failure. It is not
primarily a RAG recall failure, and the current evidence does not support a
primary QQ adapter fetch failure.

## Current Boundary Read

The current evidence points away from these ownership boundaries:

- Not primarily QQ adapter image fetching: the original row stores an inline
  image attachment shell, and a later same-group image was described normally.
- Not primarily MongoDB identity resolution: Jigsaw's platform and global user
  ids are stable, and the later request is correctly addressed to Kazusa.
- Not primarily reply target lookup: reply hydration found the target row and
  projected its stored attachment state.
- Not primarily RAG retrieval: the later request needed the native reply
  attachment description, not a separate historical fact search.

The current evidence points toward these ownership boundaries:

- `chat_input_queue` group-pruning behavior, because unaddressed/non-private
  group messages can be dropped when a noisy queue burst exceeds survivor
  thresholds.
- Dropped-item handling in `service`, because a dropped queued item is persisted
  and completed without graph execution.
- The persona graph media descriptor edge, because it runs only when a
  surviving graph state contains `user_multimedia_input`.
- Conversation attachment persistence, because generated descriptions are only
  stored after the descriptor updates the conversation row.
- Reply attachment projection, because it intentionally exposes only stored
  prompt-safe summaries and does not regenerate vision output from raw media.

## Observation-Only Scope

This completed observation record only records the failure mode and observations.

It does not authorize:

- production-code edits;
- prompt edits;
- test edits;
- queue policy changes;
- media descriptor changes;
- reply hydration changes;
- attachment storage policy changes;
- adapter image-fetch changes;
- RAG routing changes;
- dialog evaluator changes;
- database migration or data repair;
- backfilling descriptions for historical rows.

## Required Next Discovery Before Executable Plan

The next stage must inspect or reproduce the smallest safe owning boundary
before proposing a fix:

- whether unaddressed image-only group messages should receive media
  description before queue pruning, after queue pruning, or only when later
  referenced by a protected reply;
- whether dropped media rows should run a bounded descriptor-only persistence
  path without entering full cognition or dialog;
- whether reply hydration may safely request a deferred descriptor when the
  replied row has prompt-safe metadata but no stored description;
- whether any change must preserve binary payload storage limits and avoid
  exposing raw base64 or URL-only platform media to prompts;
- whether existing tests cover queue-pruned media rows that are later referenced
  by native replies.

The executable plan must be written only after this discovery identifies the
smallest owning stage and the smallest safe change surface.

## Lifecycle

- Archived on 2026-06-10 during active-plan cleanup as a completed
  observation-only incident record. Future implementation work requires a new
  active executable plan.
