# Runtime Coordination Interface Control Document

## Document Control

- Owning package: `kazusa_ai_chatbot.runtime_coordination`
- Interface boundary: in-process runtime functions -> scoped pipeline
  admission and cooperative cancellation
- Runtime consumers: brain service foreground queue, self-cognition worker,
  reflection phase group review, calendar reflection handler, and future
  foreground or background applications that can share a channel scope

## Purpose

Runtime coordination owns deterministic in-process ordering for pipelines that
can produce visible output from channel context. It does not own cognition,
RAG, dialog wording, dispatcher delivery, adapter capability, calendar
materialization, or durable cross-process leases.

For one canonical channel scope, foreground work has precedence over
background work that may send visible output. A foreground caller may request
cooperative cancellation of same-scope background runs. A background caller
must ask for admission before building visible-output context and must check
for cancellation at deterministic checkpoints before expensive stages and
before dispatcher handoff.

## Public Interface

The public entrypoint is `kazusa_ai_chatbot.runtime_coordination`.

Canonical values:

- `PipelineScope(platform, platform_channel_id, channel_type)`
- `PipelineCoordinator`
- `PipelineRunHandle`
- `PipelineCancellation`
- `PipelineCancelled`

Callers use:

- `PipelineCoordinator.start_run(...)` to request foreground or background
  admission for a `PipelineScope`;
- `PipelineCoordinator.request_cancellation(...)` to cancel lower-precedence
  same-scope background runs without importing the caller that caused the
  foreground work;
- `PipelineRunHandle.raise_if_cancelled(checkpoint)` at cooperative
  checkpoints.

The coordinator is generic. It must not import `/chat`, service queue,
self-cognition, reflection, dispatcher, adapter, QQ, Discord, or debug-client
policy.

## Scope Rules

`PipelineScope` is the exact triple `(platform, platform_channel_id,
channel_type)`. Empty or missing platform/channel fields are not a wildcard.
Different scopes must remain independent.

Foreground runs represent current user-facing work in a channel. Background
runs represent maintenance, reflection-attached self-cognition, standalone
self-cognition, or future applications that can send visible output or make
channel-visible commitments from a context snapshot.

Same-scope foreground admission cancels active background handles. Same-scope
background admission defers while a foreground handle is active. The
coordinator does not force-cancel sockets, model provider calls, or adapter
sends; callers must honor cancellation before dispatcher handoff.

## Ownership Boundaries

Deterministic code owns admission, handle release, cancellation state, and
checkpoint exceptions. LLM stages still own semantic judgment and final wording.
Dispatcher and adapters remain delivery executors; they do not decide whether
a background pipeline is stale enough to cancel.

The process-local coordinator is not a distributed lock and is not durable
scheduler state. Durable calendar runs that defer because of runtime
coordination must be requeued by the calendar scheduler rather than completed,
skipped, or failed.
