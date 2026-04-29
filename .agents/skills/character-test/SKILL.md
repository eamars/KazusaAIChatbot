---
name: character-test
description: Evaluate a chatbot character's real behavior in a simulated live environment. Use this skill whenever the user asks to test, audit, inspect, probe, or run multi-turn character conversations through the debug channel or production service, especially when they want per-turn log inspection, RAG/memory/cognition analysis, real database effects, or adaptive dialog based on the character's response. This skill should trigger even if the user says "simulate", "debug channel", "real life response", "behavior audit", "inspect her/his behavior", or asks for a series of conversations with a character.
---

# Character Test

Use this skill to run a character through realistic, evidence-backed conversation tests. The goal is not just to collect replies; it is to observe how the live system behaves across API response, debug logs, RAG, memory, cognition, dialog generation, and background consolidation.

## Core Principle

Treat each turn as an experiment:

```text
dialog plan -> present to user -> send one debug turn -> inspect logs -> evaluate -> adapt next dialog
```

Do not batch a whole conversation unless the user explicitly asks for a fixed script. Character behavior often changes because of the previous response, retrieved memory, mood state, and background writes, so the next prompt should be assembled from fresh evidence.

## Safety And Scope

Before touching a live service or database, establish the test scope from the conversation:

- Use the real production database and environment only when the user explicitly permits it.
- If permission is absent or ambiguous, avoid persistent writes and ask whether to use production memory, test data, or `no_remember=true`.
- If the user wants to measure memory impact, keep memory enabled. Do not use `no_remember` unless they requested a non-persistent dry run.
- Use a stable simulated identity for the whole test so memory, progress, and profile behavior can be attributed consistently.
- Stop sending new turns immediately if the user says to stop, pause, consolidate, or discard the run.
- Preserve raw artifacts under `test_artifacts/` because later diagnosis may need exact request payloads, responses, timestamps, global user IDs, and log lines.

## Service Startup

Run from the repository root. Prefer the project virtual environment and set `PYTHONPATH=src` when starting the service.

On Windows PowerShell, use timestamped logs:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = "test_artifacts/debug_runs/kazusa_uvicorn_$stamp.log"
$err = "test_artifacts/debug_runs/kazusa_uvicorn_$stamp.err.log"
$env:PYTHONPATH = "src"
Start-Process -FilePath "venv\Scripts\python.exe" `
  -ArgumentList "-m","uvicorn","kazusa_ai_chatbot.service:app","--host","0.0.0.0","--port","8000" `
  -RedirectStandardOutput $out `
  -RedirectStandardError $err `
  -WindowStyle Hidden `
  -PassThru
```

Then health check the service:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

If a service is already running, identify whether it is the intended process before starting another. If you start the process, record the PID and stop it when the test is done unless the user asks to leave it running.

## Simulated Channel

Use the debug platform unless the user names another transport. Keep these values stable across turns:

```json
{
  "platform": "debug",
  "platform_channel_id": "debug-character-behavior-audit",
  "channel_type": "private",
  "platform_user_id": "debug-user-character-audit",
  "platform_bot_id": "debug-bot-character",
  "display_name": "<user display name>",
  "channel_name": "character behavior audit",
  "debug_modes": {
    "listen_only": false,
    "think_only": false,
    "no_remember": false
  }
}
```

Adjust `no_remember` to `true` only for non-persistent tests. If the target character, user identity, or channel identity matters, include those values explicitly in the request payload using the repo's current API shape.

## Before Each Turn

Present the exact dialog before sending it. Include the observation target so the user can catch a risky or biased prompt.

Use this compact format:

```markdown
Turn N candidate:
用户：<exact message>

Observation target: <what this turn probes>
```

If the user requested Chinese dialog, write the candidate dialog in Chinese. Keep the first turn gentle: greet the character and avoid phrasing that implies anxiety, disbelief, interrogation, blame, or hidden evaluation unless that is the specific thing being tested.

## Sending One Turn

For each turn:

1. Record the current log length or timestamp before sending.
2. POST one message to the chat endpoint.
3. Save the raw response as `test_artifacts/debug_runs/turn_NNN_response.json`.
4. Extract the new log slice and save it as `test_artifacts/debug_runs/turn_NNN_log.txt`.
5. Wait long enough for background progress recording and consolidation logs to appear.
6. Inspect the response and the log slice before planning the next turn.

The API response alone is not enough. Many important failures appear only in logs, such as wrong identity resolution, noisy RAG slots, failed background recorders, consolidation overreach, or a mismatch between retrieved context and final dialog.

## Log Inspection Checklist

Inspect fresh logs for these stages when present:

- Request envelope: platform, channel, user ID, display name, target character, debug modes.
- Relevance decision: whether the bot decided to answer and why.
- Conversation progress load: progress summary, episode state, overused moves, active expectations.
- Decontextualizer output: rewritten query, target names, cacheability, semantic drift.
- RAG initializer: requested slots, unknown slots, cache behavior.
- RAG dispatch: chosen agent, task, context, retries, raw arguments.
- RAG agent result: retrieved evidence, empty results, wrong user/channel, source IDs.
- RAG projection/final context: what actually reached cognition.
- Cognition/boundary judgment: mood, trust, suspicion, safety, intimacy, defense, confusion.
- Dialog generation: final response, language, persona stability, whether it uses or ignores evidence.
- Consolidation/reflection: facts, promises, preferences, rejection reasons, persistence outcome.
- Background failures: tracebacks, validation errors, timeouts, malformed LLM output, cache errors.

When diagnosing memory or identity, preserve exact IDs and platform names. A display name can resolve to the wrong global user if cross-platform identity logic is involved.

## Evaluation Rubric

Evaluate each turn on both character behavior and system performance.

Behavioral observations:

- Warmth, defensiveness, skepticism, avoidance, confusion, curiosity, or over-compliance.
- Whether the response fits the user's latest message and the active character.
- Whether the character escalates a harmless topic into suspicion.
- Whether the character remembers recent visible context.
- Whether the character gracefully recovers from correction.

System observations:

- RAG necessity: no retrieval, useful retrieval, noisy retrieval, harmful retrieval.
- Memory effect: helped, hurt, ignored, or wrote questionable new memory.
- Identity correctness: correct user/channel/character vs wrong profile or cross-platform leakage.
- Progress state: useful continuity vs stale or distorting episode state.
- Consolidation quality: durable facts/promises only vs transient chat persisted too aggressively.
- Latency and retry behavior: acceptable, excessive, or hidden background failure.
- Contract drift: hard-coded character name, wrong active character, schema mismatch, invalid LLM output.

State uncertainty clearly. One run is evidence, not proof; repeated failures across turns are stronger signals.

## Per-Turn Conclusion Log

Append findings to `test_artifacts/debug_runs/behavior_audit_conclusions.md` after every turn:

```markdown
## Turn N

- User text: <exact text>
- Character response: <short exact response or summary>
- Behavioral read: <what the character did>
- Log evidence: <RAG/cognition/memory/progress/consolidation observations>
- System concern: <bug, risk, or "none observed">
- Next-turn implication: <how this should shape the next prompt>
```

Keep this file concise but evidence-rich. It should be useful for a later bug fix or architecture plan without replaying the entire log.

## Adapting The Next Dialog

Choose the next message based on the last response and logs:

- If the character becomes defensive, de-escalate before probing deeper.
- If RAG over-retrieves, use a simple turn that should not require retrieval and see whether the pipeline still searches.
- If memory seems relevant, ask a recall or continuity question and compare visible chat history, RAG evidence, and final response.
- If identity resolution looks wrong, test with a message that has unmistakable recent context, then inspect IDs and search filters.
- If consolidation writes questionable facts, avoid piling on more state until you understand the write path.
- If tracebacks appear, log them as high priority and consider stopping the behavioral run.

Do not manipulate the character with harsh prompts unless the user's test specifically requires stress testing. Most audits are more valuable when prompts are natural and low-pressure.

## Consolidation Report

When the user asks to stop or the planned turns are complete, provide a short report with:

- Highest-severity failures first.
- What worked well.
- Memory/RAG impact across turns.
- Any background errors and their likely blast radius.
- Artifact paths for response JSON, per-turn logs, and the conclusion file.
- Recommended next engineering actions.

Avoid resolving only the latest single case if the user's question is systemic. Explain whether the observed issue is character-specific, active-character-generic, identity-generic, RAG-generic, or consolidation-generic.

## Common Pitfalls

- Do not infer behavior from the chat response alone.
- Do not skip presenting the next dialog when the user requested pre-action review.
- Do not use production memory without explicit permission.
- Do not forget background tasks; recorder and consolidation failures may appear after the response returns.
- Do not hard-code Kazusa or any specific character into reusable conclusions. Use "active character" unless the test target is explicitly that character.
- Do not continue a run after the user says to consolidate.
- Do not treat a display name as a unique identity without checking platform/global user mapping in logs.
