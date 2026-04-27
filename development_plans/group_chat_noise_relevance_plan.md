# group chat noise relevance plan

## Summary

- Goal: Cut Kazusa's false-positive responses in noisy group chats by (1) fixing the broken noisy-environment detector, (2) feeding the relevance LLM a single structural boolean indicating whether the platform itself reports the current message as `@`-mentioning the bot, and (3) supplying one compact attention-noise label so the LLM has a calibrated bar in busy rooms.
- Plan class: medium
- Status: draft
- Overall cutover strategy: bigbang for in-process changes; compatible at the stored-document layer.
- Highest-risk areas: silently shipping zero behavior change because prompt selection still hits a stale field; over-tightening private chats; introducing deterministic parsing of user text in violation of the `no-prepost-user-input` skill.
- Acceptance criteria: noisy-group prompt fires whenever `channel_type` indicates a group; the relevance LLM payload for groups carries `mentioned_bot` and one `group_attention` label; the regression case modeled on `test_artifacts/qq_1082431481_recent_30_chat_history.json` (ambiguous second-person message in a noisy room with no mention or reply to bot) does not produce a Kazusa response.

## Context

The relevance agent already has a stricter `_RELEVANCE_SYSTEM_NOISY_PROMPT`, but two upstream defects keep it from firing on the cases we care about:

1. **Noisy-environment selection uses the wrong field.** [relevance_agent.py:231-232](src/kazusa_ai_chatbot/nodes/relevance_agent.py#L231-L232) keys off `channel_name`, the human-readable group title, which is empty for adapters that do not fetch a group name (notably the QQ adapter). Group requests therefore route through the *non-noisy* prompt today.
2. **Address detection has no structural mention input.** Adapters do not currently surface platform-native `@` mention information. `_should_ignore_third_party_reply` falls back to a regex over user text (`[Reply to message]` + `<@id>` extraction), which matches Discord's legacy normalization but misses QQ bare-name address ("千纱さん"). The LLM cannot distinguish "the platform reports my id was mentioned" from "the user typed my name in prose."

Together these defects let ambiguous second-person messages such as `你喜欢千纱吗？` reach the LLM with the permissive prompt and no structural signal that the message is or is not addressed to the bot.

The fix is to (a) route by `channel_type`, (b) add a single `mentioned_bot` boolean end-to-end so the LLM and the metadata short-circuit have a precise structural signal, and (c) layer one descriptive `group_attention` label on top so the LLM has a bar that scales with room business.

## Assumptions

- The bot's `platform_user_id` is stable per platform per character. Recomputing the `mentioned_bot` boolean against a different bot identity in the future is out of scope; the boolean is computed once at adapter time against the current bot id and stored.
- Each channel hosts at most one Kazusa-style character. Multi-character-per-channel scenarios are not supported by this plan.

## Mandatory Rules

These rules are non-negotiable. They duplicate critical guidance from the `no-prepost-user-input` skill so the implementing agent does not need to re-derive them.

- **No deterministic interpretation of current user text.** Do not parse, classify, or filter the current user message body to decide relevance. No keyword checks for `你`, `千纱`, aliases, commands, questions, tone, or intent.
- **Deterministic logic over structural metadata is allowed and encouraged.** `role`, `platform_user_id`, `timestamps`, `reply_context.*`, `channel_type`, `platform_bot_id`, and `mentioned_bot` are structural and may be inspected.
- **Never send raw numeric stats to the LLM.** All counts and rates produced by the noise helper must be converted to descriptive labels before being placed in the prompt payload.
- **LLM owns final social judgment.** `mentioned_bot` and the descriptor are evidence; the LLM still decides `should_respond` in every case except the single chaotic-noise metadata short-circuit defined below.
- **Group-only scope.** Private/DM relevance behavior is unchanged.
- **No downstream pipeline changes.** Do not alter cognition, dialog, RAG, consolidation, memory, or scheduler.

## Must Do

1. Fix `is_noisy_environment` to derive from `channel_type` rather than `channel_name`.
2. Add `mentioned_bot: bool` end-to-end (adapter → request → state → conversation persistence → history trim → relevance payload). New field; defaults to `False`.
3. Replace the legacy text-parsing helper `_extract_legacy_reply_target_id` with a metadata-only check using `reply_context.reply_to_platform_user_id` and `mentioned_bot`. Delete the helper.
4. Add `build_group_attention_context` helper that derives one descriptive label from `chat_history_wide` metadata only.
5. Wire `mentioned_bot` and `group_attention` into the noisy-group human payload only.
6. Update `_RELEVANCE_SYSTEM_NOISY_PROMPT` to explain how to weigh `mentioned_bot`, `reply_context.reply_to_current_bot`, and `group_attention`.
7. Add a metadata-only short-circuit: when `group_attention == "chaotic_noise"` AND `reply_context.reply_to_current_bot is not True` AND `mentioned_bot is not True`, skip the LLM and return `should_respond=False`.
8. Add the unit, payload-shape, short-circuit, and regression tests listed in the Verification section.

## Deferred

- Do not redesign relevance architecture beyond the points listed in Must Do.
- Do not change cognition, dialog, consolidation, RAG, scheduler, or dispatcher modules.
- Do not migrate historical conversation documents to populate `mentioned_bot`. Documents written before this change will read with `mentioned_bot = False`.
- Do not introduce keyword/alias matching over user text in any helper.
- Do not add raw numeric noise sub-scores to the LLM payload.
- Do not change the visible message content format.
- Do not surface the full mention list of other parties; only the boolean against the current bot id is required.

## Cutover Policy

Overall strategy: bigbang for in-process changes; compatible at the stored-document layer.

| Area | Policy | Instruction |
|---|---|---|
| Noisy-environment selection | bigbang | Switch the field used for `is_noisy_environment` to `channel_type`. Remove the `channel_name`-based branch; do not keep both. |
| `mentioned_bot` plumbing | bigbang | Add the field through every layer in one pass. New field; no compatibility shim required at the type level. |
| `mentioned_bot` storage | compatible | Conversation documents persisted before this change have no `mentioned_bot`. Readers must treat the field as optional and default to `False`. |
| Legacy reply-marker parser (`_extract_legacy_reply_target_id`) | bigbang | Delete. Replace its single call site with a metadata-only check. |
| Relevance prompt and payload | bigbang | Update `_RELEVANCE_SYSTEM_NOISY_PROMPT` and noisy human payload directly. No dual prompt path beyond the existing private-vs-group selection. |
| Private/DM relevance | compatible | Preserve current private/DM prompt and behavior. Do not attach `mentioned_bot` or `group_attention` to private payloads. |

### Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not preserve the old `channel_name`-based detection as a fallback.
- The agent must not preserve `_extract_legacy_reply_target_id` for "compatibility."
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may pick local helper names and exact internal threshold numbers, provided every count is converted to a label before reaching the LLM payload.
- The agent must not introduce semantic keyword routing or content-based filtering of user text.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or broad prompt rewrites.
- If the code shape differs from this plan, preserve the plan's intent and report the discrepancy.
- If a required instruction is impossible (for example, the QQ adapter cannot expose mention ids), stop and report the blocker rather than substituting a text-parsing fallback.

## Target State

For group chat requests, the noisy-group LLM payload includes:

```json
{
  "user_message": {
    "...": "existing fields",
    "mentioned_bot": false
  },
  "group_attention": "high_noise"
}
```

`group_attention` is one of `low_noise | medium_noise | high_noise | chaotic_noise`, computed deterministically from history metadata. It is the only new top-level key. `mentioned_bot` is a boolean reporting whether the source platform's structural mention list for the current message contains the bot's `platform_user_id`.

For private/DM requests, neither `mentioned_bot` nor `group_attention` is added to the payload.

A metadata-only fast path returns `should_respond=False` when all three conditions hold:

- `group_attention == "chaotic_noise"`,
- `reply_context.reply_to_current_bot is not True`, and
- `mentioned_bot is not True`.

In every other case the LLM owns the decision.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Noisy-detection field | Use `channel_type` | `channel_name` is empty for adapters that do not fetch group display names; the structural channel kind is the right signal. |
| Mention surface | Single boolean `mentioned_bot` | Every consumer asks one question — "was the bot in the platform's structural mention list?" Sending a list ships information no consumer reads. |
| Mention boolean computed at adapter time | Adapter resolves `bot_id in <native @ segments>` once | Matches the bot-id-stable assumption; avoids re-deriving in every consumer. |
| Legacy text-marker parser | Delete | Reading `text.startswith("[Reply to message]")` and regex-extracting `<@id>` is deterministic parsing of user text and violates the `no-prepost-user-input` skill. |
| Noise descriptor cardinality | One label, four values | A single calibrated gradient is easier for the LLM to act on than several correlated sub-scores. |
| Bot's own messages in noise calculation | Exclude | Counting Kazusa's own replies inflates noise as a function of the over-response we are trying to dampen, creating a feedback loop. |
| Metadata-only short-circuit | Allowed at `chaotic_noise` only | Operates on `mentioned_bot` and `reply_context`, never user text. The LLM still owns judgment on every other case. |
| Soft enforcement otherwise | Descriptor is guidance, not a hard skip | Outside the chaotic short-circuit, the LLM remains the decision-maker. |

## Contracts

### `mentioned_bot`

A boolean indicating whether the source platform's structural mention list for the current message contains the bot's `platform_user_id`. Default `False`. Stored documents written before this change read with `mentioned_bot = False`.

The adapter is responsible for computing this once from native mention segments at request construction time. No downstream code may compute it from user text.

### `build_group_attention_context`

```python
def build_group_attention_context(
    *,
    chat_history_wide: list[dict],
    platform_bot_id: str,
) -> dict[str, str]:
    """Describe group attention state from recent history metadata only."""
```

Rules:

- Reads only `role`, `platform_user_id`, `timestamp`, `reply_context.*`, and `mentioned_bot` from history rows. Does not read `content`.
- Excludes rows where `platform_user_id == platform_bot_id` from "active participants" and "message density" signals.
- Internal calculations may use raw counts. The returned dict must contain exactly one LLM-facing key: `group_attention`, with a value in `{low_noise, medium_noise, high_noise, chaotic_noise}`.
- Returns `{}` when called in a non-group context by mistake (defensive only — caller should not invoke it for DMs).

### Internal Signal Definitions

The helper must compute, at minimum, these internal signals before label selection. Implementations may add more, but these are required so calibration is reproducible:

- `distinct_non_bot_speakers_in_window`: number of distinct `platform_user_id` values among non-bot user rows in the recent window.
- `non_bot_message_density`: count of non-bot user messages in the recent window divided by the window's wall-clock span (messages per minute, computed but not exposed).
- `messages_since_last_address_to_bot`: count of non-bot user messages since the most recent row where `reply_context.reply_to_current_bot is True` or `mentioned_bot is True`. Treated as unbounded if no such row exists in the window.

Label-selection thresholds are at the implementing agent's discretion, provided:

- A window with one non-bot speaker and a recent direct address yields `low_noise`.
- A window with three or more distinct non-bot speakers and no direct address yields at least `high_noise`.

### Metadata-only Short-Circuit

```text
if (group_attention == "chaotic_noise"
    and reply_context.reply_to_current_bot is not True
    and mentioned_bot is not True):
    return {"should_respond": False, ...}
```

This is the only deterministic skip beyond the existing structured reply gate. It does not inspect `user_input`.

### Relevance Prompt Update

Update `_RELEVANCE_SYSTEM_NOISY_PROMPT` only, with these additions:

- Explain that `mentioned_bot=true` means the source platform's structural mention list for the current message contains the bot's id; this is strong evidence of direct address. `mentioned_bot=false` is evidence against, but does not preclude direct address via reply or other context.
- Explain that `group_attention` is a calibrated gradient describing how busy the room is, with the four labels and the "raise the bar as noise rises" intent.
- Do not add semantic keyword examples that imply text matching.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Change `is_noisy_environment` to derive from `channel_type` instead of `channel_name`.
  - Add `build_group_attention_context` helper (or import from a sibling module if it grows past ~80 lines).
  - Replace `_should_ignore_third_party_reply` body so the only signals it consumes are `reply_context.*`, `mentioned_bot`, and `platform_bot_id`. Delete `_extract_legacy_reply_target_id`.
  - Add the metadata-only chaotic-noise short-circuit before the LLM call.
  - Inject `mentioned_bot` and `group_attention` into the noisy-group human payload.
  - Tune `_RELEVANCE_SYSTEM_NOISY_PROMPT` per the Contracts section.

- `src/adapters/napcat_qq_adapter.py`
  - Resolve the bot id against QQ `at` segment ids and CQ `at` ids at request construction; emit `mentioned_bot: bool` on outbound `ChatRequest`.

- `src/kazusa_ai_chatbot/service.py`
  - Add `mentioned_bot: bool` to `ChatRequest`, initial state construction, and saved conversation documents.

- `src/kazusa_ai_chatbot/state.py`
  - Add `mentioned_bot: bool` to the relevant `IMProcessState` typing.

- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add optional `mentioned_bot: bool = Field(default=False)` to `ConversationMessageDoc`.

- `src/kazusa_ai_chatbot/utils.py`
  - Preserve `mentioned_bot` in `trim_history_dict`.

- `tests/test_relevance_agent.py`
  - See Verification for the required cases.

### Delete

- `_extract_legacy_reply_target_id` in `relevance_agent.py`.
- The `channel_name`-based branch in `is_noisy_environment`.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2*`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- All RAG, consolidator, scheduler, and dispatcher modules.
- The non-noisy `_RELEVANCE_SYSTEM_PROMPT` and the private/DM payload shape.

## Implementation Order

1. **Add `mentioned_bot` typing and persistence first.** Extend `ChatRequest`, `IMProcessState`, `ConversationMessageDoc`, and `trim_history_dict`. With the field plumbed but unused, the rest of the changes can land safely.
2. **Wire the QQ adapter** to compute `mentioned_bot` from native mention segments against the bot id.
3. **Fix `is_noisy_environment`** to use `channel_type`.
4. **Replace `_should_ignore_third_party_reply`** with metadata-only signals; delete `_extract_legacy_reply_target_id`.
5. **Add `build_group_attention_context`** with the internal signals listed in Contracts.
6. **Inject the new fields** into the noisy-group human payload and update `_RELEVANCE_SYSTEM_NOISY_PROMPT`.
7. **Add the chaotic-noise metadata-only short-circuit**.
8. **Add tests** per the Verification section.

Step ordering matters: relevance changes (3–7) need `mentioned_bot` already typed; the prompt update needs the noisy environment to actually fire so payload-shape tests are meaningful.

## Verification

### Static Greps

- `rg "channel_name" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returns no matches.
- `rg "_extract_legacy_reply_target_id|\\[Reply to message\\]" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returns no matches.
- `rg "mentioned_bot" src tests` shows the field threaded through adapter, request, state, schema, utils, relevance agent, and tests.
- `rg "你|千纱|alias|nickname" src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Allowed: prompt prose that templates `{character_name}` or describes rules to the LLM.
  - Disallowed: any deterministic Python branch over these substrings of user text.

### Tests

Run with `pytest tests/test_relevance_agent.py -q`.

- Helper unit tests:
  - `low_noise` for a window with one non-bot speaker and a recent direct address.
  - `high_noise` for a window with three or more distinct non-bot speakers and no direct address.
  - `chaotic_noise` for a window saturated with non-bot speakers and no direct address.
  - Bot's own messages excluded from "active participants" and density.
- Payload-shape tests (LLM patched):
  - Group request payload contains `mentioned_bot` and `group_attention`.
  - Private request payload contains neither.
- Short-circuit tests:
  - Group request with `group_attention="chaotic_noise"`, no `reply_to_current_bot`, and `mentioned_bot=False` returns `should_respond=False` without invoking the LLM.
  - Same setup but with `mentioned_bot=True` invokes the LLM.
- Regression test for the QQ scenario:
  - Synthetic history modeled on `test_artifacts/qq_1082431481_recent_30_chat_history.json` with an ambiguous "你..." current message, no `reply_to_current_bot`, `mentioned_bot=False`. With LLM patched to a deterministic stub, the relevance agent does not return `should_respond=True`.

### Behavior Smoke

- Five hand-curated `(history, current_message)` cases run against the configured LLM with `temperature=0`, asserting the expected `should_respond` direction for each. Record results in Execution Evidence. Use these for ship-time confidence; do not gate CI on them unless infrastructure permits.

## Acceptance Criteria

This plan is complete when:

- `is_noisy_environment` derives from `channel_type`, not `channel_name`.
- `mentioned_bot` is plumbed end-to-end and populated by the QQ adapter from native mention segments resolved against the bot id.
- `_extract_legacy_reply_target_id` is deleted and the third-party reply guard reads only structured metadata.
- The noisy-group LLM payload contains `mentioned_bot` and exactly one `group_attention` label, with no raw counts.
- The chaotic-noise metadata-only short-circuit is in place and only triggers under the three documented conditions.
- All listed tests pass and the static greps return the expected results.
- Private/DM behavior is observably unchanged in tests.

## Rollback / Recovery

- Code rollback path: revert the commits across the listed change surface.
- Data rollback path: none required. `mentioned_bot` is a new optional field; documents written with the new field can be read by old code that ignores it, and old documents read with `mentioned_bot = False` under the new code.
- Irreversible operations: none.
- Recovery verification: `pytest tests/test_relevance_agent.py -q` passes.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Implementing agent re-introduces deterministic text parsing under pressure to "get it working" | Plan forbids it; static greps explicitly check for it | Verification grep block |
| QQ adapter cannot expose mention ids reliably | Plan agent must report blocker rather than fall back to text parsing | Manual review of adapter change |
| Noise label miscalibrated and chaotic-noise short-circuit fires too often | Internal signal definitions are pinned; chaotic threshold is the strictest of the four | Helper unit tests with explicit speaker/density expectations |
| `mentioned_bot=False` on legacy stored history evaluated as "no mention to bot" when the original platform mention existed | Acceptable: legacy windows roll over quickly; mitigation is to start populating immediately | Behavior smoke after rollout |
| Bot id changes in the future, invalidating stored `mentioned_bot` values | Out of scope per Assumptions; if the bot id ever changes, run a one-off recompute job at that time | N/A this plan |
| Private chat behavior changes accidentally | New fields added to group payload only; private payload assertion in tests | Payload-shape test for DM |

## Execution Evidence

To be filled during implementation:

- Static grep results:
- Test results:
- Manual payload review:
- Offline LLM smoke results (if run):
