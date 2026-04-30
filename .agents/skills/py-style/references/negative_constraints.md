# Negative Python Constraints

Negative constraints define boundaries the code must not cross. Apply every
`N-XXX` constraint together with PEP 8 and the positive constraints. When a
negative constraint conflicts with a positive constraint, the negative
constraint wins.

---

## N-001 -- Do not hide imports inside runtime code

Inline imports inside functions, methods, or class bodies obscure dependencies
and slow down readers who need to understand what the module requires.

Test code, pytest fixtures/functions, and `if __name__ == "__main__"` blocks may
use inline imports for heavy or optional dependencies.

**Forbidden:**
```python
async def save_memory(doc):
    from datetime import datetime
    ...
```

**Correct:**
```python
from datetime import datetime


async def save_memory(doc):
    ...
```

---

## N-002 -- Do not use bare or broad exception handlers in application logic

Bare `except:` and broad `except Exception:` hide the expected failure mode.
They are only acceptable at genuine process boundaries, such as a top-level
adapter event handler or service main loop, where crashing the entire process
would be worse than logging and continuing.

Boundary catches must use `logger.exception(...)` and include the exception
message.

**Forbidden:**
```python
try:
    value = int(user_input)
except:
    value = 0
```

**Forbidden:**
```python
try:
    value = int(user_input)
except Exception:
    value = 0
```

**Correct:**
```python
try:
    value = int(user_input)
except ValueError:
    value = 0
```

When reviewing, flag every bare `except` and `except Exception` unless it is a
logged process-boundary catch.

---

## N-003 -- Do not use `try-except` as default error handling

Internal crashes are bug reports. Wrapping internal logic in `try-except`
silences the exact place where an assumption was wrong.

Use `try-except` only for genuinely external uncertainty: malformed LLM output,
network I/O, external APIs, file system operations on user-supplied paths, and
OS-level calls.

**Forbidden:**
```python
try:
    result = compute_embedding(text)
except Exception:
    result = []
```

**Forbidden:**
```python
try:
    user_id = state["global_user_id"]
except KeyError:
    user_id = "anonymous"
```

**Correct:**
```python
try:
    parsed = json.loads(llm_response)
except json.JSONDecodeError as exc:
    logger.exception(f"LLM returned non-JSON: {exc}; raw={llm_response!r}")
    parsed = {}
```

When reviewing, ask whether the caught exception comes from outside the
codebase. If not, remove the block and let the bug surface.

---

## N-004 -- Do not scatter internal defaults with `.get(key, fallback)`

Fallbacks at read sites can silently diverge and mask missing upstream writes.
For required internal data, put the default at the point of definition and use
plain indexing at read sites.

**Forbidden:**
```python
affinity = doc.get("affinity", 500)
affinity = state.get("affinity", 500)
affinity = profile.get("affinity", 500)
```

**Correct:**
```python
affinity = profile["affinity"]
```

`.get(key, fallback)` remains acceptable for genuinely external or optional
data: LLM JSON output, API responses, user-provided dictionaries, or optional
config.

---

## N-005 -- Do not chain nested dictionary retrieval

Nested retrieval chains compress several lookups into one expression and make
debugging harder. One statement should perform at most one `.get(...)` lookup.
Split traversal into named intermediate variables.

**Forbidden:**
```python
user_memory_context = (
    (global_state.get("rag_result", {}).get("user_image") or {})
    .get("user_memory_context")
    or {}
)
```

**Correct:**
```python
rag_result = global_state["rag_result"]
user_image = rag_result["user_image"]
user_memory_context = user_image["user_memory_context"]
```

This constraint does not make `.get(...)` acceptable for required internal
data. Required internal data should still use plain indexing.

---

## N-006 -- Do not return call expressions directly

Returning a function call, awaited call, constructor, or other compound
expression directly makes the produced value unavailable for debugging before
the function exits.

**Forbidden:**
```python
return project_user_memory_units(units, budget=budget)
```

**Correct:**
```python
projected_units = project_user_memory_units(units, budget=budget)
return projected_units
```

Simple attribute and index returns are allowed:

```python
return profile.affinity
return profile["affinity"]
```

---

## N-007 -- Do not omit exception text when reporting failures

Tracebacks are not a substitute for the concrete exception message. Any handler
that logs, prints, wraps, or reports an exception must include `{exc}` in the
message unless it immediately re-raises with bare `raise`.

**Forbidden:**
```python
try:
    await call_external_service()
except TimeoutError:
    logger.exception("External service failed")
```

**Correct:**
```python
try:
    await call_external_service()
except TimeoutError as exc:
    logger.exception(f"External service failed: {exc}")
```

---

## N-008 -- Do not use `%`-style string interpolation

Do not use legacy `%` formatting, and do not rely on logging's comma-argument
`%s` substitution style. Use f-strings instead.

**Forbidden:**
```python
logger.info("User %s affinity changed to %d", user_id, affinity)
message = "User %s" % user_id
```

**Correct:**
```python
logger.info(f"User {user_id} affinity changed to {affinity}")
message = f"User {user_id}"
```

---

## N-009 -- Do not separate LLM prompts from their handlers

Do not group unrelated prompt constants at the top of a file while placing the
handlers far below. Do not route distinct LLM stages through a generic helper
that hides the prompt, model, payload, parser, and validation relationship.

**Forbidden:**
```python
_FIRST_PROMPT = "..."
_SECOND_PROMPT = "..."
_shared_llm = get_llm(...)


async def first_handler(state):
    response = await _shared_llm.ainvoke(...)
```

**Correct:**
```python
_FIRST_PROMPT = "..."
_first_llm = get_llm(...)


async def first_handler(state):
    response = await _first_llm.ainvoke(...)
```

---

## N-010 -- Do not hard-code character names in reusable runtime prompts

Prompt examples and instructions are sticky for local LLMs. Reusable runtime
prompts must not hard-code a concrete character name unless that exact name is
part of a test fixture or fixed source text.

Use role-neutral wording such as "the active character", "the character", or
"the speaker" when role ownership is enough. If identity distinction is needed,
use a runtime placeholder such as `{character_name}` and pass the value from
state.

**Forbidden:**
```python
MEMORY_PROMPT = "Kazusa should summarize the user's preference."
```

**Correct:**
```python
MEMORY_PROMPT = "The active character should summarize the user's preference."
```

---

## N-011 -- Do not duplicate existing helper behavior

Adding a second function for behavior the project already implements creates
parallel contracts that drift apart. Search first, including private helpers.

**Forbidden:**
```python
def normalize_memory_text(text: str) -> str:
    return " ".join(text.strip().split())
```

when the project already has:

```python
def _normalize_text_for_memory(text: str) -> str:
    ...
```

**Correct:**
```python
normalized_text = _normalize_text_for_memory(raw_text)
```

---

## N-012 -- Do not add thin wrappers around a single simple expression

Call count alone does not justify a thin wrapper. Do not add helper functions
whose body only forwards to a single simple expression, even when that
expression appears several times.

This includes wrappers around field access, dictionary access, model
serialization, constructors, type casts, or pass-through function calls. These
wrappers add another name for readers to chase without adding behavior,
validation, or a useful abstraction.

**Forbidden:**
```python
def _request_message_envelope(req: ChatRequest) -> MessageEnvelope:
    """Return the adapter-provided typed message envelope."""

    envelope: MessageEnvelope = req.message_envelope.model_dump(
        exclude_none=True,
    )
    return envelope


message_envelope = _request_message_envelope(req)
```

**Correct:**
```python
message_envelope: MessageEnvelope = req.message_envelope.model_dump(
    exclude_none=True,
)
```

When reviewing, reject thin wrappers around a single simple expression
regardless of how many call sites they have.

---

## N-013 -- Do not mention specific development plans in code

Do not mention a specific development plan, implementation phase, migration
phase, batch name, ticket-only milestone, or temporary planning label anywhere
in Python source code. This ban includes docstrings, inline comments, block
comments, and explanatory string constants whose purpose is to describe the
code.

Plan labels such as "Phase-F", "Phase 2", "Plan B", "migration step 3", or
"per the April refactor plan" only make sense during one batch of work. Later
plans can reuse the same labels for different meanings, so these references
become misleading archaeology instead of documentation.

Describe the stable runtime behavior, domain concept, or invariant instead.

**Forbidden:**
```python
def _keyword_text_filter(pattern: str) -> dict[str, Any]:
    """Build the Phase-F regex filter for typed conversation text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {"body_text": regex_filter}
    return filter_doc
```

**Forbidden:**
```python
# Phase 3 compatibility path for the queue migration.
queued_items = load_pending_items()
```

**Correct:**
```python
def _keyword_text_filter(pattern: str) -> dict[str, Any]:
    """Build a case-insensitive filter for typed conversation text."""

    regex_filter = {"$regex": pattern, "$options": "i"}
    filter_doc: dict[str, Any] = {"body_text": regex_filter}
    return filter_doc
```

**Correct:**
```python
# Preserve queued input order while loading pending items.
queued_items = load_pending_items()
```

When reviewing, scan docstrings and comments for plan-specific vocabulary and
ask whether the sentence would still be accurate without access to the plan
that created it. If not, replace it with stable code-facing language.
