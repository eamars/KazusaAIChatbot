# Extended Examples

Additional annotated before/after examples for the py-style constraints.

---

## P-001/N-001 -- Import placement

### Conditional heavy import (not an exception)

```python
# Wrong - lazy load for performance reasons is not a valid exception
def build_embedding(text: str) -> list[float]:
    import numpy as np   # not test code
    return np.array(...)

# Right - if numpy is optional, declare it at the top with a comment
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

def build_embedding(text: str) -> list[float]:
    if not _NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required for embedding")
    ...
```

---

## P-002 -- Minimum try range

### Multiple operations, one risky

```python
# Wrong
try:
    user_id = state["global_user_id"]        # cannot raise JSONDecodeError
    raw = response.content.strip()           # cannot raise JSONDecodeError
    parsed = json.loads(raw)                 # only this can raise
    logger.debug(f"Parsed: {parsed}")        # cannot raise JSONDecodeError
except json.JSONDecodeError as exc:
    logger.exception(f"Response was not valid JSON: {exc}")
    parsed = {}

# Right
user_id = state["global_user_id"]
raw = response.content.strip()
try:
    parsed = json.loads(raw)
except json.JSONDecodeError as exc:
    logger.exception(f"Response was not valid JSON: {exc}")
    parsed = {}
logger.debug(f"Parsed: {parsed}")
```

---

## N-002/P-007 -- Specific exception types and exception messages

### Catching multiple specific types

```python
# Right - use a tuple when multiple specific types are expected
try:
    resp = await client.post(url, json=payload, timeout=10)
except (httpx.TimeoutException, httpx.ConnectError) as exc:
    logger.warning(f"Request to {url} failed: {exc}")
    raise
```

### Re-raising after logging (acceptable pattern)

```python
# Right - catch, log, re-raise so the caller still sees the failure
try:
    embedding = await get_text_embedding(text)
except httpx.ConnectError as exc:
    logger.exception(f"Embedding service unreachable: {exc}; input={text[:80]!r}")
    raise
```

---

## P-003 -- Docstrings

### Async function with side effects

```python
async def save_memory(doc: MemoryDoc, timestamp: str) -> None:
    """Persist a new memory entry to the memory collection.

    Memory entries are append-only. Deduplication and superseding are handled
    at query time via the ``status`` field, not at write time.

    Args:
        doc: A well-formed memory payload produced by ``build_memory_doc``.
            Must contain at minimum ``memory_name``, ``content``, and
            ``memory_type``.
        timestamp: ISO-8601 UTC string recording when the memory was created
            or updated. Caller is responsible for generating this value.

    Returns:
        None. Raises on DB write failure.
    """
```

### Class method

```python
async def retrieve_if_similar(
    self,
    *,
    embedding: list[float],
    cache_type: str,
    global_user_id: str | None = None,
    threshold: float | None = None,
) -> dict | None:
    """Return the cached result whose stored embedding most closely matches the query.

    Scans in-memory entries of the given ``cache_type``. If ``global_user_id``
    is provided, only entries owned by that user are considered. Expired entries
    are lazily removed from the store during the scan.

    Args:
        embedding: Query vector to compare against stored entries.
        cache_type: Namespace key, e.g. ``"user_facts"`` or ``"internal_memory"``.
        global_user_id: When given, restricts the search to a single user's entries.
        threshold: Minimum cosine similarity to count as a hit. Defaults to the
            value set at construction time.

    Returns:
        A dict with keys ``cache_id``, ``similarity``, ``results``, and
        ``metadata`` if a match is found above threshold; ``None`` on a miss.
    """
```

---

## P-004/N-004 -- Default values

### Building a new document (right pattern)

```python
# config.py - single source of truth
AFFINITY_DEFAULT = 500

# db.py - default written once, at creation time
new_profile: UserProfileDoc = {
    "global_user_id": new_id,
    "affinity": AFFINITY_DEFAULT,   # written here
    "facts": [],
    ...
}
await db.user_profiles.insert_one(new_profile)

# everywhere that reads - plain indexing is safe because the default was set on write
affinity = doc["affinity"]          # crashes if the doc is malformed
```

### Acceptable optional config

```python
# Right - SCHEDULED_TASKS_ENABLED is genuinely optional; False is a valid operational state
enabled = config.get("SCHEDULED_TASKS_ENABLED", True)
```

---

## N-003 -- try-except scope

### LLM output (right - external source)

```python
response = await llm.ainvoke([system_msg, human_msg])
try:
    parsed = json.loads(response.content)
except json.JSONDecodeError as exc:
    logger.exception(
        f"LLM returned non-JSON: {exc}; input={user_input[:80]!r}"
    )
    parsed = {}
```

### File I/O on user-supplied paths (right - external source)

```python
try:
    with open(user_provided_path, "r") as f:
        data = f.read()
except FileNotFoundError as exc:
    raise UserInputError(
        f"File not found: {user_provided_path}: {exc}"
    ) from exc
except PermissionError as exc:
    raise UserInputError(
        f"Cannot read file: {user_provided_path}: {exc}"
    ) from exc
```

### Internal DB call (wrong - should propagate)

```python
# Wrong
try:
    await db.memory.insert_one(payload)
except Exception:
    logger.warning("DB write failed")   # silences a real failure

# Right - let it propagate; the service layer catches process-level errors
await db.memory.insert_one(payload)
```
