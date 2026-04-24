

Three concrete bugs exist in `persona_supervisor2_rag.py` that should be fixed before the Phase 1 planner refactor, since they affect every request today.

### Bug 1 — `_assemble_image_text` renders recent observations as raw dict repr

**File:** `persona_supervisor2_rag.py:650`
    # Current (broken):
    for obs in recent_window:
        parts.append(f"- {obs}")  # obs is {"timestamp": ..., "summary": ...} → renders as dict

    # Fix:
    for obs in recent_window:
        parts.append(f"- {obs.get('summary', '')}")

`recent_window` entries are `{"timestamp": ..., "summary": ...}` dicts. The current code serializes them as Python dict repr into the LLM context, so the character reads `- {'timestamp': '2026-04-01 21:00', 'summary': '...'}` instead of the actual observation text. This means the user_image recent_window is silently broken in every request.

### Bug 2 — Early-exit threshold is computed but never enforced

**File:** `persona_supervisor2_rag.py:538-582, 817-820`

`INTERNAL_RAG_STRONG_THRESHOLD = 0.55` is recorded in `metadata["early_exit"]` as a post-hoc label, but `_build_rag_graph()` adds both dispatcher edges at **compile time**, before any retrieval happens. External RAG always runs in parallel with internal RAG when affinity allows — there is no actual early-exit path.

Fix options (pick one): drop `INTERNAL_RAG_STRONG_THRESHOLD` and the `early_exit` metadata key entirely until the Phase 5 evaluator is in place to enforce it properly.

### Bug 3 — `_result_confidence` is a text-length proxy with no semantic grounding

**File:** `persona_supervisor2_rag.py:430-454`
    return min(1.0, len(text) / 600.0 + 0.2)

A verbose but empty retrieval ("no relevant results found, the user asked about...") scores identically to a substantive 600-character answer. This drives cache storage decisions and the `early_exit` metadata label with a meaningless signal.

Fix: The dispatchers should emit an explicit `confidence` field alongside `next_action`. Until that lands, at minimum add a no-result guard — if the text contains known empty-retrieval markers ("未找到", "no results", empty after strip) return 0.0 explicitly.

### Consolidator dead code

**File:** `persona_supervisor2_consolidator_knowledge.py:50-208`

`_update_user_image()` and `_update_character_image()` are fully duplicated in this file but reference constants and helpers defined only in `consolidator_images.py` (`_apply_milestone_lifecycle`, `_USER_IMAGE_MAX_RECENT_WINDOW`, `_user_image_session_summary_llm`, etc.). The functions are unreachable from any call site but would fail with `NameError` if called. Delete lines 50–208 from `consolidator_knowledge.py`.

* * *

## Refactor: `_assemble_image_text` → Structured Dict

Bug 1 (dict repr in recent_window) is a symptom of a broader format problem: `_assemble_image_text` returns a markdown string that is then embedded as a value inside a `json.dumps()` HumanMessage. The LLM receives JSON with markdown strings inside it — mixed format throughout `research_facts`.

### Why structured dict is better

`research_facts` is always serialized via `json.dumps()` in cognition nodes (L2 line 250, L3 lines 374/509) and the consolidator facts harvester (line 135). The image fields should be nested JSON objects, not opaque markdown strings, for three reasons:

1. **Deduplication quality** — the facts harvester checks `research_facts.user_image` to decide if a candidate fact is already known. With markdown prose the LLM must parse text; with structured keys (`milestones`, `recent_observations`) it can compare directly.
2. **Consistent format** — eliminates the markdown-in-JSON mixing. The entire HumanMessage is uniform JSON.
3. **Fixes Bug 1 as a natural consequence** — recent_window summary strings are extracted when building the list, not rendered as dict repr.

### Target shape

    {
        "milestones": [
            {"event": "...", "category": "relationship_state", "superseded_by": None}
        ],
        "historical_summary": "...",          # narrative prose, kept as string
        "recent_observations": ["...", "..."] # summary strings only, not raw dicts
    }

Milestones and recent_observations are structured; the text content within them (`event`, `historical_summary`, summary strings) remains natural language — no semantic loss.

### What changes

| Location                                                        | Change                                                                                 |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `persona_supervisor2_rag.py`                                    | Rename `_assemble_image_text` → `_build_image_context`; return `dict` instead of `str` |
| `persona_supervisor2_rag.py` (cache hit path, ~line 733)        | Same function call — already returns dict, no extra work                               |
| `persona_supervisor2_cognition_l3.py:504-506`                   | Change default from `""` to `{}` for `user_image` / `character_image`                  |
| `persona_supervisor2_cognition_l2.py` prompt schema             | Update `"user_image": "用户画像..."` description to show the nested structure              |
| `persona_supervisor2_cognition_l3.py` prompt schemas (2 places) | Same prompt schema update                                                              |

The `research_facts` TypedDict (or dict shape) does not need a structural change — `user_image` and `character_image` simply hold a `dict` instead of a `str`. No other callers are affected.

* * *

## Input Format Consistency Issues

A broader audit of HumanMessage construction across the project found three additional format issues.

### Issue A — `character_name` is a 1-tuple in both RAG dispatchers

**File:** `persona_supervisor2_rag.py:194` and `persona_supervisor2_rag.py:310`**Severity: Correctness** — affects every request
    # Both external_rag_dispatcher and input_context_rag_dispatcher have:
    character_name=state["character_profile"]["name"],   # trailing comma!

The trailing comma makes `character_name` a 1-tuple `("Kazusa",)` instead of the string `"Kazusa"`. When passed to `.format(character_name=character_name)`, Python calls `str()` on it, producing `('Kazusa',)` in the prompt. The LLM reads the character name as a tuple repr in every dispatcher call.

Fix: remove the trailing comma on both lines.

### Issue B — `external_rag_results` / `input_context_results` are `list[str]` but treated as `str` downstream

**File:** `persona_supervisor2_rag.py:798-799`**Severity: Moderate** — type inconsistency in `research_facts`
    input_context_results = result.get("input_context_results", "")   # default: str
    external_rag_results  = result.get("external_rag_results", "")    # default: str

The RAGState TypedDict declares these as `Annotated[list[str], operator.add]`. When a dispatcher fires, the values are `["some text"]` (a list). When no dispatcher fires, the `.get()` default returns `""` (a string). So `research_facts["input_context_results"]` is sometimes `""` and sometimes `["some text"]`, and the cognition prompts describe both as plain strings.

Fix: join the list to a single string before storing in `research_facts`:
    input_context_results = "\n".join(result.get("input_context_results") or [])
    external_rag_results  = "\n".join(result.get("external_rag_results") or [])

### Issue C — `diary_entry` reads `user_profile["facts"]` (structured list) into a field the prompt treats as narrative text

**File:** `persona_supervisor2_cognition_l2.py:230`**Severity: Moderate** — prompt/data mismatch
    diary_entry = state["user_profile"]["facts"][:10]   # list of ObjectiveFactEntry dicts
    # ...
    "diary_entry": diary_entry,  # serialized as JSON array, prompt expects narrative string

The prompt at line 197 describes `diary_entry` as `"上一篇主观日记的内容"` (narrative diary content). But `user_profile["facts"]` is the `objective_facts` list — structured `{fact, category, timestamp}` dicts. Two sub-issues:

1. **Wrong field** — this should probably be `user_profile.get("character_diary", [])` or a selected recent diary entry, not `objective_facts`.
2. **Wrong type** — even if the field is correct, passing a raw list of dicts where the prompt expects narrative text is inconsistent.

Requires clarification on intent before fixing: is `diary_entry` here meant to be the character's subjective diary notes or the objective facts list? The field name and prompt description say diary; the code reads facts.
