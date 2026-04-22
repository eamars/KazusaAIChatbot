---
name: cjk-safety
description: Enforce safe handling of CJK characters — especially Chinese/Japanese quotation marks — when writing or editing Python source files. Apply whenever writing a .py file that contains CJK string content, or whenever a Python file with CJK content is being created, moved, or refactored. Trigger immediately if a Write tool call would produce a file containing CJK typographic quotes inside Python string literals delimited by ASCII double quotes.
---

# CJK Character Safety in Python Source Files

This skill prevents encoding corruption that occurs when CJK typographic characters — especially Chinese curly quotation marks — are written into Python source files via the Write tool. The Write tool may silently convert Unicode typographic quotes to their ASCII lookalikes, breaking string literals and causing `SyntaxError` at import time.

The rules below are ordered by impact. Apply all of them when writing or refactoring Python files with CJK content.

---

## Rule 1 — Use single-quoted string delimiters when the string content contains CJK typographic quotes

CJK text frequently contains typographic double quotation marks: `"` (U+201C) and `"` (U+201D). These are visually identical to ASCII `"` (U+0022). When the Write tool processes your response text, it may silently convert them to ASCII `"`, which then terminates the Python string prematurely, producing a `SyntaxError`.

**Prevent this by using single-quoted `'...'` string delimiters for any Python string whose content contains or might contain CJK typographic quotes.**

**Wrong — ASCII double-quote delimiter, CJK content with typographic quotes inside:**
```python
descriptions = [
    "你对控制信号较为敏感。对方一旦使用"必须""立刻"等结构，...",  # breaks if " converts to "
]
```

**Right — single-quote delimiter is immune to the conversion:**
```python
descriptions = [
    '你对控制信号较为敏感。对方一旦使用"必须""立刻"等结构，...',  # safe
]
```

**Also right — explicit Unicode escape if the value must be double-quoted:**
```python
descriptions = [
    "你对控制信号较为敏感。对方一旦使用\u201c必须\u201d\u201c立刻\u201d等结构，...",
]
```

When writing new Python files with CJK description lists, prompt arrays, or any multi-line string data: default to single-quoted strings throughout the list.

---

## Rule 2 — Never copy CJK string content through the Write tool when an exact-bytes source exists

If the CJK content already exists correctly in a file in the repository (e.g., a function being moved to a new module), **do not re-type or copy-paste the content through a Write tool call**. The round-trip through your response text is where corruption occurs.

Instead, extract and write the bytes directly using a Python script that reads the source file and writes the destination file without passing through tool text processing:

```python
# Extract verbatim from source, write to destination
with open('source_file.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Extract the relevant block (e.g., a function definition)
start = src.index('def my_function(')
end = src.index('\ndef ', start + 1)
block = src[start:end]

header = '"""Module docstring."""\n\n'
with open('destination_file.py', 'w', encoding='utf-8') as f:
    f.write(header + block + '\n')
```

This pattern is mandatory when:
- Refactoring CJK-heavy functions into new modules
- Splitting files that contain CJK prompt strings
- Duplicating or extending existing CJK string arrays

---

## Rule 3 — Verify syntax immediately after writing any Python file with CJK content

After every Write or Edit operation on a Python file that contains CJK strings, run a syntax check before reporting the task as complete:

```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('path/to/file.py').read_text(encoding='utf-8')); print('OK')"
```

Or using `py_compile`:
```bash
python -m py_compile path/to/file.py && echo OK
```

If the check fails with `SyntaxError: invalid character` pointing at a CJK position, the corruption has occurred. Do not proceed — fix the file using Rule 1 or Rule 2 before continuing.

---

## Rule 4 — Explicit `encoding='utf-8'` on all file open calls that touch CJK content

On Windows, Python's default file encoding is the system codepage (often `cp1252` or `gbk`), not UTF-8. Any `open()` call that reads or writes CJK text must specify `encoding='utf-8'` explicitly.

**Wrong:**
```python
with open('prompts.py') as f:         # uses system encoding on Windows
    content = f.read()
```

**Right:**
```python
with open('prompts.py', encoding='utf-8') as f:
    content = f.read()
```

This applies to:
- Scripts that read Python source files to extract prompts
- Config loaders that read JSON/YAML containing CJK strings
- Any `pathlib.Path.read_text()` — use `read_text(encoding='utf-8')`

---

## Rule 5 — Know the CJK characters at highest risk

These characters are the most common sources of corruption in this codebase:

| Character | Unicode | UTF-8 bytes | Risk |
|---|---|---|---|
| `"` Left double quotation mark | U+201C | `\xe2\x80\x9c` | **Critical** — converts to ASCII `"`, breaks string delimiters |
| `"` Right double quotation mark | U+201D | `\xe2\x80\x9d` | **Critical** — same as above |
| `'` Left single quotation mark | U+2018 | `\xe2\x80\x98` | High — converts to ASCII `'`, breaks single-quoted strings |
| `'` Right single quotation mark | U+2019 | `\xe2\x80\x99` | High — same as above |
| `…` Horizontal ellipsis | U+2026 | `\xe2\x80\xa6` | Medium — valid in strings, but corrupts if file encoding fails |
| `——` Em dash (double) | U+2014×2 | `\xe2\x80\x94` × 2 | Low — valid in strings |

Typographic quotes (`"` `"` `'` `'`) are the critical risk. They appear throughout the Chinese prompt descriptions in this codebase and will silently corrupt Python string literals if the Write tool converts them to their ASCII equivalents.

---

## Application checklist

Before writing any Python file with CJK content:

1. Does the content come from an existing file in the repo? → Use Rule 2 (byte-copy via script)
2. Are you writing new CJK string lists? → Use Rule 1 (single-quoted delimiters)
3. After writing: → Run Rule 3 syntax check
4. Any `open()` calls in helper scripts? → Apply Rule 4 (`encoding='utf-8'`)

See `references/examples.md` for annotated before/after examples of the refactoring pattern.
