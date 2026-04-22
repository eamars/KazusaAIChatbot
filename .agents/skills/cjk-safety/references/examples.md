# CJK Safety: Annotated Examples

## Example 1 — Moving a helper function with CJK strings to a new module

**Scenario:** `persona_supervisor2_cognition_l2.py` contains `get_self_integrity_description()` with Chinese descriptions. You need to move it to `boundary_profile.py`.

### Wrong approach — re-typing through Write tool

```python
# boundary_profile.py  (BROKEN — curly quotes corrupted to ASCII ")
def get_self_integrity_description(self_integrity_score: float) -> str:
    descriptions = [
        "你的自我定义较强。你高度重视"我是谁"应由自己决定，...",  # SyntaxError: " became "
    ]
```

What happened: the Write tool converted `"我是谁"` (curly quotes U+201C/U+201D) to `"我是谁"` (ASCII U+0022), terminating the string at `"我是谁`.

### Right approach — extract bytes from source via script

```python
# Run this as a Bash tool call to create boundary_profile.py safely
python - << 'PYEOF'
with open('src/path/to/source.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Find function boundaries
start = src.index('def get_self_integrity_description(')
end = src.index('\ndef ', start + 1)
block = src[start:end].rstrip()

header = '"""Boundary profile helpers."""\n\n\n'
with open('src/path/to/boundary_profile.py', 'w', encoding='utf-8') as f:
    f.write(header + block + '\n')

import py_compile
py_compile.compile('src/path/to/boundary_profile.py', doraise=True)
print('OK')
PYEOF
```

The content never passes through the Write tool's text processing, so the curly quotes are preserved.

---

## Example 2 — Writing new CJK description lists

**Scenario:** Adding a new helper function with Chinese descriptions that include typographic quotes.

### Wrong — double-quoted strings with typographic quotes inside

```python
# BROKEN on write if Write tool converts " to "
def get_trust_level_description(score: float) -> str:
    descriptions = [
        "你对对方抱有基础信任，认为"言出必行"是底线。",   # will break
        "你对承诺的解读严格，"说到做到"是基本要求。",     # will break
    ]
    return descriptions[round(max(0.0, min(1.0, score)) * 10)]
```

### Right — single-quoted strings are immune

```python
# SAFE — single-quote delimiter never conflicts with typographic " inside
def get_trust_level_description(score: float) -> str:
    descriptions = [
        '你对对方抱有基础信任，认为"言出必行"是底线。',
        '你对承诺的解读严格，"说到做到"是基本要求。',
    ]
    return descriptions[round(max(0.0, min(1.0, score)) * 10)]
```

---

## Example 3 — Diagnosing the corruption after it occurs

**Symptom:**
```
SyntaxError: invalid character '"' (U+201D)
```
or
```
SyntaxError: invalid character '?' (U+2026)
```

**Diagnosis script:**
```python
# Find all lines in a file where ASCII " appears mid-string
with open('problem_file.py', 'rb') as f:
    raw = f.read()

lines = raw.split(b'\n')
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    # String lines: start and end with ", but have > 2 total " = corruption
    if stripped.startswith(b'"') and stripped.count(b'"') > 2:
        print(f'line {i}: potentially corrupted')
```

**Fix options (in order of preference):**
1. If source exists in git: use `git show HEAD:path/to/file.py` to extract and byte-copy (Example 1)
2. If content is new: rewrite with single-quoted delimiters (Example 2)
3. If neither: replace typographic quotes with Unicode escapes `\u201c` / `\u201d`

---

## Example 4 — Safe file open on Windows

**Wrong — uses system codepage (cp1252 on Windows), fails on CJK:**
```python
with open('src/nodes/boundary_profile.py') as f:
    src = f.read()
```

**Right — explicit UTF-8:**
```python
with open('src/nodes/boundary_profile.py', encoding='utf-8') as f:
    src = f.read()

# pathlib equivalent
import pathlib
src = pathlib.Path('src/nodes/boundary_profile.py').read_text(encoding='utf-8')
```

---

## Example 5 — Post-write syntax verification

Always run after writing a CJK Python file:

```bash
python -c "
import ast, pathlib
src = pathlib.Path('src/kazusa_ai_chatbot/nodes/boundary_profile.py').read_text(encoding='utf-8')
ast.parse(src)
print('Syntax OK')
"
```

If this passes, the file is safe to import. If it raises `SyntaxError`, apply the diagnosis from Example 3.
