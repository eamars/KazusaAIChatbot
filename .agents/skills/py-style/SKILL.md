---
name: py-style
description: Enforce and review Python coding style for this project. PEP 8 is the baseline, with the project's explicit positive and negative constraints applied on top. Use this skill whenever you are writing new Python code, reviewing existing Python files, or responding to feedback about code quality. Trigger on any task that involves creating or modifying .py files, reviewing a function or class, or when the user asks about code style, best practices, or refactoring. Always apply these rules proactively -- don't wait for the user to ask.
---

# Python Style Guide

PEP 8 is the baseline style guide for all Python code in this project. On top
of PEP 8, this skill enforces the project's positive and negative constraints.
Apply both layers proactively when writing Python code, and surface every
violation when reviewing code.

When PEP 8 and a project-specific constraint overlap, follow the stricter
project-specific constraint. When PEP 8 covers an issue not mentioned here,
follow PEP 8.

The constraints exist to keep code readable, debuggable, and honest about
failure. Clever workarounds that hide bugs are more expensive than crashes that
reveal them immediately.

---

## Constraint Architecture

Constraints are organized into two reference files that can grow independently.
Anyone on the team can add new constraints by appending to the relevant file --
no renumbering, no restructuring, no touching `SKILL.md`.

```text
py-style/
|-- SKILL.md
`-- references/
    |-- positive_constraints.md
    `-- negative_constraints.md
```

### How to read the constraints

Before writing or reviewing Python code, read both constraint files:

- `references/positive_constraints.md` -- the full list of positive constraints
  (`P-XXX`), which describe what to do.
- `references/negative_constraints.md` -- the full list of negative constraints
  (`N-XXX`), which describe what not to do.

Apply every constraint from both files. Positive and negative constraints have
equal weight: a positive constraint tells you the right way; a negative
constraint tells you the boundary you must not cross.

### How to add a new constraint

1. Open the appropriate file: `positive_constraints.md` or
   `negative_constraints.md`.
2. Append a new section at the end using the next available ID, such as
   `P-014` or `N-013`.
3. Follow the existing format: ID in the heading, a rationale explaining why,
   and Wrong/Right or Forbidden/Correct examples where applicable.
4. Do not renumber or reorder existing constraints.
5. Do not modify `SKILL.md`; it automatically picks up new constraints from the
   reference files.

### When constraints conflict

If a positive and negative constraint appear to conflict, the negative
constraint wins; boundaries are harder limits than preferences. If two
constraints of the same type conflict, surface the conflict to the user with
reasoning rather than silently picking one path.

---

## Review Workflow

When asked to review a file or selection:

1. Read both `references/positive_constraints.md` and
   `references/negative_constraints.md` in full.
2. Read the code to review in full.
3. Check PEP 8 violations first, then check every positive and negative
   constraint.
4. For each violation, state the line reference, the constraint ID, and a
   one-sentence explanation.
5. Propose the corrected version inline.
6. If no violations are found, say so explicitly and mention any remaining test
   gaps or residual risk.

## Writing Workflow

When writing new Python code:

1. Read both constraint files in full before editing.
2. Apply PEP 8 and every positive and negative constraint before producing code.
3. If a constraint conflict arises or a genuine exception is needed, surface it
   to the user with reasoning rather than silently picking one path.

`references/examples.md` remains supplemental reading. The active style
contract lives in the positive and negative constraint files.
