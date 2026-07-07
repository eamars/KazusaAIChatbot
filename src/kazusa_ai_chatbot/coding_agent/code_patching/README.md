# Code Patching ICD

`code_patching` is the canonical inert patch assembly boundary for the coding
agent. It consumes structured operations selected by writing or modifying
workflows and returns reviewable patch artifacts.

## Ownership

- Validates repo-relative paths, operation kinds, anchors, and proposal caps.
- Compiles `create_file`, `insert_before`, `insert_after`, `replace`, and
  `replace_file_small` operations into unified-diff artifacts.
- Materializes review packages under a managed coding-agent workspace.
- Returns patchability diagnostics and public-safe file summaries.

## Boundaries

- It does not apply patches to the caller workspace.
- It does not run target project commands, generated tests, package
  installation, or network calls.
- It rejects missing or ambiguous existing-file anchors instead of guessing.
- It rejects mixed packages atomically when any operation is invalid.
- It rejects delete, rename, chmod, binary writes, unsafe paths, and secret-like
  paths.

## Public Entrypoints

```python
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patch_validation import (
    materialize_patch_artifacts_for_review,
)
from kazusa_ai_chatbot.coding_agent.code_patching.patcher import (
    materialize_patch_artifacts,
)
```

`compile_patch_operations(...)` owns deterministic patch assembly.
`materialize_patch_artifacts_for_review(...)` writes only to managed review
storage and uses git patch checks plus static artifact inspection. Generated or
target-project code remains inert.
