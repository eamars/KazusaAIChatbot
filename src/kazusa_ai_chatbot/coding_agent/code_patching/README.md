# Code Patching ICD

`code_patching` is the canonical patch mechanics boundary for the coding agent.
It consumes structured operations selected by writing or modifying workflows,
returns reviewable patch artifacts, and applies explicitly approved artifacts
only into managed apply copies.

## Ownership

- Validates repo-relative paths, operation kinds, anchors, and proposal caps.
- Compiles `create_file`, `insert_before`, `insert_after`, `replace`, and
  `replace_file_small` operations into unified-diff artifacts.
- Materializes review packages under a managed coding-agent workspace.
- Applies approved patch artifacts into
  `<workspace_root>/patch_apply/<apply_package_id>/source`.
- Returns patchability diagnostics and public-safe file summaries.

## Boundaries

- It does not apply patches to the caller source root or arbitrary target
  directories.
- It does not run target project commands, generated tests, package
  installation, or network calls.
- It requires structured approval and matching clean source identity before
  creating any managed apply workspace.
- It rejects missing or ambiguous existing-file anchors instead of guessing.
- It rejects mixed packages atomically when any operation is invalid.
- It rejects delete, rename, chmod, binary writes, unsafe paths, and secret-like
  paths.

## Public Entrypoints

```python
from kazusa_ai_chatbot.coding_agent.code_patching.patch_operations import (
    compile_patch_operations,
)
from kazusa_ai_chatbot.coding_agent.code_patching.apply import (
    apply_approved_patch,
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
`apply_approved_patch(...)` first runs the existing review validation against
the source root. Only patch artifacts that pass parsing, sandbox patchability,
and static review checks are copied to a managed apply workspace. It then runs
git patch checks plus git apply only inside that copy. Its public response
reports relative paths and an opaque managed workspace reference, not absolute
filesystem roots or command output.

`code_verifying` may call `apply_approved_patch(...)` multiple times for one
trusted verify-and-repair request. Each attempt must create a fresh managed
apply workspace; failed or timed-out verification does not authorize continued
mutation inside a previous apply copy.

`materialize_managed_candidate(...)` is internal. Its
`approved_verification` purpose requires structured human approval, while
`preapproval_preflight` is available only when explicitly enabled in
deployment. `resolved_source` copies a resolved checkout and
`empty_source_free` creates a Git-backed candidate for generated artifacts.
Managed copies provide process containment, not an operating-system sandbox.
