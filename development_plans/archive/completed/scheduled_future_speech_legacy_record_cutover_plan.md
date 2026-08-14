# Scheduled Future Speech Legacy Record Cutover

- Status: completed
- Plan type: active bugfix cutover
- Parent plan: `scheduled_future_speech_temporal_grounding_and_content_gate_plan.md`
- User authorization: explicit authorization to remove the legacy records in
  the current conversation on 2026-08-15.
- Final independent review:
  [post-big-bang scheduled future-speech review](../../../test_artifacts/diagnostics/scheduled_future_speech_final_review_post_bigbang_20260815.md)
- Execution boundary: the exact records listed below; no broader collection
  clear or compatibility path.

## Big-bang deletion policy

The user explicitly amended the cutover to a big-bang change and directed that
no rollback or backward-compatibility implementation be added. The earlier
state-preserving retirement left these same exact rows in identifiable legacy
states. This cutover deletes those five rows directly after compare-and-delete
validation.

| Area | Policy | Instruction |
| --- | --- | --- |
| Accepted task | exact hard deletion | Delete only the task ID below when its v2 `future_speak` identity, `enqueue_failed` state, cutover reason, and missing authority all match. |
| Calendar schedules | exact hard deletion | Delete only the four schedule IDs below when their cancelled state, cutover reason, `future_cognition` trigger, marker, idempotency key, and missing authority all match. |
| Calendar runs | preserve | The four linked runs are terminal historical rows and remain unchanged. |
| Background jobs | preserve | The bounded inventory contains no linked job; any drift blocks deletion. |
| Generic future cognition | preserve | Rows without the structural future-speech source marker remain out of scope. |

The script performs no restoration, snapshot, compatibility mapping, aliasing,
fallback, migration, or broad cleanup. The terminal runs remain audit history;
they are not active legacy runtime records targeted by this deletion.

## Exact approved targets

Accepted task:

- `task-2c1831a6217342d7a5a24743d8eae669`

Calendar schedules:

- `calendar_schedule_4944be41cc53d33b443640d10e2e7226`
- `calendar_schedule_49f5cad88af6d137fab09c108d603717`
- `calendar_schedule_59770930065b92900ffc676af106b457`
- `calendar_schedule_b812cbbd86f99e01505e319213ae0e5c`

Required structural discriminator for every schedule is
`payload.source_refs[].ref_id == "future_speak_background_work"`.

The exact legacy cutover reason is
`scheduled_future_speech_legacy_cutover_2026-08-15`.

## Preconditions and safety gates

The cutover script refuses to apply when any condition fails:

- the accepted task target is exactly one v2 `future_speak` row in the
  cutover `enqueue_failed` state, with the exact cutover reason and without
  scheduled authority;
- the schedule target set is exactly the four IDs above, each in the cutover
  `cancelled` state with the exact reason, `future_cognition` trigger,
  marker-bearing payload, and no scheduled authority;
- no linked `background_work_jobs` row exists;
- no linked `calendar_runs` row is pending or running; and
- the exact deletion confirmation phrase is supplied for apply.

The deletion filters repeat the identity and state checks at write time. A
compare-and-delete that matches zero rows fails closed for that target.

## Execution and verification

Dry-run:

~~~powershell
venv\\Scripts\\python scripts\\cutover_scheduled_future_speech_legacy_records.py
~~~

Apply, after the dry-run reports `ready=true`:

~~~powershell
venv\\Scripts\\python scripts\\cutover_scheduled_future_speech_legacy_records.py --apply --confirm DELETE_SCHEDULED_FUTURE_SPEECH_LEGACY_RECORDS
~~~

Post-deletion verification:

~~~powershell
venv\\Scripts\\python scripts\\preflight_scheduled_future_speech_contract.py
~~~

The apply operation has no rollback or restoration command by explicit user
direction. Its postcondition is zero remaining target rows, unchanged terminal
run IDs, and a passing read-only preflight.

## Test impact and traceability

| Governed path | Changed symbol or contract | Semantic owner | Exact pytest node | Test mode | Regression prevented |
| --- | --- | --- | --- | --- | --- |
| `scripts/cutover_scheduled_future_speech_legacy_records.py` | exact target validation, dry-run report, compare-and-delete filters, and post-delete verification | release cutover | `tests/test_scheduled_future_speech_legacy_cutover.py::test_exact_legacy_target_set_is_ready_for_cutover`; `tests/test_scheduled_future_speech_legacy_cutover.py::test_linked_job_blocks_cutover`; `tests/test_scheduled_future_speech_legacy_cutover.py::test_active_linked_run_blocks_cutover`; `tests/test_scheduled_future_speech_legacy_cutover.py::test_report_does_not_mutate_loaded_documents`; `tests/test_scheduled_future_speech_legacy_cutover.py::test_delete_filters_bind_exact_legacy_identity_and_provenance` | deterministic unit plus live operational apply | Target drift, active carrier races, and broad destructive clears cannot pass the deletion boundary. |
| `scripts/preflight_scheduled_future_speech_contract.py` | post-deletion active inventory verification | release gate | `tests/test_scheduled_future_speech_preflight.py::test_preflight_passes_when_active_records_carry_authority`; `tests/test_scheduled_future_speech_preflight.py::test_preflight_ignores_generic_future_cognition_rows` | deterministic plus live operational verification | The parent deployment gate cannot close while incompatible active records remain. |

## Acceptance

- [x] Dry-run reports the exact five target rows and no linked job or active run.
- [x] Apply deletes exactly one accepted task and four schedules using
  compare-and-delete filters.
- [x] No target task or schedule row remains after apply.
- [x] Terminal calendar runs remain unchanged.
- [x] Post-deletion preflight exits 0 with zero incompatible active records.
- [x] Parent plan evidence is updated and independently re-reviewed.

## Execution evidence — 2026-08-15

- Deterministic cutover tests:
  `venv\\Scripts\\python -m pytest tests\\test_scheduled_future_speech_legacy_cutover.py -q`
  passed 5 tests.
- Dry-run command:
  `venv\\Scripts\\python scripts\\cutover_scheduled_future_speech_legacy_records.py`
  reported `ready=true`, one exact `enqueue_failed` task, four exact
  `cancelled` schedules, zero linked jobs, four terminal runs, and no
  validation errors.
- Apply command:
  `venv\\Scripts\\python scripts\\cutover_scheduled_future_speech_legacy_records.py --apply --confirm DELETE_SCHEDULED_FUTURE_SPEECH_LEGACY_RECORDS`
  deleted the accepted task and all four schedule IDs. It reported no
  remaining target IDs and preserved the four terminal run IDs.
- Post-deletion exports are retained as bounded audit evidence in
  `test_artifacts/diagnostics/scheduled_future_speech_cutover_*_after_bigbang.json`.
  The accepted-task and schedule exports contain zero rows; the calendar-run
  export contains the same four terminal rows.
- Read-only post-deletion preflight:
  `venv\\Scripts\\python scripts\\preflight_scheduled_future_speech_contract.py`
  exited 0 with `deployment_blocked=false`, zero incompatible active records,
  and zero incompatible rows in every scan.
- No rollback or compatibility mechanism was implemented or retained.

## Independent sign-off — 2026-08-15

- Final independent reviewer verdict: `PASS` with no blocking findings.
- The reviewer confirmed zero remaining exact target rows, unchanged terminal
  run history, generic future-cognition separation, 86 governed exact-impact
  nodes passed, 10 cutover/preflight tests passed, and a passing read-only
  preflight.
- Review artifact:
  `test_artifacts/diagnostics/scheduled_future_speech_final_review_post_bigbang_20260815.md`.

## Exclusions

No generic future-cognition deletion, historical run deletion, background-job
collection clear, migration of unrelated scheduler rows,
conversation/memory cleanup, fallback writer, alias vocabulary, or other
backward-compatible runtime path is included in this cutover.
