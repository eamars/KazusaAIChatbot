# DSH Plan 2: Semantic Tools And Coding Capability

## Summary

- Goal: add the accepted Kazusa semantic leaf capabilities and DSH-native
  coding capability to the standalone Plan 1 resolver.
- Status: draft; coarse successor plan; not executable until Plan 1 closes and
  this document is refined and explicitly approved.
- Plan class: standalone capability expansion with no Brain-path integration.
- Governing architecture:
  `docs/architecture/dsh_integration_architecture.md`.
- Functional support at exit: a standalone canonical intake can drive
  iterative private retrieval, public research, bounded computation, and
  authorized coding work through DSH, then return a validated evidence-backed
  exhaust.
- Isolation boundary: the current Brain, action selector, task-resolution,
  RAG, internal/external resolver, coding, cognition, dialog, background, and
  delivery paths remain unchanged and continue to have no call edge to DSH.
- Effort estimate: five functional work blocks. Exact owned files and test
  nodes are frozen after Plan 1 implementation evidence is available.

## Entry Conditions

Plan 2 may be promoted to an executable plan only when:

1. Plan 1 is completed and archived with every Plan 1 gate green.
2. The implemented Plan 1 intake, exhaust, RPC, session, evidence, lease,
   process, and fault contracts are treated as the baseline.
3. A post-Plan 1 source audit identifies the callable leaf executors beneath
   the current RAG, resolver, web, text, calendar, and memory designs.
4. The user approves the exact tool catalog and the coding workspace,
   network, write, execution, and approval authority contract.
5. The refined plan contains an exact create/modify/delete inventory and exact
   test node IDs derived from the implemented Plan 1 paths.

The refinement may narrow tool names or group low-level operations, while
preserving the functional gates in this document. It does not reopen the Plan
1 transport or lifecycle boundary without an explicit architecture amendment.

## Fixed Execution Ownership

This coarse draft authorizes planning only. Its post-Plan 1 refinement requires
explicit user approval and `in_progress` status before production execution.

| Role | Fixed owner | Responsibility |
|---|---|---|
| Architecture and closure | Parent agent | Owns tool-boundary decisions, scope, consolidated material review, gate interpretation, status, and closure |
| Implementation and verification | The persistent `/root/dsh_implementation_worker` subagent on `gpt-5.6-luna`, `max` reasoning, normal execution speed | Owns all production/test edits, builds, tool fixtures, test execution, remediation, and pre-handoff self-review |

Work remains one plan at a time, reuses the Plan 1 worker, and uses no
additional production worker. The parent remains read-only for production
implementation and test execution. Changing the worker binding requires a
user-approved amendment.

Execution has at most two review iterations:

1. The Luna worker delivers the complete tool implementation and all mapped
   verification; the parent returns one consolidated set of material findings.
2. When findings exist, the same worker resolves the entire set and reruns the
   acceptance matrix; the parent passes or blocks the plan.

Minor lint, collection, fixture, typing, typo, and documentation defects are
corrected during worker self-review before handoff. No third review iteration
is available.

Evidence remains proportionate: working-tree status, owned paths, final diff,
exact commands/results, live-case inspection, and gate decisions. Runtime
catalog, scope, audience, policy, and evidence fingerprints remain functional
requirements. General workspace or artifact hashing remains outside scope.

## Capability Direction

### Canonical Tool Gateway

The Plan 1 generic tool protocol becomes the only DSH-to-Kazusa semantic
capability boundary:

```text
DSH model-owned tool selection
    -> sidecar tool proxy
       -> invisible capability token and trusted scope
       -> Kazusa leaf tool gateway
          -> bounded leaf executor
       <- StandardToolResultV1
       -> durable evidence registration
    -> further DSH tool selection or submit_resolution
```

The model owns semantic tool choice, query refinement, sequencing, and when
enough evidence exists. Deterministic code owns schema validation, capability
admission, scope, audience, pagination and byte limits, timeouts, side-effect
policy, evidence registration, and crash disposition.

A leaf may perform local search, vector lookup, filtering, pagination,
provenance extraction, media decoding, or one bounded computation. A leaf does
not hide an LLM router, specialist selector, resolver loop, or open-ended DAG
that chooses other capabilities or synthesizes the complete resolution.

### Coarse Catalog

The refinement freezes canonical names from these capability families:

| Family | Required functional support |
|---|---|
| Conversation | Search scoped conversation evidence, list bounded source turns, and aggregate approved participant/time slices |
| Memory | Search and read authorized persistent user, shared, and character-world memory evidence without treating evidence as persona or final stance |
| People and profiles | Resolve an allowed person identity and read only the profile fields admitted by the current thread capability |
| Active recall and calendar | Read bounded current recall/calendar evidence through existing deterministic ownership |
| Public research | Search the public web, read hardened public resources, and inspect approved media without unrestricted access to internal network targets |
| Text and computation | Perform bounded deterministic transforms and calculations without hiding semantic routing |
| Coding | Compose DSH built-in coding tools under a Kazusa-owned workspace, sandbox, command, network, write, and approval policy; do not wrap or retain the current Kazusa coding harness design |

The sidecar composes public DSH packages and Kazusa-owned plugins only. DSH
source remains untouched and upgradable.

### Coding Boundary

DSH-native coding replaces the future need for the Kazusa coding harness. In
Plan 2 it remains reachable only from the standalone resolver. The refined
authority contract must distinguish:

- read-only repository inspection;
- proposed edits;
- authorized writes;
- command and test execution;
- network access;
- operations requiring user approval; and
- uncertain side-effect recovery.

The model cannot grant itself a broader workspace or permission through tool
arguments. Approval remains Kazusa-owned runtime authority and is revalidated
on resume. A write with unknown outcome is verified or escalated rather than
blindly repeated.

## Work Blocks And Effort Gates

| Block | Relative effort | Work | Independent completion gate |
|---|---|---|---|
| 1. Catalog and gateway freeze | Medium | Audit leaf executors, freeze manifests/schemas/limits, and connect the generic Plan 1 proxy to one Kazusa gateway | Every exposed operation is a bounded leaf with explicit authority, result, evidence, timeout, retry, and side-effect metadata |
| 2. Private evidence tools | High | Add conversation, memory, person/profile, and active-recall/calendar capabilities | Direct and multi-step standalone tests prove correct scope, audience, pagination, provenance, empty/error behavior, and evidence registration |
| 3. Public and computation tools | Medium | Add hardened web/media and deterministic text/compute capabilities | Network-boundary, content-limit, prompt-injection, citation, timeout, and evidence gates pass |
| 4. DSH coding capability | High | Compose DSH-native coding tools with the approved Kazusa workspace and authority policy | Read/propose/write/execute/approval and uncertain-outcome cases pass without importing the old coding harness |
| 5. Integrated standalone validation | High | Exercise mixed-tool investigations, restart/checkpoint/resume, adversarial calls, and terminal exhaust | Every tool family works through canonical intake-to-exhaust; Plan 1 lifecycle gates remain green; Brain isolation remains proven |

## Functional Release Gates

| Gate | Green condition |
|---|---|
| P2-G1 — Plan 1 stability | Intake, exhaust, RPC, thread/session identity, lease, checkpoint, terminal, evidence, and sidecar process contracts remain compatible with the completed Plan 1 implementation |
| P2-G2 — Tool completeness | Every approved tool has a strict manifest and passes success, empty, invalid, unauthorized, timeout, size-limit, cancellation, and provenance cases |
| P2-G3 — No hidden resolver graph | Static ownership and behavioral tests prove each Kazusa capability terminates as a leaf and DSH alone owns cross-tool semantic selection |
| P2-G4 — Coding authority | DSH coding works under the approved deterministic workspace/permission policy and has no dependency on the legacy Kazusa coding harness |
| P2-G5 — Durable evidence | Tool evidence remains registered, authorized, reconstructable after checkpoint/restart, and valid for terminal submission |
| P2-G6 — Standalone end to end | Representative private, public, mixed-source, and coding objectives complete through real standalone DSH intake, tools, and exhaust |
| P2-G7 — Brain non-impact | Current production Brain and all legacy production resolution paths remain unchanged and untapped |

All approved catalog operations and all seven gates must be green. A tool
family may not be declared supported from direct handler tests alone; it must
also pass through the real sidecar and terminal exhaust.

## Cutover And Decommission Policy

Plan 2 performs no Brain cutover. It may extract or publish bounded leaf
functions needed by the gateway while preserving current callers until Plan 3.
It does not preserve the design of an old resolver, RAG graph, specialist DAG,
or coding harness behind a new tool name.

Legacy Brain-path decommission is reserved for Plan 3. No legacy-to-DSH
checkpoint converter, dual execution, shadow comparison, fallback router, or
compatibility facade is introduced in Plan 2.

## Out Of Scope

- A Brain import or call edge to `agentic_resolver` or the sidecar.
- Mapping exhaust into `TaskResolutionResultV1`.
- Production action-selector, background-work, cognition, dialog, or delivery
  changes.
- Removal of current Brain task-resolution, RAG, internal/external resolver,
  coding, or accepted-task code.
- Changes to the retained shared-memory prewarm path.
- DSH source edits or a forked DSH design.

## Closure

The Luna worker supplies the implemented catalog, exact test evidence, and
final diff. The parent alone determines whether every capability is
functionally supported and closes the plan. Plan 3 refinement starts from the
catalog and operational behavior actually accepted here, rather than from this
coarse projection.
