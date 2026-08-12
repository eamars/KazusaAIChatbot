# Context Lookup and Translation Glossary

Use this reference after inspecting the source document, repository usage, and
the surrounding section. It records failure patterns from the explicit
cognitive trajectory translation review; it is a decision aid, not an
authoritative word-for-word dictionary.

## Contents

- [Context lookup](#context-lookup)
- [Term ledger](#term-ledger)
- [Contextual decisions](#contextual-decisions)
- [Native rewrite patterns](#native-rewrite-patterns)
- [Final audit](#final-audit)

## Context lookup

Resolve a term in this order:

1. Read the complete source sentence.
2. Read the surrounding paragraph and section heading.
3. Identify the term’s role: object, operation, permission, state, result,
   boundary, label, or ordinary prose.
4. Inspect nearby tables, diagrams, examples, field definitions, and code.
5. Search the repository for the source term and existing Chinese renderings.
6. Check the authoritative source or ICD when the term describes behavior.
7. Compare candidate Chinese wording in the target paragraph, not in isolation.
8. Record unresolved ambiguity and ask for direction when it affects meaning.

Search examples:

```powershell
rg -n "affordance|capability|亲和度|亲近" docs src tests
rg -n "private_monologue|private monologue" .
```

Prefer local usage over generic bilingual dictionaries. If the repository uses
an established term inconsistently, preserve the authoritative contract and
surface the inconsistency instead of silently choosing a third vocabulary.

## Term ledger

Use a temporary ledger for terms whose meaning depends on context:

| Source term | Meaning in this artifact | Candidate | Evidence | Decision |
| --- | --- | --- | --- | --- |
| `term` | Local semantic role | Chinese wording | Source, code, or section | chosen / unresolved |

Record why a candidate was chosen. Include grammatical role when it changes the
translation, such as `affordance` as a registry entry versus `affordance
lookup` as an operation. Delete or keep the ledger only according to the
artifact’s requirements; do not add it to the translated document by default.
Keep one ledger across the entire artifact and every translation pass. Reconcile
it after the first complete draft so a later section can correct an earlier
local choice when the full-document context reveals a better distinction.

## Contextual decisions

| Source concept | Mechanical failure | Context-first candidates | Decision cue |
| --- | --- | --- | --- |
| `affordance` | Merge it into `能力` or translate an object as `能力查询` | `可供性` in a formal HCI/architecture vocabulary; `可行动性` or `可用行动选项` for reader-facing prose | Keep it distinct from executable `capability`; inspect whether the term names an option, registry entry, or lookup operation |
| `capability` | Use the same word as `affordance` | Usually `能力` | Use for a resolver, tool, or executable ability; verify local contract language |
| `affordance lookup` | Translate as an affordance object or a generic capability query | `可供性查询` or a context-specific `可用选项查询` | The phrase names an operation, not the returned option |
| `affordance registry` | Translate as `能力注册表` when the source distinguishes it | `可供性注册表` or a locally established registry term | Preserve the distinction from capability registries |
| `affinity` | Collapse it into `亲密度` | `亲和度` when it is an aggregate relationship construct | Check whether the source separately names closeness or intimacy axes |
| `closeness` | Render it as the same `亲密度` used for affinity | `亲近` or `亲近程度` | Use the relationship-axis meaning in the local sentence |
| `intimacy` | Treat it as a synonym for every relationship axis | `亲密度` when the source names desired/perceived intimacy | Keep it separate from aggregate affinity |
| `private monologue` | Translate as `私念`, which implies selfish or ulterior thought | `内心独白` | Preserve first-person monologue and continuity meaning |
| NLP `token-level` | Use security `令牌级` | `词元级` | Use `令牌` only when the source means a credential or security token |
| `episode` | Alternate among `事件`, `片段`, and `回合` without checking the model | Use the repository’s canonical term, often `片段` in this project | Decide once for the artifact; do not form misleading compounds such as `片段性知识` when the source means episodic knowledge |
| `episodic knowledge` | Translate mechanically as `片段性知识` | `情景性知识` or `事件性知识` | Choose the term that means knowledge from episodes, not fragmented knowledge |
| `root episodes` | Translate as `根片段` | `根源片段` or `上游片段` | Inspect whether root means causal origin, lineage, or hierarchy |
| `settled` / `settlement` | Use financial `结算` in a lifecycle or queue context | `已完结`, `已确定`, `定案`, or `落定` | Choose according to whether the state is closed, decided, or merely settled for processing |
| `bid` in semantic selection | Use auction `竞标` | `候选`, `提案`, or `意向候选` | Use auction/tender terms only when competing offers are literally being submitted |
| `persona` | Use `人格` for every context or preserve an awkward literal | `人设`, `角色形象`, or `人格` | Check whether the artifact discusses character presentation, psychological identity, or a formal persona model |
| `schema` | Translate every occurrence as `模式` | `结构`, `输出结构`, or `契约` | Use `模式` only when the source means a pattern or mode rather than a data shape |
| `stochastic provider` | Translate as `随机性供应商` | `非确定性供应商` or `随机模型供应商` | The provider supplies a non-deterministic model, not randomness as a product |
| `prose transcript` | Use literary `散文式记录` | `自然语言记录` or `非结构化记录` | Check whether the contrast is with typed/structured state |
| `fail closed` | Use opaque `失败闭合` | `失败关闭策略` or `失败时安全关闭` | Preserve the safety property and make the failure behavior readable |
| `positive regard` | Use `正向认可` when the source uses the psychology term | `积极关注` or the repository’s established term | Distinguish regard from approval or endorsement |
| `monolithic` | Use `整体式` as an English-shaped adjective | `单体式` or `单一的整体` | Describe one undivided system, not merely a complete system |
| `operational` in runtime architecture | Use business `运营` | `运行期` or `运行时` | Choose `运营` only for business operations |
| `authorization grant` | Use tautological `授权授予` | `授权本身`, `一次授权`, or `授权许可` | Check whether the source means a decision, grant, credential, or permission state |

Do not apply these candidates mechanically. For example, `人设` can be wrong
in a formal identity schema, and `可供性` can be too academic for user-facing
copy. Context decides.

## Native rewrite patterns

Use these patterns to identify English syntax that needs reconstruction:

| English shape | Literal risk | Native strategy |
| --- | --- | --- |
| `X owns Y` | `X 拥有 Y` when ownership means responsibility | Use `X 负责 Y` or `X 承担 Y`; use `拥有` only for actual possession |
| `must not make X infer Y` | `不能要求 X 推断 Y` changes causality | Use `不能让 X 推断 Y` or `不得使 X 从……推断……` |
| `earned by an evidence gap` | `由……换取` implies a trade | Use `建立在……基础上`, `需要……作为依据`, or `必须由……支撑` |
| `follows from the selected intention` | `意图之后应跟` is unidiomatic | Use `从选定的意图出发，应当……` |
| `immutable as a decision context` | Modifier order becomes ambiguous | Use `作为决策上下文，……是不可变的` |
| `What episode was admitted?` | Passive `哪个片段被准入` | Use `哪个片段通过了准入` or the project’s established active form |
| `interpreted differently based on...` | Preserve an English passive stack | Use `当前事件会根据……得到不同解读` |
| `make a response / compose dialog` | `编写对话` sounds like code or documentation | Use `生成`, `撰写`, or `组织表达` according to the stage |
| `safe for display` | `适合显示` loses the safety gate | Use `可安全用于显示` when authorization or exposure safety matters |

Do not simplify away negation, scope, causality, ownership, or modality in the
name of fluency.

## Final audit

Before delivery, confirm:

- A native Chinese reader can understand each paragraph without back-translating
  English syntax.
- Every important source distinction has a visible Chinese distinction.
- Terms are consistent within the artifact, with justified context-based
  variation where necessary.
- Contractual identifiers, links, paths, code spans, field names, enum values,
  tables, headings, and diagram topology remain intact.
- The translation did not add claims, remove constraints, or turn evidence into
  judgment.
- The diff contains only the requested translation changes.
