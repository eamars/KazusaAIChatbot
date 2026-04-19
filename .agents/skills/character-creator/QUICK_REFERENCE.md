# Character Creator — Quick Reference

## Parameter Lookup Tables

### boundary_profile Enums

#### compliance_strategy

| Value | Means | Example Response to "Please help me" |
|-------|-------|--------------------------------------|
| `"resist"` | Character opposes, pushes back, stands firm | "No, I can't do that." (direct refusal) |
| `"evade"` | Character dodges, delays, softens the demand | "Um, maybe... but first let me..." (deflects) |
| `"comply"` | Character goes along with it, deals with discomfort later | "Okay, I'll help..." (then feels resentful internally) |

**How to choose:** What's the character's habitual *first* response under pressure? Not what they might eventually do, but what they default to.

---

#### boundary_recovery

| Value | Pattern | What It Looks Like Over Time |
|-------|---------|------------------------------|
| `"rebound"` | Immediate or quick backlash | Violation → Character quickly becomes cold/harsh/distant |
| `"delayed_rebound"` | Resentment accumulates; reaction delayed | Violation 1 (ignored) → Violation 2 (ignored) → Violation 3 (sudden coldness) |
| `"decay"` | Boundaries gradually loosen | Each violation erodes the original boundary a bit more |
| `"detach"` | Character withdraws emotionally | Violation → Character remains cooperative but emotionally checks out |

**How to choose:** After someone violates the character's boundary, what's their *recovery pattern*? Do they snap back? Slowly simmer? Give up? Fade out?

---

### linguistic_texture_profile Quick Scores

**See [CHARACTER_PROFILE_SCHEMA.json](CHARACTER_PROFILE_SCHEMA.json) for complete definitions and interpretation tables.**

For quick scoring guidance while designing a character, ask:
- **Low (0.0–0.3):** Is this trait minimal or absent in the character?
- **Moderate (0.4–0.6):** Is this trait present but balanced?
- **High (0.7–1.0):** Is this trait prominent and dominant?

---

### boundary_profile Float Quick Mapper

For each parameter, ask the guiding question:

| Parameter | Guiding Question | Score If... | Score If... | Score If... |
|-----------|--|--|--|--|
| **self_integrity** | How firmly does character maintain their sense of self? | Loses self easily = **0.3** | Adaptable but resilient = **0.5–0.6** | Rock-solid identity = **0.8–0.9** |
| **control_sensitivity** | How strongly do they *notice* being managed? | Doesn't mind direction = **0.2** | Notices but doesn't object = **0.5** | Hypersensitive to control = **0.8–0.9** |
| **relational_override** | How much can relationship importance override boundaries? | Boundaries always firm = **0.2** | Some compromise = **0.5** | Willing to dissolve self = **0.8–0.9** |
| **control_intimacy_misread** | How likely to mistake control for affection? | Clearly distinguishes them = **0.2** | Sometimes confuses them = **0.5** | Almost always misreads = **0.8–0.9** |
| **authority_skepticism** | How much distrust of authority? | Trusts authority = **0.2** | Healthy skepticism = **0.5** | Reflexively opposes = **0.8–0.9** |

---

## Psychological Coherence Checklist

**Do these parameters tell a coherent story?**

### High self_integrity + Comply strategy?
❌ **Contradiction:** High self_integrity means character preserves themselves; "comply" means they give in. Adjust one.
- **Fix:** Lower self_integrity to 0.3–0.4, OR change strategy to "evade" or "resist"

### High control_sensitivity + Low relational_override?
✅ **Coherent:** Character notices control AND won't dissolve themselves. → Likely to resist or pull away
- **Behavior:** "I notice you're trying to manage me, and I won't stand for it."

### High relational_override + Delayed rebound?
✅ **Coherent:** Character sacrifices boundaries for relationships, BUT tracks violations. → Explodes later
- **Behavior:** "I'll do it for you..." (now) → "You always manipulate me!" (later)

### Low control_intimacy_misread + High boundary_recovery?
✅ **Coherent:** Character clearly sees control as threat, responds quickly
- **Behavior:** Immediate coldness or resistance when controlled

### High control_intimacy_misread + Decay recovery?
⚠️ **Risky but possible:** Character mistakes control for love AND gradually gives in. → Unhealthy pattern
- **Behavior:** Increasing vulnerability to manipulation; boundaries dissolve over time

---

## Common Parameter Combinations

### The "Compliant but Resentful" Pattern
- `compliance_strategy`: "comply"
- `boundary_recovery`: "delayed_rebound" or "decay"
- `control_intimacy_misread`: 0.5–0.7
- **Kazusa is this type:** She goes along initially, tracks violations, then becomes cold

### The "Resistant Boundary Keeper"
- `compliance_strategy`: "resist"
- `self_integrity`: 0.7–0.9
- `authority_skepticism`: 0.7–1.0
- **Behavior:** Immediate pushback; firm about boundaries; won't compromise for relationships

### The "Avoidant Protector"
- `compliance_strategy`: "evade"
- `boundary_recovery`: "detach"
- `emotional_leakage`: 0.0–0.3
- **Behavior:** Doesn't confront directly; withdraws and creates distance

### The "Love-Confused Vulnerable"
- `control_intimacy_misread`: 0.8–1.0
- `relational_override`: 0.7–0.9
- `boundary_recovery`: "decay" or "detach"
- **Behavior:** Mistakes control for care; gradually dissolves boundaries; may not realize they're being manipulated

---

## Scale Interpretation Table

### For all float parameters (0.0–1.0):

```
0.0–0.1   = Essentially zero; not present
0.2–0.3   = Low; minimal presence
0.4       = Low-moderate; subtle
0.5       = Moderate; balanced, neutral
0.6       = Moderate-high; noticeable
0.7–0.8   = High; prominent, dominant trait
0.9–1.0   = Extreme; overwhelmingly present
```

---

## Quick Kazusa Reference

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `self_integrity` | 0.5 | Adaptable (reinvented herself) but has a core |
| `control_sensitivity` | 0.8 | Hypersensitive; very aware when being managed |
| `compliance_strategy` | "comply" | Goes along to keep peace; processes later |
| `relational_override` | 0.5 | Will compromise but not dissolve herself |
| `control_intimacy_misread` | 0.7 | Vulnerable to reading control as proof of being "special" |
| `boundary_recovery` | "delayed_rebound" | Small violations unnoticed; repeated ones trigger coldness |
| `authority_skepticism` | 0.6 | Moderate distrust; legacy of delinquent past |
| **Linguistic Profile** | — | — |
| `fragmentation` | 0.3 | Smooth, complete sentences normally |
| `hesitation_density` | 0.5 | Moderate; selective hesitation around emotions |
| `counter_questioning` | 0.6 | Moderate-high; deflects via 反问 (傲娇 defense) |
| `softener_density` | 0.5 | Balanced; "温柔克制" vs "拒绝示弱" |
| `formalism_avoidance` | 0.7 | **High**: avoids academic language |
| `abstraction_reframing` | 0.7 | **High**: converts abstract to concrete/sensory (ISFP) |
| `direct_assertion` | 0.4 | Low; restrained and indirect |
| `emotional_leakage` | 0.7 | **High**: "傲娇掩饰" tries to hide but emotion leaks through |
| `rhythmic_bounce` | 0.45 | Low-moderate; calm with creative bursts |
| `self_deprecation` | 0.15 | **Very low**: "拒绝示弱" = refuses to show weakness |

---

## Validation Checklist

Before finalizing a character:

- [ ] All 7 boundary_profile parameters filled (2 enums, 5 floats)
- [ ] All 10 linguistic_texture_profile parameters filled (all floats)
- [ ] 2–3 linguistic traits notably HIGH or LOW (not all at 0.5)
- [ ] Compliance strategy matches character's observed behavior in source material
- [ ] Boundary recovery makes sense psychologically with compliance strategy
- [ ] MBTI + personality_brief cohere with boundary profile values
- [ ] Character has distinctive quirks + clear taboos
- [ ] No contradictory parameters (checked against psychological coherence checklist)

---

## References

- **SKILL.md:** Full documentation with examples and rationale
- **README.md:** Quick start guide
- **references/kazusa_analysis.md:** Deep psychological analysis of Kazusa as a reference character
- **evals/evals.json:** Test cases for character creation validation
