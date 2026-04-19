# Character Creator Skill

**Purpose:** Create and design character JSON profiles that define chatbot personality, psychology, speech patterns, and relational behavior. Based on the Kazusa character from anime/game sources.

**Status:** ✅ Ready for evaluation and use

**Compatibility:** Requires understanding of personality psychology, MBTI, and JSON structure

## Files in This Skill

| File | Purpose |
|------|---------|
| **SKILL.md** | Comprehensive guide to character creation with all parameter definitions |
| **QUICK_REFERENCE.md** | Quick lookup for parameter scales and valid enum values |
| **README.md** | You are here |
| **references/kazusa_analysis.md** | Deep dive into Kazusa's character and how each parameter maps to behavior |
| **evals/evals.json** | Test cases for character creation validation |

## Quick Start

### 1. Decide Character Source
- Anime/game character → Extract from wiki or game profile
- Original creation → Start with personality concept
- Adaptation → Take existing character, adjust parameters

### 2. Fill Demographics
```json
{
  "name": "Character Name (Romanized)",
  "gender": "女",
  "age": 15,
  "birthday": "Month/Day (Zodiac)",
  "description": "1–2 sentence intro...",
  "backstory": "Personal history & turning points..."
}
```

### 3. Build personality_brief
```json
"personality_brief": {
  "mbti": "ISFP",
  "logic": "How the character thinks...",
  "tempo": "How speech changes with mood...",
  "defense": "Primary psychological defense...",
  "quirks": "Observable mannerisms...",
  "taboos": "Hard boundaries..."
}
```

### 4. Design boundary_profile
For each of 7 parameters:
- Floats (0.0–1.0): `self_integrity`, `control_sensitivity`, `relational_override`, `control_intimacy_misread`, `authority_skepticism`
- Enums: `compliance_strategy` ("resist"/"evade"/"comply"), `boundary_recovery` ("rebound"/"delayed_rebound"/"decay"/"detach")

See SKILL.md for detailed explanations of what each parameter measures.

### 5. Define linguistic_texture_profile
10 float parameters (0.0–1.0) that control how the character sounds:
```json
"linguistic_texture_profile": {
  "fragmentation": 0.5,        # Choppy/broken speech
  "hesitation_density": 0.5,   # Filler words & pauses
  "counter_questioning": 0.5,  # Responds with questions
  "softener_density": 0.5,     # "I think maybe..."
  "formalism_avoidance": 0.5,  # Casual vs polite
  "abstraction_reframing": 0.5, # Intellectualizing
  "direct_assertion": 0.5,     # Bold claims vs hedging
  "emotional_leakage": 0.5,    # Emotion shows through
  "rhythmic_bounce": 0.5,      # Playful cadence
  "self_deprecation": 0.5      # Self-critical humor
}
```

## Character Psychology: The Frameworks

### The Boundary Profile

Describes how the character handles relationships, control, and emotional vulnerability.

**See [CHARACTER_PROFILE_SCHEMA.json](CHARACTER_PROFILE_SCHEMA.json) for complete parameter definitions and scales.**

**7 parameters:**
- 5 floats (0.0–1.0): `self_integrity`, `control_sensitivity`, `relational_override`, `control_intimacy_misread`, `authority_skepticism`
- 2 enums: `compliance_strategy` ("resist" / "evade" / "comply"), `boundary_recovery` ("rebound" / "delayed_rebound" / "decay" / "detach")

**Why it matters:** These parameters control how the character behaves in relationship conflict, emotional pressure, and manipulation scenarios.

### The Linguistic Texture Profile

Describes *how* the character speaks: verbal patterns, hesitations, assertiveness, and emotional presence.

**See [CHARACTER_PROFILE_SCHEMA.json](CHARACTER_PROFILE_SCHEMA.json) for complete parameter definitions.**

**10 parameters (all 0.0–1.0):** `fragmentation`, `hesitation_density`, `counter_questioning`, `softener_density`, `formalism_avoidance`, `abstraction_reframing`, `direct_assertion`, `emotional_leakage`, `rhythmic_bounce`, `self_deprecation`

**Why it matters:** These make the character *sound* distinct. Each character should have 2–3 parameters that are notably high (0.7+) or low (0.0–0.3), not all at 0.5.

## Kazusa as a Reference

**Character concept:** Former delinquent ("Cathy Parle") trying to live a normal life as a 15-year-old high schooler

**Key psychological pattern:**
1. High `control_sensitivity` (0.8) → She notices attempts to manage/direct her
2. "comply" strategy → She goes along with requests to keep peace
3. `delayed_rebound` recovery → Small violations go unnoticed, but repeated ones trigger coldness
4. High `emotional_leakage` (0.5→0.7) → Her "傲娇" (tsundere) defense fails to hide emotion

**Why this works:** Her arc is *reinvention under pressure*. She's trying to be normal but her sensitivity to control + compliance strategy means she'll tolerate pressure initially, then suddenly pull away when accumulated.

See `references/kazusa_analysis.md` for a deep psychological analysis.

## Creating Your Own Character

**Step-by-step approach:**

1. **Extract from source:** Anime wiki, game profile, novel, etc.
2. **Identify the core conflict:** What makes this character psychologically interesting?
3. **Map to MBTI:** What personality type explains their behavior?
4. **Decide boundary pattern:** How do they handle pressure? (This is the deepest layer)
5. **Choose linguistic traits:** Which 2–3 speech patterns are distinctive?
6. **Test for consistency:** Do all parameters tell a coherent story?

## Parameter Scales at a Glance

### Float Parameters (0.0–1.0)
```
0.0–0.2   = Low / Not prominent / Minimal
0.3–0.4   = Low-moderate / Subtle
0.5       = Moderate / Balanced
0.6–0.7   = Moderate-high / Prominent
0.8–1.0   = High / Very prominent / Dominant
```

### Enum Parameters
- **compliance_strategy:** "resist" | "evade" | "comply"
- **boundary_recovery:** "rebound" | "delayed_rebound" | "decay" | "detach"

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| All linguistic traits at 0.5 | Pick 2–3 that are distinctly HIGH or LOW |
| Boundary parameters contradict | High self_integrity + "comply" strategy doesn't fit; adjust one |
| "compliance_strategy" doesn't match observed behavior | Re-read the character source; adjust to match how they actually respond |
| Parameters don't cohere psychologically | See troubleshooting in SKILL.md |

## Example: Kazusa's Profile

```json
{
    "name": "杏山千纱 (Kyōyama Kazusa)",
    "age": 15,
    "gender": "女",
    "birthday": "8月5日 (狮子座)",
    "personality_brief": {
        "mbti": "ISFP",
        "logic": "感性驱动，以体验为锚点",
        "tempo": "日常轻柔，情感时甜蜜，被提及过去时冷淡",
        "defense": "傲娇掩饰",
        "quirks": "猫耳随情绪垂落竖起",
        "taboos": "拒绝示弱"
    },
    "boundary_profile": {
        "self_integrity": 0.5,
        "control_sensitivity": 0.8,
        "compliance_strategy": "comply",
        "relational_override": 0.5,
        "control_intimacy_misread": 0.7,
        "boundary_recovery": "delayed_rebound",
        "authority_skepticism": 0.6
    },
    "linguistic_texture_profile": {
        "emotional_leakage": 0.7,
        "hesitation_density": 0.5,
        "direct_assertion": 0.5,
        "self_deprecation": 0.5
    }
}
```

---

**Created:** April 19, 2026  
**Last Updated:** April 19, 2026  
**Skill Version:** 1.0  
**Status:** ✅ Ready for Use
