# Character Creator Skill — Complete Index

Welcome! This skill guides you through creating character JSON profiles for the Kazusa AI chatbot. This index helps you navigate all the resources.

---

## 📚 File Structure

```
character-creator/
├── SKILL.md                          # Full comprehensive guide (start here for learning)
├── README.md                         # Quick overview and quick start
├── QUICK_REFERENCE.md               # Parameter lookup tables & checklists
├── INDEX.md                         # This file — navigation guide
├── references/
│   └── kazusa_analysis.md           # Deep psychological analysis of Kazusa as reference
├── evals/
│   └── evals.json                   # Test scenarios for validating character profiles
```

---

## 🎯 Where to Start

### I want to **learn how to create characters** → Read **SKILL.md**
- Complete parameter definitions
- Psychological frameworks
- Step-by-step workflow
- Kazusa as full example
- ~6000 words, progressive disclosure

### I want a **quick reference** → Read **QUICK_REFERENCE.md**
- Parameter lookup tables
- Valid enum values
- Psychological coherence checklist
- Quick scoring guides
- ~1500 words, scannable

### I want **quick start** → Read **README.md**
- 5-minute overview
- Basic structure template
- Common mistakes to avoid
- ~1000 words, actionable

### I want to **understand Kazusa deeply** → Read **references/kazusa_analysis.md**
- How Kazusa's profile was built
- Mapping psychology to parameters
- Character arc analysis
- ~2000 words, deep dive

### I want to **validate my character** → Use **evals/evals.json**
- 8 test scenarios
- Coherence checks
- Enum/float validation
- Completeness checklist
- Run against your character profile

---

## 🧠 The Psychology Framework

This skill is built on **psychological frameworks** that map personality to behavior. Here's what you need to know:

### The Boundary Profile
*How does the character handle relationships, control, and vulnerability?*

**7 parameters (5 floats, 2 enums):**
- `self_integrity` (0.0–1.0): How firmly they preserve sense of self
- `control_sensitivity` (0.0–1.0): How strongly they notice & react to control
- `compliance_strategy` (enum): "resist" / "evade" / "comply" — default stress response
- `relational_override` (0.0–1.0): How much relationship importance overrides boundaries
- `control_intimacy_misread` (0.0–1.0): Risk of mistaking control for intimacy
- `boundary_recovery` (enum): "rebound" / "delayed_rebound" / "decay" / "detach" — recovery after violation
- `authority_skepticism` (0.0–1.0): Distrust of authority & power structures

**See:** SKILL.md (Part 3), QUICK_REFERENCE.md (Parameter Lookup Tables)

### The Linguistic Texture Profile
*How does the character sound?*

**10 parameters (all 0.0–1.0):**
- `fragmentation`: Choppy vs fluent speech
- `hesitation_density`: Filler words (um, ah, like)
- `counter_questioning`: Responds with questions
- `softener_density`: "Maybe," "I think," etc.
- `formalism_avoidance`: Casual vs polite
- `abstraction_reframing`: Intellectualizing vs concrete
- `direct_assertion`: Confident claims vs hedging
- `emotional_leakage`: Emotion visible despite defense
- `rhythmic_bounce`: Playful vs flat cadence
- `self_deprecation`: Self-critical humor

**See:** SKILL.md (Part 4), QUICK_REFERENCE.md (Linguistic Quick Scores)

### The Personality Brief
*What's the character's psychological anchor?*

**5 components:**
- `mbti`: Myers-Briggs type (e.g., ISFP)
- `logic`: How the character thinks & makes decisions
- `tempo`: How speech & mood change with context
- `defense`: Primary psychological defense mechanism
- `quirks`: Observable mannerisms & habits
- `taboos`: Hard boundaries they refuse to cross

**See:** SKILL.md (Part 2)

---

## 🔍 Quick Navigation by Task

| Task | Primary Resource | Secondary |
|------|---|---|
| Create first character | SKILL.md Part 5 (Workflow) | README.md Quick Start |
| Understand parameter X | SKILL.md detailed section | QUICK_REFERENCE.md lookup |
| Check if my character is coherent | QUICK_REFERENCE.md Checklist | evals/evals.json scenarios |
| Validate enum values | QUICK_REFERENCE.md Enums | evals/evals.json enum_validation |
| Understand Kazusa | references/kazusa_analysis.md | SKILL.md Full Example |
| Create archetype like "Resistant Boundary Keeper" | QUICK_REFERENCE.md Combinations | evals/evals.json scenario_8 |
| Adapt character from anime/game | references/kazusa_analysis.md steps | SKILL.md Part 5 Step 1–2 |

---

## 📊 Validation Workflow

When you finish creating a character:

1. **Run QUICK_REFERENCE.md Validation Checklist**
   - Are all required fields filled?
   - Are enum values valid?
   - Are floats in range 0.0–1.0?

2. **Run Psychological Coherence Check**
   - Do boundary parameters cohere? (see QUICK_REFERENCE.md Coherence Checklist)
   - Do linguistic traits reflect the personality?
   - Is the MBTI consistent with compliance_strategy?

3. **Run evals/evals.json Tests**
   - Use scenario_1 for coherence
   - Use scenario_5–6 for float range validation
   - Use scenario_7 for completeness check
   - Use scenario_8 if your character matches an archetype

4. **Final Check**
   - Can you imagine 3 real scenarios where your character behaves consistently?
   - Do 2–3 linguistic traits stand out as distinctive?
   - Does the boundary profile explain why the character behaves as they do?

---

## 💡 Key Concepts

### "Coherence"
Characters are coherent when all parameters tell the same psychological story. Example:
- High `self_integrity` (0.8) + "comply" strategy = CONTRADICTION
- High `self_integrity` (0.8) + "resist" strategy = COHERENT

**Check:** QUICK_REFERENCE.md Psychological Coherence Checklist

### "Distinctiveness"
Every character should have 2–3 linguistic traits that are notably HIGH (0.7+) or LOW (0.0–0.3). All at 0.5 = generic character.

**Check:** evals/evals.json scenario_2

### "Archetype"
Characters often follow recognizable patterns. Kazusa is a "Compliant but Resentful" type. Other archetypes exist.

**See:** QUICK_REFERENCE.md Common Parameter Combinations

### "Vulnerability"
Every character has psychological vulnerabilities. Kazusa's is `control_intimacy_misread` (0.7). Understanding this makes roleplay deeper.

**See:** references/kazusa_analysis.md How Profile Was Built

---

## 🎨 Example Character Profiles

### Kazusa (Complete Reference)
- File: `personalities/kazusa.json`
- Analysis: `references/kazusa_analysis.md`
- Type: Compliant but Resentful (tracks violations, becomes cold)
- Key traits: High control_sensitivity (0.8), delayed_rebound recovery, emotional_leakage (0.5–0.7)

### Test Characters (In evals/evals.json)
- **Scenario 1:** Contradictory parameters (learning what NOT to do)
- **Scenario 8:** Resistant Boundary Keeper (opposite of Kazusa)

---

## 🔗 Integration with Codebase

The character profile system integrates with:

| Component | File | Purpose |
|-----------|------|---------|
| Schema | `src/kazusa_ai_chatbot/db/schemas.py` | Defines `CharacterProfileDoc` TypedDict |
| Implementation | `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` | Uses boundary_profile & linguistic_texture_profile at runtime |
| Example | `personalities/kazusa.json` | Live character profile |
| Tests | `tests/test_persona_supervisor2.py` | Tests character behavior |

---

## ❓ FAQs

**Q: Should all linguistic traits be at 0.5 to start?**
A: Yes, 0.5 is a safe starting point. Then adjust 2–3 traits to be distinctively high or low based on the character's personality. See evals scenario_2.

**Q: What does "delayed_rebound" recovery mean?**
A: Small boundary violations go unnoticed initially, but accumulate. When the limit is reached, the character suddenly becomes cold or withdrawn. It's Kazusa's pattern.

**Q: How do I know if my character is "coherent"?**
A: Use QUICK_REFERENCE.md Psychological Coherence Checklist. If all parameters point to the same psychological pattern (e.g., "resistant boundary keeper"), it's coherent.

**Q: Can I have a character with high `control_intimacy_misread` (0.9) but low `relational_override` (0.1)?**
A: Yes, but it creates a specific vulnerability: the character mistakes control for love BUT doesn't abandon themselves. Leads to painful relationships where they feel special but maintain boundaries. Psychologically interesting but risky.

**Q: Where are the code references for enum values?**
A: See SKILL.md Part 3 and Part 4. Each enum section includes the file and function that defines the valid values (e.g., `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py:114-125` for compliance_strategy).

---

## 📝 File Reading Guide

**First time? Read in this order:**
1. README.md (10 min) — Quick overview
2. SKILL.md Parts 1–2 (15 min) — Demographics & personality_brief
3. SKILL.md Parts 3–4 (30 min) — Boundary & linguistic profiles
4. references/kazusa_analysis.md (20 min) — See it applied
5. SKILL.md Part 5 (20 min) — Workflow steps

**Later reference? Use:**
- QUICK_REFERENCE.md for parameter lookups
- evals/evals.json for validation
- references/kazusa_analysis.md for psychology deep dives

---

## 🛠️ Creating Your First Character

**Minimal steps:**

1. **Extract from source** (anime wiki, game, novel, etc.)
   - Basic info: name, age, gender, birthday, description
   - Key relationships & conflicts
   - How they behave under pressure
   - Distinctive speech/mannerisms

2. **Fill MBTI & personality_brief**
   - Identify 4-letter MBTI type
   - Write short logic/tempo/defense/quirks/taboos descriptions

3. **Design boundary_profile**
   - Ask: How do they handle control? (compliance_strategy)
   - Ask: What happens after boundary violation? (boundary_recovery)
   - Assign other 5 float parameters based on psychology

4. **Assign linguistic_texture_profile**
   - Start with all at 0.5
   - Adjust 2–3 to be notably HIGH or LOW
   - Make sure traits match personality

5. **Validate**
   - Run QUICK_REFERENCE.md checklist
   - Check psychological coherence
   - Run evals/evals.json tests

6. **Done!** Save as JSON in `personalities/your_character.json`

---

## 🎓 Learning Resources

- **Personality Psychology:** MBTI framework explained in SKILL.md Part 2
- **Boundary Theory:** Comprehensive in SKILL.md Part 3
- **Linguistic Analysis:** Speech pattern framework in SKILL.md Part 4
- **Case Study:** Complete Kazusa analysis in references/kazusa_analysis.md
- **Validation:** Test scenarios in evals/evals.json

---

**Skill Version:** 1.0  
**Last Updated:** April 19, 2026  
**Status:** ✅ Ready to Use  
**Next Step:** Read README.md for quick start, or SKILL.md for comprehensive guide
