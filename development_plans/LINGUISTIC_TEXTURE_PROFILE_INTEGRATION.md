# Linguistic Texture Profile Integration Plan

**Stage:** 1  
**Priority:** Medium  
**Target Release:** Stage 2.5 (Post-boundary core stabilization)  

---

## Requirement

Integrate `linguistic_texture_profile` into the L3 Linguistic Agent to make character dialogue output more vivid, contextually varied, and aligned with the character's psychological profile. The 10 linguistic parameters (fragmentation, hesitation_density, counter_questioning, softener_density, formalism_avoidance, abstraction_reframing, direct_assertion, emotional_leakage, rhythmic_bounce, self_deprecation) shall be translated into human-readable descriptions and embedded into the linguistic agent's system prompt, similar to how `boundary_profile` parameters are used in the L2 boundary core agent.

**Key Goals:**
1. Make dialogue generation more vivid and character-specific
2. Create consistent speech patterns tied to numerical parameters
3. Allow future dynamic adjustment of linguistic traits over relationship evolution
4. Provide LLM with concrete, actionable constraints on how to construct sentences

---

## Technical Blocks

### Block 1: Translation Functions (Similar to L2 Pattern)
**Challenge:** Create 10 translation functions that convert float values (0.0–1.0) into descriptive text that the LLM can understand and execute.

**Approach:**
- Follow the pattern of `get_self_integrity_description()`, `get_control_sensitivity_description()`, etc. in `persona_supervisor2_cognition_l2.py`
- Each function should return a list of 11 descriptions (levels 0–10, achieved by `round(score * 10)`)
- Each description must be actionable: rather than saying "low fragmentation," say "你的表达习惯是完整的句子，很少中断或重来"
- **Example structure:**
  ```python
  def get_fragmentation_description(fragmentation_score: float) -> str:
      descriptions = [
          "完全流畅的表达，句子结构完整，几乎不出现打断、修正或片段状态。",
          ...
          "极度碎片化的表达，多数时间是片段、打断、省略，信息呈现得支离破碎。"
      ]
      clamped_score = max(0.0, min(1.0, fragmentation_score))
      level = round(clamped_score * 10)
      return descriptions[level]
  ```

**Dependencies:**
- Read `CHARACTER_PROFILE_SCHEMA.json` and `SKILL.md` (Part 4) to extract parameter definitions
- Review `persona_supervisor2_cognition_l2.py` for existing pattern

---

### Block 2: Integration into `_LINGUISTIC_AGENT_PROMPT`
**Challenge:** Embed the translated descriptions into the system prompt in a way that is clear and actionable for local LLMs with limited interpretation capacity.

**Approach:**
- Create a new section in `_LINGUISTIC_AGENT_PROMPT` called **"# 语言质感约束 (Linguistic Texture Constraints)"**
- Use `.format()` placeholders similar to existing personality fields:
  ```python
  _LINGUISTIC_AGENT_PROMPT = """\
  ...
  # 语言质感约束 (Linguistic Texture Constraints)
  {linguistic_texture_descriptions}
  
  # 应用方式
  上述 10 个语言参数定义了你的表达"质感"。你的任务是在生成内容锚点时，
  确保对话符合这些约束...
  """
  ```
- **Format requirement (for local LLM clarity):**
  ```
  **fragmentation**: [clear, concrete description for current score]
  **hesitation_density**: [clear, concrete description for current score]
  ...
  ```
- **Critical note on descriptions:** Each description must be:
  - In Chinese
  - Self-contained and unambiguous (avoid reference to other parameters)
  - Actionable with concrete examples (e.g., instead of "moderate fragmentation," say "occasionally pause and restart sentences; use about 2-3 sentence fragments per paragraph")
  - Written in imperative/prescriptive language ("you should...", "avoid...", "use more...")
  - Avoid nuanced interpretations that require deep context

**Dependencies:**
- Must be placed *before* the LLM call in `call_linguistic_agent()`
- Need to refactor `call_linguistic_agent()` to:
  1. Extract `linguistic_texture_profile` from `state["character_profile"]`
  2. Call 10 translation functions to convert to descriptive text
  3. Construct formatted string and inject into prompt
  4. Log the formatted descriptions for debugging (local LLM may misinterpret without clarity)

---

### Block 3: LLM Prompt Architecture Update
**Challenge:** The LLM needs to understand *how* to apply linguistic constraints while still respecting logical stance (CONFIRM/REFUSE/TENTATIVE).

**Approach:**
- Add explicit guidance to the prompt:
  ```
  # 核心原则
  1. 逻辑立场 (logical_stance) 是"说什么"，**永远无法改变**。
  2. 语言质感约束 (linguistic_texture) 是"怎么说"，控制表达方式但不改变内容。
  3. 例如：
     - 若 logical_stance = CONFIRM，你必须接受。但如果 fragmentation 很高，你可以这样说：
       "嗯，我……其实想……对，我答应了……就是这样。"（碎片化的接受）
     - 若 emotional_leakage 很高，即便你试图保持冷静，情感仍会在措辞、省略号和语序中泄露。
  ```
- Add constraint check: "Linguistic texture must enhance (not override) the core decision"

**Impact:** Prevents LLM from using linguistic parameters as an excuse to soften a REFUSE into a TENTATIVE

---

### Block 4: State Management
**Challenge:** Ensure `linguistic_texture_profile` is consistently available in `CognitionState`.

**Approach:**
- Verify `character_profile` in `CognitionState` includes `linguistic_texture_profile` (already added to schema)
- Add type hints in `persona_supervisor2_cognition.py`:
  ```python
  TypedDict CognitionState should include:
  "linguistic_texture_profile": LinguisticTextureProfileDoc
  ```
- Defensive check in `call_linguistic_agent()`:
  ```python
  ltp = state["character_profile"].get("linguistic_texture_profile", {})
  if not ltp:
      logger.warning("No linguistic_texture_profile found; using defaults")
      ltp = {k: 0.5 for k in [list of 10 keys]}
  ```

---

## Impact Analysis

### Positive Impacts
1. **Dialogue Vividity:** Characters will sound more distinctive; less generic LLM output
2. **Consistency:** Linguistic patterns won't randomly shift turn-to-turn; tied to character profile
3. **Behavioral Alignment:** High `emotional_leakage` + Low `direct_assertion` will naturally produce different speech than the reverse
4. **Debugging:** When dialogue feels off, developers can check linguistic parameters vs actual output

### Risk Impacts
1. **LLM Comprehension Risk:** If descriptions are vague, LLM may ignore constraints or apply them inconsistently
   - **Mitigation:** Use concrete, grammatically unambiguous examples in descriptions
2. **Token Cost:** Adding 10 descriptions to every linguistic agent call increases prompt size
   - **Mitigation:** Use compressed format; consider lazy-loading if performance becomes an issue
3. **Evaluation Difficulty:** Harder to verify if linguistic constraints are *actually* applied
   - **Mitigation:** Create evals that check for linguistic markers (fragmentation count, hesitation particles, etc.)

### Interaction with Existing Systems
- **L2 Boundary Core:** No conflict; linguistic texture is downstream of boundary decisions
- **L3 Contextual Agent:** No conflict; provides additional context but doesn't change its outputs
- **L4 Consolidator:** Will receive more detailed linguistic guidance; may improve final output quality
- **Character State Mutations:** Linguistic traits may need to shift over time (see Future section)

---

## Before/After Baseline Capture

To validate the effectiveness of this integration, we will capture baseline outputs before implementation and compare them after.

### Baseline Capture Plan

**When:** Before beginning Stage 1.1 implementation  
**Who:** Engineering team  
**Format:** Save to this document's "Before Implementation Examples" section

**Capture Method:**
1. Run linguistic agent with Kazusa profile under current system (without linguistic texture constraints)
2. Create 3 diverse test scenarios:
   - **Scenario A (Affirmation):** User compliments Kazusa; character_mood=playful; logical_stance=CONFIRM
   - **Scenario B (Confrontation):** User questions Kazusa's past; character_mood=defensive; logical_stance=REFUSE
   - **Scenario C (Ambiguity):** User asks for commitment; character_mood=hesitant; logical_stance=TENTATIVE

3. For each scenario, capture:
   - Full `msg` input to linguistic agent (internal_monologue, logical_stance, character_intent, etc.)
   - Complete `content_anchors` output
   - Complete `rhetorical_strategy` and `linguistic_style` output

4. Save as structured JSON with timestamp

### Expected Changes After Implementation

After linguistic texture integration, we expect to see:

| Linguistic Trait | Expected Observable Change |
|---|---|
| **High emotional_leakage (0.7)** | More ellipses, incomplete thoughts, emotional markers in content_anchors |
| **Low self_deprecation (0.15)** | Fewer self-minimizing phrases; no "I'm just..." or "Sorry, but..." |
| **High abstraction_reframing (0.7)** | More concrete/sensory framing in content_anchors (e.g., translates abstract ideas into physical metaphors) |
| **Low direct_assertion (0.4)** | More hedged phrasing; more questions/deflections in rhetorical_strategy |
| **High formalism_avoidance (0.7)** | Fewer academic connectors ("moreover", "however"); more natural conversational markers |

### After Implementation Validation

**When:** After Stage 1.4 testing  
**Capture:** Run same 3 scenarios with new system  
**Comparison:** Side-by-side review of before/after `content_anchors` and `rhetorical_strategy`

**Success Indicators:**
- ✅ Scenario A output shows more playful, open linguistic patterns
- ✅ Scenario B output shows more defensive, fragmented patterns
- ✅ Scenario C output shows hesitant, hedged patterns
- ✅ All scenarios preserve logical_stance (CONFIRM/REFUSE/TENTATIVE)

---

## Before Implementation Examples

**Captured:** [To be filled in before implementation]

### Scenario A: Affirmation
**Input:**
```json
{
  "character_mood": "playful",
  "global_vibe": "cozy",
  "internal_monologue": "User compliment makes me feel seen; I want to respond naturally without being too guarded",
  "logical_stance": "CONFIRM",
  "character_intent": "BANTER",
  "decontexualized_input": "User says they like Kazusa's personality"
}
```

**Baseline Output (BEFORE implementation):**
```json
{
  "rhetorical_strategy": "[To be captured]",
  "linguistic_style": "[To be captured]",
  "content_anchors": "[To be captured]"
}
```

### Scenario B: Confrontation
**Input:**
```json
{
  "character_mood": "defensive",
  "global_vibe": "tense",
  "internal_monologue": "User is probing my past; I need to shut this down without being cruel",
  "logical_stance": "REFUSE",
  "character_intent": "DEFLECT",
  "decontexualized_input": "User asks about Kazusa's time as 'Cathy Parle'"
}
```

**Baseline Output (BEFORE implementation):**
```json
{
  "rhetorical_strategy": "[To be captured]",
  "linguistic_style": "[To be captured]",
  "content_anchors": "[To be captured]"
}
```

### Scenario C: Ambiguity
**Input:**
```json
{
  "character_mood": "hesitant",
  "global_vibe": "intimate",
  "internal_monologue": "User wants commitment but I'm not sure if I'm ready; I want to be honest but also protect myself",
  "logical_stance": "TENTATIVE",
  "character_intent": "CLARIFY",
  "decontexualized_input": "User asks if Kazusa wants to deepen their relationship"
}
```

**Baseline Output (BEFORE implementation):**
```json
{
  "rhetorical_strategy": "[To be captured]",
  "linguistic_style": "[To be captured]",
  "content_anchors": "[To be captured]"
}
```

---

## Stages

### Stage 1: Implementation & Testing (Single Stage)

**1.1: Create Translation Functions** (persona_supervisor2_cognition_l2.py or new file)
- [ ] Implement `get_fragmentation_description()`
- [ ] Implement `get_hesitation_density_description()`
- [ ] Implement `get_counter_questioning_description()`
- [ ] Implement `get_softener_density_description()`
- [ ] Implement `get_formalism_avoidance_description()`
- [ ] Implement `get_abstraction_reframing_description()`
- [ ] Implement `get_direct_assertion_description()`
- [ ] Implement `get_emotional_leakage_description()`
- [ ] Implement `get_rhythmic_bounce_description()`
- [ ] Implement `get_self_deprecation_description()`
- **All functions must use self-explanatory, concrete language suitable for local LLM interpretation**
- **Each function returns a single description (not a list)** optimized for local LLM clarity
- Example description (good):
  ```
  "你在表达时经常停顿和重新开始句子。大约每3-4句话中会出现一次短暂的中断、改口或省略号。例如：'我觉得……其实不是那样，我是想说……'"
  ```
- Example description (avoid):
  ```
  "Moderate fragmentation with occasional restarts"  # too abstract, not actionable
  ```

**1.2: Refactor `call_linguistic_agent()`** (persona_supervisor2_cognition_l3.py)
- [ ] Extract `linguistic_texture_profile` from `character_profile`
- [ ] Call all 10 translation functions to produce single descriptions (not lists)
- [ ] Construct formatted constraint string in key-value format:
  ```
  fragmentation: [description]
  hesitation_density: [description]
  ...
  ```
- [ ] Inject into system prompt via `.format(linguistic_texture_descriptions=...)`
- [ ] **Add debug logging:** Log the complete formatted descriptions before sending to LLM
  ```python
  logger.debug(f"Linguistic texture descriptions:\n{linguistic_texture_descriptions}")
  ```
  This helps diagnose if local LLM is ignoring/misinterpreting constraints

**1.3: Update `_LINGUISTIC_AGENT_PROMPT`**
- [ ] Add new section: **"# 语言质感约束 (Linguistic Texture Constraints)"**
- [ ] Define how LLM should apply constraints (show examples)
- [ ] Add explicit constraint: linguistic texture ≠ logical override
- [ ] Add application examples

**1.4: Testing & Validation**
- [ ] Manual testing: Generate responses with different profiles; check for linguistic variety
- [ ] Check response of CONFIRM/REFUSE/TENTATIVE stands firm despite high `emotional_leakage`
- [ ] Verify Kazusa's output changes from baseline when linguistic traits are applied
- [ ] Create simple evals for linguistic markers:
  - High `fragmentation` → check for ellipses, interruptions, fragments
  - High `hesitation_density` → check for fillers ("那个", "嗯", "……")
  - High `emotional_leakage` → check for inconsistent punctuation, partial words

**1.5: Documentation**
- [ ] Add code comments explaining the 10-level scale pattern
- [ ] Document how to tune linguistic values for new characters
- [ ] Add section to character-creator SKILL if needed

---

## Checklist

### Pre-Implementation
- [ ] Review `CHARACTER_PROFILE_SCHEMA.json` Part 4 (linguistic_texture_profile definitions)
- [ ] Review SKILL.md Part 4 for interpretation guidelines
- [ ] Review `persona_supervisor2_cognition_l2.py` to understand translation function pattern
- [ ] Identify if translation functions should go in L2 file or new dedicated file

### Implementation
- [ ] Create all 10 translation functions with single, concrete descriptions (not lists)
- [ ] **Verify each description is self-explanatory** with concrete examples (no abstract concepts)
- [ ] Update `call_linguistic_agent()` to extract and translate linguistic_texture_profile
- [ ] Update `_LINGUISTIC_AGENT_PROMPT` with new section
- [ ] Add placeholder variables to `.format()` call with debug logging
- [ ] Verify no imports are missing
- [ ] Add defensive null-check for missing `linguistic_texture_profile`
- [ ] Review descriptions for local LLM clarity before submitting for review

### Testing
- [ ] Unit tests for each translation function (input 0.0, 0.5, 1.0 → verify output)
- [ ] Integration test: Full L3 call with Kazusa profile → check output includes constraints
- [ ] Qualitative testing: Do responses sound more "Kazusa-like"?
- [ ] Check consistency: Same input + same profile → similar linguistic style across multiple calls

### Documentation
- [ ] Update docstrings in translation functions
- [ ] Add comment block explaining the integration pattern
- [ ] Update development plan if scope changes discovered

---

## Future: Dynamic Linguistic Trait Evolution

**Why This Matters:**  
As the relationship evolves, the character's linguistic patterns should shift. For example:
- Early in relationship (low affinity): High `direct_assertion`, Low `emotional_leakage` (guarded)
- High affinity, intimate moment: Low `direct_assertion`, High `emotional_leakage` (vulnerable)
- After boundary violation: Low `rhythmic_bounce`, High `hesitation_density` (withdrawn)

**Proposed Approach (Stage 3+):**

### Phase 1: Affinity-Based Modulation (Near-term)
- Modify translation functions to accept an optional `affinity_context` parameter
- Define modulation rules:
  ```
  If affinity < 300 (unfamiliar):
    - emotional_leakage modifier: -0.2
    - direct_assertion modifier: +0.2
    
  If affinity > 700 (intimate):
    - emotional_leakage modifier: +0.2
    - direct_assertion modifier: -0.2
    - hesitation_density modifier: +0.1 (more stuttering due to vulnerability)
  ```
- Store base profile in character_state, compute effective profile at runtime

### Phase 2: State-Based Modulation (Mid-term)
- Create modulation matrix based on `character_mood`, `boundary_violation_count`, `recent_conflict`:
  ```
  If mood == "angry" OR recent_boundary_violation:
    - rhythmic_bounce modifier: -0.3 (less playful)
    - fragmentation modifier: +0.2 (more choppy)
    - direct_assertion modifier: +0.15 (more confrontational)
  
  If in "intimate moment" state:
    - self_deprecation modifier: +0.2 (more vulnerable humor)
    - softener_density modifier: +0.2 (more hedging)
  ```

### Phase 3: Long-term Personality Shift (Future research)
- Implement slow decay/growth over many conversations
- Example: If character repeatedly exhibits high `emotional_leakage` when intimate, base `emotional_leakage` slowly increases to 0.75+ (character becomes more openly emotional)
- Would require:
  - Persistent trait tracking (store deltas in character_state)
  - Monthly review/reset of accumulated deltas
  - Evals to verify shifts feel psychologically authentic

**Estimated Timeline:**
- Phase 1 (affinity modulation): 1–2 weeks (can be done during L4 consolidation)
- Phase 2 (state-based): 2–3 weeks (requires mood/conflict event tracking)
- Phase 3 (long-term shift): Deferred to Stage 4 (requires deeper personality research)

**Current Limitation:**  
This plan assumes linguistic_texture_profile remains *constant* for the implementation stage. Future dynamic adjustment will require separate work.

---

## Notes

### Description Quality Criteria for Local LLM
Since descriptions are evaluated by a local LLM with potential interpretation limitations, follow these rules:

1. **Use imperative language:** "use more X", "avoid Y", "include Z" (not "you are X")
2. **Be concrete:** Provide examples. "Include ~2-3 hesitation particles per paragraph like '那个', '嗯', '……'" (not "moderate hesitation")
3. **Avoid meta-reference:** Don't say "compared to fragmentation_score=0.3". Each description stands alone.
4. **Use actionable markers:** Mention specific linguistic markers the LLM can control:
   - For fragmentation: "interrupted sentences", "ellipses (……)", "false starts"
   - For hesitation: "filler words like 那个/嗯/呃", "trailing particles"
   - For emotional_leakage: "punctuation changes (!!! vs...)", "word choice inconsistencies", "sentence-final particles"
5. **Keep it brief:** 2-3 sentences max. Local LLMs struggle with long context.

### Naming Convention
- Translation functions: `get_[parameter]_description(score: float) -> str`
- Constraint descriptions in prompt: `{linguistic_texture_descriptions}` (formatted string, not raw dict)

### Kazusa Example Values (For Testing)
```json
"linguistic_texture_profile": {
    "fragmentation": 0.3,
    "hesitation_density": 0.5,
    "counter_questioning": 0.6,
    "softener_density": 0.5,
    "formalism_avoidance": 0.7,
    "abstraction_reframing": 0.7,
    "direct_assertion": 0.4,
    "emotional_leakage": 0.7,
    "rhythmic_bounce": 0.45,
    "self_deprecation": 0.15
}
```

Expected behavior:
- **High values:** emotional_leakage (0.7), abstraction_reframing (0.7), formalism_avoidance (0.7)
  - → Speech should leak emotion, reframe into concrete/sensory, avoid academic connectors
- **Low values:** self_deprecation (0.15), fragmentation (0.3), direct_assertion (0.4)
  - → Should NOT self-minimize, should speak in smooth sentences, should be indirect/hedging
- **Result:** Dialogue should sound like "傲娇 with emotional leaks, sensory-grounded, indirect but smooth"

---

## Success Criteria

1. **Linguistic constraints are applied:** Output dialogue varies based on linguistic_texture_profile values
2. **Logical stance is preserved:** CONFIRM remains CONFIRM even with high emotional_leakage; REFUSE remains REFUSE
3. **Consistency:** Multiple calls with same profile produce similar linguistic patterns
4. **Kazusa sounds more vivid:** Qualitative assessment from team review
5. **No performance degradation:** LLM latency remains within acceptable range
6. **Evals pass:** Custom evals for fragmentation, hesitation, emotional markers show expected patterns

---

**Owner:** [Engineering team]  
**Timeline:** 3–4 weeks  
**Blockers:** None identified  
**Dependencies:** Completion of boundary_profile integration (currently done)

