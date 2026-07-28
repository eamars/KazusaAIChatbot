"""Inspect raw LLM call outputs from counterfactual behavior tests."""
import json
from pathlib import Path

base = Path("test_artifacts/character_identity_growth")

# Focus on self_image which has the clearest semantic diff
path = base / "behavior_counterfactual_self_image.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Show raw LLM output differences for sample 1
base_s1 = data["base_samples"][0]
changed_s1 = data["changed_samples"][0]

print("=== SELF_IMAGE: BASE sample 1 raw LLM outputs ===")
for idx, call in enumerate(base_s1["calls"]):
    print(f"  Call {idx}: {call['stage_name']}")
    print(call["raw_output"][:800])
    print()

print("=== SELF_IMAGE: CHANGED sample 1 raw LLM outputs ===")
for idx, call in enumerate(changed_s1["calls"]):
    print(f"  Call {idx}: {call['stage_name']}")
    print(call["raw_output"][:800])
    print()

# Also check personality_brief
path2 = base / "behavior_counterfactual_personality_brief.json"
with open(path2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

base_s1 = data2["base_samples"][0]
changed_s1 = data2["changed_samples"][0]

print("=== PERSONALITY_BRIEF: BASE sample 1, call 0 raw output ===")
print(base_s1["calls"][0]["raw_output"][:800])
print()
print("=== PERSONALITY_BRIEF: CHANGED sample 1, call 0 raw output ===")
print(changed_s1["calls"][0]["raw_output"][:800])
print()
