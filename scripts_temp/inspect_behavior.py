"""Inspect counterfactual behavior artifacts for quality."""
import json
from pathlib import Path

base = Path("test_artifacts/character_identity_growth")
categories = [
    "self_image",
    "personality_brief",
    "boundary_profile",
    "linguistic_texture_profile",
    "visual_characterization",
]

for cat in categories:
    path = base / f"behavior_counterfactual_{cat}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 72)
    print(f"CATEGORY: {cat}")
    print("=" * 72)

    # Show the actual changed values
    base_id = data["base_identity"]
    changed_id = data["changed_identity"]

    if cat in base_id and isinstance(base_id[cat], dict):
        for key in base_id[cat]:
            bv = base_id[cat][key]
            cv = changed_id[cat][key]
            if bv != cv:
                print(f"  CHANGED {cat}.{key}:")
                print(f"    base:    {bv}")
                print(f"    changed: {cv}")
    elif cat in base_id:
        bv = base_id[cat]
        cv = changed_id[cat]
        if bv != cv:
            print(f"  CHANGED {cat}:")
            print(f"    base:    {str(bv)[:100]}")
            print(f"    changed: {str(cv)[:100]}")

    # Show per-sample proposal decisions and key projections
    for state_name, samples in [
        ("BASE", data["base_samples"]),
        ("CHANGED", data["changed_samples"]),
    ]:
        print(f"\n  {state_name} samples:")
        for i, s in enumerate(samples):
            act = s["proposal_action"]
            ver = s["review_verdict"]
            pol = s["policy_status"]
            si = s.get("cognition_moral_identity_self_image", {})
            sc = si.get("self_concept", "N/A") if isinstance(si, dict) else "N/A"
            sp = s.get("surface_text_personality", {})
            tempo = sp.get("tempo", "N/A") if isinstance(sp, dict) else "N/A"
            vis = s.get("surface_visual", {})
            vc = vis.get("visual_characterization", "N/A") if isinstance(vis, dict) else "N/A"
            print(f"    sample {i+1}: action={act} verdict={ver} policy={pol}")
            if cat == "self_image":
                print(f"      self_concept: {str(sc)[:80]}")
            elif cat == "personality_brief":
                print(f"      tempo: {tempo}")
            elif cat == "visual_characterization":
                print(f"      visual: {str(vc)[:80]}")
    print()
