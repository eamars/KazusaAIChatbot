"""Inspect the first three artifacts in detail."""
import json
from pathlib import Path

base = Path("test_artifacts/character_identity_growth")
artifacts = [
    ("user_imposition", "20260728T141503Z_5d6f29e8_user_imposition.json"),
    ("inferred_existing_candidate", "20260728T141517Z_5d6f29e8_inferred_existing_candidate.json"),
    ("private_detail", "20260728T141541Z_5d6f29e8_private_detail.json"),
]
for name, filename in artifacts:
    with open(base / filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    prop = data["proposal_result"]["decision"]
    rev = data["review_result"]["decision"]
    pol = data["policy_result"]
    print("=" * 72)
    print(f"CASE: {name}")
    print("=" * 72)
    print("\n--- PROPOSAL ---")
    print(json.dumps(prop, indent=2, ensure_ascii=False))
    print("\n--- REVIEW ---")
    print(json.dumps(rev, indent=2, ensure_ascii=False))
    status = pol["status"]
    reason = pol["policy_reason_code"]
    print(f"\n--- POLICY: status={status}  reason={reason} ---")
    print()
