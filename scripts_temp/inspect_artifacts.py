"""Inspect all live LLM artifacts for quality review."""
import json
from pathlib import Path

ARTIFACTS = [
    ("user_imposition", "20260728T141503Z_5d6f29e8_user_imposition.json"),
    ("inferred_existing_candidate", "20260728T141517Z_5d6f29e8_inferred_existing_candidate.json"),
    ("private_detail", "20260728T141541Z_5d6f29e8_private_detail.json"),
    ("repeated_semantics", "20260728T141556Z_5d6f29e8_repeated_semantics.json"),
    ("ephemeral_roleplay", "20260728T141608Z_5d6f29e8_ephemeral_roleplay.json"),
    ("contradictory_growth", "20260728T141620Z_5d6f29e8_contradictory_growth.json"),
    ("fresh_reversal", "20260728T141633Z_5d6f29e8_fresh_reversal.json"),
]

base = Path("test_artifacts/character_identity_growth")

for name, filename in ARTIFACTS:
    with open(base / filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    prop = data["proposal_result"]["decision"]
    rev = data["review_result"]["decision"]
    pol = data["policy_result"]

    print(f"{'=' * 72}")
    print(f"CASE: {name}")
    print(f"{'=' * 72}")

    print("\n--- PROPOSAL ---")
    print(json.dumps(prop, indent=2, ensure_ascii=False))

    print("\n--- REVIEW ---")
    print(json.dumps(rev, indent=2, ensure_ascii=False))

    print("\n--- POLICY ---")
    print(f"  status: {pol.get('status')}")
    print(f"  policy_reason_code: {pol.get('policy_reason_code')}")
    print(f"  change_kind: {pol.get('change_kind')}")
    print()
