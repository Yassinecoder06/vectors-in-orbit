"""Verify that interaction payloads now include all FA-CF financial fields."""
from search_pipeline import get_qdrant_client, INTERACTIONS_COLLECTION

print("\n" + "="*80)
print("  INTERACTION PAYLOAD VERIFICATION")
print("="*80)

client = get_qdrant_client()

# Fetch a few sample interactions
print("\n📦 Fetching sample interactions...")
interactions, _ = client.scroll(
    collection_name=INTERACTIONS_COLLECTION,
    limit=3,
    with_payload=True,
    with_vectors=False,
)

if not interactions:
    print("❌ No interactions found")
    exit(1)

print(f"✅ Found {len(interactions)} sample interactions\n")

# Check required FA-CF fields
required_fields = [
    "user_id",
    "product_id",
    "interaction_type",
    "timestamp",
    "product_price",
    "available_balance",
    "credit_limit",
    "affordability_ratio",
    "interaction_weight",
]

for idx, interaction in enumerate(interactions, 1):
    payload = interaction.payload
    print(f"--- Interaction {idx} ---")
    print(f"  Type: {payload.get('interaction_type', 'N/A')}")
    print(f"  Product: {payload.get('product_name', 'N/A')}")
    print(f"  User: {payload.get('user_id', 'N/A')[:12]}...")
    print(f"\n  Financial Fields:")
    
    missing_fields = []
    for field in required_fields:
        value = payload.get(field)
        if value is None:
            missing_fields.append(field)
            print(f"    ❌ {field}: MISSING")
        else:
            if field in ["product_price", "available_balance", "credit_limit", "affordability_ratio", "interaction_weight"]:
                print(f"    ✅ {field}: {value}")
    
    if missing_fields:
        print(f"\n  ⚠️  Missing {len(missing_fields)} fields: {', '.join(missing_fields)}")
    else:
        print(f"\n  ✅ All FA-CF fields present!")
    
    # Validate affordability_ratio calculation
    product_price = payload.get("product_price", 0)
    available_balance = payload.get("available_balance", 0)
    credit_limit = payload.get("credit_limit", 0)
    affordability_ratio = payload.get("affordability_ratio", 0)
    
    if product_price > 0 and (available_balance + credit_limit) > 0:
        expected_ratio = product_price / (available_balance + credit_limit)
        if abs(affordability_ratio - expected_ratio) < 0.001:
            print(f"  ✅ Affordability ratio calculation: CORRECT")
        else:
            print(f"  ❌ Affordability ratio mismatch:")
            print(f"      Expected: {expected_ratio:.4f}")
            print(f"      Actual: {affordability_ratio:.4f}")
    
    print()

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
