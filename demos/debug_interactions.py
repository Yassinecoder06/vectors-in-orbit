from interaction_logger import _get_qdrant_client_safe
from qdrant_client import models
import json

client, _, collection = _get_qdrant_client_safe()

# Get 10 most recent interactions
points, _ = client.scroll(
    collection_name=collection,
    limit=10,
    with_payload=True,
    with_vectors=False,
    # Sort by natural insertion order often puts newest last, but no reliable sort without timestamp
    # We'll filter by timestamp
)

# Actually, let's sort by timestamp descending using python since scroll doesn't sort
sorted_points = sorted(
    points, 
    key=lambda x: x.payload.get("timestamp", 0), 
    reverse=True
)

print(f"Total points retrieved: {len(points)}")
print("--- Recent Interactions ---")
for p in sorted_points[:10]:
    pl = p.payload
    print(f"Type: {pl.get('interaction_type')} | Product: {pl.get('product_name')} (ID: {pl.get('product_id')}) | TS: {pl.get('timestamp')}")
