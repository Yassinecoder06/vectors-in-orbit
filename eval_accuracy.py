import numpy as np
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Import your existing pipeline logic
# Note: Ensure search_pipeline.py is in the same directory
from search_pipeline import SearchPipeline 

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
pipeline = SearchPipeline()

def calculate_ndcg(rank):
    """Calculates Discounted Cumulative Gain for a single relevant item."""
    if rank > 0:
        return 1 / np.log2(rank + 1)
    return 0

def run_evaluation(k=10, sample_size=50):
    print(f"--- Starting Accuracy Evaluation (K={k}, Sample={sample_size} users) ---")
    
    # 1. Get users who have purchases in interaction_memory
    interactions = client.scroll(
        collection_name="interaction_memory",
        limit=sample_size * 5,
        with_payload=True
    )[0]

    # Filter for purchases
    purchases = [i for i in interactions if i.payload.get("interaction_type") == "purchase"]
    
    if not purchases:
        print("No purchases found in interaction_memory. Using 'add_to_cart' instead...")
        purchases = [i for i in interactions if i.payload.get("interaction_type") == "add_to_cart"]

    user_hits = []
    ndcg_scores = []

    # 2. Test the pipeline for each user
    tested_users = set()
    for p in purchases:
        user_id = p.payload.get("user_id")
        target_product_id = p.payload.get("product_id")
        
        if user_id in tested_users or len(tested_users) >= sample_size:
            continue
        
        tested_users.add(user_id)

        # Run your actual pipeline
        # We assume the pipeline takes a user_id and returns a list of results
        results = pipeline.search(query="", user_id=user_id, top_k=k)
        
        # Check if the target product is in the top K
        result_ids = [res['id'] for res in results]
        
        if target_product_id in result_ids:
            rank = result_ids.index(target_product_id) + 1
            user_hits.append(1)
            ndcg_scores.append(calculate_ndcg(rank))
        else:
            user_hits.append(0)
            ndcg_scores.append(0)

    # 3. Calculate Final Metrics
    recall_k = np.mean(user_hits) if user_hits else 0
    m_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

    print("\n--- RESULTS ---")
    print(f"Recall@{k}: {recall_k:.4f}")
    print(f"NDCG@{k}:   {m_ndcg:.4f}")
    print("----------------")
    return recall_k, m_ndcg

if __name__ == "__main__":
    run_evaluation()

