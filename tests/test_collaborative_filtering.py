"""
Deterministic Test for Collaborative Filtering in FinCommerce

This script PROVES whether collaborative filtering works by:
1. Clearing all interaction memory
2. Creating controlled test users and products
3. Inserting specific interactions to create user similarity
4. Verifying that similar user behavior influences recommendations

PASS CRITERIA:
- User similarity >= 0.7 between user_A and user_B
- Product P1 (ThinkPad) moves UP in ranking for user_B after CF
- CF boost is measurably applied to shared products

FAIL CRITERIA:
- Any step fails to execute
- User similarity < 0.7
- P1 does NOT move up for user_B
- CF appears inactive
"""

import sys
import time
import logging
from typing import List, Dict, Any, Tuple
from uuid import uuid4
import numpy as np
from qdrant_client import QdrantClient, models

# Import existing modules
from search_pipeline import (
    get_qdrant_client,
    embed_query,
    PRODUCTS_COLLECTION,
    INTERACTIONS_COLLECTION,
    get_user_behavior_vector,
    get_collaborative_scores,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Test constants
USER_A = "test_user_A"
USER_B = "test_user_B"
USER_C = "test_user_C"

PRODUCT_IDS = {
    "P1": str(uuid4()),  # ThinkPad
    "P2": str(uuid4()),  # Vivobook
    "P3": str(uuid4()),  # MacBook
    "P4": str(uuid4()),  # Dell XPS
}

PRODUCT_NAMES = {
    "P1": "Lenovo ThinkPad X1 Carbon",
    "P2": "ASUS Vivobook Pro",
    "P3": "Apple MacBook Pro M3",
    "P4": "Dell XPS 15",
}

PRODUCT_DESCRIPTIONS = {
    "P1": "laptop for coding programming development business professional ThinkPad",
    "P2": "laptop for coding programming development creator Vivobook ASUS",
    "P3": "laptop for coding programming development creative MacBook Apple",
    "P4": "laptop for coding programming development premium Dell XPS",
}

INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0,
}


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def step_1_clear_interactions(client: QdrantClient):
    """STEP 1: Clear all interaction memory."""
    print_section("STEP 1: CLEAR INTERACTION MEMORY")
    
    try:
        # Delete all points in interaction_memory collection
        client.delete(
            collection_name=INTERACTIONS_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(gte=0)  # Match all
                        )
                    ]
                )
            )
        )
        logger.info("✅ Cleared all interactions from interaction_memory collection")
        
        # Verify deletion
        count = client.count(collection_name=INTERACTIONS_COLLECTION)
        print(f"✅ Interaction count after clearing: {count.count}")
        
        if count.count > 0:
            logger.warning(f"⚠️  Warning: {count.count} interactions still present")
        
    except Exception as e:
        logger.error(f"❌ Failed to clear interactions: {e}")
        raise


def step_2_create_products(client: QdrantClient) -> Dict[str, List[float]]:
    """STEP 2: Create 4 controlled test products."""
    print_section("STEP 2: CREATE CONTROLLED PRODUCTS")
    
    product_vectors = {}
    
    for pid, product_id in PRODUCT_IDS.items():
        # Generate distinct embedding for each product
        description = PRODUCT_DESCRIPTIONS[pid]
        vector = embed_query(description)
        product_vectors[pid] = vector
        
        # Check if product exists, update or insert
        try:
            existing = client.retrieve(
                collection_name=PRODUCTS_COLLECTION,
                ids=[product_id],
                with_vectors=False,
            )
            action = "Updated"
        except:
            action = "Inserted"
        
        # Upsert product
        client.upsert(
            collection_name=PRODUCTS_COLLECTION,
            points=[
                models.PointStruct(
                    id=product_id,
                    vector=vector,
                    payload={
                        "product_id": product_id,
                        "name": PRODUCT_NAMES[pid],
                        "description": description,
                        "category": "Electronics",
                        "brand": pid,  # Use P1, P2, etc. as brand for tracking
                        "price": 1200.0 + (int(pid[1]) * 100),  # P1=1300, P2=1400, etc.
                        "in_stock": True,
                    }
                )
            ],
            wait=True
        )
        
        print(f"✅ {action} {pid} ({PRODUCT_NAMES[pid]}): ID={product_id[:8]}...")
        print(f"   Vector dim: {len(vector)}, L2 norm: {np.linalg.norm(vector):.3f}")
    
    # Verify distinctness
    print("\n🔍 Verifying product vector distinctness:")
    for i, (p1, v1) in enumerate(product_vectors.items()):
        for p2, v2 in list(product_vectors.items())[i+1:]:
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            print(f"   {p1} vs {p2}: cosine similarity = {similarity:.3f}")
    
    return product_vectors


def step_3_baseline_search(client: QdrantClient) -> Dict[str, List[Dict]]:
    """STEP 3: Baseline search WITHOUT collaborative filtering."""
    print_section("STEP 3: BASELINE SEARCH (NO COLLABORATIVE FILTERING)")
    
    query = "laptop for coding"
    query_vector = embed_query(query)
    
    print(f"Query: '{query}'")
    print(f"Query vector dim: {len(query_vector)}\n")
    
    baseline_results = {}
    
    for user_id in [USER_A, USER_B]:
        # Semantic search only (no CF yet)
        results = client.query_points(
            collection_name=PRODUCTS_COLLECTION,
            query=query_vector,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        
        user_results = []
        print(f"📊 Baseline results for {user_id}:")
        for rank, point in enumerate(results.points[:4], 1):
            payload = point.payload
            product_name = payload.get("name", "Unknown")
            brand = payload.get("brand", "?")
            score = point.score
            
            user_results.append({
                "rank": rank,
                "product_id": payload.get("product_id"),
                "brand": brand,
                "name": product_name,
                "score": score,
            })
            
            print(f"   #{rank}: {brand} - {product_name}")
            print(f"        Score: {score:.4f}")
        
        baseline_results[user_id] = user_results
        print()
    
    return baseline_results


def step_4_insert_interactions(client: QdrantClient, product_vectors: Dict[str, List[float]]):
    """STEP 4: Insert controlled interactions to create user similarity."""
    print_section("STEP 4: INSERT CONTROLLED INTERACTIONS")
    
    current_time = int(time.time())
    
    # Define interactions that create similarity between user_A and user_B
    interactions = [
        # User A: Clicked + Purchased P1 (ThinkPad)
        {
            "user_id": USER_A,
            "product_key": "P1",
            "interaction_type": "click",
            "timestamp": current_time - 3600,  # 1 hour ago
        },
        {
            "user_id": USER_A,
            "product_key": "P1",
            "interaction_type": "purchase",
            "timestamp": current_time - 3500,  # 58 min ago
        },
        # User A: Viewed P2 (Vivobook)
        {
            "user_id": USER_A,
            "product_key": "P2",
            "interaction_type": "view",
            "timestamp": current_time - 7200,  # 2 hours ago
        },
        
        # User B: Clicked + Added to cart P1 (same product!)
        {
            "user_id": USER_B,
            "product_key": "P1",
            "interaction_type": "click",
            "timestamp": current_time - 3000,  # 50 min ago
        },
        {
            "user_id": USER_B,
            "product_key": "P1",
            "interaction_type": "add_to_cart",
            "timestamp": current_time - 2900,  # 48 min ago
        },
        # User B: Viewed P2 (same product!)
        {
            "user_id": USER_B,
            "product_key": "P2",
            "interaction_type": "view",
            "timestamp": current_time - 6000,  # 100 min ago
        },
        
        # User C: Different behavior (interacted with P3, P4)
        {
            "user_id": USER_C,
            "product_key": "P3",
            "interaction_type": "click",
            "timestamp": current_time - 5000,
        },
        {
            "user_id": USER_C,
            "product_key": "P4",
            "interaction_type": "view",
            "timestamp": current_time - 4000,
        },
    ]
    
    print("Inserting interactions:")
    points = []
    
    for interaction in interactions:
        product_key = interaction["product_key"]
        product_id = PRODUCT_IDS[product_key]
        product_vector = product_vectors[product_key]
        interaction_type = interaction["interaction_type"]
        weight = INTERACTION_WEIGHTS[interaction_type]
        
        # Create interaction point
        point = models.PointStruct(
            id=str(uuid4()),
            vector=product_vector,  # Use ACTUAL product vector for CF
            payload={
                "user_id": interaction["user_id"],
                "product_id": product_id,
                "interaction_type": interaction_type,
                "timestamp": interaction["timestamp"],
                "weight": weight,
                "product_name": PRODUCT_NAMES[product_key],
                "category": "Electronics",
                "brand": product_key,
                "price": 1200.0 + (int(product_key[1]) * 100),
            }
        )
        points.append(point)
        
        print(f"   ✅ {interaction['user_id']}: {interaction_type.upper()} {product_key} "
              f"(weight={weight:.1f})")
    
    # Insert all interactions
    client.upsert(
        collection_name=INTERACTIONS_COLLECTION,
        points=points,
        wait=True
    )
    
    # Verify insertion
    count = client.count(collection_name=INTERACTIONS_COLLECTION)
    print(f"\n✅ Total interactions inserted: {count.count}")
    
    if count.count != len(interactions):
        raise AssertionError(
            f"Expected {len(interactions)} interactions, found {count.count}"
        )


def step_5_verify_user_similarity(client: QdrantClient) -> float:
    """STEP 5: Verify user similarity between user_A and user_B."""
    print_section("STEP 5: USER SIMILARITY VERIFICATION")
    
    # Get behavior vectors for user_A and user_B
    vector_A = get_user_behavior_vector(client, USER_A)
    vector_B = get_user_behavior_vector(client, USER_B)
    vector_C = get_user_behavior_vector(client, USER_C)
    
    if vector_A is None:
        raise AssertionError(f"❌ FAILED: user_A has no behavior vector")
    if vector_B is None:
        raise AssertionError(f"❌ FAILED: user_B has no behavior vector")
    
    print(f"✅ user_A behavior vector: dim={len(vector_A)}, norm={np.linalg.norm(vector_A):.3f}")
    print(f"✅ user_B behavior vector: dim={len(vector_B)}, norm={np.linalg.norm(vector_B):.3f}")
    if vector_C:
        print(f"✅ user_C behavior vector: dim={len(vector_C)}, norm={np.linalg.norm(vector_C):.3f}")
    
    # Compute cosine similarity
    similarity_AB = np.dot(vector_A, vector_B) / (
        np.linalg.norm(vector_A) * np.linalg.norm(vector_B)
    )
    
    print(f"\n🔍 Cosine Similarity:")
    print(f"   user_A vs user_B: {similarity_AB:.4f}")
    
    if vector_C:
        similarity_AC = np.dot(vector_A, vector_C) / (
            np.linalg.norm(vector_A) * np.linalg.norm(vector_C)
        )
        similarity_BC = np.dot(vector_B, vector_C) / (
            np.linalg.norm(vector_B) * np.linalg.norm(vector_C)
        )
        print(f"   user_A vs user_C: {similarity_AC:.4f}")
        print(f"   user_B vs user_C: {similarity_BC:.4f}")
    
    # HARD ASSERTION: Similarity must be >= 0.7
    print(f"\n📏 Similarity threshold check: {similarity_AB:.4f} >= 0.7")
    if similarity_AB < 0.7:
        raise AssertionError(
            f"❌ FAILED: User similarity {similarity_AB:.4f} < 0.7\n"
            f"   Users A and B should be similar (both interacted with P1 and P2)"
        )
    
    print(f"✅ PASSED: Users A and B are similar (similarity={similarity_AB:.4f})")
    return similarity_AB


def step_6_cf_aware_search(client: QdrantClient, baseline: Dict[str, List[Dict]]) -> List[Dict]:
    """STEP 6: Run collaborative filtering-aware search for user_B."""
    print_section("STEP 6: COLLABORATIVE FILTERING SEARCH")
    
    query = "laptop for coding"
    query_vector = embed_query(query)
    
    # Get semantic search results
    semantic_results = client.query_points(
        collection_name=PRODUCTS_COLLECTION,
        query=query_vector,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    
    # Extract candidate product IDs
    candidate_ids = [p.payload.get("product_id") for p in semantic_results.points]
    
    # Get collaborative scores for user_B
    print(f"🔍 Computing collaborative scores for {USER_B}...")
    collab_scores = get_collaborative_scores(client, USER_B, candidate_ids)
    
    print(f"📊 Collaborative scores:")
    for pid_key, pid_uuid in PRODUCT_IDS.items():
        if pid_uuid in collab_scores:
            score = collab_scores[pid_uuid]
            print(f"   {pid_key} ({PRODUCT_NAMES[pid_key]}): {score:.4f}")
    
    # Combine semantic + collaborative scores
    cf_results = []
    for point in semantic_results.points:
        payload = point.payload
        product_id = payload.get("product_id")
        brand = payload.get("brand", "?")
        name = payload.get("name", "Unknown")
        
        semantic_score = point.score
        collab_score = collab_scores.get(product_id, 0.0)
        
        # Weighted combination: 70% semantic, 30% collaborative
        final_score = 0.7 * semantic_score + 0.3 * collab_score
        
        cf_results.append({
            "product_id": product_id,
            "brand": brand,
            "name": name,
            "semantic_score": semantic_score,
            "collab_score": collab_score,
            "final_score": final_score,
        })
    
    # Sort by final score
    cf_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Add ranks
    for rank, result in enumerate(cf_results[:4], 1):
        result["rank"] = rank
    
    print(f"\n📊 CF-aware results for {USER_B}:")
    for result in cf_results[:4]:
        print(f"   #{result['rank']}: {result['brand']} - {result['name']}")
        print(f"        Semantic: {result['semantic_score']:.4f}, "
              f"Collab: {result['collab_score']:.4f}, "
              f"Final: {result['final_score']:.4f}")
    
    return cf_results


def step_7_compare_rankings(
    baseline: Dict[str, List[Dict]],
    cf_results: List[Dict],
) -> Tuple[bool, str]:
    """STEP 7: Compare baseline vs CF-aware rankings."""
    print_section("STEP 7: RANKING COMPARISON")
    
    baseline_B = baseline[USER_B]
    
    # Find P1 in both rankings
    p1_uuid = PRODUCT_IDS["P1"]
    
    baseline_p1_rank = None
    cf_p1_rank = None
    
    for result in baseline_B:
        if result["product_id"] == p1_uuid:
            baseline_p1_rank = result["rank"]
            break
    
    for result in cf_results[:4]:
        if result["product_id"] == p1_uuid:
            cf_p1_rank = result["rank"]
            break
    
    print(f"🔍 Product P1 (ThinkPad) ranking for {USER_B}:")
    print(f"   BEFORE CF: Rank #{baseline_p1_rank if baseline_p1_rank else 'Not in top 4'}")
    print(f"   AFTER CF:  Rank #{cf_p1_rank if cf_p1_rank else 'Not in top 4'}")
    
    # Check if P1 moved up
    moved_up = False
    explanation = ""
    
    if baseline_p1_rank is not None and cf_p1_rank is not None:
        if cf_p1_rank < baseline_p1_rank:
            moved_up = True
            explanation = (
                f"✅ P1 MOVED UP: Rank {baseline_p1_rank} → {cf_p1_rank}\n"
                f"   CF ACTIVE: Product promoted due to user_A's purchase history"
            )
        elif cf_p1_rank == baseline_p1_rank:
            explanation = (
                f"⚠️  P1 UNCHANGED: Rank {baseline_p1_rank} (CF may not be strong enough)"
            )
        else:
            explanation = (
                f"❌ P1 MOVED DOWN: Rank {baseline_p1_rank} → {cf_p1_rank} (CF FAILED)"
            )
    elif cf_p1_rank is not None and baseline_p1_rank is None:
        moved_up = True
        explanation = (
            f"✅ P1 ENTERED TOP 4: Now at rank {cf_p1_rank}\n"
            f"   CF ACTIVE: Product promoted due to user_A's purchase history"
        )
    elif cf_p1_rank is None and baseline_p1_rank is not None:
        explanation = (
            f"❌ P1 DROPPED OUT: Was rank {baseline_p1_rank}, now not in top 4 (CF FAILED)"
        )
    else:
        explanation = "❌ P1 not found in either ranking (TEST SETUP ERROR)"
    
    print(f"\n{explanation}")
    
    # Print full comparison
    print("\n📊 Full ranking comparison:")
    print(f"{'Rank':<6} {'BEFORE CF':<30} {'AFTER CF':<30} {'Change':<10}")
    print("-" * 80)
    
    for rank in range(1, 5):
        before = next((r for r in baseline_B if r["rank"] == rank), None)
        after = next((r for r in cf_results[:4] if r["rank"] == rank), None)
        
        before_name = f"{before['brand']}" if before else "-"
        after_name = f"{after['brand']}" if after else "-"
        
        change = ""
        if before and after:
            if before["brand"] == after["brand"]:
                change = "SAME"
            elif after["brand"] == "P1":
                change = "⬆️ CF BOOST"
            else:
                change = "CHANGED"
        
        print(f"#{rank:<5} {before_name:<30} {after_name:<30} {change:<10}")
    
    return moved_up, explanation


def step_8_final_assertion(moved_up: bool, explanation: str):
    """STEP 8: Final assertion - CF must be active."""
    print_section("STEP 8: FINAL VALIDATION")
    
    if not moved_up:
        print("❌ TEST FAILED")
        print(explanation)
        print("\n" + "=" * 80)
        print("VERDICT: COLLABORATIVE FILTERING IS NOT WORKING")
        print("=" * 80)
        raise AssertionError(
            "❌ COLLABORATIVE FILTERING TEST FAILED\n"
            f"{explanation}\n"
            "Expected: P1 should move UP in ranking for user_B\n"
            "Reason: user_A purchased P1, and users A and B are similar"
        )
    
    print("✅ TEST PASSED")
    print(explanation)
    print("\n" + "=" * 80)
    print("VERDICT: COLLABORATIVE FILTERING IS ACTIVE AND WORKING")
    print("=" * 80)
    print("\nEvidence:")
    print("1. ✅ User similarity verified (A and B both interacted with P1 and P2)")
    print("2. ✅ Collaborative scores computed successfully")
    print("3. ✅ P1 ranking improved for user_B based on user_A's behavior")
    print("4. ✅ CF boost measurably applied to shared products")


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("  COLLABORATIVE FILTERING DETERMINISTIC TEST")
    print("=" * 80)
    print("\nObjective: Prove CF is active by creating controlled user similarity")
    print("Expected: Product P1 should rank higher for user_B after user_A purchases it")
    print("\n" + "=" * 80)
    
    try:
        # Initialize Qdrant client
        client = get_qdrant_client()
        
        # Execute test steps
        step_1_clear_interactions(client)
        product_vectors = step_2_create_products(client)
        baseline_results = step_3_baseline_search(client)
        step_4_insert_interactions(client, product_vectors)
        similarity = step_5_verify_user_similarity(client)
        cf_results = step_6_cf_aware_search(client, baseline_results)
        moved_up, explanation = step_7_compare_rankings(baseline_results, cf_results)
        step_8_final_assertion(moved_up, explanation)
        
        print("\n✅ ALL TESTS PASSED - COLLABORATIVE FILTERING IS VERIFIED")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST ASSERTION FAILED:\n{e}")
        return 1
    except Exception as e:
        print(f"\n❌ TEST EXECUTION FAILED:\n{e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
