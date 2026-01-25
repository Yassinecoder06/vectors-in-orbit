"""Demo script showing Financial-Aware Collaborative Filtering behavior."""
import time
from typing import Dict, Any

from qdrant_client import models

from search_pipeline import get_qdrant_client, embed_query, PRODUCTS_COLLECTION, rerank_products, semantic_product_search
from interaction_logger import log_interaction

USERS = {
    "low_budget_user": {"user_id": "low_budget_user", "available_balance": 300.0, "credit_limit": 200.0},
    "mid_budget_user": {"user_id": "mid_budget_user", "available_balance": 1200.0, "credit_limit": 800.0},
    "high_budget_user": {"user_id": "high_budget_user", "available_balance": 4000.0, "credit_limit": 3000.0},
}

PRODUCTS = {
    "p1": {"name": "ThinkPad Coding", "price": 900.0},
    "p2": {"name": "Creator Laptop", "price": 2200.0},
    "p3": {"name": "Budget Laptop", "price": 500.0},
}


def upsert_products(client) -> Dict[str, str]:
    import uuid
    ids = {}
    for key, meta in PRODUCTS.items():
        product_id = str(uuid.uuid4())
        vec = embed_query(meta["name"])
        client.upsert(
            collection_name=PRODUCTS_COLLECTION,
            points=[
                models.PointStruct(
                    id=product_id,
                    vector=vec,
                    payload={
                        "product_id": product_id,
                        "name": meta["name"],
                        "price": meta["price"],
                        "brand": key,
                        "categories": ["laptop"],
                        "in_stock": True,
                    },
                )
            ],
            wait=True,
        )
        ids[key] = product_id
    return ids


def simulate_interactions(client, ids):
    # High budget loves p2
    log_interaction(USERS["high_budget_user"], ids["p2"], "purchase", PRODUCTS["p2"]["price"], USERS["high_budget_user"])
    # Mid budget likes p1
    log_interaction(USERS["mid_budget_user"], ids["p1"], "add_to_cart", PRODUCTS["p1"]["price"], USERS["mid_budget_user"])
    # Low budget likes p3
    log_interaction(USERS["low_budget_user"], ids["p3"], "purchase", PRODUCTS["p3"]["price"], USERS["low_budget_user"])
    time.sleep(0.1)


def run_demo_for_user(client, user_key: str, ids):
    user = USERS[user_key]
    query_vec = embed_query("laptop for coding")
    semantic = semantic_product_search(client, query_vec, user["available_balance"] + user["credit_limit"])
    reranked = rerank_products(semantic, user, client, debug_mode=False, enable_slow_signals=True)

    print(f"\nUser: {user_key} (budget={user['available_balance'] + user['credit_limit']:.0f})")
    for item in reranked[:5]:
        payload = item.get("payload", {})
        print(
            f"- {payload.get('name')} | price={payload.get('price')} | "
            f"semantic={item.get('semantic_score', 0):.2f} "
            f"afford={item.get('affordability_score', 0):.2f} "
            f"cf={item.get('collaborative_score', 0):.2f} "
            f"final={item.get('final_score', 0):.2f} "
        )
        print(f"  Explanation: {item.get('reason')}")


def main():
    client = get_qdrant_client()
    ids = upsert_products(client)
    simulate_interactions(client, ids)

    run_demo_for_user(client, "low_budget_user", ids)
    run_demo_for_user(client, "mid_budget_user", ids)
    run_demo_for_user(client, "high_budget_user", ids)

    print("\nDemo complete: similar taste does not always mean similar recommendation when budgets differ.")


if __name__ == "__main__":
    main()
