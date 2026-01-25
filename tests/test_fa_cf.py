"""End-to-end tests for Financial-Aware Collaborative Filtering (FA-CF).

Run with:
    python -m tests.test_fa_cf
"""
import time
from typing import Dict, Any, List

import numpy as np
from qdrant_client import models

from search_pipeline import (
    get_qdrant_client,
    embed_query,
    PRODUCTS_COLLECTION,
    INTERACTIONS_COLLECTION,
    rerank_products,
    semantic_product_search,
)
from cf.fa_cf import get_fa_cf_scores
from interaction_logger import log_interaction


USERS = {
    "low": {"user_id": "low_budget_user", "available_balance": 300.0, "credit_limit": 200.0},
    "mid": {"user_id": "mid_budget_user", "available_balance": 1200.0, "credit_limit": 800.0},
    "high": {"user_id": "high_budget_user", "available_balance": 4000.0, "credit_limit": 3000.0},
}

PRODUCTS = {
    "cheap": {"name": "Budget Laptop", "price": 500.0},
    "mid": {"name": "Mid Laptop", "price": 1200.0},
    "premium": {"name": "Premium Laptop", "price": 2500.0},
}


def _delete_all_interactions(client):
    client.delete(
        collection_name=INTERACTIONS_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="timestamp", range=models.Range(gte=0))])
        ),
    )


def _upsert_product(client, key: str, text: str, price: float) -> str:
    import uuid
    vec = embed_query(text)
    product_id = str(uuid.uuid4())
    client.upsert(
        collection_name=PRODUCTS_COLLECTION,
        points=[
            models.PointStruct(
                id=product_id,
                vector=vec,
                payload={
                    "product_id": product_id,
                    "name": text,
                    "price": price,
                    "brand": key,
                    "categories": ["laptop"],
                    "in_stock": True,
                },
            )
        ],
        wait=True,
    )
    return product_id


def _seed_products(client) -> Dict[str, str]:
    ids = {}
    ids["cheap"] = _upsert_product(client, "cheap", "budget laptop for students", PRODUCTS["cheap"]["price"])
    ids["mid"] = _upsert_product(client, "mid", "balanced laptop for coding", PRODUCTS["mid"]["price"])
    ids["premium"] = _upsert_product(client, "premium", "premium laptop for creators", PRODUCTS["premium"]["price"])
    return ids


def _log(user: Dict[str, Any], product_id: str, event: str, price: float):
    log_interaction(user["user_id"], product_id, event, price, user)
    time.sleep(0.05)


def test_budget_divergence(client, ids):
    """Expensive product must not be boosted for low-budget user."""
    _delete_all_interactions(client)
    _log(USERS["high"], ids["premium"], "purchase", PRODUCTS["premium"]["price"])
    _log(USERS["low"], ids["premium"], "click", PRODUCTS["premium"]["price"])

    candidates = list(ids.values())
    cf_scores = get_fa_cf_scores(client, USERS["low"]["user_id"], candidates, USERS["low"])
    assert cf_scores[ids["premium"]] == 0.0, "Premium item should not be boosted for low budget"


def test_financial_alignment_boost(client, ids):
    """Similar affordability should boost shared products."""
    _delete_all_interactions(client)
    _log(USERS["mid"], ids["mid"], "purchase", PRODUCTS["mid"]["price"])
    _log(USERS["high"], ids["mid"], "add_to_cart", PRODUCTS["mid"]["price"])

    candidates = list(ids.values())
    scores = get_fa_cf_scores(client, USERS["high"]["user_id"], candidates, USERS["high"])
    assert scores[ids["mid"]] > 0.5, "Mid product should be boosted for aligned budgets"


def test_real_time_interaction(client, ids):
    """CF score should rise after purchase."""
    _delete_all_interactions(client)
    # Before purchase
    _log(USERS["low"], ids["cheap"], "view", PRODUCTS["cheap"]["price"])
    scores_before = get_fa_cf_scores(client, USERS["low"]["user_id"], list(ids.values()), USERS["low"])
    # After purchase
    _log(USERS["low"], ids["cheap"], "purchase", PRODUCTS["cheap"]["price"])
    scores_after = get_fa_cf_scores(client, USERS["low"]["user_id"], list(ids.values()), USERS["low"])
    assert scores_after[ids["cheap"]] > scores_before[ids["cheap"]], "CF score must increase after purchase"


def test_comparison_with_without_cf(client, ids):
    """Ranking should differ when CF is disabled vs enabled."""
    _delete_all_interactions(client)
    _log(USERS["mid"], ids["mid"], "purchase", PRODUCTS["mid"]["price"])

    query_vec = embed_query("laptop for coding")
    semantic = semantic_product_search(client, query_vec, 5000)

    # Without CF
    no_cf = rerank_products(semantic, USERS["mid"], client, debug_mode=False, enable_slow_signals=False)
    # With CF
    with_cf = rerank_products(semantic, USERS["mid"], client, debug_mode=False, enable_slow_signals=True)

    def _rank(product_id: str, results: List[Dict[str, Any]]):
        for idx, item in enumerate(results, 1):
            if str(item.get("payload", {}).get("product_id", item.get("id"))) == product_id:
                return idx
        return None

    before_rank = _rank(ids["mid"], no_cf)
    after_rank = _rank(ids["mid"], with_cf)
    assert after_rank is not None and before_rank is not None, "Product must appear in both rankings"
    assert after_rank < before_rank, "CF should move product up when enabled"


def main():
    client = get_qdrant_client()
    ids = _seed_products(client)

    tests = [
        ("TEST 1: Budget divergence", test_budget_divergence),
        ("TEST 2: Financial alignment boost", test_financial_alignment_boost),
        ("TEST 3: Real-time interaction", test_real_time_interaction),
        ("TEST 4: Comparison with and without CF", test_comparison_with_without_cf),
    ]

    for label, fn in tests:
        print(f"\n=== {label} ===")
        fn(client, ids)
        print("OK")

    print("\n✅ ALL FA-CF TESTS PASSED")


if __name__ == "__main__":
    main()
