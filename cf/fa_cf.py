"""Financial-Aware Collaborative Filtering (FA-CF).

This module builds user interaction profiles with affordability context and
computes collaborative scores that respect financial alignment.
"""
from typing import Dict, List, Any, Optional
import math
import time

import numpy as np
from qdrant_client import QdrantClient, models

from scoring import INTERACTION_WEIGHTS

INTERACTIONS_COLLECTION = "interaction_memory"


def _safe_weight(payload: Dict[str, Any]) -> float:
    return float(payload.get("interaction_weight", payload.get("weight", 0.1)))


def _safe_affordability_ratio(payload: Dict[str, Any]) -> Optional[float]:
    ratio = payload.get("affordability_ratio")
    if ratio is not None:
        try:
            return float(ratio)
        except (TypeError, ValueError):
            return None
    try:
        price = float(payload.get("product_price", payload.get("price", 0.0)))
        available = float(payload.get("available_balance", 0.0))
        credit = float(payload.get("credit_limit", 0.0))
    except (TypeError, ValueError):
        return None
    denom = available + credit
    if denom <= 0:
        return None
    return price / denom


def compute_financial_alignment(ratio_a: Optional[float], ratio_b: Optional[float]) -> float:
    if ratio_a is None or ratio_b is None:
        return 0.0
    alignment = 1.0 - abs(ratio_a - ratio_b)
    return max(0.0, min(1.0, alignment))


def build_user_interaction_profile(client: QdrantClient, user_id: str, limit: int = 200) -> Optional[Dict[str, Any]]:
    """Builds a weighted interaction vector and affordability baseline for a user."""
    interactions, _ = client.scroll(
        collection_name=INTERACTIONS_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )

    if not interactions:
        return None

    vectors = []
    weights = []
    ratios = []
    current_time = int(time.time())
    decay_constant = math.log(2) / 604800.0  # 7-day half-life
    budget_samples = []

    for point in interactions:
        if not point.vector:
            continue
        payload = point.payload or {}
        timestamp = int(payload.get("timestamp", current_time))
        base_weight = _safe_weight(payload)
        age_seconds = max(0, current_time - timestamp)
        time_decay = math.exp(-decay_constant * age_seconds)
        final_weight = base_weight * time_decay
        vectors.append(point.vector)
        weights.append(final_weight)

        ratio = _safe_affordability_ratio(payload)
        if ratio is not None:
            ratios.append((ratio, final_weight))

        try:
            available = float(payload.get("available_balance", 0.0))
            credit = float(payload.get("credit_limit", 0.0))
            budget_samples.append(max(0.0, available + credit))
        except (TypeError, ValueError):
            pass

    if not vectors or not weights:
        return None

    weighted_vector = np.average(vectors, axis=0, weights=weights).tolist()
    if ratios:
        num = sum(r * w for r, w in ratios)
        den = sum(w for _, w in ratios)
        avg_ratio = num / den if den else None
    else:
        avg_ratio = None

    budget = max(budget_samples) if budget_samples else 0.0

    return {
        "user_id": user_id,
        "vector": weighted_vector,
        "avg_affordability_ratio": avg_ratio,
        "budget": budget,
    }


def get_fa_cf_scores(
    client: QdrantClient,
    user_id: str,
    candidate_ids: List[str],
    user_context: Dict[str, Any],
    search_limit: int = 120,
) -> Dict[str, float]:
    """Compute FA-CF scores for candidate products respecting financial alignment."""
    scores = {str(pid): 0.0 for pid in candidate_ids}

    profile = build_user_interaction_profile(client, user_id)
    if profile is None:
        return scores

    target_vector = profile["vector"]
    target_ratio = profile.get("avg_affordability_ratio")
    try:
        user_budget = float(user_context.get("available_balance", 0.0)) + float(
            user_context.get("credit_limit", 0.0)
        )
    except (TypeError, ValueError):
        user_budget = profile.get("budget", 0.0)

    if not target_vector:
        return scores

    results = client.query_points(
        collection_name=INTERACTIONS_COLLECTION,
        query=target_vector,
        query_filter=models.Filter(
            must_not=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=search_limit,
        with_payload=True,
        with_vectors=False,
    )

    user_hits: Dict[str, List[Dict[str, Any]]] = {}
    for hit in results.points:
        payload = hit.payload or {}
        other_user = str(payload.get("user_id", ""))
        if not other_user:
            continue
        weight = _safe_weight(payload)
        price = float(payload.get("product_price", payload.get("price", 0.0)) or 0.0)
        ratio = _safe_affordability_ratio(payload)
        user_hits.setdefault(other_user, []).append(
            {
                "score": float(hit.score),
                "weight": weight,
                "ratio": ratio,
                "price": price,
                "product_id": str(payload.get("product_id", "")),
            }
        )

    product_scores: Dict[str, float] = {}

    for other_user, hits in user_hits.items():
        num = sum(h["score"] * h["weight"] for h in hits)
        den = sum(h["weight"] for h in hits)
        if den <= 0:
            continue
        base_similarity = num / den
        ratios = [h["ratio"] for h in hits if h.get("ratio") is not None]
        if ratios:
            avg_ratio = sum(rat for rat in ratios) / len(ratios)
        else:
            avg_ratio = None
        alignment = compute_financial_alignment(target_ratio, avg_ratio)
        if alignment < 0.5:
            continue
        final_similarity = max(0.0, base_similarity * alignment)
        if final_similarity <= 0:
            continue

        for h in hits:
            pid = h.get("product_id")
            if not pid or pid not in scores:
                continue
            price = h.get("price", 0.0)
            if user_budget and price > user_budget:
                continue
            contribution = final_similarity * h["weight"] * h["score"]
            product_scores[pid] = product_scores.get(pid, 0.0) + contribution

    if not product_scores:
        return scores

    max_score = max(product_scores.values())
    if max_score <= 0:
        return scores

    for pid in scores:
        if pid in product_scores:
            scores[pid] = product_scores[pid] / max_score

    return scores
