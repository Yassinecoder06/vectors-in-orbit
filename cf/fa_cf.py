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
    try:
        interactions, _ = client.scroll(
            collection_name=INTERACTIONS_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=limit,
            with_vectors=True,
            with_payload=True,
            timeout=30,  # 30 second timeout
        )
    except Exception as e:
        # Timeout or network error - return None to gracefully degrade
        return None

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

    try:
        results = client.query_points(
            collection_name=INTERACTIONS_COLLECTION,
            query=target_vector,
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=search_limit,
            with_payload=True,
            with_vectors=False,
            timeout=30,  # 30 second timeout
        )
    except Exception as e:
        # Timeout or network error - return zero scores to gracefully degrade
        return scores

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

    # =====================
    # SELF INTERACTION BOOST
    # =====================
    # Provide immediate reinforcement from the user's own recent interactions.
    # This ensures real-time effects (e.g., purchase) increase CF score even when
    # neighbor overlap is limited.
    try:
        self_interactions, _ = client.scroll(
            collection_name=INTERACTIONS_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=300,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        self_interactions = []

    self_scores: Dict[str, float] = {}
    if self_interactions:
        current_time = int(time.time())
        decay_constant = math.log(2) / 86400.0  # 1-day half-life for self boost
        candidate_set = {str(pid) for pid in candidate_ids}
        for p in self_interactions:
            payload = p.payload or {}
            pid = str(payload.get("product_id", ""))
            if pid not in candidate_set:
                continue
            # Respect budget limits: skip items priced above user's budget
            try:
                price = float(payload.get("product_price", payload.get("price", 0.0)) or 0.0)
            except (TypeError, ValueError):
                price = 0.0
            if user_budget and price > user_budget:
                continue
            try:
                ts = int(payload.get("timestamp", current_time))
            except (TypeError, ValueError):
                ts = current_time
            age_seconds = max(0, current_time - ts)
            time_decay = math.exp(-decay_constant * age_seconds)
            w = _safe_weight(payload)
            # Stronger boost for high-intent actions
            itype = str(payload.get("interaction_type", "view"))
            intent_multiplier = {
                "view": 0.5,
                "click": 0.8,
                "add_to_cart": 1.2,
                "purchase": 1.5,
            }.get(itype, 0.5)
            contribution = max(0.0, w * intent_multiplier * time_decay)
            self_scores[pid] = self_scores.get(pid, 0.0) + contribution

    # Normalize neighbor and self scores separately, then blend
    blended_scores: Dict[str, float] = {str(pid): 0.0 for pid in candidate_ids}
    if product_scores:
        max_neighbor = max(product_scores.values())
        if max_neighbor > 0:
            for pid, val in product_scores.items():
                blended_scores[pid] = val / max_neighbor
    if self_scores:
        for pid, val in self_scores.items():
            # Blend: 70% neighbors, 30% self boost (tunable)
            # Use capped absolute self signal to avoid normalization saturation
            self_signal = min(1.0, max(0.0, val))
            blended_scores[pid] = 0.7 * blended_scores.get(pid, 0.0) + 0.3 * self_signal

    # If both empty, keep zeros
    for pid in scores:
        scores[pid] = blended_scores.get(pid, scores[pid])

    return scores
