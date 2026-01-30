"""
Optimize FA-CF scoring coefficients with semantic dominance.

Standalone script:
- Loads interaction logs (Qdrant or local JSON/JSONL)
- Falls back to synthetic data if no logs are found
- Uses hill-climbing to maximize reward from high-intent events
- Enforces semantic weight >= 0.35 at all times
- Saves learned coefficients to scoring/learned_coefficients_realistic.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WEIGHT_KEYS = ["semantic", "affordability", "preference", "collaborative", "popularity"]
SEMANTIC_MIN = 0.35
REWARD_MAP = {
    "view": 0.0,
    "click": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}

DEFAULT_MAX_ITERS = 250
DEFAULT_PATIENCE = 50
DEFAULT_DELTA = 0.05

# Popularity decay (half-life in hours)
POPULARITY_HALF_LIFE_HOURS = 6.0
SEMANTIC_REWARD_BOOST = 2.0


@dataclass
class InteractionRecord:
    user_id: str
    product_id: str
    interaction_type: str
    timestamp: int
    price: float
    available_balance: float
    credit_limit: float
    category: str
    brand: str
    product_name: str
    original_query: str

    @property
    def budget(self) -> float:
        return max(0.0, self.available_balance + self.credit_limit)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def _jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_interactions_from_qdrant(limit_per_page: int = 1000) -> List[InteractionRecord]:
    """Load interactions from Qdrant if credentials are available."""
    try:
        import sys
        import os
        # Add parent directory to path to import search_pipeline
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from qdrant_client import models
        from search_pipeline import get_qdrant_client
    except Exception as e:
        print(f"❌ Failed to import Qdrant dependencies: {e}")
        return []

    try:
        client = get_qdrant_client()
        print("✅ Connected to Qdrant successfully")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return []

    interactions: List[InteractionRecord] = []
    offset = None

    while True:
        try:
            points, offset = client.scroll(
                collection_name="interaction_memory",
                limit=limit_per_page,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"❌ Failed to scroll interaction_memory: {e}")
            break

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            interactions.append(
                InteractionRecord(
                    user_id=str(payload.get("user_id", "")),
                    product_id=str(payload.get("product_id", "")),
                    interaction_type=str(payload.get("interaction_type", "view")),
                    timestamp=_safe_int(payload.get("timestamp", int(time.time()))),
                    price=_safe_float(payload.get("product_price", payload.get("price", 0.0))),
                    available_balance=_safe_float(payload.get("available_balance", 0.0)),
                    credit_limit=_safe_float(payload.get("credit_limit", 0.0)),
                    category=str(payload.get("category", "Unknown")),
                    brand=str(payload.get("brand", "Unknown")),
                    product_name=str(payload.get("product_name", "Unknown Product")),
                    original_query=str(payload.get("original_query", "")),
                )
            )

        if offset is None:
            break

    print(f"✅ Loaded {len(interactions)} interactions from Qdrant")
    return interactions


def _try_parse_json_records(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            data = json.loads(text)
            return data if isinstance(data, list) else []
        # JSONL
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception:
        return []


def load_interactions_from_files(root_dir: str) -> List[InteractionRecord]:
    """Search for local JSON/JSONL interaction logs."""
    candidates: List[str] = []
    for base, _, files in os.walk(root_dir):
        for name in files:
            lower = name.lower()
            if ("interaction" in lower or "interactions" in lower) and lower.endswith((".json", ".jsonl")):
                candidates.append(os.path.join(base, name))

    interactions: List[InteractionRecord] = []
    for path in candidates:
        for rec in _try_parse_json_records(path):
            payload = rec.get("payload", rec)
            interactions.append(
                InteractionRecord(
                    user_id=str(payload.get("user_id", "")),
                    product_id=str(payload.get("product_id", "")),
                    interaction_type=str(payload.get("interaction_type", "view")),
                    timestamp=_safe_int(payload.get("timestamp", int(time.time()))),
                    price=_safe_float(payload.get("product_price", payload.get("price", 0.0))),
                    available_balance=_safe_float(payload.get("available_balance", 0.0)),
                    credit_limit=_safe_float(payload.get("credit_limit", 0.0)),
                    category=str(payload.get("category", "Unknown")),
                    brand=str(payload.get("brand", "Unknown")),
                    product_name=str(payload.get("product_name", "Unknown Product")),
                    original_query=str(payload.get("original_query", "")),
                )
            )

    return interactions


def generate_synthetic_interactions(seed: Optional[int] = None, n_users: int = 18, n_products: int = 50) -> List[InteractionRecord]:
    """Generate synthetic interactions where semantic similarity drives intent."""
    if seed is not None:
        random.seed(seed)

    categories = ["electronics", "fashion", "home", "beauty", "sports"]
    brands = ["Acme", "Orbit", "Nova", "Zenith", "Pulse"]

    products = []
    for i in range(n_products):
        products.append(
            {
                "product_id": f"p{i}",
                "name": f"{random.choice(brands)} {random.choice(categories)} item {i}",
                "category": random.choice(categories),
                "brand": random.choice(brands),
                "price": random.uniform(10, 500),
            }
        )

    users = []
    for i in range(n_users):
        users.append(
            {
                "user_id": f"u{i}",
                "budget": random.uniform(200, 2500),
                "pref_category": random.choice(categories),
                "pref_brand": random.choice(brands),
            }
        )

    true_weights = normalize_weights({
        "semantic": 0.55,
        "affordability": 0.15,
        "preference": 0.12,
        "collaborative": 0.10,
        "popularity": 0.08,
    })

    def _score(prod: Dict[str, Any], user: Dict[str, Any], query: str) -> Dict[str, float]:
        sem = _jaccard_similarity(query, prod["name"])
        affordability = max(0.0, min(1.0, 1.0 - prod["price"] / user["budget"]))
        pref = 1.0 if (prod["category"] == user["pref_category"] or prod["brand"] == user["pref_brand"]) else 0.0
        collab = random.uniform(0.0, 1.0) * 0.7
        pop = random.uniform(0.0, 1.0) * 0.6
        return {
            "semantic": sem,
            "affordability": affordability,
            "preference": pref,
            "collaborative": collab,
            "popularity": pop,
        }

    interactions: List[InteractionRecord] = []
    now = int(time.time())
    for _ in range(n_users * 35):
        user = random.choice(users)
        product = random.choice(products)
        query = f"{user['pref_category']} {user['pref_brand']}"
        scores = _score(product, user, query)
        final = sum(true_weights[k] * scores[k] for k in WEIGHT_KEYS)
        final = max(0.0, min(1.0, final))

        # Stronger semantic influence on intent
        final_sem = min(1.0, final + 0.5 * scores["semantic"])
        r = random.random()
        if final_sem > 0.8 and r < 0.7:
            itype = "purchase"
        elif final_sem > 0.6 and r < 0.7:
            itype = "add_to_cart"
        elif final_sem > 0.35 and r < 0.8:
            itype = "click"
        else:
            itype = "view"

        interactions.append(
            InteractionRecord(
                user_id=user["user_id"],
                product_id=product["product_id"],
                interaction_type=itype,
                timestamp=now - random.randint(0, 7 * 24 * 3600),
                price=float(product["price"]),
                available_balance=float(user["budget"] * 0.5),
                credit_limit=float(user["budget"] * 0.5),
                category=product["category"],
                brand=product["brand"],
                product_name=product["name"],
                original_query=query,
            )
        )

    return interactions


def load_interactions(workspace_root: str) -> Tuple[List[InteractionRecord], str]:
    """Load interactions from Qdrant only (real data)."""
    interactions = load_interactions_from_qdrant()
    if interactions:
        return interactions, "qdrant"

    raise RuntimeError(
        "No Qdrant interactions found. Ensure QDRANT_URL/QDRANT_API_KEY are set "
        "and interaction_memory has data."
    )


# -----------------------------------------------------------------------------
# Feature engineering for scoring components
# -----------------------------------------------------------------------------

def build_user_preferences(interactions: Iterable[InteractionRecord]) -> Dict[str, Dict[str, List[str]]]:
    """Infer preferred categories/brands per user from interaction history."""
    user_cat: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    user_brand: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for i in interactions:
        reward = REWARD_MAP.get(i.interaction_type, 0.0)
        user_cat[i.user_id][i.category] += reward + 0.1
        user_brand[i.user_id][i.brand] += reward + 0.1

    prefs: Dict[str, Dict[str, List[str]]] = {}
    for user_id in user_cat:
        cat_sorted = sorted(user_cat[user_id].items(), key=lambda x: x[1], reverse=True)
        brand_sorted = sorted(user_brand[user_id].items(), key=lambda x: x[1], reverse=True)
        prefs[user_id] = {
            "categories": [c for c, _ in cat_sorted[:2]],
            "brands": [b for b, _ in brand_sorted[:2]],
        }
    return prefs


def compute_popularity(interactions: Iterable[InteractionRecord]) -> Dict[str, float]:
    """Compute time-decayed popularity scores per product."""
    now = int(time.time())
    decay_constant = math.log(2) / (POPULARITY_HALF_LIFE_HOURS * 3600)
    scores: Dict[str, float] = defaultdict(float)

    for i in interactions:
        reward = REWARD_MAP.get(i.interaction_type, 0.0)
        age = max(0, now - i.timestamp)
        decay = math.exp(-decay_constant * age)
        scores[i.product_id] += reward * decay

    max_score = max(scores.values()) if scores else 0.0
    if max_score <= 0:
        return {pid: 0.0 for pid in scores}

    return {pid: val / max_score for pid, val in scores.items()}


def compute_collaborative_signal(interactions: Iterable[InteractionRecord]) -> Dict[Tuple[str, str], float]:
    """Simple collaborative score: product preference from similar-budget users."""
    user_ratios: Dict[str, float] = {}
    user_counts: Dict[str, int] = defaultdict(int)

    for i in interactions:
        if i.budget <= 0:
            continue
        ratio = i.price / i.budget
        user_ratios[i.user_id] = user_ratios.get(i.user_id, 0.0) + ratio
        user_counts[i.user_id] += 1

    for user_id, total in list(user_ratios.items()):
        count = user_counts.get(user_id, 1)
        user_ratios[user_id] = total / max(1, count)

    buckets: Dict[int, List[str]] = defaultdict(list)
    for user_id, ratio in user_ratios.items():
        bucket = min(9, max(0, int(ratio * 10)))
        buckets[bucket].append(user_id)

    bucket_product_scores: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for i in interactions:
        reward = REWARD_MAP.get(i.interaction_type, 0.0)
        ratio = user_ratios.get(i.user_id, None)
        if ratio is None:
            continue
        bucket = min(9, max(0, int(ratio * 10)))
        bucket_product_scores[bucket][i.product_id] += reward

    user_product_scores: Dict[Tuple[str, str], float] = {}
    for bucket, scores in bucket_product_scores.items():
        max_score = max(scores.values()) if scores else 0.0
        if max_score <= 0:
            continue
        for user_id in buckets.get(bucket, []):
            for pid, val in scores.items():
                user_product_scores[(user_id, pid)] = val / max_score

    return user_product_scores


def build_feature_rows(interactions: List[InteractionRecord]) -> List[Dict[str, Any]]:
    prefs = build_user_preferences(interactions)
    popularity = compute_popularity(interactions)
    collab = compute_collaborative_signal(interactions)

    rows: List[Dict[str, Any]] = []
    for i in interactions:
        query = i.original_query or ""
        semantic_score = _jaccard_similarity(query, f"{i.product_name} {i.category} {i.brand}")

        if i.budget <= 0:
            affordability_score = 0.0
        else:
            affordability_score = max(0.0, min(1.0, 1.0 - (i.price / i.budget)))

        pref = prefs.get(i.user_id, {"categories": [], "brands": []})
        preference_score = 1.0 if (i.category in pref["categories"] or i.brand in pref["brands"]) else 0.0

        collaborative_score = collab.get((i.user_id, i.product_id), 0.0)
        popularity_score = popularity.get(i.product_id, 0.0)

        rows.append(
            {
                "interaction": i,
                "semantic": semantic_score,
                "affordability": affordability_score,
                "preference": preference_score,
                "collaborative": collaborative_score,
                "popularity": popularity_score,
            }
        )

    return rows


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    weights = {k: max(0.0, v) for k, v in weights.items()}
    # Enforce semantic floor
    if weights.get("semantic", 0.0) < SEMANTIC_MIN:
        weights["semantic"] = SEMANTIC_MIN
    total = sum(weights.values())
    if total <= 0:
        base = 1.0 / len(weights)
        weights = {k: base for k in weights}
        weights["semantic"] = max(SEMANTIC_MIN, weights["semantic"])
        total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def semantic_dominant(weights: Dict[str, float]) -> bool:
    sem = weights.get("semantic", 0.0)
    return all(sem >= weights[k] for k in weights if k != "semantic")


def evaluate_reward(rows: List[Dict[str, Any]], weights: Dict[str, float]) -> float:
    total = 0.0
    for r in rows:
        i: InteractionRecord = r["interaction"]
        reward = REWARD_MAP.get(i.interaction_type, 0.0)
        affordability_score = r["affordability"]
        if affordability_score <= 0:
            final_score = 0.0
        else:
            # Boost semantic contribution in reward to reflect relevance
            weighted = {
                k: (weights[k] * SEMANTIC_REWARD_BOOST if k == "semantic" else weights[k])
                for k in WEIGHT_KEYS
            }
            norm = sum(weighted.values()) or 1.0
            weighted = {k: v / norm for k, v in weighted.items()}
            final_score = sum(weighted[k] * r[k] for k in WEIGHT_KEYS)
            final_score = max(0.0, min(1.0, final_score))
        total += reward * final_score
    return total


def hill_climb_optimize(
    rows: List[Dict[str, Any]],
    max_iters: int,
    patience: int,
    delta: float,
) -> Tuple[Dict[str, float], float, Dict[str, float], float, List[Tuple[int, float, Dict[str, float]]]]:
    weights = normalize_weights({k: random.random() for k in WEIGHT_KEYS})
    if not semantic_dominant(weights):
        weights["semantic"] = max(weights.values())
        weights = normalize_weights(weights)

    initial_weights_snapshot = dict(weights)
    initial_reward = evaluate_reward(rows, weights)

    history: List[Tuple[int, float, Dict[str, float]]] = [(0, initial_reward, dict(weights))]

    best_reward = initial_reward
    best_weights = dict(weights)
    no_improve = 0

    print("Initial coefficients:")
    for k in WEIGHT_KEYS:
        print(f"  {k:>14s}: {weights[k]:.4f}")
    print(f"Initial reward: {initial_reward:.4f}\n")

    for step in range(1, max_iters + 1):
        key = random.choice(WEIGHT_KEYS)
        direction = random.choice([-1.0, 1.0])
        proposal = dict(weights)
        proposal[key] = max(0.0, proposal[key] * (1.0 + direction * delta))
        proposal = normalize_weights(proposal)

        # Ensure semantic dominance
        if not semantic_dominant(proposal):
            # Pull semantic up slightly and renormalize
            proposal["semantic"] = max(proposal.values())
            proposal = normalize_weights(proposal)

        reward = evaluate_reward(rows, proposal)
        improved = reward > best_reward + 1e-9

        status = "IMPROVED" if improved else "no change"
        print(
            f"Iter {step:03d} | reward={reward:.4f} | best={best_reward:.4f} | "
            f"probe={key} {direction:+.0f}x | {status}"
        )

        if improved:
            best_reward = reward
            best_weights = dict(proposal)
            weights = dict(proposal)
            no_improve = 0
            print("  ✅ Improvement found. Updated coefficients:")
            for k in WEIGHT_KEYS:
                print(f"     {k:>14s}: {weights[k]:.4f}")
        else:
            no_improve += 1

        history.append((step, reward, dict(weights)))

        if no_improve >= patience:
            print(f"\nStopping: no improvement after {patience} steps.")
            break

    return best_weights, best_reward, initial_weights_snapshot, initial_reward, history


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------

def build_explanations(
    rows: List[Dict[str, Any]],
    initial_weights: Dict[str, float],
    final_weights: Dict[str, float],
) -> Dict[str, str]:
    high_rewards = {"add_to_cart", "purchase"}

    explanations: Dict[str, str] = {}
    for k in WEIGHT_KEYS:
        high_vals = []
        low_vals = []
        for r in rows:
            i: InteractionRecord = r["interaction"]
            if i.interaction_type in high_rewards:
                high_vals.append(r[k])
            else:
                low_vals.append(r[k])

        high_avg = sum(high_vals) / max(1, len(high_vals))
        low_avg = sum(low_vals) / max(1, len(low_vals))
        delta = high_avg - low_avg

        direction = "increased" if final_weights[k] >= initial_weights[k] else "decreased"
        if delta > 0.05:
            reason = "higher scores align with add-to-cart/purchase events"
        elif delta < -0.05:
            reason = "higher scores align more with low-intent events"
        else:
            reason = "signal shows weak separation between high- and low-intent events"

        explanations[k] = (
            f"{direction} because {reason} (avg_high={high_avg:.2f}, avg_low={low_avg:.2f})."
        )

    return explanations


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize scoring coefficients (semantic-dominant).")
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    interactions, source = load_interactions(workspace_root)

    print(f"Loaded {len(interactions)} interactions from: {source}\n")

    rows = build_feature_rows(interactions)

    final_weights, final_reward, initial_weights, initial_reward, _ = hill_climb_optimize(
        rows,
        max_iters=args.max_iters,
        patience=args.patience,
        delta=args.delta,
    )

    print("\nFinal optimized coefficients:")
    for k in WEIGHT_KEYS:
        print(f"  {k:>14s}: {final_weights[k]:.4f}")

    print(f"\nInitial reward (random init): {initial_reward:.4f}")
    print(f"Final reward: {final_reward:.4f}")
    print(f"Improvement: {(final_reward - initial_reward):.4f}\n")

    explanations = build_explanations(rows, initial_weights, final_weights)
    print("Explanation of coefficient changes:")
    for k in WEIGHT_KEYS:
        print(f"  {k:>14s}: {explanations[k]}")

    output_path = os.path.join(os.path.dirname(__file__), "learned_coefficients_realistic.json")
    payload = {
        "weights": final_weights,
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "improvement": final_reward - initial_reward,
        "source": source,
        "timestamp": int(time.time()),
        "semantic_min": SEMANTIC_MIN,
        "semantic_reward_boost": SEMANTIC_REWARD_BOOST,
        "explanations": explanations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved learned coefficients to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
