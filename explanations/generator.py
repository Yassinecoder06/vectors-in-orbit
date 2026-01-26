"""Generate user-facing explanations for product recommendations."""
from typing import List


def build_explanations(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    cf_score: float,
    popularity_score: float,
    price: float,
    budget: float,
) -> List[str]:
    explanations: List[str] = []

    if affordability_score >= 0.8:
        explanations.append("Affordable within your current financial capacity")
    elif affordability_score >= 0.5:
        explanations.append("Fits your budget with some margin")
    elif affordability_score > 0.0:
        explanations.append("Near your budget limit; consider financing options")
    else:
        explanations.append("Filtered out unaffordable options")

    if cf_score >= 0.5:
        explanations.append("Popular among users with similar budgets")
    elif cf_score > 0.1:
        explanations.append("Similar users interacted with this product")

    if preference_score >= 0.5:
        explanations.append("Matches your stated preferences")

    if semantic_score >= 0.6:
        explanations.append("Strong match to your search intent")

    if popularity_score >= 0.4:
        explanations.append("Trending with recent shoppers")

    # Fallback to ensure at least one explanation
    if not explanations:
        explanations.append("Recommended based on relevance and affordability")

    return explanations
