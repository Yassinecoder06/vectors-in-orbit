"""Scoring constants for recommendation pipeline."""

FA_CF_WEIGHTS = {
    "semantic": 0.40,
    "affordability": 0.25,
    "preference": 0.15,
    "collaborative": 0.15,
    "popularity": 0.05,
}

# Interaction weights for FA-CF
INTERACTION_WEIGHTS = {
    "view": 0.2,
    "click": 0.5,
    "add_to_cart": 0.8,
    "purchase": 1.0,
}
