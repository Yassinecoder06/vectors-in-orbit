"""
Context-Aware FinCommerce Engine - Streamlit Demo UI

A hackathon demo showcasing personalized product recommendations
using semantic search, affordability scoring, and preference matching.

Architecture:
- This module handles UI ONLY (no search logic, no logging logic)
- Delegates to search_pipeline.py for search + ranking
- Delegates to interaction_logger.py for interaction tracking + analytics
"""

import streamlit as st
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import Swipe Component
from streamlit_swipecards import streamlit_swipecards

# Import backend search function
from search_pipeline import search_products, get_qdrant_client, PRODUCTS_COLLECTION
from interaction_logger import log_interaction, get_interaction_stats_by_type

# Import visualization functions

# =============================================================================
# Interaction Hooks
# =============================================================================

def _build_product_payload(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete product payload for interaction logging.
    
    The product dict from search results has the ID at top level, 
    but payload fields are nested. This merges them correctly.
    """
    payload = product.get("payload", {}).copy()
    # Ensure product ID is in payload (it's at top level in search results)
    if "id" not in payload and "product_id" not in payload:
        product_id = product.get("id")
        if product_id:
            payload["id"] = product_id
    return payload


def on_product_view(product: Dict[str, Any], query: str = ""):
    """Hook for product view interaction."""
    try:
        user_context = build_user_context()
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload(product),
            interaction_type="view",
            user_context=user_context,
            query=query
        )
    except Exception as e:
        pass  # Silent fail - don't show warnings for views


def on_product_click(product: Dict[str, Any], query: str = ""):
    """Hook for product click interaction."""
    try:
        user_context = build_user_context()
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload(product),
            interaction_type="click",
            user_context=user_context,
            query=query
        )
    except Exception as e:
        pass  # Silent fail


def on_add_to_cart(product: Dict[str, Any], query: str = ""):
    """Hook for add to cart interaction."""
    try:
        user_context = build_user_context()
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload(product),
            interaction_type="add_to_cart",
            user_context=user_context,
            query=query
        )
        st.toast("✅ Added to cart!", icon="🛒")
    except Exception as e:
        st.warning(f"Failed to log add to cart: {e}")


def on_purchase(product: Dict[str, Any], query: str = ""):
    """Hook for purchase interaction."""
    try:
        user_context = build_user_context()
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload(product),
            interaction_type="purchase",
            user_context=user_context,
            query=query
        )
        st.toast("✅ Purchase logged! Thank you!", icon="💳")
    except Exception as e:
        st.warning(f"Failed to log purchase: {e}")


def _build_product_payload_full(product: Dict[str, Any]) -> Dict[str, Any]:
    """Build a complete product payload for interaction logging (handles both formats)."""
    payload = product.get("payload", {}).copy()
    if "id" not in payload and "product_id" not in payload:
        product_id = product.get("id")
        if product_id:
            payload["id"] = product_id
    return payload


def load_discovery_queue():
    """Fetch a batch of products from the backend based on current filters."""
    user_context = build_user_context()
    query = st.session_state.search_query
    
    if not query:
        query = "trending"

    try:
        results = search_products(
            user_id=user_context["user_id"],
            query=query,
            top_k=30,  # Fetch 30 for the swipe stack
            debug_mode=False,
            override_context=user_context,
        )
    except Exception as e:
        st.error(f"Search error: {e}")
        results = []
    
    # Filter out duplicates (already in cart)
    cart_ids = {p.get("id") or p.get("payload", {}).get("product_id") for p in st.session_state.cart}
    
    new_items = []
    for item in results:
        it_id = item.get("id") or item.get("payload", {}).get("product_id")
        if it_id not in cart_ids:
            new_items.append(item)
            
    st.session_state.discovery_queue = new_items
    st.session_state.current_index = 0
    st.session_state.last_queue_query = query


def process_swipe_result(result: str, product: Dict[str, Any]):
    """Process the swipe result (right = like, left = pass)."""
    user_context = build_user_context()

    # Debug: log the raw result to understand what the component emits
    st.write(f"DEBUG: swipe result={result}, type={type(result)}")

    action = result
    if isinstance(result, dict):
        action = (
            result.get("action")
            or result.get("direction")
            or result.get("swipe")
            or result.get("decision")
        )
        if action is None and isinstance(result.get("like"), bool):
            action = "right" if result.get("like") else "left"

    # Normalize possible action values
    if isinstance(action, str):
        action = action.lower()
        if action in {"like", "swiperight", "right", "r"}:
            action = "right"
        elif action in {"dislike", "swipeleft", "left", "l"}:
            action = "left"
    
    if action == "right":
        # Add to cart
        st.session_state.cart.append(product)
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload_full(product),
            interaction_type="add_to_cart",
            user_context=user_context,
            query=st.session_state.search_query
        )
        st.toast("✅ Added to cart!", icon="🛒")
        
    elif action == "left":
        # Log as view/pass
        log_interaction(
            user_id=user_context["user_id"],
            product_payload=_build_product_payload_full(product),
            interaction_type="view",
            user_context=user_context,
            query=st.session_state.search_query
        )
    
    st.session_state.interaction_count += 1

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="FinCommerce Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .swipe-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# User personas for quick selection
USER_PERSONAS = {
    "Student": {"balance": 500, "credit": 1000, "risk": "Low"},
    "Professional": {"balance": 5000, "credit": 15000, "risk": "Medium"},
    "Executive": {"balance": 20000, "credit": 50000, "risk": "High"},
    "Custom": {"balance": 2500, "credit": 5000, "risk": "Medium"},
}


@st.cache_data(show_spinner=False)
def load_brand_category_options() -> Tuple[List[str], List[str]]:
    """
    Extract distinct brands and categories from the product dataset.
    Filters out numeric or placeholder values.
    """
    # First try to load from Qdrant Cloud
    qdrant_options = _load_brand_category_options_from_qdrant()
    if qdrant_options[0] or qdrant_options[1]:
        return qdrant_options

    data_path = Path(__file__).parent / "data" / "products_payload.json"
    
    try:
        with data_path.open("r", encoding="utf-8") as f:
            products = json.load(f)
    except FileNotFoundError:
        # Fallback if file doesn't exist
        return (
            ["Apple", "Samsung", "Sony", "HP", "Dell", "Nike", "Adidas"],
            ["Electronics", "Clothing", "Footwear", "Home & Kitchen", "Sports"],
        )
    
    def has_digit(value: str) -> bool:
        """Check if string contains digits (likely placeholder/ID)."""
        return any(ch.isdigit() for ch in value)
    
    # Extract unique brands and categories, filtering out invalid entries
    brands = sorted({
        b for b in ((p.get("brand") or "").strip() for p in products)
        if b and not has_digit(b) and len(b) > 1
    })
    
    categories = set()
    for p in products:
        cats = p.get("categories", [])
        if not isinstance(cats, list):
            cats = [cats] if cats else []
        for c in cats:
            c = str(c).strip()
            if c and not has_digit(c) and len(c) > 1:
                categories.add(c)

    categories = sorted(categories)
    
    return brands, categories


@st.cache_data(show_spinner=False)
def _load_brand_category_options_from_qdrant() -> Tuple[List[str], List[str]]:
    """
    Pull distinct brands/categories from Qdrant products collection.
    Falls back to empty lists if Qdrant is unavailable.
    """
    def has_digit(value: str) -> bool:
        return any(ch.isdigit() for ch in value)

    try:
        client = get_qdrant_client()
    except Exception:
        return [], []

    brands: set[str] = set()
    categories: set[str] = set()

    try:
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=PRODUCTS_COLLECTION,
                limit=4000,
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )

            if not points:
                break

            for p in points:
                payload = p.payload or {}
                brand = (payload.get("brand") or "").strip()
                cats = payload.get("categories", [])
                if not isinstance(cats, list):
                    cats = [cats] if cats else []

                if brand and not has_digit(brand) and len(brand) > 1:
                    brands.add(brand)
                for category in cats:
                    category = str(category).strip()
                    if category and not has_digit(category) and len(category) > 1:
                        categories.add(category)

            if not next_page:
                break

    except Exception:
        return [], []

    return sorted(brands), sorted(categories)


def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        "user_persona": "Professional",
        "available_balance": 5000.0,
        "credit_limit": 15000.0,
        "risk_tolerance": "Medium",  # Low, Medium, or High
        "preferred_brands": [],
        "preferred_categories": [],
        "search_query": "",
        "search_results": [],
        "has_searched": False,
        # Swipe specific
        "discovery_queue": [],
        "cart": [],
        "interaction_count": 0,
        "view_mode": "Swipe",  # 'Swipe' or 'Cart'
        "current_index": 0,
        "last_queue_query": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_user_context() -> Dict[str, Any]:
    """Build user context dictionary from session state for backend."""
    return {
        "user_id": f"demo_{st.session_state.user_persona.lower()}",
        "name": f"Demo {st.session_state.user_persona}",
        "available_balance": st.session_state.available_balance,
        "credit_limit": st.session_state.credit_limit,
        "preferred_brands": st.session_state.preferred_brands,
        "preferred_categories": st.session_state.preferred_categories,
        "risk_tolerance": st.session_state.risk_tolerance,  # Use custom value from sidebar
    }


def render_sidebar():
    """Render sidebar controls for user context configuration."""
    st.sidebar.title("🎛️ User Profile")
    st.sidebar.markdown("---")
    
    # Navigation Switch
    st.sidebar.subheader("Navigation")
    mode = st.sidebar.radio(
        "View Mode",
        ["Swipe & Shop", "My Cart"],
        index=0 if st.session_state.view_mode == "Swipe" else 1,
        format_func=lambda x: f"🛒 Cart ({len(st.session_state.cart)})" if x == "My Cart" else "🔍 Swipe & Shop"
    )
    if mode == "Swipe & Shop":
        st.session_state.view_mode = "Swipe"
    else:
        st.session_state.view_mode = "Cart"
    
    st.sidebar.markdown("---")

    # Search input
    st.sidebar.subheader("🔍 Search Products")
    def _trigger_sidebar_search():
        query = st.session_state.get("sidebar_search_input", "").strip()
        st.session_state.search_query = query
        st.session_state.discovery_queue = []
        st.session_state.current_index = 0
        with st.spinner("🔍 Searching..."):
            load_discovery_queue()

    st.sidebar.text_input(
        "What are you looking for?",
        value=st.session_state.search_query,
        placeholder="e.g., running shoes",
        key="sidebar_search_input",
        label_visibility="collapsed"
    )

    if st.sidebar.button("Search", key="sidebar_search_button", type="primary", use_container_width=True):
        _trigger_sidebar_search()

    st.sidebar.markdown("---")

    # Load options from dataset
    brand_options, category_options = load_brand_category_options()
    
    # User persona selector
    persona = st.sidebar.selectbox(
        "👤 User Type",
        options=list(USER_PERSONAS.keys()),
        index=list(USER_PERSONAS.keys()).index(st.session_state.user_persona),
        help="Select a preset user profile or customize below",
    )
    
    # Update defaults when persona changes
    if persona != st.session_state.user_persona:
        st.session_state.user_persona = persona
        if persona != "Custom":
            preset = USER_PERSONAS[persona]
            st.session_state.available_balance = float(preset["balance"])
            st.session_state.credit_limit = float(preset["credit"])
            st.session_state.discovery_queue = [] # Reset queue on persona change
    
    st.sidebar.markdown("### 💰 Financial Context")
    
    # Available balance - direct number input
    st.session_state.available_balance = st.sidebar.number_input(
        "💵 Available Balance ($)",
        min_value=0,
        max_value=1000000,
        value=int(st.session_state.available_balance),
        step=100,
        help="How much money do you have right now?"
    )
    
    # Credit limit - direct number input
    st.session_state.credit_limit = st.sidebar.number_input(
        "💳 Credit Limit ($)",
        min_value=0,
        max_value=1000000,
        value=int(st.session_state.credit_limit),
        step=100,
        help="What's your total credit allowance?"
    )
    
    # Total budget display
    total_budget = st.session_state.available_balance + st.session_state.credit_limit
    st.sidebar.metric("Total Budget", f"${total_budget:,.0f}")
    
    st.sidebar.markdown("### 🎯 Preferences")
    
    # Preferred brands multiselect
    st.session_state.preferred_brands = st.sidebar.multiselect(
        "Preferred Brands",
        options=brand_options,
        default=st.session_state.preferred_brands,
        help="Products from these brands will be prioritized",
    )
    
    # Preferred categories multiselect
    st.session_state.preferred_categories = st.sidebar.multiselect(
        "Preferred Categories",
        options=category_options,
        default=st.session_state.preferred_categories,
        help="Products in these categories will be prioritized",
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔬 Powered by Qdrant + SentenceTransformers")
    
    # Render trending section below main sidebar
    render_trending_section()



@st.cache_data(ttl=30, show_spinner=False)
def _fetch_trending_stats(timeframe_hours: int, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cached fetch of trending stats to reduce Qdrant calls.
    TTL of 30 seconds balances freshness with performance.
    """
    return get_interaction_stats_by_type(timeframe_hours=timeframe_hours, top_k=top_k)


def render_trending_section():
    """
    Render trending product analytics in sidebar.
    
    Shows breakdown of interactions by type (most viewed, carted, purchased)
    to provide social proof and engagement signals to users.
    
    Failure-safe: catches all exceptions to prevent sidebar crashes.
    Uses 30-second cache to reduce latency.
    """
    st.sidebar.markdown("---")
    
    # Header with refresh button
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.markdown("### 📊 Trending (24h)")
    with col2:
        if st.button("🔄", key="refresh_trending", help="Refresh trending data"):
            _fetch_trending_stats.clear()
            st.rerun()
    
    try:
        stats = _fetch_trending_stats(timeframe_hours=24, top_k=5)
        
        # Check if we have any stats to show
        has_data = any(len(v) > 0 for v in stats.values())
        
        if not has_data:
            st.sidebar.caption("_No recent activity_")
            return
        
        # Most Viewed
        if stats.get("viewed"):
            with st.sidebar.expander("👁️ Most Viewed", expanded=False):
                for item in stats["viewed"][:3]:
                    name = str(item.get("product_name", "Unknown"))[:25]
                    count = item.get("count", 0)
                    st.caption(f"• {name} ({count})")
        
        # Most Added to Cart
        if stats.get("carted"):
            with st.sidebar.expander("🛒 Most Added to Cart", expanded=False):
                for item in stats["carted"][:3]:
                    name = str(item.get("product_name", "Unknown"))[:25]
                    count = item.get("count", 0)
                    st.caption(f"• {name} ({count})")
        
        # Most Purchased
        if stats.get("purchased"):
            with st.sidebar.expander("💳 Most Purchased", expanded=True):
                for item in stats["purchased"][:3]:
                    name = str(item.get("product_name", "Unknown"))[:25]
                    count = item.get("count", 0)
                    st.caption(f"• {name} ({count})")
                    
    except Exception as e:
        # Never crash the sidebar
        st.sidebar.caption("_Unable to load trending data_")


def render_product_card(product: Dict[str, Any], rank: int):
    """Render a single product recommendation card with interaction logging."""
    payload = product.get("payload", {})
    
    # Extract product details with defaults
    name = payload.get("name", "Unknown Product")
    description = payload.get("description", "No description available")
    price = payload.get("price", 0.0)
    brand = payload.get("brand", "Unknown")
    categories = payload.get("categories", ["General"])
    if not isinstance(categories, list):
        categories = [categories] if categories else ["General"]
    category = categories[0] if categories else "General"
    monthly_installment = payload.get("monthly_installment", price / 12)
    in_stock = payload.get("in_stock", True)
    image_url = (payload.get("image_url") or "").strip()
    
    # Extract scores
    final_score = product.get("final_score", 0.0)
    semantic_score = product.get("semantic_score", 0.0)
    affordability_score = product.get("affordability_score", 0.0)
    preference_score = product.get("preference_score", 0.0)
    collaborative_score = product.get("collaborative_score", 0.0)
    popularity_score = product.get("popularity_score", 0.0)
    
    # Get explanations
    explanations = product.get("explanations", [])
    
    # Log view interaction automatically
    on_product_view(product, st.session_state.get("search_query", ""))
    
    # Determine score color
    if final_score >= 0.7:
        score_color = "🟢"
    elif final_score >= 0.5:
        score_color = "🟡"
    else:
        score_color = "🔴"
    
    # Card container
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if image_url:
                st.image(image_url, width=220)
            st.markdown(f"### #{rank} {name}")
            st.caption(f"**{brand}** · {category}")
        
        with col2:
            st.metric("Price", f"${price:,.2f}")
            st.caption(f"${monthly_installment:,.2f}/mo")
        
        with col3:
            st.metric("Match Score", f"{final_score:.0%}")
            st.caption(f"{score_color} {'In Stock' if in_stock else 'Out of Stock'}")
        
        # Description
        if description:
            st.markdown("**Description**")
            st.write(f"{description[:100]}{'...' if len(description) > 100 else ''}")
        
        # Display explanations prominently
        if explanations:
            st.markdown("**Why this product?**")
            for explanation in explanations[:3]:  # Show top 3 explanations
                st.caption(f"• {explanation}")
        
        # Session state key for this product's expanded state
        expand_key = f"expand_{rank}_{name[:10]}"
        if expand_key not in st.session_state:
            st.session_state[expand_key] = False
        
        # Expandable explanation section - controlled by session state or expander
        with st.expander("🔍 View Details & Breakdown", expanded=st.session_state[expand_key]):
            if description:
                st.markdown("**Full Description**")
                st.write(description)
                st.markdown("---")
            
            # Show all explanations
            if explanations:
                st.markdown("**All Reasons:**")
                for i, exp in enumerate(explanations, 1):
                    st.write(f"{i}. {exp}")
                st.markdown("---")
            
            render_explanation(
                semantic_score,
                affordability_score,
                preference_score,
                collaborative_score,
                popularity_score,
                payload,
            )
        
        # Interaction buttons - only Add to Cart and Buy Now
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🛒 Add to Cart", key=f"cart_{rank}_{name[:10]}", use_container_width=True):
                on_add_to_cart(product, st.session_state.get("search_query", ""))
        
        with col_btn2:
            if st.button("💳 Buy Now", key=f"buy_{rank}_{name[:10]}", use_container_width=True):
                on_purchase(product, st.session_state.get("search_query", ""))
        
        st.markdown("---")


def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    collaborative_score: float,
    popularity_score: float,
    payload: Dict[str, Any],
):
    """Render detailed explanation for product recommendation."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**🎯 Semantic Match**")
        st.progress(min(semantic_score, 1.0))
        st.caption(f"{semantic_score:.0%} query relevance")
        if semantic_score >= 0.7:
            st.success("Strong match to your search!")
        elif semantic_score >= 0.4:
            st.info("Moderate relevance")
        else:
            st.warning("Weak match")
    
    with col2:
        st.markdown("**💰 Affordability**")
        st.progress(min(affordability_score, 1.0))
        st.caption(f"{affordability_score:.0%} budget fit")
        
        price = payload.get("price", 0)
        total_budget = st.session_state.available_balance + st.session_state.credit_limit
        
        if affordability_score >= 0.8:
            st.success(f"Well within budget (${price:,.0f} / ${total_budget:,.0f})")
        elif affordability_score >= 0.5:
            st.info(f"Affordable (${price:,.0f} / ${total_budget:,.0f})")
        else:
            st.warning(f"Stretches budget (${price:,.0f} / ${total_budget:,.0f})")
    
    with col3:
        st.markdown("**❤️ Preference Match**")
        st.progress(min(preference_score, 1.0))
        st.caption(f"{preference_score:.0%} preference alignment")
        
        brand = payload.get("brand", "")
        categories = payload.get("categories", [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []
        
        brand_match = brand.lower() in [b.lower() for b in st.session_state.preferred_brands]
        preferred_categories = [c.lower() for c in st.session_state.preferred_categories]
        category_match = any(str(c).lower() in preferred_categories for c in categories)
        
        if brand_match and category_match:
            # Find matched category with bidirectional substring matching
            matched = None
            for cat in categories:
                cat_lower = str(cat).lower()
                for pref in preferred_categories:
                    pref_lower = pref.lower()
                    if pref_lower in cat_lower or cat_lower in pref_lower:
                        matched = cat
                        break
                if matched:
                    break
            if matched:
                st.success(f"Matches brand ({brand}) & category ({matched})")
            else:
                st.success(f"Matches brand ({brand}) & category")
        elif brand_match:
            st.success(f"Matches preferred brand: {brand}")
        elif category_match:
            # Find matched category with bidirectional substring matching
            matched = None
            for cat in categories:
                cat_lower = str(cat).lower()
                for pref in preferred_categories:
                    pref_lower = pref.lower()
                    if pref_lower in cat_lower or cat_lower in pref_lower:
                        matched = cat
                        break
                if matched:
                    break
            if matched:
                st.success(f"Matches preferred category: {matched}")
            else:
                st.success("Matches preferred category")
        elif not st.session_state.preferred_brands and not st.session_state.preferred_categories:
            st.info("No preferences set - all products considered equally")
        else:
            # Debug: show product categories when no match
            if categories:
                cat_str = ", ".join([str(c) for c in categories[:3]])
                st.warning(f"No match (product cats: {cat_str})")
            else:
                st.warning("No preference match")
    
    with col4:
        st.markdown("**🤝 Collaborative**")
        st.progress(min(collaborative_score, 1.0))
        st.caption(f"{collaborative_score:.0%}")
        if collaborative_score > 0.5:
            st.success("Similar users liked this")
        elif collaborative_score > 0.1:
            st.info("Some user overlap")
        else:
            st.caption("No collaborative signal")
    
    with col5:
        st.markdown("**🔥 Popularity**")
        st.progress(min(popularity_score, 1.0))
        st.caption(f"{popularity_score:.0%}")
        if popularity_score > 0.7:
            st.success("Trending now!")
        elif popularity_score > 0.4:
            st.info("Popular choice")
        elif popularity_score > 0.1:
            st.caption("Getting attention")
        else:
            st.caption("Niche product")


def perform_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Execute search with current user context."""
    if not query.strip():
        return []
    
    user_context = build_user_context()
    
    try:
        results = search_products(
            user_id=user_context["user_id"],
            query=query,
            top_k=top_k,
            debug_mode=False,
            override_context=user_context,  # Pass UI context to backend
        )
        return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def render_swipe_ui():
    """Render the Tinder-like swipe interface."""
    st.title("🔥 Discover Products")
    st.caption("Swipe **Right** ➡️ to Add to Cart | Swipe **Left** ⬅️ to Skip")

    # Remaining counter (from the current 30-item queue)
    if st.session_state.discovery_queue:
        remaining = len(st.session_state.discovery_queue)
        st.info(f"📚 Remaining in this batch: {remaining} / {len(st.session_state.discovery_queue)}")
    
    st.markdown("---")
    
    # Reload queue if empty
    if not st.session_state.discovery_queue:
        with st.spinner("Curating your feed..."):
            load_discovery_queue()
    
    if not st.session_state.discovery_queue:
        st.warning("No products found for your criteria. Try adjusting filters!")
        if st.button("Reset Filters"):
            st.session_state.preferred_brands = []
            st.session_state.preferred_categories = []
            st.session_state.search_query = ""
            st.rerun()
        return

    # Handle exhausted queue
    if st.session_state.current_index >= len(st.session_state.discovery_queue):
        st.info("No more products — refine your search")
        return

    # Build remaining cards as a stack
    remaining_products = st.session_state.discovery_queue[st.session_state.current_index:]
    if not remaining_products:
        st.info("No more products — refine your search")
        return

    # Get Top Card (front of stack)
    product = remaining_products[0]
    payload = product.get("payload", {})
    
    # Prepare details for card
    name = payload.get("name", "Unknown")
    price = payload.get("price", 0)
    final_score = product.get("final_score", 0.0)
    brand = payload.get("brand", "Unknown")
    
    # Determine match quality text
    match_text = "N/A"
    match_emoji = "🤔"
    if final_score > 0.8:
        match_text = "Perfect Match"
        match_emoji = "🌟"
    elif final_score > 0.6:
        match_text = "Good Match"
        match_emoji = "👍"
    elif final_score > 0.4:
        match_text = "Potential"
        match_emoji = "💭"
    
    categories = payload.get("categories", [])
    if not isinstance(categories, list):
        categories = [categories] if categories else []
    category = categories[0] if categories else "General"

    description_text = (
        f"**{name}**\n\n"
        f"Price: **${price:,.2f}**\n"
        f"Brand: **{brand}**\n"
        f"Category: **{category}**\n"
        f"Match: {match_emoji} {match_text} ({final_score:.0%})\n\n"
        f"{payload.get('description', '')[:150]}..."
    )
    
    # Prepare stack cards
    cards = []
    for idx, p in enumerate(remaining_products):
        p_payload = p.get("payload", {})
        p_name = p_payload.get("name", "Unknown")
        p_price = p_payload.get("price", 0)
        p_final_score = p.get("final_score", 0.0)
        p_brand = p_payload.get("brand", "Unknown")
        p_categories = p_payload.get("categories", [])
        if not isinstance(p_categories, list):
            p_categories = [p_categories] if p_categories else []
        p_category = p_categories[0] if p_categories else "General"
        p_match_text = "N/A"
        p_match_emoji = "🤔"
        if p_final_score > 0.8:
            p_match_text = "Perfect Match"
            p_match_emoji = "🌟"
        elif p_final_score > 0.6:
            p_match_text = "Good Match"
            p_match_emoji = "👍"
        elif p_final_score > 0.4:
            p_match_text = "Potential"
            p_match_emoji = "💭"

        p_description_text = (
            f"**{p_name}**\n\n"
            f"Price: **${p_price:,.2f}**\n"
            f"Brand: **{p_brand}**\n"
            f"Category: **{p_category}**\n"
            f"Match: {p_match_emoji} {p_match_text} ({p_final_score:.0%})\n\n"
            f"{p_payload.get('description', '')[:150]}..."
        )

        cards.append({
            "id": p.get("id") or p_payload.get("product_id") or f"idx_{st.session_state.current_index + idx}",
            "name": p_brand,
            "description": p_description_text,
            "image": p_payload.get("image_url", "https://via.placeholder.com/400x400?text=No+Image"),
        })
    
    # Unique key for React component to reset on new card
    current_key = f"swipe_{product.get('id', 'u')}_{st.session_state.interaction_count}"
    
    # Render Swipe Component (stacked cards)
    result = streamlit_swipecards(
        cards=cards,
        key=current_key
    )
    
    if result:
        # Resolve swiped product from result or fallback to top card
        swiped_product = product
        if isinstance(result, dict):
            swiped_id = result.get("id") or result.get("card", {}).get("id")
            if swiped_id:
                for p in remaining_products:
                    p_id = p.get("id") or p.get("payload", {}).get("product_id")
                    if str(p_id) == str(swiped_id):
                        swiped_product = p
                        break

        process_swipe_result(result, swiped_product)

        # Remove swiped product from queue to keep stack accurate
        swiped_id = swiped_product.get("id") or swiped_product.get("payload", {}).get("product_id")
        st.session_state.discovery_queue = [
            p for p in st.session_state.discovery_queue
            if (p.get("id") or p.get("payload", {}).get("product_id")) != swiped_id
        ]
        st.rerun()

    # Click-to-view details (logs click interaction)
    details_key = f"details_{product.get('id', 'u')}_{st.session_state.current_index}"
    if details_key not in st.session_state:
        st.session_state[details_key] = False

    if st.button("🔎 View details & why recommended", key=f"details_btn_{details_key}"):
        st.session_state[details_key] = True
        on_product_click(product, st.session_state.search_query)

    if st.session_state[details_key]:
        st.markdown("### Product Details")
        st.write(payload.get("description", "No description available"))
        st.markdown("### Why We Recommended This")
        explanations = product.get("explanations", [])
        if explanations:
            for i, exp in enumerate(explanations, 1):
                st.write(f"{i}. {exp}")
        else:
            st.caption("No explanation available.")
        st.markdown("---")
        
    # Show detailed explanation below
    with st.expander("🔍 See Why We Picked This For You", expanded=False):
        render_explanation(
            product.get("semantic_score", 0.0),
            product.get("affordability_score", 0.0),
            product.get("preference_score", 0.0),
            product.get("collaborative_score", 0.0),
            product.get("popularity_score", 0.0),
            payload,
        )


def render_cart_ui():
    """Render the shopping cart view."""
    st.title(f"🛒 My Cart ({len(st.session_state.cart)})")
    
    if not st.session_state.cart:
        st.info("Your cart is empty.")
        if st.button("Start Swiping"):
            st.session_state.view_mode = "Swipe"
            st.rerun()
        return

    total = 0.0
    items_to_delete = []
    
    for i, item in enumerate(st.session_state.cart):
        payload = item.get("payload", {})
        price = payload.get("price", 0.0)
        total += price
        
        with st.container():
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                img_url = payload.get("image_url", "")
                if img_url:
                    st.image(img_url, width=80)
            with c2:
                st.subheader(payload.get("name", "Unknown"))
                st.caption(f"{payload.get('brand', 'N/A')} - ${price:,.2f}")
            with c3:
                if st.button("❌ Remove", key=f"del_{i}"):
                    items_to_delete.append(i)
        st.divider()
    
    # Delete items in reverse order to preserve indices
    for i in reversed(items_to_delete):
        st.session_state.cart.pop(i)
    
    if items_to_delete:
        st.rerun()
    
    st.markdown(f"## Total: ${total:,.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Continue Shopping", use_container_width=True):
            st.session_state.view_mode = "Swipe"
            st.rerun()
    
    with col2:
        if st.button("💳 Checkout", type="primary", use_container_width=True):
            user_context = build_user_context()
            for item in st.session_state.cart:
                log_interaction(
                    user_id=user_context["user_id"],
                    product_payload=_build_product_payload_full(item),
                    interaction_type="purchase",
                    user_context=user_context
                )
            st.session_state.cart = []
            st.balloons()
            st.success("🎉 Purchase Complete! Thank you!")
            time.sleep(2)
            st.rerun()


def render_main_area():
    """Render main area based on current view mode."""
    if st.session_state.view_mode == "Swipe":
        render_swipe_ui()
    else:
        render_cart_ui()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
