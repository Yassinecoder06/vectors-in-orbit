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
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import Swipe Component
from streamlit_swipecards import streamlit_swipecards

# Import backend search function
from search_pipeline import search_products, get_qdrant_client, PRODUCTS_COLLECTION
from interaction_logger import log_interaction, get_interaction_stats_by_type

# Import terrain component
from terrain_component import terrain_canvas

# Import financial visualization functions
from financial_semantic_viz import (
    determine_safety_colors,
    build_search_result_terrain_payload,
)

# =============================================================================
# Image URL Helper
# =============================================================================

PLACEHOLDER_IMAGE = "https://via.placeholder.com/400x400?text=No+Image"

def normalize_image_url(url: str) -> str:
    """
    Normalize an image URL with fallback to placeholder.
    
    Handles:
    - None/empty values
    - Whitespace stripping
    - Protocol-relative URLs (//example.com)
    - Invalid schemes
    
    Returns a valid image URL or placeholder.
    """
    if not url:
        return PLACEHOLDER_IMAGE
    
    url = str(url).strip()
    
    if not url:
        return PLACEHOLDER_IMAGE
    
    # Handle protocol-relative URLs
    if url.startswith("//"):
        url = "https:" + url
    
    # Validate URL scheme
    if not url.startswith(("http://", "https://", "data:")):
        return PLACEHOLDER_IMAGE
    
    return url


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
    except Exception:
        pass  # Silent fail - don't disrupt UX with logging errors
    
    st.toast("✅ Added to cart!", icon="🛒")


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
    except Exception:
        pass  # Silent fail - don't disrupt UX with logging errors
    
    st.toast("✅ Purchase logged! Thank you!", icon="💳")


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


def process_swipe_result(result, product: Dict[str, Any]):
    """Process the swipe result (right = like, left = pass).
    
    The streamlit_swipecards component returns a dict with:
    - lastAction: {action: 'left'|'right'|'back', cardIndex: int}
    - swipedCards: [{index: int, action: str}, ...]
    - totalSwiped: int
    - remainingCards: int
    """
    user_context = build_user_context()
    
    # Debug: show in sidebar what we received
    if "debug_swipe" not in st.session_state:
        st.session_state.debug_swipe = []
    st.session_state.debug_swipe.append(f"Result: {result}")
    st.session_state.debug_swipe = st.session_state.debug_swipe[-5:]

    action = None
    
    # The component returns action nested in lastAction
    if isinstance(result, dict):
        # Primary: get action from lastAction.action
        last_action = result.get("lastAction")
        if isinstance(last_action, dict):
            action = last_action.get("action")
        
        # Fallback to top-level action key
        if action is None:
            action = result.get("action")
        
        # Fallback to other possible keys
        if action is None:
            action = (
                result.get("direction")
                or result.get("swipe")
                or result.get("decision")
            )
    elif isinstance(result, str):
        action = result
    elif isinstance(result, bool):
        action = "right" if result else "left"

    # Normalize action values
    if isinstance(action, str):
        action = action.lower().strip()
        if action in {"like", "swiperight", "right", "r", "yes", "true"}:
            action = "right"
        elif action in {"dislike", "swipeleft", "left", "l", "no", "false", "pass", "skip"}:
            action = "left"
        elif action == "back":
            action = "back"
    
    st.session_state.debug_swipe.append(f"Action: {action}")
    
    if action == "right":
        if "cart" not in st.session_state:
            st.session_state.cart = []

        # Add to cart
        st.session_state.cart.append(product)
        st.session_state.debug_swipe.append(f"✅ Added! Cart: {len(st.session_state.cart)}")
        
        try:
            log_interaction(
                user_id=user_context["user_id"],
                product_payload=_build_product_payload_full(product),
                interaction_type="add_to_cart",
                user_context=user_context,
                query=st.session_state.search_query
            )
        except Exception:
            pass
        st.toast("✅ Added to cart!", icon="🛒")
        
    elif action == "left":
        st.session_state.debug_swipe.append("⏭️ Skipped")
        try:
            log_interaction(
                user_id=user_context["user_id"],
                product_payload=_build_product_payload_full(product),
                interaction_type="view",
                user_context=user_context,
                query=st.session_state.search_query
            )
        except Exception:
            pass
            
    elif action == "back":
        st.session_state.debug_swipe.append("⬅️ Back pressed")
        
    else:
        st.session_state.debug_swipe.append(f"⚠️ Unknown: {action}")
    
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
    
    /* Style for all Streamlit images to fit properly */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stImage"] img {
        object-fit: contain;
        border-radius: 8px;
        max-height: 300px;
        width: 100%;
    }
    
    /* Specific styling for cart thumbnails */
    .element-container [data-testid="stImage"] img {
        object-fit: cover;
        border-radius: 8px;
        max-width: 100%;
        height: auto;
    }
    
    /* Make swipe cards adjust to content and center */
    iframe[title*="streamlit_swipecards"],
    iframe[title*="swipe"],
    .stCustomComponentV1 iframe {
        height: 900px !important;
        min-height: 900px !important;
        width: 75% !important;
        max-width: 75% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }
    
    /* Also target parent container */
    .stCustomComponentV1 {
        height: 900px !important;
        min-height: 900px !important;
    }
</style>
""", unsafe_allow_html=True)

# User personas for quick selection
USER_PERSONAS = {
    "Student": {"balance": 500, "credit": 1000},
    "Professional": {"balance": 5000, "credit": 15000},
    "Executive": {"balance": 20000, "credit": 50000},
    "Custom": {"balance": 2500, "credit": 5000},
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
                timeout=30,  # 30 second timeout
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
        "liked_products": [],  # Products swiped right, pending review
        "skipped_products": [],  # Products swiped left
        "interaction_count": 0,
        "view_mode": "Swipe",  # 'Swipe', 'Cart', or 'Review'
        "current_index": 0,
        "last_queue_query": None,
        # 3D Terrain
        "terrain_seed": 42,
        "selected_terrain_product": None,
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
    view_options = ["Swipe & Shop", "3D Landscape", "My Cart"]
    current_index = 0
    if st.session_state.view_mode == "3D":
        current_index = 1
    elif st.session_state.view_mode == "Cart":
        current_index = 2
    
    def format_nav(x):
        if x == "My Cart":
            return f"🛒 Cart ({len(st.session_state.cart)})"
        elif x == "3D Landscape":
            return "🗺️ 3D Landscape"
        return "🔍 Swipe & Shop"
    
    mode = st.sidebar.radio(
        "View Mode",
        view_options,
        index=current_index,
        format_func=format_nav
    )
    if mode == "Swipe & Shop":
        st.session_state.view_mode = "Swipe"
    elif mode == "3D Landscape":
        st.session_state.view_mode = "3D"
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
    
    # Debug panel for swipe results
    if st.session_state.get("debug_swipe"):
        with st.sidebar.expander("🐛 Swipe Debug", expanded=False):
            for msg in st.session_state.debug_swipe:
                st.caption(msg)



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
                st.image(image_url, width="content")
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


@st.fragment
def render_swipe_cards():
    """Fragment for swipe cards - tracks swipes without page reload."""
    if not st.session_state.discovery_queue:
        return
    
    # Initialize liked products tracking if not exists
    if "liked_products" not in st.session_state:
        st.session_state.liked_products = []
    
    remaining_products = st.session_state.discovery_queue
    if not remaining_products:
        return
    
    # Prepare stack cards - simplified with just name and price
    cards = []
    for idx, p in enumerate(remaining_products):
        p_payload = p.get("payload", {})
        p_name = p_payload.get("name", "Unknown")[:50]  # Truncate for compact display
        p_price = p_payload.get("price", 0)
        p_image = normalize_image_url(p_payload.get("image_url"))

        cards.append({
            "id": p.get("id") or p_payload.get("product_id") or f"idx_{idx}",
            "name": p_name,
            "description": f"${p_price:,.2f}",  # Plain price, no markdown
            "image": p_image,
        })
    
    # Single key for the entire session - component tracks swipes internally
    current_key = f"swipe_session_{st.session_state.last_queue_query}"
    
    # Render Swipe Component - it handles all swipes internally
    result = streamlit_swipecards(
        cards=cards,
        key=current_key,
        view="desktop"
    )
    
    # Determine current card index based on swipes
    current_card_index = 0
    swiped_cards = []
    remaining_count = len(cards)
    
    # Process result - component returns all swiped cards info
    if result and isinstance(result, dict):
        swiped_cards = result.get("swipedCards", [])
        remaining_count = result.get("remainingCards", len(cards))
        current_card_index = len(swiped_cards)  # Next card after all swiped ones
        
        # Debug info
        if "debug_swipe" not in st.session_state:
            st.session_state.debug_swipe = []
        st.session_state.debug_swipe.append(f"Swiped: {len(swiped_cards)}, Remaining: {remaining_count}")
        st.session_state.debug_swipe = st.session_state.debug_swipe[-5:]
        
        # Collect all right-swiped products
        liked_indices = [s["index"] for s in swiped_cards if s.get("action") == "right"]
        
        # Update liked products list
        st.session_state.liked_products = [
            remaining_products[i] for i in liked_indices 
            if i < len(remaining_products)
        ]
        
        # Show progress
        if swiped_cards:
            liked_count = len([s for s in swiped_cards if s.get("action") == "right"])
            st.success(f"❤️ Liked: {liked_count} | ⏭️ Skipped: {len(swiped_cards) - liked_count}")
        
        # When all cards are swiped, show the review button
        if remaining_count == 0:
            if st.session_state.liked_products:
                st.balloons()
                st.success(f"🎉 Done! You liked {len(st.session_state.liked_products)} products!")
                if st.button("📦 Add All to Cart", type="primary", use_container_width=True, key="add_all_to_cart"):
                    # Add all liked products to cart
                    if "cart" not in st.session_state:
                        st.session_state.cart = []
                    st.session_state.cart.extend(st.session_state.liked_products)
                    cart_count = len(st.session_state.cart)
                    st.session_state.liked_products = []
                    st.session_state.discovery_queue = []
                    st.session_state.view_mode = "Cart"
                    st.toast(f"✅ Added {cart_count} items to cart!", icon="🛒")
                    # Full page rerun to switch to cart view
                    st.rerun()
            else:
                st.info("You didn't like any products. Try a new search!")

    # Get current product based on swipe progress
    if current_card_index < len(remaining_products):
        product = remaining_products[current_card_index]
        payload = product.get("payload", {})
        
        # Combined details and explanation section
        with st.expander(f"🔍 View Details: {payload.get('name', 'Product')[:40]}", expanded=False):
            on_product_click(product, st.session_state.search_query)
            
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
            
            render_explanation(
                product.get("semantic_score", 0.0),
                product.get("affordability_score", 0.0),
                product.get("preference_score", 0.0),
                product.get("collaborative_score", 0.0),
                product.get("popularity_score", 0.0),
                payload,
            )


def render_swipe_ui():
    """Render the Tinder-like swipe interface."""
    st.title("🔥 Discover Products")
    st.caption("Swipe **Right** ➡️ to Add to Cart | Swipe **Left** ⬅️ to Skip")

    # Remaining counter
    if st.session_state.discovery_queue:
        remaining = len(st.session_state.discovery_queue)
        st.info(f"📚 Remaining in this batch: {remaining}")
    
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
    
    # Render the swipe cards fragment (reruns independently)
    render_swipe_cards()


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
                    st.image(img_url, width=100)
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


def render_3d_landscape_ui():
    """Render the 3D Terrain Landscape view."""
    st.title("🗺️ 3D Financial Landscape")
    st.markdown(
        "Explore your search results in an interactive 3D landscape. "
        "**Green markers** = safe & affordable. **Orange markers** = risky/stretched. "
        "**Red markers** = unsafe/unaffordable."
    )
    
    # Check if we have search results
    results = st.session_state.discovery_queue
    
    if not results:
        st.info("👋 No products loaded yet. Use the search in the sidebar to discover products.")
        if st.button("Load Default Products", type="primary"):
            with st.spinner("Loading products..."):
                load_discovery_queue()
            st.rerun()
        return
    
    try:
        if len(results) < 3:
            st.warning("Need at least 3 products for visualization. Try searching for more results.")
            return
            
        prices = []
        final_scores = []
        
        user_context = build_user_context()
        total_budget = user_context["available_balance"] + user_context["credit_limit"]
        
        # Map risk tolerance to numeric
        risk_tolerance_map = {"Low": 0.2, "Medium": 0.5, "High": 0.8}
        user_risk_tolerance = risk_tolerance_map.get(user_context["risk_tolerance"], 0.5)
        
        for result in results:
            payload = result.get("payload", {})
            prices.append(payload.get("price", 0.0))
            final_scores.append(result.get("final_score", 0.0))
        
        # Prepare metadata for safety color calculation
        finance_metadata = {
            'prices': np.array(prices),
            'user_budgets': np.array([total_budget] * len(prices)),
            'risk_tolerances': np.array([user_risk_tolerance] * len(prices)),
            'final_scores': np.array(final_scores)
        }
        
        # Display safety color statistics
        safety_colors = determine_safety_colors(finance_metadata)
        green_count = np.sum(safety_colors == 'green')
        orange_count = np.sum(safety_colors == 'orange')
        red_count = np.sum(safety_colors == 'red')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 Safe & Affordable", f"{green_count}")
        with col2:
            st.metric("🟠 Risky/Stretched", f"{orange_count}")
        with col3:
            st.metric("🔴 Unsafe", f"{red_count}")
        with col4:
            avg_score = np.mean(finance_metadata['final_scores'])
            st.metric("Avg Score", f"{avg_score:.2f}")
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Your Budget:** ${total_budget:,.2f}")
        with col2:
            st.info(f"**Avg Product Price:** ${np.mean(finance_metadata['prices']):,.2f}")

        st.markdown("---")
        st.markdown("#### ⚡ 3D Terrain Explorer")
        st.caption(
            "Interactive 3D terrain with product markers. "
            "Use WASD keys to navigate, click products to select."
        )

        toggle_col, shuffle_col = st.columns([4, 1])
        with toggle_col:
            show_terrain = st.toggle(
                "Enable 3D terrain",
                value=True,
                key="terrain_toggle",
            )
        with shuffle_col:
            if show_terrain and st.button("Shuffle seed", use_container_width=True):
                st.session_state.terrain_seed = random.randint(1, 1_000_000)
                st.rerun()

        if show_terrain:
            with st.spinner("Rendering 3D terrain..."):
                terrain_payload = build_search_result_terrain_payload(
                    results=results,
                    coords=None,  # No UMAP coords needed - function handles positioning
                    user_risk_tolerance=user_risk_tolerance,
                    budget_override=total_budget,
                    random_seed=int(st.session_state.terrain_seed),
                )

            if terrain_payload:
                selected_terrain = terrain_canvas(
                    data=terrain_payload,
                    height=650,
                    key=f"terrain_canvas_{terrain_payload.get('meta', {}).get('seed', st.session_state.terrain_seed)}",
                )
                if selected_terrain:
                    st.session_state.selected_terrain_product = selected_terrain
                st.caption(
                    "Click and drag to orbit the scene. Scroll to zoom. WASD to navigate."
                )

                selected_product = st.session_state.get("selected_terrain_product")
                if selected_product:
                    st.markdown("##### 🧭 Selected Terrain Product")
                    left, right = st.columns([1, 2])
                    with left:
                        image_url = selected_product.get("imageUrl") or ""
                        if image_url:
                            st.image(image_url, width=220)
                    with right:
                        st.markdown(f"**{selected_product.get('name', 'Unknown Product')}**")
                        st.caption(
                            f"${selected_product.get('price', 0):,.2f} · "
                            f"{selected_product.get('brand', 'Unknown')} · "
                            f"{selected_product.get('category', 'Unknown')}"
                        )
                        description = selected_product.get("description") or "No description available."
                        st.write(description)
                        
                        # Add to cart button for selected product
                        if st.button("🛒 Add to Cart", key="terrain_add_cart", use_container_width=True):
                            # Find the matching product in results
                            for result in results:
                                payload = result.get("payload", {})
                                if payload.get("name") == selected_product.get("name"):
                                    on_add_to_cart(result, st.session_state.get("search_query", ""))
                                    break
            else:
                st.warning("Unable to create the terrain payload. Try again in a moment.")
                
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_main_area():
    """Render main area based on current view mode."""
    if st.session_state.view_mode == "Swipe":
        render_swipe_ui()
    elif st.session_state.view_mode == "3D":
        render_3d_landscape_ui()
    else:
        render_cart_ui()


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
