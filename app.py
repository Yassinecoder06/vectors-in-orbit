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

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="FinCommerce Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    
    st.sidebar.markdown("### 💰 Financial Context")
    
    # Available balance slider
    st.session_state.available_balance = st.sidebar.slider(
        "Available Balance ($)",
        min_value=0,
        max_value=50000,
        value=int(st.session_state.available_balance),
        step=100,
        help="Your current available funds",
    )
    
    # Credit limit slider
    st.session_state.credit_limit = st.sidebar.slider(
        "Credit Limit ($)",
        min_value=0,
        max_value=100000,
        value=int(st.session_state.credit_limit),
        step=500,
        help="Your maximum credit allowance",
    )
    
    # Risk tolerance selector
    st.session_state.risk_tolerance = st.sidebar.selectbox(
        "🎯 Risk Tolerance",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(st.session_state.risk_tolerance),
        help="How much product price volatility or financial stretch are you comfortable with?",
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
            matched = next((c for c in categories if str(c).lower() in preferred_categories), None)
            if matched:
                st.success(f"Matches brand ({brand}) & category ({matched})")
            else:
                st.success(f"Matches brand ({brand}) & category")
        elif brand_match:
            st.success(f"Matches preferred brand: {brand}")
        elif category_match:
            matched = next((c for c in categories if str(c).lower() in preferred_categories), None)
            if matched:
                st.success(f"Matches preferred category: {matched}")
            else:
                st.success("Matches preferred category")
        elif not st.session_state.preferred_brands and not st.session_state.preferred_categories:
            st.info("No preferences set - all products considered equally")
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


def render_main_area():
    """Render main search interface and results."""
    st.title("🛒 Context-Aware FinCommerce Engine")
    st.markdown(
        "Personalized product recommendations powered by **semantic search**, "
        "**affordability scoring**, and **preference matching**."
    )
    
    # Search input section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 What are you looking for?",
            value=st.session_state.search_query,
            placeholder="e.g., 'comfortable running shoes for marathon training'",
            label_visibility="collapsed",
        )
    
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    
    # Number of results selector
    top_k = st.slider("Number of recommendations", 3, 10, 5, key="top_k_slider")
    
    # Execute search
    if search_clicked and query.strip():
        st.session_state.search_query = query
        with st.spinner("🔍 Searching across products..."):
            st.session_state.search_results = perform_search(query, top_k)
            st.session_state.has_searched = True
    
    st.markdown("---")
    
    # Display results
    if st.session_state.has_searched:
        results = st.session_state.search_results
        
        if results:
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Recommendations", "🗺️ 3D Explorer"])
            
            with tab1:
                st.markdown(f"### 🎁 Top {len(results)} Recommendations")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = sum(r.get("final_score", 0) for r in results) / len(results)
                    st.metric("Avg Match", f"{avg_score:.0%}")
                with col2:
                    avg_price = sum(r.get("payload", {}).get("price", 0) for r in results) / len(results)
                    st.metric("Avg Price", f"${avg_price:,.0f}")
                with col3:
                    in_budget = sum(
                        1 for r in results 
                        if r.get("affordability_score", 0) >= 0.5
                    )
                    st.metric("In Budget", f"{in_budget}/{len(results)}")
                with col4:
                    pref_match = sum(
                        1 for r in results 
                        if r.get("preference_score", 0) >= 1.0
                    )
                    st.metric("Pref Match", f"{pref_match}/{len(results)}")
                
                st.markdown("---")
                
                # Render product cards
                for idx, product in enumerate(results, start=1):
                    render_product_card(product, idx)
            
            with tab2:
                st.markdown("### 🗺️ 3D Terrain Explorer")
                st.markdown(
                    "Explore your search results in an interactive 3D landscape. "
                    "**Green markers** = safe & affordable. **Orange markers** = risky/stretched. "
                    "**Red markers** = unsafe/unaffordable."
                )
                
                try:
                    if len(results) < 3:
                        st.warning("Need at least 3 products for visualization. Try searching for more results.")
                    else:
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
                            else:
                                st.warning("Unable to create the terrain payload. Try again in a moment.")
                        
                except Exception as e:
                    st.error(f"Error creating visualization: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info(
                "😕 No products found matching your criteria.\n\n"
                "Try:\n"
                "- Broadening your search terms\n"
                "- Increasing your budget\n"
                "- Removing preference filters"
            )
    else:
        # Welcome state
        st.markdown(
            """
            ### 👋 Welcome!
            
            Enter a search query above to discover personalized product recommendations.
            
            **How it works:**
            1. 🔍 **Semantic Search** - We understand your intent, not just keywords
            2. 💰 **Affordability Scoring** - Products ranked by your budget
            3. ❤️ **Preference Matching** - Your favorite brands & categories prioritized
            
            Configure your profile in the sidebar to personalize results!
            """
        )


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
