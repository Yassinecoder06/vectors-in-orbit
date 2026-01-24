# Updated functions for app.py
# Copy these into app.py to replace the existing functions

# 1. Updated render_explanation with collaborative and popularity scores
def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    collaborative_score: float,
    popularity_score: float,
    payload: Dict[str, Any],
):
    """Render detailed explanation for product recommendation with all 5 scores."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**🎯 Semantic**")
        st.progress(min(semantic_score, 1.0))
        st.caption(f"{semantic_score:.0%}")
        if semantic_score >= 0.7:
            st.success("Strong")
        elif semantic_score >= 0.4:
            st.info("Moderate")
        else:
            st.warning("Weak")
    
    with col2:
        st.markdown("**💰 Affordability**")
        st.progress(min(affordability_score, 1.0))
        st.caption(f"{affordability_score:.0%}")
        
        price = payload.get("price", 0)
        total_budget = st.session_state.available_balance + st.session_state.credit_limit
        
        if affordability_score >= 0.8:
            st.success(f"${price:,.0f}")
        elif affordability_score >= 0.5:
            st.info(f"${price:,.0f}")
        else:
            st.warning(f"${price:,.0f}")
    
    with col3:
        st.markdown("**❤️ Preference**")
        st.progress(min(preference_score, 1.0))
        st.caption(f"{preference_score:.0%}")
        
        brand = payload.get("brand", "")
        categories = payload.get("categories", [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []
        
        brand_match = brand.lower() in [b.lower() for b in st.session_state.preferred_brands]
        preferred_categories = [c.lower() for c in st.session_state.preferred_categories]
        category_match = any(str(c).lower() in preferred_categories for c in categories)
        
        if brand_match or category_match:
            st.success("Match!")
        elif not st.session_state.preferred_brands and not st.session_state.preferred_categories:
            st.info("No prefs")
        else:
            st.warning("No match")
    
    with col4:
        st.markdown("**🤝 Collaborative**")
        st.progress(min(collaborative_score, 1.0))
        st.caption(f"{collaborative_score:.0%}")
        if collaborative_score > 0.5:
            st.success("High")
        elif collaborative_score > 0.1:
            st.info("Medium")
        else:
            st.caption("Low")
    
    with col5:
        st.markdown("**🔥 Popularity**")
        st.progress(min(popularity_score, 1.0))
        st.caption(f"{popularity_score:.0%}")
        if popularity_score > 0.7:
            st.success("Trending!")
        elif popularity_score > 0.4:
            st.info("Popular")
        elif popularity_score > 0.1:
            st.caption("Some buzz")
        else:
            st.caption("New/Niche")


# 2. Updated card rendering section (lines 361-396)
# Replace the section starting from "# Description" to "st.markdown("---")"
"""
        # Description
        if description:
            st.markdown("**Description**")
            st.write(f"{description[:100]}{'...' if len(description) > 100 else ''}")
        
        # Display explanations prominently
        if explanations:
            st.markdown("**Why this product?**")
            for explanation in explanations[:3]:  # Show top 3 explanations
                st.caption(f"• {explanation}")
        
        # Expandable explanation section
        with st.expander("🔍 See detailed breakdown"):
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
        
        # Interaction buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("👁️ View Details", key=f"view_{rank}_{name[:10]}", use_container_width=True):
                on_product_click(product, st.session_state.get("search_query", ""))
                st.info(f"Viewing details for {name}")
        
        with col_btn2:
            if st.button("🛒 Add to Cart", key=f"cart_{rank}_{name[:10]}", use_container_width=True):
                on_add_to_cart(product, st.session_state.get("search_query", ""))
        
        with col_btn3:
            if st.button("💳 Buy Now", key=f"buy_{rank}_{name[:10]}", use_container_width=True):
                on_purchase(product, st.session_state.get("search_query", ""))
        
        st.markdown("---")
"""
