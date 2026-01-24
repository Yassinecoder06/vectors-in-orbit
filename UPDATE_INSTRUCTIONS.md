# Instructions for updating app.py

Replace lines 361-396 in app.py with:

```python
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
```

And update render_explanation function signature (around line 400) from:

```python
def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    payload: Dict[str, Any],
):
```

to:

```python
def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    collaborative_score: float,
    popularity_score: float,
    payload: Dict[str, Any],
):
```

And add two more columns to the function body after col3.
