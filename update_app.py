#!/usr/bin/env python3
"""
Script to update app.py with new interaction logging and explainability features
"""

import re

# Read the file
with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update render_explanation function signature
old_signature = """def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    payload: Dict[str, Any],
):"""

new_signature = """def render_explanation(
    semantic_score: float,
    affordability_score: float,
    preference_score: float,
    collaborative_score: float,
    popularity_score: float,
    payload: Dict[str, Any],
):"""

content = content.replace(old_signature, new_signature)

# 2. Update the columns in render_explanation
old_columns = "col1, col2, col3 = st.columns(3)"
new_columns = "col1, col2, col3, col4, col5 = st.columns(5)"
content = content.replace(old_columns, new_columns)

# 3. Add collaborative and popularity columns (insert after col3 block)
col3_end = '''        else:
            st.warning("No preference match")'''

new_cols = '''        else:
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
            st.caption("Niche product")'''

content = content.replace(col3_end, new_cols)

# 4. Replace the TODO section with actual implementation
todo_pattern = r'''        # Expandable explanation section
        with st.expander\("🔍 Why this product\?"\):
            if description:
                st.markdown\("\*\*Full Description\*\*"\)
                st.write\(description\)
                st.markdown\("---"\)
            render_explanation\(
                semantic_score,
                affordability_score,
                preference_score,
                payload,
            \)
            
        # TODO: Hook for Interaction Logging.*?st.markdown\("---"\)'''

new_implementation = '''        # Display explanations prominently
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
        
        st.markdown("---")'''

content = re.sub(todo_pattern, new_implementation, content, flags=re.DOTALL)

# Write back
with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ app.py updated successfully!")
print("   - Updated render_explanation signature")
print("   - Added collaborative and popularity score displays")
print("   - Replaced TODO with interaction buttons")
print("   - Added explanations display")
