# 🚀 Quick Start Guide - New Features

## 🎯 What's New?

Your FinCommerce system now has **real-time interaction tracking** and **popularity-aware recommendations**!

---

## 📦 Installation

```bash
# Already done if you have existing environment
pip install qdrant-client sentence-transformers streamlit numpy
```

---

## 🎮 Usage

### 1. Run the Streamlit App
```bash
streamlit run app.py
```

**What's new in the UI:**
- ✅ **Auto-tracking**: Product views logged automatically
- ✅ **Interactive buttons**: View Details, Add to Cart, Buy Now
- ✅ **Explanations**: See why each product was recommended
- ✅ **5 Score Gauges**: Semantic, Affordability, Preference, Collaborative, **Popularity** (NEW)

---

### 2. Log Interactions Manually (Optional)

```python
from interaction_logger import log_interaction

# Log when a user buys something
log_interaction(
    user_id="user_123",
    product_payload={"id": "prod_456", "name": "MacBook", "price": 1299},
    interaction_type="purchase",  # or "view", "click", "add_to_cart"
    query="laptop for coding"
)
```

---

### 3. Check Trending Products

```python
from interaction_logger import get_top_interacted_products

# Get top 10 trending in last 24 hours
trending = get_top_interacted_products(timeframe_hours=24, top_k=10)

for p in trending:
    print(f"{p['product_id']}: popularity = {p['weighted_popularity_score']:.2f}")
```

---

### 4. Search with All Features

```python
from search_pipeline import search_products

results = search_products(
    user_id="demo_user",
    query="laptop under $1500",
    top_k=5,
    debug_mode=True  # Shows detailed logs
)

# Each result now has:
# - popularity_score (0-1)
# - explanations (list of reasons)
# - All original fields
```

---

## 🔍 What Changed?

### Scoring Formula
**OLD** (4 components):
```
final = 0.35*semantic + 0.30*affordability + 0.15*preference + 0.20*collaborative
```

**NEW** (5 components):
```
final = 0.30*semantic + 0.25*affordability + 0.15*preference + 0.20*collaborative + 0.10*popularity
```

### New Fields in Results
Every search result now includes:
- `popularity_score`: 0-1 (how trending the product is)
- `explanations`: List of human-readable reasons
- `reason`: Primary explanation (for backward compatibility)

---

## 🐛 Debugging

### Enable Debug Mode
```python
# In search
results = search_products(..., debug_mode=True)

# In popularity
trending = get_top_interacted_products(..., debug=True)
```

### Check Logs
Look for these emoji indicators:
- ✅ Success operations
- 📊 Statistics/metrics
- 🔍 Search operations
- ❌ Errors

---

## ⚙️ Configuration

### Change Interaction Weights
Edit `interaction_logger.py` line 17:
```python
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0
}
```

### Change Scoring Weights
Edit `search_pipeline.py` line ~408:
```python
final_score = (
    0.30 * semantic_score +
    0.25 * affordability_score +
    0.15 * preference_score +
    0.20 * collaborative_score +
    0.10 * popularity_score  # Adjust this!
)
```

### Change Time Decay
Edit `interaction_logger.py` line ~146:
```python
decay_constant = np.log(2) / (6 * 3600)  # 6-hour half-life
# For 12 hours: (12 * 3600)
# For 24 hours: (24 * 3600)
```

---

## 🧪 Testing

```bash
python test_new_features.py
```

Expected output:
```
✅ PASSED: Interaction Logging
✅ PASSED: Popularity Calculation
✅ PASSED: Search with Popularity
✅ PASSED: UI Hooks
✅ PASSED: Formula Weights

Total: 5/5 tests passed
🎉 ALL TESTS PASSED! System is ready for production.
```

---

## 📊 Common Tasks

### 1. Seed Some Interactions (for testing)
```python
from interaction_logger import log_interaction

products = [
    {"id": "p1", "name": "MacBook Pro", "price": 1299},
    {"id": "p2", "name": "Dell XPS", "price": 999},
    {"id": "p3", "name": "HP Laptop", "price": 699}
]

for p in products:
    log_interaction("test_user", p, "purchase", "laptop")
    log_interaction("test_user", p, "view", "laptop")
```

### 2. Check Interaction Count
```python
from search_pipeline import get_qdrant_client, INTERACTIONS_COLLECTION

client = get_qdrant_client()
info = client.get_collection(INTERACTIONS_COLLECTION)
print(f"Total interactions: {info.points_count}")
```

### 3. View Sample Explanations
```python
from search_pipeline import search_products

results = search_products("test_user", "laptop", top_k=3)

for r in results[:1]:  # Show first result
    print("\nExplanations:")
    for exp in r['explanations']:
        print(f"  • {exp}")
```

---

## ❓ FAQ

**Q: Why are popularity scores all 0?**
A: No interactions logged yet. Run sample interactions or wait for real usage.

**Q: Can I disable popularity?**
A: Yes! Set weight to 0 in scoring formula or don't call `get_top_interacted_products()`.

**Q: How do I reset interactions?**
A: Delete the `interaction_memory` collection in Qdrant or use scroll+delete.

**Q: What if search is slower now?**
A: Popularity adds ~100ms. Cache results for 5-10 min if needed.

**Q: Can I track custom events?**
A: Yes! Add to `INTERACTION_WEIGHTS` dict and use in `log_interaction()`.

---

## 🎯 Next Actions

1. ✅ Test the UI: `streamlit run app.py`
2. ✅ Generate sample interactions (use code above)
3. ✅ Search and see popularity in action
4. ✅ Check explanations on product cards
5. ✅ Monitor logs for debug output

---

## 📚 Full Documentation

- **Comprehensive Guide**: `IMPLEMENTATION_GUIDE.md`
- **Completion Report**: `COMPLETION_REPORT.md`
- **Test Suite**: `test_new_features.py`

---

## 💡 Pro Tips

1. **Popularity takes time**: Need ~100 interactions to see meaningful scores
2. **Debug mode is your friend**: Use it to understand scoring
3. **Explanations are customizable**: Edit messages in `search_pipeline.py`
4. **Weights are tunable**: Experiment to find optimal balance
5. **Cold-start is handled**: New products work fine (just no popularity boost)

---

**Ready to explore? Run `streamlit run app.py` and start shopping!** 🛒🚀
