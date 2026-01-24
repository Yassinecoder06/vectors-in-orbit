# 🛒 Vectors In Orbit - Context-Aware FinCommerce Engine

A production-ready e-commerce recommendation system that combines **semantic search**, **financial constraints**, **collaborative filtering**, and **real-time popularity tracking** using Qdrant vector database.

## ✨ Key Features

- 🎯 **Semantic Search**: Find products based on natural language understanding
- 💰 **Affordability-Aware**: Respects user's real-time balance and credit limits
- ❤️ **Preference Matching**: Learns and respects brand/category preferences
- 🤝 **Collaborative Filtering**: "Users like you also purchased..."
- 🔥 **Trending Products**: Real-time popularity with time-decay (6-hour half-life)
- 📊 **Multi-Reason Explanations**: Transparent AI with 5 explanation categories
- ⚡ **Sub-second Queries**: ~800ms end-to-end latency
- 🎨 **Interactive UI**: Streamlit-based demo with real-time interaction tracking

## 🏗️ Architecture

### 4 Qdrant Collections
- **products_multimodal** (384D): Product catalog with semantic embeddings
- **user_profiles** (384D): User preferences and demographics
- **financial_contexts** (256D): Real-time financial data (balance, credit)
- **interaction_memory** (384D): User interactions (view, click, cart, purchase)

### 5-Signal Scoring Formula
```
final_score = 0.30×semantic + 0.25×affordability + 0.15×preference + 0.20×collaborative + 0.10×popularity
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Qdrant Cloud Account**: [Sign up free](https://cloud.qdrant.io/)
- **GPU** (optional): For faster embeddings (CUDA-compatible)

## 🚀 Quick Start

### 1. Clone and Navigate
```bash
cd "C:\Work\Vectors In Orbit"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key packages:**
- `qdrant-client` - Vector database client
- `sentence-transformers` - Embeddings (all-MiniLM-L6-v2)
- `streamlit` - Interactive UI
- `numpy` - Numerical computations
- `python-dotenv` - Environment management

### 3. Configure Qdrant Cloud

Create a `.env` file in the project root:
```bash
# .env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key_here
```

Get your credentials:
1. Go to [Qdrant Cloud Dashboard](https://cloud.qdrant.io/)
2. Create a cluster (free tier available)
3. Copy the URL and API key

### 4. Setup Qdrant Collections

```bash
python qdrant_setup.py
```

This creates 4 collections with proper schemas and payload indexes.

**Expected output:**
```
✓ Successfully connected to Qdrant
✓ Collection 'products_multimodal' created (384D)
✓ Collection 'user_profiles' created (384D)
✓ Collection 'financial_contexts' created (256D)
✓ Collection 'interaction_memory' created (384D)
```

### 5. Generate and Insert Data

```bash
python generate_and_insert_data.py
```

This will:
- Load product data from CSV files (Amazon, Walmart, Lazada, Shein)
- Generate embeddings using SentenceTransformer
- Create sample user profiles and financial contexts
- Batch upload to Qdrant (~100 products per batch)

**Expected time:** 2-5 minutes depending on dataset size and GPU availability

### 6. Launch the Application

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`

## 🎮 Usage

### In the Streamlit UI

1. **Select User Persona**: Choose from Student, Professional, Executive, or Custom
2. **Set Financial Context**: Adjust balance and credit limit
3. **Choose Preferences**: Select preferred brands and categories
4. **Search**: Enter natural language queries like:
   - "Laptop for machine learning under 1500"
   - "Running shoes for marathons"
   - "Affordable smartphone with good camera"

### Interaction Tracking

The system automatically logs:
- **Views** (weight: 0.1) - When products are displayed
- **Clicks** (weight: 0.3) - "View Details" button
- **Add to Cart** (weight: 0.6) - "Add to Cart" button
- **Purchases** (weight: 1.0) - "Buy Now" button

### Understanding Results

Each product shows:
- **5 Score Breakdown**: Semantic, Affordability, Preference, Collaborative, Popularity
- **Top 3 Explanations**: Why this product was recommended
- **Financial Info**: Monthly installment options (if eligible)
- **Interactive Buttons**: Track user behavior in real-time

## 🧪 Testing

### Test the Search Pipeline
```bash
python search_pipeline.py
```

This runs a demo search and displays:
- Top 5 recommended products
- Score breakdowns
- Explanations
- Pretty-printed tables (if `rich` is installed)

### Validate Formula Weights
```bash
python test_new_features.py
```

Checks:
- ✅ Formula weights sum to 1.0
- ✅ Interaction logging works
- ✅ Popularity calculation
- ✅ All 4 interaction types

## 📁 Project Structure

```
Vectors In Orbit/
├── app.py                          # Streamlit UI (main application)
├── search_pipeline.py              # Core search & ranking logic
├── interaction_logger.py           # Real-time interaction tracking
├── qdrant_setup.py                 # Collection schema setup
├── generate_and_insert_data.py    # Data ingestion pipeline
├── requirements.txt                # Python dependencies
├── .env                            # Qdrant credentials (create this)
│
├── data/
│   ├── amazon/
│   │   ├── amazon-products.csv
│   │   ├── prepare_data.py
│   │   └── amazon_products_payload.json
│   ├── lazada/
│   ├── shein/
│   ├── walmart/
│   ├── combine_all_data.py
│   └── all_products_payload.json
│
├── report/
│   ├── report.tex        # Technical report (LaTeX)
│   ├── compile_tex_to_pdf.py       # PDF generator
│   └── [other reports]
│
└── tools/
    └── check_gpu.py                # GPU availability checker
```

## 🔧 Advanced Configuration

### Adjust Scoring Weights

Edit `search_pipeline.py` (line ~408):
```python
final_score = (
    0.30 * semantic_score +      # Query relevance
    0.25 * affordability_score + # Budget fit
    0.15 * preference_score +    # Brand/category match
    0.20 * collaborative_score + # Similar users
    0.10 * popularity_score      # Trending
)
```

### Modify Interaction Weights

Edit `interaction_logger.py` (line ~15):
```python
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0
}
```

### Change Time Decay

**Popularity** (6-hour half-life) in `interaction_logger.py` (line ~146):
```python
decay_constant = np.log(2) / (6 * 3600)  # 6 hours
```

**Collaborative** (7-day half-life) in `search_pipeline.py` (line ~213):
```python
decay_constant = np.log(2) / 604800  # 7 days
```

## 📊 Performance Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Embedding (GPU) | ~50ms | CUDA-accelerated |
| Embedding (CPU) | ~150ms | all-MiniLM-L6-v2 |
| Vector Search | ~200ms | Qdrant Cloud |
| Popularity Fetch | ~200ms | 1K interactions/24h |
| Reranking | ~150ms | 10 products |
| **Total Query** | **~800ms** | End-to-end |
| Interaction Log | <50ms | Single upsert |

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'qdrant_client'"
```bash
pip install qdrant-client sentence-transformers streamlit
```

### "Collection not found" error
Run the setup script first:
```bash
python qdrant_setup.py
```

### "No products in collection" error
Insert data:
```bash
python generate_and_insert_data.py
```

### Slow embeddings (CPU)
Check GPU availability:
```bash
python tools/check_gpu.py
```

If CUDA is available but not detected:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### UI not opening
Check if port 8501 is available:
```bash
streamlit run app.py --server.port 8502
```

## 📚 Documentation

- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md` - Comprehensive technical guide
- **Completion Report**: `COMPLETION_REPORT.md` - Feature implementation details
- **Update Instructions**: `UPDATE_INSTRUCTIONS.md` - Upgrade guide
- **Quick Start**: `QUICK_START.md` - Minimal setup guide
- **Technical Report**: `report/final_submission.tex` - Academic paper (LaTeX)

## 🎓 Educational Use

### Generate PDF Report
```bash
cd report
python compile_tex_to_pdf.py
```

Requires LaTeX distribution (TeX Live, MiKTeX, or similar).

### Understand the Math

The scoring formula balances 5 signals:

1. **Semantic**: Cosine similarity between query and product embeddings
2. **Affordability**: `1 - (price / total_budget)` clamped to [0,1]
3. **Preference**: Jaccard similarity for brands/categories
4. **Collaborative**: Weighted average of similar users' interactions
5. **Popularity**: Time-decayed interaction counts (exponential decay)

## 🔐 Security Notes

- **Never commit `.env`** - Add to `.gitignore`
- **Rotate API keys** regularly
- **Use read-only keys** for production frontends
- **Validate user inputs** before embedding

## 🚀 Production Deployment

### Recommended Optimizations

1. **Cache popularity scores** (5-10 min TTL)
2. **Batch interaction logging** (queue + periodic flush)
3. **Pre-compute user behavior vectors** (hourly refresh)
4. **Add Redis layer** for hot data
5. **Monitor Qdrant metrics** (query latency, collection size)

### Scaling Considerations

- **Qdrant**: Scales horizontally with sharding
- **Embeddings**: Use batch processing for bulk updates
- **Interactions**: Partition by date for archival
- **UI**: Deploy with Streamlit Cloud or containerize

## 🤝 Contributing

This is a hackathon demo project. Key areas for enhancement:

- [ ] A/B testing framework for scoring weights
- [ ] Category-specific trending ("Popular in Electronics")
- [ ] Seasonal boost factors
- [ ] Multi-language support
- [ ] Image-based search (CLIP embeddings)
- [ ] Real payment gateway integration

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Qdrant** - Vector database platform
- **Sentence Transformers** - Embedding models
- **Streamlit** - Rapid UI development
- **Hugging Face** - Model hosting

---

**Built with ❤️ for the Vectors In Orbit Hackathon**  
*Demonstrating the power of vector databases in modern e-commerce*

For questions or issues, please check the documentation files or create an issue.
