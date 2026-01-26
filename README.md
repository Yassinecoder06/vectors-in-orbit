# Fin-e Trip

A production-ready e-commerce recommendation system that combines semantic search, financial constraints, collaborative filtering, and real-time popularity tracking using Qdrant vector database.

## Key Features

- **Semantic Search**: Find products based on natural language understanding
- **Affordability-Aware**: Respects user's real-time balance and credit limits
- **Preference Matching**: Learns and respects brand and category preferences
- **Collaborative Filtering**: Recommendations based on users with similar behavior
- **Trending Products**: Real-time popularity with time-decay (6-hour half-life)
- **Transparent Explanations**: Shows exactly why each product was recommended
- **Fast Performance**: 150-250ms end-to-end latency for queries
- **Interactive UI**: Streamlit-based demo with real-time interaction tracking

## Architecture

### 4 Qdrant Collections
- **products_multimodal**: Product catalog with semantic embeddings (384D)
- **user_profiles**: User preferences and demographics (384D)
- **financial_contexts**: Real-time financial data (balance, credit)
- **interaction_memory**: User interactions (view, click, cart, purchase)

### 5-Signal Scoring Formula
```
final_score = 0.40×semantic + 0.25×affordability + 0.15×preference + 0.15×collaborative + 0.05×popularity
```

## Prerequisites

- Python 3.8 or higher
- Qdrant Cloud Account (free tier available at https://cloud.qdrant.io/)
- GPU optional but recommended for faster embeddings

## Quick Start

### 1. Clone and Navigate
```bash
cd "C:\Work\Vectors In Orbit"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `qdrant-client` - Vector database client
- `sentence-transformers` - Embeddings (all-MiniLM-L6-v2)
- `streamlit` - Interactive UI
- `numpy` - Numerical computations

### 3. Configure Qdrant Cloud

Create a `.env` file in the project root:
```
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key_here
```

To get your credentials:
1. Go to https://cloud.qdrant.io/
2. Create a cluster (free tier available)
3. Copy the URL and API key from the dashboard

### 4. Setup Qdrant Collections

```bash
python qdrant_setup.py
```

This creates 4 collections with proper schemas and payload indexes.

### 5. Generate and Insert Data

```bash
python generate_and_insert_data.py
```

This will:
- Load product data from CSV files (Amazon, Walmart, Lazada, Shein)
- Generate embeddings using SentenceTransformer
- Create sample user profiles and financial contexts
- Upload to Qdrant (takes 2-5 minutes depending on GPU availability)

### 6. Launch the Application

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`

## Usage

### In the Streamlit UI

1. Select a user persona (Student, Professional, Executive, or Custom)
2. Set your financial context (available balance and credit limit)
3. Choose preferred brands and categories
4. Search with natural language queries:
   - "Laptop for machine learning under 1500"
   - "Running shoes for marathons"
   - "Affordable smartphone with good camera"

### Understanding the Results

Each recommendation shows:

- **Score Breakdown**: How each of the 5 signals contributed (Semantic, Affordability, Preference, Collaborative, Popularity)
- **Reasons**: Transparent explanation of why the product was recommended
- **Financial Info**: Price, monthly installment options, and budget fit
- **Interactive Buttons**: View details, add to cart, or purchase (all tracked in real-time)

### Example UI Output

The system displays scoring breakdowns showing how each signal contributes to the final recommendation:

- **Semantic Match** (40%): How relevant to your search query
- **Affordability** (25%): How well it fits your budget
- **Preference Match** (15%): Alignment with your brand/category preferences
- **Collaborative** (15%): What similar users purchased
- **Popularity** (5%): Current trending products

Each product also lists the top reasons it was recommended, such as:
- "Trending: Very popular in last 24 hours"
- "Well within your budget"
- "Users with similar behavior purchased this"
- "Relevant to your search"

## Advanced Configuration

### Adjust Scoring Weights

Edit `search_pipeline.py` (around line 408):
```python
final_score = (
    0.40 * semantic_score +      # Query relevance
    0.25 * affordability_score + # Budget fit
    0.15 * preference_score +    # Brand/category match
    0.15 * collaborative_score + # Similar users
    0.05 * popularity_score      # Trending
)
```

### Modify Interaction Weights

Edit `interaction_logger.py` (around line 15):
```python
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0
}
```

### Change Time Decay

**Popularity** (6-hour half-life) in `interaction_logger.py`:
```python
decay_constant = np.log(2) / (6 * 3600)  # 6 hours
```

**Collaborative** (7-day half-life) in `search_pipeline.py`:
```python
decay_constant = np.log(2) / 604800  # 7 days
```

## Performance

| Operation | Latency |
|-----------|---------|
| Query embedding | 4-5ms (GPU) |
| Vector search | 50-100ms |
| Multi-signal reranking | 80-150ms |
| Total query | 150-250ms |
| Interaction logging | < 50ms |

## Testing

### Run the Search Pipeline Demo
```bash
python search_pipeline.py
```

Displays top 5 recommended products with score breakdowns and explanations.

### Validate the Recommendation Formula
```bash
python -m tests.test_fa_cf
```

Checks budget constraints, financial alignment, real-time responsiveness, and collaborative filtering.

## Troubleshooting

### "ModuleNotFoundError: No module named 'qdrant_client'"
```bash
pip install qdrant-client sentence-transformers streamlit
```

### "Collection not found" error
Run the setup script:
```bash
python qdrant_setup.py
```

### "No products in collection" error
Insert data:
```bash
python generate_and_insert_data.py
```

### Slow performance or CPU-only embeddings
Check GPU availability:
```bash
python tools/check_gpu.py
```

If you have a CUDA GPU but it's not detected:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### UI not opening
Try a different port:
```bash
streamlit run app.py --server.port 8502
```

## Documentation

- **Technical Details**: See `/report/vectors_in_orbit_technical_report.pdf` for the full 22-page technical documentation
- **Project Overview**: See `/report/vectors_in_orbit_project_report.pdf` for project roadmap and timeline

## Project Structure

```
Vectors In Orbit/
├── app.py                          Main Streamlit UI
├── search_pipeline.py              Core search and ranking
├── interaction_logger.py           Real-time interaction tracking
├── qdrant_setup.py                 Database setup
├── generate_and_insert_data.py     Data pipeline
├── requirements.txt                Python dependencies
├── .env                            Qdrant credentials (create this)
│
├── cf/
│   └── fa_cf.py                    Financial-aware collaborative filtering
│
├── data/
│   ├── amazon/                     Amazon product data
│   ├── walmart/                    Walmart product data
│   ├── lazada/                     Lazada product data
│   └── shein/                      Shein product data
│
└── vectors_in_orbit_project_report.pdf
```

## Security Notes

- Never commit `.env` file (add to `.gitignore`)
- Rotate API keys regularly
- Use read-only keys for production frontends
- Always validate user inputs before processing

## Deployment

For production use:
1. Cache popularity scores (5-10 minute TTL)
2. Batch interaction logging for higher throughput
3. Add Redis layer for hot data
4. Monitor Qdrant metrics regularly
5. Consider Kubernetes orchestration for scaling

## Contributing

This project focuses on:
- Financial-aware recommendations
- Real-time budget constraints
- Transparent, explainable AI
- Production-ready code quality

## License

MIT License

---

**Fin-e Trip**: Bringing financial awareness to e-commerce recommendations through vector search and collaborative filtering.
