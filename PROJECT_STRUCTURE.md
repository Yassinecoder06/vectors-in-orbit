# Project Organization

This document describes the structure of the Vectors In Orbit e-commerce recommendation system.

## Directory Structure

```
Vectors In Orbit/
├── Core Application Files
│   ├── app.py                          Streamlit UI application
│   ├── search_pipeline.py              Search and ranking engine
│   ├── interaction_logger.py           Interaction tracking system
│   ├── qdrant_setup.py                 Qdrant schema setup
│   └── generate_and_insert_data.py     Data generation and loading
│
├── cf/                                 Financial-Aware Collaborative Filtering
│   ├── __init__.py
│   └── fa_cf.py                        FA-CF core algorithm
│
├── explanations/                       Recommendation explanations
│   ├── __init__.py
│   └── generator.py                    Explanation generation logic
│
├── data/                               Product data
│   ├── all_products_payload.json       Combined product data
│   ├── combine_all_data.py             Data combination script
│   ├── amazon/                         Amazon product data
│   ├── walmart/                        Walmart product data
│   ├── lazada/                         Lazada product data
│   └── shein/                          Shein product data
│
├── Configuration Files
│   ├── .env                            Qdrant credentials
│   ├── .gitignore                      Git ignore rules
│   ├── requirements.txt                Python dependencies
│   └── README.md                       Main project README
│
└── PROJECT_STRUCTURE.md                This file
```

## Getting Started

1. **Setup**: Read [README.md](README.md)
2. **Install**: `pip install -r requirements.txt`
3. **Configure**: Set up `.env` with Qdrant credentials
4. **Initialize**: `python qdrant_setup.py`
5. **Load Data**: `python generate_and_insert_data.py`
6. **Run UI**: `streamlit run app.py`

## Core Components

### search_pipeline.py
Main recommendation engine that combines 5 signals:
- Semantic relevance (40%)
- Affordability alignment (25%)
- User preferences (15%)
- Collaborative filtering (15%)
- Popularity trends (5%)

### interaction_logger.py
Real-time interaction tracking system with:
- Non-blocking writes to Qdrant
- Financial context validation
- Time-decay for historical data
- Popularity cache management

### cf/fa_cf.py
Financial-aware collaborative filtering that:
- Respects budget constraints
- Aligns on affordability ratios
- Provides real-time signals via self-boost
- Validates all recommendations

### app.py
Streamlit UI featuring:
- User persona selection
- Financial context configuration
- Real-time search and recommendations
- Interactive scoring breakdowns
- Explanation display

## Data Flow

```
User Query
    ↓
[search_pipeline.py] Embed and search
    ↓
[Qdrant Cloud] Vector similarity retrieval
    ↓
[search_pipeline.py] Multi-signal reranking
    ├── Semantic matching
    ├── Affordability check
    ├── Preference alignment
    ├── [cf/fa_cf.py] Collaborative filtering
    └── [interaction_logger.py] Popularity scores
    ↓
[explanations/generator.py] Generate reasons
    ↓
[app.py] Display to user
    ↓
User Interaction Logged
    ↓
[interaction_logger.py] Store in Qdrant
```

## Testing

### Run FA-CF Tests
```bash
python -m tests.test_fa_cf
```

Validates:
- Budget divergence (expensive items get zero boost for low-budget users)
- Financial alignment (similar affordability ratios increase CF scores)
- Real-time interaction (post-purchase scores exceed pre-purchase)
- CF comparison (CF-enabled rankings differ from semantic-only)

### Run Pipeline Demo
```bash
python search_pipeline.py
```

Shows top 5 recommendations with score breakdowns.

## Production Setup

```bash
# 1. Navigate to project
cd "c:\Work\Vectors In Orbit"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Edit .env file:
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your_api_key

# 4. Initialize schema
python qdrant_setup.py

# 5. Load data
python generate_and_insert_data.py

# 6. Run application
streamlit run app.py
```

## Key Features by Module

### Financial-Aware CF (cf/fa_cf.py)
- Financial alignment scoring
- Budget gating (hard constraint)
- Weighted interaction profiles
- Cross-user similarity filtering

### Interaction Logging (interaction_logger.py)
- Real-time logging with financial context
- Automatic affordability ratio calculation
- Backward compatibility
- Popularity cache (5-minute TTL)

### Search Pipeline (search_pipeline.py)
- GPU-accelerated embeddings
- Multi-signal reranking (5 components)
- Budget-aware scoring
- Explanation generation

### UI Application (app.py)
- Streamlit-based interactive demo
- Real-time user context configuration
- Interaction tracking
- Trending products display

## Qdrant Collections

### products_multimodal
- Vector: 384D SentenceTransformer embeddings
- Payload: product name, description, price, categories, brand, in_stock, image_url
- Purpose: Semantic search for product discovery

### interaction_memory
- Vector: Behavioral embeddings
- Payload: user_id, product_id, interaction_type, timestamp, financial context
- Purpose: Collaborative filtering and popularity signals

### user_profiles
- Vector: Preference embeddings
- Payload: user name, location, preferred categories, preferred brands
- Purpose: User preference matching

### financial_contexts
- Payload: available_balance, credit_limit, current_debt, eligible_installments
- Purpose: Budget validation and affordability scoring

## Configuration

### Scoring Weights
Edit search_pipeline.py to adjust signal contributions:
```python
final_score = (
    0.40 * semantic_score +      
    0.25 * affordability_score + 
    0.15 * preference_score +    
    0.15 * collaborative_score + 
    0.05 * popularity_score      
)
```

### Interaction Weights
Edit interaction_logger.py:
```python
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0
}
```

### Time Decay Parameters
**Popularity** (6-hour half-life):
```python
decay_constant = np.log(2) / (6 * 3600)
```

**Collaborative** (7-day half-life):
```python
decay_constant = np.log(2) / 604800
```

## Performance Metrics

| Operation | Latency |
|-----------|---------|
| Query embedding (GPU) | 4-5ms |
| Vector search | 50-100ms |
| Reranking | 80-150ms |
| Total query | 150-250ms |
| Interaction logging | < 50ms |

## Development Workflow

When adding new features:
1. Create module in appropriate directory
2. Write tests in tests/
3. Update documentation
4. Run validation before committing
5. Maintain backward compatibility

## Documentation

Technical documentation is available in the `/report/` directory:
- **vectors_in_orbit_technical_report.pdf** (22 pages) - Complete technical deep-dive
- **vectors_in_orbit_project_report.pdf** (9 pages) - Executive summary and roadmap

## Security

- Never commit `.env` file
- Rotate API keys regularly
- Use read-only keys for production
- Always validate user inputs
- Monitor Qdrant collection sizes

## Maintenance

### Regular Tasks
- Weekly: Review error logs
- Monthly: Update dependencies
- Quarterly: Reindex collections if needed

### Cleanup
```bash
# Remove Python cache
Remove-Item -Recurse -Force __pycache__
```

## Architecture Highlights

**3-Layer Design**
1. **Presentation**: Streamlit UI with interactive components
2. **Business Logic**: Search pipeline and scoring modules
3. **Data**: Qdrant Cloud vector database

**Key Innovations**
- Financial constraints integrated into collaborative filtering
- Real-time affordability ratio calculations
- Multi-signal scoring with transparent breakdowns
- Non-blocking interaction logging for low latency

**Scalability Features**
- Batch embedding processing
- Qdrant Cloud for distributed storage
- Asynchronous interaction writes
- Cache-aware popularity scoring

---

**Project**: Vectors In Orbit  
**Version**: FA-CF v1.0 (Production Ready)  
**Last Updated**: January 26, 2026
