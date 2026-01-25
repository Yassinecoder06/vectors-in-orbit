# 📁 Project Organization

This document describes the organized structure of the FinCommerce Vector Search Engine project.

## 📂 Directory Structure

```
Vectors In Orbit/
├── 📄 Core Application Files
│   ├── app.py                          # Streamlit UI application
│   ├── search_pipeline.py              # Search and ranking engine
│   ├── interaction_logger.py           # Interaction tracking system
│   ├── qdrant_setup.py                 # Qdrant schema setup
│   └── generate_and_insert_data.py     # Data generation script
│
├── 📂 cf/                              # Financial-Aware Collaborative Filtering
│   ├── __init__.py
│   └── fa_cf.py                        # FA-CF core algorithm
│
├── 📂 explanations/                    # Recommendation explanations
│   ├── __init__.py
│   └── generator.py                    # Explanation generation
│
├── 📂 scoring/                         # Scoring configuration
│   └── __init__.py                     # Weight constants
│
├── 📂 interactions/                    # Interaction handling (future)
│   └── __init__.py
│
├── 📂 data/                            # Product data
│   ├── all_products_payload.json       # Combined product data
│   ├── combine_all_data.py             # Data combination script
│   ├── amazon/                         # Amazon product data
│   ├── walmart/                        # Walmart product data
│   ├── lazada/                         # Lazada product data
│   └── shein/                          # Shein product data
│
├── 📂 tests/                           # Test suite
│   ├── test_fa_cf.py                   # FA-CF comprehensive tests
│   └── test_collaborative_filtering.py  # Original CF tests
│
├── 📂 demos/                           # Demo and verification scripts
│   ├── demo_fa_cf.py                   # FA-CF demonstration
│   ├── verify_fa_cf.py                 # FA-CF verification
│   ├── verify_interaction_payload.py    # Payload verification
│   └── debug_interactions.py           # Interaction debugging
│
├── 📂 docs/                            # Documentation
│   ├── FA_CF_VALIDATION_REPORT.md      # FA-CF validation report
│   ├── IMPLEMENTATION_CHECKLIST.md     # Implementation checklist
│   ├── README_OPTIMIZATION.md          # Optimization quick start
│   ├── BEFORE_AFTER_COMPARISON.md      # Performance comparison
│   ├── DOCUMENTATION_INDEX.md          # Documentation index
│   ├── LATENCY_OPTIMIZATION_COMPLETE.md # Optimization summary
│   └── OPTIMIZATION_SUMMARY.md         # Technical optimization guide
│
├── 📂 report/                          # LaTeX reports
│   ├── fa_cf_complete_implementation.tex # FA-CF complete report (40+ pages)
│   ├── complete_project_report.tex     # Full project report
│   ├── pipeline_report.tex             # Pipeline technical report
│   ├── report.tex                      # Hackathon report
│   ├── final_idea.tex                  # Project concept
│   └── compile_tex_to_pdf.py           # PDF compilation script
│
├── 📂 tools/                           # Utility scripts
│   └── check_gpu.py                    # GPU availability checker
│
├── 📂 __pycache__/                     # Python cache (auto-generated)
│
├── 📄 Configuration Files
│   ├── .env                            # Environment variables (Qdrant credentials)
│   ├── .gitignore                      # Git ignore rules
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # Main project README
│
└── 📄 Project Organization
    └── PROJECT_STRUCTURE.md            # This file
```

## 📚 Quick Navigation

### Getting Started
1. **Setup**: Read [README.md](../README.md)
2. **Install**: `pip install -r requirements.txt`
3. **Configure**: Set up `.env` with Qdrant credentials
4. **Initialize**: `python qdrant_setup.py`
5. **Load Data**: `python generate_and_insert_data.py`
6. **Run UI**: `streamlit run app.py`

### Documentation
- **For Beginners**: [README.md](../README.md)
- **For Engineers**: [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)
- **For Managers**: [docs/LATENCY_OPTIMIZATION_COMPLETE.md](docs/LATENCY_OPTIMIZATION_COMPLETE.md)
- **FA-CF Implementation**: [docs/FA_CF_VALIDATION_REPORT.md](docs/FA_CF_VALIDATION_REPORT.md)
- **Complete Documentation Index**: [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)

### Testing & Validation
- **Run FA-CF Tests**: `python -m tests.test_fa_cf`
- **Run Original CF Test**: `python -m tests.test_collaborative_filtering`
- **Run FA-CF Demo**: `python demos/demo_fa_cf.py`
- **Verify FA-CF**: `python demos/verify_fa_cf.py`
- **Verify Payloads**: `python demos/verify_interaction_payload.py`

### Reports (Academic/Technical)
- **FA-CF Complete Report**: [report/fa_cf_complete_implementation.tex](report/fa_cf_complete_implementation.tex) (40+ pages)
- **Full Project Report**: [report/complete_project_report.tex](report/complete_project_report.tex)
- **Pipeline Report**: [report/pipeline_report.tex](report/pipeline_report.tex)
- **Compile to PDF**: `cd report && python compile_tex_to_pdf.py`

## 🏗️ Architecture Overview

### Core Modules
```
search_pipeline.py (1035 lines)
├── Semantic Search (40% weight)
├── Affordability Scoring (25% weight)
├── Preference Matching (15% weight)
├── FA-CF Collaborative (15% weight)
└── Popularity Scoring (5% weight)

cf/fa_cf.py (180 lines)
├── Financial Alignment Calculation
├── User Interaction Profile Building
├── Budget Gating (hard constraint)
└── Cross-User CF Score Aggregation

interaction_logger.py (657 lines)
├── Real-time Interaction Logging
├── Financial Context Validation
├── Affordability Ratio Calculation
└── Popularity Cache Management
```

### Data Flow
```
User Query
    ↓
[search_pipeline.py] → Embed query (GPU/CPU)
    ↓
[Qdrant Cloud] → Semantic search (384D vectors)
    ↓
[search_pipeline.py] → Multi-signal reranking
    ├── Semantic similarity
    ├── Affordability check
    ├── Preference matching
    ├── [cf/fa_cf.py] → FA-CF scores
    └── [interaction_logger.py] → Popularity scores
    ↓
[explanations/generator.py] → Generate explanations
    ↓
Ranked Results → [app.py] → Display to user
    ↓
User Interaction → [interaction_logger.py] → Store in Qdrant
```

## 🎯 Key Features by Module

### Financial-Aware CF (`cf/fa_cf.py`)
- ✅ Financial alignment scoring (threshold: 0.5)
- ✅ Budget gating (hard constraint)
- ✅ Weighted interaction profiles
- ✅ Cross-user similarity filtering

### Interaction Logging (`interaction_logger.py`)
- ✅ Real-time logging with financial context
- ✅ Automatic affordability ratio calculation
- ✅ Backward compatibility (dual-mode signature)
- ✅ Popularity cache (5-minute TTL)

### Search Pipeline (`search_pipeline.py`)
- ✅ GPU-accelerated embeddings
- ✅ Multi-signal reranking (5 components)
- ✅ Budget-aware scoring
- ✅ Explanation generation

### UI Application (`app.py`)
- ✅ Streamlit-based interactive demo
- ✅ Real-time user context configuration
- ✅ Interaction tracking (view/click/cart/purchase)
- ✅ Trending products sidebar

## 📊 Production Deployment

### Prerequisites
1. Python 3.8+ installed
2. CUDA-capable GPU (optional, for faster embeddings)
3. Qdrant Cloud account (or self-hosted Qdrant)
4. ~500MB disk space for embeddings model

### Environment Setup
```bash
# 1. Clone repository
cd "c:\Work\Vectors In Orbit"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Edit .env file with Qdrant credentials:
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your_api_key

# 4. Initialize schema
python qdrant_setup.py

# 5. Load data
python generate_and_insert_data.py

# 6. Run application
streamlit run app.py
```

### Validation Checklist
```bash
# Verify FA-CF implementation
python demos/verify_fa_cf.py

# Verify interaction payloads
python demos/verify_interaction_payload.py

# Run comprehensive tests
python -m tests.test_fa_cf

# Run demo
python demos/demo_fa_cf.py
```

## 🔧 Development Workflow

### Adding New Features
1. **Create module** in appropriate directory (cf/, explanations/, scoring/)
2. **Write tests** in tests/
3. **Update documentation** in docs/
4. **Run validation** before committing

### Code Organization Principles
- **Modular Design**: Each module has single responsibility
- **Clear Separation**: UI (app.py) ↔ Logic (search_pipeline.py) ↔ Data (Qdrant)
- **Backward Compatibility**: Old code continues to work
- **Production-Ready**: Error handling, logging, validation

## 📖 Documentation Hierarchy

### Level 1: Quick Start (5 min)
- [README.md](../README.md) - Project overview and setup

### Level 2: User Guides (10-15 min)
- [docs/README_OPTIMIZATION.md](docs/README_OPTIMIZATION.md) - Performance optimization guide
- [docs/FA_CF_VALIDATION_REPORT.md](docs/FA_CF_VALIDATION_REPORT.md) - FA-CF validation

### Level 3: Technical Deep Dive (30+ min)
- [docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) - Complete optimization details
- [report/fa_cf_complete_implementation.tex](report/fa_cf_complete_implementation.tex) - 40+ page academic report

### Level 4: Reference (as needed)
- [docs/IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md) - Task tracking
- [docs/BEFORE_AFTER_COMPARISON.md](docs/BEFORE_AFTER_COMPARISON.md) - Performance comparison
- [docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md) - Complete doc index

## 🎓 Academic Reports

All LaTeX reports are in the `report/` directory:

1. **fa_cf_complete_implementation.tex** (Recommended)
   - 40+ pages covering entire FA-CF implementation
   - Mathematical formulations with equations
   - Complete code listings
   - Architecture diagrams
   - Test results and validation
   - Future work recommendations

2. **complete_project_report.tex**
   - Full project overview
   - All 4 Qdrant collections
   - Multi-signal scoring system
   - End-to-end architecture

3. **pipeline_report.tex**
   - Technical pipeline documentation
   - Data flow diagrams
   - Performance analysis

To compile:
```bash
cd report
python compile_tex_to_pdf.py
# Or manually: pdflatex fa_cf_complete_implementation.tex
```

## 🧹 Maintenance

### Regular Tasks
- **Weekly**: Review logs for errors
- **Monthly**: Update dependencies (`pip install -U -r requirements.txt`)
- **Quarterly**: Reindex Qdrant collections if schema changes

### Cleanup Commands
```bash
# Remove Python cache
Remove-Item -Recurse -Force __pycache__

# Remove LaTeX build artifacts
cd report
Remove-Item *.aux, *.log, *.out, *.toc -Force
```

## 📝 Notes

- **Financial Context Required**: All interactions must include available_balance and credit_limit
- **Budget Gating**: Hard constraint - never bypass affordability checks
- **Alignment Threshold**: 0.5 is calibrated; changing may affect recommendation quality
- **Cache TTL**: Popularity cache refreshes every 5 minutes
- **GPU Acceleration**: Automatically detected; falls back to CPU if unavailable

## 🤝 Contributing

When adding new features:
1. Follow existing module structure
2. Add comprehensive tests
3. Update relevant documentation
4. Maintain backward compatibility
5. Run validation suite before committing

---

**Last Updated**: January 25, 2026  
**Project**: Vectors In Orbit - Context-Aware FinCommerce Engine  
**Version**: FA-CF v1.0 (Production Ready)
