# Contributor Intelligence Platform

**Enterprise-grade contributor profiling system with PostgreSQL, 90-day activity analytics, and AI-powered skill extraction.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/database-PostgreSQL-blue.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Model](#data-model)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

---

## Overview

The Contributor Intelligence Platform is a production-ready system designed to analyze contributor work patterns, extract technical skills from project descriptions, and generate comprehensive intelligence summaries for talent matching and resource allocation.

### Key Capabilities

- **Email-Based Identity Management** - Contributors indexed by verified email addresses
- **90-Day Activity Analytics** - Real-time engagement tracking with weekly hour patterns
- **Intelligent Project Selection** - Automatic identification of top 5 most relevant projects
- **AI-Powered Skill Extraction** - LLM-based extraction of technical competencies from work history
- **Compliance Tracking** - Comprehensive KYC, risk assessment, and verification management
- **Production Database** - PostgreSQL with JSONB indexing and connection pooling
- **Scalable Architecture** - Modular design supporting 100K+ contributor profiles

---

## Architecture

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | Python 3.9+ | Data processing & business logic |
| **Database** | PostgreSQL 12+ | Persistent storage with JSONB |
| **AI/ML** | Ollama (Qwen 2.5) | Intelligence extraction |
| **Data Models** | Pydantic | Type-safe validation |
| **Config** | python-dotenv | Environment management |

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│            (Upload, View, Extract, Search)              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼─────┐          ┌─────▼──────┐
    │ Data     │          │Intelligence│
    │Processor │          │ Extractor  │
    └────┬─────┘          └─────┬──────┘
         │                      │
         │    ┌─────────────────┤
         │    │                 │
    ┌────▼────▼─┐          ┌───▼────┐
    │PostgreSQL │          │ Ollama │
    │  Database │          │  LLM   │
    └───────────┘          └────────┘
```

---

## Features

### Data Processing
- **Activity-Based Profiling** - Focus on recent 90-day engagement patterns
- **Top Project Selection** - Intelligent ranking by time_90d with lifetime fallback
- **Qualification Parsing** - Automatic extraction of passed certifications and tests
- **Language Analysis** - Multi-language proficiency tracking with comprehension validation

### Intelligence Extraction
- **Skill Identification** - Extraction of technical skills from project descriptions
- **Activity Summarization** - Weekly hour patterns and engagement consistency
- **Compliance Aggregation** - KYC status, risk tiers, and verification timeline
- **140-170 Word Summaries** - Concise, actionable intelligence profiles

### Compliance & Security
- **KYC Test Tracking** - Timestamped verification history
- **Risk Tier Management** - Multi-level risk classification
- **DOTS Verification** - Identity verification status tracking
- **Audit Trail** - Comprehensive logging of all operations

---

## Prerequisites

### Required Software

| Software | Minimum Version | Purpose |
|----------|----------------|---------|
| Python | 3.9+ | Application runtime |
| CUDA | 11.8+ | GPU acceleration (required) |
| PyTorch | 2.1+ | Deep learning framework |
| pip | 21.0+ | Package management |

### System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (22GB+ recommended for optimal performance)
  - Tested on: L4 (24GB), V100 (16GB), A10G (24GB), A100 (40GB)
  - Minimum: T4 (16GB) for smaller batches
- **RAM**: 16GB system RAM minimum (32GB recommended)
- **Disk**: 15GB free space (5GB for model cache, 10GB for data)
- **CPU**: 4+ cores recommended
- **OS**: Linux (preferred), macOS with Metal, or Windows 11 with WSL2
- **CUDA Drivers**: Compatible with CUDA 11.8+

### GPU Performance Notes

- **FlashAttention-2** provides 2-3x speedup (highly recommended)
- **4-bit quantization** reduces VRAM from 28GB → 14-18GB
- **Micro-batching** achieves 90-98% GPU utilization
- **Expected throughput**: 2-5 profiles/sec on L4 GPU

---

## Installation

### Step 1: Clone Repository

```bash
cd /path/to/project
git clone <repository-url>
cd OGs
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install PyTorch with CUDA

```bash
# Install PyTorch 2.1+ with CUDA 11.8+ support
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### Step 4: Install FlashAttention-2 (Highly Recommended)

```bash
# Install FlashAttention-2 for 2-3x speedup
# Requires: CUDA, ninja, packaging
pip install flash-attn --no-build-isolation

# If build fails, the system will automatically fall back to SDPA (still fast)
# To skip FlashAttention-2, set USE_FLASH_ATTENTION=False in .env
```

### Step 5: Install Other Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Download Model (First Run Only)

The Qwen 2.5 7B model (~4.3GB) will automatically download from HuggingFace on first run:

```bash
# Pre-download to verify (optional)
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print('✅ Model cached')"
```

Model cache location: `~/.cache/huggingface/hub/`

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Database Configuration (SQLite for quick start)
DATABASE_URL=sqlite:////tmp/contributor_intelligence.db

# Activity Analysis
ACTIVITY_WINDOW_DAYS=90
WEEKS_IN_90_DAYS=13
MIN_HOURS_ACTIVE=1.0
TOP_PROJECTS_COUNT=5

# Embedded GPU LLM Configuration
EMBED_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBED_4BIT=1
MAX_TOKENS=320
TEMPERATURE=0.05
TOP_P=0.9
MAX_CONCURRENT_LLM=1  # Worker threads for batch collection

# GPU Optimization Parameters
INFER_CONCURRENCY=3           # Max concurrent GPU batches (semaphore)
MICRO_BATCH_SIZE=6            # Target prompts per batch
BATCH_LATENCY_MS=100          # Max wait time to collect batch (ms)
PREFILL_BATCH_TOKENS=4096     # Max input tokens per prefill batch
DECODE_CONCURRENCY=8          # Max concurrent decode operations
USE_FLASH_ATTENTION=True      # Enable FlashAttention-2 (auto-fallback to SDPA)
ENABLE_COMPILE=True           # Enable torch.compile optimization

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Application
APP_ENV=production
```

### GPU Tuning Guide

Adjust these parameters based on your GPU:

| GPU | VRAM | MICRO_BATCH_SIZE | INFER_CONCURRENCY | Expected Speed |
|-----|------|------------------|-------------------|----------------|
| T4 | 16GB | 4 | 2 | 1-2 profiles/sec |
| L4 | 24GB | 6-8 | 3 | 2-4 profiles/sec |
| V100 | 16GB | 6 | 3 | 2-3 profiles/sec |
| A10G | 24GB | 8-12 | 4 | 3-5 profiles/sec |
| A100 | 40GB | 16-32 | 6 | 5-10 profiles/sec |

**Tuning tips:**
- Increase `MICRO_BATCH_SIZE` until VRAM is 80-90% utilized
- Increase `BATCH_LATENCY_MS` if batches aren't filling up
- Reduce `MAX_CONCURRENT_LLM` to `1` for maximum batch efficiency
- Monitor with `nvidia-smi` during extraction

### Configuration Reference

See `src/config.py` for complete configuration options and defaults.

---

## Usage

### Starting the Application

```bash
# Verify GPU is available
nvidia-smi

# Start Streamlit application (model loads automatically on first use)
streamlit run app.py
```

Access at: **http://localhost:8501**

**First run:** Model will download (~4.3GB) and load (~30 seconds). Subsequent runs are instant.

### Workflow

#### 1. **Upload & Process** (`📤` page)
- Upload `InputWithEmail.csv` file
- Click "🚀 Process Data"
- System validates, parses, and calculates activity metrics
- Profiles saved to PostgreSQL

#### 2. **Extract Intelligence** (`🧠` page)
- Click "🚀 Extract Intelligence for All"
- AI generates 140-170 word summaries
- Summaries include skills, activity, compliance
- Stored in database for instant retrieval

#### 3. **View Profiles** (`👤` page)
- Select contributor by email
- View activity summary, top 5 projects, compliance
- Read AI-generated intelligence summary
- Export individual profiles

#### 4. **Search** (`🔍` page)
- Search contributors by email pattern
- Filter by activity status
- Quick access to intelligence summaries

---

## Data Model

### Input CSV Schema

```csv
contributor_email,contributor_id,currently_residing_country__c,...
user@example.com,003Hs00004cggv0IAA,United States,...
```

**Required Columns:**
- `contributor_email` (PRIMARY KEY)
- `contributor_id`
- `currently_residing_country__c`
- `risk_tier__c`, `kyc_status__c`, `dots_status__c`
- `languages_known` (JSON array)
- `qualifications_summary` (JSON array with passed tests)
- `projects_info` (JSON array with engagement data)

### Output JSON Structure

```json
{
  "contributor_email": "user@example.com",
  "contributor_id": "003Hs00004cggv0IAA",
  "activity_summary": {
    "is_active_90d": true,
    "weekly_hours_avg": 12.5,
    "total_projects_worked": 6,
    "projects_active_90d": 4
  },
  "top_5_projects": [...],
  "languages": [...],
  "compliance": {...},
  "intelligence_summary": "..."
}
```

---

## API Reference

### Key Modules

#### `src/processors/data_processor.py`
```python
process_contributor(row: pd.Series, settings: Settings) -> ContributorProfile
```
Processes single contributor from CSV row.

#### `src/intelligence/skill_extractor.py`
```python
generate_intelligence_summary(profile: ContributorProfile, llm_client: OllamaClient) -> str
```
Generates AI-powered intelligence summary.

#### `src/database/repositories.py`
```python
upsert_contributor(profile: ContributorProfile) -> bool
get_by_email(email: str) -> Optional[dict]
```
Database CRUD operations.

---

## Troubleshooting

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -U postgres -d contributor_intelligence -c "SELECT 1"

# Check if PostgreSQL is running
pg_isready -U postgres

# Restart PostgreSQL (macOS)
brew services restart postgresql
```

### Ollama Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama service
pkill ollama
ollama serve

# Verify model is downloaded
ollama list
```

### Common Errors

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Database does not exist` | Run `createdb contributor_intelligence` |
| `Connection refused (Ollama)` | Start Ollama: `ollama serve` |
| `Out of memory` | Use smaller model: `qwen2.5:3b` or `llama3.2:3b` |

---

## Performance

### Benchmarks

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| CSV Processing | ~10 contributors/sec | Single-threaded |
| Intelligence Extraction | ~5 sec/contributor | Depends on Ollama |
| Database Queries | <10ms | With proper indexing |
| Profile Lookup | <5ms | Email-based |

### Optimization Tips

1. **Batch Processing** - Process CSV files in batches of 100-500 rows
2. **Database Indexing** - Ensure indexes on email, activity status, country
3. **Connection Pooling** - Configure pool size based on concurrent users
4. **Model Selection** - Use `qwen2.5:3b` for faster extraction (lower RAM)

---

## Project Structure

```
OGs/
├── src/
│   ├── config.py              # Configuration management (Pydantic)
│   ├── models.py              # Data models (Pydantic)
│   ├── database/
│   │   ├── connection.py      # PostgreSQL connection pool
│   │   ├── repositories.py    # Data access layer
│   │   └── migrations.py      # Schema management
│   ├── processors/
│   │   ├── activity_analyzer.py   # 90-day metrics calculation
│   │   └── data_processor.py      # CSV processing pipeline
│   └── intelligence/
│       ├── llm_client.py      # Ollama LLM wrapper
│       └── skill_extractor.py # Skill extraction engine
├── data/
│   └── InputWithEmail.csv     # Input data file
├── logs/
│   └── app.log                # Application logs
├── app.py                     # Streamlit UI application
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## License

**Proprietary Software** - Internal use only.

---

## Support & Contribution

For technical support or feature requests, please contact the development team.

**Maintainer**: Engineering Team
**Version**: 2.0.0
**Last Updated**: 2025
