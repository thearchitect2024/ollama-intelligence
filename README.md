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
| PostgreSQL | 12+ | Primary database |
| Ollama | Latest | AI model inference |
| pip | 21.0+ | Package management |

### System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free space
- **CPU**: 2+ cores recommended
- **OS**: macOS, Linux, or Windows 10+

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

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup PostgreSQL

```bash
# Create database
createdb contributor_intelligence

# Or using psql
psql -U postgres
CREATE DATABASE contributor_intelligence;
\q
```

### Step 5: Install Ollama & Model

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull AI model (recommended)
ollama pull qwen2.5:7b-instruct-q4_0

# Alternative models for lower RAM:
# ollama pull qwen2.5:3b        # 3B params, ~2GB RAM
# ollama pull llama3.2:3b       # 3B params, ~2GB RAM

# Verify installation
ollama list
```

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=contributor_intelligence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Activity Analysis
ACTIVITY_WINDOW_DAYS=90
WEEKS_IN_90_DAYS=13
MIN_HOURS_ACTIVE=1.0
TOP_PROJECTS_COUNT=5

# LLM Configuration
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
OLLAMA_BASE_URL=http://localhost:11434
MAX_TOKENS=256
TEMPERATURE=0.3

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Application
APP_ENV=production
```

### Configuration Reference

See `src/config.py` for complete configuration options and defaults.

---

## Usage

### Starting the Application

```bash
# Ensure Ollama is running
ollama serve &

# Start Streamlit application
streamlit run app.py
```

Access at: **http://localhost:8501**

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
