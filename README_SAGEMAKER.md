# SageMaker Studio Files

This directory contains everything you need to run the Contributor Intelligence Platform in Amazon SageMaker Studio with a file-based SQLite database.

## 📁 Files Created

| File | Purpose |
|------|---------|
| `sagemaker_script.py` | **Main script** - Standalone Python script for processing |
| `sagemaker_notebook.ipynb` | **Jupyter notebook** - Interactive cell-by-cell processing |
| `setup_sagemaker.sh` | **Setup script** - Automated Ollama & dependency installation |
| `SAGEMAKER_QUICKSTART.md` | **Quick start** - Get running in 5 minutes |
| `SAGEMAKER_SETUP.md` | **Full guide** - Detailed setup and troubleshooting |
| `README_SAGEMAKER.md` | **This file** - Overview and file descriptions |

## 🚀 Quick Start

**For the impatient (5 minutes):**
```bash
# 1. Upload files to SageMaker Studio
# 2. Run setup
bash setup_sagemaker.sh

# 3. Edit CSV path in sagemaker_script.py (line 34)
# 4. Run it
python sagemaker_script.py
```

📖 **Full instructions:** See `SAGEMAKER_QUICKSTART.md`

## 🎯 What This Does

1. **Processes CSV data** → Parses contributor information
2. **Stores in SQLite** → Local file-based database (no RDS needed)
3. **Extracts intelligence** → Uses Ollama (Qwen 2.5 7B) to generate summaries
4. **Identifies skills** → AI extracts 3-5 key skills per contributor
5. **Exports results** → CSV file with all intelligence summaries

## 🔄 Choose Your Workflow

### Option A: Python Script (Recommended)

**Best for:** Running once, large batches, automation

```bash
python sagemaker_script.py
```

**Pros:**
- ✅ Fully automated
- ✅ Progress tracking
- ✅ Error handling
- ✅ Easy to schedule

### Option B: Jupyter Notebook

**Best for:** Exploration, step-by-step, learning

1. Open `sagemaker_notebook.ipynb`
2. Run cells one by one
3. Inspect results as you go

**Pros:**
- ✅ Interactive
- ✅ See intermediate results
- ✅ Easy to modify
- ✅ Great for debugging

## 💾 Database: SQLite vs PostgreSQL

### SQLite (This Implementation)

**Pros:**
- ✅ No setup required
- ✅ Fast for < 10K records
- ✅ Perfect for testing
- ✅ No cost

**Cons:**
- ❌ Not for > 100K records
- ❌ No concurrent writes
- ❌ Lost if instance stops (use /tmp)

**When to use:** 
- Testing
- Small datasets (< 10K)
- Proof of concept
- One-time processing

### PostgreSQL (Production)

**Pros:**
- ✅ Scales to millions of records
- ✅ Concurrent access
- ✅ Persistent storage
- ✅ Production-ready

**Cons:**
- ❌ Requires RDS setup
- ❌ Additional cost (~$15/month)
- ❌ More complex

**When to use:**
- Production workloads
- Large datasets (> 10K)
- Multiple users
- Long-term storage

## 🖥️ CPU vs GPU

### Performance Comparison (3,200 profiles)

| Hardware | Instance | Speed | Time | Cost |
|----------|----------|-------|------|------|
| **CPU** | ml.t3.xlarge | 0.4/min | 2h | $0.46 |
| **GPU (T4)** | ml.g4dn.xlarge | 2-3/min | 30m | $0.30 |
| **GPU (A10G)** | ml.g5.xlarge | 5-7/min | 15m | $0.30 |

💡 **Recommendation:** Use `ml.g4dn.xlarge` (T4 GPU) - Faster AND cheaper!

### How to Enable GPU

1. File → Shut Down
2. Change instance type → `ml.g4dn.xlarge`
3. Restart
4. Run `nvidia-smi` to verify
5. Run your script - Ollama auto-detects GPU!

## 📊 Expected Processing Times

| Profiles | CPU (ml.t3.xlarge) | GPU (ml.g4dn.xlarge) |
|----------|-------------------|---------------------|
| 100 | 4 minutes | 1 minute |
| 1,000 | 40 minutes | 10 minutes |
| 3,000 | 2 hours | 30 minutes |
| 10,000 | 7 hours | 90 minutes |
| 30,000 | 21 hours | 4.5 hours |

*Times are estimates. Actual speed varies based on project descriptions length.*

## 🔧 Configuration Options

Edit these variables in `sagemaker_script.py`:

```python
# Database location
SQLITE_DB_PATH = '/tmp/contributor_intelligence.db'

# Your CSV file (UPDATE THIS!)
CSV_FILE_PATH = '/home/sagemaker-user/contributor_data.csv'

# LLM model (3b = faster, 7b = better quality)
OLLAMA_MODEL = 'qwen2.5:7b-instruct-q4_0'

# Parallel LLM requests (higher = faster, but more RAM)
MAX_CONCURRENT = 10

# CSV chunk size (lower = less RAM)
CHUNK_SIZE = 500
```

## 📤 Output Files

### Database
```
/tmp/contributor_intelligence.db  # SQLite database with all data
```

### CSV Export
```
/tmp/contributor_intelligence_results.csv  # Final results
```

Columns:
- `email` - Contributor email
- `contributor_id` - Unique ID
- `intelligence_summary` - AI-generated summary (150 words)
- `skills` - Extracted skills (comma-separated)
- `total_projects` - Number of projects
- `total_hours` - Total hours worked

## 🐛 Troubleshooting

### "Connection refused" error

Ollama not running. Restart it:
```bash
pkill ollama
OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &
```

### "CSV file not found"

Check path:
```bash
ls -la /home/sagemaker-user/*.csv
```

Update `CSV_FILE_PATH` in script.

### Out of memory

Reduce batch size:
```python
MAX_CONCURRENT = 5
CHUNK_SIZE = 100
```

### Too slow

1. ✅ Use GPU instance (ml.g4dn.xlarge)
2. ✅ Use smaller model: `qwen2.5:3b-instruct-q4_0`
3. ✅ Increase `OLLAMA_NUM_PARALLEL` to 20
4. ✅ Increase `MAX_CONCURRENT` to 20

### Check logs

```bash
# Ollama logs
tail -f /tmp/ollama.log

# Database stats
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors;"
```

## 🔄 Differences from Local Version

| Feature | Local (app.py) | SageMaker (sagemaker_script.py) |
|---------|----------------|--------------------------------|
| **Database** | PostgreSQL | SQLite |
| **UI** | Streamlit | Command-line / Notebook |
| **Setup** | Manual DB setup | Auto SQLite creation |
| **Dependencies** | requirements.txt | Auto-install in script |
| **LLM** | Ollama (manual) | Ollama (auto-setup) |
| **Output** | Web UI | CSV export |

## 📈 Scaling Up

### For Large Datasets (> 10K profiles)

1. **Switch to RDS PostgreSQL**
   - See `SAGEMAKER_SETUP.md` for RDS setup
   - Modify script to use original `DatabaseManager`
   - Use original `app.py` logic

2. **Deploy Model to SageMaker Endpoint**
   - Persistent model serving
   - Auto-scaling
   - No cold start

3. **Use EMR for huge datasets (> 100K)**
   - Distributed processing
   - Spark-based
   - Process millions of records

## 💰 Cost Optimization

### For Small Jobs (< 1K profiles)
- Use CPU (ml.t3.xlarge)
- Stop instance immediately after
- **Total cost: < $0.50**

### For Medium Jobs (1K-10K profiles)
- Use GPU (ml.g4dn.xlarge)
- Use Spot instances (70% cheaper)
- **Total cost: $1-5**

### For Large Jobs (> 10K profiles)
- Use GPU (ml.g5.xlarge or ml.g5.2xlarge)
- Process overnight with Spot
- Consider SageMaker Processing Jobs
- **Total cost: $5-50**

## 🎓 Learning Resources

**SQLite:**
- Official docs: https://www.sqlite.org/docs.html
- Python sqlite3: https://docs.python.org/3/library/sqlite3.html

**Ollama:**
- Ollama docs: https://ollama.com/docs
- Model library: https://ollama.com/library

**SageMaker:**
- SageMaker Studio: https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html
- Pricing: https://aws.amazon.com/sagemaker/pricing/

## 🆘 Support

1. **Check the guides:**
   - `SAGEMAKER_QUICKSTART.md` - Quick start
   - `SAGEMAKER_SETUP.md` - Detailed setup

2. **Check logs:**
   ```bash
   tail -f /tmp/ollama.log
   ```

3. **Test with small dataset:**
   ```python
   # In script, add:
   chunk_df = chunk_df.head(10)  # Only 10 rows
   ```

4. **Enable debug logging:**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

## ✅ Checklist

Before running, make sure:

- [ ] Uploaded `sagemaker_script.py` to SageMaker
- [ ] Uploaded your CSV file
- [ ] Ran `setup_sagemaker.sh`
- [ ] Updated `CSV_FILE_PATH` in script
- [ ] Ollama is running (`ps aux | grep ollama`)
- [ ] Model is downloaded (`ollama list`)
- [ ] Enough disk space (`df -h /tmp`)

Then run:
```bash
python sagemaker_script.py
```

## 🎉 Success Indicators

You'll know it's working when you see:

```
✅ Configuration set
✅ SQLite database initialized
✅ CSV Import Complete!
⚡ Progress: 100/3200 | Speed: 25.3 profiles/min | ETA: 122.3 min
✅ Extraction Complete!
✅ Results exported to: /tmp/contributor_intelligence_results.csv
```

Good luck! 🚀

