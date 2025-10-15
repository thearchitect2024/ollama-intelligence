# SageMaker Studio Setup Guide

This guide will help you run the Contributor Intelligence Platform in Amazon SageMaker Studio with a file-based SQLite database.

## Prerequisites

- Amazon SageMaker Studio account
- CSV file with contributor data
- At least 8GB RAM instance (ml.t3.xlarge or better)
- For GPU: ml.g4dn.xlarge or ml.g5.xlarge instance

## Option 1: Quick Start (No Jupyter Notebook Needed)

### 1. Upload Files to SageMaker

Upload these files to your SageMaker Studio home directory:
- `sagemaker_script.py`
- Your CSV file (e.g., `contributor_data.csv`)

### 2. Install Ollama

```bash
# In SageMaker Studio terminal
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama with parallel processing
OLLAMA_NUM_PARALLEL=10 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > /tmp/ollama.log 2>&1 &

# Wait a few seconds, then pull the model
sleep 5
ollama pull qwen2.5:7b-instruct-q4_0
```

### 3. Update Configuration

Edit `sagemaker_script.py` and update the CSV path:

```python
CSV_FILE_PATH = '/home/sagemaker-user/contributor_data.csv'  # Your actual path
```

### 4. Run the Script

```bash
python sagemaker_script.py
```

The script will:
1. Create SQLite database at `/tmp/contributor_intelligence.db`
2. Process your CSV file
3. Extract intelligence using Ollama
4. Export results to `/tmp/contributor_intelligence_results.csv`

## Option 2: Using Jupyter Notebook

### 1. Open `sagemaker_notebook.ipynb`

Upload and open the notebook in SageMaker Studio.

### 2. Update Configuration

In cell 2, update:
```python
CSV_FILE_PATH = '/home/sagemaker-user/your_file.csv'
```

### 3. Run All Cells

Execute all cells in sequence. The notebook will guide you through:
- Installing dependencies
- Setting up Ollama
- Processing data
- Extracting intelligence
- Viewing results

## Database Options

### SQLite (Default - File-Based)

**Pros:**
- No setup required
- Fast for < 10K records
- Perfect for testing

**Cons:**
- Not suitable for > 100K records
- No concurrent writes
- File-based (not persistent if instance stops)

**When to use:** Quick testing, small datasets, proof of concept

### Amazon RDS PostgreSQL (Production)

For production workloads, use RDS:

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier contributor-intelligence-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --master-user-password YourPassword123! \
    --allocated-storage 20 \
    --publicly-accessible false
```

Then modify the code to use PostgreSQL instead of SQLite.

## Using GPU for Faster Processing

### Check GPU Availability

```bash
nvidia-smi
```

If you see GPU information, Ollama will automatically use it!

### GPU-Optimized Instances

| Instance Type | vCPU | RAM | GPU | Speed Estimate |
|--------------|------|-----|-----|----------------|
| ml.t3.xlarge | 4 | 16GB | No | ~0.4 profiles/min |
| ml.g4dn.xlarge | 4 | 16GB | T4 (16GB) | ~2-3 profiles/min |
| ml.g5.xlarge | 4 | 16GB | A10G (24GB) | ~5-7 profiles/min |

### Recommendation

For 3,200 profiles:
- **CPU (ml.t3.xlarge)**: ~2 hours, **Cost: ~$0.50**
- **GPU (ml.g4dn.xlarge)**: ~30 minutes, **Cost: ~$0.30**
- **GPU (ml.g5.xlarge)**: ~15 minutes, **Cost: ~$0.50**

💡 **Best Value:** `ml.g4dn.xlarge` (T4 GPU) - Faster and cheaper!

## Performance Tuning

### 1. Increase Parallel Requests

Edit the script:
```python
MAX_CONCURRENT = 20  # Default: 10
```

Then restart Ollama with more parallelism:
```bash
pkill ollama
OLLAMA_NUM_PARALLEL=20 ollama serve > /tmp/ollama.log 2>&1 &
```

### 2. Use Smaller Model (Faster)

```bash
ollama pull qwen2.5:3b-instruct-q4_0  # Half the size, ~2x faster
```

Update script:
```python
OLLAMA_MODEL = 'qwen2.5:3b-instruct-q4_0'
```

### 3. Batch Processing

Process large CSVs in multiple runs:
```python
# Process first 1000 rows
df = pd.read_csv(CSV_FILE_PATH, nrows=1000)
```

## Monitoring Progress

### Check Ollama Logs

```bash
tail -f /tmp/ollama.log
```

### Check Database Stats

```bash
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors;"
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors WHERE intelligence_summary IS NOT NULL;"
```

### Monitor GPU Usage (if available)

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Ollama Connection Refused

```bash
# Check if Ollama is running
ps aux | grep ollama

# Restart Ollama
pkill ollama
OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &
```

### Out of Memory

Use smaller batch size:
```python
CHUNK_SIZE = 100  # Default: 500
MAX_CONCURRENT = 5  # Default: 10
```

### Slow Processing

1. Check if GPU is being used: `nvidia-smi`
2. Increase `OLLAMA_NUM_PARALLEL`
3. Use smaller model (3B instead of 7B)
4. Use GPU instance type

### CSV Column Mismatch

The script expects these columns:
- `email` (required)
- `contributor_id`
- `projects_info` or `projects_json` (JSON format)
- `languages_known` or `languages_json` (JSON format)

If your CSV has different columns, modify the `process_contributor_row()` function.

## Cost Estimation

### Compute Costs (us-east-1, as of 2025)

| Instance Type | Hourly Rate | 3K profiles | 30K profiles |
|--------------|-------------|-------------|--------------|
| ml.t3.xlarge (CPU) | $0.23/hr | $0.50 | $5.00 |
| ml.g4dn.xlarge (GPU) | $0.60/hr | $0.30 | $3.00 |
| ml.g5.xlarge (GPU) | $1.20/hr | $0.30 | $3.00 |

### Storage Costs

- SQLite database: Free (local instance storage)
- RDS PostgreSQL: ~$15/month (db.t3.micro)

## Exporting Results

Results are automatically exported to:
```
/tmp/contributor_intelligence_results.csv
```

Download from SageMaker Studio:
1. Right-click the file in file browser
2. Select "Download"

Or use AWS CLI:
```bash
aws s3 cp /tmp/contributor_intelligence_results.csv s3://your-bucket/results/
```

## Next Steps

1. **Scale Up:** Use RDS PostgreSQL for production
2. **Deploy Model:** Deploy Qwen to SageMaker Endpoint for always-on inference
3. **Add UI:** Create Streamlit app for querying results
4. **Schedule:** Use AWS Lambda + EventBridge for scheduled processing

## Support

For issues or questions:
1. Check `/tmp/ollama.log` for LLM errors
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Test with small dataset first (100 rows)

