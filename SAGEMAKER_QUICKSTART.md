# SageMaker Studio Quick Start (5 Minutes)

Get the Contributor Intelligence Platform running in SageMaker Studio with SQLite database.

## Step 1: Upload Files (1 minute)

Upload to SageMaker Studio:
```
✓ sagemaker_script.py
✓ setup_sagemaker.sh  
✓ your_contributor_data.csv
```

## Step 2: Run Setup (3 minutes)

Open a terminal in SageMaker Studio and run:

```bash
bash setup_sagemaker.sh
```

This will:
- ✅ Install Ollama
- ✅ Start Ollama server with 10 parallel workers
- ✅ Download Qwen 2.5 7B model (~4GB)
- ✅ Install Python dependencies

## Step 3: Update Configuration (30 seconds)

Edit `sagemaker_script.py` line 34:

```python
CSV_FILE_PATH = '/home/sagemaker-user/your_actual_file.csv'
```

## Step 4: Run Processing (Varies by dataset size)

```bash
python sagemaker_script.py
```

**Processing Times:**
- 100 profiles: ~4 minutes (CPU) or ~1 minute (GPU)
- 1,000 profiles: ~40 minutes (CPU) or ~10 minutes (GPU)
- 3,000 profiles: ~2 hours (CPU) or ~30 minutes (GPU)

## Step 5: Get Results

Results are saved to:
```
/tmp/contributor_intelligence_results.csv
```

Download it or copy to S3:
```bash
aws s3 cp /tmp/contributor_intelligence_results.csv s3://your-bucket/
```

## That's It!

You now have:
- ✅ SQLite database with all contributors
- ✅ AI-generated intelligence summaries
- ✅ Extracted skills for each contributor
- ✅ CSV export ready to use

---

## Optional: Speed Up with GPU

If you want faster processing:

1. **Change SageMaker instance to GPU:**
   - Go to File → Shut Down → Change instance type
   - Select `ml.g4dn.xlarge` (T4 GPU - **Recommended**)
   - Or `ml.g5.xlarge` (A10G GPU - Faster but pricier)

2. **Verify GPU is working:**
   ```bash
   nvidia-smi
   ```

3. **Re-run the script:**
   ```bash
   python sagemaker_script.py
   ```

**Speed Improvement:** 5-10x faster than CPU!

---

## Troubleshooting

### Problem: "Connection refused"

**Solution:**
```bash
pkill ollama
OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
```

### Problem: "CSV file not found"

**Solution:** Check the path is correct:
```bash
ls -la /home/sagemaker-user/*.csv
```

Update `CSV_FILE_PATH` in the script with the actual path.

### Problem: Processing too slow

**Solutions:**
1. Use GPU instance (ml.g4dn.xlarge)
2. Use smaller model: `ollama pull qwen2.5:3b-instruct-q4_0`
3. Reduce batch size in script: `MAX_CONCURRENT = 5`

---

## What's Next?

### For Small Datasets (< 10K)
Continue using SQLite - it's perfect!

### For Large Datasets (> 10K)
Switch to RDS PostgreSQL:
1. Create RDS instance (see `SAGEMAKER_SETUP.md`)
2. Modify script to use PostgreSQL instead of SQLite
3. Use original `app.py` code

### Deploy to Production
1. Deploy Qwen to SageMaker Endpoint
2. Create API with Lambda
3. Add Streamlit UI
4. Schedule with EventBridge

---

## Cost Calculator

**Compute:**
- CPU (ml.t3.xlarge): $0.23/hour
- GPU (ml.g4dn.xlarge): $0.60/hour

**Example: 3,000 profiles**
- CPU: 2 hours × $0.23 = **$0.46**
- GPU: 0.5 hours × $0.60 = **$0.30**

💡 **GPU is actually cheaper for this workload!**

---

## Need Help?

Check logs:
```bash
# Application output (already visible)
# Ollama logs:
tail -f /tmp/ollama.log

# Database status:
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors;"
```

Happy processing! 🚀

