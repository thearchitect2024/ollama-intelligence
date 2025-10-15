# Running Streamlit App in SageMaker Studio with Public URL

## Quick Start (3 Steps)

### Step 1: Get Your Ngrok Token
1. Go to https://dashboard.ngrok.com/get-started/your-authtoken
2. Sign up/login (free)
3. Copy your auth token

### Step 2: Upload Files to SageMaker Studio
Upload these files:
- `app.py` (your Streamlit app)
- `start_streamlit_ngrok.sh` (startup script)
- All `src/` directory files
- `requirements.txt`
- Your CSV file (optional, for testing)

### Step 3: Run It!
```bash
bash start_streamlit_ngrok.sh YOUR_NGROK_TOKEN_HERE
```

**That's it!** You'll get a public URL like: `https://abc123.ngrok.io`

---

## What It Does

The script automatically:
1. ✅ Installs all dependencies (Streamlit, pyngrok, etc.)
2. ✅ Installs and starts Ollama server
3. ✅ Downloads Qwen 2.5 7B model
4. ✅ Creates SQLite database
5. ✅ Starts Streamlit app
6. ✅ Creates ngrok tunnel
7. ✅ Gives you a public URL

---

## Full Example

```bash
# In SageMaker Studio terminal:
cd /home/sagemaker-user

# Upload your files, then run:
bash start_streamlit_ngrok.sh 2abc...your_token...xyz

# Output will show:
# ============================================================
# ✅ Streamlit is now publicly accessible!
# ============================================================
#
# 🌐 Public URL: https://1234-5678-90ab.ngrok.io
#
# ============================================================
```

Visit that URL in your browser - your Streamlit app is now live!

---

## Features

### ✅ Public Access
- **Anyone** can access your app via the ngrok URL
- No VPN or SSH tunneling needed
- Works from any device/network

### ✅ Persistent Session
- Tunnel stays active as long as terminal is open
- Streamlit auto-reloads on code changes
- Database persists in `/tmp/contributor_intelligence.db`

### ✅ GPU Support
- If using ml.g4dn.xlarge or ml.g5.xlarge
- Ollama automatically uses GPU
- 5-10x faster LLM inference!

---

## Using the App

Once the public URL is live:

1. **Open URL** in your browser
2. **Upload CSV** - Click "Upload CSV" in sidebar
3. **Process Data** - CSV gets parsed and stored
4. **Extract Intelligence** - Click button to run LLM analysis
5. **View Profiles** - Browse contributor intelligence
6. **Export Results** - Download processed data

---

## Monitoring

### Check Logs

```bash
# Streamlit logs
tail -f /tmp/streamlit.log

# Ollama logs
tail -f /tmp/ollama.log

# Check processes
ps aux | grep -E "streamlit|ollama|ngrok"
```

### Check Database

```bash
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors;"
```

---

## Stopping the App

### Option 1: Ctrl+C
Press `Ctrl+C` in the terminal running the script

### Option 2: Kill Processes
```bash
pkill streamlit
pkill ollama
pkill ngrok
```

---

## Troubleshooting

### "Connection refused" from Ollama
```bash
# Restart Ollama
pkill ollama
OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
```

### Streamlit not accessible
```bash
# Check if it's running
ps aux | grep streamlit

# Check logs
tail -f /tmp/streamlit.log

# Restart
pkill streamlit
streamlit run app.py --server.port 8501 --server.headless true &
```

### Ngrok tunnel error
```bash
# Check token is valid
# Get new token from: https://dashboard.ngrok.com

# Restart script with correct token
bash start_streamlit_ngrok.sh YOUR_CORRECT_TOKEN
```

### Port already in use
```bash
# Kill everything and restart
pkill streamlit
pkill ollama
pkill ngrok
sleep 2

# Run script again
bash start_streamlit_ngrok.sh YOUR_TOKEN
```

---

## Performance

### CPU (ml.t3.xlarge)
- Setup time: 5-10 minutes (model download)
- LLM speed: ~0.4 profiles/min
- Cost: $0.23/hour

### GPU (ml.g4dn.xlarge) ⭐ Recommended
- Setup time: 5-10 minutes (model download)
- LLM speed: ~2-3 profiles/min (5-8x faster!)
- Cost: $0.60/hour

### GPU (ml.g5.xlarge) 
- Setup time: 5-10 minutes (model download)
- LLM speed: ~5-7 profiles/min (12-15x faster!)
- Cost: $1.20/hour

---

## Ngrok Free Tier Limits

- ✅ Public URL access
- ✅ HTTPS support
- ⚠️ Random URL each time (unless paid plan)
- ⚠️ 40 connections/min limit
- ⚠️ Session timeout after 2 hours idle

For production, upgrade to ngrok paid plan or deploy to AWS.

---

## Database Options

### Current: SQLite (File-Based)
```bash
Location: /tmp/contributor_intelligence.db
Best for: < 10K contributors
```

### Production: PostgreSQL RDS
For large datasets, modify the script to use RDS:
```bash
# In .env file:
POSTGRES_HOST=your-rds-endpoint.amazonaws.com
POSTGRES_DB=contributor_intelligence
POSTGRES_USER=admin
POSTGRES_PASSWORD=your-password
```

---

## Security Notes

⚠️ **Your app is publicly accessible!**

- Anyone with the URL can access it
- No authentication by default
- Don't expose sensitive data
- For production, add Streamlit authentication or use AWS private deployment

---

## Alternative: Streamlit Community Cloud

For easier public deployment:

1. Push code to GitHub
2. Deploy on https://streamlit.io/cloud
3. Free public hosting
4. No ngrok needed
5. But requires PostgreSQL (not SQLite)

---

## File Checklist

Before running, ensure you have:

- [ ] `app.py` - Main Streamlit app
- [ ] `start_streamlit_ngrok.sh` - Startup script
- [ ] `src/` directory - All Python modules
- [ ] `requirements.txt` - Dependencies
- [ ] `.env` file - Created automatically by script
- [ ] Ngrok token - From ngrok.com

---

## Complete Command Reference

```bash
# Start app with public URL
bash start_streamlit_ngrok.sh YOUR_TOKEN

# Check if services are running
ps aux | grep -E "streamlit|ollama"

# View Streamlit logs
tail -f /tmp/streamlit.log

# View Ollama logs
tail -f /tmp/ollama.log

# Stop everything
pkill streamlit && pkill ollama && pkill ngrok

# Restart just Streamlit
pkill streamlit
streamlit run app.py --server.port 8501 --server.headless true &

# Check database
sqlite3 /tmp/contributor_intelligence.db "SELECT COUNT(*) FROM contributors;"

# Export database
sqlite3 /tmp/contributor_intelligence.db ".dump" > backup.sql

# Clear database and restart
rm /tmp/contributor_intelligence.db
bash start_streamlit_ngrok.sh YOUR_TOKEN
```

---

## Success Indicators

You'll know it's working when you see:

```
✅ Streamlit is now publicly accessible!
🌐 Public URL: https://xxxx.ngrok.io
```

Then:
1. Open the URL in your browser
2. See the Streamlit app interface
3. Upload and process CSV files
4. Extract intelligence with LLM

---

## Next Steps After Setup

1. **Test with small dataset** - Try 100 rows first
2. **Monitor performance** - Check logs and speed
3. **Process full dataset** - Upload your complete CSV
4. **Export results** - Download processed data
5. **Deploy to production** - Use RDS + proper deployment

---

## Support

**Issues?**
1. Check logs: `tail -f /tmp/streamlit.log`
2. Verify services: `ps aux | grep streamlit`
3. Test Ollama: `curl http://localhost:11434/api/tags`
4. Restart everything and try again

**Ready to go!** 🚀

