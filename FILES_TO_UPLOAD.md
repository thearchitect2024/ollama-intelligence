# Files to Upload to SageMaker Studio

## ✅ Complete Checklist

### Required Files (Upload These)

1. **Your CSV file** ⭐ REQUIRED
   - Example: `contributor_data.csv`
   - Your actual contributor data
   - Can be any name, just update the path in the script

### Choose ONE of these options:

#### Option A: Standalone Python Script (Recommended)
2. `sagemaker_script.py` ⭐ MAIN FILE
   - Complete processing pipeline
   - No notebook needed
   - Run: `python sagemaker_script.py`

#### Option B: Jupyter Notebook (Interactive)
2. `sagemaker_notebook.ipynb` ⭐ MAIN FILE
   - Step-by-step processing
   - Cell-by-cell execution
   - Good for learning/debugging

### Helper Files (Highly Recommended)

3. `setup_sagemaker.sh`
   - Automated setup script
   - Installs Ollama + dependencies
   - Run: `bash setup_sagemaker.sh`

4. `test_sagemaker_setup.py`
   - Verify setup is working
   - Checks all dependencies
   - Run: `python test_sagemaker_setup.py`

### Documentation (Optional but Helpful)

5. `SAGEMAKER_QUICKSTART.md` - Quick start guide (5 min)
6. `README_SAGEMAKER.md` - Complete overview
7. `SAGEMAKER_SETUP.md` - Detailed reference
8. `SAGEMAKER_FILES_SUMMARY.txt` - Comprehensive summary

---

## 📤 Minimum Upload (Quick Test)

If you just want to test quickly:

1. Your CSV file
2. `sagemaker_script.py`
3. `setup_sagemaker.sh`

Then run:
```bash
bash setup_sagemaker.sh
# Edit CSV_FILE_PATH in sagemaker_script.py
python sagemaker_script.py
```

---

## 📤 Recommended Upload (Best Experience)

Upload all files for the complete experience:

```
your_data.csv                    ← Your CSV data
sagemaker_script.py              ← Main script
setup_sagemaker.sh               ← Setup automation
test_sagemaker_setup.py          ← Verification
SAGEMAKER_QUICKSTART.md          ← Quick guide
README_SAGEMAKER.md              ← Full guide
SAGEMAKER_SETUP.md               ← Reference
SAGEMAKER_FILES_SUMMARY.txt      ← Summary
```

---

## 📤 How to Upload to SageMaker Studio

### Method 1: Drag and Drop
1. Open SageMaker Studio
2. Open File Browser (left sidebar)
3. Drag and drop files from your computer

### Method 2: Upload Button
1. Open SageMaker Studio
2. Click "Upload Files" button (top of file browser)
3. Select files to upload

### Method 3: AWS S3
```bash
# On your local machine
aws s3 cp . s3://your-bucket/sagemaker-files/ --recursive

# In SageMaker Studio terminal
aws s3 cp s3://your-bucket/sagemaker-files/ . --recursive
```

### Method 4: Git Clone
```bash
# In SageMaker Studio terminal
git clone your-repo-url
cd your-repo
```

---

## ✅ After Upload

### Verify Files Are Present

```bash
# In SageMaker terminal
ls -lh
```

You should see:
- Your CSV file
- `sagemaker_script.py` or `sagemaker_notebook.ipynb`
- `setup_sagemaker.sh`
- `test_sagemaker_setup.py`

### Run Setup

```bash
bash setup_sagemaker.sh
```

### Test Setup

```bash
python test_sagemaker_setup.py
```

Should show: **"All tests passed!"**

### Start Processing

**For Script:**
```bash
# Edit CSV path in sagemaker_script.py first!
python sagemaker_script.py
```

**For Notebook:**
- Open `sagemaker_notebook.ipynb`
- Update CSV path in cell 4
- Run all cells

---

## 📥 Output Files (What to Download After Processing)

After processing completes, download these files:

1. **`/tmp/contributor_intelligence_results.csv`** ⭐
   - Final results with intelligence summaries
   - This is what you need!

2. **`/tmp/contributor_intelligence.db`** (optional)
   - SQLite database with all data
   - Keep for future queries

### How to Download

**Method 1: Right-click**
- Right-click file in SageMaker file browser
- Select "Download"

**Method 2: Copy to S3**
```bash
aws s3 cp /tmp/contributor_intelligence_results.csv s3://your-bucket/results/
```

**Method 3: View in Notebook**
```python
import pandas as pd
df = pd.read_csv('/tmp/contributor_intelligence_results.csv')
df.head()  # Preview results
```

---

## 🎯 Quick Reference

| File | Size | Upload? | Purpose |
|------|------|---------|---------|
| **Your CSV** | Varies | ✅ YES | Input data |
| `sagemaker_script.py` | 18 KB | ✅ YES | Main script |
| `sagemaker_notebook.ipynb` | 2 KB | Optional | Interactive notebook |
| `setup_sagemaker.sh` | 2 KB | ✅ YES | Setup automation |
| `test_sagemaker_setup.py` | 8 KB | Recommended | Verification |
| `SAGEMAKER_QUICKSTART.md` | 3 KB | Optional | Quick guide |
| `README_SAGEMAKER.md` | 8 KB | Optional | Full guide |
| `SAGEMAKER_SETUP.md` | 6 KB | Optional | Reference |
| `SAGEMAKER_FILES_SUMMARY.txt` | 13 KB | Optional | Summary |

---

## 💡 Pro Tips

1. **Upload docs first** - Read the guides in SageMaker Studio
2. **Test with small data** - Try 100 rows first
3. **Use GPU** - Change to `ml.g4dn.xlarge` for 5-8x speedup
4. **Monitor progress** - Watch the terminal output
5. **Check logs** - `tail -f /tmp/ollama.log` if issues occur

---

## 🆘 Troubleshooting

**"File not found" error**
- Check file is uploaded
- Check spelling and path
- Use `ls -la` to list files

**"Permission denied"**
- Make scripts executable: `chmod +x *.sh *.py`

**"No such file or directory"**
- You're in wrong directory
- Use `pwd` to check current directory
- Use `cd /home/sagemaker-user` to go home

---

## ✅ Ready Checklist

Before running, confirm:

- [ ] All required files uploaded
- [ ] CSV file is present
- [ ] Ran `setup_sagemaker.sh`
- [ ] Ran `test_sagemaker_setup.py` (all tests passed)
- [ ] Updated CSV_FILE_PATH in script/notebook
- [ ] Know instance type (CPU vs GPU)
- [ ] Have enough disk space (5GB+)

**You're ready to go!** 🚀

