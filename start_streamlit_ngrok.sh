#!/bin/bash
# Complete Streamlit + Ngrok Setup for SageMaker Studio
# Usage: bash start_streamlit_ngrok.sh [YOUR_NGROK_TOKEN]

set -e

NGROK_TOKEN="${1:-}"

echo "=========================================="
echo "🚀 Starting Streamlit with Public URL"
echo "=========================================="

# Check if ngrok token provided
if [ -z "$NGROK_TOKEN" ]; then
    echo ""
    echo "❌ Error: Ngrok token required!"
    echo ""
    echo "Usage: bash start_streamlit_ngrok.sh YOUR_NGROK_TOKEN"
    echo ""
    echo "Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo ""
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q streamlit pyngrok psycopg2-binary pandas pydantic pydantic-settings langchain-ollama python-dotenv 2>/dev/null || echo "Some dependencies already installed"

# Install Ollama if needed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Stop existing processes
echo "🔄 Cleaning up existing processes..."
pkill ollama 2>/dev/null || true
pkill streamlit 2>/dev/null || true
pkill ngrok 2>/dev/null || true
sleep 2

# Start Ollama
echo "🚀 Starting Ollama..."
OLLAMA_NUM_PARALLEL=10 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "   Ollama PID: $OLLAMA_PID"
sleep 5

# Check if model exists, download if not
if ! ollama list | grep -q "qwen2.5:7b-instruct-q4_0"; then
    echo "📥 Downloading Qwen model (one-time, ~4GB, 5-10 min)..."
    ollama pull qwen2.5:7b-instruct-q4_0
fi
echo "✅ Model ready"

# Setup environment for Streamlit app
echo "⚙️  Configuring environment..."
cat > .env << 'EOF'
# Using SQLite for simplicity (can change to PostgreSQL)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=contributor_intelligence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Ollama Configuration
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
OLLAMA_BASE_URL=http://localhost:11434
EOF

# Create SQLite-compatible database connection module
echo "🔧 Setting up SQLite database..."
python3 << 'PYTHON_EOF'
import sqlite3
import os

# Create SQLite database
db_path = '/tmp/contributor_intelligence.db'
conn = sqlite3.connect(db_path)
conn.execute('''
    CREATE TABLE IF NOT EXISTS contributors (
        email TEXT PRIMARY KEY,
        contributor_id TEXT UNIQUE NOT NULL,
        processed_data TEXT NOT NULL,
        intelligence_summary TEXT,
        processing_status TEXT DEFAULT 'pending',
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        intelligence_extracted_at TIMESTAMP
    )
''')
conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON contributors(processing_status)')
conn.commit()
conn.close()
print(f"✅ SQLite database created: {db_path}")

# Update .env to use SQLite
with open('.env', 'a') as f:
    f.write(f'\n# SQLite Database\n')
    f.write(f'SQLITE_DB_PATH={db_path}\n')
PYTHON_EOF

# Configure ngrok
echo "🌐 Setting up ngrok..."
pip install -q pyngrok
python3 << PYTHON_EOF
from pyngrok import ngrok, conf
import sys

# Set auth token
try:
    conf.get_default().auth_token = "$NGROK_TOKEN"
    print("✅ Ngrok configured")
except Exception as e:
    print(f"❌ Ngrok error: {e}")
    sys.exit(1)
PYTHON_EOF

# Start Streamlit in background
echo "🎨 Starting Streamlit app..."
streamlit run app.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "   Streamlit PID: $STREAMLIT_PID"
sleep 8

# Start ngrok tunnel
echo "🌍 Creating public URL tunnel..."
python3 << 'PYTHON_EOF'
from pyngrok import ngrok
import time

# Create tunnel to Streamlit
public_url = ngrok.connect(8501, bind_tls=True)
print(f"\n{'='*60}")
print(f"✅ Streamlit is now publicly accessible!")
print(f"{'='*60}")
print(f"\n🌐 Public URL: {public_url}")
print(f"\n{'='*60}")
print(f"\n📊 Access your Streamlit app at the URL above")
print(f"🔄 The tunnel will stay active - keep this terminal open")
print(f"\n📝 Logs:")
print(f"   Streamlit: tail -f /tmp/streamlit.log")
print(f"   Ollama: tail -f /tmp/ollama.log")
print(f"\n⏹️  To stop:")
print(f"   Press Ctrl+C or run: pkill streamlit && pkill ollama && pkill ngrok")
print(f"\n{'='*60}\n")

# Keep tunnel alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    ngrok.disconnect(public_url)
    print("✅ Tunnel closed")
PYTHON_EOF

# Cleanup on exit
echo ""
echo "🧹 Cleaning up..."
pkill streamlit 2>/dev/null || true
pkill ngrok 2>/dev/null || true
echo "✅ Done"

