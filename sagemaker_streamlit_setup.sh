#!/bin/bash
# SageMaker Studio - Streamlit with Ngrok Public URL Setup

set -e

echo "=========================================="
echo "Streamlit + Ngrok Setup for SageMaker"
echo "=========================================="

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q streamlit pyngrok psycopg2-binary pandas pydantic pydantic-settings langchain-ollama python-dotenv

# Install Ollama
echo ""
echo "📦 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Stop any existing processes
echo ""
echo "🔄 Stopping existing processes..."
pkill ollama || true
pkill streamlit || true
sleep 2

# Start Ollama
echo ""
echo "🚀 Starting Ollama server..."
OLLAMA_NUM_PARALLEL=10 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > /tmp/ollama.log 2>&1 &
echo "Ollama PID: $!"
sleep 5

# Download model
echo ""
echo "📥 Downloading Qwen model (this may take 5-10 minutes)..."
ollama pull qwen2.5:7b-instruct-q4_0
echo "✅ Model ready"

# Setup environment variables
echo ""
echo "⚙️  Setting up environment..."
cat > .env << 'EOF'
# PostgreSQL Configuration (using local SQLite instead)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=contributor_intelligence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Ollama Configuration
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
OLLAMA_BASE_URL=http://localhost:11434
EOF

echo "✅ Environment configured"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Get your ngrok auth token from: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "2. Run: bash start_streamlit_ngrok.sh YOUR_NGROK_TOKEN"
echo ""

