#!/bin/bash
# SageMaker Studio Setup Script
# Run this in your SageMaker Studio terminal to set up the environment

set -e

echo "=========================================="
echo "Contributor Intelligence Platform Setup"
echo "=========================================="

# Install Ollama
echo ""
echo "📦 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Kill any existing Ollama processes
echo ""
echo "🔄 Stopping existing Ollama processes..."
pkill ollama || true
sleep 2

# Start Ollama with optimized settings
echo ""
echo "🚀 Starting Ollama server..."
OLLAMA_NUM_PARALLEL=10 OLLAMA_MAX_LOADED_MODELS=1 ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "✅ Ollama started (PID: $OLLAMA_PID)"

# Wait for Ollama to be ready
echo ""
echo "⏳ Waiting for Ollama to be ready..."
sleep 5

# Check if Ollama is running
if ! ps -p $OLLAMA_PID > /dev/null; then
    echo "❌ Ollama failed to start. Check /tmp/ollama.log"
    exit 1
fi

# Pull the model
echo ""
echo "📥 Pulling Qwen 2.5 7B model (this may take 5-10 minutes)..."
ollama pull qwen2.5:7b-instruct-q4_0
echo "✅ Model downloaded"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q pandas pydantic pydantic-settings python-dotenv tqdm aiohttp
echo "✅ Dependencies installed"

# Check system info
echo ""
echo "🖥️  System Information:"
echo "----------------------------------------"
echo "Python: $(python --version)"
echo "Ollama: $(ollama --version)"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU Detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo ""
    echo "💻 No GPU detected - using CPU"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your CSV file to SageMaker Studio"
echo "2. Update CSV_FILE_PATH in sagemaker_script.py"
echo "3. Run: python sagemaker_script.py"
echo ""
echo "Or use the Jupyter notebook: sagemaker_notebook.ipynb"
echo ""
echo "Monitor Ollama logs: tail -f /tmp/ollama.log"
echo ""

