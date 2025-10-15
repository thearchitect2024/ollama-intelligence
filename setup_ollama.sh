#!/bin/bash
# Setup script for Ollama + LangChain on 8GB RAM systems

echo "=================================================="
echo "Ollama Setup for Contributor Intelligence"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "❌ Ollama is not installed"
    echo ""
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollama installed successfully"
    else
        echo "❌ Failed to install Ollama"
        echo "Please install manually from: https://ollama.com/download"
        exit 1
    fi
else
    echo "✅ Ollama is already installed"
fi

echo ""
echo "Checking Ollama service..."

# Check if Ollama is running
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "⚠️  Ollama service is not running"
    echo "Starting Ollama in background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "✅ Ollama service started"
    else
        echo "❌ Failed to start Ollama"
        echo "Please start manually: ollama serve"
        exit 1
    fi
fi

echo ""
echo "Pulling Qwen 2.5 7B Instruct Q4 model (recommended for 8GB RAM)..."
echo "This may take a few minutes on first run (~4.4GB download)..."

ollama pull qwen2.5:7b-instruct-q4_0

if [ $? -eq 0 ]; then
    echo "✅ Model downloaded successfully"
else
    echo "❌ Failed to download model"
    exit 1
fi

echo ""
echo "Verifying installation..."
ollama list

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Test the setup: python3 test_few_shot.py"
echo "2. Run the app: streamlit run app.py"
echo ""
echo "Alternative models for 8GB RAM:"
echo "  - qwen2.5:3b (faster, 2GB RAM)"
echo "  - llama3.2:3b (alternative, 2GB RAM)"
echo ""
echo "To change model, edit OLLAMA_MODEL in .env file"
echo ""

