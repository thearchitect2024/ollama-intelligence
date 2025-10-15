#!/bin/bash

echo "======================================"
echo "Contributor Intelligence Platform"
echo "Installation Script"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Ensure PostgreSQL is running"
echo "2. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
echo "3. Pull model: ollama pull qwen2.5:7b-instruct-q4_0"
echo "4. Start Ollama: ollama serve &"
echo "5. Run app: streamlit run app.py"
echo ""
