#!/bin/bash
# setup_enhanced.sh
# Setup script for Janus Unified - Zero-API-key Voice + Text Interface

set -e

echo "════════════════════════════════════════════════════════════"
echo "  JANUS UNIFIED - Setup Script"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

source venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy pyaudio requests webrtcvad cryptography

# Optional dependencies
echo ""
echo "Installing optional dependencies..."
pip install twilio python-telegram-bot scipy 2>/dev/null || echo -e "${YELLOW}⚠ Some optional dependencies failed${NC}"

echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p models
mkdir -p discovered_tools
mkdir -p sync_checkpoints
mkdir -p /tmp/janus_voice
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Download Whisper model
echo "════════════════════════════════════════════════════════════"
echo "  WHISPER.CPP SETUP"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ ! -d "whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git
    echo -e "${GREEN}✓ Whisper.cpp cloned${NC}"
else
    echo -e "${YELLOW}⚠ whisper.cpp already exists${NC}"
fi

cd whisper.cpp

# Build
echo "Building whisper.cpp..."
if [ "$OS" == "macos" ]; then
    make
elif [ "$OS" == "linux" ]; then
    make
else
    echo -e "${YELLOW}⚠ Please build whisper.cpp manually for Windows${NC}"
fi

# Download model
echo ""
echo "Downloading Whisper model (base.en)..."
if [ ! -f "models/ggml-base.en.bin" ]; then
    bash models/download-ggml-model.sh base.en
    echo -e "${GREEN}✓ Model downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Model already exists${NC}"
fi

cd ..

# Symlink model
echo "Creating model symlink..."
ln -sf whisper.cpp/models/ggml-base.en.bin models/ggml-base.en.bin 2>/dev/null || true
echo ""

# Piper TTS setup
echo "════════════════════════════════════════════════════════════"
echo "  PIPER TTS SETUP"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ "$OS" == "linux" ]; then
    echo "Downloading Piper for Linux..."
    if [ ! -d "piper" ]; then
        mkdir -p piper
        cd piper
        wget -q https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz
        tar -xzf piper_amd64.tar.gz
        rm piper_amd64.tar.gz
        cd ..
        echo -e "${GREEN}✓ Piper downloaded${NC}"
    else
        echo -e "${YELLOW}⚠ Piper already exists${NC}"
    fi
    
    # Download voice model
    echo "Downloading voice model..."
    mkdir -p models
    cd models
    if [ ! -f "en_US-lessac-medium.onnx" ]; then
        wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
        wget -q https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
        echo -e "${GREEN}✓ Voice model downloaded${NC}"
    else
        echo -e "${YELLOW}⚠ Voice model already exists${NC}"
    fi
    cd ..
    
elif [ "$OS" == "macos" ]; then
    echo -e "${YELLOW}⚠ Please install Piper manually on macOS:${NC}"
    echo "  brew install piper-tts"
    echo "  Or download from: https://github.com/rhasspy/piper/releases"
else
    echo -e "${YELLOW}⚠ Please install Piper manually for Windows:${NC}"
    echo "  Download from: https://github.com/rhasspy/piper/releases"
fi

echo ""

# Create config file
echo "════════════════════════════════════════════════════════════"
echo "  CONFIGURATION"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ ! -f "config.json" ]; then
    cat > config.json << 'EOF'
{
  "voice": {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "memory_dir": "/tmp/janus_voice",
    "whisper_model": "models/ggml-base.en.bin",
    "piper_model": "models/en_US-lessac-medium.onnx"
  },
  "messaging": {
    "host": "0.0.0.0",
    "port": 8080,
    "telegram_token": null,
    "sms": null
  },
  "tools": {
    "tools_dir": "discovered_tools",
    "discover_modules": []
  },
  "sync": {
    "device_name": null,
    "device_type": "laptop"
  }
}
EOF
    echo -e "${GREEN}✓ Config file created: config.json${NC}"
else
    echo -e "${YELLOW}⚠ Config file already exists${NC}"
fi

echo ""

# Create run script
cat > run_janus.sh << 'EOF'
#!/bin/bash
# Run Janus Unified

source venv/bin/activate
python janus_unified.py "$@"
EOF
chmod +x run_janus.sh

echo "════════════════════════════════════════════════════════════"
echo "  SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}Janus Unified is ready to use!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit config.json to add your messaging credentials"
echo ""
echo "2. Run Janus:"
echo "   ./run_janus.sh"
echo ""
echo "3. Or activate the environment and run:"
echo "   source venv/bin/activate"
echo "   python janus_unified.py"
echo ""
echo "4. Say 'Hey Janus' to start voice interaction"
echo ""
echo "5. Access messaging at: http://localhost:8080"
echo ""
echo "For more information, see UNIFIED_README.md"
echo ""
